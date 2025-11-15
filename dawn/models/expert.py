"""
Delta Expert - PNN-based Expert with Peer Context

Single expert that performs recurrent refinement through deltas.

Architecture:
1. Self processing → delta
2. Peer interpretation → deltas (Phase 2)
3. Integration → total delta
4. State update (PNN: h_new = h + delta)

Seamlessly works with 0 to N peers.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List

from .components.context.peer_context import PeerContext
from .components.delta.delta_module import DeltaModule
from .components.integration.peer_integrator import PeerIntegrator


class DeltaExpert(nn.Module):
    """
    Delta-based Expert with Peer Context

    Performs recurrent refinement with optional peer collaboration.

    Args:
        config: Dict - full model configuration
        peer_names: Optional[List[str]] - peer expert names (None for Phase 1)
        shared_embeddings: Optional[nn.ModuleDict] - shared embeddings
    """

    def __init__(
        self,
        config: Dict,
        peer_names: Optional[List[str]] = None,
        shared_embeddings: Optional[nn.ModuleDict] = None,
    ):
        super().__init__()

        self.hidden_size = config['hidden_size']
        self.num_steps = config['num_steps']
        self.vocab_size = config['vocab_size']
        self.max_length = config['max_length']  # Store max_length for truncation
        self.peer_names = sorted(peer_names) if peer_names else []

        # === Embeddings ===
        self._setup_embeddings(config, shared_embeddings)

        # === Peer Context (Phase 2) ===
        if self.peer_names:
            peer_context_config = config.get('peer_context', {})
            self.peer_context = PeerContext(
                hidden_size=self.hidden_size,
                peer_names=self.peer_names,
                config=peer_context_config,
            )
        else:
            self.peer_context = None

        # === Delta Module (Universal) ===
        delta_config = config.get('delta_module', {})
        self.delta_module = DeltaModule(
            hidden_size=self.hidden_size,
            num_blocks=delta_config.get('num_blocks', 5),
            num_heads=config['num_heads'],
            intermediate_size=config['intermediate_size'],
            dropout=config['dropout'],
            config=delta_config,
        )

        # === Peer Integrator ===
        max_sources = 1 + len(self.peer_names) if self.peer_names else 1
        integration_config = config.get('integration', {})
        self.peer_integrator = PeerIntegrator(
            hidden_size=self.hidden_size,
            max_sources=max_sources,
            config=integration_config,
        )

        self._init_weights(config)

    def _setup_embeddings(self, config: Dict, shared_embeddings: Optional[nn.ModuleDict]):
        """Setup embeddings (shared or own)"""
        if shared_embeddings is not None:
            self.token_embeddings = shared_embeddings['token']
            self.position_embeddings = shared_embeddings['position']
            self.embedding_layer_norm = shared_embeddings['layer_norm']
            self.embedding_dropout = shared_embeddings['dropout']
            self._owns_embeddings = False
        else:
            max_length = config['max_length']
            dropout = config['dropout']

            self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
            self.position_embeddings = nn.Embedding(max_length, self.hidden_size)
            self.embedding_layer_norm = nn.LayerNorm(self.hidden_size)
            self.embedding_dropout = nn.Dropout(dropout)
            self._owns_embeddings = True

    def _init_weights(self, config: Dict):
        """
        Initialize weights following BERT/PNN best practices

        Only initializes expert's own embeddings if not using shared embeddings.
        All subcomponents (delta_module, peer_context, peer_integrator) handle
        their own initialization internally, preserving special init patterns
        (e.g., zero-init in DeltaRefiner final layers).
        """
        init_std = config.get('init_std', 0.02)

        # Only initialize embeddings if we own them (not shared)
        if self._owns_embeddings:
            # Token embeddings
            self.token_embeddings.weight.data.normal_(mean=0.0, std=init_std)

            # Position embeddings
            self.position_embeddings.weight.data.normal_(mean=0.0, std=init_std)

            # LayerNorm
            self.embedding_layer_norm.bias.data.zero_()
            self.embedding_layer_norm.weight.data.fill_(1.0)

        # All other components (delta_module, peer_context, peer_integrator)
        # initialize themselves in their own __init__ methods.
        # This preserves special initialization patterns like zero-init in DeltaRefiner.

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get initial embeddings"""
        batch_size, seq_len = input_ids.shape

        # Truncate to max_length if needed (safety check)
        if seq_len > self.max_length:
            print(f"⚠️  WARNING: Input sequence length ({seq_len}) exceeds max_length ({self.max_length}). Truncating to {self.max_length}.")
            input_ids = input_ids[:, :self.max_length]
            seq_len = self.max_length

        token_embeds = self.token_embeddings(input_ids)

        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = token_embeds + position_embeds
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def _collect_deltas(
        self,
        h: torch.Tensor,
        peer_outputs: Optional[Dict[str, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Collect deltas from all sources

        Args:
            h: [B, L, D] - current state
            peer_outputs: {peer_name: [B, L, D]} - peer outputs
            attention_mask: [B, L] (True = masked)

        Returns:
            deltas: [num_sources, B, L, D]
        """
        deltas = []

        # Self delta
        delta_self = self.delta_module(h, attention_mask)
        deltas.append(delta_self)

        # Peer deltas (Phase 2)
        if self.peer_context and peer_outputs:
            interpreted_peers = self.peer_context(
                h_self=h,
                peer_outputs=peer_outputs,
                attention_mask=attention_mask,
            )

            for peer_h in interpreted_peers:
                delta_peer = self.delta_module(peer_h, attention_mask)
                deltas.append(delta_peer)

        return torch.stack(deltas, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        peer_outputs: Optional[Dict[str, torch.Tensor]] = None,
        return_all_steps: bool = False,
    ):
        """
        Unified forward pass

        Args:
            input_ids: [B, L]
            attention_mask: [B, L] (1 = keep, 0 = mask)
            peer_outputs: {peer_name: [B, L, D]} (Phase 2)
            return_all_steps: bool

        Returns:
            h: [B, L, D] or List[[B, L, D]]
        """
        # Initial embeddings
        h = self.get_embeddings(input_ids)

        # Convert attention mask (1=keep → 0=keep, 0=mask → 1=mask)
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask == 0  # True = masked

        # Storage
        all_outputs = [h] if return_all_steps else None

        # === Recurrent Refinement Loop ===
        for step in range(self.num_steps):
            # 1. Collect deltas from all sources
            deltas = self._collect_deltas(h, peer_outputs, attn_mask)

            # 2. Integrate deltas
            total_delta = self.peer_integrator(h, deltas)

            # 3. Update state (PNN!)
            h_new = h + total_delta
            # GeGLU ensures stable deltas - no clamping needed!

            # Update state - allow gradients to flow through all steps
            h = h_new

            if return_all_steps:
                all_outputs.append(h)

        return all_outputs if return_all_steps else h

    def add_peer_context(self, peer_names: List[str], config: Dict):
        """
        Add peer context after Phase 1 training

        Args:
            peer_names: List of peer expert names
            config: Full config dict
        """
        self.peer_names = sorted(peer_names)

        # Add peer context
        peer_context_config = config.get('peer_context', {})
        self.peer_context = PeerContext(
            hidden_size=self.hidden_size,
            peer_names=self.peer_names,
            config=peer_context_config,
        )

        # Update integrator if needed
        new_max_sources = 1 + len(peer_names)
        if new_max_sources > self.peer_integrator.max_sources:
            # Expand peer_to_gain modules
            for _ in range(new_max_sources - self.peer_integrator.max_sources - 1):
                new_module = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size // 2),
                    nn.LayerNorm(self.hidden_size // 2),
                    nn.GELU(),
                    nn.Linear(self.hidden_size // 2, self.hidden_size),
                    nn.Tanh(),
                )
                self.peer_integrator.peer_to_gain.append(new_module)

            # Expand peer_blend_weights parameter
            old_weights = self.peer_integrator.peer_blend_weights.data
            new_weights = torch.zeros(new_max_sources - 1, device=old_weights.device)
            new_weights[:old_weights.size(0)] = old_weights
            self.peer_integrator.peer_blend_weights = nn.Parameter(new_weights)
            self.peer_integrator.max_sources = new_max_sources

    def freeze_base(self):
        """Freeze base modules, train peer context only"""
        # Freeze delta module
        for param in self.delta_module.parameters():
            param.requires_grad = False

        # Freeze embeddings
        if self._owns_embeddings:
            for param in self.token_embeddings.parameters():
                param.requires_grad = False
            for param in self.position_embeddings.parameters():
                param.requires_grad = False

        # Freeze integrator
        for param in self.peer_integrator.parameters():
            param.requires_grad = False

        # Keep peer context trainable
        if self.peer_context:
            for param in self.peer_context.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze everything"""
        for param in self.parameters():
            param.requires_grad = True

    def compute_recurrent_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_heads,
        task_name: str,
        labels: torch.Tensor,
        step_weights: list = None,
        return_accuracies: bool = False,
    ) -> tuple:
        """
        Memory-efficient computation of weighted loss across all refinement steps.

        Key optimization: Processes one step at a time instead of storing all steps.
        This prevents OOM by only keeping current hidden state in memory.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len] (1=keep, 0=mask)
            task_heads: TaskHeads module for computing task-specific loss
            task_name: Name of the task (e.g., "mlm", "sst2")
            labels: Target labels [batch, seq_len] or [batch]
            step_weights: Weights for each step (default: [0.1, 0.2, 0.3, 0.4])
            return_accuracies: If True, also return step-wise accuracies

        Returns:
            total_loss: Weighted sum of step losses
            step_losses: List of individual losses per step
            step_accs: (optional) List of individual accuracies per step
        """
        if step_weights is None:
            step_weights = [0.1, 0.2, 0.3, 0.4]

        # Initial embeddings
        h = self.get_embeddings(input_ids)
        if not torch.isfinite(h).all():
            raise RuntimeError("NaN/Inf detected in embeddings!")

        # Convert attention mask (1=keep → 0=keep, 0=mask → 1=mask)
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask == 0  # True = masked

        total_loss = 0.0
        step_losses = []
        step_accs = [] if return_accuracies else None

        # Process each refinement step sequentially (memory efficient!)
        for step in range(self.num_steps):
            # 1. Collect deltas from all sources
            deltas = self._collect_deltas(h, peer_outputs=None, attention_mask=attn_mask)
            if not torch.isfinite(deltas).all():
                num_nan = torch.isnan(deltas).sum().item()
                num_inf = torch.isinf(deltas).sum().item()
                raise RuntimeError(
                    f"NaN/Inf detected in deltas at step {step}!\n"
                    f"  NaN count: {num_nan}, Inf count: {num_inf}\n"
                    f"  Deltas shape: {deltas.shape}\n"
                    f"  Input h range: [{h.min():.4f}, {h.max():.4f}]"
                )

            # 2. Integrate deltas
            total_delta = self.peer_integrator(h, deltas)
            if not torch.isfinite(total_delta).all():
                num_nan = torch.isnan(total_delta).sum().item()
                num_inf = torch.isinf(total_delta).sum().item()
                raise RuntimeError(
                    f"NaN/Inf detected in total_delta at step {step}!\n"
                    f"  NaN count: {num_nan}, Inf count: {num_inf}\n"
                    f"  Total_delta range: [{total_delta.min():.4f}, {total_delta.max():.4f}]\n"
                    f"  Input deltas range: [{deltas.min():.4f}, {deltas.max():.4f}]"
                )

            # 3. Update state (PNN!)
            h_new = h + total_delta
            # GeGLU ensures stable deltas - no clamping needed!

            # Detach for independent gradient per step (memory efficient)
            # This allows each step to learn independently while maintaining recurrence
            if step < self.num_steps - 1:
                h = h_new.detach()
            else:
                h = h_new

            if not torch.isfinite(h).all():
                num_nan = torch.isnan(h).sum().item()
                num_inf = torch.isinf(h).sum().item()
                raise RuntimeError(
                    f"NaN/Inf detected in h after step {step}!\n"
                    f"  NaN count: {num_nan}, Inf count: {num_inf}\n"
                    f"  h range: [{h.min():.4f}, {h.max():.4f}]\n"
                    f"  total_delta range: [{total_delta.min():.4f}, {total_delta.max():.4f}]"
                )

            # 4. Compute loss for this step
            result = task_heads(h, task_name, labels=labels)
            loss = result["loss"]
            logits = result.get("logits")

            # 5. Weight and accumulate
            weight = step_weights[step]
            total_loss += weight * loss
            step_losses.append(loss.item())

            # 6. Compute accuracy if requested
            if return_accuracies and logits is not None:
                with torch.no_grad():
                    if task_name == "sts":
                        # Regression task: no accuracy
                        step_accs.append(0.0)
                    else:
                        preds = logits.detach().argmax(dim=-1).view(-1)  # [B*L]
                        labels_flat = labels.view(-1)  # [B*L]
                        mask = (labels_flat != -100)  # [B*L]
                        correct = ((preds == labels_flat) & mask).sum().item()
                        total_tokens = mask.sum().item()
                        acc = correct / total_tokens if total_tokens > 0 else 0.0
                        step_accs.append(acc)

            # 7. CRITICAL: Delete large tensors immediately to free memory
            del logits
            del result
            del deltas
            del total_delta

        if return_accuracies:
            return total_loss, step_losses, step_accs
        else:
            return total_loss, step_losses
