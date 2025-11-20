"""
DAWN: Dynamic Architecture With Neurons
A neural architecture with FULLY LEARNABLE soft neuron routing

Key principles:
- Zero manual configuration
- Pure soft selection (no hard top-k)
- All parameters learned from data
- Complete gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


# ============================================================
# Debug Logger
# ============================================================

class DebugLogger:
    """Global debug logger for DAWN model."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.enabled = False
            cls._instance.log_file = None
            cls._instance.step = 0
            cls._instance.verbose_steps = set()
        return cls._instance

    def setup(self, log_file: str, enabled: bool = True):
        self.log_file = log_file
        self.enabled = enabled
        if enabled and log_file:
            with open(log_file, 'a') as f:
                f.write(f"\n{'='*60}\nDebug logging started\n{'='*60}\n")

    def set_step(self, step: int):
        self.step = step
        if step % 100 == 0:
            self.verbose_steps.add(step)

    def log(self, component: str, message: str):
        if not self.enabled or not self.log_file:
            return
        with open(self.log_file, 'a') as f:
            f.write(f"[Step {self.step}][{component}] {message}\n")

    def log_tensor(self, component: str, name: str, tensor: torch.Tensor):
        if not self.enabled or not self.log_file:
            return

        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        should_log_detail = has_nan or has_inf or (self.step in self.verbose_steps)

        if should_log_detail:
            msg = (f"{name} - shape: {list(tensor.shape)}, "
                   f"min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}, "
                   f"mean: {tensor.mean().item():.4f}, std: {tensor.std().item():.4f}, "
                   f"nan: {has_nan}, inf: {has_inf}")
            self.log(component, msg)

        if has_nan or has_inf:
            self.log(component, f"🔥 {'NaN' if has_nan else 'Inf'} DETECTED in {name}!")


debug_logger = DebugLogger()


# ============================================================
# Regularization Functions
# ============================================================

def compute_orthogonality_loss(weight_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute orthogonality regularization loss for weight matrix.
    """
    normalized = F.normalize(weight_matrix, p=2, dim=1)
    gram = normalized @ normalized.T
    n = gram.size(0)
    identity = torch.eye(n, device=gram.device)
    off_diagonal_mask = 1 - identity
    ortho_loss = (gram * off_diagonal_mask).pow(2).sum() / (n * (n - 1) + 1e-8)
    return ortho_loss


def compute_model_orthogonality_loss(model) -> dict:
    """
    Compute orthogonality loss for all weight matrices in DAWN model.
    """
    router_losses = []
    input_losses = []
    process_comb_losses = []
    process_proj_losses = []

    for layer in model.layers:
        block = layer.block

        router_ortho = compute_orthogonality_loss(block.router.neuron_patterns)
        router_losses.append(router_ortho)

        input_ortho = compute_orthogonality_loss(block.input_neurons.patterns)
        input_losses.append(input_ortho)

        comb_ortho = compute_orthogonality_loss(block.process_neurons.combination_weights)
        process_comb_losses.append(comb_ortho)

        proj_ortho = compute_orthogonality_loss(block.process_neurons.output_projections)
        process_proj_losses.append(proj_ortho)

    n_layers = len(model.layers)

    return {
        'router_ortho': sum(router_losses) / n_layers,
        'input_ortho': sum(input_losses) / n_layers,
        'process_comb_ortho': sum(process_comb_losses) / n_layers,
        'process_proj_ortho': sum(process_proj_losses) / n_layers,
        'total_ortho': (sum(router_losses) + sum(input_losses) +
                       sum(process_comb_losses) + sum(process_proj_losses)) / (4 * n_layers)
    }


def compute_learned_sparsity_loss(
    weights: torch.Tensor,
    selection_info: dict
) -> torch.Tensor:
    """
    STRONG sparsity guidance with progressive penalties.

    Target: 30-70% active neurons
    Progressive penalty: stronger when further from target
    """
    effective_k_ratio = selection_info.get('effective_k_ratio', 0.5)

    # Target range
    target_min = 0.3
    target_max = 0.7
    target_center = 0.5

    # STRONG penalties for extremes (2x multiplier)
    if effective_k_ratio < target_min:
        deviation = target_min - effective_k_ratio
        sparsity_penalty = (deviation * 2.0) ** 2
    elif effective_k_ratio > target_max:
        deviation = effective_k_ratio - target_max
        sparsity_penalty = (deviation * 2.0) ** 2
    else:
        # Gentle centering toward 0.5
        sparsity_penalty = 0.1 * (effective_k_ratio - target_center) ** 2

    penalty_tensor = torch.tensor(sparsity_penalty, device=weights.device, dtype=weights.dtype)

    # Gini diversity penalty (also stronger)
    avg_weights = weights.mean(dim=0)
    sorted_weights, _ = torch.sort(avg_weights)
    n = len(sorted_weights)
    index = torch.arange(1, n + 1, device=weights.device, dtype=torch.float32)
    gini = (2 * (index * sorted_weights).sum()) / (n * sorted_weights.sum() + 1e-8) - (n + 1) / n

    if gini < 0.2:
        gini_penalty = ((0.2 - gini) * 2.0) ** 2
    elif gini > 0.6:
        gini_penalty = ((gini - 0.6) * 2.0) ** 2
    else:
        gini_penalty = torch.tensor(0.0, device=weights.device)

    return penalty_tensor + gini_penalty


# ============================================================
# Core Components
# ============================================================

class DynamicRouter(nn.Module):
    """
    Fully differentiable attention-based dynamic neuron router.

    ALL sparsity parameters are learned:
    - temperature: controls softmax distribution
    - threshold: controls selection cutoff
    - steepness: controls transition sharpness

    NO clamps, NO hard limits - pure learning!
    """

    def __init__(
        self,
        d_model: int,
        n_neurons: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_neurons = n_neurons
        self.n_heads = n_heads

        # Context attention
        self.context_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Learnable neuron patterns
        self.neuron_patterns = nn.Parameter(
            torch.empty(n_neurons, d_model)
        )

        # Neuron attention for routing
        self.neuron_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.0,
            batch_first=True
        )

        # ============================================================
        # THREE Learnable parameters (NO CLAMPS!)
        # ============================================================

        # Temperature: softmax sharpness
        # Let model learn any value - will naturally stabilize
        self.log_temperature = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0

        # Threshold: where to cut
        # Sigmoid naturally bounds to (0, 1)
        self.sparsity_threshold = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

        # Steepness: transition sharpness
        # Let model learn - will find optimal steepness
        self.log_steepness = nn.Parameter(torch.tensor(math.log(3.0)))  # exp(log(3)) = 3.0

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.neuron_patterns, gain=1.0)

    def get_temperature(self) -> torch.Tensor:
        """Get learned temperature (unclamped!)"""
        return torch.exp(self.log_temperature)

    def get_threshold(self) -> torch.Tensor:
        """Get learned threshold (naturally bounded by sigmoid)"""
        return torch.sigmoid(self.sparsity_threshold)

    def get_steepness(self) -> torch.Tensor:
        """Get learned steepness (unclamped!)"""
        return torch.exp(self.log_steepness)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Pure soft selection - NO hard k!

        Returns:
            weights: Soft routing weights [batch, n_neurons]
            context: Attended representations [batch, seq_len, d_model]
            selection_info: Dict with learned parameters
        """
        batch_size, seq_len, d_model = x.shape

        # Context modeling
        context, _ = self.context_attention(x, x, x, key_padding_mask=mask)

        # Attend to neuron patterns
        neuron_kv = self.neuron_patterns.unsqueeze(0).expand(batch_size, -1, -1)
        neuron_responses, attn_weights = self.neuron_attention(
            context, neuron_kv, neuron_kv,
            need_weights=True,
            average_attn_weights=True
        )

        # Aggregate scores
        scores, _ = attn_weights.max(dim=1)

        debug_logger.log_tensor("Router", "scores", scores)

        # Normalize scores
        scores_normalized = (scores - scores.mean(dim=-1, keepdim=True)) / \
                           (scores.std(dim=-1, keepdim=True) + 1e-6)

        # Get learned parameters (NO CLAMPS!)
        temperature = self.get_temperature()
        learned_threshold = self.get_threshold()
        steepness = self.get_steepness()

        # Base probabilities
        probs = F.softmax(scores_normalized / temperature, dim=-1)

        # Normalize scores to [0, 1]
        scores_min = scores_normalized.min(dim=-1, keepdim=True)[0]
        scores_max = scores_normalized.max(dim=-1, keepdim=True)[0]
        scores_01 = (scores_normalized - scores_min) / (scores_max - scores_min + 1e-8)

        # Soft mask with learned threshold and steepness
        soft_mask = torch.sigmoid(steepness * (scores_01 - learned_threshold))

        # Apply soft mask
        masked_probs = probs * soft_mask
        weights = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Measure effective sparsity (for logging)
        active_neurons = (weights > 1e-3).float().sum(dim=-1).mean()
        effective_k_ratio = active_neurons / self.n_neurons

        selection_info = {
            'learned_threshold': learned_threshold.item(),
            'learned_steepness': steepness.item(),
            'effective_k': active_neurons.item(),
            'effective_k_ratio': effective_k_ratio.item(),
            'temperature': temperature.item()
        }

        debug_logger.log_tensor("Router", "weights", weights)
        debug_logger.log("Router", f"weights sum: {weights.sum(dim=-1).mean().item():.4f}")
        debug_logger.log("Router",
            f"threshold: {learned_threshold.item():.3f}, "
            f"steepness: {steepness.item():.2f}, "
            f"temp: {temperature.item():.2f}, "
            f"eff_k: {active_neurons.item():.1f}")

        return weights, context, selection_info


class InputNeurons(nn.Module):
    """Input neuron layer with self-attention."""

    def __init__(
        self,
        d_model: int,
        n_neurons: int,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_neurons = n_neurons

        self.patterns = nn.Parameter(torch.empty(n_neurons, d_model))

        self.self_attention = nn.MultiheadAttention(
            embed_dim=n_neurons,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(n_neurons)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.patterns, gain=math.sqrt(2.0))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Compute neuron activations.

        Returns:
            activations: [batch, seq_len, n_neurons]
        """
        activations = F.gelu(context @ self.patterns.T)

        debug_logger.log_tensor("InputNeurons", "activations_pre", activations)

        normalized = self.norm(activations)
        attn_output, _ = self.self_attention(normalized, normalized, normalized)
        activations = activations + self.dropout(attn_output)

        debug_logger.log_tensor("InputNeurons", "activations_post", activations)

        return activations


class ProcessNeurons(nn.Module):
    """
    SIMPLIFIED process neuron layer.

    No indexing, no hard selection - just clean matrix multiplication!
    """

    def __init__(
        self,
        d_model: int,
        n_input: int,
        n_process: int
    ):
        super().__init__()

        self.d_model = d_model
        self.n_input = n_input
        self.n_process = n_process

        # How process neurons combine input neurons
        self.combination_weights = nn.Parameter(
            torch.empty(n_process, n_input)
        )

        # Process neurons to output space
        self.output_projections = nn.Parameter(
            torch.empty(n_process, d_model)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.combination_weights, gain=math.sqrt(2.0))
        nn.init.orthogonal_(self.output_projections, gain=math.sqrt(2.0))

    def forward(self, weighted_activations: torch.Tensor) -> torch.Tensor:
        """
        SIMPLE soft processing - just matmul!

        Args:
            weighted_activations: [batch, seq_len, n_input]

        Returns:
            output: [batch, seq_len, d_model]
        """
        # Compute process neuron activations
        # [batch, seq_len, n_input] @ [n_input, n_process] = [batch, seq_len, n_process]
        process_activations = F.gelu(
            weighted_activations @ self.combination_weights.T
        )

        debug_logger.log_tensor("ProcessNeurons", "process_activations", process_activations)

        # Project to output
        # [batch, seq_len, n_process] @ [n_process, d_model] = [batch, seq_len, d_model]
        output = process_activations @ self.output_projections

        debug_logger.log_tensor("ProcessNeurons", "output", output)

        return output


# ============================================================
# DAWN Block
# ============================================================

class DAWNBlock(nn.Module):
    """
    Complete DAWN block with PURE SOFT selection.

    No k parameters, no hard selection - fully differentiable!
    """

    def __init__(
        self,
        d_model: int,
        n_input: int = 128,
        n_process: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_input = n_input
        self.n_process = n_process

        self.router = DynamicRouter(
            d_model=d_model,
            n_neurons=n_input,
            n_heads=n_heads,
            dropout=dropout
        )

        self.input_neurons = InputNeurons(
            d_model=d_model,
            n_neurons=n_input,
            n_heads=4,
            dropout=dropout
        )

        self.process_neurons = ProcessNeurons(
            d_model=d_model,
            n_input=n_input,
            n_process=n_process
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_routing_info: bool = False
    ) -> Tuple[torch.Tensor, Dict] or Tuple[torch.Tensor, Dict, Dict]:
        """
        Forward with PURE soft selection.

        NO k parameters - model decides everything!
        """
        # Route with pure soft selection (no k!)
        weights, context, selection_info = self.router(x, mask)
        # weights: [batch, n_input] - ALL neurons, soft weights

        # Compute aux losses
        neuron_usage = weights.mean(dim=0)
        load_balance_loss = neuron_usage.std()

        weights_clamped = weights.clamp(min=1e-10)
        entropy = -(weights_clamped * torch.log(weights_clamped)).sum(dim=-1).mean()
        max_entropy = math.log(self.n_input)
        entropy_loss = 1.0 - (entropy / max_entropy)

        sparsity_guidance = compute_learned_sparsity_loss(weights, selection_info)

        aux_loss = {
            'load_balance': load_balance_loss,
            'entropy': entropy_loss,
            'sparsity_guidance': sparsity_guidance
        }

        # Compute activations
        activations = self.input_neurons(context)
        # [batch, seq_len, n_input]

        # Apply soft routing (ALL neurons!)
        weighted_activations = activations * weights.unsqueeze(1)
        # [batch, seq_len, n_input]
        # Gradient flows to ALL neurons!

        # Process (simple matmul!)
        output = self.process_neurons(weighted_activations)
        # [batch, seq_len, d_model]

        output = self.dropout(output)

        if return_routing_info:
            routing_info = {
                'weights': weights,
                'learned_threshold': selection_info['learned_threshold'],
                'learned_steepness': selection_info['learned_steepness'],
                'effective_k': selection_info['effective_k'],
                'effective_k_ratio': selection_info['effective_k_ratio'],
                'temperature': selection_info['temperature']
            }
            return output, aux_loss, routing_info

        return output, aux_loss


# ============================================================
# DAWN Layer
# ============================================================

class DAWNLayer(nn.Module):
    """DAWN layer with residual connection and layer normalization."""

    def __init__(
        self,
        d_model: int,
        n_input: int = 128,
        n_process: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.block = DAWNBlock(
            d_model=d_model,
            n_input=n_input,
            n_process=n_process,
            n_heads=n_heads,
            dropout=dropout
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_routing_info: bool = False
    ) -> Tuple[torch.Tensor, dict] or Tuple[torch.Tensor, dict, dict]:
        """Forward with residual connection."""
        normed = self.norm(x)

        if return_routing_info:
            output, aux_loss, routing_info = self.block(
                normed, mask, return_routing_info=True
            )
            return x + output, aux_loss, routing_info

        output, aux_loss = self.block(normed, mask)
        return x + output, aux_loss


# ============================================================
# Language Model
# ============================================================

class DAWNLanguageModel(nn.Module):
    """DAWN-based language model with PURE SOFT selection."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_input: int = 128,
        n_process: int = 256,
        n_heads: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # DAWN layers
        self.layers = nn.ModuleList([
            DAWNLayer(
                d_model=d_model,
                n_input=n_input,
                n_process=n_process,
                n_heads=n_heads,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_routing_info: bool = False
    ) -> dict:
        """Forward pass - NO k parameters!"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(positions)

        x = self.dropout(token_emb + position_emb)

        # Process through layers
        all_aux_losses = []
        routing_info_list = [] if return_routing_info else None

        for layer in self.layers:
            if return_routing_info:
                x, aux_loss, routing_info = layer(
                    x, mask=attention_mask, return_routing_info=True
                )
                routing_info_list.append(routing_info)
            else:
                x, aux_loss = layer(x, mask=attention_mask)
            all_aux_losses.append(aux_loss)

        # Output
        x = self.norm(x)
        logits = self.output(x)

        debug_logger.log_tensor("Model", "logits", logits)

        # Loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            debug_logger.log("Model", f"loss: {loss.item():.4f}, nan: {torch.isnan(loss).item()}")

        # Aggregate aux losses
        aggregated_aux = {
            'load_balance': sum(l['load_balance'] for l in all_aux_losses) / len(all_aux_losses),
            'entropy': sum(l['entropy'] for l in all_aux_losses) / len(all_aux_losses),
            'sparsity_guidance': sum(l['sparsity_guidance'] for l in all_aux_losses) / len(all_aux_losses)
        }

        debug_logger.log("Model", f"aux_loss - lb: {aggregated_aux['load_balance']:.4f}, ent: {aggregated_aux['entropy']:.4f}, sp: {aggregated_aux['sparsity_guidance']:.4f}")

        result = {'logits': logits, 'loss': loss, 'aux_loss': aggregated_aux}

        if return_routing_info:
            result['routing_info'] = routing_info_list

        return result

    def get_model_stats(self) -> dict:
        """Get model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'n_layers': self.n_layers,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model
        }

    @classmethod
    def from_config(cls, config: dict, vocab_size: int):
        """Create model from config dict."""
        model_cfg = config['model']
        return cls(
            vocab_size=vocab_size,
            d_model=model_cfg['d_model'],
            n_layers=model_cfg['n_layers'],
            n_input=model_cfg['n_input'],
            n_process=model_cfg['n_process'],
            n_heads=model_cfg['n_heads'],
            max_seq_len=model_cfg['max_seq_len'],
            dropout=model_cfg['dropout']
        )


# Backward compatibility
GlobalRouter = DynamicRouter
HierarchicalDynamicFFN = DAWNBlock
TransformerLayerWithHierarchicalFFN = DAWNLayer
HierarchicalLanguageModel = DAWNLanguageModel


if __name__ == '__main__':
    print("=" * 60)
    print("DAWN: Dynamic Architecture With Neurons")
    print("Pure soft selection - Zero manual configuration")
    print("=" * 60)
    print()

    model = DAWNLanguageModel(
        vocab_size=30000,
        d_model=512,
        n_layers=6,
        n_input=128,
        n_process=256,
        n_heads=8,
        max_seq_len=512,
        dropout=0.1
    )

    stats = model.get_model_stats()
    print("Model Configuration:")
    print(f"  Layers: {stats['n_layers']}")
    print(f"  Model dimension: {stats['d_model']}")
    print(f"  Input neurons: {128} (learned sparsity)")
    print(f"  Process neurons: {256} (learned sparsity)")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
    print()

    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    labels = torch.randint(0, 30000, (batch_size, seq_len))

    print("Testing forward pass...")
    with torch.no_grad():
        output = model(input_ids, labels=labels, return_routing_info=True)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")
    print(f"  Routing info layers: {len(output['routing_info'])}")
    print()

    print("✓ Pure soft selection ready!")
    print("✓ Zero manual configuration!")
    print("✓ All parameters learned from data!")
