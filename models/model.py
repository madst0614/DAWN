"""
DAWN: Dynamic Architecture With Neurons
A neural architecture with attention-guided dynamic neuron routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict
import math


# ============================================================
# Core Components
# ============================================================

class DynamicRouter(nn.Module):
    """
    Attention-based dynamic neuron router with Gumbel-Softmax.

    Uses Gumbel-Softmax for differentiable sampling instead of
    straight-through estimator to prevent collapse.

    Args:
        d_model: Model dimension
        n_neurons: Number of available neurons to route
        n_heads: Number of attention heads
        dropout: Dropout probability
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

        # Multi-head attention for context modeling
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Content-to-neuron affinity projection
        self.affinity_proj = nn.Linear(d_model, n_neurons)

        # Temperature parameter (fixed high for exploration)
        self.register_buffer('temperature', torch.tensor(2.0))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.affinity_proj.weight)
        nn.init.zeros_(self.affinity_proj.bias)

    def _gumbel_softmax_topk(
        self,
        logits: torch.Tensor,
        k: int,
        tau: float = 1.0,
        hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gumbel-Softmax with top-k constraint.

        Args:
            logits: Logits [batch, n_neurons]
            k: Number of neurons to select
            tau: Gumbel temperature
            hard: Use straight-through in forward (but soft in backward)

        Returns:
            indices: Top-k indices [batch, k]
            weights: Soft weights [batch, n_neurons]
        """
        # Apply Gumbel-Softmax
        if self.training:
            # Sample Gumbel noise
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(logits) + 1e-10
            ) + 1e-10)

            # Add noise to logits
            logits_with_noise = (logits + gumbel_noise) / tau

            # Softmax
            soft_weights = F.softmax(logits_with_noise, dim=-1)
        else:
            # Inference: no noise
            soft_weights = F.softmax(logits / tau, dim=-1)

        # Get top-k indices (based on original logits)
        _, indices = logits.topk(k, dim=-1)

        # Create top-k mask
        topk_mask = torch.zeros_like(soft_weights)
        topk_mask.scatter_(1, indices, 1.0)

        # Apply mask to weights
        masked_weights = soft_weights * topk_mask

        # Renormalize
        weights = masked_weights / (
            masked_weights.sum(dim=-1, keepdim=True) + 1e-8
        )

        if hard and self.training:
            # Straight-through: discrete forward, continuous backward
            hard_weights = torch.zeros_like(weights)
            hard_weights.scatter_(1, indices, 1.0)
            # Trick: detach difference
            weights = hard_weights + (weights - weights.detach())

        return indices, weights

    def forward(
        self,
        x: torch.Tensor,
        k: int,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route to top-k neurons based on input content.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            k: Number of neurons to select
            mask: Optional attention mask [batch, seq_len]

        Returns:
            indices: Selected neuron indices [batch, k]
            weights: Routing weights [batch, n_neurons]
            context: Attended representations [batch, seq_len, d_model]
        """
        # Apply attention for context modeling
        context, _ = self.attention(x, x, x, key_padding_mask=mask)
        # [batch, seq_len, d_model]

        # Compute neuron affinity scores
        affinity = self.affinity_proj(context)
        # [batch, seq_len, n_neurons]

        # Max-pooling: select based on maximum need across sequence
        scores, _ = affinity.max(dim=1)
        # [batch, n_neurons]

        # Add exploration noise during training
        if self.training:
            noise = torch.randn_like(scores) * 0.1
            scores = scores + noise

        # Route using Gumbel-Softmax
        indices, weights = self._gumbel_softmax_topk(
            logits=scores / self.temperature,
            k=k,
            tau=1.0,
            hard=False  # Soft selection
        )

        return indices, weights, context


class InputNeurons(nn.Module):
    """
    Input neuron layer with self-attention.

    Transforms input representations into neuron activations,
    with neurons communicating via self-attention mechanism.

    Args:
        d_model: Model dimension
        n_neurons: Number of neurons
        n_heads: Number of attention heads for neuron communication
        dropout: Dropout probability
    """

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

        # Learnable neuron patterns
        self.patterns = nn.Parameter(torch.empty(n_neurons, d_model))

        # Self-attention for neuron communication
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

        Args:
            context: Attended input [batch, seq_len, d_model]

        Returns:
            activations: Neuron activations [batch, seq_len, n_neurons]
        """
        # Transform to neuron space
        activations = F.gelu(context @ self.patterns.T)
        # [batch, seq_len, n_neurons]

        # Neuron self-attention
        attn_output, _ = self.self_attention(
            activations, activations, activations
        )

        # Residual connection and normalization
        activations = self.norm(activations + self.dropout(attn_output))
        # [batch, seq_len, n_neurons]

        return activations


class ProcessNeurons(nn.Module):
    """
    Process neuron layer with learned input combination analysis.

    Uses a small MLP to analyze input neuron selection patterns
    and predict process neuron relevance.

    Args:
        d_model: Model dimension
        n_input: Number of input neurons
        n_process: Number of process neurons
        hidden_dim: Hidden dimension for combination analyzer
    """

    def __init__(
        self,
        d_model: int,
        n_input: int,
        n_process: int,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()

        self.d_model = d_model
        self.n_input = n_input
        self.n_process = n_process

        # Hidden dimension for combination analyzer
        if hidden_dim is None:
            hidden_dim = n_input * 2

        # Combination weights: how process neurons combine input neurons
        self.combination_weights = nn.Parameter(
            torch.empty(n_process, n_input)
        )

        # Output projections: process neurons to model space
        self.output_projections = nn.Parameter(
            torch.empty(n_process, d_model)
        )

        # Learned combination analyzer
        # Analyzes which input neurons are selected and predicts
        # which process neurons should be relevant
        self.combination_analyzer = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_process)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.combination_weights, gain=math.sqrt(2.0))
        nn.init.orthogonal_(self.output_projections, gain=math.sqrt(2.0))

        # Initialize analyzer
        for module in self.combination_analyzer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        selected_activations: torch.Tensor,
        selected_indices: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """
        Process selected input neurons with learned combination analysis.

        Args:
            selected_activations: Selected neuron activations [batch, seq_len, k_in]
            selected_indices: Indices of selected neurons [batch, k_in]
            k: Number of process neurons to select

        Returns:
            output: Processed output [batch, seq_len, d_model]
        """
        batch_size, seq_len, k_in = selected_activations.shape

        # Gather relevant combination weights
        combination_weights_expanded = self.combination_weights.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, n_process, n_input]

        indices_expanded = selected_indices.unsqueeze(1).expand(
            -1, self.n_process, -1
        )  # [batch, n_process, k_in]

        selected_weights = torch.gather(
            combination_weights_expanded, 2, indices_expanded
        )  # [batch, n_process, k_in]

        # Compute process neuron activations
        process_activations = F.gelu(
            torch.bmm(selected_activations, selected_weights.transpose(1, 2))
        )  # [batch, seq_len, n_process]

        # Create binary selection pattern
        # This encodes which input neurons were selected
        input_selection = torch.zeros(
            batch_size, self.n_input,
            dtype=torch.float32,
            device=selected_indices.device
        )
        input_selection.scatter_(1, selected_indices, 1.0)
        # [batch, n_input]

        # Analyze this combination pattern
        # The MLP learns: "Given this input selection,
        # which process neurons are likely to be useful?"
        combination_relevance = self.combination_analyzer(input_selection)
        # [batch, n_process]

        # Compute activation-based scores
        activation_scores, _ = process_activations.max(dim=1)
        # [batch, n_process]

        # Combine: activation magnitude × combination relevance
        final_scores = activation_scores * torch.sigmoid(combination_relevance)
        # [batch, n_process]

        # Select top-k process neurons
        _, process_indices = final_scores.topk(k, dim=-1)
        # [batch, k]

        # Gather selected process activations
        process_indices_expanded = process_indices.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # [batch, seq_len, k]

        selected_process_activations = torch.gather(
            process_activations, 2, process_indices_expanded
        )  # [batch, seq_len, k]

        # Gather corresponding output projections
        output_projections_expanded = self.output_projections.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, n_process, d_model]

        process_indices_for_output = process_indices.unsqueeze(2).expand(
            -1, -1, self.d_model
        )  # [batch, k, d_model]

        selected_projections = torch.gather(
            output_projections_expanded, 1, process_indices_for_output
        )  # [batch, k, d_model]

        # Combine to produce output
        output = torch.bmm(selected_process_activations, selected_projections)
        # [batch, seq_len, d_model]

        return output


# ============================================================
# DAWN Block
# ============================================================

class DAWNBlock(nn.Module):
    """
    Complete DAWN block combining routing, input neurons, and process neurons.

    Args:
        d_model: Model dimension
        n_input: Number of input neurons
        n_process: Number of process neurons
        n_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_input: int = 64,
        n_process: int = 128,
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
        k_input: Optional[int] = None,
        k_process: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        return_routing_info: bool = False
    ) -> Union[Tuple[torch.Tensor, Dict], Tuple[torch.Tensor, Dict, Dict]]:
        """
        Forward pass through DAWN block.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            k_input: Number of input neurons to select (default: n_input // 2)
            k_process: Number of process neurons to select (default: n_process // 2)
            mask: Optional attention mask [batch, seq_len]
            return_routing_info: Whether to return routing info for aux loss

        Returns:
            output: Processed tensor [batch, seq_len, d_model]
            aux_loss: Dictionary of auxiliary losses (legacy, kept for compatibility)
            routing_info: (optional) Dictionary with indices and weights
        """
        batch_size, seq_len, _ = x.shape

        # Set default k values
        if k_input is None:
            k_input = self.n_input // 2
        if k_process is None:
            k_process = self.n_process // 2

        # Route to relevant neurons
        indices, weights, context = self.router(x, k_input, mask)
        # indices: [batch, k_input]
        # weights: [batch, n_input]
        # context: [batch, seq_len, d_model]

        # === Legacy Auxiliary Losses (for backward compatibility) ===
        # These are simple losses; better ones computed in train.py
        neuron_usage = weights.mean(dim=0)  # [n_input]
        load_balance_loss = neuron_usage.std()

        entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean()
        max_entropy = math.log(self.n_input)
        entropy_loss = 1.0 - (entropy / max_entropy)

        aux_loss = {
            'load_balance': load_balance_loss,
            'entropy': entropy_loss
        }

        # Compute input neuron activations
        activations = self.input_neurons(context)
        # [batch, seq_len, n_input]

        # Apply routing weights
        weighted_activations = activations * weights.unsqueeze(1)
        # [batch, seq_len, n_input]

        # Select routed neurons
        indices_expanded = indices.unsqueeze(1).expand(-1, seq_len, -1)
        selected_activations = torch.gather(
            weighted_activations, 2, indices_expanded
        )
        # [batch, seq_len, k_input]

        # Process neurons
        output = self.process_neurons(
            selected_activations, indices, k_process
        )
        # [batch, seq_len, d_model]

        output = self.dropout(output)

        if return_routing_info:
            routing_info = {
                'indices': indices,  # [batch, k]
                'weights': weights,  # [batch, n_neurons]
            }
            return output, aux_loss, routing_info

        return output, aux_loss


# ============================================================
# DAWN Layer
# ============================================================

class DAWNLayer(nn.Module):
    """
    DAWN layer with residual connection and layer normalization.

    Args:
        d_model: Model dimension
        n_input: Number of input neurons
        n_process: Number of process neurons
        n_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_input: int = 64,
        n_process: int = 128,
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
        k_input: Optional[int] = None,
        k_process: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        return_routing_info: bool = False
    ) -> Union[Tuple[torch.Tensor, dict], Tuple[torch.Tensor, dict, dict]]:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            k_input: Number of input neurons to select
            k_process: Number of process neurons to select
            mask: Optional attention mask
            return_routing_info: Whether to return routing info

        Returns:
            output: Layer output [batch, seq_len, d_model]
            aux_loss: Auxiliary losses from block
            routing_info: (optional) Routing information
        """
        if return_routing_info:
            output, aux_loss, routing_info = self.block(
                x, k_input, k_process, mask, return_routing_info=True
            )
            return self.norm(x + output), aux_loss, routing_info

        output, aux_loss = self.block(x, k_input, k_process, mask)
        return self.norm(x + output), aux_loss


# ============================================================
# Language Model
# ============================================================

class DAWNLanguageModel(nn.Module):
    """
    DAWN-based language model.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_layers: Number of DAWN layers
        n_input: Number of input neurons per layer
        n_process: Number of process neurons per layer
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 12,
        n_input: int = 64,
        n_process: int = 128,
        n_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        **kwargs  # For compatibility
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
        k_input: Optional[int] = None,
        k_process: Optional[int] = None,
        return_routing_info: bool = False
    ) -> dict:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels for language modeling [batch, seq_len]
            k_input: Number of input neurons to select
            k_process: Number of process neurons to select
            return_routing_info: Whether to return routing info for aux loss

        Returns:
            Dictionary containing:
                - logits: Output logits [batch, seq_len, vocab_size]
                - loss: Cross-entropy loss (if labels provided)
                - aux_loss: Aggregated auxiliary losses
                - routing_info: (optional) List of routing info per layer
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(
            batch_size, -1
        )
        position_emb = self.position_embedding(positions)

        x = self.dropout(token_emb + position_emb)

        # Process through DAWN layers and collect aux losses
        all_aux_losses = []
        routing_info_list = [] if return_routing_info else None

        for layer in self.layers:
            if return_routing_info:
                x, aux_loss, routing_info = layer(
                    x, k_input, k_process, mask=attention_mask,
                    return_routing_info=True
                )
                routing_info_list.append(routing_info)
            else:
                x, aux_loss = layer(x, k_input, k_process, mask=attention_mask)
            all_aux_losses.append(aux_loss)

        # Output projection
        x = self.norm(x)
        logits = self.output(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        # Aggregate auxiliary losses
        aggregated_aux = {
            'load_balance': sum(l['load_balance'] for l in all_aux_losses) / len(all_aux_losses),
            'entropy': sum(l['entropy'] for l in all_aux_losses) / len(all_aux_losses)
        }

        result = {'logits': logits, 'loss': loss, 'aux_loss': aggregated_aux}

        if return_routing_info:
            result['routing_info'] = routing_info_list

        return result

    def get_model_stats(self) -> dict:
        """Get model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'n_layers': self.n_layers,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model
        }

    @classmethod
    def from_config(cls, config: dict, vocab_size: int):
        """
        Create model from config dict.

        Args:
            config: Config dict with 'model' section
            vocab_size: Vocabulary size

        Returns:
            DAWNLanguageModel instance
        """
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


# ============================================================
# Aliases for backward compatibility
# ============================================================

# Keep old names as aliases
GlobalRouter = DynamicRouter
HierarchicalDynamicFFN = DAWNBlock
TransformerLayerWithHierarchicalFFN = DAWNLayer
HierarchicalLanguageModel = DAWNLanguageModel


# ============================================================
# Testing
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("DAWN: Dynamic Activation in Weighted Networks")
    print("=" * 60)
    print()

    # Initialize model
    model = DAWNLanguageModel(
        vocab_size=30000,
        d_model=512,
        n_layers=6,
        n_input=64,
        n_process=128,
        n_heads=8,
        max_seq_len=512,
        dropout=0.1
    )

    # Print model statistics
    stats = model.get_model_stats()

    print("Model Configuration:")
    print(f"  Layers: {stats['n_layers']}")
    print(f"  Model dimension: {stats['d_model']}")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
    print()

    # Test forward pass
    batch_size = 4
    seq_len = 128

    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    labels = torch.randint(0, 30000, (batch_size, seq_len))

    print("Testing forward pass...")
    with torch.no_grad():
        output = model(input_ids, labels=labels)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")
    print()

    print("✓ Tests passed successfully!")
