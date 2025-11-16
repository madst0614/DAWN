"""
Contextual SPROUT Layer with Firing Pattern Combination.

Combines:
- Context routing (which neurons, what weights)
- Firing pattern combination (learned neuron specializations)
- Self-attention (information mixing)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .learned_neuron_pool import LearnedNeuronPool, ContextualNeuronPool
from .context_router import ContextRouter


class ContextualSPROUTLayer(nn.Module):
    """
    SPROUT layer with learned firing patterns.

    Flow:
    1. Context Router → select k neurons + compute pattern weights
    2. Neuron Pool → combine firing patterns → GELU → output
    3. Self-Attention → mix information
    4. Residual connections
    """

    def __init__(
        self,
        d_model: int,
        pool_size: int,
        k: int,
        d_ff: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        router_temperature: float = 1.0
    ):
        """
        Initialize contextual SPROUT layer.

        Args:
            d_model: Model dimension
            pool_size: Number of neurons in pool
            k: Number of neurons to select per token
            d_ff: Firing pattern dimension
            n_heads: Number of attention heads (default: 4)
            dropout: Dropout probability (default: 0.1)
            router_temperature: Gumbel-Softmax temperature (default: 1.0)
        """
        super().__init__()
        self.d_model = d_model
        self.pool_size = pool_size
        self.k = k
        self.d_ff = d_ff

        # Context router: selects neurons and computes pattern weights
        self.router = ContextRouter(
            d_model=d_model,
            pool_size=pool_size,
            k=k,
            temperature=router_temperature
        )

        # Neuron pool: stores learned firing patterns
        self.neuron_pool = LearnedNeuronPool(
            pool_size=pool_size,
            d_model=d_model,
            d_ff=d_ff
        )

        # Self-attention
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_routing: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through contextual SPROUT layer.

        Args:
            x: Input tensor [batch, seq, d_model]
            return_routing: If True, return routing information

        Returns:
            output: Processed tensor [batch, seq, d_model]
            routing_info: Optional routing information
        """
        # 1. Context routing: select neurons and get pattern weights
        selected_indices, pattern_weights, selection_scores = self.router(x)

        # 2. Combine firing patterns and produce output
        neuron_output = self.neuron_pool(selected_indices, pattern_weights)

        # 3. Self-attention
        residual = x
        attn_out, attn_weights = self.attention(neuron_output, neuron_output, neuron_output)
        attn_out = self.dropout(attn_out)
        x = self.norm1(residual + attn_out)

        # 4. Add neuron output with residual
        x = self.norm2(x + neuron_output)

        # Return routing info if requested
        routing_info = None
        if return_routing:
            routing_info = {
                'selected_indices': selected_indices,
                'pattern_weights': pattern_weights,
                'selection_scores': selection_scores,
                'attn_weights': attn_weights,
                'selection_entropy': self.router.get_selection_entropy(selection_scores),
            }

            # Weight distribution stats
            mean_max_weight, weight_entropy = self.router.get_weight_distribution_stats(pattern_weights)
            routing_info['mean_max_weight'] = mean_max_weight
            routing_info['weight_entropy'] = weight_entropy

        return x, routing_info


class AdvancedContextualSPROUTLayer(nn.Module):
    """
    Advanced SPROUT layer with context-modulated firing patterns.

    Uses ContextualNeuronPool for pattern modulation based on input.
    """

    def __init__(
        self,
        d_model: int,
        pool_size: int,
        k: int,
        d_ff: int,
        modulation_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        router_temperature: float = 1.0
    ):
        """
        Initialize advanced contextual SPROUT layer.

        Args:
            d_model: Model dimension
            pool_size: Number of neurons
            k: Neurons to select per token
            d_ff: Firing pattern dimension
            modulation_dim: Context modulation dimension (default: 64)
            n_heads: Attention heads (default: 4)
            dropout: Dropout probability (default: 0.1)
            router_temperature: Gumbel-Softmax temperature (default: 1.0)
        """
        super().__init__()
        self.d_model = d_model
        self.pool_size = pool_size
        self.k = k
        self.d_ff = d_ff

        # Context router
        self.router = ContextRouter(
            d_model=d_model,
            pool_size=pool_size,
            k=k,
            temperature=router_temperature
        )

        # Contextual neuron pool (with pattern modulation)
        self.neuron_pool = ContextualNeuronPool(
            pool_size=pool_size,
            d_model=d_model,
            d_ff=d_ff,
            modulation_dim=modulation_dim
        )

        # Self-attention
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_routing: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with context-modulated patterns.

        Args:
            x: Input tensor [batch, seq, d_model]
            return_routing: If True, return routing information

        Returns:
            output: Processed tensor [batch, seq, d_model]
            routing_info: Optional routing information
        """
        # 1. Context routing
        selected_indices, pattern_weights, selection_scores = self.router(x)

        # 2. Combine context-modulated firing patterns
        neuron_output = self.neuron_pool(x, selected_indices, pattern_weights)

        # 3. Self-attention
        residual = x
        attn_out, attn_weights = self.attention(neuron_output, neuron_output, neuron_output)
        attn_out = self.dropout(attn_out)
        x = self.norm1(residual + attn_out)

        # 4. Add neuron output with residual
        x = self.norm2(x + neuron_output)

        # Return routing info if requested
        routing_info = None
        if return_routing:
            routing_info = {
                'selected_indices': selected_indices,
                'pattern_weights': pattern_weights,
                'selection_scores': selection_scores,
                'attn_weights': attn_weights,
                'selection_entropy': self.router.get_selection_entropy(selection_scores),
            }

            # Weight distribution stats
            mean_max_weight, weight_entropy = self.router.get_weight_distribution_stats(pattern_weights)
            routing_info['mean_max_weight'] = mean_max_weight
            routing_info['weight_entropy'] = weight_entropy

        return x, routing_info
