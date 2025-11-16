"""
SPROUT Layer - One processing step in SPROUT.

Combines routing + neuron selection + self-attention.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from .neuron_pool import NeuronPool
from .hard_router import HardRouter


class SPROUTLayer(nn.Module):
    """
    SPROUT processing layer.

    Flow:
    1. Router selects k neurons for each token
    2. Selected neurons process the token
    3. Self-attention among neuron outputs
    4. Residual connections and normalization
    """

    def __init__(
        self,
        d_model: int,
        pool_size: int,
        k: int,
        n_heads: int,
        dropout: float = 0.1,
        router_temperature: float = 1.0
    ):
        """
        Initialize SPROUT layer.

        Args:
            d_model: Model dimension
            pool_size: Number of neurons in pool
            k: Number of neurons to select per token
            n_heads: Number of attention heads
            dropout: Dropout probability
            router_temperature: Temperature for Gumbel-Softmax
        """
        super().__init__()
        self.d_model = d_model
        self.pool_size = pool_size
        self.k = k

        # Router: selects k neurons for each token
        self.router = HardRouter(d_model, pool_size, k, router_temperature)

        # Neuron pool: shared pool of neurons
        self.neuron_pool = NeuronPool(pool_size, d_model, d_model * 4)

        # Self-attention among selected neurons
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
        Forward pass through SPROUT layer.

        Args:
            x: Input tensor [batch, seq, d_model]
            return_routing: If True, return routing information

        Returns:
            output: Processed tensor [batch, seq, d_model]
            routing_info: Optional routing information
        """
        # 1. Router selects k neurons for each token
        indices, routing_scores = self.router(x)  # indices: [batch, seq, k]

        # 2. Process with selected neurons
        neuron_outputs = self.neuron_pool(x, indices)  # [batch, seq, k, d_model]

        # Average over selected neurons
        neuron_outputs = neuron_outputs.mean(dim=2)  # [batch, seq, d_model]

        # 3. Self-attention among neuron outputs
        residual = x
        attn_out, attn_weights = self.attention(
            neuron_outputs,
            neuron_outputs,
            neuron_outputs
        )
        attn_out = self.dropout(attn_out)
        x = self.norm1(residual + attn_out)

        # 4. Add neuron outputs with residual
        x = self.norm2(x + neuron_outputs)

        # Return routing info if requested
        routing_info = None
        if return_routing:
            routing_info = {
                'indices': indices,
                'scores': routing_scores,
                'attn_weights': attn_weights,
                'entropy': self.router.get_routing_entropy(routing_scores),
                'load_balance_loss': self.router.get_load_balance_loss(indices)
            }

        return x, routing_info
