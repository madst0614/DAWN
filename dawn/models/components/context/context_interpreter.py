"""
Context Interpreter - Generic Context Interpretation

Interprets other representations in current context through:
1. Projection (other space → intermediate space)
2. Cross-Attention (current context as query)
3. Post-processing (refinement)

Used by:
- PeerContext: Interpret peer expert outputs
- ExpertContext: Interpret expert outputs
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class ContextInterpreter(nn.Module):
    """
    Generic context interpreter

    Transforms h_other into h_self's coordinate system.

    Args:
        hidden_size: int
        num_heads: int - for cross-attention
        projection_rank: int - intermediate projection dimension
        config: Dict with dropout, init_std
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        projection_rank: int,
        config: Dict,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Projection: other space → intermediate space
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_rank),
            nn.LayerNorm(projection_rank),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(projection_rank, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Cross-Attention: self as query, other as key/value
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=config.get('dropout', 0.1),
            batch_first=True,
        )

        # Post-processing
        self.post_process = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self._init_weights(config)

    def _init_weights(self, config: Dict):
        """Initialize weights carefully for gradient flow"""
        init_std = config.get('init_std', 0.02)

        # Projection: first layer normal init, output layer zero init
        nn.init.normal_(self.projection[0].weight, mean=0.0, std=init_std)  # Input layer
        nn.init.zeros_(self.projection[0].bias)
        nn.init.zeros_(self.projection[4].weight)  # Output layer (residual)
        nn.init.zeros_(self.projection[4].bias)

        # Post-process: output layer zero init (residual)
        nn.init.zeros_(self.post_process[1].weight)
        nn.init.zeros_(self.post_process[1].bias)

    def forward(
        self,
        h_self: torch.Tensor,
        h_other: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Interpret h_other in h_self's context

        Args:
            h_self: [B, L, D] - current context (query)
            h_other: [B, L, D] - representation to interpret
            attention_mask: [B, L] (True = masked)

        Returns:
            h_interpreted: [B, L, D]
        """
        # Project other to intermediate space
        h_proj = self.projection(h_other)

        # Attend based on self's context
        h_attended, _ = self.cross_attention(
            query=h_self,
            key=h_proj,
            value=h_proj,
            key_padding_mask=attention_mask,
        )

        # Post-process
        h_interpreted = self.post_process(h_attended)

        return h_interpreted
