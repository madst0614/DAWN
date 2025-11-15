"""
Router module for SPROUT.

Routes inputs to child nodes based on attention-gated compatibility.
"""

import torch
import torch.nn as nn


class Router(nn.Module):
    """Routes input to child nodes based on compatibility"""

    def __init__(self, dim: int, num_heads: int = 4):
        """
        Initialize router.

        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.gate = nn.Linear(dim, 1)

    def compute_compatibility(
        self,
        x: torch.Tensor,
        node_key: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute how compatible input is with a node.

        Args:
            x: Input tensor [batch, seq, dim]
            node_key: Node's key representation [1, 1, dim]

        Returns:
            Compatibility score [batch, seq]
        """
        # Expand node_key to match batch size
        batch_size = x.size(0)
        node_key_expanded = node_key.expand(batch_size, -1, -1)

        # Attention between input and node representation
        attn_out, _ = self.attention(x, node_key_expanded, node_key_expanded)

        # Gate for compatibility score
        compat = torch.sigmoid(self.gate(attn_out))
        return compat.squeeze(-1)
