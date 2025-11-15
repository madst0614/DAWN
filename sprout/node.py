"""
Node module for SPROUT.

Implements the recursive node structure with Attention → FFN → Router flow.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

from .router import Router


class Node(nn.Module):
    """
    Recursive node structure:
    Input → Attention (gather info) → FFN (process) → Router (route to children)
    """

    def __init__(
        self,
        dim: int,
        node_id: int,
        depth: int = 0,
        max_depth: int = 5,
        num_heads: int = 4,
        ffn_mult: int = 4
    ):
        """
        Initialize node.

        Args:
            dim: Hidden dimension
            node_id: Unique identifier for this node
            depth: Current depth in tree
            max_depth: Maximum allowed depth
            num_heads: Number of attention heads
            ffn_mult: FFN expansion multiplier
        """
        super().__init__()
        self.dim = dim
        self.node_id = node_id
        self.depth = depth
        self.max_depth = max_depth

        # Processing components
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Routing component
        self.router = Router(dim, num_heads=num_heads)

        # Node representation (learnable "key" for this node)
        self.node_key = nn.Parameter(torch.randn(1, 1, dim))

        # Children tracking (use ModuleList for proper parameter registration)
        self.child_nodes = nn.ModuleList()
        self.next_child_id = 0

        # Statistics
        self.usage_count = 0

    def forward(
        self,
        x: torch.Tensor,
        context_bias: Optional[torch.Tensor] = None,
        compatibility_threshold: float = 0.8
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Forward pass through node.

        Args:
            x: Input tensor [batch, seq, dim]
            context_bias: Optional peer context
            compatibility_threshold: Threshold for creating new branches

        Returns:
            output: Processed representation
            path_log: Routing decisions and compatibility scores
        """
        self.usage_count += 1
        path_log = []

        # 1. Attention: gather relevant information
        if context_bias is not None:
            x_with_context = x + context_bias
        else:
            x_with_context = x

        attn_out, attn_weights = self.attention(
            x_with_context,
            x_with_context,
            x_with_context
        )
        x = self.norm1(x + attn_out)

        # 2. FFN: process at this level
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # 3. Router: decide next step
        if self.depth >= self.max_depth:
            # Max depth reached, return
            return x, path_log

        if len(self.child_nodes) == 0:
            # No children yet, create first child
            new_child = self._create_child()
            child_output, child_log = new_child(
                x,
                context_bias,
                compatibility_threshold
            )
            path_log.append({
                'node_id': self.node_id,
                'depth': self.depth,
                'action': 'created_first_child',
                'child_id': new_child.node_id,
                'compatibility': 1.0
            })
            path_log.extend(child_log)
            return child_output, path_log

        # Check compatibility with existing children
        child_compatibilities = []
        for child in self.child_nodes:
            compat = self.router.compute_compatibility(x, child.node_key)
            child_compatibilities.append(compat.mean().item())

        best_idx = max(
            range(len(child_compatibilities)),
            key=lambda i: child_compatibilities[i]
        )
        best_compat = child_compatibilities[best_idx]

        if best_compat < compatibility_threshold:
            # Create new branch
            new_child = self._create_child()
            child_output, child_log = new_child(
                x,
                context_bias,
                compatibility_threshold
            )

            path_log.append({
                'node_id': self.node_id,
                'depth': self.depth,
                'action': 'branched',
                'new_child_id': new_child.node_id,
                'best_existing_compat': best_compat,
                'threshold': compatibility_threshold
            })
            path_log.extend(child_log)
        else:
            # Route to best existing child
            best_child = self.child_nodes[best_idx]

            # Soft gating based on compatibility
            gate_strength = child_compatibilities[best_idx]
            child_output, child_log = best_child(
                x,
                context_bias,
                compatibility_threshold
            )
            child_output = gate_strength * child_output + (1 - gate_strength) * x

            path_log.append({
                'node_id': self.node_id,
                'depth': self.depth,
                'action': 'routed',
                'child_id': best_child.node_id,
                'compatibility': best_compat
            })
            path_log.extend(child_log)

        return child_output, path_log

    def _create_child(self) -> 'Node':
        """Create a new child node"""
        child = Node(
            dim=self.dim,
            node_id=self.next_child_id,
            depth=self.depth + 1,
            max_depth=self.max_depth
        )
        # Move child to same device as parent
        child = child.to(self.node_key.device)
        self.child_nodes.append(child)
        self.next_child_id += 1
        return child

    def count_nodes(self) -> int:
        """Recursively count total nodes in subtree"""
        count = 1  # self
        for child in self.child_nodes:
            count += child.count_nodes()
        return count

    def get_structure_info(self) -> Dict:
        """Get structure information for visualization"""
        return {
            'node_id': self.node_id,
            'depth': self.depth,
            'num_children': len(self.child_nodes),
            'usage_count': self.usage_count,
            'children': [child.get_structure_info() for child in self.child_nodes]
        }
