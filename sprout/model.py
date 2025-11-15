"""
SPROUT main model implementation.

Self-organizing Progressive Routing with Organic Unified Trees
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

from .node import Node


class SPROUT(nn.Module):
    """
    SPROUT: Self-organizing Progressive Routing with Organic Unified Trees

    Main model with single root entry point that dynamically grows
    tree-like structures based on input compatibility.
    """

    def __init__(
        self,
        dim: int = 512,
        max_depth: int = 5,
        compatibility_threshold: float = 0.8,
        num_heads: int = 4,
        ffn_mult: int = 4
    ):
        """
        Initialize SPROUT model.

        Args:
            dim: Hidden dimension
            max_depth: Maximum tree depth
            compatibility_threshold: Threshold for creating new branches
            num_heads: Number of attention heads
            ffn_mult: FFN expansion multiplier
        """
        super().__init__()
        self.dim = dim
        self.max_depth = max_depth
        self.compatibility_threshold = compatibility_threshold
        self.num_heads = num_heads
        self.ffn_mult = ffn_mult

        # Single root entry point
        self.root = Node(
            dim=dim,
            node_id=0,
            depth=0,
            max_depth=max_depth,
            num_heads=num_heads,
            ffn_mult=ffn_mult
        )

        # Track convergence
        self.branch_history: List[int] = []

    def forward(
        self,
        x: torch.Tensor,
        context_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Forward pass through SPROUT.

        Args:
            x: Input tensor [batch, seq, dim]
            context_bias: Optional context from peers

        Returns:
            output: Final representation
            path_log: Complete routing path with decisions
        """
        output, path_log = self.root(x, context_bias, self.compatibility_threshold)

        # Track branching for convergence detection
        num_branches = sum(1 for log in path_log if log['action'] == 'branched')
        self.branch_history.append(num_branches)

        return output, path_log

    def is_converged(self, window: int = 100, threshold: float = 0.001) -> bool:
        """
        Check if structure has converged (few new branches).

        Args:
            window: Number of recent iterations to check
            threshold: Maximum acceptable branch rate

        Returns:
            True if converged
        """
        if len(self.branch_history) < window:
            return False

        recent = self.branch_history[-window:]
        branch_rate = sum(recent) / window
        return branch_rate < threshold

    def count_total_nodes(self) -> int:
        """Count total nodes in entire structure"""
        return self.root.count_nodes()

    def get_structure(self) -> Dict:
        """Get complete structure information"""
        return self.root.get_structure_info()

    def visualize_structure(self, max_depth: int = 3):
        """
        Print structure tree (simplified).

        Args:
            max_depth: Maximum depth to visualize
        """
        def print_node(node: Node, prefix: str = "", is_last: bool = True):
            if node.depth > max_depth:
                return

            connector = "└── " if is_last else "├── "
            print(
                f"{prefix}{connector}Node {node.node_id} "
                f"(depth={node.depth}, children={len(node.child_nodes)}, "
                f"usage={node.usage_count})"
            )

            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.child_nodes):
                print_node(child, new_prefix, i == len(node.child_nodes) - 1)

        print("\n=== SPROUT Structure ===")
        print_node(self.root)
        print(f"Total nodes: {self.count_total_nodes()}\n")

    def get_statistics(self) -> Dict:
        """
        Get comprehensive model statistics.

        Returns:
            Dictionary with statistics about the model
        """
        structure = self.get_structure()

        def collect_stats(node_info: Dict, stats: Dict):
            stats['total_nodes'] += 1
            stats['nodes_by_depth'][node_info['depth']] = \
                stats['nodes_by_depth'].get(node_info['depth'], 0) + 1
            stats['total_usage'] += node_info['usage_count']

            if node_info['num_children'] > stats['max_children']:
                stats['max_children'] = node_info['num_children']

            for child in node_info['children']:
                collect_stats(child, stats)

        stats = {
            'total_nodes': 0,
            'nodes_by_depth': {},
            'total_usage': 0,
            'max_children': 0,
            'total_branches': len([b for b in self.branch_history if b > 0]),
            'convergence_window': len(self.branch_history)
        }

        collect_stats(structure, stats)

        if self.branch_history:
            stats['recent_branch_rate'] = sum(self.branch_history[-100:]) / min(
                100,
                len(self.branch_history)
            )

        return stats
