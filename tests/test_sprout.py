"""Tests for SPROUT model."""

import torch
import pytest
from sprout import SPROUT, Node, Router


class TestRouter:
    """Test Router component."""

    def test_router_init(self):
        """Test router initialization."""
        router = Router(dim=512, num_heads=4)
        assert router.dim == 512
        assert router.num_heads == 4

    def test_compatibility_computation(self):
        """Test compatibility computation."""
        router = Router(dim=512)
        x = torch.randn(2, 10, 512)
        node_key = torch.randn(1, 1, 512)

        compat = router.compute_compatibility(x, node_key)

        assert compat.shape == (2, 10)
        assert torch.all(compat >= 0) and torch.all(compat <= 1)


class TestNode:
    """Test Node component."""

    def test_node_init(self):
        """Test node initialization."""
        node = Node(dim=512, node_id=0, depth=0, max_depth=5)
        assert node.dim == 512
        assert node.node_id == 0
        assert node.depth == 0
        assert node.max_depth == 5
        assert len(node.child_nodes) == 0

    def test_node_forward_basic(self):
        """Test basic node forward pass."""
        node = Node(dim=512, node_id=0, depth=0, max_depth=5)
        x = torch.randn(2, 10, 512)

        output, path_log = node(x)

        assert output.shape == (2, 10, 512)
        assert len(path_log) > 0
        assert path_log[0]['node_id'] == 0

    def test_node_creates_children(self):
        """Test that node creates children."""
        node = Node(dim=512, node_id=0, depth=0, max_depth=5)
        x = torch.randn(2, 10, 512)

        # First forward should create first child
        output, path_log = node(x)

        assert len(node.child_nodes) == 1
        assert any(log['action'] == 'created_first_child' for log in path_log)

    def test_node_max_depth(self):
        """Test that node respects max depth."""
        node = Node(dim=512, node_id=0, depth=5, max_depth=5)
        x = torch.randn(2, 10, 512)

        output, path_log = node(x)

        # Should not create children at max depth
        assert len(node.child_nodes) == 0
        assert len(path_log) == 0

    def test_node_count(self):
        """Test node counting."""
        node = Node(dim=512, node_id=0, depth=0, max_depth=3)
        x = torch.randn(2, 10, 512)

        # Process multiple inputs to create structure
        for _ in range(5):
            node(x)

        count = node.count_nodes()
        assert count >= 1  # At least the root

    def test_node_structure_info(self):
        """Test structure info retrieval."""
        node = Node(dim=512, node_id=0, depth=0, max_depth=3)
        x = torch.randn(2, 10, 512)

        node(x)
        info = node.get_structure_info()

        assert 'node_id' in info
        assert 'depth' in info
        assert 'num_children' in info
        assert 'usage_count' in info
        assert 'children' in info


class TestSPROUT:
    """Test SPROUT model."""

    def test_sprout_init(self):
        """Test SPROUT initialization."""
        model = SPROUT(
            dim=512,
            max_depth=5,
            compatibility_threshold=0.8
        )
        assert model.dim == 512
        assert model.max_depth == 5
        assert model.compatibility_threshold == 0.8
        assert model.root.node_id == 0

    def test_sprout_forward(self):
        """Test SPROUT forward pass."""
        model = SPROUT(dim=512, max_depth=5)
        x = torch.randn(2, 10, 512)

        output, path_log = model(x)

        assert output.shape == (2, 10, 512)
        assert len(path_log) > 0

    def test_sprout_with_context(self):
        """Test SPROUT with context bias."""
        model = SPROUT(dim=512, max_depth=5)
        x = torch.randn(2, 10, 512)
        context = torch.randn(2, 10, 512)

        output, path_log = model(x, context_bias=context)

        assert output.shape == (2, 10, 512)
        assert len(path_log) > 0

    def test_sprout_branching(self):
        """Test that SPROUT creates branches."""
        model = SPROUT(dim=512, max_depth=5, compatibility_threshold=0.99)
        x = torch.randn(2, 10, 512)

        # Multiple forward passes should create branches
        for _ in range(10):
            output, path_log = model(x)

        num_nodes = model.count_total_nodes()
        assert num_nodes > 1  # Should have created some branches

    def test_sprout_convergence_tracking(self):
        """Test convergence tracking."""
        model = SPROUT(dim=512, max_depth=5)
        x = torch.randn(2, 10, 512)

        # Process some inputs
        for _ in range(50):
            model(x)

        # Should have branch history
        assert len(model.branch_history) == 50

        # Convergence check should work
        is_conv = model.is_converged(window=10, threshold=0.5)
        assert isinstance(is_conv, bool)

    def test_sprout_statistics(self):
        """Test statistics gathering."""
        model = SPROUT(dim=512, max_depth=5)
        x = torch.randn(2, 10, 512)

        # Process some inputs
        for _ in range(10):
            model(x)

        stats = model.get_statistics()

        assert 'total_nodes' in stats
        assert 'nodes_by_depth' in stats
        assert 'total_usage' in stats
        assert 'max_children' in stats
        assert stats['total_nodes'] >= 1

    def test_sprout_structure(self):
        """Test structure retrieval."""
        model = SPROUT(dim=512, max_depth=5)
        x = torch.randn(2, 10, 512)

        model(x)
        structure = model.get_structure()

        assert 'node_id' in structure
        assert 'depth' in structure
        assert structure['depth'] == 0  # Root is at depth 0

    def test_sprout_visualization(self, capsys):
        """Test structure visualization."""
        model = SPROUT(dim=512, max_depth=5)
        x = torch.randn(2, 10, 512)

        model(x)
        model.visualize_structure(max_depth=2)

        captured = capsys.readouterr()
        assert "SPROUT Structure" in captured.out
        assert "Node 0" in captured.out
        assert "Total nodes:" in captured.out

    def test_different_compatibility_thresholds(self):
        """Test different compatibility thresholds."""
        high_threshold = SPROUT(dim=512, max_depth=5, compatibility_threshold=0.99)
        low_threshold = SPROUT(dim=512, max_depth=5, compatibility_threshold=0.1)

        x = torch.randn(2, 10, 512)

        # Process same input
        for _ in range(10):
            high_threshold(x)
            low_threshold(x)

        # High threshold should create more branches
        high_nodes = high_threshold.count_total_nodes()
        low_nodes = low_threshold.count_total_nodes()

        assert high_nodes >= low_nodes

    def test_batch_processing(self):
        """Test different batch sizes."""
        model = SPROUT(dim=512, max_depth=5)

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 10, 512)
            output, path_log = model(x)

            assert output.shape == (batch_size, 10, 512)

    def test_sequence_lengths(self):
        """Test different sequence lengths."""
        model = SPROUT(dim=512, max_depth=5)

        for seq_len in [1, 5, 10, 20]:
            x = torch.randn(2, seq_len, 512)
            output, path_log = model(x)

            assert output.shape == (2, seq_len, 512)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
