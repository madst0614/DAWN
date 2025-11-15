"""
Basic usage example for SPROUT.

Demonstrates:
- Model initialization
- Forward pass
- Structure visualization
- Statistics gathering
"""

import torch
from sprout import SPROUT


def main():
    print("=" * 60)
    print("SPROUT - Basic Usage Example")
    print("=" * 60)

    # Initialize model
    print("\n1. Initializing SPROUT model...")
    model = SPROUT(
        dim=512,
        max_depth=5,
        compatibility_threshold=0.8,
        num_heads=4,
        ffn_mult=4
    )
    print(f"   - Dimension: {model.dim}")
    print(f"   - Max depth: {model.max_depth}")
    print(f"   - Compatibility threshold: {model.compatibility_threshold}")

    # Create input
    print("\n2. Creating input tensor...")
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, model.dim)
    print(f"   - Shape: {x.shape} [batch, seq, dim]")

    # Forward pass
    print("\n3. Running forward pass...")
    output, path_log = model(x)
    print(f"   - Output shape: {output.shape}")
    print(f"   - Path log entries: {len(path_log)}")

    # Show routing decisions
    print("\n4. Routing decisions:")
    for i, log in enumerate(path_log[:5]):  # Show first 5
        print(f"   [{i}] Node {log['node_id']} (depth {log['depth']}): {log['action']}")
        if 'compatibility' in log:
            print(f"       Compatibility: {log['compatibility']:.4f}")

    # Multiple forward passes to grow structure
    print("\n5. Growing structure with multiple passes...")
    for i in range(20):
        x_new = torch.randn(batch_size, seq_len, model.dim)
        model(x_new)

    print(f"   - Processed 20 additional inputs")

    # Visualize structure
    print("\n6. Structure visualization:")
    model.visualize_structure(max_depth=3)

    # Get statistics
    print("7. Model statistics:")
    stats = model.get_statistics()
    print(f"   - Total nodes: {stats['total_nodes']}")
    print(f"   - Nodes by depth: {stats['nodes_by_depth']}")
    print(f"   - Total usage: {stats['total_usage']}")
    print(f"   - Max children per node: {stats['max_children']}")
    if 'recent_branch_rate' in stats:
        print(f"   - Recent branch rate: {stats['recent_branch_rate']:.4f}")

    # Check convergence
    print("\n8. Convergence status:")
    is_converged = model.is_converged(window=20, threshold=0.1)
    print(f"   - Converged: {is_converged}")
    print(f"   - Branch history length: {len(model.branch_history)}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
