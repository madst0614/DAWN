"""
Convergence tracking example for SPROUT.

Demonstrates:
- Monitoring structure growth
- Convergence detection
- Branch rate analysis
"""

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sprout import SPROUT


def main():
    print("=" * 60)
    print("SPROUT - Convergence Tracking Example")
    print("=" * 60)

    # Initialize model with high threshold to encourage branching
    print("\n1. Initializing SPROUT with high compatibility threshold...")
    model = SPROUT(
        dim=256,
        max_depth=4,
        compatibility_threshold=0.95,  # High threshold = more branching
        num_heads=4
    )

    # Track metrics over time
    num_iterations = 500
    node_counts = []
    branch_counts = []
    convergence_status = []

    print(f"\n2. Training for {num_iterations} iterations...")

    for i in range(num_iterations):
        # Generate random input
        x = torch.randn(2, 8, model.dim)

        # Forward pass
        output, path_log = model(x)

        # Track metrics
        num_nodes = model.count_total_nodes()
        num_branches = sum(1 for log in path_log if log['action'] == 'branched')
        is_conv = model.is_converged(window=50, threshold=0.01)

        node_counts.append(num_nodes)
        branch_counts.append(num_branches)
        convergence_status.append(is_conv)

        # Print progress
        if (i + 1) % 100 == 0:
            print(f"   Iteration {i+1}/{num_iterations}")
            print(f"   - Nodes: {num_nodes}")
            print(f"   - Branches this iter: {num_branches}")
            print(f"   - Converged: {is_conv}")

    # Final statistics
    print("\n3. Final statistics:")
    final_stats = model.get_statistics()
    print(f"   - Total nodes: {final_stats['total_nodes']}")
    print(f"   - Nodes by depth: {final_stats['nodes_by_depth']}")
    print(f"   - Recent branch rate: {final_stats.get('recent_branch_rate', 0):.4f}")

    # Visualize final structure
    print("\n4. Final structure:")
    model.visualize_structure(max_depth=3)

    # Plot convergence
    print("\n5. Plotting convergence metrics...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Plot 1: Node count over time
    axes[0].plot(node_counts, linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Total Nodes')
    axes[0].set_title('Node Count Growth')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Branches per iteration
    axes[1].plot(branch_counts, linewidth=1, alpha=0.7)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('New Branches')
    axes[1].set_title('Branching Activity')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Convergence status
    axes[2].plot([1 if c else 0 for c in convergence_status], linewidth=2)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Converged (1=Yes, 0=No)')
    axes[2].set_title('Convergence Status')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence_tracking.png', dpi=150, bbox_inches='tight')
    print("   - Saved plot to: convergence_tracking.png")

    # Analyze convergence
    print("\n6. Convergence analysis:")
    if convergence_status[-1]:
        conv_point = next(i for i, c in enumerate(convergence_status) if c)
        print(f"   - Converged at iteration: {conv_point}")
    else:
        print("   - Structure has not fully converged")

    total_branches = sum(model.branch_history)
    print(f"   - Total branches created: {total_branches}")
    print(f"   - Average branches per iteration: {total_branches/num_iterations:.4f}")

    # Depth distribution
    print("\n7. Depth distribution:")
    for depth, count in sorted(final_stats['nodes_by_depth'].items()):
        percentage = (count / final_stats['total_nodes']) * 100
        print(f"   - Depth {depth}: {count} nodes ({percentage:.1f}%)")

    print("\n" + "=" * 60)
    print("Convergence tracking completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
