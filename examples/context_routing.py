"""
Context-aware routing example for SPROUT.

Demonstrates:
- Using context bias for routing
- Comparing with and without context
- Analyzing routing differences
"""

import torch
from sprout import SPROUT


def main():
    print("=" * 60)
    print("SPROUT - Context-Aware Routing Example")
    print("=" * 60)

    # Initialize model
    print("\n1. Initializing SPROUT model...")
    model = SPROUT(
        dim=256,
        max_depth=4,
        compatibility_threshold=0.85
    )

    # Create input
    batch_size = 4
    seq_len = 12
    x = torch.randn(batch_size, seq_len, model.dim)
    print(f"   - Input shape: {x.shape}")

    # Forward pass without context
    print("\n2. Forward pass WITHOUT context bias...")
    output_no_context, path_no_context = model(x)
    print(f"   - Output shape: {output_no_context.shape}")
    print(f"   - Routing decisions: {len(path_no_context)}")

    # Show routing path
    print("\n   Routing path (first 5 steps):")
    for i, log in enumerate(path_no_context[:5]):
        print(f"   [{i}] Node {log['node_id']}: {log['action']}")

    # Create context bias (e.g., from peer information)
    context_bias = torch.randn(batch_size, seq_len, model.dim) * 0.1
    print(f"\n3. Creating context bias: {context_bias.shape}")

    # Forward pass with context
    print("\n4. Forward pass WITH context bias...")
    output_with_context, path_with_context = model(x, context_bias=context_bias)
    print(f"   - Output shape: {output_with_context.shape}")
    print(f"   - Routing decisions: {len(path_with_context)}")

    # Show routing path
    print("\n   Routing path (first 5 steps):")
    for i, log in enumerate(path_with_context[:5]):
        print(f"   [{i}] Node {log['node_id']}: {log['action']}")

    # Compare outputs
    print("\n5. Comparing outputs...")
    output_diff = torch.abs(output_with_context - output_no_context).mean()
    print(f"   - Mean absolute difference: {output_diff.item():.6f}")

    # Analyze routing differences
    print("\n6. Analyzing routing differences...")

    actions_no_ctx = [log['action'] for log in path_no_context]
    actions_with_ctx = [log['action'] for log in path_with_context]

    print(f"   Without context:")
    for action in ['created_first_child', 'branched', 'routed']:
        count = actions_no_ctx.count(action)
        print(f"   - {action}: {count}")

    print(f"\n   With context:")
    for action in ['created_first_child', 'branched', 'routed']:
        count = actions_with_ctx.count(action)
        print(f"   - {action}: {count}")

    # Multiple passes to grow structure
    print("\n7. Growing structure with varied contexts...")
    num_passes = 50

    for i in range(num_passes):
        x_new = torch.randn(batch_size, seq_len, model.dim)

        # Alternate between different context strategies
        if i % 3 == 0:
            # Strong context
            ctx = torch.randn(batch_size, seq_len, model.dim) * 0.3
        elif i % 3 == 1:
            # Weak context
            ctx = torch.randn(batch_size, seq_len, model.dim) * 0.05
        else:
            # No context
            ctx = None

        output, path_log = model(x_new, context_bias=ctx)

    print(f"   - Processed {num_passes} inputs with varied contexts")

    # Final structure
    print("\n8. Final structure after context-aware training:")
    model.visualize_structure(max_depth=3)

    # Statistics
    print("\n9. Final statistics:")
    stats = model.get_statistics()
    print(f"   - Total nodes: {stats['total_nodes']}")
    print(f"   - Nodes by depth: {stats['nodes_by_depth']}")
    print(f"   - Max children per node: {stats['max_children']}")

    # Demonstrate routing with different contexts
    print("\n10. Routing comparison with different contexts:")

    test_input = torch.randn(1, seq_len, model.dim)

    # Strong positive context
    strong_ctx = torch.ones(1, seq_len, model.dim) * 0.5
    _, path_strong = model(test_input, context_bias=strong_ctx)

    # Strong negative context
    negative_ctx = -torch.ones(1, seq_len, model.dim) * 0.5
    _, path_negative = model(test_input, context_bias=negative_ctx)

    # No context
    _, path_none = model(test_input, context_bias=None)

    print(f"   - Routing depth with strong positive context: {len(path_strong)}")
    print(f"   - Routing depth with strong negative context: {len(path_negative)}")
    print(f"   - Routing depth with no context: {len(path_none)}")

    # Show how context affects compatibility
    print("\n11. Context impact on compatibility scores:")
    for i, (log_strong, log_neg, log_none) in enumerate(
        zip(path_strong[:3], path_negative[:3], path_none[:3])
    ):
        print(f"\n   Step {i}:")
        if 'compatibility' in log_strong:
            print(f"   - Strong positive: {log_strong['compatibility']:.4f}")
        if 'compatibility' in log_neg:
            print(f"   - Strong negative: {log_neg['compatibility']:.4f}")
        if 'compatibility' in log_none:
            print(f"   - No context: {log_none['compatibility']:.4f}")

    print("\n" + "=" * 60)
    print("Context-aware routing example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
