"""
Test that the tensor broadcasting optimization produces identical results
to the original nested for-loop implementation.
"""

import torch
import numpy as np


def count_neuron_tokens_original(top_indices, input_ids, valid_mask, n_input, vocab_size):
    """Original nested for-loop implementation"""
    B, k_eff = top_indices.shape
    S = input_ids.shape[1]
    device = input_ids.device

    neuron_token_counts = torch.zeros(n_input, vocab_size, dtype=torch.float32, device=device)

    for b in range(B):
        batch_neurons = top_indices[b]  # [k_eff]
        batch_valid_mask = valid_mask[b]  # [S]
        batch_valid_tokens = input_ids[b][batch_valid_mask]  # [num_valid]

        for neuron_idx in batch_neurons:
            neuron_token_counts[neuron_idx].scatter_add_(
                0, batch_valid_tokens,
                torch.ones_like(batch_valid_tokens, dtype=torch.float32)
            )

    return neuron_token_counts


def count_neuron_tokens_optimized(top_indices, input_ids, valid_mask, n_input, vocab_size):
    """Optimized tensor broadcasting implementation"""
    B, k_eff = top_indices.shape
    S = input_ids.shape[1]
    device = input_ids.device

    neuron_token_counts = torch.zeros(n_input, vocab_size, dtype=torch.float32, device=device)

    # Expand to [B, k_eff, S] for all combinations
    neuron_indices = top_indices.unsqueeze(2).expand(B, k_eff, S)  # [B, k_eff, S]
    token_indices = input_ids.unsqueeze(1).expand(B, k_eff, S)     # [B, k_eff, S]
    valid_expanded = valid_mask.unsqueeze(1).expand(B, k_eff, S)   # [B, k_eff, S]

    # Filter valid tokens only
    valid_neurons = neuron_indices[valid_expanded]  # [total_valid]
    valid_tokens = token_indices[valid_expanded]    # [total_valid]

    # 2D index (neuron, token) → 1D: neuron * vocab_size + token
    flat_indices = valid_neurons * vocab_size + valid_tokens  # [total_valid]

    # Single scatter_add on flattened view!
    flat_counts = neuron_token_counts.view(-1)  # [n_input * vocab_size]
    flat_counts.scatter_add_(
        0, flat_indices,
        torch.ones_like(flat_indices, dtype=torch.float32)
    )

    return neuron_token_counts


def test_optimization():
    """Test that both implementations produce identical results"""
    print("Testing tensor optimization...")

    # Test parameters
    B = 16  # Batch size
    S = 32  # Sequence length
    k_eff = 10  # Top-k neurons
    n_input = 128  # Total input neurons
    vocab_size = 1000  # Vocab size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate random test data
    torch.manual_seed(42)
    top_indices = torch.randint(0, n_input, (B, k_eff), device=device)
    input_ids = torch.randint(0, vocab_size, (B, S), device=device)

    # Create valid mask (exclude tokens 0, 1, 2 as special tokens)
    valid_mask = (input_ids >= 3)

    print(f"\nTest data:")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {S}")
    print(f"  Top-k neurons: {k_eff}")
    print(f"  Input neurons: {n_input}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Valid tokens: {valid_mask.sum().item()} / {B * S}")

    # Test original implementation
    print("\nRunning original nested for-loop implementation...")
    import time
    start = time.time()
    counts_original = count_neuron_tokens_original(
        top_indices, input_ids, valid_mask, n_input, vocab_size
    )
    time_original = time.time() - start
    print(f"  Time: {time_original*1000:.2f}ms")

    # Test optimized implementation
    print("\nRunning optimized tensor broadcasting implementation...")
    start = time.time()
    counts_optimized = count_neuron_tokens_optimized(
        top_indices, input_ids, valid_mask, n_input, vocab_size
    )
    time_optimized = time.time() - start
    print(f"  Time: {time_optimized*1000:.2f}ms")

    # Compare results
    print("\nComparing results...")
    diff = (counts_original - counts_optimized).abs()
    max_diff = diff.max().item()

    print(f"  Max absolute difference: {max_diff}")
    print(f"  Total counts (original): {counts_original.sum().item()}")
    print(f"  Total counts (optimized): {counts_optimized.sum().item()}")

    if max_diff < 1e-6:
        print("\n✅ TEST PASSED! Results are identical.")
    else:
        print(f"\n❌ TEST FAILED! Max difference: {max_diff}")
        return False

    # Speedup analysis
    speedup = time_original / time_optimized
    print(f"\n🚀 Performance:")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Original: {time_original*1000:.2f}ms")
    print(f"  Optimized: {time_optimized*1000:.2f}ms")

    return True


def test_large_batch():
    """Test with realistic batch size (128)"""
    print("\n" + "="*60)
    print("Testing with realistic batch size (B=128, k_eff=70)...")
    print("="*60)

    B = 128
    S = 128
    k_eff = 70
    n_input = 128
    vocab_size = 30522  # BERT vocab size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(42)
    top_indices = torch.randint(0, n_input, (B, k_eff), device=device)
    input_ids = torch.randint(0, vocab_size, (B, S), device=device)
    valid_mask = (input_ids >= 3) & (input_ids != 101) & (input_ids != 102)

    print(f"  Device: {device}")
    print(f"  Expected kernel calls (original): {B * k_eff} = {B} × {k_eff}")

    # Original
    import time
    start = time.time()
    counts_original = count_neuron_tokens_original(
        top_indices, input_ids, valid_mask, n_input, vocab_size
    )
    time_original = time.time() - start

    # Optimized
    start = time.time()
    counts_optimized = count_neuron_tokens_optimized(
        top_indices, input_ids, valid_mask, n_input, vocab_size
    )
    time_optimized = time.time() - start

    # Verify
    diff = (counts_original - counts_optimized).abs().max().item()

    speedup = time_original / time_optimized
    print(f"\n  Results match: {diff < 1e-6} (max diff: {diff})")
    print(f"  Original time: {time_original*1000:.2f}ms")
    print(f"  Optimized time: {time_optimized*1000:.2f}ms")
    print(f"  🚀 Speedup: {speedup:.1f}x")

    return diff < 1e-6


if __name__ == '__main__':
    print("="*60)
    print("Tensor Broadcasting Optimization Test")
    print("="*60)

    # Run tests
    test1 = test_optimization()
    test2 = test_large_batch()

    print("\n" + "="*60)
    if test1 and test2:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*60)
