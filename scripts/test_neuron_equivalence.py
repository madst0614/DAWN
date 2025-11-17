"""
Test equivalence between neuron-based FFN and standard FFN
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.neuron_based import (
    DynamicFFNLayer,
    NeuronBasedLanguageModel
)


class StandardFFN(nn.Module):
    """Standard dense FFN for comparison"""
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.W2(F.gelu(self.W1(x)))


def test_ffn_equivalence():
    """Test that DynamicFFNLayer with all neurons == StandardFFN"""
    print("="*70)
    print("TEST 1: FFN Equivalence (all neurons)")
    print("="*70)

    d_model = 512
    d_ff = 2048
    batch = 4
    seq = 16

    # Create models
    neuron_ffn = DynamicFFNLayer(d_model=d_model, d_ff=d_ff)
    standard_ffn = StandardFFN(d_model=d_model, d_ff=d_ff)

    # Copy weights from neuron-based to standard
    with torch.no_grad():
        W1 = torch.stack([n.W_in for n in neuron_ffn.middle_neurons])  # [d_ff, d_model]
        W2 = torch.stack([n.W for n in neuron_ffn.output_neurons])      # [d_model, d_ff]

        standard_ffn.W1.weight.copy_(W1)
        standard_ffn.W2.weight.copy_(W2)

    # Test input
    x = torch.randn(batch, seq, d_model)

    # Forward pass
    with torch.no_grad():
        out_neuron = neuron_ffn(x, top_k=None)  # All neurons
        out_standard = standard_ffn(x)

    # Compare
    diff = (out_neuron - out_standard).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape (neuron): {out_neuron.shape}")
    print(f"Output shape (standard): {out_standard.shape}")
    print(f"\nMax difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-5:
        print("\n✅ EQUIVALENCE TEST PASSED")
    else:
        print(f"\n❌ EQUIVALENCE TEST FAILED (max diff: {max_diff})")

    return max_diff < 1e-5


def test_sparse_approximation():
    """Test sparse neuron selection quality"""
    print("\n" + "="*70)
    print("TEST 2: Sparse Approximation Quality")
    print("="*70)

    d_model = 512
    d_ff = 2048
    batch = 4
    seq = 16

    neuron_ffn = DynamicFFNLayer(d_model=d_model, d_ff=d_ff)
    neuron_ffn.eval()

    x = torch.randn(batch, seq, d_model)

    # Dense output (ground truth)
    with torch.no_grad():
        out_dense = neuron_ffn(x, top_k=None)

    # Test different sparsity levels
    sparsity_levels = [
        (int(0.1 * d_ff), "10%"),
        (int(0.2 * d_ff), "20%"),
        (int(0.5 * d_ff), "50%"),
    ]

    print(f"\nDense output norm: {out_dense.norm().item():.4f}")
    print("\nSparsity Level | MSE Loss | Cosine Sim | % Approximation")
    print("-" * 70)

    results = []
    for top_k, label in sparsity_levels:
        with torch.no_grad():
            out_sparse = neuron_ffn(x, top_k=top_k)

        # Metrics
        mse = F.mse_loss(out_sparse, out_dense).item()

        # Cosine similarity (flatten)
        cos_sim = F.cosine_similarity(
            out_sparse.flatten().unsqueeze(0),
            out_dense.flatten().unsqueeze(0)
        ).item()

        # Approximation quality (% of norm preserved)
        approx_quality = (out_sparse.norm() / out_dense.norm()).item() * 100

        print(f"{label:14s} | {mse:8.6f} | {cos_sim:10.6f} | {approx_quality:14.2f}%")

        results.append({
            'sparsity': label,
            'top_k': top_k,
            'mse': mse,
            'cos_sim': cos_sim,
            'quality': approx_quality
        })

    # Check if 50% sparsity achieves good approximation
    best = results[-1]  # 50% sparsity
    if best['cos_sim'] > 0.95 and best['quality'] > 90:
        print("\n✅ SPARSE APPROXIMATION TEST PASSED")
        print(f"   50% sparsity achieves {best['quality']:.1f}% quality with {best['cos_sim']:.4f} similarity")
    else:
        print("\n⚠️  SPARSE APPROXIMATION QUALITY MODERATE")
        print(f"   50% sparsity: {best['quality']:.1f}% quality, {best['cos_sim']:.4f} similarity")

    return results


def test_router_selection():
    """Test router's neuron selection"""
    print("\n" + "="*70)
    print("TEST 3: Router Selection Pattern")
    print("="*70)

    d_model = 512
    d_ff = 2048
    batch = 8
    seq = 16
    top_k = 512  # 25% sparsity

    neuron_ffn = DynamicFFNLayer(d_model=d_model, d_ff=d_ff)
    neuron_ffn.eval()

    # Test different inputs
    x1 = torch.randn(batch, seq, d_model)
    x2 = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        # Get routing scores
        x1_flat = x1.view(-1, d_model)
        x2_flat = x2.view(-1, d_model)

        scores1 = x1_flat @ neuron_ffn.router.W_router.T  # [batch*seq, d_ff]
        scores2 = x2_flat @ neuron_ffn.router.W_router.T

        # Get top-k indices
        _, top_indices1 = torch.topk(scores1, top_k, dim=-1)
        _, top_indices2 = torch.topk(scores2, top_k, dim=-1)

        # Convert to binary masks
        mask1 = torch.zeros_like(scores1)
        mask2 = torch.zeros_like(scores2)
        mask1.scatter_(-1, top_indices1, 1.0)
        mask2.scatter_(-1, top_indices2, 1.0)

        # Neuron usage statistics
        usage1 = mask1.sum(dim=0)  # [d_ff]
        usage2 = mask2.sum(dim=0)

        n_used1 = (usage1 > 0).sum().item()
        n_used2 = (usage2 > 0).sum().item()

        # Overlap between two inputs
        overlap = (mask1 * mask2).sum() / (batch * seq * top_k)

    print(f"\nTotal neurons: {d_ff}")
    print(f"Target active per position: {top_k}")
    print(f"Unique neurons used (input 1): {n_used1} / {d_ff}")
    print(f"Unique neurons used (input 2): {n_used2} / {d_ff}")
    print(f"Overlap between inputs: {overlap.item():.2%}")

    # Usage distribution
    usage_combined = usage1 + usage2
    n_never = (usage_combined == 0).sum().item()
    n_rare = ((usage_combined > 0) & (usage_combined < 5)).sum().item()
    n_common = (usage_combined >= 5).sum().item()

    print(f"\nNeuron usage distribution:")
    print(f"  Never used: {n_never} ({n_never/d_ff*100:.1f}%)")
    print(f"  Rarely (<5): {n_rare} ({n_rare/d_ff*100:.1f}%)")
    print(f"  Common (≥5): {n_common} ({n_common/d_ff*100:.1f}%)")

    if n_used1 > top_k and overlap < 0.9:
        print("\n✅ ROUTER SELECTION TEST PASSED")
        print("   Different inputs use different neurons")
    else:
        print("\n⚠️  ROUTER SELECTION NEEDS TRAINING")
        print("   Router is randomly initialized")

    return {
        'n_used1': n_used1,
        'n_used2': n_used2,
        'overlap': overlap.item()
    }


def test_backward_pass():
    """Test that gradients flow correctly"""
    print("\n" + "="*70)
    print("TEST 4: Gradient Flow")
    print("="*70)

    d_model = 512
    d_ff = 2048
    batch = 4
    seq = 16

    neuron_ffn = DynamicFFNLayer(d_model=d_model, d_ff=d_ff)

    x = torch.randn(batch, seq, d_model, requires_grad=True)

    # Forward + backward (dense)
    out_dense = neuron_ffn(x, top_k=None)
    loss_dense = out_dense.sum()
    loss_dense.backward()

    grad_dense = x.grad.clone()
    x.grad.zero_()

    # Forward + backward (sparse)
    out_sparse = neuron_ffn(x, top_k=512)
    loss_sparse = out_sparse.sum()
    loss_sparse.backward()

    grad_sparse = x.grad.clone()

    print(f"\nDense gradient norm: {grad_dense.norm().item():.4f}")
    print(f"Sparse gradient norm: {grad_sparse.norm().item():.4f}")

    # Check neuron gradients
    n_neurons_with_grad = 0
    for neuron in neuron_ffn.middle_neurons:
        if neuron.W_in.grad is not None and neuron.W_in.grad.abs().sum() > 0:
            n_neurons_with_grad += 1

    print(f"\nNeurons with non-zero gradients: {n_neurons_with_grad} / {d_ff}")

    if grad_dense.norm() > 0 and grad_sparse.norm() > 0 and n_neurons_with_grad > 0:
        print("\n✅ GRADIENT FLOW TEST PASSED")
    else:
        print("\n❌ GRADIENT FLOW TEST FAILED")

    return grad_dense.norm().item() > 0


def test_full_model():
    """Test full NeuronBasedLanguageModel"""
    print("\n" + "="*70)
    print("TEST 5: Full Model Forward Pass")
    print("="*70)

    vocab_size = 1000
    d_model = 256
    d_ff = 512
    n_layers = 2
    n_heads = 4
    batch = 2
    seq = 16

    model = NeuronBasedLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layers,
        n_heads=n_heads
    )

    tokens = torch.randint(0, vocab_size, (batch, seq))

    try:
        # Dense mode
        with torch.no_grad():
            outputs_dense = model(tokens, top_k=None)
            logits_dense = outputs_dense['logits']

        print(f"\nInput shape: {tokens.shape}")
        print(f"Output shape (dense): {logits_dense.shape}")
        print(f"Expected shape: [{batch}, {seq}, {vocab_size}]")

        # Sparse mode
        with torch.no_grad():
            outputs_sparse = model(tokens, top_k=128)
            logits_sparse = outputs_sparse['logits']

        print(f"Output shape (sparse): {logits_sparse.shape}")

        # Check output validity
        assert logits_dense.shape == (batch, seq, vocab_size), "Wrong output shape"
        assert not torch.isnan(logits_dense).any(), "NaN in output"
        assert not torch.isinf(logits_dense).any(), "Inf in output"

        # Test with labels (loss computation)
        labels = torch.randint(0, vocab_size, (batch, seq))
        with torch.no_grad():
            outputs_with_loss = model(tokens, labels=labels, top_k=None)
            loss = outputs_with_loss['loss']

        assert loss is not None, "Loss should not be None when labels provided"
        assert not torch.isnan(loss), "Loss is NaN"
        print(f"Loss shape: {loss.shape}")
        print(f"Loss value: {loss.item():.4f}")

        print("\n✅ FULL MODEL TEST PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FULL MODEL TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("NEURON-BASED MODEL EQUIVALENCE TESTS")
    print("="*70)

    torch.manual_seed(42)

    results = {}

    # Run all tests
    results['equivalence'] = test_ffn_equivalence()
    results['sparse_approx'] = test_sparse_approximation()
    results['router'] = test_router_selection()
    results['gradients'] = test_backward_pass()
    results['full_model'] = test_full_model()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum([
        results['equivalence'],
        results['gradients'],
        results['full_model']
    ])
    total = 3  # Critical tests

    print(f"\nCritical tests passed: {passed}/{total}")

    if passed == total:
        print("\n✅ ALL CRITICAL TESTS PASSED")
        print("   Model is ready for training!")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("   Fix issues before training")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
