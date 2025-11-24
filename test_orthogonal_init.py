#!/usr/bin/env python3
"""
Test orthogonal initialization and diversity in neuron coefficients
"""

import torch
import torch.nn.functional as F
from models.model import DAWN

def test_orthogonal_init():
    """Verify that neuron coefficients start orthogonal"""
    print("🔍 Testing Orthogonal Initialization\n")
    print("="*60)

    # Create model
    model = DAWN(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=3,
        n_heads=4,
        n_neurons=512,
        neuron_rank=16,
        neuron_k=16,
        n_basis=16,
        basis_rank=8,
        mod_rank=32,
        d_ff=1024,
        max_seq_len=128,
        dropout=0.1
    )

    # Check each layer
    for layer_idx, layer in enumerate(model.layers):
        print(f"\n📊 Layer {layer_idx}")
        print("-" * 60)

        # Get neuron coefficients
        coef_A = layer.basis_ffn.neuron_coef_A.detach()  # [512, 16]
        coef_B = layer.basis_ffn.neuron_coef_B.detach()  # [512, 16]

        # 1. Check orthonormality (columns should be orthonormal)
        # For n_neurons > n_basis, rows should be approximately orthonormal
        # when considering only the first n_basis neurons
        n_neurons, n_basis = coef_A.shape

        # Normalize coefficient vectors
        coef_A_norm = F.normalize(coef_A, p=2, dim=1)
        coef_B_norm = F.normalize(coef_B, p=2, dim=1)

        # Compute pairwise similarity (should be low for orthogonal)
        sim_A = torch.mm(coef_A_norm, coef_A_norm.T)
        sim_B = torch.mm(coef_B_norm, coef_B_norm.T)

        # Get off-diagonal similarities (exclude self-similarity)
        mask = ~torch.eye(n_neurons, dtype=torch.bool)
        off_diag_A = sim_A[mask].abs()
        off_diag_B = sim_B[mask].abs()

        # Statistics
        mean_sim_A = off_diag_A.mean().item()
        max_sim_A = off_diag_A.max().item()
        high_sim_A = (off_diag_A > 0.8).sum().item()

        mean_sim_B = off_diag_B.mean().item()
        max_sim_B = off_diag_B.max().item()
        high_sim_B = (off_diag_B > 0.8).sum().item()

        print(f"\n  Coefficient A:")
        print(f"    Mean similarity: {mean_sim_A:.4f} (lower is better)")
        print(f"    Max similarity:  {max_sim_A:.4f}")
        print(f"    High similarity pairs (>0.8): {high_sim_A}/{n_neurons*(n_neurons-1)}")

        print(f"\n  Coefficient B:")
        print(f"    Mean similarity: {mean_sim_B:.4f} (lower is better)")
        print(f"    Max similarity:  {max_sim_B:.4f}")
        print(f"    High similarity pairs (>0.8): {high_sim_B}/{n_neurons*(n_neurons-1)}")

        # 2. Check effective rank (diversity metric)
        U, S, V = torch.svd(coef_A_norm)
        S_normalized = S / S.sum()
        effective_rank_A = torch.exp(-(S_normalized * torch.log(S_normalized + 1e-10)).sum()).item()

        U, S, V = torch.svd(coef_B_norm)
        S_normalized = S / S.sum()
        effective_rank_B = torch.exp(-(S_normalized * torch.log(S_normalized + 1e-10)).sum()).item()

        print(f"\n  Effective rank:")
        print(f"    Coefficient A: {effective_rank_A:.2f}/{n_basis} ({effective_rank_A/n_basis*100:.1f}%)")
        print(f"    Coefficient B: {effective_rank_B:.2f}/{n_basis} ({effective_rank_B/n_basis*100:.1f}%)")

        # Evaluation
        print(f"\n  ✅ Evaluation:")
        if mean_sim_A < 0.1 and mean_sim_B < 0.1:
            print(f"    ✅ Low mean similarity (good diversity)")
        else:
            print(f"    ⚠️  High mean similarity (may lack diversity)")

        if high_sim_A < 1000 and high_sim_B < 1000:
            print(f"    ✅ Few highly similar pairs (good)")
        else:
            print(f"    ⚠️  Many highly similar pairs (redundancy issue)")

        if effective_rank_A > n_basis * 0.8 and effective_rank_B > n_basis * 0.8:
            print(f"    ✅ High effective rank (good diversity)")
        else:
            print(f"    ⚠️  Low effective rank (some collapse)")

    # Test diversity loss computation
    print(f"\n\n🔧 Testing Diversity Loss")
    print("=" * 60)

    for layer_idx, layer in enumerate(model.layers):
        diversity_loss = layer.basis_ffn.compute_diversity_loss().item()
        print(f"  Layer {layer_idx} diversity loss: {diversity_loss:.6f}")

        if diversity_loss < 0.01:
            print(f"    ✅ Very diverse (loss < 0.01)")
        elif diversity_loss < 0.1:
            print(f"    ✅ Good diversity (loss < 0.1)")
        else:
            print(f"    ⚠️  High loss (may have redundancy)")

    print("\n" + "=" * 60)
    print("✅ Orthogonal initialization test complete!")
    print("\nExpected results with QR initialization:")
    print("  - Mean similarity: ~0.02-0.05 (very low)")
    print("  - High similarity pairs: <100 (minimal)")
    print("  - Effective rank: >12/16 (>75% diversity)")
    print("  - Diversity loss: <0.01 (very diverse)")

if __name__ == "__main__":
    torch.manual_seed(42)
    test_orthogonal_init()
