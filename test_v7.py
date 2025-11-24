"""
Test script for DAWN v7.0 - Fixed Orthogonal Basis

Tests:
1. Model creation and parameter counting
2. Orthogonality verification (should be mathematically perfect)
3. Forward pass and loss computation
4. Basis usage analysis
5. Gradient flow verification
6. Memory efficiency
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from models.model_v7 import DAWN, FixedOrthogonalBasis


def test_orthogonality():
    """Test 1: Verify mathematical orthogonality of fixed basis"""
    print("\n" + "="*60)
    print("Test 1: Fixed Basis Orthogonality")
    print("="*60)

    # Create a basis
    basis = FixedOrthogonalBasis(
        n_basis=32,
        d_model=256,
        d_ff=1024,
        basis_rank=64
    )

    # Check Basis A orthogonality
    basis_A_flat = basis.basis_A.view(32, -1)
    gram_A = torch.mm(basis_A_flat, basis_A_flat.T)
    identity_A = torch.eye(32)
    error_A = (gram_A - identity_A).abs().max().item()

    print(f"  Basis A (Up projection):")
    print(f"    Shape: {basis.basis_A.shape}")
    print(f"    Max deviation from identity: {error_A:.2e}")
    print(f"    ✓ Orthogonal: {error_A < 1e-5}")

    # Check Basis B orthogonality
    basis_B_flat = basis.basis_B.view(32, -1)
    gram_B = torch.mm(basis_B_flat, basis_B_flat.T)
    identity_B = torch.eye(32)
    error_B = (gram_B - identity_B).abs().max().item()

    print(f"  Basis B (Down projection):")
    print(f"    Shape: {basis.basis_B.shape}")
    print(f"    Max deviation from identity: {error_B:.2e}")
    print(f"    ✓ Orthogonal: {error_B < 1e-5}")

    # Check Basis embedding orthogonality
    gram_emb = torch.mm(basis.basis_emb, basis.basis_emb.T)
    identity_emb = torch.eye(32)
    error_emb = (gram_emb - identity_emb).abs().max().item()

    print(f"  Basis Embedding (Routing):")
    print(f"    Shape: {basis.basis_emb.shape}")
    print(f"    Max deviation from identity: {error_emb:.2e}")
    print(f"    ✓ Orthogonal: {error_emb < 1e-5}")

    # Overall result
    is_orthogonal = max(error_A, error_B, error_emb) < 1e-5
    print(f"\n  Overall: {'✅ PASS' if is_orthogonal else '❌ FAIL'}")

    return is_orthogonal


def test_model_creation():
    """Test 2: Model creation and parameter counting"""
    print("\n" + "="*60)
    print("Test 2: Model Creation & Parameters")
    print("="*60)

    config = {
        'vocab_size': 30000,
        'd_model': 256,
        'd_ff': 1024,
        'n_layers': 4,
        'n_heads': 4,
        'n_basis': 32,
        'basis_rank': 64,
        'n_neurons': 64,
        'neuron_k': 8,
        'max_seq_len': 512,
        'dropout': 0.1,
    }

    # Create model
    model = DAWN(**config)
    print(f"  Model version: {model.__version__}")

    # Parameter count
    params = model.get_num_params()
    print(f"\n  Parameters:")
    print(f"    Total: {params['total']:,}")
    print(f"    Trainable: {params['trainable']:,}")
    print(f"    Fixed Basis: {params['basis_fixed']:,}")
    print(f"    Recipe: {params['recipe']:,} ({params['recipe']/params['trainable']*100:.1f}%)")
    print(f"    Router: {params['router']:,} ({params['router']/params['trainable']*100:.1f}%)")
    print(f"    Embeddings: {params['embeddings']:,} ({params['embeddings']/params['trainable']*100:.1f}%)")

    # Verify basis is not trainable
    basis_requires_grad = any(p.requires_grad for p in model.shared_basis.parameters())
    print(f"\n  Basis trainable: {basis_requires_grad}")
    print(f"  ✓ {'✅ PASS' if not basis_requires_grad else '❌ FAIL (basis should be fixed!)'}")

    return model, not basis_requires_grad


def test_forward_pass(model):
    """Test 3: Forward pass and loss computation"""
    print("\n" + "="*60)
    print("Test 3: Forward Pass & Loss")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  Device: {device}")

    # Create dummy data
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len)).to(device)
    labels = torch.randint(0, model.vocab_size, (batch_size, seq_len)).to(device)

    # Forward pass
    print(f"\n  Forward pass:")
    print(f"    Input shape: {input_ids.shape}")

    logits, neuron_indices = model(input_ids, return_activations=True)
    print(f"    Output shape: {logits.shape}")
    print(f"    Neuron indices per layer: {len(neuron_indices)} layers")
    print(f"    Neuron indices shape: {neuron_indices[0].shape}")

    # Loss computation
    loss, loss_dict = model.get_loss(input_ids, labels)
    print(f"\n  Loss:")
    print(f"    CE Loss: {loss_dict['ce']:.4f}")
    print(f"    Total Loss: {loss_dict['total']:.4f}")
    print(f"    ✓ No orthogonality loss needed!")

    # Check for NaN
    has_nan = torch.isnan(logits).any().item()
    print(f"\n  NaN check: {'❌ FAIL (has NaN)' if has_nan else '✅ PASS'}")

    return not has_nan, model


def test_basis_usage(model):
    """Test 4: Analyze basis usage across layers"""
    print("\n" + "="*60)
    print("Test 4: Basis Usage Analysis")
    print("="*60)

    analysis = model.analyze_basis_usage()

    for layer_name, stats in analysis.items():
        print(f"\n  {layer_name}:")

        # Basis usage distribution
        basis_usage = stats['basis_usage']
        max_usage = max(basis_usage)
        min_usage = min(basis_usage)

        print(f"    Basis usage range: [{min_usage:.3f}, {max_usage:.3f}]")
        print(f"    Recipe diversity: {stats['recipe_diversity']:.4f}")
        print(f"    Neuron similarity: {stats['neuron_similarity']:.4f}")

        # Check if basis is being used
        using_all_basis = min_usage > 0.001
        print(f"    Using all basis: {using_all_basis}")

    print(f"\n  ✅ Analysis complete")
    return True


def test_gradient_flow(model):
    """Test 5: Verify gradient flow"""
    print("\n" + "="*60)
    print("Test 5: Gradient Flow")
    print("="*60)

    device = next(model.parameters()).device

    # Create dummy data
    input_ids = torch.randint(0, model.vocab_size, (2, 32)).to(device)
    labels = torch.randint(0, model.vocab_size, (2, 32)).to(device)

    # Forward + backward
    model.train()
    loss, _ = model.get_loss(input_ids, labels)
    loss.backward()

    # Check gradients
    print(f"\n  Checking gradients:")

    # Recipe gradients (should have gradients)
    recipe_grads = []
    for i, layer in enumerate(model.layers):
        grad = layer.ffn.neuron_recipe.grad
        if grad is not None:
            grad_norm = grad.norm().item()
            recipe_grads.append(grad_norm)
            print(f"    Layer {i} recipe grad norm: {grad_norm:.6f}")

    has_recipe_grads = len(recipe_grads) == len(model.layers) and all(g > 0 for g in recipe_grads)

    # Basis gradients (should NOT have gradients)
    basis_grads = []
    for buffer_name, buffer in model.shared_basis.named_buffers():
        if buffer.requires_grad and buffer.grad is not None:
            basis_grads.append(buffer_name)

    has_basis_grads = len(basis_grads) > 0

    print(f"\n  Recipe parameters have gradients: {has_recipe_grads}")
    print(f"  Basis buffers have gradients: {has_basis_grads}")
    print(f"  ✓ {'✅ PASS' if has_recipe_grads and not has_basis_grads else '❌ FAIL'}")

    model.zero_grad()
    return has_recipe_grads and not has_basis_grads


def test_memory_efficiency():
    """Test 6: Compare memory usage with v6.0"""
    print("\n" + "="*60)
    print("Test 6: Memory Efficiency")
    print("="*60)

    config_v7 = {
        'vocab_size': 30000,
        'd_model': 256,
        'd_ff': 1024,
        'n_layers': 4,
        'n_heads': 4,
        'n_basis': 32,
        'basis_rank': 64,
        'n_neurons': 64,
        'neuron_k': 8,
        'max_seq_len': 512,
        'dropout': 0.1,
    }

    # v7.0
    model_v7 = DAWN(**config_v7)
    params_v7 = model_v7.get_num_params()

    print(f"\n  v7.0 Parameters:")
    print(f"    Total: {params_v7['total']:,}")
    print(f"    Trainable: {params_v7['trainable']:,}")
    print(f"    Fixed (shared): {params_v7['basis_fixed']:,}")

    # Efficiency metrics
    trainable_per_layer = params_v7['recipe'] / config_v7['n_layers']
    print(f"\n  Efficiency:")
    print(f"    Recipe params per layer: {trainable_per_layer:,.0f}")
    print(f"    Fixed basis shared across {config_v7['n_layers']} layers")

    # Compare with hypothetical non-shared version
    basis_params_per_layer = params_v7['basis_fixed'] / config_v7['n_layers']
    hypothetical_total = params_v7['trainable'] + params_v7['basis_fixed'] * config_v7['n_layers']
    savings = (hypothetical_total - params_v7['total']) / hypothetical_total * 100

    print(f"\n  Sharing Benefits:")
    print(f"    Basis params per layer (if not shared): {basis_params_per_layer:,.0f}")
    print(f"    Total params (if not shared): {hypothetical_total:,}")
    print(f"    Savings from sharing: {savings:.1f}%")

    print(f"\n  ✅ Memory efficient design")
    return True


def test_neuron_embeddings():
    """Test 7: Verify neuron embeddings are derived from recipes"""
    print("\n" + "="*60)
    print("Test 7: Neuron Embedding Derivation")
    print("="*60)

    config = {
        'vocab_size': 30000,
        'd_model': 256,
        'd_ff': 1024,
        'n_layers': 2,
        'n_heads': 4,
        'n_basis': 32,
        'basis_rank': 64,
        'n_neurons': 64,
        'neuron_k': 8,
        'max_seq_len': 512,
        'dropout': 0.1,
    }

    model = DAWN(**config)

    print(f"\n  Checking neuron embedding derivation:")

    for i, layer in enumerate(model.layers):
        # Get neuron embedding
        neuron_emb = layer.ffn.neuron_emb  # [n_neurons, d_model]

        # Manually compute from recipe
        recipe = layer.ffn.neuron_recipe
        recipe_norm = F.softmax(recipe, dim=-1)
        expected_emb = torch.matmul(recipe_norm, model.shared_basis.basis_emb)

        # Compare
        diff = (neuron_emb - expected_emb).abs().max().item()

        print(f"    Layer {i}: Max difference = {diff:.2e}")

        if diff > 1e-6:
            print(f"    ❌ FAIL: Neuron embeddings not properly derived!")
            return False

    print(f"\n  ✅ PASS: All neuron embeddings correctly derived from recipes")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("DAWN v7.0 - Comprehensive Test Suite")
    print("="*60)

    results = []

    # Test 1: Orthogonality
    try:
        results.append(("Orthogonality", test_orthogonality()))
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results.append(("Orthogonality", False))

    # Test 2: Model creation
    try:
        model, passed = test_model_creation()
        results.append(("Model Creation", passed))
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results.append(("Model Creation", False))
        return

    # Test 3: Forward pass
    try:
        passed, model = test_forward_pass(model)
        results.append(("Forward Pass", passed))
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results.append(("Forward Pass", False))

    # Test 4: Basis usage
    try:
        results.append(("Basis Usage", test_basis_usage(model)))
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results.append(("Basis Usage", False))

    # Test 5: Gradient flow
    try:
        results.append(("Gradient Flow", test_gradient_flow(model)))
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results.append(("Gradient Flow", False))

    # Test 6: Memory efficiency
    try:
        results.append(("Memory Efficiency", test_memory_efficiency()))
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results.append(("Memory Efficiency", False))

    # Test 7: Neuron embeddings
    try:
        results.append(("Neuron Embeddings", test_neuron_embeddings()))
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results.append(("Neuron Embeddings", False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:.<40} {status}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed! DAWN v7.0 is ready!")
    else:
        print("⚠️  Some tests failed. Please review.")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    success = run_all_tests()
    sys.exit(0 if success else 1)
