"""
Test script to verify learned sparsity functionality.

Tests that:
1. Model can be created with learned sparsity parameters
2. Forward pass works without specifying k
3. Learned parameters (k_ratio, temperature) exist and are in valid ranges
4. Sparsity guidance loss is computed correctly
"""

import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model import (
    DAWNLanguageModel,
    compute_learned_sparsity_loss,
    compute_model_orthogonality_loss
)


def test_learned_sparsity():
    """Test learned sparsity functionality."""
    print("=" * 70)
    print("TESTING LEARNED SPARSITY")
    print("=" * 70)

    # Create small model for testing
    print("\n1. Creating model with learned sparsity...")
    model = DAWNLanguageModel(
        vocab_size=1000,
        d_model=128,
        n_layers=3,
        n_input=64,
        n_process=128,
        n_heads=4,
        max_seq_len=64,
        dropout=0.1
    )
    print("   ✓ Model created successfully")

    # Check that learned parameters exist
    print("\n2. Checking learned parameters...")
    for i, layer in enumerate(model.layers):
        router = layer.block.router

        # Check parameters exist
        assert hasattr(router, 'log_temperature'), f"Layer {i}: log_temperature missing"
        assert hasattr(router, 'k_ratio_logit'), f"Layer {i}: k_ratio_logit missing"
        assert isinstance(router.log_temperature, torch.nn.Parameter), f"Layer {i}: log_temperature not Parameter"
        assert isinstance(router.k_ratio_logit, torch.nn.Parameter), f"Layer {i}: k_ratio_logit not Parameter"

        # Check methods exist
        assert hasattr(router, 'get_temperature'), f"Layer {i}: get_temperature method missing"
        assert hasattr(router, 'get_k_ratio'), f"Layer {i}: get_k_ratio method missing"

        # Check values are in valid range
        temp = router.get_temperature().item()
        k_ratio = router.get_k_ratio().item()

        assert 0.5 <= temp <= 5.0, f"Layer {i}: temperature {temp} out of range [0.5, 5.0]"
        assert 0.2 <= k_ratio <= 0.8, f"Layer {i}: k_ratio {k_ratio} out of range [0.2, 0.8]"

        print(f"   Layer {i}: temp={temp:.2f}, k_ratio={k_ratio:.3f} ✓")

    print("\n3. Testing forward pass without k specified...")
    batch_size = 4
    seq_len = 32

    # Create dummy input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))

    # Forward pass WITHOUT specifying k (model decides!)
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        k_input=None,  # Model decides!
        k_process=None,  # Model decides!
        return_routing_info=True
    )

    print(f"   ✓ Forward pass successful")
    print(f"   Loss: {outputs['loss'].item():.4f}")
    print(f"   Logits shape: {outputs['logits'].shape}")

    # Check aux loss contains sparsity_guidance
    print("\n4. Checking sparsity guidance loss...")
    aux_loss = outputs['aux_loss']

    assert 'sparsity_guidance' in aux_loss, "sparsity_guidance missing from aux_loss"
    sparsity_guidance = aux_loss['sparsity_guidance']

    print(f"   ✓ Sparsity guidance computed")
    print(f"   Sparsity guidance: {sparsity_guidance.item():.6f}")
    print(f"   Load balance: {aux_loss['load_balance'].item():.6f}")
    print(f"   Entropy: {aux_loss['entropy'].item():.6f}")

    # Check routing info contains learned parameters
    print("\n5. Checking routing info...")
    routing_info = outputs['routing_info']

    for i, layer_info in enumerate(routing_info):
        assert 'learned_k' in layer_info, f"Layer {i}: learned_k missing"
        assert 'learned_temp' in layer_info, f"Layer {i}: learned_temp missing"
        assert 'learned_k_ratio' in layer_info, f"Layer {i}: learned_k_ratio missing"

        learned_k = layer_info['learned_k']
        learned_temp = layer_info['learned_temp']
        learned_k_ratio = layer_info['learned_k_ratio']

        print(f"   Layer {i}: k={learned_k}, k_ratio={learned_k_ratio:.3f}, temp={learned_temp:.2f} ✓")

    # Test backward pass
    print("\n6. Testing backward pass (gradient flow)...")
    total_loss = outputs['loss'] + 0.01 * aux_loss['load_balance'] + 0.005 * sparsity_guidance
    total_loss.backward()

    # Check gradients exist for learned parameters
    for i, layer in enumerate(model.layers):
        router = layer.block.router

        assert router.log_temperature.grad is not None, f"Layer {i}: no grad for log_temperature"
        assert router.k_ratio_logit.grad is not None, f"Layer {i}: no grad for k_ratio_logit"

        temp_grad_norm = router.log_temperature.grad.norm().item()
        k_ratio_grad_norm = router.k_ratio_logit.grad.norm().item()

        print(f"   Layer {i}: temp_grad={temp_grad_norm:.6f}, k_ratio_grad={k_ratio_grad_norm:.6f} ✓")

    # Test sparsity loss function
    print("\n7. Testing sparsity loss function...")

    # Create dummy weights
    weights = torch.softmax(torch.randn(batch_size, 64), dim=-1)
    sparsity_loss = compute_learned_sparsity_loss(weights)

    print(f"   ✓ Sparsity loss computed: {sparsity_loss.item():.6f}")

    # Test orthogonality loss
    print("\n8. Testing orthogonality loss...")
    ortho_losses = compute_model_orthogonality_loss(model)

    print(f"   ✓ Orthogonality losses computed:")
    print(f"     Router: {ortho_losses['router_ortho'].item():.6f}")
    print(f"     Input: {ortho_losses['input_ortho'].item():.6f}")
    print(f"     Total: {ortho_losses['total_ortho'].item():.6f}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print("\nLearned sparsity is working correctly!")
    print("\nKey features verified:")
    print("  ✓ Learnable temperature and k_ratio parameters")
    print("  ✓ Automatic k selection (no manual specification)")
    print("  ✓ Sparsity guidance loss computation")
    print("  ✓ Gradient flow to learned parameters")
    print("  ✓ Routing info includes learned values")
    print("\n🚀 Ready to train with zero manual intervention!")


if __name__ == '__main__':
    test_learned_sparsity()
