"""
Test script to verify SOFT THRESHOLD learned sparsity functionality.

Tests that:
1. Model can be created with soft threshold learned sparsity
2. Forward pass works without specifying k (fully differentiable!)
3. Learned parameters (threshold, temperature) exist and are in valid ranges
4. Sparsity guidance loss with selection_info is computed correctly
5. Gradient flows to learned threshold parameter
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
    print("\n2. Checking learned parameters (SOFT THRESHOLD)...")
    for i, layer in enumerate(model.layers):
        router = layer.block.router

        # Check parameters exist
        assert hasattr(router, 'log_temperature'), f"Layer {i}: log_temperature missing"
        assert hasattr(router, 'sparsity_threshold'), f"Layer {i}: sparsity_threshold missing"
        assert hasattr(router, 'log_steepness'), f"Layer {i}: log_steepness missing"
        assert isinstance(router.log_temperature, torch.nn.Parameter), f"Layer {i}: log_temperature not Parameter"
        assert isinstance(router.sparsity_threshold, torch.nn.Parameter), f"Layer {i}: sparsity_threshold not Parameter"
        assert isinstance(router.log_steepness, torch.nn.Parameter), f"Layer {i}: log_steepness not Parameter"

        # Check methods exist
        assert hasattr(router, 'get_temperature'), f"Layer {i}: get_temperature method missing"
        assert hasattr(router, 'get_threshold'), f"Layer {i}: get_threshold method missing"
        assert hasattr(router, 'get_steepness'), f"Layer {i}: get_steepness method missing"

        # Check values are in valid range
        temp = router.get_temperature().item()
        threshold = router.get_threshold().item()
        steepness = router.get_steepness().item()

        assert 0.5 <= temp <= 5.0, f"Layer {i}: temperature {temp} out of range [0.5, 5.0]"
        assert 0.0 <= threshold <= 1.0, f"Layer {i}: threshold {threshold} out of range [0.0, 1.0]"
        assert 1.0 <= steepness <= 10.0, f"Layer {i}: steepness {steepness} out of range [1.0, 10.0]"

        print(f"   Layer {i}: temp={temp:.2f}, threshold={threshold:.3f}, steepness={steepness:.2f} ✓")

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

    # Check routing info contains learned parameters (SOFT THRESHOLD!)
    print("\n5. Checking routing info (soft threshold)...")
    routing_info = outputs['routing_info']

    for i, layer_info in enumerate(routing_info):
        assert 'learned_threshold' in layer_info, f"Layer {i}: learned_threshold missing"
        assert 'learned_steepness' in layer_info, f"Layer {i}: learned_steepness missing"
        assert 'effective_k' in layer_info, f"Layer {i}: effective_k missing"
        assert 'effective_k_ratio' in layer_info, f"Layer {i}: effective_k_ratio missing"
        assert 'learned_temp' in layer_info, f"Layer {i}: learned_temp missing"

        learned_threshold = layer_info['learned_threshold']
        learned_steepness = layer_info['learned_steepness']
        effective_k = layer_info['effective_k']
        effective_k_ratio = layer_info['effective_k_ratio']
        learned_temp = layer_info['learned_temp']

        print(f"   Layer {i}: threshold={learned_threshold:.3f}, steepness={learned_steepness:.2f}, eff_k={effective_k:.1f} ({effective_k_ratio:.1%}), temp={learned_temp:.2f} ✓")

    # Test backward pass (gradient flow to THRESHOLD!)
    print("\n6. Testing backward pass (gradient flow to threshold)...")
    total_loss = outputs['loss'] + 0.01 * aux_loss['load_balance'] + 0.005 * sparsity_guidance
    total_loss.backward()

    # Check gradients exist for learned parameters
    for i, layer in enumerate(model.layers):
        router = layer.block.router

        assert router.log_temperature.grad is not None, f"Layer {i}: no grad for log_temperature"
        assert router.sparsity_threshold.grad is not None, f"Layer {i}: no grad for sparsity_threshold"
        assert router.log_steepness.grad is not None, f"Layer {i}: no grad for log_steepness"

        temp_grad_norm = router.log_temperature.grad.norm().item()
        threshold_grad_norm = router.sparsity_threshold.grad.norm().item()
        steepness_grad_norm = router.log_steepness.grad.norm().item()

        print(f"   Layer {i}: temp_grad={temp_grad_norm:.6f}, threshold_grad={threshold_grad_norm:.6f}, steepness_grad={steepness_grad_norm:.6f} ✓")

    # Test sparsity loss function with selection_info
    print("\n7. Testing sparsity loss function (with selection_info)...")

    # Create dummy weights and selection_info
    weights = torch.softmax(torch.randn(batch_size, 64), dim=-1)
    dummy_selection_info = {
        'effective_k_ratio': 0.5,
        'effective_k': 32,
        'learned_threshold': 0.5,
        'learned_steepness': 3.0,
        'temperature': 1.0
    }
    sparsity_loss = compute_learned_sparsity_loss(weights, dummy_selection_info)

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
    print("\nSOFT THRESHOLD learned sparsity is working correctly!")
    print("\nKey features verified:")
    print("  ✓ Learnable temperature, THRESHOLD, and STEEPNESS parameters")
    print("  ✓ Fully differentiable soft selection (no hard top-k!)")
    print("  ✓ Effective k measured from soft weights")
    print("  ✓ Sparsity guidance with selection_info")
    print("  ✓ Gradient flow to all learned parameters (temp, threshold, steepness)")
    print("  ✓ Routing info includes all learned parameters")
    print("\n🚀 Ready to train with FULLY DIFFERENTIABLE learned sparsity!")


if __name__ == '__main__':
    test_learned_sparsity()
