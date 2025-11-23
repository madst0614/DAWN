"""
Quick test for DAWN v5.0 architecture
Tests: model creation, forward pass, loss computation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from models.model import DAWN

def test_v5_architecture():
    """Test DAWN v5.0 architecture"""
    print("\n" + "="*60)
    print("DAWN v5.0 Architecture Test")
    print("="*60)

    # Config
    config = {
        'vocab_size': 30522,
        'd_model': 256,
        'd_ff': 512,
        'n_layers': 2,
        'n_heads': 4,

        # Neurons (v5.0)
        'n_neurons': 512,
        'neuron_rank': 16,
        'neuron_k': 16,

        # Basis FFN (v5.0)
        'n_basis': 16,
        'basis_rank': 8,
        'mod_rank': 32,

        'max_seq_len': 128,
        'dropout': 0.1,
    }

    # Create model
    print("\n1. Creating model...")
    model = DAWN(**config)
    print(f"   ✓ Model version: {model.__version__}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Total parameters: {total_params:,}")

    # Parameter breakdown
    layer = model.layers[0]
    router_params = sum(p.numel() for p in layer.neuron_router.parameters())
    ffn_params = sum(p.numel() for p in layer.basis_ffn.parameters())

    print(f"\n2. Parameter breakdown (per layer):")
    print(f"   Router: {router_params:,}")
    print(f"     - Neuron pool (low-rank): {layer.neuron_router.neuron_A.numel() + layer.neuron_router.neuron_B.numel():,}")
    print(f"   Basis FFN: {ffn_params:,}")
    print(f"     - Basis blocks: {layer.basis_ffn.basis_A.numel() + layer.basis_ffn.basis_B.numel():,}")
    print(f"     - Neuron coefficients: {layer.basis_ffn.neuron_coef_A.numel() + layer.basis_ffn.neuron_coef_B.numel():,}")
    print(f"     - Token modulation: {layer.basis_ffn.token_mod_A.weight.numel() + layer.basis_ffn.token_mod_B.weight.numel():,}")

    # Test forward pass
    print(f"\n3. Testing forward pass...")
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))

    logits = model(input_ids)
    print(f"   ✓ Input shape: {input_ids.shape}")
    print(f"   ✓ Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config['vocab_size'])

    # Test with losses
    print(f"\n4. Testing with losses...")
    logits, losses = model(input_ids, return_losses=True)
    print(f"   ✓ Losses returned: {list(losses.keys())}")
    print(f"   ✓ Neuron ortho losses: {[f'{loss.item():.4f}' for loss in losses['neuron_ortho']]}")

    # Test with activations
    print(f"\n5. Testing with activations...")
    logits, all_selected = model(input_ids, return_activations=True)
    print(f"   ✓ Selected neurons per layer: {[sel.shape for sel in all_selected]}")

    # Test generation
    print(f"\n6. Testing generation...")
    prompt = torch.randint(0, config['vocab_size'], (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"   ✓ Prompt length: {prompt.shape[1]}")
    print(f"   ✓ Generated length: {generated.shape[1]}")
    assert generated.shape[1] == prompt.shape[1] + 20

    # Test neuron composition
    print(f"\n7. Testing neuron composition...")
    neurons_full = layer.neuron_router.neurons
    print(f"   ✓ Neuron pool shape: {neurons_full.shape}")
    assert neurons_full.shape == (config['n_neurons'], config['d_model'])

    # Verify low-rank
    neuron_A_params = layer.neuron_router.neuron_A.numel()
    neuron_B_params = layer.neuron_router.neuron_B.numel()
    full_rank_params = config['n_neurons'] * config['d_model']
    reduction = 100 * (1 - (neuron_A_params + neuron_B_params) / full_rank_params)
    print(f"   ✓ Low-rank reduction: {reduction:.1f}%")

    print(f"\n{'='*60}")
    print("✅ All tests passed!")
    print("="*60)

    return model


if __name__ == "__main__":
    model = test_v5_architecture()
