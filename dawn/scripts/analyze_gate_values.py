#!/usr/bin/env python3
"""
Analyze GeGLUGate output_scale values from checkpoint

Usage:
    python scripts/analyze_gate_values.py [checkpoint_path]

If no checkpoint provided, analyzes freshly initialized model.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dawn_model import DAWNModel
from configs.model import get_model_config


def analyze_gate_scales(model: nn.Module, verbose: bool = True):
    """
    Extract and analyze all output_scale parameters from GeGLUGates

    Returns:
        dict with gate names and their output_scale values
    """
    results = {}

    if verbose:
        print("\n" + "="*80)
        print("GeGLUGate Output Scale Analysis")
        print("="*80)

    for name, module in model.named_modules():
        # Check if this module has output_scale (GeGLUGate signature)
        if hasattr(module, 'output_scale'):
            scale_value = module.output_scale.item()
            results[name] = scale_value

            if verbose:
                status = "✓" if abs(scale_value) < 1.0 else "⚠"
                if abs(scale_value) > 5.0:
                    status = "❌"
                print(f"{status} {name:60s} = {scale_value:8.4f}")

    if verbose:
        print("="*80)

        if results:
            scales = list(results.values())
            print(f"\nStatistics:")
            print(f"  Total gates: {len(scales)}")
            print(f"  Min scale:   {min(scales):.4f}")
            print(f"  Max scale:   {max(scales):.4f}")
            print(f"  Mean scale:  {sum(scales)/len(scales):.4f}")
            print(f"  Median:      {sorted(scales)[len(scales)//2]:.4f}")

            # Danger zones
            large_scales = [s for s in scales if abs(s) > 1.0]
            very_large = [s for s in scales if abs(s) > 5.0]

            if very_large:
                print(f"\n❌ DANGER: {len(very_large)} gates have |scale| > 5.0")
                print(f"   These are likely to cause NaN soon!")
            elif large_scales:
                print(f"\n⚠ WARNING: {len(large_scales)} gates have |scale| > 1.0")
                print(f"   Monitor for potential instability")
            else:
                print(f"\n✓ OK: All scales are |scale| < 1.0 (stable range)")
        else:
            print("\nNo GeGLUGate modules found in model!")

    return results


def test_forward_pass(model: nn.Module, device: str = 'cpu'):
    """
    Run a test forward pass and capture gate outputs
    """
    print("\n" + "="*80)
    print("Forward Pass Gate Analysis")
    print("="*80)

    model.eval()
    model.to(device)

    # Create dummy input
    batch_size = 2
    seq_len = 128
    vocab_size = model.token_embeddings.num_embeddings

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Hook to capture gate outputs
    gate_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            with torch.no_grad():
                gate_outputs[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'abs_max': output.abs().max().item(),
                }
        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, 'output_scale'):
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)

    # Forward pass
    try:
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)

        # Print results
        print(f"\nGate output ranges (after forward pass):")
        print(f"{'Gate name':<60} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'|Max|':>8}")
        print("-"*100)

        for name in sorted(gate_outputs.keys()):
            stats = gate_outputs[name]
            status = "✓" if stats['abs_max'] < 10.0 else "⚠"
            if stats['abs_max'] > 50.0:
                status = "❌"

            print(f"{status} {name:<58} {stats['mean']:8.3f} {stats['std']:8.3f} "
                  f"{stats['min']:8.3f} {stats['max']:8.3f} {stats['abs_max']:8.3f}")

        print("="*80)

    except RuntimeError as e:
        print(f"\n❌ Forward pass failed: {e}")
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()


def main():
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        print(f"\nLoading checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Create model
        config = get_model_config()
        model = DAWNModel(config)

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print(f"✓ Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    else:
        print("\nNo checkpoint provided - analyzing freshly initialized model")
        config = get_model_config()
        model = DAWNModel(config)

    # Analyze scales
    results = analyze_gate_scales(model, verbose=True)

    # Test forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nRunning test forward pass on {device}...")
    test_forward_pass(model, device=device)

    return results


if __name__ == '__main__':
    main()
