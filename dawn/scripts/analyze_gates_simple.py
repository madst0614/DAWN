#!/usr/bin/env python3
"""
Comprehensive gate analysis - analyze output_scale values and gate behavior

Usage:
    python scripts/analyze_gates_simple.py --checkpoint_dir /path/to/checkpoints/
    python scripts/analyze_gates_simple.py --checkpoint /path/to/specific.pt
"""

import sys
import torch
import argparse
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in directory"""
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Find all .pt files
    checkpoints = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")

    # Sort by modification time, get latest
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return latest


def analyze_checkpoint(checkpoint_path):
    """Analyze output_scale values from checkpoint"""

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get state dict (handle different checkpoint formats)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'expert_state_dict' in checkpoint:
        # Phase 1 checkpoints: expert_state_dict
        state_dict = checkpoint['expert_state_dict']
        print(f"✓ Phase 1 checkpoint detected (expert: {checkpoint.get('expert_name', 'unknown')})")
    else:
        # Assume entire checkpoint is state_dict
        state_dict = checkpoint

    print("\n" + "="*80)
    print("GeGLUGate output_scale Analysis")
    print("="*80)

    # Debug: Show some state_dict keys
    print("\nFirst 10 state_dict keys:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {key}")
    print(f"  ... (total {len(state_dict)} keys)\n")

    # Find all output_scale parameters
    scales = {}
    for name, param in state_dict.items():
        if 'output_scale' in name:
            # Check if it's a Tensor
            if not isinstance(param, torch.Tensor):
                print(f"⚠️  Skipping non-tensor: {name} (type: {type(param)})")
                continue

            value = param.item() if param.numel() == 1 else param.cpu().numpy()
            scales[name] = value

            # Status indicator
            abs_val = abs(value) if isinstance(value, (int, float)) else abs(value).max()
            if abs_val > 5.0:
                status = "❌ DANGER"
            elif abs_val > 1.0:
                status = "⚠ WARNING"
            else:
                status = "✓ OK"

            print(f"{status:12} {name:70s} = {value:8.4f}" if isinstance(value, (int, float))
                  else f"{status:12} {name:70s} = {value}")

    print("="*80)

    if scales:
        values = [v for v in scales.values() if isinstance(v, (int, float))]
        if values:
            print(f"\nStatistics:")
            print(f"  Total output_scales: {len(values)}")
            print(f"  Min:    {min(values):8.4f}")
            print(f"  Max:    {max(values):8.4f}")
            print(f"  Mean:   {sum(values)/len(values):8.4f}")
            print(f"  Median: {sorted(values)[len(values)//2]:8.4f}")

            # Warnings
            large = [v for v in values if abs(v) > 1.0]
            very_large = [v for v in values if abs(v) > 5.0]

            if very_large:
                print(f"\n❌ CRITICAL: {len(very_large)} scales have |value| > 5.0")
                print(f"   Model is likely unstable or about to explode!")
                print(f"   Scales: {very_large}")
            elif large:
                print(f"\n⚠ WARNING: {len(large)} scales have |value| > 1.0")
                print(f"   These may contribute to instability")
                print(f"   Scales: {large}")
            else:
                print(f"\n✓ All scales are in stable range (|value| < 1.0)")
    else:
        print("\nNo output_scale parameters found!")
        print("Make sure this is a GeGLUGate-based model.")

    print("\n" + "="*80)

    # Check for NaN in checkpoint
    print("\nChecking for NaN parameters in checkpoint...")
    nan_params = []
    for name, param in state_dict.items():
        # Skip non-tensor values (like step, epoch, etc.)
        if not isinstance(param, torch.Tensor):
            continue

        if torch.isnan(param).any():
            nan_count = torch.isnan(param).sum().item()
            total = param.numel()
            nan_params.append((name, nan_count, total))

    if nan_params:
        print(f"\n❌ Found NaN in {len(nan_params)} parameters:")
        for name, nan_count, total in nan_params[:10]:  # Show first 10
            print(f"   {name}: {nan_count}/{total} ({100*nan_count/total:.1f}%)")
    else:
        print("✓ No NaN parameters found")

    print("="*80)

    return scales, state_dict


def test_forward_pass(checkpoint, state_dict):
    """
    Load expert with checkpoint and test forward pass with gate capture

    Args:
        checkpoint: Full checkpoint dict
        state_dict: expert_state_dict
    """
    from models.expert import DeltaExpert

    print("\n" + "="*80)
    print("Forward Pass Gate Analysis")
    print("="*80)

    # Get expert name
    expert_name = checkpoint.get('expert_name', 'lexical')
    print(f"\nCreating {expert_name} expert and loading checkpoint...")

    # Get config from checkpoint (saved during training)
    if 'config' in checkpoint:
        cfg_dict = checkpoint['config']
        print(f"✓ Using config from checkpoint")
    else:
        # Fallback: create minimal config dict
        print(f"⚠️  No config in checkpoint, using defaults")
        cfg_dict = {
            'vocab_size': 30522,
            'hidden_size': 768,
            'num_heads': 12,
            'intermediate_size': [1024, 1536, 2048, 1536, 1024],
            'num_steps': 4,
            'dropout': 0.1,
            'max_length': 512,
        }

    # Create expert with config dict (DeltaExpert expects Dict, not class instance)
    expert = DeltaExpert(
        config=cfg_dict,
        peer_names=None,  # Phase 1 has no peers
        shared_embeddings=None,  # Will use own embeddings
    )

    # Load checkpoint weights
    expert.load_state_dict(state_dict)
    expert.eval()

    print("✓ Expert loaded")

    # Create dummy input
    batch_size = 2
    seq_len = 32
    vocab_size = cfg_dict['vocab_size']
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Storage for gate outputs
    gate_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            gate_outputs[name] = output.detach().clone()
        return hook

    # Register hooks
    print("\nRegistering hooks on gates...")
    hooks = []
    refiner = expert.delta_module.refiner

    for i in range(5):
        hook = refiner.mini_gates[i].register_forward_hook(make_hook(f'mini_gates.{i}'))
        hooks.append(hook)

    hook = refiner.final_gate.register_forward_hook(make_hook('final_gate'))
    hooks.append(hook)

    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        try:
            _ = expert(input_ids, attention_mask=attention_mask)
            print("✓ Forward pass successful")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            for hook in hooks:
                hook.remove()
            return

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze outputs
    print("\n" + "-"*80)
    print(f"{'Gate':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'|Max|':>10}")
    print("-"*80)

    for name in sorted(gate_outputs.keys()):
        out = gate_outputs[name]
        print(f"{name:<20} {out.min():>10.4f} {out.max():>10.4f} "
              f"{out.mean():>10.4f} {out.std():>10.4f} {out.abs().max():>10.4f}")

    # Diagnostics
    print("\n" + "-"*80)
    print("Diagnostic checks:")
    print("-"*80)
    for name, out in sorted(gate_outputs.items()):
        abs_max = out.abs().max().item()
        if abs_max > 10.0:
            print(f"❌ {name}: DANGER! |max|={abs_max:.2f} (>10)")
        elif abs_max > 5.0:
            print(f"⚠️  {name}: WARNING |max|={abs_max:.2f} (>5)")
        elif abs_max > 2.0:
            print(f"⚠️  {name}: High |max|={abs_max:.2f} (>2)")
        else:
            print(f"✓ {name}: OK |max|={abs_max:.2f}")

    # Tanh simulation
    print("\n" + "-"*80)
    print("What if we apply tanh to output_scale?")
    print("-"*80)
    for name, param in expert.named_parameters():
        if 'output_scale' in name:
            raw = param.item()
            tanh_val = torch.tanh(param).item()
            diff = raw - tanh_val

            short_name = name.replace('delta_module.refiner.', '')
            print(f"{short_name:30s}  Raw: {raw:7.4f}  →  Tanh: {tanh_val:7.4f}  "
                  f"(diff: {diff:+7.4f})")

            if abs(diff) > 0.01:
                print(f"  → Tanh would clamp by {abs(diff):.4f}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze GeGLUGate output_scale values")
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to specific checkpoint file'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='Directory containing checkpoints (will use latest)'
    )

    args = parser.parse_args()

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
    elif args.checkpoint_dir:
        try:
            checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
            print(f"Found latest checkpoint: {checkpoint_path.name}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("\nTry one of these commands to find your checkpoints:")
            print("  !find /content -name '*.pt' -type f 2>/dev/null")
            print("  !ls -lh /content/drive/MyDrive/dawn/")
            return
    else:
        print("❌ Please specify either --checkpoint or --checkpoint_dir")
        print("\nExamples:")
        print("  python scripts/analyze_gates_simple.py --checkpoint_dir /content/drive/MyDrive/dawn/phase1_checkpoints/")
        print("  python scripts/analyze_gates_simple.py --checkpoint /path/to/checkpoint.pt")
        return

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Analyze checkpoint (part 1: static analysis)
    scales, state_dict = analyze_checkpoint(checkpoint_path)

    # Test forward pass (part 2: dynamic analysis)
    if state_dict:
        try:
            test_forward_pass(checkpoint, state_dict)
        except Exception as e:
            print(f"\n⚠️  Could not run forward pass analysis: {e}")
            print("Static checkpoint analysis completed successfully.")


if __name__ == '__main__':
    main()
