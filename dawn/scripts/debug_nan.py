"""
Analyze checkpoint to debug NaN issue
"""
import torch
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load checkpoint
checkpoint_path = "/content/drive/MyDrive/dawn/phase1_checkpoints/lexical_mlm_epoch4.pt"
print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

expert_state = checkpoint["expert_state_dict"]

print("\n" + "="*70)
print("DELTA REFINER ANALYSIS")
print("="*70)

# Analyze each block
for block_idx in range(5):
    print(f"\n{'─'*70}")
    print(f"Block {block_idx}")
    print(f"{'─'*70}")

    # FFN weights
    ffn_keys = [
        f'delta_module.refiner.blocks.{block_idx}.ffn.0.weight',
        f'delta_module.refiner.blocks.{block_idx}.ffn.0.bias',
        f'delta_module.refiner.blocks.{block_idx}.ffn.3.weight',  # Final layer
        f'delta_module.refiner.blocks.{block_idx}.ffn.3.bias',
    ]

    for key in ffn_keys:
        if key in expert_state:
            param = expert_state[key]
            print(f"\n{key}:")
            print(f"  Shape: {list(param.shape)}")
            print(f"  Mean: {param.mean().item():.6f}")
            print(f"  Std: {param.std().item():.6f}")
            print(f"  Min: {param.min().item():.6f}")
            print(f"  Max: {param.max().item():.6f}")
            print(f"  Abs Max: {param.abs().max().item():.6f}")

            # Check for zeros
            num_zeros = (param == 0).sum().item()
            total = param.numel()
            if num_zeros > 0:
                print(f"  ⚠️  Zeros: {num_zeros}/{total} ({100*num_zeros/total:.2f}%)")

    # Attention weights
    attn_key = f'delta_module.refiner.blocks.{block_idx}.attention.in_proj_weight'
    if attn_key in expert_state:
        param = expert_state[attn_key]
        print(f"\n{attn_key}:")
        print(f"  Shape: {list(param.shape)}")
        print(f"  Mean: {param.mean().item():.6f}")
        print(f"  Std: {param.std().item():.6f}")
        print(f"  Abs Max: {param.abs().max().item():.6f}")

    # LayerNorm weights
    for ln_type in ['attn_layer_norm', 'ffn_layer_norm']:
        ln_weight_key = f'delta_module.refiner.blocks.{block_idx}.{ln_type}.weight'
        ln_bias_key = f'delta_module.refiner.blocks.{block_idx}.{ln_type}.bias'

        if ln_weight_key in expert_state:
            weight = expert_state[ln_weight_key]
            bias = expert_state[ln_bias_key]
            print(f"\n{ln_type}:")
            print(f"  Weight - Mean: {weight.mean().item():.6f}, Std: {weight.std().item():.6f}")
            print(f"  Bias - Mean: {bias.mean().item():.6f}, Std: {bias.std().item():.6f}")

print("\n" + "="*70)
print("MINI-GATE ANALYSIS")
print("="*70)

for gate_idx in range(5):
    print(f"\n{'─'*70}")
    print(f"Mini-Gate {gate_idx}")
    print(f"{'─'*70}")

    # Temperature
    temp_key = f'delta_module.refiner.mini_gates.{gate_idx}.temperature'
    if temp_key in expert_state:
        temp = expert_state[temp_key]
        print(f"\nTemperature: {temp.item():.6f}")

    # Query/Key projections
    for proj_type in ['query_proj', 'key_proj']:
        weight_key = f'delta_module.refiner.mini_gates.{gate_idx}.{proj_type}.weight'
        bias_key = f'delta_module.refiner.mini_gates.{gate_idx}.{proj_type}.bias'

        if weight_key in expert_state:
            weight = expert_state[weight_key]
            bias = expert_state[bias_key]

            print(f"\n{proj_type}:")
            print(f"  Weight - Mean: {weight.mean().item():.6f}, Std: {weight.std().item():.6f}, Abs Max: {weight.abs().max().item():.6f}")
            print(f"  Bias - Mean: {bias.mean().item():.6f}, Std: {bias.std().item():.6f}, Abs Max: {bias.abs().max().item():.6f}")

print("\n" + "="*70)
print("FINAL GATE ANALYSIS")
print("="*70)

# Final gate temperature
final_temp_key = 'delta_module.refiner.final_gate.temperature'
if final_temp_key in expert_state:
    temp = expert_state[final_temp_key]
    print(f"\nFinal Gate Temperature: {temp.item():.6f}")

# Final gate projections
for proj_type in ['query_proj', 'key_proj']:
    weight_key = f'delta_module.refiner.final_gate.{proj_type}.weight'
    bias_key = f'delta_module.refiner.final_gate.{proj_type}.bias'

    if weight_key in expert_state:
        weight = expert_state[weight_key]
        bias = expert_state[bias_key]

        print(f"\n{proj_type}:")
        print(f"  Weight - Mean: {weight.mean().item():.6f}, Std: {weight.std().item():.6f}, Abs Max: {weight.abs().max().item():.6f}")
        print(f"  Bias - Mean: {bias.mean().item():.6f}, Std: {bias.std().item():.6f}, Abs Max: {bias.abs().max().item():.6f}")

print("\n" + "="*70)
print("PROBLEMATIC PATTERNS")
print("="*70)

# Check for all-zero weights (problematic)
all_zero_params = []
very_large_params = []
very_small_std = []

for name, param in expert_state.items():
    if 'delta_module.refiner' not in name:
        continue

    # Check all zeros
    if (param == 0).all():
        all_zero_params.append(name)

    # Check very large values
    if param.abs().max() > 10.0:
        very_large_params.append((name, param.abs().max().item()))

    # Check very small std (might indicate stuck training)
    if param.numel() > 1 and param.std() < 1e-6:
        very_small_std.append((name, param.std().item()))

if all_zero_params:
    print("\n⚠️  ALL-ZERO PARAMETERS:")
    for name in all_zero_params:
        print(f"  - {name}")

if very_large_params:
    print("\n⚠️  VERY LARGE PARAMETERS (>10.0):")
    for name, val in very_large_params:
        print(f"  - {name}: {val:.6f}")

if very_small_std:
    print("\n⚠️  VERY SMALL STD (<1e-6):")
    for name, val in very_small_std:
        print(f"  - {name}: {val:.10f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nCheckpoint: {checkpoint['expert_name']}_{checkpoint['task']}")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Train Loss: {checkpoint['metrics']['train_loss']:.4f}")
print(f"Train Acc: {checkpoint['metrics']['train_accuracy']:.2f}%")
print(f"Val Acc: {checkpoint['metrics']['val_accuracy']:.2f}%")
