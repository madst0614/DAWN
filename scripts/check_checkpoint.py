#!/usr/bin/env python3
"""체크포인트 내용 확인"""
import sys
import torch

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/dawn/checkpoints_v10_18M/run_v10.0_20251130_162625_7036/best_model.pt"

print(f"Loading: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\n=== Checkpoint Keys ===")
for key in checkpoint.keys():
    print(f"  {key}: {type(checkpoint[key])}")

print("\n=== Config ===")
config = checkpoint.get('config', {})
print(f"Type: {type(config)}")
if isinstance(config, dict):
    for k, v in config.items():
        print(f"  {k}: {v}")
else:
    for k in dir(config):
        if not k.startswith('_'):
            print(f"  {k}: {getattr(config, k)}")

print("\n=== State Dict Keys (first 20) ===")
state_dict = checkpoint.get('model_state_dict', {})
keys = list(state_dict.keys())
for k in keys[:20]:
    print(f"  {k}: {state_dict[k].shape}")
print(f"  ... total {len(keys)} keys")

print("\n=== Model Structure Detection ===")
has_per_layer = any('layers.0.attn.shared_neurons' in k for k in keys)
has_global = any(k.startswith('shared_neurons.') for k in keys)
print(f"  Per-layer shared_neurons: {has_per_layer}")
print(f"  Global shared_neurons: {has_global}")

# Check compress/expand neuron shapes
for k in keys:
    if 'compress_neurons' in k or 'expand_neurons' in k:
        print(f"  {k}: {state_dict[k].shape}")
        break
