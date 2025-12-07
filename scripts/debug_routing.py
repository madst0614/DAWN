#!/usr/bin/env python3
"""
Debug script for routing weight extraction
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    from transformers import BertTokenizer
    from models import create_model_by_version

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    # Version detection
    path_str = str(args.checkpoint).lower()
    if 'v15' in path_str:
        version = '15.0'
    elif 'v14' in path_str:
        version = '14.0'
    else:
        version = config.get('model_version', '15.0')

    print(f"Model version: {version}")
    print(f"Config: {config}")

    model_kwargs = {
        'vocab_size': config.get('vocab_size', 30522),
        'd_model': config.get('d_model', 320),
        'n_layers': config.get('n_layers', 4),
        'n_heads': config.get('n_heads', 4),
        'rank': config.get('rank', 64),
        'max_seq_len': config.get('max_seq_len', 512),
        'n_feature': config.get('n_feature', 48),
        'n_relational': config.get('n_relational', 12),
        'n_value': config.get('n_value', 12),
        'n_knowledge': config.get('n_knowledge', 80),
        'dropout': config.get('dropout', 0.1),
        'state_dim': config.get('state_dim', 64),
        'knowledge_rank': config.get('knowledge_rank', 128),
        'coarse_k': config.get('coarse_k', 20),
        'fine_k': config.get('fine_k', 10),
    }

    model = create_model_by_version(version, model_kwargs)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Test forward
    print("\n" + "="*60)
    print("Testing forward pass with return_routing_info=True")
    print("="*60)

    test_text = "The cat sat on the mat."
    tokens = tokenizer(test_text, return_tensors='pt', padding=True)
    input_ids = tokens['input_ids'].to(device)

    print(f"\nInput: '{test_text}'")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids}")

    with torch.no_grad():
        outputs = model(input_ids, return_routing_info=True)

    print(f"\n--- Output structure ---")
    print(f"Type: {type(outputs)}")

    if isinstance(outputs, tuple):
        print(f"Tuple length: {len(outputs)}")
        for i, out in enumerate(outputs):
            if isinstance(out, torch.Tensor):
                print(f"  outputs[{i}]: Tensor shape={out.shape}, dtype={out.dtype}")
            elif isinstance(out, list):
                print(f"  outputs[{i}]: List length={len(out)}")
                for j, item in enumerate(out[:3]):  # First 3 items
                    print(f"    [{j}]: {type(item)}")
                    if isinstance(item, dict):
                        for k, v in item.items():
                            if isinstance(v, torch.Tensor):
                                print(f"      '{k}': Tensor shape={v.shape}")
                            elif isinstance(v, dict):
                                print(f"      '{k}': dict with keys={list(v.keys())}")
                                for kk, vv in v.items():
                                    if isinstance(vv, torch.Tensor):
                                        print(f"        '{kk}': Tensor shape={vv.shape}, min={vv.min():.4f}, max={vv.max():.4f}, mean={vv.mean():.4f}")
                                    else:
                                        print(f"        '{kk}': {type(vv)}")
                            else:
                                print(f"      '{k}': {type(v)}")
            elif isinstance(out, dict):
                print(f"  outputs[{i}]: Dict with keys={list(out.keys())}")
            else:
                print(f"  outputs[{i}]: {type(out)}")
    else:
        print(f"Not a tuple: {type(outputs)}")

    # Try to extract routing info
    print("\n--- Extracting routing info ---")
    if isinstance(outputs, tuple):
        if len(outputs) == 2:
            logits, routing_info_list = outputs
            print("Format: (logits, routing_info_list)")
        elif len(outputs) >= 3:
            # Might be (loss, logits, routing_info) - but we didn't pass labels
            routing_info_list = outputs[-1]
            print(f"Format: tuple of {len(outputs)} elements, using last as routing_info")
        else:
            routing_info_list = []

        if routing_info_list and len(routing_info_list) > 0:
            print(f"\nRouting info list length: {len(routing_info_list)}")

            for layer_idx, info in enumerate(routing_info_list[:2]):  # First 2 layers
                print(f"\n--- Layer {layer_idx} ---")
                if isinstance(info, dict):
                    print(f"Keys: {list(info.keys())}")

                    # Check attention routing
                    if 'attention' in info:
                        attn_info = info['attention']
                        print(f"  attention keys: {list(attn_info.keys()) if isinstance(attn_info, dict) else type(attn_info)}")

                        if isinstance(attn_info, dict):
                            # Check all routing-related keys
                            all_keys = ['feature_weights', 'feature_pref', 'neuron_weights',
                                       'relational_weights_Q', 'value_weights', 'token_routing']
                            print(f"    All keys in attn_info: {list(attn_info.keys())}")

                            for key in all_keys:
                                if key in attn_info:
                                    v = attn_info[key]
                                    if isinstance(v, torch.Tensor):
                                        w = v
                                        print(f"\n    {key}: shape={w.shape}, dtype={w.dtype}")
                                        print(f"      min={w.min():.6f}, max={w.max():.6f}, mean={w.mean():.6f}")
                                        print(f"      non-zero count: {(w != 0).sum().item()} / {w.numel()}")

                                        # Print first few values
                                        if w.dim() == 3:  # [B, S, N]
                                            print(f"      Sample [0, 0, :10]: {[f'{x:.4f}' for x in w[0, 0, :10].tolist()]}")
                                            print(f"      Sample [0, 1, :10]: {[f'{x:.4f}' for x in w[0, 1, :10].tolist()]}")
                                        elif w.dim() == 2:  # [B, N] or [S, N]
                                            print(f"      Sample [0, :10]: {[f'{x:.4f}' for x in w[0, :10].tolist()]}")
                                            # Check how many neurons are non-zero per batch
                                            nonzero_per_batch = (w != 0).sum(dim=1)
                                            print(f"      Non-zero neurons per batch: {nonzero_per_batch.tolist()}")
                                    else:
                                        print(f"\n    {key}: {v}")

                    # Check ffn routing
                    if 'ffn' in info:
                        ffn_info = info['ffn']
                        print(f"  ffn keys: {list(ffn_info.keys()) if isinstance(ffn_info, dict) else type(ffn_info)}")

                        if isinstance(ffn_info, dict):
                            for key in ['feature_weights', 'neuron_weights', 'weights', 'routing_weights']:
                                if key in ffn_info:
                                    w = ffn_info[key]
                                    print(f"    {key}: shape={w.shape}, min={w.min():.6f}, max={w.max():.6f}, mean={w.mean():.6f}")
                else:
                    print(f"Not a dict: {type(info)}")
        else:
            print("No routing info in list!")

    print("\n" + "="*60)
    print("Debug complete")
    print("="*60)


if __name__ == '__main__':
    main()
