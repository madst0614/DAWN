#!/usr/bin/env python3
"""
DAWN v16 Routing Diagnostics
=============================
체크포인트로 routing 상태 진단:
1. Usage EMA 분포
2. 실제 뉴런 선택 빈도
3. Importance 집중도 (entropy)

Usage:
    python diagnose_v16_routing.py --checkpoint path/to/checkpoint.pt --val_data path/to/val.pt
"""

import argparse
import sys
import os
import torch
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_checkpoint_path(path):
    """Resolve path to actual checkpoint file"""
    import glob
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        patterns = ["*.pt", "*.pth"]
        all_ckpts = []
        for pattern in patterns:
            all_ckpts.extend(glob.glob(os.path.join(path, pattern)))
            all_ckpts.extend(glob.glob(os.path.join(path, "**", pattern), recursive=True))
        if all_ckpts:
            for ckpt in all_ckpts:
                name = os.path.basename(ckpt).lower()
                if 'best' in name or 'final' in name:
                    return ckpt
            all_ckpts.sort(key=os.path.getmtime, reverse=True)
            return all_ckpts[0]
    raise FileNotFoundError(f"No checkpoint found: {path}")


def diagnose_usage_ema(model):
    """1. Usage EMA 분포 확인"""
    print(f"\n{'='*60}")
    print("1. Usage EMA Distribution")
    print(f"{'='*60}")

    results = {}

    if not hasattr(model, 'global_routers'):
        print("  Model doesn't have global_routers")
        return results

    router = model.global_routers.neuron_router

    ema_list = [
        ('FR', getattr(router, 'usage_ema_feature_r', None)),
        ('FV', getattr(router, 'usage_ema_feature_v', None)),
        ('R', getattr(router, 'usage_ema_relational', None)),
        ('V', getattr(router, 'usage_ema_value', None)),
    ]

    for name, ema in ema_list:
        if ema is None:
            print(f"  {name}: Not found")
            continue

        usage = ema.cpu().numpy()
        dead_count = int((usage < 0.01).sum())

        print(f"\n  {name} ({len(usage)} neurons):")
        print(f"    max={usage.max():.4f} (idx {usage.argmax()})")
        print(f"    min={usage.min():.4f} (idx {usage.argmin()})")
        print(f"    mean={usage.mean():.4f}, std={usage.std():.4f}")
        print(f"    dead (<0.01): {dead_count}/{len(usage)} ({100*dead_count/len(usage):.1f}%)")

        # Top 5 most used
        top5_idx = np.argsort(usage)[-5:][::-1]
        print(f"    Top 5: {[(int(i), f'{usage[i]:.4f}') for i in top5_idx]}")

        results[name] = {
            'n_neurons': len(usage),
            'max': float(usage.max()),
            'max_idx': int(usage.argmax()),
            'min': float(usage.min()),
            'mean': float(usage.mean()),
            'std': float(usage.std()),
            'dead_count': dead_count,
            'dead_pct': 100 * dead_count / len(usage),
            'top5': [(int(i), float(usage[i])) for i in top5_idx]
        }

    return results


def diagnose_neuron_selection(model, dataloader, device, max_batches=20):
    """2. 실제 뉴런 선택 빈도 확인"""
    print(f"\n{'='*60}")
    print("2. Neuron Selection Frequency (from routing)")
    print(f"{'='*60}")

    neuron_counts = {
        'FR': Counter(),
        'FV': Counter(),
        'R': Counter(),
        'V': Counter(),
    }

    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)

            B, L = input_ids.shape
            total_tokens += B * L

            # Forward with routing info
            output = model(input_ids, return_routing_info=True)
            if not isinstance(output, tuple) or len(output) < 2:
                print("  Model doesn't return routing_info")
                return

            routing_infos = output[-1]

            # Aggregate neuron selections across layers
            for layer_info in routing_infos:
                if 'attention' not in layer_info:
                    continue

                attn = layer_info['attention']

                # Get preferences and find top-k selections
                pref_map = {
                    'FR': attn.get('feature_r_pref'),
                    'FV': attn.get('feature_v_pref'),
                    'R': attn.get('relational_q_pref'),
                    'V': attn.get('value_pref'),
                }

                topk_map = {'FR': 8, 'FV': 8, 'R': 4, 'V': 6}

                for nt, pref in pref_map.items():
                    if pref is None:
                        continue
                    k = min(topk_map.get(nt, 4), pref.shape[-1])
                    _, topk_idx = torch.topk(pref, k, dim=-1)

                    # Count selections (vectorized)
                    flat_idx = topk_idx.view(-1).cpu().numpy()
                    unique, counts = np.unique(flat_idx, return_counts=True)
                    for idx, cnt in zip(unique, counts):
                        neuron_counts[nt][int(idx)] += int(cnt)

    print(f"\n  Analyzed {max_batches} batches, {total_tokens} tokens")

    results = {'total_tokens': total_tokens}

    for nt, counts in neuron_counts.items():
        if not counts:
            print(f"\n  {nt}: No data")
            continue

        top10 = counts.most_common(10)
        total_selections = sum(counts.values())

        print(f"\n  {nt} - Top 10 neurons (total selections: {total_selections}):")
        for idx, count in top10:
            pct = 100 * count / total_selections
            print(f"    neuron {idx}: {count} ({pct:.1f}%)")

        # Coverage: how many neurons cover 50% of selections?
        sorted_counts = sorted(counts.values(), reverse=True)
        cumsum = np.cumsum(sorted_counts)
        n_for_50 = int(np.searchsorted(cumsum, total_selections * 0.5) + 1)
        n_for_90 = int(np.searchsorted(cumsum, total_selections * 0.9) + 1)
        print(f"    Coverage: {n_for_50} neurons for 50%, {n_for_90} neurons for 90%")

        results[nt] = {
            'total_selections': total_selections,
            'top10': [(int(idx), int(cnt)) for idx, cnt in top10],
            'n_for_50_coverage': n_for_50,
            'n_for_90_coverage': n_for_90,
            'n_active_neurons': len(counts)
        }

    return results


def diagnose_importance_entropy(model, dataloader, tokenizer, device, max_batches=10):
    """3. Importance 집중도 (entropy) 확인"""
    print(f"\n{'='*60}")
    print("3. Importance Entropy Analysis")
    print(f"{'='*60}")

    all_entropies = []
    sample_outputs = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)

            B, L = input_ids.shape

            output = model(input_ids, return_routing_info=True)
            if not isinstance(output, tuple) or len(output) < 2:
                continue

            routing_infos = output[-1]

            for layer_idx, layer_info in enumerate(routing_infos):
                if 'attention' not in layer_info:
                    continue

                attn = layer_info['attention']

                # Check for importance tensor
                importance = attn.get('importance')
                if importance is None:
                    # Try to compute from preferences
                    pref = attn.get('relational_q_pref')
                    if pref is not None:
                        # Normalize to get importance-like distribution
                        importance = torch.softmax(pref, dim=-1)

                if importance is not None:
                    # Entropy: -sum(p * log(p))
                    # importance shape: [B, L, N] or similar
                    if importance.dim() >= 2:
                        p = importance.float()
                        p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
                        entropy = -(p * (p + 1e-8).log()).sum(dim=-1)
                        all_entropies.append(entropy.mean().item())

                        # Sample top tokens for first batch
                        if batch_idx == 0 and layer_idx == 0 and len(sample_outputs) < 5:
                            # Find tokens with highest importance concentration (lowest entropy)
                            batch_entropy = entropy[0]  # First sequence
                            top5_pos = torch.argsort(batch_entropy)[:5]  # Lowest entropy

                            tokens = input_ids[0].cpu().tolist()
                            for pos in top5_pos:
                                pos = pos.item()
                                if pos < len(tokens):
                                    tok_str = tokenizer.decode([tokens[pos]])
                                    sample_outputs.append({
                                        'pos': pos,
                                        'token': tok_str,
                                        'entropy': batch_entropy[pos].item()
                                    })

    results = {}

    if all_entropies:
        mean_entropy = np.mean(all_entropies)
        std_entropy = np.std(all_entropies)

        print(f"\n  Mean entropy: {mean_entropy:.4f} (std: {std_entropy:.4f})")
        print(f"  (Lower = more concentrated routing)")

        # Max entropy reference
        # For uniform distribution over N neurons: log(N)
        print(f"\n  Reference: uniform over 100 neurons → entropy ≈ {np.log(100):.2f}")
        print(f"  Reference: uniform over 10 neurons → entropy ≈ {np.log(10):.2f}")

        results['mean_entropy'] = float(mean_entropy)
        results['std_entropy'] = float(std_entropy)
        results['ref_uniform_100'] = float(np.log(100))
        results['ref_uniform_10'] = float(np.log(10))

    if sample_outputs:
        print(f"\n  Sample tokens with lowest entropy (most concentrated):")
        for s in sample_outputs[:5]:
            print(f"    pos {s['pos']}: '{s['token'].strip()}' (entropy={s['entropy']:.4f})")

        results['sample_low_entropy_tokens'] = [
            {'pos': s['pos'], 'token': s['token'].strip(), 'entropy': s['entropy']}
            for s in sample_outputs[:5]
        ]

    return results


def main():
    parser = argparse.ArgumentParser(description="DAWN v16 Routing Diagnostics")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./routing_diagnosis")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve checkpoint path
    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")

    # Load model
    try:
        from models.model_v16 import DAWN
    except ImportError:
        from model_v16 import DAWN

    from transformers import BertTokenizer

    checkpoint = torch.load(ckpt_path, map_location=args.device)
    config = checkpoint.get('config', {})

    print(f"Model config: d_model={config.get('d_model')}, n_layers={config.get('n_layers')}")

    model = DAWN(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load data
    print(f"Loading data: {args.val_data}")
    val_data = torch.load(args.val_data)
    if isinstance(val_data, dict):
        input_ids = val_data.get('input_ids', val_data.get('tokens'))
    else:
        input_ids = val_data

    if input_ids.dim() == 1:
        seq_len = config.get('max_seq_len', 512)
        n_seqs = input_ids.shape[0] // seq_len
        input_ids = input_ids[:n_seqs * seq_len].view(n_seqs, seq_len)

    dataset = torch.utils.data.TensorDataset(input_ids)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\n{'='*60}")
    print("DAWN v16 Routing Diagnostics")
    print(f"{'='*60}")

    results = {}

    # 1. Usage EMA
    results['usage_ema'] = diagnose_usage_ema(model)

    # 2. Neuron Selection
    results['neuron_selection'] = diagnose_neuron_selection(model, dataloader, args.device, args.max_batches)

    # 3. Importance Entropy
    results['importance_entropy'] = diagnose_importance_entropy(model, dataloader, tokenizer, args.device, args.max_batches)

    # Save results
    import json
    output_path = os.path.join(args.output_dir, "routing_diagnosis.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    print(f"\n{'='*60}")
    print("Diagnostics Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
