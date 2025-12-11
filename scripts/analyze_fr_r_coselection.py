#!/usr/bin/env python3
"""
DAWN v16 FR-R Co-selection Analysis
=====================================
FR(Feature-R) 뉴런과 R(Relational) 뉴런의 공동 선택 패턴 분석

Usage:
    python analyze_fr_r_coselection.py --checkpoint path/to/checkpoint.pt --val_data path/to/val.pt
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


def analyze_coselection(model, dataloader, device, config, max_batches=50):
    """Analyze FR-R co-selection patterns."""

    n_feature_r = config.get('n_feature_r', 96)
    n_relational = config.get('n_relational', 96)

    print(f"\n{'='*60}")
    print("FR-R Co-selection Analysis")
    print(f"{'='*60}")
    print(f"  n_feature_r: {n_feature_r}")
    print(f"  n_relational: {n_relational}")

    # Co-selection matrix
    co_selection = torch.zeros(n_feature_r, n_relational, device=device)

    # Individual selection counts
    fr_counts = torch.zeros(n_feature_r, device=device)
    r_counts = torch.zeros(n_relational, device=device)

    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)

            B = input_ids.shape[0]
            total_samples += B

            # Forward with routing info
            output = model(input_ids, return_routing_info=True)
            if not isinstance(output, tuple) or len(output) < 2:
                print("  Warning: Model doesn't return routing_info")
                continue

            routing_infos = output[-1]

            # Use layer 0 (can extend to all layers)
            if len(routing_infos) == 0 or 'attention' not in routing_infos[0]:
                continue

            attn = routing_infos[0]['attention']

            # Get weights (soft selection weights after softmax)
            fr_weights = attn.get('feature_r_weights')  # [B, n_feature_r] or [B, S, n_feature_r]
            r_weights = attn.get('relational_weights_Q')  # [B, n_relational] or [B, S, n_relational]

            if fr_weights is None or r_weights is None:
                # Try preference tensors instead
                fr_weights = attn.get('feature_r_pref')
                r_weights = attn.get('relational_q_pref')

                if fr_weights is None or r_weights is None:
                    print(f"  Warning: No FR/R weights found in routing_info")
                    print(f"  Available keys: {list(attn.keys())}")
                    continue

            # Handle different tensor shapes
            if fr_weights.dim() == 3:
                # [B, S, N] -> average over sequence
                fr_weights = fr_weights.mean(dim=1)  # [B, N]
            if r_weights.dim() == 3:
                r_weights = r_weights.mean(dim=1)  # [B, N]

            # Binary selection (threshold at mean or use top-k)
            # Using threshold: selected if weight > uniform expectation
            fr_threshold = 1.0 / n_feature_r
            r_threshold = 1.0 / n_relational

            fr_selected = (fr_weights > fr_threshold).float()  # [B, n_feature_r]
            r_selected = (r_weights > r_threshold).float()     # [B, n_relational]

            # Count individual selections
            fr_counts += fr_selected.sum(dim=0)
            r_counts += r_selected.sum(dim=0)

            # Co-occurrence: outer product per batch, then sum
            # co_selection[i, j] = count of times FR_i and R_j both selected
            for b in range(B):
                co_selection += torch.outer(fr_selected[b], r_selected[b])

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{max_batches} batches...")

    print(f"\n  Total samples analyzed: {total_samples}")

    results = analyze_coselection_matrix(co_selection, fr_counts, r_counts, n_feature_r, n_relational)
    results['total_samples'] = total_samples

    return results, co_selection.cpu().numpy()


def analyze_coselection_matrix(co_selection, fr_counts, r_counts, n_feature_r, n_relational):
    """Analyze the co-selection matrix."""

    results = {}

    # Normalize to get joint probability
    total = co_selection.sum()
    if total > 0:
        co_prob = co_selection / total
    else:
        co_prob = co_selection

    # Marginal probabilities
    fr_prob = fr_counts / fr_counts.sum() if fr_counts.sum() > 0 else fr_counts
    r_prob = r_counts / r_counts.sum() if r_counts.sum() > 0 else r_counts

    # Top 20 pairs
    flat_co = co_selection.view(-1)
    top_k = min(20, flat_co.numel())
    top_values, top_indices = torch.topk(flat_co, top_k)

    print(f"\n{'='*60}")
    print("Top 20 FR-R Pairs (by co-selection count)")
    print(f"{'='*60}")

    top_pairs = []
    for i in range(top_k):
        idx = top_indices[i].item()
        fr_idx = idx // n_relational
        r_idx = idx % n_relational
        count = top_values[i].item()
        pct = count / total.item() * 100 if total > 0 else 0

        print(f"  FR_{fr_idx} + R_{r_idx}: {int(count)} ({pct:.2f}%)")
        top_pairs.append({
            'fr_idx': fr_idx,
            'r_idx': r_idx,
            'count': int(count),
            'pct': pct
        })

    results['top_pairs'] = top_pairs

    # Concentration analysis
    print(f"\n{'='*60}")
    print("Pair Concentration Analysis")
    print(f"{'='*60}")

    # What % of co-selections come from top-k pairs?
    cumsum = torch.cumsum(torch.sort(flat_co, descending=True)[0], dim=0)

    top10_pct = (cumsum[9] / total * 100).item() if total > 0 and len(cumsum) > 9 else 0
    top50_pct = (cumsum[49] / total * 100).item() if total > 0 and len(cumsum) > 49 else 0
    top100_pct = (cumsum[99] / total * 100).item() if total > 0 and len(cumsum) > 99 else 0

    print(f"  Top 10 pairs: {top10_pct:.1f}% of all co-selections")
    print(f"  Top 50 pairs: {top50_pct:.1f}% of all co-selections")
    print(f"  Top 100 pairs: {top100_pct:.1f}% of all co-selections")

    # Entropy of co-selection distribution
    co_prob_flat = co_prob.view(-1)
    co_prob_flat = co_prob_flat[co_prob_flat > 0]  # Remove zeros for log
    entropy = -(co_prob_flat * co_prob_flat.log()).sum().item()
    max_entropy = np.log(n_feature_r * n_relational)
    normalized_entropy = entropy / max_entropy

    print(f"\n  Co-selection entropy: {entropy:.2f} (max: {max_entropy:.2f})")
    print(f"  Normalized entropy: {normalized_entropy:.2%}")
    print(f"  Interpretation:")
    if normalized_entropy < 0.5:
        print(f"    → CONCENTRATED: Strong FR-R pairing learned")
    elif normalized_entropy < 0.8:
        print(f"    → MODERATE: Some pairing structure")
    else:
        print(f"    → UNIFORM: Shared space convergence (FR/R independent)")

    results['concentration'] = {
        'top10_pct': top10_pct,
        'top50_pct': top50_pct,
        'top100_pct': top100_pct,
        'entropy': entropy,
        'max_entropy': max_entropy,
        'normalized_entropy': normalized_entropy
    }

    # FR neuron analysis: which FR neurons have strongest pairing?
    print(f"\n{'='*60}")
    print("FR Neuron Specialization")
    print(f"{'='*60}")

    # For each FR, find its most common R partner
    fr_specialization = []
    for fr_idx in range(n_feature_r):
        row = co_selection[fr_idx]
        if row.sum() > 0:
            top_r = row.argmax().item()
            top_r_pct = (row[top_r] / row.sum() * 100).item()
            fr_specialization.append({
                'fr_idx': fr_idx,
                'top_r': top_r,
                'top_r_pct': top_r_pct,
                'total_count': int(row.sum().item())
            })

    # Sort by specialization (how concentrated on one R)
    fr_specialization.sort(key=lambda x: x['top_r_pct'], reverse=True)

    print(f"  Most specialized FR neurons (strongest R preference):")
    for item in fr_specialization[:10]:
        print(f"    FR_{item['fr_idx']}: {item['top_r_pct']:.1f}% with R_{item['top_r']} (n={item['total_count']})")

    results['fr_specialization'] = fr_specialization[:20]

    # R neuron analysis: which R neurons have strongest pairing?
    print(f"\n{'='*60}")
    print("R Neuron Specialization")
    print(f"{'='*60}")

    r_specialization = []
    for r_idx in range(n_relational):
        col = co_selection[:, r_idx]
        if col.sum() > 0:
            top_fr = col.argmax().item()
            top_fr_pct = (col[top_fr] / col.sum() * 100).item()
            r_specialization.append({
                'r_idx': r_idx,
                'top_fr': top_fr,
                'top_fr_pct': top_fr_pct,
                'total_count': int(col.sum().item())
            })

    r_specialization.sort(key=lambda x: x['top_fr_pct'], reverse=True)

    print(f"  Most specialized R neurons (strongest FR preference):")
    for item in r_specialization[:10]:
        print(f"    R_{item['r_idx']}: {item['top_fr_pct']:.1f}% with FR_{item['top_fr']} (n={item['total_count']})")

    results['r_specialization'] = r_specialization[:20]

    return results


def save_heatmap(co_selection, output_path):
    """Save co-selection heatmap as image."""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 10))
        plt.imshow(co_selection, aspect='auto', cmap='hot')
        plt.colorbar(label='Co-selection count')
        plt.xlabel('R neuron index')
        plt.ylabel('FR neuron index')
        plt.title('FR-R Co-selection Heatmap')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"\n  Heatmap saved to: {output_path}")
    except ImportError:
        print("\n  Warning: matplotlib not available, skipping heatmap")


def main():
    parser = argparse.ArgumentParser(description="DAWN v16 FR-R Co-selection Analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./coselection_analysis")

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

    checkpoint = torch.load(ckpt_path, map_location=args.device)
    config = checkpoint.get('config', {})

    # Check model version for v16.1
    model_version = config.get('model_version', '16.0')
    if model_version == '16.1':
        try:
            from models.model_v16_1 import DAWN
        except ImportError:
            from model_v16_1 import DAWN

    print(f"Model version: {model_version}")
    print(f"Model config: d_model={config.get('d_model')}, n_layers={config.get('n_layers')}")

    model = DAWN(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

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

    # Run analysis
    results, co_matrix = analyze_coselection(model, dataloader, args.device, config, args.max_batches)

    # Save heatmap
    heatmap_path = os.path.join(args.output_dir, "fr_r_coselection_heatmap.png")
    save_heatmap(co_matrix, heatmap_path)

    # Save results
    import json
    output_path = os.path.join(args.output_dir, "fr_r_coselection.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")

    # Save raw matrix
    np.save(os.path.join(args.output_dir, "co_selection_matrix.npy"), co_matrix)
    print(f"Raw matrix saved to: {args.output_dir}/co_selection_matrix.npy")

    print(f"\n{'='*60}")
    print("Analysis Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
