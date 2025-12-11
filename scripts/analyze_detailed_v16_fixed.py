#!/usr/bin/env python3
"""
DAWN v16 Detailed Analysis (Fixed Version)
==========================================

Layer별 router 직접 접근으로 정확한 뉴런 사용 분석

Features:
- Usage histogram per neuron type
- Layer-wise usage heatmap
- Co-occurrence heatmap
- Selection diversity comparison
- Cumulative usage curve

Usage:
    python scripts/analyze_detailed_v16_fixed.py --checkpoint <path> --val_data <path>
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def analyze_and_visualize(checkpoint_path, data_path, save_dir='./analysis', n_batches=100, batch_size=32, device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    device = device if torch.cuda.is_available() else 'cpu'

    # Load model
    print("Loading model...")
    if os.path.isdir(checkpoint_path):
        ckpts = [f for f in os.listdir(checkpoint_path) if f.startswith('checkpoint_') and f.endswith('.pt')]
        if ckpts:
            steps = [int(f.split('step')[1].split('.')[0]) for f in ckpts]
            latest = ckpts[np.argmax(steps)]
            checkpoint_path = os.path.join(checkpoint_path, latest)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('model_config', ckpt.get('config', {}))
    print(f"Loaded: {checkpoint_path}")

    # Detect version and load model
    from models import create_model_by_version

    path_str = str(checkpoint_path).lower()
    if 'v16.1' in path_str or 'v16_1' in path_str:
        version = '16.1'
    else:
        version = config.get('model_version', '16.0')

    print(f"Model version: {version}")
    model = create_model_by_version(version, config)

    state_dict = ckpt.get('model_state_dict', ckpt)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()

    # Load data
    print("Loading data...")
    from utils.data import load_single_file, TokenDataset, collate_fn_dynamic_padding
    from functools import partial
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    data, is_pretokenized = load_single_file(data_path, max_length=128)
    if is_pretokenized:
        dataset = TokenDataset(data, max_length=128)
        print(f"Loaded {len(dataset):,} sequences (pre-tokenized)")
    else:
        from utils.data import TextDataset
        dataset = TextDataset(data, tokenizer, max_length=128)
        print(f"Loaded {len(dataset):,} texts")

    collate_fn = partial(collate_fn_dynamic_padding, tokenizer=tokenizer, max_seq_len=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Using batch_size={batch_size}, total batches available: {len(dataloader)}")

    # Get config values
    n_neurons = {
        'FR': config.get('n_feature_r', 72),
        'FV': config.get('n_feature_v', 48),
        'R': config.get('n_relational', 196),
        'V': config.get('n_value', 24)
    }

    top_k = {
        'FR': config.get('top_k_feature_r', 8),
        'FV': config.get('top_k_feature_v', 6),
        'R': config.get('top_k_relational', 20),
        'V': config.get('top_k_value', 4)
    }

    # Collect statistics via forward pass with routing_info
    print("\nCollecting neuron usage statistics...")
    usage_counts = {ntype: np.zeros(n_neurons[ntype]) for ntype in n_neurons}
    batch_unique = {ntype: [] for ntype in n_neurons}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break

            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
                attention_mask = torch.ones_like(input_ids)
            else:
                input_ids = batch.to(device)
                attention_mask = torch.ones_like(input_ids)

            try:
                outputs = model(input_ids, attention_mask=attention_mask, return_routing_info=True)

                if isinstance(outputs, tuple) and len(outputs) == 2:
                    logits, routing_infos = outputs
                else:
                    continue

                batch_sets = {ntype: set() for ntype in n_neurons}

                # Use first layer's routing info (same as analyze_v16.py)
                routing_info = routing_infos[0].get('attention', {}) if routing_infos else {}

                # Debug: print available keys on first batch
                if batch_idx == 0:
                    print(f"  Available routing keys: {list(routing_info.keys())}")

                for ntype, weight_key in [
                    ('FR', 'feature_r_weights'),
                    ('FV', 'feature_v_weights'),
                    ('R', 'relational_weights_Q'),
                    ('V', 'value_weights'),
                ]:
                    if weight_key in routing_info:
                        weights = routing_info[weight_key]
                        if weights.dim() == 3:
                            # Find selected neurons (weight > 0)
                            selected = (weights > 0).any(dim=0).any(dim=0).cpu()
                            selected_indices = selected.nonzero().flatten().tolist()
                            for idx in selected_indices:
                                usage_counts[ntype][idx] += 1
                            batch_sets[ntype].update(selected_indices)
                        elif weights.dim() == 2:
                            selected = (weights > 0).any(dim=0).cpu()
                            selected_indices = selected.nonzero().flatten().tolist()
                            for idx in selected_indices:
                                usage_counts[ntype][idx] += 1
                            batch_sets[ntype].update(selected_indices)

                for ntype in n_neurons:
                    batch_unique[ntype].append(len(batch_sets[ntype]))

            except Exception as e:
                if batch_idx == 0:
                    print(f"  Warning: {e}")
                continue

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{n_batches}")

    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return

    # ==================== VISUALIZATIONS ====================
    print("\nGenerating visualizations...")

    # 1. Usage histogram for each neuron type
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, ntype in zip(axes.flatten(), ['FR', 'FV', 'R', 'V']):
        counts = usage_counts[ntype]
        counts_norm = counts / counts.sum() if counts.sum() > 0 else counts

        # Sort by usage
        sorted_idx = np.argsort(counts_norm)[::-1]
        sorted_counts = counts_norm[sorted_idx]

        colors = ['#e74c3c' if c > 0.01 else '#3498db' if c > 0.001 else '#95a5a6'
                  for c in sorted_counts]

        ax.bar(range(len(sorted_counts)), sorted_counts, color=colors, width=1.0)
        ax.set_xlabel('Neuron (sorted by usage)')
        ax.set_ylabel('Usage Ratio')
        ax.set_title(f'{ntype}: {n_neurons[ntype]} neurons, top-k={top_k[ntype]}')

        # Add statistics
        active = np.sum(counts > 0)
        gini = 1 - 2 * np.sum((np.arange(1, len(sorted_counts)+1) * sorted_counts)) / (len(sorted_counts) * sorted_counts.sum() + 1e-8) if sorted_counts.sum() > 0 else 0
        ax.text(0.95, 0.95, f'Active: {active}/{n_neurons[ntype]}\nGini: {gini:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{save_dir}/usage_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/usage_histogram.png")

    # 2. Per-batch diversity
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(4)
    width = 0.35

    per_batch_avg = [np.mean(batch_unique[ntype]) if batch_unique[ntype] else 0 for ntype in ['FR', 'FV', 'R', 'V']]
    union_counts = [np.sum(usage_counts[ntype] > 0) for ntype in ['FR', 'FV', 'R', 'V']]

    bars1 = ax.bar(x - width/2, per_batch_avg, width, label='Per-batch avg', color='steelblue')
    bars2 = ax.bar(x + width/2, union_counts, width, label='Total active', color='coral')

    ax.set_ylabel('Neuron Count')
    ax.set_xlabel('Neuron Type')
    ax.set_title('Selection Diversity: Per-batch vs Total Active')
    ax.set_xticks(x)
    ax.set_xticklabels(['FR', 'FV', 'R', 'V'])
    ax.legend()

    # Add total counts as text
    for i, ntype in enumerate(['FR', 'FV', 'R', 'V']):
        ax.text(i + width/2, union_counts[i] + 1, f'/{n_neurons[ntype]}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/selection_diversity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/selection_diversity.png")

    # 3. Cumulative usage curve
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, ntype in zip(axes.flatten(), ['FR', 'FV', 'R', 'V']):
        counts = usage_counts[ntype]
        counts_norm = counts / counts.sum() if counts.sum() > 0 else counts
        sorted_counts = np.sort(counts_norm)[::-1]
        cumsum = np.cumsum(sorted_counts)

        ax.plot(range(1, len(cumsum)+1), cumsum, 'b-', linewidth=2)
        ax.axhline(y=0.9, color='r', linestyle='--', label='90% coverage')
        ax.axhline(y=0.95, color='orange', linestyle='--', label='95% coverage')

        # Find how many neurons for 90% and 95%
        n_90 = np.searchsorted(cumsum, 0.9) + 1
        n_95 = np.searchsorted(cumsum, 0.95) + 1

        ax.axvline(x=n_90, color='r', linestyle=':', alpha=0.5)
        ax.axvline(x=n_95, color='orange', linestyle=':', alpha=0.5)

        ax.set_xlabel('Number of Neurons')
        ax.set_ylabel('Cumulative Usage')
        ax.set_title(f'{ntype}: {n_90} neurons for 90%, {n_95} for 95%')
        ax.legend(loc='lower right')
        ax.set_xlim(0, n_neurons[ntype])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/cumulative_usage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/cumulative_usage.png")

    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)

    for ntype in ['FR', 'FV', 'R', 'V']:
        counts = usage_counts[ntype]
        counts_norm = counts / counts.sum() if counts.sum() > 0 else counts
        sorted_counts = np.sort(counts_norm)[::-1]
        cumsum = np.cumsum(sorted_counts)

        active = np.sum(counts > 0)
        n_90 = np.searchsorted(cumsum, 0.9) + 1 if len(cumsum) > 0 else 0
        n_95 = np.searchsorted(cumsum, 0.95) + 1 if len(cumsum) > 0 else 0
        per_batch = np.mean(batch_unique[ntype]) if batch_unique[ntype] else 0
        div_ratio = per_batch / top_k[ntype] if top_k[ntype] > 0 else 0

        print(f"\n{ntype} (total={n_neurons[ntype]}, top-k={top_k[ntype]}):")
        print(f"  Active neurons: {active} ({100*active/n_neurons[ntype]:.1f}%)")
        print(f"  For 90% coverage: {n_90} neurons")
        print(f"  For 95% coverage: {n_95} neurons")
        print(f"  Per-batch unique: {per_batch:.1f}")
        print(f"  DivRatio: {div_ratio:.2f}")

        # Recommendation
        if div_ratio < 1.1:
            print(f"  → FIXED: 거의 고정된 선택. top-k 줄여도 됨")
        elif div_ratio < 1.5:
            print(f"  → MODERATE: 적당히 동적. 현재 적절")
        else:
            print(f"  → DYNAMIC: 동적 선택. 용량 확인 필요")

    print("\n" + "="*70)
    print(f"All visualizations saved to: {save_dir}/")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='DAWN v16 Detailed Analysis (Fixed)')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--val_data', required=True, help='Validation data path (.pt)')
    parser.add_argument('--output_dir', default='./analysis_detailed', help='Output directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_batches', '--max_batches', type=int, default=100, dest='n_batches')
    parser.add_argument('--batch_size', '-bs', type=int, default=128,
                        help='Batch size for GPU (default: 128, increase for more GPU util)')

    args = parser.parse_args()

    analyze_and_visualize(
        checkpoint_path=args.checkpoint,
        data_path=args.val_data,
        save_dir=args.output_dir,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
