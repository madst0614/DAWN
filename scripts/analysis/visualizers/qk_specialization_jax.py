#!/usr/bin/env python3
"""
Q/K Specialization Analysis (JAX/TPU) — Section 4.1
=====================================================
Analyze Q vs K neuron usage patterns from DAWN 400M checkpoint.

Outputs per QK pool (feature_qk, restore_qk):
  - Scatter plot: Q usage vs K usage per neuron
  - Correlation coefficient (Pearson)
  - Specialization ratio: Q-only / K-only / shared neuron counts

Designed for single-host TPU v4-8.

Usage:
    python scripts/analysis/visualizers/qk_specialization_jax.py \
        --checkpoint gs://dawn-tpu-data-c4/checkpoints/dawn_v17_1_400M_c4_20B_v4_32/run_v17.1_20260210_160828_3201 \
        --output ./section4_results \
        --n_batches 100 --batch_size 32 --seq_len 512
"""

import sys
import os
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from scripts.analysis.utils_jax import (
    load_model_jax, create_model_from_config, load_val_data_jax,
    create_batches, JAXRoutingDataExtractor, JAXRoutingData,
    QK_POOLS, convert_to_serializable, save_results,
)

# Inline style (avoids __init__.py torch import chain)
PAPER_STYLE = {
    'font_family': 'serif', 'font_size_base': 10, 'font_size_label': 14,
    'font_size_subtitle': 14, 'font_size_tick': 11, 'font_size_legend': 10,
    'font_size_annotation': 12, 'font_size_category': 14,
    'axes_linewidth': 0.8, 'spines_top': False, 'spines_right': False,
}
S = PAPER_STYLE

if HAS_MATPLOTLIB:
    plt.rcParams.update({
        'font.family': S['font_family'], 'font.size': S['font_size_base'],
        'axes.linewidth': S['axes_linewidth'],
        'axes.spines.top': S['spines_top'], 'axes.spines.right': S['spines_right'],
        'axes.labelsize': S['font_size_label'],
        'xtick.labelsize': S['font_size_tick'], 'ytick.labelsize': S['font_size_tick'],
    })

# Colors (consistent with qk_specialization.py)
COLOR_Q = '#C0392B'
COLOR_K = '#2471A3'
COLOR_SHARED = '#50C878'
COLOR_INACTIVE = '#95A5A6'


# ============================================================
# Analysis
# ============================================================

def analyze_qk_specialization(
    model_cls, params, config, val_tokens,
    n_batches=100, batch_size=32, seq_len=512,
):
    """Compute per-neuron Q/K selection counts for each QK pool.

    Returns dict keyed by pool name with q_counts, k_counts, correlation,
    specialization breakdown, etc.
    """
    model_instance = create_model_from_config(config)
    extractor = JAXRoutingDataExtractor(model_instance, params, config)

    results = {}

    for pool_name, pool_info in QK_POOLS.items():
        n_attr = pool_info['n_attr']
        n_neurons = config.get(n_attr, 0)
        if n_neurons == 0:
            continue

        q_counts = np.zeros(n_neurons, dtype=np.float64)
        k_counts = np.zeros(n_neurons, dtype=np.float64)

        if pool_name == 'feature_qk':
            std_q_key, std_k_key = 'fqk_q', 'fqk_k'
        else:
            std_q_key, std_k_key = 'rqk_q', 'rqk_k'

        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        batch_overlaps = []

        for batch in tqdm(batches, desc=f'{pool_info["display"]} Q/K'):
            input_ids = np.array(batch)
            routing_info = extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)

            w_q = routing.get_weight(std_q_key)
            w_k = routing.get_weight(std_k_key)
            if w_q is None or w_k is None:
                continue

            if w_q.ndim == 3:
                q_counts += (w_q > 0).astype(float).sum(axis=(0, 1))
                k_counts += (w_k > 0).astype(float).sum(axis=(0, 1))
            else:
                q_counts += (w_q > 0).astype(float).sum(axis=0)
                k_counts += (w_k > 0).astype(float).sum(axis=0)

            # Batch overlap (matches GPU routing_jax.py)
            if w_q.ndim >= 2:
                overlap = ((w_q > 0) & (w_k > 0)).astype(float)
                active_q = (w_q > 0).astype(float).sum(axis=-1)
                overlap_ratio = (overlap.sum(axis=-1) / (active_q + 1e-8)).mean()
                batch_overlaps.append(float(overlap_ratio))

        # Correlation
        if q_counts.sum() > 0 and k_counts.sum() > 0:
            corr = float(np.corrcoef(q_counts, k_counts)[0, 1])
        else:
            corr = 0.0

        # Specialization ratio
        total_usage = q_counts + k_counts
        q_ratio = np.zeros_like(q_counts)
        valid = total_usage > 0
        q_ratio[valid] = q_counts[valid] / total_usage[valid]

        q_specialized = int((q_ratio > 0.7).sum())
        k_specialized = int((q_ratio < 0.3).sum())
        shared = int(((q_ratio >= 0.3) & (q_ratio <= 0.7)).sum())
        inactive = int((~valid).sum())

        # Sensitivity analysis (matches GPU routing_jax.py)
        sensitivity_thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]
        sensitivity_analysis = {}
        for t in sensitivity_thresholds:
            q_spec = int((q_ratio > t).sum())
            k_spec = int((q_ratio < (1 - t)).sum())
            shared_t = int(((q_ratio >= (1 - t)) & (q_ratio <= t)).sum())
            sensitivity_analysis[str(t)] = {
                'q_specialized': q_spec, 'k_specialized': k_spec,
                'shared': shared_t, 'total': n_neurons,
            }

        results[pool_name] = {
            'display': pool_info['display'],
            'n_neurons': n_neurons,
            'q_counts': q_counts.tolist(),
            'k_counts': k_counts.tolist(),
            'correlation': corr,
            'avg_overlap': float(np.mean(batch_overlaps)) if batch_overlaps else 0,
            'std_overlap': float(np.std(batch_overlaps)) if batch_overlaps else 0,
            'q_specialized': q_specialized,
            'k_specialized': k_specialized,
            'shared': shared,
            'inactive': inactive,
            'q_total': int(q_counts.sum()),
            'k_total': int(k_counts.sum()),
            'q_ratio': q_ratio.tolist(),
            'specialization_thresholds': {'q_specialized': 0.7, 'k_specialized': 0.3},
            'sensitivity_analysis': sensitivity_analysis,
        }

    results['meta'] = {
        'n_batches': len(batches) if 'batches' in dir() else n_batches,
        'batch_size': batch_size,
        'seq_len': seq_len,
    }
    return results


# ============================================================
# Visualization
# ============================================================

def plot_qk_scatter(results, output_dir, dpi=300):
    """Generate Q vs K usage scatter + specialization bar for each pool."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")
        return []

    saved = []

    for pool_name in ['feature_qk', 'restore_qk']:
        data = results.get(pool_name)
        if data is None:
            continue

        q_counts = np.array(data['q_counts'])
        k_counts = np.array(data['k_counts'])
        total = q_counts + k_counts
        valid = total > 0
        q_ratio = np.zeros_like(q_counts)
        q_ratio[valid] = q_counts[valid] / total[valid]

        # Classify
        is_q = q_ratio > 0.7
        is_k = q_ratio < 0.3
        is_shared = (q_ratio >= 0.3) & (q_ratio <= 0.7) & valid
        is_inactive = ~valid

        # --- Figure: 2 panels ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: Scatter
        if is_inactive.any():
            ax1.scatter(q_counts[is_inactive], k_counts[is_inactive],
                        c=COLOR_INACTIVE, s=15, alpha=0.4, label='Inactive')
        if is_shared.any():
            ax1.scatter(q_counts[is_shared], k_counts[is_shared],
                        c=COLOR_SHARED, s=20, alpha=0.6, label=f'Shared ({int(is_shared.sum())})')
        if is_q.any():
            ax1.scatter(q_counts[is_q], k_counts[is_q],
                        c=COLOR_Q, s=20, alpha=0.6, label=f'Q-specialized ({int(is_q.sum())})')
        if is_k.any():
            ax1.scatter(q_counts[is_k], k_counts[is_k],
                        c=COLOR_K, s=20, alpha=0.6, label=f'K-specialized ({int(is_k.sum())})')

        # Diagonal
        lim = max(q_counts.max(), k_counts.max()) * 1.05
        ax1.plot([0, lim], [0, lim], '--', color='gray', linewidth=0.8, alpha=0.5)
        ax1.set_xlabel('Q Selection Count', fontsize=S['font_size_label'])
        ax1.set_ylabel('K Selection Count', fontsize=S['font_size_label'])
        ax1.set_title(f'{data["display"]} — Q vs K Usage  (r={data["correlation"]:.3f})',
                       fontsize=S['font_size_subtitle'], fontweight='bold')
        ax1.legend(fontsize=S['font_size_legend'], loc='upper left')

        # Panel 2: Bar chart
        categories = ['Q-only', 'Shared', 'K-only', 'Inactive']
        counts = [data['q_specialized'], data['shared'],
                  data['k_specialized'], data['inactive']]
        colors = [COLOR_Q, COLOR_SHARED, COLOR_K, COLOR_INACTIVE]

        bars = ax2.bar(categories, counts, color=colors, edgecolor='white', linewidth=0.5)
        for bar, c in zip(bars, counts):
            if c > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         str(c), ha='center', fontsize=S['font_size_annotation'])

        ax2.set_ylabel('Number of Neurons', fontsize=S['font_size_label'])
        ax2.set_title(f'{data["display"]} — Specialization Breakdown',
                       fontsize=S['font_size_subtitle'], fontweight='bold')

        plt.tight_layout()
        fname = f'qk_specialization_{pool_name}.png'
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {path}")
        saved.append(path)

    return saved


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Q/K Specialization Analysis (Section 4.1)')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path (local or gs://)')
    parser.add_argument('--output', default='./section4_results', help='Output directory')
    parser.add_argument('--val_data', default=None, help='Validation data path (.bin)')
    parser.add_argument('--n_batches', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model_cls, params, config = load_model_jax(args.checkpoint)
    print(f"  d_model={config.get('d_model')}, n_layers={config.get('n_layers')}, "
          f"n_feature_qk={config.get('n_feature_qk')}, n_restore_qk={config.get('n_restore_qk')}")

    # Load validation data (or generate random for testing)
    if args.val_data:
        print(f"Loading validation data: {args.val_data}")
        val_tokens = load_val_data_jax(args.val_data)
    else:
        print("No --val_data provided, using random tokens for testing")
        vocab_size = config.get('vocab_size', 30522)
        n_tokens = args.n_batches * args.batch_size * args.seq_len
        val_tokens = np.random.randint(1, vocab_size, size=n_tokens, dtype=np.int32)

    # Run analysis
    print("\n=== Q/K Specialization Analysis ===")
    results = analyze_qk_specialization(
        model_cls, params, config, val_tokens,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    # Save JSON
    json_path = os.path.join(args.output, 'qk_specialization.json')
    save_results(results, json_path)
    print(f"  Saved: {json_path}")

    # Print summary
    for pool_name in ['feature_qk', 'restore_qk']:
        d = results.get(pool_name)
        if d is None:
            continue
        print(f"\n  {d['display']} (n={d['n_neurons']}):")
        print(f"    Correlation:    r = {d['correlation']:.4f}")
        print(f"    Q-specialized:  {d['q_specialized']}")
        print(f"    K-specialized:  {d['k_specialized']}")
        print(f"    Shared:         {d['shared']}")
        print(f"    Inactive:       {d['inactive']}")

    # Plot
    plot_qk_scatter(results, args.output, dpi=args.dpi)

    print("\nDone.")


if __name__ == '__main__':
    main()
