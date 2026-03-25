#!/usr/bin/env python3
"""
Dead Neuron Weight Norm Analysis
=================================
Uses extract_full_routing to run full forward passes and compute
per-neuron activation frequency across all layers. Compares dead vs active
neurons by weight norm and embedding norm.

Generates scatter plots: x=weight norm, y=activation frequency per pool.

Usage:
    python scripts/analysis/visualizers/dead_neuron_analysis.py \
        --checkpoint gs://dawn-tpu-data-c4/checkpoints/... \
        --output ./results/dead_neurons

    # Or called from analyze_all_jax.py:
    from scripts.analysis.visualizers.dead_neuron_analysis import (
        analyze_dead_neuron_norms, plot_dead_neuron_scatter,
    )
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from typing import Dict, Optional

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

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
    extract_full_routing, create_batches, convert_to_serializable,
)

# Paper style
PAPER_STYLE = {
    'font_family': 'serif', 'font_size_base': 10, 'font_size_label': 14,
    'font_size_tick': 11, 'font_size_legend': 10,
    'axes_linewidth': 0.8, 'spines_top': False, 'spines_right': False,
}
if HAS_MATPLOTLIB:
    S = PAPER_STYLE
    plt.rcParams.update({
        'font.family': S['font_family'], 'font.size': S['font_size_base'],
        'axes.linewidth': S['axes_linewidth'],
        'axes.spines.top': S['spines_top'], 'axes.spines.right': S['spines_right'],
        'axes.labelsize': S['font_size_label'],
        'xtick.labelsize': S['font_size_tick'], 'ytick.labelsize': S['font_size_tick'],
    })


# Pool name -> (param_key, slice_fn) for extracting neuron parameters
POOL_PARAM_MAP = {
    'fqk_q': ('f_neurons', 'feature_qk'),
    'fqk_k': ('f_neurons', 'feature_qk'),
    'fv': ('f_neurons', 'feature_v'),
    'rqk_q': ('r_neurons', 'restore_qk'),
    'rqk_k': ('r_neurons', 'restore_qk'),
    'rv': ('r_neurons', 'restore_v'),
    'fknow': ('feature_know', 'feature_know'),
    'rknow': ('restore_know', 'restore_know'),
}

POOL_DISPLAY = {
    'fqk_q': 'F-QK (Q)', 'fqk_k': 'F-QK (K)',
    'fv': 'F-V', 'rv': 'R-V',
    'rqk_q': 'R-QK (Q)', 'rqk_k': 'R-QK (K)',
    'fknow': 'F-Know', 'rknow': 'R-Know',
}


def _get_neuron_norms(params, config):
    """Extract per-neuron weight Frobenius norms and embedding norms."""
    all_params = params.get('params', params)
    sn = all_params.get('shared_neurons', {})
    router = all_params.get('router', {}).get('neuron_router', {})

    n_fqk = config.get('n_feature_qk', 88)
    n_fv = config.get('n_feature_v', 352)
    n_rqk = config.get('n_restore_qk', 88)
    n_rv = config.get('n_restore_v', 352)

    # Neuron weight norms: shared_neurons contains [N, D, R] or [N, R, D]
    weight_norms = {}
    f_neurons = np.asarray(sn.get('f_neurons', np.array([])))
    r_neurons = np.asarray(sn.get('r_neurons', np.array([])))
    f_know = np.asarray(sn.get('feature_know', np.array([])))
    r_know = np.asarray(sn.get('restore_know', np.array([])))

    if f_neurons.size > 0:
        # f_neurons: [n_fqk + n_fv, D, R]
        norms = np.linalg.norm(f_neurons.reshape(f_neurons.shape[0], -1), axis=1)
        weight_norms['fqk_q'] = norms[:n_fqk]
        weight_norms['fqk_k'] = norms[:n_fqk]
        weight_norms['fv'] = norms[n_fqk:n_fqk + n_fv]

    if r_neurons.size > 0:
        norms = np.linalg.norm(r_neurons.reshape(r_neurons.shape[0], -1), axis=1)
        weight_norms['rqk_q'] = norms[:n_rqk]
        weight_norms['rqk_k'] = norms[:n_rqk]
        weight_norms['rv'] = norms[n_rqk:n_rqk + n_rv]

    if f_know.size > 0:
        weight_norms['fknow'] = np.linalg.norm(f_know.reshape(f_know.shape[0], -1), axis=1)

    if r_know.size > 0:
        weight_norms['rknow'] = np.linalg.norm(r_know.reshape(r_know.shape[0], -1), axis=1)

    # Neuron embedding norms from router
    emb_norms = {}
    neuron_emb = np.asarray(router.get('neuron_emb', np.array([])))
    if neuron_emb.size > 0:
        all_emb_norms = np.linalg.norm(neuron_emb, axis=1)
        # Split by pool
        fqk_end = n_fqk
        fv_end = fqk_end + n_fv
        rqk_end = fv_end + n_rqk
        rv_end = rqk_end + n_rv
        n_fk = config.get('n_feature_know', 224)
        fk_end = rv_end + n_fk

        emb_norms['fqk_q'] = all_emb_norms[:fqk_end]
        emb_norms['fqk_k'] = all_emb_norms[:fqk_end]
        emb_norms['fv'] = all_emb_norms[fqk_end:fv_end]
        emb_norms['rqk_q'] = all_emb_norms[fv_end:rqk_end]
        emb_norms['rqk_k'] = all_emb_norms[fv_end:rqk_end]
        emb_norms['rv'] = all_emb_norms[rqk_end:rv_end]
        emb_norms['fknow'] = all_emb_norms[rv_end:fk_end]
        emb_norms['rknow'] = all_emb_norms[fk_end:]

    return weight_norms, emb_norms


def analyze_dead_neuron_norms(
    params, config, val_tokens,
    n_batches: int = 50, batch_size: int = 4, seq_len: int = 512,
    threshold: float = 0.0,
) -> Dict:
    """Analyze per-neuron activation frequency via full forward routing.

    Args:
        params: Model parameters
        config: Model config dict
        val_tokens: Validation token array
        n_batches: Number of batches to process
        batch_size: Batch size
        seq_len: Sequence length
        threshold: Weight threshold for counting as "active"

    Returns:
        Dict with per-pool activation frequency, weight norms, embedding norms,
        dead/active classification.
    """
    if not HAS_JAX:
        raise RuntimeError('JAX not available')

    batches = create_batches(val_tokens, batch_size, seq_len)
    if n_batches:
        batches = batches[:n_batches]

    n_layers = config.get('n_layers', 16)
    pool_keys = ['fqk_q', 'fqk_k', 'fv', 'rqk_q', 'rqk_k', 'rv', 'fknow', 'rknow']

    # Initialize per-neuron activation counters
    activation_counts = {}
    total_tokens = 0

    print(f"  Running {len(batches)} batches through extract_full_routing...")

    for bi, batch in enumerate(tqdm(batches, desc='Dead Neuron Analysis')):
        input_ids = np.array(batch)
        B, S = input_ids.shape
        total_tokens += B * S

        routing = extract_full_routing(params, config, input_ids)

        for li in range(n_layers):
            layer_data = routing[f'layer_{li}']
            for key in pool_keys:
                w = np.asarray(layer_data[key])  # [B, S, N]
                active = (w > threshold).sum(axis=(0, 1))  # [N]

                if key not in activation_counts:
                    activation_counts[key] = np.zeros_like(active, dtype=np.float64)
                activation_counts[key] += active

    # Normalize: frequency = count / (total_tokens * n_layers)
    total_observations = total_tokens * n_layers
    activation_freq = {
        key: counts / total_observations
        for key, counts in activation_counts.items()
    }

    # Get weight norms and embedding norms
    weight_norms, emb_norms = _get_neuron_norms(params, config)

    # Build per-pool results
    results = {
        'n_batches': len(batches),
        'total_tokens': total_tokens,
        'n_layers': n_layers,
        'threshold': threshold,
        'per_pool': {},
    }

    for key in pool_keys:
        if key not in activation_freq:
            continue

        freq = activation_freq[key]
        n_total = len(freq)
        n_dead = int((freq == 0).sum())
        n_active = n_total - n_dead

        pool_result = {
            'display': POOL_DISPLAY.get(key, key),
            'n_total': n_total,
            'n_dead': n_dead,
            'n_active': n_active,
            'dead_ratio': n_dead / n_total if n_total > 0 else 0,
            'activation_freq': freq.tolist(),
        }

        # Weight norm comparison
        if key in weight_norms:
            wn = weight_norms[key]
            pool_result['weight_norms'] = wn.tolist()
            if n_dead > 0 and n_active > 0:
                dead_mask = freq == 0
                pool_result['dead_weight_norm_mean'] = float(wn[dead_mask].mean())
                pool_result['dead_weight_norm_std'] = float(wn[dead_mask].std())
                pool_result['active_weight_norm_mean'] = float(wn[~dead_mask].mean())
                pool_result['active_weight_norm_std'] = float(wn[~dead_mask].std())

        # Embedding norm comparison
        if key in emb_norms:
            en = emb_norms[key]
            pool_result['emb_norms'] = en.tolist()
            if n_dead > 0 and n_active > 0:
                dead_mask = freq == 0
                pool_result['dead_emb_norm_mean'] = float(en[dead_mask].mean())
                pool_result['active_emb_norm_mean'] = float(en[~dead_mask].mean())

        results['per_pool'][key] = pool_result

    # Summary
    total_dead = sum(p['n_dead'] for p in results['per_pool'].values())
    total_neurons = sum(p['n_total'] for p in results['per_pool'].values())
    results['summary'] = {
        'total_dead': total_dead,
        'total_neurons': total_neurons,
        'overall_dead_ratio': total_dead / total_neurons if total_neurons > 0 else 0,
    }

    return results


def plot_dead_neuron_scatter(results: Dict, output_dir: str, dpi: int = 300) -> Optional[str]:
    """Generate scatter plots: weight norm vs activation frequency per pool.

    Args:
        results: Output from analyze_dead_neuron_norms
        output_dir: Directory to save plots
        dpi: Plot resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        print("  matplotlib not available, skipping plot")
        return None

    os.makedirs(output_dir, exist_ok=True)

    per_pool = results.get('per_pool', {})
    pools_with_norms = [
        k for k, v in per_pool.items()
        if 'weight_norms' in v and 'activation_freq' in v
    ]

    if not pools_with_norms:
        print("  No pools with weight norm data, skipping plot")
        return None

    # Deduplicate: fqk_q and fqk_k share same neurons, show only fqk_q
    display_pools = []
    seen_base = set()
    for k in pools_with_norms:
        base = k.replace('_q', '').replace('_k', '')
        if base not in seen_base:
            seen_base.add(base)
            display_pools.append(k)

    n_pools = len(display_pools)
    cols = min(4, n_pools)
    rows = (n_pools + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)

    colors = {
        'fqk_q': '#E63946', 'fv': '#F4A261', 'rqk_q': '#457B9D',
        'rv': '#2A9D8F', 'fknow': '#9B5DE5', 'rknow': '#00BBF9',
    }

    for idx, key in enumerate(display_pools):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        data = per_pool[key]
        freq = np.array(data['activation_freq'])
        wn = np.array(data['weight_norms'])
        display = data.get('display', key)
        color = colors.get(key, '#666666')

        n_dead = data['n_dead']
        n_total = data['n_total']

        # Scatter: active neurons
        active_mask = freq > 0
        dead_mask = ~active_mask

        if active_mask.any():
            ax.scatter(wn[active_mask], freq[active_mask],
                       c=color, alpha=0.6, s=20, label=f'Active ({active_mask.sum()})',
                       edgecolors='none')

        if dead_mask.any():
            ax.scatter(wn[dead_mask], freq[dead_mask],
                       c='#CCCCCC', alpha=0.8, s=20, marker='x',
                       label=f'Dead ({n_dead})', linewidths=1)

        ax.set_xlabel('Weight Norm (Frobenius)')
        ax.set_ylabel('Activation Frequency')
        ax.set_title(f'{display} ({n_dead}/{n_total} dead)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)

        # Annotate weight norm comparison
        if 'dead_weight_norm_mean' in data and 'active_weight_norm_mean' in data:
            dead_wn = data['dead_weight_norm_mean']
            active_wn = data['active_weight_norm_mean']
            ax.axvline(x=dead_wn, color='#CCCCCC', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=active_wn, color=color, linestyle='--', alpha=0.5, linewidth=1)
            ax.text(0.02, 0.95, f'Dead norm: {dead_wn:.2f}\nActive norm: {active_wn:.2f}',
                    transform=ax.transAxes, fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Hide unused axes
    for idx in range(n_pools, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle('Dead Neuron Analysis: Weight Norm vs Activation Frequency',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, 'dead_neuron_scatter.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")

    # === Embedding norm scatter ===
    pools_with_emb = [k for k in display_pools if 'emb_norms' in per_pool[k]]
    if pools_with_emb:
        n_pools_e = len(pools_with_emb)
        cols_e = min(4, n_pools_e)
        rows_e = (n_pools_e + cols_e - 1) // cols_e
        fig2, axes2 = plt.subplots(rows_e, cols_e, figsize=(4.5 * cols_e, 4 * rows_e), squeeze=False)

        for idx, key in enumerate(pools_with_emb):
            row, col = divmod(idx, cols_e)
            ax = axes2[row][col]

            data = per_pool[key]
            freq = np.array(data['activation_freq'])
            en = np.array(data['emb_norms'])
            display = data.get('display', key)
            color = colors.get(key, '#666666')

            active_mask = freq > 0
            dead_mask = ~active_mask

            if active_mask.any():
                ax.scatter(en[active_mask], freq[active_mask],
                           c=color, alpha=0.6, s=20, edgecolors='none')
            if dead_mask.any():
                ax.scatter(en[dead_mask], freq[dead_mask],
                           c='#CCCCCC', alpha=0.8, s=20, marker='x', linewidths=1)

            ax.set_xlabel('Embedding Norm')
            ax.set_ylabel('Activation Frequency')
            ax.set_title(f'{display} (emb norm)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.2)

        for idx in range(n_pools_e, rows_e * cols_e):
            row, col = divmod(idx, cols_e)
            axes2[row][col].set_visible(False)

        fig2.suptitle('Dead Neuron Analysis: Embedding Norm vs Activation Frequency',
                      fontsize=14, fontweight='bold', y=1.02)
        fig2.tight_layout()

        path2 = os.path.join(output_dir, 'dead_neuron_emb_scatter.png')
        fig2.savefig(path2, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        print(f"  Saved: {path2}")

    return path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Dead Neuron Weight Norm Analysis')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--output', default='./results/dead_neurons', help='Output directory')
    parser.add_argument('--val_data', default=None, help='Validation data path')
    parser.add_argument('--n_batches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    from scripts.analysis.utils_jax import load_model_jax, load_val_data_jax, save_results

    print(f"Loading checkpoint: {args.checkpoint}")
    _, params, _, config = load_model_jax(args.checkpoint)

    if args.val_data:
        val_tokens = load_val_data_jax(args.val_data)
    else:
        vocab_size = config.get('vocab_size', 30522)
        n_tokens = args.n_batches * args.batch_size * args.seq_len
        val_tokens = np.random.randint(1, vocab_size, size=n_tokens, dtype=np.int32)

    print("\n=== Dead Neuron Analysis ===")
    results = analyze_dead_neuron_norms(
        params, config, val_tokens,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    # Print summary
    print(f"\n  Summary:")
    for key, data in results.get('per_pool', {}).items():
        display = data.get('display', key)
        print(f"    {display:12s}: {data['n_dead']:3d}/{data['n_total']:3d} dead "
              f"({data['dead_ratio']*100:.1f}%)", end='')
        if 'dead_weight_norm_mean' in data:
            print(f"  dead_wn={data['dead_weight_norm_mean']:.2f} "
                  f"active_wn={data['active_weight_norm_mean']:.2f}", end='')
        print()

    # Save
    import json
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    # Plot
    plot_dead_neuron_scatter(results, args.output, dpi=args.dpi)
    print("\nDone.")


if __name__ == '__main__':
    main()
