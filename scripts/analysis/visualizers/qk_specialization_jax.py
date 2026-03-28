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
    create_batches,
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

import jax
import jax.numpy as jnp


# Colors (consistent with qk_specialization.py)
COLOR_Q = '#C0392B'
COLOR_K = '#2471A3'
COLOR_SHARED = '#50C878'
COLOR_INACTIVE = '#95A5A6'


# ============================================================
# Multi-layer Q/K routing forward
# ============================================================

def _build_qk_forward(params, config, pad_len):
    """Build a JIT-compiled full forward that returns layer-summed Q/K weights.

    Runs all n_layers of attention + knowledge (full forward pass), collects
    Q and K routing weights for both feature_qk and restore_qk pools at each
    layer, and returns the sum across all layers.

    This matches paper Appendix D.1: "accumulated across all tokens and layers."

    Returns:
        fn(input_ids[B, pad_len]) -> (fqk_wQ, fqk_wK, rqk_wQ, rqk_wK)
        each [B, pad_len, N_pool], summed over n_layers.
    """
    from models.model_v17_1_jax import (
        _layer_norm, _router_attn_forward, _router_know_forward,
        _attention_forward, _knowledge_forward,
    )

    all_params = params.get('params', params)
    router_params = all_params.get('router', {})
    sn_params = all_params.get('shared_neurons', {})
    n_layers = config.get('n_layers', 16)

    n_fqk = config.get('n_feature_qk', 88)
    n_fv = config.get('n_feature_v', 352)
    n_rqk = config.get('n_restore_qk', 88)
    n_rv = config.get('n_restore_v', 352)
    n_fk = config.get('n_feature_know', 224)
    n_rk = config.get('n_restore_know', 224)
    d_space = config.get('d_space', 256)
    tk_fqk = config.get('top_k_feature_qk', 16)
    tk_fv = config.get('top_k_feature_v', 16)
    tk_rqk = config.get('top_k_restore_qk', 16)
    tk_rv = config.get('top_k_restore_v', 16)
    tk_fk = config.get('top_k_feature_know', 16)
    tk_rk = config.get('top_k_restore_know', 16)
    n_heads = config.get('n_heads', 8)
    d_model = config.get('d_model', 768)

    block_params_list = [all_params[f'block_{i}'] for i in range(n_layers)]
    token_emb_table = all_params['token_emb']['embedding']
    pos_emb_table = all_params['pos_emb']['embedding']
    positions = jnp.arange(pad_len)
    pos_emb_fixed = pos_emb_table[positions][jnp.newaxis, :]

    @jax.jit
    def _forward(input_ids):
        """input_ids: [B, pad_len] -> (fqk_wQ, fqk_wK, rqk_wQ, rqk_wK) each [B, pad_len, N]"""
        x = jnp.take(token_emb_table, input_ids, axis=0) + pos_emb_fixed
        rng_key = jax.random.PRNGKey(0)

        fqk_q_sum = jnp.zeros((input_ids.shape[0], pad_len, n_fqk))
        fqk_k_sum = jnp.zeros((input_ids.shape[0], pad_len, n_fqk))
        rqk_q_sum = jnp.zeros((input_ids.shape[0], pad_len, n_rqk))
        rqk_k_sum = jnp.zeros((input_ids.shape[0], pad_len, n_rqk))

        for li in range(n_layers):
            bp = block_params_list[li]
            rng_key, rng_ar, rng_kr, rng_a, rng_k = jax.random.split(rng_key, 5)

            normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
            attn_results = _router_attn_forward(
                normed, router_params,
                n_fqk, n_fv, n_rqk, n_rv, d_space,
                tk_fqk, tk_fv, tk_rqk, tk_rv,
                0.0, None, True, rng_ar,
            )
            fqk_wQ, fqk_wK, fv_w, rqk_wQ, rqk_wK, rv_w = attn_results[:6]

            fqk_q_sum = fqk_q_sum + fqk_wQ
            fqk_k_sum = fqk_k_sum + fqk_wK
            rqk_q_sum = rqk_q_sum + rqk_wQ
            rqk_k_sum = rqk_k_sum + rqk_wK

            attn_out = _attention_forward(
                normed, sn_params,
                fqk_wQ, fqk_wK, fv_w, rqk_wQ, rqk_wK, rv_w,
                bp['attn']['expand_O']['kernel'],
                n_fqk, n_rqk, n_heads, d_model,
                0.0, True, rng_a,
            )
            x = x + attn_out

            normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
            fk_w, rk_w, _ = _router_know_forward(
                normed, router_params,
                n_fqk, n_fv, n_rqk, n_rv, n_fk, n_rk,
                tk_fk, tk_rk,
                0.0, None, True, rng_kr,
            )
            know_out = _knowledge_forward(
                normed, sn_params, fk_w, rk_w,
                0.0, True, rng_k,
            )
            x = x + know_out

        return fqk_q_sum, fqk_k_sum, rqk_q_sum, rqk_k_sum

    return _forward


# ============================================================
# Analysis
# ============================================================

def analyze_qk_specialization(
    model_cls, params, config, val_tokens,
    n_batches=100, batch_size=32, seq_len=512,
):
    """Compute per-neuron Q/K selection counts for each QK pool.

    Uses full forward pass through all layers (paper Appendix D.1:
    "accumulated across all tokens and layers").

    Returns dict keyed by pool name with q_counts, k_counts, correlation,
    specialization breakdown, etc.
    """
    # Pad seq_len to hardware-aligned length
    max_seq = config.get('max_seq_len', 512)
    pad_len = min(seq_len, max_seq)
    pad_len = ((pad_len + 31) // 32) * 32
    pad_len = min(pad_len, max_seq)

    # Build JIT-compiled multi-layer forward
    print(f"  JIT compiling multi-layer Q/K forward (batch_size={batch_size}, "
          f"pad_len={pad_len})...", end=" ", flush=True)
    import time
    t0 = time.time()
    forward_fn = _build_qk_forward(params, config, pad_len)
    dummy = jnp.zeros((batch_size, pad_len), dtype=jnp.int32)
    _ = forward_fn(dummy)
    _[0].block_until_ready()
    print(f"done ({time.time() - t0:.1f}s)")

    # Pre-init accumulators
    n_fqk = config.get('n_feature_qk', 88)
    n_rqk = config.get('n_restore_qk', 88)
    pool_data = {
        'feature_qk': {
            'info': QK_POOLS['feature_qk'],
            'n_neurons': n_fqk,
            'q_counts': np.zeros(n_fqk, dtype=np.float64),
            'k_counts': np.zeros(n_fqk, dtype=np.float64),
            'overlaps': [],
        },
        'restore_qk': {
            'info': QK_POOLS['restore_qk'],
            'n_neurons': n_rqk,
            'q_counts': np.zeros(n_rqk, dtype=np.float64),
            'k_counts': np.zeros(n_rqk, dtype=np.float64),
            'overlaps': [],
        },
    }

    batches = create_batches(val_tokens, batch_size, seq_len)
    if n_batches:
        batches = batches[:n_batches]

    for batch in tqdm(batches, desc='Q/K Specialization (multi-layer)'):
        input_ids = np.array(batch)
        B, S_actual = input_ids.shape

        # Pad to fixed pad_len
        if S_actual < pad_len:
            padded = np.zeros((B, pad_len), dtype=np.int32)
            padded[:, :S_actual] = input_ids
        else:
            padded = input_ids[:, :pad_len]

        fqk_q, fqk_k, rqk_q, rqk_k = forward_fn(jnp.array(padded))

        # Convert to numpy, slice to actual length
        fqk_q_np = np.asarray(fqk_q)[:, :S_actual, :]
        fqk_k_np = np.asarray(fqk_k)[:, :S_actual, :]
        rqk_q_np = np.asarray(rqk_q)[:, :S_actual, :]
        rqk_k_np = np.asarray(rqk_k)[:, :S_actual, :]

        # Feature QK: accumulate
        pd = pool_data['feature_qk']
        pd['q_counts'] += (fqk_q_np > 0).astype(float).sum(axis=(0, 1))
        pd['k_counts'] += (fqk_k_np > 0).astype(float).sum(axis=(0, 1))
        overlap = ((fqk_q_np > 0) & (fqk_k_np > 0)).astype(float)
        active_q = (fqk_q_np > 0).astype(float).sum(axis=-1)
        pd['overlaps'].append(float((overlap.sum(axis=-1) / (active_q + 1e-8)).mean()))

        # Restore QK: accumulate
        pd = pool_data['restore_qk']
        pd['q_counts'] += (rqk_q_np > 0).astype(float).sum(axis=(0, 1))
        pd['k_counts'] += (rqk_k_np > 0).astype(float).sum(axis=(0, 1))
        overlap = ((rqk_q_np > 0) & (rqk_k_np > 0)).astype(float)
        active_q = (rqk_q_np > 0).astype(float).sum(axis=-1)
        pd['overlaps'].append(float((overlap.sum(axis=-1) / (active_q + 1e-8)).mean()))

        del fqk_q, fqk_k, rqk_q, rqk_k

    # Post-process: compute stats per pool
    results = {}
    for pool_name, pd in pool_data.items():
        q_counts = pd['q_counts']
        k_counts = pd['k_counts']
        n_neurons = pd['n_neurons']

        total_usage = q_counts + k_counts
        q_ratio = np.zeros_like(q_counts)
        valid = total_usage > 0

        if q_counts.sum() > 0 and k_counts.sum() > 0:
            corr_all = float(np.corrcoef(q_counts, k_counts)[0, 1])
        else:
            corr_all = 0.0

        if valid.sum() >= 2:
            corr_active = float(np.corrcoef(q_counts[valid], k_counts[valid])[0, 1])
        else:
            corr_active = corr_all
        q_ratio[valid] = q_counts[valid] / total_usage[valid]

        q_specialized = int((q_ratio > 0.7).sum())
        k_specialized = int((q_ratio < 0.3).sum())
        shared = int(((q_ratio >= 0.3) & (q_ratio <= 0.7)).sum())
        inactive = int((~valid).sum())

        sensitivity_thresholds = np.array([0.6, 0.65, 0.7, 0.75, 0.8])
        q_spec_all = (q_ratio[:, np.newaxis] > sensitivity_thresholds).sum(axis=0)
        k_spec_all = (q_ratio[:, np.newaxis] < (1 - sensitivity_thresholds)).sum(axis=0)
        shared_all = ((q_ratio[:, np.newaxis] >= (1 - sensitivity_thresholds)) &
                      (q_ratio[:, np.newaxis] <= sensitivity_thresholds)).sum(axis=0)
        sensitivity_analysis = {
            str(float(t)): {
                'q_specialized': int(q_spec_all[i]),
                'k_specialized': int(k_spec_all[i]),
                'shared': int(shared_all[i]),
                'total': n_neurons,
            }
            for i, t in enumerate(sensitivity_thresholds)
        }

        results[pool_name] = {
            'display': pd['info']['display'],
            'n_neurons': n_neurons,
            'q_counts': q_counts.tolist(),
            'k_counts': k_counts.tolist(),
            'correlation': corr_all,
            'correlation_active': corr_active,
            'avg_overlap': float(np.mean(pd['overlaps'])) if pd['overlaps'] else 0,
            'std_overlap': float(np.std(pd['overlaps'])) if pd['overlaps'] else 0,
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
        'n_batches': len(batches),
        'batch_size': batch_size,
        'seq_len': seq_len,
        'multi_layer': True,
        'n_layers': config.get('n_layers', 16),
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
        r_active = data.get('correlation_active', data['correlation'])
        r_all = data['correlation']
        ax1.set_title(f'{data["display"]} — Q vs K Usage  (r_active={r_active:.3f}, r_all={r_all:.3f})',
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
    model_cls, params, _, config = load_model_jax(args.checkpoint)
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
        print(f"    Correlation:    r_all = {d['correlation']:.4f},  r_active = {d.get('correlation_active', d['correlation']):.4f}")
        print(f"    Q-specialized:  {d['q_specialized']}")
        print(f"    K-specialized:  {d['k_specialized']}")
        print(f"    Shared:         {d['shared']}")
        print(f"    Inactive:       {d['inactive']}")

    # Plot
    plot_qk_scatter(results, args.output, dpi=args.dpi)

    print("\nDone.")


if __name__ == '__main__':
    main()
