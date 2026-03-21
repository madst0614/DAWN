#!/usr/bin/env python3
"""
Layer-wise Attention/Knowledge Balance (JAX/TPU) — Section 4.4
================================================================
Compute per-layer attention vs knowledge output norm to measure
circuit contribution ratio across depth.

Runs partial forward passes through all layers, recording:
  - ||attn_out||  (attention sub-block output norm)
  - ||know_out||  (knowledge sub-block output norm)
  - attention_ratio = ||attn|| / (||attn|| + ||know||)

Designed for single-host TPU v4-8.

Usage:
    python scripts/analysis/visualizers/layer_balance_jax.py \
        --checkpoint gs://dawn-tpu-data-c4/checkpoints/... \
        --output ./section4_results \
        --val_data gs://bucket/val.bin \
        --n_batches 20 --batch_size 4
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

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
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from scripts.analysis.utils_jax import (
    load_model_jax, load_val_data_jax, create_batches,
    convert_to_serializable, save_results,
)
from scripts.analysis.visualizers.style import PAPER_STYLE, apply_paper_style

if HAS_MATPLOTLIB:
    apply_paper_style(plt)
S = PAPER_STYLE

COLOR_ATTENTION = '#4A90D9'
COLOR_KNOWLEDGE = '#50C878'
COLOR_GRAY = '#7F8C8D'


# ============================================================
# Analysis
# ============================================================

def analyze_layer_balance(
    params, config, val_tokens,
    n_batches=20, batch_size=4, seq_len=512,
):
    """Run partial forward through all layers, recording attn/know output norms.

    Returns per-layer attention ratio and raw norms.
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required")

    from models.model_v17_1_jax import (
        _layer_norm, _router_attn_forward, _router_know_forward,
        _attention_forward, _knowledge_forward,
    )

    n_layers = config.get('n_layers', 16)
    all_params = params.get('params', params)
    router_params = all_params.get('router', {})
    sn_params = all_params.get('shared_neurons', {})

    batches = create_batches(val_tokens, batch_size, seq_len)
    if n_batches:
        batches = batches[:n_batches]

    # Accumulators: per-layer lists of mean norms
    attn_norms = [[] for _ in range(n_layers)]
    know_norms = [[] for _ in range(n_layers)]

    rng_key = jax.random.PRNGKey(42)

    # Config values (extract once)
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

    print(f"  {n_layers} layers, {len(batches)} batches, batch_size={batch_size}")

    for batch in tqdm(batches, desc='Layer Balance'):
        input_ids = jnp.array(batch)
        B, S_len = input_ids.shape

        # Initial embeddings
        tok_emb = all_params['token_emb']['embedding'][input_ids]
        pos_emb = all_params['pos_emb']['embedding'][jnp.arange(S_len)[jnp.newaxis, :]]
        x = tok_emb + pos_emb

        rng_key, batch_rng = jax.random.split(rng_key)

        for li in range(n_layers):
            bp = all_params[f'block_{li}']
            rng_key, rng_ar, rng_kr, rng_a, rng_k = jax.random.split(rng_key, 5)

            # --- Attention sub-block ---
            normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])

            fqk_wQ, fqk_wK, fv_w, rqk_wQ, rqk_wK, rv_w, _ = _router_attn_forward(
                normed, router_params,
                n_fqk, n_fv, n_rqk, n_rv, d_space,
                tk_fqk, tk_fv, tk_rqk, tk_rv,
                0.0, None, True, rng_ar,
            )

            attn_out = _attention_forward(
                normed, sn_params,
                fqk_wQ, fqk_wK, fv_w, rqk_wQ, rqk_wK, rv_w,
                bp['attn']['expand_O']['kernel'],
                n_fqk, n_rqk, n_heads, d_model,
                0.0, True, rng_a,
            )

            attn_norm = float(jnp.linalg.norm(attn_out, axis=-1).mean())
            x = x + attn_out

            # --- Knowledge sub-block ---
            normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])

            fk_w, rk_w, _ = _router_know_forward(
                normed, router_params,
                n_fqk, n_fv, n_rqk, n_rv, n_fk, n_rk,
                tk_fk, tk_rk,
                0.0, None, True, rng_kr,
            )

            know_out = _knowledge_forward(
                normed, sn_params,
                fk_w, rk_w,
                0.0, True, rng_k,
            )

            know_norm = float(jnp.linalg.norm(know_out, axis=-1).mean())
            x = x + know_out

            attn_norms[li].append(attn_norm)
            know_norms[li].append(know_norm)

    # Aggregate
    per_layer = []
    for li in range(n_layers):
        a_mean = float(np.mean(attn_norms[li]))
        k_mean = float(np.mean(know_norms[li]))
        total = a_mean + k_mean
        ratio = (a_mean / total * 100) if total > 0 else 50.0
        per_layer.append({
            'layer': li,
            'attn_norm': a_mean,
            'know_norm': k_mean,
            'attention_ratio': ratio,
            'knowledge_ratio': 100 - ratio,
        })

    results = {
        'n_layers': n_layers,
        'n_batches': len(batches),
        'batch_size': batch_size,
        'seq_len': seq_len,
        'per_layer': per_layer,
        'layer_stats': [p['attention_ratio'] for p in per_layer],
        'summary': {
            'mean_attention_ratio': float(np.mean([p['attention_ratio'] for p in per_layer])),
            'early_layers_attn': float(np.mean([p['attention_ratio'] for p in per_layer[:n_layers//3]])),
            'mid_layers_attn': float(np.mean([p['attention_ratio'] for p in per_layer[n_layers//3:2*n_layers//3]])),
            'late_layers_attn': float(np.mean([p['attention_ratio'] for p in per_layer[2*n_layers//3:]])),
        },
    }
    return results


# ============================================================
# Visualization
# ============================================================

def plot_layer_balance(results, output_dir, dpi=300):
    """Line plot: attention contribution % per layer with fill."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    per_layer = results['per_layer']
    n_layers = results['n_layers']
    layers = list(range(1, n_layers + 1))
    attn_ratios = [p['attention_ratio'] for p in per_layer]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Line with markers
    ax.plot(layers, attn_ratios, 'o-', color=COLOR_ATTENTION, linewidth=2,
            markersize=6, markerfacecolor='white', markeredgewidth=1.5)

    # Fill above/below 50%
    ax.fill_between(layers, attn_ratios, 50,
                    where=[a >= 50 for a in attn_ratios],
                    color=COLOR_ATTENTION, alpha=0.25, label='Attention dominant')
    ax.fill_between(layers, attn_ratios, 50,
                    where=[a < 50 for a in attn_ratios],
                    color=COLOR_KNOWLEDGE, alpha=0.25, label='Knowledge dominant')

    # 50% baseline
    ax.axhline(y=50, color=COLOR_GRAY, linestyle='--', linewidth=1.5)

    # Annotations for early/mid/late
    s = results['summary']
    thirds = n_layers // 3
    for label, val, x_pos in [
        ('Early', s['early_layers_attn'], thirds / 2 + 0.5),
        ('Mid', s['mid_layers_attn'], thirds + thirds / 2 + 0.5),
        ('Late', s['late_layers_attn'], 2 * thirds + (n_layers - 2 * thirds) / 2 + 0.5),
    ]:
        ax.annotate(f'{label}\n{val:.1f}%', xy=(x_pos, min(attn_ratios) - 2),
                    fontsize=S['font_size_legend'], ha='center', color='gray', alpha=0.7)

    ax.set_xlim(0.5, n_layers + 0.5)
    y_min = max(25, min(attn_ratios) - 5)
    y_max = min(80, max(attn_ratios) + 5)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(layers)
    ax.set_xlabel('Layer', fontsize=S['font_size_label'])
    ax.set_ylabel('Attention Contribution (%)', fontsize=S['font_size_label'])
    ax.set_title('Attention vs Knowledge Balance Across Layers',
                  fontsize=S['font_size_subtitle'], fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    legend_elements = [
        mpatches.Patch(color=COLOR_ATTENTION, alpha=0.25, label='Attention > 50%'),
        mpatches.Patch(color=COLOR_KNOWLEDGE, alpha=0.25, label='Knowledge > 50%'),
        plt.Line2D([0], [0], color=COLOR_GRAY, linestyle='--', label='50% baseline'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=S['font_size_legend'], framealpha=0.9)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'layer_balance.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Layer-wise Attention/Knowledge Balance (Section 4.4)')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--output', default='./section4_results', help='Output directory')
    parser.add_argument('--val_data', default=None, help='Validation data path (.bin)')
    parser.add_argument('--n_batches', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model_cls, params, config = load_model_jax(args.checkpoint)
    print(f"  n_layers={config.get('n_layers')}, d_model={config.get('d_model')}")

    # Load validation data
    if args.val_data:
        print(f"Loading validation data: {args.val_data}")
        val_tokens = load_val_data_jax(args.val_data)
    else:
        print("No --val_data, using random tokens for testing")
        vocab_size = config.get('vocab_size', 30522)
        n_tokens = args.n_batches * args.batch_size * args.seq_len
        val_tokens = np.random.randint(1, vocab_size, size=n_tokens, dtype=np.int32)

    print("\n=== Layer-wise Attention/Knowledge Balance ===")
    results = analyze_layer_balance(
        params, config, val_tokens,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    # Save JSON
    json_path = os.path.join(args.output, 'layer_balance.json')
    save_results(results, json_path)
    print(f"  Saved: {json_path}")

    # Print summary
    print(f"\n  Layer-wise attention contribution (%):")
    for p in results['per_layer']:
        bar = '█' * int(p['attention_ratio'] / 2) + '░' * (50 - int(p['attention_ratio'] / 2))
        print(f"    L{p['layer']:2d}: {p['attention_ratio']:5.1f}% attn | {bar}")

    s = results['summary']
    print(f"\n  Early layers: {s['early_layers_attn']:.1f}% attention")
    print(f"  Mid layers:   {s['mid_layers_attn']:.1f}% attention")
    print(f"  Late layers:  {s['late_layers_attn']:.1f}% attention")

    # Plot
    plot_layer_balance(results, args.output, dpi=args.dpi)

    print("\nDone.")


if __name__ == '__main__':
    main()
