#!/usr/bin/env python3
"""
Knowledge Neuron Coherence Analysis (JAX/TPU) — Section 4.3
=============================================================
Compare neuron activation patterns between capital-knowledge queries
and control queries to identify knowledge-specific neurons.

Capital queries:  "The capital of France is", "The capital of Japan is", ...
Control queries:  "The sky is", "The water is", ...

For F-Know and R-Know pools, classifies neurons as:
  - Shared:           fires for both capital AND control (freq > threshold)
  - Capital-specific: fires for capital queries only
  - Control-specific: fires for control queries only

Designed for single-host TPU v4-8.

Usage:
    python scripts/analysis/visualizers/knowledge_coherence_jax.py \
        --checkpoint gs://dawn-tpu-data-c4/checkpoints/... \
        --output ./section4_results
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

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
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from scripts.analysis.utils_jax import (
    load_model_jax, create_model_from_config,
    JAXRoutingDataExtractor, JAXRoutingData,
    convert_to_serializable, save_results,
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

# Colors
COLOR_SHARED = '#50C878'
COLOR_CAPITAL = '#E63946'
COLOR_CONTROL = '#457B9D'
COLOR_INACTIVE = '#95A5A6'

# ============================================================
# Default Prompts
# ============================================================

CAPITAL_PROMPTS = [
    ("France",  "The capital of France is"),
    ("UK",      "The capital of the United Kingdom is"),
    ("Japan",   "The capital of Japan is"),
    ("Germany", "The capital of Germany is"),
    ("Italy",   "The capital of Italy is"),
    ("Spain",   "The capital of Spain is"),
    ("China",   "The capital of China is"),
    ("Brazil",  "The capital of Brazil is"),
    ("India",   "The capital of India is"),
    ("Canada",  "The capital of Canada is"),
]

CONTROL_PROMPTS = [
    ("sky",     "The sky is"),
    ("water",   "The water is"),
    ("cat",     "The cat is"),
    ("book",    "The book is"),
    ("food",    "The food is"),
    ("music",   "The music is"),
    ("tree",    "The tree is"),
    ("house",   "The house is"),
    ("car",     "The car is"),
    ("sun",     "The sun is"),
]


# ============================================================
# Analysis
# ============================================================

def _get_knowledge_activations(extractor, tokenizer, prompt, config):
    """Get active neuron sets for F-Know and R-Know pools from a single prompt.

    Uses ALL token positions (not just last). With embedding-only routing,
    each token's embedding independently determines routing. Using all positions
    captures the full prompt semantics — e.g., "capital" and country names
    route to different neurons than "sky" or "water".

    Returns dict: {pool: set_of_active_neuron_indices}
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=True,
                                  max_length=config.get('max_seq_len', 512),
                                  truncation=True)
    input_ids_np = np.array([input_ids], dtype=np.int32)

    routing_info = extractor.extract_routing(input_ids_np)
    routing = JAXRoutingData(routing_info)

    result = {}
    for pool_key in ['fknow', 'rknow']:
        w = routing.get_weight(pool_key)
        if w is None:
            result[pool_key] = set()
            continue
        # Use ALL token positions — union of active neurons across the prompt
        if w.ndim == 3:
            w_all = w[0]  # [S, N]
        else:
            w_all = w  # [S, N] or [N]
            if w_all.ndim == 1:
                w_all = w_all[np.newaxis, :]  # [1, N]
        # A neuron is "active" if it fires for ANY token in the prompt
        active = set(int(i) for i in np.where((w_all > 0).any(axis=0))[0])
        result[pool_key] = active

    return result


def analyze_knowledge_coherence(
    model_cls, params, config,
    capital_prompts=None, control_prompts=None,
):
    """Compare neuron activations between capital and control queries.

    For each pool (fknow, rknow), computes:
      - Per-neuron activation frequency across capital and control queries
      - Classification: shared / capital-specific / control-specific
      - Contrastive score: capital_freq - control_freq
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    capital_prompts = capital_prompts or CAPITAL_PROMPTS
    control_prompts = control_prompts or CONTROL_PROMPTS

    model_instance = create_model_from_config(config)
    extractor = JAXRoutingDataExtractor(model_instance, params, config)

    n_fknow = config.get('n_feature_know', 224)
    n_rknow = config.get('n_restore_know', 224)
    pool_sizes = {'fknow': n_fknow, 'rknow': n_rknow}

    # Accumulate activation counts
    capital_counts = {p: np.zeros(pool_sizes[p], dtype=np.int32) for p in pool_sizes}
    control_counts = {p: np.zeros(pool_sizes[p], dtype=np.int32) for p in pool_sizes}

    # Per-query detail
    per_query = {}

    n_capital = len(capital_prompts)
    n_control = len(control_prompts)

    print(f"  Processing {n_capital} capital queries...")
    for label, prompt in capital_prompts:
        activations = _get_knowledge_activations(extractor, tokenizer, prompt, config)
        per_query[f"capital_{label}"] = {
            'prompt': prompt,
            'type': 'capital',
        }
        for pool_key in ['fknow', 'rknow']:
            active = activations[pool_key]
            for idx in active:
                if idx < pool_sizes[pool_key]:
                    capital_counts[pool_key][idx] += 1
            per_query[f"capital_{label}"][pool_key] = sorted(active)

    print(f"  Processing {n_control} control queries...")
    for label, prompt in control_prompts:
        activations = _get_knowledge_activations(extractor, tokenizer, prompt, config)
        per_query[f"control_{label}"] = {
            'prompt': prompt,
            'type': 'control',
        }
        for pool_key in ['fknow', 'rknow']:
            active = activations[pool_key]
            for idx in active:
                if idx < pool_sizes[pool_key]:
                    control_counts[pool_key][idx] += 1
            per_query[f"control_{label}"][pool_key] = sorted(active)

    # Compute per-pool stats
    results = {'per_pool': {}, 'per_query': per_query}

    for pool_key in ['fknow', 'rknow']:
        n = pool_sizes[pool_key]
        cap_freq = capital_counts[pool_key] / max(n_capital, 1)
        ctrl_freq = control_counts[pool_key] / max(n_control, 1)
        contrastive = cap_freq - ctrl_freq

        # Classify neurons — matches GPU factual_heatmap.py thresholds
        # GPU: shared = all >= 0.7; capital-specific = cap >= 0.7 AND ctrl < 0.3
        # Also compute with relaxed threshold for JAX's smaller sample size
        thresh_high = 0.7
        thresh_low = 0.3

        cap_high = cap_freq >= thresh_high
        ctrl_high = ctrl_freq >= thresh_high
        cap_low = cap_freq >= thresh_low
        ctrl_low = ctrl_freq >= thresh_low

        # Strict (GPU-compatible): matches factual_heatmap.py logic
        shared_strict = int((cap_high & ctrl_high).sum())
        capital_specific_strict = int((cap_high & (ctrl_freq < thresh_low)).sum())
        control_specific_strict = int((ctrl_high & (cap_freq < thresh_low)).sum())

        # Relaxed (for JAX's 10-query sample): uses 0.3 as active threshold
        shared = int((cap_low & ctrl_low).sum())
        capital_specific = int((cap_low & ~ctrl_low).sum())
        control_specific = int((~cap_low & ctrl_low).sum())
        inactive = int((~cap_low & ~ctrl_low).sum())

        # Top capital-specific neurons (use relaxed for ranking)
        cap_spec_idx = np.where(cap_low & ~ctrl_low)[0]
        cap_spec_sorted = cap_spec_idx[np.argsort(contrastive[cap_spec_idx])[::-1]]
        top_capital = [
            {'neuron': int(i), 'capital_freq': float(cap_freq[i]),
             'control_freq': float(ctrl_freq[i]),
             'contrastive': float(contrastive[i])}
            for i in cap_spec_sorted[:15]
        ]

        # Top shared neurons
        shared_idx = np.where(cap_low & ctrl_low)[0]
        shared_sorted = shared_idx[np.argsort(cap_freq[shared_idx] + ctrl_freq[shared_idx])[::-1]]
        top_shared = [
            {'neuron': int(i), 'capital_freq': float(cap_freq[i]),
             'control_freq': float(ctrl_freq[i])}
            for i in shared_sorted[:15]
        ]

        results['per_pool'][pool_key] = {
            'n_neurons': n,
            # Relaxed classification (threshold=0.3, for JAX sample size)
            'shared': shared,
            'capital_specific': capital_specific,
            'control_specific': control_specific,
            'inactive': inactive,
            # Strict classification (matches GPU factual_heatmap.py: 0.7/0.3)
            'shared_strict': shared_strict,
            'capital_specific_strict': capital_specific_strict,
            'control_specific_strict': control_specific_strict,
            # Raw data
            'capital_freq': cap_freq.tolist(),
            'control_freq': ctrl_freq.tolist(),
            'contrastive_scores': contrastive.tolist(),
            'top_capital_specific': top_capital,
            'top_shared': top_shared,
        }

    results['meta'] = {
        'n_capital_queries': n_capital,
        'n_control_queries': n_control,
        'threshold_relaxed': 0.3,
        'threshold_strict_high': 0.7,
        'threshold_strict_low': 0.3,
        'note': 'strict thresholds match GPU factual_heatmap.py; '
                'relaxed thresholds adapted for smaller JAX sample size',
    }

    return results


# ============================================================
# Visualization
# ============================================================

def plot_knowledge_coherence(results, output_dir, dpi=300):
    """Generate coherence figures for each knowledge pool."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")
        return []

    saved = []

    for pool_key in ['fknow', 'rknow']:
        data = results['per_pool'].get(pool_key)
        if data is None:
            continue

        cap_freq = np.array(data['capital_freq'])
        ctrl_freq = np.array(data['control_freq'])
        n = data['n_neurons']
        pool_label = 'F-Know' if pool_key == 'fknow' else 'R-Know'

        # --- Figure: 2 panels ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: Scatter — capital freq vs control freq
        thresh = 0.3
        cap_act = cap_freq >= thresh
        ctrl_act = ctrl_freq >= thresh

        shared_m = cap_act & ctrl_act
        cap_only = cap_act & ~ctrl_act
        ctrl_only = ~cap_act & ctrl_act
        inactive_m = ~cap_act & ~ctrl_act

        for mask, color, label in [
            (inactive_m, COLOR_INACTIVE, f'Inactive ({int(inactive_m.sum())})'),
            (shared_m, COLOR_SHARED, f'Shared ({int(shared_m.sum())})'),
            (cap_only, COLOR_CAPITAL, f'Capital-specific ({int(cap_only.sum())})'),
            (ctrl_only, COLOR_CONTROL, f'Control-specific ({int(ctrl_only.sum())})'),
        ]:
            if mask.any():
                ax1.scatter(cap_freq[mask], ctrl_freq[mask],
                            c=color, s=18, alpha=0.6, label=label)

        ax1.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.8, alpha=0.5)
        ax1.axhline(y=thresh, color='gray', linewidth=0.5, alpha=0.3, linestyle=':')
        ax1.axvline(x=thresh, color='gray', linewidth=0.5, alpha=0.3, linestyle=':')
        ax1.set_xlabel('Capital Query Freq', fontsize=S['font_size_label'])
        ax1.set_ylabel('Control Query Freq', fontsize=S['font_size_label'])
        ax1.set_title(f'{pool_label} — Capital vs Control Activation',
                       fontsize=S['font_size_subtitle'], fontweight='bold')
        ax1.legend(fontsize=S['font_size_legend'] - 1, loc='upper left')
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)

        # Panel 2: Bar chart — classification breakdown
        categories = ['Shared', 'Capital-\nspecific', 'Control-\nspecific', 'Inactive']
        counts = [data['shared'], data['capital_specific'],
                  data['control_specific'], data['inactive']]
        colors = [COLOR_SHARED, COLOR_CAPITAL, COLOR_CONTROL, COLOR_INACTIVE]

        bars = ax2.bar(categories, counts, color=colors, edgecolor='white', linewidth=0.5)
        for bar, c in zip(bars, counts):
            if c > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         str(c), ha='center', fontsize=S['font_size_annotation'])

        ax2.set_ylabel('Number of Neurons', fontsize=S['font_size_label'])
        ax2.set_title(f'{pool_label} — Neuron Classification (n={n})',
                       fontsize=S['font_size_subtitle'], fontweight='bold')

        plt.tight_layout()
        fname = f'knowledge_coherence_{pool_key}.png'
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {path}")
        saved.append(path)

    # Heatmap: neuron activation across all queries (fknow only, or both)
    if HAS_SEABORN:
        for pool_key in ['fknow', 'rknow']:
            data = results['per_pool'].get(pool_key)
            if data is None:
                continue
            _plot_query_heatmap(results, pool_key, output_dir, dpi)

    return saved


def _plot_query_heatmap(results, pool_key, output_dir, dpi=300):
    """Heatmap: queries (rows) x top neurons (columns)."""
    data = results['per_pool'][pool_key]
    per_query = results['per_query']
    pool_label = 'F-Know' if pool_key == 'fknow' else 'R-Know'

    # Collect active neuron sets per query
    cap_freq = np.array(data['capital_freq'])
    ctrl_freq = np.array(data['control_freq'])
    contrastive = np.array(data['contrastive_scores'])

    # Select top neurons by absolute contrastive score
    top_idx = np.argsort(np.abs(contrastive))[-30:][::-1]

    # Build matrix: queries x neurons
    query_labels = []
    rows = []

    for qname, qdata in sorted(per_query.items()):
        active_set = set(qdata.get(pool_key, []))
        row = np.array([1.0 if int(i) in active_set else 0.0 for i in top_idx])
        query_labels.append(f"{'*' if qdata['type']=='capital' else ' '} {qdata['prompt']}")
        rows.append(row)

    if not rows:
        return

    mat = np.stack(rows)

    fig, ax = plt.subplots(figsize=(max(10, len(top_idx) * 0.35),
                                     max(5, len(rows) * 0.35)))
    sns.heatmap(
        mat,
        xticklabels=[str(i) for i in top_idx],
        yticklabels=query_labels,
        cmap='YlOrRd',
        cbar_kws={'label': 'Active (1) / Inactive (0)'},
        linewidths=0.3,
        ax=ax,
    )
    ax.set_xlabel('Neuron Index', fontsize=S['font_size_label'])
    ax.set_title(f'{pool_label} — Query × Neuron Activation  (* = capital query)',
                  fontsize=S['font_size_subtitle'], fontweight='bold')
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    path = os.path.join(output_dir, f'knowledge_heatmap_{pool_key}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Knowledge Neuron Coherence (Section 4.3)')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--output', default='./section4_results', help='Output directory')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model_cls, params, config = load_model_jax(args.checkpoint)
    print(f"  n_feature_know={config.get('n_feature_know')}, "
          f"n_restore_know={config.get('n_restore_know')}")

    print("\n=== Knowledge Neuron Coherence Analysis ===")
    results = analyze_knowledge_coherence(model_cls, params, config)

    # Save JSON
    json_path = os.path.join(args.output, 'knowledge_coherence.json')
    save_results(results, json_path)
    print(f"  Saved: {json_path}")

    # Print summary
    for pool_key in ['fknow', 'rknow']:
        d = results['per_pool'].get(pool_key)
        if d is None:
            continue
        pool_label = 'F-Know' if pool_key == 'fknow' else 'R-Know'
        print(f"\n  {pool_label} (n={d['n_neurons']}):")
        print(f"    Shared:           {d['shared']}")
        print(f"    Capital-specific: {d['capital_specific']}")
        print(f"    Control-specific: {d['control_specific']}")
        print(f"    Inactive:         {d['inactive']}")
        if d['top_capital_specific']:
            top3 = d['top_capital_specific'][:3]
            print(f"    Top capital neurons: " +
                  ', '.join(f"N{t['neuron']}(Δ={t['contrastive']:.2f})" for t in top3))

    # Plot
    plot_knowledge_coherence(results, args.output, dpi=args.dpi)

    print("\nDone.")


if __name__ == '__main__':
    main()
