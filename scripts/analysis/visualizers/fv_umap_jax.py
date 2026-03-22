#!/usr/bin/env python3
"""
Feature V Pool UMAP — Intra-Pool Functional Clustering
========================================================
UMAP of Feature V pool neuron embeddings (624 neurons),
colored by most selective POS tag from POS selectivity analysis.

Designed for single-host TPU v4-8.

Usage:
    python scripts/analysis/visualizers/fv_umap_jax.py \
        --checkpoint gs://dawn-tpu-data-c4/checkpoints/... \
        --output ./section4_results \
        --max_sentences 2000
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from scripts.analysis.utils_jax import (
    load_model_jax, get_neuron_embeddings_jax,
    convert_to_serializable, save_results,
)
from scripts.analysis.visualizers.pos_selectivity_jax import (
    load_ud_ewt, analyze_pos_selectivity, UPOS_TAGS,
)

# Inline style
PAPER_STYLE = {
    'font_family': 'serif', 'font_size_base': 10, 'font_size_label': 14,
    'font_size_subtitle': 14, 'font_size_tick': 11, 'font_size_legend': 10,
    'font_size_annotation': 12,
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

# POS color palette — perceptually distinct
POS_COLORS = {
    'NOUN':  '#E63946',   # red
    'VERB':  '#457B9D',   # steel blue
    'ADJ':   '#F4A261',   # orange
    'ADV':   '#2A9D8F',   # teal
    'DET':   '#9B5DE5',   # purple
    'ADP':   '#00BBF9',   # cyan
    'PRON':  '#FF6B6B',   # coral
    'PROPN': '#C77DFF',   # lavender
    'PUNCT': '#95A5A6',   # gray
    'AUX':   '#E9C46A',   # gold
    'CCONJ': '#264653',   # dark teal
    'SCONJ': '#606C38',   # olive
    'NUM':   '#BC6C25',   # brown
    'PART':  '#A8DADC',   # light blue
    'INTJ':  '#D62828',   # dark red
    'SYM':   '#BDB2FF',   # light purple
    'X':     '#CCCCCC',   # light gray
}
COLOR_NONE = '#DDDDDD'  # neurons with no selective POS


def get_fv_embeddings(params, config):
    """Extract Feature V pool embeddings from the full neuron embedding table."""
    emb = get_neuron_embeddings_jax(params)
    if emb is None:
        return None

    n_fqk = config.get('n_feature_qk', 0)
    n_fv = config.get('n_feature_v', 0)
    if n_fv == 0:
        return None

    fv_start = n_fqk
    fv_end = n_fqk + n_fv

    if fv_end > len(emb):
        print(f"  Warning: FV range [{fv_start}:{fv_end}] exceeds embedding table size {len(emb)}")
        fv_end = min(fv_end, len(emb))

    return emb[fv_start:fv_end]  # [n_fv, d_space]


def assign_top_pos(selectivity_matrix, pos_tags, threshold=1.5):
    """Assign each neuron its most selective POS (if above threshold).

    Args:
        selectivity_matrix: [n_pos, n_neurons] selectivity scores
        pos_tags: list of POS tag strings
        threshold: minimum selectivity to count as "selective"

    Returns:
        top_pos: list of POS strings (or None) per neuron
        top_sel: array of max selectivity per neuron
    """
    sel = np.array(selectivity_matrix)  # [n_pos, N]
    max_pos_idx = sel.argmax(axis=0)    # [N]
    max_sel = sel.max(axis=0)           # [N]

    top_pos = []
    for i in range(sel.shape[1]):
        if max_sel[i] >= threshold:
            top_pos.append(pos_tags[max_pos_idx[i]])
        else:
            top_pos.append(None)

    return top_pos, max_sel


def plot_fv_umap(
    coords, top_pos, max_sel, n_fv, output_dir,
    method_label='UMAP', dpi=300,
):
    """Scatter plot of FV neurons in 2D, colored by top POS."""
    if not HAS_MATPLOTLIB:
        print("  matplotlib not available")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # Count per POS for legend ordering
    pos_counts = {}
    for pos in top_pos:
        if pos is not None:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
    n_none = sum(1 for p in top_pos if p is None)

    # Plot unassigned neurons first (background)
    none_mask = np.array([p is None for p in top_pos])
    if none_mask.any():
        ax.scatter(
            coords[none_mask, 0], coords[none_mask, 1],
            c=COLOR_NONE, s=12, alpha=0.3, edgecolors='none',
            label=f'Non-selective ({n_none})',
        )

    # Plot each POS group, sorted by count descending
    for pos, cnt in sorted(pos_counts.items(), key=lambda x: -x[1]):
        mask = np.array([p == pos for p in top_pos])
        color = POS_COLORS.get(pos, COLOR_NONE)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, s=20, alpha=0.7, edgecolors='none',
            label=f'{pos} ({cnt})',
        )

    ax.legend(
        loc='best', fontsize=8, framealpha=0.9, markerscale=1.8,
        ncol=2 if len(pos_counts) > 8 else 1,
    )
    ax.set_xlabel(f'{method_label} 1', fontsize=S['font_size_label'])
    ax.set_ylabel(f'{method_label} 2', fontsize=S['font_size_label'])
    ax.set_title(
        f'Feature V Pool — Neuron Embedding {method_label}\n'
        f'{n_fv} neurons colored by most selective POS (selectivity >= 1.5)',
        fontsize=S['font_size_subtitle'], fontweight='bold',
    )
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'fv_umap_pos.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description='Feature V Pool UMAP colored by POS selectivity')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--output', default='./section4_results')
    parser.add_argument('--pos_results', default=None,
                        help='Pre-computed pos_selectivity_fv.json (skips re-computation)')
    parser.add_argument('--ud_data', default=None, help='Local .conllu file')
    parser.add_argument('--ud_split', default='train')
    parser.add_argument('--max_sentences', type=int, default=2000)
    parser.add_argument('--multi_layer', action='store_true')
    parser.add_argument('--sel_threshold', type=float, default=1.5,
                        help='Min selectivity to assign POS color')
    parser.add_argument('--n_neighbors', type=int, default=15)
    parser.add_argument('--min_dist', type=float, default=0.1)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model_cls, params, _, config = load_model_jax(args.checkpoint)
    n_fv = config.get('n_feature_v', 0)
    print(f"  n_feature_v={n_fv}, d_space={config.get('d_space', '?')}")

    # --- Step 1: Extract FV embeddings ---
    print("\n=== Extracting Feature V embeddings ===")
    fv_emb = get_fv_embeddings(params, config)
    if fv_emb is None:
        print("  ERROR: Could not extract FV embeddings")
        return
    print(f"  FV embeddings: {fv_emb.shape}")

    # --- Step 2: UMAP / PCA ---
    if HAS_UMAP:
        print(f"\n=== UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist}) ===")
        reducer = UMAP(
            n_components=2, n_neighbors=args.n_neighbors,
            min_dist=args.min_dist, random_state=42, metric='cosine',
        )
        coords = reducer.fit_transform(fv_emb)
        method_label = 'UMAP'
    elif HAS_SKLEARN:
        print("\n=== PCA (UMAP not available) ===")
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(fv_emb)
        method_label = f'PCA (var: {pca.explained_variance_ratio_.sum():.1%})'
    else:
        print("  ERROR: Neither umap-learn nor sklearn available")
        return

    # --- Step 3: POS selectivity ---
    selectivity_matrix = None

    if args.pos_results and os.path.exists(args.pos_results):
        print(f"\n=== Loading pre-computed POS selectivity: {args.pos_results} ===")
        with open(args.pos_results) as f:
            pos_data = json.load(f)
        selectivity_matrix = np.array(pos_data['selectivity_matrix'])
        pos_tags = pos_data.get('pos_tags', UPOS_TAGS)
    else:
        print(f"\n=== Computing POS selectivity (fv, {args.max_sentences} sentences) ===")
        dataset = load_ud_ewt(args.ud_split, args.max_sentences, args.ud_data)
        pos_results, selectivity_matrix = analyze_pos_selectivity(
            model_cls, params, config, dataset,
            pool_type='fv',
            max_sentences=args.max_sentences,
            multi_layer=args.multi_layer,
        )
        pos_tags = pos_results.get('pos_tags', UPOS_TAGS)

        # Save for reuse
        json_path = os.path.join(args.output, 'pos_selectivity_fv.json')
        save_results(pos_results, json_path)
        print(f"  Saved POS results: {json_path}")

    # Assign top POS per neuron
    # selectivity_matrix: [n_pos, n_neurons_in_pool]
    # FV neurons are indices 0..n_fv-1 in this matrix (pool_type='fv')
    n_sel_neurons = selectivity_matrix.shape[1]
    if n_sel_neurons != n_fv:
        print(f"  Warning: selectivity has {n_sel_neurons} neurons, FV pool has {n_fv}")
        n_fv = min(n_fv, n_sel_neurons)
        coords = coords[:n_fv]
        fv_emb = fv_emb[:n_fv]

    top_pos, max_sel = assign_top_pos(selectivity_matrix, pos_tags, args.sel_threshold)
    top_pos = top_pos[:n_fv]
    max_sel = max_sel[:n_fv]

    n_assigned = sum(1 for p in top_pos if p is not None)
    print(f"\n  POS-assigned neurons: {n_assigned}/{n_fv} "
          f"(threshold={args.sel_threshold})")

    # Summary per POS
    from collections import Counter
    pos_dist = Counter(p for p in top_pos if p is not None)
    for pos, cnt in sorted(pos_dist.items(), key=lambda x: -x[1])[:10]:
        print(f"    {pos:8s}: {cnt}")

    # --- Step 4: Plot ---
    print(f"\n=== Plotting ===")
    plot_fv_umap(coords, top_pos, max_sel, n_fv, args.output,
                 method_label=method_label, dpi=args.dpi)

    # Save coordinates + metadata
    meta = {
        'method': method_label,
        'n_fv': n_fv,
        'n_assigned': n_assigned,
        'sel_threshold': args.sel_threshold,
        'n_neighbors': args.n_neighbors,
        'min_dist': args.min_dist,
        'pos_distribution': dict(pos_dist),
        'top_pos_per_neuron': top_pos,
        'max_selectivity_per_neuron': max_sel.tolist(),
        'coords': coords.tolist(),
    }
    meta_path = os.path.join(args.output, 'fv_umap_pos.json')
    save_results(meta, meta_path)
    print(f"  Saved: {meta_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
