#!/usr/bin/env python3
"""
POS Selectivity Analysis (JAX/TPU) — Section 4.2
==================================================
Analyze per-neuron POS selectivity using UD English Web Treebank.

Key metric: Selectivity = E[weight|POS] / E[weight|all]
  - > 1: neuron is selective for this POS
  - = 1: uniform across POS
  - < 1: neuron avoids this POS

Outputs:
  - POS x Neuron heatmap (selectivity matrix)
  - JSON with per-neuron selectivity scores

Designed for single-host TPU v4-8.

Usage:
    python scripts/analysis/visualizers/pos_selectivity_jax.py \
        --checkpoint gs://dawn-tpu-data-c4/checkpoints/... \
        --output ./section4_results \
        --pool_type fv --max_sentences 2000
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from scripts.analysis.utils_jax import (
    load_model_jax, create_model_from_config,
    JAXRoutingDataExtractor, JAXRoutingData,
    WEIGHT_KEY_MAP, convert_to_serializable, save_results,
    resolve_pool_type,
)
from scripts.analysis.visualizers.style import PAPER_STYLE, apply_paper_style

if HAS_MATPLOTLIB:
    apply_paper_style(plt)
S = PAPER_STYLE

# Universal POS tags
UPOS_TAGS = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
    'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X',
]
POS_TO_IDX = {pos: i for i, pos in enumerate(UPOS_TAGS)}


# ============================================================
# UD-EWT Dataset Loading
# ============================================================

def load_ud_ewt(split='train', max_sentences=None, data_path=None):
    """Load UD English Web Treebank. Downloads if not cached locally.

    Returns list of {'tokens': [...], 'upos': [...]}.
    """
    try:
        import conllu
    except ImportError:
        raise ImportError("pip install conllu")

    if data_path and os.path.exists(data_path):
        print(f"Loading local conllu: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = f.read()
    else:
        urls = {
            'train': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu',
            'dev': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu',
            'test': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu',
        }
        import urllib.request
        url = urls.get(split, urls['train'])
        print(f"Downloading UD-EWT ({split})...")
        with urllib.request.urlopen(url) as resp:
            data = resp.read().decode('utf-8')

    sentences = conllu.parse(data)
    if max_sentences:
        sentences = sentences[:max_sentences]

    dataset = []
    for sent in sentences:
        tokens = [t['form'] for t in sent]
        upos = [t['upos'] for t in sent]
        dataset.append({'tokens': tokens, 'upos': upos})

    print(f"Loaded {len(dataset)} sentences")
    return dataset


# ============================================================
# Token-to-POS Alignment
# ============================================================

def align_tokens_to_pos(tokenizer, ud_tokens, ud_pos):
    """Map subword token IDs to POS tags via character spans.

    Returns (pos_tags, token_ids) for the tokenizer's subwords.
    """
    text = ""
    ud_spans = []
    for tok, pos in zip(ud_tokens, ud_pos):
        start = len(text)
        text += tok
        end = len(text)
        ud_spans.append((start, end, pos))
        text += " "
    text = text.rstrip()

    try:
        enc = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=True, return_tensors=None)
        token_ids = enc['input_ids']
        offsets = enc['offset_mapping']

        pos_tags = []
        for s, e in offsets:
            assigned = 'X'
            for us, ue, pos in ud_spans:
                if s < ue and e > us:
                    assigned = pos
                    break
            pos_tags.append(assigned)
        return pos_tags, token_ids

    except (TypeError, KeyError):
        # Fallback: sequential decode matching
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        pos_tags = []
        decoded = ""
        for tid in token_ids:
            decoded += tokenizer.decode([tid])
            assigned = 'X'
            for us, ue, pos in ud_spans:
                if len(decoded.strip()) <= ue + 1:
                    assigned = pos
                    break
            pos_tags.append(assigned)
        return pos_tags, token_ids


# ============================================================
# Analysis
# ============================================================

def analyze_pos_selectivity(
    model_cls, params, config, dataset,
    pool_type='fv', max_sentences=None,
):
    """Compute POS selectivity matrix.

    For each (POS, neuron) pair, accumulates routing weights then computes:
        selectivity = mean_weight_given_pos / mean_weight_overall

    Returns dict with selectivity matrix, raw stats, top selective neurons.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    pool_type = resolve_pool_type(pool_type)

    model_instance = create_model_from_config(config)
    extractor = JAXRoutingDataExtractor(model_instance, params, config)

    # Determine neuron count from pool type
    pool_to_n = {
        'fqk_q': config.get('n_feature_qk', 88),
        'fqk_k': config.get('n_feature_qk', 88),
        'fv': config.get('n_feature_v', 352),
        'rqk_q': config.get('n_restore_qk', 88),
        'rqk_k': config.get('n_restore_qk', 88),
        'rv': config.get('n_restore_v', 352),
        'fknow': config.get('n_feature_know', 224),
        'rknow': config.get('n_restore_know', 224),
    }
    n_neurons = pool_to_n.get(pool_type, 352)
    n_pos = len(UPOS_TAGS)

    # Accumulators
    weight_sum = np.zeros((n_pos, n_neurons), dtype=np.float64)
    weight_count = np.zeros((n_pos, n_neurons), dtype=np.int64)
    pos_token_counts = np.zeros(n_pos, dtype=np.int64)

    n_sents = min(len(dataset), max_sentences) if max_sentences else len(dataset)
    skipped = 0

    for i in tqdm(range(n_sents), desc=f'POS selectivity ({pool_type})'):
        sent = dataset[i]
        try:
            pos_tags, token_ids = align_tokens_to_pos(
                tokenizer, sent['tokens'], sent['upos'])
        except Exception:
            skipped += 1
            continue

        if not token_ids:
            skipped += 1
            continue

        # Get routing weights
        input_ids = np.array([token_ids], dtype=np.int32)
        routing_info = extractor.extract_routing(input_ids)
        routing = JAXRoutingData(routing_info)
        weights = routing.get_weight(pool_type)

        if weights is None:
            skipped += 1
            continue

        # weights: [1, S, N] or [1, N]
        if weights.ndim == 3:
            w = weights[0]  # [S, N]
        else:
            # Expand batch-level to all positions
            w = np.broadcast_to(weights[0:1], (len(token_ids), weights.shape[-1]))

        seq_len = min(len(pos_tags), w.shape[0])
        w = w[:seq_len]

        # Truncate/pad neuron dim
        if w.shape[1] > n_neurons:
            w = w[:, :n_neurons]
        elif w.shape[1] < n_neurons:
            w = np.pad(w, ((0, 0), (0, n_neurons - w.shape[1])))

        # Accumulate per POS
        pos_indices = np.array([POS_TO_IDX.get(pos_tags[j], -1)
                                for j in range(seq_len)], dtype=np.int32)
        valid = pos_indices >= 0
        if valid.any():
            vp = pos_indices[valid]
            vw = w[valid]
            active = (vw > 0).astype(np.int64)
            np.add.at(weight_sum, vp, vw)
            np.add.at(weight_count, vp, active)
            np.add.at(pos_token_counts, vp, 1)

    print(f"  Processed {n_sents - skipped}/{n_sents} sentences (skipped {skipped})")

    # Compute selectivity matrix — matches GPU formula in pos_neuron.py
    # Step 1: mean_weight[pos, neuron] = weight_sum / weight_count (active only)
    #   Uses weight_count (active tokens where w>0), NOT pos_token_counts (all tokens)
    #   NaN where neuron was never active for this POS
    pos_mask = pos_token_counts > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_weight = np.where(
            weight_count > 0,
            weight_sum / weight_count,
            np.nan,
        )

    # Step 2: neuron_avg = nanmean across POS (treats each POS equally)
    #   NOT global token-weighted mean — this ensures rare POS aren't down-weighted
    with np.errstate(divide='ignore', invalid='ignore'):
        neuron_avg = np.nanmean(mean_weight, axis=0)  # [n_neurons]

    # Step 3: selectivity = mean_weight / neuron_avg
    with np.errstate(divide='ignore', invalid='ignore'):
        selectivity = np.where(
            neuron_avg > 0,
            mean_weight / neuron_avg,
            0.0,
        )

    # Clean up NaN/inf
    mean_weight = np.nan_to_num(mean_weight, nan=0.0)
    selectivity = np.nan_to_num(selectivity, nan=0.0, posinf=0.0, neginf=0.0)

    # Top selective neurons per POS + specialist stats (matches GPU)
    SPECIALIST_SEL_THRESHOLD = 2.0
    SPECIALIST_MW_THRESHOLD = 0.1

    top_per_pos = {}
    pos_specialist_stats = {}
    for p, pos in enumerate(UPOS_TAGS):
        if not pos_mask[p]:
            continue
        mw = mean_weight[p]
        sel = selectivity[p]
        active_mask = mw > 0

        if not active_mask.any():
            continue

        n_sel_gt_1_5 = int(((sel > 1.5) & active_mask).sum())
        n_sel_gt_2 = int(((sel > SPECIALIST_SEL_THRESHOLD) & active_mask).sum())
        n_specialists = int(((sel > SPECIALIST_SEL_THRESHOLD) & (mw > SPECIALIST_MW_THRESHOLD)).sum())

        pos_specialist_stats[pos] = {
            'n_active': int(active_mask.sum()),
            'n_sel_gt_1_5': n_sel_gt_1_5,
            'n_sel_gt_2': n_sel_gt_2,
            'n_specialists': n_specialists,
        }

        # Top 20 neurons (matches GPU)
        top_idx = np.argsort(sel)[-20:][::-1]
        top_per_pos[pos] = [
            {'neuron': int(idx), 'selectivity': float(sel[idx]),
             'mean_weight': float(mw[idx]),
             'is_specialist': bool(sel[idx] > SPECIALIST_SEL_THRESHOLD and mw[idx] > SPECIALIST_MW_THRESHOLD)}
            for idx in top_idx if sel[idx] > 1.0
        ]

    results = {
        'pool_type': pool_type,
        'n_neurons': n_neurons,
        'n_sentences': n_sents - skipped,
        'pos_token_counts': {UPOS_TAGS[p]: int(pos_token_counts[p])
                             for p in range(n_pos) if pos_token_counts[p] > 0},
        'selectivity_matrix': selectivity.tolist(),
        'mean_weight_matrix': mean_weight.tolist(),
        'pos_tags': UPOS_TAGS,
        'top_selective_per_pos': top_per_pos,
        'pos_specialist_stats': pos_specialist_stats,
        'neuron_avg': neuron_avg.tolist(),
    }
    return results, selectivity


# ============================================================
# Visualization
# ============================================================

def plot_pos_heatmap(selectivity, results, output_dir, pool_type='fv',
                     top_n_neurons=50, dpi=300):
    """POS x Neuron selectivity heatmap."""
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        print("matplotlib/seaborn not available, skipping heatmap")
        return None

    # Filter to POS tags that have data
    pos_counts = results['pos_token_counts']
    valid_pos = [p for p in UPOS_TAGS if p in pos_counts]
    valid_idx = [UPOS_TAGS.index(p) for p in valid_pos]

    if not valid_pos:
        print("No valid POS data, skipping heatmap")
        return None

    mat = selectivity[valid_idx]  # [n_valid_pos, N]

    # Select top neurons by max selectivity across any POS
    max_sel = mat.max(axis=0)
    top_neurons = np.argsort(max_sel)[-top_n_neurons:][::-1]
    mat_top = mat[:, top_neurons]

    fig, ax = plt.subplots(figsize=(max(12, top_n_neurons * 0.28),
                                     max(4, len(valid_pos) * 0.45)))

    sns.heatmap(
        mat_top,
        xticklabels=[str(n) for n in top_neurons],
        yticklabels=valid_pos,
        cmap='YlOrRd',
        center=1.0,
        vmin=0, vmax=min(mat_top.max(), 5.0),
        ax=ax,
        cbar_kws={'label': 'Selectivity (E[w|POS] / E[w|all])'},
        linewidths=0.3,
    )

    ax.set_xlabel('Neuron Index', fontsize=S['font_size_label'])
    ax.set_ylabel('POS Tag', fontsize=S['font_size_label'])
    ax.set_title(f'POS Selectivity — pool={pool_type}  (top {top_n_neurons} neurons)',
                  fontsize=S['font_size_subtitle'], fontweight='bold')
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=S['font_size_tick'])
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'pos_selectivity_{pool_type}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='POS Selectivity Analysis (Section 4.2)')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--output', default='./section4_results', help='Output directory')
    parser.add_argument('--pool_type', default='fv', help='Routing pool (fv, fqk_q, rv, ...)')
    parser.add_argument('--ud_split', default='train', choices=['train', 'dev', 'test'])
    parser.add_argument('--ud_data', default=None, help='Local .conllu file path')
    parser.add_argument('--max_sentences', type=int, default=2000)
    parser.add_argument('--top_n_neurons', type=int, default=50)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model_cls, params, config = load_model_jax(args.checkpoint)
    print(f"  d_model={config.get('d_model')}, n_layers={config.get('n_layers')}")

    # Load UD-EWT
    dataset = load_ud_ewt(args.ud_split, args.max_sentences, args.ud_data)

    # Analyze
    print(f"\n=== POS Selectivity Analysis (pool={args.pool_type}) ===")
    results, selectivity = analyze_pos_selectivity(
        model_cls, params, config, dataset,
        pool_type=args.pool_type,
        max_sentences=args.max_sentences,
    )

    # Save JSON
    json_path = os.path.join(args.output, f'pos_selectivity_{args.pool_type}.json')
    save_results(results, json_path)
    print(f"  Saved: {json_path}")

    # Print summary
    print(f"\n  Sentences processed: {results['n_sentences']}")
    print(f"  Token counts by POS:")
    for pos, cnt in sorted(results['pos_token_counts'].items(),
                            key=lambda x: -x[1])[:10]:
        print(f"    {pos:8s}: {cnt:6d}")

    print(f"\n  Top selective neurons per POS (selectivity > 1):")
    for pos in ['NOUN', 'VERB', 'ADJ', 'DET', 'ADP', 'PUNCT']:
        top = results['top_selective_per_pos'].get(pos, [])
        if top:
            top3 = ', '.join(f'N{t["neuron"]}({t["selectivity"]:.2f})'
                              for t in top[:3])
            print(f"    {pos:8s}: {top3}")

    # Plot
    plot_pos_heatmap(selectivity, results, args.output,
                     pool_type=args.pool_type,
                     top_n_neurons=args.top_n_neurons, dpi=args.dpi)

    print("\nDone.")


if __name__ == '__main__':
    main()
