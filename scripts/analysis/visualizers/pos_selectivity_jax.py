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
# Multi-layer routing extraction (partial forward)
# ============================================================

# Map pool_type to the variable returned by _router_attn_forward / _router_know_forward
_POOL_TO_ATTN_IDX = {
    'fqk_q': 0, 'fqk_k': 1, 'fv': 2,
    'rqk_q': 3, 'rqk_k': 4, 'rv': 5,
}
_POOL_IS_KNOW = {'fknow': 0, 'rknow': 1}


def _build_batched_forward(params, config, pool_type, pad_len):
    """Build a JIT-compiled batched forward that returns layer-averaged routing weights.

    All inputs MUST be padded to exactly ``pad_len`` so that the JIT-compiled
    function is traced once and reused for every batch (no recompilation).

    Args:
        params: FrozenDict model params
        config: Model config dict
        pool_type: Routing pool key
        pad_len: Fixed sequence length that all inputs will be padded to

    Returns:
        forward_fn(input_ids_jnp[B, pad_len]) -> weights [B, pad_len, N]
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

    is_attn_pool = pool_type in _POOL_TO_ATTN_IDX
    is_know_pool = pool_type in _POOL_IS_KNOW
    attn_idx = _POOL_TO_ATTN_IDX.get(pool_type)
    know_idx = _POOL_IS_KNOW.get(pool_type)

    block_params_list = [all_params[f'block_{i}'] for i in range(n_layers)]

    token_emb_table = all_params['token_emb']['embedding']
    pos_emb_table = all_params['pos_emb']['embedding']

    # Precompute position embeddings for the fixed pad_len OUTSIDE jit
    # so there is no dynamic shape inside the traced function.
    positions = jnp.arange(pad_len)                        # [pad_len]
    pos_emb_fixed = pos_emb_table[positions][jnp.newaxis, :]  # [1, pad_len, D]

    @jax.jit
    def _forward(input_ids):
        """input_ids: [B, pad_len] (int32, 0-padded) -> weights [B, pad_len, N]"""
        x = token_emb_table[input_ids] + pos_emb_fixed

        rng_key = jax.random.PRNGKey(0)
        weight_sum = None

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

            if is_attn_pool:
                w = attn_results[attn_idx]  # [B, pad_len, N]
                weight_sum = w if weight_sum is None else weight_sum + w

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

            if is_know_pool:
                w = fk_w if know_idx == 0 else rk_w
                weight_sum = w if weight_sum is None else weight_sum + w

            know_out = _knowledge_forward(
                normed, sn_params, fk_w, rk_w,
                0.0, True, rng_k,
            )
            x = x + know_out

        return weight_sum / n_layers

    return _forward


def _extract_multilayer_weights(
    params, config, token_ids_np, pool_type,
):
    """Run partial forward through all layers, return layer-averaged routing weights.

    Mirrors GPU POSNeuronAnalyzer.extract_routing_weights() which averages
    weights across all layers.

    Args:
        params: FrozenDict model params
        config: Model config dict
        token_ids_np: [1, S] int32 array
        pool_type: Routing pool key ('fv', 'fqk_q', 'rv', 'fknow', etc.)

    Returns:
        weights: [S, N] numpy array averaged across layers, or None
    """
    if not HAS_JAX:
        return None

    is_attn_pool = pool_type in _POOL_TO_ATTN_IDX
    is_know_pool = pool_type in _POOL_IS_KNOW
    if not is_attn_pool and not is_know_pool:
        return None

    seq_len = token_ids_np.shape[1]
    pad_len = seq_len  # single sentence: pad_len = actual length (concrete)
    forward_fn = _build_batched_forward(params, config, pool_type, pad_len)
    input_ids = jnp.array(token_ids_np)
    w = np.array(forward_fn(input_ids))
    if w.ndim == 3:
        return w[0].astype(np.float32)
    return w.astype(np.float32)


def _extract_multilayer_weights_batched(
    forward_fn, input_ids_batch, seq_lens,
):
    """Batched multi-layer extraction. Returns list of [S_i, N] arrays (unpadded).

    Args:
        forward_fn: JIT-compiled forward from _build_batched_forward
        input_ids_batch: [B, S_max] padded jnp array
        seq_lens: list of actual sequence lengths per batch item

    Returns:
        list of [S_i, N] numpy arrays
    """
    w_batch = np.array(forward_fn(input_ids_batch))  # [B, S_max, N]
    results = []
    for i, slen in enumerate(seq_lens):
        results.append(w_batch[i, :slen].astype(np.float32))
    return results


# ============================================================
# Analysis
# ============================================================

def _accumulate_weights(weight_sum, weight_count, pos_token_counts,
                        w, pos_tags, n_neurons):
    """Accumulate routing weights into POS accumulators for one sentence."""
    seq_len = min(len(pos_tags), w.shape[0])
    w = w[:seq_len]

    if w.shape[1] > n_neurons:
        w = w[:, :n_neurons]
    elif w.shape[1] < n_neurons:
        w = np.pad(w, ((0, 0), (0, n_neurons - w.shape[1])))

    pos_indices = np.array([POS_TO_IDX.get(t, -1)
                            for t in pos_tags[:seq_len]], dtype=np.int32)
    valid = pos_indices >= 0
    if valid.any():
        vp = pos_indices[valid]
        vw = w[valid]
        active = (vw > 0).astype(np.int64)
        np.add.at(weight_sum, vp, vw)
        np.add.at(weight_count, vp, active)
        np.add.at(pos_token_counts, vp, 1)


def analyze_pos_selectivity(
    model_cls, params, config, dataset,
    pool_type='fv', max_sentences=None, multi_layer=False,
    batch_size=16,
):
    """Compute POS selectivity matrix.

    For each (POS, neuron) pair, accumulates routing weights then computes:
        selectivity = mean_weight_given_pos / mean_weight_overall

    In multi-layer mode, sentences are batched (padded to max length within batch)
    and processed through a JIT-compiled full forward pass for efficiency.

    Returns dict with selectivity matrix, raw stats, top selective neurons.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    pool_type = resolve_pool_type(pool_type)

    model_instance = create_model_from_config(config)
    extractor = JAXRoutingDataExtractor(model_instance, params, config)

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

    weight_sum = np.zeros((n_pos, n_neurons), dtype=np.float64)
    weight_count = np.zeros((n_pos, n_neurons), dtype=np.int64)
    pos_token_counts = np.zeros(n_pos, dtype=np.int64)

    n_sents = min(len(dataset), max_sentences) if max_sentences else len(dataset)
    skipped = 0

    mode_label = 'multi-layer' if multi_layer else 'embedding-only'
    if multi_layer and not HAS_JAX:
        print("  Warning: JAX not available, falling back to embedding-only")
        multi_layer = False

    # ---- Phase 1: Pre-tokenize all sentences ----
    print(f"  Pre-tokenizing {n_sents} sentences...")
    tokenized = []
    for i in range(n_sents):
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
        tokenized.append((pos_tags, token_ids))

    print(f"  Tokenized {len(tokenized)}/{n_sents} sentences "
          f"(skipped {skipped})")

    # ---- Phase 2: Process ----
    if multi_layer:
        # Sort by length for efficient padding (less wasted compute)
        tokenized.sort(key=lambda x: len(x[1]))

        # Determine a fixed pad_len: use the longest sentence (clamped to max_seq_len)
        # so that JIT compiles exactly ONCE and every batch reuses the same trace.
        max_seq = config.get('max_seq_len', 512)
        pad_len = min(max(len(ids) for _, ids in tokenized), max_seq)
        # Round up to multiple of 32 for hardware alignment
        pad_len = ((pad_len + 31) // 32) * 32
        pad_len = min(pad_len, max_seq)

        # Build JIT-compiled forward with fixed pad_len and warm up
        forward_fn = _build_batched_forward(params, config, pool_type, pad_len)
        print(f"  JIT compiling batched forward "
              f"(batch_size={batch_size}, pad_len={pad_len})...",
              end=" ", flush=True)
        import time
        t0 = time.time()
        dummy = jnp.zeros((batch_size, pad_len), dtype=jnp.int32)
        _ = forward_fn(dummy)
        _.block_until_ready()
        print(f"done ({time.time() - t0:.1f}s)")

        # Process in batches
        n_batches = (len(tokenized) + batch_size - 1) // batch_size
        for bi in tqdm(range(n_batches),
                       desc=f'POS selectivity ({pool_type}, {mode_label})'):
            start = bi * batch_size
            end = min(start + batch_size, len(tokenized))
            batch_items = tokenized[start:end]

            # Pad all sentences to the fixed pad_len
            seq_lens = [min(len(ids), pad_len) for _, ids in batch_items]

            # Last batch may be smaller than batch_size — pad batch dim too
            padded = np.zeros((batch_size, pad_len), dtype=np.int32)
            for j, (_, ids) in enumerate(batch_items):
                slen = seq_lens[j]
                padded[j, :slen] = ids[:slen]

            # Batched forward (always [batch_size, pad_len] — fixed shape)
            w_list = _extract_multilayer_weights_batched(
                forward_fn, jnp.array(padded), seq_lens)

            # Accumulate per sentence (only real items, not padding rows)
            for j, (pos_tags, _) in enumerate(batch_items):
                w = w_list[j]
                if w is None:
                    continue
                _accumulate_weights(weight_sum, weight_count,
                                    pos_token_counts, w, pos_tags, n_neurons)
    else:
        # Embedding-only mode (already fast, no batching needed)
        for pos_tags, token_ids in tqdm(
                tokenized,
                desc=f'POS selectivity ({pool_type}, {mode_label})'):
            input_ids = np.array([token_ids], dtype=np.int32)
            routing_info = extractor.extract_routing(input_ids)
            routing = JAXRoutingData(routing_info)
            weights = routing.get_weight(pool_type)
            if weights is None:
                continue
            if weights.ndim == 3:
                w = weights[0]
            else:
                w = np.broadcast_to(weights[0:1],
                                    (len(token_ids), weights.shape[-1]))
            _accumulate_weights(weight_sum, weight_count,
                                pos_token_counts, w, pos_tags, n_neurons)

    print(f"  Processed {len(tokenized)}/{n_sents} sentences "
          f"(skipped {skipped})")

    # Compute selectivity matrix — frequency-based (paper Section 4.2)
    #
    # With top-k + renormalization, weight magnitudes are ~uniform (1/k),
    # so weight-based selectivity ≈ 1. Instead we use activation frequency:
    #   selectivity[p,n] = P(n active | POS=p) / P(n active | all)
    # This measures how much MORE often a neuron fires for a specific POS.
    #
    # weight_count[p, n] = number of POS-p tokens that activated neuron n
    # pos_token_counts[p] = total number of POS-p tokens

    pos_mask = pos_token_counts > 0
    total_tokens = pos_token_counts.sum()

    # Step 1: activation frequency per POS
    #   freq[p, n] = weight_count[p,n] / pos_token_counts[p]
    with np.errstate(divide='ignore', invalid='ignore'):
        activation_freq = np.where(
            pos_token_counts[:, np.newaxis] > 0,
            weight_count / pos_token_counts[:, np.newaxis],
            0.0,
        )

    # Step 2: overall activation frequency per neuron
    #   overall_freq[n] = sum_p(weight_count[p,n]) / total_tokens
    overall_count = weight_count.sum(axis=0)  # [n_neurons]
    with np.errstate(divide='ignore', invalid='ignore'):
        overall_freq = overall_count / max(total_tokens, 1)

    # Step 3: selectivity = activation_freq / overall_freq
    with np.errstate(divide='ignore', invalid='ignore'):
        selectivity = np.where(
            overall_freq > 0,
            activation_freq / overall_freq,
            0.0,
        )

    # Also compute mean_weight for backward compatibility (used in specialist stats)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_weight = np.where(
            weight_count > 0,
            weight_sum / weight_count,
            0.0,
        )

    # Neuron avg (for JSON output compatibility)
    neuron_avg = overall_freq

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
        'multi_layer': multi_layer,
        'n_layers': config.get('n_layers', 16) if multi_layer else 1,
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
    parser.add_argument('--multi_layer', action='store_true', default=True,
                        help='Average routing weights across all layers via partial forward '
                             '(matches GPU analysis). Default: True for paper-consistent results.')
    parser.add_argument('--embedding_only', action='store_true',
                        help='Use embedding-level routing only (fast, but single-layer). '
                             'Overrides --multi_layer.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for multi-layer mode (default: 16)')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    # --embedding_only overrides --multi_layer
    if args.embedding_only:
        args.multi_layer = False

    os.makedirs(args.output, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model_cls, params, _, config = load_model_jax(args.checkpoint)
    print(f"  d_model={config.get('d_model')}, n_layers={config.get('n_layers')}")
    if args.multi_layer:
        print(f"  Multi-layer mode: averaging across {config.get('n_layers', 16)} layers")

    # Load UD-EWT
    dataset = load_ud_ewt(args.ud_split, args.max_sentences, args.ud_data)

    # Analyze
    mode = 'multi-layer' if args.multi_layer else 'embedding-only'
    print(f"\n=== POS Selectivity Analysis (pool={args.pool_type}, {mode}) ===")
    results, selectivity = analyze_pos_selectivity(
        model_cls, params, config, dataset,
        pool_type=args.pool_type,
        max_sentences=args.max_sentences,
        multi_layer=args.multi_layer,
        batch_size=args.batch_size,
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

    # Specialist stats
    print(f"\n  POS specialist stats (selectivity > 2.0):")
    for pos in ['NOUN', 'VERB', 'ADJ', 'DET', 'ADP', 'PUNCT', 'PROPN', 'NUM']:
        stats = results.get('pos_specialist_stats', {}).get(pos)
        if stats:
            print(f"    {pos:8s}: {stats['n_specialists']:3d} specialists, "
                  f"{stats['n_sel_gt_1_5']:3d} sel>1.5, "
                  f"{stats['n_active']:4d} active")

    print(f"\n  Top selective neurons per POS (selectivity > 1):")
    for pos in ['NOUN', 'VERB', 'ADJ', 'DET', 'ADP', 'PUNCT']:
        top = results['top_selective_per_pos'].get(pos, [])
        if top:
            top3 = ', '.join(f'N{t["neuron"]}({t["selectivity"]:.2f}×)'
                              for t in top[:3])
            print(f"    {pos:8s}: {top3}")

    # Plot
    plot_pos_heatmap(selectivity, results, args.output,
                     pool_type=args.pool_type,
                     top_n_neurons=args.top_n_neurons, dpi=args.dpi)

    print("\nDone.")


if __name__ == '__main__':
    main()
