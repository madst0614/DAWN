#!/usr/bin/env python3
"""
DAWN Pseudo-Neuron Suppression Experiment (JAX/TPU)
=====================================================
JAX native version — runs directly on TPU without PyTorch.

Experiment protocol (same as PyTorch version):
  Phase 1: Collect activation frequencies per pool at target position
  Phase 2: Identify capital-related neurons (≥threshold), build -inf masks
  Phase 3: Run suppressed forward, measure target token hit rate delta

Suppression mechanism:
  JAX is pure-functional — no monkey-patching.
  Instead, we write a custom forward that injects jnp.where(mask, -inf, logits)
  between logit computation and softmax inside _router_attn_forward / _router_know_forward.
  Masks are static arrays → XLA fuses them into a single HLO, zero overhead.

Usage:
    python scripts/analysis/standalone/neuron_suppression_experiment_jax.py \\
        --checkpoint ~/dawn-tpu-data-c4/checkpoints/dawn_v17_1_400M_c4_20B_v4_32/run_v17.1_20260210_160828_3201 \\
        --n_runs 100 \\
        --threshold 0.7 \\
        --output results/suppression/
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import json
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from scripts.analysis.utils_jax import load_model_jax, create_model_from_config
from models.model_v17_1_jax import (
    safe_dropout, topk_sparsify, feature_fn, restore_fn, _layer_norm,
    _attention_forward, _knowledge_forward,
    _router_attn_forward, _router_know_forward,
    dawn_init_kv_cache, dawn_cached_forward_with_routing,
)


# ============================================================
# Pool definitions
# ============================================================

ATTENTION_POOLS = {
    'fqk_Q': 'fqk_weights_Q',
    'fqk_K': 'fqk_weights_K',
    'fv':    'fv_weights',
    'rqk_Q': 'rqk_weights_Q',
    'rqk_K': 'rqk_weights_K',
    'rv':    'rv_weights',
}

KNOWLEDGE_POOLS = {
    'feature_know': 'feature_know_w',
    'restore_know': 'restore_know_w',
}

ALL_POOL_NAMES = list(ATTENTION_POOLS.keys()) + list(KNOWLEDGE_POOLS.keys())


# ============================================================
# Default queries
# ============================================================

DEFAULT_CAPITAL_QUERIES = [
    {"prompt": "the capital of france is", "target": "paris"},
    {"prompt": "the capital of japan is",  "target": "tokyo"},
    {"prompt": "the capital of the united kingdom is", "target": "london"},
]

DEFAULT_CONTROL_QUERIES = [
    {"prompt": "the sky is",           "target": "blue"},
    {"prompt": "water is made of",     "target": "hydrogen"},
    {"prompt": "the sun rises in the", "target": "east"},
]


# ============================================================
# Domain-specific presets
# ============================================================

QUERY_PRESETS = {
    'capital': {
        'description': 'Capital city knowledge (original experiment)',
        'target_queries': DEFAULT_CAPITAL_QUERIES,
        'control_queries': DEFAULT_CONTROL_QUERIES,
    },
    'physics': {
        'description': 'Physics/astronomy vs biology/geography/history',
        'target_queries': [
            {"prompt": "light travels at the speed of",  "target": "light"},
            {"prompt": "the earth orbits the",           "target": "sun"},
            {"prompt": "the earth revolves around the",  "target": "sun"},
        ],
        'control_queries': [
            {"prompt": "plants need sunlight to",         "target": "grow"},
            {"prompt": "the amazon is the longest",       "target": "river"},
            {"prompt": "the lungs are used for",          "target": "breathing"},
            {"prompt": "the french revolution began in",  "target": "1789"},
            {"prompt": "mount everest is the",            "target": "highest"},
        ],
    },
}


# ============================================================
# Suppressed router pure functions (Phase 2 core)
# ============================================================

def _suppressed_router_attn_forward(
    x, router_params,
    n_feature_qk, n_feature_v, n_restore_qk, n_restore_v, d_space,
    top_k_feature_qk, top_k_feature_v, top_k_restore_qk, top_k_restore_v,
    router_dropout, attention_mask, deterministic, rng,
    # Suppression masks: bool [N_pool], True = suppress
    mask_fqk_Q, mask_fqk_K, mask_fv, mask_rqk_Q, mask_rqk_K, mask_rv,
):
    """
    _router_attn_forward with -inf mask injection.

    Identical to models.model_v17_1_jax._router_attn_forward except:
    logits = jnp.where(mask, -inf, logits) inserted before softmax.
    """
    nr = router_params['neuron_router']
    neuron_emb = nr['neuron_emb']
    emb_norm = neuron_emb / (jnp.linalg.norm(neuron_emb, axis=-1, keepdims=True) + 1e-8)

    fqk_end = n_feature_qk
    fv_end = fqk_end + n_feature_v
    rqk_end = fv_end + n_restore_qk
    rv_end = rqk_end + n_restore_v

    rng, rng1 = jax.random.split(rng)
    all_proj = x @ nr['proj_all']['kernel'] + nr['proj_all']['bias']
    all_proj = safe_dropout(all_proj, router_dropout, deterministic, rng1)
    h_fqk_Q, h_fqk_K, h_fv, h_rqk_Q, h_rqk_K, h_rv = jnp.split(all_proj, 6, axis=-1)

    fqk_emb = emb_norm[:fqk_end]
    fv_emb = emb_norm[fqk_end:fv_end]
    rqk_emb = emb_norm[fv_end:rqk_end]
    rv_emb = emb_norm[rqk_end:rv_end]

    logits_fqk_Q = jnp.einsum('bsd,nd->bsn', h_fqk_Q, fqk_emb)
    logits_fqk_K = jnp.einsum('bsd,nd->bsn', h_fqk_K, fqk_emb)
    logits_fv = jnp.einsum('bsd,nd->bsn', h_fv, fv_emb)
    logits_rqk_Q = jnp.einsum('bsd,nd->bsn', h_rqk_Q, rqk_emb)
    logits_rqk_K = jnp.einsum('bsd,nd->bsn', h_rqk_K, rqk_emb)
    logits_rv = jnp.einsum('bsd,nd->bsn', h_rv, rv_emb)

    # --- Suppression: -inf before softmax ---
    neg_inf = jnp.finfo(logits_fqk_Q.dtype).min
    logits_fqk_Q = jnp.where(mask_fqk_Q, neg_inf, logits_fqk_Q)
    logits_fqk_K = jnp.where(mask_fqk_K, neg_inf, logits_fqk_K)
    logits_fv = jnp.where(mask_fv, neg_inf, logits_fv)
    logits_rqk_Q = jnp.where(mask_rqk_Q, neg_inf, logits_rqk_Q)
    logits_rqk_K = jnp.where(mask_rqk_K, neg_inf, logits_rqk_K)
    logits_rv = jnp.where(mask_rv, neg_inf, logits_rv)

    fqk_pref_Q = jax.nn.softmax(logits_fqk_Q, axis=-1)
    fqk_pref_K = jax.nn.softmax(logits_fqk_K, axis=-1)
    fv_pref = jax.nn.softmax(logits_fv, axis=-1)
    rqk_pref_Q = jax.nn.softmax(logits_rqk_Q, axis=-1)
    rqk_pref_K = jax.nn.softmax(logits_rqk_K, axis=-1)
    rv_pref = jax.nn.softmax(logits_rv, axis=-1)

    # Skip aux_loss for inference
    aux_loss = jnp.float32(0.0)

    fqk_weights_Q, _ = topk_sparsify(fqk_pref_Q, top_k_feature_qk)
    fqk_weights_K, _ = topk_sparsify(fqk_pref_K, top_k_feature_qk)
    fv_weights, _ = topk_sparsify(fv_pref, top_k_feature_v)
    rqk_weights_Q, _ = topk_sparsify(rqk_pref_Q, top_k_restore_qk)
    rqk_weights_K, _ = topk_sparsify(rqk_pref_K, top_k_restore_qk)
    rv_weights, _ = topk_sparsify(rv_pref, top_k_restore_v)

    return (fqk_weights_Q, fqk_weights_K, fv_weights,
            rqk_weights_Q, rqk_weights_K, rv_weights,
            aux_loss)


def _suppressed_router_know_forward(
    x, router_params,
    n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
    n_feature_know, n_restore_know,
    top_k_feature_know, top_k_restore_know,
    router_dropout, attention_mask, deterministic, rng,
    # Suppression masks: bool [N_pool], True = suppress
    mask_feature_know, mask_restore_know,
):
    """
    _router_know_forward with -inf mask injection.
    """
    nr = router_params['neuron_router']
    neuron_emb = nr['neuron_emb']
    emb_norm = neuron_emb / (jnp.linalg.norm(neuron_emb, axis=-1, keepdims=True) + 1e-8)

    rv_end = n_feature_qk + n_feature_v + n_restore_qk + n_restore_v
    fk_end = rv_end + n_feature_know

    rng, rng1, rng2 = jax.random.split(rng, 3)

    h_fk = x @ nr['proj_feature_know']['kernel'] + nr['proj_feature_know']['bias']
    h_fk = safe_dropout(h_fk, router_dropout, deterministic, rng1)
    emb_fk = emb_norm[rv_end:fk_end]
    logits_fk = jnp.einsum('bsd,nd->bsn', h_fk, emb_fk)

    h_rk = x @ nr['proj_restore_know']['kernel'] + nr['proj_restore_know']['bias']
    h_rk = safe_dropout(h_rk, router_dropout, deterministic, rng2)
    emb_rk = emb_norm[fk_end:]
    logits_rk = jnp.einsum('bsd,nd->bsn', h_rk, emb_rk)

    # --- Suppression: -inf before softmax ---
    neg_inf = jnp.finfo(logits_fk.dtype).min
    logits_fk = jnp.where(mask_feature_know, neg_inf, logits_fk)
    logits_rk = jnp.where(mask_restore_know, neg_inf, logits_rk)

    pref_f = jax.nn.softmax(logits_fk, axis=-1)
    pref_r = jax.nn.softmax(logits_rk, axis=-1)

    # Skip aux_loss for inference
    aux_loss = jnp.float32(0.0)

    feature_know_w, _ = topk_sparsify(pref_f, top_k_feature_know)
    restore_know_w, _ = topk_sparsify(pref_r, top_k_restore_know)

    return feature_know_w, restore_know_w, aux_loss


# ============================================================
# Custom forward with suppression (replaces model.apply)
# ============================================================

def build_suppressed_forward(model, params, config, masks):
    """
    Build a JIT-compiled forward function with suppression masks baked in.

    Args:
        model: DAWN JAX model instance (for config only)
        params: FrozenDict {'params': {...}}
        config: model config dict
        masks: dict of pool_name → jnp.array bool [N_pool]
               Missing pools get all-False (no suppression).

    Returns:
        jit-compiled fn(input_ids) → logits [B, S, V]
    """
    all_params = params['params']
    sn_params = all_params['shared_neurons']
    router_params = all_params['router']

    token_emb_table = all_params['token_emb']['embedding']  # [V, D]
    pos_emb_table = all_params['pos_emb']['embedding']      # [max_seq, D]
    norm_scale = all_params['norm']['scale']
    norm_bias = all_params['norm']['bias']

    n_layers = config.get('n_layers', 16)
    n_feature_qk = config.get('n_feature_qk', 88)
    n_feature_v = config.get('n_feature_v', 352)
    n_restore_qk = config.get('n_restore_qk', 88)
    n_restore_v = config.get('n_restore_v', 352)
    n_feature_know = config.get('n_feature_know', 224)
    n_restore_know = config.get('n_restore_know', 224)
    n_heads = config.get('n_heads', 8)
    d_model = config.get('d_model', 768)
    d_space = config.get('d_space', 256)
    top_k_fqk = config.get('top_k_feature_qk', 16)
    top_k_fv = config.get('top_k_feature_v', 16)
    top_k_rqk = config.get('top_k_restore_qk', 16)
    top_k_rv = config.get('top_k_restore_v', 16)
    top_k_fk = config.get('top_k_feature_know', 16)
    top_k_rk = config.get('top_k_restore_know', 16)

    # Build bool masks — default all-False (no suppression)
    m_fqk_Q = masks.get('fqk_Q', jnp.zeros(n_feature_qk, dtype=jnp.bool_))
    m_fqk_K = masks.get('fqk_K', jnp.zeros(n_feature_qk, dtype=jnp.bool_))
    m_fv = masks.get('fv', jnp.zeros(n_feature_v, dtype=jnp.bool_))
    m_rqk_Q = masks.get('rqk_Q', jnp.zeros(n_restore_qk, dtype=jnp.bool_))
    m_rqk_K = masks.get('rqk_K', jnp.zeros(n_restore_qk, dtype=jnp.bool_))
    m_rv = masks.get('rv', jnp.zeros(n_restore_v, dtype=jnp.bool_))
    m_fknow = masks.get('feature_know', jnp.zeros(n_feature_know, dtype=jnp.bool_))
    m_rknow = masks.get('restore_know', jnp.zeros(n_restore_know, dtype=jnp.bool_))

    # Stack block params
    block_params_list = [all_params[f'block_{i}'] for i in range(n_layers)]
    stacked_block_params = jax.tree.map(
        lambda *arrays: jnp.stack(arrays), *block_params_list)

    @jax.jit
    def forward(input_ids):
        B, S = input_ids.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = jnp.take(token_emb_table, input_ids, axis=0) + jnp.take(pos_emb_table, positions, axis=0)

        rng = jax.random.PRNGKey(0)
        layer_rngs = jax.random.split(rng, n_layers)

        deterministic = True
        attention_mask = None

        def scan_body(carry, xs):
            x = carry
            bp = xs['params']
            rng = xs['rng']
            rng, rng_attn_router, rng_know_router, rng_attn, rng_know = \
                jax.random.split(rng, 5)

            # --- Attention ---
            normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])

            (fqk_w_Q, fqk_w_K, fv_w, rqk_w_Q, rqk_w_K, rv_w,
             _) = _suppressed_router_attn_forward(
                normed, router_params,
                n_feature_qk, n_feature_v, n_restore_qk, n_restore_v, d_space,
                top_k_fqk, top_k_fv, top_k_rqk, top_k_rv,
                0.0, attention_mask, deterministic, rng_attn_router,
                m_fqk_Q, m_fqk_K, m_fv, m_rqk_Q, m_rqk_K, m_rv)

            attn_out = _attention_forward(
                normed, sn_params,
                fqk_w_Q, fqk_w_K, fv_w, rqk_w_Q, rqk_w_K, rv_w,
                bp['attn']['expand_O']['kernel'],
                n_feature_qk, n_restore_qk, n_heads, d_model,
                0.0, deterministic, rng_attn)

            x = x + attn_out

            # --- Knowledge ---
            normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])

            feat_know_w, rest_know_w, _ = _suppressed_router_know_forward(
                normed, router_params,
                n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
                n_feature_know, n_restore_know,
                top_k_fk, top_k_rk,
                0.0, attention_mask, deterministic, rng_know_router,
                m_fknow, m_rknow)

            know_out = _knowledge_forward(
                normed, sn_params,
                feat_know_w, rest_know_w,
                0.0, deterministic, rng_know)

            x = x + know_out
            return x, None

        xs = {'params': stacked_block_params, 'rng': layer_rngs}
        x, _ = jax.lax.scan(scan_body, x, xs)

        # Final norm + logits
        x = _layer_norm(x, norm_scale, norm_bias)
        logits = x @ token_emb_table.T  # weight tying
        return logits

    return forward


def build_masks_from_sets(suppressed_neurons, config):
    """
    Convert {pool_name: set(int)} → {pool_name: jnp.bool[N_pool]}.
    """
    pool_sizes = {
        'fqk_Q': config.get('n_feature_qk', 88),
        'fqk_K': config.get('n_feature_qk', 88),
        'fv':    config.get('n_feature_v', 352),
        'rqk_Q': config.get('n_restore_qk', 88),
        'rqk_K': config.get('n_restore_qk', 88),
        'rv':    config.get('n_restore_v', 352),
        'feature_know': config.get('n_feature_know', 224),
        'restore_know': config.get('n_restore_know', 224),
    }
    masks = {}
    for pool_name, indices in suppressed_neurons.items():
        if not indices:
            continue
        size = pool_sizes.get(pool_name)
        if size is None:
            continue
        m = np.zeros(size, dtype=bool)
        for idx in indices:
            if 0 <= idx < size:
                m[idx] = True
        masks[pool_name] = jnp.array(m)
    return masks


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='DAWN Pseudo-Neuron Suppression Experiment (JAX/TPU)',
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .flax checkpoint (file or directory)')
    parser.add_argument('--min_target_count', type=int, default=100,
                        help='Min target token hits for domain queries (default: 100)')
    parser.add_argument('--control_min_target_count', type=int, default=20,
                        help='Min target token hits for control queries (default: 20). '
                             'Control queries are not used for neuron selection.')
    parser.add_argument('--max_runs', type=int, default=500,
                        help='Max generation runs per query')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k_sampling', type=int, default=50,
                        help='Top-k for sampling (0=greedy)')
    parser.add_argument('--top_n_pct', type=float, default=0.10,
                        help='Suppress top N%% neurons by contrastive score '
                             '(default: 0.10 = top 10%%)')
    parser.add_argument('--mode', type=str, default='intersection',
                        choices=['intersection', 'union'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--queries', type=str, default=None,
                        help='Custom queries JSON file')
    parser.add_argument('--preset', type=str, default=None,
                        choices=list(QUERY_PRESETS.keys()),
                        help='Use a built-in query preset '
                             f'({", ".join(QUERY_PRESETS.keys())})')
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Loading model from: {args.checkpoint}")

    model_cls, params, tokenizer, config = load_model_jax(args.checkpoint)
    model = create_model_from_config(config)

    print(f"  Model version: {config.get('model_version', 'unknown')}")
    print(f"  Pools: FQK={config.get('n_feature_qk')}, FV={config.get('n_feature_v')}, "
          f"RQK={config.get('n_restore_qk')}, RV={config.get('n_restore_v')}, "
          f"FK={config.get('n_feature_know')}, RK={config.get('n_restore_know')}")

    # Load queries: --preset > --queries > defaults
    capital_queries = DEFAULT_CAPITAL_QUERIES
    control_queries = DEFAULT_CONTROL_QUERIES
    target_label = 'target'
    control_label = 'control'
    if args.preset:
        preset = QUERY_PRESETS[args.preset]
        capital_queries = preset['target_queries']
        control_queries = preset['control_queries']
        target_label = args.preset
        print(f"  Preset: {args.preset} — {preset['description']}")
    elif args.queries:
        with open(args.queries) as f:
            qdata = json.load(f)
        capital_queries = qdata.get('capital', capital_queries)
        control_queries = qdata.get('control', control_queries)

    # Run experiment
    experiment = NeuronSuppressionExperimentJAX(
        model, params, config, tokenizer
    )
    results = experiment.run_full_experiment(
        capital_queries=capital_queries,
        control_queries=control_queries,
        min_target_count=args.min_target_count,
        max_runs=args.max_runs,
        temperature=args.temperature,
        top_k_sampling=args.top_k_sampling,
        top_n_pct=args.top_n_pct,
        mode=args.mode,
        target_label=target_label,
        control_label=control_label,
        control_min_target_count=args.control_min_target_count,
    )

    # Save
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = Path(args.checkpoint).name or 'checkpoint'
        filename = f"suppression_jax_{ckpt_name}_pct{args.top_n_pct}_{args.mode}.json"
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(make_serializable(results), f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


def make_serializable(obj):
    """Convert JAX/numpy types for JSON."""
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    try:
        return np.asarray(obj).tolist()
    except (TypeError, ValueError):
        pass
    return obj


class NeuronSuppressionExperimentJAX:
    def __init__(self, model, params, config, tokenizer):
        self.model = model
        self.params = params
        self.config = config
        self.tokenizer = tokenizer

        # Baseline forward uses model.apply (the real model, not custom forward)
        @jax.jit
        def _baseline_forward(input_ids):
            result = model.apply(
                params, input_ids,
                deterministic=True,
                rngs={'dropout': jax.random.PRNGKey(0)},
            )
            return result['logits']

        self._baseline_forward = _baseline_forward

    # ----------------------------------------------------------
    # Phase 1: Collect activation frequencies (paper method)
    # ----------------------------------------------------------
    # Mirrors behavioral.py:analyze_factual_neurons (GPU) and
    # knowledge_coherence_jax.py:analyze_knowledge_coherence (TPU).
    #
    # Method: sampling-based autoregressive generation.
    #   1. Generate tokens autoregressively (temperature=1, top_k=50)
    #   2. At each step, extract routing weights from KV-cached forward
    #   3. If generated token == target → record as "target hit"
    #      otherwise → record as "baseline"
    #   4. Repeat until min_target_count hits collected
    #   5. Compute contrastive score per neuron:
    #      target_freq / target_runs - baseline_freq / baseline_steps
    # ----------------------------------------------------------

    def collect_activation_frequencies(
        self, prompt, target_token,
        min_target_count=100, max_runs=500, max_tokens_per_run=200,
        temperature=1.0, top_k=50,
    ):
        """Paper method: sampling generation + routing at target hit."""
        target_ids = self.tokenizer.encode(target_token, add_special_tokens=False)
        if not target_ids:
            raise ValueError(f"Target token '{target_token}' not in vocabulary")
        target_token_id = target_ids[0]
        target_lower = target_token.strip().lower()

        target_tokens_decoded = self.tokenizer.convert_ids_to_tokens(target_ids)
        print(f"    [tokenizer] '{target_token}' -> ids={target_ids}, "
              f"tokens={target_tokens_decoded}")

        prompt_ids = [101] + self.tokenizer.encode(prompt, add_special_tokens=False)  # [CLS] only, no [SEP]
        prompt_len = len(prompt_ids)

        # Pool name → routing_info key mapping
        _POOL_KEY = {
            'fqk_Q': 'fqk_wQ', 'fqk_K': 'fqk_wK',
            'fv': 'fv_w', 'rqk_Q': 'rqk_wQ', 'rqk_K': 'rqk_wK',
            'rv': 'rv_w',
            'feature_know': 'fknow_w', 'restore_know': 'rknow_w',
        }

        # Pool size bounds for validation (matching knowledge_coherence_jax)
        _POOL_SIZES = {
            'fqk_Q': self.config.get('n_feature_qk', 88),
            'fqk_K': self.config.get('n_feature_qk', 88),
            'fv':    self.config.get('n_feature_v', 352),
            'rqk_Q': self.config.get('n_restore_qk', 88),
            'rqk_K': self.config.get('n_restore_qk', 88),
            'rv':    self.config.get('n_restore_v', 352),
            'feature_know': self.config.get('n_feature_know', 224),
            'restore_know': self.config.get('n_restore_know', 224),
        }

        target_neuron_counts = {pool: defaultdict(int) for pool in ALL_POOL_NAMES}
        baseline_neuron_counts = {pool: defaultdict(int) for pool in ALL_POOL_NAMES}
        successful_runs = 0
        total_runs = 0
        total_baseline_steps = 0
        sample_generations = []

        # Different seed per prompt for diverse sampling
        prompt_hash = hash(prompt) & 0xFFFFFFFF
        rng_key = jax.random.PRNGKey(prompt_hash)

        phase1_t0 = time.time()
        while successful_runs < min_target_count and total_runs < max_runs:
            total_runs += 1
            if total_runs % 10 == 0 or successful_runs == min_target_count:
                elapsed_p1 = time.time() - phase1_t0
                rate = successful_runs / elapsed_p1 if elapsed_p1 > 0 and successful_runs > 0 else 0
                eta = (min_target_count - successful_runs) / rate if rate > 0 else 0
                print(f"\r      {successful_runs}/{min_target_count} hits "
                      f"(run {total_runs}/{max_runs}) "
                      f"[{elapsed_p1:.0f}s elapsed, ~{eta:.0f}s remaining]",
                      end='', flush=True)

            # Init KV cache + prefill with routing
            kv_k, kv_v = dawn_init_kv_cache(self.config, batch_size=1)
            prompt_2d = jnp.array(np.array(prompt_ids)[np.newaxis, :])
            logits, kv_k, kv_v, routing_info = self._decode_step_with_routing(
                self.params, prompt_2d, kv_k, kv_v, 0)

            generated_ids = list(prompt_ids)
            cache_pos = prompt_len

            # Sample first token
            first_logits = np.array(logits[0, -1, :])
            rng_key, subkey = jax.random.split(rng_key)
            next_token = self._sample_token(first_logits, temperature, top_k, subkey)
            generated_ids.append(next_token)

            prev_routing_info = routing_info

            for step in range(max_tokens_per_run - 1):
                token_text = self.tokenizer.decode([next_token]).strip().lower()

                # Extract active neurons from routing (union across layers)
                step_neurons = self._extract_active_from_routing(
                    prev_routing_info, _POOL_KEY)

                if token_text == target_lower or next_token == target_token_id:
                    # Target hit — record routing
                    for pool in ALL_POOL_NAMES:
                        for n in step_neurons[pool]:
                            if n < _POOL_SIZES.get(pool, 0):
                                target_neuron_counts[pool][n] += 1
                    successful_runs += 1
                    if len(sample_generations) < 3:
                        gen_text = self.tokenizer.decode(
                            generated_ids, skip_special_tokens=True)
                        sample_generations.append(gen_text)
                    break
                else:
                    # Baseline step
                    for pool in ALL_POOL_NAMES:
                        for n in step_neurons[pool]:
                            if n < _POOL_SIZES.get(pool, 0):
                                baseline_neuron_counts[pool][n] += 1
                    total_baseline_steps += 1

                if next_token in (self.tokenizer.sep_token_id,
                                  self.tokenizer.eos_token_id):
                    break
                if cache_pos >= self.config.get('max_seq_len', 512) - 1:
                    break

                # Decode next with routing
                token_2d = jnp.array([[next_token]])
                logits, kv_k, kv_v, routing_info = \
                    self._decode_step_with_routing(
                        self.params, token_2d, kv_k, kv_v, cache_pos)
                prev_routing_info = routing_info
                cache_pos += 1

                next_logits = np.array(logits[0, 0, :])
                rng_key, subkey = jax.random.split(rng_key)
                next_token = self._sample_token(
                    next_logits, temperature, top_k, subkey)
                generated_ids.append(next_token)

        print(f"\r      {successful_runs}/{min_target_count} hits "
              f"(run {total_runs}) — Done!          ")

        match_rate = successful_runs / total_runs if total_runs > 0 else 0
        print(f"    Match rate: {match_rate*100:.1f}% "
              f"({successful_runs}/{total_runs})")
        if sample_generations:
            print(f"    Sample: \"{sample_generations[0][:80]}...\"")

        # Compute contrastive scores per neuron per pool
        neuron_scores = {}
        for pool in ALL_POOL_NAMES:
            pool_scores = {}
            all_neurons = set(target_neuron_counts[pool].keys()) | \
                          set(baseline_neuron_counts[pool].keys())
            for n in all_neurons:
                t_freq = target_neuron_counts[pool].get(n, 0) / max(successful_runs, 1)
                b_freq = baseline_neuron_counts[pool].get(n, 0) / max(total_baseline_steps, 1)
                pool_scores[n] = {
                    'target_freq': t_freq,
                    'baseline_freq': b_freq,
                    'contrastive': t_freq - b_freq,
                }
            neuron_scores[pool] = pool_scores

        return {
            'prompt': prompt,
            'target_token': target_token,
            'target_token_id': target_token_id,
            'successful_runs': successful_runs,
            'total_runs': total_runs,
            'match_rate': match_rate,
            'total_baseline_steps': total_baseline_steps,
            'sample_generations': sample_generations,
            'neuron_scores': neuron_scores,
            'target_neuron_counts': {p: dict(c) for p, c in target_neuron_counts.items()},
            'baseline_neuron_counts': {p: dict(c) for p, c in baseline_neuron_counts.items()},
        }

    # ----------------------------------------------------------
    # Helpers for sampling-based generation
    # ----------------------------------------------------------

    def _init_decode_step(self):
        """JIT compile the cached forward with routing (once)."""
        if hasattr(self, '_jit_decode_step'):
            return
        params = self.params
        config = self.config

        @jax.jit
        def _step(params, token_ids, kv_k, kv_v, cache_pos):
            return dawn_cached_forward_with_routing(
                params, config, token_ids, kv_k, kv_v, cache_pos)

        # Warmup
        print("  JIT compiling decode step (with routing)...", end=" ", flush=True)
        t0 = time.time()
        dummy_kv_k, dummy_kv_v = dawn_init_kv_cache(config, batch_size=1)
        _out = _step(params, jnp.array([[0]]), dummy_kv_k, dummy_kv_v, 0)
        _out[0].block_until_ready()
        print(f"done ({time.time() - t0:.1f}s)")

        self._jit_decode_step = _step

    def _decode_step_with_routing(self, params, token_ids, kv_k, kv_v, cache_pos):
        self._init_decode_step()
        return self._jit_decode_step(params, token_ids, kv_k, kv_v, cache_pos)

    @staticmethod
    def _sample_token(logits_np, temperature, top_k, rng_key):
        if temperature <= 0:
            return int(np.argmax(logits_np))
        logits_np = logits_np / temperature
        if top_k > 0:
            top_idx = np.argpartition(logits_np, -top_k)[-top_k:]
            mask = np.full_like(logits_np, -np.inf)
            mask[top_idx] = logits_np[top_idx]
            logits_np = mask
        probs = np.exp(logits_np - np.max(logits_np))
        probs = probs / (probs.sum() + 1e-8)
        return int(jax.random.choice(rng_key, len(probs), p=probs))

    @staticmethod
    def _extract_active_from_routing(routing_info, pool_key_map):
        """Extract active neuron indices from routing_info (union across layers)."""
        result = {}
        for pool_name, key in pool_key_map.items():
            if key not in routing_info:
                result[pool_name] = set()
                continue
            w = np.array(routing_info[key])  # [n_layers, B, S, N] or [n_layers, B, N]
            active = set()
            for li in range(w.shape[0]):
                w_layer = w[li]
                if w_layer.ndim == 3:
                    w_last = w_layer[0, -1]
                elif w_layer.ndim == 2:
                    w_last = w_layer[0]
                else:
                    continue
                active.update(int(i) for i in np.where(w_last > 0)[0])
            result[pool_name] = active
        return result

    # ----------------------------------------------------------
    # Phase 1→2: Identify suppression targets
    # ----------------------------------------------------------

    def identify_suppression_targets(self, freq_results, top_n_pct=0.10,
                                     mode='intersection'):
        """
        Select neurons to suppress based on contrastive scores.

        Uses paper method: neurons that are activated significantly MORE
        when the target token is generated vs baseline steps.

        Selects top N% neurons by contrastive score per pool.
        mode='intersection': must be in top N% for ALL capital queries.
        mode='union': in top N% for ANY capital query.
        """
        targets = {}

        for pool in ALL_POOL_NAMES:
            per_query_sets = []

            for result in freq_results:
                pool_scores = result['neuron_scores'].get(pool, {})
                if not pool_scores:
                    per_query_sets.append(set())
                    continue

                # Rank by contrastive score (target_freq - baseline_freq)
                sorted_neurons = sorted(
                    pool_scores.items(),
                    key=lambda x: x[1]['contrastive'], reverse=True)

                # Only take neurons with positive contrastive score
                positive = [(n, s) for n, s in sorted_neurons
                            if s['contrastive'] > 0]

                if not positive:
                    per_query_sets.append(set())
                    continue

                n_select = max(1, int(len(sorted_neurons) * top_n_pct))
                meeting = {int(n) for n, _ in positive[:n_select]}
                per_query_sets.append(meeting)

            if not per_query_sets:
                continue

            if mode == 'intersection':
                combined = per_query_sets[0]
                for s in per_query_sets[1:]:
                    combined = combined & s
            else:
                combined = set()
                for s in per_query_sets:
                    combined |= s

            if combined:
                targets[pool] = combined

        return targets

    # ----------------------------------------------------------
    # Top-k probability extraction (single forward, no sampling)
    # ----------------------------------------------------------

    def get_next_token_probs(self, prompt, forward_fn=None, top_k=10):
        """
        Single forward pass → softmax → top-k token probabilities.
        """
        if forward_fn is None:
            forward_fn = self._baseline_forward

        input_ids = [101] + self.tokenizer.encode(prompt, add_special_tokens=False)
        input_arr = jnp.array([input_ids])

        logits = forward_fn(input_arr)
        last_logits = np.array(logits[0, -1, :]).astype(np.float64)

        # Stable softmax
        last_logits -= last_logits.max()
        probs = np.exp(last_logits)
        probs /= probs.sum()

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_tokens = []
        for idx in top_indices:
            tok = self.tokenizer.decode([idx]).strip()
            top_tokens.append((tok, int(idx), float(probs[idx])))

        return {
            'prompt': prompt,
            'input_len': len(input_ids),
            'top_tokens': top_tokens,
        }

    # ----------------------------------------------------------
    # Full experiment orchestration
    # ----------------------------------------------------------

    def run_full_experiment(
        self,
        capital_queries=None, control_queries=None,
        min_target_count=100, max_runs=500,
        max_tokens_per_run=200,
        temperature=1.0, top_k_sampling=50,
        top_n_pct=0.10, mode='intersection',
        target_label='target', control_label='control',
        control_min_target_count=None,
    ):
        if capital_queries is None:
            capital_queries = DEFAULT_CAPITAL_QUERIES
        if control_queries is None:
            control_queries = DEFAULT_CONTROL_QUERIES
        if control_min_target_count is None:
            control_min_target_count = min_target_count

        results = {
            'config': {
                'min_target_count': min_target_count,
                'max_runs': max_runs,
                'temperature': temperature,
                'top_k_sampling': top_k_sampling,
                'top_n_pct': top_n_pct, 'mode': mode,
                'capital_queries': capital_queries,
                'control_queries': control_queries,
                'target_label': target_label,
                'control_label': control_label,
            },
            'phase1': {}, 'phase2': {}, 'phase3': {},
        }

        # === Phase 1: Sampling-based activation frequency (paper method) ===
        print("=" * 70)
        print("PHASE 1: Sampling-based routing extraction (paper method)")
        print(f"  min_target_count={min_target_count}, max_runs={max_runs}, "
              f"temp={temperature}, top_k={top_k_sampling}")
        print("=" * 70)

        # --- Baseline top-10 (before suppression) ---
        print("\n  --- Baseline next-token probabilities (pre-suppression) ---")
        all_queries = capital_queries + control_queries
        baseline_probs = {}
        for qi, q in enumerate(all_queries, 1):
            tag = target_label if qi <= len(capital_queries) else control_label
            print(f"\n  [{qi}/{len(all_queries)}] [{tag}] \"{q['prompt']}\" -> target: '{q['target']}'")
            bp = self.get_next_token_probs(q['prompt'])
            baseline_probs[q['prompt']] = bp
            target_lower = q['target'].strip().lower()
            for tok, tid, prob in bp['top_tokens']:
                marker = ' <-- TARGET' if tok.lower() == target_lower else ''
                print(f"    {prob:>6.2%}  '{tok}' (id={tid}){marker}")
        results['phase1']['baseline_top10'] = baseline_probs

        # --- Sampling-based activation collection ---
        freq_results = []
        for qi, q in enumerate(capital_queries, 1):
            print(f"\n  [{qi}/{len(capital_queries)}] Query: \"{q['prompt']}\" -> target: '{q['target']}'")
            t0 = time.time()
            freq = self.collect_activation_frequencies(
                q['prompt'], q['target'],
                min_target_count=min_target_count,
                max_runs=max_runs,
                max_tokens_per_run=max_tokens_per_run,
                temperature=temperature,
                top_k=top_k_sampling)
            elapsed = time.time() - t0
            for pool in ALL_POOL_NAMES:
                scores = freq['neuron_scores'].get(pool, {})
                if not scores:
                    continue
                top3 = sorted(scores.items(),
                              key=lambda x: x[1]['contrastive'], reverse=True)[:3]
                top3_str = ", ".join(
                    f"n{n}({s['contrastive']:+.2f})" for n, s in top3)
                print(f"    {pool}: top contrastive = {top3_str}")
            print(f"    [{elapsed:.1f}s]")
            freq_results.append(freq)

        results['phase1']['capital_frequencies'] = freq_results

        print(f"\n  --- Control queries (min_target_count={control_min_target_count}) ---")
        control_freqs = []
        for qi, q in enumerate(control_queries, 1):
            print(f"\n  [{qi}/{len(control_queries)}] Query: \"{q['prompt']}\" -> target: '{q['target']}'")
            t0 = time.time()
            freq = self.collect_activation_frequencies(
                q['prompt'], q['target'],
                min_target_count=control_min_target_count,
                max_runs=max_runs,
                max_tokens_per_run=max_tokens_per_run,
                temperature=temperature,
                top_k=top_k_sampling)
            elapsed = time.time() - t0
            print(f"    [{elapsed:.1f}s]")
            control_freqs.append(freq)
        results['phase1']['control_frequencies'] = control_freqs

        # === Phase 2: Identify suppression targets (contrastive) ===
        print("\n" + "=" * 70)
        print("PHASE 2: Identifying suppression targets (contrastive)")
        print("=" * 70)

        suppressed = self.identify_suppression_targets(
            freq_results, top_n_pct=top_n_pct, mode=mode)
        total_suppressed = sum(len(v) for v in suppressed.values())

        print(f"\n  Mode: {mode} | Top {top_n_pct:.0%} by contrastive score")
        print(f"  Total neurons to suppress: {total_suppressed}")
        for pool, indices in sorted(suppressed.items()):
            idx_preview = sorted(indices)[:10]
            print(f"    {pool}: {len(indices)} neurons — "
                  f"{idx_preview}{'...' if len(indices) > 10 else ''}")

        results['phase2']['suppressed_neurons'] = {
            k: sorted(v) for k, v in suppressed.items()}
        results['phase2']['total_suppressed'] = total_suppressed

        if total_suppressed == 0:
            print("\n  WARNING: No neurons with positive contrastive score! "
                  "Try --top_n_pct higher or --mode union")
            results['phase3']['note'] = 'no neurons to suppress'
            self._print_summary(results)
            return results

        # Build suppressed forward
        masks = build_masks_from_sets(suppressed, self.config)
        suppressed_forward = build_suppressed_forward(
            self.model, self.params, self.config, masks)
        print("  Suppressed forward compiled (JIT)")

        # === Phase 3: Post-suppression top-10 comparison ===
        print("\n" + "=" * 70)
        print("PHASE 3: Post-suppression next-token probabilities")
        print("=" * 70)

        suppressed_probs = {}
        for qi, q in enumerate(all_queries, 1):
            tag = target_label if qi <= len(capital_queries) else control_label
            print(f"\n  [{qi}/{len(all_queries)}] [{tag}] \"{q['prompt']}\" -> target: '{q['target']}'")
            sp = self.get_next_token_probs(q['prompt'], forward_fn=suppressed_forward)
            suppressed_probs[q['prompt']] = sp
            target_lower = q['target'].strip().lower()
            for tok, tid, prob in sp['top_tokens']:
                # Find baseline prob for comparison
                bp_prob = 0.0
                bp = baseline_probs.get(q['prompt'], {})
                for bt, _, bprob in bp.get('top_tokens', []):
                    if bt == tok:
                        bp_prob = bprob
                        break
                delta = prob - bp_prob
                marker = ' <-- TARGET' if tok.lower() == target_lower else ''
                print(f"    {prob:>6.2%}  '{tok}' (was {bp_prob:>5.2%}, delta={delta:>+6.2%}){marker}")
        results['phase3']['suppressed_top10'] = suppressed_probs

        # Compute selectivity metrics
        results['selectivity'] = self._compute_selectivity(
            capital_queries, control_queries, baseline_probs, suppressed_probs)

        self._print_summary(results)
        return results

    def _compute_selectivity(self, capital_queries, control_queries,
                             baseline_probs, suppressed_probs):
        """
        Compute selectivity metrics from probability shifts.

        - target_drop: avg probability drop for target token in domain queries
        - control_drop: avg probability drop for target token in control queries
        - selectivity_index: target_drop - control_drop
          (positive = selective suppression; >0.1 = strong domain specificity)
        """
        def _get_target_prob(probs_dict, prompt, target):
            target_lower = target.strip().lower()
            entry = probs_dict.get(prompt, {})
            for tok, _, prob in entry.get('top_tokens', []):
                if tok.lower() == target_lower:
                    return prob
            return 0.0

        target_drops = []
        for q in capital_queries:
            pre = _get_target_prob(baseline_probs, q['prompt'], q['target'])
            post = _get_target_prob(suppressed_probs, q['prompt'], q['target'])
            target_drops.append(pre - post)

        control_drops = []
        for q in control_queries:
            pre = _get_target_prob(baseline_probs, q['prompt'], q['target'])
            post = _get_target_prob(suppressed_probs, q['prompt'], q['target'])
            control_drops.append(pre - post)

        avg_target = float(np.mean(target_drops)) if target_drops else 0.0
        avg_control = float(np.mean(control_drops)) if control_drops else 0.0
        selectivity = avg_target - avg_control

        return {
            'target_drops': [float(d) for d in target_drops],
            'control_drops': [float(d) for d in control_drops],
            'avg_target_drop': avg_target,
            'avg_control_drop': avg_control,
            'selectivity_index': selectivity,
            'interpretation': (
                'SELECTIVE: target domain dropped significantly more than control'
                if selectivity > 0.1
                else 'WEAK: suppression affected both domains similarly'
                if selectivity > 0.0
                else 'NON-SELECTIVE: control dropped more than target'
            ),
        }

    def _print_summary(self, results):
        print("\n" + "=" * 70)
        print("SUMMARY: Pre vs Post Suppression (next-token top-10)")
        print("=" * 70)

        config = results['config']
        phase2 = results['phase2']
        print(f"  Top {config['top_n_pct']:.0%} by contrastive score | "
              f"Mode: {config['mode']}")
        print(f"  Suppressed: {phase2['total_suppressed']} neurons")
        for pool, indices in sorted(phase2['suppressed_neurons'].items()):
            print(f"    {pool}: {len(indices)}")

        baseline = results['phase1'].get('baseline_top10', {})
        suppressed = results['phase3'].get('suppressed_top10', {})
        all_queries = config['capital_queries'] + config['control_queries']
        n_capital = len(config['capital_queries'])

        print("\n" + "-" * 95)
        print(f"  {'Query':<35s} {'Target':<8s} {'Pre(target)':>11s} {'Post(target)':>12s} {'Delta':>8s}")
        print("-" * 95)

        for qi, q in enumerate(all_queries):
            tag = '' if qi < n_capital else f'  ({config.get("control_label", "ctrl")})'
            prompt = q['prompt']
            target_lower = q['target'].strip().lower()

            # Find target prob in baseline
            pre_prob = 0.0
            bp = baseline.get(prompt, {})
            for tok, _, prob in bp.get('top_tokens', []):
                if tok.lower() == target_lower:
                    pre_prob = prob
                    break

            # Find target prob in suppressed
            post_prob = 0.0
            sp = suppressed.get(prompt, {})
            for tok, _, prob in sp.get('top_tokens', []):
                if tok.lower() == target_lower:
                    post_prob = prob
                    break

            delta = post_prob - pre_prob
            print(f"  {prompt[:33]:<35s} {q['target']:<8s} "
                  f"{pre_prob:>10.2%}  {post_prob:>11.2%}  {delta:>+7.2%}{tag}")

        print("-" * 95)

        # Selectivity metrics
        sel = results.get('selectivity')
        if sel:
            print(f"\n  SELECTIVITY METRICS:")
            print(f"    Avg target prob drop:  {sel['avg_target_drop']:>+.2%}")
            print(f"    Avg control prob drop: {sel['avg_control_drop']:>+.2%}")
            print(f"    Selectivity index:     {sel['selectivity_index']:>+.2%}")
            print(f"    Verdict: {sel['interpretation']}")


if __name__ == '__main__':
    main()
