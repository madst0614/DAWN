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
        x = token_emb_table[input_ids] + pos_emb_table[positions]

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
    parser.add_argument('--n_runs', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--mode', type=str, default='intersection',
                        choices=['intersection', 'union'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--queries', type=str, default=None,
                        help='Custom queries JSON file')
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Loading model from: {args.checkpoint}")

    model_cls, params, tokenizer, config = load_model_jax(args.checkpoint)
    model = create_model_from_config(config)

    print(f"  Model version: {config.get('model_version', 'unknown')}")
    print(f"  Pools: FQK={config.get('n_feature_qk')}, FV={config.get('n_feature_v')}, "
          f"RQK={config.get('n_restore_qk')}, RV={config.get('n_restore_v')}, "
          f"FK={config.get('n_feature_know')}, RK={config.get('n_restore_know')}")

    # Load custom queries
    capital_queries = DEFAULT_CAPITAL_QUERIES
    control_queries = DEFAULT_CONTROL_QUERIES
    if args.queries:
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
        n_runs=args.n_runs,
        threshold=args.threshold,
        mode=args.mode,
    )

    # Save
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = Path(args.checkpoint).name or 'checkpoint'
        filename = f"suppression_jax_{ckpt_name}_t{args.threshold}_n{args.n_runs}_{args.mode}.json"
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

        # Build baseline forward (no suppression — empty masks)
        self._baseline_forward = build_suppressed_forward(model, params, config, {})

    # ----------------------------------------------------------
    # Phase 1: Collect activation frequencies
    # ----------------------------------------------------------

    def collect_activation_frequencies(self, prompt, target_token, n_runs=100):
        """
        Run baseline forward n_runs times, record which neurons are
        in top-k at the last token position.

        Returns dict with match_rate and per-pool neuron frequencies.
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_arr = jnp.array([input_ids])

        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)
        if len(target_id) == 0:
            raise ValueError(f"Target token '{target_token}' not in vocabulary")
        target_id = target_id[0]

        # We need routing weights for frequency analysis.
        # Use JAXRoutingDataExtractor-style extraction: compute routing
        # from embedding layer output (first layer input).
        # This captures the router's neuron selection.
        extractor = self._build_routing_extractor()

        freq = {pool: defaultdict(int) for pool in ALL_POOL_NAMES}
        match_count = 0

        for _ in range(n_runs):
            logits = self._baseline_forward(input_arr)
            next_id = int(jnp.argmax(logits[0, -1, :]))
            if next_id == target_id:
                match_count += 1

            # Extract routing weights
            routing = extractor(input_arr)
            attn_weights, know_weights = routing

            # Attention: 6 weight tensors [B, S, N_pool]
            attn_names = ['fqk_Q', 'fqk_K', 'fv', 'rqk_Q', 'rqk_K', 'rv']
            for w, name in zip(attn_weights, attn_names):
                w_last = np.asarray(w[0, -1])  # [N_pool] at last position
                active = np.where(w_last > 0)[0]
                for idx in active:
                    freq[name][int(idx)] += 1

            # Knowledge: 2 weight tensors
            know_names = ['feature_know', 'restore_know']
            for w, name in zip(know_weights, know_names):
                w_last = np.asarray(w[0, -1])
                active = np.where(w_last > 0)[0]
                for idx in active:
                    freq[name][int(idx)] += 1

        return {
            'prompt': prompt,
            'target_token': target_token,
            'target_token_id': target_id,
            'match_count': match_count,
            'total_runs': n_runs,
            'match_rate': match_count / n_runs,
            'neuron_frequencies': {pool: dict(counts) for pool, counts in freq.items()},
        }

    def _build_routing_extractor(self):
        """Build JIT-compiled function that returns routing weights only."""
        all_params = self.params['params']
        router_params = all_params['router']
        nr = router_params['neuron_router']

        token_emb_table = all_params['token_emb']['embedding']
        pos_emb_table = all_params['pos_emb']['embedding']

        neuron_emb = nr['neuron_emb']
        attn_kernel = nr['proj_all']['kernel']
        attn_bias = nr['proj_all']['bias']
        fk_kernel = nr['proj_feature_know']['kernel']
        fk_bias = nr['proj_feature_know']['bias']
        rk_kernel = nr['proj_restore_know']['kernel']
        rk_bias = nr['proj_restore_know']['bias']

        n_fqk = self.config.get('n_feature_qk', 88)
        n_fv = self.config.get('n_feature_v', 352)
        n_rqk = self.config.get('n_restore_qk', 88)
        n_rv = self.config.get('n_restore_v', 352)
        n_fk = self.config.get('n_feature_know', 224)

        fqk_end = n_fqk
        fv_end = fqk_end + n_fv
        rqk_end = fv_end + n_rqk
        rv_end = rqk_end + n_rv
        fk_end = rv_end + n_fk

        tk_fqk = self.config.get('top_k_feature_qk', 16)
        tk_fv = self.config.get('top_k_feature_v', 16)
        tk_rqk = self.config.get('top_k_restore_qk', 16)
        tk_rv = self.config.get('top_k_restore_v', 16)
        tk_fk = self.config.get('top_k_feature_know', 16)
        tk_rk = self.config.get('top_k_restore_know', 16)

        @jax.jit
        def extract(input_ids):
            B, S = input_ids.shape
            positions = jnp.arange(S)[jnp.newaxis, :]
            x = token_emb_table[input_ids] + pos_emb_table[positions]

            emb_norm = neuron_emb / (jnp.linalg.norm(neuron_emb, axis=-1, keepdims=True) + 1e-8)

            # Attention routing
            all_proj = x @ attn_kernel + attn_bias
            splits = jnp.split(all_proj, 6, axis=-1)
            embs = [emb_norm[:fqk_end], emb_norm[:fqk_end],
                    emb_norm[fqk_end:fv_end], emb_norm[fv_end:rqk_end],
                    emb_norm[fv_end:rqk_end], emb_norm[rqk_end:rv_end]]
            tks = [tk_fqk, tk_fqk, tk_fv, tk_rqk, tk_rqk, tk_rv]

            attn_results = []
            for h, emb, tk in zip(splits, embs, tks):
                logits = jnp.einsum('bsd,nd->bsn', h, emb)
                pref = jax.nn.softmax(logits, axis=-1)
                w, _ = topk_sparsify(pref, tk)
                attn_results.append(w)

            # Knowledge routing
            h_fk = x @ fk_kernel + fk_bias
            logits_fk = jnp.einsum('bsd,nd->bsn', h_fk, emb_norm[rv_end:fk_end])
            pref_fk = jax.nn.softmax(logits_fk, axis=-1)
            w_fk, _ = topk_sparsify(pref_fk, tk_fk)

            h_rk = x @ rk_kernel + rk_bias
            logits_rk = jnp.einsum('bsd,nd->bsn', h_rk, emb_norm[fk_end:])
            pref_rk = jax.nn.softmax(logits_rk, axis=-1)
            w_rk, _ = topk_sparsify(pref_rk, tk_rk)

            return attn_results, (w_fk, w_rk)

        return extract

    # ----------------------------------------------------------
    # Phase 1→2: Identify suppression targets
    # ----------------------------------------------------------

    def identify_suppression_targets(self, freq_results, threshold=0.7, mode='intersection'):
        """
        Find neurons active ≥threshold across capital queries.
        mode='intersection': must meet threshold in ALL queries.
        mode='union': meets threshold in ANY query.
        """
        targets = {}
        for pool in ALL_POOL_NAMES:
            per_query_sets = []
            for result in freq_results:
                n_runs = result['total_runs']
                min_count = int(n_runs * threshold)
                pool_freq = result['neuron_frequencies'].get(pool, {})
                meeting = {int(idx) for idx, count in pool_freq.items() if count >= min_count}
                per_query_sets.append(meeting)

            if not per_query_sets:
                continue

            if mode == 'intersection':
                combined = per_query_sets[0]
                for s in per_query_sets[1:]:
                    combined = combined & s
            else:  # union
                combined = set()
                for s in per_query_sets:
                    combined |= s

            if combined:
                targets[pool] = combined

        return targets

    # ----------------------------------------------------------
    # Phase 3: Measure target frequency
    # ----------------------------------------------------------

    def measure_target_frequency(self, prompt, target_token, n_runs=100, forward_fn=None):
        """
        Run forward n_runs times, count target token hits.
        Uses forward_fn (suppressed or baseline).
        """
        if forward_fn is None:
            forward_fn = self._baseline_forward

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_arr = jnp.array([input_ids])

        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)
        if len(target_id) == 0:
            raise ValueError(f"Target token '{target_token}' not in vocabulary")
        target_id = target_id[0]

        match_count = 0
        token_counts = defaultdict(int)

        for _ in range(n_runs):
            logits = forward_fn(input_arr)
            next_id = int(jnp.argmax(logits[0, -1, :]))
            token_counts[next_id] += 1
            if next_id == target_id:
                match_count += 1

        top5 = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_decoded = [
            (self.tokenizer.decode([tid]).strip(), count, count / n_runs)
            for tid, count in top5
        ]

        return {
            'prompt': prompt,
            'target_token': target_token,
            'target_token_id': target_id,
            'match_count': match_count,
            'total_runs': n_runs,
            'match_rate': match_count / n_runs,
            'top5': top5_decoded,
        }

    # ----------------------------------------------------------
    # Full experiment orchestration
    # ----------------------------------------------------------

    def run_full_experiment(
        self,
        capital_queries=None, control_queries=None,
        n_runs=100, threshold=0.7, mode='intersection',
    ):
        if capital_queries is None:
            capital_queries = DEFAULT_CAPITAL_QUERIES
        if control_queries is None:
            control_queries = DEFAULT_CONTROL_QUERIES

        results = {
            'config': {
                'n_runs': n_runs, 'threshold': threshold, 'mode': mode,
                'capital_queries': capital_queries,
                'control_queries': control_queries,
            },
            'phase1': {}, 'phase2': {}, 'phase3': {},
        }

        # === Phase 1: Activation frequency ===
        print("=" * 70)
        print("PHASE 1: Collecting activation frequencies")
        print("=" * 70)

        freq_results = []
        for q in capital_queries:
            print(f"\n  Query: \"{q['prompt']}\" -> target: '{q['target']}'")
            t0 = time.time()
            freq = self.collect_activation_frequencies(
                q['prompt'], q['target'], n_runs=n_runs)
            elapsed = time.time() - t0
            print(f"    Match rate: {freq['match_count']}/{n_runs} "
                  f"({freq['match_rate']:.0%})  [{elapsed:.1f}s]")
            for pool in ALL_POOL_NAMES:
                pf = freq['neuron_frequencies'].get(pool, {})
                high = {k: v for k, v in pf.items() if v >= n_runs * threshold}
                if high:
                    print(f"    {pool}: {len(high)} neurons >= {threshold:.0%}")
            freq_results.append(freq)

        results['phase1']['capital_frequencies'] = freq_results

        # Control baselines
        print(f"\n  --- Control queries (baseline) ---")
        control_baselines = []
        for q in control_queries:
            print(f"  Query: \"{q['prompt']}\" -> target: '{q['target']}'")
            baseline = self.measure_target_frequency(
                q['prompt'], q['target'], n_runs=n_runs)
            print(f"    Match rate: {baseline['match_count']}/{n_runs} "
                  f"({baseline['match_rate']:.0%})")
            control_baselines.append(baseline)
        results['phase1']['control_baselines'] = control_baselines

        # === Phase 2: Identify & build suppression ===
        print("\n" + "=" * 70)
        print("PHASE 2: Identifying suppression targets")
        print("=" * 70)

        suppressed = self.identify_suppression_targets(
            freq_results, threshold, mode)
        total_suppressed = sum(len(v) for v in suppressed.values())

        print(f"\n  Mode: {mode} | Threshold: {threshold:.0%}")
        print(f"  Total neurons to suppress: {total_suppressed}")
        for pool, indices in sorted(suppressed.items()):
            idx_preview = sorted(indices)[:10]
            print(f"    {pool}: {len(indices)} neurons — {idx_preview}{'...' if len(indices) > 10 else ''}")

        results['phase2']['suppressed_neurons'] = {
            k: sorted(v) for k, v in suppressed.items()}
        results['phase2']['total_suppressed'] = total_suppressed

        if total_suppressed == 0:
            print("\n  WARNING: No neurons met threshold! "
                  "Try --threshold lower or --mode union")
            results['phase3']['note'] = 'no neurons to suppress'
            self._print_summary(results)
            return results

        # Build suppressed forward
        masks = build_masks_from_sets(suppressed, self.config)
        suppressed_forward = build_suppressed_forward(
            self.model, self.params, self.config, masks)
        print("  Suppressed forward compiled (JIT)")

        # === Phase 3: Post-suppression measurement ===
        print("\n" + "=" * 70)
        print("PHASE 3: Measuring post-suppression effect")
        print("=" * 70)

        print("\n  --- Capital queries (suppressed) ---")
        capital_post = []
        for q in capital_queries:
            print(f"  Query: \"{q['prompt']}\" -> target: '{q['target']}'")
            post = self.measure_target_frequency(
                q['prompt'], q['target'], n_runs=n_runs,
                forward_fn=suppressed_forward)
            print(f"    Match rate: {post['match_count']}/{n_runs} "
                  f"({post['match_rate']:.0%})")
            if post['top5']:
                top_tok, _, top_pct = post['top5'][0]
                print(f"    Most frequent: '{top_tok}' ({top_pct:.0%})")
            capital_post.append(post)
        results['phase3']['capital_post_suppression'] = capital_post

        print(f"\n  --- Control queries (suppressed) ---")
        control_post = []
        for q in control_queries:
            print(f"  Query: \"{q['prompt']}\" -> target: '{q['target']}'")
            post = self.measure_target_frequency(
                q['prompt'], q['target'], n_runs=n_runs,
                forward_fn=suppressed_forward)
            print(f"    Match rate: {post['match_count']}/{n_runs} "
                  f"({post['match_rate']:.0%})")
            control_post.append(post)
        results['phase3']['control_post_suppression'] = control_post

        self._print_summary(results)
        return results

    def _print_summary(self, results):
        print("\n" + "=" * 70)
        print("SUMMARY: Pseudo-Neuron Suppression Results")
        print("=" * 70)

        config = results['config']
        phase2 = results['phase2']
        print(f"  Threshold: {config['threshold']:.0%} | "
              f"Mode: {config['mode']} | Runs: {config['n_runs']}")
        print(f"  Suppressed: {phase2['total_suppressed']} neurons")
        for pool, indices in sorted(phase2['suppressed_neurons'].items()):
            print(f"    {pool}: {len(indices)}")

        print("\n" + "-" * 90)
        print(f"  {'Query':<40s} {'Target':<8s} {'Pre':>7s} {'Post':>7s} {'Delta':>7s}")
        print("-" * 90)

        cap_freqs = results['phase1']['capital_frequencies']
        cap_posts = results['phase3'].get('capital_post_suppression', [])
        for freq, post in zip(cap_freqs, cap_posts):
            pre = freq['match_rate']
            pst = post['match_rate']
            delta = pst - pre
            print(f"  {freq['prompt'][:38]:<40s} {freq['target_token']:<8s} "
                  f"{pre:>6.0%}  {pst:>6.0%}  {delta:>+6.0%}")

        print("-" * 90)

        ctrl_bases = results['phase1'].get('control_baselines', [])
        ctrl_posts = results['phase3'].get('control_post_suppression', [])
        for base, post in zip(ctrl_bases, ctrl_posts):
            pre = base['match_rate']
            pst = post['match_rate']
            delta = pst - pre
            print(f"  {base['prompt'][:38]:<40s} {base['target_token']:<8s} "
                  f"{pre:>6.0%}  {pst:>6.0%}  {delta:>+6.0%}  (control)")

        print("-" * 90)

        if cap_posts:
            print("\n  Post-suppression top-5 tokens (capital queries):")
            for post in cap_posts:
                tokens_str = ", ".join(
                    f"'{tok}' ({pct:.0%})" for tok, _, pct in post['top5'])
                print(f"    \"{post['prompt']}\" -> {tokens_str}")


if __name__ == '__main__':
    main()
