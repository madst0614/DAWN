#!/usr/bin/env python3
"""
Knowledge Neuron Coherence Analysis (JAX/TPU) — Appendix D.3
=============================================================
JAX/TPU port of BehavioralAnalyzer.analyze_factual_neurons().

Autoregressive generation approach matching GPU paper method:
  1. For each (prompt, target) pair, generate until target token appears
  2. On target hit, extract routing weights at the last position via full forward
  3. On non-target steps, also extract routing for baseline frequency
  4. Repeat until target appears min_target_count times (default 100, max 500 runs)
  5. Compute per-neuron: target_freq, baseline_freq, contrastive_score

Uses unified naming convention: '{pool}_{neuron_id}' (e.g. 'fknow_12').
Pools analyzed: fv, rv, fknow, rknow (all V/Knowledge pools by default).

Designed for single-host TPU v4-8.

Usage:
    python scripts/analysis/visualizers/knowledge_coherence_jax.py \\
        --checkpoint gs://dawn-tpu-data-c4/checkpoints/... \\
        --output ./section4_results \\
        --min_targets 100 --max_runs 500
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs): return x

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

from scripts.analysis.utils_jax import (
    load_model_jax, create_model_from_config,
    convert_to_serializable, save_results,
)

# Inline style
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
# Default Prompts (Appendix D.3)
# ============================================================

CAPITAL_PROMPTS = [
    ("France",  "The capital of France is",  "Paris"),
    ("UK",      "The capital of the United Kingdom is", "London"),
    ("Japan",   "The capital of Japan is",   "Tokyo"),
    ("Germany", "The capital of Germany is",  "Berlin"),
    ("Italy",   "The capital of Italy is",    "Rome"),
    ("Spain",   "The capital of Spain is",    "Madrid"),
    ("China",   "The capital of China is",    "Beijing"),
    ("Brazil",  "The capital of Brazil is",   "Bras"),  # Brasilia
    ("India",   "The capital of India is",    "New"),    # New Delhi
    ("Canada",  "The capital of Canada is",   "Ottawa"),
]

CONTROL_PROMPTS = [
    ("sky",     "The sky is",    "blue"),
    ("water",   "The water is",  "clear"),
    ("cat",     "The cat is",    "a"),
    ("book",    "The book is",   "a"),
    ("food",    "The food is",   "good"),
    ("music",   "The music is",  "a"),
    ("tree",    "The tree is",   "a"),
    ("house",   "The house is",  "a"),
    ("car",     "The car is",    "a"),
    ("sun",     "The sun is",    "a"),
]


# ============================================================
# Autoregressive Generation with Routing Extraction
# ============================================================

def _extract_routing_at_last_pos(params, config, input_ids_np, pools):
    """Full forward through all layers, extract routing weights at last position.

    Returns dict: {pool_key: set_of_active_neuron_indices}
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

    # Pool index mapping for attention routing results
    _ATTN_IDX = {'fqk_q': 0, 'fqk_k': 1, 'fv': 2, 'rqk_q': 3, 'rqk_k': 4, 'rv': 5}
    _KNOW_IDX = {'fknow': 0, 'rknow': 1}

    input_ids = jnp.array(input_ids_np)
    B, Seq = input_ids.shape

    tok_emb = all_params['token_emb']['embedding'][input_ids]
    pos_emb = all_params['pos_emb']['embedding'][jnp.arange(Seq)[jnp.newaxis, :]]
    x = tok_emb + pos_emb

    rng_key = jax.random.PRNGKey(0)

    # Accumulate active neurons across all layers (union, matching GPU)
    result = {p: set() for p in pools}

    for li in range(n_layers):
        bp = all_params[f'block_{li}']
        rng_key, rng_ar, rng_kr, rng_a, rng_k = jax.random.split(rng_key, 5)

        normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
        attn_results = _router_attn_forward(
            normed, router_params,
            n_fqk, n_fv, n_rqk, n_rv, d_space,
            tk_fqk, tk_fv, tk_rqk, tk_rv,
            0.0, None, True, rng_ar,
        )
        fqk_wQ, fqk_wK, fv_w, rqk_wQ, rqk_wK, rv_w = attn_results[:6]

        # Extract attention pool activations at last position
        for pool in pools:
            if pool in _ATTN_IDX:
                w = np.array(attn_results[_ATTN_IDX[pool]])
                if w.ndim == 3:
                    w_last = w[0, -1]  # [N]
                else:
                    w_last = w[0]
                active = set(int(i) for i in np.where(w_last > 0)[0])
                result[pool].update(active)

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

        # Extract knowledge pool activations at last position
        for pool in pools:
            if pool in _KNOW_IDX:
                w = np.array(fk_w if _KNOW_IDX[pool] == 0 else rk_w)
                if w.ndim == 3:
                    w_last = w[0, -1]
                else:
                    w_last = w[0]
                active = set(int(i) for i in np.where(w_last > 0)[0])
                result[pool].update(active)

        know_out = _knowledge_forward(
            normed, sn_params, fk_w, rk_w,
            0.0, True, rng_k,
        )
        x = x + know_out

    return result


def analyze_knowledge_coherence(
    model_cls, params, config,
    capital_prompts=None, control_prompts=None,
    pools=None,
    min_target_count=100,
    max_runs=500,
    max_tokens_per_run=200,
    temperature=1.0,
    top_k=50,
):
    """Autoregressive knowledge coherence analysis (Appendix D.3).

    JAX/TPU port of BehavioralAnalyzer.analyze_factual_neurons().
    Analyzes ALL specified pools simultaneously in a single generation pass.

    For each (prompt, target) pair:
      1. Generate autoregressively until target token appears
      2. On target hit, full forward to extract routing at last position
      3. On non-target step, also extract routing for baseline frequency
      4. Repeat until min_target_count hits (or max_runs exhausted)
      5. Compute per-neuron: target_freq, baseline_freq, contrastive_score

    Uses unified naming convention: '{pool}_{neuron_id}' (e.g. 'fknow_12')
    matching BehavioralAnalyzer output schema.
    """
    if not HAS_JAX:
        return {'error': 'JAX not available'}

    from transformers import AutoTokenizer
    from models.model_v17_1_jax import dawn_init_kv_cache, dawn_cached_forward

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    capital_prompts = capital_prompts or CAPITAL_PROMPTS
    control_prompts = control_prompts or CONTROL_PROMPTS
    pools = pools or ['fv', 'rv', 'fknow', 'rknow']

    pool_sizes = {
        'fknow': config.get('n_feature_know', 224),
        'rknow': config.get('n_restore_know', 224),
        'fv': config.get('n_feature_v', 352),
        'rv': config.get('n_restore_v', 352),
        'fqk_q': config.get('n_feature_qk', 88),
        'fqk_k': config.get('n_feature_qk', 88),
        'rqk_q': config.get('n_restore_qk', 88),
        'rqk_k': config.get('n_restore_qk', 88),
    }

    print(f"    Analyzing {len(pools)} pools simultaneously: {pools}")

    # Token validation (matching BehavioralAnalyzer)
    all_prompt_pairs = (
        [(label, prompt, target, 'capital') for label, prompt, target in capital_prompts] +
        [(label, prompt, target, 'control') for label, prompt, target in control_prompts]
    )
    all_targets = list(set(t for _, _, t, _ in all_prompt_pairs))

    token_validation = {}
    for target_text in all_targets:
        tids = tokenizer.encode(target_text, add_special_tokens=False)
        is_single = len(tids) == 1
        token_validation[target_text] = {
            'is_single_token': is_single,
            'token_ids': tids,
            'tokens': tokenizer.convert_ids_to_tokens(tids),
        }
        if not is_single:
            print(f"    Warning: '{target_text}' is not a single token "
                  f"(tokenizes to {len(tids)} tokens: "
                  f"{tokenizer.convert_ids_to_tokens(tids)})")

    # JIT compile decode step
    @jax.jit
    def decode_step(params, token_ids, kv_k, kv_v, cache_pos):
        return dawn_cached_forward(params, config, token_ids, kv_k, kv_v, cache_pos)

    print("  JIT compiling decode step...", end=" ", flush=True)
    t0 = time.time()
    dummy_kv_k, dummy_kv_v = dawn_init_kv_cache(config, batch_size=1)
    _out = decode_step(params, jnp.array([[0]]), dummy_kv_k, dummy_kv_v, 0)
    _out[0].block_until_ready()
    print(f"done ({time.time() - t0:.1f}s)")

    results = {
        'pools_analyzed': pools,
        'min_target_count': min_target_count,
        'token_validation': token_validation,
        'per_pool': {p: {} for p in pools},
        'per_target': {},
    }

    pbar_overall = tqdm(
        total=len(all_prompt_pairs),
        desc="  Pairs",
        unit="pair",
        position=0,
        leave=True,
    ) if HAS_TQDM else None

    for pair_idx, (label, prompt, target_text, query_type) in enumerate(all_prompt_pairs):
        # Validate target token
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        if not target_ids:
            print(f"    Skipping '{target_text}': cannot tokenize")
            if pbar_overall:
                pbar_overall.update(1)
            continue
        target_token_id = target_ids[0]
        target_lower = target_text.strip().lower()

        # Encode prompt
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_ids)

        # Per-pool neuron counters — unified naming: '{pool}_{n}'
        target_neuron_counts = {p: Counter() for p in pools}
        baseline_neuron_counts = {p: Counter() for p in pools}
        successful_runs = 0
        total_runs = 0
        total_baseline_steps = 0
        sample_generations = []
        rng_key = jax.random.PRNGKey(pair_idx * 137)

        pair_desc = f"[{query_type}] \"{prompt}\" -> \"{target_text}\""
        if pbar_overall:
            pbar_overall.set_postfix_str(pair_desc, refresh=True)

        pbar_hits = tqdm(
            total=min_target_count,
            desc=f"    Hits",
            unit="hit",
            position=1,
            leave=False,
        ) if HAS_TQDM else None

        while successful_runs < min_target_count and total_runs < max_runs:
            total_runs += 1
            if not HAS_TQDM and (total_runs % 50 == 0 or successful_runs == min_target_count):
                print(f"\r      {successful_runs}/{min_target_count} hits "
                      f"(run {total_runs})", end='', flush=True)

            # Init KV cache and prefill
            kv_k, kv_v = dawn_init_kv_cache(config, batch_size=1)
            prompt_2d = jnp.array(np.array(prompt_ids)[np.newaxis, :])
            logits, kv_k, kv_v = dawn_cached_forward(
                params, config, prompt_2d, kv_k, kv_v, 0)

            generated_ids = list(prompt_ids)
            cache_pos = prompt_len

            # Sample first token from prefill logits
            first_logits = np.array(logits[0, -1, :])
            rng_key, subkey = jax.random.split(rng_key)
            next_token = _sample_token(first_logits, temperature, top_k, subkey)
            generated_ids.append(next_token)

            for step in range(max_tokens_per_run - 1):
                token_text = tokenizer.decode([next_token]).strip().lower()

                # Extract routing at current step via full forward (all pools)
                full_ids = np.array([generated_ids], dtype=np.int32)
                activations = _extract_routing_at_last_pos(
                    params, config, full_ids, pools)

                if token_text == target_lower or next_token == target_token_id:
                    # Target found — record neurons with unified naming
                    for pool in pools:
                        for n in activations[pool]:
                            if n < pool_sizes.get(pool, 0):
                                target_neuron_counts[pool][f'{pool}_{n}'] += 1

                    successful_runs += 1
                    if pbar_hits:
                        pbar_hits.update(1)
                        pbar_hits.set_postfix(runs=total_runs, refresh=True)

                    if len(sample_generations) < 3:
                        gen_text = tokenizer.decode(generated_ids,
                                                     skip_special_tokens=True)
                        sample_generations.append(gen_text)
                    break
                else:
                    # Non-target step — record as baseline with unified naming
                    for pool in pools:
                        for n in activations[pool]:
                            if n < pool_sizes.get(pool, 0):
                                baseline_neuron_counts[pool][f'{pool}_{n}'] += 1
                    total_baseline_steps += 1

                if next_token == tokenizer.sep_token_id:
                    break
                if next_token == tokenizer.eos_token_id:
                    break
                if cache_pos >= config.get('max_seq_len', 512) - 1:
                    break

                # Decode next token
                token_2d = jnp.array([[next_token]])
                logits, kv_k, kv_v = decode_step(
                    params, token_2d, kv_k, kv_v, cache_pos)
                cache_pos += 1

                next_logits = np.array(logits[0, 0, :])
                rng_key, subkey = jax.random.split(rng_key)
                next_token = _sample_token(next_logits, temperature, top_k, subkey)
                generated_ids.append(next_token)

        if pbar_hits:
            pbar_hits.close()

        match_rate = successful_runs / total_runs if total_runs > 0 else 0
        if not HAS_TQDM:
            print(f"\r      {successful_runs}/{min_target_count} hits "
                  f"(run {total_runs}) — Done!          ")
            print(f"      Match rate: {match_rate*100:.1f}% "
                  f"({successful_runs}/{total_runs})")

        if pbar_overall:
            pbar_overall.update(1)

        # Compute per-pool results (matching BehavioralAnalyzer schema)
        target_key = f"{query_type}_{label}"
        target_result = {
            'prompt': prompt,
            'target_token': target_text,
            'query_type': query_type,
            'successful_runs': successful_runs,
            'total_runs': total_runs,
            'match_rate': match_rate,
            'sample_generations': sample_generations,
            'per_pool': {},
        }

        if successful_runs > 0:
            for pool in pools:
                pool_target_counts = target_neuron_counts[pool]
                pool_baseline_counts = baseline_neuron_counts[pool]

                target_freq = {n: c / successful_runs
                               for n, c in pool_target_counts.items()}
                baseline_freq = ({n: c / total_baseline_steps
                                  for n, c in pool_baseline_counts.items()}
                                 if total_baseline_steps > 0 else {})

                # Contrastive = target_freq - baseline_freq (positive = target-specific)
                all_neurons_in_pool = set(target_freq.keys()) | set(baseline_freq.keys())
                contrastive_scores = {
                    n: target_freq.get(n, 0) - baseline_freq.get(n, 0)
                    for n in all_neurons_in_pool
                }

                common_100 = sorted(n for n, f in target_freq.items() if f >= 1.0)
                common_80 = sorted(n for n, f in target_freq.items() if f >= 0.8)
                common_50 = sorted(n for n, f in target_freq.items() if f >= 0.5)

                target_result['per_pool'][pool] = {
                    'common_100': common_100,
                    'common_80': common_80,
                    'common_50': common_50,
                    'n_unique': len(pool_target_counts),
                    'all_frequencies': {n: float(f) for n, f in target_freq.items()},
                    'contrastive_scores': contrastive_scores,
                    'top_neurons': sorted(
                        [{'neuron': n, 'freq': float(f * 100)}
                         for n, f in target_freq.items()],
                        key=lambda x: -x['freq']
                    )[:20],
                }
                print(f"        {pool}: {len(common_100)} neurons@100%, "
                      f"{len(common_80)} neurons@80%, "
                      f"{len(common_50)} neurons@50%")
        else:
            target_result['note'] = (
                f'Target "{target_text}" not found in {total_runs} runs')

        results['per_target'][target_key] = target_result

    if pbar_overall:
        pbar_overall.close()

    # Aggregate per-pool summary (matching BehavioralAnalyzer)
    for pool in pools:
        all_common_100 = set()
        all_common_80 = set()
        for target_data in results['per_target'].values():
            if 'per_pool' in target_data and pool in target_data['per_pool']:
                all_common_100.update(target_data['per_pool'][pool].get('common_100', []))
                all_common_80.update(target_data['per_pool'][pool].get('common_80', []))
        results['per_pool'][pool] = {
            'n_common_100': len(all_common_100),
            'n_common_80': len(all_common_80),
            'top_neurons': sorted(all_common_100)[:20],
        }

    # Also compute capital vs control aggregate (for visualization)
    pool_vis = {}
    for pool in pools:
        n = pool_sizes.get(pool, 0)
        cap_freq = np.zeros(n, dtype=np.float64)
        ctrl_freq = np.zeros(n, dtype=np.float64)
        n_cap = 0
        n_ctrl = 0

        for tkey, tdata in results['per_target'].items():
            if tdata.get('successful_runs', 0) == 0:
                continue
            pool_data = tdata.get('per_pool', {}).get(pool, {})
            all_freq = pool_data.get('all_frequencies', {})

            if tdata['query_type'] == 'capital':
                for nname, f in all_freq.items():
                    # Extract raw neuron id from unified name '{pool}_{id}'
                    nid = int(nname.split('_')[-1]) if isinstance(nname, str) and '_' in nname else int(nname)
                    if nid < n:
                        cap_freq[nid] = max(cap_freq[nid], f)
                n_cap += 1
            else:
                for nname, f in all_freq.items():
                    nid = int(nname.split('_')[-1]) if isinstance(nname, str) and '_' in nname else int(nname)
                    if nid < n:
                        ctrl_freq[nid] = max(ctrl_freq[nid], f)
                n_ctrl += 1

        contrastive = cap_freq - ctrl_freq

        # Classify (paper thresholds: 0.7/0.3)
        thresh_high = 0.7
        thresh_low = 0.3
        cap_high = cap_freq >= thresh_high
        ctrl_high = ctrl_freq >= thresh_high

        shared = int((cap_high & ctrl_high).sum())
        capital_specific = int((cap_high & (ctrl_freq < thresh_low)).sum())
        control_specific = int((ctrl_high & (cap_freq < thresh_low)).sum())
        inactive = int((~(cap_freq >= thresh_low) & ~(ctrl_freq >= thresh_low)).sum())

        # Top neurons
        cap_spec_idx = np.where(cap_high & (ctrl_freq < thresh_low))[0]
        if len(cap_spec_idx) > 0:
            cap_spec_sorted = cap_spec_idx[np.argsort(contrastive[cap_spec_idx])[::-1]]
        else:
            cap_spec_sorted = np.array([], dtype=int)
        top_capital = [
            {'neuron': f'{pool}_{int(i)}', 'capital_freq': float(cap_freq[i]),
             'control_freq': float(ctrl_freq[i]),
             'contrastive': float(contrastive[i])}
            for i in cap_spec_sorted[:15]
        ]

        shared_idx = np.where(cap_high & ctrl_high)[0]
        if len(shared_idx) > 0:
            shared_sorted = shared_idx[np.argsort(
                cap_freq[shared_idx] + ctrl_freq[shared_idx])[::-1]]
        else:
            shared_sorted = np.array([], dtype=int)
        top_shared = [
            {'neuron': f'{pool}_{int(i)}', 'capital_freq': float(cap_freq[i]),
             'control_freq': float(ctrl_freq[i])}
            for i in shared_sorted[:15]
        ]

        pool_vis[pool] = {
            'n_neurons': n,
            'shared': shared,
            'capital_specific': capital_specific,
            'control_specific': control_specific,
            'inactive': inactive,
            'capital_freq': cap_freq.tolist(),
            'control_freq': ctrl_freq.tolist(),
            'contrastive_scores': contrastive.tolist(),
            'top_capital_specific': top_capital,
            'top_shared': top_shared,
            'n_capital_queries': n_cap,
            'n_control_queries': n_ctrl,
        }

    results['per_pool_visualization'] = pool_vis

    # Summary (matching BehavioralAnalyzer)
    per_pool = results['per_pool']
    most_factual = max(per_pool.items(),
                       key=lambda x: x[1].get('n_common_80', 0)) if per_pool else ('unknown', {})
    results['summary'] = {
        'most_factual_pool': most_factual[0],
        'total_factual_neurons': sum(p.get('n_common_80', 0) for p in per_pool.values()),
    }

    results['meta'] = {
        'method': 'autoregressive_generation',
        'min_target_count': min_target_count,
        'max_runs': max_runs,
        'max_tokens_per_run': max_tokens_per_run,
        'temperature': temperature,
        'top_k': top_k,
        'pools': pools,
        'threshold_high': 0.7,
        'threshold_low': 0.3,
        'note': 'JAX/TPU port of BehavioralAnalyzer.analyze_factual_neurons(). '
                'Uses unified naming ({pool}_{neuron_id}), per-step baseline extraction, '
                'and contrastive scoring (target_freq - baseline_freq).',
    }

    return results


def _sample_token(logits_np, temperature, top_k, rng_key):
    """Sample next token from logits."""
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


# ============================================================
# Visualization (unchanged from original)
# ============================================================

def plot_knowledge_coherence(results, output_dir, dpi=300):
    """Generate coherence figures for each knowledge pool."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")
        return []

    saved = []

    for pool_key in results.get('meta', {}).get('pools', ['fv', 'rv', 'fknow', 'rknow']):
        # Use per_pool_visualization for capital vs control aggregate data
        vis_data = results.get('per_pool_visualization', results.get('per_pool', {}))
        data = vis_data.get(pool_key)
        if data is None or not isinstance(data, dict) or 'capital_freq' not in data:
            continue

        cap_freq = np.array(data['capital_freq'])
        ctrl_freq = np.array(data['control_freq'])
        n = data['n_neurons']
        pool_label = {'fknow': 'F-Know', 'rknow': 'R-Know',
                      'fv': 'F-V', 'rv': 'R-V'}.get(pool_key, pool_key)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: Scatter
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

        # Panel 2: Bar chart
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

    return saved


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Knowledge Neuron Coherence — Appendix D.3 '
                    '(autoregressive generation)')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--output', default='./section4_results')
    parser.add_argument('--min_targets', type=int, default=100,
                        help='Target token occurrences to collect per prompt')
    parser.add_argument('--max_runs', type=int, default=500,
                        help='Max generation runs per prompt (safety limit)')
    parser.add_argument('--max_tokens', type=int, default=200,
                        help='Max tokens per generation run')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--pools', nargs='+', default=['fv', 'rv', 'fknow', 'rknow'],
                        help='Pools to analyze (default: fv rv fknow rknow)')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model_cls, params, _, config = load_model_jax(args.checkpoint)
    print(f"  n_feature_know={config.get('n_feature_know')}, "
          f"n_restore_know={config.get('n_restore_know')}")

    print(f"\n=== Knowledge Neuron Coherence (Appendix D.3) ===")
    print(f"  Method: autoregressive generation")
    print(f"  Target hits: {args.min_targets}, max runs: {args.max_runs}")

    results = analyze_knowledge_coherence(
        model_cls, params, config,
        pools=args.pools,
        min_target_count=args.min_targets,
        max_runs=args.max_runs,
        max_tokens_per_run=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # Save JSON
    json_path = os.path.join(args.output, 'knowledge_coherence.json')
    save_results(results, json_path)
    print(f"  Saved: {json_path}")

    # Print summary (BehavioralAnalyzer style)
    summary = results.get('summary', {})
    print(f"\n  {'='*60}")
    print(f"  FACTUAL NEURON ANALYSIS SUMMARY")
    print(f"  {'='*60}")

    print(f"\n  Per-Pool Results:")
    for pool, pool_data in results.get('per_pool', {}).items():
        n_common = pool_data.get('n_common_80', 0)
        top_neurons = pool_data.get('top_neurons', [])[:3]
        print(f"    {pool:8s}: {n_common:3d} neurons (80%+), top: {top_neurons}")

    print(f"\n  Summary:")
    print(f"    Most factual pool: {summary.get('most_factual_pool', 'unknown')}")
    print(f"    Total factual neurons: {summary.get('total_factual_neurons', 0)}")

    # Capital vs control visualization data
    vis_data = results.get('per_pool_visualization', {})
    for pool_key in args.pools:
        d = vis_data.get(pool_key)
        if d is None or not isinstance(d, dict) or 'shared' not in d:
            continue
        pool_label = {'fknow': 'F-Know', 'rknow': 'R-Know',
                      'fv': 'F-V', 'rv': 'R-V'}.get(pool_key, pool_key)
        print(f"\n  {pool_label} (n={d['n_neurons']}):")
        print(f"    Shared:           {d['shared']}")
        print(f"    Capital-specific: {d['capital_specific']}")
        print(f"    Control-specific: {d['control_specific']}")
        print(f"    Inactive:         {d['inactive']}")
        if d.get('top_capital_specific'):
            top3 = d['top_capital_specific'][:3]
            print(f"    Top capital neurons: " +
                  ', '.join(f"{t['neuron']}(D={t['contrastive']:.2f})" for t in top3))

    print(f"\n  {'='*60}")

    # Plot
    plot_knowledge_coherence(results, args.output, dpi=args.dpi)

    print("\nDone.")


if __name__ == '__main__':
    main()
