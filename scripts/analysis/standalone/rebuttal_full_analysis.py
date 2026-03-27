#!/usr/bin/env python3
"""
DAWN Rebuttal Full Analysis — One-Touch Pipeline (JAX/TPU)
============================================================
Runs all rebuttal analyses in sequence and generates rebuttal_summary.txt.

Analyses:
  D.1  Q/K Specialization (400M)
  D.2  POS Selectivity (400M)
  D.3  Knowledge Neurons — Physics Domain (400M)
  D.4  Layer-wise Attention/Knowledge Balance (400M)
  D.5  Suppression Sweep (new contribution, 400M)

Usage:
    # Full run (TPU)
    python scripts/analysis/standalone/rebuttal_full_analysis.py \
        --checkpoint gs://dawn-tpu-data-c4/checkpoints/dawn_v17_1_400M_c4_20B_v4_32/run_v17.1_20260210_160828_3201 \
        --val_data gs://dawn-tpu-data-c4/data/c4_val.bin \
        --output results/rebuttal/

    # Fast mode (verification)
    python scripts/analysis/standalone/rebuttal_full_analysis.py \
        --checkpoint gs://... --val_data gs://... --output results/rebuttal_fast/ --fast
"""

import sys
import os
from pathlib import Path
import time
import json
import argparse
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise RuntimeError("JAX required — this script is designed for TPU")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='DAWN Rebuttal Full Analysis (JAX/TPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint path (local or gs://)')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Validation data path (.bin or .pt)')
    parser.add_argument('--output', type=str, default='results/rebuttal',
                        help='Output directory (default: results/rebuttal)')

    # Per-analysis hyperparameters
    parser.add_argument('--d1_batches', type=int, default=None,
                        help='D.1 batch count (default: 200, fast: 20)')
    parser.add_argument('--d2_sentences', type=int, default=None,
                        help='D.2 sentence count (default: 5000, fast: 500)')
    parser.add_argument('--d3_min_targets', type=int, default=None,
                        help='D.3 min target hits (default: 100, fast: 20)')
    parser.add_argument('--d3_max_runs', type=int, default=None,
                        help='D.3 max generation runs (default: 500, fast: 100)')
    parser.add_argument('--d4_batches', type=int, default=None,
                        help='D.4 batch count (default: 200, fast: 20)')

    # Fast mode
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: reduced counts for quick verification')

    # Skip individual analyses
    parser.add_argument('--skip', type=str, default='',
                        help='Comma-separated analyses to skip (e.g. "d1,d2")')

    args = parser.parse_args()

    # Resolve defaults: explicit > fast > full
    FULL = {'d1_batches': 200, 'd2_sentences': 5000,
            'd3_min_targets': 100, 'd3_max_runs': 500, 'd4_batches': 200}
    FAST = {'d1_batches': 20, 'd2_sentences': 500,
            'd3_min_targets': 20, 'd3_max_runs': 100, 'd4_batches': 20}

    defaults = FAST if args.fast else FULL
    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    args.skip_set = {s.strip().lower() for s in args.skip.split(',') if s.strip()}

    return args


# ============================================================
# Analysis functions (each returns a results dict)
# ============================================================

def run_d1_qk_specialization(model_cls, params, config, val_tokens, args):
    """D.1 Q/K Specialization reproduction."""
    from scripts.analysis.visualizers.qk_specialization_jax import analyze_qk_specialization

    print(f"  Batches: {args.d1_batches}, batch_size=16, seq_len=512")
    results = analyze_qk_specialization(
        model_cls, params, config, val_tokens,
        n_batches=args.d1_batches, batch_size=16, seq_len=512,
    )

    # Print results per pool
    for pool_name, pool_data in results.items():
        if pool_name == 'meta':
            continue
        display = pool_data['display']
        n = pool_data['n_neurons']
        q_spec = pool_data['q_specialized']
        k_spec = pool_data['k_specialized']
        shared = pool_data['shared']
        inactive = pool_data['inactive']
        active = n - inactive
        spec_pct = (q_spec + k_spec) / active * 100 if active > 0 else 0

        print(f"\n  {display} ({n} neurons):")
        print(f"    Correlation (all): r={pool_data['correlation']:.4f}")
        print(f"    Correlation (active): r={pool_data['correlation_active']:.4f}")
        print(f"    Q-only: {q_spec}  K-only: {k_spec}  Shared: {shared}  Inactive: {inactive}")
        print(f"    Specialization: {spec_pct:.1f}% (of {active} active)")
        print(f"    Avg Q/K overlap: {pool_data['avg_overlap']:.4f}")

        # Threshold sensitivity
        print(f"    Threshold sensitivity:")
        for thresh, stats in sorted(pool_data['sensitivity_analysis'].items()):
            t_active = stats['q_specialized'] + stats['k_specialized'] + stats['shared']
            t_spec = stats['q_specialized'] + stats['k_specialized']
            t_pct = t_spec / t_active * 100 if t_active > 0 else 0
            print(f"      θ={thresh}: Q={stats['q_specialized']} K={stats['k_specialized']} "
                  f"Shared={stats['shared']} → {t_pct:.1f}% specialized")

    # Save intermediate
    output_dir = Path(args.output) / 'd1_qk_specialization'
    output_dir.mkdir(parents=True, exist_ok=True)
    from scripts.analysis.standalone.neuron_suppression_experiment_jax import make_serializable
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\n  Saved: {output_dir / 'results.json'}")

    return results

def run_d2_pos_selectivity(model_cls, params, config, args):
    """D.2 POS Selectivity reproduction."""
    from scripts.analysis.visualizers.pos_selectivity_jax import (
        analyze_pos_selectivity, load_ud_ewt,
    )

    print(f"  Loading UD-EWT (max {args.d2_sentences} sentences)...")
    dataset = load_ud_ewt(split='train', max_sentences=args.d2_sentences)
    print(f"  Loaded {len(dataset)} sentences")

    # Run on key pools: F-V and R-V (paper D.2 focus)
    pools_to_analyze = ['fv', 'rv']
    all_pool_results = {}

    for pool in pools_to_analyze:
        print(f"\n  Analyzing pool: {pool} (multi-layer)")
        results, selectivity = analyze_pos_selectivity(
            model_cls, params, config, dataset,
            pool_type=pool, max_sentences=args.d2_sentences,
            multi_layer=True, batch_size=16,
        )
        all_pool_results[pool] = results

        # Print top selective neurons per POS
        print(f"\n  [{pool.upper()}] Top POS selectivity:")
        top_per_pos = results.get('top_selective_per_pos', {})
        for pos, neurons in sorted(top_per_pos.items()):
            if not neurons:
                continue
            top1 = neurons[0]
            n_specialists = sum(1 for n in neurons if n.get('is_specialist'))
            print(f"    {pos:<6s}: top neuron={top1['neuron']:3d} "
                  f"sel={top1['selectivity']:.1f}x  "
                  f"({n_specialists} specialists)")

    # Save intermediate
    output_dir = Path(args.output) / 'd2_pos_selectivity'
    output_dir.mkdir(parents=True, exist_ok=True)
    from scripts.analysis.standalone.neuron_suppression_experiment_jax import make_serializable
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(make_serializable(all_pool_results), f, indent=2)
    print(f"\n  Saved: {output_dir / 'results.json'}")

    return all_pool_results

def run_d3_knowledge_neurons(model_cls, params, config, tokenizer, args):
    """D.3 Knowledge Neurons — Physics domain via contrastive score."""
    from scripts.analysis.standalone.neuron_suppression_experiment_jax import (
        NeuronSuppressionExperimentJAX, QUERY_PRESETS,
        make_serializable,
    )
    from scripts.analysis.utils_jax import create_model_from_config

    model_instance = create_model_from_config(config)
    preset = QUERY_PRESETS['physics']
    target_queries = preset['target_queries']
    control_queries = preset['control_queries']

    experiment = NeuronSuppressionExperimentJAX(
        model_instance, params, config, tokenizer)

    print(f"  min_target_count={args.d3_min_targets}, max_runs={args.d3_max_runs}")
    print(f"  Target queries: {len(target_queries)}, Control queries: {len(control_queries)}")

    # Baseline top-10 probabilities
    print("\n  --- Baseline probabilities ---")
    baseline_probs = {}
    for q in target_queries + control_queries:
        bp = experiment.get_next_token_probs(q['prompt'])
        baseline_probs[q['prompt']] = bp
        target_lower = q['target'].strip().lower()
        target_prob = 0.0
        for tok, _, prob in bp['top_tokens']:
            if tok.lower() == target_lower:
                target_prob = prob
                break
        tag = 'physics' if q in target_queries else 'control'
        print(f"    [{tag}] \"{q['prompt']}\" → '{q['target']}': {target_prob:.2%}")

    # Collect activation frequencies (contrastive scores) for physics queries
    print("\n  --- Contrastive score collection (physics queries) ---")
    freq_results = []
    for q in target_queries:
        print(f"\n    \"{q['prompt']}\" → '{q['target']}'")
        freq = experiment.collect_activation_frequencies(
            q['prompt'], q['target'],
            min_target_count=args.d3_min_targets,
            max_runs=args.d3_max_runs,
        )
        freq_results.append(freq)

        # Top contrastive neurons per pool
        for pool in ['fv', 'rv', 'fknow', 'rknow']:
            pool_key = {'fv': 'fv', 'rv': 'rv', 'fknow': 'feature_know',
                        'rknow': 'restore_know'}.get(pool, pool)
            scores = freq['neuron_scores'].get(pool_key, {})
            if not scores:
                continue
            top3 = sorted(scores.items(),
                          key=lambda x: x[1]['contrastive'], reverse=True)[:3]
            top3_str = ", ".join(f"n{n}({s['contrastive']:+.3f})" for n, s in top3)
            print(f"      {pool}: {top3_str}")

    # Also collect for control queries (with reduced count)
    print("\n  --- Contrastive score collection (control queries) ---")
    control_freqs = []
    for q in control_queries:
        print(f"    \"{q['prompt']}\" → '{q['target']}'")
        freq = experiment.collect_activation_frequencies(
            q['prompt'], q['target'],
            min_target_count=max(20, args.d3_min_targets // 5),
            max_runs=args.d3_max_runs,
        )
        control_freqs.append(freq)

    results = {
        'baseline_probs': baseline_probs,
        'physics_frequencies': freq_results,
        'control_frequencies': control_freqs,
    }

    # Save intermediate
    output_dir = Path(args.output) / 'd3_knowledge_neurons'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\n  Saved: {output_dir / 'results.json'}")

    return results

def run_d4_layer_balance(params, config, val_tokens, args):
    """D.4 Layer-wise Attention/Knowledge Balance."""
    from scripts.analysis.visualizers.layer_balance_jax import analyze_layer_balance

    print(f"  Batches: {args.d4_batches}, batch_size=4, seq_len=512")
    results = analyze_layer_balance(
        params, config, val_tokens,
        n_batches=args.d4_batches, batch_size=4, seq_len=512,
    )

    # Print per-layer results
    n_layers = results['n_layers']
    print(f"\n  Layer-wise Attention Contribution (%):")
    for p in results['per_layer']:
        bar_len = int(p['attention_ratio'] / 2)
        bar = '#' * bar_len + '.' * (50 - bar_len)
        print(f"    L{p['layer']:2d}: {p['attention_ratio']:5.1f}% attn  "
              f"{p['knowledge_ratio']:5.1f}% know  |{bar}|")

    s = results['summary']
    print(f"\n  Early layers (L0-{n_layers//3-1}):  {s['early_layers_attn']:.1f}% attention")
    print(f"  Mid layers:              {s['mid_layers_attn']:.1f}% attention")
    print(f"  Late layers:             {s['late_layers_attn']:.1f}% attention")

    # Save intermediate
    output_dir = Path(args.output) / 'd4_layer_balance'
    output_dir.mkdir(parents=True, exist_ok=True)
    from scripts.analysis.standalone.neuron_suppression_experiment_jax import make_serializable
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\n  Saved: {output_dir / 'results.json'}")

    return results

def greedy_generate(forward_fn, tokenizer, prompt, max_tokens=50):
    """Greedy decode up to max_tokens using a forward function (baseline or suppressed)."""
    input_ids = [101] + tokenizer.encode(prompt, add_special_tokens=False)
    generated = list(input_ids)

    for _ in range(max_tokens):
        input_arr = jnp.array([generated])
        logits = forward_fn(input_arr)
        next_id = int(jnp.argmax(logits[0, -1, :]))
        if next_id in (tokenizer.sep_token_id, tokenizer.eos_token_id, 0):
            break
        generated.append(next_id)

    return tokenizer.decode(generated, skip_special_tokens=True)


def run_d5_suppression_sweep(model_cls, params, config, tokenizer, args):
    """D.5 Suppression Sweep + generation samples."""
    from scripts.analysis.standalone.neuron_suppression_experiment_jax import (
        NeuronSuppressionExperimentJAX, QUERY_PRESETS,
        build_suppressed_forward, build_masks_from_sets,
        make_serializable,
    )
    from scripts.analysis.utils_jax import create_model_from_config

    model_instance = create_model_from_config(config)
    preset = QUERY_PRESETS['physics']
    target_queries = preset['target_queries']
    control_queries = preset['control_queries']

    experiment = NeuronSuppressionExperimentJAX(
        model_instance, params, config, tokenizer)

    # --- Collect activation frequencies (reuse D.3 if available) ---
    d3_path = Path(args.output) / 'd3_knowledge_neurons' / 'results.json'
    if d3_path.exists():
        print("  Reusing D.3 activation frequencies from cache")
        with open(d3_path) as f:
            d3_data = json.load(f)
        freq_results = d3_data.get('physics_frequencies', [])
        # Need to reconstruct neuron_scores format for identify_suppression_targets
        if freq_results and 'neuron_scores' in freq_results[0]:
            print(f"  Found {len(freq_results)} cached frequency results")
        else:
            freq_results = None
    else:
        freq_results = None

    if not freq_results:
        print("  Collecting activation frequencies for physics queries...")
        freq_results = []
        for q in target_queries:
            print(f"    \"{q['prompt']}\" → '{q['target']}'")
            freq = experiment.collect_activation_frequencies(
                q['prompt'], q['target'],
                min_target_count=args.d3_min_targets,
                max_runs=args.d3_max_runs,
            )
            freq_results.append(freq)

    # --- Pre-suppression generation samples ---
    print("\n  === Pre-suppression Generation Samples ===")
    baseline_forward = experiment._baseline_forward
    pre_generations = {}
    for q in target_queries:
        text = greedy_generate(baseline_forward, tokenizer, q['prompt'], max_tokens=50)
        pre_generations[q['prompt']] = text
        print(f"    '{q['prompt']}' → '{text[:80]}...'")

    # --- Sweep over top_n_pct values ---
    sweep_pcts = [0.03, 0.04, 0.05, 0.10]
    sweep_results = []

    for pct in sweep_pcts:
        print(f"\n  --- Sweep: top_n_pct={pct:.2f}, mode=union ---")

        suppressed = experiment.identify_suppression_targets(
            freq_results, top_n_pct=pct, mode='union')
        total_neurons = sum(len(v) for v in suppressed.values())
        print(f"    Suppressed neurons: {total_neurons}")

        if total_neurons == 0:
            sweep_results.append({
                'pct': pct, 'n_neurons': 0,
                'target_drops': [], 'control_drops': [],
                'avg_target_drop': 0, 'avg_control_drop': 0,
                'selectivity_index': 0, 'verdict': 'NO NEURONS',
            })
            continue

        # Build suppressed forward
        masks = build_masks_from_sets(suppressed, config)
        suppressed_forward = build_suppressed_forward(
            model_instance, params, config, masks)

        # Measure target probs
        target_drops = []
        for q in target_queries:
            bp = experiment.get_next_token_probs(q['prompt'])
            sp = experiment.get_next_token_probs(q['prompt'], forward_fn=suppressed_forward)
            target_lower = q['target'].strip().lower()

            pre_p = next((p for t, _, p in bp['top_tokens'] if t.lower() == target_lower), 0.0)
            post_p = next((p for t, _, p in sp['top_tokens'] if t.lower() == target_lower), 0.0)
            target_drops.append(pre_p - post_p)

        control_drops = []
        for q in control_queries:
            bp = experiment.get_next_token_probs(q['prompt'])
            sp = experiment.get_next_token_probs(q['prompt'], forward_fn=suppressed_forward)
            target_lower = q['target'].strip().lower()

            pre_p = next((p for t, _, p in bp['top_tokens'] if t.lower() == target_lower), 0.0)
            post_p = next((p for t, _, p in sp['top_tokens'] if t.lower() == target_lower), 0.0)
            control_drops.append(pre_p - post_p)

        avg_td = float(np.mean(target_drops))
        avg_cd = float(np.mean(control_drops))
        sel_idx = avg_td - avg_cd

        verdict = ('SELECTIVE' if sel_idx > 0.1
                   else 'WEAK' if sel_idx > 0
                   else 'NON-SELECTIVE')

        print(f"    Target drop: {avg_td:+.2%}  Control drop: {avg_cd:+.2%}  "
              f"Selectivity: {sel_idx:+.2%}  → {verdict}")

        entry = {
            'pct': pct, 'n_neurons': total_neurons,
            'suppressed_per_pool': {k: len(v) for k, v in suppressed.items()},
            'target_drops': [float(d) for d in target_drops],
            'control_drops': [float(d) for d in control_drops],
            'avg_target_drop': avg_td,
            'avg_control_drop': avg_cd,
            'selectivity_index': sel_idx,
            'verdict': verdict,
        }

        # Generation samples at this sweep point
        if pct == sweep_pcts[-1]:  # last (most aggressive) sweep
            print(f"\n  === Post-suppression Generation (pct={pct}) ===")
            post_generations = {}
            for q in target_queries:
                text = greedy_generate(suppressed_forward, tokenizer, q['prompt'], max_tokens=50)
                post_generations[q['prompt']] = text
                print(f"    '{q['prompt']}' → '{text[:80]}...'")
            entry['post_generations'] = post_generations

        sweep_results.append(entry)

    # Print sweep summary table
    print(f"\n  === Suppression Sweep Summary ===")
    print(f"  {'pct':>5s} | {'neurons':>7s} | {'target_drop':>11s} | {'control_drop':>12s} | {'selectivity':>11s} | verdict")
    print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*11}-+-{'-'*12}-+-{'-'*11}-+--------")
    for r in sweep_results:
        print(f"  {r['pct']:5.2f} | {r['n_neurons']:7d} | {r['avg_target_drop']:>+10.2%} | "
              f"{r['avg_control_drop']:>+11.2%} | {r['selectivity_index']:>+10.2%} | {r['verdict']}")

    results = {
        'sweep': sweep_results,
        'pre_generations': pre_generations,
    }

    # Save intermediate
    output_dir = Path(args.output) / 'd5_suppression_sweep'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\n  Saved: {output_dir / 'results.json'}")

    return results

def generate_summary(all_results, args):
    """Generate rebuttal_summary.txt."""
    pass  # Part 7


# ============================================================
# Main orchestration
# ============================================================

def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_str = "FAST" if args.fast else "FULL"
    print("=" * 70)
    print(f"DAWN REBUTTAL FULL ANALYSIS — {mode_str} MODE")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Val data:   {args.val_data}")
    print(f"  Output:     {args.output}")
    print(f"  JAX devices: {jax.devices()}")
    if args.skip_set:
        print(f"  Skipping:   {', '.join(sorted(args.skip_set))}")
    print(f"  Params: d1_batches={args.d1_batches}, d2_sentences={args.d2_sentences}, "
          f"d3_min_targets={args.d3_min_targets}, d3_max_runs={args.d3_max_runs}, "
          f"d4_batches={args.d4_batches}")
    print("=" * 70)

    # --- Load model (shared across all analyses) ---
    print("\n[0/5] Loading model...")
    t0 = time.time()
    from scripts.analysis.utils_jax import load_model_jax, load_val_data_jax
    model_cls, params, tokenizer, config = load_model_jax(args.checkpoint)
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Version: {config.get('model_version', 'unknown')}")
    print(f"  n_layers={config.get('n_layers')}, d_model={config.get('d_model')}")

    # --- Load validation data (shared by D.1, D.4) ---
    print("\n  Loading validation data...")
    max_tokens = max(args.d1_batches, args.d4_batches) * 32 * 512
    val_tokens = load_val_data_jax(args.val_data, max_tokens=max_tokens)
    print(f"  Loaded {len(val_tokens):,} tokens")

    all_results = {}
    total_t0 = time.time()

    # --- D.1 ---
    if 'd1' not in args.skip_set:
        print("\n" + "=" * 70)
        print("[1/5] D.1 — Q/K Specialization")
        print("=" * 70)
        t0 = time.time()
        all_results['d1'] = run_d1_qk_specialization(
            model_cls, params, config, val_tokens, args)
        print(f"  D.1 done in {time.time() - t0:.1f}s")
    else:
        print("\n[1/5] D.1 — SKIPPED")

    # --- D.2 ---
    if 'd2' not in args.skip_set:
        print("\n" + "=" * 70)
        print("[2/5] D.2 — POS Selectivity")
        print("=" * 70)
        t0 = time.time()
        all_results['d2'] = run_d2_pos_selectivity(
            model_cls, params, config, args)
        print(f"  D.2 done in {time.time() - t0:.1f}s")
    else:
        print("\n[2/5] D.2 — SKIPPED")

    # --- D.3 ---
    if 'd3' not in args.skip_set:
        print("\n" + "=" * 70)
        print("[3/5] D.3 — Knowledge Neurons (Physics)")
        print("=" * 70)
        t0 = time.time()
        all_results['d3'] = run_d3_knowledge_neurons(
            model_cls, params, config, tokenizer, args)
        print(f"  D.3 done in {time.time() - t0:.1f}s")
    else:
        print("\n[3/5] D.3 — SKIPPED")

    # --- D.4 ---
    if 'd4' not in args.skip_set:
        print("\n" + "=" * 70)
        print("[4/5] D.4 — Layer-wise Balance")
        print("=" * 70)
        t0 = time.time()
        all_results['d4'] = run_d4_layer_balance(
            params, config, val_tokens, args)
        print(f"  D.4 done in {time.time() - t0:.1f}s")
    else:
        print("\n[4/5] D.4 — SKIPPED")

    # --- D.5 ---
    if 'd5' not in args.skip_set:
        print("\n" + "=" * 70)
        print("[5/5] D.5 — Suppression Sweep")
        print("=" * 70)
        t0 = time.time()
        all_results['d5'] = run_d5_suppression_sweep(
            model_cls, params, config, tokenizer, args)
        print(f"  D.5 done in {time.time() - t0:.1f}s")
    else:
        print("\n[5/5] D.5 — SKIPPED")

    # --- Summary ---
    total_elapsed = time.time() - total_t0
    print(f"\n  Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # Save raw results JSON
    raw_path = output_dir / 'rebuttal_results.json'
    from scripts.analysis.standalone.neuron_suppression_experiment_jax import make_serializable
    with open(raw_path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2, ensure_ascii=False)
    print(f"  Raw results: {raw_path}")

    # Generate summary
    generate_summary(all_results, args)

    print("\n" + "=" * 70)
    print("REBUTTAL ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
