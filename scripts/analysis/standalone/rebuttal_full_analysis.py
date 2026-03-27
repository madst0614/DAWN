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
    """D.3 Knowledge Neurons — Physics domain."""
    pass  # Part 4

def run_d4_layer_balance(params, config, val_tokens, args):
    """D.4 Layer-wise Attention/Knowledge Balance."""
    pass  # Part 5

def run_d5_suppression_sweep(model_cls, params, config, tokenizer, args):
    """D.5 Suppression Sweep + generation samples."""
    pass  # Part 6

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
