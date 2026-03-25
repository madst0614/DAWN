#!/usr/bin/env python3
"""
Domain-Specific Neuron Suppression Experiment
===============================================
Tests whether DAWN's pseudo-neurons encode domain-specific knowledge
by suppressing neurons identified from one domain (physics/astronomy)
and measuring selective impact: target domain accuracy should drop
while unrelated domain (biology/geography) accuracy should be preserved.

Experiment design:
  Group 1 — Suppress targets (physics/astronomy):
    "light travels at the speed of"  → light  (high confidence)
    "the earth orbits the"           → sun    (high confidence)
    "the earth revolves around the"  → sun    (medium confidence)

  Group 2 — Control (biology/geography/history):
    "plants need sunlight to"        → grow
    "the amazon is the longest"      → river
    "the lungs are used for"         → breathing
    "the french revolution began in" → 1789
    "mount everest is the"           → highest

Protocol:
  Phase 1 — Baseline: Measure prediction accuracy for all queries
  Phase 2 — Neuron identification: Collect activation frequencies
            from Group 1, identify physics/astronomy neurons
  Phase 3 — Suppression: Suppress identified neurons, re-measure
            ALL queries. Group 1 should degrade, Group 2 should not.

Usage:
    python scripts/analysis/standalone/domain_suppression_experiment.py \
        --checkpoint path/to/checkpoint \
        --device cuda \
        --n_runs 100 \
        --threshold 0.7 \
        --output results/domain_suppression/
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import argparse
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from scripts.analysis.utils import load_model, ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS
from scripts.analysis.standalone.neuron_suppression_experiment import (
    SuppressionHookManager,
    ATTENTION_POOLS,
    KNOWLEDGE_POOLS,
    make_serializable,
)


# ============================================================
# Query definitions
# ============================================================

# Group 1: Physics/Astronomy — suppression target domain
PHYSICS_QUERIES = [
    {"prompt": "light travels at the speed of",  "target": "light"},
    {"prompt": "the earth orbits the",            "target": "sun"},
    {"prompt": "the earth revolves around the",   "target": "sun"},
]

# Group 2: Control — completely different domains
CONTROL_QUERIES = [
    # Biology
    {"prompt": "plants need sunlight to",         "target": "grow"},
    {"prompt": "the lungs are used for",          "target": "breathing"},
    # Geography
    {"prompt": "the amazon is the longest",       "target": "river"},
    {"prompt": "mount everest is the",            "target": "highest"},
    # History
    {"prompt": "the french revolution began in",  "target": "1789"},
]


# ============================================================
# Core experiment class
# ============================================================

class DomainSuppressionExperiment:
    """
    Domain-specific selective suppression experiment.

    1. Baseline all queries (no suppression)
    2. Collect activation frequencies from domain queries
    3. Identify domain-specific neurons via frequency threshold
    4. Suppress those neurons and re-measure all queries
    5. Compute selectivity metrics
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hook_manager = SuppressionHookManager()
        self.model.eval()

    # ----------------------------------------------------------
    # Measure: greedy next-token match rate + top-5
    # ----------------------------------------------------------

    @torch.no_grad()
    def measure_accuracy(
        self,
        prompt: str,
        target_token: str,
        n_runs: int = 100,
    ) -> Dict:
        """Run n_runs greedy forward passes and measure target hit rate."""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)
        if len(target_id) == 0:
            raise ValueError(f"Target token '{target_token}' not in vocabulary")
        target_id = target_id[0]

        match_count = 0
        token_counts = defaultdict(int)

        for _ in range(n_runs):
            logits = self.model(input_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            next_id = logits[0, -1, :].argmax().item()
            token_counts[next_id] += 1
            if next_id == target_id:
                match_count += 1

        # Top-5 tokens
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
    # Collect activation frequencies (with routing info)
    # ----------------------------------------------------------

    @torch.no_grad()
    def collect_activation_frequencies(
        self,
        prompt: str,
        target_token: str,
        n_runs: int = 100,
    ) -> Dict:
        """
        Run greedy forward passes, recording which neurons are in
        top-k selection at the last (prediction) position.
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)
        if len(target_id) == 0:
            raise ValueError(f"Target token '{target_token}' not in vocabulary")
        target_id = target_id[0]

        freq = {pool: defaultdict(int)
                for pool in list(ATTENTION_POOLS.keys()) + list(KNOWLEDGE_POOLS.keys())}
        match_count = 0

        for _ in range(n_runs):
            logits, routing_infos = self.model(input_tensor, return_routing_info=True)

            next_token_id = logits[0, -1, :].argmax().item()
            if next_token_id == target_id:
                match_count += 1

            for layer_info in routing_infos:
                attn_info = layer_info.get('attention', {})
                know_info = layer_info.get('knowledge', {})

                for pool_name, weight_key in ATTENTION_POOLS.items():
                    weights = attn_info.get(weight_key)
                    if weights is None:
                        continue
                    w_last = weights[0, -1]
                    active_idx = (w_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                    for idx in active_idx:
                        freq[pool_name][idx] += 1

                for pool_name, weight_key in KNOWLEDGE_POOLS.items():
                    weights = know_info.get(weight_key)
                    if weights is None:
                        continue
                    w_last = weights[0, -1]
                    active_idx = (w_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                    for idx in active_idx:
                        freq[pool_name][idx] += 1

        return {
            'prompt': prompt,
            'target_token': target_token,
            'target_token_id': target_id,
            'match_count': match_count,
            'total_runs': n_runs,
            'match_rate': match_count / n_runs,
            'neuron_frequencies': {pool: dict(counts) for pool, counts in freq.items()},
        }

    # ----------------------------------------------------------
    # Identify domain neurons (union across domain queries)
    # ----------------------------------------------------------

    def identify_domain_neurons(
        self,
        freq_results: List[Dict],
        threshold: float = 0.7,
        mode: str = 'union',
    ) -> Dict[str, Set[int]]:
        """
        Find neurons active in ≥threshold fraction of runs.

        Args:
            mode: 'union' = active in ANY query (more aggressive, captures
                  broader domain knowledge); 'intersection' = active in ALL.
        """
        pool_names = list(ATTENTION_POOLS.keys()) + list(KNOWLEDGE_POOLS.keys())
        targets = {}

        for pool in pool_names:
            per_query_sets = []
            for result in freq_results:
                n_runs = result['total_runs']
                min_count = int(n_runs * threshold)
                pool_freq = result['neuron_frequencies'].get(pool, {})
                meeting = {int(idx) for idx, count in pool_freq.items()
                           if count >= min_count}
                per_query_sets.append(meeting)

            if not per_query_sets:
                continue

            if mode == 'union':
                combined = set()
                for s in per_query_sets:
                    combined |= s
            else:  # intersection
                combined = per_query_sets[0]
                for s in per_query_sets[1:]:
                    combined &= s

            if combined:
                targets[pool] = combined

        return targets

    # ----------------------------------------------------------
    # Full experiment orchestration
    # ----------------------------------------------------------

    def run_full_experiment(
        self,
        domain_queries: List[Dict] = None,
        control_queries: List[Dict] = None,
        n_runs: int = 100,
        threshold: float = 0.7,
        mode: str = 'union',
    ) -> Dict:
        """
        Run the complete domain-specific suppression experiment.

        Returns:
            Full experiment results with selectivity metrics.
        """
        if domain_queries is None:
            domain_queries = PHYSICS_QUERIES
        if control_queries is None:
            control_queries = CONTROL_QUERIES

        results = {
            'config': {
                'n_runs': n_runs,
                'threshold': threshold,
                'mode': mode,
                'domain_queries': domain_queries,
                'control_queries': control_queries,
                'domain': 'physics_astronomy',
                'control_domains': ['biology', 'geography', 'history'],
            },
            'phase1_baseline': {},
            'phase2_neuron_id': {},
            'phase3_suppressed': {},
            'selectivity': {},
        }

        # ==============================
        # Phase 1: Baseline (no suppression)
        # ==============================
        print("=" * 70)
        print("PHASE 1: Baseline accuracy (no suppression)")
        print("=" * 70)

        print("\n  --- Domain queries (physics/astronomy) ---")
        domain_baselines = []
        for q in domain_queries:
            print(f"  \"{q['prompt']}\" → '{q['target']}'")
            t0 = time.time()
            baseline = self.measure_accuracy(q['prompt'], q['target'], n_runs=n_runs)
            elapsed = time.time() - t0
            print(f"    Accuracy: {baseline['match_rate']:.0%}  [{elapsed:.1f}s]")
            if baseline['top5']:
                top_tok, _, top_pct = baseline['top5'][0]
                print(f"    Top token: '{top_tok}' ({top_pct:.0%})")
            domain_baselines.append(baseline)

        print(f"\n  --- Control queries (biology/geography/history) ---")
        control_baselines = []
        for q in control_queries:
            print(f"  \"{q['prompt']}\" → '{q['target']}'")
            t0 = time.time()
            baseline = self.measure_accuracy(q['prompt'], q['target'], n_runs=n_runs)
            elapsed = time.time() - t0
            print(f"    Accuracy: {baseline['match_rate']:.0%}  [{elapsed:.1f}s]")
            control_baselines.append(baseline)

        results['phase1_baseline']['domain'] = domain_baselines
        results['phase1_baseline']['control'] = control_baselines

        # ==============================
        # Phase 2: Identify domain neurons
        # ==============================
        print("\n" + "=" * 70)
        print("PHASE 2: Identifying physics/astronomy neurons")
        print("=" * 70)

        freq_results = []
        for q in domain_queries:
            print(f"\n  Collecting activations: \"{q['prompt']}\" → '{q['target']}'")
            t0 = time.time()
            freq = self.collect_activation_frequencies(
                q['prompt'], q['target'], n_runs=n_runs
            )
            elapsed = time.time() - t0
            print(f"    Match rate: {freq['match_rate']:.0%}  [{elapsed:.1f}s]")

            for pool in list(ATTENTION_POOLS.keys()) + list(KNOWLEDGE_POOLS.keys()):
                pool_freq = freq['neuron_frequencies'].get(pool, {})
                high_freq = {k: v for k, v in pool_freq.items()
                             if v >= n_runs * threshold}
                if high_freq:
                    print(f"    {pool}: {len(high_freq)} neurons ≥{threshold:.0%}")

            freq_results.append(freq)

        results['phase2_neuron_id']['activation_frequencies'] = freq_results

        # Identify neurons to suppress
        suppressed = self.identify_domain_neurons(freq_results, threshold, mode)
        total_suppressed = sum(len(v) for v in suppressed.values())

        print(f"\n  Mode: {mode} | Threshold: {threshold:.0%}")
        print(f"  Total physics/astronomy neurons: {total_suppressed}")
        for pool, indices in sorted(suppressed.items()):
            print(f"    {pool}: {len(indices)} neurons — "
                  f"{sorted(indices)[:10]}{'...' if len(indices) > 10 else ''}")

        results['phase2_neuron_id']['suppressed_neurons'] = {
            k: sorted(v) for k, v in suppressed.items()
        }
        results['phase2_neuron_id']['total_suppressed'] = total_suppressed

        if total_suppressed == 0:
            print("\n  WARNING: No neurons met the threshold! "
                  "Try lowering --threshold or using --mode union")
            results['phase3_suppressed']['note'] = 'no neurons to suppress'
            return results

        # ==============================
        # Phase 3: Suppression + re-measure
        # ==============================
        print("\n" + "=" * 70)
        print("PHASE 3: Measuring post-suppression (physics neurons OFF)")
        print("=" * 70)

        self.hook_manager.set_suppressed_neurons(suppressed)
        self.hook_manager.install(self.model)
        print(f"  Suppression installed: {total_suppressed} neurons")

        # Domain queries — should degrade
        print("\n  --- Domain queries (should degrade) ---")
        domain_post = []
        for q in domain_queries:
            print(f"  \"{q['prompt']}\" → '{q['target']}'")
            post = self.measure_accuracy(q['prompt'], q['target'], n_runs=n_runs)
            print(f"    Accuracy: {post['match_rate']:.0%}")
            if post['top5']:
                top_tok, _, top_pct = post['top5'][0]
                print(f"    Top token: '{top_tok}' ({top_pct:.0%})")
            domain_post.append(post)

        # Control queries — should be preserved
        print(f"\n  --- Control queries (should be preserved) ---")
        control_post = []
        for q in control_queries:
            print(f"  \"{q['prompt']}\" → '{q['target']}'")
            post = self.measure_accuracy(q['prompt'], q['target'], n_runs=n_runs)
            print(f"    Accuracy: {post['match_rate']:.0%}")
            control_post.append(post)

        self.hook_manager.remove()

        results['phase3_suppressed']['domain'] = domain_post
        results['phase3_suppressed']['control'] = control_post

        # ==============================
        # Selectivity analysis
        # ==============================
        selectivity = self._compute_selectivity(
            domain_baselines, domain_post,
            control_baselines, control_post,
        )
        results['selectivity'] = selectivity

        # Print summary
        self._print_summary(results)

        return results

    # ----------------------------------------------------------
    # Selectivity metrics
    # ----------------------------------------------------------

    def _compute_selectivity(
        self,
        domain_pre: List[Dict],
        domain_post: List[Dict],
        control_pre: List[Dict],
        control_post: List[Dict],
    ) -> Dict:
        """
        Compute selectivity metrics:
        - domain_drop: average accuracy drop in target domain
        - control_drop: average accuracy change in control domain
        - selectivity_index: domain_drop - control_drop
          (positive = selective; larger = more domain-specific)
        """
        domain_drops = []
        for pre, post in zip(domain_pre, domain_post):
            drop = pre['match_rate'] - post['match_rate']
            domain_drops.append(drop)

        control_drops = []
        for pre, post in zip(control_pre, control_post):
            drop = pre['match_rate'] - post['match_rate']
            control_drops.append(drop)

        avg_domain_drop = np.mean(domain_drops) if domain_drops else 0.0
        avg_control_drop = np.mean(control_drops) if control_drops else 0.0
        selectivity_index = avg_domain_drop - avg_control_drop

        return {
            'domain_drops': domain_drops,
            'control_drops': control_drops,
            'avg_domain_drop': float(avg_domain_drop),
            'avg_control_drop': float(avg_control_drop),
            'selectivity_index': float(selectivity_index),
            'interpretation': (
                'SELECTIVE: domain accuracy dropped significantly more than control'
                if selectivity_index > 0.1
                else 'WEAK: suppression affected both domains similarly'
                if selectivity_index > 0.0
                else 'NON-SELECTIVE: control dropped more than domain'
            ),
        }

    # ----------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------

    def _print_summary(self, results):
        """Print clean comparison table with selectivity metrics."""
        print("\n" + "=" * 90)
        print("SUMMARY: Domain-Specific Neuron Suppression Results")
        print("=" * 90)

        config = results['config']
        phase2 = results['phase2_neuron_id']
        sel = results['selectivity']

        print(f"  Domain: {config['domain']} | "
              f"Control: {', '.join(config['control_domains'])}")
        print(f"  Threshold: {config['threshold']:.0%} | "
              f"Mode: {config['mode']} | "
              f"Runs: {config['n_runs']}")
        print(f"  Suppressed: {phase2['total_suppressed']} neurons")

        # Table
        print("\n" + "-" * 95)
        print(f"  {'Group':<8s} {'Query':<40s} {'Target':<10s} "
              f"{'Pre':>6s} {'Post':>6s} {'Drop':>7s}")
        print("-" * 95)

        # Domain rows
        domain_pre = results['phase1_baseline']['domain']
        domain_post = results['phase3_suppressed'].get('domain', [])
        for pre, post in zip(domain_pre, domain_post):
            drop = pre['match_rate'] - post['match_rate']
            print(f"  {'DOMAIN':<8s} {pre['prompt'][:38]:<40s} "
                  f"{pre['target_token']:<10s} "
                  f"{pre['match_rate']:>5.0%}  {post['match_rate']:>5.0%}  "
                  f"{drop:>+6.0%}")

        print("-" * 95)

        # Control rows
        ctrl_pre = results['phase1_baseline']['control']
        ctrl_post = results['phase3_suppressed'].get('control', [])
        for pre, post in zip(ctrl_pre, ctrl_post):
            drop = pre['match_rate'] - post['match_rate']
            print(f"  {'CTRL':<8s} {pre['prompt'][:38]:<40s} "
                  f"{pre['target_token']:<10s} "
                  f"{pre['match_rate']:>5.0%}  {post['match_rate']:>5.0%}  "
                  f"{drop:>+6.0%}")

        print("-" * 95)

        # Selectivity
        print(f"\n  SELECTIVITY METRICS:")
        print(f"    Avg domain accuracy drop:  {sel['avg_domain_drop']:>+.1%}")
        print(f"    Avg control accuracy drop: {sel['avg_control_drop']:>+.1%}")
        print(f"    Selectivity index:         {sel['selectivity_index']:>+.1%}")
        print(f"    Verdict: {sel['interpretation']}")

        # Post-suppression top tokens for domain queries
        if domain_post:
            print(f"\n  Post-suppression top-5 tokens (domain queries):")
            for post in domain_post:
                tokens_str = ", ".join(
                    f"'{tok}' ({pct:.0%})" for tok, cnt, pct in post['top5']
                )
                print(f"    \"{post['prompt']}\" → {tokens_str}")

        print("=" * 90)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Domain-Specific Neuron Suppression Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (.pt file or directory, local or gs://)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--n_runs', type=int, default=100,
                        help='Number of runs per query (default: 100)')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Activation frequency threshold (default: 0.7)')
    parser.add_argument('--mode', type=str, default='union',
                        choices=['intersection', 'union'],
                        help='union = neuron active in ANY domain query (default); '
                             'intersection = active in ALL')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results JSON')
    parser.add_argument('--queries', type=str, default=None,
                        help='Path to custom queries JSON file with '
                             '"domain" and "control" keys')

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, tokenizer, config = load_model(args.checkpoint, device=args.device)
    print(f"  Model version: {model.__version__}")
    print(f"  Device: {args.device}")

    # Load custom queries if provided
    domain_queries = None
    control_queries = None
    if args.queries:
        with open(args.queries) as f:
            qdata = json.load(f)
        domain_queries = qdata.get('domain', PHYSICS_QUERIES)
        control_queries = qdata.get('control', CONTROL_QUERIES)

    # Run experiment
    experiment = DomainSuppressionExperiment(model, tokenizer, device=args.device)
    results = experiment.run_full_experiment(
        domain_queries=domain_queries,
        control_queries=control_queries,
        n_runs=args.n_runs,
        threshold=args.threshold,
        mode=args.mode,
    )

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        ckpt_name = Path(args.checkpoint).name or 'checkpoint'
        filename = (f"domain_suppression_{ckpt_name}"
                    f"_t{args.threshold}_n{args.n_runs}_{args.mode}.json")
        output_path = output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(make_serializable(results), f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
