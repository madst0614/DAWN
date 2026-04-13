#!/usr/bin/env python3
"""
DAWN Pseudo-Neuron Suppression Experiment
==========================================
Identifies capital-related pseudo-neurons via activation frequency,
suppresses them by setting routing logits to -inf, and measures
the effect on target token generation.

Experiment protocol:
  Phase 1 — Activation frequency baseline
    For each capital query (e.g. "the capital of france is"),
    run 100 greedy forward passes, collect which neurons are
    in the top-k selection at the target-token position.
    Neurons active in ≥70% of runs are "capital-related".

  Phase 2 — Suppression
    Hook into UnifiedNeuronRouter.get_all_logits() and
    get_knowledge_logits() to set the logits of identified
    neurons to -inf before softmax.  This guarantees they
    are never selected by top-k.

  Phase 3 — Measurement
    Re-run the same queries + control queries 100 times each.
    Report target-token hit rate before/after suppression.

Usage:
    python scripts/analysis/standalone/neuron_suppression_experiment.py \
        --checkpoint gs://dawn-tpu-data-c4/checkpoints/dawn_v17_1_400M_c4_20B_v4_32_fair \
        --device cuda \
        --n_runs 100 \
        --threshold 0.7 \
        --output results/suppression/
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from scripts.analysis.utils import load_model, ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS

# ============================================================
# Pool definitions — maps pool name to the routing_info keys
# used to extract active neuron indices from v17.1 weights.
#
# For attention pools, routing_info['attention'][weight_key]
# gives [B, S, N_pool] sparse weights after top-k.
# For knowledge pools, routing_info['knowledge'][weight_key].
# ============================================================

ATTENTION_POOLS = {
    'fqk_Q': 'fqk_weights_Q',   # Feature QK — Q routing
    'fqk_K': 'fqk_weights_K',   # Feature QK — K routing
    'fv':    'fv_weights',       # Feature V
    'rqk_Q': 'rqk_weights_Q',   # Restore QK — Q routing
    'rqk_K': 'rqk_weights_K',   # Restore QK — K routing
    'rv':    'rv_weights',       # Restore V
}

KNOWLEDGE_POOLS = {
    'feature_know': 'feature_know_w',
    'restore_know': 'restore_know_w',
}


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
# Suppression hooks
# ============================================================

class SuppressionHookManager:
    """
    Installs forward hooks on UnifiedNeuronRouter to mask out
    specific neuron logits with -inf before softmax.

    v17.1 routing flow:
      get_all_logits(x) → 6 logit tensors [B,S,N_pool]
      get_knowledge_logits(x) → 2 logit tensors [B,S,N_pool]
    Then softmax → cumulative/token routing → top-k sparsify.

    By setting logits[:,:,idx] = -inf, softmax produces ~0 for
    those neurons, so they can never be selected by top-k.

    TPU optimization:
      Pre-compute boolean mask tensors on-device at install time.
      Use torch.where(mask, -inf, logits) — pure element-wise op,
      no Python-side indexing, no CPU sync, no clone().
    """

    def __init__(self):
        self.hooks = []
        self.suppressed = {}  # pool_name → set of neuron indices
        self.active = False
        self._masks = {}      # pool_name → bool tensor [N_pool], True = suppress

    def set_suppressed_neurons(self, suppressed: Dict[str, Set[int]]):
        """
        Args:
            suppressed: dict mapping pool name to set of local neuron indices.
                Pool names: 'fqk_Q', 'fqk_K', 'fv', 'rqk_Q', 'rqk_K', 'rv',
                           'feature_know', 'restore_know'
        """
        self.suppressed = {k: set(v) for k, v in suppressed.items()}

    def _build_mask(self, pool_name: str, pool_size: int, device) -> Optional[torch.Tensor]:
        """
        Build a [N_pool] boolean mask tensor on device.
        True at positions to suppress. Returns None if no suppression needed.
        """
        indices = self.suppressed.get(pool_name)
        if not indices:
            return None
        mask = torch.zeros(pool_size, dtype=torch.bool, device=device)
        idx_tensor = torch.tensor(sorted(indices), dtype=torch.long, device=device)
        mask[idx_tensor] = True
        return mask

    def install(self, model):
        """Install hooks on the router's neuron_router module."""
        self.remove()  # clean up any prior hooks

        router = model.router.neuron_router
        device = router.neuron_emb.device

        # Pre-compute pool sizes from router boundaries
        pool_sizes = {
            'fqk_Q': router.n_feature_qk,
            'fqk_K': router.n_feature_qk,
            'fv':    router.n_feature_v,
            'rqk_Q': router.n_restore_qk,
            'rqk_K': router.n_restore_qk,
            'rv':    router.n_restore_v,
            'feature_know': router.n_feature_know,
            'restore_know': router.n_restore_know,
        }

        # Build all masks on-device once
        masks = {}
        for pool_name, size in pool_sizes.items():
            m = self._build_mask(pool_name, size, device)
            if m is not None:
                masks[pool_name] = m
        self._masks = masks

        neg_inf = torch.tensor(float('-inf'), device=device)

        # Hook get_all_logits — returns 6 tensors
        orig_get_all_logits = router.get_all_logits
        attn_names = ['fqk_Q', 'fqk_K', 'fv', 'rqk_Q', 'rqk_K', 'rv']

        # Snapshot which attention pools need masking (avoid dict lookup per call)
        attn_masks = [masks.get(name) for name in attn_names]
        has_any_attn_mask = any(m is not None for m in attn_masks)

        def hooked_get_all_logits(x):
            results = orig_get_all_logits(x)
            if not has_any_attn_mask:
                return results
            out = []
            for logits, mask in zip(results, attn_masks):
                if mask is not None:
                    # mask: [N], broadcast to [B, S, N] — pure element-wise, no sync
                    logits = torch.where(mask, neg_inf, logits)
                out.append(logits)
            return tuple(out)

        router.get_all_logits = hooked_get_all_logits

        # Hook get_knowledge_logits — returns 2 tensors
        orig_get_knowledge_logits = router.get_knowledge_logits
        fknow_mask = masks.get('feature_know')
        rknow_mask = masks.get('restore_know')
        has_any_know_mask = fknow_mask is not None or rknow_mask is not None

        def hooked_get_knowledge_logits(x):
            logits_f, logits_r = orig_get_knowledge_logits(x)
            if not has_any_know_mask:
                return logits_f, logits_r
            if fknow_mask is not None:
                logits_f = torch.where(fknow_mask, neg_inf, logits_f)
            if rknow_mask is not None:
                logits_r = torch.where(rknow_mask, neg_inf, logits_r)
            return logits_f, logits_r

        router.get_knowledge_logits = hooked_get_knowledge_logits

        # Store originals for removal
        self._orig_get_all_logits = orig_get_all_logits
        self._orig_get_knowledge_logits = orig_get_knowledge_logits
        self._router = router
        self.active = True

    def remove(self):
        """Restore original methods."""
        if self.active and hasattr(self, '_router'):
            self._router.get_all_logits = self._orig_get_all_logits
            self._router.get_knowledge_logits = self._orig_get_knowledge_logits
            self.active = False
            self._masks = {}


# ============================================================
# Core experiment class
# ============================================================

class NeuronSuppressionExperiment:
    """
    Full suppression experiment pipeline.

    1. collect_activation_frequencies — Phase 1
    2. identify_suppression_targets  — filter by threshold
    3. apply_suppression / remove    — Phase 2
    4. measure_target_frequency      — Phase 3
    5. run_full_experiment           — orchestrate everything
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hook_manager = SuppressionHookManager()
        self.model.eval()

    # ----------------------------------------------------------
    # Phase 1: Collect activation frequencies
    # ----------------------------------------------------------

    @torch.no_grad()
    def collect_activation_frequencies(
        self,
        prompt: str,
        target_token: str,
        n_runs: int = 100,
    ) -> Dict:
        """
        Run greedy generation n_runs times, recording which neurons
        are in the top-k selection at the first generated token position.

        Returns:
            Dict with keys:
              'match_count': how many runs produced target_token
              'total_runs': n_runs
              'match_rate': match_count / n_runs
              'target_token': target_token
              'target_token_id': int
              'neuron_frequencies': {pool_name: {neuron_idx: count}}
              'prompt': prompt
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)
        if len(target_id) == 0:
            raise ValueError(f"Target token '{target_token}' not in vocabulary")
        target_id = target_id[0]

        # Per-pool neuron activation counts
        freq = {pool: defaultdict(int) for pool in list(ATTENTION_POOLS.keys()) + list(KNOWLEDGE_POOLS.keys())}
        match_count = 0

        for run_idx in range(n_runs):
            logits, routing_infos = self.model(input_tensor, return_routing_info=True)

            # Check if greedy next token == target
            next_token_id = logits[0, -1, :].argmax().item()
            if next_token_id == target_id:
                match_count += 1

            # Extract active neurons at last position across all layers
            for layer_info in routing_infos:
                attn_info = layer_info.get('attention', {})
                know_info = layer_info.get('knowledge', {})

                # Attention pools
                for pool_name, weight_key in ATTENTION_POOLS.items():
                    weights = attn_info.get(weight_key)
                    if weights is None:
                        continue
                    # weights: [B, S, N_pool] — sparse (mostly 0)
                    w_last = weights[0, -1]  # [N_pool] at last position
                    active_idx = (w_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                    for idx in active_idx:
                        freq[pool_name][idx] += 1

                # Knowledge pools
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
    # Phase 1→2 bridge: Identify suppression targets
    # ----------------------------------------------------------

    def identify_suppression_targets(
        self,
        freq_results: List[Dict],
        threshold: float = 0.7,
    ) -> Dict[str, Set[int]]:
        """
        From multiple frequency results, find neurons that are active
        in ≥threshold fraction of runs across ALL capital queries.

        A neuron must meet the threshold in EVERY capital query
        to be considered "capital-related".

        Returns:
            Dict mapping pool_name → set of neuron indices to suppress.
        """
        if not freq_results:
            return {}

        # For each pool, find neurons meeting threshold in each query
        pool_names = list(ATTENTION_POOLS.keys()) + list(KNOWLEDGE_POOLS.keys())
        targets = {}

        for pool in pool_names:
            # Neurons meeting threshold per query
            per_query_sets = []
            for result in freq_results:
                n_runs = result['total_runs']
                min_count = int(n_runs * threshold)
                pool_freq = result['neuron_frequencies'].get(pool, {})
                meeting = {int(idx) for idx, count in pool_freq.items() if count >= min_count}
                per_query_sets.append(meeting)

            # Intersection: must be active in ALL capital queries
            if per_query_sets:
                common = per_query_sets[0]
                for s in per_query_sets[1:]:
                    common = common & s
                if common:
                    targets[pool] = common

        return targets

    def identify_suppression_targets_union(
        self,
        freq_results: List[Dict],
        threshold: float = 0.7,
    ) -> Dict[str, Set[int]]:
        """
        Union variant: neuron meets threshold in ANY capital query.
        More aggressive suppression.
        """
        pool_names = list(ATTENTION_POOLS.keys()) + list(KNOWLEDGE_POOLS.keys())
        targets = {}

        for pool in pool_names:
            union = set()
            for result in freq_results:
                n_runs = result['total_runs']
                min_count = int(n_runs * threshold)
                pool_freq = result['neuron_frequencies'].get(pool, {})
                meeting = {int(idx) for idx, count in pool_freq.items() if count >= min_count}
                union |= meeting
            if union:
                targets[pool] = union

        return targets

    # ----------------------------------------------------------
    # Phase 2: Apply / remove suppression
    # ----------------------------------------------------------

    def apply_suppression(self, suppressed_neurons: Dict[str, Set[int]]):
        """Install suppression hooks."""
        self.hook_manager.set_suppressed_neurons(suppressed_neurons)
        self.hook_manager.install(self.model)
        total = sum(len(v) for v in suppressed_neurons.values())
        print(f"  Suppression installed: {total} neurons across {len(suppressed_neurons)} pools")
        for pool, indices in sorted(suppressed_neurons.items()):
            print(f"    {pool}: {len(indices)} neurons — {sorted(indices)[:10]}{'...' if len(indices) > 10 else ''}")

    def remove_suppression(self):
        """Remove suppression hooks."""
        self.hook_manager.remove()

    # ----------------------------------------------------------
    # Phase 3: Measure target token frequency
    # ----------------------------------------------------------

    @torch.no_grad()
    def measure_target_frequency(
        self,
        prompt: str,
        target_token: str,
        n_runs: int = 100,
    ) -> Dict:
        """
        Run n_runs greedy forward passes. Count how many times
        the first generated token matches target_token.

        Also records top-5 token distribution for richer analysis.
        """
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
            next_logits = logits[0, -1, :]
            next_id = next_logits.argmax().item()
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
    # Full experiment orchestration
    # ----------------------------------------------------------

    def run_full_experiment(
        self,
        capital_queries: List[Dict] = None,
        control_queries: List[Dict] = None,
        n_runs: int = 100,
        threshold: float = 0.7,
        mode: str = 'intersection',
    ) -> Dict:
        """
        Run the complete suppression experiment.

        Args:
            capital_queries: list of {"prompt": ..., "target": ...}
            control_queries: list of {"prompt": ..., "target": ...}
            n_runs: number of runs per query
            threshold: activation frequency threshold (0-1)
            mode: 'intersection' (strict) or 'union' (aggressive)

        Returns:
            Full experiment results dict.
        """
        if capital_queries is None:
            capital_queries = DEFAULT_CAPITAL_QUERIES
        if control_queries is None:
            control_queries = DEFAULT_CONTROL_QUERIES

        results = {
            'config': {
                'n_runs': n_runs,
                'threshold': threshold,
                'mode': mode,
                'capital_queries': capital_queries,
                'control_queries': control_queries,
            },
            'phase1': {},
            'phase2': {},
            'phase3': {},
        }

        # ==============================
        # Phase 1: Activation frequency
        # ==============================
        print("=" * 70)
        print("PHASE 1: Collecting activation frequencies")
        print("=" * 70)

        freq_results = []
        for q in capital_queries:
            print(f"\n  Query: \"{q['prompt']}\" → target: '{q['target']}'")
            t0 = time.time()
            freq = self.collect_activation_frequencies(
                q['prompt'], q['target'], n_runs=n_runs
            )
            elapsed = time.time() - t0
            print(f"    Match rate: {freq['match_count']}/{freq['total_runs']} "
                  f"({freq['match_rate']:.0%})  [{elapsed:.1f}s]")

            # Summary per pool
            for pool in list(ATTENTION_POOLS.keys()) + list(KNOWLEDGE_POOLS.keys()):
                pool_freq = freq['neuron_frequencies'].get(pool, {})
                high_freq = {k: v for k, v in pool_freq.items() if v >= n_runs * threshold}
                if high_freq:
                    print(f"    {pool}: {len(high_freq)} neurons ≥{threshold:.0%} threshold")

            freq_results.append(freq)

        results['phase1']['capital_frequencies'] = freq_results

        # Also baseline control queries (no suppression)
        print(f"\n  --- Control queries (baseline) ---")
        control_baselines = []
        for q in control_queries:
            print(f"  Query: \"{q['prompt']}\" → target: '{q['target']}'")
            baseline = self.measure_target_frequency(
                q['prompt'], q['target'], n_runs=n_runs
            )
            print(f"    Match rate: {baseline['match_count']}/{baseline['total_runs']} "
                  f"({baseline['match_rate']:.0%})")
            control_baselines.append(baseline)

        results['phase1']['control_baselines'] = control_baselines

        # ==============================
        # Phase 2: Identify & suppress
        # ==============================
        print("\n" + "=" * 70)
        print("PHASE 2: Identifying suppression targets")
        print("=" * 70)

        if mode == 'intersection':
            suppressed = self.identify_suppression_targets(freq_results, threshold)
        else:
            suppressed = self.identify_suppression_targets_union(freq_results, threshold)

        total_suppressed = sum(len(v) for v in suppressed.values())
        print(f"\n  Mode: {mode}")
        print(f"  Threshold: {threshold:.0%}")
        print(f"  Total neurons to suppress: {total_suppressed}")

        results['phase2']['suppressed_neurons'] = {
            k: sorted(v) for k, v in suppressed.items()
        }
        results['phase2']['total_suppressed'] = total_suppressed

        if total_suppressed == 0:
            print("\n  WARNING: No neurons met the threshold! "
                  "Try lowering --threshold or using --mode union")
            results['phase3']['note'] = 'no neurons to suppress'
            return results

        self.apply_suppression(suppressed)

        # ==============================
        # Phase 3: Measure post-suppression
        # ==============================
        print("\n" + "=" * 70)
        print("PHASE 3: Measuring post-suppression effect")
        print("=" * 70)

        # Capital queries — should show reduced target hit rate
        print("\n  --- Capital queries (suppressed) ---")
        capital_post = []
        for q in capital_queries:
            print(f"  Query: \"{q['prompt']}\" → target: '{q['target']}'")
            post = self.measure_target_frequency(
                q['prompt'], q['target'], n_runs=n_runs
            )
            print(f"    Match rate: {post['match_count']}/{post['total_runs']} "
                  f"({post['match_rate']:.0%})")
            if post['top5']:
                top_tok, top_cnt, top_pct = post['top5'][0]
                print(f"    Most frequent token: '{top_tok}' ({top_pct:.0%})")
            capital_post.append(post)

        results['phase3']['capital_post_suppression'] = capital_post

        # Control queries — should show minimal impact
        print(f"\n  --- Control queries (suppressed) ---")
        control_post = []
        for q in control_queries:
            print(f"  Query: \"{q['prompt']}\" → target: '{q['target']}'")
            post = self.measure_target_frequency(
                q['prompt'], q['target'], n_runs=n_runs
            )
            print(f"    Match rate: {post['match_count']}/{post['total_runs']} "
                  f"({post['match_rate']:.0%})")
            control_post.append(post)

        results['phase3']['control_post_suppression'] = control_post

        # Clean up
        self.remove_suppression()

        # ==============================
        # Selectivity metrics
        # ==============================
        selectivity = self._compute_selectivity(
            freq_results, capital_post, control_baselines, control_post
        )
        results['selectivity'] = selectivity

        # ==============================
        # Summary table
        # ==============================
        self._print_summary(results)

        return results

    def _compute_selectivity(
        self,
        target_pre: List[Dict],
        target_post: List[Dict],
        control_pre: List[Dict],
        control_post: List[Dict],
    ) -> Dict:
        """
        Compute selectivity metrics for domain-specific suppression.

        - target_drop: avg accuracy drop in suppression-target queries
        - control_drop: avg accuracy change in control queries
        - selectivity_index: target_drop - control_drop
          (positive = selective; >0.1 = strong evidence of domain specificity)
        """
        target_drops = []
        for pre, post in zip(target_pre, target_post):
            target_drops.append(pre['match_rate'] - post['match_rate'])

        control_drops = []
        for pre, post in zip(control_pre, control_post):
            control_drops.append(pre['match_rate'] - post['match_rate'])

        avg_target = float(np.mean(target_drops)) if target_drops else 0.0
        avg_control = float(np.mean(control_drops)) if control_drops else 0.0
        selectivity = avg_target - avg_control

        return {
            'target_drops': target_drops,
            'control_drops': control_drops,
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
        """Print a clean comparison table."""
        print("\n" + "=" * 70)
        print("SUMMARY: Pseudo-Neuron Suppression Results")
        print("=" * 70)

        config = results['config']
        phase2 = results['phase2']
        print(f"  Threshold: {config['threshold']:.0%} | "
              f"Mode: {config['mode']} | "
              f"Runs: {config['n_runs']}")
        print(f"  Suppressed: {phase2['total_suppressed']} neurons")
        for pool, indices in sorted(phase2['suppressed_neurons'].items()):
            print(f"    {pool}: {len(indices)}")

        # Table header
        print("\n" + "-" * 90)
        print(f"  {'Query':<40s} {'Target':<8s} {'Pre':>7s} {'Post':>7s} {'Delta':>7s}")
        print("-" * 90)

        # Capital queries
        cap_freqs = results['phase1']['capital_frequencies']
        cap_posts = results['phase3'].get('capital_post_suppression', [])
        for freq, post in zip(cap_freqs, cap_posts):
            pre_rate = freq['match_rate']
            post_rate = post['match_rate']
            delta = post_rate - pre_rate
            prompt_short = freq['prompt'][:38]
            print(f"  {prompt_short:<40s} {freq['target_token']:<8s} "
                  f"{pre_rate:>6.0%}  {post_rate:>6.0%}  {delta:>+6.0%}")

        print("-" * 90)

        # Control queries
        ctrl_bases = results['phase1'].get('control_baselines', [])
        ctrl_posts = results['phase3'].get('control_post_suppression', [])
        for base, post in zip(ctrl_bases, ctrl_posts):
            pre_rate = base['match_rate']
            post_rate = post['match_rate']
            delta = post_rate - pre_rate
            prompt_short = base['prompt'][:38]
            print(f"  {prompt_short:<40s} {base['target_token']:<8s} "
                  f"{pre_rate:>6.0%}  {post_rate:>6.0%}  {delta:>+6.0%}  (control)")

        print("-" * 90)

        # Selectivity metrics
        sel = results.get('selectivity')
        if sel:
            print(f"\n  SELECTIVITY METRICS:")
            print(f"    Avg target drop:    {sel['avg_target_drop']:>+.1%}")
            print(f"    Avg control drop:   {sel['avg_control_drop']:>+.1%}")
            print(f"    Selectivity index:  {sel['selectivity_index']:>+.1%}")
            print(f"    Verdict: {sel['interpretation']}")

        # Post-suppression top tokens for capital queries
        if cap_posts:
            print("\n  Post-suppression top-5 tokens (capital queries):")
            for post in cap_posts:
                tokens_str = ", ".join(
                    f"'{tok}' ({pct:.0%})" for tok, cnt, pct in post['top5']
                )
                print(f"    \"{post['prompt']}\" → {tokens_str}")


# ============================================================
# Serialization helper
# ============================================================

def make_serializable(obj):
    """Convert sets and numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return obj


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='DAWN Pseudo-Neuron Suppression Experiment',
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
    parser.add_argument('--mode', type=str, default='intersection',
                        choices=['intersection', 'union'],
                        help='intersection = neuron must be active in ALL capital queries; '
                             'union = active in ANY (default: intersection)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results JSON')
    parser.add_argument('--queries', type=str, default=None,
                        help='Path to custom queries JSON file')
    parser.add_argument('--preset', type=str, default=None,
                        choices=list(QUERY_PRESETS.keys()),
                        help='Use a built-in query preset '
                             f'({", ".join(QUERY_PRESETS.keys())})')

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, tokenizer, config = load_model(args.checkpoint, device=args.device)
    print(f"  Model version: {model.__version__}")
    print(f"  Vocab size: {model.vocab_size}")

    # Load queries: --preset > --queries > defaults
    capital_queries = None
    control_queries = None
    if args.preset:
        preset = QUERY_PRESETS[args.preset]
        capital_queries = preset['target_queries']
        control_queries = preset['control_queries']
        print(f"  Preset: {args.preset} — {preset['description']}")
    elif args.queries:
        with open(args.queries) as f:
            qdata = json.load(f)
        capital_queries = qdata.get('capital', DEFAULT_CAPITAL_QUERIES)
        control_queries = qdata.get('control', DEFAULT_CONTROL_QUERIES)

    # Run experiment
    experiment = NeuronSuppressionExperiment(model, tokenizer, device=args.device)
    results = experiment.run_full_experiment(
        capital_queries=capital_queries,
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
        filename = f"suppression_{ckpt_name}_t{args.threshold}_n{args.n_runs}_{args.mode}.json"
        output_path = output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(make_serializable(results), f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
