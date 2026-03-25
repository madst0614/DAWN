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
# These replace _router_attn_forward / _router_know_forward
# with versions that apply -inf masks to logits before softmax.
# Will be added in next section.
# ============================================================


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


# Placeholder class — will be filled in next step
class NeuronSuppressionExperimentJAX:
    def __init__(self, model, params, config, tokenizer):
        self.model = model
        self.params = params
        self.config = config
        self.tokenizer = tokenizer

    def run_full_experiment(self, **kwargs):
        raise NotImplementedError("Next step: add experiment logic")


if __name__ == '__main__':
    main()
