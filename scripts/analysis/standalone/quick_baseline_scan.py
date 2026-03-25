#!/usr/bin/env python3
"""Quick baseline logit scan — single forward → top-5, no generation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import jax
import jax.numpy as jnp

from scripts.analysis.utils_jax import load_model_jax, create_model_from_config

QUERIES = [
    ("the capital of france is",    "paris"),
    ("the capital of germany is",   "berlin"),
    ("the capital of italy is",     "rome"),
    ("the capital of spain is",     "madrid"),
    ("the capital of china is",     "beijing"),
    ("the eiffel tower is in",      "paris"),
    ("the colosseum is in",         "rome"),
    ("mount fuji is in",            "japan"),
    ("shakespeare was born in",     "england"),
    ("the sun rises in the",        "east"),
    ("ice is made of",              "water"),
    ("the earth orbits the",        "sun"),
    ("humans have two",             "eyes"),
    ("fire is",                     "hot"),
]


def main():
    parser = argparse.ArgumentParser(description="Quick baseline logit scan")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Loading: {args.checkpoint}")

    model_cls, params, tokenizer, config = load_model_jax(args.checkpoint)
    model = create_model_from_config(config)

    @jax.jit
    def forward(input_ids):
        result = model.apply(
            params, input_ids,
            deterministic=True,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )
        return result["logits"]

    # Warmup
    print("JIT warmup...", end=" ", flush=True)
    _ = forward(jnp.array([[0]]))
    _.block_until_ready()
    print("done\n")

    hits = []
    misses = []

    print("=" * 80)
    print(f"  {'Prompt':<35s} {'Target':<10s} {'Rank':>4s}  {'Prob':>7s}  Top-{args.top_k}")
    print("=" * 80)

    for prompt, target in QUERIES:
        input_ids = [101] + tokenizer.encode(prompt, add_special_tokens=False)
        input_arr = jnp.array([input_ids])

        logits = forward(input_arr)
        last_logits = np.array(logits[0, -1, :]).astype(np.float64)
        last_logits -= last_logits.max()
        probs = np.exp(last_logits)
        probs /= probs.sum()

        top_indices = np.argsort(probs)[::-1][:args.top_k]
        top_tokens = [(tokenizer.decode([i]).strip(), float(probs[i])) for i in top_indices]

        # Find target rank
        target_lower = target.strip().lower()
        target_rank = None
        target_prob = 0.0
        for rank, (tok, prob) in enumerate(top_tokens, 1):
            if tok.lower() == target_lower:
                target_rank = rank
                target_prob = prob
                break

        top_str = ", ".join(f"{tok}({prob:.1%})" for tok, prob in top_tokens)

        if target_rank:
            marker = f"#{target_rank}"
            print(f"  {prompt:<35s} {target:<10s} {marker:>4s}  {target_prob:>6.1%}  {top_str}")
            hits.append((prompt, target, target_rank, target_prob, top_tokens))
        else:
            print(f"  {prompt:<35s} {target:<10s}    -      -  {top_str}")
            misses.append((prompt, target, top_tokens))

    print("=" * 80)
    print(f"\n  Hits (target in top-{args.top_k}): {len(hits)}/{len(QUERIES)}")
    for prompt, target, rank, prob, _ in hits:
        print(f"    #{rank} {prob:>5.1%}  \"{prompt}\" -> {target}")

    if misses:
        print(f"\n  Misses: {len(misses)}/{len(QUERIES)}")
        for prompt, target, top in misses:
            print(f"    \"{prompt}\" -> {target}  (top: {top[0][0]})")


if __name__ == "__main__":
    main()
