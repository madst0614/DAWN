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
    ("the sky is",                              "blue"),
    ("fire is",                                 "hot"),
    ("ice is",                                  "cold"),
    ("the earth is a",                          "planet"),
    ("the sun is a",                            "star"),
    ("snow is",                                 "white"),
    ("grass is",                                "green"),
    ("the moon is a",                           "satellite"),
    ("the sun is in the",                       "sky"),
    ("london is in",                            "england"),
    ("paris is in",                             "france"),
    ("tokyo is in",                             "japan"),
    ("rome is in",                              "italy"),
    ("berlin is in",                            "germany"),
    ("one plus one is",                         "two"),
    ("humans breathe",                          "air"),
    ("birds can",                               "fly"),
    ("fish live in",                            "water"),
    ("cats and dogs are",                       "animals"),
    ("the opposite of hot is",                  "cold"),
    ("the earth revolves around the",           "sun"),
    ("the moon orbits around the",              "earth"),
    ("the sky at night is",                     "dark"),
    ("sugar tastes",                            "sweet"),
    ("lemons taste",                            "sour"),
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

    all_results = []

    print(f"Scanning {len(QUERIES)} queries...\n")

    for qi, (prompt, target) in enumerate(QUERIES, 1):
        input_ids = [101] + tokenizer.encode(prompt, add_special_tokens=False)
        input_arr = jnp.array([input_ids])

        logits = forward(input_arr)
        last_logits = np.array(logits[0, -1, :]).astype(np.float64)
        last_logits -= last_logits.max()
        probs = np.exp(last_logits)
        probs /= probs.sum()

        top_indices = np.argsort(probs)[::-1][:args.top_k]
        top_tokens = [(tokenizer.decode([i]).strip(), float(probs[i])) for i in top_indices]

        # Find target in full vocab (not just top-k)
        target_lower = target.strip().lower()
        target_rank = None
        target_prob = 0.0
        sorted_indices = np.argsort(probs)[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            tok = tokenizer.decode([idx]).strip().lower()
            if tok == target_lower:
                target_rank = rank
                target_prob = float(probs[idx])
                break

        all_results.append((prompt, target, target_rank, target_prob, top_tokens))
        print(f"  [{qi}/{len(QUERIES)}] done", end="\r", flush=True)

    print(" " * 40)  # clear progress line

    # Sort by target_prob descending
    all_results.sort(key=lambda x: x[3], reverse=True)

    print("=" * 100)
    print(f"  {'#':<3s} {'Prompt':<42s} {'Target':<10s} {'Rank':>5s} {'Prob':>8s}  Top-{args.top_k}")
    print("=" * 100)

    elite = []   # >= 90%
    hits = []    # top-k but < 90%
    misses = []
    for i, (prompt, target, rank, prob, top_tokens) in enumerate(all_results, 1):
        top_str = ", ".join(f"{tok}({p:.1%})" for tok, p in top_tokens)
        in_topk = rank is not None and rank <= args.top_k
        rank_str = f"#{rank}" if rank else "-"
        prob_str = f"{prob:.2%}" if rank else "-"
        if prob >= 0.90:
            marker = " *** 90%+ ***"
        elif in_topk:
            marker = " *"
        else:
            marker = ""
        print(f"  {i:<3d} {prompt:<42s} {target:<10s} {rank_str:>5s} {prob_str:>8s}  {top_str}{marker}")

        if prob >= 0.90:
            elite.append((prompt, target, rank, prob))
        elif in_topk:
            hits.append((prompt, target, rank, prob))
        else:
            misses.append((prompt, target, rank, prob, top_tokens))

    print("=" * 100)

    if elite:
        print(f"\n  >>> 90%+ QUERIES: {len(elite)}/{len(all_results)} <<<")
        print(f"  {'Prompt':<42s} {'Target':<10s} {'Rank':>5s} {'Prob':>8s}")
        print(f"  {'-'*42} {'-'*10} {'-'*5} {'-'*8}")
        for prompt, target, rank, prob in elite:
            print(f"  {prompt:<42s} {target:<10s} #{rank:>3d} {prob:>7.2%}")

    if hits:
        print(f"\n  === OTHER HITS (target in top-{args.top_k}): {len(hits)} ===")
        for prompt, target, rank, prob in hits:
            print(f"  {prompt:<42s} {target:<10s} #{rank:>3d} {prob:>7.2%}")

    if misses:
        print(f"\n  === MISSES: {len(misses)}/{len(all_results)} ===")
        for prompt, target, rank, prob, top in misses:
            rank_str = f"#{rank}" if rank else "N/A"
            prob_str = f"{prob:.2%}" if rank else "N/A"
            print(f"  {prompt:<42s} {target:<10s} {rank_str:>5s} {prob_str:>8s}  (top1: {top[0][0]})")


if __name__ == "__main__":
    main()
