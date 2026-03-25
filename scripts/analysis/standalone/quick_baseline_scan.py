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
    # 천문/물리
    ("light travels at the speed of",           "light"),
    ("gravity pulls objects",                   "down"),
    ("the sun is a",                            "star"),
    ("atoms are made of",                       "protons"),
    ("water freezes at",                        "zero"),
    ("sound travels through",                   "air"),
    ("the force of gravity is measured in",     "newtons"),
    # 생물/신체
    ("the brain is part of the",                "nervous"),
    ("the heart is a",                          "muscle"),
    ("bones are part of the",                   "skeleton"),
    ("the lungs are used for",                  "breathing"),
    ("plants need sunlight to",                 "grow"),
    ("photosynthesis produces",                 "oxygen"),
    ("humans have five",                        "senses"),
    ("blood is pumped by the",                  "heart"),
    ("the largest organ in the body is the",    "skin"),
    # 지리/역사
    ("the amazon is the longest",               "river"),
    ("the nile flows through",                  "egypt"),
    ("mount everest is the",                    "highest"),
    ("the atlantic is an",                      "ocean"),
    ("australia is a",                          "continent"),
    ("the sahara is a",                         "desert"),
    ("the titanic sank in",                     "the"),
    ("world war two ended in",                  "1945"),
    ("the french revolution began in",          "1789"),
    ("columbus sailed in",                      "1492"),
    # 언어/문화
    ("shakespeare wrote",                       "hamlet"),
    ("the bible is a",                          "book"),
    ("english is spoken in",                    "the"),
    ("the olympics are held every",             "four"),
    ("christmas is celebrated in",              "december"),
    # 음식/일상
    ("coffee is made from",                     "beans"),
    ("bread is made from",                      "wheat"),
    ("wine is made from",                       "grapes"),
    ("cheese is made from",                     "milk"),
    ("butter is made from",                     "cream"),
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
