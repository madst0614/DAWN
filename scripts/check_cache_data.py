"""
WikiText 캐시 데이터 분석 스크립트

실제 캐시된 WikiText 데이터의 길이, 분포, 샘플을 확인합니다.
"""

import pickle
import os
from pathlib import Path
import sys

# 가능한 캐시 경로들
POSSIBLE_CACHE_PATHS = [
    "/content/drive/MyDrive/dawn_v4/cache",
    "/home/user/.cache/dawn_v4",
    "/home/user/sprout/cache",
]


def find_cache_file(split="train", dataset="wikitext"):
    """캐시 파일 찾기"""
    filename = f"{dataset}_5to1_texts.pkl"

    for base_path in POSSIBLE_CACHE_PATHS:
        cache_path = os.path.join(base_path, split, filename)
        if os.path.exists(cache_path):
            return cache_path

    return None


def analyze_texts(texts, max_samples=20, max_analysis=5000):
    """텍스트 데이터 분석"""

    print("="*70)
    print("TEXT DATA ANALYSIS")
    print("="*70)

    total_texts = len(texts)
    print(f"\nTotal texts: {total_texts:,}")

    # 샘플 출력
    print(f"\n{'='*70}")
    print(f"SAMPLE TEXTS (first {max_samples})")
    print("="*70)

    for i, text in enumerate(texts[:max_samples]):
        words = text.split()
        chars = len(text)
        print(f"\n[{i+1}] Words: {len(words)}, Chars: {chars}")
        print(f"    Preview: {text[:200]}...")
        if len(text) > 200:
            print(f"    ... (truncated, total: {chars} chars)")

    # 길이 통계 (전체 또는 샘플)
    analysis_count = min(max_analysis, total_texts)
    print(f"\n{'='*70}")
    print(f"LENGTH STATISTICS (analyzing first {analysis_count:,} texts)")
    print("="*70)

    # Word counts
    word_lengths = [len(text.split()) for text in texts[:analysis_count]]
    char_lengths = [len(text) for text in texts[:analysis_count]]

    print(f"\nWord Count Statistics:")
    print(f"  Min:    {min(word_lengths):,} words")
    print(f"  Max:    {max(word_lengths):,} words")
    print(f"  Mean:   {sum(word_lengths)/len(word_lengths):.1f} words")

    # Percentiles for words
    sorted_words = sorted(word_lengths)
    p25_idx = len(sorted_words) // 4
    p50_idx = len(sorted_words) // 2
    p75_idx = 3 * len(sorted_words) // 4
    p90_idx = int(len(sorted_words) * 0.9)
    p95_idx = int(len(sorted_words) * 0.95)

    print(f"  P25:    {sorted_words[p25_idx]:,} words")
    print(f"  P50:    {sorted_words[p50_idx]:,} words (median)")
    print(f"  P75:    {sorted_words[p75_idx]:,} words")
    print(f"  P90:    {sorted_words[p90_idx]:,} words")
    print(f"  P95:    {sorted_words[p95_idx]:,} words")

    print(f"\nCharacter Count Statistics:")
    print(f"  Min:    {min(char_lengths):,} chars")
    print(f"  Max:    {max(char_lengths):,} chars")
    print(f"  Mean:   {sum(char_lengths)/len(char_lengths):.1f} chars")

    # Distribution
    print(f"\n{'='*70}")
    print("WORD COUNT DISTRIBUTION")
    print("="*70)

    bins = [
        (0, 10, "< 10 words"),
        (10, 20, "10-20 words"),
        (20, 30, "20-30 words"),
        (30, 50, "30-50 words"),
        (50, 100, "50-100 words"),
        (100, 200, "100-200 words"),
        (200, float('inf'), "> 200 words")
    ]

    for min_w, max_w, label in bins:
        count = sum(1 for l in word_lengths if min_w <= l < max_w)
        pct = count / len(word_lengths) * 100
        print(f"  {label:20s}: {count:6,} ({pct:5.1f}%)")

    # Token estimation (rough: 1 word ≈ 1.3 tokens for English)
    print(f"\n{'='*70}")
    print("ESTIMATED TOKEN COUNTS")
    print("="*70)
    print("(Assuming ~1.3 tokens per word for English)")

    estimated_tokens = [int(w * 1.3) for w in word_lengths]
    sorted_tokens = sorted(estimated_tokens)

    print(f"\nEstimated Token Statistics:")
    print(f"  Mean:   {sum(estimated_tokens)/len(estimated_tokens):.1f} tokens")
    print(f"  P50:    {sorted_tokens[p50_idx]:,} tokens (median)")
    print(f"  P75:    {sorted_tokens[p75_idx]:,} tokens")
    print(f"  P90:    {sorted_tokens[p90_idx]:,} tokens")
    print(f"  P95:    {sorted_tokens[p95_idx]:,} tokens")

    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS FOR max_seq_len")
    print("="*70)

    p50_tokens = sorted_tokens[p50_idx]
    p75_tokens = sorted_tokens[p75_idx]
    p90_tokens = sorted_tokens[p90_idx]
    p95_tokens = sorted_tokens[p95_idx]

    print(f"\nBased on token distribution:")
    print(f"  Conservative (covers 50%): max_seq_len = {p50_tokens}")
    print(f"  Balanced (covers 75%):     max_seq_len = {p75_tokens}")
    print(f"  Generous (covers 90%):     max_seq_len = {p90_tokens}")
    print(f"  Very generous (covers 95%): max_seq_len = {p95_tokens}")

    print(f"\nPadding analysis for max_seq_len=128:")
    over_128 = sum(1 for t in estimated_tokens if t > 128)
    under_128 = len(estimated_tokens) - over_128
    avg_padding = sum(max(0, 128 - t) for t in estimated_tokens) / len(estimated_tokens)
    print(f"  Texts fitting in 128 tokens: {under_128:,} ({under_128/len(estimated_tokens)*100:.1f}%)")
    print(f"  Texts truncated:             {over_128:,} ({over_128/len(estimated_tokens)*100:.1f}%)")
    print(f"  Average padding needed:      {avg_padding:.1f} tokens")
    print(f"  Padding ratio:               {avg_padding/128*100:.1f}%")

    if avg_padding / 128 > 0.5:
        print(f"\n  ⚠️  WARNING: High padding ratio (>{50}%)!")
        print(f"     Consider reducing max_seq_len")


def main():
    print("\n" + "="*70)
    print("WIKITEXT CACHE DATA CHECKER")
    print("="*70)

    # Find train cache
    train_path = find_cache_file(split="train", dataset="wikitext")
    val_path = find_cache_file(split="validation", dataset="wikitext")

    if not train_path and not val_path:
        print("\n❌ No cache files found!")
        print("\nSearched in:")
        for path in POSSIBLE_CACHE_PATHS:
            print(f"  - {path}/{{train,validation}}/wikitext_5to1_texts.pkl")
        print("\nPlease check if cache files exist or update POSSIBLE_CACHE_PATHS")
        return

    # Analyze train data
    if train_path:
        print(f"\n✅ Found train cache: {train_path}")

        try:
            with open(train_path, 'rb') as f:
                train_texts = pickle.load(f)

            print(f"\n{'#'*70}")
            print("TRAIN SET ANALYSIS")
            print("#"*70)
            analyze_texts(train_texts, max_samples=20, max_analysis=5000)

        except Exception as e:
            print(f"\n❌ Error loading train cache: {e}")

    # Analyze validation data
    if val_path:
        print(f"\n✅ Found validation cache: {val_path}")

        try:
            with open(val_path, 'rb') as f:
                val_texts = pickle.load(f)

            print(f"\n{'#'*70}")
            print("VALIDATION SET ANALYSIS")
            print("#"*70)
            analyze_texts(val_texts, max_samples=10, max_analysis=2000)

        except Exception as e:
            print(f"\n❌ Error loading validation cache: {e}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
