"""
첫 배치 데이터 분석 스크립트

학습 데이터의 실제 길이, 패딩 비율, 마스킹 패턴을 확인합니다.
train.py의 데이터 로더를 그대로 사용합니다.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer

# Import from train script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_module",
    PROJECT_ROOT / "scripts" / "train.py"
)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)


def analyze_batch(batch, tokenizer, batch_idx=0):
    """배치 데이터 분석"""

    print("\n" + "="*70)
    print(f"BATCH {batch_idx} ANALYSIS")
    print("="*70)

    input_ids = batch['input_ids']
    B, S = input_ids.shape

    print(f"\nBatch shape: {input_ids.shape} (batch_size={B}, seq_len={S})")

    # 전체 통계
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print("="*70)

    total_tokens = B * S
    print(f"Total tokens in batch: {total_tokens:,}")

    # 각 시퀀스 분석
    print(f"\n{'='*70}")
    print("PER-SEQUENCE ANALYSIS")
    print("="*70)

    seq_lengths = []
    padding_counts = []

    for i in range(min(10, B)):  # 처음 10개 시퀀스만
        seq = input_ids[i]

        # Padding 찾기 (tokenizer.pad_token_id)
        is_padding = (seq == tokenizer.pad_token_id)
        non_padding = (~is_padding).sum().item()
        padding = is_padding.sum().item()

        seq_lengths.append(non_padding)
        padding_counts.append(padding)

        # 특수 토큰 찾기
        is_cls = (seq == tokenizer.cls_token_id).sum().item()
        is_sep = (seq == tokenizer.sep_token_id).sum().item()
        is_mask = (seq == tokenizer.mask_token_id).sum().item() if hasattr(tokenizer, 'mask_token_id') else 0

        print(f"\nSequence {i+1}:")
        print(f"  Total length:     {S}")
        print(f"  Actual tokens:    {non_padding} ({non_padding/S*100:.1f}%)")
        print(f"  Padding:          {padding} ({padding/S*100:.1f}%)")
        print(f"  [CLS] tokens:     {is_cls}")
        print(f"  [SEP] tokens:     {is_sep}")
        print(f"  [MASK] tokens:    {is_mask}")

        # 첫 30개 토큰 출력
        print(f"  First 30 tokens:  {seq[:30].tolist()}")

        # 실제 텍스트 디코딩 (padding 제외)
        if non_padding > 0:
            actual_tokens = seq[~is_padding][:50]  # 처음 50개만
            try:
                decoded = tokenizer.decode(actual_tokens, skip_special_tokens=False)
                print(f"  Decoded (first 50): {decoded[:150]}...")
            except:
                pass

    # 전체 배치 통계
    print(f"\n{'='*70}")
    print("BATCH STATISTICS")
    print("="*70)

    all_seq_lengths = []
    all_padding_counts = []

    for i in range(B):
        seq = input_ids[i]
        is_padding = (seq == tokenizer.pad_token_id)
        non_padding = (~is_padding).sum().item()
        padding = is_padding.sum().item()

        all_seq_lengths.append(non_padding)
        all_padding_counts.append(padding)

    avg_length = sum(all_seq_lengths) / len(all_seq_lengths)
    avg_padding = sum(all_padding_counts) / len(all_padding_counts)

    print(f"\nSequence lengths:")
    print(f"  Min:     {min(all_seq_lengths)}")
    print(f"  Max:     {max(all_seq_lengths)}")
    print(f"  Mean:    {avg_length:.1f}")
    print(f"  Median:  {sorted(all_seq_lengths)[len(all_seq_lengths)//2]}")

    print(f"\nPadding statistics:")
    print(f"  Avg padding per sequence: {avg_padding:.1f} tokens")
    print(f"  Padding ratio:            {avg_padding/S*100:.1f}%")
    print(f"  Actual content:           {avg_length/S*100:.1f}%")

    # 분포
    print(f"\nLength distribution:")
    bins = [
        (0, 20, "0-20"),
        (20, 40, "20-40"),
        (40, 60, "40-60"),
        (60, 80, "60-80"),
        (80, 100, "80-100"),
        (100, S, f"100-{S}")
    ]

    for min_l, max_l, label in bins:
        count = sum(1 for l in all_seq_lengths if min_l <= l < max_l)
        pct = count / B * 100
        print(f"  {label:10s}: {count:4d} sequences ({pct:5.1f}%)")

    # 권장사항
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print("="*70)

    p75_length = sorted(all_seq_lengths)[int(len(all_seq_lengths) * 0.75)]
    p90_length = sorted(all_seq_lengths)[int(len(all_seq_lengths) * 0.90)]

    print(f"\nCurrent max_seq_len: {S}")
    print(f"P75 actual length:   {p75_length} (covers 75% of sequences)")
    print(f"P90 actual length:   {p90_length} (covers 90% of sequences)")

    if avg_padding / S > 0.5:
        print(f"\n⚠️  WARNING: High padding ratio ({avg_padding/S*100:.1f}%)")
        print(f"   Recommendation: Reduce max_seq_len to ~{p75_length}-{p90_length}")
        print(f"   This will reduce padding from {avg_padding/S*100:.1f}% to ~{20-30}%")
    elif avg_padding / S > 0.3:
        print(f"\n⚠️  Moderate padding ratio ({avg_padding/S*100:.1f}%)")
        print(f"   Consider reducing max_seq_len to ~{p90_length} for efficiency")
    else:
        print(f"\n✅ Padding ratio is acceptable ({avg_padding/S*100:.1f}%)")


def analyze_labels(input_ids, labels, tokenizer):
    """라벨 분석 (MLM 마스킹 후)"""

    print("\n" + "="*70)
    print("LABEL ANALYSIS (MLM MASKING)")
    print("="*70)

    B, S = input_ids.shape

    # 유효한 라벨 카운트
    valid_mask = (labels != -100)
    total_tokens = labels.numel()
    valid_tokens = valid_mask.sum().item()
    masked_tokens = total_tokens - valid_tokens

    print(f"\nTotal tokens:   {total_tokens:,}")
    print(f"Valid tokens (for training):    {valid_tokens:,} ({valid_tokens/total_tokens*100:.1f}%)")
    print(f"Ignored tokens (labeled -100):  {masked_tokens:,} ({masked_tokens/total_tokens*100:.1f}%)")

    valid_ratio = valid_tokens / total_tokens

    # MLM에서는 15% 선택 → 패딩/special tokens 제외 후 약 10-20%가 정상
    if valid_ratio < 0.05:
        print(f"\n🚨 CRITICAL: Too few tokens selected ({valid_ratio*100:.1f}%)")
        print(f"   Expected: 10-20% for MLM (15% of non-padding tokens)")
    elif valid_ratio > 0.25:
        print(f"\n⚠️  WARNING: Too many tokens selected ({valid_ratio*100:.1f}%)")
        print(f"   Expected: 10-20% for MLM (15% of non-padding tokens)")
    else:
        print(f"\n✅ MLM masking ratio is correct!")
        print(f"   {valid_ratio*100:.1f}% ≈ 15% of (total - padding - special tokens)")
        print(f"   85-90% labeled as -100 (ignored) is NORMAL for MLM")

    # 첫 시퀀스 상세 분석
    print(f"\n{'='*70}")
    print("FIRST SEQUENCE DETAIL")
    print("="*70)

    seq_input = input_ids[0]
    seq_labels = labels[0]
    seq_valid = valid_mask[0]

    print(f"\nInput  [:30]: {seq_input[:30].tolist()}")
    print(f"Labels [:30]: {seq_labels[:30].tolist()}")
    print(f"Valid  [:30]: {seq_valid[:30].tolist()}")

    # 라벨이 있는 위치 확인
    valid_positions = seq_valid.nonzero().squeeze()
    print(f"\nValid label positions (first 20): {valid_positions[:20].tolist()}")

    # [MASK] 토큰 위치 확인
    if hasattr(tokenizer, 'mask_token_id'):
        mask_positions = (seq_input == tokenizer.mask_token_id).nonzero().squeeze()
        print(f"[MASK] token positions (first 20): {mask_positions[:20].tolist()}")

    # 예상 토큰 vs 실제 라벨 비교
    print(f"\nFirst 10 valid tokens:")
    valid_idx = valid_positions[:10]
    for i, idx in enumerate(valid_idx):
        idx_item = idx.item() if idx.dim() > 0 else idx
        input_token = seq_input[idx_item].item()
        label_token = seq_labels[idx_item].item()
        print(f"  Pos {idx_item:3d}: input={input_token:5d}, label={label_token:5d}")

    # 전체 배치 80/10/10 분포 분석
    print(f"\n{'='*70}")
    print("80/10/10 DISTRIBUTION ANALYSIS (Full Batch)")
    print("="*70)

    # 전체 배치에서 유효한 토큰들 분석
    all_valid_mask = labels != -100
    num_valid = all_valid_mask.sum().item()

    # [MASK] 토큰 카운트
    mask_token_id = tokenizer.mask_token_id
    is_mask = (input_ids == mask_token_id) & all_valid_mask
    num_mask = is_mask.sum().item()

    # Random 토큰 카운트 (input != label, input != [MASK])
    is_random = (input_ids != labels) & ~is_mask & all_valid_mask
    num_random = is_random.sum().item()

    # Keep original 카운트 (input == label)
    is_keep = (input_ids == labels) & all_valid_mask
    num_keep = is_keep.sum().item()

    print(f"\nTotal valid tokens: {num_valid:,}")
    print(f"  [MASK] tokens:     {num_mask:5d} ({num_mask/num_valid*100:.1f}%)")
    print(f"  Random tokens:     {num_random:5d} ({num_random/num_valid*100:.1f}%)")
    print(f"  Keep original:     {num_keep:5d} ({num_keep/num_valid*100:.1f}%)")

    print(f"\nExpected distribution (BERT standard):")
    print(f"  [MASK]: 80%, Random: 10%, Keep: 10%")

    # 검증
    mask_ratio = num_mask / num_valid if num_valid > 0 else 0
    random_ratio = num_random / num_valid if num_valid > 0 else 0
    keep_ratio = num_keep / num_valid if num_valid > 0 else 0

    if 0.75 <= mask_ratio <= 0.85 and 0.05 <= random_ratio <= 0.15 and 0.05 <= keep_ratio <= 0.15:
        print(f"\n✅ Distribution matches BERT standard (80/10/10)!")
    else:
        print(f"\n⚠️  Distribution deviates from expected 80/10/10")
        if mask_ratio < 0.75:
            print(f"   - Too few [MASK] tokens ({mask_ratio*100:.1f}% < 75%)")
        if random_ratio < 0.05 or random_ratio > 0.15:
            print(f"   - Random ratio out of range ({random_ratio*100:.1f}% not in 5-15%)")


def main():
    print("="*70)
    print("FIRST BATCH DATA ANALYZER")
    print("="*70)

    # 데이터 로더 생성
    print("\nLoading data...")
    try:
        train_loader, val_loader, tokenizer = train_module.load_cached_data(
            tokenizer_path="bert-base-uncased",
            max_length=128,  # 현재 설정
            batch_size=128   # 현재 설정
        )

        print(f"✅ Data loaded successfully!")
        print(f"   Tokenizer: bert-base-uncased")
        print(f"   pad_token_id: {tokenizer.pad_token_id}")
        print(f"   cls_token_id: {tokenizer.cls_token_id}")
        print(f"   sep_token_id: {tokenizer.sep_token_id}")
        if hasattr(tokenizer, 'mask_token_id'):
            print(f"   mask_token_id: {tokenizer.mask_token_id}")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("\nMake sure WikiText cache exists at:")
        print("  /content/drive/MyDrive/dawn_v4/cache/train/wikitext_5to1_texts.pkl")
        return

    # 첫 배치 가져오기 (MLM 마스킹 전)
    print("\nFetching first batch...")
    first_batch = next(iter(train_loader))

    # 배치 분석
    analyze_batch(first_batch, tokenizer, batch_idx=1)

    # MLM 마스킹 적용
    print("\n" + "="*70)
    print("APPLYING MLM MASKING")
    print("="*70)

    input_ids = first_batch['input_ids']
    input_ids_masked, labels = train_module.apply_mlm_masking(
        input_ids.clone(),
        tokenizer,
        train_module.MLM_CONFIG
    )

    # 라벨 분석 (FIXED: Use masked input_ids, not original)
    analyze_labels(input_ids_masked, labels, tokenizer)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
