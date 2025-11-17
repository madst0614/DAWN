"""
학습된 체크포인트 분석
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import argparse

from src.models.sprout_neuron_based import NeuronBasedLanguageModel


def analyze_neuron_usage(model, tokenizer, device, num_samples=100):
    """뉴런 사용 패턴 분석"""
    print("\n" + "="*70)
    print("뉴런 사용 패턴 분석")
    print("="*70)

    model.eval()

    # 테스트 문장들
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Python is a popular programming language for data science.",
        "The weather today is sunny and warm.",
        "Deep neural networks require large amounts of training data.",
    ] * 20  # 100개 샘플

    test_texts = test_texts[:num_samples]

    # 각 레이어별 뉴런 사용 추적
    n_layers = len(model.layers)
    d_ff = model.layers[0].ffn.d_ff

    layer_usage = {i: torch.zeros(d_ff, device=device) for i in range(n_layers)}

    with torch.no_grad():
        for text in test_texts:
            tokens = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=32,
                truncation=True
            )['input_ids'].to(device)

            # Forward pass로 라우터 점수 수집
            batch, seq = tokens.shape

            # Embedding
            token_emb = model.token_embedding(tokens)
            positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
            pos_emb = model.position_embedding(positions)
            x = token_emb + pos_emb

            # 각 레이어 통과하며 라우팅 기록
            for layer_idx, layer in enumerate(model.layers):
                # Attention
                x_norm = layer.norm1(x)
                attn_out, _ = layer.attention(x_norm, x_norm, x_norm)
                x = x + layer.dropout(attn_out)

                # FFN - 라우터 점수 계산
                x_norm = layer.norm2(x)
                x_flat = x_norm.view(-1, x_norm.shape[-1])

                router_scores = x_flat @ layer.ffn.router.W_router.T
                top_k = 768  # 분석용
                _, top_indices = torch.topk(router_scores, top_k, dim=-1)

                # 사용된 뉴런 기록
                for indices in top_indices:
                    layer_usage[layer_idx][indices] += 1

                # FFN 통과
                x_ffn = layer.ffn(x_norm, top_k=top_k)
                x = x + layer.dropout(x_ffn)

    # 분석 결과 출력
    print(f"\n분석한 샘플 수: {num_samples}")
    print(f"총 뉴런 수 (레이어당): {d_ff}")
    print(f"분석 시 활성화 뉴런 수: 768 (25%)")

    for layer_idx in range(n_layers):
        usage = layer_usage[layer_idx]
        total_positions = num_samples * 32  # seq_len=32

        n_never = (usage == 0).sum().item()
        n_rare = ((usage > 0) & (usage < total_positions * 0.1)).sum().item()
        n_common = (usage >= total_positions * 0.1).sum().item()

        mean_usage = usage.mean().item()
        max_usage = usage.max().item()

        print(f"\n레이어 {layer_idx}:")
        print(f"  사용 안 됨: {n_never:4d} / {d_ff} ({n_never/d_ff*100:.1f}%)")
        print(f"  드물게 사용: {n_rare:4d} / {d_ff} ({n_rare/d_ff*100:.1f}%)")
        print(f"  자주 사용: {n_common:4d} / {d_ff} ({n_common/d_ff*100:.1f}%)")
        print(f"  평균 사용: {mean_usage:.1f}, 최대: {max_usage:.0f}")

        # 상위 10개 뉴런
        top_10_usage, top_10_idx = torch.topk(usage, 10)
        print(f"  상위 10 뉴런: {top_10_idx.tolist()[:5]}...")


def analyze_mlm_quality(model, tokenizer, device):
    """MLM 예측 품질 분석"""
    print("\n" + "="*70)
    print("MLM 예측 품질 분석")
    print("="*70)

    model.eval()

    # 테스트 케이스
    test_cases = [
        ("The [MASK] is shining brightly.", "sun"),
        ("I love to [MASK] books.", "read"),
        ("Python is a programming [MASK].", "language"),
        ("She went to the [MASK] yesterday.", "store"),
        ("The cat is [MASK] on the mat.", "sitting"),
    ]

    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        for masked_text, answer in test_cases:
            # Tokenize
            tokens = tokenizer(
                masked_text,
                return_tensors='pt',
                padding='max_length',
                max_length=32,
                truncation=True
            ).to(device)

            input_ids = tokens['input_ids']

            # Forward
            outputs = model(input_ids, top_k=768)
            logits = outputs['logits']

            # Mask 위치 찾기
            mask_pos = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
            if len(mask_pos[0]) == 0:
                continue

            mask_logits = logits[mask_pos[0][0], mask_pos[1][0]]

            # Top-5 예측
            top5_probs, top5_ids = torch.topk(mask_logits, 5)
            top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids]

            # 정답 확인
            answer_lower = answer.lower().strip()
            top1_match = top5_tokens[0].lower().strip() == answer_lower
            top5_match = any(t.lower().strip() == answer_lower for t in top5_tokens)

            if top1_match:
                correct_top1 += 1
            if top5_match:
                correct_top5 += 1

            print(f"\n문장: {masked_text}")
            print(f"정답: {answer}")
            print(f"예측: {top5_tokens[0]} {'✓' if top1_match else '✗'}")
            print(f"Top-5: {top5_tokens}")

    print(f"\n정확도:")
    print(f"  Top-1: {correct_top1}/{len(test_cases)} ({correct_top1/len(test_cases)*100:.1f}%)")
    print(f"  Top-5: {correct_top5}/{len(test_cases)} ({correct_top5/len(test_cases)*100:.1f}%)")


def analyze_sparsity_quality(model, tokenizer, device):
    """스파시티 품질 분석"""
    print("\n" + "="*70)
    print("스파시티 품질 분석")
    print("="*70)

    model.eval()

    test_text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer(
        test_text,
        return_tensors='pt',
        padding='max_length',
        max_length=32,
        truncation=True
    )['input_ids'].to(device)

    # Dense 출력
    with torch.no_grad():
        outputs_dense = model(tokens, top_k=None)
        logits_dense = outputs_dense['logits']

    # 다양한 스파시티 테스트
    sparsity_levels = [
        (int(0.1 * 3072), "10%"),
        (int(0.25 * 3072), "25%"),
        (int(0.5 * 3072), "50%"),
    ]

    print(f"\nDense 출력 norm: {logits_dense.norm().item():.4f}")
    print("\nSparsity | MSE Loss | Cosine Sim | Norm %")
    print("-" * 50)

    for top_k, label in sparsity_levels:
        with torch.no_grad():
            outputs_sparse = model(tokens, top_k=top_k)
            logits_sparse = outputs_sparse['logits']

        # 메트릭
        mse = F.mse_loss(logits_sparse, logits_dense).item()

        flat_sparse = logits_sparse.flatten()
        flat_dense = logits_dense.flatten()
        cos_sim = F.cosine_similarity(flat_sparse.unsqueeze(0), flat_dense.unsqueeze(0)).item()

        norm_ratio = (logits_sparse.norm() / logits_dense.norm()).item() * 100

        print(f"{label:8s} | {mse:8.6f} | {cos_sim:10.6f} | {norm_ratio:6.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("="*70)
    print("뉴런 기반 모델 체크포인트 분석")
    print("="*70)
    print(f"체크포인트: {args.checkpoint}")
    print(f"디바이스: {args.device}")

    # 체크포인트 로드
    print("\n체크포인트 로딩 중...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # 모델 설정 복원
    model_args = checkpoint.get('args', {})

    # 모델 생성
    model = NeuronBasedLanguageModel(
        vocab_size=model_args.get('vocab_size', 30522),
        d_model=model_args.get('d_model', 768),
        d_ff=model_args.get('d_ff', 3072),
        n_layers=model_args.get('n_layers', 12),
        n_heads=model_args.get('n_heads', 12),
        max_seq_len=model_args.get('max_seq_len', 128),
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 모델 로드 완료")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Step: {checkpoint.get('global_step', '?')}")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 분석 실행
    analyze_neuron_usage(model, tokenizer, args.device, num_samples=100)
    analyze_sparsity_quality(model, tokenizer, args.device)
    analyze_mlm_quality(model, tokenizer, args.device)

    print("\n" + "="*70)
    print("분석 완료!")
    print("="*70)


if __name__ == "__main__":
    main()
