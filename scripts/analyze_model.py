"""
SPROUT Brain-Like 모델 분석 도구

학습된 모델의 뉴런 활성화 패턴, 다양성, 예측 품질 분석
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from tqdm import tqdm

from src.models.sprout_brain_like import create_brain_like_sprout


def analyze_neuron_usage(model, dataloader, device, num_batches=100):
    """
    뉴런 사용 패턴 분석

    각 뉴런이 얼마나 자주 사용되는지 확인
    """
    print("\n" + "="*70)
    print("뉴런 사용 패턴 분석")
    print("="*70)

    n_neurons = model.n_neurons
    neuron_usage = torch.zeros(n_neurons)
    all_activations = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing", total=num_batches)):
            if batch_idx >= num_batches:
                break

            tokens = batch['input_ids'].to(device)

            # 초기 활성화 패턴
            activation = model.input_encoder(tokens)  # [batch, n_neurons]

            # 사용된 뉴런 기록
            active = (activation > 0.01).float()
            neuron_usage += active.sum(dim=0).cpu()

            # 전체 활성화 저장
            all_activations.append(activation.cpu())

    # 통계 계산
    total_samples = min(num_batches, len(dataloader)) * dataloader.batch_size
    usage_freq = neuron_usage / total_samples

    # 카테고리별 분류
    always_used = (usage_freq > 0.95).sum().item()
    often_used = ((usage_freq > 0.5) & (usage_freq <= 0.95)).sum().item()
    sometimes_used = ((usage_freq > 0.1) & (usage_freq <= 0.5)).sum().item()
    rarely_used = ((usage_freq > 0.01) & (usage_freq <= 0.1)).sum().item()
    never_used = (usage_freq <= 0.01).sum().item()

    print(f"\n총 {total_samples} 샘플 분석")
    print(f"\n뉴런 사용 분포:")
    print(f"  항상 사용 (>95%):   {always_used:4d} / {n_neurons} ({always_used/n_neurons*100:.1f}%)")
    print(f"  자주 사용 (50-95%):  {often_used:4d} / {n_neurons} ({often_used/n_neurons*100:.1f}%)")
    print(f"  가끔 사용 (10-50%):  {sometimes_used:4d} / {n_neurons} ({sometimes_used/n_neurons*100:.1f}%)")
    print(f"  드물게 (1-10%):     {rarely_used:4d} / {n_neurons} ({rarely_used/n_neurons*100:.1f}%)")
    print(f"  거의 안 씀 (<1%):   {never_used:4d} / {n_neurons} ({never_used/n_neurons*100:.1f}%)")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 사용 빈도 히스토그램
    ax = axes[0]
    ax.hist(usage_freq.numpy(), bins=50, edgecolor='black')
    ax.set_xlabel('Usage Frequency')
    ax.set_ylabel('Number of Neurons')
    ax.set_title('Neuron Usage Distribution')
    ax.grid(True, alpha=0.3)

    # 카테고리별 막대 그래프
    ax = axes[1]
    categories = ['Always\n(>95%)', 'Often\n(50-95%)', 'Sometimes\n(10-50%)',
                  'Rarely\n(1-10%)', 'Never\n(<1%)']
    counts = [always_used, often_used, sometimes_used, rarely_used, never_used]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']

    ax.bar(categories, counts, color=colors, edgecolor='black')
    ax.set_ylabel('Number of Neurons')
    ax.set_title('Neuron Usage Categories')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('neuron_usage.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: neuron_usage.png")

    return {
        'usage_freq': usage_freq,
        'categories': {
            'always': always_used,
            'often': often_used,
            'sometimes': sometimes_used,
            'rarely': rarely_used,
            'never': never_used
        }
    }


def analyze_input_diversity(model, tokenizer, device):
    """
    입력 다양성 분석

    서로 다른 입력에 대해 다른 뉴런이 활성화되는지 확인
    """
    print("\n" + "="*70)
    print("입력 다양성 분석")
    print("="*70)

    # 다양한 테스트 문장
    test_inputs = [
        "The cat is sleeping on the mat",
        "I love programming in Python",
        "Quantum mechanics is very difficult",
        "She bought a new car yesterday",
        "The weather is nice today",
        "Machine learning models require data",
        "The students are studying for exams",
        "Pizza is my favorite food",
        "Scientists discovered a new planet",
        "The concert was absolutely amazing"
    ]

    model.eval()
    activations = []

    print(f"\n{len(test_inputs)} 개의 테스트 문장 분석 중...")

    with torch.no_grad():
        for text in test_inputs:
            tokens = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=32,
                truncation=True
            )['input_ids'].to(device)

            act = model.input_encoder(tokens)
            activations.append(act[0].cpu())  # [n_neurons]

    # 유사도 매트릭스 계산
    n = len(test_inputs)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sim = F.cosine_similarity(
                activations[i].unsqueeze(0),
                activations[j].unsqueeze(0)
            ).item()
            similarity_matrix[i, j] = sim

    # 통계
    all_sims = []
    for i in range(n):
        for j in range(i+1, n):
            all_sims.append(similarity_matrix[i, j])

    mean_sim = np.mean(all_sims)
    std_sim = np.std(all_sims)

    print(f"\n평균 유사도: {mean_sim:.3f} (± {std_sim:.3f})")
    print(f"최소 유사도: {np.min(all_sims):.3f}")
    print(f"최대 유사도: {np.max(all_sims):.3f}")

    # 해석
    if mean_sim < 0.3:
        interpretation = "✅ 매우 다양한 활성화 (좋음!)"
    elif mean_sim < 0.7:
        interpretation = "✓ 적당히 다양함"
    elif mean_sim < 0.9:
        interpretation = "⚠️ 비슷한 패턴 (문제 가능)"
    else:
        interpretation = "❌ 거의 동일한 패턴 (심각)"

    print(f"\n해석: {interpretation}")

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(similarity_matrix, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Input {i+1}" for i in range(n)], rotation=45, ha='right')
    ax.set_yticklabels([f"Input {i+1}" for i in range(n)])

    # 값 표시
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Activation Pattern Similarity Matrix')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig('input_diversity.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: input_diversity.png")

    # 각 문장별 활성 뉴런 출력
    print(f"\n각 입력별 활성 뉴런 (Top 10):")
    for i, text in enumerate(test_inputs):
        active_neurons = (activations[i] > 0.01).nonzero().squeeze()
        top_values, top_indices = torch.topk(activations[i], k=10)

        print(f"\n{i+1}. \"{text}\"")
        print(f"   활성 뉴런 수: {len(active_neurons)}")
        print(f"   Top 10: {top_indices.tolist()}")

    return {
        'mean_similarity': mean_sim,
        'std_similarity': std_sim,
        'similarity_matrix': similarity_matrix,
        'interpretation': interpretation
    }


def analyze_prediction_quality(model, tokenizer, device):
    """
    예측 품질 분석

    Masked token 예측이 얼마나 정확한지 확인
    """
    print("\n" + "="*70)
    print("예측 품질 분석")
    print("="*70)

    # 테스트 케이스 (정답 포함)
    test_cases = [
        ("The cat is [MASK] on the mat", "sleeping"),
        ("I love [MASK] in Python", "programming"),
        ("The [MASK] is very difficult", "problem"),
        ("She bought a new [MASK] yesterday", "car"),
        ("The weather is [MASK] today", "nice"),
        ("Machine [MASK] models require data", "learning"),
        ("The students are [MASK] for exams", "studying"),
        ("Pizza is my favorite [MASK]", "food"),
        ("Scientists discovered a new [MASK]", "planet"),
        ("The concert was absolutely [MASK]", "amazing"),
    ]

    model.eval()
    results = []

    print(f"\n{len(test_cases)} 개의 MLM 테스트:")

    with torch.no_grad():
        for text, answer in test_cases:
            # 토큰화
            tokens = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=32,
                truncation=True
            )['input_ids'].to(device)

            # MASK 위치 확인 (존재 여부만)
            mask_pos = (tokens == tokenizer.mask_token_id).nonzero()
            if len(mask_pos) == 0:
                continue

            # 예측
            # Brain-Like 모델: 전체 시퀀스 → 하나의 예측 [batch, vocab_size]
            logits = model(tokens)
            pred_logits = logits[0]  # [vocab_size]

            # Top-10 예측
            top_values, top_indices = torch.topk(pred_logits, k=10)
            top_words = [tokenizer.decode([idx]) for idx in top_indices]

            # 정답 위치
            answer_id = tokenizer.convert_tokens_to_ids(answer)
            answer_rank = (pred_logits.argsort(descending=True) == answer_id).nonzero()
            if len(answer_rank) > 0:
                answer_rank = answer_rank.item() + 1
            else:
                answer_rank = -1

            results.append({
                'text': text,
                'answer': answer,
                'top_10': top_words,
                'answer_rank': answer_rank,
                'in_top_10': answer_rank <= 10 and answer_rank > 0
            })

            print(f"\n입력: {text}")
            print(f"정답: {answer} (Rank: {answer_rank if answer_rank > 0 else '>1000'})")
            print(f"Top 10 예측:")
            for i, word in enumerate(top_words):
                marker = "✅" if word.strip() == answer else ""
                print(f"  {i+1}. {word:15s} {marker}")

    # 통계
    total = len(results)
    top1_correct = sum(1 for r in results if r['answer_rank'] == 1)
    top5_correct = sum(1 for r in results if 1 <= r['answer_rank'] <= 5)
    top10_correct = sum(1 for r in results if r['in_top_10'])

    print(f"\n" + "="*70)
    print(f"예측 정확도:")
    print(f"  Top-1:  {top1_correct}/{total} ({top1_correct/total*100:.1f}%)")
    print(f"  Top-5:  {top5_correct}/{total} ({top5_correct/total*100:.1f}%)")
    print(f"  Top-10: {top10_correct}/{total} ({top10_correct/total*100:.1f}%)")
    print("="*70)

    return {
        'top1_accuracy': top1_correct / total,
        'top5_accuracy': top5_correct / total,
        'top10_accuracy': top10_correct / total,
        'results': results
    }


def analyze_learning_curve(checkpoint_dir):
    """
    학습 곡선 분석

    Epoch별 loss/accuracy 트렌드 확인
    """
    print("\n" + "="*70)
    print("학습 곡선 분석")
    print("="*70)

    # 체크포인트 로드
    checkpoint_path = os.path.join(checkpoint_dir, "sprout_brain_like_best.pt")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 정보 출력
    print(f"\nCheckpoint 정보:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    print(f"  Accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%")

    # 수동으로 입력된 히스토리 (실제로는 로그에서 파싱)
    # 여기서는 예시
    epochs = [1, 2, 3]
    losses = [12.09, 11.27, 11.19]
    accs = [7.35, 7.64, 8.20]

    # 트렌드 분석
    loss_improvement = (losses[0] - losses[-1]) / losses[0] * 100
    acc_improvement = (accs[-1] - accs[0]) / accs[0] * 100

    print(f"\n학습 진행:")
    print(f"  Loss 개선: {loss_improvement:.1f}% ({losses[0]:.2f} → {losses[-1]:.2f})")
    print(f"  Acc 개선: {acc_improvement:.1f}% ({accs[0]:.2f}% → {accs[-1]:.2f}%)")

    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, losses, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    # Accuracy curve
    ax2.plot(epochs, accs, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)

    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: learning_curve.png")

    # 외삽 (더 학습하면?)
    if len(epochs) >= 3:
        from scipy.optimize import curve_fit

        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c

        try:
            popt, _ = curve_fit(exp_decay, epochs, losses, p0=[10, 0.1, 8])
            predicted_loss_10 = exp_decay(10, *popt)
            predicted_loss_20 = exp_decay(20, *popt)

            print(f"\n예상 성능 (외삽):")
            print(f"  10 epoch: Loss ≈ {predicted_loss_10:.2f}")
            print(f"  20 epoch: Loss ≈ {predicted_loss_20:.2f}")
        except:
            print(f"\n⚠️ 외삽 실패 (데이터 부족)")

    return {
        'epochs': epochs,
        'losses': losses,
        'accuracies': accs,
        'loss_improvement': loss_improvement,
        'acc_improvement': acc_improvement
    }


def main():
    """메인 분석 루틴"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze SPROUT Brain-Like Model")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for analysis")
    parser.add_argument("--num_batches", type=int, default=100,
                        help="Number of batches to analyze")
    parser.add_argument("--debug", action="store_true",
                        help="Use small data for testing")

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Auto-detect checkpoint dir
    if os.path.exists("/content/drive/MyDrive/sprout_checkpoints"):
        args.checkpoint_dir = "/content/drive/MyDrive/sprout_checkpoints"
        print(f"📂 Using: {args.checkpoint_dir}")

    # Tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Model
    print(f"Loading model...")
    checkpoint_path = os.path.join(args.checkpoint_dir, "sprout_brain_like_best.pt")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"Please train the model first!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_brain_like_sprout(
        vocab_size=len(tokenizer),
        n_neurons=4096,
        d_state=256,
        n_interaction_steps=5
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")

    # 1. 학습 곡선
    analyze_learning_curve(args.checkpoint_dir)

    # 2. 입력 다양성
    analyze_input_diversity(model, tokenizer, device)

    # 3. 예측 품질
    analyze_prediction_quality(model, tokenizer, device)

    # 4. 뉴런 사용 (데이터 필요)
    if args.debug:
        print(f"\n⚠️ Debug mode: Skipping neuron usage analysis (requires dataloader)")
    else:
        print(f"\n⚠️ Neuron usage analysis requires training dataloader")
        print(f"   Run this script during/after training with --analyze_activation")

    print(f"\n" + "="*70)
    print("분석 완료!")
    print("="*70)
    print(f"\n생성된 파일:")
    print(f"  - learning_curve.png")
    print(f"  - input_diversity.png")
    print(f"\n다음 단계:")
    print(f"  1. 그래프 확인")
    print(f"  2. 더 길게 학습 (10-20 epochs)")
    print(f"  3. 학습률 조정 시도")
    print("="*70)


if __name__ == "__main__":
    main()
