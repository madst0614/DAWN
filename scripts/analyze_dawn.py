"""
DAWN Checkpoint Comprehensive Analysis
Dynamic Neuron Transformer 분석 스크립트

분석 항목:
1. 뉴런 활성화 패턴 (사용 빈도, Gini coefficient, Entropy)
2. 패턴 활성화 분석 (다양성, 사용률)
3. 토큰-뉴런 전문화 (특정 토큰이 특정 뉴런 선택?)
4. Layer별 차이 (KL divergence, Cosine similarity)
5. 정확도-불확실성 관계
6. 종합 시각화 및 리포트

Usage:
    python scripts/analyze_dawn.py --checkpoint /path/to/checkpoint_folder
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import json
from collections import defaultdict, Counter
from scipy.stats import entropy
import yaml

from models.model import DAWN
from utils.data import load_data, apply_mlm_masking, MLM_CONFIG
from transformers import BertTokenizer


# ============================================================
# 데이터 수집
# ============================================================

class ActivationCollector:
    """뉴런/패턴 선택 패턴 수집"""
    def __init__(self, model, n_layers):
        self.model = model
        self.n_layers = n_layers

        # 뉴런 선택 기록
        self.neuron_selections = [[] for _ in range(n_layers)]

        # 토큰별 뉴런 선택
        self.token_neuron_map = defaultdict(lambda: [[] for _ in range(n_layers)])

        # 예측 정확도별 패턴
        self.correct_selections = [[] for _ in range(n_layers)]
        self.incorrect_selections = [[] for _ in range(n_layers)]

    def collect(self, input_ids, labels, logits, all_selected):
        """한 배치의 선택 패턴 수집"""
        B, S = input_ids.shape

        # 예측 정확도
        predictions = logits.argmax(dim=-1)  # [B, S]
        correct_mask = (predictions == labels) & (labels != -100)  # [B, S]

        for layer_idx, selected_idx in enumerate(all_selected):
            # selected_idx: [B, S, k]

            # 1. 전체 뉴런 선택 기록
            self.neuron_selections[layer_idx].append(selected_idx.cpu())

            # 2. 토큰별 뉴런 선택
            for b in range(B):
                for s in range(S):
                    token_id = input_ids[b, s].item()
                    neurons = selected_idx[b, s].cpu().tolist()
                    self.token_neuron_map[token_id][layer_idx].extend(neurons)

            # 3. 정확도별 뉴런 선택
            correct_neurons = selected_idx[correct_mask].cpu()
            incorrect_neurons = selected_idx[~correct_mask].cpu()

            if len(correct_neurons) > 0:
                self.correct_selections[layer_idx].append(correct_neurons)
            if len(incorrect_neurons) > 0:
                self.incorrect_selections[layer_idx].append(incorrect_neurons)

    def finalize(self):
        """수집 완료 후 텐서 병합"""
        for layer_idx in range(self.n_layers):
            if self.neuron_selections[layer_idx]:
                self.neuron_selections[layer_idx] = torch.cat(
                    self.neuron_selections[layer_idx], dim=0
                )

            if self.correct_selections[layer_idx]:
                self.correct_selections[layer_idx] = torch.cat(
                    self.correct_selections[layer_idx], dim=0
                )

            if self.incorrect_selections[layer_idx]:
                self.incorrect_selections[layer_idx] = torch.cat(
                    self.incorrect_selections[layer_idx], dim=0
                )


# ============================================================
# 1. 뉴런 사용 분석
# ============================================================

def analyze_neuron_usage(collector, n_neurons, n_layers):
    """뉴런 사용 빈도 및 분포 분석"""
    print("\n" + "="*70)
    print("1. NEURON USAGE ANALYSIS")
    print("="*70)

    results = {}

    for layer_idx in range(n_layers):
        selections = collector.neuron_selections[layer_idx]

        if len(selections) == 0:
            continue

        # 뉴런별 선택 빈도
        neuron_counts = torch.bincount(
            selections.flatten(),
            minlength=n_neurons
        ).numpy()

        total_selections = neuron_counts.sum()
        neuron_freq = neuron_counts / total_selections

        # Gini coefficient (불균형 측정)
        sorted_freq = np.sort(neuron_freq)
        n = len(sorted_freq)
        cumsum = np.cumsum(sorted_freq)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_freq) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])

        # 사용률
        used_neurons = (neuron_counts > 0).sum()
        usage_ratio = used_neurons / n_neurons

        # Top-k 집중도
        top_10_ratio = np.sort(neuron_freq)[-10:].sum()
        top_50_ratio = np.sort(neuron_freq)[-50:].sum()

        print(f"\nLayer {layer_idx}:")
        print(f"  Used neurons: {used_neurons}/{n_neurons} ({usage_ratio:.2%})")
        print(f"  Gini coefficient: {gini:.4f} (0=equal, 1=unequal)")
        print(f"  Entropy: {entropy(neuron_freq + 1e-10):.4f}")
        print(f"  Top-10 neurons: {top_10_ratio:.2%}")
        print(f"  Top-50 neurons: {top_50_ratio:.2%}")

        # ⚠️ 경고 시스템
        warnings = []
        if usage_ratio < 0.2:
            warnings.append(f"⚠️  SPARSE: Only {usage_ratio:.1%} neurons used - potential bottleneck!")
        if gini > 0.8:
            warnings.append(f"⚠️  UNEQUAL: Gini={gini:.2f} - heavily concentrated usage!")
        if top_10_ratio > 0.5:
            warnings.append(f"⚠️  DOMINATED: Top-10 neurons = {top_10_ratio:.1%} of usage!")

        for warning in warnings:
            print(f"  {warning}")

        results[f'layer_{layer_idx}'] = {
            'neuron_counts': neuron_counts.tolist(),
            'neuron_freq': neuron_freq.tolist(),
            'gini_coefficient': float(gini),
            'used_neurons': int(used_neurons),
            'total_neurons': int(n_neurons),
            'usage_ratio': float(usage_ratio),
            'top_10_ratio': float(top_10_ratio),
            'top_50_ratio': float(top_50_ratio),
            'entropy': float(entropy(neuron_freq + 1e-10)),
        }

    return results


# ============================================================
# 2. 토큰-뉴런 전문화 분석
# ============================================================

def analyze_token_neuron_specialization(collector, tokenizer, n_neurons, n_layers, top_k_tokens=50):
    """토큰별 뉴런 선택 패턴 분석"""
    print("\n" + "="*70)
    print("2. TOKEN-NEURON SPECIALIZATION")
    print("="*70)

    results = {}

    # 가장 많이 나온 토큰
    all_tokens = list(collector.token_neuron_map.keys())
    token_counts = {
        token_id: sum(len(neurons) for neurons in collector.token_neuron_map[token_id])
        for token_id in all_tokens
    }
    top_tokens = sorted(token_counts.keys(), key=lambda x: token_counts[x], reverse=True)[:top_k_tokens]

    for layer_idx in range(n_layers):
        print(f"\nLayer {layer_idx} - Top 10 specialized tokens:")
        token_neuron_patterns = {}

        for token_id in top_tokens:
            neurons = collector.token_neuron_map[token_id][layer_idx]

            if len(neurons) == 0:
                continue

            neuron_counts = Counter(neurons)
            total = sum(neuron_counts.values())

            top_3 = neuron_counts.most_common(3)
            concentration = sum(count for _, count in top_3) / total if total > 0 else 0
            unique_neurons = len(neuron_counts)

            token_str = tokenizer.decode([token_id]).strip()

            token_neuron_patterns[token_str] = {
                'token_id': token_id,
                'total_occurrences': total,
                'unique_neurons': unique_neurons,
                'concentration': float(concentration),
                'top_3_neurons': [(int(n), int(c)) for n, c in top_3],
            }

        # 집중도 높은 순으로 정렬
        sorted_tokens = sorted(
            token_neuron_patterns.items(),
            key=lambda x: x[1]['concentration'],
            reverse=True
        )[:10]

        for token_str, data in sorted_tokens:
            print(f"  '{token_str}': concentration={data['concentration']:.2%}, "
                  f"unique={data['unique_neurons']}, top_3={data['top_3_neurons'][:2]}")

        results[f'layer_{layer_idx}'] = token_neuron_patterns

    return results


# ============================================================
# 3. Layer별 차이 분석
# ============================================================

def analyze_layer_differences(neuron_usage_results):
    """Layer별 뉴런 사용 패턴 차이"""
    print("\n" + "="*70)
    print("3. LAYER DIFFERENCES")
    print("="*70)

    results = {}
    layers = sorted([k for k in neuron_usage_results.keys() if k.startswith('layer_')])

    for i, layer_i in enumerate(layers):
        for j, layer_j in enumerate(layers):
            if i >= j:
                continue

            freq_i = np.array(neuron_usage_results[layer_i]['neuron_freq'])
            freq_j = np.array(neuron_usage_results[layer_j]['neuron_freq'])

            kl_div = entropy(freq_i + 1e-10, freq_j + 1e-10)
            cos_sim = np.dot(freq_i, freq_j) / (np.linalg.norm(freq_i) * np.linalg.norm(freq_j))

            print(f"\n{layer_i} vs {layer_j}:")
            print(f"  KL Divergence: {kl_div:.4f} (higher = more different)")
            print(f"  Cosine Similarity: {cos_sim:.4f} (1=identical)")

            results[f'{layer_i}_vs_{layer_j}'] = {
                'kl_divergence': float(kl_div),
                'cosine_similarity': float(cos_sim),
            }

    return results


# ============================================================
# 4. 정확도-불확실성 분석
# ============================================================

def analyze_uncertainty_accuracy(collector, n_layers):
    """정답/오답 시 뉴런 선택 패턴 비교"""
    print("\n" + "="*70)
    print("4. ACCURACY-UNCERTAINTY RELATIONSHIP")
    print("="*70)

    results = {}

    for layer_idx in range(n_layers):
        correct_sel = collector.correct_selections[layer_idx]
        incorrect_sel = collector.incorrect_selections[layer_idx]

        if len(correct_sel) == 0 or len(incorrect_sel) == 0:
            continue

        correct_unique = len(torch.unique(correct_sel))
        incorrect_unique = len(torch.unique(incorrect_sel))

        # 중복 뉴런 (정답과 오답 모두 사용)
        correct_neurons_set = set(torch.unique(correct_sel).tolist())
        incorrect_neurons_set = set(torch.unique(incorrect_sel).tolist())
        overlap = correct_neurons_set & incorrect_neurons_set
        overlap_ratio = len(overlap) / len(correct_neurons_set | incorrect_neurons_set)

        # 뉴런 다양성 (평균 유니크 뉴런 수)
        correct_diversity = correct_unique / len(correct_sel) if len(correct_sel) > 0 else 0
        incorrect_diversity = incorrect_unique / len(incorrect_sel) if len(incorrect_sel) > 0 else 0

        print(f"\nLayer {layer_idx}:")
        print(f"  Correct: {len(correct_sel):,} samples, {correct_unique} unique neurons")
        print(f"  Incorrect: {len(incorrect_sel):,} samples, {incorrect_unique} unique neurons")
        print(f"  Overlap: {len(overlap)} neurons ({overlap_ratio:.2%})")
        print(f"  Diversity: Correct={correct_diversity:.4f}, Incorrect={incorrect_diversity:.4f}")

        # ⚠️ 불확실성 신호 체크
        if overlap_ratio > 0.9:
            print(f"  ⚠️  WEAK SIGNAL: {overlap_ratio:.1%} overlap - can't distinguish correct/incorrect!")
        elif abs(correct_unique - incorrect_unique) / max(correct_unique, incorrect_unique) < 0.1:
            print(f"  ⚠️  SIMILAR PATTERNS: Correct and incorrect use similar neurons!")

        results[f'layer_{layer_idx}'] = {
            'correct_samples': len(correct_sel),
            'incorrect_samples': len(incorrect_sel),
            'correct_unique_neurons': int(correct_unique),
            'incorrect_unique_neurons': int(incorrect_unique),
            'overlap_neurons': len(overlap),
            'overlap_ratio': float(overlap_ratio),
            'correct_diversity': float(correct_diversity),
            'incorrect_diversity': float(incorrect_diversity),
        }

    return results


# ============================================================
# 5. 시각화
# ============================================================

def visualize_results(neuron_usage_results, output_dir):
    """분석 결과 시각화"""
    output_dir = Path(output_dir)
    n_layers = len([k for k in neuron_usage_results.keys() if k.startswith('layer_')])

    print("\n" + "="*70)
    print("5. GENERATING VISUALIZATIONS")
    print("="*70)

    # 1. 뉴런 사용 분포
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for layer_idx in range(min(n_layers, 4)):
        layer_key = f'layer_{layer_idx}'
        neuron_freq = neuron_usage_results[layer_key]['neuron_freq']

        ax = axes[layer_idx]
        ax.hist(neuron_freq, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Neuron Selection Frequency', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'Layer {layer_idx}\nGini: {neuron_usage_results[layer_key]["gini_coefficient"]:.3f}, '
                    f'Usage: {neuron_usage_results[layer_key]["usage_ratio"]:.2%}', fontsize=11)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'neuron_usage_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()  # Colab에서 바로 표시
    plt.close()
    print("  ✓ neuron_usage_distribution.png")

    # 2. Layer별 통계
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    layers = list(range(n_layers))
    gini_coeffs = [neuron_usage_results[f'layer_{i}']['gini_coefficient'] for i in layers]
    usage_ratios = [neuron_usage_results[f'layer_{i}']['usage_ratio'] for i in layers]
    entropies = [neuron_usage_results[f'layer_{i}']['entropy'] for i in layers]

    axes[0].bar(layers, gini_coeffs, color='skyblue', edgecolor='black', width=0.6)
    axes[0].set_xlabel('Layer', fontsize=11)
    axes[0].set_ylabel('Gini Coefficient', fontsize=11)
    axes[0].set_title('Neuron Usage Inequality', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')

    axes[1].bar(layers, usage_ratios, color='lightcoral', edgecolor='black', width=0.6)
    axes[1].set_xlabel('Layer', fontsize=11)
    axes[1].set_ylabel('Usage Ratio', fontsize=11)
    axes[1].set_title('Fraction of Used Neurons', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1])

    axes[2].bar(layers, entropies, color='lightgreen', edgecolor='black', width=0.6)
    axes[2].set_xlabel('Layer', fontsize=11)
    axes[2].set_ylabel('Entropy', fontsize=11)
    axes[2].set_title('Neuron Selection Diversity', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'layer_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()  # Colab에서 바로 표시
    plt.close()
    print("  ✓ layer_statistics.png")

    # 3. Top-50 뉴런 히트맵
    fig, ax = plt.subplots(figsize=(12, 6))

    heatmap_data = []
    for layer_idx in range(n_layers):
        layer_key = f'layer_{layer_idx}'
        neuron_freq = np.array(neuron_usage_results[layer_key]['neuron_freq'])
        top_50_indices = np.argsort(neuron_freq)[-50:]
        top_50_freq = neuron_freq[top_50_indices]
        heatmap_data.append(top_50_freq)

    heatmap_data = np.array(heatmap_data)

    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Frequency'})
    ax.set_xlabel('Top-50 Neurons (sorted by frequency)', fontsize=11)
    ax.set_ylabel('Layer', fontsize=11)
    ax.set_title('Top-50 Most Active Neurons per Layer', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'neuron_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()  # Colab에서 바로 표시
    plt.close()
    print("  ✓ neuron_heatmap.png")


# ============================================================
# 6. 리포트 생성
# ============================================================

def generate_report(all_results, output_dir, checkpoint_path):
    """종합 리포트 생성"""
    output_dir = Path(output_dir)
    report_path = output_dir / 'analysis_report.txt'

    print("\n" + "="*70)
    print("6. GENERATING REPORT")
    print("="*70)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DAWN - Dynamic Neuron Transformer Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")

        # 1. 뉴런 사용
        f.write("1. NEURON USAGE ANALYSIS\n")
        f.write("-" * 80 + "\n")

        neuron_results = all_results['neuron_usage']
        for layer_key in sorted(neuron_results.keys()):
            layer_data = neuron_results[layer_key]
            f.write(f"\n{layer_key.upper()}:\n")
            f.write(f"  Used: {layer_data['used_neurons']}/{layer_data['total_neurons']} "
                   f"({layer_data['usage_ratio']:.2%})\n")
            f.write(f"  Gini: {layer_data['gini_coefficient']:.4f}, "
                   f"Entropy: {layer_data['entropy']:.4f}\n")
            f.write(f"  Top-10: {layer_data['top_10_ratio']:.2%}, "
                   f"Top-50: {layer_data['top_50_ratio']:.2%}\n")

        # 2. Layer 차이
        f.write("\n\n2. LAYER DIFFERENCES\n")
        f.write("-" * 80 + "\n")

        layer_diff = all_results['layer_differences']
        for pair_key in sorted(layer_diff.keys()):
            pair_data = layer_diff[pair_key]
            f.write(f"\n{pair_key}: KL={pair_data['kl_divergence']:.4f}, "
                   f"Cosine={pair_data['cosine_similarity']:.4f}\n")

        # 3. 토큰-뉴런 전문화 (샘플)
        f.write("\n\n3. TOKEN-NEURON SPECIALIZATION (Top 5 per layer)\n")
        f.write("-" * 80 + "\n")

        token_spec = all_results['token_neuron_specialization']
        for layer_key in sorted(token_spec.keys()):
            f.write(f"\n{layer_key.upper()}:\n")
            layer_data = token_spec[layer_key]

            sorted_tokens = sorted(
                layer_data.items(),
                key=lambda x: x[1]['concentration'],
                reverse=True
            )[:5]

            for token_str, data in sorted_tokens:
                f.write(f"  '{token_str}': {data['concentration']:.2%} concentration, "
                       f"{data['unique_neurons']} unique, top_3={data['top_3_neurons']}\n")

        # 4. 정확도-불확실성
        f.write("\n\n4. ACCURACY-UNCERTAINTY\n")
        f.write("-" * 80 + "\n")

        uncertainty = all_results['uncertainty_accuracy']
        for layer_key in sorted(uncertainty.keys()):
            layer_data = uncertainty[layer_key]
            f.write(f"\n{layer_key}:\n")
            f.write(f"  Correct: {layer_data['correct_unique_neurons']} neurons\n")
            f.write(f"  Incorrect: {layer_data['incorrect_unique_neurons']} neurons\n")
            f.write(f"  Overlap: {layer_data['overlap_neurons']} ({layer_data['overlap_ratio']:.2%})\n")
            f.write(f"  Diversity: Correct={layer_data['correct_diversity']:.4f}, "
                   f"Incorrect={layer_data['incorrect_diversity']:.4f}\n")

        # 5. 권장사항
        f.write("\n\n5. RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n\n")

        recommendations = []

        # 뉴런 희소성 체크
        neuron_results = all_results['neuron_usage']
        for layer_key in sorted(neuron_results.keys()):
            layer_data = neuron_results[layer_key]
            if layer_data['usage_ratio'] < 0.2:
                recommendations.append(
                    f"⚠️  {layer_key}: Only {layer_data['usage_ratio']:.1%} neurons used\n"
                    f"   → Consider increasing n_neurons (currently {layer_data['total_neurons']})\n"
                    f"   → Or reducing k (top-k selection parameter)"
                )
            if layer_data['gini_coefficient'] > 0.8:
                recommendations.append(
                    f"⚠️  {layer_key}: High inequality (Gini={layer_data['gini_coefficient']:.2f})\n"
                    f"   → Few neurons dominate - may need better initialization\n"
                    f"   → Or add regularization to encourage uniform usage"
                )

        # 불확실성 신호 체크
        for layer_key in sorted(uncertainty.keys()):
            layer_data = uncertainty[layer_key]
            if layer_data['overlap_ratio'] > 0.9:
                recommendations.append(
                    f"⚠️  {layer_key}: Weak uncertainty signal (overlap={layer_data['overlap_ratio']:.1%})\n"
                    f"   → Model can't distinguish correct/incorrect predictions\n"
                    f"   → May need more training or larger model capacity"
                )

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n\n")
        else:
            f.write("✓ No critical issues detected!\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"  ✓ analysis_report.txt")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze DAWN checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint folder or .pt file')
    parser.add_argument('--num_batches', type=int, default=100,
                       help='Number of batches to analyze')
    args = parser.parse_args()

    # 체크포인트 경로 처리
    checkpoint_path = Path(args.checkpoint)

    if checkpoint_path.is_dir():
        # 폴더인 경우 best_model.pt 찾기
        best_model_path = checkpoint_path / 'best_model.pt'
        config_path = checkpoint_path / 'config.json'
        output_dir = checkpoint_path / 'analysis'
    else:
        # 파일인 경우
        best_model_path = checkpoint_path
        config_path = checkpoint_path.parent / 'config.json'
        output_dir = checkpoint_path.parent / 'analysis'

    if not best_model_path.exists():
        print(f"❌ Checkpoint not found: {best_model_path}")
        return

    output_dir.mkdir(exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Config 로드
    print(f"\nLoading config: {config_path}")
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    # 체크포인트 로드
    print(f"Loading checkpoint: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)

    # 모델 생성
    print("Creating model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

    model = DAWN(
        vocab_size=vocab_size,
        hidden_dim=cfg['model']['d_model'],
        num_layers=cfg['model']['n_layers'],
        n_heads=cfg['model']['n_heads'],
        n_neurons=cfg['model']['n_neurons'],
        n_patterns=cfg['model']['n_patterns'],
        k=cfg['model']['k'],
        d_ff=cfg['model'].get('d_ff', None),
        max_seq_len=cfg['model']['max_seq_len'],
        dropout=cfg['model']['dropout']
    )

    # Load state dict (handle torch.compile() prefix)
    state_dict = checkpoint['model_state_dict']

    # Remove _orig_mod. prefix if present (from torch.compile)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    n_layers = cfg['model']['n_layers']
    n_neurons = cfg['model']['n_neurons']

    print(f"\nModel: {n_layers} layers, {n_neurons} neurons/layer")
    print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")

    # 데이터 로드
    print("\nLoading validation data...")
    _, val_loader, _ = load_data(
        cfg['data'],
        max_length=cfg['model']['max_seq_len'],
        batch_size=32
    )

    # 수집
    print(f"\nCollecting neuron patterns from {args.num_batches} batches...")
    collector = ActivationCollector(model, n_layers)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, total=args.num_batches)):
            if batch_idx >= args.num_batches:
                break

            input_ids = batch['input_ids'].to(device)
            input_ids, labels = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)

            logits, all_selected = model(input_ids, return_activations=True)
            collector.collect(input_ids, labels, logits, all_selected)

    collector.finalize()

    # 분석
    neuron_usage_results = analyze_neuron_usage(collector, n_neurons, n_layers)
    token_spec_results = analyze_token_neuron_specialization(
        collector, tokenizer, n_neurons, n_layers, top_k_tokens=100
    )
    layer_diff_results = analyze_layer_differences(neuron_usage_results)
    uncertainty_results = analyze_uncertainty_accuracy(collector, n_layers)

    all_results = {
        'neuron_usage': neuron_usage_results,
        'token_neuron_specialization': token_spec_results,
        'layer_differences': layer_diff_results,
        'uncertainty_accuracy': uncertainty_results,
    }

    # 저장
    print(f"\nSaving results to: {output_dir}")
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("  ✓ analysis_results.json")

    # 시각화
    visualize_results(neuron_usage_results, output_dir)

    # 리포트
    generate_report(all_results, output_dir, best_model_path)

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("  - analysis_results.json")
    print("  - analysis_report.txt")
    print("  - neuron_usage_distribution.png")
    print("  - layer_statistics.png")
    print("  - neuron_heatmap.png")


if __name__ == "__main__":
    main()
