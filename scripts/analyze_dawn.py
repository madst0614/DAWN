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
    """뉴런/패턴 선택 패턴 수집 (확장판)"""
    def __init__(self, model, n_layers):
        self.model = model
        self.n_layers = n_layers

        # 뉴런 선택 기록
        self.neuron_selections = [[] for _ in range(n_layers)]

        # 패턴 선택 기록 ⭐ NEW
        self.pattern_selections = [[] for _ in range(n_layers)]

        # 위치별 뉴런 선택 ⭐ NEW
        self.position_neuron_map = defaultdict(lambda: [[] for _ in range(n_layers)])

        # 토큰별 뉴런 선택
        self.token_neuron_map = defaultdict(lambda: [[] for _ in range(n_layers)])

        # 예측 정확도별 패턴
        self.correct_selections = [[] for _ in range(n_layers)]
        self.incorrect_selections = [[] for _ in range(n_layers)]

    def collect(self, input_ids, labels, logits, all_selected, all_patterns=None):
        """한 배치의 선택 패턴 수집 (최적화)"""
        B, S = input_ids.shape

        # 예측 정확도
        predictions = logits.argmax(dim=-1)  # [B, S]
        correct_mask = (predictions == labels) & (labels != -100)  # [B, S]

        # CPU로 한 번에 이동 (매 루프마다 하지 않고)
        input_ids_cpu = input_ids.cpu()

        for layer_idx, selected_idx in enumerate(all_selected):
            # selected_idx: [B, S, k]

            # CPU로 한 번에 이동
            selected_cpu = selected_idx.cpu()

            # 1. 전체 뉴런 선택 기록
            self.neuron_selections[layer_idx].append(selected_cpu)

            # 2. 패턴 선택 기록
            if all_patterns is not None:
                pattern_weights = all_patterns[layer_idx]  # [B, S, n_patterns]
                self.pattern_selections[layer_idx].append(pattern_weights.cpu())

            # 3. 토큰별 + 위치별 뉴런 선택 (vectorized)
            # Flatten: [B, S, k] → [B*S, k]
            selected_flat = selected_cpu.reshape(-1, selected_cpu.shape[-1])  # [B*S, k]
            tokens_flat = input_ids_cpu.reshape(-1)  # [B*S]

            # 토큰별: 각 unique token에 대해 뉴런 수집
            for token_id in tokens_flat.unique().tolist():
                mask = tokens_flat == token_id
                neurons = selected_flat[mask].reshape(-1).tolist()
                self.token_neuron_map[token_id][layer_idx].extend(neurons)

            # 위치별: 각 위치에 대해 뉴런 수집
            for s in range(S):
                # 모든 배치의 s번째 위치
                neurons = selected_cpu[:, s, :].reshape(-1).tolist()
                self.position_neuron_map[s][layer_idx].extend(neurons)

            # 4. 정확도별 뉴런 선택
            correct_neurons = selected_cpu[correct_mask.cpu()]
            incorrect_neurons = selected_cpu[~correct_mask.cpu()]

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

            # 패턴 정보 병합 ⭐ NEW
            if self.pattern_selections[layer_idx]:
                self.pattern_selections[layer_idx] = torch.cat(
                    self.pattern_selections[layer_idx], dim=0
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

# ============================================================
# 추가 분석: 패턴, 위치
# ============================================================

def analyze_pattern_usage(collector, n_patterns, n_layers):
    """FFN 패턴 사용 분석 ⭐"""
    print("\n" + "="*70)
    print("PATTERN (FFN) USAGE ANALYSIS")
    print("="*70)

    results = {}

    for layer_idx in range(n_layers):
        if len(collector.pattern_selections[layer_idx]) == 0:
            continue

        pattern_weights = collector.pattern_selections[layer_idx]  # [N, S, n_patterns]

        # Top-k 패턴 선택 (가장 높은 가중치)
        top_patterns = pattern_weights.argmax(dim=-1).numpy()  # [N, S]
        top_patterns = top_patterns.flatten()  # [N * S]
        pattern_counts = np.bincount(top_patterns, minlength=n_patterns)
        pattern_freq = pattern_counts / pattern_counts.sum()

        # Gini coefficient
        sorted_freq = np.sort(pattern_freq)
        n = len(sorted_freq)
        cumsum = np.cumsum(sorted_freq)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_freq) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])

        # 사용률
        used_patterns = (pattern_counts > 0).sum()
        usage_ratio = used_patterns / n_patterns

        # Top-k 집중도
        top_10_ratio = np.sort(pattern_freq)[-10:].sum()

        print(f"\nLayer {layer_idx}:")
        print(f"  Used patterns: {used_patterns}/{n_patterns} ({usage_ratio:.2%})")
        print(f"  Gini: {gini:.4f}, Entropy: {entropy(pattern_freq + 1e-10):.4f}")
        print(f"  Top-10 patterns: {top_10_ratio:.2%}")

        # 경고
        if usage_ratio < 0.2:
            print(f"  ⚠️  SPARSE: Only {usage_ratio:.1%} patterns used!")
        if gini > 0.8:
            print(f"  ⚠️  UNEQUAL: Pattern Gini={gini:.2f}!")

        results[f'layer_{layer_idx}'] = {
            'used_patterns': int(used_patterns),
            'total_patterns': int(n_patterns),
            'usage_ratio': float(usage_ratio),
            'gini_coefficient': float(gini),
            'entropy': float(entropy(pattern_freq + 1e-10)),
            'top_10_ratio': float(top_10_ratio),
        }

    return results


def analyze_pattern_collapse_detail(collector, n_patterns, n_layers):
    """패턴 collapse 상세 분석 - 어떤 패턴들이 지배하는가?"""
    print("\n" + "="*70)
    print("PATTERN COLLAPSE DETAIL ANALYSIS")
    print("="*70)

    results = {}

    for layer_idx in range(n_layers):
        if len(collector.pattern_selections[layer_idx]) == 0:
            continue

        pattern_weights = collector.pattern_selections[layer_idx]  # [N, S, n_patterns]
        top_patterns = pattern_weights.argmax(dim=-1).numpy().flatten()  # [N * S]

        # 패턴별 카운트
        pattern_counts = np.bincount(top_patterns, minlength=n_patterns)
        total_selections = pattern_counts.sum()

        # Top-10 패턴 찾기
        top_indices = np.argsort(pattern_counts)[::-1][:10]
        top_counts = pattern_counts[top_indices]
        top_ratios = top_counts / total_selections

        print(f"\nLayer {layer_idx}:")
        print(f"  Total selections: {total_selections}")
        print(f"  Top-10 patterns:")
        for rank, (idx, count, ratio) in enumerate(zip(top_indices, top_counts, top_ratios), 1):
            print(f"    #{rank}: Pattern {idx:3d} - {count:7d} times ({ratio:6.2%})")

        # 평균 gate 값 분석 (상위 5개 vs 하위 5개)
        avg_weights = pattern_weights.mean(dim=(0, 1)).numpy()  # [n_patterns]
        top5_avg = avg_weights[top_indices[:5]].mean()
        bottom5_indices = np.argsort(pattern_counts)[:5]
        bottom5_avg = avg_weights[bottom5_indices].mean()

        print(f"  Average gate values:")
        print(f"    Top-5 patterns: {top5_avg:.6f}")
        print(f"    Bottom-5 patterns: {bottom5_avg:.6f}")
        print(f"    Ratio: {top5_avg / (bottom5_avg + 1e-10):.2f}x")

        # Collapse 경고
        top1_ratio = top_ratios[0]
        top5_ratio = top_ratios[:5].sum()
        if top1_ratio > 0.5:
            print(f"  🔴 SEVERE COLLAPSE: Top-1 pattern dominates {top1_ratio:.1%}!")
        elif top5_ratio > 0.8:
            print(f"  ⚠️  COLLAPSE: Top-5 patterns dominate {top5_ratio:.1%}!")

        results[f'layer_{layer_idx}'] = {
            'top_10_patterns': top_indices.tolist(),
            'top_10_counts': top_counts.tolist(),
            'top_10_ratios': top_ratios.tolist(),
            'top5_avg_gate': float(top5_avg),
            'bottom5_avg_gate': float(bottom5_avg),
        }

    return results


def analyze_neuron_pattern_correlation(collector, n_layers, sample_size=1000):
    """뉴런 Set → 패턴 매핑 일관성 분석"""
    print("\n" + "="*70)
    print("NEURON-PATTERN CORRELATION ANALYSIS")
    print("="*70)

    results = {}

    for layer_idx in range(n_layers):
        if len(collector.neuron_selections[layer_idx]) == 0:
            continue
        if len(collector.pattern_selections[layer_idx]) == 0:
            continue

        neuron_sel = collector.neuron_selections[layer_idx]  # [N, S, k]
        pattern_weights = collector.pattern_selections[layer_idx]  # [N, S, n_patterns]

        N, S, k = neuron_sel.shape
        total_samples = N * S

        # 샘플링 (너무 크면)
        if total_samples > sample_size:
            indices = np.random.choice(total_samples, sample_size, replace=False)
        else:
            indices = np.arange(total_samples)

        # Flatten
        neuron_sel_flat = neuron_sel.reshape(-1, k).numpy()[indices]  # [sample, k]
        pattern_sel_flat = pattern_weights.argmax(dim=-1).reshape(-1).numpy()[indices]  # [sample]

        # 뉴런 Set → 패턴 매핑 빈도
        neuron_pattern_map = defaultdict(lambda: defaultdict(int))

        for neuron_set, pattern in zip(neuron_sel_flat, pattern_sel_flat):
            neuron_key = tuple(sorted(neuron_set.tolist()))
            neuron_pattern_map[neuron_key][pattern] += 1

        # 일관성 측정: 같은 뉴런 Set이 항상 같은 패턴 선택?
        consistency_scores = []
        for neuron_key, pattern_counts in neuron_pattern_map.items():
            total = sum(pattern_counts.values())
            if total > 1:  # 2번 이상 나타난 Set만
                max_count = max(pattern_counts.values())
                consistency = max_count / total
                consistency_scores.append(consistency)

        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            median_consistency = np.median(consistency_scores)

            print(f"\nLayer {layer_idx}:")
            print(f"  Unique neuron sets: {len(neuron_pattern_map)}")
            print(f"  Sets appearing >1 time: {len(consistency_scores)}")
            print(f"  Avg consistency: {avg_consistency:.4f}")
            print(f"  Median consistency: {median_consistency:.4f}")

            if avg_consistency > 0.9:
                print(f"  ✅ HIGH: Same neuron sets → same patterns ({avg_consistency:.1%})")
            elif avg_consistency < 0.5:
                print(f"  ⚠️  LOW: Neuron-pattern mapping is inconsistent")

            results[f'layer_{layer_idx}'] = {
                'unique_neuron_sets': len(neuron_pattern_map),
                'repeated_sets': len(consistency_scores),
                'avg_consistency': float(avg_consistency),
                'median_consistency': float(median_consistency),
            }

    return results


def analyze_selection_confidence(collector, n_layers):
    """선택 confidence 분석 (softmax score 분포)"""
    print("\n" + "="*70)
    print("SELECTION CONFIDENCE ANALYSIS")
    print("="*70)

    results = {}

    for layer_idx in range(n_layers):
        if len(collector.pattern_selections[layer_idx]) == 0:
            continue

        pattern_weights = collector.pattern_selections[layer_idx]  # [N, S, n_patterns]

        # Top-k 점수 분석
        top_scores, _ = pattern_weights.topk(k=3, dim=-1)  # [N, S, 3]
        top1_scores = top_scores[:, :, 0].numpy().flatten()
        top2_scores = top_scores[:, :, 1].numpy().flatten()
        top3_scores = top_scores[:, :, 2].numpy().flatten()

        # Gap 분석
        gap_1_2 = top1_scores - top2_scores
        gap_2_3 = top2_scores - top3_scores

        print(f"\nLayer {layer_idx}:")
        print(f"  Top-1 score: {top1_scores.mean():.6f} ± {top1_scores.std():.6f}")
        print(f"  Top-2 score: {top2_scores.mean():.6f} ± {top2_scores.std():.6f}")
        print(f"  Top-3 score: {top3_scores.mean():.6f} ± {top3_scores.std():.6f}")
        print(f"  Gap (1-2): {gap_1_2.mean():.6f} ± {gap_1_2.std():.6f}")
        print(f"  Gap (2-3): {gap_2_3.mean():.6f} ± {gap_2_3.std():.6f}")

        # Confidence 해석
        if gap_1_2.mean() < 0.01:
            print(f"  ⚠️  LOW CONFIDENCE: Top-1/2 gap very small ({gap_1_2.mean():.6f})")
        elif gap_1_2.mean() > 0.1:
            print(f"  ✅ HIGH CONFIDENCE: Clear winner (gap={gap_1_2.mean():.4f})")

        results[f'layer_{layer_idx}'] = {
            'top1_mean': float(top1_scores.mean()),
            'top1_std': float(top1_scores.std()),
            'gap_1_2_mean': float(gap_1_2.mean()),
            'gap_1_2_std': float(gap_1_2.std()),
        }

    return results


def analyze_position_patterns(collector, n_layers, max_positions=128):
    """시퀀스 위치별 뉴런 패턴 분석 ⭐"""
    print("\n" + "="*70)
    print("POSITION-BASED NEURON PATTERNS")
    print("="*70)

    results = {}

    for layer_idx in range(n_layers):
        position_stats = {}

        # 시작 (0-15), 중간 (48-63), 끝 (112-127)
        ranges = {
            'start': (0, 16),
            'middle': (48, 64),
            'end': (112, 128)
        }

        for range_name, (start_pos, end_pos) in ranges.items():
            all_neurons = []
            for pos in range(start_pos, min(end_pos, max_positions)):
                if pos in collector.position_neuron_map:
                    neurons = collector.position_neuron_map[pos][layer_idx]
                    all_neurons.extend(neurons)

            if all_neurons:
                unique_neurons = len(set(all_neurons))
                position_stats[range_name] = unique_neurons
            else:
                position_stats[range_name] = 0

        # 위치별 다양성 차이
        if position_stats['start'] > 0 and position_stats['end'] > 0:
            diversity_change = (position_stats['end'] - position_stats['start']) / position_stats['start']

            print(f"\nLayer {layer_idx}:")
            print(f"  Start (0-15): {position_stats['start']} unique neurons")
            print(f"  Middle (48-63): {position_stats['middle']} unique neurons")
            print(f"  End (112-127): {position_stats['end']} unique neurons")
            print(f"  Change: {diversity_change:+.1%}")

            if abs(diversity_change) > 0.5:
                print(f"  ⚠️  LARGE CHANGE: Position strongly affects neuron selection!")

        results[f'layer_{layer_idx}'] = position_stats

    return results


def analyze_neuron_coactivation(collector, n_neurons, n_layers):
    """뉴런 co-activation 패턴 분석 - 어떤 뉴런들이 함께 선택되나? (병렬 최적화)"""
    print("\n" + "="*70)
    print("NEURON CO-ACTIVATION ANALYSIS")
    print("="*70)

    results = {}

    for layer_idx in range(n_layers):
        if len(collector.neuron_selections[layer_idx]) == 0:
            continue

        selected_neurons = collector.neuron_selections[layer_idx]  # [N, S, k]
        N, S, k = selected_neurons.shape

        # 🚀 Vectorized co-activation counting using one-hot encoding
        # Create one-hot representation: [N, S, n_neurons]
        device = selected_neurons.device
        one_hot = torch.zeros(N, S, n_neurons, dtype=torch.float32, device=device)

        # Scatter ones at selected neuron positions
        one_hot.scatter_(2, selected_neurons, 1.0)

        # Activation counts: sum over batch and sequence
        neuron_activation_count = one_hot.sum(dim=(0, 1))  # [n_neurons]

        # Co-activation matrix: outer product then sum
        # [N, S, n_neurons, 1] @ [N, S, 1, n_neurons] -> [N, S, n_neurons, n_neurons]
        # Then sum over N, S -> [n_neurons, n_neurons]
        coactivation = torch.einsum('nsi,nsj->ij', one_hot, one_hot)  # Faster than matmul

        # Remove self-activation (diagonal)
        coactivation.fill_diagonal_(0)

        # Move to CPU for further processing
        coactivation = coactivation.cpu()
        neuron_activation_count = neuron_activation_count.cpu()

        # Normalize to get co-activation probability P(j | i)
        # Use broadcasting to avoid loop
        coactivation_prob = coactivation / (neuron_activation_count.unsqueeze(1) + 1e-10)

        # Find strong co-activation patterns (vectorized)
        threshold = 0.8
        # Create mask for upper triangle (avoid duplicates)
        mask = torch.triu(torch.ones_like(coactivation_prob, dtype=torch.bool), diagonal=1)
        # Mutual high probability: both P(j|i) and P(i|j) > threshold
        mutual_mask = mask & (coactivation_prob > threshold) & (coactivation_prob.T > threshold)

        # Get indices where condition is true
        strong_indices = torch.nonzero(mutual_mask, as_tuple=False)

        # Extract probabilities
        strong_pairs = []
        for idx in strong_indices:
            i, j = idx[0].item(), idx[1].item()
            prob_ij = coactivation_prob[i, j].item()
            prob_ji = coactivation_prob[j, i].item()
            strong_pairs.append((i, j, prob_ij, prob_ji))

        # Statistics
        active_neurons = (neuron_activation_count > 0).sum().item()
        active_mask = neuron_activation_count > 0
        avg_coactivation = coactivation_prob[active_mask, :][:, active_mask].mean().item() if active_mask.any() else 0.0

        # Save only summary stats (matrices are too large for JSON)
        results[f'layer_{layer_idx}'] = {
            'activation_counts': neuron_activation_count.numpy().tolist(),
            'strong_pairs': strong_pairs[:20],  # Top 20 strong pairs
            'active_neurons': int(active_neurons),
            'avg_coactivation_prob': float(avg_coactivation),
            # Store matrix shape info instead of full matrix
            'coactivation_matrix_shape': list(coactivation.shape),
            'coactivation_prob_shape': list(coactivation_prob.shape),
        }

        print(f"\nLayer {layer_idx}:")
        print(f"  Active neurons: {active_neurons}/{n_neurons}")
        print(f"  Strong co-activation pairs (>{threshold}): {len(strong_pairs)}")
        print(f"  Avg co-activation probability: {avg_coactivation:.4f}")

        # 경고
        if len(strong_pairs) > 50:
            print(f"  ⚠️  STRONG COUPLING: {len(strong_pairs)} neuron pairs almost always activate together!")
            print(f"     → May indicate redundant neurons or over-specialized combinations")

        # Show top 5 strong pairs
        if strong_pairs:
            print(f"  Top 5 co-activation pairs:")
            for idx, (i, j, prob_ij, prob_ji) in enumerate(strong_pairs[:5], 1):
                print(f"    #{idx}: Neurons ({i}, {j}) - P(j|i)={prob_ij:.3f}, P(i|j)={prob_ji:.3f}")

    return results


def analyze_neuron_diversity(model, n_layers):
    """뉴런 간 유사도 및 effective rank 분석 (clustering 포함)"""
    print("\n" + "="*70)
    print("NEURON DIVERSITY ANALYSIS")
    print("="*70)

    from scipy.cluster.hierarchy import linkage

    results = {}

    for layer_idx in range(n_layers):
        if hasattr(model, '_orig_mod'):
            router = model._orig_mod.layers[layer_idx].router
        else:
            router = model.layers[layer_idx].router

        # Get neurons (handle low-rank decomposition)
        if hasattr(router, 'neuron_codes'):
            # v3.2: Low-rank neurons
            neurons = torch.matmul(router.neuron_codes.data, router.neuron_basis.data)
        else:
            # v3.1 and earlier: Full-rank neurons
            neurons = router.neurons.data

        neurons_cpu = neurons.cpu()

        # 정규화
        neurons_norm = F.normalize(neurons, p=2, dim=1)

        # 뉴런 간 유사도 (코사인)
        similarity = torch.matmul(neurons_norm, neurons_norm.T)  # [n_neurons, n_neurons]

        # 자기 자신 제외하고 유사도 계산
        mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
        off_diag_sim = similarity[mask]

        # Find highly similar pairs (vectorized)
        threshold = 0.9
        similarity_cpu = similarity.cpu()
        n_neurons = similarity.shape[0]

        # Create upper triangular mask (avoid duplicates and diagonal)
        triu_mask = torch.triu(torch.ones(n_neurons, n_neurons, dtype=torch.bool), diagonal=1)

        # Find pairs with similarity > threshold
        high_sim_mask = triu_mask & (torch.abs(similarity_cpu) > threshold)
        pair_indices = torch.nonzero(high_sim_mask, as_tuple=False)

        # Extract pairs with their similarity values
        similar_pairs = [
            (i.item(), j.item(), similarity_cpu[i, j].item())
            for i, j in pair_indices
        ]

        # Hierarchical clustering
        try:
            linkage_matrix = linkage(neurons_cpu.numpy(), method='ward')
        except Exception as e:
            print(f"  ⚠️  Clustering failed: {e}")
            linkage_matrix = None

        # Effective rank (SVD 기반)
        U, S, V = torch.svd(neurons)
        # Normalized singular values
        S_normalized = S / S.sum()
        # Entropy-based effective rank
        entropy_val = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
        effective_rank = torch.exp(entropy_val).item()

        # 통계
        mean_sim = off_diag_sim.mean().item()
        max_sim = off_diag_sim.max().item()
        std_sim = off_diag_sim.std().item()

        # Rank ratio
        rank_ratio = effective_rank / neurons.shape[0]

        results[f'layer_{layer_idx}'] = {
            'mean_similarity': mean_sim,
            'max_similarity': max_sim,
            'std_similarity': std_sim,
            'effective_rank': effective_rank,
            'total_neurons': neurons.shape[0],
            'rank_ratio': rank_ratio,
            'similar_pairs': similar_pairs[:20],  # Top 20 similar pairs
            'linkage': linkage_matrix.tolist() if linkage_matrix is not None else None
        }

        print(f"\nLayer {layer_idx}:")
        print(f"  Mean similarity: {mean_sim:.4f}")
        print(f"  Max similarity: {max_sim:.4f}")
        print(f"  Std similarity: {std_sim:.4f}")
        print(f"  Effective rank: {effective_rank:.1f} / {neurons.shape[0]}")
        print(f"  Rank ratio: {rank_ratio:.2%}")
        print(f"  Highly similar pairs (>{threshold}): {len(similar_pairs)}")

        # 경고
        if mean_sim > 0.5:
            print(f"  ⚠️  HIGH SIMILARITY: Neurons are redundant!")
        if rank_ratio < 0.5:
            print(f"  ⚠️  LOW RANK: Limited neuron diversity!")
        if len(similar_pairs) > 50:
            print(f"  ⚠️  REDUNDANCY: {len(similar_pairs)} highly similar neuron pairs!")

    return results




def visualize_neuron_roles(diversity_results, coactivation_results, model, output_dir):
    """뉴런 역할 종합 시각화 (similarity, co-activation, clustering)"""
    print("\n=== Visualizing Neuron Roles ===")

    from scipy.cluster.hierarchy import dendrogram
    import numpy as np

    output_dir = Path(output_dir)
    roles_dir = output_dir / 'neuron_roles'
    roles_dir.mkdir(exist_ok=True)

    n_layers = len([k for k in diversity_results.keys() if k.startswith('layer_')])

    for layer_idx in range(n_layers):
        layer_key = f'layer_{layer_idx}'

        if layer_key not in diversity_results:
            continue

        div_data = diversity_results[layer_key]
        coact_data = coactivation_results.get(layer_key, {})

        # Get neurons
        if hasattr(model, '_orig_mod'):
            router = model._orig_mod.layers[layer_idx].router
        else:
            router = model.layers[layer_idx].router

        if hasattr(router, 'neuron_codes'):
            neurons = torch.matmul(router.neuron_codes.data, router.neuron_basis.data)
        else:
            neurons = router.neurons.data

        neurons_cpu = neurons.cpu()
        neurons_norm = F.normalize(neurons, p=2, dim=1)
        similarity = torch.matmul(neurons_norm, neurons_norm.T).cpu().numpy()

        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 12))

        # 1. Similarity Matrix
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(similarity, cmap='RdBu', vmin=-1, vmax=1)
        ax1.set_title(f'Layer {layer_idx}: Neuron Similarity Matrix', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Neuron ID')
        ax1.set_ylabel('Neuron ID')
        plt.colorbar(im1, ax=ax1, label='Cosine Similarity')

        # Add stats text
        stats_text = f"Mean: {div_data['mean_similarity']:.3f}\n"
        stats_text += f"Effective Rank: {div_data['effective_rank']:.1f}/{div_data['total_neurons']}\n"
        stats_text += f"Rank Ratio: {div_data['rank_ratio']:.2%}"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # 2. Similarity Distribution
        ax2 = plt.subplot(2, 3, 2)
        sim_off_diag = similarity[~np.eye(similarity.shape[0], dtype=bool)]
        ax2.hist(sim_off_diag, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.set_xlabel('Cosine Similarity')
        ax2.set_ylabel('Count')
        ax2.set_title('Similarity Distribution (off-diagonal)', fontsize=12, fontweight='bold')
        ax2.axvline(div_data['mean_similarity'], color='red', linestyle='--',
                   label=f"Mean: {div_data['mean_similarity']:.3f}")
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. Dendrogram (if available)
        ax3 = plt.subplot(2, 3, 3)
        if div_data['linkage'] is not None:
            linkage_matrix = np.array(div_data['linkage'])
            dendrogram(linkage_matrix, ax=ax3, no_labels=True, color_threshold=0)
            ax3.set_title('Hierarchical Clustering', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Distance')
        else:
            ax3.text(0.5, 0.5, 'Clustering unavailable', ha='center', va='center')
            ax3.set_title('Hierarchical Clustering', fontsize=12, fontweight='bold')

        # 4. Co-activation Summary (if available)
        ax4 = plt.subplot(2, 3, 4)
        if coact_data and 'strong_pairs' in coact_data:
            # Since we don't store full matrix, visualize strong pairs as network
            strong_pairs = coact_data.get('strong_pairs', [])

            if strong_pairs:
                # Create sparse representation of strong pairs
                n_neurons = div_data['total_neurons']
                sparse_coact = np.zeros((n_neurons, n_neurons))
                for i, j, prob_ij, prob_ji in strong_pairs:
                    sparse_coact[i, j] = prob_ij
                    sparse_coact[j, i] = prob_ji

                im4 = ax4.imshow(sparse_coact, cmap='YlOrRd', vmin=0, vmax=1)
                ax4.set_title(f'Strong Co-activation Pairs (>{len(strong_pairs)} pairs)', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Neuron j')
                ax4.set_ylabel('Neuron i')
                plt.colorbar(im4, ax=ax4, label='Probability')
            else:
                ax4.text(0.5, 0.5, 'No strong co-activation pairs', ha='center', va='center')
                ax4.set_title('Strong Co-activation Pairs', fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No co-activation data', ha='center', va='center')
            ax4.set_title('Co-activation Probability', fontsize=12, fontweight='bold')

        # 5. Activation Counts (if available)
        ax5 = plt.subplot(2, 3, 5)
        if coact_data and 'activation_counts' in coact_data:
            activation_counts = np.array(coact_data['activation_counts'])
            ax5.bar(range(len(activation_counts)), activation_counts, color='coral', alpha=0.7)
            ax5.set_xlabel('Neuron ID')
            ax5.set_ylabel('Activation Count')
            ax5.set_title('Neuron Activation Frequency', fontsize=12, fontweight='bold')
            ax5.axhline(activation_counts.mean(), color='blue', linestyle='--',
                       label=f"Mean: {activation_counts.mean():.1f}")
            ax5.legend()
            ax5.grid(alpha=0.3, axis='y')
        else:
            ax5.text(0.5, 0.5, 'No activation data', ha='center', va='center')
            ax5.set_title('Neuron Activation Frequency', fontsize=12, fontweight='bold')

        # 6. Singular Value Distribution (Rank Analysis)
        ax6 = plt.subplot(2, 3, 6)
        U, S, V = torch.svd(neurons)
        S_normalized = (S / S.sum()).cpu().numpy()  # Move to CPU before numpy conversion
        ax6.plot(S_normalized, marker='o', markersize=3, alpha=0.7)
        ax6.set_xlabel('Singular Value Index')
        ax6.set_ylabel('Normalized Magnitude')
        ax6.set_title('Singular Value Spectrum', fontsize=12, fontweight='bold')
        ax6.set_yscale('log')
        ax6.grid(alpha=0.3)
        ax6.axhline(1.0/len(S_normalized), color='red', linestyle='--',
                   label=f'Uniform (1/{len(S_normalized)})')
        ax6.legend()

        plt.tight_layout()
        plt.savefig(roles_dir / f'neuron_roles_layer_{layer_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: neuron_roles_layer_{layer_idx}.png")

    # Summary across layers
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers = list(range(n_layers))
    mean_sims = [diversity_results[f'layer_{i}']['mean_similarity'] for i in layers]
    rank_ratios = [diversity_results[f'layer_{i}']['rank_ratio'] for i in layers]
    num_similar_pairs = [len(diversity_results[f'layer_{i}']['similar_pairs']) for i in layers]

    # Mean similarity by layer
    axes[0, 0].plot(layers, mean_sims, marker='o', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Mean Cosine Similarity')
    axes[0, 0].set_title('Neuron Similarity Across Layers', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='High redundancy threshold')
    axes[0, 0].legend()

    # Rank ratio by layer
    axes[0, 1].plot(layers, rank_ratios, marker='s', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Effective Rank Ratio')
    axes[0, 1].set_title('Neuron Diversity Across Layers', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Low diversity threshold')
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])

    # Highly similar pairs
    axes[1, 0].bar(layers, num_similar_pairs, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Number of Pairs')
    axes[1, 0].set_title('Highly Similar Neuron Pairs (>0.9)', fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')

    # Co-activation stats (if available)
    if all(f'layer_{i}' in coactivation_results for i in layers):
        strong_coact_pairs = [len(coactivation_results[f'layer_{i}'].get('strong_pairs', [])) for i in layers]
        axes[1, 1].bar(layers, strong_coact_pairs, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Number of Pairs')
        axes[1, 1].set_title('Strong Co-activation Pairs (>0.8)', fontweight='bold')
        axes[1, 1].grid(alpha=0.3, axis='y')
    else:
        axes[1, 1].text(0.5, 0.5, 'No co-activation data', ha='center', va='center',
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Strong Co-activation Pairs', fontweight='bold')

    plt.tight_layout()
    plt.savefig(roles_dir / 'neuron_roles_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: neuron_roles_summary.png")


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

    # 버전 감지
    checkpoint_version = checkpoint.get('model_version', 'unknown')
    current_version = DAWN.__version__
    print(f"\n📌 Checkpoint version: {checkpoint_version}")
    print(f"📌 Current model version: {current_version}")

    if checkpoint_version == 'unknown':
        print("   ⚠️  Old checkpoint (pre-versioning)")
    elif checkpoint_version != current_version:
        print(f"   ⚠️  Version mismatch - will attempt backward compatible loading")

    # 모델 생성
    print("\nCreating model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

    # Backward compatibility for config parameters
    neuron_k = cfg['model'].get('neuron_k', cfg['model'].get('k', 8))
    pattern_k = cfg['model'].get('pattern_k', 16)

    model = DAWN(
        vocab_size=vocab_size,
        hidden_dim=cfg['model']['d_model'],
        num_layers=cfg['model']['n_layers'],
        n_heads=cfg['model']['n_heads'],
        n_neurons=cfg['model']['n_neurons'],
        n_patterns=cfg['model']['n_patterns'],
        neuron_k=neuron_k,
        pattern_k=pattern_k,
        d_ff=cfg['model'].get('d_ff', None),
        max_seq_len=cfg['model']['max_seq_len'],
        dropout=cfg['model']['dropout']
    )

    # Load state dict (handle torch.compile() prefix)
    state_dict = checkpoint['model_state_dict']

    # Remove _orig_mod. prefix if present (from torch.compile)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Load with strict=False to handle backward compatibility (connection weights)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Check if missing keys are only connection weights (expected for old checkpoints)
    connection_keys = [k for k in missing_keys if 'connection.weight' in k]
    other_missing = [k for k in missing_keys if 'connection.weight' not in k]

    if connection_keys:
        print(f"\n⚠️  Loading checkpoint without inter-layer connections ({len(connection_keys)} layers)")
        print("   Connection weights initialized to zero (equivalent to pre-connection model)")

    if other_missing:
        print(f"\n⚠️  Warning: Missing keys (not connection-related): {other_missing}")

    if unexpected_keys:
        print(f"\n⚠️  Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

    model = model.to(device)
    model.eval()

    n_layers = cfg['model']['n_layers']
    n_neurons = cfg['model']['n_neurons']
    n_patterns = cfg['model']['n_patterns']

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

            logits, all_selected, all_patterns = model(input_ids, return_activations=True)
            collector.collect(input_ids, labels, logits, all_selected, all_patterns)

    collector.finalize()

    # 분석
    neuron_usage_results = analyze_neuron_usage(collector, n_neurons, n_layers)
    token_spec_results = analyze_token_neuron_specialization(
        collector, tokenizer, n_neurons, n_layers, top_k_tokens=100
    )
    layer_diff_results = analyze_layer_differences(neuron_usage_results)
    uncertainty_results = analyze_uncertainty_accuracy(collector, n_layers)

    # ⭐ 새로운 분석
    pattern_usage_results = analyze_pattern_usage(collector, n_patterns, n_layers)
    pattern_collapse_results = analyze_pattern_collapse_detail(collector, n_patterns, n_layers)
    neuron_pattern_corr_results = analyze_neuron_pattern_correlation(collector, n_layers)
    confidence_results = analyze_selection_confidence(collector, n_layers)
    position_pattern_results = analyze_position_patterns(collector, n_layers)

    # 🧬 Neuron diversity 분석
    diversity_results = analyze_neuron_diversity(model, n_layers)

    # 🤝 Co-activation 분석
    coactivation_results = analyze_neuron_coactivation(collector, n_neurons, n_layers)

    all_results = {
        'neuron_usage': neuron_usage_results,
        'token_neuron_specialization': token_spec_results,
        'layer_differences': layer_diff_results,
        'uncertainty_accuracy': uncertainty_results,
        'pattern_usage': pattern_usage_results,  # ⭐ NEW
        'pattern_collapse_detail': pattern_collapse_results,  # ⭐ NEW
        'neuron_pattern_correlation': neuron_pattern_corr_results,  # ⭐ NEW
        'selection_confidence': confidence_results,  # ⭐ NEW
        'position_patterns': position_pattern_results,  # ⭐ NEW
        'neuron_diversity': diversity_results,  # 🧬 NEW
        'neuron_coactivation': coactivation_results,  # 🤝 NEW
    }

    # 저장
    print(f"\nSaving results to: {output_dir}")
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("  ✓ analysis_results.json")

    # 시각화
    visualize_results(neuron_usage_results, output_dir)

    # 뉴런 역할 시각화 (similarity, co-activation, clustering)
    visualize_neuron_roles(diversity_results, coactivation_results, model, output_dir)

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
    print("  - neuron_roles/ (similarity, co-activation, clustering)")


if __name__ == "__main__":
    main()
