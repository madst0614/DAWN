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
    """뉴런 선택 패턴 수집 (v5.0: neuron-only)"""
    def __init__(self, model, n_layers):
        self.model = model
        self.n_layers = n_layers

        # 뉴런 선택 기록
        self.neuron_selections = [[] for _ in range(n_layers)]

        # 위치별 뉴런 선택
        self.position_neuron_map = defaultdict(lambda: [[] for _ in range(n_layers)])

        # 토큰별 뉴런 선택
        self.token_neuron_map = defaultdict(lambda: [[] for _ in range(n_layers)])

        # 예측 정확도별 패턴
        self.correct_selections = [[] for _ in range(n_layers)]
        self.incorrect_selections = [[] for _ in range(n_layers)]

    def collect(self, input_ids, labels, logits, all_selected):
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

            # 2. 토큰별 + 위치별 뉴런 선택 (vectorized)
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

            # 3. 정확도별 뉴런 선택
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
            router = model._orig_mod.layers[layer_idx].neuron_router
        else:
            router = model.layers[layer_idx].neuron_router

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
            router = model._orig_mod.layers[layer_idx].neuron_router
        else:
            router = model.layers[layer_idx].neuron_router

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


# ============================================================
# LAYER BOTTLENECK DIAGNOSIS
# ============================================================

def analyze_layer_bottleneck(model, dataloader, device, num_batches=50, tokenizer=None):
    """Layer별 gradient 및 activation 분석"""
    print("\n" + "="*70)
    print("LAYER BOTTLENECK DIAGNOSIS")
    print("="*70)

    model.train()  # Enable gradients
    stats = {
        'gradients': defaultdict(list),
        'activations': defaultdict(list),
    }

    n_layers = len(model.layers) if hasattr(model, 'layers') else len(model._orig_mod.layers)

    for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Analyzing bottleneck")):
        if batch_idx >= num_batches:
            break

        input_ids = batch['input_ids'].to(device)

        # Apply MLM masking if tokenizer available
        if tokenizer is not None:
            input_ids, labels = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)
        else:
            labels = input_ids.clone()

        model.zero_grad()

        # Forward with losses
        logits, losses = model(input_ids, return_losses=True)

        # Compute loss
        B, S, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B * S, V),
            labels.view(B * S),
            ignore_index=-100
        )

        # Add aux losses
        aux_loss = sum(losses['pattern_load']) + sum(losses['neuron_ortho'])
        total_loss = loss + 0.1 * aux_loss

        # Backward
        total_loss.backward()

        # Collect gradients for each layer
        for layer_idx in range(n_layers):
            if hasattr(model, '_orig_mod'):
                layer = model._orig_mod.layers[layer_idx]
            else:
                layer = model.layers[layer_idx]

            # Router gradients
            if layer.neuron_router.neurons.grad is not None:
                router_grad = layer.neuron_router.neurons.grad.norm().item()
            else:
                router_grad = 0.0

            # Pattern gradients
            if layer.neuron_interaction.pattern_queries.grad is not None:
                pattern_grad = layer.neuron_interaction.pattern_queries.grad.norm().item()
            else:
                pattern_grad = 0.0

            stats['gradients'][layer_idx].append({
                'router': router_grad,
                'pattern': pattern_grad
            })

    model.eval()  # Back to eval

    # Analyze results
    results = {}
    print(f"\nGradient Flow Analysis:")
    print("-" * 70)

    for layer_idx in range(n_layers):
        router_grads = [s['router'] for s in stats['gradients'][layer_idx]]
        pattern_grads = [s['pattern'] for s in stats['gradients'][layer_idx]]

        avg_router = np.mean(router_grads)
        avg_pattern = np.mean(pattern_grads)

        print(f"\nLayer {layer_idx}:")
        print(f"  Router grad:  {avg_router:.6f}")
        print(f"  Pattern grad: {avg_pattern:.6f}")

        if avg_router < 0.001:
            print(f"  🔴 WEAK GRADIENT: Router gradient very small!")
        if avg_pattern < 0.001:
            print(f"  🔴 WEAK GRADIENT: Pattern gradient very small!")

        results[f'layer_{layer_idx}'] = {
            'router_grad_mean': float(avg_router),
            'router_grad_std': float(np.std(router_grads)),
            'pattern_grad_mean': float(avg_pattern),
            'pattern_grad_std': float(np.std(pattern_grads)),
        }

    # Compare across layers
    print(f"\n{'='*70}")
    print("Gradient Flow Comparison:")
    print("-" * 70)

    router_means = [results[f'layer_{i}']['router_grad_mean'] for i in range(n_layers)]
    max_grad = max(router_means) if max(router_means) > 0 else 1.0

    for layer_idx in range(n_layers):
        ratio = router_means[layer_idx] / max_grad
        bar = "█" * int(ratio * 50)
        marker = " 🔴" if ratio < 0.3 else ""
        print(f"  L{layer_idx}: {ratio:.3f} {bar}{marker}")

    return results


def analyze_information_flow(model, dataloader, device, num_batches=50):
    """Layer간 정보 흐름 분석 (similarity, change, rank)"""
    print("\n" + "="*70)
    print("INFORMATION FLOW ANALYSIS")
    print("="*70)

    model.eval()

    n_layers = len(model.layers) if hasattr(model, 'layers') else len(model._orig_mod.layers)
    d_model = model.d_model if hasattr(model, 'd_model') else model._orig_mod.d_model

    # Collect representations from each layer
    representations = [[] for _ in range(n_layers + 1)]  # +1 for input

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Collecting representations")):
            if batch_idx >= num_batches:
                break

            input_ids = batch['input_ids'].to(device)
            B, S = input_ids.shape

            # Get model reference
            if hasattr(model, '_orig_mod'):
                m = model._orig_mod
            else:
                m = model

            # Embedding
            pos = torch.arange(S, device=device).unsqueeze(0)
            x = m.token_emb(input_ids) + m.pos_emb(pos)
            x = m.dropout(x)

            representations[0].append(x.cpu())

            # Causal mask
            mask = torch.tril(torch.ones(S, S, device=device)).unsqueeze(0).unsqueeze(0)

            # Through layers
            for layer_idx, layer in enumerate(m.layers):
                x, _ = layer(x, mask)
                representations[layer_idx + 1].append(x.cpu())

    # Stack representations
    for i in range(n_layers + 1):
        representations[i] = torch.cat(representations[i], dim=0)  # [N, S, D]

    # Analysis
    results = {}

    # 1. Cosine similarity to input
    print(f"\nCosine Similarity to Input:")
    print("-" * 70)

    x0_flat = representations[0].view(-1, d_model)

    for layer_idx in range(1, n_layers + 1):
        x_flat = representations[layer_idx].view(-1, d_model)

        # Sample if too large
        if x0_flat.shape[0] > 10000:
            idx = torch.randperm(x0_flat.shape[0])[:10000]
            x0_sample = x0_flat[idx]
            x_sample = x_flat[idx]
        else:
            x0_sample = x0_flat
            x_sample = x_flat

        cos_sim = F.cosine_similarity(x0_sample, x_sample, dim=-1).mean().item()

        bar = "█" * int(cos_sim * 50)
        marker = " 🔴" if cos_sim < 0.5 else ""
        print(f"  After L{layer_idx-1}: {cos_sim:.3f} {bar}{marker}")

        results[f'layer_{layer_idx-1}_input_sim'] = float(cos_sim)

    # 2. Layer-to-layer change
    print(f"\nLayer-to-Layer Change (L2 distance):")
    print("-" * 70)

    for layer_idx in range(n_layers):
        x_prev = representations[layer_idx].view(-1, d_model)
        x_next = representations[layer_idx + 1].view(-1, d_model)

        # Sample if too large
        if x_prev.shape[0] > 10000:
            idx = torch.randperm(x_prev.shape[0])[:10000]
            x_prev = x_prev[idx]
            x_next = x_next[idx]

        change = ((x_next - x_prev) ** 2).sum(dim=-1).sqrt().mean().item()

        bar = "█" * int(min(change / 5.0, 1.0) * 50)
        marker = " 🔴" if change > 8.0 else " ⚠️" if change > 5.0 else ""
        print(f"  L{layer_idx}: {change:.3f} {bar}{marker}")

        results[f'layer_{layer_idx}_change'] = float(change)

    # 3. Effective rank
    print(f"\nEffective Rank (representation diversity):")
    print("-" * 70)

    for layer_idx in range(n_layers + 1):
        x_flat = representations[layer_idx].view(-1, d_model)

        # Sample for efficiency
        if x_flat.shape[0] > 5000:
            idx = torch.randperm(x_flat.shape[0])[:5000]
            x_flat = x_flat[idx]

        # Compute covariance
        x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        cov = torch.mm(x_centered.T, x_centered) / x_centered.shape[0]

        # Eigenvalues
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-8]

        if len(eigenvalues) == 0:
            eff_rank = 0.0
            rank_ratio = 0.0
        else:
            # Normalize
            eigenvalues = eigenvalues / eigenvalues.sum()

            # Effective rank
            entropy_val = -(eigenvalues * torch.log(eigenvalues + 1e-10)).sum()
            eff_rank = torch.exp(entropy_val).item()
            rank_ratio = eff_rank / d_model

        bar = "█" * int(rank_ratio * 50)
        marker = " 🔴" if rank_ratio < 0.3 else " ⚠️" if rank_ratio < 0.5 else ""
        name = "Input" if layer_idx == 0 else f"L{layer_idx-1}"
        print(f"  {name:>5s}: {rank_ratio:.3f} ({eff_rank:.1f}/{d_model}) {bar}{marker}")

        key = 'input_rank' if layer_idx == 0 else f'layer_{layer_idx-1}_rank'
        results[key] = {
            'effective_rank': float(eff_rank),
            'rank_ratio': float(rank_ratio)
        }

    return results


def analyze_task_pressure(model, dataloader, device, num_batches=30, tokenizer=None):
    """각 레이어의 task pressure 측정 (ablation study)"""
    print("\n" + "="*70)
    print("TASK PRESSURE ANALYSIS (Ablation Study)")
    print("="*70)

    model.eval()

    n_layers = len(model.layers) if hasattr(model, 'layers') else len(model._orig_mod.layers)
    layer_contributions = defaultdict(list)

    # Get model reference
    if hasattr(model, '_orig_mod'):
        m = model._orig_mod
    else:
        m = model

    # Baseline: all layers
    print("\nComputing baseline (all layers)...")
    baseline_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            input_ids = batch['input_ids'].to(device)

            # Apply MLM masking if tokenizer available
            if tokenizer is not None:
                input_ids, labels = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)
            else:
                labels = input_ids.clone()

            logits = model(input_ids)
            B, S, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * S, V),
                labels.view(B * S),
                ignore_index=-100
            ).item()
            baseline_losses.append(loss)

    baseline_loss = np.mean(baseline_losses)
    print(f"Baseline loss: {baseline_loss:.4f}")

    # Ablate each layer
    print("\nAblating each layer (identity function)...")

    for layer_idx in tqdm(range(n_layers), desc="Layer ablation"):
        # Save original forward
        original_forward = m.layers[layer_idx].forward

        # Define identity forward
        def identity_forward(x, mask=None, **kwargs):
            # Return x unchanged, with dummy outputs
            return x, None

        # Replace with identity
        m.layers[layer_idx].forward = identity_forward

        # Measure loss
        ablated_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(device)

                # Apply MLM masking if tokenizer available
                if tokenizer is not None:
                    input_ids, labels = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)
                else:
                    labels = input_ids.clone()

                logits = model(input_ids)
                B, S, V = logits.shape
                loss = F.cross_entropy(
                    logits.view(B * S, V),
                    labels.view(B * S),
                    ignore_index=-100
                ).item()
                ablated_losses.append(loss)

        # Restore original
        m.layers[layer_idx].forward = original_forward

        ablated_loss = np.mean(ablated_losses)
        contribution = ablated_loss - baseline_loss
        layer_contributions[layer_idx] = contribution

    # Results
    print(f"\n{'='*70}")
    print("Layer Importance (loss increase when removed):")
    print("-" * 70)

    max_contrib = max(layer_contributions.values()) if layer_contributions else 1.0

    results = {}
    for layer_idx in range(n_layers):
        contrib = layer_contributions[layer_idx]
        importance = contrib / max_contrib if max_contrib > 0 else 0

        bar = "█" * int(importance * 50)
        marker = " 🔥" if importance > 0.8 else " ⚠️" if importance > 0.5 else ""

        print(f"  L{layer_idx}: {contrib:+.4f} {bar}{marker}")

        results[f'layer_{layer_idx}'] = {
            'loss_increase': float(contrib),
            'importance': float(importance)
        }

    return results


def analyze_gradient_flow_detailed(model, dataloader, device, num_batches=30, tokenizer=None):
    """파라미터 수로 정규화된 gradient 분석"""
    print("\n" + "="*70)
    print("DETAILED GRADIENT FLOW ANALYSIS (Normalized by Param Count)")
    print("="*70)

    model.train()

    n_layers = len(model.layers) if hasattr(model, 'layers') else len(model._orig_mod.layers)

    # Collect gradients
    router_grads = [[] for _ in range(n_layers)]
    pattern_grads = [[] for _ in range(n_layers)]
    ffn_grads = [[] for _ in range(n_layers)]

    for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Collecting detailed gradients")):
        if batch_idx >= num_batches:
            break

        input_ids = batch['input_ids'].to(device)

        if tokenizer is not None:
            input_ids, labels = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)
        else:
            labels = input_ids.clone()

        model.zero_grad()

        # Forward
        logits, losses = model(input_ids, return_losses=True)

        # Loss
        B, S, V = logits.shape
        ce_loss = F.cross_entropy(
            logits.view(B * S, V),
            labels.view(B * S),
            ignore_index=-100
        )

        # With regularization
        pattern_load_loss = sum(losses['pattern_load'])
        neuron_ortho_loss = sum(losses['neuron_ortho'])
        total_loss = ce_loss + 0.1 * pattern_load_loss + 0.01 * neuron_ortho_loss

        total_loss.backward()

        # Collect per-layer gradients
        for layer_idx in range(n_layers):
            if hasattr(model, '_orig_mod'):
                layer = model._orig_mod.layers[layer_idx]
            else:
                layer = model.layers[layer_idx]

            # Router gradients (all params)
            router_grad_list = [
                p.grad.flatten()
                for p in layer.neuron_router.parameters()
                if p.grad is not None
            ]
            if router_grad_list:
                router_grad = torch.cat(router_grad_list).norm().item()
            else:
                router_grad = 0.0

            # Pattern query gradients specifically
            if layer.neuron_interaction.pattern_queries.grad is not None:
                pattern_grad = layer.neuron_interaction.pattern_queries.grad.norm().item()
            else:
                pattern_grad = 0.0

            # All FFN gradients
            ffn_grad_list = [
                p.grad.flatten()
                for p in layer.neuron_interaction.parameters()
                if p.grad is not None
            ]
            if ffn_grad_list:
                ffn_grad = torch.cat(ffn_grad_list).norm().item()
            else:
                ffn_grad = 0.0

            router_grads[layer_idx].append(router_grad)
            pattern_grads[layer_idx].append(pattern_grad)
            ffn_grads[layer_idx].append(ffn_grad)

    model.eval()

    # Analysis
    results = {}

    print("\n" + "-" * 70)
    print("Component Gradients (averaged across batches):")
    print("-" * 70)

    for layer_idx in range(n_layers):
        if hasattr(model, '_orig_mod'):
            layer = model._orig_mod.layers[layer_idx]
        else:
            layer = model.layers[layer_idx]

        # Parameter counts
        router_params = sum(p.numel() for p in layer.neuron_router.parameters())
        pattern_query_params = layer.neuron_interaction.pattern_queries.numel()
        ffn_params = sum(p.numel() for p in layer.neuron_interaction.parameters())

        # Average gradients
        avg_router_grad = np.mean(router_grads[layer_idx])
        avg_pattern_grad = np.mean(pattern_grads[layer_idx])
        avg_ffn_grad = np.mean(ffn_grads[layer_idx])

        # Normalized by param count (gradient per parameter)
        norm_router = avg_router_grad / router_params if router_params > 0 else 0
        norm_pattern = avg_pattern_grad / pattern_query_params if pattern_query_params > 0 else 0
        norm_ffn = avg_ffn_grad / ffn_params if ffn_params > 0 else 0

        print(f"\nLayer {layer_idx}:")
        print(f"  Router:")
        print(f"    Grad norm: {avg_router_grad:.6f}")
        print(f"    Params: {router_params:,}")
        print(f"    Grad/param: {norm_router:.9f}")
        print(f"  Pattern queries:")
        print(f"    Grad norm: {avg_pattern_grad:.6f}")
        print(f"    Params: {pattern_query_params:,}")
        print(f"    Grad/param: {norm_pattern:.9f}")
        print(f"  Full FFN:")
        print(f"    Grad norm: {avg_ffn_grad:.6f}")
        print(f"    Params: {ffn_params:,}")
        print(f"    Grad/param: {norm_ffn:.9f}")

        # Ratio analysis
        if norm_router > 0:
            pattern_to_router_ratio = norm_pattern / norm_router
            print(f"  Pattern/Router ratio: {pattern_to_router_ratio:.4f}")

            if pattern_to_router_ratio < 0.01:
                print(f"    🔴 Pattern gradient is <1% of Router!")
            elif pattern_to_router_ratio < 0.1:
                print(f"    ⚠️  Pattern gradient is <10% of Router")

        results[f'layer_{layer_idx}'] = {
            'router_grad': float(avg_router_grad),
            'router_params': int(router_params),
            'router_grad_per_param': float(norm_router),
            'pattern_grad': float(avg_pattern_grad),
            'pattern_params': int(pattern_query_params),
            'pattern_grad_per_param': float(norm_pattern),
            'ffn_grad': float(avg_ffn_grad),
            'ffn_params': int(ffn_params),
            'ffn_grad_per_param': float(norm_ffn),
        }

    return results


def analyze_pattern_necessity(model, dataloader, device, tokenizer=None, num_batches=50):
    """패턴이 실제로 필요한지 ablation 테스트"""
    print("\n" + "="*70)
    print("PATTERN NECESSITY ANALYSIS (Ablation Test)")
    print("="*70)

    model.eval()

    if hasattr(model, '_orig_mod'):
        m = model._orig_mod
    else:
        m = model

    # Save original pattern queries
    original_queries = [
        layer.neuron_interaction.pattern_queries.data.clone()
        for layer in m.layers
    ]

    def evaluate_loss():
        """Evaluate model loss"""
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(device)

                if tokenizer is not None:
                    input_ids, labels = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)
                else:
                    labels = input_ids.clone()

                logits = model(input_ids)
                B, S, V = logits.shape

                loss = F.cross_entropy(
                    logits.view(B * S, V),
                    labels.view(B * S),
                    ignore_index=-100
                )

                total_loss += loss.item() * B * S
                total_tokens += B * S

        return total_loss / total_tokens if total_tokens > 0 else 0.0

    # 1. Normal loss
    print("\n1. Normal model (all patterns active)...")
    normal_loss = evaluate_loss()
    print(f"   Loss: {normal_loss:.4f}")

    # 2. Uniform patterns (all same)
    print("\n2. Uniform patterns (all identical)...")
    for layer in m.layers:
        layer.neuron_interaction.pattern_queries.data.fill_(0.01)  # Small uniform value

    uniform_loss = evaluate_loss()
    diff_uniform = uniform_loss - normal_loss
    print(f"   Loss: {uniform_loss:.4f} (Δ = {diff_uniform:+.4f})")

    # 3. Single pattern only
    print("\n3. Single pattern per layer...")
    for layer_idx, layer in enumerate(m.layers):
        layer.neuron_interaction.pattern_queries.data.zero_()
        # Keep only first pattern
        layer.neuron_interaction.pattern_queries.data[0] = original_queries[layer_idx][0]

    single_loss = evaluate_loss()
    diff_single = single_loss - normal_loss
    print(f"   Loss: {single_loss:.4f} (Δ = {diff_single:+.4f})")

    # 4. Random patterns
    print("\n4. Random patterns (reinitialize)...")
    for layer in m.layers:
        layer.neuron_interaction.pattern_queries.data.normal_(0, 0.02)

    random_loss = evaluate_loss()
    diff_random = random_loss - normal_loss
    print(f"   Loss: {random_loss:.4f} (Δ = {diff_random:+.4f})")

    # Restore original
    for layer_idx, layer in enumerate(m.layers):
        layer.neuron_interaction.pattern_queries.data.copy_(original_queries[layer_idx])

    # Analysis
    print(f"\n{'='*70}")
    print("Pattern Necessity Summary:")
    print("-" * 70)

    if abs(diff_uniform) < 0.01:
        print("🔴 CRITICAL: Uniform patterns ≈ normal loss!")
        print("   → Patterns are NOT being used effectively")
        print("   → Model can work WITHOUT pattern selection")
    elif abs(diff_uniform) < 0.05:
        print("⚠️  WARNING: Uniform patterns only slightly worse")
        print("   → Pattern selection has minimal impact")
    else:
        print("✅ Patterns are necessary (uniform degrades loss)")

    if abs(diff_single) < 0.01:
        print("\n🔴 CRITICAL: Single pattern ≈ normal loss!")
        print("   → Only 1 pattern needed per layer")
        print("   → Pattern collapse is EXPECTED behavior")
    elif abs(diff_single) < 0.05:
        print("\n⚠️  WARNING: Single pattern almost sufficient")
        print("   → Pattern diversity may not be needed")
    else:
        print("\n✅ Multiple patterns needed (single pattern degrades)")

    if abs(diff_random) < 0.5:
        print("\n⚠️  WARNING: Random patterns not much worse")
        print("   → Trained patterns barely better than random")
        print("   → Patterns may not be learning")
    else:
        print("\n✅ Trained patterns are learned (random is much worse)")

    results = {
        'normal_loss': float(normal_loss),
        'uniform_loss': float(uniform_loss),
        'uniform_diff': float(diff_uniform),
        'single_loss': float(single_loss),
        'single_diff': float(diff_single),
        'random_loss': float(random_loss),
        'random_diff': float(diff_random),
    }

    return results


def analyze_regularization_impact(model, dataloader, device, tokenizer=None, num_batches=50):
    """Regularization이 실제로 loss에 얼마나 영향 주는지 분석"""
    print("\n" + "="*70)
    print("REGULARIZATION IMPACT ANALYSIS")
    print("="*70)

    model.eval()

    # Collect losses
    ce_losses = []
    pattern_load_losses = []
    neuron_ortho_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Analyzing regularization")):
            if batch_idx >= num_batches:
                break

            input_ids = batch['input_ids'].to(device)

            if tokenizer is not None:
                input_ids, labels = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)
            else:
                labels = input_ids.clone()

            logits, losses = model(input_ids, return_losses=True)

            B, S, V = logits.shape
            ce_loss = F.cross_entropy(
                logits.view(B * S, V),
                labels.view(B * S),
                ignore_index=-100
            ).item()

            pattern_load_loss = sum(l.item() for l in losses['pattern_load'])
            neuron_ortho_loss = sum(l.item() for l in losses['neuron_ortho'])

            ce_losses.append(ce_loss)
            pattern_load_losses.append(pattern_load_loss)
            neuron_ortho_losses.append(neuron_ortho_loss)

    # Average
    avg_ce = np.mean(ce_losses)
    avg_load = np.mean(pattern_load_losses)
    avg_ortho = np.mean(neuron_ortho_losses)

    # With current weights
    current_weight_load = 0.1
    current_weight_ortho = 0.01

    weighted_load = current_weight_load * avg_load
    weighted_ortho = current_weight_ortho * avg_ortho
    total_loss = avg_ce + weighted_load + weighted_ortho

    print(f"\nLoss Components (averaged over {num_batches} batches):")
    print("-" * 70)
    print(f"  CE Loss:              {avg_ce:.4f}")
    print(f"  Pattern Load Loss:    {avg_load:.4f} (raw)")
    print(f"    → Weighted (0.1):   {weighted_load:.4f} ({weighted_load/total_loss*100:.1f}% of total)")
    print(f"  Neuron Ortho Loss:    {avg_ortho:.4f} (raw)")
    print(f"    → Weighted (0.01):  {weighted_ortho:.4f} ({weighted_ortho/total_loss*100:.1f}% of total)")
    print(f"  Total Loss:           {total_loss:.4f}")

    # Impact analysis
    print(f"\n{'='*70}")
    print("Regularization Impact:")
    print("-" * 70)

    load_ratio = weighted_load / avg_ce
    ortho_ratio = weighted_ortho / avg_ce

    print(f"  Load balancing / CE:  {load_ratio:.4f} ({load_ratio*100:.2f}%)")
    print(f"  Orthogonality / CE:   {ortho_ratio:.4f} ({ortho_ratio*100:.2f}%)")

    if load_ratio < 0.01:
        print(f"\n  🔴 Load balancing is <1% of CE loss!")
        print(f"     → Too weak to enforce pattern diversity")
        print(f"     → Consider increasing weight: 0.1 → {0.1 * 10:.1f}")
    elif load_ratio < 0.05:
        print(f"\n  ⚠️  Load balancing is <5% of CE loss")
        print(f"     → May be too weak")
        print(f"     → Consider increasing weight: 0.1 → {0.1 * 2:.1f}")
    else:
        print(f"\n  ✅ Load balancing has significant impact")

    if ortho_ratio < 0.001:
        print(f"\n  🔴 Orthogonality is <0.1% of CE loss!")
        print(f"     → Too weak to enforce neuron diversity")
        print(f"     → Consider increasing weight: 0.01 → {0.01 * 10:.2f}")
    elif ortho_ratio < 0.01:
        print(f"\n  ⚠️  Orthogonality is <1% of CE loss")
        print(f"     → May be too weak")
        print(f"     → Consider increasing weight: 0.01 → {0.01 * 2:.2f}")
    else:
        print(f"\n  ✅ Orthogonality has significant impact")

    # Recommendations
    print(f"\n{'='*70}")
    print("Recommendations:")
    print("-" * 70)

    if load_ratio < 0.05:
        # Calculate needed weight
        target_ratio = 0.05
        needed_weight = (target_ratio * avg_ce) / avg_load if avg_load > 0 else 0.1
        print(f"  → Increase load balancing weight to {needed_weight:.2f}")
        print(f"    (to reach {target_ratio*100:.0f}% of CE loss)")

    if ortho_ratio < 0.01:
        target_ratio = 0.01
        needed_weight = (target_ratio * avg_ce) / avg_ortho if avg_ortho > 0 else 0.01
        print(f"  → Increase orthogonality weight to {needed_weight:.3f}")
        print(f"    (to reach {target_ratio*100:.0f}% of CE loss)")

    results = {
        'ce_loss': float(avg_ce),
        'pattern_load_loss_raw': float(avg_load),
        'pattern_load_loss_weighted': float(weighted_load),
        'neuron_ortho_loss_raw': float(avg_ortho),
        'neuron_ortho_loss_weighted': float(weighted_ortho),
        'total_loss': float(total_loss),
        'load_to_ce_ratio': float(load_ratio),
        'ortho_to_ce_ratio': float(ortho_ratio),
    }

    return results


def analyze_pattern_diversity(model, n_layers):
    """패턴 간 실제 차이 분석 - 패턴들이 진짜 다른가?"""
    print("\n" + "="*70)
    print("PATTERN DIVERSITY ANALYSIS (Are Patterns Actually Different?)")
    print("="*70)

    if hasattr(model, '_orig_mod'):
        m = model._orig_mod
    else:
        m = model

    results = {}

    for layer_idx in range(n_layers):
        layer = m.layers[layer_idx]
        patterns = layer.neuron_interaction.pattern_queries.data  # [n_patterns, d_model]

        # Cosine similarity matrix
        patterns_norm = F.normalize(patterns, dim=1)
        similarity = torch.mm(patterns_norm, patterns_norm.T)  # [n_patterns, n_patterns]

        # 대각선 제외
        n_patterns = patterns.shape[0]
        mask = ~torch.eye(n_patterns, dtype=torch.bool, device=similarity.device)
        off_diag = similarity[mask]

        avg_sim = off_diag.mean().item()
        max_sim = off_diag.max().item()
        min_sim = off_diag.min().item()
        std_sim = off_diag.std().item()

        # Effective diversity (1 - avg similarity)
        eff_diversity = 1 - avg_sim

        # Count highly similar pairs
        high_sim_threshold = 0.9
        high_sim_pairs = (off_diag > high_sim_threshold).sum().item()

        print(f"\nLayer {layer_idx}:")
        print(f"  Pattern similarity (off-diagonal):")
        print(f"    Mean: {avg_sim:.4f}")
        print(f"    Max:  {max_sim:.4f}")
        print(f"    Min:  {min_sim:.4f}")
        print(f"    Std:  {std_sim:.4f}")
        print(f"  Effective diversity: {eff_diversity:.4f}")
        print(f"  Highly similar pairs (>0.9): {high_sim_pairs}/{len(off_diag)}")

        # Warnings
        if avg_sim > 0.9:
            print(f"  🔴 CRITICAL: Patterns are almost identical (avg sim {avg_sim:.3f})")
            print(f"     → No diversity from initialization")
        elif avg_sim > 0.7:
            print(f"  ⚠️  WARNING: Patterns are very similar (avg sim {avg_sim:.3f})")
        else:
            print(f"  ✅ Patterns have reasonable diversity")

        results[f'layer_{layer_idx}'] = {
            'mean_similarity': float(avg_sim),
            'max_similarity': float(max_sim),
            'min_similarity': float(min_sim),
            'std_similarity': float(std_sim),
            'effective_diversity': float(eff_diversity),
            'high_sim_pairs': int(high_sim_pairs),
            'total_pairs': int(len(off_diag)),
        }

    return results


def analyze_neuron_pattern_mapping_quality(model, dataloader, device, n_layers, tokenizer=None, num_batches=50):
    """뉴런 → 패턴 매핑의 일관성 분석"""
    print("\n" + "="*70)
    print("NEURON-PATTERN MAPPING QUALITY")
    print("="*70)

    model.eval()

    # Collect neuron-pattern mappings
    neuron_to_patterns = [defaultdict(lambda: defaultdict(int)) for _ in range(n_layers)]

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Collecting neuron-pattern mappings")):
            if batch_idx >= num_batches:
                break

            input_ids = batch['input_ids'].to(device)

            if tokenizer is not None:
                input_ids, _ = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)

            _, selected_neurons = model(input_ids, return_activations=True)

            for layer_idx in range(n_layers):
                neurons = selected_neurons[layer_idx]  # [B, S, k]
                patterns = pattern_weights[layer_idx]  # [B, S, n_patterns]

                B, S, k = neurons.shape

                for b in range(B):
                    for s in range(S):
                        # 뉴런 조합을 tuple로
                        neuron_set = tuple(sorted(neurons[b, s].cpu().tolist()))

                        # 선택된 패턴 (top-1)
                        pattern_id = patterns[b, s].argmax().item()

                        neuron_to_patterns[layer_idx][neuron_set][pattern_id] += 1

    # Analysis
    results = {}

    print("\n" + "-" * 70)
    print("Neuron Set → Pattern Mapping Consistency:")
    print("-" * 70)

    for layer_idx in range(n_layers):
        mappings = neuron_to_patterns[layer_idx]

        if not mappings:
            continue

        # 각 뉴런 조합이 몇 개의 다른 패턴을 선택하나?
        pattern_counts = []
        consistencies = []

        for neuron_set, pattern_dist in mappings.items():
            num_patterns = len(pattern_dist)
            pattern_counts.append(num_patterns)

            # Consistency: 가장 많이 선택된 패턴의 비율
            total = sum(pattern_dist.values())
            max_count = max(pattern_dist.values())
            consistency = max_count / total if total > 0 else 0
            consistencies.append(consistency)

        avg_patterns = np.mean(pattern_counts)
        max_patterns = np.max(pattern_counts)

        # 1:1 매핑 비율
        one_to_one = sum(1 for c in pattern_counts if c == 1)
        one_to_one_ratio = one_to_one / len(mappings) if len(mappings) > 0 else 0

        avg_consistency = np.mean(consistencies)

        print(f"\nLayer {layer_idx}:")
        print(f"  Unique neuron sets: {len(mappings):,}")
        print(f"  Avg patterns per neuron set: {avg_patterns:.2f}")
        print(f"  Max patterns per neuron set: {max_patterns}")
        print(f"  One-to-one mappings: {one_to_one:,}/{len(mappings):,} ({one_to_one_ratio*100:.1f}%)")
        print(f"  Avg consistency: {avg_consistency:.4f}")

        # Interpretation
        if one_to_one_ratio > 0.99:
            print(f"  🔴 DETERMINISTIC: >99% one-to-one mapping!")
            print(f"     → Patterns completely determined by neurons")
            print(f"     → No context-dependent selection")
        elif one_to_one_ratio > 0.9:
            print(f"  ⚠️  HIGHLY DETERMINISTIC: {one_to_one_ratio*100:.1f}% one-to-one")
        elif avg_consistency > 0.9:
            print(f"  ⚠️  MOSTLY CONSISTENT: Same neurons → mostly same pattern")
        else:
            print(f"  ✅ Context-dependent: Same neurons can select different patterns")

        results[f'layer_{layer_idx}'] = {
            'unique_neuron_sets': int(len(mappings)),
            'avg_patterns_per_set': float(avg_patterns),
            'max_patterns_per_set': int(max_patterns),
            'one_to_one_mappings': int(one_to_one),
            'one_to_one_ratio': float(one_to_one_ratio),
            'avg_consistency': float(avg_consistency),
        }

    return results


def analyze_pattern_ffn_impact(model, dataloader, device, n_layers, tokenizer=None, num_batches=30):
    """패턴이 FFN 출력에 미치는 실제 영향 측정"""
    print("\n" + "="*70)
    print("PATTERN FFN IMPACT ANALYSIS")
    print("="*70)

    model.eval()

    if hasattr(model, '_orig_mod'):
        m = model._orig_mod
    else:
        m = model

    # Save original pattern queries
    original_queries = [
        layer.neuron_interaction.pattern_queries.data.clone()
        for layer in m.layers
    ]

    def measure_output_diff(mode='uniform'):
        """Measure output difference with modified patterns"""
        total_diff = 0.0
        total_norm = 0.0
        layer_diffs = [0.0 for _ in range(n_layers)]
        layer_norms = [0.0 for _ in range(n_layers)]

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(device)

                if tokenizer is not None:
                    input_ids, _ = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)

                # Normal forward
                logits_normal = model(input_ids)

                # Modified patterns
                if mode == 'uniform':
                    for layer in m.layers:
                        layer.neuron_interaction.pattern_queries.data.fill_(0.01)
                elif mode == 'random':
                    for layer in m.layers:
                        layer.neuron_interaction.pattern_queries.data.normal_(0, 0.02)

                logits_modified = model(input_ids)

                # Restore
                for layer, orig in zip(m.layers, original_queries):
                    layer.neuron_interaction.pattern_queries.data.copy_(orig)

                # Measure difference
                diff = (logits_normal - logits_modified).abs().mean().item()
                norm = logits_normal.abs().mean().item()

                total_diff += diff
                total_norm += norm

        avg_diff = total_diff / num_batches
        avg_norm = total_norm / num_batches
        relative_diff = avg_diff / avg_norm if avg_norm > 0 else 0

        return avg_diff, relative_diff

    # Test different modifications
    print("\nLogits change when patterns are modified:")
    print("-" * 70)

    uniform_diff, uniform_rel = measure_output_diff('uniform')
    print(f"\nUniform patterns:")
    print(f"  Absolute diff: {uniform_diff:.6f}")
    print(f"  Relative diff: {uniform_rel*100:.2f}%")

    if uniform_rel < 0.01:
        print(f"  🔴 CRITICAL: <1% change - patterns have NO effect!")
    elif uniform_rel < 0.05:
        print(f"  ⚠️  WARNING: <5% change - patterns have minimal effect")
    else:
        print(f"  ✅ Patterns have significant effect")

    random_diff, random_rel = measure_output_diff('random')
    print(f"\nRandom patterns:")
    print(f"  Absolute diff: {random_diff:.6f}")
    print(f"  Relative diff: {random_rel*100:.2f}%")

    # Layer-wise analysis
    print(f"\n{'='*70}")
    print("Layer-wise Pattern Impact:")
    print("-" * 70)

    layer_impacts = []

    for ablate_layer_idx in range(n_layers):
        # Uniform only this layer
        for layer_idx, layer in enumerate(m.layers):
            if layer_idx == ablate_layer_idx:
                layer.neuron_interaction.pattern_queries.data.fill_(0.01)

        layer_diff, layer_rel = measure_output_diff('uniform')

        # Restore
        for layer, orig in zip(m.layers, original_queries):
            layer.neuron_interaction.pattern_queries.data.copy_(orig)

        layer_impacts.append(layer_rel)

        print(f"  L{ablate_layer_idx}: {layer_rel*100:.2f}% change")

    # Identify most important layer
    max_impact_layer = np.argmax(layer_impacts)
    print(f"\n  Most impactful layer: L{max_impact_layer} ({layer_impacts[max_impact_layer]*100:.2f}%)")

    results = {
        'uniform_absolute_diff': float(uniform_diff),
        'uniform_relative_diff': float(uniform_rel),
        'random_absolute_diff': float(random_diff),
        'random_relative_diff': float(random_rel),
        'layer_impacts': {
            f'layer_{i}': float(impact)
            for i, impact in enumerate(layer_impacts)
        },
        'most_impactful_layer': int(max_impact_layer),
    }

    return results


def diagnose_bottleneck(all_results, n_layers):
    """통합 진단: bottleneck 위치 파악"""
    print("\n" + "="*70)
    print("BOTTLENECK DIAGNOSIS SUMMARY")
    print("="*70)

    issues = defaultdict(list)

    # Check each layer
    for layer_idx in range(n_layers):
        layer_key = f'layer_{layer_idx}'

        # 1. Gradient flow
        if 'bottleneck' in all_results:
            grad_mean = all_results['bottleneck'][layer_key]['router_grad_mean']
            if grad_mean < 0.001:
                issues[layer_idx].append(f"Weak gradient ({grad_mean:.6f})")

        # 2. Neuron usage
        if 'neuron_usage' in all_results:
            usage_ratio = all_results['neuron_usage'][layer_key]['usage_ratio']
            if usage_ratio < 0.7:
                issues[layer_idx].append(f"Low neuron usage ({usage_ratio:.1%})")

        # 3. Pattern usage
        if 'pattern_usage' in all_results:
            pattern_gini = all_results['pattern_usage'][layer_key]['gini_coefficient']
            if pattern_gini > 0.7:
                issues[layer_idx].append(f"Pattern collapse (Gini={pattern_gini:.2f})")

        # 4. Representation rank
        if 'information_flow' in all_results:
            rank_key = f'layer_{layer_idx}_rank'
            if rank_key in all_results['information_flow']:
                rank_ratio = all_results['information_flow'][rank_key]['rank_ratio']
                if rank_ratio < 0.4:
                    issues[layer_idx].append(f"Low diversity (rank={rank_ratio:.1%})")

        # 5. Task pressure
        if 'task_pressure' in all_results:
            importance = all_results['task_pressure'][layer_key]['importance']
            if importance > 0.8:
                issues[layer_idx].append(f"High task pressure (importance={importance:.2f})")

    # Report
    print("\n🔍 Bottleneck Indicators:")
    print("-" * 70)

    bottleneck_layers = []
    for layer_idx in range(n_layers):
        if issues[layer_idx]:
            print(f"\n  Layer {layer_idx}: 🔴 ISSUES FOUND")
            for issue in issues[layer_idx]:
                print(f"    - {issue}")
            bottleneck_layers.append(layer_idx)
        else:
            print(f"\n  Layer {layer_idx}: ✅ OK")

    # Summary
    print(f"\n{'='*70}")
    if bottleneck_layers:
        print(f"⚠️  Potential bottlenecks detected in layers: {bottleneck_layers}")
        print(f"\nRecommendations:")
        for layer_idx in bottleneck_layers:
            print(f"\n  Layer {layer_idx}:")
            if any("gradient" in i for i in issues[layer_idx]):
                print(f"    → Increase regularization weights")
                print(f"    → Check learning rate")
            if any("usage" in i for i in issues[layer_idx]):
                print(f"    → Increase capacity (more neurons/patterns)")
                print(f"    → Strengthen diversity regularization")
            if any("collapse" in i for i in issues[layer_idx]):
                print(f"    → Increase load balancing weight (0.1 → 0.2)")
                print(f"    → Add pattern dropout")
            if any("diversity" in i for i in issues[layer_idx]):
                print(f"    → Increase orthogonality weight (0.01 → 0.02)")
                print(f"    → Consider hard orthogonalization")
            if any("pressure" in i for i in issues[layer_idx]):
                print(f"    → This layer is critical - consider increasing its capacity")
    else:
        print(f"✅ No critical bottlenecks detected!")

    return {
        'bottleneck_layers': bottleneck_layers,
        'issues': dict(issues)
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze DAWN checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint folder or .pt file')
    parser.add_argument('--num_batches', type=int, default=100,
                       help='Number of batches to analyze (default: 100, use --quick for faster analysis)')
    parser.add_argument('--skip_bottleneck', action='store_true',
                       help='Skip bottleneck analysis (faster)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick analysis mode (10 batches, skip heavy analyses)')
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.num_batches = 10
        args.skip_bottleneck = True
        print("⚡ QUICK MODE: Using 10 batches, skipping bottleneck analyses")

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

    # v5.0: Model creation
    model = DAWN(
        vocab_size=vocab_size,
        hidden_dim=cfg['model']['d_model'],
        num_layers=cfg['model']['n_layers'],
        n_heads=cfg['model']['n_heads'],
        n_neurons=cfg['model']['n_neurons'],
        neuron_rank=cfg['model'].get('neuron_rank', 16),
        neuron_k=neuron_k,
        n_basis=cfg['model'].get('n_basis', 16),
        basis_rank=cfg['model'].get('basis_rank', 8),
        mod_rank=cfg['model'].get('mod_rank', 32),
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

    print(f"\nModel: {n_layers} layers, {n_neurons} neurons/layer (v5.0)")
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

    # ⭐ Pattern analysis (v5.0: disabled - no patterns in v5.0)
    # pattern_usage_results = analyze_pattern_usage(collector, n_patterns, n_layers)
    # pattern_collapse_results = analyze_pattern_collapse_detail(collector, n_patterns, n_layers)
    # neuron_pattern_corr_results = analyze_neuron_pattern_correlation(collector, n_layers)
    # confidence_results = analyze_selection_confidence(collector, n_layers)
    # position_pattern_results = analyze_position_patterns(collector, n_layers)

    # 🧬 Neuron diversity 분석 (v5.0: disabled for now)
    # diversity_results = analyze_neuron_diversity(model, n_layers)

    # 🤝 Co-activation 분석
    coactivation_results = analyze_neuron_coactivation(collector, n_neurons, n_layers)

    print("\n✅ Analysis complete! (v5.0: pattern analyses disabled)")

    # 🔬 Bottleneck 분석 (v5.0: disabled for simplicity)
    if False and not args.skip_bottleneck:
        print("\n" + "="*70)
        print("🚀 RUNNING UNIFIED BOTTLENECK ANALYSIS (OPTIMIZED)")
        print("="*70)
        print(f"Processing {args.num_batches} batches for all analyses in parallel...")

        # ⚡ 모델 사용하지 않는 분석 먼저 (즉시 실행)
        pattern_diversity_results = analyze_pattern_diversity(model, n_layers)

        # ⚡ 모든 분석에 동일한 배치 수 사용
        num_batches = args.num_batches

        bottleneck_results = analyze_layer_bottleneck(model, val_loader, device, num_batches=num_batches, tokenizer=tokenizer)
        information_flow_results = analyze_information_flow(model, val_loader, device, num_batches=num_batches)
        task_pressure_results = analyze_task_pressure(model, val_loader, device, num_batches=num_batches//3, tokenizer=tokenizer)

        # 🔍 Deep dive analyses
        gradient_detailed_results = analyze_gradient_flow_detailed(model, val_loader, device, num_batches=num_batches//3, tokenizer=tokenizer)
        pattern_necessity_results = analyze_pattern_necessity(model, val_loader, device, tokenizer=tokenizer, num_batches=num_batches//2)
        regularization_impact_results = analyze_regularization_impact(model, val_loader, device, tokenizer=tokenizer, num_batches=num_batches)

        # 🎯 Pattern-specific deep dives
        neuron_pattern_mapping_results = analyze_neuron_pattern_mapping_quality(model, val_loader, device, n_layers, tokenizer=tokenizer, num_batches=num_batches)
        pattern_ffn_impact_results = analyze_pattern_ffn_impact(model, val_loader, device, n_layers, tokenizer=tokenizer, num_batches=num_batches//3)

        print(f"✅ All bottleneck analyses completed!")
    else:
        print("\n⏩ Skipping bottleneck analysis (use --skip_bottleneck to skip)")
        bottleneck_results = {}
        information_flow_results = {}
        task_pressure_results = {}
        gradient_detailed_results = {}
        pattern_necessity_results = {}
        regularization_impact_results = {}
        pattern_diversity_results = {}
        neuron_pattern_mapping_results = {}
        pattern_ffn_impact_results = {}

    # v5.0: Simplified results (neuron-only, no patterns)
    all_results = {
        'neuron_usage': neuron_usage_results,
        'token_neuron_specialization': token_spec_results,
        'layer_differences': layer_diff_results,
        'uncertainty_accuracy': uncertainty_results,
        'neuron_coactivation': coactivation_results,
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
    print("  - neuron_roles/ (similarity, co-activation, clustering)")
    if not args.skip_bottleneck:
        print("\n🔬 Bottleneck Diagnosis:")
        print("  - Gradient flow analysis")
        print("  - Information flow (similarity, change, rank)")
        print("  - Task pressure (layer ablation)")
        print("  - Integrated bottleneck diagnosis")
        print("\n🔍 Deep Dive Analyses:")
        print("  - Detailed gradient flow (normalized by param count)")
        print("  - Pattern necessity (ablation tests)")
        print("  - Regularization impact (loss component analysis)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
