"""
Dynamic Neuron Transformer - Comprehensive Analysis
뉴런 활성화 패턴, 전문화, 다양성 종합 분석

Usage:
    python scripts/analyze_neuron_dynamics.py --checkpoint path/to/best_model.pt --output analysis_results
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
from tqdm import tqdm
import json
import argparse
from collections import defaultdict, Counter
from scipy.stats import entropy
import yaml

from models.model import DAWN
from utils.data import load_data, apply_mlm_masking, MLM_CONFIG
from transformers import BertTokenizer


# ============================================================
# 1. 뉴런 활성화 수집
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
        # 뉴런 선택을 하나의 텐서로
        for layer_idx in range(self.n_layers):
            if self.neuron_selections[layer_idx]:
                self.neuron_selections[layer_idx] = torch.cat(
                    self.neuron_selections[layer_idx], dim=0
                )  # [total_tokens, k]

            if self.correct_selections[layer_idx]:
                self.correct_selections[layer_idx] = torch.cat(
                    self.correct_selections[layer_idx], dim=0
                )

            if self.incorrect_selections[layer_idx]:
                self.incorrect_selections[layer_idx] = torch.cat(
                    self.incorrect_selections[layer_idx], dim=0
                )


# ============================================================
# 2. 뉴런 활성화 분석
# ============================================================

def analyze_neuron_usage(collector, n_neurons, n_layers):
    """뉴런 사용 빈도 및 분포 분석"""
    results = {}

    for layer_idx in range(n_layers):
        selections = collector.neuron_selections[layer_idx]  # [N, k]

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

        # Top-k 뉴런이 차지하는 비율
        top_10_ratio = np.sort(neuron_freq)[-10:].sum()
        top_50_ratio = np.sort(neuron_freq)[-50:].sum()

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
# 3. 토큰-뉴런 전문화 분석
# ============================================================

def analyze_token_neuron_specialization(collector, tokenizer, n_neurons, n_layers, top_k_tokens=100):
    """토큰별 뉴런 선택 패턴 분석"""
    results = {}

    # 가장 많이 나온 토큰 top-k
    all_tokens = list(collector.token_neuron_map.keys())
    token_counts = {
        token_id: sum(len(neurons) for neurons in collector.token_neuron_map[token_id])
        for token_id in all_tokens
    }
    top_tokens = sorted(token_counts.keys(), key=lambda x: token_counts[x], reverse=True)[:top_k_tokens]

    for layer_idx in range(n_layers):
        token_neuron_patterns = {}

        for token_id in top_tokens:
            neurons = collector.token_neuron_map[token_id][layer_idx]

            if len(neurons) == 0:
                continue

            # 뉴런 분포
            neuron_counts = Counter(neurons)
            total = sum(neuron_counts.values())

            # Top-3 뉴런
            top_3 = neuron_counts.most_common(3)

            # 집중도 (상위 3개가 차지하는 비율)
            concentration = sum(count for _, count in top_3) / total if total > 0 else 0

            # 다양성 (유니크 뉴런 수)
            unique_neurons = len(neuron_counts)

            token_str = tokenizer.decode([token_id])

            token_neuron_patterns[token_str] = {
                'token_id': token_id,
                'total_occurrences': total,
                'unique_neurons': unique_neurons,
                'concentration': float(concentration),
                'top_3_neurons': [(int(n), int(c)) for n, c in top_3],
            }

        results[f'layer_{layer_idx}'] = token_neuron_patterns

    return results


# ============================================================
# 4. Layer별 차이 분석
# ============================================================

def analyze_layer_differences(neuron_usage_results):
    """Layer별 뉴런 사용 패턴 차이 분석"""
    results = {}

    layers = sorted([k for k in neuron_usage_results.keys() if k.startswith('layer_')])

    for i, layer_i in enumerate(layers):
        for j, layer_j in enumerate(layers):
            if i >= j:
                continue

            freq_i = np.array(neuron_usage_results[layer_i]['neuron_freq'])
            freq_j = np.array(neuron_usage_results[layer_j]['neuron_freq'])

            # KL divergence
            kl_div = entropy(freq_i + 1e-10, freq_j + 1e-10)

            # Cosine similarity
            cos_sim = np.dot(freq_i, freq_j) / (np.linalg.norm(freq_i) * np.linalg.norm(freq_j))

            results[f'{layer_i}_vs_{layer_j}'] = {
                'kl_divergence': float(kl_div),
                'cosine_similarity': float(cos_sim),
            }

    return results


# ============================================================
# 5. 불확실성-정확도 분석
# ============================================================

def analyze_uncertainty_accuracy(collector, n_layers):
    """정답/오답 시 뉴런 선택 패턴 비교"""
    results = {}

    for layer_idx in range(n_layers):
        correct_sel = collector.correct_selections[layer_idx]
        incorrect_sel = collector.incorrect_selections[layer_idx]

        if len(correct_sel) == 0 or len(incorrect_sel) == 0:
            continue

        # 뉴런 다양성 비교
        correct_unique = len(torch.unique(correct_sel))
        incorrect_unique = len(torch.unique(incorrect_sel))

        results[f'layer_{layer_idx}'] = {
            'correct_samples': len(correct_sel),
            'incorrect_samples': len(incorrect_sel),
            'correct_unique_neurons': int(correct_unique),
            'incorrect_unique_neurons': int(incorrect_unique),
        }

    return results


# ============================================================
# 6. 시각화
# ============================================================

def visualize_results(neuron_usage_results, output_dir):
    """분석 결과 시각화"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_layers = len([k for k in neuron_usage_results.keys() if k.startswith('layer_')])

    # 1. 뉴런 사용 분포 히스토그램
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for layer_idx in range(min(n_layers, 4)):
        layer_key = f'layer_{layer_idx}'
        neuron_freq = neuron_usage_results[layer_key]['neuron_freq']

        ax = axes[layer_idx]
        ax.hist(neuron_freq, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Neuron Selection Frequency')
        ax.set_ylabel('Count')
        ax.set_title(f'Layer {layer_idx} - Neuron Usage Distribution\n'
                    f'Gini: {neuron_usage_results[layer_key]["gini_coefficient"]:.3f}, '
                    f'Usage: {neuron_usage_results[layer_key]["usage_ratio"]:.2%}')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'neuron_usage_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Layer별 통계 비교
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    layers = list(range(n_layers))
    gini_coeffs = [neuron_usage_results[f'layer_{i}']['gini_coefficient'] for i in layers]
    usage_ratios = [neuron_usage_results[f'layer_{i}']['usage_ratio'] for i in layers]
    entropies = [neuron_usage_results[f'layer_{i}']['entropy'] for i in layers]

    axes[0].bar(layers, gini_coeffs, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Gini Coefficient')
    axes[0].set_title('Neuron Usage Inequality\n(Higher = More Unequal)')
    axes[0].grid(alpha=0.3)

    axes[1].bar(layers, usage_ratios, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Usage Ratio')
    axes[1].set_title('Neuron Usage Ratio\n(Fraction of Used Neurons)')
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 1])

    axes[2].bar(layers, entropies, color='lightgreen', edgecolor='black')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Entropy')
    axes[2].set_title('Neuron Selection Entropy\n(Higher = More Diverse)')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'layer_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 뉴런 사용 히트맵 (전체 레이어)
    fig, ax = plt.subplots(figsize=(12, 8))

    # 각 레이어의 top-50 뉴런 사용 빈도
    heatmap_data = []
    for layer_idx in range(n_layers):
        layer_key = f'layer_{layer_idx}'
        neuron_freq = np.array(neuron_usage_results[layer_key]['neuron_freq'])
        top_50_indices = np.argsort(neuron_freq)[-50:]
        top_50_freq = neuron_freq[top_50_indices]
        heatmap_data.append(top_50_freq)

    heatmap_data = np.array(heatmap_data)

    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Selection Frequency'})
    ax.set_xlabel('Top-50 Neurons (sorted by frequency)')
    ax.set_ylabel('Layer')
    ax.set_title('Top-50 Most Used Neurons per Layer')

    plt.tight_layout()
    plt.savefig(output_dir / 'neuron_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualizations saved to {output_dir}")


# ============================================================
# 7. 리포트 생성
# ============================================================

def generate_report(all_results, output_dir):
    """종합 리포트 생성"""
    output_dir = Path(output_dir)
    report_path = output_dir / 'analysis_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Dynamic Neuron Transformer - Comprehensive Analysis Report\n")
        f.write("=" * 80 + "\n\n")

        # 1. 뉴런 사용 분석
        f.write("1. NEURON USAGE ANALYSIS\n")
        f.write("-" * 80 + "\n")

        neuron_results = all_results['neuron_usage']
        for layer_key in sorted(neuron_results.keys()):
            layer_data = neuron_results[layer_key]
            f.write(f"\n{layer_key.upper()}:\n")
            f.write(f"  Used neurons: {layer_data['used_neurons']}/{layer_data['total_neurons']} "
                   f"({layer_data['usage_ratio']:.2%})\n")
            f.write(f"  Gini coefficient: {layer_data['gini_coefficient']:.4f} "
                   f"(0=equal, 1=unequal)\n")
            f.write(f"  Entropy: {layer_data['entropy']:.4f}\n")
            f.write(f"  Top-10 neurons: {layer_data['top_10_ratio']:.2%} of selections\n")
            f.write(f"  Top-50 neurons: {layer_data['top_50_ratio']:.2%} of selections\n")

        # 2. Layer별 차이
        f.write("\n\n2. LAYER DIFFERENCES\n")
        f.write("-" * 80 + "\n")

        layer_diff = all_results['layer_differences']
        for pair_key in sorted(layer_diff.keys()):
            pair_data = layer_diff[pair_key]
            f.write(f"\n{pair_key}:\n")
            f.write(f"  KL Divergence: {pair_data['kl_divergence']:.4f} "
                   f"(higher = more different)\n")
            f.write(f"  Cosine Similarity: {pair_data['cosine_similarity']:.4f} "
                   f"(1=identical, 0=orthogonal)\n")

        # 3. 토큰-뉴런 전문화
        f.write("\n\n3. TOKEN-NEURON SPECIALIZATION (Sample)\n")
        f.write("-" * 80 + "\n")

        token_spec = all_results['token_neuron_specialization']
        for layer_key in sorted(token_spec.keys())[:2]:  # 처음 2개 레이어만
            f.write(f"\n{layer_key.upper()} (Top 10 tokens):\n")
            layer_data = token_spec[layer_key]

            # 가장 집중도가 높은 10개 토큰
            sorted_tokens = sorted(
                layer_data.items(),
                key=lambda x: x[1]['concentration'],
                reverse=True
            )[:10]

            for token_str, data in sorted_tokens:
                top_neurons = data['top_3_neurons']
                f.write(f"  '{token_str}': ")
                f.write(f"concentration={data['concentration']:.2%}, ")
                f.write(f"unique={data['unique_neurons']}, ")
                f.write(f"top_3={top_neurons}\n")

        # 4. 정확도-불확실성
        f.write("\n\n4. ACCURACY-UNCERTAINTY RELATIONSHIP\n")
        f.write("-" * 80 + "\n")

        uncertainty = all_results['uncertainty_accuracy']
        for layer_key in sorted(uncertainty.keys()):
            layer_data = uncertainty[layer_key]
            f.write(f"\n{layer_key.upper()}:\n")
            f.write(f"  Correct predictions: {layer_data['correct_samples']:,} samples, "
                   f"{layer_data['correct_unique_neurons']} unique neurons\n")
            f.write(f"  Incorrect predictions: {layer_data['incorrect_samples']:,} samples, "
                   f"{layer_data['incorrect_unique_neurons']} unique neurons\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Analysis complete!\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Report saved to {report_path}")


# ============================================================
# 8. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze Dynamic Neuron Transformer')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Config file')
    parser.add_argument('--output', type=str, default='analysis_results', help='Output directory')
    parser.add_argument('--num_batches', type=int, default=100, help='Number of batches to analyze')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create model
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

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    n_layers = cfg['model']['n_layers']
    n_neurons = cfg['model']['n_neurons']

    print(f"Model loaded: {n_layers} layers, {n_neurons} neurons per layer")

    # Load data
    print("\nLoading validation data...")
    _, val_loader, _ = load_data(
        cfg['data'],
        max_length=cfg['model']['max_seq_len'],
        batch_size=32
    )

    # Collect activations
    print(f"\nCollecting neuron selections from {args.num_batches} batches...")
    collector = ActivationCollector(model, n_layers)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, total=args.num_batches)):
            if batch_idx >= args.num_batches:
                break

            input_ids = batch['input_ids'].to(device)
            input_ids, labels = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)

            # Forward pass
            logits, all_selected = model(input_ids, return_activations=True)

            # Collect
            collector.collect(input_ids, labels, logits, all_selected)

    collector.finalize()
    print("✓ Collection complete")

    # Analyze
    print("\nAnalyzing neuron usage...")
    neuron_usage_results = analyze_neuron_usage(collector, n_neurons, n_layers)

    print("Analyzing token-neuron specialization...")
    token_spec_results = analyze_token_neuron_specialization(
        collector, tokenizer, n_neurons, n_layers, top_k_tokens=100
    )

    print("Analyzing layer differences...")
    layer_diff_results = analyze_layer_differences(neuron_usage_results)

    print("Analyzing uncertainty-accuracy relationship...")
    uncertainty_results = analyze_uncertainty_accuracy(collector, n_layers)

    # Combine results
    all_results = {
        'neuron_usage': neuron_usage_results,
        'token_neuron_specialization': token_spec_results,
        'layer_differences': layer_diff_results,
        'uncertainty_accuracy': uncertainty_results,
    }

    # Save JSON
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to {output_dir}...")
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("✓ JSON saved")

    # Visualize
    print("\nGenerating visualizations...")
    visualize_results(neuron_usage_results, output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(all_results, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"  - analysis_results.json")
    print(f"  - analysis_report.txt")
    print(f"  - neuron_usage_distribution.png")
    print(f"  - layer_statistics.png")
    print(f"  - neuron_heatmap.png")


if __name__ == "__main__":
    main()
