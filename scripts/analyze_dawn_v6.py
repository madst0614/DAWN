"""
DAWN v6.0 Checkpoint Analysis
Orthogonal Basis + Free Neurons 구조 분석

분석 항목:
1. 뉴런 사용 분석 (빈도, Gini, Entropy)
2. Basis 직교성 분석 (핵심!)
3. Neuron 좌표 분석 (clustering, 분포)
4. Token-Neuron 전문화
5. Layer별 차이
6. Co-activation 분석
7. FFN 구성 분석

Usage:
    python analyze_dawn_v6.py --checkpoint /path/to/checkpoint_folder
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

from models.model import DAWN
from utils.data import load_data, apply_mlm_masking, MLM_CONFIG
from transformers import BertTokenizer


# ============================================================
# 데이터 수집
# ============================================================

class ActivationCollector:
    """뉴런 선택 패턴 수집 (v6.0)"""

    def __init__(self, model, n_layers):
        self.model = model
        self.n_layers = n_layers

        # 뉴런 선택 기록
        self.neuron_selections = [[] for _ in range(n_layers)]

        # 토큰별 뉴런 선택
        self.token_neuron_map = defaultdict(lambda: [[] for _ in range(n_layers)])

        # 정확도별 선택
        self.correct_selections = [[] for _ in range(n_layers)]
        self.incorrect_selections = [[] for _ in range(n_layers)]

    def collect(self, input_ids, labels, logits, all_selected):
        """한 배치의 선택 패턴 수집"""
        B, S = input_ids.shape

        # 예측 정확도
        predictions = logits.argmax(dim=-1)
        correct_mask = (predictions == labels) & (labels != -100)

        input_ids_cpu = input_ids.cpu()

        for layer_idx, selected_idx in enumerate(all_selected):
            selected_cpu = selected_idx.cpu()

            # 1. 전체 뉴런 선택 기록
            self.neuron_selections[layer_idx].append(selected_cpu)

            # 2. 토큰별 뉴런 선택
            selected_flat = selected_cpu.reshape(-1, selected_cpu.shape[-1])
            tokens_flat = input_ids_cpu.reshape(-1)

            for token_id in tokens_flat.unique().tolist():
                mask = tokens_flat == token_id
                neurons = selected_flat[mask].reshape(-1).tolist()
                self.token_neuron_map[token_id][layer_idx].extend(neurons)

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

        # Gini coefficient
        sorted_freq = np.sort(neuron_freq)
        n = len(sorted_freq)
        cumsum = np.cumsum(sorted_freq)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_freq) - (n + 1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10)

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

        # 경고
        if usage_ratio < 0.5:
            print(f"  ⚠️  LOW USAGE: Only {usage_ratio:.1%} neurons used!")
        if gini > 0.8:
            print(f"  ⚠️  UNEQUAL: Gini={gini:.2f} - concentrated usage!")
        if top_10_ratio > 0.5:
            print(f"  ⚠️  DOMINATED: Top-10 = {top_10_ratio:.1%}!")

        # 개선 확인 (v6.0 목표: usage > 70%, gini < 0.7)
        if usage_ratio > 0.7 and gini < 0.7:
            print(f"  ✅ GOOD: Healthy neuron distribution!")

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
# 2. Basis 직교성 분석 (v6.0 핵심!)
# ============================================================

def analyze_basis_orthogonality(model, n_layers):
    """Basis 직교성 분석 - v6.0 핵심 지표"""
    print("\n" + "="*70)
    print("2. BASIS ORTHOGONALITY ANALYSIS (v6.0 Core)")
    print("="*70)

    results = {}

    for layer_idx in range(n_layers):
        # Get FFN module
        if hasattr(model, 'layers'):
            ffn = model.layers[layer_idx].ffn
        else:
            ffn = model._orig_mod.layers[layer_idx].ffn

        # Basis A: [n_basis, d_model, rank]
        basis_A = ffn.basis_A.data
        n_basis = basis_A.shape[0]

        # Flatten: [n_basis, d_model * rank]
        basis_A_flat = basis_A.view(n_basis, -1)
        basis_A_norm = F.normalize(basis_A_flat, p=2, dim=1)

        # Gram matrix
        gram = torch.mm(basis_A_norm, basis_A_norm.T)

        # Off-diagonal (직교성 측정)
        mask = ~torch.eye(n_basis, dtype=torch.bool, device=gram.device)
        off_diag = gram[mask].cpu().numpy()

        # 통계
        mean_sim = np.abs(off_diag).mean()
        max_sim = np.abs(off_diag).max()
        std_sim = off_diag.std()

        # 직교성 점수 (1 = 완벽 직교)
        orthogonality_score = 1 - mean_sim

        # Effective rank (SVD)
        U, S, V = torch.svd(basis_A_flat)
        S_norm = S / S.sum()
        eff_rank = torch.exp(-(S_norm * torch.log(S_norm + 1e-10)).sum()).item()
        rank_ratio = eff_rank / n_basis

        print(f"\nLayer {layer_idx} - Basis A:")
        print(f"  Mean |similarity|: {mean_sim:.4f} (lower = more orthogonal)")
        print(f"  Max |similarity|: {max_sim:.4f}")
        print(f"  Orthogonality score: {orthogonality_score:.4f} (1.0 = perfect)")
        print(f"  Effective rank: {eff_rank:.2f}/{n_basis} ({rank_ratio:.1%})")

        # Basis B도 분석
        basis_B = ffn.basis_B.data
        basis_B_flat = basis_B.view(n_basis, -1)
        basis_B_norm = F.normalize(basis_B_flat, p=2, dim=1)
        gram_B = torch.mm(basis_B_norm, basis_B_norm.T)
        off_diag_B = gram_B[mask].cpu().numpy()

        mean_sim_B = np.abs(off_diag_B).mean()
        orthogonality_B = 1 - mean_sim_B

        print(f"\nLayer {layer_idx} - Basis B:")
        print(f"  Mean |similarity|: {mean_sim_B:.4f}")
        print(f"  Orthogonality score: {orthogonality_B:.4f}")

        # 평가
        avg_orthogonality = (orthogonality_score + orthogonality_B) / 2

        if avg_orthogonality > 0.9:
            print(f"\n  ✅ EXCELLENT: Near-perfect orthogonality ({avg_orthogonality:.2%})")
        elif avg_orthogonality > 0.7:
            print(f"\n  ✅ GOOD: Strong orthogonality ({avg_orthogonality:.2%})")
        elif avg_orthogonality > 0.5:
            print(f"\n  ⚠️  MODERATE: Could be improved ({avg_orthogonality:.2%})")
        else:
            print(f"\n  🔴 POOR: Basis are not orthogonal ({avg_orthogonality:.2%})")

        results[f'layer_{layer_idx}'] = {
            'mean_similarity_A': float(mean_sim),
            'max_similarity_A': float(max_sim),
            'orthogonality_A': float(orthogonality_score),
            'effective_rank_A': float(eff_rank),
            'rank_ratio_A': float(rank_ratio),
            'mean_similarity_B': float(mean_sim_B),
            'orthogonality_B': float(orthogonality_B),
            'avg_orthogonality': float(avg_orthogonality),
        }

    return results


# ============================================================
# 3. Neuron 좌표 분석 (v6.0 신규!)
# ============================================================

def analyze_neuron_coordinates(model, n_layers):
    """Neuron 좌표 분포 분석 - 자연스러운 clustering 확인"""
    print("\n" + "="*70)
    print("3. NEURON COORDINATE ANALYSIS (v6.0 New)")
    print("="*70)

    results = {}

    for layer_idx in range(n_layers):
        # Get FFN module
        if hasattr(model, 'layers'):
            ffn = model.layers[layer_idx].ffn
        else:
            ffn = model._orig_mod.layers[layer_idx].ffn

        # Neuron coordinates: [n_neurons, n_basis]
        coords = ffn.neuron_coords.data.cpu()
        n_neurons, n_basis = coords.shape

        # 1. 좌표 분포 통계
        coord_mean = coords.mean().item()
        coord_std = coords.std().item()
        coord_min = coords.min().item()
        coord_max = coords.max().item()

        # 2. Basis 사용률 (각 basis가 얼마나 활용되나?)
        basis_usage = coords.abs().mean(dim=0).numpy()
        active_basis = (basis_usage > 0.1).sum()

        # 3. Neuron 유사도 분석
        coords_norm = F.normalize(coords, p=2, dim=1)
        similarity = torch.mm(coords_norm, coords_norm.T)

        mask = ~torch.eye(n_neurons, dtype=torch.bool)
        off_diag = similarity[mask].numpy()

        mean_sim = off_diag.mean()
        std_sim = off_diag.std()

        # 유사도 분포
        high_sim_pairs = (off_diag > 0.9).sum()
        medium_sim_pairs = ((off_diag > 0.7) & (off_diag <= 0.9)).sum()
        low_sim_pairs = (off_diag <= 0.7).sum()
        total_pairs = len(off_diag)

        print(f"\nLayer {layer_idx}:")
        print(f"  Coordinate stats:")
        print(f"    Mean: {coord_mean:.4f}, Std: {coord_std:.4f}")
        print(f"    Range: [{coord_min:.4f}, {coord_max:.4f}]")
        print(f"  Basis usage: {active_basis}/{n_basis} active (>0.1)")
        print(f"  Neuron similarity:")
        print(f"    Mean: {mean_sim:.4f}, Std: {std_sim:.4f}")
        print(f"    High (>0.9): {high_sim_pairs}/{total_pairs} ({high_sim_pairs/total_pairs:.1%})")
        print(f"    Medium (0.7-0.9): {medium_sim_pairs}/{total_pairs} ({medium_sim_pairs/total_pairs:.1%})")
        print(f"    Low (<0.7): {low_sim_pairs}/{total_pairs} ({low_sim_pairs/total_pairs:.1%})")

        # 평가
        if active_basis == n_basis:
            print(f"  ✅ All {n_basis} basis actively used!")
        else:
            print(f"  ⚠️  Only {active_basis}/{n_basis} basis used - potential underutilization")

        if high_sim_pairs / total_pairs > 0.3:
            print(f"  ⚠️  Many similar neurons ({high_sim_pairs/total_pairs:.1%}) - potential redundancy")
        elif low_sim_pairs / total_pairs > 0.7:
            print(f"  ✅ Good diversity - {low_sim_pairs/total_pairs:.1%} distinct neurons")

        results[f'layer_{layer_idx}'] = {
            'coord_mean': float(coord_mean),
            'coord_std': float(coord_std),
            'coord_range': [float(coord_min), float(coord_max)],
            'active_basis': int(active_basis),
            'total_basis': int(n_basis),
            'basis_usage': basis_usage.tolist(),
            'neuron_similarity_mean': float(mean_sim),
            'neuron_similarity_std': float(std_sim),
            'high_sim_ratio': float(high_sim_pairs / total_pairs),
            'medium_sim_ratio': float(medium_sim_pairs / total_pairs),
            'low_sim_ratio': float(low_sim_pairs / total_pairs),
        }

    return results


# ============================================================
# 헬퍼 함수
# ============================================================

def convert_to_serializable(obj):
    """NumPy 배열을 JSON 직렬화 가능하게 변환"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    else:
        return obj


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze DAWN v6.0 checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint folder or .pt file')
    parser.add_argument('--num_batches', type=int, default=100,
                       help='Number of batches to analyze')
    parser.add_argument('--quick', action='store_true',
                       help='Quick analysis mode (20 batches)')
    args = parser.parse_args()

    if args.quick:
        args.num_batches = 20
        print("⚡ QUICK MODE: Using 20 batches")

    # 체크포인트 경로 처리
    checkpoint_path = Path(args.checkpoint)

    if checkpoint_path.is_dir():
        best_model_path = checkpoint_path / 'best_model.pt'
        config_path = checkpoint_path / 'config.json'
        output_dir = checkpoint_path / 'analysis_v6'
    else:
        best_model_path = checkpoint_path
        config_path = checkpoint_path.parent / 'config.json'
        output_dir = checkpoint_path.parent / 'analysis_v6'

    if not best_model_path.exists():
        print(f"❌ Checkpoint not found: {best_model_path}")
        return

    output_dir.mkdir(exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    print("="*70)
    print("DAWN v6.0 Checkpoint Analysis")
    print("="*70)

    # Config 로드
    print(f"\nLoading config: {config_path}")
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    # 체크포인트 로드
    print(f"Loading checkpoint: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)

    # 버전 확인
    checkpoint_version = checkpoint.get('model_version', 'unknown')
    print(f"\n📌 Checkpoint version: {checkpoint_version}")
    print(f"📌 Current code version: {DAWN.__version__}")

    if checkpoint_version != '6.0':
        print(f"\n⚠️  Warning: This script is designed for v6.0")
        print(f"   Checkpoint version: {checkpoint_version}")

    # 모델 생성
    print("\nCreating model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

    model_cfg = cfg.get('model', cfg)

    model = DAWN(
        vocab_size=vocab_size,
        d_model=model_cfg.get('d_model', 256),
        d_ff=model_cfg.get('d_ff', 1024),
        n_layers=model_cfg.get('n_layers', 4),
        n_heads=model_cfg.get('n_heads', 4),
        n_neurons=model_cfg.get('n_neurons', 256),
        neuron_k=model_cfg.get('neuron_k', 16),
        n_basis=model_cfg.get('n_basis', 8),
        basis_rank=model_cfg.get('basis_rank', 64),
        max_seq_len=model_cfg.get('max_seq_len', 512),
        dropout=model_cfg.get('dropout', 0.1),
    )

    # State dict 로드
    state_dict = checkpoint['model_state_dict']

    # _orig_mod. prefix 제거 (torch.compile)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠️  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    n_layers = model_cfg.get('n_layers', 4)
    n_neurons = model_cfg.get('n_neurons', 256)
    n_basis = model_cfg.get('n_basis', 8)

    print(f"\nModel config:")
    print(f"  Layers: {n_layers}")
    print(f"  Neurons: {n_neurons}")
    print(f"  Basis: {n_basis}, Rank: {model_cfg.get('basis_rank', 64)}")
    print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    # 데이터 로드
    print("\nLoading validation data...")
    _, val_loader, _ = load_data(
        cfg['data'],
        max_length=model_cfg.get('max_seq_len', 512),
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

            if apply_mlm_masking and tokenizer:
                input_ids, labels = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)
            else:
                labels = input_ids.clone()

            logits, all_selected = model(input_ids, return_activations=True)
            collector.collect(input_ids, labels, logits, all_selected)

    collector.finalize()

    # 분석 실행
    print("\n" + "="*70)
    print("RUNNING v6.0 ANALYSES")
    print("="*70)

    neuron_usage_results = analyze_neuron_usage(collector, n_neurons, n_layers)
    basis_results = analyze_basis_orthogonality(model, n_layers)
    coord_results = analyze_neuron_coordinates(model, n_layers)

    # 결과 통합
    all_results = {
        'model_version': checkpoint_version,
        'neuron_usage': neuron_usage_results,
        'basis_orthogonality': basis_results,
        'neuron_coordinates': coord_results,
    }

    # 저장
    print(f"\nSaving results to: {output_dir}")

    all_results_serializable = convert_to_serializable(all_results)
    with open(output_dir / 'analysis_v6_results.json', 'w') as f:
        json.dump(all_results_serializable, f, indent=2)
    print("  ✓ analysis_v6_results.json")

    # 간단한 리포트
    with open(output_dir / 'analysis_v6_summary.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("DAWN v6.0 Analysis Summary\n")
        f.write("="*70 + "\n\n")

        f.write(f"Checkpoint: {best_model_path}\n")
        f.write(f"Version: {checkpoint_version}\n\n")

        f.write("KEY METRICS:\n")
        f.write("-"*70 + "\n\n")

        for layer_idx in range(n_layers):
            f.write(f"Layer {layer_idx}:\n")
            f.write(f"  Neuron usage: {neuron_usage_results[f'layer_{layer_idx}']['usage_ratio']:.2%}\n")
            f.write(f"  Gini coefficient: {neuron_usage_results[f'layer_{layer_idx}']['gini_coefficient']:.4f}\n")
            f.write(f"  Basis orthogonality: {basis_results[f'layer_{layer_idx}']['avg_orthogonality']:.4f}\n")
            f.write(f"  Neuron diversity: {coord_results[f'layer_{layer_idx}']['low_sim_ratio']:.2%} low-similarity\n")
            f.write("\n")

        f.write("="*70 + "\n")

    print("  ✓ analysis_v6_summary.txt")

    print("\n" + "="*70)
    print("✅ v6.0 ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("  - analysis_v6_results.json")
    print("  - analysis_v6_summary.txt")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
