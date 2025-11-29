#!/usr/bin/env python3
"""
DAWN v9.1 Analysis Script

v9.0 → v9.1 변경사항 반영:
1. Compress/Expand: soft (weighted sum) → hard (argmax)
2. Reflection: hard 100% → gated (sigmoid gate)
3. ReflectionNeurons: shared → separate pools (n_reflect_d, n_reflect_r)

분석 항목:
1. ReflectionNeurons 사용 분포 (reflect_d, reflect_r - 분리된 풀)
2. Q-K Overlap 분석 (Jaccard similarity)
3. CompressNeurons/ExpandNeurons 선택 패턴 (hard selection)
4. Gating 분석 (sigmoid gate 값 분포)
5. 레이어별 비교

Usage:
    python scripts/analyze_dawn_v91.py --checkpoint path/to/best_model.pt
    python scripts/analyze_dawn_v91.py --checkpoint path/to/best_model.pt --visualize
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from models import create_model_by_version
from utils.data import TextDataset, collate_fn_dynamic_padding


def load_model(checkpoint_path, device='cuda', version=None):
    """체크포인트에서 모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 모델 config 추출
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # 기본값 사용
        config = {
            'vocab_size': 30522,
            'd_model': 256,
            'n_layers': 4,
            'n_heads': 4,
            'rank': 64,
            'n_compress': 4,
            'n_expand': 4,
            'n_reflect_d': 64,
            'n_reflect_r': 64,
            'reflect_k': 3,
            'n_knowledge': 64,
            'knowledge_k': 8,
        }

    # 버전 감지
    if version is None:
        version = config.get('model_version', '9.1')
        if version is None:
            # n_reflect_d가 있으면 9.1, 없으면 9.0
            if 'n_reflect_d' in config:
                version = '9.1'
            elif 'n_reflect' in config:
                version = '9.0'
            else:
                version = '9.1'

    print(f"Detected model version: {version}")

    model = create_model_by_version(version, config)

    # state_dict 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, config, version


def collect_routing_info(model, dataloader, device, max_batches=50):
    """Forward pass하면서 routing info 수집"""
    all_routing = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Collecting routing info", total=max_batches)):
            if i >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)

            # return_routing_info=True로 forward
            outputs = model(input_ids, return_routing_info=True)

            if isinstance(outputs, tuple):
                if len(outputs) == 2:
                    logits, routing_infos = outputs
                else:
                    logits = outputs[0]
                    routing_infos = outputs[-1]

            # 레이어별 routing info 저장
            for layer_idx, layer_routing in enumerate(routing_infos):
                attn_routing = layer_routing['attention']
                mem_routing = layer_routing['memory']

                # Q/K/V/O routing
                for qkvo in ['Q', 'K', 'V', 'O']:
                    key = f'routing_{qkvo}'
                    if key in attn_routing:
                        r = attn_routing[key]

                        # reflect_d indices
                        if 'indices_d' in r:
                            all_routing[f'layer{layer_idx}_{qkvo}_indices_d'].append(
                                r['indices_d'].cpu()
                            )
                        # reflect_d gates (v9.1: gated application)
                        if 'gates_d' in r:
                            all_routing[f'layer{layer_idx}_{qkvo}_gates_d'].append(
                                r['gates_d'].cpu()
                            )

                        # compress neuron index (v9.1: hard selection)
                        if 'idx_compress' in r:
                            all_routing[f'layer{layer_idx}_{qkvo}_idx_compress'].append(
                                r['idx_compress'].cpu()
                            )
                        # Legacy: soft weights (v9.0)
                        if 'weights_compress' in r:
                            all_routing[f'layer{layer_idx}_{qkvo}_weights_compress'].append(
                                r['weights_compress'].cpu()
                            )

                        # reflect_r indices (Expander only)
                        if 'indices_r' in r:
                            all_routing[f'layer{layer_idx}_{qkvo}_indices_r'].append(
                                r['indices_r'].cpu()
                            )
                        # reflect_r gates
                        if 'gates_r' in r:
                            all_routing[f'layer{layer_idx}_{qkvo}_gates_r'].append(
                                r['gates_r'].cpu()
                            )

                        # expand neuron index (v9.1: hard selection)
                        if 'idx_expand' in r:
                            all_routing[f'layer{layer_idx}_{qkvo}_idx_expand'].append(
                                r['idx_expand'].cpu()
                            )
                        # Legacy: soft weights (v9.0)
                        if 'weights_expand' in r:
                            all_routing[f'layer{layer_idx}_{qkvo}_weights_expand'].append(
                                r['weights_expand'].cpu()
                            )

                # Memory (M) routing
                if 'query_routing' in mem_routing:
                    m_routing = mem_routing['query_routing']
                    if 'indices_d' in m_routing:
                        all_routing[f'layer{layer_idx}_M_indices_d'].append(
                            m_routing['indices_d'].cpu()
                        )
                    if 'gates_d' in m_routing:
                        all_routing[f'layer{layer_idx}_M_gates_d'].append(
                            m_routing['gates_d'].cpu()
                        )
                    if 'idx_compress' in m_routing:
                        all_routing[f'layer{layer_idx}_M_idx_compress'].append(
                            m_routing['idx_compress'].cpu()
                        )
                    if 'weights_compress' in m_routing:
                        all_routing[f'layer{layer_idx}_M_weights_compress'].append(
                            m_routing['weights_compress'].cpu()
                        )

    # 텐서들 합치기
    for key in all_routing:
        all_routing[key] = torch.cat(all_routing[key], dim=0)

    return dict(all_routing)


def compute_gini(counts):
    """Gini coefficient 계산 (0=균등, 1=편중)"""
    counts = np.array(counts, dtype=float)
    if counts.sum() == 0:
        return 0.0
    counts = np.sort(counts)
    n = len(counts)
    cumsum = np.cumsum(counts)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def compute_effective_rank(counts):
    """Effective rank (엔트로피 기반)"""
    counts = np.array(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]  # 0 제외
    entropy = -np.sum(probs * np.log(probs))
    return np.exp(entropy)


def analyze_reflection_usage(routing_info, n_reflect_d, n_reflect_r, n_layers):
    """ReflectionNeurons 사용 분포 분석 (v9.1: separate pools)"""
    print("\n" + "="*60)
    print("[ReflectionNeurons Usage - Separate Pools]")
    print("="*60)

    # reflect_d 사용량 집계
    reflect_d_counts = np.zeros(n_reflect_d)
    reflect_r_counts = np.zeros(n_reflect_r)

    for layer_idx in range(n_layers):
        for qkvo in ['Q', 'K', 'V', 'O', 'M']:
            key_d = f'layer{layer_idx}_{qkvo}_indices_d'
            key_r = f'layer{layer_idx}_{qkvo}_indices_r'

            if key_d in routing_info:
                indices = routing_info[key_d].numpy().flatten()
                for idx in indices:
                    if 0 <= idx < n_reflect_d:
                        reflect_d_counts[idx] += 1

            if key_r in routing_info:
                indices = routing_info[key_r].numpy().flatten()
                for idx in indices:
                    if 0 <= idx < n_reflect_r:
                        reflect_r_counts[idx] += 1

    # reflect_d 통계
    print(f"\n[reflect_d pool] (size={n_reflect_d})")
    print(f"  Top-10 indices: {np.argsort(reflect_d_counts)[-10:][::-1].tolist()}")
    print(f"  Top-10 counts:  {np.sort(reflect_d_counts)[-10:][::-1].astype(int).tolist()}")
    eff_rank_d = compute_effective_rank(reflect_d_counts)
    gini_d = compute_gini(reflect_d_counts)
    print(f"  Effective rank: {eff_rank_d:.1f} / {n_reflect_d}")
    print(f"  Gini coefficient: {gini_d:.3f}")

    # reflect_r 통계
    print(f"\n[reflect_r pool] (size={n_reflect_r})")
    if reflect_r_counts.sum() > 0:
        print(f"  Top-10 indices: {np.argsort(reflect_r_counts)[-10:][::-1].tolist()}")
        print(f"  Top-10 counts:  {np.sort(reflect_r_counts)[-10:][::-1].astype(int).tolist()}")
        eff_rank_r = compute_effective_rank(reflect_r_counts)
        gini_r = compute_gini(reflect_r_counts)
        print(f"  Effective rank: {eff_rank_r:.1f} / {n_reflect_r}")
        print(f"  Gini coefficient: {gini_r:.3f}")
    else:
        print("  (No reflect_r data collected)")

    return reflect_d_counts, reflect_r_counts


def analyze_gating(routing_info, n_layers):
    """Gating 분석 (v9.1: sigmoid gate 값 분포)"""
    print("\n" + "="*60)
    print("[Gating Analysis - Sigmoid Gate Values]")
    print("="*60)

    gate_stats = {}

    for layer_idx in range(n_layers):
        for qkvo in ['Q', 'K', 'V', 'O', 'M']:
            for gate_type in ['gates_d', 'gates_r']:
                key = f'layer{layer_idx}_{qkvo}_{gate_type}'
                if key in routing_info:
                    gates = routing_info[key].numpy().flatten()
                    mean_gate = gates.mean()
                    std_gate = gates.std()

                    gate_key = f'{qkvo}_{gate_type}'
                    if gate_key not in gate_stats:
                        gate_stats[gate_key] = []
                    gate_stats[gate_key].append({
                        'layer': layer_idx,
                        'mean': mean_gate,
                        'std': std_gate
                    })

    # 타입별 평균 출력
    for gate_key, stats in gate_stats.items():
        avg_mean = np.mean([s['mean'] for s in stats])
        avg_std = np.mean([s['std'] for s in stats])
        print(f"{gate_key}: avg_mean={avg_mean:.3f}, avg_std={avg_std:.3f}")

    return gate_stats


def analyze_qk_overlap(routing_info, n_layers):
    """Q-K Overlap 분석 (Jaccard similarity)"""
    print("\n" + "="*60)
    print("[Q-K Overlap Analysis]")
    print("="*60)

    jaccard_scores = []

    for layer_idx in range(n_layers):
        key_q = f'layer{layer_idx}_Q_indices_d'
        key_k = f'layer{layer_idx}_K_indices_d'

        if key_q not in routing_info or key_k not in routing_info:
            continue

        q_indices = routing_info[key_q].numpy()  # [B, S, k]
        k_indices = routing_info[key_k].numpy()

        # 배치/시퀀스별 Jaccard 계산 후 평균
        jaccards = []
        for b in range(q_indices.shape[0]):
            for s in range(q_indices.shape[1]):
                q_set = set(q_indices[b, s])
                k_set = set(k_indices[b, s])
                intersection = len(q_set & k_set)
                union = len(q_set | k_set)
                if union > 0:
                    jaccards.append(intersection / union)

        avg_jaccard = np.mean(jaccards) if jaccards else 0.0
        jaccard_scores.append(avg_jaccard)
        print(f"Layer {layer_idx}: Jaccard {avg_jaccard:.3f}")

    if jaccard_scores:
        print(f"\nAverage Q-K Jaccard: {np.mean(jaccard_scores):.3f}")

    return jaccard_scores


def analyze_compress_expand_selection(routing_info, n_layers, n_compress, n_expand):
    """Compress/Expand Neuron 선택 패턴 분석 (v9.1: hard selection)"""
    print("\n" + "="*60)
    print("[CompressNeurons Hard Selection]")
    print("="*60)

    # v9.1: hard selection (idx_compress)
    compress_counts = {}
    for qkvo in ['Q', 'K', 'V', 'M']:
        counts = np.zeros(n_compress)
        for layer_idx in range(n_layers):
            key = f'layer{layer_idx}_{qkvo}_idx_compress'
            if key in routing_info:
                indices = routing_info[key].numpy().flatten()
                for idx in indices:
                    if 0 <= idx < n_compress:
                        counts[idx] += 1

        if counts.sum() > 0:
            probs = counts / counts.sum()
            compress_counts[qkvo] = probs
            print(f"{qkvo}: {[f'{p:.3f}' for p in probs]}")

    # Fallback: soft weights (v9.0)
    if not compress_counts:
        print("(Using soft weights - v9.0 style)")
        for qkvo in ['Q', 'K', 'V', 'M']:
            weights_list = []
            for layer_idx in range(n_layers):
                key = f'layer{layer_idx}_{qkvo}_weights_compress'
                if key in routing_info:
                    weights_list.append(routing_info[key].numpy())

            if weights_list:
                all_weights = np.concatenate(weights_list, axis=0)
                avg_weights = all_weights.mean(axis=(0, 1))
                compress_counts[qkvo] = avg_weights
                print(f"{qkvo}: {[f'{w:.3f}' for w in avg_weights]}")

    print("\n" + "="*60)
    print("[ExpandNeurons Hard Selection]")
    print("="*60)

    # v9.1: hard selection (idx_expand)
    expand_counts = np.zeros(n_expand)
    for layer_idx in range(n_layers):
        key = f'layer{layer_idx}_O_idx_expand'
        if key in routing_info:
            indices = routing_info[key].numpy().flatten()
            for idx in indices:
                if 0 <= idx < n_expand:
                    expand_counts[idx] += 1

    if expand_counts.sum() > 0:
        probs = expand_counts / expand_counts.sum()
        print(f"O: {[f'{p:.3f}' for p in probs]}")
    else:
        # Fallback: soft weights (v9.0)
        expand_weights_list = []
        for layer_idx in range(n_layers):
            key = f'layer{layer_idx}_O_weights_expand'
            if key in routing_info:
                expand_weights_list.append(routing_info[key].numpy())

        if expand_weights_list:
            print("(Using soft weights - v9.0 style)")
            all_weights = np.concatenate(expand_weights_list, axis=0)
            avg_weights = all_weights.mean(axis=(0, 1))
            print(f"O: {[f'{w:.3f}' for w in avg_weights]}")

    return compress_counts


def analyze_layer_patterns(routing_info, n_layers, n_reflect_d, n_reflect_r):
    """레이어별 패턴 분석"""
    print("\n" + "="*60)
    print("[Layer-wise Patterns]")
    print("="*60)

    for layer_idx in range(n_layers):
        # 해당 레이어의 reflect_d 사용량
        layer_counts_d = np.zeros(n_reflect_d)
        layer_counts_r = np.zeros(n_reflect_r)

        for qkvo in ['Q', 'K', 'V', 'O', 'M']:
            key_d = f'layer{layer_idx}_{qkvo}_indices_d'
            key_r = f'layer{layer_idx}_{qkvo}_indices_r'

            if key_d in routing_info:
                indices = routing_info[key_d].numpy().flatten()
                for idx in indices:
                    if 0 <= idx < n_reflect_d:
                        layer_counts_d[idx] += 1

            if key_r in routing_info:
                indices = routing_info[key_r].numpy().flatten()
                for idx in indices:
                    if 0 <= idx < n_reflect_r:
                        layer_counts_r[idx] += 1

        top3_d = np.argsort(layer_counts_d)[-3:][::-1]
        top3_counts_d = layer_counts_d[top3_d].astype(int)

        print(f"Layer {layer_idx}:")
        print(f"  reflect_d top-3 = {top3_d.tolist()} (counts: {top3_counts_d.tolist()})")

        if layer_counts_r.sum() > 0:
            top3_r = np.argsort(layer_counts_r)[-3:][::-1]
            top3_counts_r = layer_counts_r[top3_r].astype(int)
            print(f"  reflect_r top-3 = {top3_r.tolist()} (counts: {top3_counts_r.tolist()})")


def visualize_results(reflect_d_counts, reflect_r_counts, jaccard_scores,
                     compress_counts, gate_stats, n_layers, save_path=None):
    """결과 시각화"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. reflect_d usage bar chart
    ax1 = axes[0, 0]
    ax1.bar(range(len(reflect_d_counts)), reflect_d_counts, color='steelblue', alpha=0.7)
    ax1.set_xlabel('reflect_d index')
    ax1.set_ylabel('Usage count')
    ax1.set_title('reflect_d Usage Distribution (d_model space)')

    # 2. reflect_r usage bar chart
    ax2 = axes[0, 1]
    if reflect_r_counts.sum() > 0:
        ax2.bar(range(len(reflect_r_counts)), reflect_r_counts, color='darkorange', alpha=0.7)
        ax2.set_xlabel('reflect_r index')
        ax2.set_ylabel('Usage count')
        ax2.set_title('reflect_r Usage Distribution (rank space)')
    else:
        ax2.text(0.5, 0.5, 'No reflect_r data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('reflect_r Usage Distribution')

    # 3. Q-K Jaccard per layer
    ax3 = axes[0, 2]
    if jaccard_scores:
        ax3.plot(range(len(jaccard_scores)), jaccard_scores, 'o-', color='green', linewidth=2, markersize=8)
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Jaccard Similarity')
        ax3.set_title('Q-K Overlap by Layer')
        ax3.set_xticks(range(len(jaccard_scores)))
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)

    # 4. Compress selection heatmap
    ax4 = axes[1, 0]
    if compress_counts:
        types = list(compress_counts.keys())
        n_compress = len(list(compress_counts.values())[0])
        heatmap_data = np.array([compress_counts[t] for t in types])

        im = ax4.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
        ax4.set_yticks(range(len(types)))
        ax4.set_yticklabels(types)
        ax4.set_xticks(range(n_compress))
        ax4.set_xticklabels([f'C{i}' for i in range(n_compress)])
        ax4.set_xlabel('CompressNeuron')
        ax4.set_ylabel('Type')
        ax4.set_title('CompressNeurons Selection (Hard)')

        for i in range(len(types)):
            for j in range(n_compress):
                ax4.text(j, i, f'{heatmap_data[i, j]:.2f}',
                        ha='center', va='center', fontsize=10)

        plt.colorbar(im, ax=ax4)

    # 5. Gate values distribution
    ax5 = axes[1, 1]
    if gate_stats:
        gate_keys = list(gate_stats.keys())
        means = [np.mean([s['mean'] for s in gate_stats[k]]) for k in gate_keys]
        stds = [np.mean([s['std'] for s in gate_stats[k]]) for k in gate_keys]

        x = range(len(gate_keys))
        ax5.bar(x, means, yerr=stds, capsize=5, color='purple', alpha=0.7)
        ax5.set_xticks(x)
        ax5.set_xticklabels(gate_keys, rotation=45, ha='right')
        ax5.set_ylabel('Gate Value')
        ax5.set_title('Gating Values (mean ± std)')
        ax5.set_ylim(0, 1)
        ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    else:
        ax5.text(0.5, 0.5, 'No gate data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Gating Values')

    # 6. Summary text
    ax6 = axes[1, 2]
    ax6.axis('off')

    summary_text = "DAWN v9.1 Analysis Summary\n"
    summary_text += "=" * 30 + "\n\n"

    if reflect_d_counts.sum() > 0:
        eff_rank_d = compute_effective_rank(reflect_d_counts)
        gini_d = compute_gini(reflect_d_counts)
        summary_text += f"reflect_d:\n"
        summary_text += f"  Effective rank: {eff_rank_d:.1f}/{len(reflect_d_counts)}\n"
        summary_text += f"  Gini: {gini_d:.3f}\n\n"

    if reflect_r_counts.sum() > 0:
        eff_rank_r = compute_effective_rank(reflect_r_counts)
        gini_r = compute_gini(reflect_r_counts)
        summary_text += f"reflect_r:\n"
        summary_text += f"  Effective rank: {eff_rank_r:.1f}/{len(reflect_r_counts)}\n"
        summary_text += f"  Gini: {gini_r:.3f}\n\n"

    if jaccard_scores:
        summary_text += f"Q-K Overlap:\n"
        summary_text += f"  Avg Jaccard: {np.mean(jaccard_scores):.3f}\n\n"

    if gate_stats:
        summary_text += f"Gating:\n"
        for k, v in gate_stats.items():
            avg = np.mean([s['mean'] for s in v])
            summary_text += f"  {k}: {avg:.3f}\n"

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='DAWN v9.1 Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint (best_model.pt)')
    parser.add_argument('--version', type=str, default=None,
                       help='Model version (auto-detected if not specified)')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to validation data (optional)')
    parser.add_argument('--max-batches', type=int, default=50,
                       help='Maximum batches to analyze')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for analysis')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    # 디바이스 설정
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 모델 로드
    print(f"\nLoading model from: {args.checkpoint}")
    model, config, version = load_model(args.checkpoint, device, args.version)

    n_layers = config.get('n_layers', 4)
    n_compress = config.get('n_compress', 4)
    n_expand = config.get('n_expand', 4)

    # v9.1: separate pools
    n_reflect_d = config.get('n_reflect_d', config.get('n_reflect', 64))
    n_reflect_r = config.get('n_reflect_r', config.get('n_reflect', 64))

    print(f"\nModel config:")
    print(f"  Version: {version}")
    print(f"  n_layers: {n_layers}")
    print(f"  n_compress: {n_compress}, n_expand: {n_expand}")
    print(f"  n_reflect_d: {n_reflect_d}, n_reflect_r: {n_reflect_r}")

    # 데이터 로드
    if args.data_path and os.path.exists(args.data_path):
        from torch.utils.data import DataLoader
        dataset = TextDataset([args.data_path], max_length=128)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_fn_dynamic_padding)
    else:
        # 랜덤 데이터로 분석
        print("\nNo data path provided, using random input for analysis")
        class DummyLoader:
            def __init__(self, batch_size, seq_len, vocab_size, num_batches):
                self.batch_size = batch_size
                self.seq_len = seq_len
                self.vocab_size = vocab_size
                self.num_batches = num_batches
                self.idx = 0

            def __iter__(self):
                self.idx = 0
                return self

            def __next__(self):
                if self.idx >= self.num_batches:
                    raise StopIteration
                self.idx += 1
                return {
                    'input_ids': torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
                }

        dataloader = DummyLoader(
            batch_size=args.batch_size,
            seq_len=128,
            vocab_size=config.get('vocab_size', 30522),
            num_batches=args.max_batches
        )

    # Routing info 수집
    print("\n" + "="*60)
    print(f"=== DAWN v{version} Analysis ===")
    print("="*60)

    routing_info = collect_routing_info(model, dataloader, device, args.max_batches)

    # 분석 실행
    reflect_d_counts, reflect_r_counts = analyze_reflection_usage(
        routing_info, n_reflect_d, n_reflect_r, n_layers
    )

    gate_stats = analyze_gating(routing_info, n_layers)

    jaccard_scores = analyze_qk_overlap(routing_info, n_layers)

    compress_counts = analyze_compress_expand_selection(
        routing_info, n_layers, n_compress, n_expand
    )

    analyze_layer_patterns(routing_info, n_layers, n_reflect_d, n_reflect_r)

    # 시각화
    if args.visualize or args.save_plot:
        visualize_results(
            reflect_d_counts, reflect_r_counts, jaccard_scores,
            compress_counts, gate_stats, n_layers,
            save_path=args.save_plot
        )

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
