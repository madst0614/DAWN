"""
DAWN v10.0 Comprehensive Neuron Analysis
=========================================

분석 항목:
1. 뉴런 활성화 분석 - Top 토큰, 토큰별 프로파일, POS 특화
2. 뉴런 클러스터링 - 활성화 패턴, weight 기반, 상관관계
3. 레이어별 역할 분석 - 특화도, 정보 흐름, Q/K/V/M 분화
4. Knowledge 뉴런 분석 - 특화, 클러스터
5. Ablation Study - 뉴런/그룹/레이어 제거
6. Attention 패턴과 뉴런 관계
7. 시각화

Usage:
    python analyze_neurons_comprehensive.py \
        --checkpoint <path> \
        --val_data <path> \
        --output_dir ./neuron_analysis
"""

import argparse
import json
import math
import os
import pickle
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def get_underlying_model(model):
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def simple_pos_tag(token: str) -> str:
    """Simple rule-based POS tagging"""
    token_lower = token.lower().strip()
    if not token_lower or token_lower.startswith('[') or token_lower.startswith('##'):
        return 'OTHER'
    if token_lower in {'the', 'a', 'an', 'this', 'that', 'these', 'those'}:
        return 'DET'
    if token_lower in {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'}:
        return 'AUX'
    if token_lower in {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'}:
        return 'PRON'
    if token_lower in {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into', 'through', 'during', 'before', 'after'}:
        return 'ADP'
    if token_lower in {'and', 'or', 'but', 'if', 'when', 'while', 'because', 'although', 'unless', 'since'}:
        return 'CONJ'
    if token_lower.isdigit() or token_lower.replace('.', '').replace(',', '').isdigit():
        return 'NUM'
    if token_lower.endswith('ing'):
        return 'VERB_ING'
    if token_lower.endswith('ed'):
        return 'VERB_ED'
    if token_lower.endswith('ly'):
        return 'ADV'
    if token_lower.endswith(('tion', 'sion', 'ment', 'ness', 'ity')):
        return 'NOUN_SUFFIX'
    if token_lower.endswith(('ful', 'less', 'ous', 'ive', 'able', 'ible')):
        return 'ADJ_SUFFIX'
    return 'OTHER'


class NeuronAnalyzer:
    """종합 뉴런 분석기"""

    def __init__(self, model, tokenizer, device):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device

        self.n_layers = self.model.n_layers
        self.d_model = self.model.d_model
        self.rank = self.model.rank
        self.n_compress = self.model.n_compress
        self.n_expand = self.model.n_expand
        self.n_knowledge = self.model.n_knowledge
        self.knowledge_k = self.model.knowledge_k

        print(f"\n{'='*60}")
        print("Neuron Analyzer Initialized")
        print(f"{'='*60}")
        print(f"n_compress: {self.n_compress}, n_expand: {self.n_expand}")
        print(f"n_knowledge: {self.n_knowledge}")
        print(f"n_layers: {self.n_layers}")

    # ============================================================
    # 1. 뉴런 활성화 분석 (GPU 최적화)
    # ============================================================

    @torch.no_grad()
    def analyze_neuron_activations(self, dataloader, max_batches: int = 100) -> Dict:
        """
        GPU 최적화된 뉴런 활성화 분석
        - 벡터화 연산으로 Python 루프 제거
        - GPU에서 직접 카운트
        """
        print(f"\n{'='*60}")
        print("1. NEURON ACTIVATION ANALYSIS (GPU Optimized)")
        print(f"{'='*60}")

        self.model.eval()
        vocab_size = self.tokenizer.vocab_size

        # GPU 텐서로 카운트 (Layer 0만 저장 - 메모리 절약)
        # [comp][neuron_id, token_id] = count
        neuron_token_matrix = {
            comp: torch.zeros(self.n_compress, vocab_size, device=self.device, dtype=torch.float32)
            for comp in ['Q', 'K', 'V', 'M']
        }

        # 레이어별 뉴런 사용 빈도 [comp][layer, neuron] = count
        layer_neuron_counts = {
            comp: torch.zeros(self.n_layers, self.n_compress, device=self.device, dtype=torch.float32)
            for comp in ['Q', 'K', 'V', 'M']
        }

        # POS별 뉴런 카운트를 위한 token→POS 매핑 (배치로 처리)
        # 미리 전체 vocab에 대해 POS 계산
        print("  Building POS mapping for vocabulary...")
        token_to_pos = {}
        pos_to_idx = {}
        pos_list = ['DET', 'AUX', 'PRON', 'ADP', 'CONJ', 'NUM', 'VERB_ING', 'VERB_ED', 'ADV', 'NOUN_SUFFIX', 'ADJ_SUFFIX', 'OTHER']
        for i, pos in enumerate(pos_list):
            pos_to_idx[pos] = i

        # Vocab의 각 토큰에 대해 POS 태그 미리 계산
        pos_indices = torch.zeros(vocab_size, dtype=torch.long, device=self.device)
        for tid in range(min(vocab_size, 50000)):  # 상위 50K만
            token_str = self.tokenizer.decode([tid]).strip()
            pos = simple_pos_tag(token_str)
            pos_indices[tid] = pos_to_idx.get(pos, pos_to_idx['OTHER'])

        # POS별 뉴런 카운트 [pos, neuron]
        pos_neuron_matrix = {
            comp: torch.zeros(len(pos_list), self.n_compress, device=self.device, dtype=torch.float32)
            for comp in ['Q', 'K', 'V', 'M']
        }

        print("  Processing batches...")
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Activation Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)  # [B, S]
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                attn = routing_info['attention']
                mem = routing_info['memory']

                comp_data = {
                    'Q': attn['Q'],
                    'K': attn['K'],
                    'V': attn['V'],
                    'M': mem['M'],
                }

                for comp, data in comp_data.items():
                    # v10.0: weights only [B, S, N], v10.1: weights + indices
                    weights = data['weights']  # [B, S, N] or [B, S, k]

                    if 'indices' in data:
                        # v10.1: Top-K selected
                        indices = data['indices']  # [B, S, k]
                        k = indices.shape[-1]
                    else:
                        # v10.0: All neurons, get top-k from weights
                        k = min(8, weights.shape[-1])
                        _, indices = torch.topk(weights, k, dim=-1)  # [B, S, k]

                    # 뉴런 사용 빈도 업데이트 (GPU에서 bincount)
                    flat_indices = indices.view(-1)  # [B*S*k]
                    counts = torch.bincount(flat_indices, minlength=self.n_compress).float()
                    layer_neuron_counts[comp][layer_idx] += counts

                    # Layer 0에서만 토큰-뉴런 매핑 (메모리 절약)
                    if layer_idx == 0:
                        # [B, S] -> [B, S, k] (브로드캐스트용)
                        token_ids_exp = input_ids.unsqueeze(-1).expand(-1, -1, k)  # [B, S, k]

                        # Flatten
                        flat_tokens = token_ids_exp.reshape(-1)  # [B*S*k]
                        flat_neurons = indices.reshape(-1)  # [B*S*k]

                        # 2D index로 scatter_add
                        # neuron_token_matrix[comp][neuron, token] += 1
                        idx_2d = flat_neurons * vocab_size + flat_tokens
                        ones = torch.ones_like(idx_2d, dtype=torch.float32)
                        neuron_token_matrix[comp].view(-1).scatter_add_(0, idx_2d, ones)

                        # POS별 뉴런 카운트
                        token_pos = pos_indices[flat_tokens]  # [B*S*k]
                        idx_pos = token_pos * self.n_compress + flat_neurons
                        pos_neuron_matrix[comp].view(-1).scatter_add_(0, idx_pos, ones)

        # CPU로 이동 및 Counter 형식으로 변환 (호환성)
        print("  Converting to analysis format...")
        neuron_token_counts = {
            comp: {0: {n: Counter() for n in range(self.n_compress)}}
            for comp in ['Q', 'K', 'V', 'M']
        }
        neuron_token_weights = {
            comp: {0: {n: defaultdict(float) for n in range(self.n_compress)}}
            for comp in ['Q', 'K', 'V', 'M']
        }

        # 각 뉴런의 Top 토큰만 추출 (메모리 절약)
        for comp in ['Q', 'K', 'V', 'M']:
            matrix = neuron_token_matrix[comp]  # [n_compress, vocab_size]
            for neuron_id in range(self.n_compress):
                row = matrix[neuron_id]
                topk_counts, topk_tokens = torch.topk(row, k=min(100, (row > 0).sum().item() or 1))
                for cnt, tid in zip(topk_counts.cpu().tolist(), topk_tokens.cpu().tolist()):
                    if cnt > 0:
                        neuron_token_counts[comp][0][neuron_id][tid] = int(cnt)

        # POS 카운트 변환
        pos_neuron_counts = defaultdict(lambda: {comp: Counter() for comp in ['Q', 'K', 'V', 'M']})
        for comp in ['Q', 'K', 'V', 'M']:
            matrix = pos_neuron_matrix[comp].cpu()  # [n_pos, n_compress]
            for pos_idx, pos in enumerate(pos_list):
                row = matrix[pos_idx]
                for neuron_id in range(self.n_compress):
                    cnt = int(row[neuron_id].item())
                    if cnt > 0:
                        pos_neuron_counts[pos][comp][neuron_id] = cnt

        # 토큰별 뉴런 프로파일 (Q, Layer 0 기준)
        token_neuron_profile = defaultdict(lambda: {comp: Counter() for comp in ['Q', 'K', 'V', 'M']})
        matrix = neuron_token_matrix['Q'].T  # [vocab_size, n_compress]
        # Top 활성화 토큰만
        token_totals = matrix.sum(dim=1)  # [vocab_size]
        top_tokens = torch.topk(token_totals, k=min(1000, (token_totals > 0).sum().item() or 1))[1]
        for tid in top_tokens.cpu().tolist():
            row = matrix[tid].cpu()
            for neuron_id in range(self.n_compress):
                cnt = int(row[neuron_id].item())
                if cnt > 0:
                    token_neuron_profile[tid]['Q'][neuron_id] = cnt

        results = {
            'neuron_top_tokens': {},
            'token_neuron_profiles': {},
            'pos_neuron_specialization': {},
        }

        # 1.1 뉴런별 Top 토큰
        print("\n--- 1.1 Neuron Top Tokens (Layer 0, Q) ---")
        for neuron_id in range(min(10, self.n_compress)):
            token_counts = neuron_token_counts['Q'][0][neuron_id]
            top_tokens = token_counts.most_common(10)

            if top_tokens:
                tokens_str = []
                for tid, cnt in top_tokens[:5]:
                    tok = self.tokenizer.decode([tid]).strip()
                    tokens_str.append(f"'{tok}'({cnt})")
                print(f"  Neuron {neuron_id}: {', '.join(tokens_str)}")

        # Save detailed results (Layer 0 only - memory optimized)
        for comp in ['Q', 'K', 'V', 'M']:
            results['neuron_top_tokens'][comp] = {}
            results['neuron_top_tokens'][comp]['L0'] = {}
            for neuron_id in range(self.n_compress):
                top_tokens = neuron_token_counts[comp][0][neuron_id].most_common(50)
                if top_tokens:
                    results['neuron_top_tokens'][comp]['L0'][neuron_id] = [
                        {'token': self.tokenizer.decode([tid]).strip(),
                         'token_id': tid,
                         'count': cnt}
                        for tid, cnt in top_tokens
                    ]

        # 1.2 토큰별 뉴런 프로파일
        print("\n--- 1.2 Token Neuron Profiles (Top tokens) ---")
        # Find most common tokens
        all_token_counts = Counter()
        for token_id, profile in token_neuron_profile.items():
            total = sum(profile['Q'].values())
            all_token_counts[token_id] = total

        top_tokens = all_token_counts.most_common(20)
        for token_id, total_count in top_tokens[:10]:
            token_str = self.tokenizer.decode([token_id]).strip()
            profile = token_neuron_profile[token_id]

            top_neurons_q = profile['Q'].most_common(5)
            neurons_str = ', '.join([f'{n}' for n, _ in top_neurons_q])
            print(f"  '{token_str}' (n={total_count}): Q neurons [{neurons_str}]")

            results['token_neuron_profiles'][token_str] = {
                'token_id': token_id,
                'total_count': total_count,
                'Q_top_neurons': [(n, c) for n, c in profile['Q'].most_common(10)],
                'K_top_neurons': [(n, c) for n, c in profile['K'].most_common(10)],
                'V_top_neurons': [(n, c) for n, c in profile['V'].most_common(10)],
                'M_top_neurons': [(n, c) for n, c in profile['M'].most_common(10)],
            }

        # 1.3 품사별 뉴런 특화
        print("\n--- 1.3 POS Neuron Specialization ---")
        for pos in sorted(pos_neuron_counts.keys()):
            if pos == 'OTHER':
                continue

            q_neurons = pos_neuron_counts[pos]['Q']
            total = sum(q_neurons.values())
            if total < 100:
                continue

            top_neurons = q_neurons.most_common(5)
            neurons_str = ', '.join([f'{n}({c})' for n, c in top_neurons[:3]])
            print(f"  {pos:12s} (n={total:5d}): [{neurons_str}]")

            results['pos_neuron_specialization'][pos] = {
                'total_count': total,
                'Q_top_neurons': [(n, c) for n, c in q_neurons.most_common(20)],
            }

        return results, neuron_token_counts, pos_neuron_counts

    # ============================================================
    # 2. 뉴런 클러스터링
    # ============================================================

    def analyze_neuron_clustering(self, neuron_token_counts: Dict, max_tokens: int = 1000) -> Dict:
        """
        2.1 활성화 패턴 기반 클러스터링
        2.2 뉴런 weight 기반 클러스터링
        2.3 뉴런 상관관계
        """
        if not HAS_SKLEARN:
            print("sklearn not available, skipping clustering")
            return {}

        print(f"\n{'='*60}")
        print("2. NEURON CLUSTERING")
        print(f"{'='*60}")

        results = {}

        # 2.1 활성화 패턴 기반 클러스터링
        print("\n--- 2.1 Activation Pattern Clustering ---")

        # Build activation vectors for each neuron (Q, Layer 0)
        all_tokens = set()
        for neuron_id in range(self.n_compress):
            all_tokens.update(neuron_token_counts['Q'][0][neuron_id].keys())

        # Limit to top tokens
        token_totals = Counter()
        for neuron_id in range(self.n_compress):
            for tid, cnt in neuron_token_counts['Q'][0][neuron_id].items():
                token_totals[tid] += cnt

        top_tokens = [tid for tid, _ in token_totals.most_common(max_tokens)]
        token_to_idx = {tid: i for i, tid in enumerate(top_tokens)}

        # Build activation matrix [n_compress, n_tokens]
        activation_matrix = np.zeros((self.n_compress, len(top_tokens)))
        for neuron_id in range(self.n_compress):
            for tid, cnt in neuron_token_counts['Q'][0][neuron_id].items():
                if tid in token_to_idx:
                    activation_matrix[neuron_id, token_to_idx[tid]] = cnt

        # Normalize
        row_sums = activation_matrix.sum(axis=1, keepdims=True) + 1e-10
        activation_matrix_norm = activation_matrix / row_sums

        # K-means clustering
        results['activation_clustering'] = {}
        for n_clusters in [8, 16, 32]:
            if n_clusters > self.n_compress:
                continue

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(activation_matrix_norm)

            cluster_sizes = np.bincount(labels, minlength=n_clusters)

            # Find representative tokens for each cluster
            cluster_tokens = {}
            for c in range(n_clusters):
                cluster_neurons = np.where(labels == c)[0]
                # Average activation for cluster
                cluster_avg = activation_matrix_norm[cluster_neurons].mean(axis=0)
                top_token_indices = np.argsort(cluster_avg)[-10:][::-1]
                cluster_tokens[c] = [
                    self.tokenizer.decode([top_tokens[i]]).strip()
                    for i in top_token_indices
                ]

            results['activation_clustering'][f'k{n_clusters}'] = {
                'labels': labels.tolist(),
                'sizes': cluster_sizes.tolist(),
                'cluster_tokens': cluster_tokens,
            }

            print(f"  K={n_clusters}: sizes={cluster_sizes.tolist()[:8]}...")
            for c in range(min(4, n_clusters)):
                tokens_str = ', '.join(cluster_tokens[c][:5])
                print(f"    Cluster {c} (n={cluster_sizes[c]}): [{tokens_str}]")

        # 2.2 뉴런 weight 기반 클러스터링
        print("\n--- 2.2 Weight-based Clustering ---")

        compress_neurons = self.model.shared_neurons.compress_neurons.data.cpu().numpy()
        N, D, R = compress_neurons.shape
        neurons_flat = compress_neurons.reshape(N, D * R)

        # PCA for dimensionality reduction
        pca = PCA(n_components=min(50, N))
        neurons_pca = pca.fit_transform(neurons_flat)

        # K-means on PCA
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        weight_labels = kmeans.fit_predict(neurons_pca)

        results['weight_clustering'] = {
            'labels': weight_labels.tolist(),
            'sizes': np.bincount(weight_labels).tolist(),
        }

        print(f"  Weight clustering (k=8): sizes={np.bincount(weight_labels).tolist()}")

        # 2.3 뉴런 상관관계 (공출현)
        print("\n--- 2.3 Neuron Co-occurrence ---")

        # Co-occurrence matrix
        cooccur = np.zeros((self.n_compress, self.n_compress))
        for neuron_id in range(self.n_compress):
            tokens = set(neuron_token_counts['Q'][0][neuron_id].keys())
            for other_id in range(neuron_id + 1, self.n_compress):
                other_tokens = set(neuron_token_counts['Q'][0][other_id].keys())
                overlap = len(tokens & other_tokens)
                cooccur[neuron_id, other_id] = overlap
                cooccur[other_id, neuron_id] = overlap

        # Normalize by total tokens
        cooccur_norm = cooccur / (cooccur.max() + 1e-10)

        # Find most correlated pairs
        top_pairs = []
        for i in range(self.n_compress):
            for j in range(i + 1, self.n_compress):
                top_pairs.append((i, j, cooccur_norm[i, j]))
        top_pairs.sort(key=lambda x: x[2], reverse=True)

        results['cooccurrence'] = {
            'top_pairs': [(i, j, float(c)) for i, j, c in top_pairs[:50]],
            'matrix_sample': cooccur_norm[:20, :20].tolist(),
        }

        print(f"  Top correlated pairs:")
        for i, j, corr in top_pairs[:5]:
            print(f"    Neuron {i} - Neuron {j}: {corr:.3f}")

        return results

    # ============================================================
    # 3. 레이어별 역할 분석
    # ============================================================

    @torch.no_grad()
    def analyze_layer_roles(self, dataloader, max_batches: int = 50) -> Dict:
        """
        3.1 레이어별 뉴런 특화도
        3.2 정보 흐름
        3.3 Q/K/V/M 역할 분화
        """
        print(f"\n{'='*60}")
        print("3. LAYER ROLE ANALYSIS")
        print(f"{'='*60}")

        self.model.eval()
        results = {}

        # Layer-wise neuron usage
        layer_neuron_usage = {
            comp: {l: torch.zeros(self.n_compress, device=self.device)
                   for l in range(self.n_layers)}
            for comp in ['Q', 'K', 'V', 'M']
        }

        # Token trajectory: same token across layers
        token_trajectories = defaultdict(lambda: {l: [] for l in range(self.n_layers)})

        total_tokens = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Layer Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape
            total_tokens += B * S

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                attn = routing_info['attention']
                mem = routing_info['memory']

                comp_data = {
                    'Q': attn['Q']['weights'],
                    'K': attn['K']['weights'],
                    'V': attn['V']['weights'],
                    'M': mem['M']['weights'],
                }

                for comp, weights in comp_data.items():
                    layer_neuron_usage[comp][layer_idx] += weights.sum(dim=(0, 1))

                # Track trajectories for sample tokens
                if batch_idx < 5:
                    for b in range(min(2, B)):
                        for s in range(min(10, S)):
                            token_id = input_ids[b, s].item()
                            q_weights = attn['Q']['weights'][b, s]
                            top_neuron = q_weights.argmax().item()
                            token_trajectories[token_id][layer_idx].append(top_neuron)

        # 3.1 레이어별 뉴런 특화도
        print("\n--- 3.1 Layer Neuron Specialization ---")
        results['layer_specialization'] = {}

        print(f"{'Layer':<8} {'Comp':<6} {'Eff.Rank':<12} {'Gini':<10} {'Top5%':<10}")
        print("-" * 46)

        for layer_idx in range(self.n_layers):
            results['layer_specialization'][f'L{layer_idx}'] = {}

            for comp in ['Q', 'K', 'V', 'M']:
                usage = layer_neuron_usage[comp][layer_idx].cpu().numpy()
                usage_norm = usage / (usage.sum() + 1e-10)

                # Effective rank
                entropy = -np.sum(usage_norm * np.log(usage_norm + 1e-10))
                eff_rank = np.exp(entropy)

                # Gini coefficient
                sorted_usage = np.sort(usage_norm)
                n = len(sorted_usage)
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * sorted_usage) - (n + 1) * np.sum(sorted_usage)) / (n * np.sum(sorted_usage) + 1e-10)

                # Top 5% concentration
                top_k = max(1, self.n_compress // 20)
                top_5_pct = np.sort(usage_norm)[-top_k:].sum()

                results['layer_specialization'][f'L{layer_idx}'][comp] = {
                    'eff_rank': float(eff_rank),
                    'gini': float(gini),
                    'top_5_pct': float(top_5_pct),
                }

                if comp == 'Q':
                    print(f"L{layer_idx:<7} {comp:<6} {eff_rank:<12.1f} {gini:<10.3f} {top_5_pct:<10.1%}")

        # 3.2 정보 흐름 (토큰 trajectory)
        print("\n--- 3.2 Information Flow (Token Trajectories) ---")
        results['token_trajectories'] = {}

        sample_tokens = list(token_trajectories.keys())[:10]
        for token_id in sample_tokens:
            token_str = self.tokenizer.decode([token_id]).strip()
            trajectory = []
            for layer_idx in range(self.n_layers):
                neurons = token_trajectories[token_id][layer_idx]
                if neurons:
                    most_common = Counter(neurons).most_common(1)[0][0]
                    trajectory.append(most_common)

            if len(trajectory) == self.n_layers:
                results['token_trajectories'][token_str] = trajectory
                traj_str = ' → '.join(map(str, trajectory))
                print(f"  '{token_str}': {traj_str}")

        # 3.3 Q/K/V/M 역할 분화
        print("\n--- 3.3 Q/K/V/M Role Differentiation ---")
        results['qkvm_correlation'] = {}

        for layer_idx in range(self.n_layers):
            q_usage = layer_neuron_usage['Q'][layer_idx].cpu().numpy()
            k_usage = layer_neuron_usage['K'][layer_idx].cpu().numpy()
            v_usage = layer_neuron_usage['V'][layer_idx].cpu().numpy()
            m_usage = layer_neuron_usage['M'][layer_idx].cpu().numpy()

            # Normalize
            q_norm = q_usage / (np.linalg.norm(q_usage) + 1e-10)
            k_norm = k_usage / (np.linalg.norm(k_usage) + 1e-10)
            v_norm = v_usage / (np.linalg.norm(v_usage) + 1e-10)
            m_norm = m_usage / (np.linalg.norm(m_usage) + 1e-10)

            # Cosine similarities
            qk_sim = float(np.dot(q_norm, k_norm))
            qv_sim = float(np.dot(q_norm, v_norm))
            qm_sim = float(np.dot(q_norm, m_norm))
            kv_sim = float(np.dot(k_norm, v_norm))

            results['qkvm_correlation'][f'L{layer_idx}'] = {
                'Q-K': qk_sim,
                'Q-V': qv_sim,
                'Q-M': qm_sim,
                'K-V': kv_sim,
            }

            if layer_idx == 0:
                print(f"  L{layer_idx}: Q-K={qk_sim:.3f}, Q-V={qv_sim:.3f}, Q-M={qm_sim:.3f}, K-V={kv_sim:.3f}")

        return results

    # ============================================================
    # 4. Knowledge 뉴런 분석
    # ============================================================

    @torch.no_grad()
    def analyze_knowledge_neurons(self, dataloader, max_batches: int = 50) -> Dict:
        """
        4.1 Knowledge 뉴런 특화
        4.2 Knowledge 클러스터
        """
        print(f"\n{'='*60}")
        print("4. KNOWLEDGE NEURON ANALYSIS")
        print(f"{'='*60}")

        self.model.eval()
        results = {}

        # Knowledge neuron usage - GPU vectorized
        knowledge_usage = torch.zeros(self.n_knowledge, device=self.device, dtype=torch.float32)
        vocab_size = self.tokenizer.vocab_size
        # Matrix for neuron-token co-occurrence (on GPU)
        knowledge_token_matrix = torch.zeros(self.n_knowledge, vocab_size, device=self.device, dtype=torch.float32)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Knowledge Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for routing_info in routing_infos:
                mem = routing_info['memory']
                k_idx = mem['knowledge_indices']  # [B, S, knowledge_k]
                k_weights = mem['knowledge_weights']  # [B, S, knowledge_k]
                knowledge_k = k_idx.shape[-1]

                # GPU vectorized: scatter_add for usage
                flat_idx = k_idx.view(-1)  # [B*S*k]
                flat_weights = k_weights.view(-1).float()  # [B*S*k]
                knowledge_usage.scatter_add_(0, flat_idx, flat_weights)

                # GPU vectorized: token co-occurrence matrix
                # Expand token ids to match k_idx shape
                tokens_expanded = input_ids.unsqueeze(-1).expand(-1, -1, knowledge_k)  # [B, S, k]
                flat_tokens = tokens_expanded.reshape(-1)  # [B*S*k]
                flat_neurons = flat_idx  # [B*S*k]

                # Create 1D index for 2D matrix and scatter_add
                idx_2d = flat_neurons * vocab_size + flat_tokens
                ones = torch.ones_like(idx_2d, dtype=torch.float32)
                knowledge_token_matrix.view(-1).scatter_add_(0, idx_2d, ones)

        # Convert matrix to Counter-like dict for compatibility with rest of code
        knowledge_token_counts = {}
        knowledge_token_matrix_cpu = knowledge_token_matrix.cpu().numpy()
        for n in range(self.n_knowledge):
            row = knowledge_token_matrix_cpu[n]
            nonzero_idx = np.nonzero(row)[0]
            knowledge_token_counts[n] = Counter({int(tid): int(row[tid]) for tid in nonzero_idx})

        # 4.1 Knowledge 뉴런 특화
        print("\n--- 4.1 Knowledge Neuron Specialization ---")

        usage_cpu = knowledge_usage.cpu().numpy()
        top_neurons = np.argsort(usage_cpu)[-20:][::-1]

        results['knowledge_top_neurons'] = {}
        for rank, neuron_id in enumerate(top_neurons[:10]):
            top_tokens = knowledge_token_counts[neuron_id].most_common(10)
            tokens_str = ', '.join([f"'{self.tokenizer.decode([tid]).strip()}'" for tid, _ in top_tokens[:5]])
            print(f"  #{rank+1} Neuron {neuron_id} (usage={usage_cpu[neuron_id]:.1f}): {tokens_str}")

            results['knowledge_top_neurons'][int(neuron_id)] = {
                'usage': float(usage_cpu[neuron_id]),
                'top_tokens': [
                    {'token': self.tokenizer.decode([tid]).strip(), 'count': cnt}
                    for tid, cnt in top_tokens
                ]
            }

        # 4.2 Knowledge 클러스터
        print("\n--- 4.2 Knowledge Clustering ---")

        if HAS_SKLEARN:
            knowledge_K = self.model.shared_neurons.knowledge_K.data.cpu().numpy()

            kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
            k_labels = kmeans.fit_predict(knowledge_K)

            cluster_sizes = np.bincount(k_labels)
            print(f"  K-means (k=8): sizes={cluster_sizes.tolist()}")

            results['knowledge_clustering'] = {
                'labels': k_labels.tolist(),
                'sizes': cluster_sizes.tolist(),
            }

            # Find representative neurons for each cluster
            for c in range(8):
                cluster_neurons = np.where(k_labels == c)[0]
                # Highest usage in cluster
                cluster_usage = [(n, usage_cpu[n]) for n in cluster_neurons]
                cluster_usage.sort(key=lambda x: x[1], reverse=True)

                if cluster_usage:
                    top_neuron = cluster_usage[0][0]
                    top_tokens = knowledge_token_counts[top_neuron].most_common(3)
                    tokens_str = ', '.join([f"'{self.tokenizer.decode([tid]).strip()}'" for tid, _ in top_tokens])
                    print(f"    Cluster {c} (n={cluster_sizes[c]}): top neuron {top_neuron} → {tokens_str}")

        return results

    # ============================================================
    # 5. Ablation Study
    # ============================================================

    @torch.no_grad()
    def run_ablation_study(self, dataloader, max_batches: int = 20) -> Dict:
        """
        5.1 개별 뉴런 제거
        5.2 뉴런 그룹 제거
        5.3 레이어별 뉴런 제거
        """
        print(f"\n{'='*60}")
        print("5. ABLATION STUDY")
        print(f"{'='*60}")

        self.model.eval()
        results = {}

        # Baseline perplexity
        print("\n--- Computing baseline perplexity ---")
        baseline_loss = 0.0
        n_tokens = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Baseline", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            loss, _ = self.model(input_ids, labels=labels)
            baseline_loss += loss.item() * input_ids.numel()
            n_tokens += input_ids.numel()

        baseline_ppl = math.exp(baseline_loss / n_tokens)
        results['baseline_ppl'] = baseline_ppl
        print(f"  Baseline PPL: {baseline_ppl:.2f}")

        # Note: Full ablation study requires modifying model weights temporarily
        # This is a simplified version that shows the framework

        # 5.1 개별 뉴런 제거 (top neurons by usage)
        print("\n--- 5.1 Individual Neuron Ablation (simulated) ---")
        print("  Note: Full ablation requires model modification")
        print("  Top neurons by usage would be tested")

        results['individual_ablation'] = {
            'note': 'Full ablation requires temporary weight modification',
            'methodology': 'Zero out specific neuron weights and measure PPL change'
        }

        # 5.2 뉴런 그룹 제거
        print("\n--- 5.2 Neuron Group Ablation (simulated) ---")
        print("  Cluster-based ablation would test:")
        print("  - 'Grammar' cluster removal")
        print("  - 'Semantic' cluster removal")

        results['group_ablation'] = {
            'note': 'Requires clustering results + weight modification'
        }

        # 5.3 레이어별 뉴런 제거
        print("\n--- 5.3 Layer-wise Ablation (simulated) ---")
        print("  Would test: randomizing neurons at each layer")

        results['layer_ablation'] = {
            'note': 'Test importance of each layer by randomizing its routing'
        }

        return results

    # ============================================================
    # 6. Attention 패턴과 뉴런 관계
    # ============================================================

    @torch.no_grad()
    def analyze_attention_neuron_relation(self, dataloader, max_batches: int = 30) -> Dict:
        """
        6.1 Attention head별 뉴런 사용
        6.2 위치별 뉴런
        """
        print(f"\n{'='*60}")
        print("6. ATTENTION-NEURON RELATION")
        print(f"{'='*60}")

        self.model.eval()
        results = {}

        # Position-wise neuron usage
        position_neuron_usage = torch.zeros(128, self.n_compress, device=self.device)
        position_counts = torch.zeros(128, device=self.device)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Attention-Neuron", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # Layer 0 Q routing
            q_weights = routing_infos[0]['attention']['Q']['weights']  # [B, S, n_compress]

            for s in range(min(S, 128)):
                position_neuron_usage[s] += q_weights[:, s, :].sum(dim=0)
                position_counts[s] += B

        # 6.2 위치별 뉴런
        print("\n--- 6.2 Position-wise Neuron Usage ---")

        position_neuron_usage_norm = position_neuron_usage / (position_counts.unsqueeze(1) + 1e-10)

        # Compare early vs late positions
        early_avg = position_neuron_usage_norm[:16].mean(dim=0).cpu().numpy()
        late_avg = position_neuron_usage_norm[-16:].mean(dim=0).cpu().numpy()

        # Top neurons for early positions
        early_top = np.argsort(early_avg)[-10:][::-1]
        late_top = np.argsort(late_avg)[-10:][::-1]

        print(f"  Early positions (0-15) top neurons: {early_top.tolist()}")
        print(f"  Late positions (112-127) top neurons: {late_top.tolist()}")

        # Overlap
        overlap = len(set(early_top) & set(late_top))
        print(f"  Overlap: {overlap}/10")

        results['position_analysis'] = {
            'early_top_neurons': early_top.tolist(),
            'late_top_neurons': late_top.tolist(),
            'overlap': overlap,
        }

        return results

    # ============================================================
    # 7. 시각화
    # ============================================================

    def visualize_all(self, all_results: Dict, output_dir: str):
        """종합 시각화"""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        print(f"\n{'='*60}")
        print("7. VISUALIZATION")
        print(f"{'='*60}")

        os.makedirs(output_dir, exist_ok=True)

        # 7.1 뉴런 활성화 히트맵 (POS x Neuron)
        if 'pos_neuron_specialization' in all_results:
            fig, ax = plt.subplots(figsize=(16, 8))

            pos_list = sorted([p for p in all_results['pos_neuron_specialization'].keys() if p != 'OTHER'])
            heatmap_data = np.zeros((len(pos_list), min(50, self.n_compress)))

            for i, pos in enumerate(pos_list):
                top_neurons = all_results['pos_neuron_specialization'][pos]['Q_top_neurons']
                for neuron_id, count in top_neurons[:50]:
                    if neuron_id < 50:
                        heatmap_data[i, neuron_id] = count

            # Normalize rows
            row_sums = heatmap_data.sum(axis=1, keepdims=True) + 1e-10
            heatmap_data = heatmap_data / row_sums

            im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
            ax.set_yticks(range(len(pos_list)))
            ax.set_yticklabels(pos_list)
            ax.set_xlabel('Neuron ID')
            ax.set_ylabel('POS')
            ax.set_title('POS-Neuron Activation Heatmap (Q, Layer 0)')
            plt.colorbar(im, ax=ax)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pos_neuron_heatmap.png'), dpi=150)
            plt.close()
            print(f"  Saved: pos_neuron_heatmap.png")

        # 7.2 레이어별 뉴런 사용 분포
        if 'layer_specialization' in all_results:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            for layer_idx in range(min(8, self.n_layers)):
                ax = axes[layer_idx // 4, layer_idx % 4]

                data = all_results['layer_specialization'][f'L{layer_idx}']
                comps = ['Q', 'K', 'V', 'M']
                eff_ranks = [data[c]['eff_rank'] for c in comps]

                ax.bar(comps, eff_ranks, color=['steelblue', 'coral', 'green', 'purple'])
                ax.axhline(y=self.n_compress, color='r', linestyle='--', alpha=0.5)
                ax.set_title(f'Layer {layer_idx}')
                ax.set_ylabel('Effective Rank')
                ax.set_ylim(0, self.n_compress * 1.1)

            plt.suptitle('Layer-wise Effective Rank by Component')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'layer_effective_rank.png'), dpi=150)
            plt.close()
            print(f"  Saved: layer_effective_rank.png")

        # 7.3 Q/K/V/M 상관관계 히트맵
        if 'qkvm_correlation' in all_results:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            for layer_idx in range(min(8, self.n_layers)):
                ax = axes[layer_idx // 4, layer_idx % 4]

                data = all_results['qkvm_correlation'][f'L{layer_idx}']

                # Build correlation matrix
                comps = ['Q', 'K', 'V', 'M']
                corr_matrix = np.eye(4)
                corr_matrix[0, 1] = corr_matrix[1, 0] = data['Q-K']
                corr_matrix[0, 2] = corr_matrix[2, 0] = data['Q-V']
                corr_matrix[0, 3] = corr_matrix[3, 0] = data['Q-M']
                corr_matrix[1, 2] = corr_matrix[2, 1] = data['K-V']

                im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=0, vmax=1)
                ax.set_xticks(range(4))
                ax.set_xticklabels(comps)
                ax.set_yticks(range(4))
                ax.set_yticklabels(comps)
                ax.set_title(f'Layer {layer_idx}')

                for i in range(4):
                    for j in range(4):
                        ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)

            plt.suptitle('Q/K/V/M Usage Correlation by Layer')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'qkvm_correlation.png'), dpi=150)
            plt.close()
            print(f"  Saved: qkvm_correlation.png")

        # 7.4 t-SNE of neurons (if sklearn available) - SKIP for speed
        # t-SNE is very slow with high-dimensional neuron data
        # Uncomment below if you want t-SNE visualization
        print("  Skipping t-SNE (slow). Use analyze_neuron_svd.py for PCA visualization.")

        print(f"\nAll visualizations saved to: {output_dir}")

    # ============================================================
    # RUN ALL
    # ============================================================

    def run_all(self, dataloader, max_batches: int = 100, output_dir: str = './neuron_analysis') -> Dict:
        """모든 분석 실행"""
        all_results = {}

        # 1. 뉴런 활성화 분석
        activation_results, neuron_token_counts, pos_neuron_counts = \
            self.analyze_neuron_activations(dataloader, max_batches)
        all_results.update(activation_results)

        # 2. 뉴런 클러스터링
        clustering_results = self.analyze_neuron_clustering(neuron_token_counts)
        all_results['clustering'] = clustering_results

        # 3. 레이어별 역할 분석
        layer_results = self.analyze_layer_roles(dataloader, min(max_batches, 50))
        all_results.update(layer_results)

        # 4. Knowledge 뉴런 분석
        knowledge_results = self.analyze_knowledge_neurons(dataloader, min(max_batches, 50))
        all_results['knowledge'] = knowledge_results

        # 5. Ablation Study
        ablation_results = self.run_ablation_study(dataloader, min(max_batches, 20))
        all_results['ablation'] = ablation_results

        # 6. Attention-Neuron 관계
        attention_results = self.analyze_attention_neuron_relation(dataloader, min(max_batches, 30))
        all_results['attention_neuron'] = attention_results

        # 7. 시각화
        self.visualize_all(all_results, output_dir)

        return all_results


def convert_to_serializable(obj):
    """Convert numpy/torch types to Python native types"""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, Counter):
        return dict(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description='DAWN Comprehensive Neuron Analysis')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--val_data', type=str,
                        default='/content/drive/MyDrive/data/val/wikitext_5to1_texts.pkl')
    parser.add_argument('--max_batches', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='./neuron_analysis')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        best = checkpoint_path / 'best_model.pt'
        if best.exists():
            checkpoint_path = best
        else:
            pt_files = list(checkpoint_path.glob('*.pt'))
            if pt_files:
                checkpoint_path = max(pt_files, key=lambda p: p.stat().st_mtime)

    print(f"\nLoading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('model_config', checkpoint.get('config', {}))
    version = config.get('model_version', '10.0')
    print(f"Model version: {version}")

    from models.model_v10 import DAWN

    model = DAWN(
        vocab_size=config.get('vocab_size', 30522),
        d_model=config.get('d_model', 320),
        n_layers=config.get('n_layers', 4),
        n_heads=config.get('n_heads', 4),
        rank=config.get('rank', 64),
        max_seq_len=config.get('max_seq_len', 128),
        n_compress=config.get('n_compress', 224),
        n_expand=config.get('n_expand', 56),
        n_knowledge=config.get('n_knowledge', 256),
        knowledge_k=config.get('knowledge_k', 12),
        dropout=config.get('dropout', 0.1),
    )

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Data
    print(f"\nLoading data: {args.val_data}")
    with open(args.val_data, 'rb') as f:
        val_texts = pickle.load(f)

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_len=128):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.max_len,
                padding='max_length',
                return_tensors='pt'
            )
            return {'input_ids': encoding['input_ids'].squeeze(0)}

    dataset = SimpleDataset(val_texts, tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Run analysis
    analyzer = NeuronAnalyzer(model, tokenizer, device)
    all_results = analyzer.run_all(dataloader, max_batches=args.max_batches, output_dir=args.output_dir)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'neuron_analysis.json')

    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\nResults saved: {results_path}")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
