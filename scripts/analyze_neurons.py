#!/usr/bin/env python3
"""
DAWN Neuron Interpretability Analysis
=====================================

Comprehensive neuron analysis for interpretability research.

Analyses:
1. Neuron-Word Mapping - 뉴런별 반응 단어
2. Neuron Clustering (t-SNE/UMAP) - 뉴런 임베딩 시각화
3. Neuron Similarity Heatmap - 뉴런간 유사도
4. Ablation Study - 뉴런 기여도 측정
5. Probing Classifier - 정보 인코딩 분석
6. Layer-wise Role Change - 레이어별 역할 변화
7. Token Trajectory - 토큰별 뉴런 경로
8. Activation Distribution - 활성화 분포
9. Context Dependency - 문맥 의존성

Usage:
    python analyze_neurons.py --checkpoint <path> --val_data <path> --mode all
    python analyze_neurons.py --checkpoint <path> --val_data <path> --mode word_map
    python analyze_neurons.py --checkpoint <path> --val_data <path> --mode ablation
"""

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mixed precision
try:
    from torch.cuda.amp import autocast
    HAS_AMP = True
except ImportError:
    HAS_AMP = False
    def autocast(enabled=True):
        return torch.no_grad()

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ============================================================
# Utilities
# ============================================================

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
    if token_lower in {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}:
        return 'PRON'
    if token_lower in {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about'}:
        return 'ADP'
    if token_lower in {'and', 'or', 'but', 'if', 'when', 'while', 'because', 'although'}:
        return 'CONJ'
    if token_lower.endswith('ing'):
        return 'VERB_ING'
    if token_lower.endswith('ed'):
        return 'VERB_ED'
    if token_lower.endswith('ly'):
        return 'ADV'
    if token_lower.endswith('tion') or token_lower.endswith('ness') or token_lower.endswith('ment'):
        return 'NOUN'
    return 'OTHER'


class VersionDetector:
    """Detect model version and get routing info"""

    def __init__(self, model):
        self.model = get_underlying_model(model)
        self.version = self._detect_version()
        self.n_neurons = self._get_neuron_count()

    def _detect_version(self) -> str:
        # v15: has knowledge_encoder (shared across layers)
        if hasattr(self.model, 'knowledge_encoder'):
            return '15'
        if hasattr(self.model, 'global_routers') and hasattr(self.model.global_routers, 'neuron_router'):
            router = self.model.global_routers.neuron_router
            if hasattr(router, 'usage_ema_feature'):
                return '14'
        if hasattr(self.model, 'shared_neurons'):
            if hasattr(self.model, 'global_ssm') and hasattr(self.model.global_ssm, 'context_proj'):
                return '13'
            return '12'
        return '10'

    def _get_neuron_count(self) -> int:
        if self.version in ['14', '15']:
            return self.model.global_routers.neuron_router.n_feature
        elif hasattr(self.model, 'shared_neurons'):
            return self.model.shared_neurons.compress_neurons.shape[0]
        return getattr(self.model, 'n_compress', 48)

    def get_weights_from_routing(self, routing_info: Dict) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract weights and indices from routing_info"""
        attn = routing_info.get('attention', routing_info)

        if 'feature_pref' in attn:
            # v14 FRTK: token-level preferences [B, S, N] - 우선 사용!
            weights = attn['feature_pref']
            indices = None
        elif 'feature_weights' in attn:
            # v14 FRTK: batch-level fallback
            weights = attn['feature_weights']
            indices = None
        elif 'compress_pref' in attn:
            # v13.2: token-level preferences [B, S, N] - 우선 사용!
            weights = attn['compress_pref']
            indices = None
        elif 'compress_weights_dense' in attn:
            # v12.7/v13: dense batch-level
            weights = attn['compress_weights_dense']
            indices = attn.get('compress_topk_idx')
        elif 'compress_weights' in attn:
            # v12/v13 fallback
            weights = attn['compress_weights']
            indices = attn.get('compress_topk_idx')
        elif 'Q' in attn and isinstance(attn['Q'], dict):
            # v10
            weights = attn['Q']['weights']
            indices = attn['Q'].get('indices')
        else:
            return None, None

        return weights, indices


# ============================================================
# 1. Neuron-Word Mapping
# ============================================================

class NeuronWordMapper:
    """Map neurons to their most activated words"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.detector = VersionDetector(model)
        self.n_neurons = self.detector.n_neurons

    @torch.no_grad()
    def build_mapping(self, dataloader, max_batches: int = 100, top_k: int = 20) -> Dict:
        """Build neuron -> word mapping"""
        print(f"\n{'='*60}")
        print(f"NEURON-WORD MAPPING (v{self.detector.version})")
        print(f"{'='*60}")

        self.model.eval()

        # neuron_id -> {word: activation_sum}
        neuron_words = [Counter() for _ in range(self.n_neurons)]
        neuron_total = np.zeros(self.n_neurons)

        # Cache token_id -> word mapping
        token_cache = {}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Building word map", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Pre-decode tokens with cache
            input_ids_cpu = input_ids.cpu().numpy()
            for b in range(B):
                for s in range(S):
                    tid = input_ids_cpu[b, s]
                    if tid not in token_cache:
                        token_cache[tid] = self.tokenizer.decode([tid]).strip()

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # Use layer 0 for word mapping
            weights, indices = self.detector.get_weights_from_routing(routing_infos[0])
            if weights is None:
                continue

            # Handle batch-level vs token-level
            is_batch_level = (len(weights.shape) == 2)

            if indices is None:
                k = min(8, weights.shape[-1])
                _, indices = torch.topk(weights, k, dim=-1)

            # Move to CPU once
            indices_cpu = indices.cpu().numpy()
            weights_cpu = weights.cpu().numpy()

            for b in range(B):
                for s in range(S):
                    word = token_cache[input_ids_cpu[b, s]]

                    # Skip special tokens
                    if word.lower() in ['[pad]', '[cls]', '[sep]', '[unk]', '<pad>', '<s>', '</s>', '<unk>']:
                        continue

                    if is_batch_level:
                        top_neurons = indices_cpu[b]
                        top_weights = weights_cpu[b]
                    else:
                        top_neurons = indices_cpu[b, s]
                        top_weights = weights_cpu[b, s]

                    for ni, w in zip(top_neurons, top_weights[:len(top_neurons)]):
                        neuron_words[ni][word] += float(w)
                        neuron_total[ni] += float(w)

        # Build result
        results = {'neurons': {}, 'summary': {}}

        pos_distribution = Counter()

        for n in range(self.n_neurons):
            top_words = neuron_words[n].most_common(top_k)

            # Get POS of top words
            pos_counts = Counter()
            for word, _ in top_words[:10]:
                pos = simple_pos_tag(word)
                pos_counts[pos] += 1

            primary_pos = pos_counts.most_common(1)[0][0] if pos_counts else 'UNK'
            pos_distribution[primary_pos] += 1

            results['neurons'][n] = {
                'top_words': top_words,
                'total_activation': float(neuron_total[n]),
                'primary_pos': primary_pos,
            }

        results['summary'] = {
            'pos_distribution': dict(pos_distribution),
            'total_neurons': self.n_neurons,
        }

        # Print summary
        print(f"\n--- POS Distribution ---")
        for pos, count in pos_distribution.most_common():
            print(f"  {pos}: {count} neurons ({100*count/self.n_neurons:.1f}%)")

        # Print example neurons
        print(f"\n--- Example Neurons ---")
        for n in [0, self.n_neurons//2, self.n_neurons-1]:
            words = [w for w, _ in results['neurons'][n]['top_words'][:5]]
            print(f"  Neuron {n}: {words}")

        return results

    def visualize(self, results: Dict, output_dir: str):
        """Create word clouds for top neurons"""
        if not HAS_MATPLOTLIB:
            return

        try:
            from wordcloud import WordCloud
            has_wordcloud = True
        except ImportError:
            has_wordcloud = False

        os.makedirs(output_dir, exist_ok=True)

        # Sort neurons by total activation
        sorted_neurons = sorted(
            results['neurons'].items(),
            key=lambda x: x[1]['total_activation'],
            reverse=True
        )

        # Plot top 9 neurons
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()

        for idx, (neuron_id, data) in enumerate(sorted_neurons[:9]):
            ax = axes[idx]

            if has_wordcloud and data['top_words']:
                word_freq = dict(data['top_words'])
                wc = WordCloud(width=400, height=300, background_color='white')
                wc.generate_from_frequencies(word_freq)
                ax.imshow(wc, interpolation='bilinear')
            else:
                # Simple bar chart
                words = [w for w, _ in data['top_words'][:10]]
                freqs = [f for _, f in data['top_words'][:10]]
                ax.barh(range(len(words)), freqs)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words)

            ax.set_title(f"Neuron {neuron_id} ({data['primary_pos']})")
            ax.axis('off' if has_wordcloud else 'on')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'neuron_words.png'), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/neuron_words.png")


# ============================================================
# 2. Neuron Clustering (t-SNE/UMAP)
# ============================================================

class NeuronClusterer:
    """Cluster neurons based on their embeddings"""

    def __init__(self, model, device='cuda'):
        self.model = get_underlying_model(model)
        self.device = device
        self.detector = VersionDetector(model)

    def extract_neuron_embeddings(self) -> np.ndarray:
        """Extract neuron weight vectors"""
        shared = self.model.shared_neurons

        if self.detector.version in ['14', '15']:
            neurons = shared.feature_neurons.data.cpu().numpy()
        else:
            neurons = shared.compress_neurons.data.cpu().numpy()

        # Flatten: [N, D, R] -> [N, D*R]
        N = neurons.shape[0]
        return neurons.reshape(N, -1)

    def cluster(self, n_clusters: int = 8, method: str = 'tsne') -> Dict:
        """Perform clustering and dimensionality reduction"""
        print(f"\n{'='*60}")
        print(f"NEURON CLUSTERING ({method.upper()})")
        print(f"{'='*60}")

        if not HAS_SKLEARN:
            print("sklearn not available")
            return {}

        embeddings = self.extract_neuron_embeddings()
        print(f"Embeddings shape: {embeddings.shape}")

        # Normalize
        scaler = StandardScaler()
        embeddings_norm = scaler.fit_transform(embeddings)

        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1), random_state=42)
            coords_2d = reducer.fit_transform(embeddings_norm)
        elif method == 'umap' and HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(embeddings_norm)
        else:
            # PCA fallback
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            coords_2d = reducer.fit_transform(embeddings_norm)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_norm)

        results = {
            'coords_2d': coords_2d.tolist(),
            'cluster_labels': cluster_labels.tolist(),
            'n_clusters': n_clusters,
            'method': method,
        }

        # Cluster sizes
        cluster_sizes = Counter(cluster_labels)
        print(f"\n--- Cluster Sizes ---")
        for c, size in sorted(cluster_sizes.items()):
            print(f"  Cluster {c}: {size} neurons")

        return results

    def visualize(self, results: Dict, output_path: str):
        """Visualize clustering"""
        if not HAS_MATPLOTLIB or not results:
            return

        coords = np.array(results['coords_2d'])
        labels = np.array(results['cluster_labels'])

        fig, ax = plt.subplots(figsize=(10, 10))

        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                            c=labels, cmap='tab10', s=100, alpha=0.7)

        # Label each point
        for i, (x, y) in enumerate(coords):
            ax.annotate(str(i), (x, y), fontsize=8, alpha=0.7)

        plt.colorbar(scatter, label='Cluster')
        ax.set_title(f"Neuron Clustering ({results['method'].upper()})")
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")


# ============================================================
# 3. Neuron Similarity Heatmap
# ============================================================

class NeuronSimilarityAnalyzer:
    """Analyze pairwise neuron similarity"""

    def __init__(self, model, device='cuda'):
        self.model = get_underlying_model(model)
        self.device = device
        self.detector = VersionDetector(model)

    def compute_similarity(self) -> Dict:
        """Compute cosine similarity matrix (GPU accelerated)"""
        print(f"\n{'='*60}")
        print(f"NEURON SIMILARITY ANALYSIS (GPU)")
        print(f"{'='*60}")

        shared = self.model.shared_neurons

        # Get neurons on GPU
        if self.detector.version in ['14', '15']:
            neurons = shared.feature_neurons.data
        else:
            neurons = shared.compress_neurons.data

        N = neurons.shape[0]
        neurons_flat = neurons.reshape(N, -1)

        # GPU-based normalization and similarity
        with torch.no_grad():
            neurons_norm = F.normalize(neurons_flat, p=2, dim=1)
            sim_matrix_gpu = torch.mm(neurons_norm, neurons_norm.T)

            # Stats on GPU
            mask = ~torch.eye(N, dtype=torch.bool, device=sim_matrix_gpu.device)
            off_diag = sim_matrix_gpu[mask]

            stats = {
                'mean': float(off_diag.mean().cpu()),
                'std': float(off_diag.std().cpu()),
                'max': float(off_diag.max().cpu()),
                'min': float(off_diag.min().cpu()),
            }

            # Find most similar pairs on GPU
            sim_for_topk = sim_matrix_gpu.clone()
            sim_for_topk.fill_diagonal_(-1)
            flat_sim = sim_for_topk.flatten()
            topk_vals, topk_idx = torch.topk(flat_sim, 10)

            sim_matrix = sim_matrix_gpu.cpu().numpy()

        results = {
            'similarity_matrix': sim_matrix.tolist(),
            'stats': stats,
            'n_neurons': N,
        }

        print(f"Similarity stats:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")

        print(f"\n--- Most Similar Pairs ---")
        for val, idx in zip(topk_vals, topk_idx):
            i, j = idx.item() // N, idx.item() % N
            print(f"  Neuron {i} - {j}: {val.item():.4f}")

        return results

    def visualize(self, results: Dict, output_path: str):
        """Create similarity heatmap"""
        if not HAS_MATPLOTLIB or not results:
            return

        sim_matrix = np.array(results['similarity_matrix'])

        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, label='Cosine Similarity')

        ax.set_title('Neuron Similarity Matrix')
        ax.set_xlabel('Neuron ID')
        ax.set_ylabel('Neuron ID')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")


# ============================================================
# 4. Ablation Study
# ============================================================

class NeuronAblation:
    """Measure neuron contribution via ablation (GPU optimized)"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.detector = VersionDetector(model)
        self.use_amp = HAS_AMP and device != 'cpu'

    @torch.no_grad()
    def run_ablation(self, dataloader, max_batches: int = 20,
                     neurons_to_test: List[int] = None,
                     parallel_batch: int = 4) -> Dict:
        """Run ablation study with GPU optimization"""
        print(f"\n{'='*60}")
        print(f"ABLATION STUDY (GPU, AMP={self.use_amp})")
        print(f"{'='*60}")

        self.model.eval()

        if neurons_to_test is None:
            neurons_to_test = list(range(min(10, self.detector.n_neurons)))

        # Cache batches to avoid reloading
        print("Caching data batches...")
        cached_batches = []
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            cached_batches.append(batch["input_ids"].to(self.device))

        # Get baseline perplexity
        baseline_loss = self._compute_loss_cached(cached_batches)
        baseline_ppl = math.exp(baseline_loss)
        print(f"Baseline perplexity: {baseline_ppl:.2f}")

        results = {
            'baseline_ppl': baseline_ppl,
            'ablations': {},
        }

        # Ablate each neuron
        shared = self.model.shared_neurons
        if self.detector.version in ['14', '15']:
            neurons = shared.feature_neurons
        else:
            neurons = shared.compress_neurons

        original_weights = neurons.data.clone()

        # Process neurons in batches for progress display
        for n in tqdm(neurons_to_test, desc="Ablating neurons"):
            # Zero out neuron n
            neurons.data[n] = 0

            ablated_loss = self._compute_loss_cached(cached_batches)
            ablated_ppl = math.exp(ablated_loss)

            delta_ppl = ablated_ppl - baseline_ppl
            results['ablations'][n] = {
                'ppl': ablated_ppl,
                'delta_ppl': delta_ppl,
                'importance': delta_ppl / baseline_ppl,
            }

            # Restore
            neurons.data = original_weights.clone()

        # Sort by importance
        sorted_neurons = sorted(results['ablations'].items(),
                               key=lambda x: x[1]['importance'], reverse=True)

        print(f"\n--- Most Important Neurons ---")
        for n, data in sorted_neurons[:5]:
            print(f"  Neuron {n}: +{data['delta_ppl']:.2f} PPL ({data['importance']*100:.1f}%)")

        print(f"\n--- Least Important Neurons ---")
        for n, data in sorted_neurons[-5:]:
            print(f"  Neuron {n}: +{data['delta_ppl']:.2f} PPL ({data['importance']*100:.1f}%)")

        return results

    def _compute_loss_cached(self, cached_batches: List[torch.Tensor]) -> float:
        """Compute average loss from cached batches with AMP"""
        total_loss = 0
        total_tokens = 0

        for input_ids in cached_batches:
            with autocast(enabled=self.use_amp):
                outputs = self.model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

            # Shift for LM loss
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += shift_labels.numel()

        return total_loss / total_tokens if total_tokens > 0 else 0

    def _compute_loss(self, dataloader, max_batches: int) -> float:
        """Compute average loss (legacy, for compatibility)"""
        total_loss = 0
        total_tokens = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)

            outputs = self.model(input_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Shift for LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += shift_labels.numel()

        return total_loss / total_tokens if total_tokens > 0 else 0

    def visualize(self, results: Dict, output_path: str):
        """Visualize ablation results"""
        if not HAS_MATPLOTLIB or not results.get('ablations'):
            return

        neurons = list(results['ablations'].keys())
        importances = [results['ablations'][n]['importance'] for n in neurons]

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['red' if imp > 0.05 else 'orange' if imp > 0.01 else 'green'
                  for imp in importances]
        ax.bar(range(len(neurons)), importances, color=colors)
        ax.set_xticks(range(len(neurons)))
        ax.set_xticklabels([f'N{n}' for n in neurons], rotation=45)
        ax.set_ylabel('Importance (relative PPL increase)')
        ax.set_title('Neuron Ablation Study')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")


# ============================================================
# 5. Probing Classifier
# ============================================================

class NeuronProbe:
    """Probe neuron activations for linguistic features"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.detector = VersionDetector(model)

    @torch.no_grad()
    def collect_activations(self, dataloader, max_batches: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Collect neuron activations with POS labels"""
        print(f"\n--- Collecting activations ---")

        self.model.eval()

        all_activations = []
        all_labels = []

        # Cache token_id -> (word, pos) mapping
        token_cache = {}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Pre-decode tokens with cache
            input_ids_cpu = input_ids.cpu().numpy()
            for b in range(B):
                for s in range(S):
                    tid = input_ids_cpu[b, s]
                    if tid not in token_cache:
                        word = self.tokenizer.decode([tid]).strip()
                        token_cache[tid] = (word, simple_pos_tag(word))

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            weights, indices = self.detector.get_weights_from_routing(routing_infos[0])
            if weights is None:
                continue

            is_batch_level = (len(weights.shape) == 2)
            weights_cpu = weights.cpu().numpy()

            for b in range(B):
                for s in range(S):
                    _, pos = token_cache[input_ids_cpu[b, s]]

                    if pos == 'OTHER':
                        continue

                    if is_batch_level:
                        act = weights_cpu[b]
                    else:
                        act = weights_cpu[b, s]

                    all_activations.append(act)
                    all_labels.append(pos)

        return np.array(all_activations), np.array(all_labels)

    def train_probe(self, dataloader, max_batches: int = 50) -> Dict:
        """Train probing classifier"""
        print(f"\n{'='*60}")
        print(f"PROBING CLASSIFIER")
        print(f"{'='*60}")

        if not HAS_SKLEARN:
            print("sklearn not available")
            return {}

        X, y = self.collect_activations(dataloader, max_batches)

        if len(X) == 0:
            print("No data collected")
            return {}

        print(f"Collected {len(X)} samples")

        # Filter rare classes
        label_counts = Counter(y)
        valid_labels = {l for l, c in label_counts.items() if c >= 10}
        mask = np.array([l in valid_labels for l in y])
        X, y = X[mask], y[mask]

        print(f"After filtering: {len(X)} samples, {len(set(y))} classes")

        # Train/test split
        n_train = int(0.8 * len(X))
        indices = np.random.permutation(len(X))
        train_idx, test_idx = indices[:n_train], indices[n_train:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial', n_jobs=-1)
        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        print(f"\n--- Results ---")
        print(f"Train accuracy: {train_acc*100:.1f}%")
        print(f"Test accuracy:  {test_acc*100:.1f}%")

        # Per-class accuracy
        y_pred = clf.predict(X_test)
        class_acc = {}
        for label in set(y_test):
            mask = y_test == label
            if mask.sum() > 0:
                class_acc[label] = float((y_pred[mask] == y_test[mask]).mean())

        print(f"\n--- Per-class Accuracy ---")
        for label, acc in sorted(class_acc.items(), key=lambda x: -x[1]):
            print(f"  {label}: {acc*100:.1f}%")

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'class_accuracy': class_acc,
            'n_samples': len(X),
            'n_classes': len(set(y)),
        }


# ============================================================
# 6. Layer-wise Role Change
# ============================================================

class LayerRoleAnalyzer:
    """Analyze how neuron roles change across layers"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.detector = VersionDetector(model)
        self.n_layers = self.model.n_layers

    @torch.no_grad()
    def analyze(self, dataloader, max_batches: int = 50) -> Dict:
        """Analyze layer-wise neuron usage"""
        print(f"\n{'='*60}")
        print(f"LAYER-WISE ROLE ANALYSIS")
        print(f"{'='*60}")

        self.model.eval()

        n_neurons = self.detector.n_neurons

        # layer -> neuron -> {word: count}
        layer_neuron_words = {
            l: [Counter() for _ in range(n_neurons)]
            for l in range(self.n_layers)
        }

        # Cache token_id -> word mapping to avoid repeated decode calls
        token_cache = {}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing layers", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Pre-decode all tokens in batch (cache lookup)
            input_ids_cpu = input_ids.cpu().numpy()
            words_batch = []
            for b in range(B):
                words_seq = []
                for s in range(S):
                    tid = input_ids_cpu[b, s]
                    if tid not in token_cache:
                        token_cache[tid] = self.tokenizer.decode([tid]).strip()
                    words_seq.append(token_cache[tid])
                words_batch.append(words_seq)

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                weights, indices = self.detector.get_weights_from_routing(routing_info)
                if weights is None:
                    continue

                is_batch_level = (len(weights.shape) == 2)

                if indices is None:
                    k = min(8, weights.shape[-1])
                    _, indices = torch.topk(weights, k, dim=-1)

                # Move to CPU once per layer
                indices_cpu = indices.cpu().numpy()

                for b in range(B):
                    for s in range(S):
                        word = words_batch[b][s]

                        if is_batch_level:
                            top_neurons = indices_cpu[b]
                        else:
                            top_neurons = indices_cpu[b, s]

                        for ni in top_neurons[:4]:  # Top 4 neurons
                            layer_neuron_words[layer_idx][ni][word] += 1

        # Analyze role changes
        results = {'layers': {}, 'role_changes': []}

        for layer_idx in range(self.n_layers):
            layer_data = {}
            for n in range(n_neurons):
                top_words = layer_neuron_words[layer_idx][n].most_common(5)
                if top_words:
                    layer_data[n] = [w for w, _ in top_words]
            results['layers'][layer_idx] = layer_data

        # Find neurons with changing roles
        print(f"\n--- Role Changes Across Layers ---")
        for n in range(min(5, n_neurons)):
            print(f"\nNeuron {n}:")
            for l in range(self.n_layers):
                words = results['layers'][l].get(n, [])[:3]
                print(f"  L{l}: {words}")

        return results


# ============================================================
# 7. Token Trajectory
# ============================================================

class TokenTrajectory:
    """Track which neurons process each token"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.detector = VersionDetector(model)

    @torch.no_grad()
    def trace(self, text: str) -> Dict:
        """Trace token through layers"""
        print(f"\n{'='*60}")
        print(f"TOKEN TRAJECTORY")
        print(f"{'='*60}")
        print(f"Text: {text}")

        self.model.eval()

        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor([tokens], device=self.device)

        token_strs = [self.tokenizer.decode([t]).strip() for t in tokens]

        _, routing_infos = self.model(input_ids, return_routing_info=True)

        trajectories = {t: [] for t in range(len(tokens))}

        for layer_idx, routing_info in enumerate(routing_infos):
            weights, indices = self.detector.get_weights_from_routing(routing_info)
            if weights is None:
                continue

            is_batch_level = (len(weights.shape) == 2)

            if indices is None:
                k = min(4, weights.shape[-1])
                _, indices = torch.topk(weights, k, dim=-1)

            for t in range(len(tokens)):
                if is_batch_level:
                    top_neurons = indices[0].cpu().numpy()[:4]
                else:
                    top_neurons = indices[0, t].cpu().numpy()[:4]

                trajectories[t].append(top_neurons.tolist())

        results = {
            'tokens': token_strs,
            'trajectories': trajectories,
            'n_layers': len(routing_infos),
        }

        # Print
        print(f"\n--- Trajectories ---")
        for t, token in enumerate(token_strs):
            traj = trajectories[t]
            traj_str = ' → '.join([str(neurons[:2]) for neurons in traj])
            print(f"  '{token}': {traj_str}")

        return results


# ============================================================
# 8. Activation Distribution
# ============================================================

class ActivationDistribution:
    """Analyze neuron activation distributions"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.detector = VersionDetector(model)

    @torch.no_grad()
    def analyze(self, dataloader, max_batches: int = 50) -> Dict:
        """Analyze activation distributions"""
        print(f"\n{'='*60}")
        print(f"ACTIVATION DISTRIBUTION")
        print(f"{'='*60}")

        self.model.eval()

        n_neurons = self.detector.n_neurons
        all_activations = [[] for _ in range(n_neurons)]

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting activations", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            weights, _ = self.detector.get_weights_from_routing(routing_infos[0])
            if weights is None:
                continue

            # Flatten batch dimension
            if len(weights.shape) == 3:
                weights = weights.reshape(-1, weights.shape[-1])

            weights_np = weights.cpu().numpy()

            for n in range(min(n_neurons, weights_np.shape[-1])):
                all_activations[n].extend(weights_np[:, n].tolist())

        # Compute stats
        results = {'neurons': {}, 'summary': {}}

        sparsity_scores = []
        for n in range(n_neurons):
            acts = np.array(all_activations[n])
            if len(acts) == 0:
                continue

            # Gini coefficient (sparsity measure)
            sorted_acts = np.sort(acts)
            index = np.arange(1, len(sorted_acts) + 1)
            gini = ((2 * index - len(sorted_acts) - 1) * sorted_acts).sum() / (len(sorted_acts) * sorted_acts.sum() + 1e-10)

            results['neurons'][n] = {
                'mean': float(np.mean(acts)),
                'std': float(np.std(acts)),
                'max': float(np.max(acts)),
                'gini': float(gini),
                'sparsity': float((acts < 0.01).mean()),
            }
            sparsity_scores.append(gini)

        results['summary'] = {
            'avg_gini': float(np.mean(sparsity_scores)),
            'avg_sparsity': float(np.mean([r['sparsity'] for r in results['neurons'].values()])),
        }

        print(f"\n--- Summary ---")
        print(f"Average Gini (sparsity): {results['summary']['avg_gini']:.4f}")
        print(f"Average zero-rate: {results['summary']['avg_sparsity']*100:.1f}%")

        # Most/least sparse neurons
        sorted_neurons = sorted(results['neurons'].items(), key=lambda x: x[1]['gini'], reverse=True)

        print(f"\n--- Most Sparse (selective) ---")
        for n, data in sorted_neurons[:5]:
            print(f"  Neuron {n}: Gini={data['gini']:.4f}")

        print(f"\n--- Least Sparse (always active) ---")
        for n, data in sorted_neurons[-5:]:
            print(f"  Neuron {n}: Gini={data['gini']:.4f}")

        return results

    def visualize(self, results: Dict, output_path: str):
        """Visualize activation distributions"""
        if not HAS_MATPLOTLIB or not results.get('neurons'):
            return

        neurons = list(results['neurons'].keys())
        ginis = [results['neurons'][n]['gini'] for n in neurons]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Gini distribution
        ax = axes[0]
        ax.hist(ginis, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(ginis), color='red', linestyle='--', label=f'Mean: {np.mean(ginis):.3f}')
        ax.set_xlabel('Gini Coefficient')
        ax.set_ylabel('Count')
        ax.set_title('Neuron Sparsity Distribution')
        ax.legend()

        # Per-neuron bar
        ax = axes[1]
        ax.bar(range(len(neurons)), ginis, alpha=0.7)
        ax.set_xlabel('Neuron ID')
        ax.set_ylabel('Gini Coefficient')
        ax.set_title('Sparsity per Neuron')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")


# ============================================================
# 9. Context Dependency
# ============================================================

class ContextDependency:
    """Analyze context-dependent routing"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.detector = VersionDetector(model)

    @torch.no_grad()
    def analyze_word(self, word: str, contexts: List[str]) -> Dict:
        """Analyze how same word routes differently in contexts"""
        print(f"\n{'='*60}")
        print(f"CONTEXT DEPENDENCY: '{word}'")
        print(f"{'='*60}")

        self.model.eval()

        results = {'word': word, 'contexts': []}

        for ctx in contexts:
            tokens = self.tokenizer.encode(ctx, add_special_tokens=True)
            input_ids = torch.tensor([tokens], device=self.device)

            # Find target word position
            token_strs = [self.tokenizer.decode([t]).strip().lower() for t in tokens]
            try:
                word_pos = token_strs.index(word.lower())
            except ValueError:
                print(f"  '{word}' not found in: {ctx}")
                continue

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            weights, indices = self.detector.get_weights_from_routing(routing_infos[0])
            if weights is None:
                continue

            is_batch_level = (len(weights.shape) == 2)

            if indices is None:
                k = min(8, weights.shape[-1])
                _, indices = torch.topk(weights, k, dim=-1)

            if is_batch_level:
                top_neurons = indices[0].cpu().numpy()[:8]
                top_weights = weights[0].cpu().numpy()
            else:
                top_neurons = indices[0, word_pos].cpu().numpy()[:8]
                top_weights = weights[0, word_pos].cpu().numpy()

            results['contexts'].append({
                'text': ctx,
                'top_neurons': top_neurons.tolist(),
                'weights': [float(top_weights[i]) for i in range(min(8, len(top_weights)))],
            })

            print(f"\n  Context: {ctx}")
            print(f"  Top neurons: {top_neurons[:5]}")

        # Compare across contexts
        if len(results['contexts']) >= 2:
            neurons_sets = [set(c['top_neurons'][:4]) for c in results['contexts']]
            intersection = set.intersection(*neurons_sets) if neurons_sets else set()
            union = set.union(*neurons_sets) if neurons_sets else set()

            jaccard = len(intersection) / len(union) if union else 0

            results['comparison'] = {
                'shared_neurons': list(intersection),
                'jaccard_similarity': jaccard,
            }

            print(f"\n--- Comparison ---")
            print(f"  Shared neurons: {list(intersection)}")
            print(f"  Jaccard similarity: {jaccard:.4f}")

        return results


# ============================================================
# 10. Semantic Path Similarity
# ============================================================

class SemanticPathAnalyzer:
    """의미 유사도 vs 경로 유사도 상관관계 분석"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.detector = VersionDetector(model)

    @torch.no_grad()
    def get_path(self, text: str) -> Dict:
        """텍스트의 뉴런 경로 추출"""
        self.model.eval()

        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor([tokens], device=self.device)

        _, routing_infos = self.model(input_ids, return_routing_info=True)

        path = {
            'feature': [],
            'relational_Q': [],
            'relational_K': [],
            'value': [],
            'knowledge_coarse': [],
            'knowledge_fine': [],
        }

        for layer_idx, routing_info in enumerate(routing_infos):
            attn = routing_info.get('attention', {})
            mem = routing_info.get('memory', {})

            # Feature weights
            if 'feature_pref' in attn:
                weights = attn['feature_pref']
                if weights.dim() == 3:
                    weights = weights.mean(dim=1)
                path['feature'].append(weights[0].cpu().numpy())
            elif 'feature_weights' in attn:
                path['feature'].append(attn['feature_weights'][0].cpu().numpy())

            # Relational Q/K
            if 'relational_weights_Q' in attn:
                path['relational_Q'].append(attn['relational_weights_Q'][0].cpu().numpy())
            if 'relational_weights_K' in attn:
                path['relational_K'].append(attn['relational_weights_K'][0].cpu().numpy())

            # Value
            if 'value_weights' in attn:
                path['value'].append(attn['value_weights'][0].cpu().numpy())

            # Knowledge (v15 2-stage)
            if 'coarse_indices' in mem:
                path['knowledge_coarse'].append(mem['coarse_indices'][0].cpu().numpy())
            if 'fine_indices' in mem:
                path['knowledge_fine'].append(mem['fine_indices'][0].cpu().numpy())

        return path

    def path_similarity(self, path1: Dict, path2: Dict) -> Dict:
        """두 경로 간 유사도 계산"""
        similarities = {}

        for key in ['feature', 'relational_Q', 'relational_K', 'value']:
            if path1.get(key) and path2.get(key):
                sims = []
                for w1, w2 in zip(path1[key], path2[key]):
                    w1_flat = w1.flatten() if len(w1.shape) > 1 else w1
                    w2_flat = w2.flatten() if len(w2.shape) > 1 else w2
                    cos_sim = np.dot(w1_flat, w2_flat) / (np.linalg.norm(w1_flat) * np.linalg.norm(w2_flat) + 1e-8)
                    sims.append(cos_sim)
                similarities[key] = float(np.mean(sims))

        # Knowledge: Jaccard similarity (set overlap)
        for key in ['knowledge_coarse', 'knowledge_fine']:
            if path1.get(key) and path2.get(key):
                jaccards = []
                for idx1, idx2 in zip(path1[key], path2[key]):
                    set1 = set(idx1.flatten().tolist())
                    set2 = set(idx2.flatten().tolist())
                    jaccard = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
                    jaccards.append(jaccard)
                similarities[key] = float(np.mean(jaccards))

        if similarities:
            similarities['overall'] = float(np.mean(list(similarities.values())))

        return similarities

    def analyze_pairs(self, text_pairs: List[Tuple[str, str, str]]) -> Dict:
        """텍스트 쌍들의 경로 유사도 분석"""
        print(f"\n{'='*60}")
        print(f"SEMANTIC PATH SIMILARITY ANALYSIS")
        print(f"{'='*60}")

        results = {'pairs': [], 'summary': {}}

        for text1, text2, relation in text_pairs:
            path1 = self.get_path(text1)
            path2 = self.get_path(text2)

            sim = self.path_similarity(path1, path2)

            results['pairs'].append({
                'text1': text1,
                'text2': text2,
                'relation': relation,
                'similarity': sim,
            })

            print(f"\n[{relation}]")
            print(f"  '{text1}'")
            print(f"  '{text2}'")
            print(f"  → Overall: {sim.get('overall', 0):.4f}")
            for k, v in sim.items():
                if k != 'overall':
                    print(f"     {k}: {v:.4f}")

        # Summary by relation type
        by_relation = defaultdict(list)
        for pair in results['pairs']:
            by_relation[pair['relation']].append(pair['similarity'].get('overall', 0))

        print(f"\n--- Summary by Relation ---")
        for relation, sims in by_relation.items():
            avg = np.mean(sims)
            results['summary'][relation] = float(avg)
            print(f"  {relation}: {avg:.4f} (n={len(sims)})")

        return results

    def run_default_analysis(self) -> Dict:
        """기본 예시들로 분석 실행"""
        similar_pairs = [
            ("The cat sleeps on the bed.", "The dog rests on the couch.", "similar"),
            ("She bought a new car.", "He purchased a new vehicle.", "similar"),
            ("The child is happy.", "The kid is joyful.", "similar"),
        ]

        different_pairs = [
            ("The cat sleeps on the bed.", "Stock prices rose sharply.", "different"),
            ("She bought a new car.", "The algorithm converges quickly.", "different"),
            ("The child is happy.", "Quantum mechanics is complex.", "different"),
        ]

        polysemy_pairs = [
            ("I deposited money at the bank.", "I sat on the river bank.", "polysemy_bank"),
            ("The bat flew at night.", "He swung the bat hard.", "polysemy_bat"),
        ]

        all_pairs = similar_pairs + different_pairs + polysemy_pairs
        return self.analyze_pairs(all_pairs)


# ============================================================
# 11. Semantic Category Neuron Mapping
# ============================================================

class SemanticCategoryAnalyzer:
    """의미 카테고리별 뉴런 활성화 분석"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.detector = VersionDetector(model)

        self.categories = {
            'animals': ['cat', 'dog', 'bird', 'fish', 'horse', 'elephant', 'tiger', 'lion', 'bear', 'rabbit'],
            'food': ['pizza', 'pasta', 'bread', 'rice', 'apple', 'banana', 'chicken', 'beef', 'soup', 'salad'],
            'emotions': ['happy', 'sad', 'angry', 'afraid', 'excited', 'worried', 'calm', 'anxious', 'proud', 'lonely'],
            'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'pink', 'brown'],
            'numbers': ['one', 'two', 'three', 'four', 'five', 'ten', 'hundred', 'thousand', 'first', 'second'],
            'actions': ['run', 'walk', 'eat', 'sleep', 'read', 'write', 'speak', 'listen', 'think', 'work'],
            'places': ['house', 'school', 'office', 'hospital', 'park', 'beach', 'mountain', 'city', 'country', 'room'],
            'time': ['today', 'tomorrow', 'yesterday', 'morning', 'evening', 'night', 'week', 'month', 'year', 'hour'],
            'tech': ['computer', 'phone', 'internet', 'software', 'algorithm', 'data', 'network', 'server', 'code', 'system'],
            'nature': ['tree', 'flower', 'river', 'ocean', 'sky', 'cloud', 'rain', 'sun', 'moon', 'star'],
        }

    @torch.no_grad()
    def get_word_activations(self, word: str, n_contexts: int = 5) -> np.ndarray:
        """단어의 평균 뉴런 활성화 추출"""
        self.model.eval()

        contexts = [
            f"The {word} is here.",
            f"I see a {word}.",
            f"This is about {word}.",
            f"Look at the {word}.",
            f"There is a {word} nearby.",
        ][:n_contexts]

        all_activations = []

        for ctx in contexts:
            tokens = self.tokenizer.encode(ctx, add_special_tokens=True)
            input_ids = torch.tensor([tokens], device=self.device)

            token_strs = [self.tokenizer.decode([t]).strip().lower() for t in tokens]
            try:
                word_pos = token_strs.index(word.lower())
            except ValueError:
                continue

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            weights, _ = self.detector.get_weights_from_routing(routing_infos[0])
            if weights is None:
                continue

            if weights.dim() == 3:
                act = weights[0, word_pos].cpu().numpy()
            else:
                act = weights[0].cpu().numpy()

            all_activations.append(act)

        if all_activations:
            return np.mean(all_activations, axis=0)
        return None

    def analyze_categories(self) -> Dict:
        """카테고리별 뉴런 활성화 분석"""
        print(f"\n{'='*60}")
        print(f"SEMANTIC CATEGORY NEURON ANALYSIS")
        print(f"{'='*60}")

        results = {
            'category_activations': {},
            'category_top_neurons': {},
            'neuron_categories': {},
            'category_separation': {},
        }

        category_acts = {}

        for category, words in tqdm(self.categories.items(), desc="Analyzing categories"):
            word_acts = []
            for word in words:
                act = self.get_word_activations(word)
                if act is not None:
                    word_acts.append(act)

            if word_acts:
                category_acts[category] = np.mean(word_acts, axis=0)

                top_k = 5
                top_neurons = np.argsort(category_acts[category])[-top_k:][::-1]
                results['category_top_neurons'][category] = top_neurons.tolist()

        results['category_activations'] = {k: v.tolist() for k, v in category_acts.items()}

        # 뉴런별 주요 카테고리 매핑
        if category_acts:
            n_neurons = len(list(category_acts.values())[0])
            for n in range(n_neurons):
                neuron_scores = {cat: acts[n] for cat, acts in category_acts.items()}
                top_cat = max(neuron_scores, key=neuron_scores.get)
                results['neuron_categories'][n] = {
                    'primary': top_cat,
                    'score': float(neuron_scores[top_cat]),
                }

        # 카테고리 간 분리도
        categories = list(category_acts.keys())
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                v1, v2 = category_acts[cat1], category_acts[cat2]
                cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                results['category_separation'][f"{cat1}_vs_{cat2}"] = float(1 - cos_sim)

        # Print summary
        print(f"\n--- Top Neurons by Category ---")
        for cat, neurons in results['category_top_neurons'].items():
            print(f"  {cat}: {neurons}")

        print(f"\n--- Category Separation (higher = more separated) ---")
        separations = list(results['category_separation'].items())
        separations.sort(key=lambda x: x[1], reverse=True)
        for pair, sep in separations[:5]:
            print(f"  {pair}: {sep:.4f}")

        return results

    def visualize(self, results: Dict, output_path: str):
        """카테고리-뉴런 히트맵 시각화"""
        if not HAS_MATPLOTLIB or not results.get('category_activations'):
            return

        categories = list(results['category_activations'].keys())
        activations = np.array([results['category_activations'][c] for c in categories])

        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(activations, aspect='auto', cmap='YlOrRd')

        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)
        ax.set_xlabel('Neuron ID')
        ax.set_ylabel('Semantic Category')
        ax.set_title('Neuron Activation by Semantic Category')

        plt.colorbar(im, label='Average Activation')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")


# ============================================================
# 12. Knowledge Neuron Content Analysis
# ============================================================

class KnowledgeNeuronAnalyzer:
    """v15 Knowledge Neuron 내용 분석"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.detector = VersionDetector(model)

    @torch.no_grad()
    def collect_knowledge_activations(self, dataloader, max_batches: int = 50) -> Dict:
        """Knowledge neuron별 활성화 문맥 수집"""
        print(f"\n{'='*60}")
        print(f"KNOWLEDGE NEURON CONTENT ANALYSIS")
        print(f"{'='*60}")

        self.model.eval()

        knowledge_contexts = defaultdict(list)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting knowledge activations", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                mem = routing_info.get('memory', {})

                if 'fine_indices' not in mem or 'fine_weights' not in mem:
                    continue

                fine_indices = mem['fine_indices'].cpu().numpy()
                fine_weights = mem['fine_weights'].cpu().numpy()

                for b in range(B):
                    tokens = input_ids[b].cpu().tolist()
                    context = self.tokenizer.decode(tokens, skip_special_tokens=True)

                    for s in range(S):
                        token_str = self.tokenizer.decode([tokens[s]]).strip()

                        # Skip special tokens
                        if token_str.lower() in ['[pad]', '[cls]', '[sep]', '[unk]', '<pad>', '<s>', '</s>', '<unk>']:
                            continue

                        for k_idx, k_weight in zip(fine_indices[b, s], fine_weights[b, s]):
                            if k_weight > 0.1:
                                knowledge_contexts[int(k_idx)].append({
                                    'context': context,
                                    'token': token_str,
                                    'position': s,
                                    'weight': float(k_weight),
                                    'layer': layer_idx,
                                })

        return dict(knowledge_contexts)

    def analyze_knowledge_content(self, knowledge_contexts: Dict, top_n: int = 10) -> Dict:
        """Knowledge neuron별 공통 패턴 분석"""
        results = {'neurons': {}, 'summary': {}}

        for k_id, contexts in knowledge_contexts.items():
            if len(contexts) < 5:
                continue

            token_counts = Counter([c['token'].lower() for c in contexts])
            top_tokens = token_counts.most_common(top_n)

            avg_weight = np.mean([c['weight'] for c in contexts])
            layer_dist = Counter([c['layer'] for c in contexts])

            results['neurons'][k_id] = {
                'n_activations': len(contexts),
                'top_tokens': top_tokens,
                'avg_weight': float(avg_weight),
                'layer_distribution': dict(layer_dist),
                'example_contexts': [c['context'][:100] for c in contexts[:3]],
            }

        total_neurons = len(results['neurons'])
        if total_neurons > 0:
            avg_activations = np.mean([n['n_activations'] for n in results['neurons'].values()])
            results['summary'] = {
                'analyzed_neurons': total_neurons,
                'avg_activations_per_neuron': float(avg_activations),
            }

        print(f"\n--- Knowledge Neuron Analysis ---")
        print(f"Analyzed {total_neurons} neurons")

        sorted_neurons = sorted(results['neurons'].items(), key=lambda x: x[1]['n_activations'], reverse=True)

        print(f"\n--- Most Active Knowledge Neurons ---")
        for k_id, data in sorted_neurons[:5]:
            tokens = [t for t, _ in data['top_tokens'][:5]]
            print(f"  K_{k_id}: {data['n_activations']} activations")
            print(f"    Top tokens: {tokens}")

        return results

    def run_analysis(self, dataloader, max_batches: int = 50) -> Dict:
        """전체 분석 실행"""
        contexts = self.collect_knowledge_activations(dataloader, max_batches)
        return self.analyze_knowledge_content(contexts)


# ============================================================
# 13. Cross-layer Semantic Consistency
# ============================================================

class CrossLayerConsistencyAnalyzer:
    """레이어 간 의미 표현 일관성 분석"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.detector = VersionDetector(model)
        self.n_layers = self.model.n_layers

    @torch.no_grad()
    def analyze_consistency(self, dataloader, max_batches: int = 30) -> Dict:
        """레이어 간 뉴런 활성화 일관성 분석"""
        print(f"\n{'='*60}")
        print(f"CROSS-LAYER CONSISTENCY ANALYSIS")
        print(f"{'='*60}")

        self.model.eval()

        layer_correlations = defaultdict(list)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing layers", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            layer_weights = []
            for routing_info in routing_infos:
                weights, _ = self.detector.get_weights_from_routing(routing_info)
                if weights is not None:
                    if weights.dim() == 3:
                        weights = weights.mean(dim=1)
                    layer_weights.append(weights.cpu().numpy())

            for i in range(len(layer_weights)):
                for j in range(i+1, len(layer_weights)):
                    for b in range(layer_weights[i].shape[0]):
                        corr = np.corrcoef(layer_weights[i][b], layer_weights[j][b])[0, 1]
                        if not np.isnan(corr):
                            layer_correlations[(i, j)].append(corr)

        results = {
            'layer_pairs': {},
            'summary': {},
        }

        correlation_matrix = np.zeros((self.n_layers, self.n_layers))

        for (i, j), corrs in layer_correlations.items():
            avg_corr = float(np.mean(corrs))
            results['layer_pairs'][f"L{i}_L{j}"] = {
                'correlation': avg_corr,
                'std': float(np.std(corrs)),
                'n_samples': len(corrs),
            }
            correlation_matrix[i, j] = avg_corr
            correlation_matrix[j, i] = avg_corr

        np.fill_diagonal(correlation_matrix, 1.0)
        results['correlation_matrix'] = correlation_matrix.tolist()

        adjacent_corrs = [results['layer_pairs'].get(f"L{i}_L{i+1}", {}).get('correlation', 0)
                         for i in range(self.n_layers - 1)]
        distant_corrs = [results['layer_pairs'].get(f"L0_L{self.n_layers-1}", {}).get('correlation', 0)]

        results['summary'] = {
            'avg_adjacent_correlation': float(np.mean(adjacent_corrs)) if adjacent_corrs else 0,
            'first_last_correlation': distant_corrs[0] if distant_corrs else 0,
        }

        print(f"\n--- Layer Pair Correlations ---")
        for pair, data in sorted(results['layer_pairs'].items()):
            print(f"  {pair}: {data['correlation']:.4f} (±{data['std']:.4f})")

        print(f"\n--- Summary ---")
        print(f"  Adjacent layers avg: {results['summary']['avg_adjacent_correlation']:.4f}")
        print(f"  First-Last correlation: {results['summary']['first_last_correlation']:.4f}")

        return results

    def visualize(self, results: Dict, output_path: str):
        """레이어 상관관계 히트맵"""
        if not HAS_MATPLOTLIB or 'correlation_matrix' not in results:
            return

        corr_matrix = np.array(results['correlation_matrix'])

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

        ax.set_xticks(range(len(corr_matrix)))
        ax.set_yticks(range(len(corr_matrix)))
        ax.set_xticklabels([f'L{i}' for i in range(len(corr_matrix))])
        ax.set_yticklabels([f'L{i}' for i in range(len(corr_matrix))])

        ax.set_title('Cross-Layer Neuron Activation Correlation')
        plt.colorbar(im, label='Correlation')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def load_model_and_data(args):
    """Load model, tokenizer, and data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        best = checkpoint_path / 'best_model.pt'
        checkpoint_path = best if best.exists() else max(checkpoint_path.glob('*.pt'), key=lambda p: p.stat().st_mtime)

    print(f"\nLoading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    # Detect version
    path_str = str(checkpoint_path).lower()
    if 'v15' in path_str:
        version = '15.0'
    elif 'v14' in path_str:
        version = '14.0'
    elif 'v13' in path_str:
        version = '13.0'
    else:
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        keys_str = ' '.join(state_dict.keys())
        if 'knowledge_encoder' in keys_str:
            version = '15.0'
        elif 'feature_neurons' in keys_str:
            version = '14.0'
        elif 'context_proj' in keys_str:
            version = '13.0'
        else:
            version = '12.0'

    print(f"Model version: {version}")

    # Create model
    from models import create_model_by_version

    model_kwargs = {
        'vocab_size': config.get('vocab_size', 30522),
        'd_model': config.get('d_model', 320),
        'n_layers': config.get('n_layers', 4),
        'n_heads': config.get('n_heads', 4),
        'rank': config.get('rank', 64),
        'max_seq_len': config.get('max_seq_len', 128),
        'n_compress': config.get('n_compress', 48),
        'n_expand': config.get('n_expand', 12),
        'n_knowledge': config.get('n_knowledge', 80),
        'dropout': config.get('dropout', 0.1),
    }

    if version.startswith('12') or version.startswith('13') or version.startswith('14') or version.startswith('15'):
        model_kwargs['state_dim'] = config.get('state_dim', 64)

    if version.startswith('15'):
        model_kwargs['n_feature'] = config.get('n_feature', 48)
        model_kwargs['n_relational'] = config.get('n_relational', 12)
        model_kwargs['n_value'] = config.get('n_value', 12)
        model_kwargs['knowledge_rank'] = config.get('knowledge_rank', 128)
        model_kwargs['coarse_k'] = config.get('coarse_k', 20)
        model_kwargs['fine_k'] = config.get('fine_k', 10)
    elif version.startswith('14'):
        model_kwargs['n_feature'] = config.get('n_feature', config.get('n_compress', 48))
        model_kwargs['n_relational'] = config.get('n_relational', config.get('n_expand', 12))
        model_kwargs['n_transfer'] = config.get('n_transfer', config.get('n_expand', 12))

    model = create_model_by_version(version, model_kwargs)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Data
    dataloader = None
    if args.val_data:
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
                    self.texts[idx], truncation=True, max_length=self.max_len,
                    padding='max_length', return_tensors='pt'
                )
                return {'input_ids': encoding['input_ids'].squeeze(0)}

        dataset = SimpleDataset(val_texts, tokenizer)
        num_workers = getattr(args, 'num_workers', 4)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None
        )

    return model, tokenizer, dataloader, device


def main():
    parser = argparse.ArgumentParser(description='DAWN Neuron Analysis')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./neuron_analysis')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'word_map', 'cluster', 'similarity',
                                'ablation', 'probe', 'layer', 'trajectory',
                                'distribution', 'context',
                                'semantic_path', 'semantic_category', 'knowledge_content', 'cross_layer'])
    parser.add_argument('--max_batches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--text', type=str, default="The bank by the river was steep.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer, dataloader, device = load_model_and_data(args)

    all_results = {}

    # 1. Word Mapping
    if args.mode in ['all', 'word_map'] and dataloader:
        mapper = NeuronWordMapper(model, tokenizer, device)
        results = mapper.build_mapping(dataloader, args.max_batches)
        mapper.visualize(results, args.output_dir)
        all_results['word_map'] = {'summary': results['summary']}

    # 2. Clustering
    if args.mode in ['all', 'cluster']:
        clusterer = NeuronClusterer(model, device)
        results = clusterer.cluster(n_clusters=8, method='tsne')
        clusterer.visualize(results, os.path.join(args.output_dir, 'clustering.png'))
        all_results['cluster'] = results

    # 3. Similarity
    if args.mode in ['all', 'similarity']:
        sim_analyzer = NeuronSimilarityAnalyzer(model, device)
        results = sim_analyzer.compute_similarity()
        sim_analyzer.visualize(results, os.path.join(args.output_dir, 'similarity.png'))
        all_results['similarity'] = {'stats': results['stats']}

    # 4. Ablation
    if args.mode in ['all', 'ablation'] and dataloader:
        ablation = NeuronAblation(model, tokenizer, device)
        results = ablation.run_ablation(dataloader, max_batches=20)
        ablation.visualize(results, os.path.join(args.output_dir, 'ablation.png'))
        all_results['ablation'] = results

    # 5. Probing
    if args.mode in ['all', 'probe'] and dataloader:
        probe = NeuronProbe(model, tokenizer, device)
        results = probe.train_probe(dataloader, args.max_batches)
        all_results['probe'] = results

    # 6. Layer-wise
    if args.mode in ['all', 'layer'] and dataloader:
        layer_analyzer = LayerRoleAnalyzer(model, tokenizer, device)
        results = layer_analyzer.analyze(dataloader, args.max_batches)
        all_results['layer'] = {
            'n_layers': layer_analyzer.n_layers,
            'layers_analyzed': len(results['layers']),
            'role_changes': len(results['role_changes'])
        }

    # 7. Trajectory
    if args.mode in ['all', 'trajectory']:
        trajectory = TokenTrajectory(model, tokenizer, device)
        results = trajectory.trace(args.text)
        all_results['trajectory'] = results

    # 8. Distribution
    if args.mode in ['all', 'distribution'] and dataloader:
        dist_analyzer = ActivationDistribution(model, tokenizer, device)
        results = dist_analyzer.analyze(dataloader, args.max_batches)
        dist_analyzer.visualize(results, os.path.join(args.output_dir, 'distribution.png'))
        all_results['distribution'] = {'summary': results['summary']}

    # 9. Context
    if args.mode in ['all', 'context']:
        ctx_analyzer = ContextDependency(model, tokenizer, device)
        contexts = [
            "The bank by the river was steep.",
            "I went to the bank to deposit money.",
        ]
        results = ctx_analyzer.analyze_word("bank", contexts)
        all_results['context'] = results

    # 10. Semantic Path Similarity
    if args.mode in ['all', 'semantic_path']:
        path_analyzer = SemanticPathAnalyzer(model, tokenizer, device)
        results = path_analyzer.run_default_analysis()
        all_results['semantic_path'] = results

    # 11. Semantic Category
    if args.mode in ['all', 'semantic_category']:
        cat_analyzer = SemanticCategoryAnalyzer(model, tokenizer, device)
        results = cat_analyzer.analyze_categories()
        cat_analyzer.visualize(results, os.path.join(args.output_dir, 'category_neurons.png'))
        all_results['semantic_category'] = {
            'category_top_neurons': results['category_top_neurons'],
            'category_separation': results['category_separation'],
        }

    # 12. Knowledge Neuron Content (v15)
    if args.mode in ['all', 'knowledge_content'] and dataloader:
        base_model = get_underlying_model(model)
        if hasattr(base_model, 'knowledge_encoder'):
            knowledge_analyzer = KnowledgeNeuronAnalyzer(model, tokenizer, device)
            results = knowledge_analyzer.run_analysis(dataloader, args.max_batches)
            all_results['knowledge_content'] = results

    # 13. Cross-layer Consistency
    if args.mode in ['all', 'cross_layer'] and dataloader:
        cross_analyzer = CrossLayerConsistencyAnalyzer(model, tokenizer, device)
        results = cross_analyzer.analyze_consistency(dataloader, args.max_batches)
        cross_analyzer.visualize(results, os.path.join(args.output_dir, 'cross_layer.png'))
        all_results['cross_layer'] = results

    # Save results
    results_path = os.path.join(args.output_dir, 'neuron_analysis.json')

    def convert_to_serializable(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"Results saved: {results_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
