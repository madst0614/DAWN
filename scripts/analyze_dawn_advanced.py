#!/usr/bin/env python3
"""
DAWN Advanced Analysis Script
============================

Advanced analysis features for DAWN models.

Features:
1. SVD/PCA Weight Analysis - 뉴런 가중치 저차원 분석
2. Sentence Visualization - 문장별 뉴런 활성화 시각화
3. Ablation Experiments - 뉴런 제거 실험
4. Semantic Analysis - 의미론적 뉴런 특화
5. Neuron Catalog - 뉴런 역할 카탈로그

Usage:
    python analyze_dawn_advanced.py --checkpoint <path> --val_data <path>
    python analyze_dawn_advanced.py --checkpoint <path> --val_data <path> --mode svd
    python analyze_dawn_advanced.py --checkpoint <path> --val_data <path> --mode sentence --text "The cat sat on the mat."
"""

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            IN_NOTEBOOK = True
        else:
            IN_NOTEBOOK = False
            matplotlib.use('Agg')
    except (ImportError, AttributeError):
        IN_NOTEBOOK = False
        matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    IN_NOTEBOOK = False

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================
# Utilities (shared with analyze_dawn.py)
# ============================================================

def get_underlying_model(model):
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def simple_pos_tag(token: str) -> str:
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
    if token_lower.endswith('ing'):
        return 'VERB_ING'
    if token_lower.endswith('ed'):
        return 'VERB_ED'
    if token_lower.endswith('ly'):
        return 'ADV'
    return 'OTHER'


def convert_to_serializable(obj):
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


# ============================================================
# 1. SVD/PCA Weight Analysis
# ============================================================

class NeuronSVDAnalyzer:
    """Analyze neuron weights using SVD/PCA"""

    def __init__(self, model, device='cuda'):
        self.model = get_underlying_model(model)
        self.device = device
        self.d_model = self.model.d_model
        self.rank = getattr(self.model, 'rank', self.d_model // 4)

        # Detect version and get neuron counts
        if hasattr(self.model, 'global_routers') and hasattr(self.model.global_routers, 'neuron_router'):
            router = self.model.global_routers.neuron_router
            if hasattr(router, 'usage_ema_feature'):
                self.version = '14'
                self.n_feature = router.n_feature
                self.n_relational = router.n_relational
                self.n_transfer = router.n_transfer
                # Legacy aliases
                self.n_compress = self.n_feature
                self.n_expand = self.n_relational
            else:
                self.version = '13'
                self.n_compress = self.model.shared_neurons.compress_neurons.shape[0]
                self.n_expand = self._get_expand_count()
                self.n_feature = self.n_compress
                self.n_relational = self.n_expand
                self.n_transfer = self.n_expand
        elif hasattr(self.model, 'shared_neurons'):
            self.version = 'legacy'
            self.n_compress = self.model.shared_neurons.compress_neurons.shape[0]
            self.n_expand = self._get_expand_count()
            self.n_feature = self.n_compress
            self.n_relational = self.n_expand
            self.n_transfer = self.n_expand
        else:
            self.version = 'unknown'
            self.n_compress = getattr(self.model, 'n_compress', 48)
            self.n_expand = getattr(self.model, 'n_expand', 12)
            self.n_feature = self.n_compress
            self.n_relational = self.n_expand
            self.n_transfer = self.n_expand

        print(f"\n{'='*60}")
        print(f"Neuron SVD/PCA Analyzer (v{self.version})")
        print(f"{'='*60}")
        if self.version == '14':
            print(f"n_feature: {self.n_feature}, n_relational: {self.n_relational}, n_transfer: {self.n_transfer}")
        else:
            print(f"n_compress: {self.n_compress}, n_expand: {self.n_expand}")
        print(f"d_model: {self.d_model}, rank: {self.rank}")

    def _get_expand_count(self):
        """Get expand neuron count from shared_neurons"""
        shared = self.model.shared_neurons
        if hasattr(shared, 'expand_neurons_pool'):
            return shared.expand_neurons_pool.shape[0]
        elif hasattr(shared, 'expand_neurons'):
            return shared.expand_neurons.shape[0]
        return self.n_compress

    def extract_neurons(self) -> Dict[str, np.ndarray]:
        """Extract neuron weights from model"""
        neurons = {}

        if self.version == '14':
            # v14: neurons stored in global_neurons
            global_neurons = self.model.global_neurons
            # Use FRTK naming
            neurons['feature'] = global_neurons.feature_neurons.data.cpu().numpy()
            neurons['relational'] = global_neurons.relational_neurons.data.cpu().numpy()
            neurons['transfer'] = global_neurons.transfer_neurons.data.cpu().numpy()

            if hasattr(global_neurons, 'knowledge_K'):
                neurons['knowledge_K'] = global_neurons.knowledge_K.data.cpu().numpy()
                neurons['knowledge_V'] = global_neurons.knowledge_V.data.cpu().numpy()

            # Legacy aliases for compatibility
            neurons['compress'] = neurons['feature']
            neurons['expand'] = neurons['relational']
        else:
            # Legacy versions: use shared_neurons
            shared = self.model.shared_neurons
            neurons['compress'] = shared.compress_neurons.data.cpu().numpy()

            if hasattr(shared, 'expand_neurons_pool'):
                neurons['expand'] = shared.expand_neurons_pool.data.cpu().numpy()
            elif hasattr(shared, 'expand_neurons'):
                neurons['expand'] = shared.expand_neurons.data.cpu().numpy()

            if hasattr(shared, 'knowledge_K'):
                neurons['knowledge_K'] = shared.knowledge_K.data.cpu().numpy()
                neurons['knowledge_V'] = shared.knowledge_V.data.cpu().numpy()

            # v14 aliases for compatibility
            neurons['feature'] = neurons['compress']
            neurons['relational'] = neurons.get('expand', neurons['compress'])
            neurons['transfer'] = neurons.get('expand', neurons['compress'])

        print(f"\nExtracted neurons:")
        for name, arr in neurons.items():
            if name not in ['compress', 'expand']:  # Skip legacy aliases for v14
                print(f"  {name}: {arr.shape}")

        return neurons

    def analyze_pca(self, neurons_flat: np.ndarray, name: str,
                    k_values: List[int] = [4, 8, 16, 32, 64]) -> Dict:
        """Perform PCA analysis on flattened neurons"""
        print(f"\n--- {name} PCA Analysis ---")
        print(f"Shape: {neurons_flat.shape}")

        N, F = neurons_flat.shape
        max_components = min(N, F)

        if HAS_SKLEARN:
            pca = PCA()
            pca.fit(neurons_flat)
            explained_ratio = pca.explained_variance_ratio_
            singular_values = pca.singular_values_
        else:
            # Torch SVD fallback
            mean = neurons_flat.mean(axis=0, keepdims=True)
            centered = neurons_flat - mean
            U, S, Vh = np.linalg.svd(centered, full_matrices=False)
            total_var = (S ** 2).sum()
            explained_ratio = (S ** 2) / total_var
            singular_values = S

        cumsum = np.cumsum(explained_ratio)

        results = {'k_explained': {}, 'cumsum': cumsum[:100].tolist()}

        print(f"\nVariance explained by K components:")
        for k in k_values:
            if k <= len(cumsum):
                pct = cumsum[k-1] * 100
                results['k_explained'][k] = pct
                status = "excellent" if pct > 95 else ("good" if pct > 90 else "moderate")
                print(f"  K={k:3d}: {pct:6.2f}% - {status}")

        # Effective dimensionality
        p = explained_ratio + 1e-10
        entropy = -np.sum(p * np.log(p))
        eff_dim = np.exp(entropy)
        results['effective_dim'] = float(eff_dim)
        results['max_dim'] = max_components
        print(f"\nEffective dimensionality: {eff_dim:.1f}/{max_components}")

        # K for thresholds
        results['k_for_threshold'] = {}
        for thresh in [0.90, 0.95, 0.99]:
            k_needed = np.searchsorted(cumsum, thresh) + 1
            results['k_for_threshold'][f"{int(thresh*100)}%"] = int(k_needed)

        return results, explained_ratio, singular_values

    def analyze_similarity(self, neurons_flat: np.ndarray) -> Dict:
        """Analyze neuron similarity"""
        N = neurons_flat.shape[0]
        neurons_norm = neurons_flat / (np.linalg.norm(neurons_flat, axis=1, keepdims=True) + 1e-10)
        sim_matrix = neurons_norm @ neurons_norm.T
        mask = ~np.eye(N, dtype=bool)
        off_diag = sim_matrix[mask]

        return {
            'mean': float(np.mean(off_diag)),
            'std': float(np.std(off_diag)),
            'max': float(np.max(off_diag)),
            'min': float(np.min(off_diag)),
        }, sim_matrix

    def run_all(self) -> Dict:
        """Run all SVD/PCA analyses"""
        neurons = self.extract_neurons()
        results = {}

        # v14: use FRTK naming
        is_v14 = self.version == '14'
        feature_name = "FeatureNeurons" if is_v14 else "CompressNeurons"
        relational_name = "RelationalNeurons" if is_v14 else "ExpandNeurons"

        # Feature/Compress neurons
        feature = neurons['feature']
        N, D, R = feature.shape
        feature_flat = feature.reshape(N, D * R)

        pca_results, ratio, sv = self.analyze_pca(feature_flat, feature_name)
        sim_results, sim_matrix = self.analyze_similarity(feature_flat)

        results['feature'] = {
            'pca': pca_results,
            'similarity': sim_results
        }
        # Legacy alias
        results['compress'] = results['feature']

        # Relational/Expand neurons
        if 'relational' in neurons:
            relational = neurons['relational']
            if len(relational.shape) == 3:
                N, R, D = relational.shape
                relational_flat = relational.reshape(N, R * D)
            else:
                relational_flat = relational

            pca_results, _, _ = self.analyze_pca(relational_flat, relational_name)
            sim_results, _ = self.analyze_similarity(relational_flat)

            results['relational'] = {
                'pca': pca_results,
                'similarity': sim_results
            }
            # Legacy alias
            results['expand'] = results['relational']

        # v14: Transfer neurons (separate from relational)
        if is_v14 and 'transfer' in neurons:
            transfer = neurons['transfer']
            if len(transfer.shape) == 3:
                N, R, D = transfer.shape
                transfer_flat = transfer.reshape(N, R * D)
            else:
                transfer_flat = transfer

            pca_results, _, _ = self.analyze_pca(transfer_flat, "TransferNeurons")
            sim_results, _ = self.analyze_similarity(transfer_flat)

            results['transfer'] = {
                'pca': pca_results,
                'similarity': sim_results
            }

        # Knowledge neurons
        if 'knowledge_K' in neurons:
            K = neurons['knowledge_K']
            V = neurons['knowledge_V']

            pca_K, _, _ = self.analyze_pca(K, "Knowledge K")
            pca_V, _, _ = self.analyze_pca(V, "Knowledge V")

            results['knowledge'] = {
                'K_pca': pca_K,
                'V_pca': pca_V
            }

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        if is_v14:
            print(f"\n[FeatureNeurons]")
            print(f"  Effective dim: {results['feature']['pca']['effective_dim']:.1f}/{self.n_feature}")
            print(f"  Similarity: {results['feature']['similarity']['mean']:.4f}")

            if 'relational' in results:
                print(f"\n[RelationalNeurons]")
                print(f"  Effective dim: {results['relational']['pca']['effective_dim']:.1f}/{self.n_relational}")
                print(f"  Similarity: {results['relational']['similarity']['mean']:.4f}")

            if 'transfer' in results:
                print(f"\n[TransferNeurons]")
                print(f"  Effective dim: {results['transfer']['pca']['effective_dim']:.1f}/{self.n_transfer}")
                print(f"  Similarity: {results['transfer']['similarity']['mean']:.4f}")
        else:
            print(f"\n[CompressNeurons]")
            print(f"  Effective dim: {results['compress']['pca']['effective_dim']:.1f}/{self.n_compress}")
            print(f"  Similarity: {results['compress']['similarity']['mean']:.4f}")

            if 'expand' in results:
                print(f"\n[ExpandNeurons]")
                print(f"  Effective dim: {results['expand']['pca']['effective_dim']:.1f}/{self.n_expand}")
                print(f"  Similarity: {results['expand']['similarity']['mean']:.4f}")

        return results

    def visualize(self, results: Dict, output_path: str):
        """Visualize SVD/PCA results"""
        if not HAS_MATPLOTLIB:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Compress cumulative variance
        ax = axes[0, 0]
        cumsum = results['compress']['pca']['cumsum']
        ax.plot(cumsum[:min(50, len(cumsum))], 'b-', linewidth=2)
        ax.axhline(y=0.9, color='r', linestyle='--', label='90%')
        ax.axhline(y=0.95, color='g', linestyle='--', label='95%')
        ax.set_xlabel('Component')
        ax.set_ylabel('Cumulative Variance')
        ax.set_title('CompressNeurons - Cumulative Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Expand cumulative variance
        ax = axes[0, 1]
        if 'expand' in results:
            cumsum = results['expand']['pca']['cumsum']
            ax.plot(cumsum[:min(50, len(cumsum))], 'b-', linewidth=2)
            ax.axhline(y=0.9, color='r', linestyle='--')
            ax.axhline(y=0.95, color='g', linestyle='--')
            ax.set_xlabel('Component')
            ax.set_ylabel('Cumulative Variance')
            ax.set_title('ExpandNeurons - Cumulative Variance')
            ax.grid(True, alpha=0.3)

        # 3. Similarity comparison
        ax = axes[1, 0]
        names = ['Compress']
        sims = [results['compress']['similarity']['mean']]
        if 'expand' in results:
            names.append('Expand')
            sims.append(results['expand']['similarity']['mean'])

        colors = ['red' if s > 0.5 else 'orange' if s > 0.3 else 'green' for s in sims]
        ax.bar(names, sims, color=colors)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        ax.set_ylabel('Mean Cosine Similarity')
        ax.set_title('Neuron Similarity')
        ax.set_ylim(0, 1)

        # 4. Effective dimensionality
        ax = axes[1, 1]
        names = ['Compress']
        eff_dims = [results['compress']['pca']['effective_dim']]
        max_dims = [results['compress']['pca']['max_dim']]

        if 'expand' in results:
            names.append('Expand')
            eff_dims.append(results['expand']['pca']['effective_dim'])
            max_dims.append(results['expand']['pca']['max_dim'])

        x = range(len(names))
        ax.bar(x, max_dims, label='Max', alpha=0.3, color='gray')
        ax.bar(x, eff_dims, label='Effective', color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel('Dimensionality')
        ax.set_title('Effective vs Max Dimensionality')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()

        print(f"\nSaved: {output_path}")


# ============================================================
# 2. Sentence Visualization
# ============================================================

class SentenceVisualizer:
    """Visualize token-neuron mapping for sentences"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def analyze_sentence(self, text: str) -> Dict:
        """Analyze neuron activations for a sentence"""
        self.model.eval()

        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor([tokens], device=self.device)

        _, routing_infos = self.model(input_ids, return_routing_info=True)

        token_strs = [self.tokenizer.decode([t]).strip() for t in tokens]

        layer_data = []
        for layer_idx, routing_info in enumerate(routing_infos):
            layer_result = {'layer': layer_idx, 'Q': None, 'K': None, 'V': None}

            attn = routing_info.get('attention', routing_info)

            # Handle different version formats
            if 'Q' in attn and isinstance(attn['Q'], dict):
                # v10 format
                for comp in ['Q', 'K', 'V']:
                    if comp in attn:
                        data = attn[comp]
                        weights = data['weights'][0].cpu().numpy()
                        if 'indices' in data:
                            indices = data['indices'][0].cpu().numpy()
                        else:
                            k = min(8, weights.shape[-1])
                            _, idx = torch.topk(torch.tensor(weights), k, dim=-1)
                            indices = idx.numpy()

                        layer_result[comp] = {'weights': weights, 'indices': indices}

            elif 'feature_weights' in attn or 'feature_pref' in attn:
                # v14 FRTK format
                n_tokens = len(tokens)

                # Feature weights (for Q/K/V compress equivalent)
                if 'feature_weights' in attn:
                    raw_weights = attn['feature_weights'][0].cpu()
                    if len(raw_weights.shape) == 1:
                        weights = raw_weights.unsqueeze(0).expand(n_tokens, -1).numpy()
                    else:
                        weights = raw_weights.numpy()

                    k = min(8, weights.shape[-1])
                    _, idx = torch.topk(torch.tensor(weights), k, dim=-1)
                    indices = idx.numpy()

                    # Use feature weights for all components (they share feature neurons)
                    for comp in ['Q', 'K', 'V']:
                        layer_result[comp] = {'weights': weights, 'indices': indices}

            elif 'compress_weights' in attn:
                # v12 format (including v12.7/v13 with top-k)
                # Try dense weights first for better visualization
                if 'compress_weights_dense' in attn:
                    raw_weights = attn['compress_weights_dense'][0].cpu()  # v12.7/v13
                else:
                    raw_weights = attn['compress_weights'][0].cpu()  # First batch

                # v12.3+: [n_compress] (batch-level) -> expand to [S, n_compress]
                # v12.x: [S, n_compress] (token-level) -> keep as is
                if len(raw_weights.shape) == 1:
                    # Batch-level routing: expand to all tokens
                    n_tokens = len(tokens)
                    weights = raw_weights.unsqueeze(0).expand(n_tokens, -1).numpy()
                else:
                    weights = raw_weights.numpy()

                k = min(8, weights.shape[-1])
                _, idx = torch.topk(torch.tensor(weights), k, dim=-1)
                indices = idx.numpy()

                for comp in ['Q', 'K', 'V']:
                    layer_result[comp] = {'weights': weights, 'indices': indices}

            layer_data.append(layer_result)

        return {
            'text': text,
            'tokens': token_strs,
            'token_ids': tokens,
            'layer_data': layer_data,
            'n_layers': len(routing_infos)
        }

    def visualize_sentence(self, analysis: Dict, output_path: str):
        """Create token-neuron heatmap"""
        if not HAS_MATPLOTLIB:
            return

        tokens = analysis['tokens']
        n_tokens = len(tokens)
        layer_data = analysis['layer_data']

        # Show first 4 layers
        n_show = min(4, len(layer_data))

        fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show))
        if n_show == 1:
            axes = [axes]

        for row_idx in range(n_show):
            ax = axes[row_idx]
            layer = layer_data[row_idx]

            if layer['Q'] is not None:
                indices = layer['Q']['indices']
                weights = layer['Q']['weights']

                # Create heatmap
                unique_neurons = sorted(set(indices.flatten()))[:40]
                neuron_to_idx = {n: i for i, n in enumerate(unique_neurons)}

                heatmap = np.zeros((n_tokens, len(unique_neurons)))
                for t in range(n_tokens):
                    for ki in range(indices.shape[1]):
                        n_idx = indices[t, ki]
                        if n_idx in neuron_to_idx:
                            w_idx = ki if weights.shape[-1] == indices.shape[1] else n_idx
                            if w_idx < weights.shape[-1]:
                                heatmap[t, neuron_to_idx[n_idx]] += weights[t, w_idx]

                im = ax.imshow(heatmap.T, aspect='auto', cmap='YlOrRd')
                ax.set_xticks(range(n_tokens))
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel(f'Layer {row_idx}\nNeuron')
                plt.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, f'Layer {row_idx}: No data', ha='center', va='center')
                ax.axis('off')

        plt.suptitle(f'Token-Neuron Activation: "{analysis["text"][:50]}..."', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()

        print(f"Saved: {output_path}")


# ============================================================
# 3. Semantic Analysis
# ============================================================

class SemanticAnalyzer:
    """Analyze semantic specialization of neurons"""

    SEMANTIC_CATEGORIES = {
        'PERSON': ['he', 'she', 'man', 'woman', 'president', 'king', 'queen',
                   'doctor', 'teacher', 'father', 'mother', 'boy', 'girl'],
        'PLACE': ['city', 'country', 'building', 'street', 'house', 'room',
                  'world', 'place', 'town', 'village', 'area'],
        'TIME': ['year', 'day', 'month', 'week', 'hour', 'time',
                 'yesterday', 'today', 'tomorrow', 'morning', 'night'],
        'ACTION': ['go', 'come', 'run', 'walk', 'move', 'take', 'give',
                   'make', 'do', 'see', 'know', 'think', 'say'],
        'EMOTION': ['love', 'hate', 'fear', 'happy', 'sad', 'angry',
                    'feel', 'want', 'like', 'enjoy'],
    }

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device

        # Detect version and get n_neurons
        if hasattr(self.model, 'global_routers') and hasattr(self.model.global_routers, 'neuron_router'):
            router = self.model.global_routers.neuron_router
            if hasattr(router, 'usage_ema_feature'):
                self.version = '14'
                self.n_neurons = router.n_feature
            else:
                self.version = '13'
                self.n_neurons = self.model.shared_neurons.compress_neurons.shape[0]
        elif hasattr(self.model, 'shared_neurons'):
            self.version = 'legacy'
            self.n_neurons = self.model.shared_neurons.compress_neurons.shape[0]
        else:
            self.version = 'unknown'
            self.n_neurons = getattr(self.model, 'n_compress', 48)

        # Build token ID sets
        self.category_token_ids = {}
        for category, words in self.SEMANTIC_CATEGORIES.items():
            token_ids = set()
            for word in words:
                for variant in [word, word.capitalize(), ' ' + word]:
                    ids = self.tokenizer.encode(variant, add_special_tokens=False)
                    token_ids.update(ids)
            self.category_token_ids[category] = token_ids

    @torch.no_grad()
    def analyze(self, dataloader, max_batches: int = 50) -> Dict:
        """Find neurons specialized for semantic categories"""
        print(f"\n{'='*60}")
        print(f"SEMANTIC NEURON ANALYSIS (v{self.version})")
        print(f"{'='*60}")

        self.model.eval()

        n_neurons = self.n_neurons
        category_counts = {cat: np.zeros(n_neurons) for cat in self.SEMANTIC_CATEGORIES}
        total_counts = np.zeros(n_neurons)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Semantic Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # Use layer 0
            routing_info = routing_infos[0]
            attn = routing_info.get('attention', routing_info)

            # Get weights based on version
            if 'Q' in attn and isinstance(attn['Q'], dict):
                # v10 format
                weights = attn['Q']['weights']
                indices = attn['Q'].get('indices')
                is_batch_level = False
            elif 'feature_weights' in attn:
                # v14 FRTK format
                weights = attn['feature_weights']
                indices = None  # v14 doesn't expose top-k indices separately
                is_batch_level = (len(weights.shape) == 2)
            elif 'compress_weights' in attn:
                # v12.7/v13: prefer dense weights for analysis
                if 'compress_weights_dense' in attn:
                    weights = attn['compress_weights_dense']
                else:
                    weights = attn['compress_weights']
                indices = attn.get('compress_topk_idx')
                # v12.3+: [B, n_compress] (batch-level), v12.x: [B, S, n_compress] (token-level)
                is_batch_level = (len(weights.shape) == 2)
            else:
                continue

            if indices is None:
                k = min(8, weights.shape[-1])
                _, indices = torch.topk(weights, k, dim=-1)

            for b in range(B):
                if is_batch_level:
                    # Batch-level routing: same weights for all tokens in batch
                    batch_neurons = indices[b].cpu().numpy()
                    batch_weights = weights[b].cpu().numpy()

                    for s in range(S):
                        token_id = input_ids[b, s].item()
                        for ni, w in zip(batch_neurons, batch_weights[:len(batch_neurons)]):
                            total_counts[ni] += w

                            for category, token_ids in self.category_token_ids.items():
                                if token_id in token_ids:
                                    category_counts[category][ni] += w
                else:
                    # Token-level routing: different weights per token
                    for s in range(S):
                        token_id = input_ids[b, s].item()
                        top_neurons = indices[b, s].cpu().numpy()
                        top_weights = weights[b, s].cpu().numpy()

                        for ni, w in zip(top_neurons, top_weights[:len(top_neurons)]):
                            total_counts[ni] += w

                            for category, token_ids in self.category_token_ids.items():
                                if token_id in token_ids:
                                    category_counts[category][ni] += w

        results = {}
        print("\n--- Semantic Category Neurons ---")

        for category in self.SEMANTIC_CATEGORIES:
            counts = category_counts[category]
            normalized = counts / (total_counts + 1e-10)
            top_neurons = np.argsort(normalized)[-10:][::-1]

            results[category] = {
                'top_neurons': [(int(n), float(normalized[n])) for n in top_neurons],
                'total': float(counts.sum())
            }

            print(f"  {category}: neurons {top_neurons[:5].tolist()}")

        return results

    def visualize(self, results: Dict, output_path: str):
        """Visualize semantic heatmap"""
        if not HAS_MATPLOTLIB:
            return

        categories = list(results.keys())

        # Get unique neurons
        all_neurons = set()
        for cat_data in results.values():
            for n, _ in cat_data['top_neurons'][:10]:
                all_neurons.add(n)
        neurons = sorted(all_neurons)

        # Build matrix
        data = np.zeros((len(categories), len(neurons)))
        for i, cat in enumerate(categories):
            for n, score in results[cat]['top_neurons']:
                if n in neurons:
                    j = neurons.index(n)
                    data[i, j] = score

        fig, ax = plt.subplots(figsize=(max(10, len(neurons) * 0.4), 6))
        im = ax.imshow(data, aspect='auto', cmap='YlOrRd')

        ax.set_xticks(range(len(neurons)))
        ax.set_xticklabels([f'N{n}' for n in neurons], rotation=45, ha='right')
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)

        plt.colorbar(im, ax=ax)
        ax.set_title('Semantic Category - Neuron Specialization')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()

        print(f"Saved: {output_path}")


# ============================================================
# 4. Neuron Catalog
# ============================================================

class NeuronCatalog:
    """Build catalog of neuron roles"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device

        # Detect version and get n_neurons
        if hasattr(self.model, 'global_routers') and hasattr(self.model.global_routers, 'neuron_router'):
            router = self.model.global_routers.neuron_router
            if hasattr(router, 'usage_ema_feature'):
                self.version = '14'
                self.n_neurons = router.n_feature
            else:
                self.version = '13'
                self.n_neurons = self.model.shared_neurons.compress_neurons.shape[0]
        elif hasattr(self.model, 'shared_neurons'):
            self.version = 'legacy'
            self.n_neurons = self.model.shared_neurons.compress_neurons.shape[0]
        else:
            self.version = 'unknown'
            self.n_neurons = getattr(self.model, 'n_compress', 48)

    @torch.no_grad()
    def build(self, dataloader, max_batches: int = 100) -> Dict:
        """Build neuron catalog"""
        print(f"\n{'='*60}")
        print(f"BUILDING NEURON CATALOG (v{self.version})")
        print(f"{'='*60}")

        self.model.eval()

        neuron_token_counts = [Counter() for _ in range(self.n_neurons)]
        neuron_pos_counts = [Counter() for _ in range(self.n_neurons)]
        neuron_total_usage = np.zeros(self.n_neurons)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Building Catalog", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # Use layer 0
            routing_info = routing_infos[0]
            attn = routing_info.get('attention', routing_info)

            if 'Q' in attn and isinstance(attn['Q'], dict):
                # v10 format
                weights = attn['Q']['weights']
                indices = attn['Q'].get('indices')
                is_batch_level = False
            elif 'feature_weights' in attn:
                # v14 FRTK format
                weights = attn['feature_weights']
                indices = None  # v14 doesn't expose top-k indices separately
                is_batch_level = (len(weights.shape) == 2)
            elif 'compress_weights' in attn:
                # v12.7/v13: prefer dense weights for catalog building
                if 'compress_weights_dense' in attn:
                    weights = attn['compress_weights_dense']
                else:
                    weights = attn['compress_weights']
                indices = attn.get('compress_topk_idx')
                # v12.3+: [B, n_compress] (batch-level), v12.x: [B, S, n_compress] (token-level)
                is_batch_level = (len(weights.shape) == 2)
            else:
                continue

            if indices is None:
                k = min(8, weights.shape[-1])
                _, indices = torch.topk(weights, k, dim=-1)

            for b in range(B):
                if is_batch_level:
                    # Batch-level routing: same weights for all tokens in batch
                    batch_neurons = indices[b].cpu().numpy()
                    batch_weights = weights[b].cpu().numpy()

                    for s in range(S):
                        token_id = input_ids[b, s].item()
                        token_str = self.tokenizer.decode([token_id]).strip()
                        pos = simple_pos_tag(token_str)

                        for ni, w in zip(batch_neurons, batch_weights[:len(batch_neurons)]):
                            neuron_token_counts[ni][token_str] += 1
                            neuron_pos_counts[ni][pos] += 1
                            neuron_total_usage[ni] += w
                else:
                    # Token-level routing: different weights per token
                    for s in range(S):
                        token_id = input_ids[b, s].item()
                        token_str = self.tokenizer.decode([token_id]).strip()
                        pos = simple_pos_tag(token_str)

                        top_neurons = indices[b, s].cpu().numpy()
                        top_weights = weights[b, s].cpu().numpy()

                        for ni, w in zip(top_neurons, top_weights[:len(top_neurons)]):
                            neuron_token_counts[ni][token_str] += 1
                            neuron_pos_counts[ni][pos] += 1
                            neuron_total_usage[ni] += w

        # Build catalog
        catalog = {}
        for n in range(self.n_neurons):
            top_tokens = neuron_token_counts[n].most_common(20)

            pos_dist = dict(neuron_pos_counts[n])
            total_pos = sum(pos_dist.values())
            if total_pos > 0:
                pos_dist = {k: v / total_pos for k, v in pos_dist.items()}
                primary_pos = max(pos_dist, key=pos_dist.get) if pos_dist else 'UNK'
            else:
                primary_pos = 'UNK'

            # Classify role
            grammar_pos = {'DET', 'ADP', 'AUX', 'CONJ'}
            if primary_pos in grammar_pos and pos_dist.get(primary_pos, 0) > 0.5:
                role = 'grammar'
            elif primary_pos in {'VERB_ING', 'VERB_ED'}:
                role = 'syntactic'
            else:
                role = 'semantic'

            catalog[n] = {
                'top_tokens': [(t, c) for t, c in top_tokens[:10]],
                'primary_pos': primary_pos,
                'pos_distribution': pos_dist,
                'role': role,
                'total_usage': float(neuron_total_usage[n])
            }

        role_counts = Counter(v['role'] for v in catalog.values())

        print("\n--- Neuron Role Distribution ---")
        for role, count in role_counts.most_common():
            print(f"  {role}: {count} neurons ({100*count/self.n_neurons:.1f}%)")

        return {
            'neurons': catalog,
            'role_distribution': dict(role_counts),
            'total_neurons': self.n_neurons
        }


# ============================================================
# Main Runner
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN Advanced Analysis')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./advanced_analysis')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'svd', 'sentence', 'semantic', 'catalog'])
    parser.add_argument('--text', type=str, default="The cat sat on the mat.",
                        help='Text for sentence visualization')
    parser.add_argument('--max_batches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
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
    version = config.get('model_version', 'auto')

    # Auto-detect version
    if version == 'auto':
        # 1. Try checkpoint path
        path_str = str(checkpoint_path).lower()
        if 'v14' in path_str:
            version = '14.0'
        elif 'v13' in path_str:
            version = '13.0'
        elif 'v12_8' in path_str or 'v12.8' in path_str:
            version = '12.8'
        elif 'v12_7' in path_str or 'v12.7' in path_str:
            version = '12.7'
        elif 'v12_6' in path_str or 'v12.6' in path_str:
            version = '12.6'
        elif 'v12_5' in path_str or 'v12.5' in path_str:
            version = '12.5'
        elif 'v12_4' in path_str or 'v12.4' in path_str:
            version = '12.4'
        elif 'v12_3' in path_str or 'v12.3' in path_str:
            version = '12.3'
        elif 'v12_2' in path_str or 'v12.2' in path_str:
            version = '12.2'
        elif 'v12_1' in path_str or 'v12.1' in path_str:
            version = '12.1'
        elif 'v12_0' in path_str or 'v12.0' in path_str or 'v12' in path_str:
            version = '12.0'
        elif 'v11' in path_str:
            version = '11.0'
        elif 'v10' in path_str:
            version = '10.0'
        else:
            # 2. Detect from state_dict keys
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            keys = list(state_dict.keys())
            keys_str = ' '.join(keys)

            if 'feature_neurons' in keys_str or 'usage_ema_feature' in keys_str:
                version = '14.0'  # v14 has FRTK naming
            elif 'context_proj' in keys_str:
                version = '13.0'  # v13 has context_proj in SSM
            elif 'A_log' in keys_str:
                version = '12.7'  # v12.7 has Mamba-style SSM
            elif 'O_compress_pool' in keys_str or 'O_expand_pool' in keys_str:
                version = '12.4'
            elif 'O_pool' in keys_str:
                version = '12.4'
            elif 'expand_neurons_pool' in keys_str:
                version = '12.3'
            elif 'ssm' in keys_str or 'SSM' in keys_str:
                version = '12.0'
            else:
                version = '10.0'

        print(f"Auto-detected version: {version}")

    print(f"Model version: {version}")

    # Import and create model
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
        'knowledge_k': config.get('knowledge_k', 10),
        'dropout': config.get('dropout', 0.1),
    }

    if version.startswith('12') or version.startswith('13') or version.startswith('14'):
        model_kwargs['state_dim'] = config.get('state_dim', 64)
        if 'knowledge_rank' in config:
            model_kwargs['knowledge_rank'] = config['knowledge_rank']

    # v12.7, v12.8, v13 have top-k sparse routing
    if version in ['12.7', '12.8', '13.0'] or version.startswith('13'):
        model_kwargs['top_k_compress'] = config.get('top_k_compress', 8)
        model_kwargs['top_k_expand'] = config.get('top_k_expand', 4)

    # v14: FRTK neuron counts
    if version.startswith('14'):
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

    all_results = {}

    # 1. SVD Analysis
    if args.mode in ['all', 'svd']:
        print("\n" + "="*60)
        print("1. SVD/PCA ANALYSIS")
        print("="*60)

        svd_analyzer = NeuronSVDAnalyzer(model, device)
        svd_results = svd_analyzer.run_all()
        all_results['svd'] = svd_results

        svd_analyzer.visualize(svd_results, os.path.join(args.output_dir, 'svd_analysis.png'))

    # 2. Sentence Visualization
    if args.mode in ['all', 'sentence']:
        print("\n" + "="*60)
        print("2. SENTENCE VISUALIZATION")
        print("="*60)

        viz = SentenceVisualizer(model, tokenizer, device)
        analysis = viz.analyze_sentence(args.text)
        viz.visualize_sentence(analysis, os.path.join(args.output_dir, 'sentence_viz.png'))

        all_results['sentence'] = {
            'text': args.text,
            'tokens': analysis['tokens']
        }

    # Load data for semantic and catalog
    if args.mode in ['all', 'semantic', 'catalog']:
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

        # 3. Semantic Analysis
        if args.mode in ['all', 'semantic']:
            print("\n" + "="*60)
            print("3. SEMANTIC ANALYSIS")
            print("="*60)

            semantic = SemanticAnalyzer(model, tokenizer, device)
            semantic_results = semantic.analyze(dataloader, args.max_batches)
            all_results['semantic'] = semantic_results

            semantic.visualize(semantic_results, os.path.join(args.output_dir, 'semantic_heatmap.png'))

        # 4. Neuron Catalog
        if args.mode in ['all', 'catalog']:
            print("\n" + "="*60)
            print("4. NEURON CATALOG")
            print("="*60)

            catalog_builder = NeuronCatalog(model, tokenizer, device)
            catalog = catalog_builder.build(dataloader, args.max_batches)
            all_results['catalog'] = catalog

    # Save results
    results_path = os.path.join(args.output_dir, 'advanced_analysis.json')
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\nResults saved: {results_path}")

    print(f"\n{'='*60}")
    print("ADVANCED ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
