#!/usr/bin/env python3
"""
DAWN v16 Analysis Script
========================

v16 Split Feature R/V 모델 전용 분석 도구.

Features:
1. Neuron Usage - FR/FV/R/V/K 타입별 활성화 분석
2. Excitability - Langevin dynamics 상태
3. Gini Coefficient - 뉴런 활용 불균형
4. Neuron Embedding - 뉴런 임베딩 시각화 (t-SNE/PCA)
5. Word-Neuron Mapping - 단어별 뉴런 활성화
6. Weight Analysis - SVD/PCA 가중치 분석
7. Single Neuron Analysis - 단일 뉴런 심층 분석
8. Ablation Study - 뉴런 제거 실험
9. Sentence Visualization - 문장별 활성화 시각화
10. Neuron Similarity - 뉴런간 유사도 히트맵

Usage:
    python scripts/analyze_v16.py --checkpoint <path>
    python scripts/analyze_v16.py --checkpoint <path> --mode all
    python scripts/analyze_v16.py --checkpoint <path> --mode usage
    python scripts/analyze_v16.py --checkpoint <path> --mode embedding
    python scripts/analyze_v16.py --checkpoint <path> --mode neuron --neuron_id 5 --neuron_type feature_r
    python scripts/analyze_v16.py --checkpoint <path> --mode sentence --text "The cat sat on the mat."
    python scripts/analyze_v16.py --checkpoint <path> --mode ablation --val_data <path>
    python scripts/analyze_v16.py --checkpoint <path> --mode weight_svd
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ============================================================
# Utilities
# ============================================================

def get_underlying_model(model):
    """Get the underlying model from torch.compile wrapper"""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def gini_coefficient(x: torch.Tensor) -> float:
    """Calculate Gini coefficient (0=equal, 1=unequal)"""
    x = x.flatten().float()
    if x.sum() == 0:
        return 0.0
    x = x / x.sum()
    x_sorted = torch.sort(x)[0]
    n = len(x_sorted)
    gini = (2 * torch.sum((torch.arange(1, n+1, device=x.device).float()) * x_sorted) - (n + 1) * x_sorted.sum()) / (n * x_sorted.sum() + 1e-8)
    return gini.item()


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
    if token_lower in {'and', 'or', 'but', 'if', 'when', 'because'}:
        return 'CONJ'
    if token_lower.isdigit():
        return 'NUM'
    if token_lower.endswith('ing'):
        return 'VERB_ING'
    if token_lower.endswith('ed'):
        return 'VERB_ED'
    if token_lower.endswith('ly'):
        return 'ADV'
    return 'OTHER'


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


# ============================================================
# Model Loading
# ============================================================

def find_latest_checkpoint(path: str) -> str:
    """Find the latest checkpoint in a directory or return the path if it's a file"""
    from pathlib import Path

    path = Path(path)

    # If it's a file, return it directly
    if path.is_file():
        return str(path)

    # If it's a directory, find the latest .pt file
    if path.is_dir():
        pt_files = list(path.glob('*.pt'))

        if not pt_files:
            # Try subdirectories (e.g., checkpoints/run_xxx/)
            pt_files = list(path.glob('**/*.pt'))

        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {path}")

        # Sort by modification time (newest first)
        pt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Prefer 'best' or 'latest' checkpoints
        for f in pt_files:
            if 'best' in f.name.lower():
                print(f"Found best checkpoint: {f}")
                return str(f)

        for f in pt_files:
            if 'latest' in f.name.lower():
                print(f"Found latest checkpoint: {f}")
                return str(f)

        # Otherwise return the most recently modified
        print(f"Using most recent checkpoint: {pt_files[0]}")
        return str(pt_files[0])

    raise FileNotFoundError(f"Path not found: {path}")


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load v16 model from checkpoint (supports directory auto-discovery)"""
    from transformers import BertTokenizer
    from models import create_model_by_version

    # Auto-discover checkpoint if directory is given
    checkpoint_path = find_latest_checkpoint(checkpoint_path)
    print(f"Loading from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    # Detect version
    path_str = str(checkpoint_path).lower()
    if 'v16.1' in path_str or 'v16_1' in path_str:
        version = '16.1'
    elif 'v16' in path_str:
        version = '16.0'
    else:
        version = config.get('model_version', '16.0')

    print(f"Loading model version: {version}")
    print(f"Config: {config}")

    model = create_model_by_version(version, config)

    # Load state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return model, tokenizer, config


# ============================================================
# V16 Analyzer Class
# ============================================================

class V16Analyzer:
    """v16 모델 분석기 - 모든 분석 기능 통합"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.router = self._get_router()
        self.neurons = self._get_neurons()

    def _get_router(self):
        """Get neuron router"""
        if hasattr(self.model, 'global_routers'):
            return self.model.global_routers.neuron_router
        return None

    def _get_neurons(self):
        """Get shared neurons"""
        if hasattr(self.model, 'shared_neurons'):
            return self.model.shared_neurons
        return None

    # ==========================================================
    # 1. Usage Analysis
    # ==========================================================
    def analyze_usage(self) -> Dict:
        """Analyze neuron usage patterns from EMA buffers"""
        if self.router is None:
            return {'error': 'No router found'}

        results = {}
        threshold = 0.01

        for name, attr in [
            ('feature_r', 'usage_ema_feature_r'),
            ('feature_v', 'usage_ema_feature_v'),
            ('relational', 'usage_ema_relational'),
            ('value', 'usage_ema_value'),
            ('knowledge', 'usage_ema_knowledge'),
        ]:
            ema = getattr(self.router, attr)
            n_total = getattr(self.router, f'n_{name}')
            active = (ema > threshold).sum().item()
            results[name] = {
                'total': n_total,
                'active': int(active),
                'ratio': active / n_total,
                'gini': gini_coefficient(ema),
                'ema_stats': {
                    'min': ema.min().item(),
                    'max': ema.max().item(),
                    'mean': ema.mean().item(),
                    'std': ema.std().item(),
                }
            }

        return results

    # ==========================================================
    # 2. Excitability Analysis
    # ==========================================================
    def analyze_excitability(self) -> Dict:
        """Analyze excitability state"""
        if self.router is None:
            return {'error': 'No router found'}

        tau = self.router.tau
        weight = self.router.excitability_weight.item() if hasattr(self.router.excitability_weight, 'item') else self.router.excitability_weight

        results = {
            'tau': tau,
            'weight': weight,
            'langevin_alpha': getattr(self.router, 'langevin_alpha', 0),
            'langevin_beta': getattr(self.router, 'langevin_beta', 0),
        }

        # Per-type excitability
        for name in ['feature_r', 'feature_v', 'relational', 'value', 'knowledge']:
            ema = getattr(self.router, f'usage_ema_{name}')
            exc = torch.clamp(1.0 - ema / tau, min=0.0, max=1.0)
            results[f'{name}_excitability'] = {
                'min': exc.min().item(),
                'max': exc.max().item(),
                'mean': exc.mean().item(),
            }

        return results

    # ==========================================================
    # 3. Neuron Embedding Analysis (t-SNE/PCA)
    # ==========================================================
    def analyze_embeddings(self, output_dir: str = None) -> Dict:
        """Analyze neuron embeddings with t-SNE/PCA"""
        if self.router is None:
            return {'error': 'No router found'}

        emb = self.router.neuron_emb.detach().cpu().numpy()

        # Type labels
        labels = []
        labels.extend(['FR'] * self.router.n_feature_r)
        labels.extend(['FV'] * self.router.n_feature_v)
        labels.extend(['R'] * self.router.n_relational)
        labels.extend(['V'] * self.router.n_value)
        labels.extend(['K'] * self.router.n_knowledge)

        results = {
            'total_neurons': len(labels),
            'embedding_dim': emb.shape[1],
            'type_counts': {
                'FR': self.router.n_feature_r,
                'FV': self.router.n_feature_v,
                'R': self.router.n_relational,
                'V': self.router.n_value,
                'K': self.router.n_knowledge,
            }
        }

        # Cosine similarity between types
        type_centroids = {}
        for t in ['FR', 'FV', 'R', 'V', 'K']:
            mask = [l == t for l in labels]
            type_centroids[t] = emb[mask].mean(axis=0)

        sim_matrix = {}
        for t1 in type_centroids:
            for t2 in type_centroids:
                if t1 <= t2:
                    c1 = type_centroids[t1]
                    c2 = type_centroids[t2]
                    sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
                    sim_matrix[f'{t1}-{t2}'] = float(sim)

        results['type_similarity'] = sim_matrix

        # Visualization
        if HAS_SKLEARN and HAS_MATPLOTLIB and output_dir:
            os.makedirs(output_dir, exist_ok=True)

            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb)-1))
            emb_2d = tsne.fit_transform(emb)

            pca = PCA(n_components=2)
            emb_pca = pca.fit_transform(emb)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            colors = {'FR': 'red', 'FV': 'orange', 'R': 'blue', 'V': 'green', 'K': 'purple'}

            for ax, data, title in [(axes[0], emb_2d, 't-SNE'), (axes[1], emb_pca, 'PCA')]:
                for t in colors:
                    mask = np.array([l == t for l in labels])
                    ax.scatter(data[mask, 0], data[mask, 1], c=colors[t], label=t, alpha=0.6, s=20)
                ax.set_title(f'Neuron Embeddings ({title})')
                ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'neuron_embeddings.png'), dpi=150)
            plt.close()
            results['visualization'] = os.path.join(output_dir, 'neuron_embeddings.png')

        return results

    # ==========================================================
    # 4. Weight SVD/PCA Analysis
    # ==========================================================
    def analyze_weight_svd(self, output_dir: str = None) -> Dict:
        """Analyze neuron weight matrices with SVD"""
        if self.neurons is None:
            return {'error': 'No shared neurons found'}

        results = {}

        # Analyze each neuron type's weight matrix
        weight_configs = [
            ('feature_r', 'feature_r_neurons'),  # [n, d_model, rank]
            ('feature_v', 'feature_v_neurons'),
            ('relational', 'relational_neurons'),  # [n, rank, d_model]
            ('value', 'value_neurons'),
        ]

        for name, attr in weight_configs:
            if not hasattr(self.neurons, attr):
                continue

            W = getattr(self.neurons, attr).detach().cpu()  # [n, ...]
            n_neurons = W.shape[0]

            # Flatten each neuron's weight to a vector
            W_flat = W.view(n_neurons, -1).numpy()

            # SVD on the neuron weight matrix
            U, S, Vh = np.linalg.svd(W_flat, full_matrices=False)

            # Effective rank (using singular value entropy)
            S_norm = S / (S.sum() + 1e-8)
            entropy = -np.sum(S_norm * np.log(S_norm + 1e-8))
            effective_rank = np.exp(entropy)

            # Explained variance ratio
            var_ratio = (S ** 2) / (np.sum(S ** 2) + 1e-8)
            cumvar = np.cumsum(var_ratio)

            results[name] = {
                'n_neurons': n_neurons,
                'weight_shape': list(W.shape),
                'effective_rank': float(effective_rank),
                'top_5_singular_values': S[:5].tolist(),
                'var_explained_by_top5': float(cumvar[4]) if len(cumvar) > 4 else float(cumvar[-1]),
                'var_explained_by_top10': float(cumvar[9]) if len(cumvar) > 9 else float(cumvar[-1]),
            }

        # Visualization
        if HAS_MATPLOTLIB and output_dir:
            os.makedirs(output_dir, exist_ok=True)

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for ax, (name, attr) in zip(axes, weight_configs):
                if name not in results:
                    continue
                W = getattr(self.neurons, attr).detach().cpu()
                W_flat = W.view(W.shape[0], -1).numpy()
                _, S, _ = np.linalg.svd(W_flat, full_matrices=False)

                ax.plot(S[:min(50, len(S))], 'b-o', markersize=3)
                ax.set_title(f'{name} Singular Values')
                ax.set_xlabel('Index')
                ax.set_ylabel('Singular Value')
                ax.set_yscale('log')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'weight_svd.png'), dpi=150)
            plt.close()
            results['visualization'] = os.path.join(output_dir, 'weight_svd.png')

        return results

    # ==========================================================
    # 5. Single Neuron Analysis
    # ==========================================================
    def analyze_single_neuron(self, neuron_id: int, neuron_type: str = 'feature_r', top_k: int = 50) -> Dict:
        """Analyze what a specific neuron encodes"""
        if self.neurons is None:
            return {'error': 'No shared neurons found'}

        results = {'neuron_id': neuron_id, 'neuron_type': neuron_type}

        # Get weight matrix for this neuron type
        attr_map = {
            'feature_r': 'feature_r_neurons',
            'feature_v': 'feature_v_neurons',
            'relational': 'relational_neurons',
            'value': 'value_neurons',
        }

        if neuron_type not in attr_map:
            return {'error': f'Unknown neuron type: {neuron_type}'}

        W_all = getattr(self.neurons, attr_map[neuron_type]).data
        n_neurons = W_all.shape[0]

        if neuron_id >= n_neurons:
            return {'error': f'Neuron ID {neuron_id} out of range (max: {n_neurons-1})'}

        W_n = W_all[neuron_id]  # [d_model, rank] or [rank, d_model]

        # Get token embeddings
        embed = self.model.token_emb.weight.data  # [vocab, d_model]

        # Compute activation strength for each token
        with torch.no_grad():
            if neuron_type in ['feature_r', 'feature_v']:
                # W: [d_model, rank], embed: [vocab, d_model]
                h = embed @ W_n  # [vocab, rank]
            else:
                # W: [rank, d_model], need to transpose
                h = embed @ W_n.T  # [vocab, rank]

            activation_strength = h.norm(dim=1)  # [vocab]

        # Filter special tokens
        all_tokens = self.tokenizer.convert_ids_to_tokens(list(range(embed.shape[0])))
        valid_mask = torch.tensor([
            not (tok.startswith('[unused') or tok.startswith('##unused') or tok.startswith('['))
            for tok in all_tokens
        ], device=activation_strength.device)

        valid_indices = valid_mask.nonzero().squeeze(-1)
        valid_strengths = activation_strength[valid_indices]
        sorted_order = valid_strengths.argsort(descending=True)

        # Top-k tokens
        top_tokens = []
        for i in range(min(top_k, len(sorted_order))):
            idx = valid_indices[sorted_order[i]].item()
            token = all_tokens[idx]
            strength = activation_strength[idx].item()
            pos = simple_pos_tag(token)
            top_tokens.append({
                'token': token,
                'strength': strength,
                'pos': pos
            })

        results['top_tokens'] = top_tokens

        # POS distribution
        pos_counts = Counter([t['pos'] for t in top_tokens])
        results['pos_distribution'] = dict(pos_counts)

        # Neuron stats
        results['weight_stats'] = {
            'mean': W_n.mean().item(),
            'std': W_n.std().item(),
            'norm': W_n.norm().item(),
        }

        # Usage EMA for this neuron
        ema_attr = f'usage_ema_{neuron_type}'
        if self.router and hasattr(self.router, ema_attr):
            ema = getattr(self.router, ema_attr)
            results['usage_ema'] = ema[neuron_id].item()
            results['excitability'] = max(0, 1 - ema[neuron_id].item() / self.router.tau)

        return results

    # ==========================================================
    # 6. Neuron Similarity Heatmap
    # ==========================================================
    def analyze_similarity(self, output_dir: str = None) -> Dict:
        """Analyze similarity between neurons within each type"""
        if self.router is None:
            return {'error': 'No router found'}

        results = {}
        emb = self.router.neuron_emb.detach()  # [total, d_space]

        # Boundaries
        boundaries = {
            'FR': (0, self.router.n_feature_r),
            'FV': (self.router.n_feature_r, self.router.n_feature_r + self.router.n_feature_v),
            'R': (self.router.n_feature_r + self.router.n_feature_v,
                  self.router.n_feature_r + self.router.n_feature_v + self.router.n_relational),
            'V': (self.router.n_feature_r + self.router.n_feature_v + self.router.n_relational,
                  self.router.n_feature_r + self.router.n_feature_v + self.router.n_relational + self.router.n_value),
            'K': (self.router.n_feature_r + self.router.n_feature_v + self.router.n_relational + self.router.n_value,
                  self.router.total_neurons),
        }

        for name, (start, end) in boundaries.items():
            type_emb = emb[start:end]
            type_emb_norm = F.normalize(type_emb, dim=-1)
            sim_matrix = (type_emb_norm @ type_emb_norm.T).cpu().numpy()

            # Stats (excluding diagonal)
            n = sim_matrix.shape[0]
            off_diag = sim_matrix[~np.eye(n, dtype=bool)]

            results[name] = {
                'n_neurons': n,
                'avg_similarity': float(off_diag.mean()),
                'max_similarity': float(off_diag.max()),
                'min_similarity': float(off_diag.min()),
                'std_similarity': float(off_diag.std()),
            }

        # Visualization
        if HAS_MATPLOTLIB and HAS_SEABORN and output_dir:
            os.makedirs(output_dir, exist_ok=True)

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for ax, (name, (start, end)) in zip(axes, boundaries.items()):
                type_emb = emb[start:end]
                type_emb_norm = F.normalize(type_emb, dim=-1)
                sim_matrix = (type_emb_norm @ type_emb_norm.T).cpu().numpy()

                sns.heatmap(sim_matrix, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1)
                ax.set_title(f'{name} Neuron Similarity')

            axes[-1].axis('off')  # Hide last subplot

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'neuron_similarity.png'), dpi=150)
            plt.close()
            results['visualization'] = os.path.join(output_dir, 'neuron_similarity.png')

        return results

    # ==========================================================
    # 7. Active Neuron Diversity (가장 중요!)
    # ==========================================================
    def analyze_selection_diversity(self, dataloader, n_batches: int = 100) -> Dict:
        """
        Selection Diversity - 배치별 실제 선택 다양성 측정
        EMA 기반이 아닌 실제 forward pass에서의 선택 패턴 분석
        """
        if self.router is None:
            return {'error': 'No router found'}

        # Track selected neurons
        union_selected = {
            'feature_r': set(),
            'feature_v': set(),
            'relational': set(),
            'value': set(),
        }
        per_batch_counts = {
            'feature_r': [],
            'feature_v': [],
            'relational': [],
            'value': [],
        }

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc='Selection Diversity', total=n_batches)):
                if i >= n_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(self.device)

                try:
                    outputs = self.model(input_ids, attention_mask=attention_mask, return_routing_info=True)
                    routing_info = outputs.get('routing_info', {})
                except Exception as e:
                    continue

                # Track selections for each neuron type
                for ntype, weight_key in [
                    ('feature_r', 'feature_r_weights'),
                    ('feature_v', 'feature_v_weights'),
                    ('relational', 'relational_weights_Q'),
                    ('value', 'value_weights'),
                ]:
                    if weight_key in routing_info:
                        weights = routing_info[weight_key]
                        # weights: [B, S, N] - batch, seq, neurons
                        if weights.dim() == 3:
                            # Neurons selected if weight > 0 for any token in batch
                            selected = (weights > 0).any(dim=0).any(dim=0).cpu()  # [N]
                            union_selected[ntype].update(selected.nonzero().flatten().tolist())
                            per_batch_counts[ntype].append(selected.sum().item())
                        elif weights.dim() == 2:
                            selected = (weights > 0).any(dim=0).cpu()
                            union_selected[ntype].update(selected.nonzero().flatten().tolist())
                            per_batch_counts[ntype].append(selected.sum().item())

        # Calculate results
        results = {}
        n_totals = {
            'feature_r': self.router.n_feature_r,
            'feature_v': self.router.n_feature_v,
            'relational': self.router.n_relational,
            'value': self.router.n_value,
        }

        for ntype in union_selected:
            n_total = n_totals[ntype]
            union_count = len(union_selected[ntype])
            batch_counts = per_batch_counts[ntype]

            if len(batch_counts) > 0:
                per_batch_avg = np.mean(batch_counts)
                per_batch_std = np.std(batch_counts)
                per_batch_min = np.min(batch_counts)
                per_batch_max = np.max(batch_counts)
            else:
                per_batch_avg = per_batch_std = per_batch_min = per_batch_max = 0

            results[ntype] = {
                'n_total': n_total,
                'per_batch_avg': float(per_batch_avg),
                'per_batch_std': float(per_batch_std),
                'per_batch_min': int(per_batch_min),
                'per_batch_max': int(per_batch_max),
                'union_count': union_count,
                'union_coverage': float(union_count / n_total) if n_total > 0 else 0,
                'diversity_ratio': float(union_count / per_batch_avg) if per_batch_avg > 0 else 0,
            }

        # Summary
        total_union = sum(len(union_selected[k]) for k in union_selected)
        total_neurons = sum(n_totals.values())
        results['summary'] = {
            'n_batches_processed': min(n_batches, len(per_batch_counts['feature_r'])),
            'total_union_coverage': float(total_union / total_neurons) if total_neurons > 0 else 0,
            'interpretation': (
                'High diversity_ratio (>2) = 많은 뉴런이 batch마다 다르게 선택됨\n'
                'Low diversity_ratio (~1) = 항상 같은 뉴런만 선택됨'
            ),
        }

        return results

    def analyze_diversity(self) -> Dict:
        """
        Active Neuron Diversity 분석
        - 활성 뉴런들이 얼마나 다양하게 사용되는지
        - Entropy, Effective Count, Coverage 등
        """
        if self.router is None:
            return {'error': 'No router found'}

        results = {}
        threshold = 0.01

        for name in ['feature_r', 'feature_v', 'relational', 'value', 'knowledge']:
            ema = getattr(self.router, f'usage_ema_{name}')
            n_total = getattr(self.router, f'n_{name}')

            # Active neurons
            active_mask = ema > threshold
            n_active = active_mask.sum().item()
            active_ema = ema[active_mask]

            if n_active == 0:
                results[name] = {
                    'n_total': n_total,
                    'n_active': 0,
                    'diversity_entropy': 0.0,
                    'effective_count': 0.0,
                    'coverage': 0.0,
                    'concentration_ratio': 1.0,
                    'top5_share': 1.0,
                }
                continue

            # Normalize to distribution
            p = active_ema / (active_ema.sum() + 1e-8)

            # Shannon entropy (higher = more diverse)
            entropy = -torch.sum(p * torch.log(p + 1e-8)).item()
            max_entropy = np.log(n_active)  # Maximum possible entropy

            # Effective count (exponential of entropy)
            effective_count = np.exp(entropy)

            # Normalized entropy (0-1 scale)
            normalized_entropy = entropy / (max_entropy + 1e-8) if max_entropy > 0 else 0

            # Coverage: what fraction of total neurons are active
            coverage = n_active / n_total

            # Concentration ratio: top-5 neurons' share
            top5_ema = torch.topk(ema, min(5, len(ema))).values
            top5_share = top5_ema.sum().item() / (ema.sum().item() + 1e-8)

            # Gini coefficient (0=equal, 1=concentrated)
            gini = gini_coefficient(ema)

            results[name] = {
                'n_total': n_total,
                'n_active': int(n_active),
                'diversity_entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'effective_count': float(effective_count),
                'coverage': float(coverage),
                'concentration_ratio': float(1 - normalized_entropy),
                'top5_share': float(top5_share),
                'gini': float(gini),
            }

        # Overall diversity score (weighted average)
        weights = {'feature_r': 2, 'feature_v': 2, 'relational': 1.5, 'value': 1.5, 'knowledge': 1}
        total_weight = sum(weights.values())
        overall_diversity = sum(
            results[name].get('normalized_entropy', 0) * weights[name]
            for name in weights if name in results
        ) / total_weight

        results['overall'] = {
            'diversity_score': float(overall_diversity),
            'health': 'good' if overall_diversity > 0.7 else 'warning' if overall_diversity > 0.4 else 'critical'
        }

        return results

    # ==========================================================
    # 8. Dead Neuron Potential (축소 결정용)
    # ==========================================================
    def analyze_dead_neurons(self, output_dir: str = None) -> Dict:
        """
        Dead Neuron 분석 - 축소 가능성 판단
        - Dead: EMA < threshold, 거의 사용 안됨
        - Dying: EMA가 decay 중
        - Revivable: Dead지만 excitability 높음
        """
        if self.router is None:
            return {'error': 'No router found'}

        results = {}
        threshold = 0.01
        dying_threshold = 0.05  # 죽어가는 중
        tau = self.router.tau

        for name in ['feature_r', 'feature_v', 'relational', 'value', 'knowledge']:
            ema = getattr(self.router, f'usage_ema_{name}')
            n_total = getattr(self.router, f'n_{name}')

            # Categories
            dead_mask = ema < threshold
            dying_mask = (ema >= threshold) & (ema < dying_threshold)
            active_mask = ema >= dying_threshold

            n_dead = dead_mask.sum().item()
            n_dying = dying_mask.sum().item()
            n_active = active_mask.sum().item()

            # Excitability for dead neurons
            exc = torch.clamp(1.0 - ema / tau, min=0.0, max=1.0)
            dead_exc = exc[dead_mask]

            # Revivable: dead but high excitability
            revivable_mask = dead_mask & (exc > 0.8)
            n_revivable = revivable_mask.sum().item()

            # Candidates for removal (dead + low excitability)
            removable_mask = dead_mask & (exc < 0.3)
            n_removable = removable_mask.sum().item()

            results[name] = {
                'n_total': n_total,
                'n_dead': int(n_dead),
                'n_dying': int(n_dying),
                'n_active': int(n_active),
                'n_revivable': int(n_revivable),
                'n_removable': int(n_removable),
                'dead_ratio': float(n_dead / n_total),
                'removable_ratio': float(n_removable / n_total),
                'dead_neuron_ids': dead_mask.nonzero().squeeze(-1).tolist() if n_dead > 0 else [],
                'removable_neuron_ids': removable_mask.nonzero().squeeze(-1).tolist() if n_removable > 0 else [],
            }

        # Shrink recommendations
        total_removable = sum(r['n_removable'] for r in results.values() if isinstance(r, dict))
        total_neurons = sum(r['n_total'] for r in results.values() if isinstance(r, dict))

        results['shrink_recommendation'] = {
            'total_removable': total_removable,
            'shrink_ratio': float(total_removable / total_neurons),
            'recommended_action': 'shrink' if total_removable > total_neurons * 0.2 else 'keep',
            'per_type': {
                name: {
                    'current': results[name]['n_total'],
                    'recommended': results[name]['n_total'] - results[name]['n_removable'],
                    'remove': results[name]['n_removable'],
                }
                for name in ['feature_r', 'feature_v', 'relational', 'value', 'knowledge']
                if name in results
            }
        }

        # Visualization
        if HAS_MATPLOTLIB and output_dir:
            os.makedirs(output_dir, exist_ok=True)

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            colors = ['green', 'yellow', 'red']
            labels = ['Active', 'Dying', 'Dead']

            for ax, name in zip(axes[:5], ['feature_r', 'feature_v', 'relational', 'value', 'knowledge']):
                data = results[name]
                sizes = [data['n_active'], data['n_dying'], data['n_dead']]
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title(f'{name}\n(removable: {data["n_removable"]})')

            # Summary bar
            ax = axes[5]
            names = ['FR', 'FV', 'R', 'V', 'K']
            removable = [results[n]['n_removable'] for n in ['feature_r', 'feature_v', 'relational', 'value', 'knowledge']]
            ax.bar(names, removable, color='red', alpha=0.7)
            ax.set_title('Removable Neurons')
            ax.set_ylabel('Count')

            plt.tight_layout()
            path = os.path.join(output_dir, 'dead_neurons.png')
            plt.savefig(path, dpi=150)
            plt.close()
            results['visualization'] = path

        return results

    # ==========================================================
    # 9. Neuron Clustering
    # ==========================================================
    def analyze_clustering(self, n_clusters: int = 5, output_dir: str = None) -> Dict:
        """
        Neuron Clustering - 뉴런들의 기능적 그룹 분석
        """
        if self.router is None or not HAS_SKLEARN:
            return {'error': 'No router found or sklearn not available'}

        results = {}
        emb = self.router.neuron_emb.detach().cpu().numpy()

        # Type boundaries
        boundaries = {
            'FR': (0, self.router.n_feature_r),
            'FV': (self.router.n_feature_r, self.router.n_feature_r + self.router.n_feature_v),
            'R': (self.router.n_feature_r + self.router.n_feature_v,
                  self.router.n_feature_r + self.router.n_feature_v + self.router.n_relational),
            'V': (self.router.n_feature_r + self.router.n_feature_v + self.router.n_relational,
                  self.router.n_feature_r + self.router.n_feature_v + self.router.n_relational + self.router.n_value),
            'K': (self.router.n_feature_r + self.router.n_feature_v + self.router.n_relational + self.router.n_value,
                  self.router.total_neurons),
        }

        for name, (start, end) in boundaries.items():
            type_emb = emb[start:end]
            n_neurons = type_emb.shape[0]

            if n_neurons < n_clusters:
                results[name] = {'error': f'Not enough neurons for {n_clusters} clusters'}
                continue

            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(type_emb)

            # Cluster stats
            cluster_stats = []
            for c in range(n_clusters):
                cluster_mask = labels == c
                cluster_size = cluster_mask.sum()

                # Get usage EMA for this type
                ema_attr = f'usage_ema_{name.lower()}' if name != 'FR' else 'usage_ema_feature_r'
                if name == 'FV':
                    ema_attr = 'usage_ema_feature_v'
                elif name == 'R':
                    ema_attr = 'usage_ema_relational'
                elif name == 'V':
                    ema_attr = 'usage_ema_value'
                elif name == 'K':
                    ema_attr = 'usage_ema_knowledge'

                ema = getattr(self.router, ema_attr).cpu().numpy()
                cluster_ema = ema[cluster_mask]

                cluster_stats.append({
                    'cluster_id': c,
                    'size': int(cluster_size),
                    'avg_usage': float(cluster_ema.mean()),
                    'active_count': int((cluster_ema > 0.01).sum()),
                    'neuron_ids': np.where(cluster_mask)[0].tolist(),
                })

            # Sort by average usage
            cluster_stats.sort(key=lambda x: x['avg_usage'], reverse=True)

            results[name] = {
                'n_clusters': n_clusters,
                'clusters': cluster_stats,
                'inertia': float(kmeans.inertia_),
            }

        # Visualization
        if HAS_MATPLOTLIB and output_dir:
            os.makedirs(output_dir, exist_ok=True)

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for ax, (name, (start, end)) in zip(axes[:5], boundaries.items()):
                if name not in results or 'error' in results[name]:
                    ax.axis('off')
                    continue

                type_emb = emb[start:end]

                # PCA for visualization
                pca = PCA(n_components=2)
                emb_2d = pca.fit_transform(type_emb)

                # Get cluster labels
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(type_emb)

                scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
                ax.set_title(f'{name} Clusters')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')

            axes[5].axis('off')

            plt.tight_layout()
            path = os.path.join(output_dir, 'neuron_clusters.png')
            plt.savefig(path, dpi=150)
            plt.close()
            results['visualization'] = path

        return results

    # ==========================================================
    # 10. Token Trajectory
    # ==========================================================
    def analyze_token_trajectory(self, dataloader, max_batches: int = 20) -> Dict:
        """
        Token Trajectory - 토큰이 어떤 뉴런 경로를 따르는지 분석
        POS별로 뉴런 사용 패턴 분석
        """
        if self.router is None:
            return {'error': 'No router found'}

        # POS -> neuron usage accumulator
        pos_neuron_usage = {
            'feature_r': defaultdict(lambda: defaultdict(float)),
            'feature_v': defaultdict(lambda: defaultdict(float)),
            'relational': defaultdict(lambda: defaultdict(float)),
            'value': defaultdict(lambda: defaultdict(float)),
        }
        pos_counts = Counter()

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc='Token Trajectory', total=max_batches)):
                if batch_idx >= max_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                try:
                    outputs = self.model(input_ids, attention_mask=attention_mask, return_routing_info=True)
                    routing_info = outputs.get('routing_info', {})
                except:
                    continue

                # Process each token
                for b in range(input_ids.shape[0]):
                    for s in range(input_ids.shape[1]):
                        if attention_mask[b, s] == 0:
                            continue

                        token_id = input_ids[b, s].item()
                        token = self.tokenizer.decode([token_id])
                        pos = simple_pos_tag(token)
                        pos_counts[pos] += 1

                        # Accumulate neuron activations per POS
                        for ntype, key in [
                            ('feature_r', 'feature_r_idx'),
                            ('feature_v', 'feature_v_idx'),
                            ('relational', 'relational_idx_Q'),
                            ('value', 'value_idx'),
                        ]:
                            if key in routing_info:
                                idx = routing_info[key]
                                if idx.dim() >= 2:
                                    neuron_ids = idx[b, s].cpu().tolist()
                                    if isinstance(neuron_ids, int):
                                        neuron_ids = [neuron_ids]
                                    for nid in neuron_ids:
                                        pos_neuron_usage[ntype][pos][nid] += 1

        # Aggregate results
        results = {'pos_counts': dict(pos_counts)}

        for ntype in pos_neuron_usage:
            results[ntype] = {}
            for pos in pos_neuron_usage[ntype]:
                neuron_counts = pos_neuron_usage[ntype][pos]
                if not neuron_counts:
                    continue

                # Top neurons for this POS
                sorted_neurons = sorted(neuron_counts.items(), key=lambda x: x[1], reverse=True)
                top_neurons = sorted_neurons[:10]

                # Entropy of neuron usage
                counts = np.array(list(neuron_counts.values()))
                p = counts / (counts.sum() + 1e-8)
                entropy = -np.sum(p * np.log(p + 1e-8))

                results[ntype][pos] = {
                    'top_neurons': [(int(n), float(c)) for n, c in top_neurons],
                    'unique_neurons': len(neuron_counts),
                    'entropy': float(entropy),
                }

        return results

    # ==========================================================
    # 11. Probing Classifier
    # ==========================================================
    def run_probing(self, dataloader, max_batches: int = 50) -> Dict:
        """
        Probing Classifier - 뉴런이 어떤 정보를 인코딩하는지
        POS 분류 태스크로 뉴런 활성화 패턴 분석
        """
        if not HAS_SKLEARN:
            return {'error': 'sklearn not available'}

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report

        # Collect activations and labels
        X_data = {
            'feature_r': [],
            'feature_v': [],
            'relational': [],
            'value': [],
        }
        y_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc='Probing', total=max_batches)):
                if batch_idx >= max_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                try:
                    outputs = self.model(input_ids, attention_mask=attention_mask, return_routing_info=True)
                    routing_info = outputs.get('routing_info', {})
                except:
                    continue

                # Get routing weights
                for ntype, key in [
                    ('feature_r', 'feature_r_weights'),
                    ('feature_v', 'feature_v_weights'),
                    ('relational', 'relational_weights_Q'),
                    ('value', 'value_weights'),
                ]:
                    if key in routing_info:
                        w = routing_info[key]
                        if w.dim() == 3:  # [B, S, N]
                            for b in range(w.shape[0]):
                                for s in range(w.shape[1]):
                                    if attention_mask[b, s] == 0:
                                        continue
                                    X_data[ntype].append(w[b, s].cpu().numpy())

                # Labels (POS tags)
                for b in range(input_ids.shape[0]):
                    for s in range(input_ids.shape[1]):
                        if attention_mask[b, s] == 0:
                            continue
                        token = self.tokenizer.decode([input_ids[b, s].item()])
                        pos = simple_pos_tag(token)
                        y_labels.append(pos)

        results = {}

        # Train probing classifier for each neuron type
        for ntype in X_data:
            X = np.array(X_data[ntype])
            y = np.array(y_labels[:len(X)])

            if len(X) < 100 or len(np.unique(y)) < 2:
                results[ntype] = {'error': 'Not enough data'}
                continue

            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train
            clf = LogisticRegression(max_iter=500, random_state=42, multi_class='ovr')
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                # Per-class accuracy
                unique_classes = np.unique(y_test)
                per_class_acc = {}
                for cls in unique_classes:
                    mask = y_test == cls
                    if mask.sum() > 0:
                        per_class_acc[cls] = float((y_pred[mask] == cls).mean())

                results[ntype] = {
                    'accuracy': float(acc),
                    'n_samples': len(X),
                    'n_classes': len(unique_classes),
                    'per_class_accuracy': per_class_acc,
                }
            except Exception as e:
                results[ntype] = {'error': str(e)}

        return results

    # ==========================================================
    # 8. Ablation Study
    # ==========================================================
    def run_ablation(self, dataloader, max_batches: int = 50) -> Dict:
        """Run ablation study to measure neuron importance"""
        from copy import deepcopy

        results = {}

        # Get baseline loss
        baseline_loss = self._compute_loss(dataloader, max_batches)
        results['baseline_loss'] = baseline_loss

        # Ablate each neuron type
        neuron_types = ['feature_r', 'feature_v', 'relational', 'value']

        for neuron_type in neuron_types:
            attr_map = {
                'feature_r': 'feature_r_neurons',
                'feature_v': 'feature_v_neurons',
                'relational': 'relational_neurons',
                'value': 'value_neurons',
            }

            if not hasattr(self.neurons, attr_map[neuron_type]):
                continue

            W = getattr(self.neurons, attr_map[neuron_type])
            n_neurons = W.shape[0]
            original_W = W.data.clone()

            # Ablate top-k used neurons
            ema = getattr(self.router, f'usage_ema_{neuron_type}')
            top_indices = ema.argsort(descending=True)[:5]  # Top 5 most used

            ablation_results = []
            for idx in top_indices:
                idx = idx.item()
                # Zero out this neuron
                W.data[idx] = 0

                ablated_loss = self._compute_loss(dataloader, max_batches)
                loss_delta = ablated_loss - baseline_loss

                ablation_results.append({
                    'neuron_id': idx,
                    'usage_ema': ema[idx].item(),
                    'ablated_loss': ablated_loss,
                    'loss_delta': loss_delta,
                    'importance': loss_delta / (baseline_loss + 1e-8),
                })

                # Restore
                W.data[idx] = original_W[idx]

            results[neuron_type] = ablation_results

        return results

    def _compute_loss(self, dataloader, max_batches: int) -> float:
        """Compute average loss on dataloader"""
        self.model.eval()
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch.get('labels', input_ids).to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    # ==========================================================
    # 9. Usage Histogram Visualization
    # ==========================================================
    def visualize_usage(self, output_dir: str):
        """Create usage histogram plots"""
        if not HAS_MATPLOTLIB:
            return {'error': 'matplotlib not available'}

        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        data = [
            ('Feature R', self.router.usage_ema_feature_r, 'red'),
            ('Feature V', self.router.usage_ema_feature_v, 'orange'),
            ('Relational', self.router.usage_ema_relational, 'blue'),
            ('Value', self.router.usage_ema_value, 'green'),
            ('Knowledge', self.router.usage_ema_knowledge, 'purple'),
        ]

        for ax, (name, ema, color) in zip(axes, data):
            values = ema.detach().cpu().numpy()
            ax.hist(values, bins=50, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(x=0.01, color='red', linestyle='--', label='threshold=0.01')
            ax.set_title(f'{name} Usage EMA')
            ax.set_xlabel('EMA Value')
            ax.set_ylabel('Count')

            active = (values > 0.01).sum()
            total = len(values)
            ax.text(0.95, 0.95, f'Active: {active}/{total}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=10)

        # Summary bar chart
        ax = axes[5]
        names = [d[0] for d in data]
        active_ratios = [(d[1] > 0.01).float().mean().item() for d in data]
        colors = [d[2] for d in data]
        ax.bar(names, active_ratios, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Active Neuron Ratio by Type')
        ax.set_ylabel('Active Ratio')
        ax.set_ylim(0, 1)

        plt.tight_layout()
        path = os.path.join(output_dir, 'usage_histogram.png')
        plt.savefig(path, dpi=150)
        plt.close()

        return {'visualization': path}

    # ==========================================================
    # 10. Full Report
    # ==========================================================
    def generate_report(self, output_dir: str = None) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'usage': self.analyze_usage(),
            'excitability': self.analyze_excitability(),
        }

        if output_dir:
            report['embeddings'] = self.analyze_embeddings(output_dir)
            report['weight_svd'] = self.analyze_weight_svd(output_dir)
            report['similarity'] = self.analyze_similarity(output_dir)
            report['usage_viz'] = self.visualize_usage(output_dir)

        return report


# ============================================================
# Print Functions
# ============================================================

def print_usage_summary(usage: Dict):
    print("\n" + "="*60)
    print("NEURON USAGE SUMMARY")
    print("="*60)
    print(f"{'Type':<12} {'Active':>8} {'Total':>8} {'Ratio':>8} {'Gini':>8} {'Mean':>10}")
    print("-"*60)
    for key in ['feature_r', 'feature_v', 'relational', 'value', 'knowledge']:
        d = usage[key]
        print(f"{key:<12} {d['active']:>8} {d['total']:>8} {d['ratio']:>8.2%} {d['gini']:>8.2f} {d['ema_stats']['mean']:>10.4f}")


def print_excitability_summary(exc: Dict):
    print("\n" + "="*60)
    print("EXCITABILITY STATE")
    print("="*60)
    print(f"tau: {exc['tau']:.2f}")
    print(f"weight: {exc['weight']:.4f}")
    print(f"langevin_alpha: {exc.get('langevin_alpha', 'N/A')}")
    print(f"langevin_beta: {exc.get('langevin_beta', 'N/A')}")
    print()
    print(f"{'Type':<12} {'Min':>8} {'Mean':>8} {'Max':>8}")
    print("-"*40)
    for key in ['feature_r', 'feature_v', 'relational', 'value', 'knowledge']:
        d = exc[f'{key}_excitability']
        print(f"{key:<12} {d['min']:>8.2f} {d['mean']:>8.2f} {d['max']:>8.2f}")


def print_neuron_summary(neuron: Dict):
    print("\n" + "="*60)
    print(f"SINGLE NEURON ANALYSIS: {neuron['neuron_type']} #{neuron['neuron_id']}")
    print("="*60)
    print(f"Usage EMA: {neuron.get('usage_ema', 'N/A'):.4f}")
    print(f"Excitability: {neuron.get('excitability', 'N/A'):.4f}")
    print(f"\nWeight Stats:")
    for k, v in neuron.get('weight_stats', {}).items():
        print(f"  {k}: {v:.4f}")
    print(f"\nTop Tokens:")
    for t in neuron.get('top_tokens', [])[:20]:
        print(f"  {t['token']:<20} {t['strength']:>8.4f}  [{t['pos']}]")
    print(f"\nPOS Distribution:")
    for pos, count in neuron.get('pos_distribution', {}).items():
        print(f"  {pos}: {count}")


# ============================================================
# Main
# ============================================================

def print_diversity_summary(div: Dict):
    print("\n" + "="*60)
    print("ACTIVE NEURON DIVERSITY")
    print("="*60)
    print(f"{'Type':<12} {'Active':>8} {'EffCount':>10} {'Coverage':>10} {'Top5Share':>10} {'Entropy':>10}")
    print("-"*70)
    for key in ['feature_r', 'feature_v', 'relational', 'value', 'knowledge']:
        if key not in div:
            continue
        d = div[key]
        print(f"{key:<12} {d['n_active']:>8} {d['effective_count']:>10.1f} {d['coverage']:>10.2%} {d['top5_share']:>10.2%} {d['normalized_entropy']:>10.2f}")
    if 'overall' in div:
        print("-"*70)
        print(f"Overall Diversity Score: {div['overall']['diversity_score']:.2f} ({div['overall']['health']})")


def print_selection_diversity_summary(sel: Dict):
    print("\n" + "="*60)
    print("SELECTION DIVERSITY (실시간 Forward Pass)")
    print("="*60)
    print(f"{'Type':<12} {'PerBatch':>10} {'Union':>8} {'Total':>8} {'Coverage':>10} {'DivRatio':>10}")
    print("-"*70)
    for key in ['feature_r', 'feature_v', 'relational', 'value']:
        if key not in sel:
            continue
        d = sel[key]
        print(f"{key:<12} {d['per_batch_avg']:>10.1f} {d['union_count']:>8} {d['n_total']:>8} {d['union_coverage']:>10.1%} {d['diversity_ratio']:>10.2f}")
    if 'summary' in sel:
        print("-"*70)
        print(f"Total Union Coverage: {sel['summary']['total_union_coverage']:.1%}")
        print(f"Batches processed: {sel['summary']['n_batches_processed']}")


def print_dead_neurons_summary(dead: Dict):
    print("\n" + "="*60)
    print("DEAD NEURON ANALYSIS")
    print("="*60)
    print(f"{'Type':<12} {'Dead':>8} {'Dying':>8} {'Active':>8} {'Revivable':>10} {'Removable':>10}")
    print("-"*70)
    for key in ['feature_r', 'feature_v', 'relational', 'value', 'knowledge']:
        if key not in dead:
            continue
        d = dead[key]
        print(f"{key:<12} {d['n_dead']:>8} {d['n_dying']:>8} {d['n_active']:>8} {d['n_revivable']:>10} {d['n_removable']:>10}")
    if 'shrink_recommendation' in dead:
        rec = dead['shrink_recommendation']
        print("-"*70)
        print(f"Shrink Recommendation: {rec['recommended_action'].upper()}")
        print(f"Total Removable: {rec['total_removable']} ({rec['shrink_ratio']:.1%})")


def main():
    parser = argparse.ArgumentParser(description='DAWN v16 Analysis')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path (file or directory)')
    parser.add_argument('--val_data', help='Validation data path')
    parser.add_argument('--output_dir', default='./analysis_v16', help='Output directory')
    parser.add_argument('--mode', default='all',
                        choices=['all', 'usage', 'excitability', 'embedding', 'weight_svd',
                                 'neuron', 'similarity', 'diversity', 'selection_diversity',
                                 'dead_neurons', 'clustering', 'trajectory', 'probing', 'ablation'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_batches', type=int, default=100)

    # Single neuron analysis
    parser.add_argument('--neuron_id', type=int, default=0)
    parser.add_argument('--neuron_type', default='feature_r',
                        choices=['feature_r', 'feature_v', 'relational', 'value'])

    # Clustering
    parser.add_argument('--n_clusters', type=int, default=5)

    args = parser.parse_args()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, config = load_model(args.checkpoint, args.device)

    # Create analyzer
    analyzer = V16Analyzer(model, tokenizer, args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Run analysis based on mode
    if args.mode in ['all', 'usage']:
        usage = analyzer.analyze_usage()
        print_usage_summary(usage)
        with open(os.path.join(args.output_dir, 'usage.json'), 'w') as f:
            json.dump(usage, f, indent=2)

    if args.mode in ['all', 'excitability']:
        exc = analyzer.analyze_excitability()
        print_excitability_summary(exc)
        with open(os.path.join(args.output_dir, 'excitability.json'), 'w') as f:
            json.dump(exc, f, indent=2)

    if args.mode in ['all', 'embedding']:
        emb = analyzer.analyze_embeddings(args.output_dir)
        print(f"\nEmbedding visualization saved to: {emb.get('visualization', 'N/A')}")
        with open(os.path.join(args.output_dir, 'embeddings.json'), 'w') as f:
            json.dump(emb, f, indent=2)

    if args.mode in ['all', 'weight_svd']:
        svd = analyzer.analyze_weight_svd(args.output_dir)
        print("\n" + "="*60)
        print("WEIGHT SVD ANALYSIS")
        print("="*60)
        for name, data in svd.items():
            if name == 'visualization':
                continue
            print(f"\n{name}:")
            print(f"  Effective rank: {data['effective_rank']:.2f}")
            print(f"  Var explained by top 5: {data['var_explained_by_top5']:.2%}")
        with open(os.path.join(args.output_dir, 'weight_svd.json'), 'w') as f:
            json.dump(convert_to_serializable(svd), f, indent=2)

    if args.mode in ['all', 'similarity']:
        sim = analyzer.analyze_similarity(args.output_dir)
        print("\n" + "="*60)
        print("NEURON SIMILARITY")
        print("="*60)
        for name, data in sim.items():
            if name == 'visualization':
                continue
            print(f"{name}: avg={data['avg_similarity']:.3f}, max={data['max_similarity']:.3f}")
        with open(os.path.join(args.output_dir, 'similarity.json'), 'w') as f:
            json.dump(sim, f, indent=2)

    if args.mode == 'neuron':
        neuron = analyzer.analyze_single_neuron(args.neuron_id, args.neuron_type)
        print_neuron_summary(neuron)
        with open(os.path.join(args.output_dir, f'neuron_{args.neuron_type}_{args.neuron_id}.json'), 'w') as f:
            json.dump(convert_to_serializable(neuron), f, indent=2)

    # NEW: Diversity analysis (most important!)
    # Shows BOTH EMA-based diversity AND selection diversity
    if args.mode in ['all', 'diversity']:
        # 1. EMA-based diversity
        div = analyzer.analyze_diversity()
        print_diversity_summary(div)
        with open(os.path.join(args.output_dir, 'diversity_ema.json'), 'w') as f:
            json.dump(div, f, indent=2)

        # 2. Selection diversity (requires data)
        if args.val_data:
            from functools import partial
            from utils.data import TextDataset, collate_fn_dynamic_padding
            from torch.utils.data import DataLoader

            dataset = TextDataset(args.val_data, tokenizer, max_length=128)
            collate_fn = partial(collate_fn_dynamic_padding, tokenizer=tokenizer)
            dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

            sel_div = analyzer.analyze_selection_diversity(dataloader, args.max_batches)
            print_selection_diversity_summary(sel_div)
            with open(os.path.join(args.output_dir, 'diversity_selection.json'), 'w') as f:
                json.dump(sel_div, f, indent=2)
        else:
            print("\n[Info] Selection diversity requires --val_data")

    # Selection diversity only mode
    if args.mode == 'selection_diversity' and args.val_data:
        from functools import partial
        from utils.data import TextDataset, collate_fn_dynamic_padding
        from torch.utils.data import DataLoader

        dataset = TextDataset(args.val_data, tokenizer, max_length=128)
        collate_fn = partial(collate_fn_dynamic_padding, tokenizer=tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

        sel_div = analyzer.analyze_selection_diversity(dataloader, args.max_batches)
        print_selection_diversity_summary(sel_div)
        with open(os.path.join(args.output_dir, 'diversity_selection.json'), 'w') as f:
            json.dump(sel_div, f, indent=2)

    # NEW: Dead neuron analysis (for shrink decisions)
    if args.mode in ['all', 'dead_neurons']:
        dead = analyzer.analyze_dead_neurons(args.output_dir)
        print_dead_neurons_summary(dead)
        with open(os.path.join(args.output_dir, 'dead_neurons.json'), 'w') as f:
            json.dump(convert_to_serializable(dead), f, indent=2)

    # NEW: Neuron clustering
    if args.mode in ['all', 'clustering']:
        clusters = analyzer.analyze_clustering(args.n_clusters, args.output_dir)
        print("\n" + "="*60)
        print("NEURON CLUSTERING")
        print("="*60)
        for name, data in clusters.items():
            if name == 'visualization' or 'error' in str(data):
                continue
            print(f"\n{name}: {data['n_clusters']} clusters")
            for c in data.get('clusters', [])[:3]:
                print(f"  Cluster {c['cluster_id']}: size={c['size']}, active={c['active_count']}, usage={c['avg_usage']:.4f}")
        with open(os.path.join(args.output_dir, 'clustering.json'), 'w') as f:
            json.dump(convert_to_serializable(clusters), f, indent=2)

    # NEW: Token trajectory (requires data)
    if args.mode == 'trajectory' and args.val_data:
        from functools import partial
        from utils.data import TextDataset, collate_fn_dynamic_padding
        from torch.utils.data import DataLoader

        dataset = TextDataset(args.val_data, tokenizer, max_length=128)
        collate_fn = partial(collate_fn_dynamic_padding, tokenizer=tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

        traj = analyzer.analyze_token_trajectory(dataloader, args.max_batches)
        print("\n" + "="*60)
        print("TOKEN TRAJECTORY")
        print("="*60)
        print(f"POS counts: {traj.get('pos_counts', {})}")
        with open(os.path.join(args.output_dir, 'trajectory.json'), 'w') as f:
            json.dump(convert_to_serializable(traj), f, indent=2)

    # NEW: Probing classifier (requires data)
    if args.mode == 'probing' and args.val_data:
        from functools import partial
        from utils.data import TextDataset, collate_fn_dynamic_padding
        from torch.utils.data import DataLoader

        dataset = TextDataset(args.val_data, tokenizer, max_length=128)
        collate_fn = partial(collate_fn_dynamic_padding, tokenizer=tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

        probing = analyzer.run_probing(dataloader, args.max_batches)
        print("\n" + "="*60)
        print("PROBING CLASSIFIER")
        print("="*60)
        for ntype, data in probing.items():
            if 'error' in data:
                print(f"{ntype}: {data['error']}")
            else:
                print(f"{ntype}: accuracy={data['accuracy']:.2%}, samples={data['n_samples']}")
        with open(os.path.join(args.output_dir, 'probing.json'), 'w') as f:
            json.dump(convert_to_serializable(probing), f, indent=2)

    if args.mode == 'ablation' and args.val_data:
        from functools import partial
        from utils.data import TextDataset, collate_fn_dynamic_padding
        from torch.utils.data import DataLoader

        dataset = TextDataset(args.val_data, tokenizer, max_length=128)
        collate_fn = partial(collate_fn_dynamic_padding, tokenizer=tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

        ablation = analyzer.run_ablation(dataloader, args.max_batches)
        print("\n" + "="*60)
        print("ABLATION STUDY")
        print("="*60)
        print(f"Baseline loss: {ablation['baseline_loss']:.4f}")
        for ntype, results in ablation.items():
            if ntype == 'baseline_loss':
                continue
            print(f"\n{ntype}:")
            for r in results:
                print(f"  Neuron {r['neuron_id']}: delta={r['loss_delta']:.4f}, importance={r['importance']:.2%}")
        with open(os.path.join(args.output_dir, 'ablation.json'), 'w') as f:
            json.dump(convert_to_serializable(ablation), f, indent=2)

    if args.mode == 'all':
        viz = analyzer.visualize_usage(args.output_dir)
        print(f"\nUsage visualization saved to: {viz.get('visualization', 'N/A')}")

    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
