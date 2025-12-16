#!/usr/bin/env python3
"""
DAWN Model Analysis Suite
=========================
Analysis toolkit for the DAWN (Dynamic Attention and Weight Network) model.

Analysis Categories:
1. Usage Analysis     - EMA distribution, active/dead neurons, diversity
2. Routing Analysis   - Routing patterns, entropy, selection frequency
3. Embedding Analysis - Neuron embeddings, similarity, clustering

Usage:
    python analyze_dawn.py --checkpoint path/to/ckpt --mode all
    python analyze_dawn.py --checkpoint path/to/ckpt --mode usage
    python analyze_dawn.py --checkpoint path/to/ckpt --mode routing --val_data path/to/data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs): return x

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
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


# ============================================================
# DAWN Neuron Types Configuration
# ============================================================

NEURON_TYPES = {
    # (display_name, ema_attr, n_attr, color)
    'feature_qk':   ('FQK',  'usage_ema_feature_qk',   'n_feature_qk',   'red'),
    'feature_v':    ('FV',   'usage_ema_feature_v',    'n_feature_v',    'orange'),
    'restore_qk':   ('RQK',  'usage_ema_restore_qk',   'n_restore_qk',   'blue'),
    'restore_v':    ('RV',   'usage_ema_restore_v',    'n_restore_v',    'green'),
    'feature_know': ('FK',   'usage_ema_feature_know', 'n_feature_know', 'purple'),
    'restore_know': ('RK',   'usage_ema_restore_know', 'n_restore_know', 'cyan'),
}

ROUTING_KEYS = {
    # (display_name, pref_key, weight_key, pool_type)
    'fqk_q': ('FQK_Q', 'fqk_q_pref', 'fqk_weights_Q', 'feature_qk'),
    'fqk_k': ('FQK_K', 'fqk_k_pref', 'fqk_weights_K', 'feature_qk'),
    'fv':    ('FV',    'fv_pref',    'fv_weights',    'feature_v'),
    'rqk_q': ('RQK_Q', 'rqk_q_pref', 'rqk_weights_Q', 'restore_qk'),
    'rqk_k': ('RQK_K', 'rqk_k_pref', 'rqk_weights_K', 'restore_qk'),
    'rv':    ('RV',    'rv_pref',    'rv_weights',    'restore_v'),
}


# ============================================================
# Utility Functions
# ============================================================

def gini_coefficient(values: torch.Tensor) -> float:
    """Calculate Gini coefficient (0=equal, 1=unequal)"""
    values = values.flatten().float()
    if values.sum() == 0:
        return 0.0
    sorted_vals = torch.sort(values)[0]
    n = len(sorted_vals)
    cumsum = torch.cumsum(sorted_vals, dim=0)
    return float(1 - 2 * cumsum.sum() / (n * sorted_vals.sum()) + 1/n)


def calc_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Calculate entropy along dimension"""
    probs = probs.clamp(min=1e-8)
    return -torch.sum(probs * torch.log(probs), dim=dim)


def calc_entropy_ratio(probs: torch.Tensor) -> float:
    """Calculate entropy as percentage of maximum"""
    if probs.numel() == 0:
        return 0.0
    ent = calc_entropy(probs.mean(dim=0) if probs.dim() > 1 else probs)
    max_ent = np.log(probs.shape[-1])
    return float(ent / max_ent * 100) if max_ent > 0 else 0.0


def convert_to_serializable(obj):
    """Convert numpy/torch types to JSON-serializable"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load v17.1 model from checkpoint"""
    from models import create_model_by_version
    from transformers import BertTokenizer

    # Find checkpoint file
    path = Path(checkpoint_path)
    if path.is_dir():
        pt_files = list(path.glob('*.pt'))
        for f in pt_files:
            if 'best' in f.name.lower() or 'final' in f.name.lower():
                checkpoint_path = str(f)
                break
        else:
            if pt_files:
                checkpoint_path = str(sorted(pt_files, key=os.path.getmtime)[-1])

    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    # Force v17.1
    version = '17.1'
    print(f"Model version: {version}")

    model = create_model_by_version(version, config)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer, config


def get_router(model):
    """Get neuron router from model"""
    if hasattr(model, 'router') and hasattr(model.router, 'neuron_router'):
        return model.router.neuron_router
    if hasattr(model, 'global_routers'):
        return model.global_routers.neuron_router
    # Handle compiled models
    if hasattr(model, '_orig_mod'):
        return get_router(model._orig_mod)
    return None


def get_neurons(model):
    """Get shared neurons from model"""
    if hasattr(model, 'shared_neurons'):
        return model.shared_neurons
    if hasattr(model, '_orig_mod'):
        return get_neurons(model._orig_mod)
    return None


# ============================================================
# 1. Usage Analysis
# ============================================================

class UsageAnalyzer:
    """뉴런 사용 패턴 분석"""

    def __init__(self, router):
        self.router = router

    def analyze_ema_distribution(self) -> Dict:
        """EMA 분포 분석"""
        results = {}
        threshold = 0.01

        for name, (display, ema_attr, n_attr, _) in NEURON_TYPES.items():
            if not hasattr(self.router, ema_attr):
                continue

            ema = getattr(self.router, ema_attr)
            n_total = getattr(self.router, n_attr)

            active = (ema > threshold).sum().item()
            dead = (ema < threshold).sum().item()

            results[name] = {
                'display': display,
                'total': n_total,
                'active': int(active),
                'dead': int(dead),
                'active_ratio': active / n_total,
                'gini': gini_coefficient(ema),
                'stats': {
                    'min': float(ema.min()),
                    'max': float(ema.max()),
                    'mean': float(ema.mean()),
                    'std': float(ema.std()),
                    'median': float(ema.median()),
                }
            }

        return results

    def analyze_excitability(self) -> Dict:
        """Excitability 상태 분석"""
        tau = self.router.tau
        weight = getattr(self.router, 'excitability_weight', 0)
        if hasattr(weight, 'item'):
            weight = weight.item()

        results = {
            'tau': tau,
            'weight': weight,
        }

        for name, (display, ema_attr, _, _) in NEURON_TYPES.items():
            if not hasattr(self.router, ema_attr):
                continue

            ema = getattr(self.router, ema_attr)
            exc = torch.clamp(1.0 - ema / tau, min=0.0, max=1.0)

            results[name] = {
                'display': display,
                'min': float(exc.min()),
                'max': float(exc.max()),
                'mean': float(exc.mean()),
                'high_exc_count': int((exc > 0.8).sum()),  # 재활성화 가능
            }

        return results

    def analyze_diversity(self) -> Dict:
        """뉴런 다양성 분석 (엔트로피, effective count)"""
        results = {}
        threshold = 0.01

        for name, (display, ema_attr, n_attr, _) in NEURON_TYPES.items():
            if not hasattr(self.router, ema_attr):
                continue

            ema = getattr(self.router, ema_attr)
            n_total = getattr(self.router, n_attr)

            active_mask = ema > threshold
            n_active = active_mask.sum().item()

            if n_active == 0:
                results[name] = {
                    'display': display,
                    'n_active': 0,
                    'entropy': 0,
                    'effective_count': 0,
                    'coverage': 0,
                }
                continue

            # Normalize to probability
            active_ema = ema[active_mask]
            p = active_ema / active_ema.sum()

            entropy = -torch.sum(p * torch.log(p + 1e-8)).item()
            max_entropy = np.log(n_active)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            effective_count = np.exp(entropy)

            # Top-k concentration
            top5 = torch.topk(active_ema, min(5, n_active))[0]
            top5_share = top5.sum() / active_ema.sum()

            results[name] = {
                'display': display,
                'n_active': n_active,
                'n_total': n_total,
                'entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'effective_count': float(effective_count),
                'coverage': n_active / n_total,
                'top5_share': float(top5_share),
                'gini': gini_coefficient(ema),
            }

        # Overall health score
        entropies = [r['normalized_entropy'] for r in results.values() if isinstance(r, dict) and 'normalized_entropy' in r]
        overall = sum(entropies) / len(entropies) if entropies else 0

        results['overall'] = {
            'diversity_score': overall,
            'health': 'good' if overall > 0.7 else 'warning' if overall > 0.4 else 'critical'
        }

        return results

    def analyze_dead_neurons(self) -> Dict:
        """죽은 뉴런 분석 및 축소 권장"""
        results = {}
        threshold = 0.01
        dying_threshold = 0.05
        tau = self.router.tau

        for name, (display, ema_attr, n_attr, _) in NEURON_TYPES.items():
            if not hasattr(self.router, ema_attr):
                continue

            ema = getattr(self.router, ema_attr)
            n_total = getattr(self.router, n_attr)
            exc = torch.clamp(1.0 - ema / tau, min=0.0, max=1.0)

            dead_mask = ema < threshold
            dying_mask = (ema >= threshold) & (ema < dying_threshold)
            active_mask = ema >= dying_threshold

            # Revivable: dead but high excitability
            revivable_mask = dead_mask & (exc > 0.8)
            # Removable: dead and low excitability
            removable_mask = dead_mask & (exc < 0.3)

            results[name] = {
                'display': display,
                'n_total': n_total,
                'n_active': int(active_mask.sum()),
                'n_dying': int(dying_mask.sum()),
                'n_dead': int(dead_mask.sum()),
                'n_revivable': int(revivable_mask.sum()),
                'n_removable': int(removable_mask.sum()),
            }

        # Shrink recommendation
        total_removable = sum(r['n_removable'] for r in results.values() if isinstance(r, dict))
        total_neurons = sum(r['n_total'] for r in results.values() if isinstance(r, dict))

        results['recommendation'] = {
            'total_removable': total_removable,
            'shrink_ratio': total_removable / total_neurons if total_neurons > 0 else 0,
            'action': 'shrink' if total_removable > total_neurons * 0.2 else 'keep'
        }

        return results


# ============================================================
# 2. Routing Analysis
# ============================================================

class RoutingAnalyzer:
    """라우팅 패턴 분석"""

    def __init__(self, model, router, device='cuda'):
        self.model = model
        self.router = router
        self.device = device

    def analyze_entropy(self, dataloader, n_batches: int = 50) -> Dict:
        """라우팅 엔트로피 분석"""
        entropy_data = {name: [] for name in ROUTING_KEYS.keys()}

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Entropy')):
                if i >= n_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                outputs = self.model(input_ids, return_routing_info=True)

                if not isinstance(outputs, tuple) or len(outputs) < 2:
                    continue

                routing_infos = outputs[-1]

                # Average entropy across all layers
                for layer_info in routing_infos:
                    attn = layer_info.get('attention', {})

                    for key, (display, pref_key, _, _) in ROUTING_KEYS.items():
                        pref = attn.get(pref_key)
                        if pref is not None:
                            ent = calc_entropy_ratio(pref)
                            entropy_data[key].append(ent)

        results = {}
        for key, (display, _, _, pool) in ROUTING_KEYS.items():
            if entropy_data[key]:
                results[key] = {
                    'display': display,
                    'pool': pool,
                    'mean_entropy': float(np.mean(entropy_data[key])),
                    'std_entropy': float(np.std(entropy_data[key])),
                    'min_entropy': float(np.min(entropy_data[key])),
                    'max_entropy': float(np.max(entropy_data[key])),
                }

        return results

    def analyze_selection_frequency(self, dataloader, n_batches: int = 50) -> Dict:
        """뉴런 선택 빈도 분석"""
        selection_counts = {name: Counter() for name in ROUTING_KEYS.keys()}

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Selection')):
                if i >= n_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                outputs = self.model(input_ids, return_routing_info=True)

                if not isinstance(outputs, tuple) or len(outputs) < 2:
                    continue

                routing_infos = outputs[-1]

                for layer_info in routing_infos:
                    attn = layer_info.get('attention', {})

                    for key, (_, pref_key, _, _) in ROUTING_KEYS.items():
                        pref = attn.get(pref_key)
                        if pref is None:
                            continue

                        # Get top-k selections
                        k = min(8, pref.shape[-1])
                        _, topk_idx = torch.topk(pref, k, dim=-1)
                        flat_idx = topk_idx.view(-1).cpu().numpy()

                        for idx in flat_idx:
                            selection_counts[key][int(idx)] += 1

        results = {}
        for key, (display, _, _, pool) in ROUTING_KEYS.items():
            counts = selection_counts[key]
            if not counts:
                continue

            total = sum(counts.values())
            top10 = counts.most_common(10)
            unique = len(counts)

            # Get pool size
            pool_info = NEURON_TYPES.get(pool, {})
            n_attr = pool_info[2] if pool_info else None
            n_total = getattr(self.router, n_attr, 0) if n_attr else 0

            results[key] = {
                'display': display,
                'pool': pool,
                'total_selections': total,
                'unique_selected': unique,
                'coverage': unique / n_total if n_total > 0 else 0,
                'top10': [(idx, cnt, cnt/total) for idx, cnt in top10],
                'concentration': sum(cnt for _, cnt in top10) / total if total > 0 else 0,
            }

        return results

    def analyze_qk_overlap(self, dataloader, n_batches: int = 50) -> Dict:
        """Q/K 라우팅 오버랩 분석"""
        overlap_data = {'fqk': [], 'rqk': []}

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Q/K Overlap')):
                if i >= n_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                outputs = self.model(input_ids, return_routing_info=True)

                if not isinstance(outputs, tuple) or len(outputs) < 2:
                    continue

                routing_infos = outputs[-1]

                for layer_info in routing_infos:
                    attn = layer_info.get('attention', {})

                    # Feature QK overlap
                    fqk_q = attn.get('fqk_q_pref')
                    fqk_k = attn.get('fqk_k_pref')
                    if fqk_q is not None and fqk_k is not None:
                        k = min(8, fqk_q.shape[-1])
                        q_top = torch.topk(fqk_q, k, dim=-1)[1]
                        k_top = torch.topk(fqk_k, k, dim=-1)[1]

                        # Calculate Jaccard overlap
                        for b in range(q_top.shape[0]):
                            q_set = set(q_top[b].view(-1).cpu().tolist())
                            k_set = set(k_top[b].view(-1).cpu().tolist())
                            overlap = len(q_set & k_set) / len(q_set | k_set) if (q_set | k_set) else 0
                            overlap_data['fqk'].append(overlap)

                    # Restore QK overlap
                    rqk_q = attn.get('rqk_q_pref')
                    rqk_k = attn.get('rqk_k_pref')
                    if rqk_q is not None and rqk_k is not None:
                        k = min(8, rqk_q.shape[-1])
                        q_top = torch.topk(rqk_q, k, dim=-1)[1]
                        k_top = torch.topk(rqk_k, k, dim=-1)[1]

                        for b in range(q_top.shape[0]):
                            q_set = set(q_top[b].view(-1).cpu().tolist())
                            k_set = set(k_top[b].view(-1).cpu().tolist())
                            overlap = len(q_set & k_set) / len(q_set | k_set) if (q_set | k_set) else 0
                            overlap_data['rqk'].append(overlap)

        results = {}
        for key in ['fqk', 'rqk']:
            if overlap_data[key]:
                results[key] = {
                    'mean_overlap': float(np.mean(overlap_data[key])),
                    'std_overlap': float(np.std(overlap_data[key])),
                    'interpretation': 'Q와 K가 비슷한 뉴런 선택' if np.mean(overlap_data[key]) > 0.3 else 'Q와 K가 다른 뉴런 선택'
                }

        return results


# ============================================================
# 3. Embedding Analysis
# ============================================================

class EmbeddingAnalyzer:
    """뉴런 임베딩 분석"""

    def __init__(self, router):
        self.router = router

    def get_embeddings_by_type(self) -> Dict[str, np.ndarray]:
        """타입별 임베딩 추출"""
        emb = self.router.neuron_emb.detach().cpu().numpy()

        result = {}
        offset = 0
        for name, (display, _, n_attr, _) in NEURON_TYPES.items():
            if hasattr(self.router, n_attr):
                n = getattr(self.router, n_attr)
                result[name] = emb[offset:offset + n]
                offset += n

        return result

    def analyze_similarity(self) -> Dict:
        """타입 내 유사도 분석"""
        embeddings = self.get_embeddings_by_type()
        results = {}

        for name, emb in embeddings.items():
            if len(emb) < 2:
                continue

            # Normalize and compute similarity
            emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
            sim_matrix = emb_norm @ emb_norm.T

            # Stats (excluding diagonal)
            n = sim_matrix.shape[0]
            off_diag = sim_matrix[~np.eye(n, dtype=bool)]

            display = NEURON_TYPES[name][0]
            results[name] = {
                'display': display,
                'n_neurons': n,
                'avg_similarity': float(off_diag.mean()),
                'max_similarity': float(off_diag.max()),
                'min_similarity': float(off_diag.min()),
                'std_similarity': float(off_diag.std()),
            }

        return results

    def analyze_cross_type_similarity(self) -> Dict:
        """타입 간 유사도 분석"""
        embeddings = self.get_embeddings_by_type()

        # Compute centroids
        centroids = {}
        for name, emb in embeddings.items():
            centroids[name] = emb.mean(axis=0)

        # Compute pairwise similarity
        results = {}
        names = list(centroids.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                c1, c2 = centroids[n1], centroids[n2]
                sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
                key = f"{NEURON_TYPES[n1][0]}-{NEURON_TYPES[n2][0]}"
                results[key] = float(sim)

        return results

    def visualize(self, output_dir: str) -> str:
        """t-SNE/PCA 시각화"""
        if not HAS_MATPLOTLIB or not HAS_SKLEARN:
            return None

        os.makedirs(output_dir, exist_ok=True)

        emb = self.router.neuron_emb.detach().cpu().numpy()

        # Build labels
        labels = []
        colors_map = {}
        for name, (display, _, n_attr, color) in NEURON_TYPES.items():
            if hasattr(self.router, n_attr):
                n = getattr(self.router, n_attr)
                labels.extend([display] * n)
                colors_map[display] = color

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb)-1))
        emb_tsne = tsne.fit_transform(emb)

        # PCA
        pca = PCA(n_components=2)
        emb_pca = pca.fit_transform(emb)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, data, title in [(axes[0], emb_tsne, 't-SNE'), (axes[1], emb_pca, 'PCA')]:
            for t in set(labels):
                mask = np.array([l == t for l in labels])
                ax.scatter(data[mask, 0], data[mask, 1],
                          c=colors_map.get(t, 'gray'), label=t, alpha=0.6, s=20)
            ax.set_title(f'DAWN Neuron Embeddings ({title})')
            ax.legend()

        plt.tight_layout()
        path = os.path.join(output_dir, 'dawn_embeddings.png')
        plt.savefig(path, dpi=150)
        plt.close()

        return path


# ============================================================
# Main Analyzer Class
# ============================================================

class DAWNAnalyzer:
    """DAWN model integrated analyzer"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.router = get_router(model)
        self.neurons = get_neurons(model)

        if self.router is None:
            raise ValueError("Could not find router in model")

        self.usage = UsageAnalyzer(self.router)
        self.embedding = EmbeddingAnalyzer(self.router)
        self.routing = RoutingAnalyzer(model, self.router, device)

        print("DAWN Analyzer initialized")

    def run_usage_analysis(self, output_dir: str = None) -> Dict:
        """Usage 분석 실행"""
        print("\n" + "="*60)
        print("USAGE ANALYSIS")
        print("="*60)

        results = {
            'ema_distribution': self.usage.analyze_ema_distribution(),
            'excitability': self.usage.analyze_excitability(),
            'diversity': self.usage.analyze_diversity(),
            'dead_neurons': self.usage.analyze_dead_neurons(),
        }

        # Print summary
        print("\n--- EMA Distribution ---")
        for name, data in results['ema_distribution'].items():
            print(f"  {data['display']:>4}: {data['active']:>4}/{data['total']:<4} active ({data['active_ratio']:.1%}), gini={data['gini']:.2f}")

        print("\n--- Diversity ---")
        div = results['diversity']
        for name, data in div.items():
            if name == 'overall':
                continue
            if isinstance(data, dict) and 'display' in data:
                print(f"  {data['display']:>4}: entropy={data.get('normalized_entropy', 0):.2f}, effective={data.get('effective_count', 0):.1f}")
        print(f"\n  Overall health: {div['overall']['health']} (score={div['overall']['diversity_score']:.2f})")

        print("\n--- Dead Neurons ---")
        dead = results['dead_neurons']
        for name, data in dead.items():
            if name == 'recommendation':
                continue
            if isinstance(data, dict) and 'display' in data:
                print(f"  {data['display']:>4}: {data['n_dead']} dead, {data['n_revivable']} revivable, {data['n_removable']} removable")
        print(f"\n  Recommendation: {dead['recommendation']['action']} ({dead['recommendation']['total_removable']} removable)")

        return results

    def run_routing_analysis(self, dataloader, n_batches: int = 50, output_dir: str = None) -> Dict:
        """Routing 분석 실행"""
        print("\n" + "="*60)
        print("ROUTING ANALYSIS")
        print("="*60)

        results = {
            'entropy': self.routing.analyze_entropy(dataloader, n_batches),
            'selection': self.routing.analyze_selection_frequency(dataloader, n_batches),
            'qk_overlap': self.routing.analyze_qk_overlap(dataloader, n_batches),
        }

        # Print summary
        print("\n--- Routing Entropy ---")
        for key, data in results['entropy'].items():
            print(f"  {data['display']:>6}: {data['mean_entropy']:.1f}% (std={data['std_entropy']:.1f})")

        print("\n--- Selection Coverage ---")
        for key, data in results['selection'].items():
            print(f"  {data['display']:>6}: {data['unique_selected']}/{data['coverage']:.1%} coverage, top10 conc={data['concentration']:.1%}")

        print("\n--- Q/K Overlap ---")
        for key, data in results['qk_overlap'].items():
            print(f"  {key.upper()}: {data['mean_overlap']:.2f} - {data['interpretation']}")

        return results

    def run_embedding_analysis(self, output_dir: str = None) -> Dict:
        """Embedding 분석 실행"""
        print("\n" + "="*60)
        print("EMBEDDING ANALYSIS")
        print("="*60)

        results = {
            'intra_similarity': self.embedding.analyze_similarity(),
            'cross_similarity': self.embedding.analyze_cross_type_similarity(),
        }

        # Print summary
        print("\n--- Intra-type Similarity ---")
        for name, data in results['intra_similarity'].items():
            print(f"  {data['display']:>4}: avg={data['avg_similarity']:.3f}, max={data['max_similarity']:.3f}")

        print("\n--- Cross-type Similarity (centroids) ---")
        for key, sim in results['cross_similarity'].items():
            print(f"  {key}: {sim:.3f}")

        # Visualization
        if output_dir:
            viz_path = self.embedding.visualize(output_dir)
            if viz_path:
                results['visualization'] = viz_path
                print(f"\n  Visualization saved: {viz_path}")

        return results

    def run_all(self, dataloader=None, output_dir: str = './dawn_analysis') -> Dict:
        """전체 분석 실행"""
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'usage': self.run_usage_analysis(output_dir),
            'embedding': self.run_embedding_analysis(output_dir),
        }

        if dataloader:
            results['routing'] = self.run_routing_analysis(dataloader, output_dir=output_dir)

        # Save results
        output_path = os.path.join(output_dir, 'dawn_analysis.json')
        with open(output_path, 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


# ============================================================
# CLI
# ============================================================

def create_dataloader(data_path: str, tokenizer, batch_size: int = 32):
    """Create dataloader from parquet/json data"""
    from torch.utils.data import DataLoader, Dataset

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len=128):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {k: v.squeeze(0) for k, v in enc.items()}

    # Load data
    if data_path.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(data_path)
        texts = df['text'].tolist()[:10000]
    elif data_path.endswith('.json'):
        import json
        with open(data_path) as f:
            data = json.load(f)
        texts = [d['text'] for d in data[:10000]]
    else:
        raise ValueError(f"Unsupported data format: {data_path}")

    dataset = TextDataset(texts, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main():
    parser = argparse.ArgumentParser(description='DAWN Model Analysis Suite')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'usage', 'routing', 'embedding'],
                        help='Analysis mode')
    parser.add_argument('--val_data', type=str, default=None, help='Validation data (for routing analysis)')
    parser.add_argument('--output_dir', type=str, default='./dawn_analysis', help='Output directory')
    parser.add_argument('--n_batches', type=int, default=50, help='Number of batches for routing analysis')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, tokenizer, config = load_model(args.checkpoint, device)

    # Create analyzer
    analyzer = DAWNAnalyzer(model, tokenizer, device)

    # Create dataloader if needed
    dataloader = None
    if args.val_data and args.mode in ['all', 'routing']:
        dataloader = create_dataloader(args.val_data, tokenizer, args.batch_size)

    # Run analysis
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'all':
        results = analyzer.run_all(dataloader, args.output_dir)
    elif args.mode == 'usage':
        results = analyzer.run_usage_analysis(args.output_dir)
    elif args.mode == 'routing':
        if dataloader is None:
            print("ERROR: --val_data required for routing analysis")
            return
        results = analyzer.run_routing_analysis(dataloader, args.n_batches, args.output_dir)
    elif args.mode == 'embedding':
        results = analyzer.run_embedding_analysis(args.output_dir)

    # Save results
    output_path = os.path.join(args.output_dir, f'dawn_{args.mode}.json')
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"\n{'='*60}")
    print(f"Analysis complete! Results: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
