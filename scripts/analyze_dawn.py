#!/usr/bin/env python3
"""
DAWN Model Analysis Suite
=========================
Comprehensive analysis toolkit for the DAWN (Dynamic Attention and Weight Network) model.
Designed for paper-ready analysis and visualization.

Analysis Categories:
1. Usage Analysis       - EMA distribution, excitability, diversity, dead neurons
2. Routing Analysis     - Entropy, selection frequency, Q/K overlap, selection diversity
3. Embedding Analysis   - Similarity, clustering, t-SNE/PCA visualization
4. Weight Analysis      - SVD decomposition, effective rank
5. Behavioral Analysis  - Token trajectory, probing classifier, ablation study

Usage:
    python analyze_dawn.py --checkpoint path/to/ckpt --mode all
    python analyze_dawn.py --checkpoint path/to/ckpt --mode usage
    python analyze_dawn.py --checkpoint path/to/ckpt --mode routing --val_data path/to/data
    python analyze_dawn.py --checkpoint path/to/ckpt --mode weight_svd
    python analyze_dawn.py --checkpoint path/to/ckpt --mode clustering
    python analyze_dawn.py --checkpoint path/to/ckpt --mode probing --val_data path/to/data
    python analyze_dawn.py --checkpoint path/to/ckpt --mode ablation --val_data path/to/data
    python analyze_dawn.py --checkpoint path/to/ckpt --mode trajectory --val_data path/to/data
    python analyze_dawn.py --checkpoint path/to/ckpt --mode neuron --neuron_type feature_qk --neuron_id 0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
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

# Neuron attribute names for weight analysis
NEURON_ATTRS = {
    'feature_qk': 'feature_qk_neurons',
    'feature_v': 'feature_v_neurons',
    'restore_qk': 'restore_qk_neurons',
    'restore_v': 'restore_v_neurons',
    'feature_know': 'feature_know',
    'restore_know': 'restore_know',
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


def simple_pos_tag(token: str) -> str:
    """Simple POS tagging for probing analysis"""
    token = token.lower().strip()
    if token in ['the', 'a', 'an']:
        return 'DET'
    elif token in ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'has', 'have', 'had', 'do', 'does', 'did']:
        return 'VERB'
    elif token in ['in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about']:
        return 'PREP'
    elif token in ['and', 'or', 'but', 'so', 'yet']:
        return 'CONJ'
    elif token in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']:
        return 'PRON'
    elif token in ['.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}']:
        return 'PUNCT'
    elif token.isdigit():
        return 'NUM'
    elif token.startswith('[') and token.endswith(']'):
        return 'SPECIAL'
    else:
        return 'OTHER'


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load DAWN model from checkpoint"""
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
    """Neuron usage pattern analysis"""

    def __init__(self, router):
        self.router = router

    def analyze_ema_distribution(self) -> Dict:
        """EMA distribution analysis"""
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
        """Excitability state analysis"""
        tau = self.router.tau
        weight = getattr(self.router, 'excitability_weight', 0)
        if hasattr(weight, 'item'):
            weight = weight.item()

        results = {
            'tau': tau,
            'weight': weight,
            'langevin_alpha': getattr(self.router, 'langevin_alpha', 0),
            'langevin_beta': getattr(self.router, 'langevin_beta', 0),
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
                'high_exc_count': int((exc > 0.8).sum()),
            }

        return results

    def analyze_diversity(self) -> Dict:
        """Neuron diversity analysis (entropy, effective count)"""
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

            active_ema = ema[active_mask]
            p = active_ema / active_ema.sum()

            entropy = -torch.sum(p * torch.log(p + 1e-8)).item()
            max_entropy = np.log(n_active)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            effective_count = np.exp(entropy)

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

        entropies = [r['normalized_entropy'] for r in results.values() if isinstance(r, dict) and 'normalized_entropy' in r]
        overall = sum(entropies) / len(entropies) if entropies else 0

        results['overall'] = {
            'diversity_score': overall,
            'health': 'good' if overall > 0.7 else 'warning' if overall > 0.4 else 'critical'
        }

        return results

    def analyze_dead_neurons(self, output_dir: str = None) -> Dict:
        """Dead neuron analysis and shrink recommendations"""
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

            revivable_mask = dead_mask & (exc > 0.8)
            removable_mask = dead_mask & (exc < 0.3)

            results[name] = {
                'display': display,
                'n_total': n_total,
                'n_active': int(active_mask.sum()),
                'n_dying': int(dying_mask.sum()),
                'n_dead': int(dead_mask.sum()),
                'n_revivable': int(revivable_mask.sum()),
                'n_removable': int(removable_mask.sum()),
                'dead_neuron_ids': dead_mask.nonzero().squeeze(-1).tolist() if dead_mask.sum() > 0 else [],
                'removable_neuron_ids': removable_mask.nonzero().squeeze(-1).tolist() if removable_mask.sum() > 0 else [],
            }

        total_removable = sum(r['n_removable'] for r in results.values() if isinstance(r, dict) and 'n_removable' in r)
        total_neurons = sum(r['n_total'] for r in results.values() if isinstance(r, dict) and 'n_total' in r)

        type_names = [name for name in results.keys() if isinstance(results[name], dict) and 'n_total' in results[name]]

        results['recommendation'] = {
            'total_removable': total_removable,
            'shrink_ratio': total_removable / total_neurons if total_neurons > 0 else 0,
            'action': 'shrink' if total_removable > total_neurons * 0.2 else 'keep',
            'per_type': {
                name: {
                    'current': results[name]['n_total'],
                    'recommended': results[name]['n_total'] - results[name]['n_removable'],
                    'remove': results[name]['n_removable'],
                }
                for name in type_names
            }
        }

        # Visualization
        if HAS_MATPLOTLIB and output_dir:
            os.makedirs(output_dir, exist_ok=True)

            n_types = len(type_names)
            n_cols = 3
            n_rows = (n_types + 1 + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_types == 0 else list(axes)

            colors = ['green', 'yellow', 'red']
            labels = ['Active', 'Dying', 'Dead']

            for ax, name in zip(axes[:n_types], type_names):
                data = results[name]
                sizes = [data['n_active'], data['n_dying'], data['n_dead']]
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title(f'{name.upper()}\n(removable: {data["n_removable"]})')

            if n_types < len(axes):
                ax = axes[n_types]
                display_names = [n.upper()[:3] for n in type_names]
                removable = [results[n]['n_removable'] for n in type_names]
                ax.bar(display_names, removable, color='red', alpha=0.7)
                ax.set_title('Removable Neurons')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)

            for i in range(n_types + 1, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            path = os.path.join(output_dir, 'dead_neurons.png')
            plt.savefig(path, dpi=150)
            plt.close()
            results['visualization'] = path

        return results

    def visualize_usage(self, output_dir: str) -> Dict:
        """Create usage histogram plots"""
        if not HAS_MATPLOTLIB:
            return {'error': 'matplotlib not available'}

        os.makedirs(output_dir, exist_ok=True)

        data = []
        for name, (display, ema_attr, _, color) in NEURON_TYPES.items():
            if hasattr(self.router, ema_attr):
                ema = getattr(self.router, ema_attr)
                data.append((display, ema, color))

        n_plots = len(data) + 1
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_plots == 1 else list(axes)

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

        if len(data) < len(axes):
            ax = axes[len(data)]
            names = [d[0] for d in data]
            active_ratios = [(d[1] > 0.01).float().mean().item() for d in data]
            colors = [d[2] for d in data]
            ax.bar(names, active_ratios, color=colors, alpha=0.7, edgecolor='black')
            ax.set_title('Active Neuron Ratio by Type')
            ax.set_ylabel('Active Ratio')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)

        for i in range(len(data) + 1, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        path = os.path.join(output_dir, 'usage_histogram.png')
        plt.savefig(path, dpi=150)
        plt.close()

        return {'visualization': path}


# ============================================================
# 2. Routing Analysis
# ============================================================

class RoutingAnalyzer:
    """Routing pattern analysis"""

    def __init__(self, model, router, device='cuda'):
        self.model = model
        self.router = router
        self.device = device

    def analyze_entropy(self, dataloader, n_batches: int = 50) -> Dict:
        """Routing entropy analysis"""
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
        """Neuron selection frequency analysis"""
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

    def analyze_selection_diversity(self, dataloader, n_batches: int = 100) -> Dict:
        """Selection diversity - measure actual selection diversity per batch"""
        union_selected = {key: set() for key in ROUTING_KEYS.keys()}
        per_batch_counts = {key: [] for key in ROUTING_KEYS.keys()}

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc='Selection Diversity', total=n_batches)):
                if i >= n_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)

                try:
                    outputs = self.model(input_ids, return_routing_info=True)
                    if not isinstance(outputs, tuple) or len(outputs) < 2:
                        continue
                    routing_infos = outputs[-1]
                    routing_info = routing_infos[0].get('attention', {}) if routing_infos else {}
                except:
                    continue

                for key, (_, _, weight_key, _) in ROUTING_KEYS.items():
                    if weight_key in routing_info:
                        weights = routing_info[weight_key]
                        if weights.dim() == 3:
                            selected = (weights > 0).any(dim=0).any(dim=0).cpu()
                            union_selected[key].update(selected.nonzero().flatten().tolist())
                            per_batch_counts[key].append(selected.sum().item())
                        elif weights.dim() == 2:
                            selected = (weights > 0).any(dim=0).cpu()
                            union_selected[key].update(selected.nonzero().flatten().tolist())
                            per_batch_counts[key].append(selected.sum().item())

        # Map routing keys to pool types for n_total lookup
        key_to_pool = {key: info[3] for key, info in ROUTING_KEYS.items()}

        results = {}
        for key in ROUTING_KEYS.keys():
            pool = key_to_pool[key]
            pool_info = NEURON_TYPES.get(pool, {})
            n_attr = pool_info[2] if pool_info else None
            n_total = getattr(self.router, n_attr, 0) if n_attr else 0

            union_count = len(union_selected[key])
            batch_counts = per_batch_counts[key]

            if len(batch_counts) > 0:
                per_batch_avg = np.mean(batch_counts)
                per_batch_std = np.std(batch_counts)
            else:
                per_batch_avg = 0
                per_batch_std = 0

            results[key] = {
                'display': ROUTING_KEYS[key][0],
                'pool': pool,
                'n_total': n_total,
                'per_batch_avg': float(per_batch_avg),
                'per_batch_std': float(per_batch_std),
                'union_count': union_count,
                'union_coverage': float(union_count / n_total) if n_total > 0 else 0,
                'diversity_ratio': float(union_count / per_batch_avg) if per_batch_avg > 0 else 0,
            }

        total_union = sum(len(union_selected[k]) for k in union_selected)
        total_neurons = sum(getattr(self.router, NEURON_TYPES[ROUTING_KEYS[k][3]][2], 0) for k in ROUTING_KEYS.keys())

        results['summary'] = {
            'n_batches_processed': min(n_batches, len(per_batch_counts[next(iter(per_batch_counts))])),
            'total_union_coverage': float(total_union / total_neurons) if total_neurons > 0 else 0,
            'interpretation': 'High diversity_ratio (>2) = many neurons selected differently per batch\n'
                             'Low diversity_ratio (~1) = same neurons always selected'
        }

        return results

    def analyze_qk_overlap(self, dataloader, n_batches: int = 50) -> Dict:
        """Q/K routing overlap analysis"""
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

                    fqk_q = attn.get('fqk_q_pref')
                    fqk_k = attn.get('fqk_k_pref')
                    if fqk_q is not None and fqk_k is not None:
                        k = min(8, fqk_q.shape[-1])
                        q_top = torch.topk(fqk_q, k, dim=-1)[1]
                        k_top = torch.topk(fqk_k, k, dim=-1)[1]

                        for b in range(q_top.shape[0]):
                            q_set = set(q_top[b].view(-1).cpu().tolist())
                            k_set = set(k_top[b].view(-1).cpu().tolist())
                            overlap = len(q_set & k_set) / len(q_set | k_set) if (q_set | k_set) else 0
                            overlap_data['fqk'].append(overlap)

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
                    'interpretation': 'Q and K select similar neurons' if np.mean(overlap_data[key]) > 0.3 else 'Q and K select different neurons'
                }

        return results


# ============================================================
# 3. Embedding Analysis
# ============================================================

class EmbeddingAnalyzer:
    """Neuron embedding analysis"""

    def __init__(self, router):
        self.router = router

    def get_embeddings_by_type(self) -> Dict[str, np.ndarray]:
        """Extract embeddings by type"""
        emb = self.router.neuron_emb.detach().cpu().numpy()

        result = {}
        offset = 0
        for name, (display, _, n_attr, _) in NEURON_TYPES.items():
            if hasattr(self.router, n_attr):
                n = getattr(self.router, n_attr)
                result[name] = emb[offset:offset + n]
                offset += n

        return result

    def analyze_similarity(self, output_dir: str = None) -> Dict:
        """Intra-type similarity analysis"""
        embeddings = self.get_embeddings_by_type()
        results = {}

        for name, emb in embeddings.items():
            if len(emb) < 2:
                continue

            emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
            sim_matrix = emb_norm @ emb_norm.T

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

        # Visualization
        if HAS_MATPLOTLIB and output_dir:
            os.makedirs(output_dir, exist_ok=True)

            n_types = len(embeddings)
            fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4))
            if n_types == 1:
                axes = [axes]

            for ax, (name, emb) in zip(axes, embeddings.items()):
                emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
                sim_matrix = emb_norm @ emb_norm.T
                sns.heatmap(sim_matrix, ax=ax, cmap='coolwarm', vmin=-1, vmax=1)
                ax.set_title(f'{NEURON_TYPES[name][0]} Similarity')

            plt.tight_layout()
            path = os.path.join(output_dir, 'similarity_heatmap.png')
            plt.savefig(path, dpi=150)
            plt.close()
            results['visualization'] = path

        return results

    def analyze_cross_type_similarity(self) -> Dict:
        """Cross-type similarity analysis"""
        embeddings = self.get_embeddings_by_type()

        centroids = {}
        for name, emb in embeddings.items():
            centroids[name] = emb.mean(axis=0)

        results = {}
        names = list(centroids.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                c1, c2 = centroids[n1], centroids[n2]
                sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
                key = f"{NEURON_TYPES[n1][0]}-{NEURON_TYPES[n2][0]}"
                results[key] = float(sim)

        return results

    def analyze_clustering(self, n_clusters: int = 5, output_dir: str = None) -> Dict:
        """Neuron clustering analysis"""
        if not HAS_SKLEARN:
            return {'error': 'sklearn not available'}

        results = {}
        emb = self.router.neuron_emb.detach().cpu().numpy()

        boundaries = {}
        offset = 0
        for name, (display, ema_attr, n_attr, _) in NEURON_TYPES.items():
            if hasattr(self.router, n_attr):
                n = getattr(self.router, n_attr)
                boundaries[name] = (offset, offset + n, ema_attr)
                offset += n

        for name, (start, end, ema_attr) in boundaries.items():
            type_emb = emb[start:end]
            n_neurons = type_emb.shape[0]

            if n_neurons < n_clusters:
                results[name] = {'error': f'Not enough neurons for {n_clusters} clusters'}
                continue

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(type_emb)

            cluster_stats = []
            ema = getattr(self.router, ema_attr).cpu().numpy() if hasattr(self.router, ema_attr) else None

            for c in range(n_clusters):
                cluster_mask = labels == c
                cluster_size = cluster_mask.sum()

                if ema is not None:
                    cluster_ema = ema[cluster_mask]
                    cluster_stats.append({
                        'cluster_id': c,
                        'size': int(cluster_size),
                        'avg_usage': float(cluster_ema.mean()),
                        'max_usage': float(cluster_ema.max()),
                        'min_usage': float(cluster_ema.min()),
                        'active_count': int((cluster_ema > 0.01).sum()),
                    })
                else:
                    cluster_stats.append({
                        'cluster_id': c,
                        'size': int(cluster_size),
                    })

            results[name] = {
                'display': NEURON_TYPES[name][0],
                'n_clusters': n_clusters,
                'clusters': sorted(cluster_stats, key=lambda x: -x.get('avg_usage', 0)),
                'labels': labels.tolist(),
            }

        # Visualization
        if HAS_MATPLOTLIB and output_dir:
            os.makedirs(output_dir, exist_ok=True)

            n_types = len([k for k in results if 'error' not in results.get(k, {})])
            if n_types > 0:
                fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5))
                if n_types == 1:
                    axes = [axes]

                ax_idx = 0
                for name, (start, end, _) in boundaries.items():
                    if name not in results or 'error' in results[name]:
                        continue

                    type_emb = emb[start:end]
                    pca = PCA(n_components=2)
                    emb_2d = pca.fit_transform(type_emb)

                    labels = results[name]['labels']
                    scatter = axes[ax_idx].scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
                    axes[ax_idx].set_title(f'{NEURON_TYPES[name][0]} Clusters')
                    ax_idx += 1

                plt.tight_layout()
                path = os.path.join(output_dir, 'clustering.png')
                plt.savefig(path, dpi=150)
                plt.close()
                results['visualization'] = path

        return results

    def visualize(self, output_dir: str) -> str:
        """t-SNE/PCA visualization"""
        if not HAS_MATPLOTLIB or not HAS_SKLEARN:
            return None

        os.makedirs(output_dir, exist_ok=True)

        emb = self.router.neuron_emb.detach().cpu().numpy()

        labels = []
        colors_map = {}
        for name, (display, _, n_attr, color) in NEURON_TYPES.items():
            if hasattr(self.router, n_attr):
                n = getattr(self.router, n_attr)
                labels.extend([display] * n)
                colors_map[display] = color

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb)-1))
        emb_tsne = tsne.fit_transform(emb)

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
# 4. Weight Analysis
# ============================================================

class WeightAnalyzer:
    """Neuron weight matrix analysis"""

    def __init__(self, neurons):
        self.neurons = neurons

    def analyze_weight_svd(self, output_dir: str = None) -> Dict:
        """Analyze neuron weights with SVD"""
        if self.neurons is None:
            return {'error': 'No shared neurons found'}

        results = {}

        for name, attr in NEURON_ATTRS.items():
            if not hasattr(self.neurons, attr):
                continue

            W = getattr(self.neurons, attr).detach().cpu()
            n_neurons = W.shape[0]

            if W.dim() > 2:
                W = W.view(n_neurons, -1)

            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except:
                results[name] = {'error': 'SVD failed'}
                continue

            S_normalized = S / S.sum()
            cumsum = torch.cumsum(S_normalized, dim=0)
            effective_rank = float((S > S.max() * 0.01).sum())
            var_top5 = float(cumsum[min(4, len(cumsum)-1)])
            var_top10 = float(cumsum[min(9, len(cumsum)-1)])

            results[name] = {
                'display': NEURON_TYPES.get(name, (name,))[0],
                'n_neurons': n_neurons,
                'weight_shape': list(W.shape),
                'effective_rank': effective_rank,
                'var_explained_by_top5': var_top5,
                'var_explained_by_top10': var_top10,
                'top_singular_values': S[:10].tolist(),
                'condition_number': float(S[0] / S[-1]) if S[-1] > 0 else float('inf'),
            }

        # Visualization
        if HAS_MATPLOTLIB and output_dir:
            os.makedirs(output_dir, exist_ok=True)

            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            if valid_results:
                n_plots = len(valid_results)
                fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
                if n_plots == 1:
                    axes = [axes]

                for ax, (name, data) in zip(axes, valid_results.items()):
                    sv = data['top_singular_values']
                    ax.bar(range(len(sv)), sv)
                    ax.set_title(f'{data["display"]} Singular Values')
                    ax.set_xlabel('Index')
                    ax.set_ylabel('Value')

                plt.tight_layout()
                path = os.path.join(output_dir, 'weight_svd.png')
                plt.savefig(path, dpi=150)
                plt.close()
                results['visualization'] = path

        return results


# ============================================================
# 5. Behavioral Analysis
# ============================================================

class BehavioralAnalyzer:
    """Token-level behavioral analysis"""

    def __init__(self, model, router, tokenizer, device='cuda'):
        self.model = model
        self.router = router
        self.tokenizer = tokenizer
        self.device = device

    def analyze_single_neuron(self, neuron_id: int, neuron_type: str) -> Dict:
        """Analyze a single neuron"""
        results = {
            'neuron_type': neuron_type,
            'neuron_id': neuron_id,
        }

        # Get EMA
        type_info = NEURON_TYPES.get(neuron_type)
        if type_info:
            ema_attr = type_info[1]
            if hasattr(self.router, ema_attr):
                ema = getattr(self.router, ema_attr)
                if neuron_id < len(ema):
                    results['usage_ema'] = float(ema[neuron_id])
                    tau = self.router.tau
                    exc = max(0, min(1, 1.0 - ema[neuron_id].item() / tau))
                    results['excitability'] = float(exc)

        # Get embedding
        emb = self.router.neuron_emb.detach().cpu().numpy()

        offset = 0
        for name, (_, _, n_attr, _) in NEURON_TYPES.items():
            if hasattr(self.router, n_attr):
                n = getattr(self.router, n_attr)
                if name == neuron_type:
                    if neuron_id < n:
                        neuron_emb = emb[offset + neuron_id]
                        results['embedding_norm'] = float(np.linalg.norm(neuron_emb))
                        results['embedding_mean'] = float(neuron_emb.mean())
                        results['embedding_std'] = float(neuron_emb.std())
                    break
                offset += n

        return results

    def analyze_token_trajectory(self, dataloader, n_batches: int = 20) -> Dict:
        """Analyze how routing changes across sequence positions"""
        position_routing = defaultdict(lambda: defaultdict(list))

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Trajectory')):
                if i >= n_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                outputs = self.model(input_ids, return_routing_info=True)

                if not isinstance(outputs, tuple) or len(outputs) < 2:
                    continue

                routing_infos = outputs[-1]
                if not routing_infos:
                    continue

                attn = routing_infos[0].get('attention', {})

                for key, (_, pref_key, _, _) in ROUTING_KEYS.items():
                    pref = attn.get(pref_key)
                    if pref is None:
                        continue

                    if pref.dim() == 3:  # [B, S, N]
                        for pos in range(min(pref.shape[1], 128)):
                            ent = calc_entropy_ratio(pref[:, pos, :])
                            position_routing[key][pos].append(ent)

        results = {}
        for key in ROUTING_KEYS.keys():
            if position_routing[key]:
                pos_avg = {}
                for pos, values in position_routing[key].items():
                    pos_avg[pos] = float(np.mean(values))
                results[key] = {
                    'display': ROUTING_KEYS[key][0],
                    'position_entropy': pos_avg,
                    'early_avg': float(np.mean([v for p, v in pos_avg.items() if p < 10])),
                    'late_avg': float(np.mean([v for p, v in pos_avg.items() if p >= 10])),
                }

        return results

    def run_probing(self, dataloader, max_batches: int = 50) -> Dict:
        """Probing classifier for POS prediction"""
        if not HAS_SKLEARN:
            return {'error': 'sklearn not available'}

        X_data = {key: [] for key in ROUTING_KEYS.keys()}
        y_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc='Probing', total=max_batches)):
                if batch_idx >= max_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(self.device)

                try:
                    outputs = self.model(input_ids, return_routing_info=True)
                    if not isinstance(outputs, tuple) or len(outputs) < 2:
                        continue
                    routing_infos = outputs[-1]
                    routing_info = routing_infos[0].get('attention', {}) if routing_infos else {}
                except:
                    continue

                for key, (_, _, weight_key, _) in ROUTING_KEYS.items():
                    if weight_key in routing_info:
                        w = routing_info[weight_key]
                        if w.dim() == 3:
                            for b in range(w.shape[0]):
                                for s in range(w.shape[1]):
                                    if attention_mask[b, s] == 0:
                                        continue
                                    X_data[key].append(w[b, s].cpu().numpy())

                for b in range(input_ids.shape[0]):
                    for s in range(input_ids.shape[1]):
                        if attention_mask[b, s] == 0:
                            continue
                        token = self.tokenizer.decode([input_ids[b, s].item()])
                        pos = simple_pos_tag(token)
                        y_labels.append(pos)

        results = {}

        for key in X_data:
            X = np.array(X_data[key])
            y = np.array(y_labels[:len(X)])

            if len(X) < 100 or len(np.unique(y)) < 2:
                results[key] = {'error': 'Not enough data'}
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            clf = LogisticRegression(max_iter=500, multi_class='multinomial', random_state=42)
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                results[key] = {
                    'display': ROUTING_KEYS[key][0],
                    'accuracy': float(acc),
                    'n_samples': len(X),
                    'n_classes': len(np.unique(y)),
                    'classes': list(np.unique(y)),
                }
            except Exception as e:
                results[key] = {'error': str(e)}

        return results

    def run_ablation(self, dataloader, max_batches: int = 20) -> Dict:
        """Ablation study - measure impact of zeroing out neurons"""
        results = {'baseline_loss': 0.0}

        # Get baseline loss
        total_loss = 0
        n_batches = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                labels = input_ids.clone()

                try:
                    outputs = self.model(input_ids, labels=labels)
                    if isinstance(outputs, tuple):
                        loss = outputs[0]
                    else:
                        loss = outputs.get('loss', torch.tensor(0.0))
                    total_loss += loss.item()
                    n_batches += 1
                except:
                    continue

        baseline_loss = total_loss / max(n_batches, 1)
        results['baseline_loss'] = baseline_loss

        # Ablate top neurons per type
        for name, (display, ema_attr, n_attr, _) in NEURON_TYPES.items():
            if not hasattr(self.router, ema_attr):
                continue

            ema = getattr(self.router, ema_attr)
            top_indices = torch.topk(ema, min(5, len(ema)))[1].tolist()

            ablation_results = []
            for neuron_idx in top_indices:
                # Store original and zero out
                original_ema = ema[neuron_idx].item()
                ema[neuron_idx] = 0.0

                ablated_loss = 0
                n_batches = 0

                with torch.no_grad():
                    for batch_idx, batch in enumerate(dataloader):
                        if batch_idx >= max_batches // 2:
                            break

                        input_ids = batch['input_ids'].to(self.device)
                        labels = input_ids.clone()

                        try:
                            outputs = self.model(input_ids, labels=labels)
                            if isinstance(outputs, tuple):
                                loss = outputs[0]
                            else:
                                loss = outputs.get('loss', torch.tensor(0.0))
                            ablated_loss += loss.item()
                            n_batches += 1
                        except:
                            continue

                # Restore
                ema[neuron_idx] = original_ema

                if n_batches > 0:
                    ablated_loss /= n_batches
                    delta = ablated_loss - baseline_loss

                    ablation_results.append({
                        'neuron_id': neuron_idx,
                        'original_ema': original_ema,
                        'ablated_loss': ablated_loss,
                        'loss_delta': delta,
                        'importance': abs(delta) / max(baseline_loss, 0.001),
                    })

            results[name] = sorted(ablation_results, key=lambda x: -abs(x['loss_delta']))

        return results


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
        self.weight = WeightAnalyzer(self.neurons)
        self.behavioral = BehavioralAnalyzer(model, self.router, tokenizer, device)

        print("DAWN Analyzer initialized")

    def run_usage_analysis(self, output_dir: str = None) -> Dict:
        """Usage analysis"""
        print("\n" + "="*60)
        print("USAGE ANALYSIS")
        print("="*60)

        results = {
            'ema_distribution': self.usage.analyze_ema_distribution(),
            'excitability': self.usage.analyze_excitability(),
            'diversity': self.usage.analyze_diversity(),
            'dead_neurons': self.usage.analyze_dead_neurons(output_dir),
        }

        if output_dir:
            viz = self.usage.visualize_usage(output_dir)
            results['usage_visualization'] = viz.get('visualization')

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
            if name in ['recommendation', 'visualization']:
                continue
            if isinstance(data, dict) and 'display' in data:
                print(f"  {data['display']:>4}: {data['n_dead']} dead, {data['n_revivable']} revivable, {data['n_removable']} removable")
        print(f"\n  Recommendation: {dead['recommendation']['action']} ({dead['recommendation']['total_removable']} removable)")

        return results

    def run_routing_analysis(self, dataloader, n_batches: int = 50, output_dir: str = None) -> Dict:
        """Routing analysis"""
        print("\n" + "="*60)
        print("ROUTING ANALYSIS")
        print("="*60)

        results = {
            'entropy': self.routing.analyze_entropy(dataloader, n_batches),
            'selection': self.routing.analyze_selection_frequency(dataloader, n_batches),
            'selection_diversity': self.routing.analyze_selection_diversity(dataloader, n_batches),
            'qk_overlap': self.routing.analyze_qk_overlap(dataloader, n_batches),
        }

        print("\n--- Routing Entropy ---")
        for key, data in results['entropy'].items():
            print(f"  {data['display']:>6}: {data['mean_entropy']:.1f}% (std={data['std_entropy']:.1f})")

        print("\n--- Selection Coverage ---")
        for key, data in results['selection'].items():
            print(f"  {data['display']:>6}: {data['unique_selected']}/{data['coverage']:.1%} coverage, top10 conc={data['concentration']:.1%}")

        print("\n--- Selection Diversity ---")
        for key, data in results['selection_diversity'].items():
            if key == 'summary':
                continue
            print(f"  {data['display']:>6}: batch_avg={data['per_batch_avg']:.1f}, union={data['union_count']}, diversity_ratio={data['diversity_ratio']:.2f}")

        print("\n--- Q/K Overlap ---")
        for key, data in results['qk_overlap'].items():
            print(f"  {key.upper()}: {data['mean_overlap']:.2f} - {data['interpretation']}")

        return results

    def run_embedding_analysis(self, output_dir: str = None) -> Dict:
        """Embedding analysis"""
        print("\n" + "="*60)
        print("EMBEDDING ANALYSIS")
        print("="*60)

        results = {
            'intra_similarity': self.embedding.analyze_similarity(output_dir),
            'cross_similarity': self.embedding.analyze_cross_type_similarity(),
        }

        print("\n--- Intra-type Similarity ---")
        for name, data in results['intra_similarity'].items():
            if name == 'visualization':
                continue
            print(f"  {data['display']:>4}: avg={data['avg_similarity']:.3f}, max={data['max_similarity']:.3f}")

        print("\n--- Cross-type Similarity (centroids) ---")
        for key, sim in results['cross_similarity'].items():
            print(f"  {key}: {sim:.3f}")

        if output_dir:
            viz_path = self.embedding.visualize(output_dir)
            if viz_path:
                results['visualization'] = viz_path
                print(f"\n  Visualization saved: {viz_path}")

        return results

    def run_weight_analysis(self, output_dir: str = None) -> Dict:
        """Weight SVD analysis"""
        print("\n" + "="*60)
        print("WEIGHT SVD ANALYSIS")
        print("="*60)

        results = self.weight.analyze_weight_svd(output_dir)

        for name, data in results.items():
            if name == 'visualization' or 'error' in str(data):
                continue
            print(f"\n{data.get('display', name)}:")
            print(f"  Effective rank: {data['effective_rank']:.2f}")
            print(f"  Var explained by top 5: {data['var_explained_by_top5']:.2%}")
            print(f"  Condition number: {data['condition_number']:.2f}")

        return results

    def run_clustering_analysis(self, n_clusters: int = 5, output_dir: str = None) -> Dict:
        """Clustering analysis"""
        print("\n" + "="*60)
        print("CLUSTERING ANALYSIS")
        print("="*60)

        results = self.embedding.analyze_clustering(n_clusters, output_dir)

        for name, data in results.items():
            if name == 'visualization' or 'error' in str(data):
                continue
            print(f"\n{data['display']}: {data['n_clusters']} clusters")
            for c in data.get('clusters', [])[:3]:
                print(f"  Cluster {c['cluster_id']}: size={c['size']}, active={c.get('active_count', 'N/A')}, usage={c.get('avg_usage', 0):.4f}")

        return results

    def run_trajectory_analysis(self, dataloader, n_batches: int = 20, output_dir: str = None) -> Dict:
        """Token trajectory analysis"""
        print("\n" + "="*60)
        print("TOKEN TRAJECTORY ANALYSIS")
        print("="*60)

        results = self.behavioral.analyze_token_trajectory(dataloader, n_batches)

        for key, data in results.items():
            print(f"\n{data['display']}:")
            print(f"  Early positions (0-9) avg entropy: {data['early_avg']:.1f}%")
            print(f"  Late positions (10+) avg entropy: {data['late_avg']:.1f}%")

        return results

    def run_probing_analysis(self, dataloader, max_batches: int = 50, output_dir: str = None) -> Dict:
        """Probing classifier analysis"""
        print("\n" + "="*60)
        print("PROBING CLASSIFIER")
        print("="*60)

        results = self.behavioral.run_probing(dataloader, max_batches)

        for key, data in results.items():
            if 'error' in data:
                print(f"{key}: {data['error']}")
            else:
                print(f"{data['display']}: accuracy={data['accuracy']:.2%}, samples={data['n_samples']}")

        return results

    def run_ablation_analysis(self, dataloader, max_batches: int = 20, output_dir: str = None) -> Dict:
        """Ablation study"""
        print("\n" + "="*60)
        print("ABLATION STUDY")
        print("="*60)

        results = self.behavioral.run_ablation(dataloader, max_batches)

        print(f"Baseline loss: {results['baseline_loss']:.4f}")
        for name, data in results.items():
            if name == 'baseline_loss' or not isinstance(data, list):
                continue
            print(f"\n{NEURON_TYPES.get(name, (name,))[0]}:")
            for r in data[:3]:
                print(f"  Neuron {r['neuron_id']}: delta={r['loss_delta']:.4f}, importance={r['importance']:.2%}")

        return results

    def analyze_single_neuron(self, neuron_id: int, neuron_type: str) -> Dict:
        """Single neuron analysis"""
        print("\n" + "="*60)
        print(f"SINGLE NEURON ANALYSIS: {neuron_type} #{neuron_id}")
        print("="*60)

        results = self.behavioral.analyze_single_neuron(neuron_id, neuron_type)

        print(f"Usage EMA: {results.get('usage_ema', 'N/A'):.4f}")
        print(f"Excitability: {results.get('excitability', 'N/A'):.4f}")
        print(f"Embedding norm: {results.get('embedding_norm', 'N/A'):.4f}")

        return results

    def run_all(self, dataloader=None, output_dir: str = './dawn_analysis') -> Dict:
        """Run all analyses"""
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'usage': self.run_usage_analysis(output_dir),
            'embedding': self.run_embedding_analysis(output_dir),
            'weight_svd': self.run_weight_analysis(output_dir),
            'clustering': self.run_clustering_analysis(output_dir=output_dir),
        }

        if dataloader:
            results['routing'] = self.run_routing_analysis(dataloader, output_dir=output_dir)
            results['trajectory'] = self.run_trajectory_analysis(dataloader, output_dir=output_dir)

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

    if data_path.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(data_path)
        texts = df['text'].tolist()[:10000]
    elif data_path.endswith('.json'):
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
                        choices=['all', 'usage', 'routing', 'embedding', 'weight_svd',
                                'clustering', 'trajectory', 'probing', 'ablation', 'neuron'],
                        help='Analysis mode')
    parser.add_argument('--val_data', type=str, default=None, help='Validation data path')
    parser.add_argument('--output_dir', type=str, default='./dawn_analysis', help='Output directory')
    parser.add_argument('--n_batches', type=int, default=50, help='Number of batches')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters')
    parser.add_argument('--neuron_type', type=str, default='feature_qk', help='Neuron type for single neuron analysis')
    parser.add_argument('--neuron_id', type=int, default=0, help='Neuron ID for single neuron analysis')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, tokenizer, config = load_model(args.checkpoint, device)

    analyzer = DAWNAnalyzer(model, tokenizer, device)

    dataloader = None
    if args.val_data and args.mode in ['all', 'routing', 'trajectory', 'probing', 'ablation']:
        dataloader = create_dataloader(args.val_data, tokenizer, args.batch_size)

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
    elif args.mode == 'weight_svd':
        results = analyzer.run_weight_analysis(args.output_dir)
    elif args.mode == 'clustering':
        results = analyzer.run_clustering_analysis(args.n_clusters, args.output_dir)
    elif args.mode == 'trajectory':
        if dataloader is None:
            print("ERROR: --val_data required for trajectory analysis")
            return
        results = analyzer.run_trajectory_analysis(dataloader, args.n_batches, args.output_dir)
    elif args.mode == 'probing':
        if dataloader is None:
            print("ERROR: --val_data required for probing analysis")
            return
        results = analyzer.run_probing_analysis(dataloader, args.n_batches, args.output_dir)
    elif args.mode == 'ablation':
        if dataloader is None:
            print("ERROR: --val_data required for ablation analysis")
            return
        results = analyzer.run_ablation_analysis(dataloader, args.n_batches, args.output_dir)
    elif args.mode == 'neuron':
        results = analyzer.analyze_single_neuron(args.neuron_id, args.neuron_type)

    output_path = os.path.join(args.output_dir, f'dawn_{args.mode}.json')
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"\n{'='*60}")
    print(f"Analysis complete! Results: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
