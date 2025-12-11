#!/usr/bin/env python3
"""
DAWN v16 Analysis Script
========================

v16 Split Feature R/V 모델 전용 분석 도구.

Features:
1. Neuron Usage - FR/FV/R/V/K 타입별 활성화 분석
2. Excitability - Langevin dynamics 상태
3. Gini Coefficient - 뉴런 활용 불균형
4. Neuron Embedding - 뉴런 임베딩 시각화
5. Word-Neuron Mapping - 단어별 뉴런 활성화
6. Knowledge Health - Knowledge 뉴런 상태

Usage:
    python scripts/analyze_v16.py --checkpoint <path>
    python scripts/analyze_v16.py --checkpoint <path> --val_data <path> --mode all
    python scripts/analyze_v16.py --checkpoint <path> --mode usage
    python scripts/analyze_v16.py --checkpoint <path> --mode embedding
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
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
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


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
    cumsum = torch.cumsum(x_sorted, dim=0)
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


# ============================================================
# Model Loading
# ============================================================

def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load v16 model from checkpoint"""
    from transformers import BertTokenizer
    from models import create_model_by_version

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
# Analysis Functions
# ============================================================

class V16Analyzer:
    """v16 모델 분석기"""

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

    # --------------------------------------------------
    # 1. Usage Analysis
    # --------------------------------------------------
    def analyze_usage(self) -> Dict:
        """Analyze neuron usage patterns from EMA buffers"""
        if self.router is None:
            return {'error': 'No router found'}

        results = {}
        threshold = 0.01

        # Feature R
        ema_fr = self.router.usage_ema_feature_r
        active_fr = (ema_fr > threshold).sum().item()
        results['feature_r'] = {
            'total': self.router.n_feature_r,
            'active': int(active_fr),
            'ratio': active_fr / self.router.n_feature_r,
            'gini': gini_coefficient(ema_fr),
            'ema_stats': {
                'min': ema_fr.min().item(),
                'max': ema_fr.max().item(),
                'mean': ema_fr.mean().item(),
                'std': ema_fr.std().item(),
            }
        }

        # Feature V
        ema_fv = self.router.usage_ema_feature_v
        active_fv = (ema_fv > threshold).sum().item()
        results['feature_v'] = {
            'total': self.router.n_feature_v,
            'active': int(active_fv),
            'ratio': active_fv / self.router.n_feature_v,
            'gini': gini_coefficient(ema_fv),
            'ema_stats': {
                'min': ema_fv.min().item(),
                'max': ema_fv.max().item(),
                'mean': ema_fv.mean().item(),
                'std': ema_fv.std().item(),
            }
        }

        # Relational
        ema_r = self.router.usage_ema_relational
        active_r = (ema_r > threshold).sum().item()
        results['relational'] = {
            'total': self.router.n_relational,
            'active': int(active_r),
            'ratio': active_r / self.router.n_relational,
            'gini': gini_coefficient(ema_r),
            'ema_stats': {
                'min': ema_r.min().item(),
                'max': ema_r.max().item(),
                'mean': ema_r.mean().item(),
                'std': ema_r.std().item(),
            }
        }

        # Value
        ema_v = self.router.usage_ema_value
        active_v = (ema_v > threshold).sum().item()
        results['value'] = {
            'total': self.router.n_value,
            'active': int(active_v),
            'ratio': active_v / self.router.n_value,
            'gini': gini_coefficient(ema_v),
            'ema_stats': {
                'min': ema_v.min().item(),
                'max': ema_v.max().item(),
                'mean': ema_v.mean().item(),
                'std': ema_v.std().item(),
            }
        }

        # Knowledge
        ema_k = self.router.usage_ema_knowledge
        active_k = (ema_k > threshold).sum().item()
        results['knowledge'] = {
            'total': self.router.n_knowledge,
            'active': int(active_k),
            'ratio': active_k / self.router.n_knowledge,
            'gini': gini_coefficient(ema_k),
            'ema_stats': {
                'min': ema_k.min().item(),
                'max': ema_k.max().item(),
                'mean': ema_k.mean().item(),
                'std': ema_k.std().item(),
            }
        }

        return results

    # --------------------------------------------------
    # 2. Excitability Analysis
    # --------------------------------------------------
    def analyze_excitability(self) -> Dict:
        """Analyze excitability state"""
        if self.router is None:
            return {'error': 'No router found'}

        tau = self.router.tau
        weight = self.router.excitability_weight.item() if hasattr(self.router.excitability_weight, 'item') else self.router.excitability_weight

        results = {
            'tau': tau,
            'weight': weight,
            'langevin_alpha': self.router.langevin_alpha,
            'langevin_beta': self.router.langevin_beta,
        }

        # Per-type excitability
        for name, ema in [
            ('feature_r', self.router.usage_ema_feature_r),
            ('feature_v', self.router.usage_ema_feature_v),
            ('relational', self.router.usage_ema_relational),
            ('value', self.router.usage_ema_value),
            ('knowledge', self.router.usage_ema_knowledge),
        ]:
            exc = torch.clamp(1.0 - ema / tau, min=0.0, max=1.0)
            results[f'{name}_excitability'] = {
                'min': exc.min().item(),
                'max': exc.max().item(),
                'mean': exc.mean().item(),
            }

        return results

    # --------------------------------------------------
    # 3. Neuron Embedding Analysis
    # --------------------------------------------------
    def analyze_embeddings(self, output_dir: str = None) -> Dict:
        """Analyze neuron embeddings with t-SNE/PCA"""
        if self.router is None:
            return {'error': 'No router found'}

        emb = self.router.neuron_emb.detach().cpu().numpy()  # [total, d_space]

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

            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb)-1))
            emb_2d = tsne.fit_transform(emb)

            # PCA
            pca = PCA(n_components=2)
            emb_pca = pca.fit_transform(emb)

            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            colors = {'FR': 'red', 'FV': 'orange', 'R': 'blue', 'V': 'green', 'K': 'purple'}

            for ax, data, title in [(axes[0], emb_2d, 't-SNE'), (axes[1], emb_pca, 'PCA')]:
                for t in colors:
                    mask = [l == t for l in labels]
                    ax.scatter(data[mask, 0], data[mask, 1], c=colors[t], label=t, alpha=0.6, s=20)
                ax.set_title(f'Neuron Embeddings ({title})')
                ax.legend()
                ax.set_xlabel('Dim 1')
                ax.set_ylabel('Dim 2')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'neuron_embeddings.png'), dpi=150)
            plt.close()

            results['visualization'] = os.path.join(output_dir, 'neuron_embeddings.png')

        return results

    # --------------------------------------------------
    # 4. Word-Neuron Mapping (requires data)
    # --------------------------------------------------
    def analyze_word_neuron_mapping(self, dataloader, max_batches: int = 100) -> Dict:
        """Analyze which words activate which neurons"""
        if self.router is None:
            return {'error': 'No router found'}

        # Token -> neuron activation counts
        token_neuron_counts = {
            'feature_r': defaultdict(lambda: defaultdict(float)),
            'feature_v': defaultdict(lambda: defaultdict(float)),
            'relational': defaultdict(lambda: defaultdict(float)),
            'value': defaultdict(lambda: defaultdict(float)),
        }

        token_counts = Counter()

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc='Word-Neuron', total=max_batches)):
                if batch_idx >= max_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Forward pass to get routing info
                try:
                    outputs = self.model(input_ids, attention_mask=attention_mask, return_routing_info=True)
                    routing_info = outputs.get('routing_info', {})
                except Exception as e:
                    print(f"Forward pass error: {e}")
                    continue

                # Decode tokens
                for b in range(input_ids.shape[0]):
                    for s in range(input_ids.shape[1]):
                        if attention_mask[b, s] == 0:
                            continue

                        token_id = input_ids[b, s].item()
                        token = self.tokenizer.decode([token_id])
                        token_counts[token] += 1

                        # Get neuron activations for this token
                        for neuron_type, key in [
                            ('feature_r', 'feature_r_weights'),
                            ('feature_v', 'feature_v_weights'),
                            ('relational', 'relational_weights_Q'),
                            ('value', 'value_weights'),
                        ]:
                            if key in routing_info:
                                weights = routing_info[key]
                                if weights.dim() == 3:  # [B, S, N]
                                    w = weights[b, s].cpu().numpy()
                                    top_neurons = np.argsort(w)[-5:]  # top 5
                                    for n in top_neurons:
                                        token_neuron_counts[neuron_type][token][int(n)] += w[n]

        # Aggregate results: top tokens per neuron
        results = {}
        for neuron_type in token_neuron_counts:
            neuron_top_tokens = defaultdict(list)
            for token, neuron_weights in token_neuron_counts[neuron_type].items():
                for neuron_id, weight in neuron_weights.items():
                    neuron_top_tokens[neuron_id].append((token, weight))

            # Sort and keep top 10 tokens per neuron
            for neuron_id in neuron_top_tokens:
                neuron_top_tokens[neuron_id] = sorted(
                    neuron_top_tokens[neuron_id], key=lambda x: x[1], reverse=True
                )[:10]

            results[neuron_type] = dict(neuron_top_tokens)

        return results

    # --------------------------------------------------
    # 5. Usage Histogram Visualization
    # --------------------------------------------------
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

        # Last subplot: summary bar chart
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

    # --------------------------------------------------
    # 6. Full Report
    # --------------------------------------------------
    def generate_report(self, output_dir: str = None) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'usage': self.analyze_usage(),
            'excitability': self.analyze_excitability(),
        }

        if output_dir:
            report['embeddings'] = self.analyze_embeddings(output_dir)
            report['usage_viz'] = self.visualize_usage(output_dir)

        return report


# ============================================================
# Main
# ============================================================

def print_usage_summary(usage: Dict):
    """Print usage summary to console"""
    print("\n" + "="*60)
    print("NEURON USAGE SUMMARY")
    print("="*60)

    headers = ['Type', 'Active', 'Total', 'Ratio', 'Gini', 'EMA Mean']
    print(f"{'Type':<12} {'Active':>8} {'Total':>8} {'Ratio':>8} {'Gini':>8} {'Mean':>10}")
    print("-"*60)

    for key in ['feature_r', 'feature_v', 'relational', 'value', 'knowledge']:
        d = usage[key]
        print(f"{key:<12} {d['active']:>8} {d['total']:>8} {d['ratio']:>8.2%} {d['gini']:>8.2f} {d['ema_stats']['mean']:>10.4f}")


def print_excitability_summary(exc: Dict):
    """Print excitability summary to console"""
    print("\n" + "="*60)
    print("EXCITABILITY STATE")
    print("="*60)
    print(f"tau: {exc['tau']:.2f}")
    print(f"weight: {exc['weight']:.4f}")
    print(f"langevin_alpha: {exc['langevin_alpha']}")
    print(f"langevin_beta: {exc['langevin_beta']}")
    print()

    print(f"{'Type':<12} {'Min':>8} {'Mean':>8} {'Max':>8}")
    print("-"*40)
    for key in ['feature_r', 'feature_v', 'relational', 'value', 'knowledge']:
        d = exc[f'{key}_excitability']
        print(f"{key:<12} {d['min']:>8.2f} {d['mean']:>8.2f} {d['max']:>8.2f}")


def main():
    parser = argparse.ArgumentParser(description='DAWN v16 Analysis')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--val_data', help='Validation data path')
    parser.add_argument('--output_dir', default='./analysis_v16', help='Output directory')
    parser.add_argument('--mode', default='all', choices=['all', 'usage', 'excitability', 'embedding', 'word_neuron'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_batches', type=int, default=100, help='Max batches for word-neuron analysis')
    args = parser.parse_args()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, config = load_model(args.checkpoint, args.device)

    # Create analyzer
    analyzer = V16Analyzer(model, tokenizer, args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Run analysis
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
        print("\n" + "="*60)
        print("EMBEDDING ANALYSIS")
        print("="*60)
        print(f"Total neurons: {emb['total_neurons']}")
        print(f"Embedding dim: {emb['embedding_dim']}")
        print("\nType similarity (cosine):")
        for k, v in emb.get('type_similarity', {}).items():
            print(f"  {k}: {v:.3f}")
        with open(os.path.join(args.output_dir, 'embeddings.json'), 'w') as f:
            json.dump(emb, f, indent=2)

    if args.mode in ['all']:
        viz = analyzer.visualize_usage(args.output_dir)
        print(f"\nUsage visualization saved to: {viz.get('visualization', 'N/A')}")

    if args.mode == 'word_neuron' and args.val_data:
        from utils.data import TextDataset, collate_fn_dynamic_padding
        from torch.utils.data import DataLoader

        dataset = TextDataset(args.val_data, tokenizer, max_length=128)
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_dynamic_padding)

        word_neuron = analyzer.analyze_word_neuron_mapping(dataloader, args.max_batches)
        with open(os.path.join(args.output_dir, 'word_neuron.json'), 'w') as f:
            json.dump(word_neuron, f, indent=2)
        print(f"\nWord-neuron mapping saved to: {os.path.join(args.output_dir, 'word_neuron.json')}")

    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
