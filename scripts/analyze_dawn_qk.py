#!/usr/bin/env python3
"""
DAWN Q/K Usage Pattern Analysis
================================
Q/K neuron usage pattern analysis for v17.1:
1. Per-neuron Q selection count vs K selection count
2. Q/K overlap analysis
3. Scatter plot visualization

Usage:
    python analyze_dawn_qk.py --checkpoint path/to/ckpt --val_data path/to/data
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# v17.1 Q/K Routing Configuration
QK_POOLS = {
    'feature_qk': {
        'display': 'FQK',
        'q_pref': 'fqk_q_pref',
        'k_pref': 'fqk_k_pref',
        'q_weight': 'fqk_weights_Q',
        'k_weight': 'fqk_weights_K',
        'n_attr': 'n_feature_qk',
        'color': 'red',
    },
    'restore_qk': {
        'display': 'RQK',
        'q_pref': 'rqk_q_pref',
        'k_pref': 'rqk_k_pref',
        'q_weight': 'rqk_weights_Q',
        'k_weight': 'rqk_weights_K',
        'n_attr': 'n_restore_qk',
        'color': 'blue',
    },
}


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load DAWN v17.1 model"""
    from models import create_model_by_version
    from transformers import BertTokenizer

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


def create_dataloader(data_path: str, tokenizer, batch_size: int = 32):
    """Create dataloader"""
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
        raise ValueError(f"Unsupported format: {data_path}")

    dataset = TextDataset(texts, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class QKAnalyzer:
    """Q/K Usage Pattern Analyzer for DAWN v17.1"""

    def __init__(self, model, router, tokenizer, device='cuda'):
        self.model = model
        self.router = router
        self.tokenizer = tokenizer
        self.device = device

    def analyze_qk_usage(self, dataloader, n_batches: int = 100) -> dict:
        """
        Analyze per-neuron Q/K selection counts

        Returns:
            - q_counts: Q selection count per neuron
            - k_counts: K selection count per neuron
            - correlation: Q/K correlation coefficient
            - specialization: Q-only, K-only, shared, inactive counts
        """
        results = {}

        for pool_name, pool_info in QK_POOLS.items():
            n_neurons = getattr(self.router, pool_info['n_attr'], 0)
            if n_neurons == 0:
                continue

            q_counts = torch.zeros(n_neurons, device=self.device)
            k_counts = torch.zeros(n_neurons, device=self.device)
            batch_overlaps = []

            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(dataloader, desc=f'{pool_info["display"]} Q/K', total=n_batches)):
                    if i >= n_batches:
                        break

                    input_ids = batch['input_ids'].to(self.device)

                    try:
                        outputs = self.model(input_ids, return_routing_info=True)
                        if not isinstance(outputs, tuple) or len(outputs) < 2:
                            continue
                        routing_infos = outputs[-1]
                        if not routing_infos:
                            continue
                        attn = routing_infos[0].get('attention', {})
                    except:
                        continue

                    # Get Q/K weights
                    w_q = attn.get(pool_info['q_weight'])
                    w_k = attn.get(pool_info['k_weight'])

                    if w_q is None or w_k is None:
                        # Try preference tensors
                        w_q = attn.get(pool_info['q_pref'])
                        w_k = attn.get(pool_info['k_pref'])
                        if w_q is None or w_k is None:
                            continue

                    # Handle different shapes
                    if w_q.dim() == 3:  # [B, S, N]
                        selected_q = (w_q > 0).float().sum(dim=[0, 1])
                        selected_k = (w_k > 0).float().sum(dim=[0, 1])
                    else:  # [B, N]
                        selected_q = (w_q > 0).float().sum(dim=0)
                        selected_k = (w_k > 0).float().sum(dim=0)

                    q_counts += selected_q
                    k_counts += selected_k

                    # Calculate batch overlap
                    if w_q.dim() >= 2:
                        overlap = ((w_q > 0) & (w_k > 0)).float()
                        active_q = (w_q > 0).float().sum(-1)
                        overlap_ratio = (overlap.sum(-1) / (active_q + 1e-8)).mean().item()
                        batch_overlaps.append(overlap_ratio)

            # Calculate statistics
            q_np = q_counts.cpu().numpy()
            k_np = k_counts.cpu().numpy()

            # Correlation
            if q_np.sum() > 0 and k_np.sum() > 0:
                corr = float(np.corrcoef(q_np, k_np)[0, 1])
            else:
                corr = 0.0

            # Specialization analysis
            threshold = (q_np.sum() + k_np.sum()) / (2 * len(q_np)) * 0.1
            q_only = int(((q_np > threshold) & (k_np < threshold)).sum())
            k_only = int(((k_np > threshold) & (q_np < threshold)).sum())
            shared = int(((q_np > threshold) & (k_np > threshold)).sum())
            inactive = int(((q_np < threshold) & (k_np < threshold)).sum())

            results[pool_name] = {
                'display': pool_info['display'],
                'n_neurons': n_neurons,
                'q_counts': q_np.tolist(),
                'k_counts': k_np.tolist(),
                'correlation': corr,
                'avg_overlap': float(np.mean(batch_overlaps)) if batch_overlaps else 0,
                'std_overlap': float(np.std(batch_overlaps)) if batch_overlaps else 0,
                'q_specialized': q_only,
                'k_specialized': k_only,
                'shared': shared,
                'inactive': inactive,
                'q_total': int(q_np.sum()),
                'k_total': int(k_np.sum()),
            }

        results['n_batches'] = min(n_batches, i + 1) if 'i' in dir() else 0
        return results

    def analyze_qk_entropy(self, dataloader, n_batches: int = 50) -> dict:
        """Analyze Q/K routing entropy patterns"""
        results = {}

        for pool_name, pool_info in QK_POOLS.items():
            q_entropy = []
            k_entropy = []

            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(dataloader, desc=f'{pool_info["display"]} Entropy', total=n_batches)):
                    if i >= n_batches:
                        break

                    input_ids = batch['input_ids'].to(self.device)

                    try:
                        outputs = self.model(input_ids, return_routing_info=True)
                        if not isinstance(outputs, tuple) or len(outputs) < 2:
                            continue
                        routing_infos = outputs[-1]
                        if not routing_infos:
                            continue
                        attn = routing_infos[0].get('attention', {})
                    except:
                        continue

                    # Get preference tensors
                    pref_q = attn.get(pool_info['q_pref'])
                    pref_k = attn.get(pool_info['k_pref'])

                    if pref_q is not None:
                        p = pref_q.mean(dim=0) if pref_q.dim() > 1 else pref_q
                        p = p.clamp(min=1e-8)
                        ent = -torch.sum(p * torch.log(p)).item()
                        max_ent = np.log(pref_q.shape[-1])
                        q_entropy.append(ent / max_ent * 100)

                    if pref_k is not None:
                        p = pref_k.mean(dim=0) if pref_k.dim() > 1 else pref_k
                        p = p.clamp(min=1e-8)
                        ent = -torch.sum(p * torch.log(p)).item()
                        max_ent = np.log(pref_k.shape[-1])
                        k_entropy.append(ent / max_ent * 100)

            if q_entropy and k_entropy:
                results[pool_name] = {
                    'display': pool_info['display'],
                    'q_entropy_mean': float(np.mean(q_entropy)),
                    'q_entropy_std': float(np.std(q_entropy)),
                    'k_entropy_mean': float(np.mean(k_entropy)),
                    'k_entropy_std': float(np.std(k_entropy)),
                    'entropy_diff': float(np.mean(q_entropy) - np.mean(k_entropy)),
                }

        return results

    def visualize(self, usage_results: dict, output_dir: str) -> str:
        """Visualize Q/K usage patterns"""
        if not HAS_MATPLOTLIB:
            return None

        os.makedirs(output_dir, exist_ok=True)

        n_pools = len([k for k in usage_results if k not in ['n_batches']])
        fig, axes = plt.subplots(n_pools, 3, figsize=(18, 6 * n_pools))
        if n_pools == 1:
            axes = axes.reshape(1, -1)

        row = 0
        for pool_name, data in usage_results.items():
            if pool_name == 'n_batches':
                continue

            q_counts = np.array(data['q_counts'])
            k_counts = np.array(data['k_counts'])

            # 1. Scatter: Q vs K
            ax = axes[row, 0]
            ax.scatter(q_counts, k_counts, alpha=0.6, s=30, c=QK_POOLS[pool_name]['color'])
            max_val = max(q_counts.max(), k_counts.max())
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Q=K')
            ax.set_xlabel('Q Selection Count')
            ax.set_ylabel('K Selection Count')
            ax.set_title(f'{data["display"]}: Q vs K Usage\n(corr={data["correlation"]:.3f})')
            ax.legend()

            # 2. Bar: Specialization
            ax = axes[row, 1]
            categories = ['Q-only', 'K-only', 'Shared', 'Inactive']
            values = [data['q_specialized'], data['k_specialized'], data['shared'], data['inactive']]
            colors = ['blue', 'orange', 'green', 'gray']
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            ax.set_ylabel('Neuron Count')
            ax.set_title(f'{data["display"]}: Neuron Specialization')
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(val), ha='center', va='bottom')

            # 3. Histogram: Q/(Q+K) ratio
            ax = axes[row, 2]
            total = q_counts + k_counts + 1e-8
            q_ratio = q_counts / total
            active_mask = (q_counts + k_counts) > 0
            if active_mask.sum() > 0:
                ax.hist(q_ratio[active_mask], bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax.axvline(x=0.5, color='red', linestyle='--', label='Q=K')
            ax.set_xlabel('Q Ratio (Q / (Q+K))')
            ax.set_ylabel('Neuron Count')
            ax.set_title(f'{data["display"]}: Q/K Balance Distribution')
            ax.legend()

            row += 1

        plt.tight_layout()
        path = os.path.join(output_dir, 'qk_usage_analysis.png')
        plt.savefig(path, dpi=150)
        plt.close()

        return path


def print_summary(usage: dict, entropy: dict):
    """Print analysis summary"""
    print("\n" + "="*70)
    print("Q/K USAGE PATTERN ANALYSIS (DAWN v17.1)")
    print("="*70)
    print(f"Batches analyzed: {usage.get('n_batches', 0)}")

    for pool_name, data in usage.items():
        if pool_name == 'n_batches':
            continue

        print(f"\n{data['display']} ({data['n_neurons']} neurons):")
        print(f"  Q/K Correlation: {data['correlation']:.3f}")
        print(f"  Avg Overlap: {data['avg_overlap']:.3f} +/- {data['std_overlap']:.3f}")
        print(f"  Total: Q={data['q_total']:,}, K={data['k_total']:,}")
        print(f"  Specialization:")
        print(f"    - Q-specialized: {data['q_specialized']}")
        print(f"    - K-specialized: {data['k_specialized']}")
        print(f"    - Shared (both): {data['shared']}")
        print(f"    - Inactive: {data['inactive']}")

        # Entropy info if available
        if pool_name in entropy:
            ent = entropy[pool_name]
            print(f"  Entropy:")
            print(f"    - Q: {ent['q_entropy_mean']:.1f}% +/- {ent['q_entropy_std']:.1f}")
            print(f"    - K: {ent['k_entropy_mean']:.1f}% +/- {ent['k_entropy_std']:.1f}")
            print(f"    - Diff (Q-K): {ent['entropy_diff']:+.1f}%")

        # Interpretation
        if data['correlation'] > 0.7:
            print(f"  [Interpretation] High correlation -> neurons used similarly for Q and K")
            print(f"                   -> Pool separation may NOT be beneficial")
        elif data['correlation'] < 0.3:
            print(f"  [Interpretation] Low correlation -> Q and K use different neurons")
            print(f"                   -> Pool separation IS beneficial!")
        else:
            print(f"  [Interpretation] Moderate correlation -> mixed pattern")


def main():
    parser = argparse.ArgumentParser(description='DAWN Q/K Usage Pattern Analysis')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--val_data', required=True, help='Validation data path')
    parser.add_argument('--output_dir', default='./dawn_qk_analysis', help='Output directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_batches', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, tokenizer, config = load_model(args.checkpoint, device)
    router = get_router(model)

    if router is None:
        print("ERROR: Could not find router in model")
        return

    dataloader = create_dataloader(args.val_data, tokenizer, args.batch_size)
    analyzer = QKAnalyzer(model, router, tokenizer, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Run analyses
    print("\n--- Analyzing Q/K Usage Patterns ---")
    usage_results = analyzer.analyze_qk_usage(dataloader, args.max_batches)

    print("\n--- Analyzing Q/K Entropy ---")
    entropy_results = analyzer.analyze_qk_entropy(dataloader, args.max_batches // 2)

    # Print summary
    print_summary(usage_results, entropy_results)

    # Visualize
    if HAS_MATPLOTLIB:
        viz_path = analyzer.visualize(usage_results, args.output_dir)
        if viz_path:
            print(f"\nVisualization saved: {viz_path}")

    # Save results
    results = {
        'usage': usage_results,
        'entropy': entropy_results,
    }

    with open(os.path.join(args.output_dir, 'qk_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
