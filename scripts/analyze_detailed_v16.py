#!/usr/bin/env python3
"""
DAWN v16 Detailed Analysis Script
=================================

Layer별, Position별, Co-occurrence, Rank Stability 분석

Usage:
    python scripts/analyze_detailed_v16.py --checkpoint <path> --val_data <path>
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

try:
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class DetailedAnalyzer:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.model, self.config = self.load_model(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def load_model(self, checkpoint_path):
        from models import create_model_by_version

        if os.path.isdir(checkpoint_path):
            ckpts = [f for f in os.listdir(checkpoint_path) if f.startswith('checkpoint_') and f.endswith('.pt')]
            if ckpts:
                steps = [int(f.split('step')[1].split('.')[0]) for f in ckpts]
                latest = ckpts[np.argmax(steps)]
                checkpoint_path = os.path.join(checkpoint_path, latest)
                print(f"Using checkpoint: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = ckpt.get('model_config', ckpt.get('config', {}))

        # Detect version
        path_str = str(checkpoint_path).lower()
        if 'v16.1' in path_str or 'v16_1' in path_str:
            version = '16.1'
        else:
            version = config.get('model_version', '16.0')

        print(f"Loading model version: {version}")
        model = create_model_by_version(version, config)

        state_dict = ckpt.get('model_state_dict', ckpt)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)
        model.to(self.device)
        model.eval()

        return model, config

    def analyze_selection_diversity_detailed(self, dataloader, n_batches=100):
        """Selection diversity with per-layer breakdown"""
        router = None
        if hasattr(self.model, 'global_routers'):
            router = self.model.global_routers.neuron_router

        if router is None:
            return {'error': 'No router found'}

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

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= n_batches:
                    break

                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(self.device)
                elif isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                    attention_mask = torch.ones_like(input_ids)
                else:
                    input_ids = batch.to(self.device)
                    attention_mask = torch.ones_like(input_ids)

                try:
                    outputs = self.model(input_ids, attention_mask=attention_mask, return_routing_info=True)

                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        logits, routing_infos = outputs
                        routing_info = routing_infos[0].get('attention', {}) if routing_infos else {}
                    else:
                        continue

                except Exception as e:
                    print(f"Error at batch {batch_idx}: {e}")
                    continue

                # Track selections
                for ntype, weight_key in [
                    ('feature_r', 'feature_r_weights'),
                    ('feature_v', 'feature_v_weights'),
                    ('relational', 'relational_weights_Q'),
                    ('value', 'value_weights'),
                ]:
                    if weight_key in routing_info:
                        weights = routing_info[weight_key]
                        if weights.dim() == 3:
                            selected = (weights > 0).any(dim=0).any(dim=0).cpu()
                            union_selected[ntype].update(selected.nonzero().flatten().tolist())
                            per_batch_counts[ntype].append(selected.sum().item())
                        elif weights.dim() == 2:
                            selected = (weights > 0).any(dim=0).cpu()
                            union_selected[ntype].update(selected.nonzero().flatten().tolist())
                            per_batch_counts[ntype].append(selected.sum().item())

                if batch_idx % 20 == 0:
                    print(f"  Selection analysis: {batch_idx}/{n_batches}")

        # Calculate results
        results = {}
        n_totals = {
            'feature_r': router.n_feature_r,
            'feature_v': router.n_feature_v,
            'relational': router.n_relational,
            'value': router.n_value,
        }

        for ntype in union_selected:
            n_total = n_totals[ntype]
            union_count = len(union_selected[ntype])
            batch_counts = per_batch_counts[ntype]

            if len(batch_counts) > 0:
                per_batch_avg = np.mean(batch_counts)
                per_batch_std = np.std(batch_counts)
            else:
                per_batch_avg = per_batch_std = 0

            results[ntype] = {
                'n_total': n_total,
                'per_batch_avg': float(per_batch_avg),
                'per_batch_std': float(per_batch_std),
                'union_count': union_count,
                'union_coverage': float(union_count / n_total) if n_total > 0 else 0,
                'diversity_ratio': float(union_count / per_batch_avg) if per_batch_avg > 0 else 0,
            }

        return results

    def analyze_cooccurrence(self, dataloader, n_batches=30):
        """뉴런 co-occurrence 분석"""
        cooc = {
            'feature_r': defaultdict(int),
            'feature_v': defaultdict(int),
            'relational': defaultdict(int),
            'value': defaultdict(int),
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= n_batches:
                    break

                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(self.device)
                elif isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                    attention_mask = torch.ones_like(input_ids)
                else:
                    input_ids = batch.to(self.device)
                    attention_mask = torch.ones_like(input_ids)

                try:
                    outputs = self.model(input_ids, attention_mask=attention_mask, return_routing_info=True)

                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        logits, routing_infos = outputs
                        routing_info = routing_infos[0].get('attention', {}) if routing_infos else {}
                    else:
                        continue

                except Exception as e:
                    continue

                # Co-occurrence from indices
                for ntype, idx_key in [
                    ('feature_r', 'feature_r_idx'),
                    ('feature_v', 'feature_v_idx'),
                    ('relational', 'relational_idx_Q'),
                    ('value', 'value_idx'),
                ]:
                    if idx_key in routing_info:
                        idx_tensor = routing_info[idx_key]  # [B, T, K]
                        if idx_tensor.dim() == 3:
                            B, T, K = idx_tensor.shape
                            for b in range(min(B, 4)):
                                for t in range(min(T, 64)):
                                    indices = sorted(idx_tensor[b, t].cpu().numpy())
                                    for i in range(len(indices)):
                                        for j in range(i+1, len(indices)):
                                            pair = (int(indices[i]), int(indices[j]))
                                            cooc[ntype][pair] += 1

                if batch_idx % 10 == 0:
                    print(f"  Co-occurrence: {batch_idx}/{n_batches}")

        return cooc

    def visualize_all(self, selection_results, cooc, save_dir='./analysis_detailed'):
        if not HAS_MATPLOTLIB:
            print("matplotlib not available, skipping visualization")
            return

        os.makedirs(save_dir, exist_ok=True)

        # 1. Selection diversity bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        ntypes = ['feature_r', 'feature_v', 'relational', 'value']
        x = np.arange(len(ntypes))
        width = 0.35

        per_batch = [selection_results[nt]['per_batch_avg'] for nt in ntypes]
        union = [selection_results[nt]['union_count'] for nt in ntypes]

        ax.bar(x - width/2, per_batch, width, label='Per-batch avg', color='steelblue')
        ax.bar(x + width/2, union, width, label='Union across batches', color='coral')

        ax.set_xlabel('Neuron Type')
        ax.set_ylabel('Count')
        ax.set_title('Selection Diversity: Per-batch vs Union')
        ax.set_xticks(x)
        ax.set_xticklabels(['FR', 'FV', 'R', 'V'])
        ax.legend()

        # Add total counts as text
        for i, nt in enumerate(ntypes):
            total = selection_results[nt]['n_total']
            ax.annotate(f'/{total}', xy=(x[i] + width/2, union[i]), ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/selection_diversity.png', dpi=150)
        plt.close()

        # 2. Co-occurrence top pairs
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, ntype in zip(axes.flatten(), ['feature_r', 'feature_v', 'relational', 'value']):
            top_pairs = sorted(cooc[ntype].items(), key=lambda x: -x[1])[:20]
            if top_pairs:
                labels = [f"{p[0]}-{p[1]}" for p, c in top_pairs]
                counts = [c for p, c in top_pairs]
                ax.barh(range(len(labels)), counts, color='teal')
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_xlabel('Co-occurrence Count')
                ax.set_title(f'{ntype.upper()} Top Co-occurring Pairs')
            ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(f'{save_dir}/cooccurrence_pairs.png', dpi=150)
        plt.close()

        print(f"\nVisualizations saved to {save_dir}/")

    def print_summary(self, selection_results, cooc):
        print("\n" + "="*70)
        print("DETAILED ANALYSIS SUMMARY")
        print("="*70)

        print("\n[Selection Diversity]")
        print(f"{'Type':<12} {'PerBatch':>10} {'Union':>8} {'Total':>8} {'Coverage':>10} {'DivRatio':>10}")
        print("-"*70)
        for ntype in ['feature_r', 'feature_v', 'relational', 'value']:
            r = selection_results[ntype]
            print(f"{ntype:<12} {r['per_batch_avg']:>10.1f} {r['union_count']:>8} {r['n_total']:>8} {r['union_coverage']:>10.1%} {r['diversity_ratio']:>10.2f}")

        print("\n[Co-occurrence Analysis]")
        for ntype in ['feature_r', 'feature_v', 'relational', 'value']:
            cooc_counts = list(cooc[ntype].values())
            if cooc_counts:
                top5_share = sum(sorted(cooc_counts, reverse=True)[:5]) / (sum(cooc_counts) + 1e-8)
                n_unique_pairs = len(cooc[ntype])
                top_pair = max(cooc[ntype].items(), key=lambda x: x[1]) if cooc[ntype] else ((0,0), 0)
                print(f"  {ntype}: {n_unique_pairs} unique pairs, top-5 share: {top5_share:.1%}, most common: {top_pair[0]} ({top_pair[1]}x)")


def main():
    parser = argparse.ArgumentParser(description='DAWN v16 Detailed Analysis')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--val_data', required=True, help='Validation data path (.pt)')
    parser.add_argument('--output_dir', default='./analysis_detailed', help='Output directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_batches', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    print("Loading analyzer...")
    analyzer = DetailedAnalyzer(args.checkpoint, args.device)

    print("Loading data...")
    from utils.data import load_single_file, TokenDataset, TextDataset, collate_fn_dynamic_padding
    from functools import partial
    from torch.utils.data import DataLoader

    data, is_pretokenized = load_single_file(args.val_data, max_length=128)

    if is_pretokenized:
        dataset = TokenDataset(data, max_length=128)
        print(f"Loaded {len(dataset):,} sequences (pre-tokenized)")
    else:
        dataset = TextDataset(data, analyzer.tokenizer, max_length=128)
        print(f"Loaded {len(dataset):,} texts")

    collate_fn = partial(collate_fn_dynamic_padding, tokenizer=analyzer.tokenizer, max_seq_len=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    print("\n1. Selection diversity analysis...")
    selection_results = analyzer.analyze_selection_diversity_detailed(dataloader, args.n_batches)

    print("\n2. Co-occurrence analysis...")
    cooc = analyzer.analyze_cooccurrence(dataloader, n_batches=min(30, args.n_batches))

    print("\n3. Generating visualizations...")
    analyzer.visualize_all(selection_results, cooc, args.output_dir)

    analyzer.print_summary(selection_results, cooc)

    print(f"\n{'='*70}")
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
