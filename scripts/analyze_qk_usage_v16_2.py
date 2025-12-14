#!/usr/bin/env python3
"""
DAWN v16.2 Q/K Usage Pattern Analysis
=====================================

Q/K 뉴런 사용 패턴 분석:
1. 뉴런별 Q 선택 횟수 vs K 선택 횟수
2. Q/K 겹침 분석
3. Scatter plot 시각화

Usage:
    python scripts/analyze_qk_usage_v16_2.py --checkpoint <path> --val_data <path>
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


def get_underlying_model(model):
    """Get the underlying model from torch.compile wrapper"""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def find_latest_checkpoint(path: str) -> str:
    """Find the latest checkpoint in a directory or return the path if it's a file"""
    path = Path(path)

    if path.is_file():
        return str(path)

    if path.is_dir():
        pt_files = list(path.glob('*.pt'))
        if not pt_files:
            pt_files = list(path.glob('**/*.pt'))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {path}")

        pt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for f in pt_files:
            if 'best' in f.name.lower():
                print(f"Found best checkpoint: {f}")
                return str(f)

        for f in pt_files:
            if 'latest' in f.name.lower():
                print(f"Found latest checkpoint: {f}")
                return str(f)

        print(f"Using most recent checkpoint: {pt_files[0]}")
        return str(pt_files[0])

    raise FileNotFoundError(f"Path not found: {path}")


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load v16.2 model from checkpoint"""
    from transformers import BertTokenizer
    from models import create_model_by_version

    checkpoint_path = find_latest_checkpoint(checkpoint_path)
    print(f"Loading from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    # Detect version
    path_str = str(checkpoint_path).lower()
    if 'v16.2' in path_str or 'v16_2' in path_str:
        version = '16.2'
    elif 'v16.1' in path_str or 'v16_1' in path_str:
        version = '16.1'
    elif 'v16' in path_str:
        version = '16.0'
    else:
        version = config.get('model_version', '16.2')

    print(f"Loading model version: {version}")

    model = create_model_by_version(version, config)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return model, tokenizer, config, version


def create_dataloader(data_path: str, tokenizer, max_length: int = 128, batch_size: int = 32):
    """Create dataloader for analysis"""
    from functools import partial
    from torch.utils.data import DataLoader
    from utils.data import load_single_file, TokenDataset, TextDataset, collate_fn_dynamic_padding

    data, is_pretokenized = load_single_file(data_path, max_length)

    if is_pretokenized:
        dataset = TokenDataset(data, max_length)
        print(f"Loaded {len(dataset):,} sequences (pre-tokenized)")
    else:
        dataset = TextDataset(data, tokenizer, max_length)
        print(f"Loaded {len(dataset):,} texts")

    collate_fn = partial(collate_fn_dynamic_padding, tokenizer=tokenizer, max_seq_len=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    return dataloader


class QKUsageAnalyzer:
    """v16.2 Q/K 사용 패턴 분석기"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.router = self._get_router()

    def _get_router(self):
        if hasattr(self.model, 'global_routers'):
            return self.model.global_routers.neuron_router
        return None

    def analyze_qk_usage(self, dataloader, n_batches: int = 100):
        """
        뉴런별 Q/K 사용 횟수 분석

        Returns:
            - feature_r_q_counts: [n_feature_r] Q로 선택된 횟수
            - feature_r_k_counts: [n_feature_r] K로 선택된 횟수
            - relational_q_counts: [n_relational] Q로 선택된 횟수
            - relational_k_counts: [n_relational] K로 선택된 횟수
        """
        if self.router is None:
            return {'error': 'No router found'}

        n_feature_r = self.router.n_feature_r
        n_relational = self.router.n_relational

        # 카운터 초기화
        fr_q_counts = torch.zeros(n_feature_r, device=self.device)
        fr_k_counts = torch.zeros(n_feature_r, device=self.device)
        rel_q_counts = torch.zeros(n_relational, device=self.device)
        rel_k_counts = torch.zeros(n_relational, device=self.device)

        # 배치별 겹침 추적
        batch_overlaps_fr = []
        batch_overlaps_rel = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc='Analyzing Q/K usage', total=n_batches)):
                if i >= n_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(self.device)

                try:
                    outputs = self.model(input_ids, attention_mask=attention_mask, return_routing_info=True)

                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        logits, routing_infos = outputs
                        attn = routing_infos[0].get('attention', {}) if routing_infos else {}
                    else:
                        continue

                except Exception as e:
                    print(f"Error at batch {i}: {e}")
                    continue

                # v16.2: feature_r_weights_Q, feature_r_weights_K
                w_frq = attn.get('feature_r_weights_Q')
                w_frk = attn.get('feature_r_weights_K')
                w_rq = attn.get('relational_weights_Q')
                w_rk = attn.get('relational_weights_K')

                if w_frq is not None and w_frk is not None:
                    # Feature R: 선택된 뉴런 카운트
                    selected_frq = (w_frq > 0).float().sum(dim=0)  # [N_FR]
                    selected_frk = (w_frk > 0).float().sum(dim=0)
                    fr_q_counts += selected_frq
                    fr_k_counts += selected_frk

                    # 배치별 겹침 계산
                    overlap_fr = ((w_frq > 0) & (w_frk > 0)).float()
                    active_q = (w_frq > 0).float().sum(-1)
                    overlap_ratio_fr = (overlap_fr.sum(-1) / (active_q + 1e-8)).mean().item()
                    batch_overlaps_fr.append(overlap_ratio_fr)

                if w_rq is not None and w_rk is not None:
                    # Relational: 선택된 뉴런 카운트
                    selected_rq = (w_rq > 0).float().sum(dim=0)  # [N_R]
                    selected_rk = (w_rk > 0).float().sum(dim=0)
                    rel_q_counts += selected_rq
                    rel_k_counts += selected_rk

                    # 배치별 겹침 계산
                    overlap_r = ((w_rq > 0) & (w_rk > 0)).float()
                    active_q_r = (w_rq > 0).float().sum(-1)
                    overlap_ratio_r = (overlap_r.sum(-1) / (active_q_r + 1e-8)).mean().item()
                    batch_overlaps_rel.append(overlap_ratio_r)

        # 결과 정리
        results = {
            'feature_r': {
                'n_neurons': n_feature_r,
                'q_counts': fr_q_counts.cpu().numpy(),
                'k_counts': fr_k_counts.cpu().numpy(),
                'avg_overlap': np.mean(batch_overlaps_fr) if batch_overlaps_fr else 0,
                'std_overlap': np.std(batch_overlaps_fr) if batch_overlaps_fr else 0,
            },
            'relational': {
                'n_neurons': n_relational,
                'q_counts': rel_q_counts.cpu().numpy(),
                'k_counts': rel_k_counts.cpu().numpy(),
                'avg_overlap': np.mean(batch_overlaps_rel) if batch_overlaps_rel else 0,
                'std_overlap': np.std(batch_overlaps_rel) if batch_overlaps_rel else 0,
            },
            'n_batches': min(n_batches, i + 1),
        }

        # 통계 계산
        for key in ['feature_r', 'relational']:
            q = results[key]['q_counts']
            k = results[key]['k_counts']

            # 상관계수
            if q.sum() > 0 and k.sum() > 0:
                corr = np.corrcoef(q, k)[0, 1]
            else:
                corr = 0

            # Q 전용, K 전용, 공용 뉴런
            threshold = (q.sum() + k.sum()) / (2 * len(q)) * 0.1  # 평균의 10%
            q_only = ((q > threshold) & (k < threshold)).sum()
            k_only = ((k > threshold) & (q < threshold)).sum()
            both = ((q > threshold) & (k > threshold)).sum()
            neither = ((q < threshold) & (k < threshold)).sum()

            results[key]['correlation'] = float(corr)
            results[key]['q_specialized'] = int(q_only)
            results[key]['k_specialized'] = int(k_only)
            results[key]['shared'] = int(both)
            results[key]['inactive'] = int(neither)

        return results

    def visualize_qk_usage(self, results: dict, output_dir: str):
        """Q/K 사용 패턴 시각화"""
        if not HAS_MATPLOTLIB:
            return {'error': 'matplotlib not available'}

        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        for row, key in enumerate(['feature_r', 'relational']):
            data = results[key]
            q_counts = data['q_counts']
            k_counts = data['k_counts']

            # 1. Scatter plot: Q count vs K count
            ax = axes[row, 0]
            ax.scatter(q_counts, k_counts, alpha=0.6, s=30)

            # 대각선 (Q=K)
            max_val = max(q_counts.max(), k_counts.max())
            ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Q=K')

            ax.set_xlabel('Q Selection Count')
            ax.set_ylabel('K Selection Count')
            ax.set_title(f'{key.upper()}: Q vs K Usage\n(corr={data["correlation"]:.3f})')
            ax.legend()

            # 2. Bar chart: 뉴런 분류
            ax = axes[row, 1]
            categories = ['Q-only', 'K-only', 'Shared', 'Inactive']
            values = [data['q_specialized'], data['k_specialized'],
                     data['shared'], data['inactive']]
            colors = ['blue', 'orange', 'green', 'gray']

            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            ax.set_ylabel('Neuron Count')
            ax.set_title(f'{key.upper()}: Neuron Specialization')

            # 값 표시
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       str(val), ha='center', va='bottom')

            # 3. 히스토그램: Q/K 비율
            ax = axes[row, 2]

            # Q/(Q+K) 비율 계산
            total = q_counts + k_counts + 1e-8
            q_ratio = q_counts / total

            # 활성 뉴런만
            active_mask = (q_counts + k_counts) > 0
            if active_mask.sum() > 0:
                ax.hist(q_ratio[active_mask], bins=20, alpha=0.7, color='purple', edgecolor='black')

            ax.axvline(x=0.5, color='red', linestyle='--', label='Q=K')
            ax.set_xlabel('Q Ratio (Q / (Q+K))')
            ax.set_ylabel('Neuron Count')
            ax.set_title(f'{key.upper()}: Q/K Balance Distribution')
            ax.legend()

        plt.tight_layout()
        path = os.path.join(output_dir, 'qk_usage_analysis.png')
        plt.savefig(path, dpi=150)
        plt.close()

        print(f"Visualization saved to: {path}")
        return {'visualization': path}

    def print_summary(self, results: dict):
        """결과 요약 출력"""
        print("\n" + "="*70)
        print("Q/K USAGE PATTERN ANALYSIS (v16.2)")
        print("="*70)
        print(f"Batches analyzed: {results['n_batches']}")

        for key in ['feature_r', 'relational']:
            data = results[key]
            print(f"\n{key.upper()} ({data['n_neurons']} neurons):")
            print(f"  Q/K Correlation: {data['correlation']:.3f}")
            print(f"  Avg Overlap: {data['avg_overlap']:.3f} +/- {data['std_overlap']:.3f}")
            print(f"  Specialization:")
            print(f"    - Q-specialized: {data['q_specialized']}")
            print(f"    - K-specialized: {data['k_specialized']}")
            print(f"    - Shared (both): {data['shared']}")
            print(f"    - Inactive: {data['inactive']}")

            # 해석
            if data['correlation'] > 0.7:
                print(f"  [Interpretation] High correlation -> neurons used similarly for Q and K")
                print(f"                   -> Pool separation may NOT be beneficial")
            elif data['correlation'] < 0.3:
                print(f"  [Interpretation] Low correlation -> Q and K use different neurons")
                print(f"                   -> Pool separation IS beneficial!")
            else:
                print(f"  [Interpretation] Moderate correlation -> mixed pattern")


def main():
    parser = argparse.ArgumentParser(description='DAWN v16.2 Q/K Usage Analysis')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--val_data', required=True, help='Validation data path')
    parser.add_argument('--output_dir', default='./analysis_qk_v16_2', help='Output directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_batches', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, config, version = load_model(args.checkpoint, args.device)
    print(f"Model version: {version}")

    if version not in ['16.2']:
        print(f"Warning: This script is designed for v16.2, but loaded {version}")
        print("         Results may not be accurate for other versions.")

    # Create dataloader
    dataloader = create_dataloader(args.val_data, tokenizer, batch_size=args.batch_size)

    # Create analyzer
    analyzer = QKUsageAnalyzer(model, tokenizer, args.device)

    # Run analysis
    os.makedirs(args.output_dir, exist_ok=True)

    results = analyzer.analyze_qk_usage(dataloader, args.n_batches)

    # Print summary
    analyzer.print_summary(results)

    # Visualize
    if HAS_MATPLOTLIB:
        analyzer.visualize_qk_usage(results, args.output_dir)

    # Save results
    # Convert numpy arrays to lists for JSON
    save_results = {}
    for key, val in results.items():
        if isinstance(val, dict):
            save_results[key] = {}
            for k, v in val.items():
                if isinstance(v, np.ndarray):
                    save_results[key][k] = v.tolist()
                else:
                    save_results[key][k] = v
        else:
            save_results[key] = val

    with open(os.path.join(args.output_dir, 'qk_usage_results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
