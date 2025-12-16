#!/usr/bin/env python3
"""
DAWN Co-selection Analysis
===========================
Analyze neuron co-selection patterns between pools for v17.1:
1. FQK + RQK co-selection (Q/K processing pairs)
2. FV + RV co-selection (Value processing pairs)
3. FK + RK co-selection (Knowledge pairs)
4. Cross-pool analysis
5. Subspace diversity analysis

Usage:
    python analyze_dawn_coselection.py --checkpoint path/to/ckpt --val_data path/to/data
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

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


# v17.1 Pool Pairs for Co-selection Analysis
COSELECTION_PAIRS = {
    'fqk_rqk': {
        'name': 'F-QK / R-QK (Q/K Processing)',
        'pool_a': {
            'type': 'feature_qk',
            'display': 'F-QK',
            'pref_key': 'fqk_q_pref',  # Use Q preference
            'source': 'attention',
            'n_attr': 'n_feature_qk',
            'neuron_attr': 'feature_qk_neurons',
        },
        'pool_b': {
            'type': 'restore_qk',
            'display': 'R-QK',
            'pref_key': 'rqk_q_pref',
            'source': 'attention',
            'n_attr': 'n_restore_qk',
            'neuron_attr': 'restore_qk_neurons',
        },
    },
    'fv_rv': {
        'name': 'F-V / R-V (Value Processing)',
        'pool_a': {
            'type': 'feature_v',
            'display': 'F-V',
            'pref_key': 'fv_pref',
            'source': 'attention',
            'n_attr': 'n_feature_v',
            'neuron_attr': 'feature_v_neurons',
        },
        'pool_b': {
            'type': 'restore_v',
            'display': 'R-V',
            'pref_key': 'rv_pref',
            'source': 'attention',
            'n_attr': 'n_restore_v',
            'neuron_attr': 'restore_v_neurons',
        },
    },
    'fk_rk': {
        'name': 'F-Know / R-Know (Knowledge Processing)',
        'pool_a': {
            'type': 'feature_know',
            'display': 'F-Know',
            'pref_key': 'feature_know_w',  # v17.1 uses feature_know_w
            'source': 'knowledge',
            'n_attr': 'n_feature_know',
            'neuron_attr': 'feature_know',
        },
        'pool_b': {
            'type': 'restore_know',
            'display': 'R-Know',
            'pref_key': 'restore_know_w',  # v17.1 uses restore_know_w
            'source': 'knowledge',
            'n_attr': 'n_restore_know',
            'neuron_attr': 'restore_know',
        },
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


def get_shared_neurons(model):
    """Get shared neurons from model"""
    if hasattr(model, 'shared_neurons'):
        return model.shared_neurons
    if hasattr(model, '_orig_mod'):
        return get_shared_neurons(model._orig_mod)
    return None


def create_dataloader(data_path: str, tokenizer, batch_size: int = 32, max_samples: int = 10000):
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
        texts = df['text'].tolist()[:max_samples]
    elif data_path.endswith('.json'):
        with open(data_path) as f:
            data = json.load(f)
        texts = [d['text'] for d in data[:max_samples]]
    elif data_path.endswith('.pt'):
        data = torch.load(data_path)
        if isinstance(data, dict):
            input_ids = data.get('input_ids', data.get('tokens'))
        else:
            input_ids = data
        # Return tensor-based loader
        if input_ids.dim() == 1:
            seq_len = 512
            n_seqs = input_ids.shape[0] // seq_len
            input_ids = input_ids[:n_seqs * seq_len].view(n_seqs, seq_len)
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(input_ids[:max_samples])
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Unsupported format: {data_path}")

    dataset = TextDataset(texts, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class CoselectionAnalyzer:
    """Co-selection Pattern Analyzer for DAWN v17.1"""

    def __init__(self, model, router, shared_neurons, device='cuda'):
        self.model = model
        self.router = router
        self.shared = shared_neurons
        self.device = device

    def analyze_coselection(self, dataloader, pair_key: str, n_batches: int = 50) -> dict:
        """
        Analyze co-selection patterns between two neuron pools.

        Returns:
            - co_matrix: [n_a, n_b] co-selection count matrix
            - top_pairs: Top co-selected pairs
            - concentration: How concentrated co-selections are
            - specialization: Per-neuron specialization metrics
        """
        pair_info = COSELECTION_PAIRS.get(pair_key)
        if pair_info is None:
            return {}

        pool_a = pair_info['pool_a']
        pool_b = pair_info['pool_b']

        n_a = getattr(self.router, pool_a['n_attr'], 0)
        n_b = getattr(self.router, pool_b['n_attr'], 0)

        if n_a == 0 or n_b == 0:
            print(f"  Skip {pair_key}: n_a={n_a}, n_b={n_b}")
            return {}

        print(f"\n  {pair_info['name']}: {pool_a['display']}({n_a}) x {pool_b['display']}({n_b})")

        # Co-selection matrix
        co_matrix = torch.zeros(n_a, n_b, device=self.device)
        a_counts = torch.zeros(n_a, device=self.device)
        b_counts = torch.zeros(n_b, device=self.device)
        total_samples = 0

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc=f'{pair_key}', total=n_batches)):
                if i >= n_batches:
                    break

                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                elif isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                B = input_ids.shape[0]
                total_samples += B

                try:
                    outputs = self.model(input_ids, return_routing_info=True)
                    if not isinstance(outputs, tuple) or len(outputs) < 2:
                        continue
                    routing_infos = outputs[-1]
                    if not routing_infos:
                        continue
                    layer_info = routing_infos[0]
                except Exception as e:
                    continue

                # Get preferences from the right source dict (attention or knowledge)
                source_a = layer_info.get(pool_a.get('source', 'attention'), {})
                source_b = layer_info.get(pool_b.get('source', 'attention'), {})
                pref_a = source_a.get(pool_a['pref_key'])
                pref_b = source_b.get(pool_b['pref_key'])

                if pref_a is None or pref_b is None:
                    continue

                # Handle different shapes: [B, S, N] or [B, N]
                if pref_a.dim() == 3:
                    pref_a = pref_a.mean(dim=1)  # [B, N]
                if pref_b.dim() == 3:
                    pref_b = pref_b.mean(dim=1)  # [B, N]

                # Binary selection (above uniform threshold)
                thresh_a = 1.0 / n_a
                thresh_b = 1.0 / n_b

                selected_a = (pref_a > thresh_a).float()  # [B, n_a]
                selected_b = (pref_b > thresh_b).float()  # [B, n_b]

                # Count individual selections
                a_counts += selected_a.sum(dim=0)
                b_counts += selected_b.sum(dim=0)

                # Co-occurrence: outer product per batch, then sum
                for b in range(B):
                    co_matrix += torch.outer(selected_a[b], selected_b[b])

        # Analyze results
        results = self._analyze_coselection_matrix(
            co_matrix, a_counts, b_counts, n_a, n_b,
            pool_a['display'], pool_b['display']
        )
        results['total_samples'] = total_samples
        results['pair_name'] = pair_info['name']
        results['co_matrix'] = co_matrix.cpu().numpy().tolist()

        return results

    def _analyze_coselection_matrix(self, co_matrix, a_counts, b_counts, n_a, n_b, name_a, name_b):
        """Analyze co-selection matrix"""
        results = {}

        # Normalize to joint probability
        total = co_matrix.sum()
        if total > 0:
            co_prob = co_matrix / total
        else:
            return {'error': 'No co-selections found'}

        # Top pairs
        flat_co = co_matrix.view(-1)
        top_k = min(20, flat_co.numel())
        top_values, top_indices = torch.topk(flat_co, top_k)

        top_pairs = []
        for i in range(top_k):
            idx = top_indices[i].item()
            a_idx = idx // n_b
            b_idx = idx % n_b
            count = top_values[i].item()
            pct = count / total.item() * 100

            top_pairs.append({
                'a_idx': a_idx,
                'b_idx': b_idx,
                'a_name': f'{name_a}_{a_idx}',
                'b_name': f'{name_b}_{b_idx}',
                'count': int(count),
                'pct': pct
            })

        results['top_pairs'] = top_pairs

        # Concentration analysis
        cumsum = torch.cumsum(torch.sort(flat_co, descending=True)[0], dim=0)

        top10_pct = (cumsum[9] / total * 100).item() if len(cumsum) > 9 else 0
        top50_pct = (cumsum[49] / total * 100).item() if len(cumsum) > 49 else 0
        top100_pct = (cumsum[99] / total * 100).item() if len(cumsum) > 99 else 0

        # Entropy
        co_prob_flat = co_prob.view(-1)
        co_prob_flat = co_prob_flat[co_prob_flat > 0]
        entropy = -(co_prob_flat * co_prob_flat.log()).sum().item()
        max_entropy = np.log(n_a * n_b)
        norm_entropy = entropy / max_entropy

        results['concentration'] = {
            'top10_pct': top10_pct,
            'top50_pct': top50_pct,
            'top100_pct': top100_pct,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'normalized_entropy': norm_entropy,
        }

        # Specialization analysis: for each A, find best B partner
        a_specialization = []
        for a_idx in range(n_a):
            row = co_matrix[a_idx]
            if row.sum() > 0:
                top_b = row.argmax().item()
                top_b_pct = (row[top_b] / row.sum() * 100).item()
                a_specialization.append({
                    'a_idx': a_idx,
                    'top_b': top_b,
                    'top_b_pct': top_b_pct,
                    'total_count': int(row.sum().item())
                })

        a_specialization.sort(key=lambda x: x['top_b_pct'], reverse=True)
        results['a_specialization'] = a_specialization[:20]

        # For each B, find best A partner
        b_specialization = []
        for b_idx in range(n_b):
            col = co_matrix[:, b_idx]
            if col.sum() > 0:
                top_a = col.argmax().item()
                top_a_pct = (col[top_a] / col.sum() * 100).item()
                b_specialization.append({
                    'b_idx': b_idx,
                    'top_a': top_a,
                    'top_a_pct': top_a_pct,
                    'total_count': int(col.sum().item())
                })

        b_specialization.sort(key=lambda x: x['top_a_pct'], reverse=True)
        results['b_specialization'] = b_specialization[:20]

        return results

    def analyze_subspace_diversity(self, pair_key: str) -> dict:
        """
        Analyze neuron subspace diversity within a pool.
        Measures how different the neurons are from each other.
        """
        pair_info = COSELECTION_PAIRS.get(pair_key)
        if pair_info is None or self.shared is None:
            return {}

        results = {}

        for pool in ['pool_a', 'pool_b']:
            pool_info = pair_info[pool]
            neuron_attr = pool_info['neuron_attr']

            neurons = getattr(self.shared, neuron_attr, None)
            if neurons is None:
                continue

            n_neurons = neurons.shape[0]
            display = pool_info['display']

            print(f"\n  {display} Subspace Diversity ({n_neurons} neurons, shape={neurons.shape})")

            # Flatten each neuron's matrix
            neurons_flat = neurons.view(n_neurons, -1)  # [N, d*rank or rank*d]

            # Compute pairwise cosine similarity
            neurons_norm = F.normalize(neurons_flat, dim=-1)
            sim_matrix = torch.mm(neurons_norm, neurons_norm.t())  # [N, N]

            # Remove diagonal
            mask = ~torch.eye(n_neurons, dtype=torch.bool, device=self.device)
            sim_off_diag = sim_matrix[mask]

            avg_sim = sim_off_diag.mean().item()
            std_sim = sim_off_diag.std().item()
            min_sim = sim_off_diag.min().item()
            max_sim = sim_off_diag.max().item()

            # Find most similar pairs
            sim_flat = sim_matrix.clone().view(-1)
            for i in range(n_neurons):
                sim_flat[i * n_neurons + i] = -2  # Exclude diagonal

            top_k = min(10, n_neurons * (n_neurons - 1) // 2)
            top_vals, top_idx = torch.topk(sim_flat, top_k)

            top_similar = []
            for i in range(top_k):
                idx = top_idx[i].item()
                n_i = idx // n_neurons
                n_j = idx % n_neurons
                top_similar.append((n_i, n_j, top_vals[i].item()))

            # Interpretation
            if avg_sim < 0.3:
                interpretation = 'DIVERSE: Neurons use distinct subspaces (good!)'
            elif avg_sim < 0.6:
                interpretation = 'MODERATE: Some overlap in neuron subspaces'
            else:
                interpretation = 'COLLAPSED: Neurons converging to similar subspaces (bad!)'

            results[display] = {
                'n_neurons': n_neurons,
                'mean_similarity': avg_sim,
                'std_similarity': std_sim,
                'min_similarity': min_sim,
                'max_similarity': max_sim,
                'top_similar_pairs': top_similar,
                'interpretation': interpretation,
            }

            print(f"    Mean pairwise similarity: {avg_sim:.4f} +/- {std_sim:.4f}")
            print(f"    Interpretation: {interpretation}")

        return results

    def analyze_cross_pool_alignment(self, pair_key: str, dataloader, n_batches: int = 20) -> dict:
        """
        Analyze alignment between pool A and pool B in the rank space.
        Uses actual data to compute output directions.
        """
        pair_info = COSELECTION_PAIRS.get(pair_key)
        if pair_info is None or self.shared is None:
            return {}

        pool_a = pair_info['pool_a']
        pool_b = pair_info['pool_b']

        neurons_a = getattr(self.shared, pool_a['neuron_attr'], None)
        neurons_b = getattr(self.shared, pool_b['neuron_attr'], None)

        if neurons_a is None or neurons_b is None:
            return {}

        n_a = neurons_a.shape[0]
        n_b = neurons_b.shape[0]
        rank = neurons_a.shape[-1] if neurons_a.dim() == 3 else neurons_a.shape[1]

        print(f"\n  {pool_a['display']}-{pool_b['display']} Alignment Analysis")
        print(f"    {pool_a['display']}: {neurons_a.shape}, {pool_b['display']}: {neurons_b.shape}")

        # Compute output vectors for pool A using data
        a_outputs = [[] for _ in range(n_a)]

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc='Computing outputs', total=n_batches)):
                if i >= n_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                elif isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                # Get embeddings
                x = self.model.token_emb(input_ids)
                if hasattr(self.model, 'pos_emb'):
                    seq_len = input_ids.shape[1]
                    positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
                    x = x + self.model.pos_emb(positions)

                # Compute pool A outputs
                for j in range(n_a):
                    # neurons_a[j]: [d_model, rank] or similar
                    neuron = neurons_a[j]
                    if neuron.dim() == 2:
                        out = torch.matmul(x, neuron)  # [B, S, rank]
                    else:
                        out = torch.matmul(x, neuron.view(-1, rank))
                    a_outputs[j].append(out.mean(dim=[0, 1]))  # [rank]

        # Average A outputs across batches
        a_vecs = torch.stack([torch.stack(outputs).mean(0) for outputs in a_outputs if outputs])
        a_vecs_norm = F.normalize(a_vecs, dim=-1)

        # Compute B preferred input directions via SVD
        b_vecs = []
        for j in range(n_b):
            W = neurons_b[j]
            if W.dim() == 2:
                # Shape: [rank, d_model] or [d_model, rank]
                # Transpose if needed to get [rank, d_model]
                if W.shape[0] > W.shape[1]:
                    W = W.t()
            else:
                W = W.view(W.shape[0], -1)

            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            b_vecs.append(U[:, 0])  # First left singular vector

        b_vecs = torch.stack(b_vecs)
        b_vecs_norm = F.normalize(b_vecs, dim=-1)

        # Compute alignment matrix
        alignment = torch.mm(a_vecs_norm, b_vecs_norm.t())  # [n_a, n_b]

        # Statistics
        avg_align = alignment.abs().mean().item()
        max_align = alignment.abs().max().item()

        # Top aligned pairs
        align_flat = alignment.abs().view(-1)
        top_k = min(20, align_flat.numel())
        top_vals, top_idx = torch.topk(align_flat, top_k)

        top_aligned = []
        for i in range(top_k):
            idx = top_idx[i].item()
            a_idx = idx // n_b
            b_idx = idx % n_b
            val = alignment[a_idx, b_idx].item()
            top_aligned.append({
                'a_idx': a_idx,
                'b_idx': b_idx,
                'alignment': val
            })

        results = {
            'pair_name': pair_info['name'],
            'mean_abs_alignment': avg_align,
            'max_abs_alignment': max_align,
            'top_aligned_pairs': top_aligned,
            'alignment_matrix': alignment.detach().cpu().numpy().tolist(),
        }

        print(f"    Mean |alignment|: {avg_align:.4f}")
        print(f"    Max |alignment|: {max_align:.4f}")
        print(f"    Top 5 aligned:")
        for item in top_aligned[:5]:
            print(f"      {pool_a['display']}_{item['a_idx']} -> {pool_b['display']}_{item['b_idx']}: {item['alignment']:+.4f}")

        return results


def visualize_coselection(results: dict, output_dir: str):
    """Visualize co-selection patterns"""
    if not HAS_MATPLOTLIB:
        return

    os.makedirs(output_dir, exist_ok=True)

    for pair_key, pair_results in results.items():
        if 'co_matrix' not in pair_results:
            continue

        co_matrix = np.array(pair_results['co_matrix'])
        pair_name = pair_results.get('pair_name', pair_key)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Heatmap
        ax = axes[0]
        im = ax.imshow(co_matrix, aspect='auto', cmap='hot')
        fig.colorbar(im, ax=ax, label='Co-selection count')
        ax.set_xlabel('Pool B neuron')
        ax.set_ylabel('Pool A neuron')
        ax.set_title(f'{pair_name}\nCo-selection Heatmap')

        # 2. Concentration bar chart
        ax = axes[1]
        conc = pair_results.get('concentration', {})
        labels = ['Top 10', 'Top 50', 'Top 100']
        values = [conc.get('top10_pct', 0), conc.get('top50_pct', 0), conc.get('top100_pct', 0)]
        bars = ax.bar(labels, values, color=['red', 'orange', 'green'], alpha=0.7)
        ax.set_ylabel('% of all co-selections')
        ax.set_title(f'{pair_name}\nPair Concentration')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center')

        # 3. Top pairs
        ax = axes[2]
        top_pairs = pair_results.get('top_pairs', [])[:10]
        if top_pairs:
            labels = [f"{p['a_name']}\n{p['b_name']}" for p in top_pairs]
            values = [p['pct'] for p in top_pairs]
            ax.barh(range(len(values)), values, color='steelblue', alpha=0.7)
            ax.set_yticks(range(len(values)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('% of co-selections')
            ax.set_title(f'{pair_name}\nTop 10 Pairs')
            ax.invert_yaxis()

        plt.tight_layout()
        path = os.path.join(output_dir, f'coselection_{pair_key}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


def print_summary(results: dict):
    """Print analysis summary"""
    print("\n" + "="*70)
    print("CO-SELECTION ANALYSIS SUMMARY (DAWN v17.1)")
    print("="*70)

    for pair_key, pair_results in results.items():
        if 'pair_name' not in pair_results:
            continue

        print(f"\n{pair_results['pair_name']}")
        print("-" * 50)

        # Concentration
        conc = pair_results.get('concentration', {})
        print(f"  Concentration:")
        print(f"    Top 10 pairs: {conc.get('top10_pct', 0):.1f}%")
        print(f"    Top 50 pairs: {conc.get('top50_pct', 0):.1f}%")
        print(f"    Normalized entropy: {conc.get('normalized_entropy', 0):.2%}")

        # Interpretation
        norm_ent = conc.get('normalized_entropy', 1.0)
        if norm_ent < 0.5:
            interp = "CONCENTRATED: Strong neuron pairing learned"
        elif norm_ent < 0.8:
            interp = "MODERATE: Some pairing structure"
        else:
            interp = "UNIFORM: Pools operate independently"
        print(f"    Interpretation: {interp}")

        # Top pairs
        top_pairs = pair_results.get('top_pairs', [])[:5]
        if top_pairs:
            print(f"  Top 5 pairs:")
            for p in top_pairs:
                print(f"    {p['a_name']} + {p['b_name']}: {p['count']} ({p['pct']:.2f}%)")

        # Specialization
        a_spec = pair_results.get('a_specialization', [])[:3]
        if a_spec:
            print(f"  Most specialized (Pool A):")
            for s in a_spec:
                print(f"    A_{s['a_idx']}: {s['top_b_pct']:.1f}% with B_{s['top_b']}")


def main():
    parser = argparse.ArgumentParser(description='DAWN Co-selection Analysis')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--val_data', required=True, help='Validation data path')
    parser.add_argument('--output_dir', default='./dawn_coselection', help='Output directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_batches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pairs', type=str, default='all',
                       help='Pairs to analyze: all, fqk_rqk, fv_rv, fk_rk')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, tokenizer, config = load_model(args.checkpoint, device)
    router = get_router(model)
    shared = get_shared_neurons(model)

    if router is None:
        print("ERROR: Could not find router in model")
        return

    dataloader = create_dataloader(args.val_data, tokenizer, args.batch_size)
    analyzer = CoselectionAnalyzer(model, router, shared, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Select pairs to analyze
    if args.pairs == 'all':
        pairs_to_analyze = list(COSELECTION_PAIRS.keys())
    else:
        pairs_to_analyze = [p.strip() for p in args.pairs.split(',')]

    all_results = {}

    print("\n--- Analyzing Co-selection Patterns ---")
    for pair_key in pairs_to_analyze:
        if pair_key not in COSELECTION_PAIRS:
            print(f"  Skip unknown pair: {pair_key}")
            continue

        # Co-selection analysis
        cosel_results = analyzer.analyze_coselection(dataloader, pair_key, args.max_batches)
        if cosel_results:
            all_results[pair_key] = cosel_results

        # Subspace diversity
        div_results = analyzer.analyze_subspace_diversity(pair_key)
        if div_results:
            all_results[pair_key]['subspace_diversity'] = div_results

        # Cross-pool alignment (only if shared neurons exist)
        if shared is not None:
            align_results = analyzer.analyze_cross_pool_alignment(pair_key, dataloader, args.max_batches // 2)
            if align_results:
                all_results[pair_key]['alignment'] = align_results

    # Print summary
    print_summary(all_results)

    # Visualize
    if HAS_MATPLOTLIB:
        print("\n--- Generating Visualizations ---")
        visualize_coselection(all_results, args.output_dir)

    # Save results
    # Remove numpy arrays for JSON serialization
    json_results = {}
    for k, v in all_results.items():
        json_results[k] = {kk: vv for kk, vv in v.items() if kk != 'co_matrix'}
        if 'alignment' in v and 'alignment_matrix' in v['alignment']:
            json_results[k]['alignment'] = {kk: vv for kk, vv in v['alignment'].items()
                                            if kk != 'alignment_matrix'}

    with open(os.path.join(args.output_dir, 'coselection_analysis.json'), 'w') as f:
        json.dump(json_results, f, indent=2)

    # Save raw matrices as numpy
    for pair_key, results in all_results.items():
        if 'co_matrix' in results:
            np.save(os.path.join(args.output_dir, f'{pair_key}_co_matrix.npy'),
                   np.array(results['co_matrix']))
        if 'alignment' in results and 'alignment_matrix' in results['alignment']:
            np.save(os.path.join(args.output_dir, f'{pair_key}_alignment.npy'),
                   np.array(results['alignment']['alignment_matrix']))

    print(f"\nResults saved to: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
