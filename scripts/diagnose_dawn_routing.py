#!/usr/bin/env python3
"""
DAWN Routing Diagnostics
=========================
Comprehensive routing health diagnostics for DAWN v17.1:
1. Usage EMA distribution (neuron utilization)
2. Neuron selection frequency (actual selections during inference)
3. Routing entropy analysis (how concentrated routing is)
4. Neuron role analysis (what tokens activate each neuron)
5. Dead neuron detection

Usage:
    python diagnose_dawn_routing.py --checkpoint path/to/ckpt --val_data path/to/data
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
    import spacy
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except:
    HAS_SPACY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# DAWN v17.1 Neuron Pool Configuration
NEURON_POOLS = {
    'feature_qk': {
        'display': 'FQK',
        'ema_attr': 'usage_ema_feature_qk',
        'n_attr': 'n_feature_qk',
        'pref_keys': ['fqk_q_pref', 'fqk_k_pref'],
        'top_k_config': 'top_k_feature_qk',
        'default_top_k': 8,
        'color': 'red',
    },
    'feature_v': {
        'display': 'FV',
        'ema_attr': 'usage_ema_feature_v',
        'n_attr': 'n_feature_v',
        'pref_keys': ['fv_pref'],
        'top_k_config': 'top_k_feature_v',
        'default_top_k': 6,
        'color': 'orange',
    },
    'restore_qk': {
        'display': 'RQK',
        'ema_attr': 'usage_ema_restore_qk',
        'n_attr': 'n_restore_qk',
        'pref_keys': ['rqk_q_pref', 'rqk_k_pref'],
        'top_k_config': 'top_k_restore_qk',
        'default_top_k': 8,
        'color': 'blue',
    },
    'restore_v': {
        'display': 'RV',
        'ema_attr': 'usage_ema_restore_v',
        'n_attr': 'n_restore_v',
        'pref_keys': ['rv_pref'],
        'top_k_config': 'top_k_restore_v',
        'default_top_k': 4,
        'color': 'green',
    },
    'feature_know': {
        'display': 'FK',
        'ema_attr': 'usage_ema_feature_know',
        'n_attr': 'n_feature_know',
        'pref_keys': ['feature_know_w', 'fk_pref'],  # v17.1 uses feature_know_w
        'top_k_config': 'top_k_feature_know',
        'default_top_k': 8,
        'color': 'purple',
    },
    'restore_know': {
        'display': 'RK',
        'ema_attr': 'usage_ema_restore_know',
        'n_attr': 'n_restore_know',
        'pref_keys': ['restore_know_w', 'rk_pref'],  # v17.1 uses restore_know_w
        'top_k_config': 'top_k_restore_know',
        'default_top_k': 4,
        'color': 'cyan',
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
        dataset = TextDataset(texts, tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False), None
    elif data_path.endswith('.json'):
        with open(data_path) as f:
            data = json.load(f)
        texts = [d['text'] for d in data[:max_samples]]
        dataset = TextDataset(texts, tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False), None
    elif data_path.endswith('.pt'):
        data = torch.load(data_path)
        if isinstance(data, dict):
            input_ids = data.get('input_ids', data.get('tokens'))
        else:
            input_ids = data
        if input_ids.dim() == 1:
            seq_len = 512
            n_seqs = input_ids.shape[0] // seq_len
            input_ids = input_ids[:n_seqs * seq_len].view(n_seqs, seq_len)
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(input_ids[:max_samples])
        return DataLoader(dataset, batch_size=batch_size, shuffle=False), input_ids
    else:
        raise ValueError(f"Unsupported format: {data_path}")


class RoutingDiagnostics:
    """Routing Health Diagnostics for DAWN v17.1"""

    def __init__(self, model, router, tokenizer, config, device='cuda'):
        self.model = model
        self.router = router
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

    def diagnose_usage_ema(self) -> dict:
        """Analyze Usage EMA distribution for each neuron pool"""
        print("\n" + "="*60)
        print("1. USAGE EMA DISTRIBUTION")
        print("="*60)

        results = {}

        for pool_name, pool_info in NEURON_POOLS.items():
            ema = getattr(self.router, pool_info['ema_attr'], None)
            if ema is None:
                continue

            usage = ema.cpu().numpy()
            n_neurons = len(usage)

            dead_count = int((usage < 0.01).sum())
            low_usage = int(((usage >= 0.01) & (usage < 0.1)).sum())

            # Gini coefficient (measure of inequality)
            sorted_usage = np.sort(usage)
            n = len(sorted_usage)
            cumsum = np.cumsum(sorted_usage)
            gini = (2 * np.sum((np.arange(1, n+1) * sorted_usage)) / (n * np.sum(sorted_usage))) - (n + 1) / n if np.sum(sorted_usage) > 0 else 0

            # Top 5 neurons
            top5_idx = np.argsort(usage)[-5:][::-1]

            print(f"\n  {pool_info['display']} ({n_neurons} neurons):")
            print(f"    Usage: max={usage.max():.4f}, min={usage.min():.4f}, mean={usage.mean():.4f}")
            print(f"    Dead (<0.01): {dead_count} ({100*dead_count/n_neurons:.1f}%)")
            print(f"    Low usage (0.01-0.1): {low_usage} ({100*low_usage/n_neurons:.1f}%)")
            print(f"    Gini coefficient: {gini:.3f}", end="")
            if gini > 0.5:
                print(" [HIGH INEQUALITY]")
            elif gini > 0.3:
                print(" [MODERATE]")
            else:
                print(" [BALANCED]")
            print(f"    Top 5: {[(int(i), f'{usage[i]:.4f}') for i in top5_idx]}")

            results[pool_name] = {
                'display': pool_info['display'],
                'n_neurons': n_neurons,
                'max': float(usage.max()),
                'min': float(usage.min()),
                'mean': float(usage.mean()),
                'std': float(usage.std()),
                'dead_count': dead_count,
                'dead_pct': 100 * dead_count / n_neurons,
                'low_usage_count': low_usage,
                'gini': gini,
                'top5': [(int(i), float(usage[i])) for i in top5_idx],
                'all_usage': usage.tolist(),
            }

        return results

    def diagnose_neuron_selection(self, dataloader, n_batches: int = 20) -> dict:
        """Analyze actual neuron selection frequency during inference"""
        print("\n" + "="*60)
        print("2. NEURON SELECTION FREQUENCY")
        print("="*60)

        results = {}
        neuron_counts = {pool: Counter() for pool in NEURON_POOLS.keys()}
        total_tokens = 0

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc='Selection freq', total=n_batches)):
                if i >= n_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                elif isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                B, L = input_ids.shape
                total_tokens += B * L

                try:
                    outputs = self.model(input_ids, return_routing_info=True)
                    if not isinstance(outputs, tuple) or len(outputs) < 2:
                        continue
                    routing_infos = outputs[-1]
                    if not routing_infos:
                        continue
                except:
                    continue

                # Aggregate across all layers
                for layer_info in routing_infos:
                    attn = layer_info.get('attention', {})

                    for pool_name, pool_info in NEURON_POOLS.items():
                        top_k = self.config.get(pool_info['top_k_config'], pool_info['default_top_k'])

                        for pref_key in pool_info['pref_keys']:
                            pref = attn.get(pref_key)
                            if pref is None:
                                continue

                            k = min(top_k, pref.shape[-1])
                            _, topk_idx = torch.topk(pref, k, dim=-1)

                            flat_idx = topk_idx.view(-1).cpu().numpy()
                            unique, counts = np.unique(flat_idx, return_counts=True)
                            for idx, cnt in zip(unique, counts):
                                neuron_counts[pool_name][int(idx)] += int(cnt)

        print(f"\n  Analyzed {total_tokens:,} tokens across {n_batches} batches")
        results['total_tokens'] = total_tokens

        for pool_name, counts in neuron_counts.items():
            if not counts:
                continue

            pool_info = NEURON_POOLS[pool_name]
            n_neurons = getattr(self.router, pool_info['n_attr'], 0)

            top10 = counts.most_common(10)
            total_selections = sum(counts.values())
            n_active = len(counts)

            # Coverage analysis
            sorted_counts = sorted(counts.values(), reverse=True)
            cumsum = np.cumsum(sorted_counts)
            n_for_50 = int(np.searchsorted(cumsum, total_selections * 0.5) + 1)
            n_for_90 = int(np.searchsorted(cumsum, total_selections * 0.9) + 1)

            print(f"\n  {pool_info['display']}:")
            print(f"    Total selections: {total_selections:,}")
            print(f"    Active neurons: {n_active}/{n_neurons}")
            print(f"    Coverage: {n_for_50} neurons for 50%, {n_for_90} for 90%")
            print(f"    Top 10: {[(idx, f'{cnt/total_selections*100:.1f}%') for idx, cnt in top10]}")

            results[pool_name] = {
                'display': pool_info['display'],
                'total_selections': total_selections,
                'n_active': n_active,
                'n_total': n_neurons,
                'n_for_50_coverage': n_for_50,
                'n_for_90_coverage': n_for_90,
                'top10': [(int(idx), int(cnt)) for idx, cnt in top10],
            }

        return results

    def diagnose_routing_entropy(self, dataloader, n_batches: int = 20) -> dict:
        """Analyze routing entropy (how concentrated/diverse routing is)"""
        print("\n" + "="*60)
        print("3. ROUTING ENTROPY ANALYSIS")
        print("="*60)

        results = {}
        entropies = {pool: [] for pool in NEURON_POOLS.keys()}

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc='Entropy', total=n_batches)):
                if i >= n_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                elif isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                try:
                    outputs = self.model(input_ids, return_routing_info=True)
                    if not isinstance(outputs, tuple) or len(outputs) < 2:
                        continue
                    routing_infos = outputs[-1]
                    if not routing_infos:
                        continue
                except:
                    continue

                # Use first layer for entropy analysis
                attn = routing_infos[0].get('attention', {})

                for pool_name, pool_info in NEURON_POOLS.items():
                    for pref_key in pool_info['pref_keys']:
                        pref = attn.get(pref_key)
                        if pref is None:
                            continue

                        # Normalize to probability distribution
                        if pref.dim() == 3:  # [B, S, N]
                            p = F.softmax(pref, dim=-1)
                            ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean().item()
                        else:  # [B, N]
                            p = F.softmax(pref, dim=-1)
                            ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean().item()

                        max_ent = np.log(pref.shape[-1])
                        norm_ent = ent / max_ent * 100

                        entropies[pool_name].append(norm_ent)

        print(f"\n  Entropy (% of maximum):")
        print(f"  {'Pool':<10} {'Mean':<10} {'Std':<10} {'Interpretation'}")
        print(f"  {'-'*50}")

        for pool_name, pool_info in NEURON_POOLS.items():
            if not entropies[pool_name]:
                continue

            mean_ent = np.mean(entropies[pool_name])
            std_ent = np.std(entropies[pool_name])

            if mean_ent < 30:
                interp = "CONCENTRATED (few neurons dominate)"
            elif mean_ent < 60:
                interp = "MODERATE"
            else:
                interp = "DIVERSE (many neurons active)"

            print(f"  {pool_info['display']:<10} {mean_ent:.1f}%{'':<5} {std_ent:.1f}%{'':<5} {interp}")

            results[pool_name] = {
                'display': pool_info['display'],
                'mean_entropy_pct': mean_ent,
                'std_entropy_pct': std_ent,
                'interpretation': interp,
            }

        return results

    def diagnose_dead_neurons(self, dataloader, n_batches: int = 30) -> dict:
        """Identify dead or rarely-used neurons"""
        print("\n" + "="*60)
        print("4. DEAD NEURON DETECTION")
        print("="*60)

        results = {}
        activation_counts = {}

        for pool_name, pool_info in NEURON_POOLS.items():
            n_neurons = getattr(self.router, pool_info['n_attr'], 0)
            if n_neurons > 0:
                activation_counts[pool_name] = torch.zeros(n_neurons, device=self.device)

        self.model.eval()
        total_batches = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc='Dead neurons', total=n_batches)):
                if i >= n_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                elif isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                total_batches += 1

                try:
                    outputs = self.model(input_ids, return_routing_info=True)
                    if not isinstance(outputs, tuple) or len(outputs) < 2:
                        continue
                    routing_infos = outputs[-1]
                    if not routing_infos:
                        continue
                except:
                    continue

                for layer_info in routing_infos:
                    attn = layer_info.get('attention', {})

                    for pool_name, pool_info in NEURON_POOLS.items():
                        if pool_name not in activation_counts:
                            continue

                        for pref_key in pool_info['pref_keys']:
                            pref = attn.get(pref_key)
                            if pref is None:
                                continue

                            # Count neurons that were selected (above threshold)
                            thresh = 1.0 / pref.shape[-1]
                            selected = (pref > thresh).float()

                            if selected.dim() == 3:
                                selected = selected.sum(dim=[0, 1])
                            else:
                                selected = selected.sum(dim=0)

                            n_pool = activation_counts[pool_name].shape[0]
                            if selected.shape[0] == n_pool:
                                activation_counts[pool_name] += selected

        print(f"\n  Analyzed {total_batches} batches")
        print(f"\n  {'Pool':<10} {'Dead':<10} {'Rare':<10} {'Active':<10} {'Status'}")
        print(f"  {'-'*50}")

        for pool_name, counts in activation_counts.items():
            pool_info = NEURON_POOLS[pool_name]
            n_neurons = len(counts)

            counts_np = counts.cpu().numpy()
            dead = int((counts_np == 0).sum())
            rare = int(((counts_np > 0) & (counts_np < total_batches)).sum())
            active = int((counts_np >= total_batches).sum())

            if dead > n_neurons * 0.3:
                status = "CRITICAL"
            elif dead > n_neurons * 0.1:
                status = "WARNING"
            else:
                status = "OK"

            print(f"  {pool_info['display']:<10} {dead:<10} {rare:<10} {active:<10} {status}")

            results[pool_name] = {
                'display': pool_info['display'],
                'dead': dead,
                'dead_pct': 100 * dead / n_neurons,
                'rare': rare,
                'active': active,
                'status': status,
                'dead_indices': np.where(counts_np == 0)[0].tolist()[:20],  # First 20
            }

        return results

    def diagnose_neuron_roles(self, dataloader, target_neurons: dict = None, n_batches: int = 15) -> dict:
        """Analyze what tokens each neuron prefers"""
        print("\n" + "="*60)
        print("5. NEURON ROLE ANALYSIS")
        print("="*60)

        # Determine target neurons (top 3 per pool if not specified)
        if target_neurons is None:
            target_neurons = {}
            for pool_name, pool_info in NEURON_POOLS.items():
                ema = getattr(self.router, pool_info['ema_attr'], None)
                if ema is not None:
                    top3 = torch.argsort(ema, descending=True)[:3].cpu().tolist()
                    target_neurons[pool_name] = top3

        print(f"  Analyzing neurons: {target_neurons}")

        results = {}
        neuron_tokens = defaultdict(lambda: defaultdict(list))

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc='Neuron roles', total=n_batches)):
                if i >= n_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                elif isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                B, L = input_ids.shape
                tokens_batch = input_ids.cpu().tolist()

                try:
                    outputs = self.model(input_ids, return_routing_info=True)
                    if not isinstance(outputs, tuple) or len(outputs) < 2:
                        continue
                    routing_infos = outputs[-1]
                    if not routing_infos:
                        continue
                except:
                    continue

                attn = routing_infos[0].get('attention', {})

                for pool_name, neuron_indices in target_neurons.items():
                    pool_info = NEURON_POOLS.get(pool_name)
                    if pool_info is None:
                        continue

                    for pref_key in pool_info['pref_keys']:
                        pref = attn.get(pref_key)
                        if pref is None:
                            continue

                        for neuron_idx in neuron_indices:
                            if neuron_idx >= pref.shape[-1]:
                                continue

                            if pref.dim() == 3:  # [B, S, N]
                                neuron_pref = pref[:, :, neuron_idx]  # [B, S]

                                for b in range(B):
                                    tokens = tokens_batch[b]
                                    prefs_seq = neuron_pref[b].cpu().numpy()

                                    top_pos = np.argsort(prefs_seq)[-5:][::-1]

                                    for pos in top_pos:
                                        if pos >= len(tokens):
                                            continue
                                        tok = self.tokenizer.decode([tokens[pos]]).strip()
                                        score = float(prefs_seq[pos])
                                        neuron_tokens[pool_name][neuron_idx].append((tok, score))

        # Analyze collected tokens
        for pool_name, neurons_data in neuron_tokens.items():
            pool_info = NEURON_POOLS[pool_name]
            results[pool_name] = {}

            for neuron_idx, token_list in neurons_data.items():
                if not token_list:
                    continue

                # Count token frequencies
                token_counts = Counter(tok for tok, _ in token_list)
                top_tokens = token_counts.most_common(15)

                print(f"\n  {pool_info['display']}_{neuron_idx}:")
                print(f"    Top tokens: {[t for t, _ in top_tokens[:10]]}")

                # POS analysis if spacy available
                pos_counts = Counter()
                if HAS_SPACY:
                    for tok, _ in token_list[:200]:
                        if tok and tok not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                            try:
                                doc = nlp(tok)
                                if doc:
                                    pos_counts[doc[0].pos_] += 1
                            except:
                                pass

                    if pos_counts:
                        print(f"    POS: {dict(pos_counts.most_common(5))}")

                results[pool_name][neuron_idx] = {
                    'top_tokens': top_tokens,
                    'pos_distribution': dict(pos_counts.most_common(10)) if pos_counts else {},
                    'n_samples': len(token_list),
                }

        return results


def visualize_diagnostics(ema_results: dict, selection_results: dict, output_dir: str):
    """Visualize diagnostic results"""
    if not HAS_MATPLOTLIB:
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Usage EMA distribution
    n_pools = len([k for k in ema_results if 'all_usage' in ema_results.get(k, {})])
    if n_pools > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (pool_name, data) in enumerate(ema_results.items()):
            if 'all_usage' not in data or idx >= 6:
                continue

            ax = axes[idx]
            usage = np.array(data['all_usage'])
            pool_info = NEURON_POOLS.get(pool_name, {})

            ax.hist(usage, bins=30, alpha=0.7, color=pool_info.get('color', 'gray'), edgecolor='black')
            ax.axvline(x=0.01, color='red', linestyle='--', label='Dead threshold')
            ax.set_xlabel('Usage EMA')
            ax.set_ylabel('Count')
            ax.set_title(f'{data["display"]}: {data["dead_count"]} dead ({data["dead_pct"]:.1f}%)')
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'usage_ema_distribution.png'), dpi=150)
        plt.close()

    # 2. Selection coverage comparison
    if selection_results:
        pools = [k for k in selection_results if k != 'total_tokens' and 'n_for_50_coverage' in selection_results.get(k, {})]

        if pools:
            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(len(pools))
            width = 0.35

            coverage_50 = [selection_results[p]['n_for_50_coverage'] for p in pools]
            coverage_90 = [selection_results[p]['n_for_90_coverage'] for p in pools]

            bars1 = ax.bar(x - width/2, coverage_50, width, label='50% coverage', color='steelblue')
            bars2 = ax.bar(x + width/2, coverage_90, width, label='90% coverage', color='coral')

            ax.set_xlabel('Neuron Pool')
            ax.set_ylabel('Number of Neurons')
            ax.set_title('Neurons Needed for X% of Selections')
            ax.set_xticks(x)
            ax.set_xticklabels([selection_results[p]['display'] for p in pools])
            ax.legend()

            for bar, val in zip(bars1, coverage_50):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(val), ha='center', va='bottom')
            for bar, val in zip(bars2, coverage_90):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(val), ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'selection_coverage.png'), dpi=150)
            plt.close()

    print(f"\n  Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='DAWN Routing Diagnostics')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--val_data', required=True, help='Validation data path')
    parser.add_argument('--output_dir', default='./dawn_diagnostics', help='Output directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_batches', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--skip_roles', action='store_true', help='Skip neuron role analysis')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, tokenizer, config = load_model(args.checkpoint, device)
    router = get_router(model)

    if router is None:
        print("ERROR: Could not find router in model")
        return

    dataloader, _ = create_dataloader(args.val_data, tokenizer, args.batch_size)
    diagnostics = RoutingDiagnostics(model, router, tokenizer, config, device)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("DAWN ROUTING DIAGNOSTICS (v17.1)")
    print("="*60)

    all_results = {}

    # Run diagnostics
    all_results['usage_ema'] = diagnostics.diagnose_usage_ema()
    all_results['selection'] = diagnostics.diagnose_neuron_selection(dataloader, args.max_batches)
    all_results['entropy'] = diagnostics.diagnose_routing_entropy(dataloader, args.max_batches // 2)
    all_results['dead_neurons'] = diagnostics.diagnose_dead_neurons(dataloader, args.max_batches)

    if not args.skip_roles:
        all_results['neuron_roles'] = diagnostics.diagnose_neuron_roles(dataloader, n_batches=args.max_batches // 2)

    # Visualize
    if HAS_MATPLOTLIB:
        visualize_diagnostics(all_results['usage_ema'], all_results['selection'], args.output_dir)

    # Save results
    # Clean for JSON
    json_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            json_results[k] = {kk: {kkk: vvv for kkk, vvv in vv.items() if kkk != 'all_usage'}
                              if isinstance(vv, dict) else vv
                              for kk, vv in v.items()}
        else:
            json_results[k] = v

    with open(os.path.join(args.output_dir, 'routing_diagnostics.json'), 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {args.output_dir}")
    print("="*60)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Overall health assessment
    issues = []

    for pool_name, data in all_results.get('dead_neurons', {}).items():
        if isinstance(data, dict) and data.get('status') == 'CRITICAL':
            issues.append(f"{data['display']}: {data['dead']} dead neurons (CRITICAL)")
        elif isinstance(data, dict) and data.get('status') == 'WARNING':
            issues.append(f"{data['display']}: {data['dead']} dead neurons (WARNING)")

    for pool_name, data in all_results.get('usage_ema', {}).items():
        if isinstance(data, dict) and data.get('gini', 0) > 0.5:
            issues.append(f"{data['display']}: High usage inequality (Gini={data['gini']:.3f})")

    if issues:
        print("\n  ISSUES DETECTED:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  All routing metrics within normal range.")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
