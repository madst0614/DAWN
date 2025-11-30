"""
DAWN v10.1 Comprehensive Analysis & Bottleneck Diagnosis
=========================================================

v10.1 구조 (Top-K Sparse):
- CompressNeurons: Q/K/V/M 공유 [n_compress, d_model, rank] + top-k selection
- ExpandNeurons: O 공유 [n_expand, rank, d_model] + top-k selection
- KnowledgeNeurons: [n_knowledge, rank] + [n_knowledge, d_model]
- Hard routing (top-k) with soft weighting

분석 항목:
1. Compression Quality - 정보 손실 측정
2. Top-K Selection Patterns - 뉴런 선택 패턴
3. Load Balance - 균등 분포 확인
4. Memory Retrieval Quality - Query 구분력
5. Q/K/V/M/O Diversity - 컴포넌트별 차이
6. Knowledge Neuron Health - 사용 분포
7. Routing Specialization - 토큰별 특화
8. Recommendations - 자동 진단

Usage:
    python analyze_v10_1.py --checkpoint <path> --val_data <path>
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
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
            get_ipython().run_line_magic('matplotlib', 'inline')
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
    print("Warning: matplotlib not available")


def get_underlying_model(model):
    """Get the underlying model from torch.compile wrapper"""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def simple_pos_tag(token: str) -> str:
    """Simple rule-based POS tagging"""
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
    if token_lower.isdigit() or token_lower.replace('.', '').replace(',', '').isdigit():
        return 'NUM'
    if token_lower.endswith('ing'):
        return 'VERB_ING'
    if token_lower.endswith('ed'):
        return 'VERB_ED'
    if token_lower.endswith('ly'):
        return 'ADV'
    if token_lower.endswith(('tion', 'sion', 'ment', 'ness', 'ity')):
        return 'NOUN_SUFFIX'
    if token_lower.endswith(('ful', 'less', 'ous', 'ive', 'able', 'ible')):
        return 'ADJ_SUFFIX'
    return 'OTHER'


class DAWNv10_1Analyzer:
    """Comprehensive analyzer for DAWN v10.1 (Top-K Sparse)"""

    def __init__(self, model, tokenizer, device):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device

        # Model config
        self.n_layers = self.model.n_layers
        self.d_model = self.model.d_model
        self.rank = self.model.rank
        self.n_compress = self.model.n_compress
        self.n_expand = self.model.n_expand
        self.n_knowledge = self.model.n_knowledge
        self.knowledge_k = self.model.knowledge_k
        self.n_heads = self.model.n_heads

        # v10.1 specific
        self.compress_top_k = self.model.compress_top_k
        self.expand_top_k = self.model.expand_top_k

        # Components
        self.components = ['Q', 'K', 'V', 'O', 'M']

        print(f"\n{'='*60}")
        print(f"DAWN v10.1 Analyzer Initialized (Top-K Sparse)")
        print(f"{'='*60}")
        print(f"d_model: {self.d_model}, rank: {self.rank}")
        print(f"n_compress: {self.n_compress}, n_expand: {self.n_expand}")
        print(f"compress_top_k: {self.compress_top_k}, expand_top_k: {self.expand_top_k}")
        print(f"n_knowledge: {self.n_knowledge}, knowledge_k: {self.knowledge_k}")
        print(f"n_layers: {self.n_layers}, n_heads: {self.n_heads}")

    # ============================================================
    # 1. COMPRESSION QUALITY ANALYSIS
    # ============================================================

    @torch.no_grad()
    def analyze_compression_quality(self, dataloader, max_batches: int = 50) -> Dict:
        """
        핵심 분석: Compress/Expand 품질 측정

        측정:
        1. Query Discriminability - 서로 다른 입력이 다른 Query 만드는지
        2. Output Diversity - expand 결과가 다양한지
        3. Effective Rank - SVD 기반 실제 사용 차원
        4. Variance Retention - 정보 보존률
        """
        print(f"\n{'='*60}")
        print("1. COMPRESSION QUALITY ANALYSIS")
        print(f"{'='*60}")
        print(f"   d_model({self.d_model}) → rank({self.rank})")
        print(f"   Using top-{self.compress_top_k} of {self.n_compress} compress neurons")

        self.model.eval()

        # Storage for analysis
        query_discriminability = {comp: [] for comp in ['Q', 'K', 'V', 'M']}
        output_diversity = []
        variance_retention = {comp: [] for comp in ['Q', 'K', 'V', 'M']}
        layer_discriminability = {l: {comp: [] for comp in ['Q', 'K', 'V', 'M']} for l in range(self.n_layers)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Compression Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Get embeddings
            pos = torch.arange(S, device=self.device).unsqueeze(0).expand(B, S)
            x = self.model.token_emb(input_ids) + self.model.pos_emb(pos)

            # Causal mask
            mask = torch.triu(torch.ones(S, S, device=self.device), diagonal=1).bool()
            mask = ~mask.unsqueeze(0).unsqueeze(0)

            for layer_idx, layer in enumerate(self.model.layers):
                x_norm = layer.norm1(x)

                # Analyze each compressor
                compressors = {
                    'Q': layer.attn.compressor_Q,
                    'K': layer.attn.compressor_K,
                    'V': layer.attn.compressor_V,
                }

                for comp_name, compressor in compressors.items():
                    # Get compressed representation (no noise during analysis)
                    compressed, _ = compressor(x_norm, add_noise=False)  # [B, S, rank]

                    # 1. Query Discriminability
                    compressed_flat = compressed.reshape(B * S, self.rank)
                    compressed_norm = F.normalize(compressed_flat, dim=-1)

                    n_samples = min(256, B * S)
                    idx = torch.randperm(B * S)[:n_samples]
                    sampled = compressed_norm[idx]

                    sim_matrix = sampled @ sampled.T
                    sim_mask = ~torch.eye(n_samples, dtype=torch.bool, device=self.device)
                    avg_sim = sim_matrix[sim_mask].mean().item()
                    discriminability = 1.0 - avg_sim

                    query_discriminability[comp_name].append(discriminability)
                    layer_discriminability[layer_idx][comp_name].append(discriminability)

                    # 2. Variance Retention
                    original_var = x_norm.var(dim=-1).mean()
                    compressed_var = compressed.var(dim=-1).mean()
                    ratio = compressed_var / (original_var + 1e-10)
                    variance_retention[comp_name].append(ratio.item())

                # Memory compressor
                x_norm2 = layer.norm2(x)
                mem_compressor = layer.memory.query_compressor
                mem_compressed, _ = mem_compressor(x_norm2, add_noise=False)

                # Memory Query Discriminability
                mem_flat = mem_compressed.reshape(B * S, self.rank)
                mem_norm = F.normalize(mem_flat, dim=-1)
                n_samples = min(256, B * S)
                idx = torch.randperm(B * S)[:n_samples]
                sampled = mem_norm[idx]
                sim_matrix = sampled @ sampled.T
                sim_mask = ~torch.eye(n_samples, dtype=torch.bool, device=self.device)
                avg_sim = sim_matrix[sim_mask].mean().item()
                query_discriminability['M'].append(1.0 - avg_sim)
                layer_discriminability[layer_idx]['M'].append(1.0 - avg_sim)

                # Memory variance retention
                original_var = x_norm2.var(dim=-1).mean()
                compressed_var = mem_compressed.var(dim=-1).mean()
                variance_retention['M'].append((compressed_var / (original_var + 1e-10)).item())

                # 4. Output Diversity (O expander)
                attn_out, attn_routing = layer.attn(layer.norm1(x), mask, add_noise=False)
                o_flat = attn_out.reshape(B * S, self.d_model)
                o_norm = F.normalize(o_flat, dim=-1)
                n_samples = min(256, B * S)
                idx = torch.randperm(B * S)[:n_samples]
                sampled = o_norm[idx]
                sim_matrix = sampled @ sampled.T
                sim_mask = ~torch.eye(n_samples, dtype=torch.bool, device=self.device)
                avg_sim = sim_matrix[sim_mask].mean().item()
                output_diversity.append(1.0 - avg_sim)

                # Forward through layer
                x = x + attn_out
                mem_out, _ = layer.memory(layer.norm2(x), add_noise=False)
                x = x + mem_out

        # Aggregate results
        results = {
            'query_discriminability': {},
            'output_diversity': {},
            'variance_retention': {},
            'layer_breakdown': {},
            'diagnosis': {}
        }

        print(f"\n--- Query Discriminability (higher = better, max 1.0) ---")
        print(f"{'Component':<12} {'Mean':<12} {'Std':<12} {'Status'}")
        print("-" * 48)

        for comp in ['Q', 'K', 'V', 'M']:
            if query_discriminability[comp]:
                mean_disc = np.mean(query_discriminability[comp])
                std_disc = np.std(query_discriminability[comp])

                if mean_disc < 0.3:
                    status = "LOW - queries too similar"
                elif mean_disc < 0.5:
                    status = "MODERATE"
                else:
                    status = "GOOD"

                results['query_discriminability'][comp] = {'mean': mean_disc, 'std': std_disc}
                print(f"{comp:<12} {mean_disc:<12.4f} {std_disc:<12.4f} {status}")

        print(f"\n--- Output Diversity (O Expander) ---")
        mean_div = np.mean(output_diversity)
        std_div = np.std(output_diversity)
        results['output_diversity'] = {'mean': mean_div, 'std': std_div}
        status = "GOOD" if mean_div > 0.5 else ("MODERATE" if mean_div > 0.3 else "LOW")
        print(f"Diversity: {mean_div:.4f} (std: {std_div:.4f}) {status}")

        print(f"\n--- Variance Retention (higher = better) ---")
        for comp in ['Q', 'K', 'V', 'M']:
            if variance_retention[comp]:
                mean_ratio = np.mean(variance_retention[comp])
                results['variance_retention'][comp] = mean_ratio
                status = "GOOD" if mean_ratio > 0.3 else "LOW"
                print(f"{comp}: {mean_ratio:.2%} {status}")

        # Layer breakdown
        print(f"\n--- Layer-wise Discriminability ---")
        print(f"{'Layer':<8}", end='')
        for comp in ['Q', 'K', 'V', 'M']:
            print(f"{comp:<10}", end='')
        print()

        for layer_idx in range(self.n_layers):
            print(f"L{layer_idx:<7}", end='')
            for comp in ['Q', 'K', 'V', 'M']:
                if layer_discriminability[layer_idx][comp]:
                    disc = np.mean(layer_discriminability[layer_idx][comp])
                    results['layer_breakdown'][f"L{layer_idx}_{comp}"] = disc
                    print(f"{disc:<10.4f}", end='')
            print()

        # Overall diagnosis
        avg_disc = np.mean([results['query_discriminability'][c]['mean'] for c in ['Q', 'K', 'V'] if c in results['query_discriminability']])

        if avg_disc < 0.3:
            results['diagnosis']['quality'] = "CRITICAL: Queries too similar"
            results['diagnosis']['recommendation'] = "Increase rank or compress_top_k"
        elif avg_disc < 0.5:
            results['diagnosis']['quality'] = "WARNING: Moderate query discriminability"
            results['diagnosis']['recommendation'] = "Consider increasing compress_top_k"
        else:
            results['diagnosis']['quality'] = "OK: Good query discriminability"
            results['diagnosis']['recommendation'] = "Current settings are adequate"

        print(f"\n DIAGNOSIS: {results['diagnosis']['quality']}")
        print(f" RECOMMENDATION: {results['diagnosis']['recommendation']}")

        return results

    # ============================================================
    # 2. TOP-K SELECTION PATTERNS (v10.1 specific)
    # ============================================================

    @torch.no_grad()
    def analyze_topk_patterns(self, dataloader, max_batches: int = 50) -> Dict:
        """
        v10.1 핵심: Top-K 선택 패턴 분석

        측정:
        1. Neuron selection frequency - 각 뉴런이 선택된 횟수
        2. Selection diversity - 다양한 뉴런이 선택되는지
        3. Co-selection patterns - 어떤 뉴런들이 함께 선택되는지
        4. Layer-wise patterns - 레이어별 선택 패턴 차이
        """
        print(f"\n{'='*60}")
        print("2. TOP-K SELECTION PATTERNS (v10.1)")
        print(f"{'='*60}")
        print(f"   Compress: top-{self.compress_top_k} of {self.n_compress}")
        print(f"   Expand: top-{self.expand_top_k} of {self.n_expand}")

        self.model.eval()

        # Selection counters
        compress_counts = {comp: torch.zeros(self.n_layers, self.n_compress, device=self.device)
                          for comp in ['Q', 'K', 'V', 'M']}
        expand_counts = torch.zeros(self.n_layers, self.n_expand, device=self.device)

        # Co-selection matrix (for first layer only to save memory)
        coselect_Q = torch.zeros(self.n_compress, self.n_compress, device=self.device)

        total_tokens = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Top-K Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape
            total_tokens += B * S

            _, routing_infos = self.model(input_ids, return_routing_info=True, add_noise=False)

            for layer_idx, routing_info in enumerate(routing_infos):
                attn_routing = routing_info['attention']
                mem_routing = routing_info['memory']

                # Compress indices
                for comp in ['Q', 'K', 'V']:
                    indices = attn_routing[comp]['indices']  # [B, S, k]
                    flat_idx = indices.reshape(-1)
                    for idx in flat_idx:
                        compress_counts[comp][layer_idx, idx] += 1

                # Memory compress
                m_indices = mem_routing['M']['indices']  # [B, S, k]
                flat_idx = m_indices.reshape(-1)
                for idx in flat_idx:
                    compress_counts['M'][layer_idx, idx] += 1

                # Expand indices
                o_indices = attn_routing['O']['indices']  # [B, S, k]
                flat_idx = o_indices.reshape(-1)
                for idx in flat_idx:
                    expand_counts[layer_idx, idx] += 1

                # Co-selection (first layer Q only)
                if layer_idx == 0:
                    q_indices = attn_routing['Q']['indices']  # [B, S, k]
                    for b in range(B):
                        for s in range(S):
                            idxs = q_indices[b, s]
                            for i in idxs:
                                for j in idxs:
                                    if i != j:
                                        coselect_Q[i, j] += 1

        results = {'compress': {}, 'expand': {}, 'coselection': {}, 'summary': {}}

        # Normalize counts
        expected_count = total_tokens * self.compress_top_k / self.n_compress

        print(f"\n--- Compress Neuron Selection Frequency ---")
        print(f"Expected per neuron: {expected_count:.1f}")
        print(f"{'Comp':<6} {'Layer':<8} {'Min':<10} {'Max':<10} {'Std':<10} {'Dead':<8} {'Status'}")
        print("-" * 60)

        for comp in ['Q', 'K', 'V', 'M']:
            results['compress'][comp] = {}
            for layer_idx in range(self.n_layers):
                counts = compress_counts[comp][layer_idx]
                min_c = counts.min().item()
                max_c = counts.max().item()
                std_c = counts.std().item()
                dead = (counts < expected_count * 0.01).sum().item()

                # Imbalance ratio
                imbalance = max_c / (min_c + 1)

                results['compress'][comp][f"L{layer_idx}"] = {
                    'min': min_c, 'max': max_c, 'std': std_c,
                    'dead': dead, 'imbalance': imbalance
                }

                status = "GOOD" if imbalance < 5 else ("MODERATE" if imbalance < 10 else "IMBALANCED")
                print(f"{comp:<6} L{layer_idx:<7} {min_c:<10.0f} {max_c:<10.0f} {std_c:<10.1f} {dead:<8} {status}")

        print(f"\n--- Expand Neuron Selection Frequency ---")
        expected_expand = total_tokens * self.expand_top_k / self.n_expand
        print(f"Expected per neuron: {expected_expand:.1f}")

        for layer_idx in range(self.n_layers):
            counts = expand_counts[layer_idx]
            min_c = counts.min().item()
            max_c = counts.max().item()
            std_c = counts.std().item()
            dead = (counts < expected_expand * 0.01).sum().item()
            imbalance = max_c / (min_c + 1)

            results['expand'][f"L{layer_idx}"] = {
                'min': min_c, 'max': max_c, 'std': std_c,
                'dead': dead, 'imbalance': imbalance
            }

            status = "GOOD" if imbalance < 5 else ("MODERATE" if imbalance < 10 else "IMBALANCED")
            print(f"L{layer_idx}: min={min_c:.0f}, max={max_c:.0f}, dead={dead} {status}")

        # Co-selection analysis
        coselect_norm = coselect_Q / (coselect_Q.sum() + 1e-10)
        top_pairs = []
        for i in range(self.n_compress):
            for j in range(i+1, self.n_compress):
                if coselect_Q[i, j] > 0:
                    top_pairs.append((i, j, coselect_Q[i, j].item()))
        top_pairs.sort(key=lambda x: -x[2])

        print(f"\n--- Top Co-selected Neuron Pairs (Q, Layer 0) ---")
        for i, j, count in top_pairs[:10]:
            print(f"  Neurons ({i}, {j}): {count:.0f} times")

        results['coselection']['top_pairs'] = top_pairs[:20]

        # Summary
        all_imbalances = []
        for comp in results['compress']:
            for layer in results['compress'][comp]:
                all_imbalances.append(results['compress'][comp][layer]['imbalance'])

        avg_imbalance = np.mean(all_imbalances)
        results['summary']['avg_imbalance'] = avg_imbalance

        if avg_imbalance > 10:
            results['summary']['diagnosis'] = "HIGH imbalance - increase load_balance_weight"
        elif avg_imbalance > 5:
            results['summary']['diagnosis'] = "MODERATE imbalance - consider tuning"
        else:
            results['summary']['diagnosis'] = "GOOD balance"

        print(f"\n SUMMARY: Avg imbalance ratio = {avg_imbalance:.2f}")
        print(f" {results['summary']['diagnosis']}")

        return results

    # ============================================================
    # 3. LOAD BALANCE ANALYSIS
    # ============================================================

    @torch.no_grad()
    def analyze_load_balance(self, dataloader, max_batches: int = 50) -> Dict:
        """
        Load balance 분석 (v10.1 핵심)

        측정:
        1. Gini coefficient - 불균등도
        2. Dead neuron count - 사용되지 않는 뉴런
        3. Hot neuron count - 과도하게 사용되는 뉴런
        4. Entropy - 분포 균등성
        """
        print(f"\n{'='*60}")
        print("3. LOAD BALANCE ANALYSIS")
        print(f"{'='*60}")

        self.model.eval()

        # Accumulate selection counts
        compress_usage = {comp: torch.zeros(self.n_compress, device=self.device)
                        for comp in ['Q', 'K', 'V', 'M']}
        expand_usage = torch.zeros(self.n_expand, device=self.device)

        total_selections = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Load Balance Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape
            total_selections += B * S * self.n_layers

            _, routing_infos = self.model(input_ids, return_routing_info=True, add_noise=False)

            for layer_idx, routing_info in enumerate(routing_infos):
                attn_routing = routing_info['attention']
                mem_routing = routing_info['memory']

                for comp in ['Q', 'K', 'V']:
                    indices = attn_routing[comp]['indices'].reshape(-1)
                    compress_usage[comp].scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float))

                m_indices = mem_routing['M']['indices'].reshape(-1)
                compress_usage['M'].scatter_add_(0, m_indices, torch.ones_like(m_indices, dtype=torch.float))

                o_indices = attn_routing['O']['indices'].reshape(-1)
                expand_usage.scatter_add_(0, o_indices, torch.ones_like(o_indices, dtype=torch.float))

        results = {'compress': {}, 'expand': {}, 'overall': {}}

        def compute_balance_metrics(usage, n_neurons, top_k, name):
            """Compute load balance metrics"""
            # Expected usage
            total = usage.sum().item()
            expected = total / n_neurons

            # Normalize to probability
            probs = usage / (usage.sum() + 1e-10)

            # Gini coefficient
            sorted_p, _ = torch.sort(probs)
            n = len(probs)
            index = torch.arange(1, n + 1, dtype=torch.float32, device=probs.device)
            gini = ((2 * index - n - 1) * sorted_p).sum() / (n * sorted_p.sum() + 1e-10)

            # Entropy (normalized)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            max_entropy = math.log(n_neurons)
            norm_entropy = (entropy / max_entropy).item()

            # Dead neurons (< 1% of expected)
            dead = (usage < expected * 0.01).sum().item()

            # Hot neurons (> 300% of expected)
            hot = (usage > expected * 3.0).sum().item()

            # Load balance loss (like in training)
            target = total / n_neurons
            lb_loss = ((usage - target) ** 2).mean() / (target ** 2 + 1e-10)

            return {
                'gini': gini.item(),
                'entropy': norm_entropy,
                'dead': int(dead),
                'hot': int(hot),
                'lb_loss': lb_loss.item(),
                'expected': expected,
                'min': usage.min().item(),
                'max': usage.max().item(),
            }

        print(f"\n--- Compress Neuron Load Balance ---")
        print(f"{'Comp':<8} {'Gini':<10} {'Entropy':<10} {'Dead':<8} {'Hot':<8} {'LB Loss':<12}")
        print("-" * 56)

        for comp in ['Q', 'K', 'V', 'M']:
            metrics = compute_balance_metrics(compress_usage[comp], self.n_compress, self.compress_top_k, comp)
            results['compress'][comp] = metrics

            status = "GOOD" if metrics['gini'] < 0.3 else ("MODERATE" if metrics['gini'] < 0.5 else "POOR")
            print(f"{comp:<8} {metrics['gini']:<10.4f} {metrics['entropy']:<10.4f} {metrics['dead']:<8} {metrics['hot']:<8} {metrics['lb_loss']:<12.4f} {status}")

        print(f"\n--- Expand Neuron Load Balance ---")
        metrics = compute_balance_metrics(expand_usage, self.n_expand, self.expand_top_k, 'O')
        results['expand']['O'] = metrics
        status = "GOOD" if metrics['gini'] < 0.3 else ("MODERATE" if metrics['gini'] < 0.5 else "POOR")
        print(f"{'O':<8} {metrics['gini']:<10.4f} {metrics['entropy']:<10.4f} {metrics['dead']:<8} {metrics['hot']:<8} {metrics['lb_loss']:<12.4f} {status}")

        # Overall summary
        all_ginis = [results['compress'][c]['gini'] for c in ['Q', 'K', 'V', 'M']]
        all_ginis.append(results['expand']['O']['gini'])
        avg_gini = np.mean(all_ginis)

        all_dead = sum(results['compress'][c]['dead'] for c in ['Q', 'K', 'V', 'M'])
        all_dead += results['expand']['O']['dead']

        results['overall'] = {
            'avg_gini': avg_gini,
            'total_dead': all_dead,
        }

        print(f"\n OVERALL:")
        print(f"   Average Gini: {avg_gini:.4f}")
        print(f"   Total Dead Neurons: {all_dead}")

        if avg_gini > 0.5:
            print(f" RECOMMENDATION: Increase load_balance_weight (currently seeing high imbalance)")
        elif all_dead > 10:
            print(f" RECOMMENDATION: Increase router_noise for exploration")
        else:
            print(f" Load balance is acceptable")

        return results

    # ============================================================
    # 4. MEMORY RETRIEVAL QUALITY
    # ============================================================

    @torch.no_grad()
    def analyze_memory_retrieval(self, dataloader, max_batches: int = 50) -> Dict:
        """Memory retrieval 품질 분석"""
        print(f"\n{'='*60}")
        print("4. MEMORY RETRIEVAL QUALITY")
        print(f"{'='*60}")

        self.model.eval()

        score_margins = []
        score_entropies = []
        coverage_per_layer = {l: set() for l in range(self.n_layers)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Memory Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)

            _, routing_infos = self.model(input_ids, return_routing_info=True, add_noise=False)

            for layer_idx, routing_info in enumerate(routing_infos):
                mem_routing = routing_info['memory']

                k_indices = mem_routing['knowledge_indices']
                k_weights = mem_routing['knowledge_weights']

                # Coverage
                unique_indices = k_indices.unique().cpu().tolist()
                coverage_per_layer[layer_idx].update(unique_indices)

                # Weight entropy
                entropy = -(k_weights * torch.log(k_weights + 1e-10)).sum(dim=-1)
                max_entropy = math.log(self.knowledge_k)
                norm_entropy = (entropy / max_entropy).mean().item()
                score_entropies.append(norm_entropy)

                # Margin
                sorted_weights, _ = torch.sort(k_weights, dim=-1, descending=True)
                margin = (sorted_weights[:, :, 0] - sorted_weights[:, :, -1]).mean().item()
                score_margins.append(margin)

        results = {
            'score_margin': {'mean': np.mean(score_margins), 'std': np.std(score_margins)},
            'score_entropy': {'mean': np.mean(score_entropies), 'std': np.std(score_entropies)},
            'coverage': {},
        }

        print(f"\n--- Retrieval Statistics ---")
        print(f"Score margin (top1 - topk): {results['score_margin']['mean']:.4f}")
        print(f"Weight entropy (normalized): {results['score_entropy']['mean']:.4f}")

        print(f"\n--- Knowledge Coverage by Layer ---")
        for layer_idx in range(self.n_layers):
            coverage = len(coverage_per_layer[layer_idx])
            pct = coverage / self.n_knowledge * 100
            results['coverage'][f"L{layer_idx}"] = coverage
            status = "GOOD" if pct > 70 else ("MODERATE" if pct > 40 else "LOW")
            print(f"  Layer {layer_idx}: {coverage}/{self.n_knowledge} ({pct:.1f}%) {status}")

        return results

    # ============================================================
    # 5. RUN ALL ANALYSES
    # ============================================================

    def run_all(self, dataloader, max_batches: int = 50, save_path: Optional[str] = None) -> Dict:
        """Run all analyses"""
        print(f"\n{'='*70}")
        print(f"DAWN v10.1 COMPREHENSIVE ANALYSIS")
        print(f"{'='*70}")

        all_results = {}

        # 1. Compression Quality
        all_results['compression'] = self.analyze_compression_quality(dataloader, max_batches)

        # 2. Top-K Patterns
        all_results['topk'] = self.analyze_topk_patterns(dataloader, max_batches)

        # 3. Load Balance
        all_results['load_balance'] = self.analyze_load_balance(dataloader, max_batches)

        # 4. Memory Retrieval
        all_results['memory'] = self.analyze_memory_retrieval(dataloader, max_batches)

        # Final recommendations
        print(f"\n{'='*70}")
        print(f"FINAL RECOMMENDATIONS")
        print(f"{'='*70}")

        recommendations = []

        # Compression quality
        if 'diagnosis' in all_results['compression']:
            if 'CRITICAL' in all_results['compression']['diagnosis'].get('quality', ''):
                recommendations.append("- Increase rank or compress_top_k for better compression quality")

        # Load balance
        avg_gini = all_results['load_balance']['overall']['avg_gini']
        if avg_gini > 0.5:
            recommendations.append(f"- Increase load_balance_weight (Gini={avg_gini:.3f} is high)")

        total_dead = all_results['load_balance']['overall']['total_dead']
        if total_dead > 10:
            recommendations.append(f"- Increase router_noise for exploration ({total_dead} dead neurons)")

        # Top-K imbalance
        if all_results['topk']['summary']['avg_imbalance'] > 10:
            recommendations.append("- High selection imbalance - check load_balance_weight")

        if not recommendations:
            recommendations.append("- Model looks healthy! Continue training.")

        for rec in recommendations:
            print(rec)

        if save_path:
            # Convert tensors to serializable format
            def convert_to_serializable(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.cpu().tolist()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(v) for v in obj]
                else:
                    return obj

            serializable_results = convert_to_serializable(all_results)
            with open(save_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"\nResults saved to: {save_path}")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="DAWN v10.1 Analysis")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data')
    parser.add_argument('--max_batches', type=int, default=50, help='Max batches to analyze')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load model
    from models import create_model_by_version
    from utils.checkpoint import load_checkpoint_smart

    checkpoint_path = Path(args.checkpoint)
    config_path = checkpoint_path.parent / 'config.json'

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_version = config['model'].get('model_version', '10.1')

        model_kwargs = {
            'vocab_size': tokenizer.vocab_size,
            'd_model': config['model']['d_model'],
            'n_layers': config['model']['n_layers'],
            'n_heads': config['model']['n_heads'],
            'rank': config['model'].get('rank', config['model'].get('basis_rank', 64)),
            'max_seq_len': config['model'].get('max_seq_len', 128),
            'n_compress': config['model']['n_compress'],
            'n_expand': config['model']['n_expand'],
            'n_knowledge': config['model']['n_knowledge'],
            'knowledge_k': config['model']['knowledge_k'],
        }

        # v10.1 specific
        if model_version == '10.1':
            model_kwargs.update({
                'compress_top_k': config['model'].get('compress_top_k', 8),
                'expand_top_k': config['model'].get('expand_top_k', 4),
                'router_noise': config['model'].get('router_noise', 0.1),
            })

        model = create_model_by_version(model_version, model_kwargs)
    else:
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Load weights
    checkpoint, _ = load_checkpoint_smart(model, str(checkpoint_path), device=device)
    model = model.to(device)
    model.eval()

    # Load data
    import pickle
    from torch.utils.data import DataLoader, Dataset

    class SimpleDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len=128):
            self.tokenizer = tokenizer
            self.texts = texts
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            encoded = self.tokenizer(
                text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {'input_ids': encoded['input_ids'].squeeze(0)}

    with open(args.val_data, 'rb') as f:
        val_texts = pickle.load(f)

    val_dataset = SimpleDataset(val_texts[:5000], tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Run analysis
    analyzer = DAWNv10_1Analyzer(model, tokenizer, device)
    results = analyzer.run_all(val_loader, max_batches=args.max_batches, save_path=args.output)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
