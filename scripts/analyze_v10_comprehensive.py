"""
DAWN v10.0 Comprehensive Analysis & Bottleneck Diagnosis
=========================================================

v10 구조:
- CompressNeurons: Q/K/V/M 공유 [n_compress, d_model, rank]
- ExpandNeurons: O 공유 [n_expand, rank, d_model]
- KnowledgeNeurons: [n_knowledge, rank] + [n_knowledge, d_model]
- Soft routing (no Householder)

분석 항목:
1. Compression Bottleneck - 정보 손실 측정
2. Memory Retrieval Quality - Query 구분력
3. Routing Patterns - 뉴런 활용도
4. Q/K/V/M/O Diversity - 컴포넌트별 차이
5. Knowledge Neuron Health - 사용 분포
6. Attention vs Memory Balance - 기여도 분석
7. Gradient Flow Analysis - 학습 병목
8. Information Flow - Norm 변화
9. Token/POS Specialization - 뉴런 특화
10. Recommendations - 자동 진단

Usage:
    python analyze_v10_comprehensive.py --checkpoint <path> --val_data <path>
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


class DAWNv10Analyzer:
    """Comprehensive analyzer for DAWN v10.0"""

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

        # Components
        self.components = ['Q', 'K', 'V', 'O', 'M']

        print(f"\n{'='*60}")
        print(f"DAWN v10.0 Analyzer Initialized")
        print(f"{'='*60}")
        print(f"d_model: {self.d_model}, rank: {self.rank}")
        print(f"n_compress: {self.n_compress}, n_expand: {self.n_expand}")
        print(f"n_knowledge: {self.n_knowledge}, knowledge_k: {self.knowledge_k}")
        print(f"n_layers: {self.n_layers}, n_heads: {self.n_heads}")

    # ============================================================
    # 1. COMPRESSION QUALITY ANALYSIS (핵심!)
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

        self.model.eval()

        # Storage for analysis
        query_discriminability = {comp: [] for comp in ['Q', 'K', 'V', 'M']}
        output_diversity = []  # O expander output diversity
        variance_retention = {comp: [] for comp in ['Q', 'K', 'V', 'M']}
        layer_discriminability = {l: {comp: [] for comp in ['Q', 'K', 'V', 'M']} for l in range(self.n_layers)}

        # Singular value storage
        compressed_sv = {comp: [] for comp in ['Q', 'K', 'V', 'M']}

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
                    # Get compressed representation
                    compressed, _ = compressor(x_norm)  # [B, S, rank]

                    # 1. Query Discriminability: cosine distance between different positions
                    # Higher = better (different inputs → different outputs)
                    compressed_flat = compressed.reshape(B * S, self.rank)
                    compressed_norm = F.normalize(compressed_flat, dim=-1)

                    # Sample pairs to compute discriminability
                    n_samples = min(256, B * S)
                    idx = torch.randperm(B * S)[:n_samples]
                    sampled = compressed_norm[idx]

                    # Pairwise cosine similarity
                    sim_matrix = sampled @ sampled.T
                    # Exclude diagonal (self-similarity)
                    sim_mask = ~torch.eye(n_samples, dtype=torch.bool, device=self.device)
                    avg_sim = sim_matrix[sim_mask].mean().item()
                    discriminability = 1.0 - avg_sim  # Higher = more discriminative

                    query_discriminability[comp_name].append(discriminability)
                    layer_discriminability[layer_idx][comp_name].append(discriminability)

                    # 2. Variance Retention
                    original_var = x_norm.var(dim=-1).mean()
                    compressed_var = compressed.var(dim=-1).mean()
                    ratio = compressed_var / (original_var + 1e-10)
                    variance_retention[comp_name].append(ratio.item())

                    # 3. Effective rank from singular values (sample)
                    if batch_idx < 5:
                        if compressed_flat.shape[0] > self.rank:
                            _, sv, _ = torch.linalg.svd(compressed_flat[:256], full_matrices=False)
                            compressed_sv[comp_name].append(sv.cpu().numpy())

                # Memory compressor
                x_norm2 = layer.norm2(x)
                mem_compressor = layer.memory.query_compressor
                mem_compressed, _ = mem_compressor(x_norm2)

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
                # Get attention output
                attn_out, attn_routing = layer.attn(layer.norm1(x), mask)

                # Measure diversity of O outputs
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
                mem_out, _ = layer.memory(layer.norm2(x))
                x = x + mem_out

        # Aggregate results
        results = {
            'query_discriminability': {},
            'output_diversity': {},
            'variance_retention': {},
            'effective_rank': {},
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

                # Diagnosis
                if mean_disc < 0.3:
                    status = "⚠️ LOW - queries too similar"
                elif mean_disc < 0.5:
                    status = "⚡ MODERATE"
                else:
                    status = "✅ GOOD"

                results['query_discriminability'][comp] = {'mean': mean_disc, 'std': std_disc}
                print(f"{comp:<12} {mean_disc:<12.4f} {std_disc:<12.4f} {status}")

        print(f"\n--- Output Diversity (O Expander) ---")
        mean_div = np.mean(output_diversity)
        std_div = np.std(output_diversity)
        results['output_diversity'] = {'mean': mean_div, 'std': std_div}
        status = "✅" if mean_div > 0.5 else ("⚡" if mean_div > 0.3 else "⚠️")
        print(f"Diversity: {mean_div:.4f} (std: {std_div:.4f}) {status}")

        print(f"\n--- Variance Retention (higher = better) ---")
        for comp in ['Q', 'K', 'V', 'M']:
            if variance_retention[comp]:
                mean_ratio = np.mean(variance_retention[comp])
                results['variance_retention'][comp] = mean_ratio
                status = "✅" if mean_ratio > 0.3 else "⚠️"
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

        # Effective rank from singular values
        print(f"\n--- Effective Rank (from singular values) ---")
        for comp in ['Q', 'K', 'V']:
            if compressed_sv[comp]:
                all_sv = np.concatenate(compressed_sv[comp])
                sv_norm = all_sv / (all_sv.sum() + 1e-10)
                entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-10))
                eff_rank = np.exp(entropy)
                results['effective_rank'][comp] = {
                    'effective_rank': eff_rank,
                    'max_rank': self.rank,
                    'utilization': eff_rank / self.rank
                }
                print(f"{comp}: effective_rank={eff_rank:.1f}/{self.rank} ({eff_rank/self.rank:.1%})")

        # Overall diagnosis
        avg_disc = np.mean([results['query_discriminability'][c]['mean'] for c in ['Q', 'K', 'V'] if c in results['query_discriminability']])

        if avg_disc < 0.3:
            results['diagnosis']['quality'] = "CRITICAL: Queries too similar, compression losing distinctiveness"
            results['diagnosis']['recommendation'] = "Increase rank or add contrastive loss"
        elif avg_disc < 0.5:
            results['diagnosis']['quality'] = "WARNING: Moderate query discriminability"
            results['diagnosis']['recommendation'] = "Consider increasing rank"
        else:
            results['diagnosis']['quality'] = "OK: Good query discriminability"
            results['diagnosis']['recommendation'] = "Current compression is adequate"

        if mean_div < 0.3:
            results['diagnosis']['output'] = "⚠️ Output diversity low - expander may be collapsing"
        else:
            results['diagnosis']['output'] = "✅ Output diversity OK"

        print(f"\n🔍 DIAGNOSIS: {results['diagnosis']['quality']}")
        print(f"   {results['diagnosis']['output']}")
        print(f"💡 RECOMMENDATION: {results['diagnosis']['recommendation']}")

        return results

    # Alias for backward compatibility
    def analyze_compression_bottleneck(self, dataloader, max_batches: int = 50) -> Dict:
        return self.analyze_compression_quality(dataloader, max_batches)

    # ============================================================
    # 2. MEMORY RETRIEVAL QUALITY
    # ============================================================

    @torch.no_grad()
    def analyze_memory_retrieval(self, dataloader, max_batches: int = 50) -> Dict:
        """
        Memory retrieval 품질 분석

        측정:
        1. Top-k score margin (구분력)
        2. Score distribution (entropy)
        3. Knowledge neuron coverage
        4. Query-Knowledge alignment
        """
        print(f"\n{'='*60}")
        print("2. MEMORY RETRIEVAL QUALITY")
        print(f"{'='*60}")

        self.model.eval()

        # Storage
        score_margins = []  # top1 - top(k+1)
        score_entropies = []
        coverage_per_layer = {l: set() for l in range(self.n_layers)}
        topk_concentration = []  # top-k가 전체의 몇 %?

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Memory Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                mem_routing = routing_info['memory']

                # Get scores before top-k (need to recompute)
                # For now, analyze the weights and indices
                k_indices = mem_routing['knowledge_indices']  # [B, S, k]
                k_weights = mem_routing['knowledge_weights']  # [B, S, k]

                # Coverage
                unique_indices = k_indices.unique().cpu().tolist()
                coverage_per_layer[layer_idx].update(unique_indices)

                # Top-k concentration (how much of softmax mass in top-k)
                # k_weights are already softmaxed over top-k
                # We need the raw scores to measure true margin

                # Weight entropy (lower = more concentrated)
                entropy = -(k_weights * torch.log(k_weights + 1e-10)).sum(dim=-1)
                max_entropy = math.log(self.knowledge_k)
                norm_entropy = (entropy / max_entropy).mean().item()
                score_entropies.append(norm_entropy)

                # Margin approximation: difference between top weights
                sorted_weights, _ = torch.sort(k_weights, dim=-1, descending=True)
                margin = (sorted_weights[:, :, 0] - sorted_weights[:, :, -1]).mean().item()
                score_margins.append(margin)

                # Top-1 concentration
                top1_weight = sorted_weights[:, :, 0].mean().item()
                topk_concentration.append(top1_weight)

        results = {
            'score_margin': {'mean': np.mean(score_margins), 'std': np.std(score_margins)},
            'score_entropy': {'mean': np.mean(score_entropies), 'std': np.std(score_entropies)},
            'top1_concentration': np.mean(topk_concentration),
            'coverage': {},
            'diagnosis': {}
        }

        print(f"\n--- Retrieval Statistics ---")
        print(f"Score margin (top1 - topk): {results['score_margin']['mean']:.4f}")
        print(f"Weight entropy (normalized): {results['score_entropy']['mean']:.4f}")
        print(f"Top-1 concentration: {results['top1_concentration']:.2%}")

        print(f"\n--- Knowledge Coverage by Layer ---")
        for layer_idx in range(self.n_layers):
            coverage = len(coverage_per_layer[layer_idx])
            pct = coverage / self.n_knowledge * 100
            results['coverage'][f"L{layer_idx}"] = coverage

            status = "✅" if pct > 70 else ("⚡" if pct > 40 else "⚠️")
            print(f"  Layer {layer_idx}: {coverage}/{self.n_knowledge} ({pct:.1f}%) {status}")

        # Diagnosis
        avg_coverage = np.mean(list(results['coverage'].values())) / self.n_knowledge

        if results['score_margin']['mean'] < 0.1:
            results['diagnosis']['margin'] = "⚠️ LOW margin - Query lacks discriminative power"
            results['diagnosis']['margin_fix'] = "Increase Memory query rank"
        else:
            results['diagnosis']['margin'] = "✅ Good margin"
            results['diagnosis']['margin_fix'] = None

        if avg_coverage < 0.5:
            results['diagnosis']['coverage'] = f"⚠️ LOW coverage ({avg_coverage:.1%}) - Many dead neurons"
            results['diagnosis']['coverage_fix'] = "Add diversity loss or reduce n_knowledge"
        else:
            results['diagnosis']['coverage'] = f"✅ Good coverage ({avg_coverage:.1%})"
            results['diagnosis']['coverage_fix'] = None

        print(f"\n🔍 DIAGNOSIS:")
        print(f"   Margin: {results['diagnosis']['margin']}")
        print(f"   Coverage: {results['diagnosis']['coverage']}")

        return results

    # ============================================================
    # 3. ROUTING PATTERNS
    # ============================================================

    @torch.no_grad()
    def analyze_routing_patterns(self, dataloader, max_batches: int = 50) -> Dict:
        """
        Soft routing 패턴 분석

        측정:
        1. Effective neurons used (from routing weights)
        2. Gini coefficient (불균등도)
        3. Routing entropy
        4. Dead neurons
        """
        print(f"\n{'='*60}")
        print("3. ROUTING PATTERNS")
        print(f"{'='*60}")

        self.model.eval()

        # Accumulate routing weights
        compress_weights = {comp: torch.zeros(self.n_layers, self.n_compress, device=self.device)
                          for comp in ['Q', 'K', 'V', 'M']}
        expand_weights = torch.zeros(self.n_layers, self.n_expand, device=self.device)

        total_tokens = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Routing Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape
            total_tokens += B * S

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                attn_routing = routing_info['attention']
                mem_routing = routing_info['memory']

                # Compress weights (soft routing)
                for comp in ['Q', 'K', 'V']:
                    weights = attn_routing[comp]['weights']  # [B, S, n_compress]
                    compress_weights[comp][layer_idx] += weights.sum(dim=(0, 1))

                # Memory compress
                m_weights = mem_routing['M']['weights']  # [B, S, n_compress]
                compress_weights['M'][layer_idx] += m_weights.sum(dim=(0, 1))

                # Expand weights
                o_weights = attn_routing['O']['weights']  # [B, S, n_expand]
                expand_weights[layer_idx] += o_weights.sum(dim=(0, 1))

        # Normalize
        for comp in compress_weights:
            compress_weights[comp] /= total_tokens
        expand_weights /= total_tokens

        results = {'compress': {}, 'expand': {}, 'summary': {}}

        def compute_stats(weights):
            """Compute effective rank, Gini, entropy from weight distribution"""
            # Normalize to probability
            probs = weights / (weights.sum() + 1e-10)

            # Effective rank
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            eff_rank = torch.exp(entropy).item()

            # Gini
            sorted_p, _ = torch.sort(probs)
            n = len(probs)
            index = torch.arange(1, n + 1, dtype=torch.float32, device=probs.device)
            gini = ((2 * index - n - 1) * sorted_p).sum() / (n * sorted_p.sum() + 1e-10)

            # Dead neurons (< 1% of average)
            threshold = 1.0 / len(probs) * 0.01
            dead = (probs < threshold).sum().item()

            return {
                'eff_rank': eff_rank,
                'max_neurons': len(probs),
                'utilization': eff_rank / len(probs),
                'gini': gini.item(),
                'dead_count': int(dead)
            }

        print(f"\n--- Compress Neuron Usage ---")
        print(f"{'Comp':<6} {'Layer':<8} {'Eff.Rank':<12} {'Util%':<10} {'Gini':<8} {'Dead':<8}")
        print("-" * 52)

        for comp in ['Q', 'K', 'V', 'M']:
            results['compress'][comp] = {}
            for layer_idx in range(self.n_layers):
                stats = compute_stats(compress_weights[comp][layer_idx])
                results['compress'][comp][f"L{layer_idx}"] = stats

                status = "✅" if stats['utilization'] > 0.5 else ("⚡" if stats['utilization'] > 0.3 else "⚠️")
                print(f"{comp:<6} L{layer_idx:<7} {stats['eff_rank']:<12.1f} {stats['utilization']*100:<10.1f} {stats['gini']:<8.3f} {stats['dead_count']:<8} {status}")

        print(f"\n--- Expand Neuron Usage ---")
        for layer_idx in range(self.n_layers):
            stats = compute_stats(expand_weights[layer_idx])
            results['expand'][f"L{layer_idx}"] = stats
            status = "✅" if stats['utilization'] > 0.5 else "⚠️"
            print(f"L{layer_idx}: eff_rank={stats['eff_rank']:.1f}/{self.n_expand}, gini={stats['gini']:.3f} {status}")

        # Summary
        all_utils = []
        for comp in results['compress']:
            for layer in results['compress'][comp]:
                all_utils.append(results['compress'][comp][layer]['utilization'])

        results['summary'] = {
            'avg_utilization': np.mean(all_utils),
            'min_utilization': np.min(all_utils),
            'diagnosis': None
        }

        if results['summary']['avg_utilization'] < 0.3:
            results['summary']['diagnosis'] = "⚠️ LOW utilization - reduce n_compress or add load balancing"
        elif results['summary']['avg_utilization'] > 0.8:
            results['summary']['diagnosis'] = "✅ HIGH utilization - consider increasing n_compress"
        else:
            results['summary']['diagnosis'] = "✅ Good utilization"

        print(f"\n🔍 DIAGNOSIS: {results['summary']['diagnosis']}")

        return results, compress_weights, expand_weights

    # ============================================================
    # 4. Q/K/V/M/O DIVERSITY
    # ============================================================

    @torch.no_grad()
    def analyze_component_diversity(self, dataloader, max_batches: int = 50) -> Dict:
        """
        Q/K/V/M/O가 서로 다른 뉴런을 사용하는지 분석

        같은 compress_neurons 풀을 공유하니까,
        다양하게 사용해야 각자 역할이 다른 것
        """
        print(f"\n{'='*60}")
        print("4. Q/K/V/M/O DIVERSITY")
        print(f"{'='*60}")

        self.model.eval()

        # Per-layer weight correlation
        weight_correlations = {l: {} for l in range(self.n_layers)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Diversity Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                attn = routing_info['attention']
                mem = routing_info['memory']

                # Get average weights per component
                weights = {
                    'Q': attn['Q']['weights'].mean(dim=(0, 1)),  # [n_compress]
                    'K': attn['K']['weights'].mean(dim=(0, 1)),
                    'V': attn['V']['weights'].mean(dim=(0, 1)),
                    'O': attn['O']['weights'].mean(dim=(0, 1)),  # [n_expand]
                    'M': mem['M']['weights'].mean(dim=(0, 1)),   # [n_compress]
                }

                # Pairwise correlation (only same-size)
                compress_comps = ['Q', 'K', 'V', 'M']
                for i, c1 in enumerate(compress_comps):
                    for c2 in compress_comps[i+1:]:
                        pair = f"{c1}-{c2}"
                        corr = F.cosine_similarity(
                            weights[c1].unsqueeze(0),
                            weights[c2].unsqueeze(0)
                        ).item()

                        if pair not in weight_correlations[layer_idx]:
                            weight_correlations[layer_idx][pair] = []
                        weight_correlations[layer_idx][pair].append(corr)

        results = {'layer_correlations': {}, 'avg_correlations': {}, 'diagnosis': {}}

        print(f"\n--- Weight Correlation (1.0 = identical, 0.0 = orthogonal) ---")

        # Collect all pairs
        all_pairs = set()
        for layer_idx in range(self.n_layers):
            all_pairs.update(weight_correlations[layer_idx].keys())

        print(f"{'Pair':<10}", end='')
        for layer_idx in range(self.n_layers):
            print(f"{'L'+str(layer_idx):<10}", end='')
        print(f"{'Avg':<10}")
        print("-" * (10 + 10 * (self.n_layers + 1)))

        for pair in sorted(all_pairs):
            print(f"{pair:<10}", end='')
            pair_avg = []
            for layer_idx in range(self.n_layers):
                if pair in weight_correlations[layer_idx]:
                    avg_corr = np.mean(weight_correlations[layer_idx][pair])
                    pair_avg.append(avg_corr)
                    results['layer_correlations'][f"L{layer_idx}_{pair}"] = avg_corr
                    print(f"{avg_corr:<10.3f}", end='')
                else:
                    print(f"{'N/A':<10}", end='')

            overall_avg = np.mean(pair_avg) if pair_avg else 0
            results['avg_correlations'][pair] = overall_avg
            status = "⚠️" if overall_avg > 0.8 else "✅"
            print(f"{overall_avg:<10.3f} {status}")

        # Q-K correlation is critical
        qk_corr = results['avg_correlations'].get('Q-K', 0)
        if qk_corr > 0.7:
            results['diagnosis']['QK'] = f"⚠️ Q-K too similar ({qk_corr:.2f}) - may hurt attention diversity"
        else:
            results['diagnosis']['QK'] = f"✅ Q-K diverse ({qk_corr:.2f})"

        # V-M should be different (attention vs memory)
        vm_corr = results['avg_correlations'].get('V-M', 0)
        if vm_corr > 0.7:
            results['diagnosis']['VM'] = f"⚠️ V-M too similar ({vm_corr:.2f}) - attention and memory overlap"
        else:
            results['diagnosis']['VM'] = f"✅ V-M diverse ({vm_corr:.2f})"

        print(f"\n🔍 DIAGNOSIS:")
        for key, msg in results['diagnosis'].items():
            print(f"   {msg}")

        return results

    # ============================================================
    # 5. KNOWLEDGE NEURON HEALTH
    # ============================================================

    @torch.no_grad()
    def analyze_knowledge_health(self, dataloader, max_batches: int = 50) -> Dict:
        """Knowledge neuron 건강도 분석"""
        print(f"\n{'='*60}")
        print("5. KNOWLEDGE NEURON HEALTH")
        print(f"{'='*60}")

        self.model.eval()

        # Usage tracking
        usage = torch.zeros(self.n_layers, self.n_knowledge, device=self.device)
        weight_sum = torch.zeros(self.n_layers, self.n_knowledge, device=self.device)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Knowledge Health", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                mem = routing_info['memory']
                k_idx = mem['knowledge_indices']  # [B, S, k]
                k_weights = mem['knowledge_weights']  # [B, S, k]

                # Count usage
                idx_flat = k_idx.reshape(-1)
                usage[layer_idx] += torch.bincount(idx_flat, minlength=self.n_knowledge).float()

                # Weight sum
                for i in range(self.knowledge_k):
                    weight_sum[layer_idx].scatter_add_(
                        0, k_idx[:, :, i].reshape(-1), k_weights[:, :, i].reshape(-1)
                    )

        results = {'layers': {}, 'global': {}, 'diagnosis': {}}

        # Per-layer analysis
        print(f"\n--- Knowledge Usage by Layer ---")
        print(f"{'Layer':<8} {'Used':<12} {'Dead':<10} {'Top10%':<12} {'Entropy':<10}")
        print("-" * 52)

        global_usage = usage.sum(dim=0)

        for layer_idx in range(self.n_layers):
            layer_usage = usage[layer_idx]
            layer_probs = layer_usage / (layer_usage.sum() + 1e-10)

            # Metrics
            used = (layer_usage > 0).sum().item()
            dead = self.n_knowledge - used

            # Top 10% concentration
            top_k = max(1, self.n_knowledge // 10)
            top_usage = torch.topk(layer_probs, top_k)[0].sum().item()

            # Entropy
            entropy = -(layer_probs * torch.log(layer_probs + 1e-10)).sum()
            max_entropy = math.log(self.n_knowledge)
            norm_entropy = (entropy / max_entropy).item()

            results['layers'][f"L{layer_idx}"] = {
                'used': used,
                'dead': dead,
                'top10_concentration': top_usage,
                'entropy': norm_entropy
            }

            status = "✅" if dead < self.n_knowledge * 0.2 else "⚠️"
            print(f"L{layer_idx:<7} {used:<12} {dead:<10} {top_usage:<12.1%} {norm_entropy:<10.3f} {status}")

        # Global
        global_probs = global_usage / (global_usage.sum() + 1e-10)
        global_dead = (global_usage == 0).sum().item()

        results['global'] = {
            'dead_neurons': global_dead,
            'dead_pct': global_dead / self.n_knowledge,
        }

        # Top neurons
        top_neurons = torch.topk(global_probs, 10)
        print(f"\n--- Top 10 Knowledge Neurons (global) ---")
        for i, (idx, prob) in enumerate(zip(top_neurons.indices.tolist(), top_neurons.values.tolist())):
            print(f"  #{idx}: {prob*100:.2f}%")

        # Dead neurons
        dead_indices = (global_usage == 0).nonzero().flatten().tolist()
        if dead_indices:
            print(f"\n⚠️ Dead neurons ({len(dead_indices)}): {dead_indices[:20]}...")

        # Diagnosis
        if global_dead > self.n_knowledge * 0.3:
            results['diagnosis']['dead'] = f"⚠️ Too many dead neurons ({global_dead}/{self.n_knowledge})"
            results['diagnosis']['fix'] = "Reduce n_knowledge or add diversity regularization"
        else:
            results['diagnosis']['dead'] = f"✅ Reasonable neuron utilization"
            results['diagnosis']['fix'] = None

        print(f"\n🔍 DIAGNOSIS: {results['diagnosis']['dead']}")

        return results, usage

    # ============================================================
    # 6. ATTENTION VS MEMORY BALANCE
    # ============================================================

    @torch.no_grad()
    def analyze_attn_mem_balance(self, dataloader, max_batches: int = 30) -> Dict:
        """Attention과 Memory의 기여도 비교"""
        print(f"\n{'='*60}")
        print("6. ATTENTION VS MEMORY BALANCE")
        print(f"{'='*60}")

        self.model.eval()

        layer_stats = {l: {
            'attn_norm': [], 'mem_norm': [],
            'attn_contribution': [], 'cosine_sim': []
        } for l in range(self.n_layers)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Attn-Mem Balance", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Manual forward
            pos = torch.arange(S, device=self.device).unsqueeze(0).expand(B, S)
            x = self.model.token_emb(input_ids) + self.model.pos_emb(pos)

            mask = torch.triu(torch.ones(S, S, device=self.device), diagonal=1).bool()
            mask = ~mask.unsqueeze(0).unsqueeze(0)

            for layer_idx, layer in enumerate(self.model.layers):
                # Attention
                x_norm = layer.norm1(x)
                attn_out, _ = layer.attn(x_norm, mask)
                x = x + attn_out

                # Memory
                x_norm = layer.norm2(x)
                mem_out, _ = layer.memory(x_norm)
                x = x + mem_out

                # Stats
                attn_norm = attn_out.norm(dim=-1).mean().item()
                mem_norm = mem_out.norm(dim=-1).mean().item()

                layer_stats[layer_idx]['attn_norm'].append(attn_norm)
                layer_stats[layer_idx]['mem_norm'].append(mem_norm)
                layer_stats[layer_idx]['attn_contribution'].append(
                    attn_norm / (attn_norm + mem_norm + 1e-10)
                )

                # Cosine similarity
                cos = F.cosine_similarity(
                    attn_out.reshape(-1, self.d_model),
                    mem_out.reshape(-1, self.d_model)
                ).mean().item()
                layer_stats[layer_idx]['cosine_sim'].append(cos)

        results = {'layers': {}, 'diagnosis': {}}

        print(f"\n--- Layer-wise Contribution ---")
        print(f"{'Layer':<8} {'Attn Norm':<12} {'Mem Norm':<12} {'Attn%':<10} {'Cos Sim':<10}")
        print("-" * 52)

        for layer_idx in range(self.n_layers):
            stats = layer_stats[layer_idx]

            avg_attn = np.mean(stats['attn_norm'])
            avg_mem = np.mean(stats['mem_norm'])
            avg_contrib = np.mean(stats['attn_contribution'])
            avg_cos = np.mean(stats['cosine_sim'])

            results['layers'][f"L{layer_idx}"] = {
                'attn_norm': avg_attn,
                'mem_norm': avg_mem,
                'attn_contribution': avg_contrib,
                'cosine_sim': avg_cos
            }

            # Balance check
            if avg_contrib > 0.8:
                status = "⚠️ Attn dominant"
            elif avg_contrib < 0.2:
                status = "⚠️ Mem dominant"
            else:
                status = "✅ Balanced"

            print(f"L{layer_idx:<7} {avg_attn:<12.3f} {avg_mem:<12.3f} {avg_contrib:<10.1%} {avg_cos:<10.3f} {status}")

        # Diagnosis
        avg_contrib_all = np.mean([results['layers'][f"L{l}"]['attn_contribution'] for l in range(self.n_layers)])

        if avg_contrib_all > 0.7:
            results['diagnosis']['balance'] = "⚠️ Attention dominates - Memory underutilized"
            results['diagnosis']['fix'] = "Memory capacity or query quality issue"
        elif avg_contrib_all < 0.3:
            results['diagnosis']['balance'] = "⚠️ Memory dominates - Attention may be weak"
            results['diagnosis']['fix'] = "Check attention compression quality"
        else:
            results['diagnosis']['balance'] = "✅ Reasonable balance"
            results['diagnosis']['fix'] = None

        print(f"\n🔍 DIAGNOSIS: {results['diagnosis']['balance']}")

        return results, layer_stats

    # ============================================================
    # 7. GRADIENT FLOW ANALYSIS
    # ============================================================

    def analyze_gradient_flow(self, dataloader, max_batches: int = 10) -> Dict:
        """Gradient flow 분석 - 학습 병목 찾기"""
        print(f"\n{'='*60}")
        print("7. GRADIENT FLOW ANALYSIS")
        print(f"{'='*60}")

        self.model.train()

        grad_norms = defaultdict(list)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Gradient Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            self.model.zero_grad()
            loss, _ = self.model(input_ids, labels=labels)
            loss.backward()

            # Collect gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()

                    # Categorize
                    if 'compress_neurons' in name:
                        grad_norms['compress_neurons'].append(grad_norm)
                    elif 'expand_neurons' in name:
                        grad_norms['expand_neurons'].append(grad_norm)
                    elif 'knowledge_K' in name:
                        grad_norms['knowledge_K'].append(grad_norm)
                    elif 'knowledge_V' in name:
                        grad_norms['knowledge_V'].append(grad_norm)
                    elif 'router' in name:
                        if 'compressor_Q' in name:
                            grad_norms['router_Q'].append(grad_norm)
                        elif 'compressor_K' in name:
                            grad_norms['router_K'].append(grad_norm)
                        elif 'compressor_V' in name:
                            grad_norms['router_V'].append(grad_norm)
                        elif 'expander_O' in name:
                            grad_norms['router_O'].append(grad_norm)
                        elif 'memory' in name:
                            grad_norms['router_M'].append(grad_norm)
                    elif 'token_emb' in name:
                        grad_norms['embedding'].append(grad_norm)
                    elif 'norm' in name:
                        grad_norms['layernorm'].append(grad_norm)

        self.model.eval()

        results = {'components': {}, 'diagnosis': {}}

        print(f"\n--- Gradient Norms by Component ---")
        print(f"{'Component':<20} {'Mean':<12} {'Std':<12} {'Status'}")
        print("-" * 56)

        all_means = []
        for comp, norms in sorted(grad_norms.items()):
            if norms:
                mean_norm = np.mean(norms)
                std_norm = np.std(norms)
                all_means.append(mean_norm)
                results['components'][comp] = {'mean': mean_norm, 'std': std_norm}

        # Relative comparison
        global_mean = np.mean(all_means) if all_means else 1.0

        for comp in sorted(grad_norms.keys()):
            if comp in results['components']:
                mean = results['components'][comp]['mean']
                std = results['components'][comp]['std']
                relative = mean / (global_mean + 1e-10)

                if relative < 0.1:
                    status = "⚠️ VANISHING"
                elif relative > 10:
                    status = "⚠️ EXPLODING"
                else:
                    status = "✅ OK"

                print(f"{comp:<20} {mean:<12.6f} {std:<12.6f} {status}")

        # Check for issues
        knowledge_grad = results['components'].get('knowledge_K', {}).get('mean', 0)
        compress_grad = results['components'].get('compress_neurons', {}).get('mean', 0)

        if knowledge_grad < compress_grad * 0.1:
            results['diagnosis']['knowledge'] = "⚠️ Knowledge gradient weak - retrieval not learning"
        else:
            results['diagnosis']['knowledge'] = "✅ Knowledge gradient OK"

        print(f"\n🔍 DIAGNOSIS:")
        for key, msg in results['diagnosis'].items():
            print(f"   {msg}")

        return results

    # ============================================================
    # 8. INFORMATION FLOW
    # ============================================================

    @torch.no_grad()
    def analyze_information_flow(self, dataloader, max_batches: int = 20) -> Dict:
        """레이어별 norm 변화 추적"""
        print(f"\n{'='*60}")
        print("8. INFORMATION FLOW")
        print(f"{'='*60}")

        self.model.eval()

        flow_stats = {l: {'input': [], 'after_attn': [], 'after_mem': []}
                     for l in range(self.n_layers)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Info Flow", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            pos = torch.arange(S, device=self.device).unsqueeze(0).expand(B, S)
            x = self.model.token_emb(input_ids) + self.model.pos_emb(pos)

            mask = torch.triu(torch.ones(S, S, device=self.device), diagonal=1).bool()
            mask = ~mask.unsqueeze(0).unsqueeze(0)

            for layer_idx, layer in enumerate(self.model.layers):
                flow_stats[layer_idx]['input'].append(x.norm(dim=-1).mean().item())

                attn_out, _ = layer.attn(layer.norm1(x), mask)
                x = x + attn_out
                flow_stats[layer_idx]['after_attn'].append(x.norm(dim=-1).mean().item())

                mem_out, _ = layer.memory(layer.norm2(x))
                x = x + mem_out
                flow_stats[layer_idx]['after_mem'].append(x.norm(dim=-1).mean().item())

        results = {'layers': {}}

        print(f"\n--- Norm Progression ---")
        print(f"{'Layer':<8} {'Input':<12} {'→Attn':<12} {'→Mem':<12} {'Growth':<10}")
        print("-" * 54)

        for layer_idx in range(self.n_layers):
            stats = flow_stats[layer_idx]

            input_norm = np.mean(stats['input'])
            attn_norm = np.mean(stats['after_attn'])
            mem_norm = np.mean(stats['after_mem'])
            growth = mem_norm / (input_norm + 1e-10)

            results['layers'][f"L{layer_idx}"] = {
                'input': input_norm,
                'after_attn': attn_norm,
                'after_mem': mem_norm,
                'growth': growth
            }

            status = "⚠️ HIGH" if growth > 2.0 else ("⚠️ LOW" if growth < 0.5 else "✅")
            print(f"L{layer_idx:<7} {input_norm:<12.2f} {attn_norm:<12.2f} {mem_norm:<12.2f} {growth:<10.2f}x {status}")

        return results

    # ============================================================
    # 9. TOKEN/POS SPECIALIZATION
    # ============================================================

    @torch.no_grad()
    def analyze_token_specialization(self, dataloader, max_batches: int = 50) -> Dict:
        """토큰/POS별 뉴런 특화 분석"""
        print(f"\n{'='*60}")
        print("9. TOKEN/POS SPECIALIZATION")
        print(f"{'='*60}")

        self.model.eval()

        # Track token -> neuron mapping (Q router, layer 0)
        pos_neuron_weights = defaultdict(lambda: torch.zeros(self.n_compress, device=self.device))
        pos_counts = defaultdict(float)

        token_neuron_weights = defaultdict(lambda: torch.zeros(self.n_compress, device=self.device))
        token_counts = defaultdict(float)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Specialization", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # Layer 0 Q routing
            q_weights = routing_infos[0]['attention']['Q']['weights']  # [B, S, n_compress]

            for b in range(B):
                for s in range(S):
                    tid = input_ids[b, s].item()
                    token = self.tokenizer.decode([tid]).strip()
                    pos = simple_pos_tag(token)

                    weights = q_weights[b, s]  # [n_compress]

                    pos_neuron_weights[pos] += weights
                    pos_counts[pos] += 1

                    token_neuron_weights[token] += weights
                    token_counts[token] += 1

        results = {'pos': {}, 'tokens': {}, 'neurons': {}}

        # POS patterns
        print(f"\n--- POS -> Preferred Neurons (L0 Q) ---")
        for pos in sorted(pos_counts.keys()):
            if pos_counts[pos] < 100:
                continue

            avg_weights = pos_neuron_weights[pos] / pos_counts[pos]
            top_neurons = torch.topk(avg_weights, 5)

            results['pos'][pos] = {
                'count': int(pos_counts[pos]),
                'top_neurons': top_neurons.indices.tolist(),
                'top_weights': top_neurons.values.tolist()
            }

            neurons_str = ', '.join([f'{n}({w:.2f})' for n, w in
                                    zip(top_neurons.indices.tolist()[:3], top_neurons.values.tolist()[:3])])
            print(f"  {pos:12s} (n={int(pos_counts[pos]):5d}): [{neurons_str}]")

        # Token patterns (top tokens)
        print(f"\n--- Top Tokens -> Preferred Neurons ---")
        top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:15]

        for token, count in top_tokens:
            if count < 10:
                continue

            avg_weights = token_neuron_weights[token] / count
            top_neurons = torch.topk(avg_weights, 3)

            results['tokens'][token] = {
                'count': int(count),
                'top_neurons': top_neurons.indices.tolist()
            }

            neurons_str = ', '.join([f'{n}' for n in top_neurons.indices.tolist()])
            print(f"  '{token:10s}' (n={int(count):4d}): neurons [{neurons_str}]")

        # Neuron -> preferred POS (reverse)
        print(f"\n--- Neuron -> Preferred POS ---")
        for neuron_idx in range(min(8, self.n_compress)):
            pos_prefs = []
            for pos in pos_counts:
                if pos_counts[pos] > 50:
                    avg_weight = (pos_neuron_weights[pos][neuron_idx] / pos_counts[pos]).item()
                    pos_prefs.append((pos, avg_weight))

            pos_prefs.sort(key=lambda x: x[1], reverse=True)
            top_pos = pos_prefs[:3]

            results['neurons'][neuron_idx] = [p[0] for p in top_pos]

            pos_str = ', '.join([f'{p[0]}({p[1]:.3f})' for p in top_pos])
            print(f"  Neuron {neuron_idx:2d}: [{pos_str}]")

        return results

    # ============================================================
    # 10. SHARED NEURONS HEALTH
    # ============================================================

    def analyze_shared_neurons_health(self) -> Dict:
        """SharedNeurons 파라미터 건강도"""
        print(f"\n{'='*60}")
        print("10. SHARED NEURONS HEALTH")
        print(f"{'='*60}")

        shared = self.model.shared_neurons
        results = {}

        # Compress neurons
        compress = shared.compress_neurons.data  # [n_compress, d_model, rank]

        print(f"\n--- Compress Neurons [{self.n_compress}, {self.d_model}, {self.rank}] ---")

        # Per-neuron analysis
        orthogonality_errors = []
        condition_numbers = []

        for i in range(self.n_compress):
            W = compress[i]  # [d_model, rank]

            # Orthogonality: W.T @ W should be ~I
            WtW = W.T @ W
            I = torch.eye(self.rank, device=W.device)
            orth_error = (WtW - I).abs().max().item()
            orthogonality_errors.append(orth_error)

            # Condition number
            _, s, _ = torch.linalg.svd(W, full_matrices=False)
            cond = (s[0] / (s[-1] + 1e-10)).item()
            condition_numbers.append(cond)

        results['compress'] = {
            'orth_error_mean': np.mean(orthogonality_errors),
            'orth_error_max': np.max(orthogonality_errors),
            'condition_mean': np.mean(condition_numbers),
            'condition_max': np.max(condition_numbers)
        }

        status = "✅" if results['compress']['orth_error_mean'] < 0.5 else "⚠️"
        print(f"  Orth error: mean={results['compress']['orth_error_mean']:.4f}, max={results['compress']['orth_error_max']:.4f} {status}")
        print(f"  Condition: mean={results['compress']['condition_mean']:.2f}, max={results['compress']['condition_max']:.2f}")

        # Expand neurons
        expand = shared.expand_neurons.data  # [n_expand, rank, d_model]

        print(f"\n--- Expand Neurons [{self.n_expand}, {self.rank}, {self.d_model}] ---")

        expand_orth_errors = []
        expand_conditions = []

        for i in range(self.n_expand):
            W = expand[i]  # [rank, d_model]

            # W @ W.T should be ~I
            WWt = W @ W.T
            I = torch.eye(self.rank, device=W.device)
            orth_error = (WWt - I).abs().max().item()
            expand_orth_errors.append(orth_error)

            _, s, _ = torch.linalg.svd(W, full_matrices=False)
            cond = (s[0] / (s[-1] + 1e-10)).item()
            expand_conditions.append(cond)

        results['expand'] = {
            'orth_error_mean': np.mean(expand_orth_errors),
            'orth_error_max': np.max(expand_orth_errors),
            'condition_mean': np.mean(expand_conditions),
            'condition_max': np.max(expand_conditions)
        }

        status = "✅" if results['expand']['orth_error_mean'] < 0.5 else "⚠️"
        print(f"  Orth error: mean={results['expand']['orth_error_mean']:.4f}, max={results['expand']['orth_error_max']:.4f} {status}")
        print(f"  Condition: mean={results['expand']['condition_mean']:.2f}, max={results['expand']['condition_max']:.2f}")

        # Knowledge neurons
        K = shared.knowledge_K.data  # [n_knowledge, rank]
        V = shared.knowledge_V.data  # [n_knowledge, d_model]

        print(f"\n--- Knowledge Neurons [{self.n_knowledge}] ---")

        # K diversity
        K_norm = F.normalize(K, dim=-1)
        K_sim = K_norm @ K_norm.T
        K_mask = ~torch.eye(self.n_knowledge, dtype=torch.bool, device=K_sim.device)
        K_off_diag = K_sim[K_mask]

        results['knowledge'] = {
            'K_norm_mean': K.norm(dim=-1).mean().item(),
            'K_sim_mean': K_off_diag.abs().mean().item(),
            'K_sim_max': K_off_diag.abs().max().item(),
            'V_norm_mean': V.norm(dim=-1).mean().item()
        }

        status = "✅" if results['knowledge']['K_sim_mean'] < 0.5 else "⚠️"
        print(f"  K norm: {results['knowledge']['K_norm_mean']:.4f}")
        print(f"  K similarity: mean={results['knowledge']['K_sim_mean']:.4f}, max={results['knowledge']['K_sim_max']:.4f} {status}")
        print(f"  V norm: {results['knowledge']['V_norm_mean']:.4f}")

        # Auxiliary losses
        print(f"\n--- Auxiliary Losses ---")
        orth_loss = self.model.orthogonality_loss().item()
        div_loss = self.model.knowledge_diversity_loss().item()

        results['losses'] = {
            'orthogonality': orth_loss,
            'knowledge_diversity': div_loss
        }

        print(f"  Orthogonality loss: {orth_loss:.6f}")
        print(f"  Knowledge diversity loss: {div_loss:.6f}")

        return results

    # ============================================================
    # RECOMMENDATIONS
    # ============================================================

    def generate_recommendations(self, all_results: Dict) -> List[str]:
        """분석 결과 기반 자동 권장사항 생성"""
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")

        recommendations = []

        # 1. Compression quality
        if 'compression' in all_results:
            comp = all_results['compression']
            if 'diagnosis' in comp:
                if 'quality' in comp['diagnosis'] and 'CRITICAL' in comp['diagnosis']['quality']:
                    recommendations.append(f"🔴 COMPRESSION: {comp['diagnosis'].get('recommendation', '')}")
                if 'output' in comp['diagnosis'] and '⚠️' in comp['diagnosis']['output']:
                    recommendations.append(f"🟡 OUTPUT: {comp['diagnosis']['output']}")

        # 2. Memory retrieval
        if 'memory_retrieval' in all_results:
            mem = all_results['memory_retrieval']
            if 'diagnosis' in mem:
                if 'margin_fix' in mem['diagnosis'] and mem['diagnosis']['margin_fix']:
                    recommendations.append(f"🟡 MEMORY: {mem['diagnosis']['margin_fix']}")
                if 'coverage_fix' in mem['diagnosis'] and mem['diagnosis']['coverage_fix']:
                    recommendations.append(f"🟡 KNOWLEDGE: {mem['diagnosis']['coverage_fix']}")

        # 3. Routing patterns
        if 'routing' in all_results:
            routing = all_results['routing']
            if 'summary' in routing and routing['summary'].get('diagnosis'):
                if '⚠️' in routing['summary']['diagnosis']:
                    recommendations.append(f"🟡 ROUTING: {routing['summary']['diagnosis']}")

        # 4. Component diversity
        if 'diversity' in all_results:
            div = all_results['diversity']
            if 'diagnosis' in div:
                for key, msg in div['diagnosis'].items():
                    if '⚠️' in msg:
                        recommendations.append(f"🟡 DIVERSITY: {msg}")

        # 5. Attention-Memory balance
        if 'attn_mem' in all_results:
            am = all_results['attn_mem']
            if 'diagnosis' in am and am['diagnosis'].get('fix'):
                recommendations.append(f"🟡 BALANCE: {am['diagnosis']['fix']}")

        # 6. Gradient flow
        if 'gradients' in all_results:
            grad = all_results['gradients']
            if 'diagnosis' in grad:
                for key, msg in grad['diagnosis'].items():
                    if '⚠️' in msg:
                        recommendations.append(f"🔴 GRADIENT: {msg}")

        if not recommendations:
            recommendations.append("✅ All metrics look healthy!")

        for rec in recommendations:
            print(f"  {rec}")

        return recommendations

    # ============================================================
    # RUN ALL
    # ============================================================

    def run_all(self, dataloader, max_batches: int = 50) -> Dict:
        """모든 분석 실행"""
        print(f"\n{'='*60}")
        print("DAWN v10.0 COMPREHENSIVE ANALYSIS")
        print(f"{'='*60}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        all_results = {}

        # 1. Compression bottleneck (CRITICAL)
        all_results['compression'] = self.analyze_compression_bottleneck(dataloader, max_batches)

        # 2. Memory retrieval
        all_results['memory_retrieval'] = self.analyze_memory_retrieval(dataloader, max_batches)

        # 3. Routing patterns
        routing_results, _, _ = self.analyze_routing_patterns(dataloader, max_batches)
        all_results['routing'] = routing_results

        # 4. Component diversity
        all_results['diversity'] = self.analyze_component_diversity(dataloader, max_batches)

        # 5. Knowledge health
        knowledge_results, _ = self.analyze_knowledge_health(dataloader, max_batches)
        all_results['knowledge'] = knowledge_results

        # 6. Attention-Memory balance
        attn_mem_results, _ = self.analyze_attn_mem_balance(dataloader, min(max_batches, 30))
        all_results['attn_mem'] = attn_mem_results

        # 7. Gradient flow
        all_results['gradients'] = self.analyze_gradient_flow(dataloader, min(max_batches, 10))

        # 8. Information flow
        all_results['info_flow'] = self.analyze_information_flow(dataloader, min(max_batches, 20))

        # 9. Token specialization
        all_results['specialization'] = self.analyze_token_specialization(dataloader, max_batches)

        # 10. SharedNeurons health
        all_results['shared_health'] = self.analyze_shared_neurons_health()

        # Final recommendations
        recommendations = self.generate_recommendations(all_results)
        all_results['recommendations'] = recommendations

        return all_results

    # ============================================================
    # VISUALIZATION
    # ============================================================

    def visualize(self, all_results: Dict, output_path: str):
        """종합 시각화"""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available, skipping visualization")
            return

        print(f"\n{'='*60}")
        print("CREATING VISUALIZATION")
        print(f"{'='*60}")

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))

        # 1. Query Discriminability
        ax = axes[0, 0]
        if 'compression' in all_results:
            comp = all_results['compression']
            if 'query_discriminability' in comp:
                comps = list(comp['query_discriminability'].keys())
                discs = [comp['query_discriminability'][c]['mean'] for c in comps]
                colors = ['red' if d < 0.3 else 'orange' if d < 0.5 else 'green' for d in discs]
                ax.bar(comps, discs, color=colors)
                ax.axhline(y=0.5, color='g', linestyle='--', alpha=0.5)
                ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5)
                ax.set_title('Query Discriminability (higher=better)')
                ax.set_ylabel('Discriminability')
                ax.set_ylim(0, 1)

        # 2. Output Diversity & Variance Retention
        ax = axes[0, 1]
        if 'compression' in all_results:
            comp = all_results['compression']
            metrics = ['O Diversity']
            values = [comp.get('output_diversity', {}).get('mean', 0)]
            if 'variance_retention' in comp:
                for c in ['Q', 'K', 'V', 'M']:
                    if c in comp['variance_retention']:
                        metrics.append(f'{c} Var')
                        values.append(comp['variance_retention'][c])
            colors = ['coral' if v < 0.3 else 'steelblue' for v in values]
            ax.bar(metrics, values, color=colors)
            ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.5)
            ax.set_title('Output Diversity & Variance')
            ax.set_ylabel('Value')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)

        # 3. Memory retrieval metrics
        ax = axes[0, 2]
        if 'memory_retrieval' in all_results:
            mem = all_results['memory_retrieval']
            metrics = ['score_margin', 'score_entropy', 'top1_concentration']
            values = [mem['score_margin']['mean'], mem['score_entropy']['mean'], mem['top1_concentration']]
            ax.bar(metrics, values, color='steelblue')
            ax.set_title('Memory Retrieval Quality')
            ax.set_xticklabels(metrics, rotation=45, ha='right')

        # 4. Knowledge coverage
        ax = axes[0, 3]
        if 'memory_retrieval' in all_results:
            mem = all_results['memory_retrieval']
            if 'coverage' in mem:
                layers = list(mem['coverage'].keys())
                coverage = [mem['coverage'][l] / self.n_knowledge * 100 for l in layers]
                ax.bar(layers, coverage, color='steelblue')
                ax.axhline(y=70, color='g', linestyle='--', alpha=0.5)
                ax.axhline(y=40, color='orange', linestyle='--', alpha=0.5)
                ax.set_title('Knowledge Coverage (%)')
                ax.set_ylabel('% Used')

        # 5. Routing utilization
        ax = axes[1, 0]
        if 'routing' in all_results:
            routing = all_results['routing']
            if 'compress' in routing:
                comps = []
                utils = []
                for comp in ['Q', 'K', 'V', 'M']:
                    if comp in routing['compress']:
                        for layer in routing['compress'][comp]:
                            comps.append(f"{comp}_{layer}")
                            utils.append(routing['compress'][comp][layer]['utilization'] * 100)

                if comps:
                    ax.bar(range(len(comps)), utils, color='steelblue')
                    ax.set_xticks(range(len(comps)))
                    ax.set_xticklabels(comps, rotation=45, ha='right', fontsize=8)
                    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
                    ax.set_title('Routing Utilization (%)')

        # 6. Component diversity (correlation heatmap)
        ax = axes[1, 1]
        if 'diversity' in all_results:
            div = all_results['diversity']
            if 'avg_correlations' in div:
                pairs = list(div['avg_correlations'].keys())
                corrs = [div['avg_correlations'][p] for p in pairs]
                colors = ['red' if c > 0.7 else 'steelblue' for c in corrs]
                ax.barh(pairs, corrs, color=colors)
                ax.axvline(x=0.7, color='r', linestyle='--', alpha=0.5)
                ax.set_title('Component Correlation')
                ax.set_xlabel('Cosine Similarity')

        # 7. Attention vs Memory
        ax = axes[1, 2]
        if 'attn_mem' in all_results:
            am = all_results['attn_mem']
            if 'layers' in am:
                layers = list(am['layers'].keys())
                attn_contrib = [am['layers'][l]['attn_contribution'] for l in layers]
                mem_contrib = [1 - c for c in attn_contrib]

                x = range(len(layers))
                ax.bar(x, attn_contrib, label='Attention', color='steelblue')
                ax.bar(x, mem_contrib, bottom=attn_contrib, label='Memory', color='coral')
                ax.set_xticks(x)
                ax.set_xticklabels(layers)
                ax.set_title('Attn vs Memory Contribution')
                ax.legend()

        # 8. Gradient flow
        ax = axes[1, 3]
        if 'gradients' in all_results:
            grad = all_results['gradients']
            if 'components' in grad:
                comps = list(grad['components'].keys())
                means = [grad['components'][c]['mean'] for c in comps]

                ax.barh(comps, means, color='steelblue')
                ax.set_title('Gradient Norms')
                ax.set_xlabel('Mean Gradient Norm')

        # 9. Information flow
        ax = axes[2, 0]
        if 'info_flow' in all_results:
            flow = all_results['info_flow']
            if 'layers' in flow:
                for stage in ['input', 'after_attn', 'after_mem']:
                    values = [flow['layers'][f"L{l}"][stage] for l in range(self.n_layers)]
                    ax.plot(range(self.n_layers), values, marker='o', label=stage)
                ax.set_title('Information Flow (Norms)')
                ax.set_xlabel('Layer')
                ax.legend()

        # 10. Knowledge neuron usage
        ax = axes[2, 1]
        if 'knowledge' in all_results:
            k = all_results['knowledge']
            if 'layers' in k:
                layers = list(k['layers'].keys())
                dead = [k['layers'][l]['dead'] for l in layers]
                used = [self.n_knowledge - d for d in dead]

                x = range(len(layers))
                ax.bar(x, used, label='Used', color='steelblue')
                ax.bar(x, dead, bottom=used, label='Dead', color='coral')
                ax.set_xticks(x)
                ax.set_xticklabels(layers)
                ax.set_title('Knowledge Neuron Usage')
                ax.legend()

        # 11. Shared neurons health
        ax = axes[2, 2]
        if 'shared_health' in all_results:
            health = all_results['shared_health']
            metrics = []
            values = []

            if 'compress' in health:
                metrics.append('Compress\nOrth')
                values.append(health['compress']['orth_error_mean'])
            if 'expand' in health:
                metrics.append('Expand\nOrth')
                values.append(health['expand']['orth_error_mean'])
            if 'knowledge' in health:
                metrics.append('Know\nSim')
                values.append(health['knowledge']['K_sim_mean'])

            if metrics:
                colors = ['red' if v > 0.5 else 'steelblue' for v in values]
                ax.bar(metrics, values, color=colors)
                ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
                ax.set_title('SharedNeurons Health')

        # 12. Recommendations summary
        ax = axes[2, 3]
        ax.axis('off')
        if 'recommendations' in all_results:
            recs = all_results['recommendations']
            text = "RECOMMENDATIONS:\n\n" + "\n\n".join(recs[:5])
            ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()

        print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN v10.0 Comprehensive Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--val_data', type=str,
                        default='/content/drive/MyDrive/data/validation/wikitext_5to1_texts.pkl',
                        help='Path to validation data')
    parser.add_argument('--max_batches', type=int, default=50,
                        help='Max batches for analysis')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./analysis_v10',
                        help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        best_model = checkpoint_path / 'best_model.pt'
        if best_model.exists():
            checkpoint_path = best_model
        else:
            pt_files = list(checkpoint_path.glob('*.pt'))
            if pt_files:
                checkpoint_path = max(pt_files, key=lambda p: p.stat().st_mtime)

    print(f"\nLoading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    # Import model
    try:
        from model_v10 import DAWN
    except ImportError:
        from models import create_model_by_version
        model_kwargs = {
            'vocab_size': config.get('vocab_size', 30522),
            'd_model': config.get('d_model', 320),
            'n_layers': config.get('n_layers', 4),
            'n_heads': config.get('n_heads', 4),
            'rank': config.get('rank', 80),
            'max_seq_len': config.get('max_seq_len', 128),
            'n_compress': config.get('n_compress', 48),
            'n_expand': config.get('n_expand', 12),
            'n_knowledge': config.get('n_knowledge', 80),
            'knowledge_k': config.get('knowledge_k', 10),
            'dropout': config.get('dropout', 0.1),
        }
        model = create_model_by_version('10.0', model_kwargs)
    else:
        model = DAWN(
            vocab_size=config.get('vocab_size', 30522),
            d_model=config.get('d_model', 320),
            n_layers=config.get('n_layers', 4),
            n_heads=config.get('n_heads', 4),
            rank=config.get('rank', 80),
            max_seq_len=config.get('max_seq_len', 128),
            n_compress=config.get('n_compress', 48),
            n_expand=config.get('n_expand', 12),
            n_knowledge=config.get('n_knowledge', 80),
            knowledge_k=config.get('knowledge_k', 10),
            dropout=config.get('dropout', 0.1),
        )

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Data
    print(f"\nLoading data: {args.val_data}")
    import pickle
    with open(args.val_data, 'rb') as f:
        val_texts = pickle.load(f)

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_len=128):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.max_len,
                padding='max_length',
                return_tensors='pt'
            )
            return {'input_ids': encoding['input_ids'].squeeze(0)}

    dataset = SimpleDataset(val_texts, tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Run analysis
    analyzer = DAWNv10Analyzer(model, tokenizer, device)
    all_results = analyzer.run_all(dataloader, max_batches=args.max_batches)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    results_path = os.path.join(args.output_dir, 'analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Visualization
    viz_path = os.path.join(args.output_dir, 'analysis_visualization.png')
    analyzer.visualize(all_results, viz_path)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
