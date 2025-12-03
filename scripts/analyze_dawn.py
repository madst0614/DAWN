#!/usr/bin/env python3
"""
DAWN Unified Analysis Script
============================

Version-aware analysis for v10.x and v12.x models.

Features:
1. Compression Quality - 압축 품질 측정
2. Routing Patterns - 뉴런 활용도
3. Knowledge Health - Knowledge 뉴런 건강도
4. Layer Analysis - 레이어별 역할
5. Token/POS Specialization - 뉴런 특화
6. Attention vs Memory Balance - 기여도 분석
7. Visualization - 종합 시각화

Usage:
    python analyze_dawn.py --checkpoint <path> --val_data <path>
    python analyze_dawn.py --checkpoint <path> --val_data <path> --output_dir ./analysis
"""

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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

try:
    from sklearn.cluster import KMeans
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


def convert_to_serializable(obj):
    """Convert numpy/torch types to Python native types"""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, Counter):
        return dict(obj)
    return obj


# ============================================================
# Version-Aware Routing Info Parser
# ============================================================

class RoutingInfoParser:
    """
    Version-aware routing info parser.

    v10.x format (TOKEN-LEVEL routing):
        routing_info['attention']['Q']['weights']  # [B, S, N] - per-token weights
        routing_info['attention']['Q']['indices']  # optional [B, S, k]
        routing_info['attention']['K']['weights']
        routing_info['attention']['V']['weights']
        routing_info['attention']['O']['weights']  # [B, S, n_expand]
        routing_info['memory']['M']['weights']
        routing_info['memory']['knowledge_indices']
        routing_info['memory']['knowledge_weights']

    v12.3-v12.6 format:
        # Token-level preferences (before SSM weighting):
        routing_info['attention']['compress_pref']    # [B, S, n_compress] - TOKEN-LEVEL!
        routing_info['attention']['expand_pref_Q']    # [B, S, n_expand] - TOKEN-LEVEL!
        routing_info['attention']['expand_pref_K/V']  # [B, S, n_expand]

        # Batch-level weights (after SSM importance weighting):
        # einsum('bs,bsn->bn', importance, pref)
        routing_info['attention']['compress_weights']  # [B, n_compress] - BATCH-LEVEL
        routing_info['attention']['expand_weights_Q']  # [B, n_expand] - BATCH-LEVEL
        routing_info['attention']['expand_weights_K/V']

        routing_info['memory']['neuron_weights']  # [B, n_compress]
        routing_info['memory']['knowledge_indices']
        routing_info['memory']['knowledge_weights']

    v12.7/v13 format (Top-k sparse routing):
        # Same as v12.3-v12.6 but with additional sparse/dense weights and top-k indices
        routing_info['attention']['compress_weights']        # [B, n_compress] - SPARSE (top-k)
        routing_info['attention']['compress_weights_dense']  # [B, n_compress] - DENSE (pre-top-k)
        routing_info['attention']['compress_topk_idx']       # [B, top_k_compress]
        routing_info['attention']['expand_weights_Q']        # [B, n_expand] - SPARSE
        routing_info['attention']['expand_weights_Q_dense']  # [B, n_expand] - DENSE
        routing_info['attention']['expand_topk_idx_Q']       # [B, top_k_expand]
        # ... same for K, V

        routing_info['memory']['memory_weights']       # [B, n_compress] - SPARSE
        routing_info['memory']['memory_weights_dense'] # [B, n_compress] - DENSE
        routing_info['memory']['memory_topk_idx']      # [B, top_k_compress]
        routing_info['memory']['knowledge_indices']
        routing_info['memory']['knowledge_weights']
    """

    def __init__(self, model_version: str):
        self.version = model_version
        self.is_v10 = model_version.startswith('10') or model_version in ['10.0', '11.0']
        self.is_v12 = model_version.startswith('12')
        self.is_v13 = model_version.startswith('13')
        self.is_topk = model_version in ['12.7', '12.8', '13', '13.0'] or model_version.startswith('13')

    def detect_version(self, routing_info: Dict) -> str:
        """Auto-detect version from routing_info structure"""
        if 'attention' in routing_info:
            attn = routing_info['attention']
            if 'Q' in attn and isinstance(attn['Q'], dict):
                return 'v10'
            elif 'compress_weights_dense' in attn:
                # v12.7, v12.8, v13 have dense weights for load balance loss
                return 'v12_topk'
            elif 'compress_weights' in attn:
                return 'v12'
        return 'unknown'

    def get_compress_weights(self, routing_info: Dict, comp: str = 'Q', use_dense: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get compress weights for a component.

        Args:
            routing_info: The routing info dict
            comp: Component name ('Q', 'K', 'V', 'M')
            use_dense: If True and available, return dense weights (pre-top-k) instead of sparse

        Returns:
            weights: [B, S, N] for v10.x (token-level) or [B, N] for v12.x (batch-level)
            indices: [B, S, k] or None (top-k indices for v12.7/v13)
        """
        version = self.detect_version(routing_info)

        if version == 'v10':
            if comp in ['Q', 'K', 'V']:
                data = routing_info['attention'].get(comp, {})
            else:  # M
                data = routing_info['memory'].get('M', {})

            weights = data.get('weights')
            indices = data.get('indices')
            return weights, indices

        elif version in ['v12', 'v12_topk']:
            attn = routing_info['attention']
            mem = routing_info['memory']

            if comp in ['Q', 'K', 'V']:
                # v12.7/v13: use dense weights if requested, else sparse
                if use_dense and 'compress_weights_dense' in attn:
                    weights = attn.get('compress_weights_dense')
                else:
                    weights = attn.get('compress_weights')
                indices = attn.get('compress_topk_idx')  # Top-k indices for v12.7/v13
            else:  # M
                if use_dense and 'memory_weights_dense' in mem:
                    weights = mem.get('memory_weights_dense')
                else:
                    weights = mem.get('memory_weights', mem.get('neuron_weights'))
                indices = mem.get('memory_topk_idx')

            return weights, indices

        return None, None

    def get_expand_weights(self, routing_info: Dict, comp: str = 'Q', use_dense: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get expand weights for a component.

        Args:
            routing_info: The routing info dict
            comp: Component name ('Q', 'K', 'V')
            use_dense: If True and available, return dense weights (pre-top-k) instead of sparse

        Returns:
            weights: [B, S, n_expand] for v10.x (token-level) or [B, n_expand] for v12.x (batch-level)
            indices: [B, S, k] or None (top-k indices for v12.7/v13)
        """
        version = self.detect_version(routing_info)

        if version == 'v10':
            data = routing_info['attention'].get('O', {})
            return data.get('weights'), data.get('indices')

        elif version in ['v12', 'v12_topk']:
            attn = routing_info['attention']

            # v12.7/v13: use dense weights if requested, else sparse
            if use_dense:
                key = f'expand_weights_{comp}_dense'
                if key in attn:
                    return attn.get(key), attn.get(f'expand_topk_idx_{comp}')

            key = f'expand_weights_{comp}'
            weights = attn.get(key)
            indices = attn.get(f'expand_topk_idx_{comp}')  # Top-k indices for v12.7/v13
            return weights, indices

        return None, None

    def get_knowledge_info(self, routing_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get knowledge neuron indices and weights.

        Returns:
            indices: [B, S, k]
            weights: [B, S, k]
        """
        mem = routing_info.get('memory', {})
        return mem.get('knowledge_indices'), mem.get('knowledge_weights')

    def get_token_level_pref(self, routing_info: Dict, comp: str = 'Q') -> Optional[torch.Tensor]:
        """
        Get token-level routing preferences (before SSM importance weighting).

        v10.x: same as get_compress_weights (already token-level)
        v12.3: compress_pref [B, S, n_compress] or expand_pref_Q/K/V [B, S, n_expand]

        Returns:
            pref: [B, S, N] - token-level preferences
        """
        version = self.detect_version(routing_info)

        if version == 'v10':
            # v10 is already token-level
            weights, _ = self.get_compress_weights(routing_info, comp)
            return weights

        elif version in ['v12', 'v12_topk']:
            attn = routing_info['attention']
            mem = routing_info.get('memory', {})

            if comp == 'compress':
                # compress_pref [B, S, n_compress]
                return attn.get('compress_pref')
            elif comp in ['Q', 'K', 'V']:
                # expand_pref_Q/K/V [B, S, n_expand]
                return attn.get(f'expand_pref_{comp}')
            elif comp == 'M':
                # Memory's token_neuron_pref (if saved in routing_info)
                # Note: NeuronMemory needs to save 'compress_pref' for this to work
                return mem.get('compress_pref')

        return None


# ============================================================
# Main Analyzer
# ============================================================

class DAWNAnalyzer:
    """Unified DAWN analyzer for all versions"""

    def __init__(self, model, tokenizer, device, model_version: str = 'auto'):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device

        # Detect version
        if model_version == 'auto':
            if hasattr(self.model, 'shared_neurons'):
                shared = self.model.shared_neurons

                # Check for v13 (has context in SSM)
                if hasattr(self.model, 'global_ssm') and hasattr(self.model.global_ssm, 'context_proj'):
                    model_version = '13.0'
                # Check for v12.7 (has A_log in SSM, no context)
                elif hasattr(self.model, 'global_ssm') and hasattr(self.model.global_ssm, 'A_log'):
                    model_version = '12.7'
                elif hasattr(shared, 'expand_neurons_pool'):
                    model_version = '12.3'
                elif hasattr(shared, 'expand_neurons'):
                    model_version = '10.0'
                else:
                    model_version = '12.0'
            else:
                model_version = '10.0'

        self.version = model_version
        self.parser = RoutingInfoParser(model_version)

        # Model config
        self.n_layers = self.model.n_layers
        self.d_model = self.model.d_model
        self.rank = self.model.rank
        self.n_heads = self.model.n_heads

        # Neuron counts
        if hasattr(self.model, 'shared_neurons'):
            shared = self.model.shared_neurons
            self.n_compress = shared.compress_neurons.shape[0]
            if hasattr(shared, 'expand_neurons_pool'):
                self.n_expand = shared.expand_neurons_pool.shape[0]
            elif hasattr(shared, 'expand_neurons'):
                self.n_expand = shared.expand_neurons.shape[0]
            else:
                self.n_expand = self.n_compress

            if hasattr(shared, 'knowledge_K'):
                self.n_knowledge = shared.knowledge_K.shape[0]
            else:
                self.n_knowledge = 0
        else:
            self.n_compress = getattr(self.model, 'n_compress', 48)
            self.n_expand = getattr(self.model, 'n_expand', 12)
            self.n_knowledge = getattr(self.model, 'n_knowledge', 80)

        self.knowledge_k = getattr(self.model, 'knowledge_k', 10)

        print(f"\n{'='*60}")
        print(f"DAWN Analyzer (v{self.version})")
        print(f"{'='*60}")
        print(f"d_model: {self.d_model}, rank: {self.rank}")
        print(f"n_compress: {self.n_compress}, n_expand: {self.n_expand}")
        print(f"n_knowledge: {self.n_knowledge}, n_layers: {self.n_layers}")

    # ============================================================
    # 1. COMPRESSION QUALITY
    # ============================================================

    @torch.no_grad()
    def analyze_compression_quality(self, dataloader, max_batches: int = 50) -> Dict:
        """Analyze compression quality (discriminability, variance retention)"""
        print(f"\n{'='*60}")
        print("1. COMPRESSION QUALITY ANALYSIS")
        print(f"{'='*60}")

        self.model.eval()

        discriminability = {comp: [] for comp in ['Q', 'K', 'V', 'M']}
        layer_disc = {l: {comp: [] for comp in ['Q', 'K', 'V', 'M']} for l in range(self.n_layers)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Compression Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                for comp in ['Q', 'K', 'V', 'M']:
                    weights, indices = self.parser.get_compress_weights(routing_info, comp)

                    if weights is None:
                        continue

                    # Handle different shapes:
                    # v10.x: [B, S, N] (token-level) -> flatten to [B*S, N]
                    # v12.3: [B, N] (batch-level) -> keep as [B, N]
                    if len(weights.shape) == 3:
                        weights_flat = weights.reshape(-1, weights.shape[-1])  # [B*S, N]
                    else:
                        weights_flat = weights  # [B, N]

                    weights_norm = F.normalize(weights_flat, dim=-1)

                    # Sample for efficiency
                    n_samples = min(256, weights_flat.shape[0])
                    if n_samples < 2:
                        continue  # Need at least 2 samples for pairwise comparison

                    idx = torch.randperm(weights_flat.shape[0])[:n_samples]
                    sampled = weights_norm[idx]

                    # Pairwise similarity
                    sim_matrix = sampled @ sampled.T
                    mask = ~torch.eye(n_samples, dtype=torch.bool, device=self.device)
                    avg_sim = sim_matrix[mask].mean().item()
                    disc = 1.0 - avg_sim

                    discriminability[comp].append(disc)
                    layer_disc[layer_idx][comp].append(disc)

        results = {'discriminability': {}, 'layer_breakdown': {}, 'diagnosis': {}}

        print(f"\n--- Discriminability (higher = better, max 1.0) ---")
        print(f"{'Component':<12} {'Mean':<12} {'Std':<12} {'Status'}")
        print("-" * 48)

        for comp in ['Q', 'K', 'V', 'M']:
            if discriminability[comp]:
                mean_d = np.mean(discriminability[comp])
                std_d = np.std(discriminability[comp])

                if mean_d < 0.3:
                    status = "LOW"
                elif mean_d < 0.5:
                    status = "MODERATE"
                else:
                    status = "GOOD"

                results['discriminability'][comp] = {'mean': mean_d, 'std': std_d}
                print(f"{comp:<12} {mean_d:<12.4f} {std_d:<12.4f} {status}")

        # Layer breakdown
        print(f"\n--- Layer-wise Discriminability ---")
        for layer_idx in range(self.n_layers):
            results['layer_breakdown'][f'L{layer_idx}'] = {}
            for comp in ['Q', 'K', 'V', 'M']:
                if layer_disc[layer_idx][comp]:
                    disc = np.mean(layer_disc[layer_idx][comp])
                    results['layer_breakdown'][f'L{layer_idx}'][comp] = disc

        return results

    # ============================================================
    # 2. ROUTING PATTERNS
    # ============================================================

    @torch.no_grad()
    def analyze_routing_patterns(self, dataloader, max_batches: int = 50) -> Dict:
        """Analyze neuron utilization patterns"""
        print(f"\n{'='*60}")
        print("2. ROUTING PATTERNS")
        print(f"{'='*60}")

        self.model.eval()

        compress_usage = {comp: torch.zeros(self.n_layers, self.n_compress, device=self.device)
                         for comp in ['Q', 'K', 'V', 'M']}
        expand_usage = {comp: torch.zeros(self.n_layers, self.n_expand, device=self.device)
                       for comp in ['Q', 'K', 'V']}
        total_tokens = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Routing Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape
            total_tokens += B * S

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                for comp in ['Q', 'K', 'V', 'M']:
                    weights, _ = self.parser.get_compress_weights(routing_info, comp)
                    if weights is not None:
                        # Handle different shapes: [B, S, N] or [B, N]
                        if len(weights.shape) == 3:
                            compress_usage[comp][layer_idx] += weights.sum(dim=(0, 1))
                        elif len(weights.shape) == 2:
                            compress_usage[comp][layer_idx] += weights.sum(dim=0)

                for comp in ['Q', 'K', 'V']:
                    weights, _ = self.parser.get_expand_weights(routing_info, comp)
                    if weights is not None:
                        # Handle different shapes: [B, S, N] or [B, N]
                        if len(weights.shape) == 3:
                            expand_usage[comp][layer_idx] += weights.sum(dim=(0, 1))
                        elif len(weights.shape) == 2:
                            expand_usage[comp][layer_idx] += weights.sum(dim=0)

        # Normalize
        for comp in compress_usage:
            compress_usage[comp] /= total_tokens
        for comp in expand_usage:
            expand_usage[comp] /= total_tokens

        def compute_stats(weights):
            probs = weights / (weights.sum() + 1e-10)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            eff_rank = torch.exp(entropy).item()

            sorted_p, _ = torch.sort(probs)
            n = len(probs)
            index = torch.arange(1, n + 1, dtype=torch.float32, device=probs.device)
            gini = ((2 * index - n - 1) * sorted_p).sum() / (n * sorted_p.sum() + 1e-10)

            threshold = 1.0 / len(probs) * 0.01
            dead = (probs < threshold).sum().item()

            return {
                'eff_rank': eff_rank,
                'max_neurons': len(probs),
                'utilization': eff_rank / len(probs),
                'gini': gini.item(),
                'dead_count': int(dead)
            }

        results = {'compress': {}, 'expand': {}, 'summary': {}}

        print(f"\n--- Compress Neuron Usage ---")
        print(f"{'Comp':<6} {'Layer':<8} {'Eff.Rank':<12} {'Util%':<10} {'Dead':<8}")
        print("-" * 44)

        all_utils = []
        for comp in ['Q', 'K', 'V', 'M']:
            results['compress'][comp] = {}
            for layer_idx in range(self.n_layers):
                stats = compute_stats(compress_usage[comp][layer_idx])
                results['compress'][comp][f'L{layer_idx}'] = stats
                all_utils.append(stats['utilization'])

                if comp == 'Q':
                    print(f"{comp:<6} L{layer_idx:<7} {stats['eff_rank']:<12.1f} {stats['utilization']*100:<10.1f} {stats['dead_count']:<8}")

        print(f"\n--- Expand Neuron Usage ---")
        for comp in ['Q', 'K', 'V']:
            results['expand'][comp] = {}
            for layer_idx in range(self.n_layers):
                stats = compute_stats(expand_usage[comp][layer_idx])
                results['expand'][comp][f'L{layer_idx}'] = stats

                if comp == 'Q':
                    print(f"L{layer_idx}: eff_rank={stats['eff_rank']:.1f}/{self.n_expand}, util={stats['utilization']:.1%}")

        results['summary'] = {
            'avg_utilization': np.mean(all_utils),
            'min_utilization': np.min(all_utils),
        }

        return results

    # ============================================================
    # 3. KNOWLEDGE HEALTH
    # ============================================================

    @torch.no_grad()
    def analyze_knowledge_health(self, dataloader, max_batches: int = 50) -> Dict:
        """Analyze knowledge neuron health"""
        print(f"\n{'='*60}")
        print("3. KNOWLEDGE NEURON HEALTH")
        print(f"{'='*60}")

        if self.n_knowledge == 0:
            print("No knowledge neurons found")
            return {}

        self.model.eval()

        usage = torch.zeros(self.n_layers, self.n_knowledge, device=self.device)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Knowledge Health", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                k_indices, k_weights = self.parser.get_knowledge_info(routing_info)

                if k_indices is not None:
                    idx_flat = k_indices.reshape(-1)
                    usage[layer_idx] += torch.bincount(idx_flat, minlength=self.n_knowledge).float()

        results = {'layers': {}, 'global': {}}

        print(f"\n--- Knowledge Usage by Layer ---")
        print(f"{'Layer':<8} {'Used':<12} {'Dead':<10} {'Top10%':<12}")
        print("-" * 42)

        global_usage = usage.sum(dim=0)

        for layer_idx in range(self.n_layers):
            layer_usage = usage[layer_idx]
            layer_probs = layer_usage / (layer_usage.sum() + 1e-10)

            used = (layer_usage > 0).sum().item()
            dead = self.n_knowledge - used

            top_k = max(1, self.n_knowledge // 10)
            top_usage = torch.topk(layer_probs, top_k)[0].sum().item()

            results['layers'][f'L{layer_idx}'] = {
                'used': used,
                'dead': dead,
                'top10_concentration': top_usage
            }

            print(f"L{layer_idx:<7} {used:<12} {dead:<10} {top_usage:<12.1%}")

        global_dead = (global_usage == 0).sum().item()
        results['global'] = {
            'dead_neurons': global_dead,
            'dead_pct': global_dead / self.n_knowledge,
        }

        print(f"\nGlobal dead neurons: {global_dead}/{self.n_knowledge} ({global_dead/self.n_knowledge:.1%})")

        return results

    # ============================================================
    # 4. LAYER ANALYSIS
    # ============================================================

    @torch.no_grad()
    def analyze_layer_roles(self, dataloader, max_batches: int = 50) -> Dict:
        """Analyze layer-wise neuron specialization"""
        print(f"\n{'='*60}")
        print("4. LAYER ROLE ANALYSIS")
        print(f"{'='*60}")

        self.model.eval()

        # Compress usage (Q=K=V in v12.3, separate in v10)
        layer_comp_usage = {l: {comp: torch.zeros(self.n_compress, device=self.device)
                               for comp in ['Q', 'K', 'V', 'M']}
                          for l in range(self.n_layers)}

        # Expand usage (Q/K/V different in v12.3)
        layer_expand_usage = {l: {comp: torch.zeros(self.n_expand, device=self.device)
                                  for comp in ['Q', 'K', 'V']}
                             for l in range(self.n_layers)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Layer Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                # Compress routing
                for comp in ['Q', 'K', 'V', 'M']:
                    weights, _ = self.parser.get_compress_weights(routing_info, comp)
                    if weights is not None:
                        if len(weights.shape) == 3:
                            layer_comp_usage[layer_idx][comp] += weights.sum(dim=(0, 1))
                        else:
                            layer_comp_usage[layer_idx][comp] += weights.sum(dim=0)

                # Expand routing (v12.3 has different expand_weights for Q/K/V)
                for comp in ['Q', 'K', 'V']:
                    weights, _ = self.parser.get_expand_weights(routing_info, comp)
                    if weights is not None:
                        if len(weights.shape) == 3:
                            layer_expand_usage[layer_idx][comp] += weights.sum(dim=(0, 1))
                        else:
                            layer_expand_usage[layer_idx][comp] += weights.sum(dim=0)

        results = {'layer_correlation': {}, 'expand_correlation': {}}

        # Check if v12.3 (Q=K=V for compress)
        is_v12 = self.version.startswith('12')

        print(f"\n--- Compress Routing: Q/K/V/M Correlation by Layer ---")
        if is_v12:
            print("(Note: v12.3 uses shared compress_weights for Q/K/V, so Q-K, Q-V = 1.0)")

        for layer_idx in range(self.n_layers):
            q_usage = layer_comp_usage[layer_idx]['Q']
            k_usage = layer_comp_usage[layer_idx]['K']
            v_usage = layer_comp_usage[layer_idx]['V']
            m_usage = layer_comp_usage[layer_idx]['M']

            q_norm = F.normalize(q_usage.unsqueeze(0), dim=-1)
            k_norm = F.normalize(k_usage.unsqueeze(0), dim=-1)
            v_norm = F.normalize(v_usage.unsqueeze(0), dim=-1)
            m_norm = F.normalize(m_usage.unsqueeze(0), dim=-1)

            qk = F.cosine_similarity(q_norm, k_norm).item()
            qv = F.cosine_similarity(q_norm, v_norm).item()
            qm = F.cosine_similarity(q_norm, m_norm).item()

            results['layer_correlation'][f'L{layer_idx}'] = {
                'Q-K': qk, 'Q-V': qv, 'Q-M': qm
            }

            if layer_idx == 0:
                print(f"  L{layer_idx}: Q-K={qk:.3f}, Q-V={qv:.3f}, Q-M={qm:.3f}")

        # Expand correlation (more meaningful for v12.3)
        if is_v12:
            print(f"\n--- Expand Routing: Q/K/V Differentiation by Layer ---")
            print("(v12.3 uses different expand_weights for Q/K/V)")

            for layer_idx in range(self.n_layers):
                q_exp = layer_expand_usage[layer_idx]['Q']
                k_exp = layer_expand_usage[layer_idx]['K']
                v_exp = layer_expand_usage[layer_idx]['V']

                q_norm = F.normalize(q_exp.unsqueeze(0), dim=-1)
                k_norm = F.normalize(k_exp.unsqueeze(0), dim=-1)
                v_norm = F.normalize(v_exp.unsqueeze(0), dim=-1)

                qk = F.cosine_similarity(q_norm, k_norm).item()
                qv = F.cosine_similarity(q_norm, v_norm).item()
                kv = F.cosine_similarity(k_norm, v_norm).item()

                results['expand_correlation'][f'L{layer_idx}'] = {
                    'Q-K': qk, 'Q-V': qv, 'K-V': kv
                }

                print(f"  L{layer_idx}: Q-K={qk:.3f}, Q-V={qv:.3f}, K-V={kv:.3f}")

        return results

    # ============================================================
    # 5. TOKEN SPECIALIZATION
    # ============================================================

    @torch.no_grad()
    def analyze_token_specialization(self, dataloader, max_batches: int = 50) -> Dict:
        """Analyze token/POS neuron specialization"""
        print(f"\n{'='*60}")
        print("5. TOKEN/POS SPECIALIZATION")
        print(f"{'='*60}")

        self.model.eval()

        pos_neuron_weights = defaultdict(lambda: torch.zeros(self.n_compress, device=self.device))
        pos_counts = defaultdict(float)
        use_token_pref = False  # Whether we're using token-level compress_pref

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Specialization", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # Try to get token-level compress_pref first (v12.3)
            token_pref = self.parser.get_token_level_pref(routing_infos[0], 'compress')

            if token_pref is not None and len(token_pref.shape) == 3:
                # v12.3: Use compress_pref [B, S, n_compress] for true token-level analysis
                use_token_pref = True
                _, seq_len, n_neurons = token_pref.shape

                for b in range(B):
                    for s in range(min(S, seq_len)):
                        tid = input_ids[b, s].item()
                        token = self.tokenizer.decode([tid]).strip()
                        pos = simple_pos_tag(token)

                        weights = token_pref[b, s]
                        if weights.shape[0] == self.n_compress:
                            pos_neuron_weights[pos] += weights
                            pos_counts[pos] += 1
            else:
                # v10.x or fallback: Use compress_weights
                q_weights, _ = self.parser.get_compress_weights(routing_infos[0], 'Q')

                if q_weights is None:
                    continue

                if len(q_weights.shape) == 3:
                    _, seq_len, n_neurons = q_weights.shape
                    for b in range(B):
                        for s in range(min(S, seq_len)):
                            tid = input_ids[b, s].item()
                            token = self.tokenizer.decode([tid]).strip()
                            pos = simple_pos_tag(token)

                            weights = q_weights[b, s]
                            if weights.shape[0] == self.n_compress:
                                pos_neuron_weights[pos] += weights
                                pos_counts[pos] += 1
                elif len(q_weights.shape) == 2:
                    # Batch-level fallback (shouldn't happen if compress_pref exists)
                    for b in range(B):
                        batch_weights = q_weights[b]
                        for s in range(S):
                            tid = input_ids[b, s].item()
                            token = self.tokenizer.decode([tid]).strip()
                            pos = simple_pos_tag(token)

                            if batch_weights.shape[0] == self.n_compress:
                                pos_neuron_weights[pos] += batch_weights
                                pos_counts[pos] += 1

        results = {'pos': {}, 'use_token_pref': use_token_pref}

        print(f"\n--- POS -> Preferred Neurons (L0 compress_pref) ---")
        if use_token_pref:
            print("(Using token-level compress_pref [B,S,N] for true per-token analysis)")

        # Calculate POS discriminability (cosine sim between POS weight vectors)
        pos_vectors = {}
        for pos in sorted(pos_counts.keys()):
            if pos_counts[pos] < 100 or pos == 'OTHER':
                continue
            avg_weights = pos_neuron_weights[pos] / pos_counts[pos]
            pos_vectors[pos] = F.normalize(avg_weights.unsqueeze(0), dim=-1)

            top_neurons = torch.topk(avg_weights, 5)
            results['pos'][pos] = {
                'count': int(pos_counts[pos]),
                'top_neurons': top_neurons.indices.tolist(),
            }

            neurons_str = ', '.join([f'{n}' for n in top_neurons.indices.tolist()[:3]])
            print(f"  {pos:12s} (n={int(pos_counts[pos]):5d}): [{neurons_str}]")

        # Compute POS discriminability (average pairwise distance)
        if len(pos_vectors) >= 2:
            pos_list = list(pos_vectors.keys())
            similarities = []
            for i in range(len(pos_list)):
                for j in range(i+1, len(pos_list)):
                    sim = F.cosine_similarity(pos_vectors[pos_list[i]], pos_vectors[pos_list[j]]).item()
                    similarities.append(sim)

            avg_sim = np.mean(similarities)
            pos_disc = 1.0 - avg_sim
            results['pos_discriminability'] = pos_disc

            print(f"\n  POS Discriminability: {pos_disc:.4f}")
            if pos_disc > 0.3:
                print(f"  (Good differentiation! Neurons specialize for different POS)")
            elif pos_disc > 0.1:
                print(f"  (Moderate differentiation)")
            else:
                print(f"  (Low differentiation - neurons may be general-purpose)")

        return results

    # ============================================================
    # 6. ATTENTION VS MEMORY
    # ============================================================

    @torch.no_grad()
    def analyze_attn_mem_balance(self, dataloader, max_batches: int = 30) -> Dict:
        """Analyze attention vs memory contribution"""
        print(f"\n{'='*60}")
        print("6. ATTENTION VS MEMORY BALANCE")
        print(f"{'='*60}")

        # v12.7/v13 require routing weights for forward pass
        # Use full model forward to get layer outputs
        if self.version in ['12.7', '12.8', '13.0'] or self.version.startswith('13'):
            return self._analyze_attn_mem_balance_v12_topk(dataloader, max_batches)

        self.model.eval()

        layer_stats = {l: {'attn_norm': [], 'mem_norm': []} for l in range(self.n_layers)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Attn-Mem Balance", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            pos = torch.arange(S, device=self.device).unsqueeze(0).expand(B, S)
            x = self.model.token_emb(input_ids) + self.model.pos_emb(pos)

            mask = torch.triu(torch.ones(S, S, device=self.device), diagonal=1).bool()
            mask = ~mask.unsqueeze(0).unsqueeze(0)

            for layer_idx, layer in enumerate(self.model.layers):
                attn_out, _ = layer.attn(layer.norm1(x), mask)
                x = x + attn_out

                mem_out, _ = layer.memory(layer.norm2(x))
                x = x + mem_out

                layer_stats[layer_idx]['attn_norm'].append(attn_out.norm(dim=-1).mean().item())
                layer_stats[layer_idx]['mem_norm'].append(mem_out.norm(dim=-1).mean().item())

        results = {'layers': {}}

        print(f"\n--- Layer-wise Contribution ---")
        print(f"{'Layer':<8} {'Attn Norm':<12} {'Mem Norm':<12} {'Attn%':<10}")
        print("-" * 42)

        for layer_idx in range(self.n_layers):
            avg_attn = np.mean(layer_stats[layer_idx]['attn_norm'])
            avg_mem = np.mean(layer_stats[layer_idx]['mem_norm'])
            contrib = avg_attn / (avg_attn + avg_mem + 1e-10)

            results['layers'][f'L{layer_idx}'] = {
                'attn_norm': avg_attn,
                'mem_norm': avg_mem,
                'attn_contribution': contrib
            }

            print(f"L{layer_idx:<7} {avg_attn:<12.3f} {avg_mem:<12.3f} {contrib:<10.1%}")

        return results

    @torch.no_grad()
    def _analyze_attn_mem_balance_v12_topk(self, dataloader, max_batches: int = 30) -> Dict:
        """Analyze attention vs memory contribution for v12.7/v13 (requires routing weights)"""
        self.model.eval()

        layer_stats = {l: {'attn_norm': [], 'mem_norm': []} for l in range(self.n_layers)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Attn-Mem Balance", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Full forward pass with hooks to capture intermediate outputs
            pos = torch.arange(S, device=self.device).unsqueeze(0).expand(B, S)
            x = self.model.token_emb(input_ids) + self.model.pos_emb(pos)

            mask = torch.triu(torch.ones(S, S, device=self.device), diagonal=1).bool()
            mask = ~mask.unsqueeze(0).unsqueeze(0)

            # Get importance and routing weights from SSM
            if hasattr(self.model, 'global_ssm'):
                ssm_result = self.model.global_ssm(x)
                if isinstance(ssm_result, tuple):
                    importance, context = ssm_result
                else:
                    importance = ssm_result
                    context = None
            else:
                importance = torch.ones(B, S, device=self.device) / S
                context = None

            # Get routing weights
            compress_weights, expand_weights_Q, expand_weights_K, expand_weights_V, _ = \
                self.model.global_routers.get_attention_weights(x, importance)
            memory_weights, _ = self.model.global_routers.get_memory_weights(x, importance)

            # Process through layers
            for layer_idx, layer in enumerate(self.model.layers):
                x_norm = layer.norm1(x)

                # Add context if available (v13)
                if context is not None and hasattr(layer, 'attn'):
                    x_for_attn = x_norm + context
                else:
                    x_for_attn = x_norm

                attn_out, _ = layer.attn(x_for_attn, compress_weights, expand_weights_Q,
                                         expand_weights_K, expand_weights_V, mask)
                x = x + attn_out

                mem_out, _ = layer.memory(layer.norm2(x), memory_weights)
                x = x + mem_out

                layer_stats[layer_idx]['attn_norm'].append(attn_out.norm(dim=-1).mean().item())
                layer_stats[layer_idx]['mem_norm'].append(mem_out.norm(dim=-1).mean().item())

        results = {'layers': {}}

        print(f"\n--- Layer-wise Contribution ---")
        print(f"{'Layer':<8} {'Attn Norm':<12} {'Mem Norm':<12} {'Attn%':<10}")
        print("-" * 42)

        for layer_idx in range(self.n_layers):
            avg_attn = np.mean(layer_stats[layer_idx]['attn_norm'])
            avg_mem = np.mean(layer_stats[layer_idx]['mem_norm'])
            contrib = avg_attn / (avg_attn + avg_mem + 1e-10)

            results['layers'][f'L{layer_idx}'] = {
                'attn_norm': avg_attn,
                'mem_norm': avg_mem,
                'attn_contribution': contrib
            }

            print(f"L{layer_idx:<7} {avg_attn:<12.3f} {avg_mem:<12.3f} {contrib:<10.1%}")

        return results

    # ============================================================
    # 7. ROUTING DIVERSITY ANALYSIS (Collapse 진단)
    # ============================================================

    @torch.no_grad()
    def analyze_routing_diversity(self, dataloader, max_batches: int = 50) -> Dict:
        """
        Analyze expand/compress router diversity to diagnose collapse.

        Checks:
        1. compress_pref vs expand_pref entropy (router 자체 문제?)
        2. Token-level variance (토큰마다 다른가?)
        3. Before/after importance weighting (SSM이 diversity 죽이나?)
        """
        print(f"\n{'='*60}")
        print("7. ROUTING DIVERSITY ANALYSIS (Collapse 진단)")
        print(f"{'='*60}")

        self.model.eval()

        # Accumulators
        stats = {
            'compress_pref': {'entropy': [], 'token_var': [], 'neuron_var': []},
            'expand_pref_Q': {'entropy': [], 'token_var': [], 'neuron_var': []},
            'expand_pref_K': {'entropy': [], 'token_var': [], 'neuron_var': []},
            'expand_pref_V': {'entropy': [], 'token_var': [], 'neuron_var': []},
            'compress_weights': {'entropy': []},
            'expand_weights_Q': {'entropy': []},
            'expand_weights_K': {'entropy': []},
            'expand_weights_V': {'entropy': []},
            'importance': {'entropy': [], 'concentration': []},
        }

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Diversity Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # Use layer 0 for analysis
            routing_info = routing_infos[0]
            attn = routing_info.get('attention', routing_info)

            # Skip if not v12.3 format
            if 'compress_pref' not in attn:
                print("\n[SKIP] compress_pref not found in routing_info")
                print("This analysis requires v12.3 format with token-level preferences.")
                return {}

            # Get token-level preferences
            compress_pref = attn['compress_pref']  # [B, S, n_compress]
            expand_pref_Q = attn.get('expand_pref_Q')  # [B, S, n_expand]
            expand_pref_K = attn.get('expand_pref_K')
            expand_pref_V = attn.get('expand_pref_V')

            # Get batch-level weights
            compress_weights = attn['compress_weights']  # [B, n_compress]
            expand_weights_Q = attn.get('expand_weights_Q')  # [B, n_expand]
            expand_weights_K = attn.get('expand_weights_K')
            expand_weights_V = attn.get('expand_weights_V')

            # Get importance
            importance = attn.get('importance')  # [B, S]

            def compute_entropy(p, dim=-1):
                """Compute entropy along dimension"""
                return -(p * torch.log(p + 1e-10)).sum(dim).mean().item()

            def compute_token_var(p):
                """Variance across tokens (dim=1)"""
                return p.var(dim=1).mean().item()

            def compute_neuron_var(p):
                """Variance across neurons (dim=-1)"""
                return p.var(dim=-1).mean().item()

            # Analyze compress_pref [B, S, n_compress]
            stats['compress_pref']['entropy'].append(compute_entropy(compress_pref))
            stats['compress_pref']['token_var'].append(compute_token_var(compress_pref))
            stats['compress_pref']['neuron_var'].append(compute_neuron_var(compress_pref))

            # Analyze expand_pref_Q/K/V [B, S, n_expand]
            for name, pref in [('expand_pref_Q', expand_pref_Q),
                               ('expand_pref_K', expand_pref_K),
                               ('expand_pref_V', expand_pref_V)]:
                if pref is not None:
                    stats[name]['entropy'].append(compute_entropy(pref))
                    stats[name]['token_var'].append(compute_token_var(pref))
                    stats[name]['neuron_var'].append(compute_neuron_var(pref))

            # Analyze batch-level weights [B, n]
            stats['compress_weights']['entropy'].append(compute_entropy(compress_weights))
            for name, weights in [('expand_weights_Q', expand_weights_Q),
                                  ('expand_weights_K', expand_weights_K),
                                  ('expand_weights_V', expand_weights_V)]:
                if weights is not None:
                    stats[name]['entropy'].append(compute_entropy(weights))

            # Analyze importance [B, S]
            if importance is not None:
                stats['importance']['entropy'].append(compute_entropy(importance))
                # Concentration: how much does top token dominate?
                top_importance = importance.max(dim=1).values / (importance.sum(dim=1) + 1e-10)
                stats['importance']['concentration'].append(top_importance.mean().item())

        # Compute averages and build results
        results = {}

        # Max entropies for reference
        max_entropy_compress = math.log(self.n_compress)
        max_entropy_expand = math.log(self.n_expand)

        print(f"\n--- Token-Level Preferences (Before SSM) ---")
        print(f"{'Metric':<20} {'Entropy':<12} {'Max':<8} {'Ratio':<8} {'Token Var':<12} {'Neuron Var':<12}")
        print("-" * 72)

        for name in ['compress_pref', 'expand_pref_Q', 'expand_pref_K', 'expand_pref_V']:
            if stats[name]['entropy']:
                entropy = np.mean(stats[name]['entropy'])
                token_var = np.mean(stats[name]['token_var'])
                neuron_var = np.mean(stats[name]['neuron_var'])
                max_ent = max_entropy_compress if 'compress' in name else max_entropy_expand
                ratio = entropy / max_ent

                results[name] = {
                    'entropy': entropy,
                    'max_entropy': max_ent,
                    'entropy_ratio': ratio,
                    'token_variance': token_var,
                    'neuron_variance': neuron_var,
                }

                print(f"{name:<20} {entropy:<12.4f} {max_ent:<8.2f} {ratio:<8.2%} {token_var:<12.6f} {neuron_var:<12.6f}")

        print(f"\n--- Batch-Level Weights (After SSM) ---")
        print(f"{'Metric':<20} {'Entropy':<12} {'Max':<8} {'Ratio':<8}")
        print("-" * 40)

        for name in ['compress_weights', 'expand_weights_Q', 'expand_weights_K', 'expand_weights_V']:
            if stats[name]['entropy']:
                entropy = np.mean(stats[name]['entropy'])
                max_ent = max_entropy_compress if 'compress' in name else max_entropy_expand
                ratio = entropy / max_ent

                results[name] = {
                    'entropy': entropy,
                    'max_entropy': max_ent,
                    'entropy_ratio': ratio,
                }

                print(f"{name:<20} {entropy:<12.4f} {max_ent:<8.2f} {ratio:<8.2%}")

        # SSM importance analysis
        if stats['importance']['entropy']:
            imp_entropy = np.mean(stats['importance']['entropy'])
            imp_concentration = np.mean(stats['importance']['concentration'])

            print(f"\n--- SSM Importance ---")
            print(f"Importance entropy: {imp_entropy:.4f}")
            print(f"Top token concentration: {imp_concentration:.2%}")

            results['importance'] = {
                'entropy': imp_entropy,
                'concentration': imp_concentration,
            }

        # Diagnosis
        print(f"\n--- Diagnosis ---")

        # Compare compress vs expand
        if 'compress_pref' in results and 'expand_pref_Q' in results:
            c_ratio = results['compress_pref']['entropy_ratio']
            e_ratio = results['expand_pref_Q']['entropy_ratio']

            print(f"compress_pref entropy ratio: {c_ratio:.2%}")
            print(f"expand_pref_Q entropy ratio: {e_ratio:.2%}")

            if e_ratio < 0.3:
                print("  → expand_pref collapsed! Router 자체 문제")
            elif c_ratio > 0.5 and e_ratio > 0.5:
                print("  → Both diverse! 좋은 상태")

        # Compare before/after SSM
        if 'expand_pref_Q' in results and 'expand_weights_Q' in results:
            before = results['expand_pref_Q']['entropy_ratio']
            after = results['expand_weights_Q']['entropy_ratio']
            drop = before - after

            print(f"\nExpand entropy drop (pref→weights): {before:.2%} → {after:.2%} (Δ={drop:.2%})")

            if drop > 0.2:
                print("  → SSM importance가 diversity 죽임!")
            elif drop < 0.05:
                print("  → SSM 영향 적음, pref 자체가 문제")

        # Token variance check
        if 'expand_pref_Q' in results:
            token_var = results['expand_pref_Q']['token_variance']
            if token_var < 0.001:
                print(f"\nToken variance 매우 낮음 ({token_var:.6f})")
                print("  → 모든 토큰이 같은 뉴런 선호. Router가 토큰 구분 못함")
            elif token_var > 0.01:
                print(f"\nToken variance 양호 ({token_var:.6f})")
                print("  → 토큰별 분화 있음")

        return results

    # ============================================================
    # 8. IMPORTANCE DISTRIBUTION ANALYSIS
    # ============================================================

    @torch.no_grad()
    def analyze_importance_distribution(self, dataloader, max_batches: int = 50) -> Dict:
        """
        Analyze SSM importance distribution.

        Computes:
        1. Importance entropy (higher = more uniform)
        2. Importance sparsity (Gini coefficient)
        3. Top 10% concentration (how much weight on top tokens)
        4. Per-layer importance statistics
        """
        print(f"\n{'='*60}")
        print("8. IMPORTANCE DISTRIBUTION ANALYSIS")
        print(f"{'='*60}")

        self.model.eval()

        # Check if model has SSM
        if not hasattr(self.model, 'global_ssm'):
            print("No SSM found in model, skipping importance analysis")
            return {}

        stats = {
            'entropy': [],
            'gini': [],
            'top_10_concentration': [],
            'top_20_concentration': [],
            'max_importance': [],
            'min_importance': [],
            'std_importance': [],
            # Raw importance (before softmax) stats
            'raw_std': [],
            'raw_min': [],
            'raw_max': [],
            'raw_range': [],
        }

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Importance Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Get embeddings
            pos = torch.arange(S, device=self.device).unsqueeze(0).expand(B, S)
            x = self.model.token_emb(input_ids) + self.model.pos_emb(pos)

            # Get importance from SSM
            ssm_result = self.model.global_ssm(x)
            if len(ssm_result) == 3:
                importance, _, raw_importance = ssm_result
            elif len(ssm_result) == 2:
                importance, _ = ssm_result
                raw_importance = None
            else:
                importance = ssm_result
                raw_importance = None

            # importance: [B, S] - normalized probability over tokens

            # 1. Entropy (higher = more uniform)
            entropy = -(importance * torch.log(importance + 1e-10)).sum(dim=-1)  # [B]
            max_entropy = math.log(S)
            normalized_entropy = entropy / max_entropy  # [B]
            stats['entropy'].append(normalized_entropy.mean().item())

            # 2. Gini coefficient (0 = perfect equality, 1 = max inequality)
            sorted_imp, _ = torch.sort(importance, dim=-1)  # [B, S]
            n = S
            index = torch.arange(1, n + 1, dtype=torch.float32, device=self.device)
            gini = ((2 * index - n - 1) * sorted_imp).sum(dim=-1) / (n * sorted_imp.sum(dim=-1) + 1e-10)  # [B]
            stats['gini'].append(gini.mean().item())

            # 3. Top-k concentration
            k_10 = max(1, S // 10)  # top 10%
            k_20 = max(1, S // 5)   # top 20%

            top_10_vals = torch.topk(importance, k_10, dim=-1).values  # [B, k_10]
            top_20_vals = torch.topk(importance, k_20, dim=-1).values  # [B, k_20]

            top_10_conc = top_10_vals.sum(dim=-1) / (importance.sum(dim=-1) + 1e-10)  # [B]
            top_20_conc = top_20_vals.sum(dim=-1) / (importance.sum(dim=-1) + 1e-10)  # [B]

            stats['top_10_concentration'].append(top_10_conc.mean().item())
            stats['top_20_concentration'].append(top_20_conc.mean().item())

            # 4. Basic statistics
            stats['max_importance'].append(importance.max(dim=-1).values.mean().item())
            stats['min_importance'].append(importance.min(dim=-1).values.mean().item())
            stats['std_importance'].append(importance.std(dim=-1).mean().item())

            # 5. Raw importance statistics (before softmax)
            if raw_importance is not None:
                stats['raw_std'].append(raw_importance.std(dim=-1).mean().item())
                stats['raw_min'].append(raw_importance.min(dim=-1).values.mean().item())
                stats['raw_max'].append(raw_importance.max(dim=-1).values.mean().item())
                raw_range = raw_importance.max(dim=-1).values - raw_importance.min(dim=-1).values
                stats['raw_range'].append(raw_range.mean().item())

        # Aggregate results
        results = {}
        for key in stats:
            if stats[key]:
                results[key] = {
                    'mean': np.mean(stats[key]),
                    'std': np.std(stats[key]),
                }

        # Print summary
        print(f"\n--- Importance Distribution Statistics (after softmax) ---")
        print(f"{'Metric':<25} {'Mean':<12} {'Std':<12}")
        print("-" * 49)

        for key in ['entropy', 'gini', 'top_10_concentration', 'top_20_concentration',
                    'max_importance', 'min_importance', 'std_importance']:
            if key in results:
                mean = results[key]['mean']
                std = results[key]['std']
                print(f"{key:<25} {mean:<12.4f} {std:<12.4f}")

        # Raw importance stats (before softmax)
        if 'raw_std' in results:
            print(f"\n--- Raw Importance Statistics (before softmax) ---")
            print(f"{'Metric':<25} {'Mean':<12} {'Std':<12}")
            print("-" * 49)
            for key in ['raw_std', 'raw_min', 'raw_max', 'raw_range']:
                if key in results:
                    mean = results[key]['mean']
                    std = results[key]['std']
                    print(f"{key:<25} {mean:<12.4f} {std:<12.4f}")

        # Diagnosis
        print(f"\n--- Diagnosis ---")

        if 'entropy' in results:
            entropy = results['entropy']['mean']
            if entropy < 0.3:
                print(f"Entropy={entropy:.2%} (LOW): Importance very concentrated")
                print("  → Few tokens dominate, SSM may be collapsing")
            elif entropy < 0.6:
                print(f"Entropy={entropy:.2%} (MODERATE): Some concentration")
            else:
                print(f"Entropy={entropy:.2%} (HIGH): Uniform distribution")

        if 'gini' in results:
            gini = results['gini']['mean']
            if gini > 0.7:
                print(f"Gini={gini:.3f} (HIGH): Very unequal distribution")
            elif gini > 0.4:
                print(f"Gini={gini:.3f} (MODERATE): Some inequality")
            else:
                print(f"Gini={gini:.3f} (LOW): Relatively uniform")

        if 'top_10_concentration' in results:
            top10 = results['top_10_concentration']['mean']
            print(f"\nTop 10% tokens hold {top10:.1%} of total importance")
            if top10 > 0.5:
                print("  → High concentration! SSM focusing on few tokens")
            elif top10 > 0.3:
                print("  → Moderate concentration")
            else:
                print("  → Low concentration, fairly distributed")

        # Raw importance diagnosis
        if 'raw_std' in results and 'raw_range' in results:
            raw_std = results['raw_std']['mean']
            raw_range = results['raw_range']['mean']
            print(f"\n--- Raw Score Analysis ---")
            print(f"Raw std: {raw_std:.4f}, Raw range: {raw_range:.4f}")
            if raw_std < 0.1:
                print("  ⚠ Raw scores very similar (std < 0.1)")
                print("  → Problem is in SSM output, not softmax temperature")
            elif raw_range < 1.0:
                print("  ⚠ Raw score range small (< 1.0)")
                print("  → Softmax will produce near-uniform distribution")
            else:
                print("  ✓ Raw scores have reasonable variance")
                print("  → SSM is differentiating tokens")

        return results

    # ============================================================
    # 9. VISUALIZATION
    # ============================================================

    def visualize(self, all_results: Dict, output_path: str):
        """Generate summary visualization"""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        print(f"\n{'='*60}")
        print("7. VISUALIZATION")
        print(f"{'='*60}")

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # 1. Discriminability
        ax = axes[0, 0]
        if 'compression' in all_results and 'discriminability' in all_results['compression']:
            disc = all_results['compression']['discriminability']
            comps = list(disc.keys())
            vals = [disc[c]['mean'] for c in comps]
            colors = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in vals]
            ax.bar(comps, vals, color=colors)
            ax.axhline(y=0.5, color='g', linestyle='--', alpha=0.5)
            ax.set_title('Compression Discriminability')
            ax.set_ylim(0, 1)

        # 2. Routing utilization
        ax = axes[0, 1]
        if 'routing' in all_results and 'compress' in all_results['routing']:
            routing = all_results['routing']['compress']
            utils = []
            labels = []
            for comp in ['Q', 'K', 'V', 'M']:
                if comp in routing:
                    for layer, stats in routing[comp].items():
                        utils.append(stats['utilization'] * 100)
                        labels.append(f'{comp}_{layer}')
            if utils:
                ax.bar(range(len(utils)), utils, color='steelblue')
                ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
                ax.set_title('Routing Utilization (%)')
                ax.set_xticks(range(0, len(labels), max(1, len(labels)//8)))

        # 3. Knowledge usage
        ax = axes[0, 2]
        if 'knowledge' in all_results and 'layers' in all_results['knowledge']:
            k = all_results['knowledge']
            layers = list(k['layers'].keys())
            dead = [k['layers'][l]['dead'] for l in layers]
            used = [self.n_knowledge - d for d in dead]

            ax.bar(range(len(layers)), used, label='Used', color='steelblue')
            ax.bar(range(len(layers)), dead, bottom=used, label='Dead', color='coral')
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layers)
            ax.set_title('Knowledge Neuron Usage')
            ax.legend()

        # 4. Layer correlation
        ax = axes[1, 0]
        if 'layer_roles' in all_results and 'layer_correlation' in all_results['layer_roles']:
            corr = all_results['layer_roles']['layer_correlation']
            layers = list(corr.keys())
            qk = [corr[l]['Q-K'] for l in layers]
            qv = [corr[l]['Q-V'] for l in layers]
            qm = [corr[l]['Q-M'] for l in layers]

            x = range(len(layers))
            ax.plot(x, qk, 'o-', label='Q-K')
            ax.plot(x, qv, 's-', label='Q-V')
            ax.plot(x, qm, '^-', label='Q-M')
            ax.set_xticks(x)
            ax.set_xticklabels(layers)
            ax.set_title('Component Correlation')
            ax.legend()

        # 5. Attn vs Memory
        ax = axes[1, 1]
        if 'attn_mem' in all_results and 'layers' in all_results['attn_mem']:
            am = all_results['attn_mem']['layers']
            layers = list(am.keys())
            attn = [am[l]['attn_contribution'] for l in layers]
            mem = [1 - a for a in attn]

            ax.bar(range(len(layers)), attn, label='Attention', color='steelblue')
            ax.bar(range(len(layers)), mem, bottom=attn, label='Memory', color='coral')
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layers)
            ax.set_title('Attn vs Memory')
            ax.legend()

        # 6. POS specialization
        ax = axes[1, 2]
        if 'specialization' in all_results and 'pos' in all_results['specialization']:
            pos_data = all_results['specialization']['pos']
            pos_list = sorted(pos_data.keys())[:10]
            counts = [pos_data[p]['count'] for p in pos_list]

            ax.barh(pos_list, counts, color='steelblue')
            ax.set_title('POS Token Counts')
            ax.set_xlabel('Count')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()

        print(f"Saved: {output_path}")

    # ============================================================
    # RUN ALL
    # ============================================================

    def run_all(self, dataloader, max_batches: int = 50, output_dir: str = './analysis') -> Dict:
        """Run all analyses"""
        print(f"\n{'='*60}")
        print(f"DAWN COMPREHENSIVE ANALYSIS (v{self.version})")
        print(f"{'='*60}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        all_results = {'version': self.version}

        # 1. Compression quality
        all_results['compression'] = self.analyze_compression_quality(dataloader, max_batches)

        # 2. Routing patterns
        all_results['routing'] = self.analyze_routing_patterns(dataloader, max_batches)

        # 3. Knowledge health
        all_results['knowledge'] = self.analyze_knowledge_health(dataloader, max_batches)

        # 4. Layer roles
        all_results['layer_roles'] = self.analyze_layer_roles(dataloader, max_batches)

        # 5. Token specialization
        all_results['specialization'] = self.analyze_token_specialization(dataloader, max_batches)

        # 6. Attn-Mem balance
        all_results['attn_mem'] = self.analyze_attn_mem_balance(dataloader, min(max_batches, 30))

        # 7. Routing diversity (collapse diagnosis)
        all_results['routing_diversity'] = self.analyze_routing_diversity(dataloader, max_batches)

        # 8. Importance distribution analysis
        all_results['importance_distribution'] = self.analyze_importance_distribution(dataloader, max_batches)

        # 9. Visualization
        os.makedirs(output_dir, exist_ok=True)
        viz_path = os.path.join(output_dir, 'analysis_summary.png')
        self.visualize(all_results, viz_path)

        return all_results


# ============================================================
# EPOCH TRACKING
# ============================================================

def analyze_epoch_progression(checkpoint_dir: str, val_data_path: str,
                               output_dir: str = './epoch_analysis',
                               epochs: List[int] = None,
                               max_batches: int = 30,
                               batch_size: int = 32) -> Dict:
    """
    Compare metrics across training epochs.

    Args:
        checkpoint_dir: Directory containing epoch checkpoints (epoch_1.pt, epoch_2.pt, ...)
        val_data_path: Path to validation data
        output_dir: Output directory for results
        epochs: Specific epochs to analyze (e.g., [1, 3, 5, 10]). If None, find all.
        max_batches: Max batches per epoch analysis
        batch_size: Batch size for dataloader

    Returns:
        Dictionary with metrics per epoch
    """
    from pathlib import Path
    import re

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(checkpoint_dir)

    print(f"\n{'='*60}")
    print("EPOCH PROGRESSION ANALYSIS")
    print(f"{'='*60}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Device: {device}")

    # Find epoch checkpoints
    if epochs is None:
        # Auto-detect epoch checkpoints
        epoch_files = {}
        for f in checkpoint_path.glob('*.pt'):
            # Match patterns like: epoch_1.pt, checkpoint_epoch_3.pt, etc.
            match = re.search(r'epoch[_-]?(\d+)', f.stem, re.IGNORECASE)
            if match:
                epoch_num = int(match.group(1))
                epoch_files[epoch_num] = f
        epochs = sorted(epoch_files.keys())
    else:
        # Find files for specified epochs
        epoch_files = {}
        for epoch in epochs:
            patterns = [
                f'epoch_{epoch}.pt',
                f'epoch{epoch}.pt',
                f'checkpoint_epoch_{epoch}.pt',
                f'epoch-{epoch}.pt',
            ]
            for pattern in patterns:
                f = checkpoint_path / pattern
                if f.exists():
                    epoch_files[epoch] = f
                    break

    if not epoch_files:
        print(f"No epoch checkpoints found in {checkpoint_dir}")
        return {}

    print(f"Found epochs: {sorted(epoch_files.keys())}")

    # Load validation data
    print(f"\nLoading validation data: {val_data_path}")
    with open(val_data_path, 'rb') as f:
        val_texts = pickle.load(f)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

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

    # Track metrics across epochs
    epoch_metrics = {}
    model = None
    version = None

    for epoch in sorted(epoch_files.keys()):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}")
        print(f"{'='*60}")

        checkpoint = torch.load(epoch_files[epoch], map_location=device)
        config = checkpoint.get('model_config', checkpoint.get('config', {}))

        # Detect version from first checkpoint
        if version is None:
            version = config.get('model_version', 'auto')
            path_str = str(epoch_files[epoch]).lower()
            if version == 'auto':
                if 'v13' in path_str:
                    version = '13.0'
                elif 'v12_7' in path_str or 'v12.7' in path_str:
                    version = '12.7'
                else:
                    version = '12.3'
            print(f"Model version: {version}")

            # Create model once
            from models import create_model_by_version
            model_kwargs = {
                'vocab_size': config.get('vocab_size', 30522),
                'd_model': config.get('d_model', 320),
                'n_layers': config.get('n_layers', 4),
                'n_heads': config.get('n_heads', 4),
                'rank': config.get('rank', 64),
                'max_seq_len': config.get('max_seq_len', 128),
                'n_compress': config.get('n_compress', 48),
                'n_expand': config.get('n_expand', 12),
                'n_knowledge': config.get('n_knowledge', 80),
                'knowledge_k': config.get('knowledge_k', 10),
                'dropout': config.get('dropout', 0.1),
                'state_dim': config.get('state_dim', 64),
            }
            if version in ['12.7', '12.8', '13.0'] or version.startswith('13'):
                model_kwargs['top_k_compress'] = config.get('top_k_compress', 8)
                model_kwargs['top_k_expand'] = config.get('top_k_expand', 4)

            model = create_model_by_version(version, model_kwargs)
            model = model.to(device)

            dataset = SimpleDataset(val_texts, tokenizer, model_kwargs['max_seq_len'])
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=2
            )

        # Load weights
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Get training metrics from checkpoint
        train_loss = checkpoint.get('train_loss', checkpoint.get('loss'))
        val_loss = checkpoint.get('val_loss')

        # Run quick analysis
        analyzer = DAWNAnalyzer(model, tokenizer, device, model_version=version)

        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }

        # Quick routing diversity check (key metric for collapse)
        try:
            diversity = analyzer.analyze_routing_diversity(dataloader, max_batches=min(max_batches, 20))
            if diversity:
                metrics['compress_pref_entropy'] = diversity.get('compress_pref', {}).get('entropy_ratio')
                metrics['expand_pref_Q_entropy'] = diversity.get('expand_pref_Q', {}).get('entropy_ratio')
                metrics['compress_weights_entropy'] = diversity.get('compress_weights', {}).get('entropy_ratio')
                metrics['expand_weights_Q_entropy'] = diversity.get('expand_weights_Q', {}).get('entropy_ratio')
        except Exception as e:
            print(f"  Warning: Routing diversity analysis failed: {e}")

        # Quick importance distribution check
        try:
            importance = analyzer.analyze_importance_distribution(dataloader, max_batches=min(max_batches, 20))
            if importance:
                metrics['importance_entropy'] = importance.get('entropy', {}).get('mean')
                metrics['importance_gini'] = importance.get('gini', {}).get('mean')
                metrics['importance_top10_conc'] = importance.get('top_10_concentration', {}).get('mean')
        except Exception as e:
            print(f"  Warning: Importance analysis failed: {e}")

        epoch_metrics[epoch] = metrics
        print(f"\nEpoch {epoch} Summary: train_loss={train_loss}, val_loss={val_loss}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("EPOCH COMPARISON")
    print(f"{'='*60}")

    epochs_sorted = sorted(epoch_metrics.keys())

    # Header
    header = f"{'Metric':<30}"
    for e in epochs_sorted:
        header += f"E{e:<8}"
    print(header)
    print("-" * (30 + 9 * len(epochs_sorted)))

    # Rows
    metric_names = [
        ('train_loss', 'Train Loss'),
        ('val_loss', 'Val Loss'),
        ('compress_pref_entropy', 'Compress Pref Entropy'),
        ('expand_pref_Q_entropy', 'Expand Pref Q Entropy'),
        ('compress_weights_entropy', 'Compress Weights Entropy'),
        ('expand_weights_Q_entropy', 'Expand Weights Q Entropy'),
        ('importance_entropy', 'Importance Entropy'),
        ('importance_gini', 'Importance Gini'),
        ('importance_top10_conc', 'Importance Top10%'),
    ]

    for key, name in metric_names:
        row = f"{name:<30}"
        for e in epochs_sorted:
            val = epoch_metrics[e].get(key)
            if val is not None:
                if isinstance(val, float):
                    row += f"{val:<9.4f}"
                else:
                    row += f"{val:<9}"
            else:
                row += f"{'N/A':<9}"
        print(row)

    # Diagnose trends
    print(f"\n--- Trend Analysis ---")

    if len(epochs_sorted) >= 2:
        first_epoch = epochs_sorted[0]
        last_epoch = epochs_sorted[-1]

        # Check entropy trend
        e_first = epoch_metrics[first_epoch].get('expand_pref_Q_entropy')
        e_last = epoch_metrics[last_epoch].get('expand_pref_Q_entropy')

        if e_first is not None and e_last is not None:
            delta = e_last - e_first
            if delta < -0.1:
                print(f"⚠ Expand entropy DECREASING: {e_first:.2%} → {e_last:.2%} (Δ={delta:.2%})")
                print("  → Router may be collapsing during training!")
            elif delta > 0.1:
                print(f"✓ Expand entropy INCREASING: {e_first:.2%} → {e_last:.2%} (Δ={delta:.2%})")
                print("  → Router becoming more diverse (good!)")
            else:
                print(f"  Expand entropy stable: {e_first:.2%} → {e_last:.2%}")

        # Check importance concentration trend
        c_first = epoch_metrics[first_epoch].get('importance_top10_conc')
        c_last = epoch_metrics[last_epoch].get('importance_top10_conc')

        if c_first is not None and c_last is not None:
            delta = c_last - c_first
            if delta > 0.1:
                print(f"⚠ Top10% concentration INCREASING: {c_first:.1%} → {c_last:.1%}")
                print("  → SSM becoming more concentrated")
            elif delta < -0.1:
                print(f"✓ Top10% concentration DECREASING: {c_first:.1%} → {c_last:.1%}")
            else:
                print(f"  Top10% concentration stable: {c_first:.1%} → {c_last:.1%}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'epoch_progression.json')
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(epoch_metrics), f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Visualization
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss curves
        ax = axes[0, 0]
        train_losses = [epoch_metrics[e].get('train_loss') for e in epochs_sorted]
        val_losses = [epoch_metrics[e].get('val_loss') for e in epochs_sorted]
        if any(v is not None for v in train_losses):
            ax.plot(epochs_sorted, train_losses, 'o-', label='Train')
        if any(v is not None for v in val_losses):
            ax.plot(epochs_sorted, val_losses, 's-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()

        # Entropy trends
        ax = axes[0, 1]
        compress_ent = [epoch_metrics[e].get('compress_pref_entropy') for e in epochs_sorted]
        expand_ent = [epoch_metrics[e].get('expand_pref_Q_entropy') for e in epochs_sorted]
        if any(v is not None for v in compress_ent):
            ax.plot(epochs_sorted, compress_ent, 'o-', label='Compress Pref')
        if any(v is not None for v in expand_ent):
            ax.plot(epochs_sorted, expand_ent, 's-', label='Expand Pref Q')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Entropy Ratio')
        ax.set_title('Router Entropy (higher=diverse)')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% target')
        ax.legend()

        # Importance distribution
        ax = axes[1, 0]
        imp_ent = [epoch_metrics[e].get('importance_entropy') for e in epochs_sorted]
        imp_gini = [epoch_metrics[e].get('importance_gini') for e in epochs_sorted]
        if any(v is not None for v in imp_ent):
            ax.plot(epochs_sorted, imp_ent, 'o-', label='Entropy')
        if any(v is not None for v in imp_gini):
            ax.plot(epochs_sorted, imp_gini, 's-', label='Gini')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title('Importance Distribution')
        ax.legend()

        # Top-k concentration
        ax = axes[1, 1]
        top10 = [epoch_metrics[e].get('importance_top10_conc') for e in epochs_sorted]
        if any(v is not None for v in top10):
            ax.plot(epochs_sorted, top10, 'o-', color='coral')
            ax.fill_between(epochs_sorted, 0, top10, alpha=0.3, color='coral')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Concentration')
        ax.set_title('Top 10% Token Concentration')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% threshold')

        plt.tight_layout()
        viz_path = os.path.join(output_dir, 'epoch_progression.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"Visualization saved: {viz_path}")

    return epoch_metrics


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN Unified Analysis')
    parser.add_argument('--checkpoint', type=str, required=False,
                        help='Path to checkpoint (or checkpoint dir for --epoch_tracking)')
    parser.add_argument('--val_data', type=str, required=False,
                        help='Path to validation data (pkl)')
    parser.add_argument('--epoch_tracking', action='store_true',
                        help='Run epoch progression analysis instead of single checkpoint')
    parser.add_argument('--epochs', type=str, default=None,
                        help='Specific epochs to analyze (comma-separated, e.g., "1,3,5,10")')
    parser.add_argument('--max_batches', type=int, default=50,
                        help='Max batches for analysis')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                        help='Output directory')
    parser.add_argument('--version', type=str, default='auto',
                        help='Model version (auto, 10.0, 12.3, etc.)')
    args = parser.parse_args()

    # Epoch tracking mode
    if args.epoch_tracking:
        if not args.checkpoint or not args.val_data:
            print("Error: --checkpoint (checkpoint dir) and --val_data are required for epoch tracking")
            return

        epochs = None
        if args.epochs:
            epochs = [int(e.strip()) for e in args.epochs.split(',')]

        analyze_epoch_progression(
            checkpoint_dir=args.checkpoint,
            val_data_path=args.val_data,
            output_dir=args.output_dir,
            epochs=epochs,
            max_batches=args.max_batches,
            batch_size=args.batch_size,
        )
        return

    # Regular single-checkpoint analysis
    if not args.checkpoint or not args.val_data:
        print("Error: --checkpoint and --val_data are required")
        print("Usage:")
        print("  Single checkpoint: python analyze_dawn.py --checkpoint <path> --val_data <path>")
        print("  Epoch tracking:    python analyze_dawn.py --epoch_tracking --checkpoint <dir> --val_data <path>")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        best = checkpoint_path / 'best_model.pt'
        if best.exists():
            checkpoint_path = best
        else:
            pt_files = list(checkpoint_path.glob('*.pt'))
            if pt_files:
                checkpoint_path = max(pt_files, key=lambda p: p.stat().st_mtime)

    print(f"\nLoading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('model_config', checkpoint.get('config', {}))
    version = config.get('model_version', args.version)

    # Auto-detect version
    if version == 'auto':
        # 1. Try checkpoint path
        path_str = str(checkpoint_path).lower()
        if 'v13' in path_str:
            version = '13.0'
        elif 'v12_8' in path_str or 'v12.8' in path_str:
            version = '12.8'
        elif 'v12_7' in path_str or 'v12.7' in path_str:
            version = '12.7'
        elif 'v12_6' in path_str or 'v12.6' in path_str:
            version = '12.6'
        elif 'v12_5' in path_str or 'v12.5' in path_str:
            version = '12.5'
        elif 'v12_4' in path_str or 'v12.4' in path_str:
            version = '12.4'
        elif 'v12_3' in path_str or 'v12.3' in path_str:
            version = '12.3'
        elif 'v12_2' in path_str or 'v12.2' in path_str:
            version = '12.2'
        elif 'v12_1' in path_str or 'v12.1' in path_str:
            version = '12.1'
        elif 'v12_0' in path_str or 'v12.0' in path_str or 'v12' in path_str:
            version = '12.0'
        elif 'v11' in path_str:
            version = '11.0'
        elif 'v10' in path_str:
            version = '10.0'
        else:
            # 2. Detect from state_dict keys
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            keys = list(state_dict.keys())
            keys_str = ' '.join(keys)

            if 'context_proj' in keys_str:
                version = '13.0'  # v13 has context_proj in SSM
            elif 'A_log' in keys_str:
                version = '12.7'  # v12.7 has Mamba-style SSM
            elif 'O_compress_pool' in keys_str or 'O_expand_pool' in keys_str:
                version = '12.4'  # v12.4c with low_rank_O
            elif 'O_pool' in keys_str:
                version = '12.4'  # v12.4a
            elif 'expand_neurons_pool' in keys_str:
                version = '12.3'
            elif 'ssm' in keys_str or 'SSM' in keys_str:
                version = '12.0'
            else:
                version = '10.0'

        print(f"Auto-detected version: {version}")

    print(f"Model version: {version}")

    # Import model
    from models import create_model_by_version

    model_kwargs = {
        'vocab_size': config.get('vocab_size', 30522),
        'd_model': config.get('d_model', 320),
        'n_layers': config.get('n_layers', 4),
        'n_heads': config.get('n_heads', 4),
        'rank': config.get('rank', 64),
        'max_seq_len': config.get('max_seq_len', 128),
        'n_compress': config.get('n_compress', 48),
        'n_expand': config.get('n_expand', 12),
        'n_knowledge': config.get('n_knowledge', 80),
        'knowledge_k': config.get('knowledge_k', 10),
        'dropout': config.get('dropout', 0.1),
    }

    # Add version-specific params
    if version.startswith('12') or version.startswith('13'):
        model_kwargs['state_dim'] = config.get('state_dim', 64)
        if 'knowledge_rank' in config:
            model_kwargs['knowledge_rank'] = config['knowledge_rank']

    # v12.7, v12.8, v13 have top-k sparse routing
    if version in ['12.7', '12.8', '13.0'] or version.startswith('13'):
        model_kwargs['top_k_compress'] = config.get('top_k_compress', 8)
        model_kwargs['top_k_expand'] = config.get('top_k_expand', 4)

    model = create_model_by_version(version, model_kwargs)

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
    analyzer = DAWNAnalyzer(model, tokenizer, device, model_version=version)
    all_results = analyzer.run_all(dataloader, max_batches=args.max_batches, output_dir=args.output_dir)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\nResults saved: {results_path}")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
