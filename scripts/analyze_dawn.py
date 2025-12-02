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

    v12.3 format:
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
    """

    def __init__(self, model_version: str):
        self.version = model_version
        self.is_v10 = model_version.startswith('10') or model_version in ['10.0', '11.0']
        self.is_v12 = model_version.startswith('12')

    def detect_version(self, routing_info: Dict) -> str:
        """Auto-detect version from routing_info structure"""
        if 'attention' in routing_info:
            attn = routing_info['attention']
            if 'Q' in attn and isinstance(attn['Q'], dict):
                return 'v10'
            elif 'compress_weights' in attn:
                return 'v12'
        return 'unknown'

    def get_compress_weights(self, routing_info: Dict, comp: str = 'Q') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get compress weights for a component.

        Returns:
            weights: [B, S, N] for v10.x (token-level) or [B, N] for v12.3 (batch-level)
            indices: [B, S, k] or None
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

        elif version == 'v12':
            attn = routing_info['attention']
            mem = routing_info['memory']

            if comp in ['Q', 'K', 'V']:
                # v12.3: shared compress_weights, separate expand_weights
                weights = attn.get('compress_weights')
            else:  # M
                weights = mem.get('neuron_weights')

            return weights, None

        return None, None

    def get_expand_weights(self, routing_info: Dict, comp: str = 'Q') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get expand weights for a component.

        Returns:
            weights: [B, S, n_expand] for v10.x (token-level) or [B, n_expand] for v12.3 (batch-level)
            indices: [B, S, k] or None
        """
        version = self.detect_version(routing_info)

        if version == 'v10':
            data = routing_info['attention'].get('O', {})
            return data.get('weights'), data.get('indices')

        elif version == 'v12':
            attn = routing_info['attention']
            key = f'expand_weights_{comp}'
            weights = attn.get(key)
            return weights, None

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

        elif version == 'v12':
            attn = routing_info['attention']

            if comp == 'compress':
                # compress_pref [B, S, n_compress]
                return attn.get('compress_pref')
            elif comp in ['Q', 'K', 'V']:
                # expand_pref_Q/K/V [B, S, n_expand]
                return attn.get(f'expand_pref_{comp}')
            elif comp == 'M':
                # Memory uses compress routing
                return attn.get('compress_pref')

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
                if hasattr(shared, 'expand_neurons_pool'):
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

        layer_comp_usage = {l: {comp: torch.zeros(self.n_compress, device=self.device)
                               for comp in ['Q', 'K', 'V', 'M']}
                          for l in range(self.n_layers)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Layer Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                for comp in ['Q', 'K', 'V', 'M']:
                    weights, _ = self.parser.get_compress_weights(routing_info, comp)
                    if weights is not None:
                        # Handle different shapes:
                        # v10.x: [B, S, N] -> sum over (B, S)
                        # v12.3: [B, N] -> sum over (B)
                        if len(weights.shape) == 3:
                            layer_comp_usage[layer_idx][comp] += weights.sum(dim=(0, 1))
                        else:
                            layer_comp_usage[layer_idx][comp] += weights.sum(dim=0)

        results = {'layer_correlation': {}}

        print(f"\n--- Q/K/V/M Correlation by Layer ---")

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

    # ============================================================
    # 7. VISUALIZATION
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

        # 7. Visualization
        os.makedirs(output_dir, exist_ok=True)
        viz_path = os.path.join(output_dir, 'analysis_summary.png')
        self.visualize(all_results, viz_path)

        return all_results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN Unified Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation data (pkl)')
    parser.add_argument('--max_batches', type=int, default=50,
                        help='Max batches for analysis')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                        help='Output directory')
    parser.add_argument('--version', type=str, default='auto',
                        help='Model version (auto, 10.0, 12.3, etc.)')
    args = parser.parse_args()

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
        if 'v12_4' in path_str or 'v12.4' in path_str:
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

            if 'O_compress_pool' in keys_str or 'O_expand_pool' in keys_str:
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
    if version.startswith('12'):
        model_kwargs['state_dim'] = config.get('state_dim', 64)
        if 'knowledge_rank' in config:
            model_kwargs['knowledge_rank'] = config['knowledge_rank']

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
