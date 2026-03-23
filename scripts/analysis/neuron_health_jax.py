"""
Neuron Health Analysis (Forward-based) - JAX Version
=====================================================
Analyze neuron health status using forward pass data.

JAX/Flax compatible version for TPU analysis.

All metrics computed from actual routing weights during inference,
not from EMA statistics.

NOTE: This is a JAX port of neuron_health.py (354 lines).
"""

import os
import numpy as np
from typing import Dict, Optional
from collections import defaultdict

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs): return x

from .base_jax import BaseAnalyzerJAX
from .utils_jax import (
    ALL_ROUTING_KEYS, POOL_N_ATTR,
    gini_coefficient,
    create_batches,
    JAXRoutingData,
    extract_full_routing,
)

# Map from full-routing pool keys to health pool names
_ROUTING_KEY_TO_POOL = {
    'fqk_q': 'feature_qk', 'fqk_k': 'feature_qk',
    'fv': 'feature_v',
    'rqk_q': 'restore_qk', 'rqk_k': 'restore_qk',
    'rv': 'restore_v',
    'fknow': 'feature_know', 'rknow': 'restore_know',
}
_ALL_ROUTING_POOL_KEYS = ['fqk_q', 'fqk_k', 'fv', 'rqk_q', 'rqk_k', 'rv', 'fknow', 'rknow']


class NeuronHealthAnalyzerJAX(BaseAnalyzerJAX):
    """Forward-based neuron health analyzer (JAX version)."""

    def __init__(self, model, params, config: Dict):
        """
        Initialize analyzer.

        Args:
            model: JAX/Flax model class
            params: FrozenDict of model parameters
            config: Model configuration dict
        """
        super().__init__(model, params, config)

    def analyze_activation_distribution(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        threshold: float = 0.01,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze neuron activation distribution using full forward routing.

        Uses extract_full_routing for all-layer routing extraction,
        ensuring dead neuron detection accounts for activation across all layers.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            threshold: Weight threshold for "active" classification
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with per-pool activation statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        # Get pool configuration
        pools = {
            'feature_qk': self.n_feature_qk,
            'feature_v': self.n_feature_v,
            'restore_qk': self.n_restore_qk,
            'restore_v': self.n_restore_v,
            'feature_know': self.n_feature_know,
            'restore_know': self.n_restore_know,
        }

        # Initialize accumulators
        activation_counts = {pool: np.zeros(n) for pool, n in pools.items() if n > 0}
        total_tokens = 0

        n_layers = self.config.get('n_layers', 16)

        # Create batches
        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Health Analysis (full routing)'):
            input_ids = np.array(batch)
            batch_tokens = input_ids.size
            total_tokens += batch_tokens

            # Full forward routing across all layers
            routing = extract_full_routing(self.params, self.config, input_ids)

            for li in range(n_layers):
                layer_data = routing.get(f'layer_{li}', {})
                for key in _ALL_ROUTING_POOL_KEYS:
                    pool = _ROUTING_KEY_TO_POOL.get(key)
                    if pool not in activation_counts:
                        continue
                    weights = layer_data.get(key)
                    if weights is None:
                        continue
                    w = np.asarray(weights)
                    # [B, S, N] -> count activations per neuron
                    active = (w > threshold).astype(np.float32).sum(axis=(0, 1))
                    activation_counts[pool] += active

        # Compute statistics
        results = {}
        for pool, counts in activation_counts.items():
            n_total = len(counts)
            n_active = int((counts > 0).sum())
            n_dead = n_total - n_active

            # Normalize: total observations = total_tokens * n_layers
            total_obs = total_tokens * n_layers
            freq = counts / (total_obs + 1e-8)

            results[pool] = {
                'total': n_total,
                'active': n_active,
                'dead': n_dead,
                'active_ratio': n_active / n_total if n_total > 0 else 0,
                'dead_ratio': n_dead / n_total if n_total > 0 else 0,
                'gini': gini_coefficient(freq),
                'stats': {
                    'min_freq': float(freq.min()),
                    'max_freq': float(freq.max()),
                    'mean_freq': float(freq.mean()),
                    'std_freq': float(freq.std()),
                    'median_freq': float(np.median(freq)),
                },
                'total_tokens': total_tokens,
                'n_layers': n_layers,
                'routing_mode': 'full_forward',
            }

        return results

    def analyze_dead_neurons(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        threshold: float = 0.01,
        output_dir: Optional[str] = None,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Identify dead neurons using full forward routing across all layers.

        Uses extract_full_routing so that a neuron activated in ANY layer
        is correctly classified as active (not just embedding-level).

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            threshold: Weight threshold for activation
            output_dir: Directory for visualization output
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with dead neuron analysis
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        # Get pool configuration
        pools = {
            'feature_qk': self.n_feature_qk,
            'feature_v': self.n_feature_v,
            'restore_qk': self.n_restore_qk,
            'restore_v': self.n_restore_v,
            'feature_know': self.n_feature_know,
            'restore_know': self.n_restore_know,
        }

        n_layers = self.config.get('n_layers', 16)

        # Track which neurons were ever activated
        ever_activated = {pool: np.zeros(n, dtype=bool) for pool, n in pools.items() if n > 0}

        # Create batches
        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Dead Neuron Analysis (full routing)'):
            input_ids = np.array(batch)

            # Full forward routing across all layers
            routing = extract_full_routing(self.params, self.config, input_ids)

            for li in range(n_layers):
                layer_data = routing.get(f'layer_{li}', {})
                for key in _ALL_ROUTING_POOL_KEYS:
                    pool = _ROUTING_KEY_TO_POOL.get(key)
                    if pool not in ever_activated:
                        continue
                    weights = layer_data.get(key)
                    if weights is None:
                        continue
                    w = np.asarray(weights)
                    # [B, S, N] -> any activation across batch & sequence
                    active = (w > threshold).any(axis=0).any(axis=0)
                    ever_activated[pool] |= active

        # Compile results
        results = {}
        total_dead = 0
        total_neurons = 0

        for pool, activated in ever_activated.items():
            n_total = len(activated)
            n_active = int(activated.sum())
            n_dead = n_total - n_active
            dead_ids = np.where(~activated)[0].tolist()

            total_dead += n_dead
            total_neurons += n_total

            results[pool] = {
                'n_total': n_total,
                'n_active': n_active,
                'n_dead': n_dead,
                'dead_ratio': n_dead / n_total if n_total > 0 else 0,
                'dead_neuron_ids': dead_ids,
            }

        results['summary'] = {
            'total_dead': total_dead,
            'total_neurons': total_neurons,
            'dead_ratio': total_dead / total_neurons if total_neurons > 0 else 0,
            'n_layers': n_layers,
            'routing_mode': 'full_forward',
        }

        # Visualization
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        return results

    def analyze_diversity(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        threshold: float = 0.01,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Analyze neuron usage diversity using full forward routing.

        Uses extract_full_routing for all-layer routing extraction.

        Args:
            val_tokens: Validation token array
            n_batches: Number of batches to process
            threshold: Weight threshold for activation
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with diversity metrics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        # Get pool configuration
        pools = {
            'feature_qk': self.n_feature_qk,
            'feature_v': self.n_feature_v,
            'restore_qk': self.n_restore_qk,
            'restore_v': self.n_restore_v,
            'feature_know': self.n_feature_know,
            'restore_know': self.n_restore_know,
        }

        n_layers = self.config.get('n_layers', 16)

        # Initialize accumulators
        activation_counts = {pool: np.zeros(n) for pool, n in pools.items() if n > 0}

        # Create batches
        batches = create_batches(val_tokens, batch_size, seq_len)
        if n_batches:
            batches = batches[:n_batches]

        for batch in tqdm(batches, desc='Diversity Analysis (full routing)'):
            input_ids = np.array(batch)

            # Full forward routing across all layers
            routing = extract_full_routing(self.params, self.config, input_ids)

            for li in range(n_layers):
                layer_data = routing.get(f'layer_{li}', {})
                for key in _ALL_ROUTING_POOL_KEYS:
                    pool = _ROUTING_KEY_TO_POOL.get(key)
                    if pool not in activation_counts:
                        continue
                    weights = layer_data.get(key)
                    if weights is None:
                        continue
                    w = np.asarray(weights)
                    active = (w > threshold).astype(np.float32).sum(axis=(0, 1))
                    activation_counts[pool] += active

        # Compute diversity metrics
        results = {}
        entropies = []

        for pool, counts in activation_counts.items():
            n_total = len(counts)
            active_mask = counts > 0
            n_active = int(active_mask.sum())

            if n_active == 0:
                results[pool] = {
                    'n_active': 0,
                    'n_total': n_total,
                    'entropy': 0,
                    'normalized_entropy': 0,
                    'effective_count': 0,
                    'coverage': 0,
                }
                continue

            # Compute entropy from activation distribution
            active_counts = counts[active_mask]
            p = active_counts / active_counts.sum()

            entropy = -np.sum(p * np.log(p + 1e-8))
            max_entropy = np.log(n_active)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            effective_count = np.exp(entropy)

            # Top-k concentration
            top5_indices = np.argsort(active_counts)[-min(5, n_active):]
            top5 = active_counts[top5_indices]
            top5_share = float(top5.sum() / active_counts.sum())

            results[pool] = {
                'n_active': n_active,
                'n_total': n_total,
                'entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'effective_count': float(effective_count),
                'coverage': n_active / n_total,
                'top5_share': top5_share,
                'gini': gini_coefficient(counts),
            }
            entropies.append(normalized_entropy)

        # Overall score
        overall = sum(entropies) / len(entropies) if entropies else 0
        results['overall'] = {
            'diversity_score': overall,
            'health': 'good' if overall > 0.7 else 'warning' if overall > 0.4 else 'critical'
        }

        return results

    def run_all(
        self,
        val_tokens: np.ndarray,
        output_dir: str = './neuron_health',
        n_batches: int = 50,
        batch_size: int = 32,
        seq_len: int = 512
    ) -> Dict:
        """
        Run all neuron health analyses (forward-based).

        Args:
            val_tokens: Validation token array
            output_dir: Directory for outputs
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'activation_distribution': self.analyze_activation_distribution(
                val_tokens, n_batches, batch_size=batch_size, seq_len=seq_len
            ),
            'diversity': self.analyze_diversity(
                val_tokens, n_batches, batch_size=batch_size, seq_len=seq_len
            ),
            'dead_neurons': self.analyze_dead_neurons(
                val_tokens, n_batches, output_dir=output_dir, batch_size=batch_size, seq_len=seq_len
            ),
        }

        return results
