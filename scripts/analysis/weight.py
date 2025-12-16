"""
Weight Matrix Analysis
======================
Analyze neuron weight matrices in DAWN v17.1 models.

Includes:
- SVD decomposition analysis
- Effective rank calculation
- Condition number analysis
"""

import os
import numpy as np
import torch
from typing import Dict, Optional

from .utils import (
    NEURON_TYPES, NEURON_ATTRS,
    HAS_MATPLOTLIB, plt
)


class WeightAnalyzer:
    """Neuron weight matrix analyzer."""

    def __init__(self, neurons):
        """
        Initialize analyzer.

        Args:
            neurons: SharedNeurons instance from DAWN model
        """
        self.neurons = neurons

    def analyze_weight_svd(self, output_dir: Optional[str] = None) -> Dict:
        """
        Analyze neuron weights using SVD decomposition.

        Computes:
        - Singular value distribution
        - Effective rank (number of significant singular values)
        - Condition number
        - Variance explained by top singular values

        Args:
            output_dir: Directory for visualization output

        Returns:
            Dictionary with SVD analysis results
        """
        if self.neurons is None:
            return {'error': 'No shared neurons found'}

        results = {}

        for name, attr in NEURON_ATTRS.items():
            if not hasattr(self.neurons, attr):
                continue

            W = getattr(self.neurons, attr).detach().cpu()
            n_neurons = W.shape[0]

            # Flatten to 2D for SVD
            if W.dim() > 2:
                W = W.view(n_neurons, -1)

            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception:
                results[name] = {'error': 'SVD failed'}
                continue

            # Compute metrics
            S_normalized = S / S.sum()
            cumsum = torch.cumsum(S_normalized, dim=0)
            effective_rank = float((S > S.max() * 0.01).sum())
            var_top5 = float(cumsum[min(4, len(cumsum)-1)])
            var_top10 = float(cumsum[min(9, len(cumsum)-1)])

            results[name] = {
                'display': NEURON_TYPES.get(name, (name,))[0],
                'n_neurons': n_neurons,
                'weight_shape': list(W.shape),
                'effective_rank': effective_rank,
                'var_explained_by_top5': var_top5,
                'var_explained_by_top10': var_top10,
                'top_singular_values': S[:10].tolist(),
                'condition_number': float(S[0] / S[-1]) if S[-1] > 0 else float('inf'),
            }

        # Visualization
        if HAS_MATPLOTLIB and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._visualize_svd(results, output_dir)

        return results

    def _visualize_svd(self, results: Dict, output_dir: str):
        """Generate SVD visualization."""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if not valid_results:
            return

        n_plots = len(valid_results)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for ax, (name, data) in zip(axes, valid_results.items()):
            sv = data['top_singular_values']
            ax.bar(range(len(sv)), sv)
            ax.set_title(f'{data["display"]} Singular Values')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')

        plt.tight_layout()
        path = os.path.join(output_dir, 'weight_svd.png')
        plt.savefig(path, dpi=150)
        plt.close()
        results['visualization'] = path

    def analyze_weight_norms(self) -> Dict:
        """
        Analyze weight matrix norms per neuron.

        Returns:
            Dictionary with norm statistics
        """
        if self.neurons is None:
            return {'error': 'No shared neurons found'}

        results = {}

        for name, attr in NEURON_ATTRS.items():
            if not hasattr(self.neurons, attr):
                continue

            W = getattr(self.neurons, attr).detach().cpu()
            n_neurons = W.shape[0]

            # Compute per-neuron norms
            norms = torch.norm(W.view(n_neurons, -1), dim=1)

            results[name] = {
                'display': NEURON_TYPES.get(name, (name,))[0],
                'n_neurons': n_neurons,
                'mean_norm': float(norms.mean()),
                'std_norm': float(norms.std()),
                'min_norm': float(norms.min()),
                'max_norm': float(norms.max()),
            }

        return results

    def analyze_weight_similarity(self) -> Dict:
        """
        Analyze pairwise similarity between neuron weight matrices.

        Returns:
            Dictionary with similarity statistics
        """
        if self.neurons is None:
            return {'error': 'No shared neurons found'}

        results = {}

        for name, attr in NEURON_ATTRS.items():
            if not hasattr(self.neurons, attr):
                continue

            W = getattr(self.neurons, attr).detach().cpu()
            n_neurons = W.shape[0]

            if n_neurons < 2:
                continue

            # Flatten and normalize
            W_flat = W.view(n_neurons, -1)
            W_norm = W_flat / (W_flat.norm(dim=1, keepdim=True) + 1e-8)

            # Compute similarity matrix
            sim = W_norm @ W_norm.T

            # Extract off-diagonal
            mask = ~torch.eye(n_neurons, dtype=torch.bool)
            off_diag = sim[mask]

            results[name] = {
                'display': NEURON_TYPES.get(name, (name,))[0],
                'n_neurons': n_neurons,
                'mean_similarity': float(off_diag.mean()),
                'std_similarity': float(off_diag.std()),
                'max_similarity': float(off_diag.max()),
                'min_similarity': float(off_diag.min()),
            }

        return results

    def run_all(self, output_dir: str = './weight_analysis') -> Dict:
        """
        Run all weight analyses.

        Args:
            output_dir: Directory for outputs

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'svd': self.analyze_weight_svd(output_dir),
            'norms': self.analyze_weight_norms(),
            'similarity': self.analyze_weight_similarity(),
        }

        return results
