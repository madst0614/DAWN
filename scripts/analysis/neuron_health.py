"""
Neuron Health Analysis
=======================
Analyze neuron health status in DAWN v17.1 models.

Includes:
- EMA distribution analysis
- Dead/Active neuron ratio
- Excitability state analysis
- Diversity metrics (Gini, entropy)
- Visualization utilities
"""

import os
import numpy as np
import torch
from typing import Dict, Optional

from .utils import (
    NEURON_TYPES, gini_coefficient,
    HAS_MATPLOTLIB, plt
)


class NeuronHealthAnalyzer:
    """Neuron health and usage pattern analyzer."""

    def __init__(self, router):
        """
        Initialize analyzer with router.

        Args:
            router: NeuronRouter instance from DAWN model
        """
        self.router = router

    def analyze_ema_distribution(self, threshold: float = 0.01) -> Dict:
        """
        Analyze EMA distribution across all neuron types.

        Args:
            threshold: Threshold for active/dead neuron classification

        Returns:
            Dictionary with per-type EMA statistics
        """
        results = {}

        for name, (display, ema_attr, n_attr, _) in NEURON_TYPES.items():
            if not hasattr(self.router, ema_attr):
                continue

            ema = getattr(self.router, ema_attr)
            n_total = getattr(self.router, n_attr)

            active = (ema > threshold).sum().item()
            dead = (ema < threshold).sum().item()

            results[name] = {
                'display': display,
                'total': n_total,
                'active': int(active),
                'dead': int(dead),
                'active_ratio': active / n_total,
                'dead_ratio': dead / n_total,
                'gini': gini_coefficient(ema),
                'stats': {
                    'min': float(ema.min()),
                    'max': float(ema.max()),
                    'mean': float(ema.mean()),
                    'std': float(ema.std()),
                    'median': float(ema.median()),
                }
            }

        return results

    def analyze_excitability(self) -> Dict:
        """
        Analyze excitability state of neurons.

        Excitability = 1 - EMA/tau, clamped to [0, 1]
        Higher excitability means the neuron is more likely to be selected.

        Returns:
            Dictionary with excitability statistics (empty if excitability not supported)
        """
        tau = getattr(self.router, 'tau', None)
        if tau is None:
            return {'supported': False, 'message': 'Excitability not supported in this model version'}

        weight = getattr(self.router, 'excitability_weight', 0)
        if hasattr(weight, 'item'):
            weight = weight.item()

        results = {
            'tau': tau,
            'weight': weight,
            'langevin_alpha': getattr(self.router, 'langevin_alpha', 0),
            'langevin_beta': getattr(self.router, 'langevin_beta', 0),
        }

        for name, (display, ema_attr, _, _) in NEURON_TYPES.items():
            if not hasattr(self.router, ema_attr):
                continue

            ema = getattr(self.router, ema_attr)
            exc = torch.clamp(1.0 - ema / tau, min=0.0, max=1.0)

            results[name] = {
                'display': display,
                'min': float(exc.min()),
                'max': float(exc.max()),
                'mean': float(exc.mean()),
                'std': float(exc.std()),
                'high_exc_count': int((exc > 0.8).sum()),
                'low_exc_count': int((exc < 0.2).sum()),
            }

        return results

    def analyze_diversity(self, threshold: float = 0.01) -> Dict:
        """
        Analyze neuron diversity using entropy and effective count.

        Args:
            threshold: Threshold for active neuron classification

        Returns:
            Dictionary with diversity metrics
        """
        results = {}

        for name, (display, ema_attr, n_attr, _) in NEURON_TYPES.items():
            if not hasattr(self.router, ema_attr):
                continue

            ema = getattr(self.router, ema_attr)
            n_total = getattr(self.router, n_attr)

            active_mask = ema > threshold
            n_active = active_mask.sum().item()

            if n_active == 0:
                results[name] = {
                    'display': display,
                    'n_active': 0,
                    'entropy': 0,
                    'effective_count': 0,
                    'coverage': 0,
                }
                continue

            active_ema = ema[active_mask]
            p = active_ema / active_ema.sum()

            entropy = -torch.sum(p * torch.log(p + 1e-8)).item()
            max_entropy = np.log(n_active)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            effective_count = np.exp(entropy)

            top5 = torch.topk(active_ema, min(5, n_active))[0]
            top5_share = top5.sum() / active_ema.sum()

            results[name] = {
                'display': display,
                'n_active': n_active,
                'n_total': n_total,
                'entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'effective_count': float(effective_count),
                'coverage': n_active / n_total,
                'top5_share': float(top5_share),
                'gini': gini_coefficient(ema),
            }

        # Overall diversity score
        entropies = [
            r['normalized_entropy']
            for r in results.values()
            if isinstance(r, dict) and 'normalized_entropy' in r
        ]
        overall = sum(entropies) / len(entropies) if entropies else 0

        results['overall'] = {
            'diversity_score': overall,
            'health': 'good' if overall > 0.7 else 'warning' if overall > 0.4 else 'critical'
        }

        return results

    def analyze_dead_neurons(self, output_dir: Optional[str] = None) -> Dict:
        """
        Analyze dead neurons and provide shrink recommendations.

        Args:
            output_dir: Directory for visualization output

        Returns:
            Dictionary with dead neuron analysis and recommendations
        """
        results = {}
        threshold = 0.01
        dying_threshold = 0.05
        tau = getattr(self.router, 'tau', None)

        for name, (display, ema_attr, n_attr, _) in NEURON_TYPES.items():
            if not hasattr(self.router, ema_attr):
                continue

            ema = getattr(self.router, ema_attr)
            n_total = getattr(self.router, n_attr)

            dead_mask = ema < threshold
            dying_mask = (ema >= threshold) & (ema < dying_threshold)
            active_mask = ema >= dying_threshold

            # Excitability-based classification (only if tau exists)
            if tau is not None:
                exc = torch.clamp(1.0 - ema / tau, min=0.0, max=1.0)
                # Revivable: dead but high excitability
                revivable_mask = dead_mask & (exc > 0.8)
                # Removable: dead and low excitability
                removable_mask = dead_mask & (exc < 0.3)
                n_revivable = int(revivable_mask.sum())
                n_removable = int(removable_mask.sum())
            else:
                n_revivable = 0
                n_removable = 0

            results[name] = {
                'display': display,
                'n_total': n_total,
                'n_active': int(active_mask.sum()),
                'n_dying': int(dying_mask.sum()),
                'n_dead': int(dead_mask.sum()),
                'n_revivable': n_revivable,
                'n_removable': n_removable,
                'dead_neuron_ids': dead_mask.nonzero().squeeze(-1).tolist()
                                   if dead_mask.sum() > 0 else [],
                'removable_neuron_ids': (removable_mask.nonzero().squeeze(-1).tolist()
                                        if tau is not None and removable_mask.sum() > 0 else []),
            }

        # Calculate recommendations
        type_names = [
            name for name in results.keys()
            if isinstance(results[name], dict) and 'n_total' in results[name]
        ]

        total_removable = sum(results[n]['n_removable'] for n in type_names)
        total_neurons = sum(results[n]['n_total'] for n in type_names)

        results['recommendation'] = {
            'total_removable': total_removable,
            'shrink_ratio': total_removable / total_neurons if total_neurons > 0 else 0,
            'action': 'shrink' if total_removable > total_neurons * 0.2 else 'keep',
            'per_type': {
                name: {
                    'current': results[name]['n_total'],
                    'recommended': results[name]['n_total'] - results[name]['n_removable'],
                    'remove': results[name]['n_removable'],
                }
                for name in type_names
            }
        }

        # Visualization
        if HAS_MATPLOTLIB and output_dir:
            self._visualize_dead_neurons(results, type_names, output_dir)

        return results

    def _visualize_dead_neurons(self, results: Dict, type_names: list, output_dir: str):
        """Generate dead neuron visualization."""
        os.makedirs(output_dir, exist_ok=True)

        n_types = len(type_names)
        n_cols = 3
        n_rows = (n_types + 1 + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_types == 0 else list(axes)

        colors = ['green', 'yellow', 'red']
        labels = ['Active', 'Dying', 'Dead']

        for ax, name in zip(axes[:n_types], type_names):
            data = results[name]
            sizes = [data['n_active'], data['n_dying'], data['n_dead']]
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'{name.upper()}\n(removable: {data["n_removable"]})')

        if n_types < len(axes):
            ax = axes[n_types]
            display_names = [n.upper()[:3] for n in type_names]
            removable = [results[n]['n_removable'] for n in type_names]
            ax.bar(display_names, removable, color='red', alpha=0.7)
            ax.set_title('Removable Neurons')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)

        for i in range(n_types + 1, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        path = os.path.join(output_dir, 'dead_neurons.png')
        plt.savefig(path, dpi=150)
        plt.close()
        results['visualization'] = path

    def visualize_usage(self, output_dir: str) -> Dict:
        """
        Create usage histogram plots.

        Args:
            output_dir: Directory for output

        Returns:
            Dictionary with visualization path
        """
        if not HAS_MATPLOTLIB:
            return {'error': 'matplotlib not available'}

        os.makedirs(output_dir, exist_ok=True)

        data = []
        for name, (display, ema_attr, _, color) in NEURON_TYPES.items():
            if hasattr(self.router, ema_attr):
                ema = getattr(self.router, ema_attr)
                data.append((display, ema, color))

        n_plots = len(data) + 1
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_plots == 1 else list(axes)

        for ax, (name, ema, color) in zip(axes, data):
            values = ema.detach().cpu().numpy()
            ax.hist(values, bins=50, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(x=0.01, color='red', linestyle='--', label='threshold=0.01')
            ax.set_title(f'{name} Usage EMA')
            ax.set_xlabel('EMA Value')
            ax.set_ylabel('Count')

            active = (values > 0.01).sum()
            total = len(values)
            ax.text(0.95, 0.95, f'Active: {active}/{total}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=10)

        # Summary bar chart
        if len(data) < len(axes):
            ax = axes[len(data)]
            names = [d[0] for d in data]
            active_ratios = [(d[1] > 0.01).float().mean().item() for d in data]
            colors = [d[2] for d in data]
            ax.bar(names, active_ratios, color=colors, alpha=0.7, edgecolor='black')
            ax.set_title('Active Neuron Ratio by Type')
            ax.set_ylabel('Active Ratio')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)

        for i in range(len(data) + 1, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        path = os.path.join(output_dir, 'usage_histogram.png')
        plt.savefig(path, dpi=150)
        plt.close()

        return {'visualization': path}

    def run_all(self, output_dir: str = './neuron_health') -> Dict:
        """
        Run all neuron health analyses.

        Args:
            output_dir: Directory for outputs

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'ema_distribution': self.analyze_ema_distribution(),
            'excitability': self.analyze_excitability(),
            'diversity': self.analyze_diversity(),
            'dead_neurons': self.analyze_dead_neurons(output_dir),
        }

        # Visualizations
        results['visualization'] = self.visualize_usage(output_dir)

        return results
