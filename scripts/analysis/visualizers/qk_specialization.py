"""
Q/K Specialization Visualizations
=================================
Visualization functions for Q/K neuron usage and specialization patterns.

Paper Figure 3: Q/K Specialization
- Left: Q vs K scatter plot with correlation
- Middle: Q-only/K-only/Shared/Inactive bar chart
- Right: Q/(Q+K) ratio histogram
"""

import os
import numpy as np
from typing import Dict, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def plot_qk_specialization(
    qk_usage_data: Dict,
    output_path: str,
    pool_colors: Dict = None,
    figsize_per_pool: tuple = (18, 6),
    dpi: int = 150
) -> Optional[str]:
    """
    Generate Q/K specialization figure (Paper Figure 3).

    Creates 3-panel figure for each QK pool:
    - Left: Q vs K usage scatter plot
    - Middle: Neuron specialization bar chart
    - Right: Q/(Q+K) ratio distribution histogram

    Args:
        qk_usage_data: Results from RoutingAnalyzer.analyze_qk_usage()
        output_path: Path to save the figure
        pool_colors: Optional dict mapping pool names to colors
        figsize_per_pool: Figure size per pool row
        dpi: Output resolution

    Returns:
        Path to saved figure or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    # Default colors
    if pool_colors is None:
        pool_colors = {
            'feature_qk': 'red',
            'restore_qk': 'blue',
        }

    # Filter out non-pool keys
    pools = {k: v for k, v in qk_usage_data.items() if isinstance(v, dict) and 'q_counts' in v}
    n_pools = len(pools)

    if n_pools == 0:
        return None

    fig, axes = plt.subplots(n_pools, 3, figsize=(figsize_per_pool[0], figsize_per_pool[1] * n_pools))
    if n_pools == 1:
        axes = axes.reshape(1, -1)

    for row, (pool_name, data) in enumerate(pools.items()):
        q_counts = np.array(data['q_counts'])
        k_counts = np.array(data['k_counts'])
        color = pool_colors.get(pool_name, 'gray')

        # 1. Scatter: Q vs K usage
        ax = axes[row, 0]
        ax.scatter(q_counts, k_counts, alpha=0.6, s=30, c=color)
        max_val = max(q_counts.max(), k_counts.max()) if len(q_counts) > 0 else 1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Q=K')
        ax.set_xlabel('Q Selection Count')
        ax.set_ylabel('K Selection Count')
        ax.set_title(f'{data["display"]}: Q vs K Usage\n(corr={data["correlation"]:.3f})')
        ax.legend()

        # 2. Bar: Specialization categories
        ax = axes[row, 1]
        categories = ['Q-only', 'K-only', 'Shared', 'Inactive']
        values = [
            data.get('q_specialized', 0),
            data.get('k_specialized', 0),
            data.get('shared', 0),
            data.get('inactive', 0)
        ]
        colors = ['blue', 'orange', 'green', 'gray']
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_ylabel('Neuron Count')
        ax.set_title(f'{data["display"]}: Neuron Specialization')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(val),
                   ha='center', va='bottom')

        # 3. Histogram: Q/(Q+K) ratio distribution
        ax = axes[row, 2]
        total = q_counts + k_counts + 1e-8
        q_ratio = q_counts / total
        active_mask = (q_counts + k_counts) > 0
        if active_mask.sum() > 0:
            ax.hist(q_ratio[active_mask], bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(x=0.5, color='red', linestyle='--', label='Q=K')
        ax.set_xlabel('Q Ratio (Q / (Q+K))')
        ax.set_ylabel('Neuron Count')
        ax.set_title(f'{data["display"]}: Q/K Balance Distribution')
        ax.legend()

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path


def plot_qk_usage(
    qk_usage_data: Dict,
    output_dir: str,
    filename: str = 'qk_usage_analysis.png'
) -> Optional[str]:
    """
    Convenience wrapper for plot_qk_specialization.

    Args:
        qk_usage_data: Results from RoutingAnalyzer.analyze_qk_usage()
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved figure or None
    """
    output_path = os.path.join(output_dir, filename)
    return plot_qk_specialization(qk_usage_data, output_path)
