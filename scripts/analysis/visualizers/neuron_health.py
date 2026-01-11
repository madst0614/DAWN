"""
Neuron Health Visualizations
============================
Visualization functions for neuron health analysis.

Paper Figure 4: Neuron Health
- EMA distribution histograms
- Active/Dying/Dead pie charts
- Summary bar charts
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def plot_dead_neurons(
    results: Dict,
    type_names: List[str],
    output_path: str,
    dpi: int = 150
) -> Optional[str]:
    """
    Generate dead neuron visualization.

    Creates pie charts showing active/dying/dead breakdown per neuron type,
    plus a summary bar chart of removable neurons.

    Args:
        results: Dead neuron analysis results
        type_names: List of neuron type names to visualize
        output_path: Path to save the figure
        dpi: Output resolution

    Returns:
        Path to saved figure or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

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
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path


def plot_usage_histogram(
    ema_data: List[Tuple[str, np.ndarray, str]],
    output_path: str,
    threshold: float = 0.01,
    dpi: int = 150
) -> Optional[str]:
    """
    Create usage histogram plots for all neuron types.

    Args:
        ema_data: List of (display_name, ema_values, color) tuples
        output_path: Path to save the figure
        threshold: Threshold for active neuron classification
        dpi: Output resolution

    Returns:
        Path to saved figure or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    n_plots = len(ema_data) + 1
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_plots == 1 else list(axes)

    for ax, (name, ema, color) in zip(axes, ema_data):
        values = ema if isinstance(ema, np.ndarray) else ema.detach().cpu().numpy()
        ax.hist(values, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(x=threshold, color='red', linestyle='--', label=f'threshold={threshold}')
        ax.set_title(f'{name} Usage EMA')
        ax.set_xlabel('EMA Value')
        ax.set_ylabel('Count')

        active = (values > threshold).sum()
        total = len(values)
        ax.text(0.95, 0.95, f'Active: {active}/{total}', transform=ax.transAxes,
                ha='right', va='top', fontsize=10)

    # Summary bar chart
    if len(ema_data) < len(axes):
        ax = axes[len(ema_data)]
        names = [d[0] for d in ema_data]
        active_ratios = []
        for _, ema, _ in ema_data:
            values = ema if isinstance(ema, np.ndarray) else ema.detach().cpu().numpy()
            active_ratios.append((values > threshold).mean())
        colors = [d[2] for d in ema_data]
        ax.bar(names, active_ratios, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Active Neuron Ratio by Type')
        ax.set_ylabel('Active Ratio')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)

    for i in range(len(ema_data) + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path


def plot_qk_ema_overlap(
    qk_results: Dict,
    output_path: str,
    dpi: int = 150
) -> Optional[str]:
    """
    Visualize Q/K EMA overlap analysis.

    Creates scatter plots of Q vs K EMA for each QK pool,
    showing neuron specialization patterns.

    Args:
        qk_results: Results from analyze_qk_ema_overlap()
        output_path: Path to save the figure
        dpi: Output resolution

    Returns:
        Path to saved figure or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    pools = {k: v for k, v in qk_results.items() if isinstance(v, dict) and 'q_ema' in v}
    if not pools:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    n_pools = len(pools)
    fig, axes = plt.subplots(1, n_pools, figsize=(7 * n_pools, 6))
    if n_pools == 1:
        axes = [axes]

    colors = {'fqk': 'red', 'rqk': 'blue'}

    for ax, (pool_name, data) in zip(axes, pools.items()):
        q_ema = np.array(data['q_ema'])
        k_ema = np.array(data['k_ema'])
        color = colors.get(pool_name, 'gray')

        ax.scatter(q_ema, k_ema, alpha=0.6, s=30, c=color)
        max_val = max(q_ema.max(), k_ema.max()) if len(q_ema) > 0 else 1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Q=K')
        ax.set_xlabel('Q EMA')
        ax.set_ylabel('K EMA')
        ax.set_title(f'{pool_name.upper()}: Q vs K EMA\n'
                    f'corr={data["correlation"]:.3f}, '
                    f'Q-only={data["q_only"]}, K-only={data["k_only"]}, '
                    f'shared={data["shared"]}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path
