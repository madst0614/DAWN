"""
Layer Contribution Visualizations
=================================
Visualization for layer-wise circuit contribution analysis.

Paper Figure 6b: Layer-wise Attention vs Knowledge Contribution
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


def plot_layer_contribution(
    contribution_data: Dict,
    output_path: str,
    dpi: int = 150
) -> Optional[str]:
    """
    Generate layer-wise contribution visualization.

    Paper Figure 6b: Shows attention vs knowledge circuit contribution per layer.

    Args:
        contribution_data: Results from RoutingAnalyzer.analyze_layer_contribution()
        output_path: Path to save the figure
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    per_layer = contribution_data.get('per_layer', {})
    if not per_layer:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Extract data
    layers = sorted(per_layer.keys(), key=lambda x: int(x[1:]))
    layer_indices = [int(l[1:]) for l in layers]
    attn_ratios = [per_layer[l]['attention_ratio'] * 100 for l in layers]
    know_ratios = [per_layer[l]['knowledge_ratio'] * 100 for l in layers]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Line plot
    ax = axes[0]
    ax.plot(layer_indices, attn_ratios, 'b-o', label='Attention', linewidth=2, markersize=6)
    ax.plot(layer_indices, know_ratios, 'r-o', label='Knowledge', linewidth=2, markersize=6)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Contribution (%)')
    ax.set_title('Layer-wise Circuit Contribution')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # 2. Stacked bar chart
    ax = axes[1]
    width = 0.6
    ax.bar(layer_indices, attn_ratios, width, label='Attention', color='steelblue')
    ax.bar(layer_indices, know_ratios, width, bottom=attn_ratios, label='Knowledge', color='coral')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Contribution (%)')
    ax.set_title('Stacked Layer Contribution')
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return output_path


def plot_layer_contribution_detail(
    contribution_data: Dict,
    output_path: str,
    dpi: int = 150
) -> Optional[str]:
    """
    Generate detailed layer contribution heatmap.

    Args:
        contribution_data: Results from RoutingAnalyzer.analyze_layer_contribution()
        output_path: Path to save the figure
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    per_layer = contribution_data.get('per_layer', {})
    if not per_layer:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Extract data
    layers = sorted(per_layer.keys(), key=lambda x: int(x[1:]))
    attn_sums = [per_layer[l]['attention_sum'] for l in layers]
    know_sums = [per_layer[l]['knowledge_sum'] for l in layers]

    # Normalize
    max_val = max(max(attn_sums), max(know_sums)) if attn_sums else 1
    attn_norm = [v / max_val for v in attn_sums]
    know_norm = [v / max_val for v in know_sums]

    # Build matrix
    matrix = np.array([attn_norm, know_norm])

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.5), 3))

    im = ax.imshow(matrix, aspect='auto', cmap='Blues')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Attention', 'Knowledge'])
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([l[1:] for l in layers])
    ax.set_xlabel('Layer')
    ax.set_title('Normalized Circuit Activity by Layer')
    plt.colorbar(im, ax=ax, label='Normalized Activity')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return output_path
