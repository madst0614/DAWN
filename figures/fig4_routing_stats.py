#!/usr/bin/env python3
"""
Figure 4: Routing Statistics
Paper-style figure showing neuron utilization and layer-wise contribution.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np

# Style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Colors
COLOR_ATTENTION = '#4A90D9'
COLOR_ATTENTION_LIGHT = '#A8D0F0'
COLOR_KNOWLEDGE = '#50C878'
COLOR_KNOWLEDGE_LIGHT = '#A8E6C0'
COLOR_BLACK = '#2C3E50'
COLOR_GRAY = '#7F8C8D'

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), dpi=300)

    # === (a) Neuron Utilization ===
    utilization = {
        'Feature_Q': 54.2,
        'Feature_K': 81.7,
        'Feature_V': 91.7,
        'Restore_Q': 55.0,
        'Restore_K': 64.2,
        'Restore_V': 87.5,
        'Feature_Know': 87.5,
        'Restore_Know': 91.7,
    }

    # Reverse for bottom-to-top display
    pools = list(utilization.keys())[::-1]
    values = [utilization[p] for p in pools]

    # Colors based on type
    colors = []
    for p in pools:
        if 'Know' in p:
            colors.append(COLOR_KNOWLEDGE)
        elif 'Q' in p:
            colors.append('#E74C3C')  # Red for Q
        elif 'K' in p:
            colors.append('#3498DB')  # Blue for K
        else:
            colors.append('#9B59B6')  # Purple for V

    y_pos = np.arange(len(pools))

    # Create horizontal bar chart
    bars = ax1.barh(y_pos, values, height=0.7, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax1.text(val + 1, i, f'{val:.0f}%', va='center', fontsize=8, color=COLOR_BLACK)

    # Formatting
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(pools, fontsize=8)
    ax1.set_xlim(0, 105)
    ax1.set_xlabel('Active Neurons (%)', fontsize=9)
    ax1.set_title('(a) Neuron Utilization', fontsize=10, fontweight='bold', pad=10)

    # Add gridlines
    ax1.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)

    # Add vertical line at 50%
    ax1.axvline(x=50, color=COLOR_GRAY, linestyle=':', linewidth=1, alpha=0.7)

    # Legend for (a)
    legend_elements = [
        mpatches.Patch(color='#E74C3C', label='Q routing', alpha=0.85),
        mpatches.Patch(color='#3498DB', label='K routing', alpha=0.85),
        mpatches.Patch(color='#9B59B6', label='V routing', alpha=0.85),
        mpatches.Patch(color=COLOR_KNOWLEDGE, label='Knowledge', alpha=0.85),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0.9)

    # === (b) Layer-wise Attention Contribution ===
    layer_attn = [49, 59, 61, 62, 63, 63, 62, 61, 61, 61, 48, 46]
    layers = list(range(1, 13))

    # Plot line
    ax2.plot(layers, layer_attn, 'o-', color=COLOR_ATTENTION, linewidth=2,
             markersize=6, markerfacecolor='white', markeredgewidth=1.5)

    # Fill areas
    ax2.fill_between(layers, layer_attn, 50, where=[a >= 50 for a in layer_attn],
                     color=COLOR_ATTENTION, alpha=0.3, label='Attention dominant')
    ax2.fill_between(layers, layer_attn, 50, where=[a < 50 for a in layer_attn],
                     color=COLOR_KNOWLEDGE, alpha=0.3, label='Knowledge dominant')

    # 50% baseline
    ax2.axhline(y=50, color=COLOR_GRAY, linestyle='--', linewidth=1.5, label='50% baseline')

    # Annotations for interesting points
    ax2.annotate('Early: balanced', xy=(1, 49), xytext=(1.5, 42),
                fontsize=7, color=COLOR_GRAY,
                arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=0.8))
    ax2.annotate('Middle: attention\ndominant', xy=(6, 63), xytext=(7.5, 68),
                fontsize=7, color=COLOR_ATTENTION,
                arrowprops=dict(arrowstyle='->', color=COLOR_ATTENTION, lw=0.8))
    ax2.annotate('Late: knowledge\nincreases', xy=(12, 46), xytext=(9.5, 38),
                fontsize=7, color=COLOR_KNOWLEDGE,
                arrowprops=dict(arrowstyle='->', color=COLOR_KNOWLEDGE, lw=0.8))

    # Formatting
    ax2.set_xlim(0.5, 12.5)
    ax2.set_ylim(35, 75)
    ax2.set_xticks(layers)
    ax2.set_xlabel('Layer', fontsize=9)
    ax2.set_ylabel('Attention Contribution (%)', fontsize=9)
    ax2.set_title('(b) Layer-wise Circuit Contribution', fontsize=10, fontweight='bold', pad=10)

    # Add gridlines
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)

    # Legend
    legend_elements2 = [
        mpatches.Patch(color=COLOR_ATTENTION, alpha=0.3, label='Attention > 50%'),
        mpatches.Patch(color=COLOR_KNOWLEDGE, alpha=0.3, label='Knowledge > 50%'),
        plt.Line2D([0], [0], color=COLOR_GRAY, linestyle='--', label='50% baseline'),
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', fontsize=7, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('figures/fig4_routing_stats.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig4_routing_stats.png', format='png', dpi=300, bbox_inches='tight')
    print("Saved: figures/fig4_routing_stats.pdf")
    plt.close()

if __name__ == '__main__':
    main()
