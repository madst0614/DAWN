#!/usr/bin/env python3
"""
Figure 1: DAWN Architecture Overview
Paper-style figure showing the overall DAWN architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# Style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8

# Colors
COLOR_ATTENTION = '#4A90D9'
COLOR_KNOWLEDGE = '#50C878'
COLOR_POOLS = '#F5A623'
COLOR_ROUTER = '#9B59B6'
COLOR_BLACK = '#2C3E50'
COLOR_GRAY = '#7F8C8D'
COLOR_BG = '#F8F9FA'

def draw_box(ax, x, y, w, h, text, facecolor='white', edgecolor='black',
             fontsize=9, textcolor='black', lw=1.5, alpha=1.0, rounded=True):
    """Draw a rounded box with centered text."""
    if rounded:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                             facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, alpha=alpha)
    else:
        box = FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.02",
                             facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=textcolor, fontweight='normal')
    return box

def draw_arrow(ax, start, end, color='black', style='-', lw=1.2, arrow_style='->',
               connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(start, end, arrowstyle=arrow_style, color=color,
                           linestyle=style, linewidth=lw,
                           connectionstyle=connectionstyle,
                           mutation_scale=12)
    ax.add_patch(arrow)
    return arrow

def main():
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_aspect('equal')

    # === Left side: Main flow ===

    # Input Embedding
    draw_box(ax, 0.5, 6.8, 2.0, 0.5, 'Input Embedding',
             facecolor='#E8E8E8', edgecolor=COLOR_BLACK, fontsize=8)

    # Arrow to DAWN Block
    draw_arrow(ax, (1.5, 6.8), (1.5, 6.4), color=COLOR_BLACK)

    # DAWN Block (main container)
    main_block = FancyBboxPatch((0.3, 2.2), 3.4, 4.1, boxstyle="round,pad=0.02,rounding_size=0.05",
                                 facecolor='white', edgecolor=COLOR_BLACK, linewidth=2)
    ax.add_patch(main_block)
    ax.text(2.0, 6.1, 'DAWN Block (×12)', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLOR_BLACK)

    # Inner block
    inner_y = 2.5
    inner_h = 3.3

    # LayerNorm -> Attention Circuit
    draw_box(ax, 0.5, 5.2, 1.2, 0.4, 'LayerNorm', facecolor='#F5F5F5',
             edgecolor=COLOR_GRAY, fontsize=7, lw=0.8)
    draw_arrow(ax, (1.7, 5.4), (2.0, 5.4), color=COLOR_BLACK, lw=1.0)
    draw_box(ax, 2.0, 5.0, 1.5, 0.8, 'Attention\nCircuit',
             facecolor=COLOR_ATTENTION, edgecolor=COLOR_ATTENTION,
             fontsize=8, textcolor='white', alpha=0.9)

    # Residual notation
    ax.text(0.55, 4.7, '+', fontsize=10, ha='center', va='center', color=COLOR_GRAY)
    draw_arrow(ax, (1.1, 5.0), (1.1, 4.4), color=COLOR_GRAY, lw=0.8)

    # LayerNorm -> Knowledge Circuit
    draw_box(ax, 0.5, 3.8, 1.2, 0.4, 'LayerNorm', facecolor='#F5F5F5',
             edgecolor=COLOR_GRAY, fontsize=7, lw=0.8)
    draw_arrow(ax, (1.7, 4.0), (2.0, 4.0), color=COLOR_BLACK, lw=1.0)
    draw_box(ax, 2.0, 3.6, 1.5, 0.8, 'Knowledge\nCircuit',
             facecolor=COLOR_KNOWLEDGE, edgecolor=COLOR_KNOWLEDGE,
             fontsize=8, textcolor='white', alpha=0.9)

    # Residual notation
    ax.text(0.55, 3.3, '+', fontsize=10, ha='center', va='center', color=COLOR_GRAY)
    draw_arrow(ax, (1.1, 3.6), (1.1, 2.6), color=COLOR_GRAY, lw=0.8)

    # Arrow from block
    draw_arrow(ax, (1.5, 2.2), (1.5, 1.8), color=COLOR_BLACK)

    # Final LayerNorm
    draw_box(ax, 0.5, 1.3, 2.0, 0.4, 'LayerNorm', facecolor='#F5F5F5',
             edgecolor=COLOR_GRAY, fontsize=8, lw=0.8)
    draw_arrow(ax, (1.5, 1.3), (1.5, 0.9), color=COLOR_BLACK)

    # LM Head
    draw_box(ax, 0.5, 0.4, 2.0, 0.4, 'LM Head',
             facecolor='#E8E8E8', edgecolor=COLOR_BLACK, fontsize=8)

    # === Right side: Shared Neuron Pools ===

    # Dotted arrows from circuits to pools
    draw_arrow(ax, (3.5, 5.4), (5.2, 5.2), color=COLOR_ATTENTION, style='--', lw=1.0)
    draw_arrow(ax, (3.5, 4.0), (5.2, 3.2), color=COLOR_KNOWLEDGE, style='--', lw=1.0)

    # Shared Neuron Pools box
    pool_box = FancyBboxPatch((5.2, 2.0), 2.8, 4.5, boxstyle="round,pad=0.02,rounding_size=0.05",
                               facecolor='#FFF8E7', edgecolor=COLOR_POOLS, linewidth=2)
    ax.add_patch(pool_box)
    ax.text(6.6, 6.25, 'Shared Neuron Pools', ha='center', va='center',
            fontsize=9, fontweight='bold', color=COLOR_POOLS)

    # Attention pools
    ax.text(5.4, 5.7, 'Attention:', fontsize=8, fontweight='bold', color=COLOR_ATTENTION)
    pools_attn = [
        ('Feature_QK', '120'),
        ('Feature_V', '24'),
        ('Restore_QK', '120'),
        ('Restore_V', '24'),
    ]
    for i, (name, size) in enumerate(pools_attn):
        y_pos = 5.35 - i * 0.35
        ax.text(5.5, y_pos, f'{name}', fontsize=7, color=COLOR_BLACK)
        ax.text(7.6, y_pos, f'({size})', fontsize=7, color=COLOR_GRAY, ha='right')

    # Separator line
    ax.plot([5.4, 7.8], [3.85, 3.85], color=COLOR_GRAY, linewidth=0.5, linestyle='-')

    # Knowledge pools
    ax.text(5.4, 3.6, 'Knowledge:', fontsize=8, fontweight='bold', color=COLOR_KNOWLEDGE)
    pools_know = [
        ('Feature_K', '24'),
        ('Restore_K', '24'),
    ]
    for i, (name, size) in enumerate(pools_know):
        y_pos = 3.25 - i * 0.35
        ax.text(5.5, y_pos, f'{name}', fontsize=7, color=COLOR_BLACK)
        ax.text(7.6, y_pos, f'({size})', fontsize=7, color=COLOR_GRAY, ha='right')

    # === Global Router ===

    # Router box
    draw_box(ax, 5.5, 0.5, 2.2, 0.9, 'Global Router\n(SSM + top-k)',
             facecolor='#F3E5F5', edgecolor=COLOR_ROUTER, fontsize=8, lw=1.5)

    # Arrow from router to pools
    draw_arrow(ax, (6.6, 1.4), (6.6, 2.0), color=COLOR_ROUTER, style='--', lw=1.2)
    ax.text(6.9, 1.7, 'routing\nweights', fontsize=6, color=COLOR_ROUTER, va='center')

    # Legend
    legend_y = 0.2
    legend_elements = [
        (COLOR_ATTENTION, 'Attention Circuit'),
        (COLOR_KNOWLEDGE, 'Knowledge Circuit'),
        (COLOR_POOLS, 'Neuron Pools'),
        (COLOR_ROUTER, 'Router'),
    ]

    ax.text(8.5, 7.5, 'Legend:', fontsize=8, fontweight='bold', color=COLOR_BLACK)
    for i, (color, label) in enumerate(legend_elements):
        y = 7.1 - i * 0.4
        ax.add_patch(plt.Rectangle((8.5, y-0.1), 0.3, 0.25, facecolor=color, edgecolor='none'))
        ax.text(8.95, y, label, fontsize=7, va='center', color=COLOR_BLACK)

    # Line styles
    ax.text(8.5, 5.5, 'Arrows:', fontsize=8, fontweight='bold', color=COLOR_BLACK)
    ax.plot([8.5, 9.0], [5.2, 5.2], 'k-', linewidth=1.2)
    ax.text(9.1, 5.2, 'data', fontsize=7, va='center')
    ax.plot([8.5, 9.0], [4.9, 4.9], 'k--', linewidth=1.2)
    ax.text(9.1, 4.9, 'routing', fontsize=7, va='center')

    plt.tight_layout()
    plt.savefig('figures/fig1_architecture.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig1_architecture.png', format='png', dpi=300, bbox_inches='tight')
    print("Saved: figures/fig1_architecture.pdf")
    plt.close()

if __name__ == '__main__':
    main()
