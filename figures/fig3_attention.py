#!/usr/bin/env python3
"""
Figure 3: Attention Circuit with Q/K Shared Pool
Paper-style figure showing the attention routing mechanism.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
import numpy as np

# Style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8

# Colors
COLOR_Q = '#E74C3C'  # Red
COLOR_K = '#3498DB'  # Blue
COLOR_V = '#9B59B6'  # Purple
COLOR_BLACK = '#2C3E50'
COLOR_GRAY = '#7F8C8D'
COLOR_SHARED = '#E8EAF6'  # Light purple-blue for shared

def draw_box(ax, x, y, w, h, text, facecolor='white', edgecolor='black',
             fontsize=9, textcolor='black', lw=1.5, alpha=1.0):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.03",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=lw, alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color=textcolor)
    return box

def draw_arrow(ax, start, end, color='black', lw=1.2, style='-'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(start, end, arrowstyle='->', color=color,
                           linewidth=lw, mutation_scale=12, linestyle=style)
    ax.add_patch(arrow)
    return arrow

def main():
    fig, ax = plt.subplots(figsize=(5, 6), dpi=300)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # === Top: Input x ===
    draw_box(ax, 3.0, 7.3, 1.2, 0.5, 'Input x',
             facecolor='#F5F5F5', edgecolor=COLOR_BLACK, fontsize=9)

    # Three routing paths
    # Arrows from input
    draw_arrow(ax, (2.4, 7.05), (1.2, 6.5), color=COLOR_Q, lw=1.0)
    draw_arrow(ax, (3.0, 7.05), (3.0, 6.5), color=COLOR_K, lw=1.0)
    draw_arrow(ax, (3.6, 7.05), (4.8, 6.5), color=COLOR_V, lw=1.0)

    # Route boxes
    draw_box(ax, 1.2, 6.2, 1.0, 0.5, 'Route Q',
             facecolor='#FFEBEE', edgecolor=COLOR_Q, fontsize=8)
    draw_box(ax, 3.0, 6.2, 1.0, 0.5, 'Route K',
             facecolor='#E3F2FD', edgecolor=COLOR_K, fontsize=8)
    draw_box(ax, 4.8, 6.2, 1.0, 0.5, 'Route V',
             facecolor='#F3E5F5', edgecolor=COLOR_V, fontsize=8)

    # Weight labels
    ax.text(1.2, 5.75, '$w^Q$', fontsize=8, ha='center', color=COLOR_Q)
    ax.text(3.0, 5.75, '$w^K$', fontsize=8, ha='center', color=COLOR_K)
    ax.text(4.8, 5.75, '$w^V$', fontsize=8, ha='center', color=COLOR_V)

    # Arrows to pools
    draw_arrow(ax, (1.2, 5.95), (1.8, 5.1), color=COLOR_Q, lw=1.0)
    draw_arrow(ax, (3.0, 5.95), (2.4, 5.1), color=COLOR_K, lw=1.0)
    draw_arrow(ax, (4.8, 5.95), (4.8, 5.1), color=COLOR_V, lw=1.0)

    # === Shared Q/K Pool ===
    # Create gradient-like effect with two overlapping boxes
    shared_box = FancyBboxPatch((0.7, 4.2), 2.6, 1.6,
                                 boxstyle="round,pad=0.02,rounding_size=0.05",
                                 facecolor=COLOR_SHARED, edgecolor=COLOR_BLACK,
                                 linewidth=1.5)
    ax.add_patch(shared_box)

    # Add gradient stripe effect
    for i, (color, alpha) in enumerate([(COLOR_Q, 0.15), (COLOR_K, 0.15)]):
        stripe = Rectangle((0.7 + i * 1.3, 4.2), 1.3, 1.6,
                           facecolor=color, alpha=alpha, edgecolor='none')
        ax.add_patch(stripe)

    ax.text(2.0, 5.5, 'Shared Q/K Pool', fontsize=9, ha='center',
            fontweight='bold', color=COLOR_BLACK)
    ax.text(2.0, 5.0, '$F^{QK}$ (120)', fontsize=8, ha='center', color=COLOR_GRAY)
    ax.text(2.0, 4.6, '$R^{QK}$ (120)', fontsize=8, ha='center', color=COLOR_GRAY)

    # V Pool
    draw_box(ax, 4.8, 4.6, 1.4, 1.2, 'V Pool\n$F^V$ (24)\n$R^V$ (24)',
             facecolor='#F3E5F5', edgecolor=COLOR_V, fontsize=8, alpha=0.8)

    # Arrows from pools to Q, K, V
    draw_arrow(ax, (1.3, 4.2), (1.3, 3.5), color=COLOR_Q, lw=1.0)
    draw_arrow(ax, (2.7, 4.2), (2.7, 3.5), color=COLOR_K, lw=1.0)
    draw_arrow(ax, (4.8, 4.0), (4.8, 3.5), color=COLOR_V, lw=1.0)

    # Q, K, V outputs
    draw_box(ax, 1.3, 3.2, 0.8, 0.5, 'Q',
             facecolor=COLOR_Q, edgecolor=COLOR_Q, fontsize=10, textcolor='white')
    draw_box(ax, 2.7, 3.2, 0.8, 0.5, 'K',
             facecolor=COLOR_K, edgecolor=COLOR_K, fontsize=10, textcolor='white')
    draw_box(ax, 4.8, 3.2, 0.8, 0.5, 'V',
             facecolor=COLOR_V, edgecolor=COLOR_V, fontsize=10, textcolor='white')

    # Arrows to attention
    draw_arrow(ax, (1.3, 2.95), (2.5, 2.2), color=COLOR_Q, lw=1.0)
    draw_arrow(ax, (2.7, 2.95), (2.9, 2.2), color=COLOR_K, lw=1.0)
    draw_arrow(ax, (4.8, 2.95), (3.5, 2.2), color=COLOR_V, lw=1.0)

    # Multi-Head Attention
    draw_box(ax, 3.0, 1.8, 2.4, 0.7, 'Multi-Head Attention',
             facecolor='#FFF8E1', edgecolor='#F57C00', fontsize=9, lw=1.5)

    # Arrow to output
    draw_arrow(ax, (3.0, 1.45), (3.0, 0.9), color=COLOR_BLACK, lw=1.2)

    # Output
    draw_box(ax, 3.0, 0.6, 1.2, 0.5, 'Output',
             facecolor='#F5F5F5', edgecolor=COLOR_BLACK, fontsize=9)

    # === Legend ===
    ax.text(5.3, 7.5, 'Legend:', fontsize=8, fontweight='bold')

    legend_items = [
        (COLOR_Q, 'Q path'),
        (COLOR_K, 'K path'),
        (COLOR_V, 'V path'),
    ]

    for i, (color, label) in enumerate(legend_items):
        y = 7.1 - i * 0.35
        ax.add_patch(Rectangle((5.3, y - 0.1), 0.25, 0.2, facecolor=color, edgecolor='none'))
        ax.text(5.65, y, label, fontsize=7, va='center')

    # Annotation about shared pool
    ax.text(3.0, 0.1, 'Q and K share neuron pool but use\nindependent routing weights',
            fontsize=7, ha='center', color=COLOR_GRAY, style='italic')

    plt.tight_layout()
    plt.savefig('figures/fig3_attention.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig3_attention.png', format='png', dpi=300, bbox_inches='tight')
    print("Saved: figures/fig3_attention.pdf")
    plt.close()

if __name__ == '__main__':
    main()
