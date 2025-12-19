#!/usr/bin/env python3
"""
Figure 3: Parameter Efficiency Comparison
Scatter plot showing DAWN's efficiency vs vanilla transformer scaling.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Colors
COLOR_VANILLA = '#7F8C8D'
COLOR_DAWN = '#E74C3C'
COLOR_BLACK = '#2C3E50'
COLOR_GRAY = '#BDC3C7'

def main():
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

    # Data
    models = {
        'Vanilla-22M': {'params': 22.6, 'ppl': 53.1},
        'Vanilla-54M': {'params': 53.7, 'ppl': 38.3},
        'Vanilla-108M': {'params': 108.9, 'ppl': 32.7},
        'DAWN-24M': {'params': 23.9, 'ppl': 8.4},
    }

    # Extract vanilla data for trend line
    vanilla_params = [models['Vanilla-22M']['params'],
                      models['Vanilla-54M']['params'],
                      models['Vanilla-108M']['params']]
    vanilla_ppl = [models['Vanilla-22M']['ppl'],
                   models['Vanilla-54M']['ppl'],
                   models['Vanilla-108M']['ppl']]

    # Plot vanilla points and connect with line
    ax.plot(vanilla_params, vanilla_ppl, 'o-', color=COLOR_VANILLA,
            markersize=10, linewidth=2, label='Vanilla Transformer', zorder=2)

    # Add vanilla labels
    for name, data in models.items():
        if 'Vanilla' in name:
            offset = (5, 8) if '22M' in name else (5, 5)
            ax.annotate(name.replace('Vanilla-', ''),
                       (data['params'], data['ppl']),
                       textcoords='offset points', xytext=offset,
                       fontsize=8, color=COLOR_VANILLA)

    # Extrapolate vanilla trend (power law fit)
    log_params = np.log(vanilla_params)
    log_ppl = np.log(vanilla_ppl)
    coeffs = np.polyfit(log_params, log_ppl, 1)

    # Extended trend line
    params_extended = np.linspace(15, 150, 100)
    ppl_trend = np.exp(coeffs[1]) * params_extended ** coeffs[0]
    ax.plot(params_extended, ppl_trend, '--', color=COLOR_GRAY,
            linewidth=1.5, alpha=0.7, label='Vanilla scaling trend', zorder=1)

    # Plot DAWN point (star marker, larger)
    dawn = models['DAWN-24M']
    ax.scatter([dawn['params']], [dawn['ppl']], marker='*', s=400,
               color=COLOR_DAWN, edgecolors='white', linewidths=1,
               label='DAWN (Ours)', zorder=3)

    # DAWN label
    ax.annotate('DAWN-24M', (dawn['params'], dawn['ppl']),
               textcoords='offset points', xytext=(10, -5),
               fontsize=9, fontweight='bold', color=COLOR_DAWN)

    # Annotation arrow showing improvement
    # From where vanilla would be at 24M params to DAWN
    vanilla_at_24M = np.exp(coeffs[1]) * 24 ** coeffs[0]

    ax.annotate('',
                xy=(dawn['params'], dawn['ppl']),
                xytext=(dawn['params'], vanilla_at_24M),
                arrowprops=dict(arrowstyle='->', color=COLOR_DAWN,
                              lw=2, shrinkA=5, shrinkB=5))

    # Text annotation
    improvement_text = f'{vanilla_at_24M / dawn["ppl"]:.1f}× better PPL\nat same params'
    ax.annotate(improvement_text,
               xy=(dawn['params'] + 2, (dawn['ppl'] + vanilla_at_24M) / 2),
               fontsize=8, color=COLOR_DAWN, fontweight='bold',
               va='center')

    # Add horizontal reference line from DAWN to show param savings
    # Find where vanilla achieves same PPL as DAWN
    # ppl = exp(b) * params^a => params = (ppl / exp(b))^(1/a)
    params_for_dawn_ppl = (dawn['ppl'] / np.exp(coeffs[1])) ** (1 / coeffs[0])

    if params_for_dawn_ppl > 150:  # Off the chart
        ax.annotate(f'Vanilla would need\n>{150:.0f}M params\nfor same PPL',
                   xy=(130, dawn['ppl']), fontsize=7, color=COLOR_GRAY,
                   ha='center', va='center', style='italic')
    else:
        ax.plot([dawn['params'], params_for_dawn_ppl], [dawn['ppl'], dawn['ppl']],
               ':', color=COLOR_GRAY, linewidth=1)
        ax.annotate(f'{params_for_dawn_ppl/dawn["params"]:.0f}× fewer params',
                   xy=((dawn['params'] + params_for_dawn_ppl)/2, dawn['ppl']),
                   textcoords='offset points', xytext=(0, -15),
                   fontsize=7, color=COLOR_GRAY, ha='center')

    # Formatting
    ax.set_xlabel('Parameters (M)', fontsize=11)
    ax.set_ylabel('Perplexity', fontsize=11)
    ax.set_xlim(15, 140)
    ax.set_ylim(0, 65)

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

    # Title (optional, often omitted in papers)
    # ax.set_title('Parameter Efficiency', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/fig3_param_efficiency.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig3_param_efficiency.png', format='png', dpi=300, bbox_inches='tight')
    print("Saved: figures/fig3_param_efficiency.pdf")
    plt.close()

if __name__ == '__main__':
    main()
