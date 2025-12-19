#!/usr/bin/env python3
"""
Figure 5: Training Loss Curve
Line plot comparing DAWN vs Vanilla training dynamics.

Usage:
    # With actual log files:
    python fig5_loss_curve.py --dawn-log path/to/dawn.log --vanilla-log path/to/vanilla.log

    # With demo data (for testing):
    python fig5_loss_curve.py --demo
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
from pathlib import Path

# Style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Colors
COLOR_DAWN = '#4A90D9'
COLOR_VANILLA_22M = '#7F8C8D'
COLOR_VANILLA_108M = '#2C3E50'
COLOR_GRAY = '#BDC3C7'


def parse_log_file(log_path):
    """
    Parse training log file to extract step and val_loss.

    Expected log format (from train.py):
        epoch=1,step=5000,val_loss=3.456789,val_acc=0.1234
        EPOCH,1,3.45,0.12,3.40,0.13,0.00055,1234.5

    Returns:
        steps: list of step numbers
        losses: list of validation losses
    """
    steps = []
    losses = []

    with open(log_path, 'r') as f:
        for line in f:
            # Match step-level validation logs
            match = re.search(r'step=(\d+),val_loss=([\d.]+)', line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                continue

            # Match epoch summary (EPOCH,epoch,train_loss,train_acc,val_loss,...)
            if line.startswith('EPOCH,'):
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    try:
                        # Approximate step from epoch (assuming ~100k steps/epoch)
                        epoch = int(parts[1])
                        val_loss = float(parts[4])
                        steps.append(epoch * 100000)
                        losses.append(val_loss)
                    except (ValueError, IndexError):
                        pass

    return steps, losses


def generate_demo_data():
    """Generate synthetic training curves for demonstration."""
    steps = np.arange(0, 102000, 2000)

    # DAWN: faster convergence, lower final loss
    dawn_loss = 4.5 * np.exp(-steps / 30000) + 2.1 + 0.05 * np.random.randn(len(steps))
    dawn_loss = np.maximum(dawn_loss, 2.1)

    # Vanilla-22M: slower convergence, higher final loss
    vanilla_22m_loss = 5.0 * np.exp(-steps / 40000) + 3.95 + 0.05 * np.random.randn(len(steps))
    vanilla_22m_loss = np.maximum(vanilla_22m_loss, 3.95)

    # Vanilla-108M: medium convergence
    vanilla_108m_loss = 4.8 * np.exp(-steps / 35000) + 3.45 + 0.05 * np.random.randn(len(steps))
    vanilla_108m_loss = np.maximum(vanilla_108m_loss, 3.45)

    return {
        'DAWN-24M': (steps.tolist(), dawn_loss.tolist()),
        'Vanilla-22M': (steps.tolist(), vanilla_22m_loss.tolist()),
        'Vanilla-108M': (steps.tolist(), vanilla_108m_loss.tolist()),
    }


def main():
    parser = argparse.ArgumentParser(description='Generate training loss curve figure')
    parser.add_argument('--dawn-log', type=str, help='Path to DAWN training log')
    parser.add_argument('--vanilla-22m-log', type=str, help='Path to Vanilla-22M training log')
    parser.add_argument('--vanilla-108m-log', type=str, help='Path to Vanilla-108M training log')
    parser.add_argument('--demo', action='store_true', help='Use demo data')
    parser.add_argument('--output', type=str, default='figures/fig5_loss_curve',
                       help='Output path (without extension)')
    args = parser.parse_args()

    # Load or generate data
    data = {}

    if args.demo:
        print("Using demo data...")
        data = generate_demo_data()
    else:
        if args.dawn_log and Path(args.dawn_log).exists():
            steps, losses = parse_log_file(args.dawn_log)
            if steps:
                data['DAWN-24M'] = (steps, losses)
                print(f"Loaded DAWN data: {len(steps)} points")

        if args.vanilla_22m_log and Path(args.vanilla_22m_log).exists():
            steps, losses = parse_log_file(args.vanilla_22m_log)
            if steps:
                data['Vanilla-22M'] = (steps, losses)
                print(f"Loaded Vanilla-22M data: {len(steps)} points")

        if args.vanilla_108m_log and Path(args.vanilla_108m_log).exists():
            steps, losses = parse_log_file(args.vanilla_108m_log)
            if steps:
                data['Vanilla-108M'] = (steps, losses)
                print(f"Loaded Vanilla-108M data: {len(steps)} points")

    if not data:
        print("No data available. Use --demo for demonstration or provide log files.")
        print("Example: python fig5_loss_curve.py --demo")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # Plot each model
    styles = {
        'DAWN-24M': {'color': COLOR_DAWN, 'linestyle': '-', 'linewidth': 2.5, 'label': 'DAWN-24M (Ours)'},
        'Vanilla-22M': {'color': COLOR_VANILLA_22M, 'linestyle': '--', 'linewidth': 2, 'label': 'Vanilla-22M'},
        'Vanilla-108M': {'color': COLOR_VANILLA_108M, 'linestyle': '-.', 'linewidth': 2, 'label': 'Vanilla-108M'},
    }

    for model_name, (steps, losses) in data.items():
        style = styles.get(model_name, {'color': 'gray', 'linestyle': '-', 'linewidth': 1.5, 'label': model_name})
        ax.plot(steps, losses, **style)

    # Formatting
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Validation Loss', fontsize=11)

    # Use log scale for y-axis if range is large
    loss_range = []
    for _, (_, losses) in data.items():
        loss_range.extend(losses)
    if max(loss_range) / min(loss_range) > 5:
        ax.set_yscale('log')

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Format x-axis with K notation
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x > 0 else '0'))

    # Legend
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

    # Annotations (if DAWN data exists)
    if 'DAWN-24M' in data and 'Vanilla-22M' in data:
        dawn_final = data['DAWN-24M'][1][-1]
        vanilla_final = data['Vanilla-22M'][1][-1]

        ax.annotate(f'DAWN achieves\n{vanilla_final/dawn_final:.1f}× lower loss',
                   xy=(data['DAWN-24M'][0][-1], dawn_final),
                   textcoords='offset points', xytext=(-80, 20),
                   fontsize=8, color=COLOR_DAWN,
                   arrowprops=dict(arrowstyle='->', color=COLOR_DAWN, lw=1))

    plt.tight_layout()

    # Save
    output_base = args.output
    plt.savefig(f'{output_base}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.png', format='png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_base}.pdf")
    plt.close()


if __name__ == '__main__':
    main()
