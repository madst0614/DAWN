#!/usr/bin/env python3
"""
Generate All Paper Figures

Single command to generate all figures for the DAWN paper.

Usage:
    # Generate all figures (architecture, pathway, efficiency, routing):
    python figures/generate_all.py

    # Include training curves (requires checkpoint paths):
    python figures/generate_all.py \\
        --dawn_ckpt path/to/dawn/checkpoints \\
        --vanilla_ckpt path/to/vanilla/checkpoints

    # Demo mode for all figures including loss curve:
    python figures/generate_all.py --demo

Output:
    figures/fig1_architecture.pdf
    figures/fig2_feature_restore.pdf
    figures/fig3_param_efficiency.pdf
    figures/fig4_routing_stats.pdf
    figures/fig5_loss_curve.pdf (if --demo or checkpoint paths provided)
"""

import subprocess
import sys
import os
from pathlib import Path

# Get the figures directory
FIGURES_DIR = Path(__file__).parent
PROJECT_ROOT = FIGURES_DIR.parent


def run_script(script_name, extra_args=None):
    """Run a figure generation script."""
    script_path = FIGURES_DIR / script_name
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate all paper figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic figures only:
    python figures/generate_all.py

    # All figures with training curves:
    python figures/generate_all.py \\
        --dawn_ckpt /content/drive/MyDrive/dawn/checkpoints_v17.1_20M \\
        --vanilla_ckpt /content/drive/MyDrive/dawn/checkpoints_baseline_22M

    # Demo mode (synthetic data for loss curves):
    python figures/generate_all.py --demo
        """
    )

    parser.add_argument('--dawn_ckpt', type=str,
                       help='DAWN checkpoint directory (for fig5)')
    parser.add_argument('--vanilla_ckpt', type=str,
                       help='Vanilla checkpoint directory (for fig5)')
    parser.add_argument('--demo', action='store_true',
                       help='Use demo data for fig5 loss curves')
    parser.add_argument('--skip', nargs='+', default=[],
                       help='Skip specific figures (e.g., --skip fig1 fig2)')

    args = parser.parse_args()

    # Ensure figures directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = {}

    # Figure 1: Architecture
    if 'fig1' not in args.skip:
        results['fig1'] = run_script('fig1_architecture.py')

    # Figure 2: Feature-Restore Pathway
    if 'fig2' not in args.skip:
        results['fig2'] = run_script('fig2_feature_restore.py')

    # Figure 3: Parameter Efficiency
    if 'fig3' not in args.skip:
        results['fig3'] = run_script('fig3_param_efficiency.py')

    # Figure 4: Routing Statistics
    if 'fig4' not in args.skip:
        results['fig4'] = run_script('fig4_routing_stats.py')

    # Figure 5: Loss Curves (optional)
    if 'fig5' not in args.skip:
        if args.demo:
            results['fig5'] = run_script('fig5_loss_curve.py', ['--demo'])
        elif args.dawn_ckpt or args.vanilla_ckpt:
            fig5_args = []
            ckpts = []
            labels = []

            if args.dawn_ckpt:
                ckpts.append(args.dawn_ckpt)
                labels.append('DAWN-24M')
            if args.vanilla_ckpt:
                ckpts.append(args.vanilla_ckpt)
                labels.append('Vanilla-22M')

            if ckpts:
                fig5_args.extend(['--checkpoints'] + ckpts)
                fig5_args.extend(['--labels'] + labels)
                results['fig5'] = run_script('fig5_loss_curve.py', fig5_args)
        else:
            print("\n[fig5] Skipped: No checkpoint paths or --demo flag provided")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for fig, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {fig}")

    print(f"\nGenerated {success_count}/{total_count} figures")
    print(f"Output directory: {FIGURES_DIR}")

    # List generated files
    print("\nGenerated files:")
    for f in sorted(FIGURES_DIR.glob('*.pdf')):
        size = f.stat().st_size / 1024
        print(f"  {f.name} ({size:.1f} KB)")

    return 0 if success_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())
