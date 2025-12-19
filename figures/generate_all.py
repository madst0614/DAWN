#!/usr/bin/env python3
"""
Generate All Paper Figures

Single command to generate all figures for the DAWN paper.

Usage:
    # Generate fig1-4 only:
    python figures/generate_all.py

    # Include fig5 with demo data:
    python figures/generate_all.py --demo

    # Include fig5 with actual log files:
    python figures/generate_all.py \\
        --dawn_log path/to/dawn/training_log.txt \\
        --vanilla_22m_log path/to/vanilla_22m/training_log.txt \\
        --vanilla_108m_log path/to/vanilla_108m/training_log.txt

    # Generate and zip for download:
    python figures/generate_all.py --demo --zip

Output:
    figures/fig1_architecture.pdf
    figures/fig2_feature_restore.pdf
    figures/fig3_param_efficiency.pdf
    figures/fig4_routing_stats.pdf
    figures/fig5_loss_curve.pdf (if --demo or log paths provided)
"""

import subprocess
import sys
import os
import zipfile
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
    if extra_args:
        print(f"Args: {' '.join(extra_args)}")
    print('='*60)

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def create_zip():
    """Create zip file of all generated figures."""
    zip_path = PROJECT_ROOT / 'dawn_figures.zip'

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for ext in ['*.png', '*.pdf']:
            for f in FIGURES_DIR.glob(ext):
                zf.write(f, f.name)

    print(f"\nCreated: {zip_path}")
    return zip_path


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate all paper figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic figures only (fig1-4):
    python figures/generate_all.py

    # All figures with demo loss curves:
    python figures/generate_all.py --demo

    # All figures with actual training logs:
    python figures/generate_all.py \\
        --dawn_log /content/drive/MyDrive/dawn/logs_v17.1/training_log.txt \\
        --vanilla_22m_log /content/drive/MyDrive/dawn/logs_baseline_22M/training_log.txt \\
        --vanilla_108m_log /content/drive/MyDrive/dawn/logs_baseline_125M/training_log.txt

    # Generate and zip:
    python figures/generate_all.py --demo --zip
        """
    )

    # Fig5 options - direct log file paths
    parser.add_argument('--dawn_log', type=str,
                       help='DAWN training_log.txt path (for fig5)')
    parser.add_argument('--vanilla_22m_log', type=str,
                       help='Vanilla-22M training_log.txt path (for fig5)')
    parser.add_argument('--vanilla_108m_log', type=str,
                       help='Vanilla-108M training_log.txt path (for fig5)')
    parser.add_argument('--demo', action='store_true',
                       help='Use demo data for fig5 loss curves')

    # General options
    parser.add_argument('--skip', nargs='+', default=[],
                       help='Skip specific figures (e.g., --skip fig1 fig2)')
    parser.add_argument('--zip', action='store_true',
                       help='Create zip file of all figures after generation')
    parser.add_argument('--no_annotations', action='store_true',
                       help='Disable annotations on fig5')

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
        fig5_args = []

        if args.demo:
            fig5_args.append('--demo')
        elif args.dawn_log or args.vanilla_22m_log or args.vanilla_108m_log:
            logs = []
            labels = []

            if args.dawn_log:
                logs.append(args.dawn_log)
                labels.append('DAWN-24M')
            if args.vanilla_22m_log:
                logs.append(args.vanilla_22m_log)
                labels.append('Vanilla-22M')
            if args.vanilla_108m_log:
                logs.append(args.vanilla_108m_log)
                labels.append('Vanilla-108M')

            if logs:
                fig5_args.extend(['--logs'] + logs)
                fig5_args.extend(['--labels'] + labels)

        if args.no_annotations:
            fig5_args.append('--no_annotations')

        if fig5_args:
            results['fig5'] = run_script('fig5_loss_curve.py', fig5_args)
        else:
            print("\n[fig5] Skipped: No --demo or log paths provided")

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

    # Create zip if requested
    if args.zip:
        zip_path = create_zip()
        print(f"\nTo download in Colab:")
        print(f"  from google.colab import files")
        print(f"  files.download('{zip_path}')")

    return 0 if success_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())
