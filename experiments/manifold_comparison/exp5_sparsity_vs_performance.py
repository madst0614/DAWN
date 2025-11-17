"""
Experiment 5: Sparsity vs Performance
목표: 다양한 top_k에서 매니폴드 효과

분석 내용:
- 다양한 sparsity level에서 성능 측정
- Manifold가 매우 sparse할 때 더 유리한지 확인
- Dense일 때는 비슷한지 확인
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.neuron_based import NeuronBasedLanguageModel as BaselineModel
from models.neuron_based_manifold import NeuronBasedLanguageModel as ManifoldModel
from utils.data import create_dummy_dataset
from torch.utils.data import DataLoader


def evaluate_at_sparsity(model, dataloader, device, top_k):
    """Evaluate model at specific sparsity level"""

    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating top_k={top_k}", leave=False):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels, top_k=top_k)
            loss = outputs['loss']

            batch_size, seq_len = input_ids.shape
            num_tokens = batch_size * seq_len
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return avg_loss, perplexity


def run_sparsity_sweep(model, dataloader, device, sparsity_levels, model_name):
    """Run evaluation across different sparsity levels"""

    print(f"\n{'='*80}")
    print(f"Sparsity Sweep: {model_name}")
    print(f"{'='*80}")

    results = []

    for top_k in sparsity_levels:
        print(f"\nEvaluating at top_k={top_k} ({top_k/model.d_ff*100:.1f}% of neurons)")

        loss, ppl = evaluate_at_sparsity(model, dataloader, device, top_k)

        result = {
            'top_k': top_k,
            'sparsity_ratio': top_k / model.d_ff,
            'loss': loss,
            'perplexity': ppl
        }
        results.append(result)

        print(f"  Loss: {loss:.4f}, Perplexity: {ppl:.2f}")

    return results


def plot_sparsity_comparison(results_baseline, results_manifold, d_ff, save_dir):
    """Plot performance vs sparsity comparison"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Extract data
    top_ks_b = [r['top_k'] for r in results_baseline]
    ppls_b = [r['perplexity'] for r in results_baseline]

    top_ks_m = [r['top_k'] for r in results_manifold]
    ppls_m = [r['perplexity'] for r in results_manifold]

    # Plot 1: Perplexity vs Top-K
    ax = axes[0]
    ax.plot(top_ks_b, ppls_b, marker='o', label='Baseline', linewidth=2)
    ax.plot(top_ks_m, ppls_m, marker='s', label='Manifold', linewidth=2)
    ax.set_xlabel('Top-K (Number of Neurons)')
    ax.set_ylabel('Perplexity')
    ax.set_title('Performance vs Sparsity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Perplexity vs Sparsity Ratio
    ax = axes[1]
    ratios_b = [r['sparsity_ratio'] * 100 for r in results_baseline]
    ratios_m = [r['sparsity_ratio'] * 100 for r in results_manifold]

    ax.plot(ratios_b, ppls_b, marker='o', label='Baseline', linewidth=2)
    ax.plot(ratios_m, ppls_m, marker='s', label='Manifold', linewidth=2)
    ax.set_xlabel('Sparsity Ratio (%)')
    ax.set_ylabel('Perplexity')
    ax.set_title('Performance vs Sparsity Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'sparsity_comparison.png', dpi=150)
    print(f"\nPlot saved to {save_dir / 'sparsity_comparison.png'}")

    # Compute improvements
    print("\n" + "="*80)
    print("MANIFOLD IMPROVEMENT AT EACH SPARSITY LEVEL")
    print("="*80)
    print(f"{'Top-K':<10} {'Sparsity %':<12} {'Baseline PPL':<15} {'Manifold PPL':<15} {'Improvement':<12}")
    print("-"*80)

    for rb, rm in zip(results_baseline, results_manifold):
        improvement = ((rb['perplexity'] - rm['perplexity']) / rb['perplexity']) * 100
        print(f"{rb['top_k']:<10} {rb['sparsity_ratio']*100:>11.1f} "
              f"{rb['perplexity']:>14.2f} {rm['perplexity']:>14.2f} "
              f"{improvement:>11.2f}%")

    print("="*80)


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save directory
    save_dir = Path(__file__).parent / 'results' / 'exp5_sparsity'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Model config
    config = {
        'vocab_size': 10000,
        'd_model': 256,
        'd_ff': 1024,
        'n_heads': 8,
        'n_layers': 4,
        'max_seq_len': 128,
        'dropout': 0.1,
        'batch_size': 32,
        'num_eval_samples': 1000
    }

    # Sparsity levels to test
    # From very sparse (3%) to dense (100%)
    sparsity_levels = [32, 64, 128, 256, 512, 768, 1024]  # Last one is dense

    # Create test dataset
    print("Creating test dataset...")
    test_dataset = create_dummy_dataset(
        config['num_eval_samples'],
        config['max_seq_len'],
        config['vocab_size']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Create models
    print("Creating models...")

    baseline_model = BaselineModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    ).to(device)

    manifold_model = ManifoldModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        use_manifold=True,
        manifold_d_hidden=64
    ).to(device)

    # Load trained models if available
    exp1_results_dir = Path(__file__).parent / 'results' / 'exp1_performance'
    baseline_ckpt = exp1_results_dir / 'Baseline_best.pt'
    manifold_ckpt = exp1_results_dir / 'Manifold_best.pt'

    if baseline_ckpt.exists():
        print(f"Loading baseline from {baseline_ckpt}")
        ckpt = torch.load(baseline_ckpt, map_location=device)
        baseline_model.load_state_dict(ckpt['model_state_dict'])

    if manifold_ckpt.exists():
        print(f"Loading manifold from {manifold_ckpt}")
        ckpt = torch.load(manifold_ckpt, map_location=device)
        manifold_model.load_state_dict(ckpt['model_state_dict'])

    # Run sparsity sweeps
    results_baseline = run_sparsity_sweep(
        baseline_model, test_loader, device, sparsity_levels, "Baseline"
    )

    results_manifold = run_sparsity_sweep(
        manifold_model, test_loader, device, sparsity_levels, "Manifold"
    )

    # Plot comparison
    plot_sparsity_comparison(
        results_baseline, results_manifold, config['d_ff'], save_dir
    )

    # Save results
    all_results = {
        'baseline': results_baseline,
        'manifold': results_manifold,
        'config': config,
        'sparsity_levels': sparsity_levels
    }

    with open(save_dir / 'sparsity_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Experiment 5 completed!")
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
