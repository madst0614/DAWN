"""
Experiment 6: Learning Stability
목표: 매니폴드가 학습을 불안정하게 만들지 않는가?

분석 내용:
- Gradient norm 추적
- Loss variance 측정
- Neuron usage entropy (다양성)
- 학습 곡선 smoothness
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


def compute_gradient_norm(model):
    """Compute total gradient norm"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def analyze_neuron_usage(model, dataloader, device, layer_idx=0):
    """Analyze neuron usage distribution"""

    model.eval()
    neuron_counts = torch.zeros(model.d_ff, device=device)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)

            # Forward to get selected neurons
            batch_size, seq_len = input_ids.shape
            token_emb = model.token_embedding(input_ids)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = model.position_embedding(positions)
            x = token_emb + pos_emb

            # Get to FFN
            layer = model.layers[layer_idx]
            x_norm = layer.norm1(x)
            x_attn, _ = layer.attention(x_norm, x_norm, x_norm)
            x = x + layer.dropout(x_attn)

            x_ffn = layer.norm2(x)
            x_flat = x_ffn.view(-1, model.d_model)

            # Get selected neurons
            scores = layer.ffn.router.compute_scores(x_flat)
            top_k = model.sparse_k if model.sparse_k is not None else model.d_ff
            _, top_indices = torch.topk(scores, top_k, dim=-1)

            # Count usage
            for indices in top_indices:
                neuron_counts[indices] += 1

    # Compute entropy
    probs = neuron_counts / neuron_counts.sum()
    probs = probs[probs > 0]  # Remove zeros
    entropy = -(probs * torch.log(probs)).sum().item()

    # Normalize by max entropy
    max_entropy = np.log(model.d_ff)
    normalized_entropy = entropy / max_entropy

    usage_stats = {
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'usage_std': neuron_counts.std().item(),
        'usage_mean': neuron_counts.mean().item(),
        'num_used': (neuron_counts > 0).sum().item(),
        'usage_distribution': neuron_counts.cpu().numpy()
    }

    return usage_stats


def train_with_monitoring(model, train_loader, val_loader, device, num_epochs, model_name):
    """Train model while monitoring stability metrics"""

    print(f"\n{'='*80}")
    print(f"Training with monitoring: {model_name}")
    print(f"{'='*80}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    metrics = {
        'train_losses': [],
        'val_losses': [],
        'gradient_norms': [],
        'neuron_usage_entropy': [],
        'loss_variance': []
    }

    loss_window = []
    window_size = 10

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training
        model.train()
        epoch_losses = []
        epoch_grad_norms = []

        pbar = tqdm(train_loader, desc=f"Training {model_name}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']

            loss.backward()

            # Compute gradient norm before clipping
            grad_norm = compute_gradient_norm(model)
            epoch_grad_norms.append(grad_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            loss_window.append(loss.item())
            if len(loss_window) > window_size:
                loss_window.pop(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'grad_norm': f'{grad_norm:.2f}'
            })

        # Metrics for this epoch
        avg_train_loss = np.mean(epoch_losses)
        avg_grad_norm = np.mean(epoch_grad_norms)
        loss_variance = np.var(loss_window) if len(loss_window) >= window_size else 0

        metrics['train_losses'].append(avg_train_loss)
        metrics['gradient_norms'].append(avg_grad_norm)
        metrics['loss_variance'].append(loss_variance)

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                val_losses.append(outputs['loss'].item())

        avg_val_loss = np.mean(val_losses)
        metrics['val_losses'].append(avg_val_loss)

        # Neuron usage analysis
        usage_stats = analyze_neuron_usage(model, val_loader, device)
        metrics['neuron_usage_entropy'].append(usage_stats['normalized_entropy'])

        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Grad Norm: {avg_grad_norm:.2f}, Loss Var: {loss_variance:.4f}")
        print(f"Neuron Entropy: {usage_stats['normalized_entropy']:.4f}")

    return metrics


def plot_stability_comparison(metrics_baseline, metrics_manifold, save_dir):
    """Plot stability metrics comparison"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    epochs_b = range(1, len(metrics_baseline['train_losses']) + 1)
    epochs_m = range(1, len(metrics_manifold['train_losses']) + 1)

    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(epochs_b, metrics_baseline['train_losses'], marker='o', label='Baseline', alpha=0.7)
    ax.plot(epochs_m, metrics_manifold['train_losses'], marker='s', label='Manifold', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Gradient Norm
    ax = axes[0, 1]
    ax.plot(epochs_b, metrics_baseline['gradient_norms'], marker='o', label='Baseline', alpha=0.7)
    ax.plot(epochs_m, metrics_manifold['gradient_norms'], marker='s', label='Manifold', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Loss Variance
    ax = axes[0, 2]
    ax.plot(epochs_b, metrics_baseline['loss_variance'], marker='o', label='Baseline', alpha=0.7)
    ax.plot(epochs_m, metrics_manifold['loss_variance'], marker='s', label='Manifold', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Variance')
    ax.set_title('Loss Variance (Stability)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Validation Loss
    ax = axes[1, 0]
    ax.plot(epochs_b, metrics_baseline['val_losses'], marker='o', label='Baseline', alpha=0.7)
    ax.plot(epochs_m, metrics_manifold['val_losses'], marker='s', label='Manifold', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Neuron Usage Entropy
    ax = axes[1, 1]
    ax.plot(epochs_b, metrics_baseline['neuron_usage_entropy'], marker='o', label='Baseline', alpha=0.7)
    ax.plot(epochs_m, metrics_manifold['neuron_usage_entropy'], marker='s', label='Manifold', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Entropy')
    ax.set_title('Neuron Usage Diversity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary Comparison
    ax = axes[1, 2]
    metrics_names = ['Grad\nNorm', 'Loss\nVar', 'Neuron\nEntropy']
    baseline_vals = [
        np.mean(metrics_baseline['gradient_norms']),
        np.mean(metrics_baseline['loss_variance']),
        np.mean(metrics_baseline['neuron_usage_entropy'])
    ]
    manifold_vals = [
        np.mean(metrics_manifold['gradient_norms']),
        np.mean(metrics_manifold['loss_variance']),
        np.mean(metrics_manifold['neuron_usage_entropy'])
    ]

    # Normalize for comparison
    baseline_vals_norm = [v / max(b, m) for v, b, m in zip(baseline_vals, baseline_vals, manifold_vals)]
    manifold_vals_norm = [v / max(b, m) for v, b, m in zip(manifold_vals, baseline_vals, manifold_vals)]

    x = np.arange(len(metrics_names))
    width = 0.35

    ax.bar(x - width/2, baseline_vals_norm, width, label='Baseline', alpha=0.7)
    ax.bar(x + width/2, manifold_vals_norm, width, label='Manifold', alpha=0.7)

    ax.set_ylabel('Normalized Value')
    ax.set_title('Average Stability Metrics (Normalized)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'stability_comparison.png', dpi=150)
    print(f"\nPlot saved to {save_dir / 'stability_comparison.png'}")


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save directory
    save_dir = Path(__file__).parent / 'results' / 'exp6_stability'
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
        'sparse_k': 256,
        'batch_size': 32,
        'num_epochs': 10,
        'num_train_samples': 5000,
        'num_val_samples': 500
    }

    # Create datasets
    print("Creating datasets...")
    train_dataset = create_dummy_dataset(
        config['num_train_samples'],
        config['max_seq_len'],
        config['vocab_size']
    )
    val_dataset = create_dummy_dataset(
        config['num_val_samples'],
        config['max_seq_len'],
        config['vocab_size']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Create models
    print("Creating models...")

    baseline_model = BaselineModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        sparse_k=config['sparse_k']
    ).to(device)

    manifold_model = ManifoldModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        sparse_k=config['sparse_k'],
        use_manifold=True,
        manifold_d_hidden=64
    ).to(device)

    # Train baseline
    metrics_baseline = train_with_monitoring(
        baseline_model, train_loader, val_loader, device,
        config['num_epochs'], "Baseline"
    )

    # Train manifold
    metrics_manifold = train_with_monitoring(
        manifold_model, train_loader, val_loader, device,
        config['num_epochs'], "Manifold"
    )

    # Plot comparison
    plot_stability_comparison(metrics_baseline, metrics_manifold, save_dir)

    # Summary statistics
    print("\n" + "="*80)
    print("STABILITY SUMMARY")
    print("="*80)

    def print_stats(name, metrics):
        print(f"\n{name}:")
        print(f"  Avg Gradient Norm: {np.mean(metrics['gradient_norms']):.4f} "
              f"(std: {np.std(metrics['gradient_norms']):.4f})")
        print(f"  Avg Loss Variance: {np.mean(metrics['loss_variance']):.6f} "
              f"(std: {np.std(metrics['loss_variance']):.6f})")
        print(f"  Avg Neuron Entropy: {np.mean(metrics['neuron_usage_entropy']):.4f} "
              f"(std: {np.std(metrics['neuron_usage_entropy']):.4f})")
        print(f"  Final Val Loss: {metrics['val_losses'][-1]:.4f}")

    print_stats("Baseline", metrics_baseline)
    print_stats("Manifold", metrics_manifold)

    print("="*80)

    # Save results
    all_results = {
        'baseline': metrics_baseline,
        'manifold': metrics_manifold,
        'config': config
    }

    with open(save_dir / 'stability_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Experiment 6 completed!")
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
