"""
Experiment 1: Basic Performance Comparison
목표: 매니폴드가 성능을 해치지 않는지 확인

측정 지표:
- Training loss curve
- Validation perplexity
- 수렴 속도
- 최종 성능
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.neuron_based import NeuronBasedLanguageModel as BaselineModel
from models.neuron_based_manifold import NeuronBasedLanguageModel as ManifoldModel
from utils.data import create_dummy_dataset


def train_epoch(model, dataloader, optimizer, device, desc="Training"):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0

    pbar = tqdm(dataloader, desc=desc)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size, seq_len = input_ids.shape
        num_tokens = batch_size * seq_len
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / total_tokens


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']

            batch_size, seq_len = input_ids.shape
            num_tokens = batch_size * seq_len
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return avg_loss, perplexity


def run_experiment(config, train_loader, val_loader, device, save_dir):
    """Run training experiment with given config"""

    print(f"\n{'='*60}")
    print(f"Running experiment: {config['name']}")
    print(f"{'='*60}")

    # Create model
    if config['use_manifold']:
        model = ManifoldModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout'],
            sparse_k=config.get('sparse_k'),
            use_manifold=True,
            manifold_d_hidden=config.get('manifold_d_hidden', 64)
        ).to(device)
    else:
        model = BaselineModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout'],
            sparse_k=config.get('sparse_k')
        ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )

    # Training loop
    train_losses = []
    val_losses = []
    val_perplexities = []

    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            desc=f"Epoch {epoch+1} - {config['name']}"
        )
        train_losses.append(train_loss)

        # Evaluate
        val_loss, val_ppl = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        val_perplexities.append(val_ppl)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_perplexity': val_ppl,
                'config': config
            }, save_dir / f"{config['name']}_best.pt")

    # Save results
    results = {
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_perplexities': val_perplexities,
        'best_val_loss': best_val_loss,
        'final_val_perplexity': val_perplexities[-1],
        'num_params': num_params
    }

    with open(save_dir / f"{config['name']}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def plot_comparison(results_list, save_dir):
    """Plot comparison between baseline and manifold"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Training Loss
    ax = axes[0, 0]
    for results in results_list:
        epochs = range(1, len(results['train_losses']) + 1)
        ax.plot(epochs, results['train_losses'],
                marker='o', label=results['config']['name'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for results in results_list:
        epochs = range(1, len(results['val_losses']) + 1)
        ax.plot(epochs, results['val_losses'],
                marker='s', label=results['config']['name'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Validation Perplexity
    ax = axes[1, 0]
    for results in results_list:
        epochs = range(1, len(results['val_perplexities']) + 1)
        ax.plot(epochs, results['val_perplexities'],
                marker='^', label=results['config']['name'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.set_title('Validation Perplexity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Final Comparison Bar Chart
    ax = axes[1, 1]
    names = [r['config']['name'] for r in results_list]
    final_ppls = [r['final_val_perplexity'] for r in results_list]

    bars = ax.bar(names, final_ppls, alpha=0.7)
    ax.set_ylabel('Final Validation Perplexity')
    ax.set_title('Final Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, ppl in zip(bars, final_ppls):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ppl:.2f}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_dir / 'performance_comparison.png', dpi=150)
    print(f"\nPlot saved to {save_dir / 'performance_comparison.png'}")

    # Summary table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Params':<15} {'Best Val Loss':<15} {'Final PPL':<15}")
    print("-"*80)
    for results in results_list:
        print(f"{results['config']['name']:<20} "
              f"{results['num_params']:>14,} "
              f"{results['best_val_loss']:>14.4f} "
              f"{results['final_val_perplexity']:>14.2f}")
    print("="*80)


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save directory
    save_dir = Path(__file__).parent / 'results' / 'exp1_performance'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Experiment configurations
    base_config = {
        'vocab_size': 10000,
        'd_model': 256,
        'd_ff': 1024,
        'n_heads': 8,
        'n_layers': 4,
        'max_seq_len': 128,
        'dropout': 0.1,
        'sparse_k': 256,  # 25% sparsity
        'num_epochs': 20,
        'batch_size': 32,
        'lr': 3e-4,
        'num_train_samples': 10000,
        'num_val_samples': 1000
    }

    configs = [
        {**base_config, 'use_manifold': False, 'name': 'Baseline'},
        {**base_config, 'use_manifold': True, 'name': 'Manifold', 'manifold_d_hidden': 64}
    ]

    # Create datasets
    print("Creating datasets...")
    train_dataset = create_dummy_dataset(
        num_samples=base_config['num_train_samples'],
        seq_len=base_config['max_seq_len'],
        vocab_size=base_config['vocab_size']
    )
    val_dataset = create_dummy_dataset(
        num_samples=base_config['num_val_samples'],
        seq_len=base_config['max_seq_len'],
        vocab_size=base_config['vocab_size']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=base_config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=base_config['batch_size'],
        shuffle=False,
        num_workers=2
    )

    # Run experiments
    results_list = []
    for config in configs:
        results = run_experiment(config, train_loader, val_loader, device, save_dir)
        results_list.append(results)

    # Plot comparison
    plot_comparison(results_list, save_dir)

    print("\n✓ Experiment 1 completed!")
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
