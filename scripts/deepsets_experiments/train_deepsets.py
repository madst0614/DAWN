"""
DeepSets FFN Training Script

Baseline vs DeepSets-Basic vs DeepSets-Context 비교 학습

Usage:
    python train_deepsets.py --config small   # Quick test
    python train_deepsets.py --config medium  # Full training
    python train_deepsets.py --config large   # Large scale
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.neuron_based import NeuronBasedLanguageModel as BaselineModel
from models.neuron_based_deepsets import DeepSetsLanguageModel
from utils.data import create_dummy_dataset


# ============================================================
# Configurations
# ============================================================

CONFIGS = {
    'small': {
        'vocab_size': 10000,
        'd_model': 256,
        'd_ff': 1024,
        'n_heads': 8,
        'n_layers': 4,
        'max_seq_len': 128,
        'dropout': 0.1,
        'sparse_k': 256,
        'd_neuron': 64,
        'd_hidden': 128,
        'batch_size': 32,
        'num_epochs': 20,
        'lr': 3e-4,
        'num_train_samples': 10000,
        'num_val_samples': 1000,
        'description': 'Small config for quick testing'
    },
    'medium': {
        'vocab_size': 30000,
        'd_model': 512,
        'd_ff': 2048,
        'n_heads': 8,
        'n_layers': 6,
        'max_seq_len': 256,
        'dropout': 0.1,
        'sparse_k': 512,
        'd_neuron': 128,
        'd_hidden': 256,
        'batch_size': 32,
        'num_epochs': 30,
        'lr': 3e-4,
        'num_train_samples': 50000,
        'num_val_samples': 5000,
        'description': 'Medium config for full training'
    },
    'large': {
        'vocab_size': 50000,
        'd_model': 768,
        'd_ff': 3072,
        'n_heads': 12,
        'n_layers': 8,
        'max_seq_len': 512,
        'dropout': 0.1,
        'sparse_k': 1024,
        'd_neuron': 192,
        'd_hidden': 384,
        'batch_size': 16,
        'num_epochs': 50,
        'lr': 2e-4,
        'num_train_samples': 100000,
        'num_val_samples': 10000,
        'description': 'Large config for full-scale experiments'
    }
}


# ============================================================
# Training Functions
# ============================================================

def train_epoch(model, dataloader, optimizer, device, epoch, desc="Training"):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} - {desc}")
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
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
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


def train_model(model, train_loader, val_loader, config, device, model_name, save_dir):
    """Complete training loop"""

    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}\n")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['lr'] * 0.1
    )

    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_perplexities': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, model_name
        )
        history['train_losses'].append(train_loss)

        # Evaluate
        val_loss, val_ppl = evaluate(model, val_loader, device)
        history['val_losses'].append(val_loss)
        history['val_perplexities'].append(val_ppl)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_perplexity': val_ppl,
                'config': config
            }, save_dir / f"{model_name}_best.pt")

            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch} epochs")
                break

        # Update learning rate
        scheduler.step()

        # Save checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, save_dir / f"{model_name}_epoch{epoch}.pt")

    return history, best_val_loss


# ============================================================
# Plotting
# ============================================================

def plot_training_curves(results, save_dir):
    """Plot training curves for all models"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Training Loss
    ax = axes[0, 0]
    for name, history in results.items():
        epochs = range(1, len(history['train_losses']) + 1)
        ax.plot(epochs, history['train_losses'], marker='o', label=name, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for name, history in results.items():
        epochs = range(1, len(history['val_losses']) + 1)
        ax.plot(epochs, history['val_losses'], marker='s', label=name, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Validation Perplexity
    ax = axes[1, 0]
    for name, history in results.items():
        epochs = range(1, len(history['val_perplexities']) + 1)
        ax.plot(epochs, history['val_perplexities'], marker='^', label=name, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.set_title('Validation Perplexity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Learning Rates
    ax = axes[1, 1]
    for name, history in results.items():
        epochs = range(1, len(history['learning_rates']) + 1)
        ax.plot(epochs, history['learning_rates'], marker='d', label=name, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150)
    print(f"\nPlot saved to {save_dir / 'training_curves.png'}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train DeepSets FFN models')
    parser.add_argument('--config', type=str, default='small',
                        choices=['small', 'medium', 'large'],
                        help='Config to use')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['baseline', 'deepsets-basic', 'deepsets-context'],
                        help='Models to train')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Save directory')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Config
    config = CONFIGS[args.config]
    print(f"\nConfig: {args.config}")
    print(f"Description: {config['description']}")
    print("\nHyperparameters:")
    for key, value in config.items():
        if key != 'description':
            print(f"  {key}: {value}")

    # Save directory
    if args.save_dir is None:
        save_dir = Path(__file__).parent / 'results' / f'training_{args.config}'
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSave directory: {save_dir}")

    # Create datasets
    print("\nCreating datasets...")
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Train models
    results = {}
    best_losses = {}

    model_configs = []
    if 'baseline' in args.models:
        model_configs.append(('Baseline', BaselineModel, {}))
    if 'deepsets-basic' in args.models:
        model_configs.append(('DeepSets-Basic', DeepSetsLanguageModel, {
            'd_neuron': config['d_neuron'],
            'd_hidden': config['d_hidden'],
            'use_context': False
        }))
    if 'deepsets-context' in args.models:
        model_configs.append(('DeepSets-Context', DeepSetsLanguageModel, {
            'd_neuron': config['d_neuron'],
            'd_hidden': config['d_hidden'],
            'use_context': True
        }))

    for model_name, model_class, model_kwargs in model_configs:
        # Create model
        model = model_class(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout'],
            sparse_k=config['sparse_k'],
            **model_kwargs
        ).to(device)

        # Train
        history, best_val_loss = train_model(
            model, train_loader, val_loader, config, device, model_name, save_dir
        )

        results[model_name] = history
        best_losses[model_name] = best_val_loss

    # Plot results
    plot_training_curves(results, save_dir)

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Best Val Loss':<15} {'Final PPL':<15}")
    print("-"*80)
    for name in results.keys():
        final_ppl = results[name]['val_perplexities'][-1]
        print(f"{name:<20} {best_losses[name]:>14.4f} {final_ppl:>14.2f}")
    print("="*80)

    # Save results
    results_to_save = {
        'config': config,
        'results': {
            name: {
                'history': {
                    k: [float(v) for v in vals]
                    for k, vals in history.items()
                },
                'best_val_loss': float(best_losses[name])
            }
            for name, history in results.items()
        }
    }

    with open(save_dir / 'training_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n✓ Training completed!")
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
