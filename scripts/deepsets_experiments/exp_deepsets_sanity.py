"""
DeepSets Sanity Check Experiment

DeepSets FFN이 제대로 작동하는지 검증:
1. Forward/backward 작동 확인
2. Baseline FFN과 성능 비교
3. 학습 안정성 확인
4. Context 추가 효과
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
from models.neuron_based_deepsets import DeepSetsLanguageModel
from utils.data import create_dummy_dataset


def train_quick(model, dataloader, optimizer, device, num_steps, desc):
    """Quick training"""
    model.train()
    losses = []

    pbar = tqdm(dataloader, desc=desc, total=num_steps)
    for step, batch in enumerate(pbar):
        if step >= num_steps:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']

        loss.backward()

        # Check for NaN gradients
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"  ⚠️  NaN gradient in {name}")
                has_nan = True

        if has_nan:
            print("  ⚠️  Skipping step due to NaN")
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return losses


def eval_quick(model, dataloader, device, num_steps):
    """Quick evaluation"""
    model.eval()
    losses = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            losses.append(outputs['loss'].item())

    return np.mean(losses) if losses else float('inf')


def test_forward_backward(model, sample_input, sample_labels, device):
    """Test forward and backward passes"""
    print("\nTesting forward/backward...")

    # Forward
    try:
        outputs = model(input_ids=sample_input, labels=sample_labels)
        loss = outputs['loss']
        print(f"  ✓ Forward pass successful, loss: {loss.item():.4f}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False

    # Backward
    try:
        loss.backward()
        print(f"  ✓ Backward pass successful")

        # Check gradients
        has_grad = False
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                if torch.isnan(param.grad).any():
                    has_nan = True
                    print(f"    ⚠️  NaN in gradient of {name}")

        if has_grad and not has_nan:
            print(f"  ✓ Gradients computed successfully")
        elif has_nan:
            print(f"  ✗ NaN gradients detected")
            return False
        else:
            print(f"  ⚠️  No gradients found")
            return False

    except Exception as e:
        print(f"  ✗ Backward pass failed: {e}")
        return False

    return True


def compare_architectures(config, train_loader, val_loader, device, save_dir):
    """Compare Baseline vs DeepSets (basic) vs DeepSets (context)"""

    print("\n" + "="*80)
    print("ARCHITECTURE COMPARISON")
    print("="*80)

    models_config = [
        {
            'name': 'Baseline',
            'model_class': BaselineModel,
            'kwargs': {}
        },
        {
            'name': 'DeepSets-Basic',
            'model_class': DeepSetsLanguageModel,
            'kwargs': {
                'd_neuron': config['d_neuron'],
                'd_hidden': config['d_hidden'],
                'use_context': False
            }
        },
        {
            'name': 'DeepSets-Context',
            'model_class': DeepSetsLanguageModel,
            'kwargs': {
                'd_neuron': config['d_neuron'],
                'd_hidden': config['d_hidden'],
                'use_context': True
            }
        }
    ]

    results = []

    for model_cfg in models_config:
        print(f"\n{'='*80}")
        print(f"Testing: {model_cfg['name']}")
        print(f"{'='*80}")

        # Create model
        model = model_cfg['model_class'](
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout'],
            sparse_k=config['sparse_k'],
            **model_cfg['kwargs']
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {num_params:,}")

        # Forward/backward test
        sample_batch = next(iter(train_loader))
        sample_input = sample_batch['input_ids'].to(device)
        sample_labels = sample_batch['labels'].to(device)

        if not test_forward_backward(model, sample_input, sample_labels, device):
            print(f"✗ {model_cfg['name']} failed basic tests")
            continue

        # Training
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr']
        )

        train_losses = train_quick(
            model, train_loader, optimizer, device,
            config['num_train_steps'],
            f"Training {model_cfg['name']}"
        )

        # Evaluation
        val_loss = eval_quick(model, val_loader, device, config['num_eval_steps'])

        results.append({
            'name': model_cfg['name'],
            'num_params': num_params,
            'train_losses': train_losses,
            'final_train_loss': train_losses[-1] if train_losses else float('inf'),
            'val_loss': val_loss
        })

        print(f"\nResults for {model_cfg['name']}:")
        print(f"  Final train loss: {results[-1]['final_train_loss']:.4f}")
        print(f"  Val loss: {val_loss:.4f}")

    return results


def plot_results(results, save_dir):
    """Plot comparison results"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training curves
    ax = axes[0]
    for r in results:
        if r['train_losses']:
            ax.plot(r['train_losses'], label=r['name'], marker='o', alpha=0.7)

    ax.set_xlabel('Step')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Final comparison
    ax = axes[1]
    names = [r['name'] for r in results]
    train_losses = [r['final_train_loss'] for r in results]
    val_losses = [r['val_loss'] for r in results]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, train_losses, width, label='Final Train Loss', alpha=0.7)
    ax.bar(x + width/2, val_losses, width, label='Val Loss', alpha=0.7)

    ax.set_ylabel('Loss')
    ax.set_title('Final Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'deepsets_sanity_check.png', dpi=150)
    print(f"\nPlot saved to {save_dir / 'deepsets_sanity_check.png'}")


def main():
    print("="*80)
    print("DEEPSETS SANITY CHECK")
    print("="*80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Save directory
    save_dir = Path(__file__).parent / 'results' / 'exp_deepsets_sanity'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Small config for quick testing
    config = {
        'vocab_size': 5000,
        'd_model': 128,
        'd_ff': 512,
        'n_heads': 4,
        'n_layers': 2,
        'max_seq_len': 64,
        'dropout': 0.1,
        'sparse_k': 128,
        'd_neuron': 64,   # Neuron info vector size
        'd_hidden': 128,  # φ output size
        'batch_size': 16,
        'num_train_samples': 500,
        'num_val_samples': 100,
        'lr': 3e-4,
        'num_train_steps': 100,
        'num_eval_steps': 20
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Run comparison
    results = compare_architectures(config, train_loader, val_loader, device, save_dir)

    # Plot
    plot_results(results, save_dir)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Params':<15} {'Final Train':<15} {'Val Loss':<15}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<20} {r['num_params']:>14,} "
              f"{r['final_train_loss']:>14.4f} {r['val_loss']:>14.4f}")
    print("="*80)

    # Analysis
    print("\nAnalysis:")
    if len(results) >= 3:
        baseline = results[0]
        basic = results[1]
        context = results[2]

        print(f"\n1. DeepSets-Basic vs Baseline:")
        if basic['val_loss'] < baseline['val_loss'] * 1.1:
            print(f"   ✓ DeepSets-Basic achieves comparable performance")
            print(f"     ({basic['val_loss']:.4f} vs {baseline['val_loss']:.4f})")
        else:
            print(f"   ⚠️  DeepSets-Basic underperforms")
            print(f"     ({basic['val_loss']:.4f} vs {baseline['val_loss']:.4f})")

        print(f"\n2. Context Effect:")
        if context['val_loss'] < basic['val_loss']:
            improvement = (basic['val_loss'] - context['val_loss']) / basic['val_loss'] * 100
            print(f"   ✓ Context improves performance by {improvement:.1f}%")
        else:
            print(f"   → Context doesn't improve (may need more training)")

        print(f"\n3. Parameter Efficiency:")
        baseline_params = baseline['num_params']
        basic_params = basic['num_params']
        overhead = (basic_params - baseline_params) / baseline_params * 100
        print(f"   DeepSets adds {overhead:.1f}% more parameters")
        print(f"   ({basic_params - baseline_params:,} additional params)")

    # Save results
    results_to_save = {
        'config': config,
        'results': [
            {k: v for k, v in r.items() if k != 'train_losses'}
            for r in results
        ]
    }

    with open(save_dir / 'deepsets_sanity_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n✓ Sanity check completed!")
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
