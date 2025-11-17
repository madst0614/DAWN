"""
Quick Validation Script

빠른 검증용 작은 실험
- 작은 모델로 DeepSets 작동 확인
- 간단한 데이터셋으로 학습/평가
- Baseline vs DeepSets 기본 비교

Usage:
    python quick_validation.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.neuron_based import NeuronBasedLanguageModel as BaselineModel
from models.neuron_based_deepsets import DeepSetsLanguageModel
from utils.data import create_dummy_dataset


def quick_train(model, dataloader, optimizer, device, num_steps=50):
    """Quick training loop"""
    model.train()
    losses = []

    print(f"Training {num_steps} steps...")
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")

    avg_loss = sum(losses) / len(losses)
    print(f"  Average Loss: {avg_loss:.4f}\n")

    return avg_loss


def quick_eval(model, dataloader, device, num_steps=10):
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

    avg_loss = sum(losses) / len(losses)
    return avg_loss


def main():
    print("="*80)
    print("QUICK VALIDATION: Baseline vs DeepSets")
    print("="*80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Small model config
    config = {
        'vocab_size': 5000,
        'd_model': 128,
        'd_ff': 512,
        'n_heads': 4,
        'n_layers': 2,
        'max_seq_len': 64,
        'dropout': 0.1,
        'sparse_k': 128,
        'd_neuron': 64,
        'd_hidden': 128,
        'batch_size': 16,
        'num_train_samples': 500,
        'num_val_samples': 100,
        'lr': 3e-4,
        'num_train_steps': 50,
        'num_eval_steps': 10
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

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")

    # Create models
    print("-" * 80)
    print("Creating Baseline Model...")
    print("-" * 80)

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

    baseline_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    print(f"Parameters: {baseline_params:,}\n")

    print("-" * 80)
    print("Creating DeepSets Model...")
    print("-" * 80)

    deepsets_model = DeepSetsLanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        sparse_k=config['sparse_k'],
        d_neuron=config['d_neuron'],
        d_hidden=config['d_hidden'],
        use_context=False
    ).to(device)

    deepsets_params = sum(p.numel() for p in deepsets_model.parameters() if p.requires_grad)
    print(f"Parameters: {deepsets_params:,}")
    print(f"Additional params: {deepsets_params - baseline_params:,} "
          f"({(deepsets_params - baseline_params) / baseline_params * 100:.2f}%)\n")

    # Test forward pass
    print("-" * 80)
    print("Testing forward passes...")
    print("-" * 80)

    sample_batch = next(iter(train_loader))
    sample_input = sample_batch['input_ids'].to(device)
    sample_labels = sample_batch['labels'].to(device)

    print("Baseline forward pass...")
    baseline_output = baseline_model(input_ids=sample_input, labels=sample_labels)
    print(f"  ✓ Output shape: {baseline_output['logits'].shape}")
    print(f"  ✓ Loss: {baseline_output['loss'].item():.4f}")

    print("\nDeepSets forward pass...")
    deepsets_output = deepsets_model(input_ids=sample_input, labels=sample_labels)
    print(f"  ✓ Output shape: {deepsets_output['logits'].shape}")
    print(f"  ✓ Loss: {deepsets_output['loss'].item():.4f}\n")

    # Quick training
    print("="*80)
    print("BASELINE TRAINING")
    print("="*80)
    baseline_optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=config['lr']
    )
    baseline_train_loss = quick_train(
        baseline_model, train_loader, baseline_optimizer, device,
        num_steps=config['num_train_steps']
    )

    print("="*80)
    print("DEEPSETS TRAINING")
    print("="*80)
    deepsets_optimizer = torch.optim.AdamW(
        deepsets_model.parameters(),
        lr=config['lr']
    )
    deepsets_train_loss = quick_train(
        deepsets_model, train_loader, deepsets_optimizer, device,
        num_steps=config['num_train_steps']
    )

    # Quick evaluation
    print("="*80)
    print("EVALUATION")
    print("="*80)

    print("Baseline evaluation...")
    baseline_val_loss = quick_eval(
        baseline_model, val_loader, device,
        num_steps=config['num_eval_steps']
    )
    print(f"  Validation Loss: {baseline_val_loss:.4f}\n")

    print("DeepSets evaluation...")
    deepsets_val_loss = quick_eval(
        deepsets_model, val_loader, device,
        num_steps=config['num_eval_steps']
    )
    print(f"  Validation Loss: {deepsets_val_loss:.4f}\n")

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Model':<15} {'Params':<15} {'Train Loss':<15} {'Val Loss':<15}")
    print("-"*80)
    print(f"{'Baseline':<15} {baseline_params:>14,} {baseline_train_loss:>14.4f} {baseline_val_loss:>14.4f}")
    print(f"{'DeepSets':<15} {deepsets_params:>14,} {deepsets_train_loss:>14.4f} {deepsets_val_loss:>14.4f}")
    print("="*80)

    # Test different sparsity levels
    print("\nTesting different sparsity levels...")
    print("-"*80)

    sparsity_tests = [64, 128, 256, 512]

    print(f"{'Top-K':<10} {'Baseline Loss':<20} {'DeepSets Loss':<20}")
    print("-"*80)

    for top_k in sparsity_tests:
        # Baseline
        baseline_model.eval()
        with torch.no_grad():
            bl_out = baseline_model(input_ids=sample_input, labels=sample_labels, top_k=top_k)
            bl_loss = bl_out['loss'].item()

        # DeepSets
        deepsets_model.eval()
        with torch.no_grad():
            ds_out = deepsets_model(input_ids=sample_input, labels=sample_labels, top_k=top_k)
            ds_loss = ds_out['loss'].item()

        print(f"{top_k:<10} {bl_loss:>19.4f} {ds_loss:>19.4f}")

    print("="*80)
    print("✓ Quick validation completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
