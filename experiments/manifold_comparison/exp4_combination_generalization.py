"""
Experiment 4: Combination Generalization
목표: 안 본 뉴런 조합도 잘 처리하는가?

분석 내용:
- 학습 중 뉴런 조합 빈도 추적
- 테스트에서 드문 조합 찾기
- 드문 조합에서의 성능 vs 흔한 조합에서의 성능
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.neuron_based import NeuronBasedLanguageModel as BaselineModel
from models.neuron_based_manifold import NeuronBasedLanguageModel as ManifoldModel
from utils.data import create_dummy_dataset
from torch.utils.data import DataLoader


def track_neuron_combinations(model, dataloader, device, max_batches=None):
    """Track frequency of neuron combinations during inference"""

    model.eval()
    combination_counter = Counter()
    all_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Tracking combinations")):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            batch_size, seq_len = input_ids.shape
            token_emb = model.token_embedding(input_ids)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = model.position_embedding(positions)
            x = token_emb + pos_emb

            # Track combinations in first FFN layer
            layer = model.layers[0]

            # Attention
            x_norm = layer.norm1(x)
            x_attn, _ = layer.attention(x_norm, x_norm, x_norm)
            x = x + layer.dropout(x_attn)

            # FFN input
            x_ffn = layer.norm2(x)
            x_flat = x_ffn.view(-1, model.d_model)

            # Get selected neurons
            scores = layer.ffn.router.compute_scores(x_flat)
            top_k = model.sparse_k if model.sparse_k is not None else model.d_ff
            _, top_indices = torch.topk(scores, top_k, dim=-1)

            # Record combinations (as sorted tuples)
            for indices in top_indices.cpu().numpy():
                combo = tuple(sorted(indices))
                combination_counter[combo] += 1

            # Compute loss for this batch
            outputs = model(input_ids=input_ids, labels=labels)
            all_losses.append(outputs['loss'].item())

    return combination_counter, all_losses


def evaluate_on_combinations(model, dataloader, device, target_combinations):
    """Evaluate model on specific neuron combinations"""

    model.eval()
    losses_by_combo = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating combinations"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward to get combinations
            batch_size, seq_len = input_ids.shape
            token_emb = model.token_embedding(input_ids)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = model.position_embedding(positions)
            x = token_emb + pos_emb

            layer = model.layers[0]
            x_norm = layer.norm1(x)
            x_attn, _ = layer.attention(x_norm, x_norm, x_norm)
            x = x + layer.dropout(x_attn)

            x_ffn = layer.norm2(x)
            x_flat = x_ffn.view(-1, model.d_model)

            scores = layer.ffn.router.compute_scores(x_flat)
            top_k = model.sparse_k if model.sparse_k is not None else model.d_ff
            _, top_indices = torch.topk(scores, top_k, dim=-1)

            # Full forward for loss
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss'].item()

            # Match combinations
            for idx, indices in enumerate(top_indices.cpu().numpy()):
                combo = tuple(sorted(indices))
                if combo in target_combinations:
                    losses_by_combo[combo].append(loss)

    # Average losses
    avg_losses = {combo: np.mean(losses) for combo, losses in losses_by_combo.items()}

    return avg_losses


def analyze_combination_generalization(model, train_loader, test_loader, device, model_name):
    """Analyze how well model generalizes to rare combinations"""

    print(f"\n{'='*80}")
    print(f"Analyzing {model_name}")
    print(f"{'='*80}")

    # Track training combinations
    print("Tracking training combinations...")
    train_combinations, train_losses = track_neuron_combinations(
        model, train_loader, device, max_batches=50
    )

    # Track test combinations
    print("Tracking test combinations...")
    test_combinations, test_losses = track_neuron_combinations(
        model, test_loader, device, max_batches=20
    )

    # Identify rare vs common combinations
    threshold_rare = 3
    threshold_common = 10

    rare_combos = set([combo for combo, count in train_combinations.items()
                       if count < threshold_rare])
    common_combos = set([combo for combo, count in train_combinations.items()
                         if count >= threshold_common])

    # Find test combinations that are rare in training
    test_rare = set(test_combinations.keys()) & rare_combos
    test_common = set(test_combinations.keys()) & common_combos

    print(f"\nTotal unique training combinations: {len(train_combinations)}")
    print(f"Rare combinations (< {threshold_rare}): {len(rare_combos)}")
    print(f"Common combinations (>= {threshold_common}): {len(common_combos)}")
    print(f"\nTest combinations matching rare: {len(test_rare)}")
    print(f"Test combinations matching common: {len(test_common)}")

    # Evaluate on rare vs common
    if len(test_rare) > 0:
        print("\nEvaluating on rare combinations...")
        losses_rare = evaluate_on_combinations(model, test_loader, device, test_rare)
        avg_loss_rare = np.mean(list(losses_rare.values())) if losses_rare else float('nan')
    else:
        avg_loss_rare = float('nan')
        print("No rare combinations found in test set")

    if len(test_common) > 0:
        print("Evaluating on common combinations...")
        losses_common = evaluate_on_combinations(model, test_loader, device, test_common)
        avg_loss_common = np.mean(list(losses_common.values())) if losses_common else float('nan')
    else:
        avg_loss_common = float('nan')
        print("No common combinations found in test set")

    print(f"\nResults:")
    print(f"  Loss on rare combinations: {avg_loss_rare:.4f}")
    print(f"  Loss on common combinations: {avg_loss_common:.4f}")

    if not np.isnan(avg_loss_rare) and not np.isnan(avg_loss_common):
        gap = avg_loss_rare - avg_loss_common
        print(f"  Gap (rare - common): {gap:.4f}")
        print(f"    → {'Manifold helps!' if gap < 0.1 else 'Significant gap'}")

    return {
        'num_train_combos': len(train_combinations),
        'num_rare': len(rare_combos),
        'num_common': len(common_combos),
        'num_test_rare': len(test_rare),
        'num_test_common': len(test_common),
        'loss_on_rare': avg_loss_rare,
        'loss_on_common': avg_loss_common,
        'gap': avg_loss_rare - avg_loss_common if not np.isnan(avg_loss_rare) else float('nan')
    }


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save directory
    save_dir = Path(__file__).parent / 'results' / 'exp4_generalization'
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
        'batch_size': 32
    }

    # Create datasets
    print("Creating datasets...")
    train_dataset = create_dummy_dataset(5000, config['max_seq_len'], config['vocab_size'])
    test_dataset = create_dummy_dataset(500, config['max_seq_len'], config['vocab_size'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

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

    # Analyze both models
    results_baseline = analyze_combination_generalization(
        baseline_model, train_loader, test_loader, device, "Baseline"
    )

    results_manifold = analyze_combination_generalization(
        manifold_model, train_loader, test_loader, device, "Manifold"
    )

    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<15} {'Loss (Rare)':<15} {'Loss (Common)':<15} {'Gap':<10}")
    print("-"*80)
    print(f"{'Baseline':<15} {results_baseline['loss_on_rare']:>14.4f} "
          f"{results_baseline['loss_on_common']:>14.4f} "
          f"{results_baseline['gap']:>9.4f}")
    print(f"{'Manifold':<15} {results_manifold['loss_on_rare']:>14.4f} "
          f"{results_manifold['loss_on_common']:>14.4f} "
          f"{results_manifold['gap']:>9.4f}")
    print("="*80)

    # Save results
    all_results = {
        'baseline': results_baseline,
        'manifold': results_manifold,
        'config': config
    }

    with open(save_dir / 'generalization_results.json', 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.floating, float)) and np.isnan(obj):
                return None
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj

        json.dump(all_results, f, indent=2, default=convert)

    print(f"\n✓ Experiment 4 completed!")
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
