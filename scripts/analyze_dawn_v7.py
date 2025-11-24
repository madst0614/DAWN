"""
DAWN v7.0 Checkpoint Analysis
Fixed Orthogonal Basis + Recipe-based Neurons 구조 분석

분석 항목:
1. Basis 직교성 검증 (완벽해야 함!)
2. Recipe 분석 (basis 사용 패턴)
3. Neuron embedding 다양성
4. Layer별 Recipe 차이
5. Neuron 선택 패턴
6. Token-Neuron 전문화

v7.0 특징:
- Basis는 고정되어 완벽한 직교성을 가짐
- Neuron embedding은 recipe로부터 파생됨
- Recipe가 학습되는 유일한 FFN 파라미터

Usage:
    python scripts/analyze_dawn_v7.py --checkpoint /path/to/checkpoint_folder
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import json
from collections import defaultdict, Counter

from models.model_v7 import DAWN
from utils.data import load_data, apply_mlm_masking, MLM_CONFIG
from transformers import BertTokenizer


# ============================================================
# 1. Basis Orthogonality Analysis (핵심!)
# ============================================================

def analyze_basis_orthogonality(model):
    """Verify that basis is perfectly orthogonal"""
    print("\n" + "="*60)
    print("1. Basis Orthogonality Verification")
    print("="*60)

    basis = model.shared_basis
    n_basis = basis.n_basis

    results = {}

    # Basis A orthogonality
    basis_A_flat = basis.basis_A.view(n_basis, -1)
    gram_A = torch.mm(basis_A_flat, basis_A_flat.T)
    identity_A = torch.eye(n_basis, device=gram_A.device)
    error_A = (gram_A - identity_A).abs()

    results['basis_A'] = {
        'max_off_diagonal': error_A.fill_diagonal_(0).max().item(),
        'mean_off_diagonal': error_A.mean().item(),
        'diagonal_mean': torch.diag(gram_A).mean().item(),
        'diagonal_std': torch.diag(gram_A).std().item(),
    }

    # Basis B orthogonality
    basis_B_flat = basis.basis_B.view(n_basis, -1)
    gram_B = torch.mm(basis_B_flat, basis_B_flat.T)
    identity_B = torch.eye(n_basis, device=gram_B.device)
    error_B = (gram_B - identity_B).abs()

    results['basis_B'] = {
        'max_off_diagonal': error_B.fill_diagonal_(0).max().item(),
        'mean_off_diagonal': error_B.mean().item(),
        'diagonal_mean': torch.diag(gram_B).mean().item(),
        'diagonal_std': torch.diag(gram_B).std().item(),
    }

    # Basis embedding orthogonality
    gram_emb = torch.mm(basis.basis_emb, basis.basis_emb.T)
    identity_emb = torch.eye(n_basis, device=gram_emb.device)
    error_emb = (gram_emb - identity_emb).abs()

    results['basis_emb'] = {
        'max_off_diagonal': error_emb.fill_diagonal_(0).max().item(),
        'mean_off_diagonal': error_emb.mean().item(),
        'diagonal_mean': torch.diag(gram_emb).mean().item(),
        'diagonal_std': torch.diag(gram_emb).std().item(),
    }

    # Print results
    for basis_name, stats in results.items():
        print(f"\n{basis_name}:")
        print(f"  Max off-diagonal: {stats['max_off_diagonal']:.2e}")
        print(f"  Mean off-diagonal: {stats['mean_off_diagonal']:.2e}")
        print(f"  Diagonal mean: {stats['diagonal_mean']:.6f}")
        print(f"  Diagonal std: {stats['diagonal_std']:.6f}")

    # Overall verdict
    max_error = max(
        results['basis_A']['max_off_diagonal'],
        results['basis_B']['max_off_diagonal'],
        results['basis_emb']['max_off_diagonal']
    )

    print(f"\n{'✅' if max_error < 1e-5 else '❌'} Overall: Max error = {max_error:.2e}")
    print(f"   Orthogonality {'PERFECT' if max_error < 1e-5 else 'IMPERFECT'}!")

    return results


# ============================================================
# 2. Recipe Analysis
# ============================================================

def analyze_recipes(model):
    """Analyze how neurons combine basis elements"""
    print("\n" + "="*60)
    print("2. Recipe Analysis")
    print("="*60)

    results = {}

    for layer_idx, layer in enumerate(model.layers):
        recipe = layer.ffn.neuron_recipe  # [n_neurons, n_basis]
        recipe_norm = F.softmax(recipe, dim=-1)  # Normalized

        # 1. Basis usage distribution
        basis_usage = recipe_norm.mean(dim=0)  # [n_basis]

        # 2. Recipe sparsity (entropy per neuron)
        recipe_entropy = -torch.sum(
            recipe_norm * torch.log(recipe_norm + 1e-10), dim=-1
        )  # [n_neurons]

        # 3. Recipe diversity (variance across neurons)
        recipe_std = recipe_norm.std(dim=0)  # [n_basis]

        # 4. Dominant basis per neuron
        dominant_basis = recipe_norm.argmax(dim=-1)
        dominant_counts = torch.bincount(
            dominant_basis, minlength=model.n_basis
        )

        # 5. Neuron specialization (max weight per neuron)
        max_weights = recipe_norm.max(dim=-1)[0]

        results[f'layer_{layer_idx}'] = {
            'basis_usage_mean': basis_usage.mean().item(),
            'basis_usage_std': basis_usage.std().item(),
            'basis_usage_min': basis_usage.min().item(),
            'basis_usage_max': basis_usage.max().item(),
            'recipe_entropy_mean': recipe_entropy.mean().item(),
            'recipe_entropy_std': recipe_entropy.std().item(),
            'recipe_diversity_mean': recipe_std.mean().item(),
            'neuron_specialization_mean': max_weights.mean().item(),
            'neuron_specialization_std': max_weights.std().item(),
            'dominant_basis_dist': dominant_counts.cpu().numpy().tolist(),
        }

        print(f"\nLayer {layer_idx}:")
        print(f"  Basis usage: {basis_usage.mean().item():.4f} ± {basis_usage.std().item():.4f}")
        print(f"  Recipe entropy: {recipe_entropy.mean().item():.4f} ± {recipe_entropy.std().item():.4f}")
        print(f"  Recipe diversity: {recipe_std.mean().item():.4f}")
        print(f"  Neuron specialization: {max_weights.mean().item():.4f} ± {max_weights.std().item():.4f}")

    return results


# ============================================================
# 3. Neuron Embedding Analysis
# ============================================================

def analyze_neuron_embeddings(model):
    """Analyze neuron embeddings derived from recipes"""
    print("\n" + "="*60)
    print("3. Neuron Embedding Analysis")
    print("="*60)

    results = {}

    for layer_idx, layer in enumerate(model.layers):
        neuron_emb = layer.ffn.neuron_emb  # [n_neurons, d_model]
        neuron_emb_norm = F.normalize(neuron_emb, dim=-1)

        # 1. Pairwise similarity
        similarity = torch.mm(neuron_emb_norm, neuron_emb_norm.T)
        mask = 1 - torch.eye(model.n_neurons, device=similarity.device)
        off_diag_sim = similarity * mask

        # 2. Embedding norm
        emb_norms = torch.norm(neuron_emb, dim=-1)

        # 3. Clustering measure (average similarity)
        avg_similarity = off_diag_sim.sum() / mask.sum()

        results[f'layer_{layer_idx}'] = {
            'avg_similarity': avg_similarity.item(),
            'max_similarity': off_diag_sim.max().item(),
            'min_similarity': off_diag_sim[mask.bool()].min().item(),
            'emb_norm_mean': emb_norms.mean().item(),
            'emb_norm_std': emb_norms.std().item(),
        }

        print(f"\nLayer {layer_idx}:")
        print(f"  Avg similarity: {avg_similarity.item():.4f}")
        print(f"  Similarity range: [{off_diag_sim[mask.bool()].min().item():.4f}, {off_diag_sim.max().item():.4f}]")
        print(f"  Embedding norm: {emb_norms.mean().item():.4f} ± {emb_norms.std().item():.4f}")

    return results


# ============================================================
# 4. Activation Analysis (requires data)
# ============================================================

class ActivationCollector:
    """Collect neuron selection patterns"""

    def __init__(self, model, n_layers):
        self.model = model
        self.n_layers = n_layers
        self.n_neurons = model.n_neurons

        # Neuron selection counts
        self.selection_counts = [
            torch.zeros(self.n_neurons) for _ in range(n_layers)
        ]

        # Token-neuron mapping
        self.token_neuron_map = defaultdict(lambda: [[] for _ in range(n_layers)])

    def collect(self, input_ids, neuron_indices):
        """Collect from one batch"""
        input_ids_cpu = input_ids.cpu()

        for layer_idx, selected_idx in enumerate(neuron_indices):
            selected_cpu = selected_idx.cpu()

            # Count selections
            unique, counts = torch.unique(selected_cpu, return_counts=True)
            self.selection_counts[layer_idx][unique] += counts

            # Token-neuron mapping
            B, S, k = selected_cpu.shape
            selected_flat = selected_cpu.reshape(-1, k)
            tokens_flat = input_ids_cpu.reshape(-1)

            for token_id in tokens_flat.unique().tolist():
                mask = tokens_flat == token_id
                neurons = selected_flat[mask].reshape(-1).tolist()
                self.token_neuron_map[token_id][layer_idx].extend(neurons)


def analyze_activations(model, data_loader, device, max_batches=100):
    """Analyze neuron selection patterns on data"""
    print("\n" + "="*60)
    print("4. Activation Analysis")
    print("="*60)

    collector = ActivationCollector(model, model.n_layers)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Collecting activations", total=max_batches)):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            logits, neuron_indices = model(input_ids, return_activations=True)

            collector.collect(input_ids, neuron_indices)

    # Analyze selection patterns
    results = {}

    for layer_idx in range(model.n_layers):
        counts = collector.selection_counts[layer_idx]
        total = counts.sum()

        if total == 0:
            continue

        # Normalized distribution
        probs = counts / total

        # Gini coefficient
        sorted_probs = torch.sort(probs)[0]
        n = len(sorted_probs)
        index = torch.arange(1, n + 1, dtype=torch.float32)
        gini = ((2 * index - n - 1) * sorted_probs).sum() / (n * sorted_probs.sum())

        # Entropy
        probs_nonzero = probs[probs > 0]
        entropy = -torch.sum(probs_nonzero * torch.log(probs_nonzero))

        # Usage stats
        used_neurons = (counts > 0).sum().item()
        usage_rate = used_neurons / model.n_neurons

        results[f'layer_{layer_idx}'] = {
            'gini': gini.item(),
            'entropy': entropy.item(),
            'used_neurons': used_neurons,
            'usage_rate': usage_rate,
            'mean_selections': counts.mean().item(),
            'max_selections': counts.max().item(),
        }

        print(f"\nLayer {layer_idx}:")
        print(f"  Used neurons: {used_neurons}/{model.n_neurons} ({usage_rate*100:.1f}%)")
        print(f"  Gini coefficient: {gini.item():.4f}")
        print(f"  Entropy: {entropy.item():.4f}")
        print(f"  Max selections: {counts.max().item():.0f}")

    return results, collector


# ============================================================
# 5. Main Analysis
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze DAWN v7.0 checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint folder')
    parser.add_argument('--data-dir', type=str, default='/content/drive/MyDrive/data',
                        help='Data directory')
    parser.add_argument('--max-batches', type=int, default=100,
                        help='Max batches for activation analysis')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for activation analysis')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to checkpoint folder')

    args = parser.parse_args()

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint) / 'best_model.pt'
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model
    config = checkpoint['config']
    print(f"\nModel config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    model = DAWN(**config)
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"\nModel version: {DAWN.__version__}")
    print(f"Device: {device}")

    # Analysis results
    all_results = {}

    # 1. Basis orthogonality
    all_results['orthogonality'] = analyze_basis_orthogonality(model)

    # 2. Recipe analysis
    all_results['recipes'] = analyze_recipes(model)

    # 3. Neuron embedding analysis
    all_results['neuron_embeddings'] = analyze_neuron_embeddings(model)

    # 4. Activation analysis (if data available)
    if Path(args.data_dir).exists():
        try:
            print(f"\nLoading validation data from {args.data_dir}...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            # Load validation data
            from torch.utils.data import DataLoader
            from utils.data import TextDataset, collate_fn_dynamic_padding

            val_file = Path(args.data_dir) / 'validation' / 'wikitext_5to1_texts.pkl'
            if val_file.exists():
                val_dataset = TextDataset(str(val_file))
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn_dynamic_padding
                )

                activation_results, collector = analyze_activations(
                    model, val_loader, device, max_batches=args.max_batches
                )
                all_results['activations'] = activation_results

        except Exception as e:
            print(f"\n⚠️  Could not load data for activation analysis: {e}")
    else:
        print(f"\n⚠️  Data directory not found: {args.data_dir}")
        print("   Skipping activation analysis")

    # Save results
    results_file = Path(args.checkpoint) / 'analysis_v7.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Analysis complete!")
    print(f"   Results saved to: {results_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
