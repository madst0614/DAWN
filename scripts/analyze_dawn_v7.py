"""
DAWN v7.0 Checkpoint Analysis
Fixed Orthogonal Basis + Recipe-based Neurons 구조 분석

분석 항목:
1. Basis 직교성 검증 (완벽해야 함!)
2. Recipe 분석 (basis 사용 패턴)
3. Neuron embedding 다양성
3.5 Recipe 유사도 분석 (NEW!)
3.6 Basis 기여도 분석 (NEW!)
3.7 Recipe Sparsity 상세 (NEW!)
3.8 Layer 역할 분석 (NEW!)
4. Neuron 선택 패턴
5. Token-Recipe 매핑 (NEW!)

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
import argparse
from tqdm import tqdm
import json
from collections import defaultdict, Counter


# ============================================================
# Helper Functions
# ============================================================

def compute_gini(values):
    """Compute Gini coefficient"""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu()
        sorted_values = torch.sort(values.float())[0]
    else:
        sorted_values = torch.sort(torch.tensor(values, dtype=torch.float32))[0]

    n = len(sorted_values)
    if n == 0 or sorted_values.sum() == 0:
        return 0.0
    index = torch.arange(1, n + 1, dtype=torch.float32)
    return ((2 * index - n - 1) * sorted_values).sum() / (n * sorted_values.sum())


# ============================================================
# 1. Basis Orthogonality Analysis (핵심!)
# ============================================================

def analyze_basis_orthogonality(model):
    """Verify that basis is perfectly orthogonal"""
    print("\n" + "="*60)
    print("1. BASIS ORTHOGONALITY VERIFICATION")
    print("="*60)

    basis = model.shared_basis
    n_basis = basis.n_basis

    results = {}

    # Basis A orthogonality
    basis_A_flat = basis.basis_A.view(n_basis, -1)
    gram_A = torch.mm(basis_A_flat, basis_A_flat.T)
    identity_A = torch.eye(n_basis, device=gram_A.device)
    error_A = (gram_A - identity_A).abs()
    error_A_offdiag = error_A.clone()
    error_A_offdiag.fill_diagonal_(0)

    results['basis_A'] = {
        'max_off_diagonal': error_A_offdiag.max().item(),
        'mean_off_diagonal': error_A_offdiag.sum().item() / (n_basis * (n_basis - 1)),
        'diagonal_mean': torch.diag(gram_A).mean().item(),
        'diagonal_std': torch.diag(gram_A).std().item(),
    }

    # Basis B orthogonality
    basis_B_flat = basis.basis_B.view(n_basis, -1)
    gram_B = torch.mm(basis_B_flat, basis_B_flat.T)
    error_B = (gram_B - identity_A).abs()
    error_B_offdiag = error_B.clone()
    error_B_offdiag.fill_diagonal_(0)

    results['basis_B'] = {
        'max_off_diagonal': error_B_offdiag.max().item(),
        'mean_off_diagonal': error_B_offdiag.sum().item() / (n_basis * (n_basis - 1)),
        'diagonal_mean': torch.diag(gram_B).mean().item(),
        'diagonal_std': torch.diag(gram_B).std().item(),
    }

    # Basis embedding orthogonality
    gram_emb = torch.mm(basis.basis_emb, basis.basis_emb.T)
    error_emb = (gram_emb - identity_A).abs()
    error_emb_offdiag = error_emb.clone()
    error_emb_offdiag.fill_diagonal_(0)

    results['basis_emb'] = {
        'max_off_diagonal': error_emb_offdiag.max().item(),
        'mean_off_diagonal': error_emb_offdiag.sum().item() / (n_basis * (n_basis - 1)),
        'diagonal_mean': torch.diag(gram_emb).mean().item(),
        'diagonal_std': torch.diag(gram_emb).std().item(),
    }

    # Print results
    for basis_name, stats in results.items():
        print(f"\n{basis_name}:")
        print(f"  Max off-diagonal: {stats['max_off_diagonal']:.2e}")
        print(f"  Mean off-diagonal: {stats['mean_off_diagonal']:.2e}")
        print(f"  Diagonal mean: {stats['diagonal_mean']:.6f}")

    # Overall verdict
    max_error = max(
        results['basis_A']['max_off_diagonal'],
        results['basis_B']['max_off_diagonal'],
        results['basis_emb']['max_off_diagonal']
    )

    print(f"\n{'✅' if max_error < 1e-5 else '⚠️'} Overall: Max error = {max_error:.2e}")
    print(f"   Orthogonality {'PERFECT' if max_error < 1e-5 else 'APPROXIMATE'}!")

    return results


# ============================================================
# 2. Recipe Analysis
# ============================================================

def analyze_recipes(model):
    """Analyze how neurons combine basis elements"""
    print("\n" + "="*60)
    print("2. RECIPE ANALYSIS")
    print("="*60)

    results = {}

    for layer_idx, layer in enumerate(model.layers):
        recipe = layer.ffn.neuron_recipe  # [n_neurons, n_basis]
        recipe_norm = F.softmax(recipe, dim=-1)  # Normalized

        # 1. Basis usage distribution
        basis_usage = recipe_norm.mean(dim=0)  # [n_basis]

        # 2. Recipe entropy (how spread out each recipe is)
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
        print(f"  Neuron specialization: {max_weights.mean().item():.4f} ± {max_weights.std().item():.4f}")

    return results


# ============================================================
# 3. Neuron Embedding Analysis
# ============================================================

def analyze_neuron_embeddings(model):
    """Analyze neuron embeddings derived from recipes"""
    print("\n" + "="*60)
    print("3. NEURON EMBEDDING ANALYSIS")
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
# 3.5 Recipe Similarity Analysis (NEW!)
# ============================================================

def analyze_recipe_similarity(model):
    """Analyze similarity between neuron recipes within and across layers"""
    print("\n" + "="*60)
    print("3.5 RECIPE SIMILARITY ANALYSIS")
    print("="*60)

    results = {}
    all_recipes = []

    for layer_idx, layer in enumerate(model.layers):
        recipe = layer.ffn.neuron_recipe  # [n_neurons, n_basis]
        recipe_norm = F.softmax(recipe, dim=-1)
        all_recipes.append(recipe_norm)

        # Within-layer recipe similarity
        recipe_normalized = F.normalize(recipe_norm, dim=-1)
        similarity = torch.mm(recipe_normalized, recipe_normalized.T)
        mask = 1 - torch.eye(model.n_neurons, device=similarity.device)
        off_diag = similarity * mask

        # Find highly similar recipe pairs
        high_sim_threshold = 0.9
        high_sim_count = (off_diag > high_sim_threshold).sum().item() // 2

        # Effective number of distinct recipes (using eigenvalue analysis)
        cov_matrix = torch.mm(recipe_norm.T, recipe_norm) / model.n_neurons
        eigenvalues = torch.linalg.eigvalsh(cov_matrix)
        eigenvalues = eigenvalues.clamp(min=1e-10)
        normalized_eig = eigenvalues / eigenvalues.sum()
        effective_rank = torch.exp(-torch.sum(normalized_eig * torch.log(normalized_eig))).item()

        results[f'layer_{layer_idx}'] = {
            'avg_similarity': (off_diag.sum() / mask.sum()).item(),
            'max_similarity': off_diag.max().item(),
            'high_sim_pairs': high_sim_count,
            'effective_rank': effective_rank,
            'diversity_ratio': effective_rank / model.n_basis,
        }

        print(f"\nLayer {layer_idx}:")
        print(f"  Avg recipe similarity: {results[f'layer_{layer_idx}']['avg_similarity']:.4f}")
        print(f"  Max recipe similarity: {off_diag.max().item():.4f}")
        print(f"  High similarity pairs (>0.9): {high_sim_count}")
        print(f"  Effective rank: {effective_rank:.2f}/{model.n_basis} ({results[f'layer_{layer_idx}']['diversity_ratio']*100:.1f}%)")

    # Cross-layer recipe comparison
    print("\n  Cross-layer Recipe Correlation:")
    cross_layer_sim = {}
    for i in range(len(all_recipes)):
        for j in range(i+1, len(all_recipes)):
            # Average recipe per layer
            avg_recipe_i = all_recipes[i].mean(dim=0)
            avg_recipe_j = all_recipes[j].mean(dim=0)
            cos_sim = F.cosine_similarity(avg_recipe_i, avg_recipe_j, dim=0).item()
            cross_layer_sim[f'layer_{i}_vs_{j}'] = cos_sim
            print(f"    Layer {i} vs Layer {j}: {cos_sim:.4f}")

    results['cross_layer'] = cross_layer_sim

    return results


# ============================================================
# 3.6 Basis Contribution Analysis (NEW!)
# ============================================================

def analyze_basis_contribution(model):
    """Analyze how much each basis contributes across all neurons"""
    print("\n" + "="*60)
    print("3.6 BASIS CONTRIBUTION ANALYSIS")
    print("="*60)

    results = {}
    n_basis = model.n_basis

    # Aggregate across all layers
    all_basis_usage = torch.zeros(n_basis)

    for layer_idx, layer in enumerate(model.layers):
        recipe = layer.ffn.neuron_recipe  # [n_neurons, n_basis]
        recipe_norm = F.softmax(recipe, dim=-1)

        # Per-basis usage (averaged over neurons)
        basis_usage = recipe_norm.mean(dim=0)  # [n_basis]
        all_basis_usage += basis_usage.detach().cpu()

        # Identify dead/underused bases
        dead_threshold = 1.0 / (n_basis * 10)  # Less than 10% of uniform
        underused_bases = (basis_usage < dead_threshold).sum().item()

        # Identify dominant bases
        dominant_threshold = 2.0 / n_basis  # More than 2x uniform
        dominant_bases = (basis_usage > dominant_threshold).sum().item()

        # Gini coefficient for basis usage
        gini = compute_gini(basis_usage)

        # Entropy
        usage_entropy = -torch.sum(basis_usage * torch.log(basis_usage + 1e-10)).item()

        results[f'layer_{layer_idx}'] = {
            'basis_usage': basis_usage.detach().cpu().numpy().tolist(),
            'underused_bases': underused_bases,
            'dominant_bases': dominant_bases,
            'usage_gini': gini.item() if isinstance(gini, torch.Tensor) else gini,
            'usage_entropy': usage_entropy,
        }

        print(f"\nLayer {layer_idx}:")
        print(f"  Underused bases (<10% uniform): {underused_bases}/{n_basis}")
        print(f"  Dominant bases (>2x uniform): {dominant_bases}/{n_basis}")
        print(f"  Usage Gini: {gini:.4f}")

        # Top 5 and bottom 5 bases
        top_k = 5
        top_indices = torch.topk(basis_usage, top_k).indices.cpu().numpy()
        bottom_indices = torch.topk(basis_usage, top_k, largest=False).indices.cpu().numpy()
        print(f"  Top {top_k} bases: {top_indices.tolist()}")
        print(f"  Bottom {top_k} bases: {bottom_indices.tolist()}")

    # Overall basis usage
    all_basis_usage /= len(model.layers)
    overall_gini = compute_gini(all_basis_usage)

    print(f"\nOverall Basis Usage (averaged across layers):")
    print(f"  Gini: {overall_gini:.4f}")

    # Check for consistently unused bases
    consistently_low = (all_basis_usage < (1.0 / (n_basis * 5))).sum().item()
    if consistently_low > 0:
        print(f"  ⚠️  {consistently_low} bases consistently underused across all layers!")
    else:
        print(f"  ✅ All bases actively used!")

    results['overall'] = {
        'basis_usage': all_basis_usage.numpy().tolist(),
        'gini': overall_gini.item() if isinstance(overall_gini, torch.Tensor) else overall_gini,
        'consistently_underused': consistently_low,
    }

    return results


# ============================================================
# 3.7 Recipe Sparsity Detail (NEW!)
# ============================================================

def analyze_recipe_sparsity(model):
    """Detailed analysis of recipe sparsity patterns"""
    print("\n" + "="*60)
    print("3.7 RECIPE SPARSITY ANALYSIS")
    print("="*60)

    results = {}

    for layer_idx, layer in enumerate(model.layers):
        recipe = layer.ffn.neuron_recipe  # [n_neurons, n_basis]
        recipe_norm = F.softmax(recipe, dim=-1)

        # Effective number of bases per neuron
        # Using entropy: exp(H) gives effective count
        entropy = -torch.sum(recipe_norm * torch.log(recipe_norm + 1e-10), dim=-1)
        effective_bases = torch.exp(entropy)

        layer_results = {}

        # Concentration ratio (top-k bases contribution)
        for k in [1, 3, 5]:
            topk_sum = torch.topk(recipe_norm, k, dim=-1)[0].sum(dim=-1)
            mean_topk = topk_sum.mean().item()
            layer_results[f'top{k}_concentration'] = mean_topk

        # Sparsity categories
        sparse_threshold = 3  # Using less than 3 bases effectively
        dense_threshold = model.n_basis * 0.5  # Using more than half

        sparse_neurons = (effective_bases < sparse_threshold).sum().item()
        dense_neurons = (effective_bases > dense_threshold).sum().item()
        moderate_neurons = model.n_neurons - sparse_neurons - dense_neurons

        layer_results.update({
            'effective_bases_mean': effective_bases.mean().item(),
            'effective_bases_std': effective_bases.std().item(),
            'effective_bases_min': effective_bases.min().item(),
            'effective_bases_max': effective_bases.max().item(),
            'sparse_neurons': sparse_neurons,
            'moderate_neurons': moderate_neurons,
            'dense_neurons': dense_neurons,
        })

        results[f'layer_{layer_idx}'] = layer_results

        print(f"\nLayer {layer_idx}:")
        print(f"  Effective bases per neuron: {effective_bases.mean().item():.2f} ± {effective_bases.std().item():.2f}")
        print(f"  Range: [{effective_bases.min().item():.2f}, {effective_bases.max().item():.2f}]")
        print(f"  Top-1 concentration: {layer_results['top1_concentration']*100:.1f}%")
        print(f"  Top-3 concentration: {layer_results['top3_concentration']*100:.1f}%")
        print(f"  Top-5 concentration: {layer_results['top5_concentration']*100:.1f}%")
        print(f"  Neurons: {sparse_neurons} sparse / {moderate_neurons} moderate / {dense_neurons} dense")

    return results


# ============================================================
# 3.8 Layer Role Analysis (NEW!)
# ============================================================

def analyze_layer_roles(model):
    """Analyze if different layers serve different roles"""
    print("\n" + "="*60)
    print("3.8 LAYER ROLE ANALYSIS")
    print("="*60)

    results = {}

    recipes = []
    for layer in model.layers:
        recipe = layer.ffn.neuron_recipe
        recipe_norm = F.softmax(recipe, dim=-1)
        recipes.append(recipe_norm)

    # Compare layer-average recipes
    avg_recipes = torch.stack([r.mean(dim=0) for r in recipes])  # [n_layers, n_basis]

    # Which bases does each layer prefer?
    for layer_idx in range(len(model.layers)):
        layer_pref = avg_recipes[layer_idx]
        top_bases = torch.topk(layer_pref, 5).indices.cpu().numpy()

        # Compare to other layers
        other_layers = [i for i in range(len(model.layers)) if i != layer_idx]
        if other_layers:
            other_avg = avg_recipes[other_layers].mean(dim=0)
            # Difference from other layers
            diff = layer_pref - other_avg
            unique_high = torch.topk(diff, 3).indices.cpu().numpy()
            unique_low = torch.topk(diff, 3, largest=False).indices.cpu().numpy()
        else:
            unique_high = []
            unique_low = []

        results[f'layer_{layer_idx}'] = {
            'top_bases': top_bases.tolist(),
            'unique_high': unique_high.tolist() if len(unique_high) else [],
            'unique_low': unique_low.tolist() if len(unique_low) else [],
        }

        print(f"\nLayer {layer_idx}:")
        print(f"  Top bases: {top_bases.tolist()}")
        print(f"  Uniquely high (vs other layers): {unique_high.tolist() if len(unique_high) else 'N/A'}")
        print(f"  Uniquely low (vs other layers): {unique_low.tolist() if len(unique_low) else 'N/A'}")

    # Layer progression analysis
    print("\n  Layer Progression (first 5 bases):")
    for basis_idx in range(min(5, model.n_basis)):
        progression = [avg_recipes[l][basis_idx].item() for l in range(len(model.layers))]
        trend = "↑" if progression[-1] > progression[0] * 1.1 else "↓" if progression[-1] < progression[0] * 0.9 else "→"
        print(f"    Basis {basis_idx}: {' → '.join([f'{p:.3f}' for p in progression])} {trend}")

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

        # Batch count
        self.n_batches = 0

    def collect(self, input_ids, neuron_indices):
        """Collect from one batch"""
        input_ids_cpu = input_ids.cpu()
        self.n_batches += 1

        for layer_idx, selected_idx in enumerate(neuron_indices):
            selected_cpu = selected_idx.cpu()

            # Count selections
            flat_idx = selected_cpu.reshape(-1)
            for idx in flat_idx.tolist():
                self.selection_counts[layer_idx][idx] += 1

            # Token-neuron mapping (sample to avoid memory issues)
            if self.n_batches <= 10:  # Only first 10 batches
                B, S, k = selected_cpu.shape
                for b in range(B):
                    for s in range(S):
                        token_id = input_ids_cpu[b, s].item()
                        neurons = selected_cpu[b, s].tolist()
                        self.token_neuron_map[token_id][layer_idx].extend(neurons)


def analyze_activations(model, data_loader, device, max_batches=100):
    """Analyze neuron selection patterns on data"""
    print("\n" + "="*60)
    print("4. NEURON SELECTION ANALYSIS")
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
        gini = compute_gini(probs)

        # Entropy
        probs_nonzero = probs[probs > 0]
        entropy = -torch.sum(probs_nonzero * torch.log(probs_nonzero))

        # Usage stats
        used_neurons = (counts > 0).sum().item()
        usage_rate = used_neurons / model.n_neurons

        # Top neurons
        top_neurons = torch.topk(counts, 10).indices.cpu().numpy()
        top_counts = torch.topk(counts, 10).values.cpu().numpy()
        top_10_ratio = top_counts.sum() / total.item()

        results[f'layer_{layer_idx}'] = {
            'gini': gini.item() if isinstance(gini, torch.Tensor) else gini,
            'entropy': entropy.item(),
            'used_neurons': used_neurons,
            'usage_rate': usage_rate,
            'top_10_ratio': top_10_ratio,
            'top_neurons': top_neurons.tolist(),
        }

        print(f"\nLayer {layer_idx}:")
        print(f"  Used neurons: {used_neurons}/{model.n_neurons} ({usage_rate*100:.1f}%)")
        print(f"  Gini coefficient: {gini:.4f}")
        print(f"  Entropy: {entropy.item():.4f}")
        print(f"  Top-10 neurons share: {top_10_ratio*100:.1f}%")

        # Warning flags
        if usage_rate < 0.5:
            print(f"  ⚠️  LOW USAGE: Only {usage_rate*100:.1f}% neurons used!")
        if gini > 0.8:
            print(f"  ⚠️  CONCENTRATED: Gini = {gini:.2f}")

    return results, collector


# ============================================================
# 5. Token-Recipe Analysis (NEW!)
# ============================================================

def analyze_token_recipe_mapping(collector, tokenizer, model, top_k_tokens=20):
    """Analyze which recipes/bases different tokens prefer"""
    print("\n" + "="*60)
    print("5. TOKEN-RECIPE MAPPING ANALYSIS")
    print("="*60)

    results = {}

    # Get recipes for each layer
    recipes = []
    for layer in model.layers:
        recipe = layer.ffn.neuron_recipe
        recipe_norm = F.softmax(recipe, dim=-1)
        recipes.append(recipe_norm)

    # Find tokens with most selections
    token_counts = {tid: sum(len(neurons) for neurons in layers)
                    for tid, layers in collector.token_neuron_map.items()}
    top_tokens = sorted(token_counts.items(), key=lambda x: -x[1])[:top_k_tokens]

    for token_id, count in top_tokens:
        token_str = tokenizer.decode([token_id])

        for layer_idx in range(model.n_layers):
            neurons = collector.token_neuron_map[token_id][layer_idx]
            if not neurons:
                continue

            # Get the recipe for each selected neuron
            neuron_counts = Counter(neurons)
            top_neurons = neuron_counts.most_common(5)

            # Average recipe across selected neurons
            neuron_ids = list(set(neurons))
            if neuron_ids:
                avg_recipe = recipes[layer_idx][neuron_ids].mean(dim=0)
                top_bases = torch.topk(avg_recipe, 3).indices.cpu().numpy()

                results.setdefault(token_str, {})[f'layer_{layer_idx}'] = {
                    'top_neurons': [n for n, c in top_neurons],
                    'top_bases': top_bases.tolist(),
                }

    # Print summary
    print(f"\nTop {min(5, len(top_tokens))} tokens:")
    for token_id, count in top_tokens[:5]:
        token_str = tokenizer.decode([token_id])
        print(f"\n  '{token_str}' ({count} selections):")
        if token_str in results:
            for layer_key, data in results[token_str].items():
                print(f"    {layer_key}: neurons {data['top_neurons'][:3]}, bases {data['top_bases']}")

    return results


# ============================================================
# 6. Summary
# ============================================================

def print_summary(all_results):
    """Print overall summary with recommendations"""
    print("\n" + "="*60)
    print("📊 ANALYSIS SUMMARY")
    print("="*60)

    # Orthogonality
    orth = all_results.get('orthogonality', {})
    max_orth_error = max(
        orth.get('basis_A', {}).get('max_off_diagonal', 0),
        orth.get('basis_B', {}).get('max_off_diagonal', 0),
        orth.get('basis_emb', {}).get('max_off_diagonal', 0),
    )
    print(f"\n1. Basis Orthogonality: {'✅ PERFECT' if max_orth_error < 1e-5 else '⚠️ APPROXIMATE'}")

    # Recipe diversity
    recipe_sim = all_results.get('recipe_similarity', {})
    if recipe_sim:
        avg_sim = np.mean([v.get('avg_similarity', 0) for k, v in recipe_sim.items() if 'layer' in k])
        print(f"\n2. Recipe Diversity: {'✅ GOOD' if avg_sim < 0.5 else '⚠️ HIGH SIMILARITY'} (avg sim: {avg_sim:.3f})")

    # Basis usage
    basis_contrib = all_results.get('basis_contribution', {})
    if 'overall' in basis_contrib:
        unused = basis_contrib['overall'].get('consistently_underused', 0)
        print(f"\n3. Basis Usage: {'✅ ALL USED' if unused == 0 else f'⚠️ {unused} UNUSED'}")

    # Neuron usage
    activations = all_results.get('activations', {})
    if activations:
        usage_rates = [v.get('usage_rate', 1) for k, v in activations.items() if 'layer' in k]
        avg_usage = np.mean(usage_rates) if usage_rates else 0
        print(f"\n4. Neuron Usage: {'✅ GOOD' if avg_usage > 0.5 else '⚠️ LOW'} (avg: {avg_usage*100:.1f}%)")

    print("\n" + "="*60)


# ============================================================
# Main
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

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    config_file = checkpoint_path / 'config.json'
    model_file = checkpoint_path / 'best_model.pt'

    if not model_file.exists():
        print(f"❌ Checkpoint not found: {model_file}")
        return

    print(f"\nLoading config: {config_file}")
    print(f"Loading checkpoint: {model_file}")

    # Load checkpoint first to get full config
    checkpoint = torch.load(model_file, map_location='cpu')

    # Try to get config from checkpoint or separate file
    if config_file.exists():
        with open(config_file, 'r') as f:
            file_config = json.load(f)
            # Check if config is nested (training format) or flat (model format)
            if 'model' in file_config:
                # Nested config from training
                model_config = file_config['model'].copy()
            else:
                # Flat config
                model_config = file_config.copy()
    else:
        model_config = checkpoint.get('config', {})

    # Ensure vocab_size is in the config (might be in checkpoint)
    if 'vocab_size' not in model_config:
        # Try to infer from checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Get vocab_size from token_emb weight shape
        # Handle both normal and torch.compile() wrapped models
        token_emb_key = None
        if 'token_emb.weight' in state_dict:
            token_emb_key = 'token_emb.weight'
        elif '_orig_mod.token_emb.weight' in state_dict:
            token_emb_key = '_orig_mod.token_emb.weight'

        if token_emb_key:
            vocab_size = state_dict[token_emb_key].shape[0]
            model_config['vocab_size'] = vocab_size
            print(f"  Inferred vocab_size from checkpoint: {vocab_size}")
        else:
            print("  ⚠️  Warning: Could not find vocab_size, using default 30522")
            model_config['vocab_size'] = 30522

    print(f"\nModel config:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")

    # Import model (try different paths)
    try:
        from models.model_v7 import DAWN
    except ImportError:
        try:
            from models.model import DAWN
        except ImportError:
            from dawn_v7 import DAWN

    # Create model
    model = DAWN(**model_config)

    # Load weights (checkpoint already loaded above)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle torch.compile() wrapped models (strip "_orig_mod." prefix)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  Detected torch.compile() wrapped model, stripping prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print(f"\n📌 Model version: {getattr(DAWN, '__version__', 'unknown')}")

    # Analysis results
    all_results = {}

    # 1. Basis orthogonality
    all_results['orthogonality'] = analyze_basis_orthogonality(model)

    # 2. Recipe analysis
    all_results['recipes'] = analyze_recipes(model)

    # 3. Neuron embedding analysis
    all_results['neuron_embeddings'] = analyze_neuron_embeddings(model)

    # 3.5 Recipe similarity
    all_results['recipe_similarity'] = analyze_recipe_similarity(model)

    # 3.6 Basis contribution
    all_results['basis_contribution'] = analyze_basis_contribution(model)

    # 3.7 Recipe sparsity
    all_results['recipe_sparsity'] = analyze_recipe_sparsity(model)

    # 3.8 Layer roles
    all_results['layer_roles'] = analyze_layer_roles(model)

    # 4. Activation analysis (if data available)
    data_path = Path(args.data_dir)
    if data_path.exists():
        try:
            from transformers import BertTokenizer
            from torch.utils.data import DataLoader

            print(f"\nLoading validation data from {args.data_dir}...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            # Try to load data
            try:
                from utils.data import TextDataset, collate_fn_dynamic_padding
                val_file = data_path / 'validation' / 'wikitext_5to1_texts.pkl'
                if val_file.exists():
                    val_dataset = TextDataset(str(val_file), tokenizer)

                    # Create collate_fn with tokenizer
                    def collate_fn(batch):
                        return collate_fn_dynamic_padding(batch, tokenizer)

                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=collate_fn
                    )

                    activation_results, collector = analyze_activations(
                        model, val_loader, device, max_batches=args.max_batches
                    )
                    all_results['activations'] = activation_results

                    # 5. Token-recipe mapping
                    all_results['token_mapping'] = analyze_token_recipe_mapping(
                        collector, tokenizer, model
                    )
            except Exception as e:
                print(f"\n⚠️  Could not load data: {e}")

        except Exception as e:
            print(f"\n⚠️  Could not run activation analysis: {e}")
    else:
        print(f"\n⚠️  Data directory not found: {args.data_dir}")
        print("   Skipping activation analysis")

    # Print summary
    print_summary(all_results)

    # Save results
    output_dir = checkpoint_path / 'analysis_v7'
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / 'analysis_v7_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Analysis complete!")
    print(f"   Results saved to: {results_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
