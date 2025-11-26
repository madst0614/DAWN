"""
DAWN v7.5/v7.6 Checkpoint Analysis
Dynamic Q/K/V Generation (v8 design)

분석 항목:
1. Basis 직교성 검증 (완벽해야 함!)
2. Recipe 분석 (Q/K/V recipe 3개)
3. ⭐ Runtime 분석 (라우팅 + 뉴런 사용)
4. 종합 메트릭 요약
5. W_Q/K/V 동적성 검증 (토큰별 유사도)
6. Basis 커버리지 분석 (공간 커버리지)
7. ⭐ O Projection 정보 손실 분석 (핵심!)
8. Attention 패턴 분석

v7.5 특징:
- 라우터: x만 보고 뉴런 선택
- Q/K/V 모두 동적 생성 (recipe_Q/K/V)
- O projection: basis.T 사용 (종속)

v7.6 특징 (개선):
- basis_down (Q/K/V용)과 basis_up (O용) 분리
- 독립적인 O projection 학습
- Recipe diversity loss 추가

핵심 지표 해석 가이드:
| 지표                    | 정상 범위   | 문제 신호                      |
|------------------------|------------|-------------------------------|
| W_Q token similarity   | < 0.7      | > 0.9 → 동적 생성 무의미        |
| Basis effective rank   | > 20       | < 10 → 중복된 basis           |
| O proj relative error  | < 0.2      | > 0.4 → 심각한 정보 손실       |
| Attention entropy      | 1.5 ~ 3.0  | < 1.0 또는 > 4.0              |
| Basis correspondence   | < 0.3      | > 0.7 → basis_up 미학습 (v7.6) |

Usage:
    python scripts/analyze_dawn_v75.py --checkpoint /path/to/checkpoint_folder
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
# 1. Basis Orthogonality Analysis
# ============================================================

def analyze_basis_orthogonality(model):
    """Verify that basis is perfectly orthogonal

    Supports both v7.5 (single basis) and v7.6 (separate basis_down/basis_up)
    """
    print("\n" + "="*60)
    print("1. BASIS ORTHOGONALITY VERIFICATION")
    print("="*60)

    # Detect model version
    is_v76 = hasattr(model.shared_basis, 'get_basis_down')
    n_basis = model.n_basis
    results = {}

    def check_orthogonality(basis, name):
        """Check orthogonality for a given basis"""
        max_errors = []
        mean_errors = []

        for i in range(n_basis):
            basis_i = basis[i]  # [D, rank] or [rank, D]
            # Ensure we get the right shape for gram matrix
            if basis_i.shape[0] > basis_i.shape[1]:
                gram = basis_i.T @ basis_i  # [rank, rank]
            else:
                gram = basis_i @ basis_i.T  # [rank, rank] for basis_up
            identity = torch.eye(gram.shape[0], device=gram.device)
            error = (gram - identity).abs()
            error_offdiag = error.clone()
            error_offdiag.fill_diagonal_(0)

            max_errors.append(error_offdiag.max().item())
            mean_errors.append(error_offdiag.mean().item())

        return {
            'max_off_diagonal': max(max_errors),
            'mean_off_diagonal': np.mean(mean_errors),
            'max_across_basis': max_errors,
        }

    if is_v76:
        print("\n📌 v7.6 model: Checking both basis_down and basis_up")

        # Check basis_down
        basis_down = model.shared_basis.get_basis_down()
        results['basis_down'] = check_orthogonality(basis_down, 'basis_down')
        print(f"\nBasis_down orthogonality:")
        print(f"  Max off-diagonal: {results['basis_down']['max_off_diagonal']:.2e}")
        print(f"  Mean off-diagonal: {results['basis_down']['mean_off_diagonal']:.2e}")

        # Check basis_up
        basis_up = model.shared_basis.get_basis_up()
        results['basis_up'] = check_orthogonality(basis_up, 'basis_up')
        print(f"\nBasis_up orthogonality:")
        print(f"  Max off-diagonal: {results['basis_up']['max_off_diagonal']:.2e}")
        print(f"  Mean off-diagonal: {results['basis_up']['mean_off_diagonal']:.2e}")

        max_error = max(results['basis_down']['max_off_diagonal'],
                       results['basis_up']['max_off_diagonal'])
        results['basis'] = {'max_off_diagonal': max_error}
    else:
        basis = model.shared_basis.basis
        results['basis'] = check_orthogonality(basis, 'basis')

        print(f"\nBasis orthogonality:")
        print(f"  Max off-diagonal: {results['basis']['max_off_diagonal']:.2e}")
        print(f"  Mean off-diagonal: {results['basis']['mean_off_diagonal']:.2e}")

        max_error = results['basis']['max_off_diagonal']

    print(f"\n{'✅' if max_error < 1e-5 else '⚠️'} Overall: Max error = {max_error:.2e}")
    print(f"   Orthogonality {'PERFECT' if max_error < 1e-5 else 'APPROXIMATE'}!")

    return results


# ============================================================
# 2. Recipe Analysis (Q/K/V)
# ============================================================

def analyze_recipes(model):
    """Analyze how neurons combine basis elements for Q/K/V/O"""
    print("\n" + "="*60)
    print("2. RECIPE ANALYSIS (Q/K/V/O)")
    print("="*60)

    results = {}

    for layer_idx, layer in enumerate(model.layers):
        qkv_module = layer.qkv_dynamic

        # Get Q/K/V/O recipes
        recipe_Q = qkv_module.neuron_recipe_Q  # [n_neurons, n_basis]
        recipe_K = qkv_module.neuron_recipe_K
        recipe_V = qkv_module.neuron_recipe_V
        recipe_O = qkv_module.neuron_recipe_O

        layer_results = {}

        for recipe_name, recipe in [('Q', recipe_Q), ('K', recipe_K), ('V', recipe_V), ('O', recipe_O)]:
            recipe_norm = F.softmax(recipe, dim=-1)  # Normalized

            # 1. Basis usage distribution
            basis_usage = recipe_norm.mean(dim=0)  # [n_basis]

            # 2. Recipe entropy (how spread out each recipe is)
            recipe_entropy = -torch.sum(
                recipe_norm * torch.log(recipe_norm + 1e-10), dim=-1
            )  # [n_neurons]

            # 3. Neuron specialization (max weight per neuron)
            max_weights = recipe_norm.max(dim=-1)[0]

            # 4. Dominant basis per neuron
            dominant_basis = recipe_norm.argmax(dim=-1)
            dominant_counts = torch.bincount(
                dominant_basis, minlength=model.n_basis
            )

            layer_results[f'recipe_{recipe_name}'] = {
                'basis_usage_mean': basis_usage.mean().item(),
                'basis_usage_std': basis_usage.std().item(),
                'recipe_entropy_mean': recipe_entropy.mean().item(),
                'recipe_entropy_std': recipe_entropy.std().item(),
                'neuron_specialization_mean': max_weights.mean().item(),
                'neuron_specialization_std': max_weights.std().item(),
                'dominant_basis_dist': dominant_counts.cpu().numpy().tolist(),
            }

        # Compare Q/K/V/O recipes
        recipe_Q_norm = F.softmax(recipe_Q, dim=-1)
        recipe_K_norm = F.softmax(recipe_K, dim=-1)
        recipe_V_norm = F.softmax(recipe_V, dim=-1)
        recipe_O_norm = F.softmax(recipe_O, dim=-1)

        # Cosine similarity between recipes
        qk_sim = F.cosine_similarity(recipe_Q_norm, recipe_K_norm, dim=-1).mean().item()
        qv_sim = F.cosine_similarity(recipe_Q_norm, recipe_V_norm, dim=-1).mean().item()
        qo_sim = F.cosine_similarity(recipe_Q_norm, recipe_O_norm, dim=-1).mean().item()
        kv_sim = F.cosine_similarity(recipe_K_norm, recipe_V_norm, dim=-1).mean().item()
        ko_sim = F.cosine_similarity(recipe_K_norm, recipe_O_norm, dim=-1).mean().item()
        vo_sim = F.cosine_similarity(recipe_V_norm, recipe_O_norm, dim=-1).mean().item()

        layer_results['recipe_similarity'] = {
            'Q_K_similarity': qk_sim,
            'Q_V_similarity': qv_sim,
            'Q_O_similarity': qo_sim,
            'K_V_similarity': kv_sim,
            'K_O_similarity': ko_sim,
            'V_O_similarity': vo_sim,
        }

        results[f'layer_{layer_idx}'] = layer_results

        print(f"\nLayer {layer_idx}:")
        for recipe_name in ['Q', 'K', 'V', 'O']:
            r = layer_results[f'recipe_{recipe_name}']
            print(f"  Recipe {recipe_name}:")
            print(f"    Basis usage: {r['basis_usage_mean']:.4f} ± {r['basis_usage_std']:.4f}")
            print(f"    Entropy: {r['recipe_entropy_mean']:.4f} ± {r['recipe_entropy_std']:.4f}")
            print(f"    Specialization: {r['neuron_specialization_mean']:.4f} ± {r['neuron_specialization_std']:.4f}")
        print(f"  Recipe similarity: Q-K={qk_sim:.4f}, Q-V={qv_sim:.4f}, Q-O={qo_sim:.4f}")
        print(f"                     K-V={kv_sim:.4f}, K-O={ko_sim:.4f}, V-O={vo_sim:.4f}")

    return results


# ============================================================
# 3. ⭐ Runtime Behavior Analysis
# ============================================================

def analyze_runtime_behavior(model, dataloader, device, max_batches=10):
    """Analyze neuron routing and usage during inference"""
    print("\n" + "="*60)
    print("3. ⭐ RUNTIME BEHAVIOR ANALYSIS")
    print("="*60)

    model.eval()

    # ⚡ GPU-optimized accumulators
    n_layers = len(model.layers)
    all_neuron_usage = {f'layer_{i}': torch.zeros(model.n_neurons, device=device, dtype=torch.long)
                        for i in range(n_layers)}

    # Accumulate statistics per layer on GPU
    layer_stats = {
        'router_scores': torch.zeros(n_layers, device=device),
        'routing_weights': torch.zeros(n_layers, device=device),
        'attn_self': torch.zeros(n_layers, device=device),
        'counts': torch.zeros(n_layers, device=device),
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)

            # Forward pass with routing info
            logits, routing_infos = model(input_ids, return_routing_info=True)

            B, S = input_ids.shape

            # Compute attention self-attention for analysis
            pos = torch.arange(S, device=device).unsqueeze(0)
            x = model.token_emb(input_ids) + model.pos_emb(pos)
            x = model.dropout(x)
            mask = model.causal_mask[:, :, :S, :S]

            for layer_idx, layer in enumerate(model.layers):
                routing_info = routing_infos[layer_idx]
                neuron_indices = routing_info['neuron_indices']  # [B, S, k]
                neuron_weights = routing_info['neuron_weights']  # [B, S, k]

                # ⚡ Statistics accumulation (no CPU sync)
                layer_stats['routing_weights'][layer_idx] += neuron_weights.mean()

                # ⚡ Vectorized neuron usage update
                neuron_idx_flat = neuron_indices.reshape(-1)
                usage_update = torch.bincount(neuron_idx_flat, minlength=model.n_neurons)
                all_neuron_usage[f'layer_{layer_idx}'] += usage_update

                # Get router scores for analysis
                residual = x
                normed = layer.norm1(x)
                router_scores = layer.qkv_dynamic.W_router(normed)  # [B, S, n_neurons]
                layer_stats['router_scores'][layer_idx] += router_scores.abs().mean()

                # Compute attention for self-attention metric
                qkv_module = layer.qkv_dynamic
                attn_out, _ = qkv_module(normed, mask)

                # For self-attention metric, we need to recompute attention weights
                # Just use a simple approximation based on the output
                layer_stats['attn_self'][layer_idx] += attn_out.abs().mean()

                layer_stats['counts'][layer_idx] += 1

                # Forward through layer
                x = residual + layer.dropout(attn_out)
                residual = x
                normed = layer.norm2(x)
                ffn_out = layer.w_down(F.gelu(layer.w_up(normed)))
                x = residual + layer.dropout(ffn_out)

    # ⚡ Single CPU transfer at the end
    for key in ['router_scores', 'routing_weights', 'attn_self']:
        layer_stats[key] = (layer_stats[key] / layer_stats['counts']).cpu().numpy()

    # Compute results
    results = {}

    # Routing Analysis
    print("\n⭐ ROUTING ANALYSIS:")
    print(f"  Router score (mean): {layer_stats['router_scores'].mean():.4f}")
    print(f"  Routing weight (mean): {layer_stats['routing_weights'].mean():.4f}")

    results['routing'] = {
        'router_score_mean': float(layer_stats['router_scores'].mean()),
        'routing_weight_mean': float(layer_stats['routing_weights'].mean()),
    }

    # Neuron Usage Analysis
    print("\n⭐ NEURON USAGE:")
    for layer_idx in range(len(model.layers)):
        usage = all_neuron_usage[f'layer_{layer_idx}']
        total = usage.sum()
        usage_rate = (usage > 0).float().mean().item()
        gini = compute_gini(usage).item()

        print(f"\nLayer {layer_idx}:")
        print(f"  Active neurons: {(usage > 0).sum().item()}/{model.n_neurons} ({usage_rate*100:.1f}%)")
        print(f"  Usage Gini: {gini:.4f}")
        print(f"  Top-10 usage: {usage.topk(10)[0].cpu().numpy()}")

        results[f'usage_layer_{layer_idx}'] = {
            'active_neurons': int((usage > 0).sum().item()),
            'usage_rate': float(usage_rate),
            'gini': float(gini),
            'top10_usage': usage.topk(10)[0].cpu().numpy().tolist(),
        }

    return results


# ============================================================
# 4. Summary Metrics
# ============================================================

def compute_summary_metrics(all_results):
    """Compute summary statistics across all analyses"""
    print("\n" + "="*60)
    print("4. SUMMARY METRICS")
    print("="*60)

    summary = {}

    # Basis orthogonality
    if 'basis' in all_results:
        summary['basis_orthogonal'] = all_results['basis']['basis']['max_off_diagonal'] < 1e-5

    # Recipe diversity (average across Q/K/V/O and layers)
    recipe_entropies = []
    recipe_specializations = []
    if 'recipes' in all_results:
        for layer_key, layer_data in all_results['recipes'].items():
            if layer_key.startswith('layer_'):
                for recipe_type in ['Q', 'K', 'V', 'O']:
                    recipe_key = f'recipe_{recipe_type}'
                    if recipe_key in layer_data:
                        recipe_entropies.append(layer_data[recipe_key]['recipe_entropy_mean'])
                        recipe_specializations.append(layer_data[recipe_key]['neuron_specialization_mean'])

    summary['recipe_entropy_avg'] = float(np.mean(recipe_entropies)) if recipe_entropies else 0.0
    summary['recipe_specialization_avg'] = float(np.mean(recipe_specializations)) if recipe_specializations else 0.0

    # Neuron usage (average across layers)
    usage_rates = []
    gini_coeffs = []
    if 'runtime' in all_results:
        for key, value in all_results['runtime'].items():
            if key.startswith('usage_layer_'):
                usage_rates.append(value['usage_rate'])
                gini_coeffs.append(value['gini'])

    summary['avg_neuron_usage_rate'] = float(np.mean(usage_rates)) if usage_rates else 0.0
    summary['avg_usage_gini'] = float(np.mean(gini_coeffs)) if gini_coeffs else 0.0

    print(f"\n📊 Summary:")
    print(f"  Basis orthogonal: {summary.get('basis_orthogonal', False)}")
    print(f"  Avg recipe entropy: {summary['recipe_entropy_avg']:.4f}")
    print(f"  Avg recipe specialization: {summary['recipe_specialization_avg']:.4f}")
    print(f"  Avg neuron usage: {summary['avg_neuron_usage_rate']*100:.1f}%")
    print(f"  Avg usage Gini: {summary['avg_usage_gini']:.4f}")

    return summary


# ============================================================
# 5. W_Q/K/V Dynamics Analysis
# ============================================================

def analyze_w_dynamics(model, dataloader, device, max_batches=5):
    """Measure how different W_Q/K/V are across tokens"""
    print("\n" + "="*60)
    print("5. W_Q/K/V DYNAMICS ANALYSIS")
    print("="*60)

    model.eval()
    results = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="W Dynamics", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            B, S = input_ids.shape

            pos = torch.arange(S, device=device).unsqueeze(0)
            x = model.token_emb(input_ids) + model.pos_emb(pos)
            x = model.dropout(x)

            for layer_idx, layer in enumerate(model.layers):
                normed = layer.norm1(x)
                qkv = layer.qkv_dynamic

                # Perform routing
                scores = qkv.W_router(normed)
                topk_scores, topk_idx = torch.topk(scores, qkv.k, dim=-1)
                weights = F.softmax(topk_scores, dim=-1)

                # Get recipe
                recipe_Q = qkv.neuron_recipe_Q[topk_idx]
                token_recipe_Q = (recipe_Q * weights.unsqueeze(-1)).sum(dim=2)
                token_recipe_Q = F.softmax(token_recipe_Q, dim=-1)

                # Generate W_Q
                basis = qkv.shared_basis()
                W_Q = torch.einsum('bsn,ndr->bsdr', token_recipe_Q, basis)  # [B, S, D, rank]

                # Flatten W_Q and compute token-wise similarity
                W_Q_flat = W_Q.view(B * S, -1)  # [B*S, D*rank]
                W_Q_norm = F.normalize(W_Q_flat, dim=-1)

                # Cosine similarity between tokens (sampling)
                n_samples = min(100, B * S)
                indices = torch.randperm(B * S, device=device)[:n_samples]
                W_Q_sampled = W_Q_norm[indices]
                sim_matrix = W_Q_sampled @ W_Q_sampled.T

                # Off-diagonal similarity
                mask = ~torch.eye(n_samples, dtype=torch.bool, device=device)
                off_diag_sim = sim_matrix[mask]

                key = f'layer_{layer_idx}'
                if key not in results:
                    results[key] = {'w_q_sim': [], 'w_q_std': []}

                results[key]['w_q_sim'].append(off_diag_sim.mean().item())
                results[key]['w_q_std'].append(W_Q_flat.std(dim=0).mean().item())

                # Forward to next layer
                attn_out, _ = qkv(normed, model.causal_mask[:, :, :S, :S])
                x = x + layer.dropout(attn_out)
                x = x + layer.dropout(layer.w_down(F.gelu(layer.w_up(layer.norm2(x)))))

    # Output results
    print("\n⭐ W_Q DYNAMICS:")
    for layer_idx in range(len(model.layers)):
        key = f'layer_{layer_idx}'
        sim = np.mean(results[key]['w_q_sim'])
        std = np.mean(results[key]['w_q_std'])
        print(f"  Layer {layer_idx}: Token-wise W_Q similarity = {sim:.4f}, W_Q std = {std:.6f}")

        if sim > 0.95:
            print(f"    ⚠️  WARNING: W_Q nearly identical across tokens!")

    return results


# ============================================================
# 6. Basis Coverage Analysis
# ============================================================

def analyze_basis_coverage(model, dataloader, device, max_batches=5):
    """Analyze how well basis covers the space

    Supports both v7.5 (single basis) and v7.6 (separate basis_down/basis_up)
    """
    print("\n" + "="*60)
    print("6. BASIS COVERAGE ANALYSIS")
    print("="*60)

    # Detect model version and get basis
    is_v76 = hasattr(model.shared_basis, 'get_basis_down')
    if is_v76:
        basis = model.shared_basis.get_basis_down()  # [n_basis, D, rank]
        print("📌 Analyzing basis_down (v7.6)")
    else:
        basis = model.shared_basis.basis  # [n_basis, D, rank]
    n_basis, D, rank = basis.shape

    # 1. Flatten basis for analysis
    basis_flat = basis.view(n_basis, -1)  # [n_basis, D*rank]

    # SVD for effective rank
    U, S, V = torch.linalg.svd(basis_flat, full_matrices=False)

    # Effective rank (entropy-based)
    S_norm = S / S.sum()
    entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
    effective_rank = torch.exp(entropy).item()

    # Top-k singular value ratios
    top_k_ratios = []
    for k in [1, 5, 10, 20]:
        ratio = S[:k].sum() / S.sum()
        top_k_ratios.append((k, ratio.item()))

    print(f"\n📊 Basis Span Analysis:")
    print(f"  Effective rank: {effective_rank:.2f} / {n_basis}")
    print(f"  Singular value concentration:")
    for k, ratio in top_k_ratios:
        print(f"    Top-{k}: {ratio*100:.1f}%")

    # 2. How well token embeddings project onto basis space
    model.eval()
    projection_errors = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Coverage", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            x = model.token_emb(input_ids)  # [B, S, D]

            # Measure reconstruction error for each basis
            for i in range(n_basis):
                basis_i = basis[i]  # [D, rank]
                x_proj = x @ basis_i  # [B, S, rank]
                x_recon = x_proj @ basis_i.T  # [B, S, D]
                error = (x - x_recon).norm(dim=-1).mean()
                projection_errors.append(error.item())

    avg_error = np.mean(projection_errors)
    print(f"\n📊 Projection Quality:")
    print(f"  Avg reconstruction error per basis: {avg_error:.4f}")

    # 3. Basis diversity (span perspective)
    basis_centered = basis_flat - basis_flat.mean(dim=0, keepdim=True)
    cov = basis_centered.T @ basis_centered / n_basis
    cov_eigenvalues = torch.linalg.eigvalsh(cov)

    print(f"\n📊 Basis Diversity:")
    print(f"  Covariance top eigenvalue: {cov_eigenvalues[-1].item():.4f}")
    print(f"  Covariance eigenvalue ratio (top/bottom): {cov_eigenvalues[-1].item() / (cov_eigenvalues[0].item() + 1e-10):.2f}")

    return {
        'effective_rank': effective_rank,
        'sv_concentration': top_k_ratios,
        'projection_error': avg_error,
    }


# ============================================================
# 7. ⭐ O Projection Information Loss Analysis (KEY SUSPECT)
# ============================================================

def analyze_o_projection_loss(model, dataloader, device, max_batches=5):
    """Measure information loss in O projection

    Supports both v7.5 (shared basis.T) and v7.6 (independent basis_up)
    """
    print("\n" + "="*60)
    print("7. ⭐ O PROJECTION INFORMATION LOSS (KEY SUSPECT)")
    print("="*60)

    model.eval()
    results = {f'layer_{i}': [] for i in range(len(model.layers))}

    # Detect model version
    is_v76 = hasattr(model.shared_basis, 'get_basis_up')
    if is_v76:
        print("\n📌 Detected v7.6 model with independent basis_up")

        # Analyze basis down-up correspondence
        basis_down = model.shared_basis.get_basis_down()  # [n_basis, D, rank]
        basis_up = model.shared_basis.get_basis_up()      # [n_basis, rank, D]

        correspondence = []
        for i in range(model.n_basis):
            down_i = basis_down[i]  # [D, rank]
            up_i = basis_up[i]      # [rank, D]

            # Compare down.T with up
            down_T = down_i.T  # [rank, D]
            sim = F.cosine_similarity(down_T.flatten().unsqueeze(0),
                                       up_i.flatten().unsqueeze(0)).item()
            correspondence.append(sim)

        avg_correspondence = np.mean(correspondence)
        print(f"\n📊 Basis Down-Up Correspondence:")
        print(f"  Average similarity (down.T vs up): {avg_correspondence:.4f}")
        print(f"  (Lower = more independent learning, < 0.3 recommended)")
        results['basis_correspondence'] = avg_correspondence
    else:
        print("\n📌 Detected v7.5 model with shared basis.T")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="O Proj Loss", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            B, S = input_ids.shape

            pos = torch.arange(S, device=device).unsqueeze(0)
            x = model.token_emb(input_ids) + model.pos_emb(pos)
            mask = model.causal_mask[:, :, :S, :S]

            for layer_idx, layer in enumerate(model.layers):
                normed = layer.norm1(x)
                qkv = layer.qkv_dynamic

                # Replicate attention internals
                scores = qkv.W_router(normed)
                topk_scores, topk_idx = torch.topk(scores, qkv.k, dim=-1)
                weights = F.softmax(topk_scores, dim=-1)

                # Generate Q, K, V
                recipe_Q = F.softmax((qkv.neuron_recipe_Q[topk_idx] * weights.unsqueeze(-1)).sum(dim=2), dim=-1)
                recipe_K = F.softmax((qkv.neuron_recipe_K[topk_idx] * weights.unsqueeze(-1)).sum(dim=2), dim=-1)
                recipe_V = F.softmax((qkv.neuron_recipe_V[topk_idx] * weights.unsqueeze(-1)).sum(dim=2), dim=-1)

                # Get basis (v7.6: use get_basis_down, v7.5: use forward)
                if is_v76:
                    basis = qkv.shared_basis.get_basis_down()
                else:
                    basis = qkv.shared_basis()

                W_Q = torch.einsum('bsn,ndr->bsdr', recipe_Q, basis)
                W_K = torch.einsum('bsn,ndr->bsdr', recipe_K, basis)
                W_V = torch.einsum('bsn,ndr->bsdr', recipe_V, basis)

                Q = torch.einsum('bsd,bsdr->bsr', normed, W_Q)
                K = torch.einsum('bsd,bsdr->bsr', normed, W_K)
                V = torch.einsum('bsd,bsdr->bsr', normed, W_V)

                # Perform attention
                d_head = qkv.d_head
                n_heads = qkv.n_heads
                Q = Q.view(B, S, n_heads, d_head).transpose(1, 2)
                K = K.view(B, S, n_heads, d_head).transpose(1, 2)
                V = V.view(B, S, n_heads, d_head).transpose(1, 2)

                attn_scores = Q @ K.transpose(-2, -1) / (d_head ** 0.5)
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_out = attn_weights @ V  # [B, H, S, d_head]
                attn_out = attn_out.transpose(1, 2).reshape(B, S, qkv.basis_rank)  # [B, S, rank]

                # ⭐ KEY: Compare before and after O projection
                attn_out_before_O = attn_out.clone()  # [B, S, rank]

                # Perform O projection (v7.6: use get_basis_up, v7.5: use transpose)
                recipe_O = F.softmax((qkv.neuron_recipe_O[topk_idx] * weights.unsqueeze(-1)).sum(dim=2), dim=-1)
                if is_v76:
                    basis_up = qkv.shared_basis.get_basis_up()  # [n_basis, rank, D]
                else:
                    basis_up = qkv.shared_basis().transpose(-1, -2)  # [n_basis, rank, D]
                W_O = torch.einsum('bsn,nrd->bsrd', recipe_O, basis_up)  # [B, S, rank, D]
                attn_out_after_O = torch.einsum('bsr,bsrd->bsd', attn_out, W_O)  # [B, S, D]

                # Measure info loss 1: Reconstruction possibility after back-projection
                W_down = torch.einsum('bsn,ndr->bsdr', recipe_Q, basis)  # Use any down projection
                attn_reconstructed = torch.einsum('bsd,bsdr->bsr', attn_out_after_O, W_down)

                recon_error = (attn_out_before_O - attn_reconstructed).norm(dim=-1).mean()
                original_norm = attn_out_before_O.norm(dim=-1).mean()
                relative_error = recon_error / (original_norm + 1e-10)

                # Measure info loss 2: W_O effective rank
                W_O_flat = W_O.view(B * S, -1)
                W_O_sample = W_O_flat[torch.randperm(B * S, device=device)[:100]]
                _, S_vals, _ = torch.linalg.svd(W_O_sample, full_matrices=False)
                S_norm = S_vals / S_vals.sum()
                entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
                eff_rank = torch.exp(entropy).item()

                results[f'layer_{layer_idx}'].append({
                    'recon_error': recon_error.item(),
                    'relative_error': relative_error.item(),
                    'w_o_effective_rank': eff_rank,
                })

                # Forward to next layer
                x = x + layer.dropout(attn_out_after_O)
                x = x + layer.dropout(layer.w_down(F.gelu(layer.w_up(layer.norm2(x)))))

    # Output results
    print("\n⭐ O PROJECTION ANALYSIS:")
    for layer_idx in range(len(model.layers)):
        key = f'layer_{layer_idx}'
        data = results[key]
        avg_rel_error = np.mean([d['relative_error'] for d in data])
        avg_eff_rank = np.mean([d['w_o_effective_rank'] for d in data])

        print(f"\n  Layer {layer_idx}:")
        print(f"    Relative reconstruction error: {avg_rel_error:.4f}")
        print(f"    W_O effective rank: {avg_eff_rank:.2f}")

        if avg_rel_error > 0.3:
            print(f"    ⚠️  HIGH INFO LOSS - O projection losing significant information!")

    return results


# ============================================================
# 8. Attention Pattern Analysis
# ============================================================

def analyze_attention_patterns(model, dataloader, device, max_batches=5):
    """Analyze attention score distribution"""
    print("\n" + "="*60)
    print("8. ATTENTION PATTERN ANALYSIS")
    print("="*60)

    model.eval()
    results = {f'layer_{i}': {'entropy': [], 'sparsity': [], 'self_attn': []}
               for i in range(len(model.layers))}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Attn Patterns", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            B, S = input_ids.shape

            pos = torch.arange(S, device=device).unsqueeze(0)
            x = model.token_emb(input_ids) + model.pos_emb(pos)
            mask = model.causal_mask[:, :, :S, :S]

            for layer_idx, layer in enumerate(model.layers):
                normed = layer.norm1(x)
                qkv = layer.qkv_dynamic

                # Compute attention weights
                scores = qkv.W_router(normed)
                topk_scores, topk_idx = torch.topk(scores, qkv.k, dim=-1)
                weights = F.softmax(topk_scores, dim=-1)

                recipe_Q = F.softmax((qkv.neuron_recipe_Q[topk_idx] * weights.unsqueeze(-1)).sum(dim=2), dim=-1)
                recipe_K = F.softmax((qkv.neuron_recipe_K[topk_idx] * weights.unsqueeze(-1)).sum(dim=2), dim=-1)

                basis = qkv.shared_basis()
                W_Q = torch.einsum('bsn,ndr->bsdr', recipe_Q, basis)
                W_K = torch.einsum('bsn,ndr->bsdr', recipe_K, basis)

                Q = torch.einsum('bsd,bsdr->bsr', normed, W_Q)
                K = torch.einsum('bsd,bsdr->bsr', normed, W_K)

                d_head = qkv.d_head
                n_heads = qkv.n_heads
                Q = Q.view(B, S, n_heads, d_head).transpose(1, 2)
                K = K.view(B, S, n_heads, d_head).transpose(1, 2)

                attn_scores = Q @ K.transpose(-2, -1) / (d_head ** 0.5)
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, S, S]

                # Entropy (higher = more uniform)
                attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1)
                avg_entropy = attn_entropy.mean().item()

                # Sparsity (top-1 weight ratio)
                top1_weight = attn_weights.max(dim=-1)[0].mean().item()

                # Self-attention (diagonal ratio)
                diag_mask = torch.eye(S, device=device).bool()
                self_attn = attn_weights[:, :, diag_mask].mean().item()

                results[f'layer_{layer_idx}']['entropy'].append(avg_entropy)
                results[f'layer_{layer_idx}']['sparsity'].append(top1_weight)
                results[f'layer_{layer_idx}']['self_attn'].append(self_attn)

                # Forward to next layer
                attn_out, _ = qkv(normed, mask)
                x = x + layer.dropout(attn_out)
                x = x + layer.dropout(layer.w_down(F.gelu(layer.w_up(layer.norm2(x)))))

    print("\n📊 ATTENTION PATTERNS:")
    for layer_idx in range(len(model.layers)):
        key = f'layer_{layer_idx}'
        entropy = np.mean(results[key]['entropy'])
        sparsity = np.mean(results[key]['sparsity'])
        self_attn = np.mean(results[key]['self_attn'])

        print(f"  Layer {layer_idx}: entropy={entropy:.4f}, top1_weight={sparsity:.4f}, self_attn={self_attn:.4f}")

    return results


# ============================================================
# Main Analysis Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN v7.5 Checkpoint Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint folder')
    parser.add_argument('--max-batches', type=int, default=10,
                        help='Max batches for runtime analysis')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for data loading (default: 128, increase for faster GPU processing)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint) / 'best_model.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model
    from models import create_model_by_version

    # First, detect actual model version from checkpoint
    actual_version = checkpoint.get('model_version', 'unknown')
    print(f"Checkpoint model version: {actual_version}")

    # Get state_dict for version inference
    state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model'
    state_dict = checkpoint[state_dict_key]

    # Detect v7.6 from state_dict (has basis_down/basis_up instead of basis)
    has_split_basis = any('basis_down' in k or 'basis_up' in k for k in state_dict.keys())
    if has_split_basis:
        detected_version = '7.6'
        print("  Detected v7.6 from state_dict (split basis_down/basis_up)")
    elif 'layers.0.qkv_dynamic.neuron_recipe_Q' in state_dict:
        detected_version = '7.5'
        print("  Detected v7.5 from state_dict (Dynamic Q/K/V)")
    else:
        detected_version = actual_version if actual_version != 'unknown' else '7.5'

    # Use detected version (more reliable than config)
    model_version = detected_version

    # Get model config (with backward compatibility)
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        # Override model_version with detected version
        model_config['model_version'] = model_version
    else:
        # Infer model config from state_dict
        sample_layer = 'layers.0.qkv_dynamic'
        n_neurons = state_dict[f'{sample_layer}.neuron_recipe_Q'].shape[0]
        n_basis = state_dict[f'{sample_layer}.neuron_recipe_Q'].shape[1]

        model_config = {
            'vocab_size': state_dict['token_emb.weight'].shape[0],
            'd_model': state_dict['token_emb.weight'].shape[1],
            'n_layers': sum(1 for k in state_dict.keys() if k.startswith('layers.') and '.norm1.weight' in k),
            'n_neurons': n_neurons,
            'n_basis': n_basis,
            'model_version': model_version,
        }
        print(f"  Inferred config: {model_config}")

    # Create model
    print(f"\nCreating model v{model_version}...")
    model = create_model_by_version(model_version, model_config)

    # Load state dict
    state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model'
    state_dict = checkpoint[state_dict_key]

    # Handle torch.compile wrapper (remove _orig_mod. prefix)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  Removing torch.compile wrapper prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model: DAWN v{model.__version__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ⚡ Use torch.compile for faster inference (PyTorch 2.0+)
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        print("\n⚡ Compiling model with torch.compile for faster GPU execution...")
        print("   (Suppressing autotune logs...)")

        # Aggressive suppression of all compilation/autotune output
        import os
        import logging
        import sys
        import io

        # Set environment variables to suppress Triton/Inductor output
        os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
        os.environ['TRITON_PRINT_AUTOTUNING'] = '0'
        os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '0'

        # Suppress all logging from compilation-related modules
        logging.getLogger("torch._inductor").setLevel(logging.CRITICAL)
        logging.getLogger("torch._inductor.utils").setLevel(logging.CRITICAL)
        logging.getLogger("torch._inductor.compile_fx").setLevel(logging.CRITICAL)
        logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)
        logging.getLogger("torch._dynamo.output_graph").setLevel(logging.CRITICAL)

        # Disable autotune in inductor config
        try:
            import torch._inductor.config as inductor_config
            inductor_config.triton.autotune_at_compile_time = False
            inductor_config.trace.enabled = False
            inductor_config.trace.log_autotuning_results = False
            inductor_config.max_autotune = False
        except:
            pass

        try:
            # Redirect stdout/stderr during compilation to suppress print output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            try:
                # Use 'reduce-overhead' mode instead of 'max-autotune' to avoid autotune entirely
                model = torch.compile(model, mode='reduce-overhead')
            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            print("   ✅ Model compiled successfully!")
        except Exception as e:
            print(f"   ⚠️  Compilation failed: {e}")
            print("   Continuing with uncompiled model...")

    # Prepare dataloader
    from utils.data import TextDataset, collate_fn_dynamic_padding
    from torch.utils.data import DataLoader
    from functools import partial
    import pickle
    import os

    data_config = checkpoint.get('data_config', {
        'base_dir': '/content/drive/MyDrive/data',
        'val_file': 'validation/wikitext_5to1_texts.pkl'
    })

    # Load validation texts directly
    val_path = os.path.join(data_config['base_dir'], data_config['val_file'])
    print(f"\nLoading validation data from: {val_path}")

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found: {val_path}")

    with open(val_path, 'rb') as f:
        val_texts = pickle.load(f)

    print(f"Loaded {len(val_texts)} validation texts")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    max_seq_len = model_config.get('max_seq_len', 128)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=max_seq_len)

    # ⚡ GPU-optimized DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn_dynamic_padding, tokenizer=tokenizer),
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster CPU->GPU transfer
        prefetch_factor=2,  # Prefetch batches
    )

    # Run analysis
    all_results = {}

    print("\n" + "="*60)
    print("STARTING ANALYSIS")
    print("="*60)

    all_results['basis'] = analyze_basis_orthogonality(model)
    all_results['recipes'] = analyze_recipes(model)
    all_results['runtime'] = analyze_runtime_behavior(model, val_loader, device, args.max_batches)
    all_results['w_dynamics'] = analyze_w_dynamics(model, val_loader, device, args.max_batches)
    all_results['basis_coverage'] = analyze_basis_coverage(model, val_loader, device, args.max_batches)
    all_results['o_projection'] = analyze_o_projection_loss(model, val_loader, device, args.max_batches)
    all_results['attention'] = analyze_attention_patterns(model, val_loader, device, args.max_batches)
    all_results['summary'] = compute_summary_metrics(all_results)

    # Save results
    output_path = Path(args.checkpoint) / 'analysis_v75.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Analysis complete! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
