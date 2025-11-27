"""
DAWN v7.5/v7.6/v7.7/v7.8 Checkpoint Analysis
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

v7.7 특징 (개선):
- basis_qk (Q/K용)과 basis_vo (V/O용) 분리
- O는 basis_vo.T 사용 (V와 대칭)
- Gradient 균형: QK 2개, VO 2개

v7.8 특징 (개선):
- Basis 제거: recipe @ basis 혼합 없음
- 뉴런별 독립 W_Q/K/V/O: 각 뉴런이 완전한 projection 행렬 소유
- 직교 초기화: 각 뉴런의 W에 orthogonal init
- Condition number 폭발 방지

핵심 지표 해석 가이드:
| 지표                    | 정상 범위   | 문제 신호                      |
|------------------------|------------|-------------------------------|
| W_Q token similarity   | < 0.7      | > 0.9 → 동적 생성 무의미        |
| Basis effective rank   | > 20       | < 10 → 중복된 basis           |
| O proj variance ratio  | 0.5 ~ 2.0  | < 0.5 collapse, > 2.0 explode |
| O proj condition #     | < 100      | > 100 → ill-conditioned W_O   |
| Attention entropy      | 1.5 ~ 3.0  | < 1.0 또는 > 4.0              |
| Basis correspondence   | < 0.3      | > 0.7 → basis_up/vo 미학습     |

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


def detect_model_version(model):
    """Detect model version from shared_basis structure

    Returns:
        str: "7.8", "7.7", "7.6", or "7.5"
    """
    sb = model.shared_basis
    # v7.8: NeuronBank with W_Q/K/V/O (no basis)
    if hasattr(sb, 'W_Q') and hasattr(sb, 'W_K'):
        return "7.8"
    elif hasattr(sb, 'basis_qk'):
        return "7.7"
    elif hasattr(sb, 'get_basis_down'):
        return "7.6"
    else:
        return "7.5"


def get_basis_for_qkv(model):
    """Get basis used for Q/K/V projection

    Returns:
        basis: Tensor [n_basis/n_neurons, D, rank]
        name: str describing the basis
    """
    version = detect_model_version(model)
    sb = model.shared_basis

    if version == "7.8":
        # v7.8: No basis, return W_Q directly (뉴런별 독립 W)
        return sb.W_Q, "neuron_W_Q"
    elif version == "7.7":
        # v7.7: Q/K use basis_qk, V uses basis_vo
        return sb.basis_qk, "basis_qk"
    elif version == "7.6":
        return sb.get_basis_down(), "basis_down"
    else:
        return sb.basis, "basis"


def get_basis_for_o(model):
    """Get basis used for O projection

    Returns:
        basis: Tensor [n_basis/n_neurons, rank, D]
        name: str describing the basis
    """
    version = detect_model_version(model)
    sb = model.shared_basis

    if version == "7.8":
        # v7.8: No basis, return W_O directly (뉴런별 독립 W)
        return sb.W_O, "neuron_W_O"
    elif version == "7.7":
        # v7.7: O uses basis_vo.T
        return sb.get_basis_o(), "basis_vo.T"
    elif version == "7.6":
        return sb.get_basis_up(), "basis_up"
    else:
        return sb.basis.transpose(-1, -2), "basis.T"


# ============================================================
# 1. Basis Orthogonality Analysis
# ============================================================

def analyze_basis_orthogonality(model):
    """Verify that basis is perfectly orthogonal

    Supports v7.5 (single basis), v7.6 (basis_down/basis_up), v7.7 (basis_qk/basis_vo),
    and v7.8 (neuron W_Q/K/V/O orthogonality)
    """
    print("\n" + "="*60)
    print("1. BASIS ORTHOGONALITY VERIFICATION")
    print("="*60)

    version = detect_model_version(model)
    n_basis = model.n_basis
    results = {}

    def check_orthogonality(basis, name, n_items=None):
        """Check orthogonality for a given basis"""
        max_errors = []
        mean_errors = []
        if n_items is None:
            n_items = n_basis

        for i in range(n_items):
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

    if version == "7.8":
        print(f"\n📌 v7.8 model: Checking neuron W_Q/K/V/O orthogonality")
        print(f"   (뉴런별 독립 W, basis mixing 없음)")

        nb = model.shared_basis  # NeuronBank
        n_neurons = nb.n_neurons

        # Check each W type
        for W_name, W in [('W_Q', nb.W_Q), ('W_K', nb.W_K), ('W_V', nb.W_V), ('W_O', nb.W_O)]:
            result = check_orthogonality(W, W_name, n_items=n_neurons)
            results[W_name] = result
            print(f"\n{W_name} orthogonality (across {n_neurons} neurons):")
            print(f"  Max off-diagonal: {result['max_off_diagonal']:.2e}")
            print(f"  Mean off-diagonal: {result['mean_off_diagonal']:.2e}")

        max_error = max(results[k]['max_off_diagonal'] for k in ['W_Q', 'W_K', 'W_V', 'W_O'])
        results['basis'] = {'max_off_diagonal': max_error}  # For compatibility

    elif version == "7.7":
        print(f"\n📌 v7.7 model: Checking both basis_qk and basis_vo")

        # Check basis_qk
        basis_qk = model.shared_basis.basis_qk
        results['basis_qk'] = check_orthogonality(basis_qk, 'basis_qk')
        print(f"\nBasis_QK orthogonality:")
        print(f"  Max off-diagonal: {results['basis_qk']['max_off_diagonal']:.2e}")
        print(f"  Mean off-diagonal: {results['basis_qk']['mean_off_diagonal']:.2e}")

        # Check basis_vo
        basis_vo = model.shared_basis.basis_vo
        results['basis_vo'] = check_orthogonality(basis_vo, 'basis_vo')
        print(f"\nBasis_VO orthogonality:")
        print(f"  Max off-diagonal: {results['basis_vo']['max_off_diagonal']:.2e}")
        print(f"  Mean off-diagonal: {results['basis_vo']['mean_off_diagonal']:.2e}")

        max_error = max(results['basis_qk']['max_off_diagonal'],
                       results['basis_vo']['max_off_diagonal'])
        results['basis'] = {'max_off_diagonal': max_error}

    elif version == "7.6":
        print(f"\n📌 v7.6 model: Checking both basis_down and basis_up")

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

    version = detect_model_version(model)

    # v7.8: No recipes - analyze neuron W matrices directly
    if version == "7.8":
        print(f"\n📌 v7.8 model: No recipes (뉴런별 독립 W)")
        print(f"   대신 뉴런 W 행렬 간 유사도 분석")

        nb = model.shared_basis  # NeuronBank
        n_neurons = nb.n_neurons
        results = {}

        # Analyze W matrix similarity between neurons
        for W_name, W in [('W_Q', nb.W_Q), ('W_K', nb.W_K), ('W_V', nb.W_V), ('W_O', nb.W_O)]:
            # W: [n_neurons, D, rank] or [n_neurons, rank, D]
            W_flat = W.view(n_neurons, -1)  # [n_neurons, D*rank]
            W_norm = F.normalize(W_flat, dim=-1)
            sim_matrix = W_norm @ W_norm.T  # [n_neurons, n_neurons]

            # Off-diagonal similarity (neuron 간 유사도)
            mask = ~torch.eye(n_neurons, dtype=torch.bool, device=W.device)
            off_diag_sim = sim_matrix[mask].mean().item()
            off_diag_std = sim_matrix[mask].std().item()

            results[W_name] = {
                'neuron_similarity_mean': off_diag_sim,
                'neuron_similarity_std': off_diag_std,
            }
            print(f"\n  {W_name} neuron similarity: {off_diag_sim:.4f} ± {off_diag_std:.4f}")

        # Cross-type similarity (Q vs K, Q vs V, etc.)
        W_types = {'Q': nb.W_Q, 'K': nb.W_K, 'V': nb.W_V, 'O': nb.W_O}
        cross_sim = {}
        pairs = [('Q', 'K'), ('Q', 'V'), ('Q', 'O'), ('K', 'V'), ('K', 'O'), ('V', 'O')]
        for name1, name2 in pairs:
            W1 = W_types[name1]
            W2 = W_types[name2]
            # Per-neuron similarity
            W1_flat = W1.view(n_neurons, -1)
            W2_flat = W2.view(n_neurons, -1)
            W1_norm = F.normalize(W1_flat, dim=-1)
            W2_norm = F.normalize(W2_flat, dim=-1)
            sim = (W1_norm * W2_norm).sum(dim=-1).mean().item()
            cross_sim[f'{name1}_{name2}'] = sim

        results['cross_similarity'] = cross_sim
        print(f"\n  Cross-type similarity:")
        print(f"    Q-K: {cross_sim['Q_K']:.4f}, Q-V: {cross_sim['Q_V']:.4f}, Q-O: {cross_sim['Q_O']:.4f}")
        print(f"    K-V: {cross_sim['K_V']:.4f}, K-O: {cross_sim['K_O']:.4f}, V-O: {cross_sim['V_O']:.4f}")

        return results

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

    # Detect model version from first layer
    qkv_sample = model.layers[0].qkv_dynamic
    is_v78 = hasattr(qkv_sample, 'neuron_bank') and hasattr(qkv_sample.neuron_bank, 'W_Q')

    if is_v78:
        print("📌 v7.8 model: Analyzing effective W_Q from weighted neuron selection")

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

                if is_v78:
                    # v7.8: Direct W matrices per neuron
                    # W_Q: [n_neurons, D, rank]
                    # topk_idx: [B, S, k]
                    nb = qkv.neuron_bank
                    D, rank = nb.W_Q.shape[1], nb.W_Q.shape[2]

                    # Gather selected neurons' W_Q
                    idx_expanded = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(B, S, qkv.k, D, rank)
                    W_Q_selected = nb.W_Q.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1, -1).gather(2, idx_expanded)
                    # [B, S, k, D, rank]

                    # Weighted sum
                    W_Q = (W_Q_selected * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)  # [B, S, D, rank]
                else:
                    # v7.5/7.6/7.7: Recipe-based
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

    Supports v7.5 (single basis), v7.6 (basis_down/basis_up), v7.7 (basis_qk/basis_vo),
    and v7.8 (neuron W_Q - analyzes neuron diversity instead of basis)
    """
    print("\n" + "="*60)
    print("6. BASIS COVERAGE ANALYSIS")
    print("="*60)

    # Detect model version and get basis using helper
    version = detect_model_version(model)
    basis, basis_name = get_basis_for_qkv(model)
    print(f"📌 Analyzing {basis_name} (v{version})")

    if version == "7.8":
        print(f"   (v7.8: 뉴런별 W 행렬 분석 - basis 없음)")

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

    Supports v7.5 (shared basis.T), v7.6 (independent basis_up), v7.7 (basis_qk/basis_vo),
    and v7.8 (neuron W_Q/K/V/O - no basis mixing)
    """
    print("\n" + "="*60)
    print("7. ⭐ O PROJECTION INFORMATION LOSS (KEY SUSPECT)")
    print("="*60)

    model.eval()
    results = {f'layer_{i}': [] for i in range(len(model.layers))}

    # Detect model version
    version = detect_model_version(model)

    if version == "7.8":
        print("\n📌 Detected v7.8 model with independent neuron W_Q/K/V/O")
        print(f"   (뉴런별 독립 W, basis mixing 없음)")
        results['basis_correspondence'] = None  # N/A for v7.8

    elif version == "7.7":
        print("\n📌 Detected v7.7 model with basis_qk (Q/K) and basis_vo (V/O)")

        # Analyze basis_vo - O uses basis_vo.T, so they share the same basis
        basis_vo = model.shared_basis.basis_vo  # [n_basis, D, rank]
        basis_o = model.shared_basis.get_basis_o()  # [n_basis, rank, D] = basis_vo.T

        print(f"\n📊 V-O Basis Relationship (v7.7):")
        print(f"  V uses basis_vo: {list(basis_vo.shape)}")
        print(f"  O uses basis_vo.T: {list(basis_o.shape)}")
        print(f"  (Symmetric design - gradient balance 2:2)")

        # For v7.7, correspondence should be 1.0 by design
        results['basis_correspondence'] = 1.0  # By design V and O share basis_vo

    elif version == "7.6":
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

                # Get W_Q, W_K, W_V based on version
                if version == "7.8":
                    # v7.8: Direct neuron W matrices (no recipe/basis)
                    nb = qkv.neuron_bank
                    W_Q_neurons = nb.get_W_Q(topk_idx)  # [B, S, k, D, rank]
                    W_K_neurons = nb.get_W_K(topk_idx)
                    W_V_neurons = nb.get_W_V(topk_idx)

                    # Weighted average of neuron W matrices
                    weights_exp = weights.unsqueeze(-1).unsqueeze(-1)  # [B, S, k, 1, 1]
                    W_Q = (W_Q_neurons * weights_exp).sum(dim=2)  # [B, S, D, rank]
                    W_K = (W_K_neurons * weights_exp).sum(dim=2)
                    W_V = (W_V_neurons * weights_exp).sum(dim=2)
                else:
                    # Generate Q, K, V recipes (v7.5/7.6/7.7)
                    recipe_Q = F.softmax((qkv.neuron_recipe_Q[topk_idx] * weights.unsqueeze(-1)).sum(dim=2), dim=-1)
                    recipe_K = F.softmax((qkv.neuron_recipe_K[topk_idx] * weights.unsqueeze(-1)).sum(dim=2), dim=-1)
                    recipe_V = F.softmax((qkv.neuron_recipe_V[topk_idx] * weights.unsqueeze(-1)).sum(dim=2), dim=-1)

                    # Get basis based on version
                    if version == "7.7":
                        basis_qk = qkv.shared_basis.basis_qk  # Q/K share this
                        basis_vo = qkv.shared_basis.basis_vo  # V uses this
                        W_Q = torch.einsum('bsn,ndr->bsdr', recipe_Q, basis_qk)
                        W_K = torch.einsum('bsn,ndr->bsdr', recipe_K, basis_qk)
                        W_V = torch.einsum('bsn,ndr->bsdr', recipe_V, basis_vo)
                    elif version == "7.6":
                        basis = qkv.shared_basis.get_basis_down()
                        W_Q = torch.einsum('bsn,ndr->bsdr', recipe_Q, basis)
                        W_K = torch.einsum('bsn,ndr->bsdr', recipe_K, basis)
                        W_V = torch.einsum('bsn,ndr->bsdr', recipe_V, basis)
                    else:
                        basis = qkv.shared_basis()
                        W_Q = torch.einsum('bsn,ndr->bsdr', recipe_Q, basis)
                        W_K = torch.einsum('bsn,ndr->bsdr', recipe_K, basis)
                        W_V = torch.einsum('bsn,ndr->bsdr', recipe_V, basis)

                # ============================================================
                # STAGE 0: Input (normed x)
                # ============================================================
                var_input = normed.var(dim=-1).mean()  # [D] 공간 분산

                # ============================================================
                # STAGE 1: Q/K/V Projection (x → Q, K, V)
                # ============================================================
                Q = torch.einsum('bsd,bsdr->bsr', normed, W_Q)
                K = torch.einsum('bsd,bsdr->bsr', normed, W_K)
                V = torch.einsum('bsd,bsdr->bsr', normed, W_V)  # [B, S, rank]

                # Q projection analysis
                var_after_q = Q.var(dim=-1).mean()
                var_ratio_q = (var_after_q / (var_input + 1e-10)).item()

                W_Q_flat = W_Q.view(B * S, -1)
                W_Q_sample = W_Q_flat[torch.randperm(B * S, device=device)[:min(100, B * S)]]
                _, S_vals_q, _ = torch.linalg.svd(W_Q_sample, full_matrices=False)
                cond_num_q = (S_vals_q[0] / (S_vals_q[-1] + 1e-10)).item()
                # W_Q effective rank
                S_norm_q = S_vals_q / S_vals_q.sum()
                entropy_q = -torch.sum(S_norm_q * torch.log(S_norm_q + 1e-10))
                eff_rank_q = torch.exp(entropy_q).item()

                # K projection analysis
                var_after_k = K.var(dim=-1).mean()
                var_ratio_k = (var_after_k / (var_input + 1e-10)).item()

                W_K_flat = W_K.view(B * S, -1)
                W_K_sample = W_K_flat[torch.randperm(B * S, device=device)[:min(100, B * S)]]
                _, S_vals_k, _ = torch.linalg.svd(W_K_sample, full_matrices=False)
                cond_num_k = (S_vals_k[0] / (S_vals_k[-1] + 1e-10)).item()
                # W_K effective rank
                S_norm_k = S_vals_k / S_vals_k.sum()
                entropy_k = -torch.sum(S_norm_k * torch.log(S_norm_k + 1e-10))
                eff_rank_k = torch.exp(entropy_k).item()

                # V projection analysis
                var_after_v = V.var(dim=-1).mean()
                var_ratio_v = (var_after_v / (var_input + 1e-10)).item()

                W_V_flat = W_V.view(B * S, -1)
                W_V_sample = W_V_flat[torch.randperm(B * S, device=device)[:min(100, B * S)]]
                _, S_vals_v, _ = torch.linalg.svd(W_V_sample, full_matrices=False)
                cond_num_v = (S_vals_v[0] / (S_vals_v[-1] + 1e-10)).item()
                # W_V effective rank
                S_norm_v = S_vals_v / S_vals_v.sum()
                entropy_v = -torch.sum(S_norm_v * torch.log(S_norm_v + 1e-10))
                eff_rank_v = torch.exp(entropy_v).item()

                # ============================================================
                # Token Similarity Analysis (Q/K/V)
                # ============================================================
                # Compute cosine similarity between tokens after projection
                # High similarity = tokens are not distinguishable (collapse)

                def compute_token_similarity(X):
                    """Compute avg cosine similarity between tokens"""
                    # X: [B, S, rank]
                    X_flat = X.view(B * S, -1)  # [B*S, rank]
                    X_norm = F.normalize(X_flat, dim=-1)

                    # Sample tokens for efficiency
                    n_samples = min(200, B * S)
                    sample_idx = torch.randperm(B * S, device=device)[:n_samples]
                    X_sampled = X_norm[sample_idx]

                    # Compute pairwise similarity
                    sim_matrix = X_sampled @ X_sampled.T  # [n_samples, n_samples]

                    # Get off-diagonal mean (exclude self-similarity)
                    mask_diag = ~torch.eye(n_samples, dtype=torch.bool, device=device)
                    off_diag_sim = sim_matrix[mask_diag].mean().item()
                    return off_diag_sim

                token_sim_q = compute_token_similarity(Q)
                token_sim_k = compute_token_similarity(K)
                token_sim_v = compute_token_similarity(V)

                # ============================================================
                # STAGE 2: Attention Mixing (V → Attention(V))
                # ============================================================
                d_head = qkv.d_head
                n_heads = qkv.n_heads
                Q_heads = Q.view(B, S, n_heads, d_head).transpose(1, 2)
                K_heads = K.view(B, S, n_heads, d_head).transpose(1, 2)
                V_heads = V.view(B, S, n_heads, d_head).transpose(1, 2)

                attn_scores = Q_heads @ K_heads.transpose(-2, -1) / (d_head ** 0.5)
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                attn_weights = F.softmax(attn_scores, dim=-1)

                # Attention entropy (정보 혼합 정도)
                attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1).mean().item()

                attn_out = attn_weights @ V_heads  # [B, H, S, d_head]
                attn_out = attn_out.transpose(1, 2).reshape(B, S, qkv.basis_rank)  # [B, S, rank]

                var_after_attn = attn_out.var(dim=-1).mean()  # Attention 후 분산
                var_ratio_attn = (var_after_attn / (var_after_v + 1e-10)).item()

                # ============================================================
                # STAGE 3: O Projection (Attention → Output)
                # ============================================================
                attn_out_before_O = attn_out.clone()  # [B, S, rank]

                if version == "7.8":
                    # v7.8: Direct neuron W_O matrices
                    W_O_neurons = nb.get_W_O(topk_idx)  # [B, S, k, rank, D]
                    W_O = (W_O_neurons * weights_exp).sum(dim=2)  # [B, S, rank, D]
                else:
                    recipe_O = F.softmax((qkv.neuron_recipe_O[topk_idx] * weights.unsqueeze(-1)).sum(dim=2), dim=-1)
                    if version == "7.7":
                        basis_up = qkv.shared_basis.get_basis_o()  # [n_basis, rank, D] = basis_vo.T
                    elif version == "7.6":
                        basis_up = qkv.shared_basis.get_basis_up()  # [n_basis, rank, D]
                    else:
                        basis_up = qkv.shared_basis().transpose(-1, -2)  # [n_basis, rank, D]
                    W_O = torch.einsum('bsn,nrd->bsrd', recipe_O, basis_up)  # [B, S, rank, D]
                attn_out_after_O = torch.einsum('bsr,bsrd->bsd', attn_out, W_O)  # [B, S, D]

                var_after_o = attn_out_after_O.var(dim=-1).mean()  # O projection 후 분산
                var_ratio_o = (var_after_o / (var_after_attn + 1e-10)).item()

                # W_O condition number
                W_O_flat = W_O.view(B * S, -1)
                W_O_sample = W_O_flat[torch.randperm(B * S, device=device)[:min(100, B * S)]]
                _, S_vals_o, _ = torch.linalg.svd(W_O_sample, full_matrices=False)
                cond_num_o = (S_vals_o[0] / (S_vals_o[-1] + 1e-10)).item()

                # W_O effective rank
                S_norm_o = S_vals_o / S_vals_o.sum()
                entropy_o = -torch.sum(S_norm_o * torch.log(S_norm_o + 1e-10))
                eff_rank_o = torch.exp(entropy_o).item()

                # ============================================================
                # FULL PIPELINE: x → V → Attn → O
                # ============================================================
                var_ratio_total = (var_after_o / (var_input + 1e-10)).item()

                results[f'layer_{layer_idx}'].append({
                    # Stage 1: Q/K/V projection
                    'var_ratio_q': var_ratio_q,
                    'cond_num_q': cond_num_q,
                    'eff_rank_q': eff_rank_q,
                    'token_sim_q': token_sim_q,
                    'var_ratio_k': var_ratio_k,
                    'cond_num_k': cond_num_k,
                    'eff_rank_k': eff_rank_k,
                    'token_sim_k': token_sim_k,
                    'var_ratio_v': var_ratio_v,
                    'cond_num_v': cond_num_v,
                    'eff_rank_v': eff_rank_v,
                    'token_sim_v': token_sim_v,
                    # Stage 2: Attention
                    'var_ratio_attn': var_ratio_attn,
                    'attn_entropy': attn_entropy,
                    # Stage 3: O projection
                    'var_ratio_o': var_ratio_o,
                    'cond_num_o': cond_num_o,
                    'eff_rank_o': eff_rank_o,
                    # Full pipeline
                    'var_ratio_total': var_ratio_total,
                })

                # Forward to next layer
                x = x + layer.dropout(attn_out_after_O)
                x = x + layer.dropout(layer.w_down(F.gelu(layer.w_up(layer.norm2(x)))))

    # Output results
    print("\n⭐ FULL PIPELINE ANALYSIS: x → Q/K/V → Attention → O")
    print("\n  Pipeline stages:")
    print("    Stage 1: Q/K/V Projection (x → Q, K, V)")
    print("    Stage 2: Attention Mixing (V → Attn(V))")
    print("    Stage 3: O Projection (Attn → Output)")
    print("\n  Metrics explanation:")
    print("    - var_ratio: var(after)/var(before), ~1 = good preservation")
    print("    - cond_num: σ_max/σ_min, lower = better conditioned")
    print("    - token_sim: cosine similarity between tokens, <0.7 = distinguishable")
    print("    - attn_entropy: higher = more uniform attention")

    for layer_idx in range(len(model.layers)):
        key = f'layer_{layer_idx}'
        data = results[key]

        # Aggregate metrics - Q/K/V
        avg_var_q = np.mean([d['var_ratio_q'] for d in data])
        avg_cond_q = np.mean([d['cond_num_q'] for d in data])
        avg_eff_q = np.mean([d['eff_rank_q'] for d in data])
        avg_sim_q = np.mean([d['token_sim_q'] for d in data])
        avg_var_k = np.mean([d['var_ratio_k'] for d in data])
        avg_cond_k = np.mean([d['cond_num_k'] for d in data])
        avg_eff_k = np.mean([d['eff_rank_k'] for d in data])
        avg_sim_k = np.mean([d['token_sim_k'] for d in data])
        avg_var_v = np.mean([d['var_ratio_v'] for d in data])
        avg_cond_v = np.mean([d['cond_num_v'] for d in data])
        avg_eff_v = np.mean([d['eff_rank_v'] for d in data])
        avg_sim_v = np.mean([d['token_sim_v'] for d in data])
        # Attention
        avg_var_attn = np.mean([d['var_ratio_attn'] for d in data])
        avg_entropy = np.mean([d['attn_entropy'] for d in data])
        # O projection
        avg_var_o = np.mean([d['var_ratio_o'] for d in data])
        avg_cond_o = np.mean([d['cond_num_o'] for d in data])
        avg_eff_rank = np.mean([d['eff_rank_o'] for d in data])
        avg_total = np.mean([d['var_ratio_total'] for d in data])

        print(f"\n  Layer {layer_idx}:")
        print(f"    ┌──────────────────────────────────────────────────────────────────────────")

        # Stage 1: Q/K/V Projection comparison
        print(f"    │ [Stage 1: Q/K/V Projection]")

        status_q = "⚠️ COLLAPSE" if avg_var_q < 0.3 else ("⚠️ EXPLODE" if avg_var_q > 3.0 else "✓")
        cond_q_status = "⚠️" if avg_cond_q > 100 else ""
        sim_q_status = "⚠️" if avg_sim_q > 0.9 else ""
        print(f"    │   Q: var={avg_var_q:.3f} {status_q}  cond={avg_cond_q:.1f} {cond_q_status}  eff_rank={avg_eff_q:.1f}  token_sim={avg_sim_q:.3f} {sim_q_status}")

        status_k = "⚠️ COLLAPSE" if avg_var_k < 0.3 else ("⚠️ EXPLODE" if avg_var_k > 3.0 else "✓")
        cond_k_status = "⚠️" if avg_cond_k > 100 else ""
        sim_k_status = "⚠️" if avg_sim_k > 0.9 else ""
        print(f"    │   K: var={avg_var_k:.3f} {status_k}  cond={avg_cond_k:.1f} {cond_k_status}  eff_rank={avg_eff_k:.1f}  token_sim={avg_sim_k:.3f} {sim_k_status}")

        status_v = "⚠️ COLLAPSE" if avg_var_v < 0.3 else ("⚠️ EXPLODE" if avg_var_v > 3.0 else "✓")
        cond_v_status = "⚠️" if avg_cond_v > 100 else ""
        sim_v_status = "⚠️" if avg_sim_v > 0.9 else ""
        print(f"    │   V: var={avg_var_v:.3f} {status_v}  cond={avg_cond_v:.1f} {cond_v_status}  eff_rank={avg_eff_v:.1f}  token_sim={avg_sim_v:.3f} {sim_v_status}")

        # Q/K/V comparison summary
        qkv_vars = [('Q', avg_var_q), ('K', avg_var_k), ('V', avg_var_v)]
        min_proj = min(qkv_vars, key=lambda x: x[1])
        max_proj = max(qkv_vars, key=lambda x: x[1])
        if max_proj[1] / (min_proj[1] + 1e-10) > 2.0:
            print(f"    │   ⚠️  Var imbalance: {max_proj[0]} >> {min_proj[0]} (ratio={max_proj[1]/min_proj[1]:.2f}x)")

        # Token similarity warning
        max_sim = max(avg_sim_q, avg_sim_k, avg_sim_v)
        if max_sim > 0.9:
            worst = 'Q' if avg_sim_q == max_sim else ('K' if avg_sim_k == max_sim else 'V')
            print(f"    │   ⚠️  High token similarity in {worst} ({max_sim:.3f}) - tokens not distinguishable!")

        print(f"    ├─────────────────────────────────────────────────────────────────")

        # Stage 2: Attention
        print(f"    │ [Stage 2: Attention]")
        status_attn = "⚠️ COLLAPSE" if avg_var_attn < 0.3 else ("⚠️ EXPLODE" if avg_var_attn > 3.0 else "✓")
        print(f"    │   var_ratio={avg_var_attn:.4f} {status_attn}  entropy={avg_entropy:.2f}")

        print(f"    ├─────────────────────────────────────────────────────────────────")

        # Stage 3: O Projection
        print(f"    │ [Stage 3: O Projection]")
        status_o = "⚠️ COLLAPSE" if avg_var_o < 0.3 else ("⚠️ EXPLODE" if avg_var_o > 3.0 else "✓")
        cond_o_status = "⚠️" if avg_cond_o > 100 else ""
        print(f"    │   var_ratio={avg_var_o:.4f} {status_o}  cond_num={avg_cond_o:.1f} {cond_o_status}  eff_rank={avg_eff_rank:.1f}")

        # Total pipeline
        print(f"    └─────────────────────────────────────────────────────────────────")
        status_total = "⚠️ COLLAPSE" if avg_total < 0.1 else ("⚠️ EXPLODE" if avg_total > 10.0 else "✓")
        print(f"      [TOTAL] x→O var_ratio={avg_total:.4f} {status_total}")

        # Identify bottleneck - check all stages
        bottleneck = None
        min_var = min(avg_var_q, avg_var_k, avg_var_v)
        if min_var < 0.3:
            if avg_var_q == min_var:
                bottleneck = "Q Projection"
            elif avg_var_k == min_var:
                bottleneck = "K Projection"
            else:
                bottleneck = "V Projection"
        elif avg_var_attn < 0.3:
            bottleneck = "Attention"
        elif avg_var_o < 0.3:
            bottleneck = "O Projection"

        if bottleneck:
            print(f"      🎯 BOTTLENECK: {bottleneck}")

    return results


# ============================================================
# 7.5. ⭐ Mixing Stage Condition Analysis (NEW)
# ============================================================

def analyze_mixing_condition(model, dataloader, device, max_batches=3):
    """Analyze where condition number explosion happens

    Compares condition numbers at each mixing stage:
    1. Single basis: Each basis_i has its own condition
    2. Single neuron recipe → W: recipe selects/combines basis
    3. Mixed neurons → W: topk neurons mixed together

    Goal: Identify if explosion is from neuron mixing or basis mixing
    """
    print("\n" + "="*60)
    print("7.5. ⭐ MIXING STAGE CONDITION ANALYSIS")
    print("="*60)

    model.eval()
    version = detect_model_version(model)
    results = {f'layer_{i}': [] for i in range(len(model.layers))}

    # ============================================================
    # STAGE A: Single Basis/Neuron Condition Numbers
    # ============================================================
    print("\n📊 [Stage A] Single Basis/Neuron Condition Numbers")

    if version == "7.8":
        print(f"  (v7.8: 뉴런별 W 조건수 - basis 없음)")
        nb = model.shared_basis  # NeuronBank
        n_neurons = nb.n_neurons

        cond_per_neuron = {'W_Q': [], 'W_K': [], 'W_V': [], 'W_O': []}
        for i in range(n_neurons):
            for name, W in [('W_Q', nb.W_Q), ('W_K', nb.W_K), ('W_V', nb.W_V), ('W_O', nb.W_O)]:
                _, s, _ = torch.linalg.svd(W[i], full_matrices=False)
                cond_per_neuron[name].append((s[0] / (s[-1] + 1e-10)).item())

        for name in ['W_Q', 'W_K', 'W_V', 'W_O']:
            conds = cond_per_neuron[name]
            print(f"  {name}: mean={np.mean(conds):.2f}, max={np.max(conds):.2f}, min={np.min(conds):.2f}")
        results['single_neuron_cond'] = cond_per_neuron

    elif version == "7.7":
        basis_qk = model.shared_basis.basis_qk  # [n_basis, D, rank]
        basis_vo = model.shared_basis.basis_vo

        cond_per_basis_qk = []
        cond_per_basis_vo = []
        for i in range(model.n_basis):
            # basis_i: [D, rank] - condition of this projection matrix
            _, s_qk, _ = torch.linalg.svd(basis_qk[i], full_matrices=False)
            _, s_vo, _ = torch.linalg.svd(basis_vo[i], full_matrices=False)
            cond_per_basis_qk.append((s_qk[0] / (s_qk[-1] + 1e-10)).item())
            cond_per_basis_vo.append((s_vo[0] / (s_vo[-1] + 1e-10)).item())

        print(f"  basis_qk: mean={np.mean(cond_per_basis_qk):.2f}, max={np.max(cond_per_basis_qk):.2f}, min={np.min(cond_per_basis_qk):.2f}")
        print(f"  basis_vo: mean={np.mean(cond_per_basis_vo):.2f}, max={np.max(cond_per_basis_vo):.2f}, min={np.min(cond_per_basis_vo):.2f}")
        results['single_basis_cond_qk'] = cond_per_basis_qk
        results['single_basis_cond_vo'] = cond_per_basis_vo

    elif version == "7.6":
        basis_down = model.shared_basis.get_basis_down()
        basis_up = model.shared_basis.get_basis_up()

        cond_per_basis_down = []
        cond_per_basis_up = []
        for i in range(model.n_basis):
            _, s_down, _ = torch.linalg.svd(basis_down[i], full_matrices=False)
            _, s_up, _ = torch.linalg.svd(basis_up[i], full_matrices=False)
            cond_per_basis_down.append((s_down[0] / (s_down[-1] + 1e-10)).item())
            cond_per_basis_up.append((s_up[0] / (s_up[-1] + 1e-10)).item())

        print(f"  basis_down: mean={np.mean(cond_per_basis_down):.2f}, max={np.max(cond_per_basis_down):.2f}")
        print(f"  basis_up: mean={np.mean(cond_per_basis_up):.2f}, max={np.max(cond_per_basis_up):.2f}")
        results['single_basis_cond_down'] = cond_per_basis_down
        results['single_basis_cond_up'] = cond_per_basis_up
    else:
        basis = model.shared_basis.basis
        cond_per_basis = []
        for i in range(model.n_basis):
            _, s, _ = torch.linalg.svd(basis[i], full_matrices=False)
            cond_per_basis.append((s[0] / (s[-1] + 1e-10)).item())
        print(f"  basis: mean={np.mean(cond_per_basis):.2f}, max={np.max(cond_per_basis):.2f}")
        results['single_basis_cond'] = cond_per_basis

    # ============================================================
    # STAGE B & C: Single Neuron vs Mixed Neurons
    # ============================================================
    print("\n📊 [Stage B] Single Neuron Recipe → W Condition")
    print("📊 [Stage C] Mixed Neurons (top-k) → W Condition")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Mixing Analysis", total=max_batches)):
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

                # Get routing info
                scores = qkv.W_router(normed)  # [B, S, n_neurons]
                topk_scores, topk_idx = torch.topk(scores, qkv.k, dim=-1)
                weights = F.softmax(topk_scores, dim=-1)  # [B, S, k]

                # --------------------------------------------------------
                # Stage B & C: Version-specific analysis
                # --------------------------------------------------------
                if version == "7.8":
                    # v7.8: No basis mixing, only neuron W mixing
                    nb = qkv.neuron_bank

                    # Stage B: Single neuron W (no mixing)
                    single_neuron_conds = []
                    for k_idx in range(qkv.k):
                        neuron_idx = topk_idx[:, :, k_idx]  # [B, S]
                        W_V_single = nb.W_V[neuron_idx]  # [B, S, D, rank] - direct lookup
                        W_flat = W_V_single.view(B * S, -1)
                        sample_idx = torch.randperm(B * S, device=device)[:min(50, B * S)]
                        W_sample = W_flat[sample_idx]
                        _, s_vals, _ = torch.linalg.svd(W_sample, full_matrices=False)
                        cond = (s_vals[0] / (s_vals[-1] + 1e-10)).item()
                        single_neuron_conds.append(cond)

                    avg_single_neuron_cond = np.mean(single_neuron_conds)

                    # Stage C: Mixed neurons → W
                    W_V_neurons = nb.get_W_V(topk_idx)  # [B, S, k, D, rank]
                    weights_exp = weights.unsqueeze(-1).unsqueeze(-1)
                    W_mixed = (W_V_neurons * weights_exp).sum(dim=2)  # [B, S, D, rank]

                    W_flat = W_mixed.view(B * S, -1)
                    sample_idx = torch.randperm(B * S, device=device)[:min(50, B * S)]
                    W_sample = W_flat[sample_idx]
                    _, s_vals, _ = torch.linalg.svd(W_sample, full_matrices=False)
                    mixed_neuron_cond = (s_vals[0] / (s_vals[-1] + 1e-10)).item()

                    # Stage D: Neuron weight entropy (instead of recipe entropy)
                    neuron_entropy = -torch.sum(
                        weights * torch.log(weights + 1e-10), dim=-1
                    ).mean().item()
                    max_neuron_weight = weights.max(dim=-1)[0].mean().item()

                    results[f'layer_{layer_idx}'].append({
                        'single_neuron_cond': avg_single_neuron_cond,
                        'mixed_neuron_cond': mixed_neuron_cond,
                        'cond_ratio': mixed_neuron_cond / (avg_single_neuron_cond + 1e-10),
                        'recipe_entropy': neuron_entropy,  # Using neuron weight entropy
                        'max_recipe_weight': max_neuron_weight,
                    })
                else:
                    # v7.5/7.6/7.7: Basis mixing analysis
                    # Get basis
                    if version == "7.7":
                        basis = qkv.shared_basis.basis_vo  # Use V basis for analysis
                    elif version == "7.6":
                        basis = qkv.shared_basis.get_basis_down()
                    else:
                        basis = qkv.shared_basis()

                    # Stage B: Single neuron recipe → W (no mixing)
                    single_neuron_conds = []
                    for k_idx in range(qkv.k):
                        neuron_idx = topk_idx[:, :, k_idx]  # [B, S]
                        recipe = qkv.neuron_recipe_V[neuron_idx]  # [B, S, n_basis]
                        recipe_norm = F.softmax(recipe, dim=-1)

                        # W from single neuron: [B, S, D, rank]
                        W_single = torch.einsum('bsn,ndr->bsdr', recipe_norm, basis)

                        # Sample and compute condition
                        W_flat = W_single.view(B * S, -1)
                        sample_idx = torch.randperm(B * S, device=device)[:min(50, B * S)]
                        W_sample = W_flat[sample_idx]
                        _, s_vals, _ = torch.linalg.svd(W_sample, full_matrices=False)
                        cond = (s_vals[0] / (s_vals[-1] + 1e-10)).item()
                        single_neuron_conds.append(cond)

                    avg_single_neuron_cond = np.mean(single_neuron_conds)

                    # Stage C: Mixed neurons (top-k weighted) → W
                    recipe_mixed = (qkv.neuron_recipe_V[topk_idx] * weights.unsqueeze(-1)).sum(dim=2)
                    recipe_mixed_norm = F.softmax(recipe_mixed, dim=-1)  # [B, S, n_basis]

                    W_mixed = torch.einsum('bsn,ndr->bsdr', recipe_mixed_norm, basis)

                    W_flat = W_mixed.view(B * S, -1)
                    sample_idx = torch.randperm(B * S, device=device)[:min(50, B * S)]
                    W_sample = W_flat[sample_idx]
                    _, s_vals, _ = torch.linalg.svd(W_sample, full_matrices=False)
                    mixed_neuron_cond = (s_vals[0] / (s_vals[-1] + 1e-10)).item()

                    # Stage D: Recipe entropy (how spread is the recipe?)
                    recipe_entropy = -torch.sum(
                        recipe_mixed_norm * torch.log(recipe_mixed_norm + 1e-10), dim=-1
                    ).mean().item()

                    # Max weight in recipe (specialization)
                    max_recipe_weight = recipe_mixed_norm.max(dim=-1)[0].mean().item()

                    results[f'layer_{layer_idx}'].append({
                        'single_neuron_cond': avg_single_neuron_cond,
                        'mixed_neuron_cond': mixed_neuron_cond,
                        'cond_ratio': mixed_neuron_cond / (avg_single_neuron_cond + 1e-10),
                        'recipe_entropy': recipe_entropy,
                        'max_recipe_weight': max_recipe_weight,
                    })

                # Forward to next layer
                attn_out, _ = qkv(normed, mask)
                x = x + layer.dropout(attn_out)
                x = x + layer.dropout(layer.w_down(F.gelu(layer.w_up(layer.norm2(x)))))

    # ============================================================
    # Output Results
    # ============================================================
    print("\n⭐ MIXING CONDITION ANALYSIS RESULTS:")

    if version == "7.8":
        print("\n  Stages (v7.8 - No Basis Mixing):")
        print("    A: Single neuron W condition (orthogonal init → should be ~1)")
        print("    B: Single neuron W (no mixing)")
        print("    C: Mixed neurons → W (neuron mixing only)")
        print("\n  If C >> B: Neuron mixing causes explosion")
        print("  (v7.8 eliminates basis mixing)")
    else:
        print("\n  Stages:")
        print("    A: Single basis condition (orthogonal init → should be ~1)")
        print("    B: Single neuron recipe → W (basis mixing only)")
        print("    C: Mixed neurons → W (neuron + basis mixing)")
        print("\n  If B >> A: Basis mixing causes explosion")
        print("  If C >> B: Neuron mixing causes explosion")

    for layer_idx in range(len(model.layers)):
        key = f'layer_{layer_idx}'
        data = results[key]

        avg_single = np.mean([d['single_neuron_cond'] for d in data])
        avg_mixed = np.mean([d['mixed_neuron_cond'] for d in data])
        avg_ratio = np.mean([d['cond_ratio'] for d in data])
        avg_entropy = np.mean([d['recipe_entropy'] for d in data])
        avg_max_weight = np.mean([d['max_recipe_weight'] for d in data])

        print(f"\n  Layer {layer_idx}:")
        print(f"    ┌─────────────────────────────────────────────────────────")
        print(f"    │ [B] Single neuron → W:  cond_num = {avg_single:.1f}")
        print(f"    │ [C] Mixed neurons → W:  cond_num = {avg_mixed:.1f}")
        print(f"    │")

        # Identify explosion source
        if avg_ratio > 2.0:
            print(f"    │ ⚠️  Neuron mixing causes {avg_ratio:.1f}x condition explosion!")
            explosion_source = "NEURON_MIXING"
        elif avg_single > 50 and version != "7.8":
            print(f"    │ ⚠️  Basis mixing already causes high condition!")
            explosion_source = "BASIS_MIXING"
        elif avg_single > 50 and version == "7.8":
            print(f"    │ ⚠️  Single neuron W has high condition (check orthogonality)!")
            explosion_source = "INIT_DEGRADED"
        else:
            print(f"    │ ✓  Mixing ratio = {avg_ratio:.2f}x (acceptable)")
            explosion_source = None

        print(f"    │")
        if version == "7.8":
            print(f"    │ Neuron weight stats: entropy={avg_entropy:.2f}, max_weight={avg_max_weight:.3f}")
        else:
            print(f"    │ Recipe stats: entropy={avg_entropy:.2f}, max_weight={avg_max_weight:.3f}")

        if avg_max_weight > 0.5:
            print(f"    │ ⚠️  High specialization (max_weight > 0.5)")

        print(f"    └─────────────────────────────────────────────────────────")

        if explosion_source:
            print(f"      🎯 EXPLOSION SOURCE: {explosion_source}")

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

    # Detect model version from state_dict
    has_neuron_bank_W = any('neuron_bank.W_Q' in k for k in state_dict.keys())
    has_qk_vo_basis = any('basis_qk' in k or 'basis_vo' in k for k in state_dict.keys())
    has_split_basis = any('basis_down' in k or 'basis_up' in k for k in state_dict.keys())

    if has_neuron_bank_W:
        detected_version = '7.8'
        print("  Detected v7.8 from state_dict (neuron_bank.W_Q/K/V/O)")
    elif has_qk_vo_basis:
        detected_version = '7.7'
        print("  Detected v7.7 from state_dict (basis_qk/basis_vo)")
    elif has_split_basis:
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
        if model_version == '7.8':
            # v7.8: NeuronBank W matrices
            n_neurons = state_dict['neuron_bank.W_Q'].shape[0]
            rank = state_dict['neuron_bank.W_Q'].shape[2]
            n_basis = n_neurons  # For compatibility

            model_config = {
                'vocab_size': state_dict['token_emb.weight'].shape[0],
                'd_model': state_dict['token_emb.weight'].shape[1],
                'n_layers': sum(1 for k in state_dict.keys() if k.startswith('layers.') and '.norm1.weight' in k),
                'n_neurons': n_neurons,
                'rank': rank,
                'model_version': model_version,
            }
        else:
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
    all_results['mixing_condition'] = analyze_mixing_condition(model, val_loader, device, max_batches=3)
    all_results['attention'] = analyze_attention_patterns(model, val_loader, device, args.max_batches)
    all_results['summary'] = compute_summary_metrics(all_results)

    # Save results
    output_path = Path(args.checkpoint) / 'analysis_v75.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Analysis complete! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
