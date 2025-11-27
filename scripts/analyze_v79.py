"""
DAWN v7.9 Analysis Script
NeuronCircuit with Householder Transformations

분석 항목:
1. Householder Transform 분석
2. Input/Output Neurons 직교성
3. 라우팅 패턴 심층 분석
4. Compressor/Expander 정보 흐름
5. Layer별 특성 비교
6. Attention 패턴 분석
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_v79 import DAWN

# Optional: matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def get_underlying_model(model):
    """Get the underlying model from torch.compile wrapper"""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def compute_gini(values):
    """Compute Gini coefficient for measuring inequality"""
    sorted_values = torch.sort(values.float())[0]
    n = len(sorted_values)
    if n == 0 or sorted_values.sum() == 0:
        return 0.0
    index = torch.arange(1, n + 1, dtype=torch.float32, device=values.device)
    return ((2 * index - n - 1) * sorted_values).sum() / (n * sorted_values.sum())


# ============================================================
# 1. Householder Transform Analysis
# ============================================================

def analyze_householder_transforms(model):
    """Analyze Householder transform vectors (process_neurons)"""
    print("\n" + "=" * 60)
    print("1. HOUSEHOLDER TRANSFORM ANALYSIS")
    print("=" * 60)

    model = get_underlying_model(model)
    results = {'layers': {}}

    # Collect all process_neurons from all layers
    all_process_neurons = []

    for layer_idx, layer in enumerate(model.layers):
        qkv = layer.qkv_circuit
        layer_results = {}

        # Get process neurons from Q, K, V circuits (Down)
        circuits_down = {
            'Q': qkv.circuit_Q,
            'K': qkv.circuit_K,
            'V': qkv.circuit_V,
        }
        # Get process neurons from O circuit (Up)
        circuit_up = qkv.circuit_O

        print(f"\n📌 Layer {layer_idx}:")

        # Analyze Down circuits (Q/K/V)
        for name, circuit in circuits_down.items():
            process_v = circuit.process_neurons.data  # [n_process, rank]
            n_process, rank = process_v.shape

            # 1. Norm distribution (should be ≈ 1)
            norms = process_v.norm(dim=-1)  # [n_process]
            norm_mean = norms.mean().item()
            norm_std = norms.std().item()
            norm_min = norms.min().item()
            norm_max = norms.max().item()

            # 2. Cosine similarity matrix
            v_normalized = F.normalize(process_v, dim=-1)
            cos_sim = v_normalized @ v_normalized.T  # [n_process, n_process]

            # Off-diagonal similarity (should be low for diversity)
            mask = ~torch.eye(n_process, dtype=torch.bool, device=cos_sim.device)
            off_diag_sim = cos_sim[mask]
            sim_mean = off_diag_sim.abs().mean().item()
            sim_max = off_diag_sim.abs().max().item()

            # 3. Householder matrix condition number (H = I - 2vv^T)
            # For unit vectors, H is orthogonal so cond(H) = 1
            # Check effective condition after k transforms
            cond_numbers = []
            for i in range(n_process):
                v = process_v[i]
                v_norm = v / (v.norm() + 1e-8)
                H = torch.eye(rank, device=v.device) - 2 * torch.outer(v_norm, v_norm)
                _, s, _ = torch.linalg.svd(H)
                cond = (s[0] / (s[-1] + 1e-10)).item()
                cond_numbers.append(cond)

            layer_results[f'{name}_process'] = {
                'norm_mean': norm_mean,
                'norm_std': norm_std,
                'norm_range': [norm_min, norm_max],
                'cosine_sim_mean': sim_mean,
                'cosine_sim_max': sim_max,
                'condition_mean': np.mean(cond_numbers),
                'condition_max': np.max(cond_numbers),
            }

            print(f"  {name} process: ||v||={norm_mean:.4f}±{norm_std:.4f}, "
                  f"cos_sim={sim_mean:.4f}, cond={np.mean(cond_numbers):.2f}")

        # Analyze Up circuit (O)
        process_v = circuit_up.process_neurons.data
        norms = process_v.norm(dim=-1)
        v_normalized = F.normalize(process_v, dim=-1)
        cos_sim = v_normalized @ v_normalized.T
        mask = ~torch.eye(process_v.shape[0], dtype=torch.bool, device=cos_sim.device)
        off_diag_sim = cos_sim[mask]

        layer_results['O_process'] = {
            'norm_mean': norms.mean().item(),
            'norm_std': norms.std().item(),
            'cosine_sim_mean': off_diag_sim.abs().mean().item(),
        }
        print(f"  O process: ||v||={norms.mean().item():.4f}±{norms.std().item():.4f}, "
              f"cos_sim={off_diag_sim.abs().mean().item():.4f}")

        results['layers'][f'layer_{layer_idx}'] = layer_results
        all_process_neurons.append(process_v)

    # Global statistics across all layers
    all_neurons = torch.cat(all_process_neurons, dim=0)
    results['global'] = {
        'total_process_neurons': all_neurons.shape[0],
        'norm_mean': all_neurons.norm(dim=-1).mean().item(),
        'norm_std': all_neurons.norm(dim=-1).std().item(),
    }

    return results


# ============================================================
# 2. Input/Output Neurons Orthogonality
# ============================================================

def analyze_neuron_orthogonality(model):
    """Analyze orthogonality of input and output neurons"""
    print("\n" + "=" * 60)
    print("2. INPUT/OUTPUT NEURONS ORTHOGONALITY")
    print("=" * 60)

    model = get_underlying_model(model)
    results = {'input': {}, 'output': {}}

    # Get neurons from first layer (representative)
    layer = model.layers[0]
    qkv = layer.qkv_circuit

    # Input neurons from Q circuit (all Q/K/V share similar structure)
    input_neurons = qkv.circuit_Q.input_neurons.data  # [n_input, d_model, rank]
    n_input, d_model, rank = input_neurons.shape

    print(f"\n📌 Input Neurons: [{n_input}, {d_model}, {rank}]")

    # Analyze each input neuron
    input_conds = []
    input_singular_values = []
    for i in range(n_input):
        W = input_neurons[i]  # [d_model, rank]
        _, s, _ = torch.linalg.svd(W, full_matrices=False)
        cond = (s[0] / (s[-1] + 1e-10)).item()
        input_conds.append(cond)
        input_singular_values.append(s.cpu().numpy())

        # Check orthogonality: W^T @ W should be identity
        WtW = W.T @ W  # [rank, rank]
        I = torch.eye(rank, device=W.device)
        orth_error = (WtW - I).abs().max().item()
        print(f"  Neuron {i}: cond={cond:.2f}, orth_error={orth_error:.2e}, "
              f"σ_range=[{s[-1].item():.4f}, {s[0].item():.4f}]")

    results['input']['condition_numbers'] = input_conds
    results['input']['condition_mean'] = np.mean(input_conds)
    results['input']['condition_max'] = np.max(input_conds)

    # Input neuron overlap (should be low for diversity)
    print(f"\n  Input neuron overlap (W_i^T @ W_j Frobenius norm):")
    overlap_matrix = torch.zeros(n_input, n_input)
    for i in range(n_input):
        for j in range(n_input):
            if i != j:
                overlap = (input_neurons[i].T @ input_neurons[j]).norm().item()
                overlap_matrix[i, j] = overlap
    off_diag = overlap_matrix[overlap_matrix != 0]
    print(f"    Mean: {off_diag.mean().item():.4f}, Max: {off_diag.max().item():.4f}")
    results['input']['overlap_mean'] = off_diag.mean().item()
    results['input']['overlap_max'] = off_diag.max().item()

    # Output neurons
    output_neurons = qkv.circuit_O.output_neurons.data  # [n_output, rank, d_model]
    n_output, rank, d_model = output_neurons.shape

    print(f"\n📌 Output Neurons: [{n_output}, {rank}, {d_model}]")

    output_conds = []
    for i in range(n_output):
        W = output_neurons[i]  # [rank, d_model]
        _, s, _ = torch.linalg.svd(W, full_matrices=False)
        cond = (s[0] / (s[-1] + 1e-10)).item()
        output_conds.append(cond)

        # Check orthogonality: W @ W^T should be identity
        WWt = W @ W.T  # [rank, rank]
        I = torch.eye(rank, device=W.device)
        orth_error = (WWt - I).abs().max().item()
        print(f"  Neuron {i}: cond={cond:.2f}, orth_error={orth_error:.2e}, "
              f"σ_range=[{s[-1].item():.4f}, {s[0].item():.4f}]")

    results['output']['condition_numbers'] = output_conds
    results['output']['condition_mean'] = np.mean(output_conds)
    results['output']['condition_max'] = np.max(output_conds)

    # Output neuron overlap
    print(f"\n  Output neuron overlap:")
    overlap_matrix = torch.zeros(n_output, n_output)
    for i in range(n_output):
        for j in range(n_output):
            if i != j:
                overlap = (output_neurons[i] @ output_neurons[j].T).norm().item()
                overlap_matrix[i, j] = overlap
    off_diag = overlap_matrix[overlap_matrix != 0]
    print(f"    Mean: {off_diag.mean().item():.4f}, Max: {off_diag.max().item():.4f}")
    results['output']['overlap_mean'] = off_diag.mean().item()
    results['output']['overlap_max'] = off_diag.max().item()

    return results


# ============================================================
# 3. Routing Pattern Deep Analysis
# ============================================================

def analyze_routing_patterns(model, dataloader, device, max_batches=10):
    """Deep analysis of routing patterns"""
    print("\n" + "=" * 60)
    print("3. ROUTING PATTERN DEEP ANALYSIS")
    print("=" * 60)

    model = get_underlying_model(model)
    model.eval()

    n_layers = len(model.layers)
    n_process = model.n_process
    process_k = model.process_k

    # Accumulators
    # Process neuron usage per Q/K/V/O
    usage_down = {f'layer_{i}': {'Q': torch.zeros(n_process, device=device),
                                  'K': torch.zeros(n_process, device=device),
                                  'V': torch.zeros(n_process, device=device)}
                  for i in range(n_layers)}
    usage_up = {f'layer_{i}': torch.zeros(n_process, device=device) for i in range(n_layers)}

    # Co-occurrence matrix (which neurons are selected together)
    cooccurrence = {f'layer_{i}': torch.zeros(n_process, n_process, device=device)
                    for i in range(n_layers)}

    # Q-K-V correlation (do same tokens select similar neurons?)
    qkv_correlations = []

    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Routing Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            B, S = input_ids.shape
            total_tokens += B * S

            # Forward pass with routing info
            _, routing_infos = model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                routing_down = routing_info['routing_down']
                routing_up = routing_info['routing_up']

                # Process indices from down routing (shared for Q/K/V in current impl)
                process_idx = routing_down['process_indices']  # [B, S, k]

                # Update usage counts
                for qkv_type in ['Q', 'K', 'V']:
                    idx_flat = process_idx.reshape(-1)
                    counts = torch.bincount(idx_flat, minlength=n_process).float()
                    usage_down[f'layer_{layer_idx}'][qkv_type] += counts

                # Update co-occurrence
                for b in range(B):
                    for s in range(S):
                        indices = process_idx[b, s]  # [k]
                        for i in range(process_k):
                            for j in range(process_k):
                                cooccurrence[f'layer_{layer_idx}'][indices[i], indices[j]] += 1

                # Up routing
                up_idx = routing_up['process_indices']  # [B, S, k]
                idx_flat = up_idx.reshape(-1)
                counts = torch.bincount(idx_flat, minlength=n_process).float()
                usage_up[f'layer_{layer_idx}'] += counts

    # Compute statistics
    results = {'layers': {}, 'global': {}}

    print("\n📌 Per-Layer Routing Statistics:")
    all_ginis = []
    all_entropies = []

    for layer_idx in range(n_layers):
        layer_key = f'layer_{layer_idx}'

        # Combine Q/K/V usage
        total_usage = (usage_down[layer_key]['Q'] +
                       usage_down[layer_key]['K'] +
                       usage_down[layer_key]['V'])
        total_usage = total_usage / total_usage.sum()

        # Entropy (higher = more uniform)
        entropy = -(total_usage * torch.log(total_usage + 1e-10)).sum().item()
        max_entropy = math.log(n_process)
        normalized_entropy = entropy / max_entropy

        # Gini (lower = more uniform)
        gini = compute_gini(total_usage).item()

        # Usage rate
        usage_rate = (total_usage > 1e-6).float().mean().item()

        # Top-k concentration
        top5_usage = torch.topk(total_usage, 5)[0].sum().item()
        top10_usage = torch.topk(total_usage, 10)[0].sum().item()

        all_ginis.append(gini)
        all_entropies.append(normalized_entropy)

        print(f"  Layer {layer_idx}: entropy={normalized_entropy:.3f}, gini={gini:.3f}, "
              f"usage={usage_rate:.1%}, top5={top5_usage:.1%}, top10={top10_usage:.1%}")

        # Co-occurrence analysis
        cooc = cooccurrence[layer_key]
        cooc_diag = cooc.diag()
        cooc_offdiag = cooc.clone()
        cooc_offdiag.fill_diagonal_(0)

        results['layers'][layer_key] = {
            'entropy': normalized_entropy,
            'gini': gini,
            'usage_rate': usage_rate,
            'top5_concentration': top5_usage,
            'top10_concentration': top10_usage,
            'cooccurrence_self_ratio': (cooc_diag.sum() / cooc.sum()).item() if cooc.sum() > 0 else 0,
        }

    results['global'] = {
        'avg_entropy': np.mean(all_entropies),
        'avg_gini': np.mean(all_ginis),
    }

    print(f"\n  Global: avg_entropy={results['global']['avg_entropy']:.3f}, "
          f"avg_gini={results['global']['avg_gini']:.3f}")

    return results


# ============================================================
# 4. Compressor/Expander Information Flow
# ============================================================

def analyze_information_flow(model, dataloader, device, max_batches=5):
    """Analyze information flow through Compressor and Expander"""
    print("\n" + "=" * 60)
    print("4. COMPRESSOR/EXPANDER INFORMATION FLOW")
    print("=" * 60)

    model = get_underlying_model(model)
    model.eval()

    n_layers = len(model.layers)

    # Track norms at each stage
    stage_norms = {
        'input': [],           # x before compression
        'after_compress': [],  # after input neurons
        'after_process': [],   # after Householder
        'after_attention': [], # attention output
        'after_expand': [],    # after expander
        'output': [],          # final output
    }

    # Per-layer tracking
    layer_stats = {f'layer_{i}': {k: [] for k in stage_norms.keys()} for i in range(n_layers)}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Info Flow", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            B, S = input_ids.shape

            # Manual forward pass to track intermediate values
            pos = torch.arange(S, device=device).unsqueeze(0)
            x = model.token_emb(input_ids) + model.pos_emb(pos)
            x = model.dropout(x)

            mask = model.causal_mask[:, :, :S, :S]

            for layer_idx, layer in enumerate(model.layers):
                layer_key = f'layer_{layer_idx}'
                qkv = layer.qkv_circuit

                # Input norm
                residual = x
                x_norm = layer.norm1(x)
                layer_stats[layer_key]['input'].append(x_norm.norm(dim=-1).mean().item())

                # Get routing
                routing_down = qkv.router_down(x_norm)
                input_weights = routing_down.get('input_weights')
                process_idx = routing_down['process_indices']

                # Track Q compression (representative)
                circuit_Q = qkv.circuit_Q

                # After input projection
                all_proj = torch.einsum('bsd,ndr->bsnr', x_norm, circuit_Q.input_neurons)
                compressed = (all_proj * input_weights.unsqueeze(-1)).sum(dim=2)
                layer_stats[layer_key]['after_compress'].append(compressed.norm(dim=-1).mean().item())

                # After Householder
                k = process_idx.shape[-1]
                idx_expanded = process_idx.unsqueeze(-1).expand(B, S, k, circuit_Q.rank)
                selected_v = circuit_Q.process_neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
                selected_v = selected_v.gather(2, idx_expanded)

                processed = compressed.clone()
                for i in range(k):
                    v = selected_v[:, :, i, :]
                    processed = circuit_Q.apply_householder(processed, v)
                layer_stats[layer_key]['after_process'].append(processed.norm(dim=-1).mean().item())

                # Full attention forward
                attn_out, _ = qkv(x_norm, mask)
                layer_stats[layer_key]['after_attention'].append(attn_out.norm(dim=-1).mean().item())

                # After residual
                x = residual + layer.dropout(attn_out)
                layer_stats[layer_key]['output'].append(x.norm(dim=-1).mean().item())

                # FFN
                residual = x
                x_norm = layer.norm2(x)
                ffn_out = layer.w_down(F.gelu(layer.w_up(x_norm)))
                x = residual + layer.dropout(ffn_out)

    # Compute averages
    results = {'layers': {}, 'global': {}}

    print("\n📌 Information Flow (Norm Ratios):")
    for layer_idx in range(n_layers):
        layer_key = f'layer_{layer_idx}'
        stats = layer_stats[layer_key]

        avg_input = np.mean(stats['input'])
        avg_compress = np.mean(stats['after_compress'])
        avg_process = np.mean(stats['after_process'])
        avg_attn = np.mean(stats['after_attention'])
        avg_output = np.mean(stats['output'])

        # Ratios
        compress_ratio = avg_compress / (avg_input + 1e-10)
        process_ratio = avg_process / (avg_compress + 1e-10)
        attn_ratio = avg_attn / (avg_process + 1e-10)
        total_ratio = avg_output / (avg_input + 1e-10)

        results['layers'][layer_key] = {
            'input_norm': avg_input,
            'compress_ratio': compress_ratio,
            'process_ratio': process_ratio,
            'attention_ratio': attn_ratio,
            'total_ratio': total_ratio,
        }

        print(f"  Layer {layer_idx}: input={avg_input:.2f} → compress={compress_ratio:.2f}x "
              f"→ process={process_ratio:.2f}x → attn={attn_ratio:.2f}x → total={total_ratio:.2f}x")

    return results


# ============================================================
# 5. Layer-wise Comparison
# ============================================================

def analyze_layer_comparison(model):
    """Compare characteristics across layers"""
    print("\n" + "=" * 60)
    print("5. LAYER-WISE COMPARISON")
    print("=" * 60)

    model = get_underlying_model(model)
    results = {'layers': {}}

    n_layers = len(model.layers)

    print("\n📌 Router Weight Statistics:")
    for layer_idx, layer in enumerate(model.layers):
        qkv = layer.qkv_circuit

        # Router weights
        router_down = qkv.router_down
        router_up = qkv.router_up

        input_router_norm = router_down.input_router.weight.norm().item()
        process_router_down_norm = router_down.process_router.weight.norm().item()
        process_router_up_norm = router_up.process_router.weight.norm().item()
        output_router_norm = router_up.output_router.weight.norm().item()

        results['layers'][f'layer_{layer_idx}'] = {
            'input_router_norm': input_router_norm,
            'process_router_down_norm': process_router_down_norm,
            'process_router_up_norm': process_router_up_norm,
            'output_router_norm': output_router_norm,
        }

        print(f"  Layer {layer_idx}: input={input_router_norm:.4f}, "
              f"proc_down={process_router_down_norm:.4f}, "
              f"proc_up={process_router_up_norm:.4f}, "
              f"output={output_router_norm:.4f}")

    # Compare process neurons across layers
    print("\n📌 Process Neuron Diversity Across Layers:")
    for layer_idx, layer in enumerate(model.layers):
        qkv = layer.qkv_circuit

        # Get Q circuit's process neurons
        process_v = qkv.circuit_Q.process_neurons.data

        # Check diversity
        v_norm = F.normalize(process_v, dim=-1)
        cos_sim = v_norm @ v_norm.T
        mask = ~torch.eye(process_v.shape[0], dtype=torch.bool, device=cos_sim.device)
        mean_sim = cos_sim[mask].abs().mean().item()

        results['layers'][f'layer_{layer_idx}']['process_diversity'] = 1 - mean_sim

        print(f"  Layer {layer_idx}: diversity={1-mean_sim:.4f} (1-cos_sim)")

    return results


# ============================================================
# 6. Attention Pattern Analysis
# ============================================================

def analyze_attention_patterns(model, dataloader, device, max_batches=5):
    """Analyze attention patterns in NeuronAttention"""
    print("\n" + "=" * 60)
    print("6. ATTENTION PATTERN ANALYSIS")
    print("=" * 60)

    model = get_underlying_model(model)
    model.eval()

    n_layers = len(model.layers)
    n_heads = model.n_heads

    # Track attention statistics
    attn_stats = {f'layer_{i}': {
        'entropy': [],
        'sparsity': [],  # fraction of weights < 0.01
        'max_weight': [],
        'head_specialization': [],
    } for i in range(n_layers)}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Attention", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            B, S = input_ids.shape

            # Manual forward to get attention weights
            pos = torch.arange(S, device=device).unsqueeze(0)
            x = model.token_emb(input_ids) + model.pos_emb(pos)
            x = model.dropout(x)

            mask = model.causal_mask[:, :, :S, :S]

            for layer_idx, layer in enumerate(model.layers):
                layer_key = f'layer_{layer_idx}'
                qkv = layer.qkv_circuit

                residual = x
                x_norm = layer.norm1(x)

                # Get Q, K, V
                routing_down = qkv.router_down(x_norm)
                input_weights = routing_down.get('input_weights')
                input_idx = routing_down.get('input_idx')
                process_indices = routing_down['process_indices']

                Q = qkv.circuit_Q(x_norm, input_idx, input_weights, process_indices)
                K = qkv.circuit_K(x_norm, input_idx, input_weights, process_indices)
                V = qkv.circuit_V(x_norm, input_idx, input_weights, process_indices)

                # Reshape for multi-head
                d_head = qkv.d_head
                Q = Q.view(B, S, n_heads, d_head).transpose(1, 2)
                K = K.view(B, S, n_heads, d_head).transpose(1, 2)
                V = V.view(B, S, n_heads, d_head).transpose(1, 2)

                # Compute attention
                attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(d_head)
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, S, S]

                # Compute statistics
                # Entropy per head
                entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1)  # [B, H, S]
                max_entropy = math.log(S)
                norm_entropy = entropy / max_entropy

                attn_stats[layer_key]['entropy'].append(norm_entropy.mean().item())

                # Sparsity
                sparsity = (attn_weights < 0.01).float().mean().item()
                attn_stats[layer_key]['sparsity'].append(sparsity)

                # Max weight
                max_weight = attn_weights.max(dim=-1)[0].mean().item()
                attn_stats[layer_key]['max_weight'].append(max_weight)

                # Head specialization (std of entropy across heads)
                head_entropy = norm_entropy.mean(dim=(0, 2))  # [H]
                head_std = head_entropy.std().item()
                attn_stats[layer_key]['head_specialization'].append(head_std)

                # Continue forward
                attn_out = attn_weights @ V
                attn_out = attn_out.transpose(1, 2).reshape(B, S, qkv.rank)

                routing_up = qkv.router_up(attn_out)
                output_weights = routing_up.get('output_weights')
                output_idx = routing_up.get('output_idx')
                process_indices_up = routing_up['process_indices']

                attn_out = qkv.circuit_O(attn_out, output_idx, output_weights, process_indices_up)

                x = residual + layer.dropout(attn_out)

                # FFN
                residual = x
                x_norm = layer.norm2(x)
                ffn_out = layer.w_down(F.gelu(layer.w_up(x_norm)))
                x = residual + layer.dropout(ffn_out)

    # Compute averages
    results = {'layers': {}}

    print("\n📌 Attention Statistics:")
    for layer_idx in range(n_layers):
        layer_key = f'layer_{layer_idx}'
        stats = attn_stats[layer_key]

        avg_entropy = np.mean(stats['entropy'])
        avg_sparsity = np.mean(stats['sparsity'])
        avg_max = np.mean(stats['max_weight'])
        avg_head_spec = np.mean(stats['head_specialization'])

        results['layers'][layer_key] = {
            'entropy': avg_entropy,
            'sparsity': avg_sparsity,
            'max_weight': avg_max,
            'head_specialization': avg_head_spec,
        }

        print(f"  Layer {layer_idx}: entropy={avg_entropy:.3f}, sparsity={avg_sparsity:.1%}, "
              f"max_weight={avg_max:.3f}, head_spec={avg_head_spec:.4f}")

    return results


# ============================================================
# Visualization
# ============================================================

def create_visualizations(all_results, output_dir):
    """Create visualization plots"""
    if not HAS_MATPLOTLIB:
        print("\nSkipping visualizations (matplotlib not available)")
        return

    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Routing entropy/gini per layer
    if 'routing' in all_results:
        routing = all_results['routing']
        n_layers = len([k for k in routing['layers'].keys()])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        layers = list(range(n_layers))
        entropies = [routing['layers'][f'layer_{i}']['entropy'] for i in layers]
        ginis = [routing['layers'][f'layer_{i}']['gini'] for i in layers]

        axes[0].bar(layers, entropies, color='steelblue')
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Normalized Entropy')
        axes[0].set_title('Routing Entropy per Layer')
        axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Max (uniform)')

        axes[1].bar(layers, ginis, color='coral')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Gini Coefficient')
        axes[1].set_title('Routing Gini per Layer')
        axes[1].axhline(y=0.0, color='g', linestyle='--', alpha=0.5, label='Min (uniform)')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'routing_stats.png'), dpi=150)
        plt.close()
        print(f"  Saved: routing_stats.png")

    # 2. Information flow
    if 'info_flow' in all_results:
        info_flow = all_results['info_flow']
        n_layers = len([k for k in info_flow['layers'].keys()])

        fig, ax = plt.subplots(figsize=(10, 6))

        stages = ['compress', 'process', 'attention', 'total']
        x = np.arange(len(stages))
        width = 0.15

        for layer_idx in range(n_layers):
            layer_data = info_flow['layers'][f'layer_{layer_idx}']
            ratios = [layer_data[f'{s}_ratio'] for s in stages]
            ax.bar(x + layer_idx * width, ratios, width, label=f'Layer {layer_idx}')

        ax.set_ylabel('Norm Ratio')
        ax.set_title('Information Flow (Norm Ratios) per Stage')
        ax.set_xticks(x + width * (n_layers - 1) / 2)
        ax.set_xticklabels(['Compress', 'Process', 'Attention', 'Total'])
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'info_flow.png'), dpi=150)
        plt.close()
        print(f"  Saved: info_flow.png")

    # 3. Attention patterns
    if 'attention' in all_results:
        attention = all_results['attention']
        n_layers = len([k for k in attention['layers'].keys()])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        layers = list(range(n_layers))
        entropies = [attention['layers'][f'layer_{i}']['entropy'] for i in layers]
        sparsities = [attention['layers'][f'layer_{i}']['sparsity'] for i in layers]
        max_weights = [attention['layers'][f'layer_{i}']['max_weight'] for i in layers]

        axes[0].bar(layers, entropies, color='steelblue')
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Normalized Entropy')
        axes[0].set_title('Attention Entropy')

        axes[1].bar(layers, sparsities, color='coral')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Sparsity (< 0.01)')
        axes[1].set_title('Attention Sparsity')

        axes[2].bar(layers, max_weights, color='seagreen')
        axes[2].set_xlabel('Layer')
        axes[2].set_ylabel('Max Weight')
        axes[2].set_title('Attention Max Weight')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attention_stats.png'), dpi=150)
        plt.close()
        print(f"  Saved: attention_stats.png")

    print(f"\n  All visualizations saved to: {output_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN v7.9 Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--val_data', type=str,
                        default='/content/drive/MyDrive/data/validation/wikitext_5to1_texts.pkl',
                        help='Path to validation data')
    parser.add_argument('--max_batches', type=int, default=10,
                        help='Max batches for runtime analysis')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./analysis_v79_output',
                        help='Output directory for results')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Handle checkpoint path (directory or file)
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        # Look for best_model.pt or latest checkpoint
        best_model = checkpoint_path / 'best_model.pt'
        if best_model.exists():
            checkpoint_path = best_model
        else:
            # Find most recent .pt file
            pt_files = list(checkpoint_path.glob('*.pt'))
            if pt_files:
                checkpoint_path = max(pt_files, key=lambda p: p.stat().st_mtime)
            else:
                raise FileNotFoundError(f"No .pt files found in {args.checkpoint}")
        print(f"Found checkpoint: {checkpoint_path}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    config = checkpoint.get('model_config', checkpoint.get('config', {}))
    model_version = config.get('model_version', checkpoint.get('model_version', '7.9'))
    print(f"Checkpoint model version: {model_version}")

    if model_version != "7.9":
        print(f"Warning: This script is for v7.9, but checkpoint is v{model_version}")

    # Create model
    print(f"\nCreating model v7.9...")
    model = DAWN(
        vocab_size=config.get('vocab_size', 30522),
        d_model=config.get('d_model', 256),
        n_layers=config.get('n_layers', 4),
        n_heads=config.get('n_heads', 4),
        d_ff=config.get('d_ff', 1024),
        max_seq_len=config.get('max_seq_len', 128),
        rank=config.get('rank', config.get('basis_rank', 64)),
        n_input=config.get('n_input', 8),
        n_process=config.get('n_process', 32),
        n_output=config.get('n_output', 8),
        process_k=config.get('process_k', 3),
        dropout=config.get('dropout', 0.1),
        use_soft_selection=config.get('use_soft_selection', True),
    )

    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Handle torch.compile prefix
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  Removing torch.compile wrapper prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model: DAWN v{model.__version__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optional: compile for faster inference
    if hasattr(torch, 'compile'):
        print("\n⚡ Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')

    # Load data
    print(f"\nLoading validation data from: {args.val_data}")
    import pickle
    with open(args.val_data, 'rb') as f:
        val_texts = pickle.load(f)
    print(f"Loaded {len(val_texts)} validation texts")

    # Create simple dataloader
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_len=128):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                padding='max_length',
                return_tensors='pt'
            )
            return {'input_ids': encoding['input_ids'].squeeze(0)}

    dataset = SimpleDataset(val_texts, tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    # ============================================================
    # Run Analyses
    # ============================================================
    print("\n" + "=" * 60)
    print("STARTING ANALYSIS")
    print("=" * 60)

    all_results = {}

    # 1. Householder Transform Analysis
    all_results['householder'] = analyze_householder_transforms(model)

    # 2. Neuron Orthogonality
    all_results['orthogonality'] = analyze_neuron_orthogonality(model)

    # 3. Routing Patterns
    all_results['routing'] = analyze_routing_patterns(model, dataloader, device, args.max_batches)

    # 4. Information Flow
    all_results['info_flow'] = analyze_information_flow(model, dataloader, device, max_batches=5)

    # 5. Layer Comparison
    all_results['layer_comparison'] = analyze_layer_comparison(model)

    # 6. Attention Patterns
    all_results['attention'] = analyze_attention_patterns(model, dataloader, device, max_batches=5)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    print("\n📊 Key Findings:")
    print(f"  Orthogonality loss: {model.orthogonality_loss().item():.6f}")
    print(f"  Process norm loss: {model.process_norm_loss().item():.6f}")
    print(f"  Avg routing entropy: {all_results['routing']['global']['avg_entropy']:.3f}")
    print(f"  Avg routing gini: {all_results['routing']['global']['avg_gini']:.3f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    results_path = os.path.join(args.output_dir, 'analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Create visualizations
    create_visualizations(all_results, args.output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
