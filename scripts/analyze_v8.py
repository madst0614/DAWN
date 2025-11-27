"""
DAWN v8.0 Analysis Script
SharedNeurons + NeuronMemory Architecture

분석 항목:
1. SharedNeurons 분석 (Input/Process/Output/Knowledge Neurons)
2. Routing Pattern 분석 (Q/K/V/O별 라우팅)
3. Information Flow 분석 (Attention + Memory)
4. Memory/Knowledge 분석 (지식 검색 패턴)
5. Layer별 라우터 비교
6. Attention Pattern 분석
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_v8 import DAWN

# Optional: matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            IN_NOTEBOOK = True
            get_ipython().run_line_magic('matplotlib', 'inline')
        else:
            IN_NOTEBOOK = False
            matplotlib.use('Agg')
    except (ImportError, AttributeError):
        IN_NOTEBOOK = False
        matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    IN_NOTEBOOK = False
    print("Warning: matplotlib not available, skipping visualizations")

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def get_underlying_model(model):
    """Get the underlying model from torch.compile wrapper"""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


# ============================================================
# 1. SharedNeurons Analysis
# ============================================================

def analyze_shared_neurons(model):
    """Analyze SharedNeurons - the shared neuron pool across all layers"""
    print("\n" + "=" * 60)
    print("1. SHARED NEURONS ANALYSIS")
    print("=" * 60)

    model = get_underlying_model(model)
    shared = model.shared_neurons
    results = {}

    # ========== Input Neurons ==========
    print("\n📌 Input Neurons: [n_input, d_model, rank]")
    input_neurons = shared.input_neurons.data  # [n_input, d_model, rank]
    n_input, d_model, rank = input_neurons.shape

    input_results = {'shape': [n_input, d_model, rank], 'neurons': {}}

    # Condition numbers and orthogonality
    input_conds = []
    input_orth_errors = []
    for i in range(n_input):
        W = input_neurons[i]  # [d_model, rank]
        _, s, _ = torch.linalg.svd(W, full_matrices=False)
        cond = (s[0] / (s[-1] + 1e-10)).item()
        input_conds.append(cond)

        # W.T @ W should be identity (rank × rank)
        WtW = W.T @ W
        I = torch.eye(rank, device=W.device)
        orth_error = (WtW - I).abs().max().item()
        input_orth_errors.append(orth_error)

        input_results['neurons'][i] = {
            'condition': cond,
            'orthogonality_error': orth_error,
            'singular_range': [s[-1].item(), s[0].item()]
        }

        print(f"  Neuron {i}: cond={cond:.2f}, orth_err={orth_error:.2e}, σ=[{s[-1].item():.4f}, {s[0].item():.4f}]")

    input_results['avg_condition'] = np.mean(input_conds)
    input_results['avg_orth_error'] = np.mean(input_orth_errors)

    # Input neuron overlap (how similar are different input neurons)
    print(f"\n  Input neuron overlap (W_i.T @ W_j Frobenius norm):")
    input_T = input_neurons.transpose(1, 2)  # [n_input, rank, d_model]
    products = torch.einsum('iad,jdb->ijab', input_T, input_neurons)  # [n_input, n_input, rank, rank]
    overlap_matrix = products.norm(dim=(-2, -1))  # [n_input, n_input]
    mask = ~torch.eye(n_input, dtype=torch.bool, device=overlap_matrix.device)
    off_diag = overlap_matrix[mask]
    print(f"    Mean: {off_diag.mean().item():.4f}, Max: {off_diag.max().item():.4f}")
    input_results['overlap_mean'] = off_diag.mean().item()
    input_results['overlap_max'] = off_diag.max().item()

    results['input'] = input_results

    # ========== Process Neurons (Householder vectors) ==========
    print("\n📌 Process Neurons (Householder): [n_process, rank]")
    process_neurons = shared.process_neurons.data  # [n_process, rank]
    n_process = process_neurons.shape[0]

    process_results = {'shape': [n_process, rank], 'neurons': {}}

    # Norm distribution (should be ≈ 1 for Householder)
    norms = process_neurons.norm(dim=-1)
    print(f"  Norms: mean={norms.mean().item():.4f}, std={norms.std().item():.4f}, range=[{norms.min().item():.4f}, {norms.max().item():.4f}]")

    # Cosine similarity (diversity)
    v_normalized = F.normalize(process_neurons, dim=-1)
    cos_sim = v_normalized @ v_normalized.T
    mask = ~torch.eye(n_process, dtype=torch.bool, device=cos_sim.device)
    off_diag_sim = cos_sim[mask]
    print(f"  Cosine similarity: mean={off_diag_sim.abs().mean().item():.4f}, max={off_diag_sim.abs().max().item():.4f}")

    process_results['norm_mean'] = norms.mean().item()
    process_results['norm_std'] = norms.std().item()
    process_results['cosine_sim_mean'] = off_diag_sim.abs().mean().item()
    process_results['cosine_sim_max'] = off_diag_sim.abs().max().item()

    # Householder matrix condition numbers
    cond_numbers = []
    for i in range(n_process):
        v = process_neurons[i]
        v_norm = v / (v.norm() + 1e-8)
        H = torch.eye(rank, device=v.device) - 2 * torch.outer(v_norm, v_norm)
        _, s, _ = torch.linalg.svd(H)
        cond = (s[0] / (s[-1] + 1e-10)).item()
        cond_numbers.append(cond)
        process_results['neurons'][i] = {
            'norm': norms[i].item(),
            'condition': cond
        }

    print(f"  Householder condition: mean={np.mean(cond_numbers):.4f}, max={np.max(cond_numbers):.4f}")
    process_results['condition_mean'] = np.mean(cond_numbers)

    results['process'] = process_results

    # ========== Output Neurons ==========
    print("\n📌 Output Neurons: [n_output, rank, d_model]")
    output_neurons = shared.output_neurons.data  # [n_output, rank, d_model]
    n_output = output_neurons.shape[0]

    output_results = {'shape': [n_output, rank, d_model], 'neurons': {}}

    output_conds = []
    output_orth_errors = []
    for i in range(n_output):
        W = output_neurons[i]  # [rank, d_model]
        _, s, _ = torch.linalg.svd(W, full_matrices=False)
        cond = (s[0] / (s[-1] + 1e-10)).item()
        output_conds.append(cond)

        # W @ W.T should be identity (rank × rank)
        WWt = W @ W.T
        I = torch.eye(rank, device=W.device)
        orth_error = (WWt - I).abs().max().item()
        output_orth_errors.append(orth_error)

        output_results['neurons'][i] = {
            'condition': cond,
            'orthogonality_error': orth_error
        }

        print(f"  Neuron {i}: cond={cond:.2f}, orth_err={orth_error:.2e}")

    output_results['avg_condition'] = np.mean(output_conds)
    output_results['avg_orth_error'] = np.mean(output_orth_errors)

    # Output neuron overlap
    print(f"\n  Output neuron overlap:")
    products = torch.einsum('iad,jbd->ijab', output_neurons, output_neurons)
    overlap_matrix = products.norm(dim=(-2, -1))
    mask = ~torch.eye(n_output, dtype=torch.bool, device=overlap_matrix.device)
    off_diag = overlap_matrix[mask]
    print(f"    Mean: {off_diag.mean().item():.4f}, Max: {off_diag.max().item():.4f}")
    output_results['overlap_mean'] = off_diag.mean().item()
    output_results['overlap_max'] = off_diag.max().item()

    results['output'] = output_results

    # ========== Knowledge Neurons ==========
    print("\n📌 Knowledge Neurons: K=[n_knowledge, rank], V=[n_knowledge, d_model]")
    knowledge_K = shared.knowledge_K.data  # [n_knowledge, rank]
    knowledge_V = shared.knowledge_V.data  # [n_knowledge, d_model]
    n_knowledge = knowledge_K.shape[0]

    knowledge_results = {'n_knowledge': n_knowledge}

    # K statistics
    K_norms = knowledge_K.norm(dim=-1)
    K_normalized = F.normalize(knowledge_K, dim=-1)
    K_sim = K_normalized @ K_normalized.T
    K_mask = ~torch.eye(n_knowledge, dtype=torch.bool, device=K_sim.device)
    K_off_diag = K_sim[K_mask]

    print(f"  Knowledge K: norm_mean={K_norms.mean().item():.4f}, sim_mean={K_off_diag.abs().mean().item():.4f}")
    knowledge_results['K_norm_mean'] = K_norms.mean().item()
    knowledge_results['K_sim_mean'] = K_off_diag.abs().mean().item()
    knowledge_results['K_sim_max'] = K_off_diag.abs().max().item()

    # V statistics
    V_norms = knowledge_V.norm(dim=-1)
    print(f"  Knowledge V: norm_mean={V_norms.mean().item():.4f}, std={V_norms.std().item():.4f}")
    knowledge_results['V_norm_mean'] = V_norms.mean().item()
    knowledge_results['V_norm_std'] = V_norms.std().item()

    results['knowledge'] = knowledge_results

    return results


# ============================================================
# 2. Routing Pattern Analysis (Q/K/V/O)
# ============================================================

def analyze_routing_patterns(model, dataloader, device, max_batches=10):
    """Analyze routing patterns for Q/K/V/O separately"""
    print("\n" + "=" * 60)
    print("2. ROUTING PATTERN ANALYSIS (Q/K/V/O)")
    print("=" * 60)

    model = get_underlying_model(model)
    model.eval()

    n_layers = len(model.layers)
    n_process = model.n_process
    process_k = model.process_k

    # Track usage per component
    components = ['Q', 'K', 'V', 'O']
    usage = {comp: torch.zeros(n_layers, n_process, device=device) for comp in components}
    cooccurrence = {comp: torch.zeros(n_layers, n_process, n_process, device=device) for comp in components}

    # Track Q-K-V alignment
    qkv_alignment = torch.zeros(n_layers, device=device)  # How often Q/K/V select same neurons
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Routing Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            B, S = input_ids.shape
            total_tokens += B * S

            _, routing_infos = model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                attn_routing = routing_info['attention']

                # Process each component
                for comp in components:
                    if comp == 'O':
                        routing = attn_routing['routing_O']
                    else:
                        routing = attn_routing[f'routing_{comp}']

                    process_idx = routing['process_indices']  # [B, S, k]

                    # Usage count
                    idx_flat = process_idx.reshape(-1)
                    usage[comp][layer_idx] += torch.bincount(idx_flat, minlength=n_process).float()

                    # Co-occurrence
                    idx_2d = process_idx.reshape(-1, process_k)
                    idx_i = idx_2d.unsqueeze(2).expand(-1, -1, process_k)
                    idx_j = idx_2d.unsqueeze(1).expand(-1, process_k, -1)
                    linear_idx = idx_i.reshape(-1) * n_process + idx_j.reshape(-1)
                    ones = torch.ones_like(linear_idx, dtype=torch.float32)
                    cooccurrence[comp][layer_idx].view(-1).scatter_add_(0, linear_idx, ones)

                # Q-K-V alignment
                Q_idx = set(attn_routing['routing_Q']['process_indices'].reshape(-1).tolist())
                K_idx = set(attn_routing['routing_K']['process_indices'].reshape(-1).tolist())
                V_idx = set(attn_routing['routing_V']['process_indices'].reshape(-1).tolist())
                qkv_intersection = len(Q_idx & K_idx & V_idx)
                qkv_union = len(Q_idx | K_idx | V_idx)
                qkv_alignment[layer_idx] += qkv_intersection / (qkv_union + 1e-10)

    # Compute statistics
    results = {'components': {}, 'global': {}}

    print("\n📌 Per-Component Routing Statistics:")
    for comp in components:
        usage_sum = usage[comp].sum(dim=1, keepdim=True) + 1e-10
        usage_norm = usage[comp] / usage_sum

        # Entropy
        entropy = -(usage_norm * torch.log(usage_norm + 1e-10)).sum(dim=1)
        max_entropy = math.log(n_process)
        norm_entropy = entropy / max_entropy

        # Gini
        sorted_usage, _ = torch.sort(usage_norm, dim=1)
        n = n_process
        index = torch.arange(1, n + 1, dtype=torch.float32, device=device).unsqueeze(0)
        gini = ((2 * index - n - 1) * sorted_usage).sum(dim=1) / (n * sorted_usage.sum(dim=1) + 1e-10)

        # Top-k concentration
        top5 = torch.topk(usage_norm, 5, dim=1)[0].sum(dim=1)
        top10 = torch.topk(usage_norm, 10, dim=1)[0].sum(dim=1)

        results['components'][comp] = {
            'entropy_per_layer': norm_entropy.tolist(),
            'gini_per_layer': gini.tolist(),
            'top5_per_layer': top5.tolist(),
            'top10_per_layer': top10.tolist(),
            'avg_entropy': norm_entropy.mean().item(),
            'avg_gini': gini.mean().item(),
        }

        print(f"\n  {comp}:")
        for layer_idx in range(n_layers):
            print(f"    Layer {layer_idx}: entropy={norm_entropy[layer_idx].item():.3f}, "
                  f"gini={gini[layer_idx].item():.3f}, top5={top5[layer_idx].item():.1%}")

    # QKV alignment
    qkv_alignment /= max_batches
    results['qkv_alignment'] = qkv_alignment.tolist()
    print(f"\n📌 Q-K-V Alignment (Jaccard):")
    for layer_idx in range(n_layers):
        print(f"  Layer {layer_idx}: {qkv_alignment[layer_idx].item():.3f}")

    # Global averages
    results['global'] = {
        'avg_entropy': np.mean([results['components'][c]['avg_entropy'] for c in components]),
        'avg_gini': np.mean([results['components'][c]['avg_gini'] for c in components]),
    }

    return results, usage, cooccurrence


# ============================================================
# 3. Information Flow Analysis
# ============================================================

def analyze_information_flow(model, dataloader, device, max_batches=5):
    """Analyze information flow through Attention + Memory"""
    print("\n" + "=" * 60)
    print("3. INFORMATION FLOW ANALYSIS")
    print("=" * 60)

    model = get_underlying_model(model)
    model.eval()

    n_layers = len(model.layers)

    # Track norms at each stage
    layer_stats = {f'layer_{i}': {
        'input_norm': [],
        'after_compress_Q': [],
        'after_process_Q': [],
        'after_attention': [],
        'after_expand': [],
        'after_memory': [],
        'output_norm': [],
    } for i in range(n_layers)}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Info Flow", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            B, S = input_ids.shape

            # Manual forward pass
            pos = torch.arange(S, device=device).unsqueeze(0)
            x = model.token_emb(input_ids) + model.pos_emb(pos)
            x = model.dropout(x)

            mask = model.causal_mask[:, :, :S, :S]

            for layer_idx, layer in enumerate(model.layers):
                layer_key = f'layer_{layer_idx}'

                # Input
                residual = x
                x_norm = layer.norm1(x)
                layer_stats[layer_key]['input_norm'].append(x_norm.norm(dim=-1).mean().item())

                # Attention forward (get intermediate values)
                attn = layer.attention

                # Q compression
                Q, routing_Q = attn.compressor_Q(x_norm)
                layer_stats[layer_key]['after_compress_Q'].append(Q.norm(dim=-1).mean().item())

                # Full attention
                attn_out, _ = attn(x_norm, mask)
                layer_stats[layer_key]['after_attention'].append(attn_out.norm(dim=-1).mean().item())

                x = residual + layer.dropout(attn_out)

                # Memory
                residual = x
                x_norm = layer.norm2(x)
                mem_out, _ = layer.memory(x_norm)
                layer_stats[layer_key]['after_memory'].append(mem_out.norm(dim=-1).mean().item())

                x = residual + layer.dropout(mem_out)
                layer_stats[layer_key]['output_norm'].append(x.norm(dim=-1).mean().item())

    # Compute averages
    results = {'layers': {}}

    print("\n📌 Information Flow (Norm at each stage):")
    for layer_idx in range(n_layers):
        layer_key = f'layer_{layer_idx}'
        stats = layer_stats[layer_key]

        avg_input = np.mean(stats['input_norm'])
        avg_compress = np.mean(stats['after_compress_Q'])
        avg_attn = np.mean(stats['after_attention'])
        avg_mem = np.mean(stats['after_memory'])
        avg_output = np.mean(stats['output_norm'])

        # Ratios
        compress_ratio = avg_compress / (avg_input + 1e-10)
        attn_ratio = avg_attn / (avg_input + 1e-10)
        mem_ratio = avg_mem / (avg_attn + 1e-10)
        total_ratio = avg_output / (avg_input + 1e-10)

        results['layers'][layer_key] = {
            'input_norm': avg_input,
            'compress_ratio': compress_ratio,
            'attention_ratio': attn_ratio,
            'memory_ratio': mem_ratio,
            'total_ratio': total_ratio,
        }

        print(f"  Layer {layer_idx}: input={avg_input:.2f} → compress={compress_ratio:.2f}x "
              f"→ attn={attn_ratio:.2f}x → mem={mem_ratio:.2f}x → total={total_ratio:.2f}x")

    return results


# ============================================================
# 4. Memory/Knowledge Analysis
# ============================================================

def analyze_memory_patterns(model, dataloader, device, max_batches=10):
    """Analyze NeuronMemory knowledge retrieval patterns"""
    print("\n" + "=" * 60)
    print("4. MEMORY/KNOWLEDGE ANALYSIS")
    print("=" * 60)

    model = get_underlying_model(model)
    model.eval()

    n_layers = len(model.layers)
    n_knowledge = model.n_knowledge
    knowledge_k = model.knowledge_k

    # Track knowledge usage
    knowledge_usage = torch.zeros(n_layers, n_knowledge, device=device)
    knowledge_weights = torch.zeros(n_layers, n_knowledge, device=device)

    # Track per-layer retrieval entropy
    retrieval_entropy = []

    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Memory Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            B, S = input_ids.shape
            total_tokens += B * S

            _, routing_infos = model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                mem_routing = routing_info['memory']
                k_idx = mem_routing['knowledge_indices']  # [B, S, k]
                k_weights = mem_routing['knowledge_weights']  # [B, S, k]

                # Usage count
                idx_flat = k_idx.reshape(-1)
                knowledge_usage[layer_idx] += torch.bincount(idx_flat, minlength=n_knowledge).float()

                # Weighted usage
                for i in range(knowledge_k):
                    idx_i = k_idx[:, :, i].reshape(-1)
                    w_i = k_weights[:, :, i].reshape(-1)
                    knowledge_weights[layer_idx].scatter_add_(0, idx_i, w_i)

    # Compute statistics
    results = {'layers': {}, 'global': {}}

    print("\n📌 Knowledge Retrieval Statistics:")

    # Normalize usage
    usage_sum = knowledge_usage.sum(dim=1, keepdim=True) + 1e-10
    usage_norm = knowledge_usage / usage_sum

    # Per-layer analysis
    for layer_idx in range(n_layers):
        layer_usage = usage_norm[layer_idx]

        # Entropy
        entropy = -(layer_usage * torch.log(layer_usage + 1e-10)).sum()
        max_entropy = math.log(n_knowledge)
        norm_entropy = entropy / max_entropy

        # Gini
        sorted_usage, _ = torch.sort(layer_usage)
        n = n_knowledge
        index = torch.arange(1, n + 1, dtype=torch.float32, device=device)
        gini = ((2 * index - n - 1) * sorted_usage).sum() / (n * sorted_usage.sum() + 1e-10)

        # Top-k
        top5 = torch.topk(layer_usage, 5)[0].sum().item()
        top10 = torch.topk(layer_usage, 10)[0].sum().item()

        # Usage rate (how many knowledge neurons are used)
        usage_rate = (layer_usage > 1e-6).float().mean().item()

        results['layers'][f'layer_{layer_idx}'] = {
            'entropy': norm_entropy.item(),
            'gini': gini.item(),
            'top5_concentration': top5,
            'top10_concentration': top10,
            'usage_rate': usage_rate,
        }

        print(f"  Layer {layer_idx}: entropy={norm_entropy.item():.3f}, gini={gini.item():.3f}, "
              f"usage={usage_rate:.1%}, top5={top5:.1%}")

    # Global statistics
    global_usage = knowledge_usage.sum(dim=0)
    global_usage = global_usage / (global_usage.sum() + 1e-10)

    # Most used knowledge neurons
    top_neurons = torch.topk(global_usage, 10)
    print("\n📌 Top 10 Most Used Knowledge Neurons:")
    for i, (idx, prob) in enumerate(zip(top_neurons.indices.tolist(), top_neurons.values.tolist())):
        print(f"  {i+1}. Knowledge #{idx}: {prob:.2%}")
        results[f'top_knowledge_{i}'] = {'index': idx, 'usage': prob}

    # Least used (potential dead neurons)
    bottom_neurons = torch.topk(global_usage, 10, largest=False)
    dead_count = (global_usage < 1e-6).sum().item()
    print(f"\n📌 Potentially Dead Knowledge Neurons: {dead_count}/{n_knowledge}")
    results['global']['dead_count'] = dead_count
    results['global']['avg_entropy'] = np.mean([results['layers'][f'layer_{i}']['entropy'] for i in range(n_layers)])

    return results, knowledge_usage


# ============================================================
# 5. Layer Router Comparison
# ============================================================

def analyze_layer_routers(model):
    """Compare routers across layers (since neurons are shared)"""
    print("\n" + "=" * 60)
    print("5. LAYER ROUTER COMPARISON")
    print("=" * 60)

    model = get_underlying_model(model)
    n_layers = len(model.layers)

    results = {'layers': {}}

    print("\n📌 Router Weight Norms:")
    print("  (Each layer has independent routers but shares the same neurons)")

    router_types = ['input', 'process_Q', 'process_K', 'process_V', 'process_O', 'output', 'memory_Q']

    for layer_idx, layer in enumerate(model.layers):
        attn = layer.attention
        mem = layer.memory

        layer_results = {}

        # Input routers (Q, K, V)
        for comp, compressor in [('Q', attn.compressor_Q), ('K', attn.compressor_K), ('V', attn.compressor_V)]:
            layer_results[f'input_{comp}'] = compressor.input_router.weight.norm().item()
            layer_results[f'process_{comp}'] = compressor.process_router.weight.norm().item()

        # Output router (O)
        layer_results['process_O'] = attn.expander_O.process_router.weight.norm().item()
        layer_results['output_O'] = attn.expander_O.output_router.weight.norm().item()

        # Memory query
        layer_results['memory_Q'] = mem.W_Q.weight.norm().item()

        results['layers'][f'layer_{layer_idx}'] = layer_results

        print(f"\n  Layer {layer_idx}:")
        print(f"    Input routers (Q/K/V): {layer_results['input_Q']:.4f} / {layer_results['input_K']:.4f} / {layer_results['input_V']:.4f}")
        print(f"    Process routers (Q/K/V/O): {layer_results['process_Q']:.4f} / {layer_results['process_K']:.4f} / {layer_results['process_V']:.4f} / {layer_results['process_O']:.4f}")
        print(f"    Output router: {layer_results['output_O']:.4f}")
        print(f"    Memory W_Q: {layer_results['memory_Q']:.4f}")

    # Cross-layer similarity
    print("\n📌 Router Similarity Across Layers:")
    for comp in ['Q', 'K', 'V']:
        routers = [getattr(model.layers[i].attention, f'compressor_{comp}').input_router.weight.data for i in range(n_layers)]
        routers = torch.stack([r.flatten() for r in routers])
        routers_norm = F.normalize(routers, dim=-1)
        sim = routers_norm @ routers_norm.T

        print(f"  {comp} input router similarity: {sim.mean().item():.4f} (mean), {sim.min().item():.4f} (min)")

    return results


# ============================================================
# 6. Attention Pattern Analysis
# ============================================================

def analyze_attention_patterns(model, dataloader, device, max_batches=5):
    """Analyze attention patterns"""
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
        'sparsity': [],
        'max_weight': [],
        'head_variance': [],
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
                attn = layer.attention

                residual = x
                x_norm = layer.norm1(x)

                # Get Q, K, V
                Q, _ = attn.compressor_Q(x_norm)
                K, _ = attn.compressor_K(x_norm)
                V, _ = attn.compressor_V(x_norm)

                # Multi-head
                d_head = attn.d_head
                Q = Q.view(B, S, n_heads, d_head).transpose(1, 2)
                K = K.view(B, S, n_heads, d_head).transpose(1, 2)
                V = V.view(B, S, n_heads, d_head).transpose(1, 2)

                # Attention weights
                attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(d_head)
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, S, S]

                # Statistics
                # Entropy
                entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1)
                max_entropy = math.log(S)
                norm_entropy = entropy / max_entropy

                attn_stats[layer_key]['entropy'].append(norm_entropy.mean().item())
                attn_stats[layer_key]['sparsity'].append((attn_weights < 0.01).float().mean().item())
                attn_stats[layer_key]['max_weight'].append(attn_weights.max(dim=-1)[0].mean().item())

                # Head variance (how different are heads)
                head_entropy = norm_entropy.mean(dim=(0, 2))  # [H]
                attn_stats[layer_key]['head_variance'].append(head_entropy.var().item())

                # Continue forward
                attn_out = attn_weights @ V
                attn_out = attn_out.transpose(1, 2).reshape(B, S, attn.rank)
                out, _ = attn.expander_O(attn_out)

                x = residual + layer.dropout(out)

                # Memory
                residual = x
                mem_out, _ = layer.memory(layer.norm2(x))
                x = residual + layer.dropout(mem_out)

    # Compute averages
    results = {'layers': {}}

    print("\n📌 Attention Statistics:")
    for layer_idx in range(n_layers):
        layer_key = f'layer_{layer_idx}'
        stats = attn_stats[layer_key]

        results['layers'][layer_key] = {
            'entropy': np.mean(stats['entropy']),
            'sparsity': np.mean(stats['sparsity']),
            'max_weight': np.mean(stats['max_weight']),
            'head_variance': np.mean(stats['head_variance']),
        }

        print(f"  Layer {layer_idx}: entropy={np.mean(stats['entropy']):.3f}, "
              f"sparsity={np.mean(stats['sparsity']):.1%}, max_weight={np.mean(stats['max_weight']):.3f}")

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

    # 1. Routing comparison across Q/K/V/O
    if 'routing' in all_results:
        routing = all_results['routing']
        components = ['Q', 'K', 'V', 'O']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Entropy comparison
        for comp in components:
            entropies = routing['components'][comp]['entropy_per_layer']
            axes[0].plot(entropies, marker='o', label=comp)

        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Normalized Entropy')
        axes[0].set_title('Routing Entropy by Component')
        axes[0].legend()
        axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.3)

        # Gini comparison
        for comp in components:
            ginis = routing['components'][comp]['gini_per_layer']
            axes[1].plot(ginis, marker='o', label=comp)

        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Gini Coefficient')
        axes[1].set_title('Routing Gini by Component')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'routing_qkvo.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"  Saved: routing_qkvo.png")

    # 2. Information flow
    if 'info_flow' in all_results:
        info_flow = all_results['info_flow']
        n_layers = len(info_flow['layers'])

        fig, ax = plt.subplots(figsize=(10, 6))

        stages = ['compress', 'attention', 'memory', 'total']
        x = np.arange(len(stages))
        width = 0.15

        for layer_idx in range(n_layers):
            layer_data = info_flow['layers'][f'layer_{layer_idx}']
            ratios = [layer_data[f'{s}_ratio'] for s in stages]
            ax.bar(x + layer_idx * width, ratios, width, label=f'Layer {layer_idx}')

        ax.set_ylabel('Norm Ratio')
        ax.set_title('Information Flow Through Attention + Memory')
        ax.set_xticks(x + width * (n_layers - 1) / 2)
        ax.set_xticklabels(['Compress', 'Attention', 'Memory', 'Total'])
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'info_flow.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"  Saved: info_flow.png")

    # 3. Memory usage
    if 'memory' in all_results:
        memory = all_results['memory']
        n_layers = len([k for k in memory['layers'].keys()])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        layers = list(range(n_layers))
        entropies = [memory['layers'][f'layer_{i}']['entropy'] for i in layers]
        ginis = [memory['layers'][f'layer_{i}']['gini'] for i in layers]

        axes[0].bar(layers, entropies, color='steelblue')
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Normalized Entropy')
        axes[0].set_title('Knowledge Retrieval Entropy')
        axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)

        axes[1].bar(layers, ginis, color='coral')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Gini Coefficient')
        axes[1].set_title('Knowledge Retrieval Gini')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_stats.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"  Saved: memory_stats.png")

    # 4. Attention patterns
    if 'attention' in all_results:
        attention = all_results['attention']
        n_layers = len(attention['layers'])

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
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"  Saved: attention_stats.png")

    # 5. SharedNeurons visualization
    if 'shared_neurons' in all_results:
        shared = all_results['shared_neurons']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Input neurons condition
        if 'input' in shared:
            n_input = len(shared['input']['neurons'])
            input_conds = [shared['input']['neurons'][i]['condition'] for i in range(n_input)]
            axes[0, 0].bar(range(len(input_conds)), input_conds, color='steelblue')
            axes[0, 0].set_xlabel('Input Neuron')
            axes[0, 0].set_ylabel('Condition Number')
            axes[0, 0].set_title('Input Neurons Condition Numbers')

        # Process neurons norms
        if 'process' in shared:
            n_process = len(shared['process']['neurons'])
            process_norms = [shared['process']['neurons'][i]['norm'] for i in range(n_process)]
            axes[0, 1].bar(range(len(process_norms)), process_norms, color='coral')
            axes[0, 1].set_xlabel('Process Neuron')
            axes[0, 1].set_ylabel('Norm')
            axes[0, 1].set_title('Process Neurons Norms (should be ~1)')
            axes[0, 1].axhline(y=1.0, color='g', linestyle='--', alpha=0.5)

        # Output neurons condition
        if 'output' in shared:
            n_output = len(shared['output']['neurons'])
            output_conds = [shared['output']['neurons'][i]['condition'] for i in range(n_output)]
            axes[1, 0].bar(range(len(output_conds)), output_conds, color='seagreen')
            axes[1, 0].set_xlabel('Output Neuron')
            axes[1, 0].set_ylabel('Condition Number')
            axes[1, 0].set_title('Output Neurons Condition Numbers')

        # Knowledge statistics
        if 'knowledge' in shared:
            knowledge = shared['knowledge']
            labels = ['K_norm', 'K_sim', 'V_norm']
            values = [knowledge['K_norm_mean'], knowledge['K_sim_mean'], knowledge['V_norm_mean']]
            axes[1, 1].bar(labels, values, color='purple')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_title('Knowledge Neurons Statistics')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shared_neurons.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"  Saved: shared_neurons.png")

    print(f"\n  All visualizations saved to: {output_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN v8.0 Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--val_data', type=str,
                        default='/content/drive/MyDrive/data/validation/wikitext_5to1_texts.pkl',
                        help='Path to validation data')
    parser.add_argument('--max_batches', type=int, default=10,
                        help='Max batches for runtime analysis')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./analysis_v8_output',
                        help='Output directory for results')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Handle checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        best_model = checkpoint_path / 'best_model.pt'
        if best_model.exists():
            checkpoint_path = best_model
        else:
            pt_files = list(checkpoint_path.glob('*.pt'))
            if pt_files:
                checkpoint_path = max(pt_files, key=lambda p: p.stat().st_mtime)
            else:
                raise FileNotFoundError(f"No .pt files found in {args.checkpoint}")
        print(f"Found checkpoint: {checkpoint_path}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('model_config', checkpoint.get('config', {}))
    model_version = config.get('model_version', '8.0')
    print(f"Checkpoint model version: {model_version}")

    if model_version != "8.0":
        print(f"Warning: This script is for v8.0, but checkpoint is v{model_version}")
        print("Consider using analyze_v79.py instead.")

    # Create model
    print(f"\nCreating model v8.0...")
    model = DAWN(
        vocab_size=config.get('vocab_size', 30522),
        d_model=config.get('d_model', 256),
        n_layers=config.get('n_layers', 4),
        n_heads=config.get('n_heads', 4),
        rank=config.get('rank', config.get('basis_rank', 64)),
        max_seq_len=config.get('max_seq_len', 128),
        n_input=config.get('n_input', 8),
        n_process=config.get('n_process', 32),
        n_output=config.get('n_output', 8),
        process_k=config.get('process_k', 3),
        n_knowledge=config.get('n_knowledge', 64),
        knowledge_k=config.get('knowledge_k', 8),
        dropout=config.get('dropout', 0.1),
    )

    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  Removing torch.compile wrapper prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model: DAWN v{model.__version__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optional compile
    if hasattr(torch, 'compile'):
        print("\n⚡ Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')

    # Load data
    print(f"\nLoading validation data from: {args.val_data}")
    import pickle
    with open(args.val_data, 'rb') as f:
        val_texts = pickle.load(f)
    print(f"Loaded {len(val_texts)} validation texts")

    # Tokenizer and dataloader
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
    print("STARTING DAWN v8.0 ANALYSIS")
    print("=" * 60)

    all_results = {}

    # 1. SharedNeurons Analysis
    all_results['shared_neurons'] = analyze_shared_neurons(model)

    # 2. Routing Patterns (Q/K/V/O)
    routing_results, usage, cooccurrence = analyze_routing_patterns(model, dataloader, device, args.max_batches)
    all_results['routing'] = routing_results

    # 3. Information Flow
    all_results['info_flow'] = analyze_information_flow(model, dataloader, device, max_batches=5)

    # 4. Memory/Knowledge Analysis
    memory_results, knowledge_usage = analyze_memory_patterns(model, dataloader, device, args.max_batches)
    all_results['memory'] = memory_results

    # 5. Layer Router Comparison
    all_results['routers'] = analyze_layer_routers(model)

    # 6. Attention Patterns
    all_results['attention'] = analyze_attention_patterns(model, dataloader, device, max_batches=5)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    underlying = get_underlying_model(model)
    print("\n📊 Key Findings:")
    print(f"  Orthogonality loss: {underlying.orthogonality_loss().item():.6f}")
    print(f"  Process norm loss: {underlying.process_norm_loss().item():.6f}")
    print(f"  Knowledge diversity loss: {underlying.knowledge_diversity_loss().item():.6f}")
    print(f"  Avg routing entropy: {all_results['routing']['global']['avg_entropy']:.3f}")
    print(f"  Avg routing gini: {all_results['routing']['global']['avg_gini']:.3f}")
    print(f"  Avg memory entropy: {all_results['memory']['global']['avg_entropy']:.3f}")
    print(f"  Dead knowledge neurons: {all_results['memory']['global']['dead_count']}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

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
