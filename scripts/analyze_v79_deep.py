"""
DAWN Deep Analysis Script (v7.9 / v8.0 compatible)
심층 분석: 뉴런 흐름, Householder 변환, Attention-FFN/Memory 상호작용

분석 항목:
1. 레이어간 뉴런 흐름 추적 (Sankey diagram)
2. Householder 변환 효과 분석
3. Attention-FFN 상호작용 분석 (v7.9) / Attention-Memory 분석 (v8.0)
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Version-agnostic utilities
from scripts.analysis_utils import (
    load_model, get_underlying_model, get_routing_info_compat,
    get_neurons, get_layer_neurons, has_ffn, has_memory
)

# Optional imports
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
    print("Warning: matplotlib not available")

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not available, Sankey diagrams will use matplotlib")


def get_underlying_model(model):
    """Get underlying model from torch.compile wrapper"""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def simple_pos_tag(token):
    """Simple rule-based POS tagging"""
    token_lower = token.lower().strip()

    if token in '.,!?;:\'"()-[]{}':
        return 'PUNCT'
    if token.isdigit() or token.replace('.', '').replace(',', '').isdigit():
        return 'NUM'
    if token_lower in ['a', 'an', 'the']:
        return 'DET'
    if token_lower in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']:
        return 'PRON'
    if token_lower in ['in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about']:
        return 'ADP'
    if token_lower in ['and', 'or', 'but', 'so', 'yet', 'nor']:
        return 'CONJ'
    if token_lower in ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had']:
        return 'AUX'
    if token_lower.endswith('ing') or token_lower.endswith('ed'):
        return 'VERB'
    if token_lower.endswith('ly'):
        return 'ADV'
    return 'NOUN'


class DeepAnalyzer:
    """Deep analysis for DAWN v7.9 / v8.0"""

    def __init__(self, model, tokenizer, device, version="7.9"):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.version = version

        self.n_layers = len(self.model.layers)
        self.n_process = self.model.n_process
        self.process_k = self.model.process_k
        self.vocab_size = self.tokenizer.vocab_size

        # Version-specific
        self.has_ffn = has_ffn(version)
        self.has_memory = has_memory(version)
        if self.has_memory:
            self.n_knowledge = self.model.n_knowledge
            self.knowledge_k = self.model.knowledge_k

    # ============================================================
    # Analysis 1: Layer-wise Neuron Flow
    # ============================================================

    @torch.no_grad()
    def analyze_neuron_flow(self, dataloader, max_batches=50, target_tokens=None):
        """Track neuron selection across layers for specific tokens"""
        print("\n" + "=" * 60)
        print("ANALYSIS 1: LAYER-WISE NEURON FLOW")
        print("=" * 60)

        self.model.eval()

        # Default target tokens
        if target_tokens is None:
            target_tokens = ['the', 'a', 'of', 'to', 'and', 'in', 'is', '.', ',', '[PAD]', '[CLS]', '[SEP]']

        # Get token IDs
        target_token_ids = {}
        for token in target_tokens:
            if token.startswith('['):
                tid = self.tokenizer.convert_tokens_to_ids(token)
            else:
                tid = self.tokenizer.convert_tokens_to_ids(token)
            if tid != self.tokenizer.unk_token_id:
                target_token_ids[token] = tid

        print(f"  Tracking {len(target_token_ids)} target tokens")

        # Flow tracking: {token: {layer_path: count}}
        # layer_path is tuple of tuples: ((L0_neurons), (L1_neurons), ...)
        flow_counts = {token: defaultdict(int) for token in target_token_ids}
        token_counts = {token: 0 for token in target_token_ids}

        # Also track per-layer neuron usage per token
        token_layer_neurons = {token: {f'layer_{l}': defaultdict(int)
                                       for l in range(self.n_layers)}
                              for token in target_token_ids}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Neuron Flow", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Forward with routing info
            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # For each target token
            for token, tid in target_token_ids.items():
                # Find positions where this token appears
                mask = (input_ids == tid)
                if not mask.any():
                    continue

                positions = mask.nonzero()  # [N, 2] (batch_idx, seq_idx)
                token_counts[token] += len(positions)

                for pos in positions:
                    b, s = pos[0].item(), pos[1].item()

                    # Collect neuron indices across layers
                    layer_path = []
                    for layer_idx, routing_info in enumerate(routing_infos):
                        compat = get_routing_info_compat(routing_info, self.version)
                        neurons = compat['process_indices'][b, s].tolist()
                        neurons_tuple = tuple(sorted(neurons))
                        layer_path.append(neurons_tuple)

                        # Per-layer tracking
                        for n in neurons:
                            token_layer_neurons[token][f'layer_{layer_idx}'][n] += 1

                    flow_counts[token][tuple(layer_path)] += 1

        # Analyze results
        results = {'tokens': {}, 'common_paths': []}

        print("\n  Token-wise neuron flow patterns:")
        for token in target_token_ids:
            if token_counts[token] == 0:
                continue

            # Get top paths for this token
            paths = flow_counts[token]
            top_paths = sorted(paths.items(), key=lambda x: -x[1])[:5]

            # Get top neurons per layer
            layer_top_neurons = {}
            for layer_key, neuron_counts in token_layer_neurons[token].items():
                if neuron_counts:
                    top = sorted(neuron_counts.items(), key=lambda x: -x[1])[:3]
                    layer_top_neurons[layer_key] = [n for n, c in top]

            results['tokens'][token] = {
                'count': token_counts[token],
                'top_paths': [(list(p), c) for p, c in top_paths],
                'layer_top_neurons': layer_top_neurons,
                'n_unique_paths': len(paths)
            }

            print(f"\n    '{token}' (n={token_counts[token]}, {len(paths)} unique paths):")
            for layer_idx in range(self.n_layers):
                neurons = layer_top_neurons.get(f'layer_{layer_idx}', [])
                print(f"      L{layer_idx}: top neurons {neurons}")

        # Find globally common paths
        all_paths = Counter()
        for token_paths in flow_counts.values():
            all_paths.update(token_paths)

        common_paths = all_paths.most_common(10)
        results['common_paths'] = [(list(p), c) for p, c in common_paths]

        print("\n  Top 10 most common neuron flow paths:")
        for i, (path, count) in enumerate(common_paths):
            path_str = " -> ".join([str(list(p)) for p in path])
            print(f"    {i+1}. ({count}x): {path_str}")

        return results, flow_counts, token_layer_neurons

    def visualize_neuron_flow(self, flow_counts, token_layer_neurons, output_dir, top_tokens=5):
        """Create Sankey diagram for neuron flow"""
        print("\n  Creating neuron flow visualizations...")

        os.makedirs(output_dir, exist_ok=True)

        if HAS_PLOTLY:
            self._create_sankey_plotly(flow_counts, token_layer_neurons, output_dir, top_tokens)
        else:
            self._create_flow_matplotlib(token_layer_neurons, output_dir, top_tokens)

    def _create_sankey_plotly(self, flow_counts, token_layer_neurons, output_dir, top_tokens):
        """Create Sankey diagram using Plotly"""
        # Select top tokens by count
        token_counts = {t: sum(paths.values()) for t, paths in flow_counts.items()}
        selected_tokens = sorted(token_counts.keys(), key=lambda x: -token_counts[x])[:top_tokens]

        for token in selected_tokens:
            if not token_layer_neurons[token]['layer_0']:
                continue

            # Build Sankey data
            nodes = []
            node_idx = {}
            sources = []
            targets = []
            values = []

            # Create nodes for each layer's neurons
            for layer_idx in range(self.n_layers):
                layer_data = token_layer_neurons[token][f'layer_{layer_idx}']
                top_neurons = sorted(layer_data.keys(), key=lambda x: -layer_data[x])[:8]

                for n in top_neurons:
                    node_name = f"L{layer_idx}_N{n}"
                    node_idx[node_name] = len(nodes)
                    nodes.append(node_name)

            # Create links between consecutive layers
            for layer_idx in range(self.n_layers - 1):
                layer_data = token_layer_neurons[token][f'layer_{layer_idx}']
                next_layer_data = token_layer_neurons[token][f'layer_{layer_idx + 1}']

                top_neurons = sorted(layer_data.keys(), key=lambda x: -layer_data[x])[:8]
                next_top = sorted(next_layer_data.keys(), key=lambda x: -next_layer_data[x])[:8]

                for n1 in top_neurons:
                    for n2 in next_top:
                        src_name = f"L{layer_idx}_N{n1}"
                        tgt_name = f"L{layer_idx + 1}_N{n2}"

                        if src_name in node_idx and tgt_name in node_idx:
                            # Estimate flow (use geometric mean of counts)
                            flow = int(np.sqrt(layer_data[n1] * next_layer_data[n2]))
                            if flow > 0:
                                sources.append(node_idx[src_name])
                                targets.append(node_idx[tgt_name])
                                values.append(flow)

            if not sources:
                continue

            # Create Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes,
                    color="blue"
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values
                )
            )])

            fig.update_layout(title_text=f"Neuron Flow for '{token}'", font_size=10)

            safe_token = token.replace('[', '').replace(']', '').replace('.', 'dot')
            fig.write_html(os.path.join(output_dir, f'neuron_flow_{safe_token}.html'))
            print(f"    Saved: neuron_flow_{safe_token}.html")

    def _create_flow_matplotlib(self, token_layer_neurons, output_dir, top_tokens):
        """Create flow visualization using matplotlib (fallback)"""
        if not HAS_MATPLOTLIB:
            return

        # Select top tokens
        token_counts = {t: sum(sum(d.values()) for d in layers.values())
                       for t, layers in token_layer_neurons.items()}
        selected_tokens = sorted(token_counts.keys(), key=lambda x: -token_counts[x])[:top_tokens]

        fig, axes = plt.subplots(len(selected_tokens), 1, figsize=(14, 4 * len(selected_tokens)))
        if len(selected_tokens) == 1:
            axes = [axes]

        for ax, token in zip(axes, selected_tokens):
            # Create heatmap of neuron usage per layer
            data = np.zeros((self.n_layers, self.n_process))

            for layer_idx in range(self.n_layers):
                layer_data = token_layer_neurons[token][f'layer_{layer_idx}']
                total = sum(layer_data.values()) + 1e-10
                for n, count in layer_data.items():
                    if n < self.n_process:
                        data[layer_idx, n] = count / total

            im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
            ax.set_xlabel('Neuron Index')
            ax.set_ylabel('Layer')
            ax.set_yticks(range(self.n_layers))
            ax.set_yticklabels([f'L{i}' for i in range(self.n_layers)])
            ax.set_title(f"Neuron Usage Flow for '{token}'")
            plt.colorbar(im, ax=ax, label='Probability')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'neuron_flow_heatmap.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: neuron_flow_heatmap.png")

    # ============================================================
    # Analysis 2: Householder Transformation Effect
    # ============================================================

    @torch.no_grad()
    def analyze_householder_effect(self, dataloader, max_batches=30):
        """Analyze how Householder transformations change representations"""
        print("\n" + "=" * 60)
        print("ANALYSIS 2: HOUSEHOLDER TRANSFORMATION EFFECT")
        print("=" * 60)

        # v8.0 uses SharedNeurons - different internal structure
        if self.version == "8.0":
            print("  Note: v8.0 uses SharedNeurons (different structure)")
            print("  Analyzing shared process neurons...")
            return self._analyze_householder_v8(dataloader, max_batches)

        self.model.eval()

        # Track transformation effects
        # combination_effects: {(n1, n2, n3): [cos_sim_changes]}
        combination_counts = Counter()
        combination_cos_sims = defaultdict(list)
        combination_norm_changes = defaultdict(list)

        # Per-neuron effects
        neuron_effects = {n: {'cos_sim': [], 'norm_change': []} for n in range(self.n_process)}

        # Layer-wise effects
        layer_effects = {l: {'before_after_sim': [], 'norm_ratio': []} for l in range(self.n_layers)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Householder Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Manual forward to track intermediate values
            pos = torch.arange(S, device=self.device).unsqueeze(0)
            x = self.model.token_emb(input_ids) + self.model.pos_emb(pos)
            x = self.model.dropout(x)

            mask = self.model.causal_mask[:, :, :S, :S]

            for layer_idx, layer in enumerate(self.model.layers):
                residual = x
                x_norm = layer.norm1(x)

                qkv = layer.qkv_circuit

                # Get routing
                routing_down = qkv.router_down(x_norm)
                input_weights = routing_down.get('input_weights')
                process_indices = routing_down['process_indices']  # [B, S, k]

                # Track combinations
                for b in range(min(B, 4)):  # Sample a few batches
                    for s in range(min(S, 32)):  # Sample positions
                        combo = tuple(sorted(process_indices[b, s].tolist()))
                        combination_counts[combo] += 1

                # Analyze Q circuit transformation
                circuit_Q = qkv.circuit_Q

                # Before Householder: after input projection
                all_proj = torch.einsum('bsd,ndr->bsnr', x_norm, circuit_Q.input_neurons)
                x_compressed = (all_proj * input_weights.unsqueeze(-1)).sum(dim=2)  # [B, S, rank]

                # Apply Householder step by step
                x_before = x_compressed.clone()

                k = process_indices.shape[-1]
                idx_expanded = process_indices.unsqueeze(-1).expand(B, S, k, circuit_Q.rank)
                selected_v = circuit_Q.process_neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
                selected_v = selected_v.gather(2, idx_expanded)

                x_current = x_before.clone()
                for i in range(k):
                    v = selected_v[:, :, i, :]
                    x_next = circuit_Q.apply_householder(x_current, v)

                    # Track per-neuron effect (sample)
                    cos_sim = F.cosine_similarity(x_current[:2, :16], x_next[:2, :16], dim=-1).mean().item()
                    norm_change = (x_next[:2, :16].norm(dim=-1) / (x_current[:2, :16].norm(dim=-1) + 1e-10)).mean().item()

                    # Record for each selected neuron
                    for b in range(min(2, B)):
                        for s in range(min(16, S)):
                            n = process_indices[b, s, i].item()
                            neuron_effects[n]['cos_sim'].append(cos_sim)
                            neuron_effects[n]['norm_change'].append(norm_change)

                    x_current = x_next

                x_after = x_current

                # Layer-wise before/after comparison
                cos_sim = F.cosine_similarity(x_before, x_after, dim=-1).mean().item()
                norm_ratio = (x_after.norm(dim=-1) / (x_before.norm(dim=-1) + 1e-10)).mean().item()

                layer_effects[layer_idx]['before_after_sim'].append(cos_sim)
                layer_effects[layer_idx]['norm_ratio'].append(norm_ratio)

                # Full forward for next layer
                attn_out, _ = qkv(x_norm, mask)
                x = residual + layer.dropout(attn_out)

                residual = x
                x_norm = layer.norm2(x)
                ffn_out = layer.w_down(F.gelu(layer.w_up(x_norm)))
                x = residual + layer.dropout(ffn_out)

        # Compile results
        results = {
            'combinations': {},
            'neurons': {},
            'layers': {}
        }

        # Top combinations
        print("\n  Top 10 most common process neuron combinations:")
        top_combos = combination_counts.most_common(20)
        for i, (combo, count) in enumerate(top_combos[:10]):
            print(f"    {i+1}. {combo}: {count}x")
            results['combinations'][str(combo)] = {'count': count}

        # Per-neuron statistics
        print("\n  Per-neuron transformation statistics:")
        neuron_stats = []
        for n in range(self.n_process):
            if neuron_effects[n]['cos_sim']:
                avg_cos = np.mean(neuron_effects[n]['cos_sim'])
                avg_norm = np.mean(neuron_effects[n]['norm_change'])
                neuron_stats.append((n, avg_cos, avg_norm, len(neuron_effects[n]['cos_sim'])))
                results['neurons'][n] = {
                    'avg_cos_sim': avg_cos,
                    'avg_norm_change': avg_norm,
                    'count': len(neuron_effects[n]['cos_sim'])
                }

        # Sort by cos_sim (most transformative = lowest cos_sim)
        neuron_stats.sort(key=lambda x: x[1])
        print("    Most transformative neurons (lowest cos_sim after Householder):")
        for n, cos, norm, count in neuron_stats[:5]:
            print(f"      Neuron {n}: cos_sim={cos:.4f}, norm_change={norm:.4f} (n={count})")

        print("\n    Least transformative neurons (highest cos_sim):")
        for n, cos, norm, count in neuron_stats[-5:]:
            print(f"      Neuron {n}: cos_sim={cos:.4f}, norm_change={norm:.4f} (n={count})")

        # Layer statistics
        print("\n  Layer-wise Householder effects:")
        for layer_idx in range(self.n_layers):
            avg_sim = np.mean(layer_effects[layer_idx]['before_after_sim'])
            avg_norm = np.mean(layer_effects[layer_idx]['norm_ratio'])
            results['layers'][f'layer_{layer_idx}'] = {
                'avg_cos_sim': avg_sim,
                'avg_norm_ratio': avg_norm
            }
            print(f"    Layer {layer_idx}: before/after cos_sim={avg_sim:.4f}, norm_ratio={avg_norm:.4f}")

        return results, combination_counts, neuron_effects

    def _analyze_householder_v8(self, dataloader, max_batches=30):
        """v8.0 version: Analyze SharedNeurons Householder transformations"""
        self.model.eval()

        # v8.0 has shared neurons across all layers
        shared = self.model.shared_neurons
        process_neurons = shared.process_neurons.data  # [n_process, rank]

        # Analyze process neuron properties
        results = {'neurons': {}, 'combinations': {}, 'layers': {}}
        combination_counts = Counter()
        neuron_effects = {n: {'cos_sim': [], 'norm_change': []} for n in range(self.n_process)}

        # Compute cosine similarity matrix of process neurons
        v_norm = F.normalize(process_neurons, dim=-1)
        cos_sim_matrix = v_norm @ v_norm.T
        mask = ~torch.eye(self.n_process, dtype=torch.bool, device=cos_sim_matrix.device)
        avg_sim = cos_sim_matrix[mask].abs().mean().item()

        print(f"\n  Process neurons similarity: avg={avg_sim:.4f}")

        # Collect routing info
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Householder (v8)", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for routing_info in routing_infos:
                compat = get_routing_info_compat(routing_info, self.version)
                process_idx = compat['process_indices']
                if process_idx is not None:
                    for b in range(min(process_idx.shape[0], 4)):
                        for s in range(min(process_idx.shape[1], 32)):
                            combo = tuple(sorted(process_idx[b, s].tolist()))
                            combination_counts[combo] += 1

        # Top combinations
        print("\n  Top 10 process neuron combinations:")
        for i, (combo, count) in enumerate(combination_counts.most_common(10)):
            print(f"    {i+1}. {combo}: {count}x")
            results['combinations'][str(combo)] = {'count': count}

        # Neuron norms
        norms = process_neurons.norm(dim=-1).cpu().numpy()
        print(f"\n  Process neuron norms: mean={norms.mean():.4f}, std={norms.std():.4f}")

        for n in range(self.n_process):
            results['neurons'][n] = {
                'norm': float(norms[n]),
                'avg_sim_to_others': float(cos_sim_matrix[n, mask[n]].abs().mean().item())
            }

        return results, combination_counts, neuron_effects

    def visualize_householder(self, combination_counts, neuron_effects, output_dir):
        """Visualize Householder transformation effects"""
        print("\n  Creating Householder visualizations...")

        if not HAS_MATPLOTLIB:
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. Process neuron vectors heatmap
        print("    Creating process neuron heatmap...")
        neurons_data = get_neurons(self.model, self.version)
        process_neurons = neurons_data['process_neurons'].cpu().numpy()

        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(process_neurons, aspect='auto', cmap='RdBu', vmin=-0.3, vmax=0.3)
        ax.set_xlabel('Rank Dimension')
        ax.set_ylabel('Process Neuron')
        ax.set_title('Householder Vectors (Process Neurons)')
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'householder_vectors_heatmap.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: householder_vectors_heatmap.png")

        # 2. Neuron transformation effect
        print("    Creating transformation effect plot...")
        neuron_ids = []
        cos_sims = []
        norm_changes = []
        counts = []

        for n in range(self.n_process):
            if neuron_effects[n]['cos_sim']:
                neuron_ids.append(n)
                cos_sims.append(np.mean(neuron_effects[n]['cos_sim']))
                norm_changes.append(np.mean(neuron_effects[n]['norm_change']))
                counts.append(len(neuron_effects[n]['cos_sim']))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Cos similarity
        colors = plt.cm.viridis(np.array(counts) / max(counts))
        axes[0].bar(neuron_ids, cos_sims, color=colors)
        axes[0].set_xlabel('Neuron Index')
        axes[0].set_ylabel('Average Cosine Similarity')
        axes[0].set_title('Transformation Effect (lower = more transformative)')
        axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)

        # Norm change
        axes[1].bar(neuron_ids, norm_changes, color=colors)
        axes[1].set_xlabel('Neuron Index')
        axes[1].set_ylabel('Average Norm Ratio')
        axes[1].set_title('Norm Change (1.0 = preserved)')
        axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'transformation_effect.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: transformation_effect.png")

        # 3. Combination frequency
        print("    Creating combination frequency plot...")
        top_combos = combination_counts.most_common(20)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(len(top_combos))
        counts = [c for _, c in top_combos]
        labels = [str(combo) for combo, _ in top_combos]

        ax.bar(x, counts, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Neuron Combination')
        ax.set_ylabel('Frequency')
        ax.set_title('Top 20 Process Neuron Combinations')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'process_neuron_combinations.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: process_neuron_combinations.png")

    # ============================================================
    # Analysis 3: Attention-FFN Interaction
    # ============================================================

    @torch.no_grad()
    def analyze_attention_ffn(self, dataloader, max_batches=30):
        """Analyze interaction between NeuronAttention and FFN/Memory"""
        print("\n" + "=" * 60)
        if self.has_memory:
            print("ANALYSIS 3: ATTENTION-MEMORY INTERACTION (v8.0)")
        else:
            print("ANALYSIS 3: ATTENTION-FFN INTERACTION")
        print("=" * 60)

        self.model.eval()

        # Track contributions (ffn = FFN for v7.9, Memory for v8.0)
        layer_stats = {l: {
            'attn_norm': [],
            'ffn_norm': [],  # Or memory_norm for v8.0
            'attn_contribution': [],  # |attn| / (|attn| + |ffn/mem|)
            'residual_growth': [],
            'attn_ffn_cos': [],  # cosine sim between attn and ffn/mem outputs
        } for l in range(self.n_layers)}

        # POS-specific tracking
        pos_stats = defaultdict(lambda: {
            'attn_norm': [],
            'ffn_norm': [],
            'attn_contribution': []
        })

        # Neuron-specific FFN patterns
        neuron_ffn_norms = {n: [] for n in range(self.n_process)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Attention-FFN", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Get token strings for POS tagging
            token_strs = []
            for b in range(min(B, 4)):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[b].tolist())
                token_strs.append(tokens)

            # Manual forward
            pos = torch.arange(S, device=self.device).unsqueeze(0)
            x = self.model.token_emb(input_ids) + self.model.pos_emb(pos)
            x = self.model.dropout(x)

            mask = self.model.causal_mask[:, :, :S, :S]

            for layer_idx, layer in enumerate(self.model.layers):
                # Attention block
                residual = x
                x_norm = layer.norm1(x)

                qkv = layer.qkv_circuit
                routing_down = qkv.router_down(x_norm)
                process_indices = routing_down['process_indices']

                attn_out, _ = qkv(x_norm, mask)
                attn_norm = attn_out.norm(dim=-1)  # [B, S]

                x_after_attn = residual + layer.dropout(attn_out)

                # FFN block
                residual = x_after_attn
                x_norm = layer.norm2(x_after_attn)
                ffn_out = layer.w_down(F.gelu(layer.w_up(x_norm)))
                ffn_norm = ffn_out.norm(dim=-1)  # [B, S]

                x = residual + layer.dropout(ffn_out)

                # Compute statistics
                attn_contribution = attn_norm / (attn_norm + ffn_norm + 1e-10)
                residual_growth = x.norm(dim=-1) / (residual.norm(dim=-1) + 1e-10)
                attn_ffn_cos = F.cosine_similarity(attn_out, ffn_out, dim=-1)

                # Aggregate
                layer_stats[layer_idx]['attn_norm'].append(attn_norm.mean().item())
                layer_stats[layer_idx]['ffn_norm'].append(ffn_norm.mean().item())
                layer_stats[layer_idx]['attn_contribution'].append(attn_contribution.mean().item())
                layer_stats[layer_idx]['residual_growth'].append(residual_growth.mean().item())
                layer_stats[layer_idx]['attn_ffn_cos'].append(attn_ffn_cos.mean().item())

                # POS-specific (sample)
                for b in range(min(len(token_strs), 4)):
                    for s in range(min(S, 32)):
                        token = token_strs[b][s]
                        pos_tag = simple_pos_tag(token)

                        pos_stats[pos_tag]['attn_norm'].append(attn_norm[b, s].item())
                        pos_stats[pos_tag]['ffn_norm'].append(ffn_norm[b, s].item())
                        pos_stats[pos_tag]['attn_contribution'].append(attn_contribution[b, s].item())

                # Neuron-specific FFN patterns (sample)
                for b in range(min(2, B)):
                    for s in range(min(16, S)):
                        neurons = process_indices[b, s].tolist()
                        ffn_val = ffn_norm[b, s].item()
                        for n in neurons:
                            neuron_ffn_norms[n].append(ffn_val)

        # Compile results
        results = {'layers': {}, 'pos': {}, 'neurons': {}}

        print("\n  Layer-wise Attention vs FFN contribution:")
        for layer_idx in range(self.n_layers):
            stats = layer_stats[layer_idx]
            avg_attn = np.mean(stats['attn_norm'])
            avg_ffn = np.mean(stats['ffn_norm'])
            avg_contrib = np.mean(stats['attn_contribution'])
            avg_cos = np.mean(stats['attn_ffn_cos'])
            avg_growth = np.mean(stats['residual_growth'])

            results['layers'][f'layer_{layer_idx}'] = {
                'avg_attn_norm': avg_attn,
                'avg_ffn_norm': avg_ffn,
                'attn_contribution': avg_contrib,
                'attn_ffn_cosine': avg_cos,
                'residual_growth': avg_growth
            }

            print(f"    Layer {layer_idx}: attn={avg_attn:.3f}, ffn={avg_ffn:.3f}, "
                  f"attn_ratio={avg_contrib:.1%}, cos={avg_cos:.3f}")

        print("\n  POS-specific patterns:")
        for pos_tag in sorted(pos_stats.keys()):
            stats = pos_stats[pos_tag]
            if len(stats['attn_norm']) < 10:
                continue

            avg_attn = np.mean(stats['attn_norm'])
            avg_ffn = np.mean(stats['ffn_norm'])
            avg_contrib = np.mean(stats['attn_contribution'])

            results['pos'][pos_tag] = {
                'avg_attn_norm': avg_attn,
                'avg_ffn_norm': avg_ffn,
                'attn_contribution': avg_contrib,
                'count': len(stats['attn_norm'])
            }

            print(f"    {pos_tag:6s}: attn={avg_attn:.3f}, ffn={avg_ffn:.3f}, ratio={avg_contrib:.1%}")

        # Neuron-FFN correlation
        print("\n  Neuron → FFN norm correlation:")
        neuron_ffn_avg = []
        for n in range(self.n_process):
            if neuron_ffn_norms[n]:
                avg = np.mean(neuron_ffn_norms[n])
                neuron_ffn_avg.append((n, avg, len(neuron_ffn_norms[n])))
                results['neurons'][n] = {
                    'avg_ffn_norm': avg,
                    'count': len(neuron_ffn_norms[n])
                }

        neuron_ffn_avg.sort(key=lambda x: -x[1])
        print("    Neurons associated with highest FFN output:")
        for n, avg, count in neuron_ffn_avg[:5]:
            print(f"      Neuron {n}: avg_ffn_norm={avg:.3f} (n={count})")

        return results, layer_stats, pos_stats, neuron_ffn_norms

    def visualize_attention_ffn(self, layer_stats, pos_stats, output_dir):
        """Visualize Attention-FFN interaction"""
        print("\n  Creating Attention-FFN visualizations...")

        if not HAS_MATPLOTLIB:
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. Layer-wise contribution
        print("    Creating layer contribution plot...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        layers = list(range(self.n_layers))
        attn_norms = [np.mean(layer_stats[l]['attn_norm']) for l in layers]
        ffn_norms = [np.mean(layer_stats[l]['ffn_norm']) for l in layers]
        attn_contribs = [np.mean(layer_stats[l]['attn_contribution']) for l in layers]
        attn_ffn_cos = [np.mean(layer_stats[l]['attn_ffn_cos']) for l in layers]

        # Stacked bar chart
        x = np.arange(len(layers))
        width = 0.35
        axes[0, 0].bar(x - width/2, attn_norms, width, label='Attention', color='steelblue')
        axes[0, 0].bar(x + width/2, ffn_norms, width, label='FFN', color='coral')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Output Norm')
        axes[0, 0].set_title('Attention vs FFN Output Norm')
        axes[0, 0].legend()
        axes[0, 0].set_xticks(x)

        # Contribution ratio
        axes[0, 1].bar(layers, attn_contribs, color='steelblue')
        axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Attention Contribution Ratio')
        axes[0, 1].set_title('Attention Contribution (vs FFN)')
        axes[0, 1].set_ylim(0, 1)

        # Cosine similarity
        axes[1, 0].bar(layers, attn_ffn_cos, color='seagreen')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].set_title('Attention-FFN Output Similarity')

        # POS-specific contribution
        pos_tags = sorted([p for p in pos_stats.keys() if len(pos_stats[p]['attn_norm']) >= 10])
        pos_contribs = [np.mean(pos_stats[p]['attn_contribution']) for p in pos_tags]

        axes[1, 1].barh(pos_tags, pos_contribs, color='purple')
        axes[1, 1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Attention Contribution Ratio')
        axes[1, 1].set_ylabel('POS Tag')
        axes[1, 1].set_title('Attention Contribution by POS')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attention_ffn_correlation.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: attention_ffn_correlation.png")

        # 2. Residual contribution pie chart
        print("    Creating residual contribution plot...")

        fig, axes = plt.subplots(1, self.n_layers, figsize=(4 * self.n_layers, 4))
        if self.n_layers == 1:
            axes = [axes]

        for layer_idx, ax in enumerate(axes):
            attn = np.mean(layer_stats[layer_idx]['attn_norm'])
            ffn = np.mean(layer_stats[layer_idx]['ffn_norm'])

            ax.pie([attn, ffn], labels=['Attention', 'FFN'], autopct='%1.1f%%',
                   colors=['steelblue', 'coral'])
            ax.set_title(f'Layer {layer_idx}')

        plt.suptitle('Attention vs FFN Contribution by Layer')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'residual_contribution.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: residual_contribution.png")

    # ============================================================
    # Export
    # ============================================================

    def export_results(self, all_results, output_dir):
        """Export all results to files"""
        print("\n" + "=" * 60)
        print("EXPORTING RESULTS")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)

        # 1. Neuron flow patterns CSV
        if 'flow' in all_results:
            csv_path = os.path.join(output_dir, 'neuron_flow_patterns.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['token', 'count', 'L0_neurons', 'L1_neurons', 'L2_neurons', 'L3_neurons', 'n_unique_paths'])

                for token, data in all_results['flow']['tokens'].items():
                    layer_neurons = data.get('layer_top_neurons', {})
                    row = [
                        token,
                        data['count'],
                        str(layer_neurons.get('layer_0', [])),
                        str(layer_neurons.get('layer_1', [])),
                        str(layer_neurons.get('layer_2', [])),
                        str(layer_neurons.get('layer_3', [])),
                        data['n_unique_paths']
                    ]
                    writer.writerow(row)
            print(f"  Saved: {csv_path}")

        # 2. Process neuron combinations CSV
        if 'householder' in all_results:
            csv_path = os.path.join(output_dir, 'process_neuron_combinations.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['combination', 'count'])

                for combo, data in all_results['householder']['combinations'].items():
                    writer.writerow([combo, data['count']])
            print(f"  Saved: {csv_path}")

        # 3. Attention-FFN by POS CSV
        if 'attention_ffn' in all_results:
            csv_path = os.path.join(output_dir, 'attention_ffn_by_pos.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['pos_tag', 'avg_attn_norm', 'avg_ffn_norm', 'attn_contribution', 'count'])

                for pos, data in all_results['attention_ffn']['pos'].items():
                    writer.writerow([pos, data['avg_attn_norm'], data['avg_ffn_norm'],
                                    data['attn_contribution'], data['count']])
            print(f"  Saved: {csv_path}")

        # 4. Full JSON
        def convert_to_serializable(obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(i) for i in obj]
            return obj

        json_path = os.path.join(output_dir, 'deep_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(convert_to_serializable(all_results), f, indent=2)
        print(f"  Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='DAWN v7.9 Deep Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--val_data', type=str,
                        default='/content/drive/MyDrive/data/validation/wikitext_5to1_texts.pkl',
                        help='Path to validation data')
    parser.add_argument('--max_batches', type=int, default=50,
                        help='Max batches for analysis')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./analysis_v79_deep',
                        help='Output directory')
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

    # Load model (version-agnostic)
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model, version, config = load_model(checkpoint_path, device)

    print(f"Model: DAWN v{model.__version__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load data
    print(f"\nLoading validation data from: {args.val_data}")
    import pickle
    with open(args.val_data, 'rb') as f:
        val_texts = pickle.load(f)
    print(f"Loaded {len(val_texts)} validation texts")

    # Create dataloader
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
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # ============================================================
    # Run Analysis
    # ============================================================

    analyzer = DeepAnalyzer(model, tokenizer, device, version=version)
    all_results = {}

    # Analysis 1: Neuron Flow
    flow_results, flow_counts, token_layer_neurons = analyzer.analyze_neuron_flow(
        dataloader, max_batches=args.max_batches
    )
    all_results['flow'] = flow_results
    analyzer.visualize_neuron_flow(flow_counts, token_layer_neurons, args.output_dir)

    # Analysis 2: Householder Effect
    householder_results, combo_counts, neuron_effects = analyzer.analyze_householder_effect(
        dataloader, max_batches=args.max_batches
    )
    all_results['householder'] = householder_results
    analyzer.visualize_householder(combo_counts, neuron_effects, args.output_dir)

    # Analysis 3: Attention-FFN Interaction
    attn_ffn_results, layer_stats, pos_stats, _ = analyzer.analyze_attention_ffn(
        dataloader, max_batches=args.max_batches
    )
    all_results['attention_ffn'] = attn_ffn_results
    analyzer.visualize_attention_ffn(layer_stats, pos_stats, args.output_dir)

    # Export all results
    analyzer.export_results(all_results, args.output_dir)

    print("\n" + "=" * 60)
    print("DEEP ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
