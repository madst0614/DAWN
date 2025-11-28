"""
DAWN v8.x Deep Analysis Script
심층 분석: 뉴런 흐름, SharedNeurons 효과, Attention-Memory 상호작용

분석 항목:
1. 레이어간 뉴런 흐름 추적 (Sankey diagram)
2. SharedNeurons Householder 변환 효과 분석
3. Attention-Memory 상호작용 분석
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

from models import create_model_by_version

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
    print("Warning: plotly not available, Sankey diagrams disabled")


def get_underlying_model(model):
    """Get the underlying model from torch.compile wrapper"""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def simple_pos_tag(token):
    """Simple POS tagging"""
    token_lower = token.lower().strip()
    if not token_lower or token_lower.startswith('[') or token_lower.startswith('##'):
        return 'OTHER'
    if token_lower in {'the', 'a', 'an', 'this', 'that', 'these', 'those'}:
        return 'DET'
    if token_lower in {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'}:
        return 'AUX'
    if token_lower in {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}:
        return 'PRON'
    if token_lower in {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about'}:
        return 'ADP'
    if token_lower in {'and', 'or', 'but', 'if', 'when', 'while', 'because'}:
        return 'CONJ'
    if token_lower.isdigit():
        return 'NUM'
    if token_lower.endswith('ing') or token_lower.endswith('ed'):
        return 'VERB'
    if token_lower.endswith('ly'):
        return 'ADV'
    return 'NOUN'


class DeepAnalyzerV8:
    """Deep analysis for DAWN v8.0"""

    def __init__(self, model, tokenizer, device):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device

        self.n_layers = len(self.model.layers)
        self.n_process = self.model.n_process
        self.n_knowledge = self.model.n_knowledge
        self.process_k = self.model.process_k
        self.knowledge_k = self.model.knowledge_k
        self.vocab_size = self.tokenizer.vocab_size

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

        if target_tokens is None:
            target_tokens = ['the', 'is', 'in', 'and', 'to', 'of', 'a', 'that', 'it', 'for',
                             'on', 'with', 'as', 'at', 'by', 'from', 'be', 'was', 'are', 'have']

        target_ids = {self.tokenizer.encode(t, add_special_tokens=False)[0]: t
                      for t in target_tokens if len(self.tokenizer.encode(t, add_special_tokens=False)) > 0}

        # Flow tracking: {token: {(layer0_combo, layer1_combo, ...): count}}
        flow_counts = defaultdict(lambda: Counter())
        token_layer_neurons = defaultdict(lambda: defaultdict(lambda: Counter()))
        token_counts = Counter()

        # Also track Q/K/V/O/M separately (M for v8.3+)
        has_memory_routing = hasattr(self.model.layers[0].memory, 'query_compressor')
        components = ['Q', 'K', 'V', 'O']
        if has_memory_routing:
            components.append('M')
        component_flows = {comp: defaultdict(lambda: Counter()) for comp in components}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Neuron Flow", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # Find target tokens
            for tid, token in target_ids.items():
                mask = (input_ids == tid)
                if not mask.any():
                    continue

                positions = torch.where(mask)
                token_counts[token] += len(positions[0])

                for pos in zip(*positions):
                    b, s = pos[0].item(), pos[1].item()

                    # Collect process indices across layers (Q routing)
                    layer_path = []
                    for layer_idx, routing_info in enumerate(routing_infos):
                        attn_routing = routing_info['attention']
                        neurons = attn_routing['neuron_indices'][b, s].tolist()  # Q routing
                        neurons_tuple = tuple(sorted(neurons))
                        layer_path.append(neurons_tuple)

                        for n in neurons:
                            token_layer_neurons[token][f'layer_{layer_idx}'][n] += 1

                    flow_counts[token][tuple(layer_path)] += 1

                    # Track Q/K/V/O/M separately
                    for comp in components:
                        comp_path = []
                        for layer_idx, routing_info in enumerate(routing_infos):
                            attn_r = routing_info['attention']
                            mem_r = routing_info.get('memory', {})

                            if comp == 'O':
                                r = attn_r['routing_O']
                            elif comp == 'M':
                                # v8.3: Memory Query routing
                                r = mem_r.get('query_routing', None)
                                if r is None:
                                    break
                            else:
                                r = attn_r[f'routing_{comp}']
                            neurons = r['process_indices'][b, s].tolist()
                            comp_path.append(tuple(sorted(neurons)))

                        if comp_path:  # Only add if we got routing for all layers
                            component_flows[comp][token][tuple(comp_path)] += 1

        # Build results
        results = {'tokens': {}, 'summary': {}}

        print("\n📌 Token Flow Patterns (Q routing):")
        for token in sorted(flow_counts.keys(), key=lambda t: token_counts[t], reverse=True)[:10]:
            count = token_counts[token]
            flows = flow_counts[token]

            top_flows = flows.most_common(3)
            top_path, top_count = top_flows[0] if top_flows else (None, 0)

            results['tokens'][token] = {
                'total_count': count,
                'unique_paths': len(flows),
                'top_path': [list(p) for p in top_path] if top_path else None,
                'top_path_count': top_count,
                'concentration': top_count / count if count > 0 else 0
            }

            print(f"\n  '{token}' (n={count}):")
            print(f"    Unique flow paths: {len(flows)}")
            print(f"    Most common path ({top_count}x): {top_path}")
            print(f"    Path concentration: {top_count/count:.1%}")

        return results, flow_counts, token_layer_neurons

    def visualize_neuron_flow(self, flow_counts, token_layer_neurons, output_dir):
        """Create Sankey diagram for neuron flow"""
        print("\n  Creating neuron flow visualizations...")

        if not HAS_PLOTLY:
            print("    plotly not available, skipping Sankey diagram")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Select token with most data
        top_token = max(flow_counts.keys(), key=lambda t: sum(flow_counts[t].values()))

        # Build Sankey data
        # Nodes: layer_0_neuron_0, layer_0_neuron_1, ..., layer_1_neuron_0, ...
        node_labels = []
        node_map = {}

        for layer_idx in range(self.n_layers):
            for neuron_idx in range(self.n_process):
                label = f"L{layer_idx}_N{neuron_idx}"
                node_map[(layer_idx, neuron_idx)] = len(node_labels)
                node_labels.append(label)

        # Links: transitions between layers
        links = defaultdict(int)

        for path, count in flow_counts[top_token].items():
            for layer_idx in range(len(path) - 1):
                src_neurons = path[layer_idx]
                dst_neurons = path[layer_idx + 1]

                for src in src_neurons:
                    for dst in dst_neurons:
                        src_idx = node_map[(layer_idx, src)]
                        dst_idx = node_map[(layer_idx + 1, dst)]
                        links[(src_idx, dst_idx)] += count

        # Filter weak links
        min_count = max(1, sum(links.values()) * 0.01)
        filtered_links = {k: v for k, v in links.items() if v >= min_count}

        if filtered_links:
            sources = [k[0] for k in filtered_links.keys()]
            targets = [k[1] for k in filtered_links.keys()]
            values = list(filtered_links.values())

            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                )
            )])

            fig.update_layout(
                title_text=f"Neuron Flow for '{top_token}'",
                font_size=10
            )

            fig.write_html(os.path.join(output_dir, f'neuron_flow_{top_token}.html'))
            print(f"    Saved: neuron_flow_{top_token}.html")

        # Also create heatmap of layer transitions
        if HAS_MATPLOTLIB:
            print("    Creating layer transition heatmap...")

            # Aggregate all tokens
            transition_matrix = torch.zeros(self.n_layers - 1, self.n_process, self.n_process)

            for token, flows in flow_counts.items():
                for path, count in flows.items():
                    for layer_idx in range(len(path) - 1):
                        for src in path[layer_idx]:
                            for dst in path[layer_idx + 1]:
                                transition_matrix[layer_idx, src, dst] += count

            # Normalize
            transition_matrix = transition_matrix / (transition_matrix.sum(dim=(1, 2), keepdim=True) + 1e-10)

            fig, axes = plt.subplots(1, self.n_layers - 1, figsize=(5 * (self.n_layers - 1), 5))
            if self.n_layers == 2:
                axes = [axes]

            for layer_idx in range(self.n_layers - 1):
                im = axes[layer_idx].imshow(transition_matrix[layer_idx].numpy(), cmap='YlOrRd')
                axes[layer_idx].set_xlabel(f'Layer {layer_idx + 1} Neuron')
                axes[layer_idx].set_ylabel(f'Layer {layer_idx} Neuron')
                axes[layer_idx].set_title(f'L{layer_idx} → L{layer_idx + 1} Transition')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'layer_transitions.png'), dpi=150)
            if IN_NOTEBOOK:
                plt.show()
            else:
                plt.close()
            print(f"    Saved: layer_transitions.png")

    # ============================================================
    # Analysis 2: SharedNeurons Householder Effect
    # ============================================================

    @torch.no_grad()
    def analyze_householder_effect(self, dataloader, max_batches=30):
        """Analyze how SharedNeurons Householder transformations change representations"""
        print("\n" + "=" * 60)
        print("ANALYSIS 2: SHAREDNEURONS HOUSEHOLDER EFFECT")
        print("=" * 60)

        self.model.eval()
        shared = self.model.shared_neurons

        # Get process neurons (v8.x version aware)
        # For Q compressor analysis, use QK pool if available
        if hasattr(shared, 'process_neurons_qk'):
            process_neurons = shared.process_neurons_qk.data  # v8.1+
            pool_version = 'QK (v8.1+)'
        else:
            process_neurons = shared.process_neurons.data  # v8.0
            pool_version = 'single (v8.0)'

        # Analyze process neuron properties
        print(f"\n📌 Process Neurons (Householder Vectors) - {pool_version}:")

        # Norms (should be ~1)
        norms = process_neurons.norm(dim=-1)
        print(f"  Norms: mean={norms.mean().item():.4f}, std={norms.std().item():.4f}")

        # Cosine similarity
        v_norm = F.normalize(process_neurons, dim=-1)
        cos_sim = v_norm @ v_norm.T
        n_neurons = process_neurons.shape[0]
        mask = ~torch.eye(n_neurons, dtype=torch.bool, device=cos_sim.device)
        off_diag = cos_sim[mask]
        print(f"  Cosine similarity: mean={off_diag.abs().mean().item():.4f}, max={off_diag.abs().max().item():.4f}")

        # Track combination effects
        combination_counts = Counter()
        combination_effects = defaultdict(list)  # {combo: [(before_norm, after_norm, cos_sim), ...]}
        n_process_neurons = process_neurons.shape[0]
        neuron_effects = {n: {'cos_sim': [], 'norm_change': []} for n in range(n_process_neurons)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Householder Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Forward through embedding
            pos = torch.arange(S, device=self.device).unsqueeze(0)
            x = self.model.token_emb(input_ids) + self.model.pos_emb(pos)
            x = self.model.dropout(x)

            # Get routing info
            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # Analyze first layer Q compressor
            layer0 = self.model.layers[0]
            x_norm = layer0.norm1(x)

            # Get Q compression details
            compressor = layer0.attention.compressor_Q

            # Input projection
            input_scores = compressor.input_router(x_norm)
            input_weights = F.softmax(input_scores, dim=-1)
            all_proj = torch.einsum('bsd,ndr->bsnr', x_norm, shared.input_neurons)
            x_compressed = (all_proj * input_weights.unsqueeze(-1)).sum(dim=2)  # [B, S, rank]

            # Process routing
            process_scores = compressor.process_router(x_compressed)
            _, process_indices = torch.topk(process_scores, self.process_k, dim=-1)  # [B, S, k]

            # Track effects for sample positions
            for b in range(min(B, 4)):
                for s in range(min(S, 32)):
                    combo = tuple(sorted(process_indices[b, s].tolist()))
                    combination_counts[combo] += 1

                    # Before Householder
                    x_before = x_compressed[b, s].clone()  # [rank]
                    before_norm = x_before.norm().item()

                    # Apply Householder transforms
                    x_after = x_before.clone()
                    for i in range(self.process_k):
                        v = process_neurons[process_indices[b, s, i]]
                        x_after = shared.apply_householder(x_after.unsqueeze(0).unsqueeze(0), v.unsqueeze(0).unsqueeze(0))
                        x_after = x_after.squeeze()

                    after_norm = x_after.norm().item()

                    # Cosine similarity before/after
                    cos = F.cosine_similarity(x_before.unsqueeze(0), x_after.unsqueeze(0)).item()

                    combination_effects[combo].append((before_norm, after_norm, cos))

                    for n in process_indices[b, s].tolist():
                        neuron_effects[n]['cos_sim'].append(cos)
                        neuron_effects[n]['norm_change'].append(after_norm / (before_norm + 1e-10))

        # Build results
        results = {'combinations': {}, 'neurons': {}, 'summary': {}}

        print("\n📌 Top 10 Process Neuron Combinations:")
        for i, (combo, count) in enumerate(combination_counts.most_common(10)):
            effects = combination_effects[combo]
            if effects:
                avg_cos = np.mean([e[2] for e in effects])
                avg_norm_ratio = np.mean([e[1] / (e[0] + 1e-10) for e in effects])

                results['combinations'][str(combo)] = {
                    'count': count,
                    'avg_cos_sim': avg_cos,
                    'avg_norm_ratio': avg_norm_ratio
                }

                print(f"  {i+1}. {combo}: {count}x, cos_sim={avg_cos:.4f}, norm_ratio={avg_norm_ratio:.4f}")

        print("\n📌 Per-Neuron Effects:")
        for n in range(min(self.n_process, 16)):
            if neuron_effects[n]['cos_sim']:
                avg_cos = np.mean(neuron_effects[n]['cos_sim'])
                avg_norm = np.mean(neuron_effects[n]['norm_change'])

                results['neurons'][n] = {
                    'avg_cos_sim': avg_cos,
                    'avg_norm_change': avg_norm,
                    'usage_count': len(neuron_effects[n]['cos_sim'])
                }

                print(f"  Neuron {n:2d}: cos_sim={avg_cos:.4f}, norm_ratio={avg_norm:.4f}")

        # Summary
        all_cos = [e[2] for effects in combination_effects.values() for e in effects]
        all_norm = [e[1] / (e[0] + 1e-10) for effects in combination_effects.values() for e in effects]

        results['summary'] = {
            'avg_cos_sim': np.mean(all_cos) if all_cos else 0,
            'avg_norm_ratio': np.mean(all_norm) if all_norm else 0,
            'total_combinations': len(combination_counts),
            'unique_combinations': len([c for c, cnt in combination_counts.items() if cnt > 1])
        }

        print(f"\n📌 Summary:")
        print(f"  Total combinations seen: {len(combination_counts)}")
        print(f"  Average cos similarity (before/after): {results['summary']['avg_cos_sim']:.4f}")
        print(f"  Average norm ratio: {results['summary']['avg_norm_ratio']:.4f}")

        return results, combination_counts, neuron_effects

    def visualize_householder(self, combination_counts, neuron_effects, output_dir):
        """Visualize Householder transformation effects"""
        print("\n  Creating Householder visualizations...")

        if not HAS_MATPLOTLIB:
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. Process neuron vectors heatmap (v8.x version aware)
        print("    Creating process neuron heatmap...")
        shared = self.model.shared_neurons
        if hasattr(shared, 'process_neurons_qk'):
            process_neurons = shared.process_neurons_qk.data.cpu().numpy()
            title_suffix = ' (QK pool)'
        else:
            process_neurons = shared.process_neurons.data.cpu().numpy()
            title_suffix = ''

        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(process_neurons, aspect='auto', cmap='RdBu', vmin=-0.3, vmax=0.3)
        ax.set_xlabel('Rank Dimension')
        ax.set_ylabel('Process Neuron')
        ax.set_title(f'SharedNeurons Process Vectors (Householder){title_suffix}')
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'process_neurons_heatmap.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: process_neurons_heatmap.png")

        # 2. Combination frequency
        print("    Creating combination frequency plot...")
        top_combos = combination_counts.most_common(20)
        combo_labels = [str(c[0]) for c in top_combos]
        combo_counts = [c[1] for c in top_combos]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(range(len(combo_labels)), combo_counts, color='steelblue')
        ax.set_yticks(range(len(combo_labels)))
        ax.set_yticklabels(combo_labels, fontsize=8)
        ax.set_xlabel('Count')
        ax.set_title('Top 20 Process Neuron Combinations')
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'process_combinations.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: process_combinations.png")

        # 3. Neuron cosine similarity matrix (v8.x version aware)
        print("    Creating neuron similarity matrix...")
        if hasattr(shared, 'process_neurons_qk'):
            process_neurons_tensor = shared.process_neurons_qk.data
            sim_title = 'Process Neuron Cosine Similarity (QK pool)'
        else:
            process_neurons_tensor = shared.process_neurons.data
            sim_title = 'Process Neuron Cosine Similarity'
        v_norm = F.normalize(process_neurons_tensor, dim=-1)
        cos_sim = (v_norm @ v_norm.T).cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cos_sim, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_xlabel('Process Neuron')
        ax.set_ylabel('Process Neuron')
        ax.set_title(sim_title)
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'neuron_similarity.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: neuron_similarity.png")

    # ============================================================
    # Analysis 3: Attention-Memory Interaction
    # ============================================================

    @torch.no_grad()
    def analyze_attention_memory(self, dataloader, max_batches=30):
        """Analyze interaction between NeuronAttention and NeuronMemory"""
        print("\n" + "=" * 60)
        print("ANALYSIS 3: ATTENTION-MEMORY INTERACTION")
        print("=" * 60)

        self.model.eval()

        # Track contributions
        layer_stats = {l: {
            'attn_norm': [],
            'mem_norm': [],
            'attn_contribution': [],  # |attn| / (|attn| + |mem|)
            'residual_growth': [],
            'attn_mem_cos': [],  # cosine sim between attn and mem outputs
        } for l in range(self.n_layers)}

        # POS-specific tracking
        pos_stats = defaultdict(lambda: {
            'attn_norm': [],
            'mem_norm': [],
            'attn_contribution': []
        })

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Attn-Mem Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Token POS tags
            token_pos = {}
            for b in range(B):
                for s in range(S):
                    tid = input_ids[b, s].item()
                    token = self.tokenizer.decode([tid]).strip()
                    token_pos[(b, s)] = simple_pos_tag(token)

            # Manual forward
            pos = torch.arange(S, device=self.device).unsqueeze(0)
            x = self.model.token_emb(input_ids) + self.model.pos_emb(pos)
            x = self.model.dropout(x)

            mask = self.model.causal_mask[:, :, :S, :S]

            for layer_idx, layer in enumerate(self.model.layers):
                residual = x
                x_norm = layer.norm1(x)

                # Attention output
                attn_out, _ = layer.attention(x_norm, mask)
                x = residual + layer.dropout(attn_out)

                attn_norm = attn_out.norm(dim=-1)  # [B, S]

                # Memory output
                residual = x
                x_norm = layer.norm2(x)
                mem_out, _ = layer.memory(x_norm)
                x = residual + layer.dropout(mem_out)

                mem_norm = mem_out.norm(dim=-1)  # [B, S]

                # Statistics
                attn_contribution = attn_norm / (attn_norm + mem_norm + 1e-10)

                # Cosine similarity between attn and mem
                attn_flat = attn_out.reshape(-1, attn_out.shape[-1])
                mem_flat = mem_out.reshape(-1, mem_out.shape[-1])
                cos_sim = F.cosine_similarity(attn_flat, mem_flat, dim=-1)

                layer_stats[layer_idx]['attn_norm'].append(attn_norm.mean().item())
                layer_stats[layer_idx]['mem_norm'].append(mem_norm.mean().item())
                layer_stats[layer_idx]['attn_contribution'].append(attn_contribution.mean().item())
                layer_stats[layer_idx]['attn_mem_cos'].append(cos_sim.mean().item())

                # POS-specific
                for b in range(min(B, 4)):
                    for s in range(S):
                        pos_tag = token_pos[(b, s)]
                        pos_stats[pos_tag]['attn_norm'].append(attn_norm[b, s].item())
                        pos_stats[pos_tag]['mem_norm'].append(mem_norm[b, s].item())
                        pos_stats[pos_tag]['attn_contribution'].append(attn_contribution[b, s].item())

        # Build results
        results = {'layers': {}, 'pos': {}}

        print("\n📌 Per-Layer Attention-Memory Statistics:")
        for layer_idx in range(self.n_layers):
            stats = layer_stats[layer_idx]

            avg_attn = np.mean(stats['attn_norm'])
            avg_mem = np.mean(stats['mem_norm'])
            avg_contrib = np.mean(stats['attn_contribution'])
            avg_cos = np.mean(stats['attn_mem_cos'])

            results['layers'][f'layer_{layer_idx}'] = {
                'avg_attn_norm': avg_attn,
                'avg_mem_norm': avg_mem,
                'attn_contribution': avg_contrib,
                'attn_mem_cos': avg_cos
            }

            print(f"  Layer {layer_idx}: attn={avg_attn:.3f}, mem={avg_mem:.3f}, "
                  f"attn_ratio={avg_contrib:.2%}, cos={avg_cos:.3f}")

        print("\n📌 POS-Specific Attention-Memory Balance:")
        for pos_tag in sorted(pos_stats.keys()):
            if len(pos_stats[pos_tag]['attn_contribution']) < 100:
                continue

            stats = pos_stats[pos_tag]
            avg_contrib = np.mean(stats['attn_contribution'])
            avg_attn = np.mean(stats['attn_norm'])
            avg_mem = np.mean(stats['mem_norm'])

            results['pos'][pos_tag] = {
                'attn_contribution': avg_contrib,
                'avg_attn_norm': avg_attn,
                'avg_mem_norm': avg_mem,
                'count': len(stats['attn_contribution'])
            }

            print(f"  {pos_tag:6s}: attn_ratio={avg_contrib:.2%}, attn={avg_attn:.3f}, mem={avg_mem:.3f}")

        return results, layer_stats, pos_stats

    def visualize_attention_memory(self, layer_stats, pos_stats, output_dir):
        """Visualize Attention-Memory interaction"""
        print("\n  Creating Attention-Memory visualizations...")

        if not HAS_MATPLOTLIB:
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. Layer-wise contribution
        print("    Creating layer contribution plot...")
        layers = list(range(self.n_layers))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Norms
        attn_norms = [np.mean(layer_stats[l]['attn_norm']) for l in layers]
        mem_norms = [np.mean(layer_stats[l]['mem_norm']) for l in layers]

        x = np.arange(len(layers))
        width = 0.35

        axes[0].bar(x - width/2, attn_norms, width, label='Attention', color='steelblue')
        axes[0].bar(x + width/2, mem_norms, width, label='Memory', color='coral')
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Output Norm')
        axes[0].set_title('Attention vs Memory Output Norms')
        axes[0].legend()
        axes[0].set_xticks(x)

        # Contribution ratio
        contribs = [np.mean(layer_stats[l]['attn_contribution']) for l in layers]
        axes[1].bar(layers, contribs, color='seagreen')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Attention Contribution')
        axes[1].set_title('Attention / (Attention + Memory) Ratio')
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attn_mem_layers.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: attn_mem_layers.png")

        # 2. POS-specific
        print("    Creating POS contribution plot...")
        pos_tags = [p for p in sorted(pos_stats.keys()) if len(pos_stats[p]['attn_contribution']) >= 100]
        pos_contribs = [np.mean(pos_stats[p]['attn_contribution']) for p in pos_tags]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(pos_tags, pos_contribs, color='steelblue')
        ax.set_xlabel('Attention Contribution')
        ax.set_title('Attention Contribution by POS Tag')
        ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attn_mem_pos.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: attn_mem_pos.png")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN v8.x Deep Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--val_data', type=str,
                        default='/content/drive/MyDrive/data/validation/wikitext_5to1_texts.pkl',
                        help='Path to validation data')
    parser.add_argument('--max_batches', type=int, default=50,
                        help='Max batches for analysis')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./analysis_v8_deep',
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

    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('model_config', checkpoint.get('config', {}))
    model_version = config.get('model_version', '8.0')
    print(f"Checkpoint model version: {model_version}")

    model_kwargs = {
        'vocab_size': config.get('vocab_size', 30522),
        'd_model': config.get('d_model', 256),
        'n_layers': config.get('n_layers', 4),
        'n_heads': config.get('n_heads', 4),
        'rank': config.get('rank', config.get('basis_rank', 64)),
        'max_seq_len': config.get('max_seq_len', 128),
        'n_input': config.get('n_input', 8),
        'n_process': config.get('n_process', 32),
        'n_output': config.get('n_output', 8),
        'process_k': config.get('process_k', 3),
        'n_knowledge': config.get('n_knowledge', 64),
        'knowledge_k': config.get('knowledge_k', 8),
        'dropout': config.get('dropout', 0.1),
    }
    model = create_model_by_version(model_version, model_kwargs)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  Removing torch.compile wrapper prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Handle checkpoint conversion for version compatibility
    if any('process_neurons_vo' in k for k in state_dict.keys()):
        print("  Converting v8.1 checkpoint (process_neurons_vo → v + o + m)...")
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'process_neurons_vo' in k:
                new_state_dict[k.replace('process_neurons_vo', 'process_neurons_v')] = v.clone()
                new_state_dict[k.replace('process_neurons_vo', 'process_neurons_o')] = v.clone()
                new_state_dict[k.replace('process_neurons_vo', 'process_neurons_m')] = v.clone()
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    elif any('process_neurons_o' in k for k in state_dict.keys()) and not any('process_neurons_m' in k for k in state_dict.keys()):
        print("  Converting v8.2 checkpoint (adding process_neurons_m)...")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k] = v
            if 'process_neurons_o' in k:
                new_state_dict[k.replace('process_neurons_o', 'process_neurons_m')] = v.clone()
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model: DAWN v{model.__version__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load data
    print(f"\nLoading validation data from: {args.val_data}")
    import pickle
    with open(args.val_data, 'rb') as f:
        val_texts = pickle.load(f)
    print(f"Loaded {len(val_texts)} validation texts")

    # Dataset
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

    analyzer = DeepAnalyzerV8(model, tokenizer, device)
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

    # Analysis 3: Attention-Memory Interaction
    attn_mem_results, layer_stats, pos_stats = analyzer.analyze_attention_memory(
        dataloader, max_batches=args.max_batches
    )
    all_results['attention_memory'] = attn_mem_results
    analyzer.visualize_attention_memory(layer_stats, pos_stats, args.output_dir)

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
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    results_path = os.path.join(args.output_dir, 'deep_results.json')
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("DEEP ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
