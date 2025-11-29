"""
DAWN v8 Comprehensive Analysis Script (Colab-Ready)
====================================================

종합 분석 항목:
1. Neuron Usage Patterns - Effective rank, Gini coefficient
2. Q/K Diversity - Jaccard similarity, routing overlap
3. Knowledge Neurons - Usage distribution, dead neurons
4. Layer-wise Patterns - Where are bottlenecks?
5. Attention vs Memory Contribution - Which dominates?
6. Householder Transform Effect - How much does it change?
7. Token/POS Preferences - What do neurons specialize in?
8. Information Flow - Norm changes through layers
9. SharedNeurons Health - Orthogonality, condition numbers
10. Automatic Recommendations - What to fix next

Usage (Colab):
    %run scripts/analyze_v8_comprehensive.py --checkpoint <path> --val_data <path>

Or import and run:
    from scripts.analyze_v8_comprehensive import ComprehensiveAnalyzer
    analyzer = ComprehensiveAnalyzer(model, tokenizer, device)
    results = analyzer.run_all(dataloader)
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

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def get_underlying_model(model):
    """Get the underlying model from torch.compile wrapper"""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def simple_pos_tag(token):
    """Simple POS tagging based on token patterns"""
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


class ComprehensiveAnalyzer:
    """Comprehensive analyzer for DAWN v8.0"""

    def __init__(self, model, tokenizer, device):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device

        self.n_layers = len(self.model.layers)
        self.n_process = self.model.n_process
        self.n_knowledge = self.model.n_knowledge
        self.n_input = self.model.n_input
        self.n_output = self.model.n_output
        self.process_k = self.model.process_k
        self.knowledge_k = self.model.knowledge_k
        self.vocab_size = self.tokenizer.vocab_size

        # Check which pools exist
        shared = self.model.shared_neurons
        self.has_qk_split = hasattr(shared, 'process_neurons_qk')
        self.has_vo_split = hasattr(shared, 'process_neurons_v')
        self.has_memory_routing = hasattr(self.model.layers[0].memory, 'query_compressor')

        # Components to analyze
        self.components = ['Q', 'K', 'V', 'O']
        if self.has_memory_routing:
            self.components.append('M')

    # ============================================================
    # 1. Neuron Usage Patterns
    # ============================================================

    @torch.no_grad()
    def analyze_neuron_usage(self, dataloader, max_batches=50):
        """Analyze neuron usage patterns: effective rank and Gini coefficient"""
        print("\n" + "=" * 60)
        print("1. NEURON USAGE PATTERNS")
        print("=" * 60)

        self.model.eval()

        # Track usage per component per layer
        usage = {comp: torch.zeros(self.n_layers, self.n_process, device=self.device)
                 for comp in self.components}
        knowledge_usage = torch.zeros(self.n_layers, self.n_knowledge, device=self.device)

        total_tokens = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Usage Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape
            total_tokens += B * S

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                attn_routing = routing_info['attention']
                mem_routing = routing_info.get('memory', {})

                # Process neurons
                for comp in self.components:
                    if comp == 'O':
                        routing = attn_routing['routing_O']
                    elif comp == 'M':
                        routing = mem_routing.get('query_routing', None)
                        if routing is None:
                            continue
                    else:
                        routing = attn_routing[f'routing_{comp}']

                    idx = routing['process_indices'].reshape(-1)
                    usage[comp][layer_idx] += torch.bincount(idx, minlength=self.n_process).float()

                # Knowledge neurons
                k_idx = mem_routing['knowledge_indices'].reshape(-1)
                knowledge_usage[layer_idx] += torch.bincount(k_idx, minlength=self.n_knowledge).float()

        # Compute statistics
        results = {'components': {}, 'knowledge': {}, 'summary': {}}

        def compute_effective_rank_gini(counts):
            """Compute effective rank and Gini from usage counts"""
            probs = counts / (counts.sum() + 1e-10)

            # Effective rank = exp(entropy)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            eff_rank = torch.exp(entropy).item()

            # Gini coefficient
            sorted_probs, _ = torch.sort(probs)
            n = len(probs)
            index = torch.arange(1, n + 1, dtype=torch.float32, device=probs.device)
            gini = ((2 * index - n - 1) * sorted_probs).sum() / (n * sorted_probs.sum() + 1e-10)

            return eff_rank, gini.item(), probs

        print("\nEffective Rank / Gini by Layer and Component:")
        print(f"{'Comp':<6}", end='')
        for layer_idx in range(self.n_layers):
            print(f"{'L'+str(layer_idx):>12}", end='')
        print()
        print("-" * (6 + 12 * self.n_layers))

        for comp in self.components:
            results['components'][comp] = {'layers': {}}
            print(f"{comp:<6}", end='')

            for layer_idx in range(self.n_layers):
                eff_rank, gini, probs = compute_effective_rank_gini(usage[comp][layer_idx])
                results['components'][comp]['layers'][layer_idx] = {
                    'eff_rank': eff_rank,
                    'gini': gini,
                    'max_neurons': self.n_process
                }
                print(f"{eff_rank:>5.1f}/{self.n_process:<3} G{gini:.2f}", end='')
            print()

        # Knowledge neurons
        print(f"\n{'Know':<6}", end='')
        for layer_idx in range(self.n_layers):
            eff_rank, gini, probs = compute_effective_rank_gini(knowledge_usage[layer_idx])
            results['knowledge'][layer_idx] = {
                'eff_rank': eff_rank,
                'gini': gini,
                'max_neurons': self.n_knowledge
            }
            print(f"{eff_rank:>5.1f}/{self.n_knowledge:<3} G{gini:.2f}", end='')
        print()

        # Summary
        avg_eff_ranks = []
        avg_ginis = []
        for comp in self.components:
            for layer_idx in range(self.n_layers):
                layer_data = results['components'][comp]['layers'][layer_idx]
                avg_eff_ranks.append(layer_data['eff_rank'] / layer_data['max_neurons'])
                avg_ginis.append(layer_data['gini'])

        results['summary'] = {
            'avg_utilization': np.mean(avg_eff_ranks),
            'avg_gini': np.mean(avg_ginis),
            'total_tokens': total_tokens
        }

        return results, usage, knowledge_usage

    # ============================================================
    # 2. Q/K Diversity Analysis
    # ============================================================

    @torch.no_grad()
    def analyze_qk_diversity(self, dataloader, max_batches=50):
        """Analyze Q/K routing diversity using Jaccard similarity"""
        print("\n" + "=" * 60)
        print("2. Q/K DIVERSITY (Router Overlap)")
        print("=" * 60)

        self.model.eval()

        # Per-layer Jaccard accumulator
        qk_jaccards = torch.zeros(self.n_layers, device=self.device)
        qk_counts = torch.zeros(self.n_layers, device=self.device)

        # Also track component pairs
        pair_jaccards = {f'{c1}-{c2}': torch.zeros(self.n_layers, device=self.device)
                        for i, c1 in enumerate(self.components[:4])
                        for c2 in self.components[i+1:4]}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Q/K Diversity", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                attn_routing = routing_info['attention']

                # Get indices
                q_idx = attn_routing['routing_Q']['process_indices']  # [B, S, k]
                k_idx = attn_routing['routing_K']['process_indices']
                v_idx = attn_routing['routing_V']['process_indices']
                o_idx = attn_routing['routing_O']['process_indices']

                indices = {'Q': q_idx, 'K': k_idx, 'V': v_idx, 'O': o_idx}

                # Sample tokens for Jaccard (full computation is expensive)
                sample_size = min(100, B * S)
                for _ in range(sample_size):
                    b = torch.randint(B, (1,)).item()
                    s = torch.randint(S, (1,)).item()

                    q_set = set(q_idx[b, s].tolist())
                    k_set = set(k_idx[b, s].tolist())

                    intersection = len(q_set & k_set)
                    union = len(q_set | k_set)
                    if union > 0:
                        qk_jaccards[layer_idx] += intersection / union
                        qk_counts[layer_idx] += 1

                    # All pairs
                    for pair_name in pair_jaccards:
                        c1, c2 = pair_name.split('-')
                        set1 = set(indices[c1][b, s].tolist())
                        set2 = set(indices[c2][b, s].tolist())
                        inter = len(set1 & set2)
                        uni = len(set1 | set2)
                        if uni > 0:
                            pair_jaccards[pair_name][layer_idx] += inter / uni

        # Normalize
        qk_jaccards /= (qk_counts + 1e-10)
        for pair_name in pair_jaccards:
            pair_jaccards[pair_name] /= (qk_counts + 1e-10)

        results = {
            'qk_jaccard': qk_jaccards.tolist(),
            'pair_jaccards': {k: v.tolist() for k, v in pair_jaccards.items()},
            'avg_qk_jaccard': qk_jaccards.mean().item()
        }

        print(f"\nQ-K Jaccard Similarity by Layer:")
        for layer_idx in range(self.n_layers):
            jaccard = qk_jaccards[layer_idx].item()
            bar = '█' * int(jaccard * 20)
            print(f"  Layer {layer_idx}: {jaccard:.3f} {bar}")

        print(f"\nComponent Pair Jaccard (averaged):")
        for pair_name in sorted(pair_jaccards.keys()):
            avg = pair_jaccards[pair_name].mean().item()
            print(f"  {pair_name}: {avg:.3f}")

        return results

    # ============================================================
    # 3. Knowledge Neurons Analysis
    # ============================================================

    @torch.no_grad()
    def analyze_knowledge_neurons(self, dataloader, max_batches=50):
        """Analyze knowledge neuron usage patterns"""
        print("\n" + "=" * 60)
        print("3. KNOWLEDGE NEURONS ANALYSIS")
        print("=" * 60)

        self.model.eval()

        # Track usage and weights
        knowledge_usage = torch.zeros(self.n_layers, self.n_knowledge, device=self.device)
        knowledge_weights = torch.zeros(self.n_layers, self.n_knowledge, device=self.device)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Knowledge Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                mem_routing = routing_info['memory']
                k_idx = mem_routing['knowledge_indices']  # [B, S, k]
                k_weights = mem_routing['knowledge_weights']  # [B, S, k]

                # Count usage
                idx_flat = k_idx.reshape(-1)
                knowledge_usage[layer_idx] += torch.bincount(idx_flat, minlength=self.n_knowledge).float()

                # Weighted usage
                for i in range(self.knowledge_k):
                    idx_i = k_idx[:, :, i].reshape(-1)
                    w_i = k_weights[:, :, i].reshape(-1)
                    knowledge_weights[layer_idx].scatter_add_(0, idx_i, w_i)

        # Analyze
        results = {'layers': {}, 'top_neurons': [], 'dead_neurons': []}

        print(f"\nKnowledge Usage Statistics by Layer:")
        for layer_idx in range(self.n_layers):
            usage = knowledge_usage[layer_idx]
            usage_norm = usage / (usage.sum() + 1e-10)

            # Entropy
            entropy = -(usage_norm * torch.log(usage_norm + 1e-10)).sum()
            max_entropy = math.log(self.n_knowledge)
            norm_entropy = (entropy / max_entropy).item()

            # Dead neurons (< 0.1% of average usage)
            threshold = usage.sum() / self.n_knowledge * 0.001
            dead_count = (usage < threshold).sum().item()

            # Top-k concentration
            top8 = torch.topk(usage_norm, 8)[0].sum().item()

            results['layers'][layer_idx] = {
                'entropy': norm_entropy,
                'dead_count': dead_count,
                'top8_concentration': top8
            }

            print(f"  Layer {layer_idx}: entropy={norm_entropy:.3f}, dead={dead_count}/{self.n_knowledge}, top8={top8:.1%}")

        # Global top neurons
        global_usage = knowledge_usage.sum(dim=0)
        global_norm = global_usage / (global_usage.sum() + 1e-10)
        top_neurons = torch.topk(global_norm, 10)

        print(f"\nTop 10 Most Used Knowledge Neurons (global):")
        for i, (idx, prob) in enumerate(zip(top_neurons.indices.tolist(), top_neurons.values.tolist())):
            results['top_neurons'].append({'index': idx, 'usage': prob})
            print(f"  #{idx:2d}: {prob*100:.1f}%")

        # Dead neurons
        dead_threshold = global_usage.sum() / self.n_knowledge * 0.001
        dead_indices = (global_usage < dead_threshold).nonzero().flatten().tolist()
        results['dead_neurons'] = dead_indices
        print(f"\nDead neurons (global): {len(dead_indices)}")

        return results, knowledge_usage

    # ============================================================
    # 4. Attention vs Memory Contribution
    # ============================================================

    @torch.no_grad()
    def analyze_attn_mem_contribution(self, dataloader, max_batches=30):
        """Analyze relative contribution of Attention vs Memory"""
        print("\n" + "=" * 60)
        print("4. ATTENTION vs MEMORY CONTRIBUTION")
        print("=" * 60)

        self.model.eval()

        # Track norms and contributions
        layer_stats = {l: {
            'attn_norm': [],
            'mem_norm': [],
            'attn_contribution': [],
            'attn_mem_cos': []
        } for l in range(self.n_layers)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Attn-Mem Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Manual forward to capture intermediate values
            pos = torch.arange(S, device=self.device).unsqueeze(0)
            x = self.model.token_emb(input_ids) + self.model.pos_emb(pos)
            x = self.model.dropout(x)
            mask = self.model.causal_mask[:, :, :S, :S]

            for layer_idx, layer in enumerate(self.model.layers):
                # Attention
                residual = x
                x_norm = layer.norm1(x)
                attn_out, _ = layer.attention(x_norm, mask)
                x = residual + layer.dropout(attn_out)

                attn_norm = attn_out.norm(dim=-1)  # [B, S]

                # Memory
                residual = x
                x_norm = layer.norm2(x)
                mem_out, _ = layer.memory(x_norm)
                x = residual + layer.dropout(mem_out)

                mem_norm = mem_out.norm(dim=-1)  # [B, S]

                # Statistics
                attn_contribution = attn_norm / (attn_norm + mem_norm + 1e-10)

                # Cosine similarity
                attn_flat = attn_out.reshape(-1, attn_out.shape[-1])
                mem_flat = mem_out.reshape(-1, mem_out.shape[-1])
                cos_sim = F.cosine_similarity(attn_flat, mem_flat, dim=-1)

                layer_stats[layer_idx]['attn_norm'].append(attn_norm.mean().item())
                layer_stats[layer_idx]['mem_norm'].append(mem_norm.mean().item())
                layer_stats[layer_idx]['attn_contribution'].append(attn_contribution.mean().item())
                layer_stats[layer_idx]['attn_mem_cos'].append(cos_sim.mean().item())

        results = {'layers': {}}

        print(f"\nAttention vs Memory by Layer:")
        print(f"{'Layer':<8} {'Attn Norm':<12} {'Mem Norm':<12} {'Attn Ratio':<12} {'Cos Sim':<10}")
        print("-" * 54)

        for layer_idx in range(self.n_layers):
            stats = layer_stats[layer_idx]

            avg_attn = np.mean(stats['attn_norm'])
            avg_mem = np.mean(stats['mem_norm'])
            avg_contrib = np.mean(stats['attn_contribution'])
            avg_cos = np.mean(stats['attn_mem_cos'])

            results['layers'][layer_idx] = {
                'attn_norm': avg_attn,
                'mem_norm': avg_mem,
                'attn_contribution': avg_contrib,
                'attn_mem_cos': avg_cos
            }

            print(f"L{layer_idx:<7} {avg_attn:<12.3f} {avg_mem:<12.3f} {avg_contrib:<12.1%} {avg_cos:<10.3f}")

        return results, layer_stats

    # ============================================================
    # 5. Householder Transform Effect
    # ============================================================

    @torch.no_grad()
    def analyze_householder_effect(self, dataloader, max_batches=20):
        """Analyze how Householder transforms change representations in d_model space"""
        print("\n" + "=" * 60)
        print("5. HOUSEHOLDER TRANSFORM EFFECT")
        print("=" * 60)
        print("  Comparing in d_model (256) space after expand")

        self.model.eval()
        shared = self.model.shared_neurons

        # Get process neurons (for Householder)
        if hasattr(shared, 'process_neurons_qk'):
            process_neurons = shared.process_neurons_qk.data
        else:
            process_neurons = shared.process_neurons.data

        # Get output neurons (for expand)
        output_neurons = shared.output_neurons.data  # [n_output, rank, d_model]

        # Track transformation effects
        combination_effects = defaultdict(list)  # {combo: [(cos_sim, norm_ratio), ...]}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Householder Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Get routing info
            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # Manual forward through first layer Q compressor
            layer0 = self.model.layers[0]
            pos = torch.arange(S, device=self.device).unsqueeze(0)
            x = self.model.token_emb(input_ids) + self.model.pos_emb(pos)
            x = self.model.dropout(x)
            x_norm = layer0.norm1(x)

            # Get Q compression details
            compressor = layer0.attention.compressor_Q

            # Input projection (256 → 64)
            input_scores = compressor.input_router(x_norm)
            input_weights = F.softmax(input_scores, dim=-1)

            # Get input neurons
            if hasattr(shared, 'input_neurons_qk'):
                input_neurons = shared.input_neurons_qk
            elif hasattr(shared, 'input_neurons_q'):
                input_neurons = shared.input_neurons_q
            else:
                input_neurons = shared.input_neurons

            all_proj = torch.einsum('bsd,ndr->bsnr', x_norm, input_neurons)
            x_compressed = (all_proj * input_weights.unsqueeze(-1)).sum(dim=2)  # [B, S, rank]

            # Process routing
            process_scores = compressor.process_router(x_compressed)
            _, process_indices = torch.topk(process_scores, self.process_k, dim=-1)

            # Get output routing (for expand)
            # Use uniform output weights for simplicity (or get from expander)
            output_weights = torch.ones(self.n_output, device=self.device) / self.n_output

            # Sample and analyze
            for _ in range(min(50, B * S)):
                b = torch.randint(B, (1,)).item()
                s = torch.randint(S, (1,)).item()

                combo = tuple(sorted(process_indices[b, s].tolist()))

                x_before = x_compressed[b, s].clone()  # [rank] = 64

                # Apply Householder transforms (in rank space)
                x_transformed = x_before.clone()
                for i in range(self.process_k):
                    v = process_neurons[process_indices[b, s, i]]
                    x_transformed = shared.apply_householder(
                        x_transformed.unsqueeze(0).unsqueeze(0),
                        v.unsqueeze(0).unsqueeze(0)
                    ).squeeze()

                # Expand BOTH to d_model space (64 → 256)
                # x_with_transform = expand(x_transformed)
                # x_no_transform = expand(x_before)

                # Expand: weighted sum of output neurons
                # output_neurons: [n_output, rank, d_model]
                x_with_transform = torch.einsum('r,nrd->d', x_transformed, output_neurons)
                x_with_transform = (x_with_transform * output_weights.sum())  # scale

                x_no_transform = torch.einsum('r,nrd->d', x_before, output_neurons)
                x_no_transform = (x_no_transform * output_weights.sum())

                # Compare in d_model space (256)
                cos_sim = F.cosine_similarity(
                    x_with_transform.unsqueeze(0),
                    x_no_transform.unsqueeze(0)
                ).item()

                norm_with = x_with_transform.norm().item()
                norm_without = x_no_transform.norm().item()

                combination_effects[combo].append((cos_sim, norm_with / (norm_without + 1e-10)))

        # Aggregate results
        results = {'combinations': {}, 'summary': {}}

        all_cos_sims = []
        all_norm_ratios = []

        print(f"\nTop 10 Most Common Process Neuron Combinations:")
        combo_counts = {c: len(effects) for c, effects in combination_effects.items()}
        top_combos = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        for combo, count in top_combos:
            effects = combination_effects[combo]
            avg_cos = np.mean([e[0] for e in effects])
            avg_norm = np.mean([e[1] for e in effects])

            all_cos_sims.extend([e[0] for e in effects])
            all_norm_ratios.extend([e[1] for e in effects])

            results['combinations'][str(combo)] = {
                'count': count,
                'avg_cos_sim': avg_cos,
                'avg_norm_ratio': avg_norm
            }

            print(f"  {combo}: {count}x, cos={avg_cos:.3f}, norm_ratio={avg_norm:.3f}")

        results['summary'] = {
            'avg_cos_sim': np.mean(all_cos_sims) if all_cos_sims else 0,
            'avg_norm_ratio': np.mean(all_norm_ratios) if all_norm_ratios else 0,
            'unique_combinations': len(combination_effects)
        }

        print(f"\nSummary (in d_model=256 space):")
        print(f"  cos(with_H, no_H): {results['summary']['avg_cos_sim']:.3f}  (1.0=no effect, 0.0=orthogonal)")
        print(f"  norm ratio (with/without): {results['summary']['avg_norm_ratio']:.3f}")
        print(f"  Unique combinations: {results['summary']['unique_combinations']}")

        return results, combination_effects

    # ============================================================
    # 6. Token/POS Preferences
    # ============================================================

    @torch.no_grad()
    def analyze_token_preferences(self, dataloader, max_batches=50, top_k_tokens=20):
        """Analyze which tokens prefer which neurons"""
        print("\n" + "=" * 60)
        print("6. TOKEN/POS PREFERENCES")
        print("=" * 60)

        self.model.eval()

        # Token -> Neuron mapping (Q component, layer 0)
        token_neuron_count = torch.zeros(self.vocab_size, self.n_process, device=self.device)
        token_count = torch.zeros(self.vocab_size, device=self.device)

        # POS -> Neuron mapping
        pos_neuron_count = defaultdict(lambda: torch.zeros(self.n_process, device=self.device))
        pos_count = defaultdict(float)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Token Preferences", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # Layer 0 Q routing
            q_idx = routing_infos[0]['attention']['routing_Q']['process_indices']  # [B, S, k]

            # Token mapping
            for b in range(B):
                for s in range(S):
                    tid = input_ids[b, s].item()
                    token_count[tid] += 1
                    for k in range(self.process_k):
                        neuron_idx = q_idx[b, s, k].item()
                        token_neuron_count[tid, neuron_idx] += 1

                    # POS mapping
                    token = self.tokenizer.decode([tid]).strip()
                    pos = simple_pos_tag(token)
                    pos_count[pos] += 1
                    for k in range(self.process_k):
                        neuron_idx = q_idx[b, s, k].item()
                        pos_neuron_count[pos][neuron_idx] += 1

        results = {'tokens': {}, 'pos': {}, 'neurons': {}}

        # Top tokens by frequency
        top_tokens = torch.argsort(token_count, descending=True)[:top_k_tokens]

        print(f"\nTop {top_k_tokens} Token Preferences (L0 Q routing):")
        for tid in top_tokens.tolist():
            token = self.tokenizer.decode([tid]).strip()
            count = int(token_count[tid].item())
            if count < 10:
                continue

            probs = token_neuron_count[tid] / (count * self.process_k + 1e-10)
            top_neurons = torch.topk(probs, 3)

            results['tokens'][token] = {
                'count': count,
                'top_neurons': top_neurons.indices.tolist(),
                'probs': top_neurons.values.tolist()
            }

            neurons_str = ', '.join([f'{n}({p:.2f})' for n, p in
                                    zip(top_neurons.indices.tolist(), top_neurons.values.tolist())])
            print(f"  '{token}' (n={count}): [{neurons_str}]")

        # POS patterns
        print(f"\nPOS Tag -> Preferred Neurons:")
        for pos in sorted(pos_count.keys()):
            if pos_count[pos] < 100:
                continue

            probs = pos_neuron_count[pos] / (pos_count[pos] * self.process_k + 1e-10)
            top_neurons = torch.topk(probs, 3)

            results['pos'][pos] = {
                'count': int(pos_count[pos]),
                'top_neurons': top_neurons.indices.tolist()
            }

            print(f"  {pos:6s} (n={int(pos_count[pos]):5d}): neurons {top_neurons.indices.tolist()}")

        # Neuron -> preferred tokens (reverse mapping)
        print(f"\nNeuron -> Preferred Tokens (L0 Q):")
        neuron_token_count = token_neuron_count.T  # [n_process, vocab_size]
        for neuron_idx in range(min(8, self.n_process)):
            counts = neuron_token_count[neuron_idx]
            top_tokens = torch.topk(counts, 5)
            tokens = [self.tokenizer.decode([tid]).strip() for tid in top_tokens.indices.tolist()]
            results['neurons'][neuron_idx] = tokens
            print(f"  Neuron {neuron_idx:2d}: {', '.join(tokens)}")

        return results

    # ============================================================
    # 7. Information Flow
    # ============================================================

    @torch.no_grad()
    def analyze_information_flow(self, dataloader, max_batches=20):
        """Analyze norm changes through layers"""
        print("\n" + "=" * 60)
        print("7. INFORMATION FLOW (Norm Changes)")
        print("=" * 60)

        self.model.eval()

        layer_norms = {l: {
            'input': [],
            'after_attn': [],
            'after_mem': [],
            'output': []
        } for l in range(self.n_layers)}

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Info Flow", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            pos = torch.arange(S, device=self.device).unsqueeze(0)
            x = self.model.token_emb(input_ids) + self.model.pos_emb(pos)
            x = self.model.dropout(x)
            mask = self.model.causal_mask[:, :, :S, :S]

            for layer_idx, layer in enumerate(self.model.layers):
                layer_norms[layer_idx]['input'].append(x.norm(dim=-1).mean().item())

                # Attention
                residual = x
                x_norm = layer.norm1(x)
                attn_out, _ = layer.attention(x_norm, mask)
                x = residual + layer.dropout(attn_out)
                layer_norms[layer_idx]['after_attn'].append(x.norm(dim=-1).mean().item())

                # Memory
                residual = x
                x_norm = layer.norm2(x)
                mem_out, _ = layer.memory(x_norm)
                x = residual + layer.dropout(mem_out)
                layer_norms[layer_idx]['after_mem'].append(x.norm(dim=-1).mean().item())
                layer_norms[layer_idx]['output'].append(x.norm(dim=-1).mean().item())

        results = {'layers': {}}

        print(f"\nNorm Progression Through Layers:")
        print(f"{'Layer':<8} {'Input':<10} {'→Attn':<10} {'→Mem':<10} {'Output':<10} {'Growth':<10}")
        print("-" * 58)

        for layer_idx in range(self.n_layers):
            norms = layer_norms[layer_idx]
            avg_input = np.mean(norms['input'])
            avg_attn = np.mean(norms['after_attn'])
            avg_mem = np.mean(norms['after_mem'])
            avg_output = np.mean(norms['output'])
            growth = avg_output / (avg_input + 1e-10)

            results['layers'][layer_idx] = {
                'input_norm': avg_input,
                'after_attn': avg_attn,
                'after_mem': avg_mem,
                'output_norm': avg_output,
                'growth_ratio': growth
            }

            print(f"L{layer_idx:<7} {avg_input:<10.2f} {avg_attn:<10.2f} {avg_mem:<10.2f} {avg_output:<10.2f} {growth:<10.2f}x")

        return results

    # ============================================================
    # 8. SharedNeurons Health
    # ============================================================

    def analyze_shared_neurons_health(self):
        """Analyze SharedNeurons orthogonality and condition"""
        print("\n" + "=" * 60)
        print("8. SHARED NEURONS HEALTH")
        print("=" * 60)

        shared = self.model.shared_neurons
        results = {}

        # Get process neurons
        if hasattr(shared, 'process_neurons_qk'):
            process_neurons = shared.process_neurons_qk.data
            pool_name = 'QK'
        else:
            process_neurons = shared.process_neurons.data
            pool_name = 'single'

        # Norms (should be ~1 for Householder)
        norms = process_neurons.norm(dim=-1)
        results['process_norms'] = {
            'mean': norms.mean().item(),
            'std': norms.std().item(),
            'min': norms.min().item(),
            'max': norms.max().item()
        }

        print(f"\nProcess Neurons ({pool_name}):")
        print(f"  Norms: mean={norms.mean().item():.4f}, std={norms.std().item():.4f}")
        print(f"  Range: [{norms.min().item():.4f}, {norms.max().item():.4f}]")

        # Cosine similarity (orthogonality)
        v_norm = F.normalize(process_neurons, dim=-1)
        cos_sim = v_norm @ v_norm.T
        mask = ~torch.eye(self.n_process, dtype=torch.bool, device=cos_sim.device)
        off_diag = cos_sim[mask]

        results['cosine_sim'] = {
            'mean': off_diag.abs().mean().item(),
            'max': off_diag.abs().max().item()
        }

        print(f"  Cosine sim: mean={off_diag.abs().mean().item():.4f}, max={off_diag.abs().max().item():.4f}")

        # Input neurons
        if hasattr(shared, 'input_neurons_qk'):
            input_neurons = shared.input_neurons_qk.data
        elif hasattr(shared, 'input_neurons_q'):
            input_neurons = shared.input_neurons_q.data
        else:
            input_neurons = shared.input_neurons.data

        print(f"\nInput Neurons:")
        input_conds = []
        input_orth_errors = []
        n_input = input_neurons.shape[0]
        rank = input_neurons.shape[2]

        for i in range(n_input):
            W = input_neurons[i]  # [d_model, rank]
            _, s, _ = torch.linalg.svd(W, full_matrices=False)
            cond = (s[0] / (s[-1] + 1e-10)).item()
            input_conds.append(cond)

            WtW = W.T @ W
            I = torch.eye(rank, device=W.device)
            orth_error = (WtW - I).abs().max().item()
            input_orth_errors.append(orth_error)

        results['input_neurons'] = {
            'avg_condition': np.mean(input_conds),
            'max_condition': max(input_conds),
            'avg_orth_error': np.mean(input_orth_errors),
            'max_orth_error': max(input_orth_errors)
        }

        print(f"  Condition: avg={np.mean(input_conds):.2f}, max={max(input_conds):.2f}")
        print(f"  Orth error: avg={np.mean(input_orth_errors):.2e}, max={max(input_orth_errors):.2e}")

        # Knowledge neurons
        knowledge_K = shared.knowledge_K.data
        K_norms = knowledge_K.norm(dim=-1)
        K_norm_v = F.normalize(knowledge_K, dim=-1)
        K_sim = K_norm_v @ K_norm_v.T
        K_mask = ~torch.eye(self.n_knowledge, dtype=torch.bool, device=K_sim.device)
        K_off_diag = K_sim[K_mask]

        results['knowledge'] = {
            'K_norm_mean': K_norms.mean().item(),
            'K_sim_mean': K_off_diag.abs().mean().item()
        }

        print(f"\nKnowledge Neurons:")
        print(f"  K norm: mean={K_norms.mean().item():.4f}")
        print(f"  K similarity: mean={K_off_diag.abs().mean().item():.4f}")

        # Auxiliary losses
        print(f"\nAuxiliary Losses:")
        orth_loss = self.model.orthogonality_loss().item()
        norm_loss = self.model.process_norm_loss().item()
        div_loss = self.model.knowledge_diversity_loss().item()

        results['losses'] = {
            'orthogonality': orth_loss,
            'process_norm': norm_loss,
            'knowledge_diversity': div_loss
        }

        print(f"  Orthogonality: {orth_loss:.6f}")
        print(f"  Process norm: {norm_loss:.6f}")
        print(f"  Knowledge diversity: {div_loss:.6f}")

        return results

    # ============================================================
    # 9. Recommendations
    # ============================================================

    def generate_recommendations(self, all_results):
        """Generate recommendations based on analysis results"""
        print("\n" + "=" * 60)
        print("9. RECOMMENDATIONS")
        print("=" * 60)

        recommendations = []

        # Check Q-K diversity
        if 'qk_diversity' in all_results:
            qk = all_results['qk_diversity']
            avg_jaccard = qk.get('avg_qk_jaccard', 0)
            if avg_jaccard > 0.5:
                recommendations.append(
                    f"⚠️  High Q-K overlap (Jaccard={avg_jaccard:.2f}). "
                    "Consider adding diversity loss or separate Q/K pools."
                )

        # Check utilization
        if 'neuron_usage' in all_results:
            usage = all_results['neuron_usage']
            avg_util = usage['summary'].get('avg_utilization', 0)
            if avg_util < 0.3:
                recommendations.append(
                    f"⚠️  Low neuron utilization ({avg_util:.1%}). "
                    "Consider reducing n_process or adding load balancing loss."
                )
            elif avg_util > 0.9:
                recommendations.append(
                    f"✅ High neuron utilization ({avg_util:.1%}). "
                    "Consider increasing n_process for more capacity."
                )

        # Check knowledge neurons
        if 'knowledge' in all_results:
            k = all_results['knowledge']
            if 'dead_neurons' in k and len(k['dead_neurons']) > self.n_knowledge * 0.2:
                recommendations.append(
                    f"⚠️  Many dead knowledge neurons ({len(k['dead_neurons'])}/{self.n_knowledge}). "
                    "Consider reducing n_knowledge or adding usage regularization."
                )

        # Check attention-memory balance
        if 'attn_mem' in all_results:
            am = all_results['attn_mem']
            for layer_idx, layer_data in am['layers'].items():
                contrib = layer_data['attn_contribution']
                if contrib < 0.3:
                    recommendations.append(
                        f"ℹ️  Layer {layer_idx}: Memory dominates ({1-contrib:.1%}). "
                        "This may be intentional for knowledge retrieval."
                    )
                elif contrib > 0.8:
                    recommendations.append(
                        f"ℹ️  Layer {layer_idx}: Attention dominates ({contrib:.1%}). "
                        "Memory may be underutilized."
                    )

        # Check SharedNeurons health
        if 'shared_health' in all_results:
            health = all_results['shared_health']
            if health['process_norms']['std'] > 0.1:
                recommendations.append(
                    f"⚠️  Process neuron norms vary significantly (std={health['process_norms']['std']:.3f}). "
                    "Consider adding norm regularization."
                )
            if health['cosine_sim']['max'] > 0.5:
                recommendations.append(
                    f"⚠️  High process neuron similarity (max={health['cosine_sim']['max']:.3f}). "
                    "Consider adding orthogonality loss."
                )

        # Check Householder effect
        if 'householder' in all_results:
            h = all_results['householder']['summary']
            if h['avg_cos_sim'] < 0.3:
                recommendations.append(
                    f"ℹ️  Large Householder transformations (cos={h['avg_cos_sim']:.3f}). "
                    "This indicates strong representation changes."
                )

        if not recommendations:
            recommendations.append("✅ All metrics look healthy!")

        for rec in recommendations:
            print(rec)

        return recommendations

    # ============================================================
    # Run All
    # ============================================================

    def run_all(self, dataloader, max_batches=50):
        """Run all analyses"""
        print("\n" + "=" * 60)
        print("DAWN v8 COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        print(f"Model: DAWN v{self.model.__version__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Layers: {self.n_layers}, Process: {self.n_process}, Knowledge: {self.n_knowledge}")

        all_results = {}

        # 1. Neuron usage
        usage_results, usage, knowledge_usage = self.analyze_neuron_usage(dataloader, max_batches)
        all_results['neuron_usage'] = usage_results

        # 2. Q/K diversity
        all_results['qk_diversity'] = self.analyze_qk_diversity(dataloader, max_batches)

        # 3. Knowledge neurons
        knowledge_results, _ = self.analyze_knowledge_neurons(dataloader, max_batches)
        all_results['knowledge'] = knowledge_results

        # 4. Attention vs Memory
        attn_mem_results, _ = self.analyze_attn_mem_contribution(dataloader, min(max_batches, 30))
        all_results['attn_mem'] = attn_mem_results

        # 5. Householder effect
        householder_results, _ = self.analyze_householder_effect(dataloader, min(max_batches, 20))
        all_results['householder'] = householder_results

        # 6. Token preferences
        all_results['token_prefs'] = self.analyze_token_preferences(dataloader, max_batches)

        # 7. Information flow
        all_results['info_flow'] = self.analyze_information_flow(dataloader, min(max_batches, 20))

        # 8. SharedNeurons health
        all_results['shared_health'] = self.analyze_shared_neurons_health()

        # 9. Recommendations
        recommendations = self.generate_recommendations(all_results)
        all_results['recommendations'] = recommendations

        return all_results

    # ============================================================
    # Visualization
    # ============================================================

    def visualize(self, all_results, output_path):
        """Create comprehensive visualization"""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available, skipping visualization")
            return

        print("\n" + "=" * 60)
        print("CREATING VISUALIZATION")
        print("=" * 60)

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # 1. Neuron utilization by component
        ax = axes[0, 0]
        if 'neuron_usage' in all_results:
            usage = all_results['neuron_usage']
            x = np.arange(self.n_layers)
            width = 0.15

            for i, comp in enumerate(self.components):
                if comp in usage['components']:
                    values = [usage['components'][comp]['layers'][l]['eff_rank'] /
                             usage['components'][comp]['layers'][l]['max_neurons'] * 100
                             for l in range(self.n_layers)]
                    ax.bar(x + i * width, values, width, label=comp)

            ax.set_xlabel('Layer')
            ax.set_ylabel('Effective Rank (%)')
            ax.set_title('Neuron Utilization by Component')
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels([f'L{i}' for i in range(self.n_layers)])
            ax.legend()
            ax.set_ylim(0, 100)

        # 2. Q-K Jaccard by layer
        ax = axes[0, 1]
        if 'qk_diversity' in all_results:
            qk = all_results['qk_diversity']
            jaccards = qk['qk_jaccard']
            ax.bar(range(len(jaccards)), jaccards, color='steelblue')
            ax.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Q-K Jaccard')
            ax.set_title('Q-K Router Overlap (lower = more diverse)')
            ax.set_xticks(range(len(jaccards)))
            ax.set_xticklabels([f'L{i}' for i in range(len(jaccards))])
            ax.legend()

        # 3. Knowledge usage by layer
        ax = axes[0, 2]
        if 'knowledge' in all_results:
            k = all_results['knowledge']
            entropies = [k['layers'][l]['entropy'] for l in range(self.n_layers)]
            dead_pcts = [k['layers'][l]['dead_count'] / self.n_knowledge * 100 for l in range(self.n_layers)]

            x = np.arange(self.n_layers)
            ax.bar(x - 0.2, entropies, 0.4, label='Entropy', color='steelblue')
            ax.bar(x + 0.2, [d/100 for d in dead_pcts], 0.4, label='Dead %', color='coral')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Value')
            ax.set_title('Knowledge Neuron Usage')
            ax.set_xticks(x)
            ax.set_xticklabels([f'L{i}' for i in range(self.n_layers)])
            ax.legend()

        # 4. Attention vs Memory contribution
        ax = axes[1, 0]
        if 'attn_mem' in all_results:
            am = all_results['attn_mem']
            attn_contribs = [am['layers'][l]['attn_contribution'] for l in range(self.n_layers)]
            mem_contribs = [1 - c for c in attn_contribs]

            x = np.arange(self.n_layers)
            ax.bar(x, attn_contribs, label='Attention', color='steelblue')
            ax.bar(x, mem_contribs, bottom=attn_contribs, label='Memory', color='coral')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Contribution')
            ax.set_title('Attention vs Memory Contribution')
            ax.set_xticks(x)
            ax.set_xticklabels([f'L{i}' for i in range(self.n_layers)])
            ax.legend()

        # 5. Information flow
        ax = axes[1, 1]
        if 'info_flow' in all_results:
            flow = all_results['info_flow']
            stages = ['input', 'after_attn', 'after_mem', 'output']

            for l in range(self.n_layers):
                norms = [flow['layers'][l][f'{s}_norm' if s != 'after_attn' and s != 'after_mem' else s]
                        for s in stages]
                ax.plot(stages, norms, marker='o', label=f'L{l}')

            ax.set_xlabel('Stage')
            ax.set_ylabel('Norm')
            ax.set_title('Information Flow (Norm)')
            ax.legend()

        # 6. Process neuron norms distribution
        ax = axes[1, 2]
        if 'shared_health' in all_results:
            health = all_results['shared_health']
            shared = self.model.shared_neurons

            if hasattr(shared, 'process_neurons_qk'):
                norms = shared.process_neurons_qk.data.norm(dim=-1).cpu().numpy()
            else:
                norms = shared.process_neurons.data.norm(dim=-1).cpu().numpy()

            ax.hist(norms, bins=20, color='steelblue', edgecolor='black')
            ax.axvline(x=1.0, color='r', linestyle='--', label='Target (1.0)')
            ax.set_xlabel('Norm')
            ax.set_ylabel('Count')
            ax.set_title('Process Neuron Norms')
            ax.legend()

        # 7. Gini coefficients
        ax = axes[2, 0]
        if 'neuron_usage' in all_results:
            usage = all_results['neuron_usage']
            x = np.arange(self.n_layers)
            width = 0.15

            for i, comp in enumerate(self.components):
                if comp in usage['components']:
                    values = [usage['components'][comp]['layers'][l]['gini']
                             for l in range(self.n_layers)]
                    ax.bar(x + i * width, values, width, label=comp)

            ax.set_xlabel('Layer')
            ax.set_ylabel('Gini Coefficient')
            ax.set_title('Usage Concentration (higher = more concentrated)')
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels([f'L{i}' for i in range(self.n_layers)])
            ax.legend()

        # 8. Component pair Jaccard
        ax = axes[2, 1]
        if 'qk_diversity' in all_results:
            qk = all_results['qk_diversity']
            pairs = list(qk['pair_jaccards'].keys())
            avg_jaccards = [np.mean(qk['pair_jaccards'][p]) for p in pairs]

            ax.barh(pairs, avg_jaccards, color='steelblue')
            ax.set_xlabel('Average Jaccard')
            ax.set_title('Component Pair Similarity')
            ax.axvline(x=0.5, color='r', linestyle='--')

        # 9. Householder effect
        ax = axes[2, 2]
        if 'householder' in all_results:
            h = all_results['householder']
            if h['combinations']:
                combos = list(h['combinations'].keys())[:10]
                cos_sims = [h['combinations'][c]['avg_cos_sim'] for c in combos]

                ax.barh(range(len(combos)), cos_sims, color='steelblue')
                ax.set_yticks(range(len(combos)))
                ax.set_yticklabels([c[:20] + '...' if len(c) > 20 else c for c in combos], fontsize=8)
                ax.set_xlabel('Cos Similarity')
                ax.set_title('Householder Effect (top 10 combos)')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()

        print(f"Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN v8 Comprehensive Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--val_data', type=str,
                        default='/content/drive/MyDrive/data/validation/wikitext_5to1_texts.pkl',
                        help='Path to validation data')
    parser.add_argument('--max_batches', type=int, default=50,
                        help='Max batches for analysis')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./analysis_comprehensive',
                        help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    from models import create_model_by_version

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        best_model = checkpoint_path / 'best_model.pt'
        if best_model.exists():
            checkpoint_path = best_model
        else:
            pt_files = list(checkpoint_path.glob('*.pt'))
            if pt_files:
                checkpoint_path = max(pt_files, key=lambda p: p.stat().st_mtime)

    print(f"\nLoading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('model_config', checkpoint.get('config', {}))
    model_version = config.get('model_version', '8.0')

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
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Data
    print(f"\nLoading data: {args.val_data}")
    import pickle
    with open(args.val_data, 'rb') as f:
        val_texts = pickle.load(f)

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_len=128):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
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

    # Run analysis
    analyzer = ComprehensiveAnalyzer(model, tokenizer, device)
    all_results = analyzer.run_all(dataloader, max_batches=args.max_batches)

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

    results_path = os.path.join(args.output_dir, 'comprehensive_results.json')
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Visualization
    viz_path = os.path.join(args.output_dir, 'comprehensive_analysis.png')
    analyzer.visualize(all_results, viz_path)

    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
