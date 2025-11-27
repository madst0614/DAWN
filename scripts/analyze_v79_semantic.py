"""
DAWN Semantic Analysis Script (v7.9 / v8.0 compatible)
뉴런이 무엇을 배웠는지 분석

핵심 질문:
- 각 뉴런이 어떤 토큰/패턴에 반응하는가?
- 특정 뉴런 조합이 특정 의미를 처리하는가?

분석 항목:
1. 토큰별 뉴런 활성화 패턴
2. 뉴런별 선호 토큰
3. 위치별 뉴런 활성화
4. 문맥 의존성 분석
5. 뉴런 클러스터링
6. Layer별 역할 분화
7. Attention과 뉴런 상관관계
8. Q/K/V별 뉴런 역할 차이 (v7.9) / Memory 분석 (v8.0)
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict
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
    get_neurons, has_memory, VersionAdapter
)

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    # Check if running in notebook/Colab
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            IN_NOTEBOOK = True
            # Use inline backend for notebooks
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
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from scipy.cluster.hierarchy import dendrogram, linkage
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, skipping clustering")

try:
    import nltk
    from nltk.tokenize import word_tokenize
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("Warning: nltk not available, using simple POS rules")


def get_underlying_model(model):
    """Get underlying model from torch.compile wrapper"""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def simple_pos_tag(token):
    """Simple rule-based POS tagging"""
    token_lower = token.lower()

    # Punctuation
    if token in '.,!?;:\'"()-[]{}':
        return 'PUNCT'
    # Numbers
    if token.isdigit() or token.replace('.', '').replace(',', '').isdigit():
        return 'NUM'
    # Articles
    if token_lower in ['a', 'an', 'the']:
        return 'DET'
    # Pronouns
    if token_lower in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                       'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs']:
        return 'PRON'
    # Prepositions
    if token_lower in ['in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into', 'through']:
        return 'ADP'
    # Conjunctions
    if token_lower in ['and', 'or', 'but', 'so', 'yet', 'nor', 'for']:
        return 'CONJ'
    # Auxiliaries
    if token_lower in ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might']:
        return 'AUX'
    # Common verbs (ending patterns)
    if token_lower.endswith('ing') or token_lower.endswith('ed') or token_lower.endswith('es'):
        return 'VERB'
    # Adjectives (ending patterns)
    if token_lower.endswith('ly'):
        return 'ADV'
    if token_lower.endswith('ful') or token_lower.endswith('less') or token_lower.endswith('ive'):
        return 'ADJ'
    # Default to NOUN
    return 'NOUN'


class SemanticAnalyzer:
    """GPU-optimized semantic analysis for DAWN v7.9 / v8.0"""

    def __init__(self, model, tokenizer, device, version="7.9"):
        self.model = get_underlying_model(model)
        self.tokenizer = tokenizer
        self.device = device
        self.version = version

        self.n_layers = len(self.model.layers)
        self.n_process = self.model.n_process
        self.process_k = self.model.process_k
        self.vocab_size = self.tokenizer.vocab_size

        # v8.0 specific
        self.has_memory = has_memory(version)
        if self.has_memory:
            self.n_knowledge = self.model.n_knowledge
            self.knowledge_k = self.model.knowledge_k

        # ⚡ GPU tensors for accumulation
        # Token -> Neuron mapping: [vocab_size, n_layers, n_process]
        self.token_neuron_count = torch.zeros(
            self.vocab_size, self.n_layers, self.n_process,
            device=device, dtype=torch.float32
        )
        self.token_count = torch.zeros(self.vocab_size, device=device, dtype=torch.float32)

        # Position -> Neuron mapping: [max_pos_bins, n_layers, n_process]
        self.max_pos = 128
        self.pos_bins = [0, 5, 10, 20, 50, 100, 128]  # Position bins
        self.n_pos_bins = len(self.pos_bins)
        self.pos_neuron_count = torch.zeros(
            self.n_pos_bins, self.n_layers, self.n_process,
            device=device, dtype=torch.float32
        )
        self.pos_count = torch.zeros(self.n_pos_bins, device=device, dtype=torch.float32)

        # Context tracking (prev_token -> curr_neuron)
        self.context_neuron_count = torch.zeros(
            self.vocab_size, self.n_layers, self.n_process,
            device=device, dtype=torch.float32
        )

        # Attention statistics per neuron
        self.neuron_attn_entropy = torch.zeros(
            self.n_layers, self.n_process, device=device, dtype=torch.float32
        )
        self.neuron_attn_count = torch.zeros(
            self.n_layers, self.n_process, device=device, dtype=torch.float32
        )

        self.total_tokens = 0

    def get_pos_bin(self, position):
        """Get position bin index"""
        for i, threshold in enumerate(self.pos_bins[1:], 1):
            if position < threshold:
                return i - 1
        return self.n_pos_bins - 1

    @torch.no_grad()
    def collect_data(self, dataloader, max_batches=100):
        """Collect semantic data from validation set - GPU optimized"""
        print("\n" + "=" * 60)
        print("COLLECTING SEMANTIC DATA")
        print("=" * 60)

        self.model.eval()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape
            self.total_tokens += B * S

            # Forward with routing info
            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # ⚡ Vectorized token counting
            token_flat = input_ids.reshape(-1)  # [B*S]
            self.token_count.scatter_add_(
                0, token_flat,
                torch.ones_like(token_flat, dtype=torch.float32)
            )

            # ⚡ Position bin counting
            positions = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)  # [B, S]
            pos_flat = positions.reshape(-1)

            # Map positions to bins (vectorized)
            pos_bins_tensor = torch.tensor(self.pos_bins[1:], device=self.device)
            pos_bin_idx = torch.bucketize(pos_flat, pos_bins_tensor)  # [B*S]

            for layer_idx, routing_info in enumerate(routing_infos):
                # Version-agnostic routing info access
                compat = get_routing_info_compat(routing_info, self.version)
                process_idx = compat['process_indices']  # [B, S, k]

                # ⚡ Token-Neuron mapping (GPU scatter)
                # For each (token, layer), increment selected neurons
                idx_flat = process_idx.reshape(-1)  # [B*S*k]
                token_expanded = token_flat.unsqueeze(1).expand(-1, self.process_k).reshape(-1)  # [B*S*k]

                # Create combined index for scatter
                # index = token * (n_layers * n_process) + layer * n_process + neuron
                combined_idx = (token_expanded * self.n_layers * self.n_process +
                               layer_idx * self.n_process + idx_flat)

                self.token_neuron_count.view(-1).scatter_add_(
                    0, combined_idx,
                    torch.ones_like(combined_idx, dtype=torch.float32)
                )

                # ⚡ Position-Neuron mapping
                pos_expanded = pos_bin_idx.unsqueeze(1).expand(-1, self.process_k).reshape(-1)
                pos_combined_idx = (pos_expanded * self.n_layers * self.n_process +
                                   layer_idx * self.n_process + idx_flat)

                self.pos_neuron_count.view(-1).scatter_add_(
                    0, pos_combined_idx,
                    torch.ones_like(pos_combined_idx, dtype=torch.float32)
                )

                # ⚡ Context tracking (previous token -> current neuron)
                if S > 1:
                    prev_tokens = input_ids[:, :-1].reshape(-1)  # [B*(S-1)]
                    curr_neurons = process_idx[:, 1:, :].reshape(-1)  # [B*(S-1)*k]
                    prev_expanded = prev_tokens.unsqueeze(1).expand(-1, self.process_k).reshape(-1)

                    ctx_combined_idx = (prev_expanded * self.n_layers * self.n_process +
                                       layer_idx * self.n_process + curr_neurons)

                    self.context_neuron_count.view(-1).scatter_add_(
                        0, ctx_combined_idx,
                        torch.ones_like(ctx_combined_idx, dtype=torch.float32)
                    )

            # Position bin counts
            for b in range(self.n_pos_bins):
                mask = (pos_bin_idx == b)
                self.pos_count[b] += mask.sum().float()

        print(f"  Total tokens processed: {self.total_tokens:,}")

    def analyze_token_neuron_mapping(self, min_count=10, top_k=10):
        """Analyze which neurons each token activates"""
        print("\n" + "=" * 60)
        print("1. TOKEN -> NEURON MAPPING")
        print("=" * 60)

        results = {'tokens': {}}

        # ⚡ Normalize counts (GPU)
        # [vocab_size, n_layers, n_process]
        token_count_expanded = self.token_count.unsqueeze(1).unsqueeze(2)
        token_neuron_prob = self.token_neuron_count / (token_count_expanded * self.process_k + 1e-10)

        # Find tokens with enough samples
        valid_tokens = (self.token_count >= min_count).nonzero().squeeze(-1)
        print(f"\n  Tokens with >= {min_count} occurrences: {len(valid_tokens)}")

        # Get top neurons per token per layer
        top_neurons = torch.topk(token_neuron_prob, k=min(top_k, self.n_process), dim=2)

        # Sample interesting tokens
        interesting_tokens = []

        # Most common tokens
        common_tokens = torch.topk(self.token_count, k=50)[1]
        interesting_tokens.extend(common_tokens.tolist())

        # Decode and analyze
        print("\n  Top 20 most common tokens and their preferred neurons:")
        for i, token_id in enumerate(common_tokens[:20].tolist()):
            token_str = self.tokenizer.decode([token_id])
            count = int(self.token_count[token_id].item())

            # Get preferred neurons across all layers
            layer_neurons = []
            for layer_idx in range(self.n_layers):
                neurons = top_neurons.indices[token_id, layer_idx, :3].tolist()
                probs = top_neurons.values[token_id, layer_idx, :3].tolist()
                layer_neurons.append((neurons, probs))

            # Store results
            results['tokens'][token_str] = {
                'token_id': token_id,
                'count': count,
                'layer_neurons': {f'layer_{l}': {'neurons': n, 'probs': p}
                                 for l, (n, p) in enumerate(layer_neurons)}
            }

            # Print summary
            l0_neurons = layer_neurons[0][0]
            print(f"    '{token_str}' (n={count}): L0 neurons {l0_neurons}")

        return results, token_neuron_prob

    def analyze_neuron_token_preferences(self, token_neuron_prob, top_k=15):
        """Analyze which tokens each neuron prefers"""
        print("\n" + "=" * 60)
        print("2. NEURON -> TOKEN PREFERENCES")
        print("=" * 60)

        results = {'neurons': {}}

        # ⚡ Transpose: [n_layers, n_process, vocab_size]
        neuron_token_prob = token_neuron_prob.permute(1, 2, 0)

        # Weight by token frequency (more meaningful)
        token_freq = self.token_count / self.token_count.sum()
        weighted_prob = neuron_token_prob * token_freq.unsqueeze(0).unsqueeze(0)

        print("\n  Neuron profiles (Layer 0):")
        for neuron_idx in range(min(self.n_process, 16)):  # Show first 16 neurons
            layer_idx = 0

            # Top tokens for this neuron
            top_tokens = torch.topk(weighted_prob[layer_idx, neuron_idx], k=top_k)

            token_strs = [self.tokenizer.decode([tid]) for tid in top_tokens.indices.tolist()]
            token_probs = top_tokens.values.tolist()

            # Categorize by POS
            pos_counts = defaultdict(int)
            for token_str in token_strs[:10]:
                pos = simple_pos_tag(token_str.strip())
                pos_counts[pos] += 1

            dominant_pos = max(pos_counts.items(), key=lambda x: x[1])[0] if pos_counts else 'UNKNOWN'

            results['neurons'][f'L{layer_idx}_N{neuron_idx}'] = {
                'top_tokens': token_strs,
                'probs': token_probs,
                'dominant_pos': dominant_pos,
                'pos_distribution': dict(pos_counts)
            }

            top3 = ', '.join([f"'{t.strip()}'" for t in token_strs[:3]])
            print(f"    Neuron {neuron_idx:2d} ({dominant_pos:5s}): {top3}")

        return results, neuron_token_prob

    def analyze_position_patterns(self):
        """Analyze position-dependent neuron activation"""
        print("\n" + "=" * 60)
        print("3. POSITION -> NEURON PATTERNS")
        print("=" * 60)

        results = {'position_bins': {}}

        # ⚡ Normalize (GPU)
        pos_count_expanded = self.pos_count.unsqueeze(1).unsqueeze(2)
        pos_neuron_prob = self.pos_neuron_count / (pos_count_expanded * self.process_k + 1e-10)

        bin_names = ['0-4', '5-9', '10-19', '20-49', '50-99', '100+']

        print("\n  Position-dependent neuron preferences:")
        for bin_idx, bin_name in enumerate(bin_names):
            # Get top neurons for this position bin
            top_neurons_per_layer = {}

            for layer_idx in range(self.n_layers):
                top = torch.topk(pos_neuron_prob[bin_idx, layer_idx], k=5)
                top_neurons_per_layer[f'layer_{layer_idx}'] = {
                    'neurons': top.indices.tolist(),
                    'probs': top.values.tolist()
                }

            results['position_bins'][bin_name] = top_neurons_per_layer

            l0_neurons = top_neurons_per_layer['layer_0']['neurons'][:3]
            print(f"    Position {bin_name:6s}: L0 top neurons {l0_neurons}")

        # Check if position affects neuron selection
        # Compute entropy of neuron distribution per position
        entropy_per_pos = -(pos_neuron_prob * torch.log(pos_neuron_prob + 1e-10)).sum(dim=2)

        print("\n  Entropy of neuron selection by position (higher = more uniform):")
        for bin_idx, bin_name in enumerate(bin_names):
            ent = entropy_per_pos[bin_idx].mean().item()
            print(f"    Position {bin_name:6s}: entropy = {ent:.3f}")

        return results, pos_neuron_prob

    def analyze_context_dependency(self, token_neuron_prob, min_count=20):
        """Analyze how context affects neuron selection"""
        print("\n" + "=" * 60)
        print("4. CONTEXT DEPENDENCY ANALYSIS")
        print("=" * 60)

        results = {'context_effects': []}

        # ⚡ Normalize context counts (GPU)
        # context_neuron_count: [vocab_size, n_layers, n_process]
        # This tracks: given prev_token, what neurons are selected for curr_token

        # Compare with unconditional distribution
        # For each prev_token, compute KL divergence from marginal distribution

        marginal = self.token_neuron_count.sum(dim=0) / self.token_neuron_count.sum()  # [n_layers, n_process]

        context_count_sum = self.context_neuron_count.sum(dim=(1, 2), keepdim=True)
        context_prob = self.context_neuron_count / (context_count_sum + 1e-10)  # [vocab_size, n_layers, n_process]

        # Find tokens that significantly change neuron distribution
        # KL(context || marginal) for each prev_token
        kl_div = (context_prob * torch.log((context_prob + 1e-10) / (marginal.unsqueeze(0) + 1e-10))).sum(dim=(1, 2))

        # Top context-sensitive triggers
        valid_mask = context_count_sum.squeeze() > min_count
        kl_div_masked = kl_div.clone()
        kl_div_masked[~valid_mask] = -float('inf')

        top_triggers = torch.topk(kl_div_masked, k=20)

        print("\n  Tokens that most affect next token's neuron selection:")
        for i, (token_id, kl) in enumerate(zip(top_triggers.indices.tolist(), top_triggers.values.tolist())):
            if kl < 0:
                continue
            token_str = self.tokenizer.decode([token_id])

            # What neurons are preferred after this token?
            top_neurons = torch.topk(context_prob[token_id, 0], k=5)
            neurons = top_neurons.indices.tolist()

            results['context_effects'].append({
                'trigger_token': token_str,
                'kl_divergence': kl,
                'preferred_neurons_L0': neurons
            })

            print(f"    After '{token_str}': KL={kl:.3f}, L0 neurons {neurons[:3]}")

        return results

    def analyze_pos_neuron_mapping(self, token_neuron_prob):
        """Analyze POS tag to neuron mapping"""
        print("\n" + "=" * 60)
        print("5. POS TAG -> NEURON ANALYSIS")
        print("=" * 60)

        results = {'pos_neurons': {}}

        # Group tokens by POS
        pos_neuron_sum = defaultdict(lambda: torch.zeros(self.n_layers, self.n_process, device=self.device))
        pos_counts = defaultdict(int)

        # Process valid tokens
        valid_tokens = (self.token_count >= 5).nonzero().squeeze(-1)

        for token_id in tqdm(valid_tokens.tolist(), desc="POS Analysis"):
            token_str = self.tokenizer.decode([token_id]).strip()
            if not token_str or token_str.startswith('['):  # Skip special tokens
                continue

            pos = simple_pos_tag(token_str)
            count = self.token_count[token_id].item()

            pos_neuron_sum[pos] += token_neuron_prob[token_id] * count
            pos_counts[pos] += count

        # Normalize and analyze
        print("\n  POS-specific neuron preferences:")
        for pos in sorted(pos_counts.keys()):
            if pos_counts[pos] < 100:
                continue

            avg_prob = pos_neuron_sum[pos] / pos_counts[pos]

            # Top neurons for this POS at Layer 0
            top = torch.topk(avg_prob[0], k=5)

            results['pos_neurons'][pos] = {
                'count': pos_counts[pos],
                'top_neurons_L0': top.indices.tolist(),
                'probs': top.values.tolist()
            }

            neurons = top.indices.tolist()[:3]
            print(f"    {pos:6s} (n={int(pos_counts[pos]):6d}): L0 neurons {neurons}")

        return results

    def analyze_layer_specialization(self, token_neuron_prob):
        """Analyze how layers differ in their neuron usage"""
        print("\n" + "=" * 60)
        print("6. LAYER SPECIALIZATION")
        print("=" * 60)

        results = {'layers': {}}

        # ⚡ Compute layer-wise statistics (GPU)
        # Entropy of neuron usage per layer
        layer_avg_prob = token_neuron_prob.mean(dim=0)  # [n_layers, n_process]
        layer_avg_prob = layer_avg_prob / layer_avg_prob.sum(dim=1, keepdim=True)

        entropy = -(layer_avg_prob * torch.log(layer_avg_prob + 1e-10)).sum(dim=1)
        max_entropy = math.log(self.n_process)

        # Gini coefficient per layer
        sorted_prob, _ = torch.sort(layer_avg_prob, dim=1)
        n = self.n_process
        index = torch.arange(1, n + 1, dtype=torch.float32, device=self.device).unsqueeze(0)
        gini = ((2 * index - n - 1) * sorted_prob).sum(dim=1) / (n * sorted_prob.sum(dim=1) + 1e-10)

        print("\n  Layer-wise neuron usage patterns:")
        for layer_idx in range(self.n_layers):
            ent = entropy[layer_idx].item() / max_entropy
            g = gini[layer_idx].item()

            # Top neurons
            top = torch.topk(layer_avg_prob[layer_idx], k=5)

            results['layers'][f'layer_{layer_idx}'] = {
                'entropy': ent,
                'gini': g,
                'top_neurons': top.indices.tolist(),
                'top_probs': top.values.tolist()
            }

            print(f"    Layer {layer_idx}: entropy={ent:.3f}, gini={g:.3f}, top={top.indices.tolist()[:3]}")

        # Cross-layer neuron correlation
        print("\n  Cross-layer neuron correlation:")
        for i in range(self.n_layers):
            for j in range(i + 1, self.n_layers):
                corr = F.cosine_similarity(
                    layer_avg_prob[i].unsqueeze(0),
                    layer_avg_prob[j].unsqueeze(0)
                ).item()
                print(f"    Layer {i} <-> Layer {j}: {corr:.3f}")

        return results

    def cluster_neurons(self, token_neuron_prob, n_clusters=8):
        """Cluster neurons based on token preferences"""
        print("\n" + "=" * 60)
        print("7. NEURON CLUSTERING")
        print("=" * 60)

        if not HAS_SKLEARN:
            print("  Skipping (sklearn not available)")
            return {}

        results = {'clusters': {}}

        # Use Layer 0 for clustering
        layer_idx = 0

        # ⚡ Transpose: [n_process, vocab_size]
        neuron_profiles = token_neuron_prob[:, layer_idx, :].T.cpu().numpy()

        # Reduce dimensionality first for better clustering
        print("\n  Computing PCA...")
        pca = PCA(n_components=min(50, self.n_process))
        neuron_pca = pca.fit_transform(neuron_profiles)

        # K-means clustering
        print(f"  K-means clustering (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(neuron_pca)

        # Analyze clusters
        print("\n  Cluster analysis:")
        for cluster_id in range(n_clusters):
            neuron_ids = np.where(cluster_labels == cluster_id)[0]

            # Get representative tokens for this cluster
            cluster_profile = neuron_profiles[neuron_ids].mean(axis=0)
            top_token_ids = np.argsort(cluster_profile)[-10:][::-1]
            top_tokens = [self.tokenizer.decode([tid]) for tid in top_token_ids]

            # Categorize by POS
            pos_counts = defaultdict(int)
            for t in top_tokens:
                pos_counts[simple_pos_tag(t.strip())] += 1
            dominant_pos = max(pos_counts.items(), key=lambda x: x[1])[0]

            results['clusters'][f'cluster_{cluster_id}'] = {
                'neurons': neuron_ids.tolist(),
                'n_neurons': len(neuron_ids),
                'representative_tokens': top_tokens,
                'dominant_pos': dominant_pos
            }

            neurons_str = ', '.join(map(str, neuron_ids[:5]))
            tokens_str = ', '.join([f"'{t.strip()}'" for t in top_tokens[:3]])
            print(f"    Cluster {cluster_id} ({dominant_pos:5s}): neurons [{neurons_str}...], tokens [{tokens_str}]")

        return results, neuron_pca, cluster_labels

    def create_visualizations(self, token_neuron_prob, neuron_pca=None, cluster_labels=None, output_dir='./'):
        """Create visualization plots"""
        if not HAS_MATPLOTLIB:
            print("\n  Skipping visualizations (matplotlib not available)")
            return

        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)

        # 1. Token-Neuron heatmap (top tokens)
        print("  Creating token-neuron heatmap...")
        top_tokens = torch.topk(self.token_count, k=50)[1]

        fig, axes = plt.subplots(1, self.n_layers, figsize=(4 * self.n_layers, 12))
        if self.n_layers == 1:
            axes = [axes]

        for layer_idx, ax in enumerate(axes):
            data = token_neuron_prob[top_tokens, layer_idx, :].cpu().numpy()

            im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
            ax.set_xlabel('Neuron')
            ax.set_ylabel('Token')
            ax.set_title(f'Layer {layer_idx}')

            # Token labels
            token_labels = [self.tokenizer.decode([tid]).strip()[:10] for tid in top_tokens.tolist()]
            ax.set_yticks(range(len(token_labels)))
            ax.set_yticklabels(token_labels, fontsize=6)

        plt.colorbar(im, ax=axes[-1])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'token_neuron_heatmap.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: token_neuron_heatmap.png")

        # 2. Neuron clustering t-SNE
        if neuron_pca is not None and HAS_SKLEARN:
            print("  Creating t-SNE visualization...")

            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, self.n_process - 1))
            neuron_tsne = tsne.fit_transform(neuron_pca)

            fig, ax = plt.subplots(figsize=(10, 8))

            if cluster_labels is not None:
                scatter = ax.scatter(neuron_tsne[:, 0], neuron_tsne[:, 1],
                                    c=cluster_labels, cmap='tab10', s=100)
                plt.colorbar(scatter, label='Cluster')
            else:
                ax.scatter(neuron_tsne[:, 0], neuron_tsne[:, 1], s=100)

            # Label neurons
            for i in range(self.n_process):
                ax.annotate(str(i), (neuron_tsne[i, 0], neuron_tsne[i, 1]), fontsize=8)

            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title('Neuron Clustering (Layer 0)')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'neuron_tsne.png'), dpi=150)
            if IN_NOTEBOOK:
                plt.show()
            else:
                plt.close()
            print(f"    Saved: neuron_tsne.png")

        # 3. Position-based activation
        print("  Creating position pattern plot...")

        pos_count_expanded = self.pos_count.unsqueeze(1).unsqueeze(2)
        pos_neuron_prob = self.pos_neuron_count / (pos_count_expanded * self.process_k + 1e-10)

        fig, axes = plt.subplots(1, self.n_layers, figsize=(4 * self.n_layers, 6))
        if self.n_layers == 1:
            axes = [axes]

        bin_names = ['0-4', '5-9', '10-19', '20-49', '50-99', '100+']

        for layer_idx, ax in enumerate(axes):
            data = pos_neuron_prob[:, layer_idx, :].cpu().numpy()

            im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
            ax.set_xlabel('Neuron')
            ax.set_ylabel('Position Bin')
            ax.set_title(f'Layer {layer_idx}')
            ax.set_yticks(range(len(bin_names)))
            ax.set_yticklabels(bin_names)

        plt.colorbar(im, ax=axes[-1])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'position_neuron_pattern.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: position_neuron_pattern.png")

        # 4. Layer specialization comparison
        print("  Creating layer comparison...")

        layer_avg_prob = token_neuron_prob.mean(dim=0).cpu().numpy()  # [n_layers, n_process]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(self.n_process)
        width = 0.8 / self.n_layers

        for layer_idx in range(self.n_layers):
            ax.bar(x + layer_idx * width, layer_avg_prob[layer_idx], width,
                   label=f'Layer {layer_idx}', alpha=0.8)

        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Average Activation Probability')
        ax.set_title('Neuron Usage by Layer')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_neuron_usage.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: layer_neuron_usage.png")

        print(f"\n  All visualizations saved to: {output_dir}")

    def export_csv(self, results, output_dir='./'):
        """Export results to CSV files"""
        print("\n" + "=" * 60)
        print("EXPORTING CSV FILES")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)

        # 1. Token-Neuron mapping
        csv_path = os.path.join(output_dir, 'token_neuron_mapping.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['token', 'token_id', 'count', 'layer', 'top_neurons', 'probs'])

            if 'token_neuron' in results and 'tokens' in results['token_neuron']:
                for token_str, data in results['token_neuron']['tokens'].items():
                    for layer_key, layer_data in data.get('layer_neurons', {}).items():
                        writer.writerow([
                            token_str, data['token_id'], data['count'],
                            layer_key,
                            str(layer_data['neurons']),
                            str([f"{p:.4f}" for p in layer_data['probs']])
                        ])
        print(f"  Saved: {csv_path}")

        # 2. Neuron profiles
        csv_path = os.path.join(output_dir, 'neuron_profiles.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['neuron_id', 'layer', 'dominant_pos', 'top_tokens'])

            if 'neuron_token' in results and 'neurons' in results['neuron_token']:
                for neuron_key, data in results['neuron_token']['neurons'].items():
                    writer.writerow([
                        neuron_key, 0, data['dominant_pos'],
                        ', '.join(data['top_tokens'][:10])
                    ])
        print(f"  Saved: {csv_path}")

        # 3. POS analysis
        csv_path = os.path.join(output_dir, 'pos_neuron_analysis.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['pos_tag', 'count', 'top_neurons_L0', 'probs'])

            if 'pos_neurons' in results:
                for pos, data in results['pos_neurons'].get('pos_neurons', {}).items():
                    writer.writerow([
                        pos, data['count'],
                        str(data['top_neurons_L0']),
                        str([f"{p:.4f}" for p in data['probs']])
                    ])
        print(f"  Saved: {csv_path}")

        # 4. Cluster analysis
        if 'clusters' in results:
            csv_path = os.path.join(output_dir, 'neuron_clusters.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['cluster_id', 'n_neurons', 'neurons', 'dominant_pos', 'representative_tokens'])

                for cluster_key, data in results['clusters'].get('clusters', {}).items():
                    writer.writerow([
                        cluster_key, data['n_neurons'],
                        str(data['neurons']),
                        data['dominant_pos'],
                        ', '.join(data['representative_tokens'][:5])
                    ])
            print(f"  Saved: {csv_path}")

    def generate_report(self, results):
        """Generate interesting findings report"""
        print("\n" + "=" * 60)
        print("INTERESTING FINDINGS")
        print("=" * 60)

        findings = []

        # 1. Specialized neurons
        if 'neuron_token' in results and 'neurons' in results['neuron_token']:
            pos_specialists = defaultdict(list)
            for neuron_key, data in results['neuron_token']['neurons'].items():
                pos = data['dominant_pos']
                pos_specialists[pos].append(neuron_key)

            for pos, neurons in pos_specialists.items():
                if len(neurons) >= 2:
                    finding = f"Neurons {neurons[:3]} specialize in {pos} tokens"
                    findings.append(finding)
                    print(f"  - {finding}")

        # 2. Position sensitivity
        if 'position' in results:
            pos_data = results['position'].get('position_bins', {})
            if pos_data:
                start_neurons = set(pos_data.get('0-4', {}).get('layer_0', {}).get('neurons', [])[:3])
                end_neurons = set(pos_data.get('100+', {}).get('layer_0', {}).get('neurons', [])[:3])

                if start_neurons != end_neurons:
                    finding = f"Position-sensitive: start={list(start_neurons)}, end={list(end_neurons)}"
                    findings.append(finding)
                    print(f"  - {finding}")

        # 3. Layer differences
        if 'layer_spec' in results and 'layers' in results['layer_spec']:
            layers = results['layer_spec']['layers']
            if len(layers) >= 2:
                l0_entropy = layers.get('layer_0', {}).get('entropy', 0)
                l_last = list(layers.values())[-1]
                l_last_entropy = l_last.get('entropy', 0)

                if abs(l0_entropy - l_last_entropy) > 0.1:
                    direction = "more uniform" if l_last_entropy > l0_entropy else "more concentrated"
                    finding = f"Layer progression: deeper layers are {direction} in neuron usage"
                    findings.append(finding)
                    print(f"  - {finding}")

        # 4. Context effects
        if 'context' in results and 'context_effects' in results['context']:
            effects = results['context']['context_effects']
            if effects:
                top_trigger = effects[0]['trigger_token']
                finding = f"Most context-sensitive trigger: '{top_trigger}'"
                findings.append(finding)
                print(f"  - {finding}")

        if not findings:
            print("  No significant patterns found (try with more data)")

        return findings


def main():
    parser = argparse.ArgumentParser(description='DAWN v7.9 Semantic Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--val_data', type=str,
                        default='/content/drive/MyDrive/data/validation/wikitext_5to1_texts.pkl',
                        help='Path to validation data')
    parser.add_argument('--max_batches', type=int, default=100,
                        help='Max batches for analysis')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./analysis_v79_semantic',
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

    analyzer = SemanticAnalyzer(model, tokenizer, device, version=version)

    # 1. Collect data
    analyzer.collect_data(dataloader, max_batches=args.max_batches)

    # Store all results
    all_results = {}

    # 2. Token -> Neuron mapping
    token_results, token_neuron_prob = analyzer.analyze_token_neuron_mapping()
    all_results['token_neuron'] = token_results

    # 3. Neuron -> Token preferences
    neuron_results, neuron_token_prob = analyzer.analyze_neuron_token_preferences(token_neuron_prob)
    all_results['neuron_token'] = neuron_results

    # 4. Position patterns
    pos_results, _ = analyzer.analyze_position_patterns()
    all_results['position'] = pos_results

    # 5. Context dependency
    context_results = analyzer.analyze_context_dependency(token_neuron_prob)
    all_results['context'] = context_results

    # 6. POS analysis
    pos_neuron_results = analyzer.analyze_pos_neuron_mapping(token_neuron_prob)
    all_results['pos_neurons'] = pos_neuron_results

    # 7. Layer specialization
    layer_results = analyzer.analyze_layer_specialization(token_neuron_prob)
    all_results['layer_spec'] = layer_results

    # 8. Neuron clustering
    neuron_pca, cluster_labels = None, None
    if HAS_SKLEARN:
        cluster_results, neuron_pca, cluster_labels = analyzer.cluster_neurons(token_neuron_prob)
        all_results['clusters'] = cluster_results

    # 9. Create visualizations
    analyzer.create_visualizations(token_neuron_prob, neuron_pca, cluster_labels, args.output_dir)

    # 10. Export CSV
    analyzer.export_csv(all_results, args.output_dir)

    # 11. Generate report
    findings = analyzer.generate_report(all_results)

    # Save JSON results
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
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    json_path = os.path.join(args.output_dir, 'semantic_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\n  Full results saved to: {json_path}")

    print("\n" + "=" * 60)
    print("SEMANTIC ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
