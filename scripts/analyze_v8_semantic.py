"""
DAWN v8.x Semantic Analysis Script
뉴런이 무엇을 배웠는지 분석

핵심 질문:
- 각 뉴런이 어떤 토큰/패턴에 반응하는가?
- Q/K/V별로 뉴런 선택이 어떻게 다른가?
- Knowledge neurons은 어떤 토큰에 활성화되는가?

분석 항목:
1. 토큰별 뉴런 활성화 패턴 (Q/K/V 각각)
2. 뉴런별 선호 토큰
3. 위치별 뉴런 활성화
4. 문맥 의존성 분석
5. 뉴런 클러스터링
6. Layer별 라우터 역할 분화
7. Q/K/V/O별 뉴런 역할 차이
8. Knowledge neurons 분석
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
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, some visualizations disabled")


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
    if token_lower in {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'}:
        return 'AUX'
    if token_lower in {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}:
        return 'PRON'
    if token_lower in {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about'}:
        return 'ADP'
    if token_lower in {'and', 'or', 'but', 'if', 'when', 'while', 'because', 'although'}:
        return 'CONJ'
    if token_lower.isdigit():
        return 'NUM'
    if token_lower.endswith('ing') or token_lower.endswith('ed'):
        return 'VERB'
    if token_lower.endswith('ly'):
        return 'ADV'
    if token_lower.endswith('ful') or token_lower.endswith('less') or token_lower.endswith('ive'):
        return 'ADJ'
    return 'NOUN'


class SemanticAnalyzerV8:
    """GPU-optimized semantic analysis for DAWN v8.0"""

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

        # ⚡ GPU tensors for accumulation
        # Token -> Neuron mapping per component: [vocab_size, n_layers, n_process]
        self.components = ['Q', 'K', 'V', 'O']
        self.token_neuron_count = {
            comp: torch.zeros(self.vocab_size, self.n_layers, self.n_process,
                              device=device, dtype=torch.float32)
            for comp in self.components
        }
        self.token_count = torch.zeros(self.vocab_size, device=device, dtype=torch.float32)

        # Token -> Knowledge mapping: [vocab_size, n_layers, n_knowledge]
        self.token_knowledge_count = torch.zeros(
            self.vocab_size, self.n_layers, self.n_knowledge,
            device=device, dtype=torch.float32
        )

        # Position bins
        self.pos_bins = [0, 4, 8, 16, 32, 64, 128]
        self.n_pos_bins = len(self.pos_bins) - 1

        # Position -> Neuron: [n_pos_bins, n_layers, n_process]
        self.pos_neuron_count = {
            comp: torch.zeros(self.n_pos_bins, self.n_layers, self.n_process,
                              device=device, dtype=torch.float32)
            for comp in self.components
        }
        self.pos_count = torch.zeros(self.n_pos_bins, device=device, dtype=torch.float32)

    @torch.no_grad()
    def collect_data(self, dataloader, max_batches=100):
        """Collect token-neuron activation data"""
        print("\n" + "=" * 60)
        print("COLLECTING SEMANTIC DATA (v8.0)")
        print("=" * 60)

        self.model.eval()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # Token counts
            token_flat = input_ids.reshape(-1)  # [B*S]
            self.token_count.scatter_add_(0, token_flat, torch.ones_like(token_flat, dtype=torch.float32))

            # Position bins
            positions = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)
            pos_flat = positions.reshape(-1)
            pos_bins_tensor = torch.tensor(self.pos_bins[1:], device=self.device)
            pos_bin_idx = torch.bucketize(pos_flat, pos_bins_tensor)

            for layer_idx, routing_info in enumerate(routing_infos):
                attn_routing = routing_info['attention']
                mem_routing = routing_info['memory']

                # Process Q/K/V/O routing
                for comp in self.components:
                    if comp == 'O':
                        routing = attn_routing['routing_O']
                    else:
                        routing = attn_routing[f'routing_{comp}']

                    process_idx = routing['process_indices']  # [B, S, k]
                    idx_flat = process_idx.reshape(-1)  # [B*S*k]
                    token_expanded = token_flat.unsqueeze(1).expand(-1, self.process_k).reshape(-1)

                    # Token-Neuron mapping
                    combined_idx = token_expanded * (self.n_layers * self.n_process) + \
                                   layer_idx * self.n_process + idx_flat
                    self.token_neuron_count[comp].view(-1).scatter_add_(
                        0, combined_idx, torch.ones_like(combined_idx, dtype=torch.float32)
                    )

                    # Position-Neuron mapping
                    pos_expanded = pos_bin_idx.unsqueeze(1).expand(-1, self.process_k).reshape(-1)
                    pos_combined_idx = pos_expanded * (self.n_layers * self.n_process) + \
                                       layer_idx * self.n_process + idx_flat
                    self.pos_neuron_count[comp].view(-1).scatter_add_(
                        0, pos_combined_idx, torch.ones_like(pos_combined_idx, dtype=torch.float32)
                    )

                # Knowledge routing
                k_idx = mem_routing['knowledge_indices']  # [B, S, knowledge_k]
                k_flat = k_idx.reshape(-1)
                token_expanded_k = token_flat.unsqueeze(1).expand(-1, self.knowledge_k).reshape(-1)

                combined_idx_k = token_expanded_k * (self.n_layers * self.n_knowledge) + \
                                 layer_idx * self.n_knowledge + k_flat
                self.token_knowledge_count.view(-1).scatter_add_(
                    0, combined_idx_k, torch.ones_like(combined_idx_k, dtype=torch.float32)
                )

            # Position counts
            for p in range(self.n_pos_bins):
                self.pos_count[p] += (pos_bin_idx == p).float().sum()

        print(f"\n  Total tokens processed: {int(self.token_count.sum().item()):,}")
        print(f"  Unique tokens seen: {(self.token_count > 0).sum().item():,}")

    def analyze_token_neuron_mapping(self, component='Q', top_k=20):
        """Analyze which neurons are activated for each token"""
        print("\n" + "=" * 60)
        print(f"1. TOKEN -> NEURON MAPPING ({component})")
        print("=" * 60)

        token_count_expanded = self.token_count.unsqueeze(1).unsqueeze(2)
        token_neuron_prob = self.token_neuron_count[component] / (token_count_expanded * self.process_k + 1e-10)

        # Find tokens with enough samples
        min_count = 100
        valid_tokens = self.token_count >= min_count
        n_valid = valid_tokens.sum().item()
        print(f"  Tokens with >= {min_count} occurrences: {n_valid}")

        # High-frequency tokens
        top_tokens_idx = torch.argsort(self.token_count, descending=True)[:top_k]

        results = {'tokens': {}}

        print(f"\n📌 Top {top_k} Most Frequent Tokens ({component} routing):")
        for i, tid in enumerate(top_tokens_idx.tolist()):
            token = self.tokenizer.decode([tid]).strip()
            count = int(self.token_count[tid].item())
            probs = token_neuron_prob[tid]  # [n_layers, n_process]

            # Top neurons per layer
            layer0_top = torch.topk(probs[0], 5)
            layer0_neurons = layer0_top.indices.tolist()
            layer0_probs = layer0_top.values.tolist()

            results['tokens'][token] = {
                'id': tid,
                'count': count,
                'top_neurons_L0': layer0_neurons,
                'probs_L0': layer0_probs
            }

            neurons_str = ', '.join([f'{n}({p:.2f})' for n, p in zip(layer0_neurons[:3], layer0_probs[:3])])
            print(f"  {i+1:2d}. '{token}' (n={count:,}): L0=[{neurons_str}]")

        return results, token_neuron_prob

    def analyze_neuron_preferences(self, component='Q', top_k=10):
        """Analyze which tokens each neuron prefers"""
        print("\n" + "=" * 60)
        print(f"2. NEURON -> TOKEN PREFERENCES ({component})")
        print("=" * 60)

        # Aggregate across all layers
        neuron_token_count = self.token_neuron_count[component].sum(dim=1)  # [vocab_size, n_process]
        neuron_token_count = neuron_token_count.T  # [n_process, vocab_size]

        # Normalize to probabilities
        neuron_total = neuron_token_count.sum(dim=1, keepdim=True) + 1e-10
        neuron_token_prob = neuron_token_count / neuron_total

        results = {'neurons': {}}

        print(f"\n📌 Process Neuron Preferences ({component}):")
        for neuron_idx in range(min(self.n_process, 16)):  # Show first 16 neurons
            probs = neuron_token_prob[neuron_idx]
            top = torch.topk(probs, top_k)

            top_tokens = [self.tokenizer.decode([tid]).strip() for tid in top.indices.tolist()]
            top_probs = top.values.tolist()

            results['neurons'][neuron_idx] = {
                'top_tokens': top_tokens,
                'probs': top_probs
            }

            tokens_str = ', '.join([f"'{t}'" for t in top_tokens[:5]])
            print(f"  Neuron {neuron_idx:2d}: {tokens_str}")

        return results

    def analyze_knowledge_patterns(self, top_k=10):
        """Analyze which tokens activate which knowledge neurons"""
        print("\n" + "=" * 60)
        print("3. KNOWLEDGE NEURON ANALYSIS")
        print("=" * 60)

        # Knowledge neuron -> Token preferences
        knowledge_token_count = self.token_knowledge_count.sum(dim=1)  # [vocab_size, n_knowledge]
        knowledge_token_count = knowledge_token_count.T  # [n_knowledge, vocab_size]

        knowledge_total = knowledge_token_count.sum(dim=1, keepdim=True) + 1e-10
        knowledge_token_prob = knowledge_token_count / knowledge_total

        results = {'knowledge_neurons': {}}

        print(f"\n📌 Knowledge Neuron Token Preferences:")
        for k_idx in range(min(self.n_knowledge, 16)):
            probs = knowledge_token_prob[k_idx]
            top = torch.topk(probs, top_k)

            top_tokens = [self.tokenizer.decode([tid]).strip() for tid in top.indices.tolist()]
            top_probs = top.values.tolist()

            results['knowledge_neurons'][k_idx] = {
                'top_tokens': top_tokens,
                'probs': top_probs
            }

            tokens_str = ', '.join([f"'{t}'" for t in top_tokens[:5]])
            print(f"  Knowledge {k_idx:2d}: {tokens_str}")

        # Token -> Knowledge preferences
        print(f"\n📌 Token -> Knowledge Mapping (Top 20 tokens):")
        token_count_expanded = self.token_count.unsqueeze(1).unsqueeze(2)
        token_knowledge_prob = self.token_knowledge_count / (token_count_expanded * self.knowledge_k + 1e-10)

        top_tokens = torch.argsort(self.token_count, descending=True)[:20]
        for tid in top_tokens.tolist():
            token = self.tokenizer.decode([tid]).strip()
            probs = token_knowledge_prob[tid]  # [n_layers, n_knowledge]
            layer0_top = torch.topk(probs[0], 3)

            k_neurons = layer0_top.indices.tolist()
            print(f"  '{token}': Knowledge neurons {k_neurons}")

        return results

    def analyze_qkv_differences(self):
        """Analyze how Q/K/V routing differs"""
        print("\n" + "=" * 60)
        print("4. Q/K/V/O ROUTING DIFFERENCES")
        print("=" * 60)

        results = {}

        # Aggregate counts per component
        comp_totals = {}
        for comp in self.components:
            comp_totals[comp] = self.token_neuron_count[comp].sum(dim=0)  # [n_layers, n_process]

        # Compare Q vs K vs V
        print("\n📌 Neuron Usage Correlation Between Components:")
        for layer_idx in range(self.n_layers):
            correlations = {}
            for i, c1 in enumerate(self.components):
                for c2 in self.components[i+1:]:
                    v1 = comp_totals[c1][layer_idx]
                    v2 = comp_totals[c2][layer_idx]
                    v1_norm = F.normalize(v1.unsqueeze(0), dim=-1)
                    v2_norm = F.normalize(v2.unsqueeze(0), dim=-1)
                    corr = (v1_norm @ v2_norm.T).item()
                    correlations[f'{c1}-{c2}'] = corr

            print(f"  Layer {layer_idx}: Q-K={correlations['Q-K']:.3f}, Q-V={correlations['Q-V']:.3f}, "
                  f"K-V={correlations['K-V']:.3f}, Q-O={correlations['Q-O']:.3f}")

            results[f'layer_{layer_idx}'] = correlations

        # Component-specific top neurons
        print("\n📌 Most Used Neurons Per Component (Layer 0):")
        for comp in self.components:
            usage = comp_totals[comp][0]  # Layer 0
            top = torch.topk(usage, 5)
            print(f"  {comp}: neurons {top.indices.tolist()}")

        return results

    def analyze_pos_patterns(self):
        """Analyze position-based activation patterns"""
        print("\n" + "=" * 60)
        print("5. POSITION-BASED PATTERNS (Q routing)")
        print("=" * 60)

        results = {}

        bin_names = ['0-3', '4-7', '8-15', '16-31', '32-63', '64-127']

        pos_count_expanded = self.pos_count.unsqueeze(1).unsqueeze(2)
        pos_neuron_prob = self.pos_neuron_count['Q'] / (pos_count_expanded * self.process_k + 1e-10)

        print(f"\n📌 Position -> Top Neurons (Layer 0):")
        for pos_idx, bin_name in enumerate(bin_names):
            probs = pos_neuron_prob[pos_idx, 0]  # [n_process]
            top = torch.topk(probs, 5)

            results[bin_name] = {
                'count': int(self.pos_count[pos_idx].item()),
                'top_neurons': top.indices.tolist(),
                'probs': top.values.tolist()
            }

            neurons = top.indices.tolist()[:3]
            print(f"  Pos {bin_name:8s} (n={int(self.pos_count[pos_idx].item()):6d}): neurons {neurons}")

        return results

    def analyze_neuron_clustering(self, component='Q'):
        """Cluster neurons based on their token preferences"""
        print("\n" + "=" * 60)
        print(f"6. NEURON CLUSTERING ({component})")
        print("=" * 60)

        if not HAS_SKLEARN:
            print("  sklearn not available, skipping clustering")
            return None

        # Get neuron-token probability matrix
        neuron_token_count = self.token_neuron_count[component].sum(dim=1).T  # [n_process, vocab_size]
        neuron_total = neuron_token_count.sum(dim=1, keepdim=True) + 1e-10
        neuron_token_prob = neuron_token_count / neuron_total

        # PCA for dimensionality reduction
        neuron_features = neuron_token_prob.cpu().numpy()
        pca = PCA(n_components=min(50, self.n_process))
        neuron_pca = pca.fit_transform(neuron_features)

        print(f"  PCA explained variance: {pca.explained_variance_ratio_[:5].sum():.2%} (top 5 components)")

        # K-means clustering
        n_clusters = min(8, self.n_process // 4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(neuron_pca)

        results = {'clusters': {}}

        print(f"\n📌 Neuron Clusters (k={n_clusters}):")
        for c in range(n_clusters):
            members = np.where(clusters == c)[0].tolist()
            results['clusters'][c] = members
            print(f"  Cluster {c}: neurons {members[:10]}{'...' if len(members) > 10 else ''}")

        return results, neuron_pca, clusters

    def analyze_layer_specialization(self, component='Q'):
        """Analyze how layers differ in routing"""
        print("\n" + "=" * 60)
        print(f"7. LAYER SPECIALIZATION ({component})")
        print("=" * 60)

        # Per-layer neuron usage
        layer_usage = self.token_neuron_count[component].sum(dim=0)  # [n_layers, n_process]
        layer_total = layer_usage.sum(dim=1, keepdim=True) + 1e-10
        layer_prob = layer_usage / layer_total

        results = {'layers': {}}

        print("\n📌 Neuron Usage by Layer:")
        for layer_idx in range(self.n_layers):
            usage = layer_prob[layer_idx]

            # Entropy
            entropy = -(usage * torch.log(usage + 1e-10)).sum()
            max_entropy = math.log(self.n_process)
            norm_entropy = entropy / max_entropy

            # Gini
            sorted_usage, _ = torch.sort(usage)
            n = self.n_process
            index = torch.arange(1, n + 1, dtype=torch.float32, device=self.device)
            gini = ((2 * index - n - 1) * sorted_usage).sum() / (n * sorted_usage.sum() + 1e-10)

            # Top neurons
            top = torch.topk(usage, 5)

            results['layers'][layer_idx] = {
                'entropy': norm_entropy.item(),
                'gini': gini.item(),
                'top_neurons': top.indices.tolist()
            }

            print(f"  Layer {layer_idx}: entropy={norm_entropy.item():.3f}, gini={gini.item():.3f}, "
                  f"top={top.indices.tolist()[:3]}")

        # Cross-layer correlation
        print("\n📌 Cross-Layer Usage Correlation:")
        layer_prob_norm = F.normalize(layer_prob, dim=-1)
        corr = layer_prob_norm @ layer_prob_norm.T
        for i in range(self.n_layers):
            for j in range(i+1, self.n_layers):
                print(f"  L{i}-L{j}: {corr[i, j].item():.3f}")

        return results

    def analyze_pos_tagging(self):
        """Analyze neuron activation by POS tag"""
        print("\n" + "=" * 60)
        print("8. POS TAG ANALYSIS (Q routing)")
        print("=" * 60)

        # Categorize tokens by POS
        pos_neuron_counts = defaultdict(lambda: torch.zeros(self.n_layers, self.n_process, device=self.device))
        pos_counts = defaultdict(float)

        for tid in range(self.vocab_size):
            if self.token_count[tid] < 10:
                continue

            token = self.tokenizer.decode([tid]).strip()
            pos = simple_pos_tag(token)

            pos_neuron_counts[pos] += self.token_neuron_count['Q'][tid]
            pos_counts[pos] += self.token_count[tid].item()

        results = {}

        print("\n📌 POS -> Top Neurons (Layer 0):")
        for pos in sorted(pos_counts.keys()):
            if pos_counts[pos] < 100:
                continue

            probs = pos_neuron_counts[pos][0] / (pos_counts[pos] * self.process_k + 1e-10)
            top = torch.topk(probs, 3)

            results[pos] = {
                'count': int(pos_counts[pos]),
                'top_neurons': top.indices.tolist()
            }

            print(f"  {pos:6s} (n={int(pos_counts[pos]):6d}): neurons {top.indices.tolist()}")

        return results

    def visualize(self, token_neuron_prob, neuron_pca, clusters, output_dir):
        """Create visualizations"""
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)

        if not HAS_MATPLOTLIB:
            print("  matplotlib not available")
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. Token-Neuron heatmap (Q component)
        print("  Creating token-neuron heatmap...")
        top_k = min(50, (self.token_count > 100).sum().item())
        top_tokens = torch.argsort(self.token_count, descending=True)[:top_k]

        fig, axes = plt.subplots(1, self.n_layers, figsize=(4 * self.n_layers, 10))
        if self.n_layers == 1:
            axes = [axes]

        for layer_idx in range(self.n_layers):
            probs = token_neuron_prob[top_tokens, layer_idx].cpu().numpy()
            im = axes[layer_idx].imshow(probs, aspect='auto', cmap='YlOrRd')
            axes[layer_idx].set_xlabel('Neuron Index')
            axes[layer_idx].set_ylabel('Token')
            axes[layer_idx].set_title(f'Layer {layer_idx}')

            if layer_idx == 0:
                token_labels = [self.tokenizer.decode([tid]).strip()[:10] for tid in top_tokens.tolist()]
                axes[layer_idx].set_yticks(range(len(token_labels)))
                axes[layer_idx].set_yticklabels(token_labels, fontsize=6)

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
            scatter = ax.scatter(neuron_tsne[:, 0], neuron_tsne[:, 1],
                                 c=clusters, cmap='tab10', alpha=0.7, s=100)

            for i in range(self.n_process):
                ax.annotate(str(i), (neuron_tsne[i, 0], neuron_tsne[i, 1]), fontsize=8)

            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title('Process Neuron Clustering (Q routing)')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'neuron_tsne.png'), dpi=150)
            if IN_NOTEBOOK:
                plt.show()
            else:
                plt.close()
            print(f"    Saved: neuron_tsne.png")

        # 3. Q/K/V/O comparison
        print("  Creating Q/K/V/O comparison...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for idx, comp in enumerate(self.components):
            ax = axes[idx // 2, idx % 2]
            layer_usage = self.token_neuron_count[comp].sum(dim=0)  # [n_layers, n_process]
            layer_total = layer_usage.sum(dim=1, keepdim=True) + 1e-10
            layer_prob = (layer_usage / layer_total).cpu().numpy()

            for layer_idx in range(self.n_layers):
                ax.plot(layer_prob[layer_idx], label=f'Layer {layer_idx}', alpha=0.7)

            ax.set_xlabel('Neuron Index')
            ax.set_ylabel('Usage Probability')
            ax.set_title(f'{comp} Neuron Usage')
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'qkvo_comparison.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: qkvo_comparison.png")

        # 4. Knowledge neuron usage
        print("  Creating knowledge neuron visualization...")

        knowledge_usage = self.token_knowledge_count.sum(dim=0)  # [n_layers, n_knowledge]
        knowledge_total = knowledge_usage.sum(dim=1, keepdim=True) + 1e-10
        knowledge_prob = (knowledge_usage / knowledge_total).cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 6))
        for layer_idx in range(self.n_layers):
            ax.bar(np.arange(self.n_knowledge) + layer_idx * 0.2,
                   knowledge_prob[layer_idx], width=0.2, label=f'Layer {layer_idx}', alpha=0.7)

        ax.set_xlabel('Knowledge Neuron Index')
        ax.set_ylabel('Usage Probability')
        ax.set_title('Knowledge Neuron Usage by Layer')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'knowledge_usage.png'), dpi=150)
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()
        print(f"    Saved: knowledge_usage.png")

        print(f"\n  All visualizations saved to: {output_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN v8.x Semantic Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--val_data', type=str,
                        default='/content/drive/MyDrive/data/validation/wikitext_5to1_texts.pkl',
                        help='Path to validation data')
    parser.add_argument('--max_batches', type=int, default=100,
                        help='Max batches for analysis')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./analysis_v8_semantic',
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

    analyzer = SemanticAnalyzerV8(model, tokenizer, device)

    # 1. Collect data
    analyzer.collect_data(dataloader, max_batches=args.max_batches)

    all_results = {}

    # 2. Token -> Neuron mapping (Q)
    token_results, token_neuron_prob = analyzer.analyze_token_neuron_mapping('Q')
    all_results['token_neuron_Q'] = token_results

    # 3. Neuron -> Token preferences
    all_results['neuron_preferences_Q'] = analyzer.analyze_neuron_preferences('Q')

    # 4. Knowledge analysis
    all_results['knowledge'] = analyzer.analyze_knowledge_patterns()

    # 5. Q/K/V/O differences
    all_results['qkvo_diff'] = analyzer.analyze_qkv_differences()

    # 6. Position patterns
    all_results['position'] = analyzer.analyze_pos_patterns()

    # 7. Neuron clustering
    cluster_results = analyzer.analyze_neuron_clustering('Q')
    if cluster_results:
        all_results['clustering'], neuron_pca, clusters = cluster_results
    else:
        neuron_pca, clusters = None, None

    # 8. Layer specialization
    all_results['layer_spec'] = analyzer.analyze_layer_specialization('Q')

    # 9. POS tag analysis
    all_results['pos'] = analyzer.analyze_pos_tagging()

    # Visualizations
    analyzer.visualize(token_neuron_prob, neuron_pca, clusters, args.output_dir)

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

    results_path = os.path.join(args.output_dir, 'semantic_results.json')
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("SEMANTIC ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
