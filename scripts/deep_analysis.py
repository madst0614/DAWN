#!/usr/bin/env python3
"""
DAWN Deep Analysis Script
========================
1. Sentence-level visualization
2. Actual ablation experiments
3. Semantic neuron analysis
4. Neuron function catalog
5. Advanced visualizations

Usage:
    python scripts/deep_analysis.py \
        --checkpoint /path/to/checkpoint.pt \
        --data /path/to/data.pkl \
        --output_dir ./dawn_analysis
"""

import os
import sys
import json
import pickle
import argparse
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    print("Warning: spacy not available, POS/NER tagging disabled")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ============================================================
# Data Classes
# ============================================================

@dataclass
class NeuronProfile:
    """Profile for a single neuron"""
    neuron_id: int
    top_tokens: List[Tuple[str, int]]
    primary_pos: str
    pos_distribution: Dict[str, float]
    layer_usage: List[float]
    co_occurring_neurons: List[int]
    semantic_category: str
    ablation_impact: Dict[str, float]


# ============================================================
# 1. Sentence Visualization
# ============================================================

class SentenceVisualizer:
    """Token-neuron mapping and layer progression visualization"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        if HAS_SPACY:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                self.nlp = None
        else:
            self.nlp = None

    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """Get POS tags for text"""
        if self.nlp is None:
            return [(t, 'UNK') for t in text.split()]

        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]

    def get_ner_tags(self, text: str) -> List[Tuple[str, str, str]]:
        """Get NER tags for text"""
        if self.nlp is None:
            return []

        doc = self.nlp(text)
        return [(ent.text, ent.start_char, ent.label_) for ent in doc.ents]

    @torch.no_grad()
    def analyze_sentence(self, text: str) -> Dict:
        """Analyze a single sentence through all layers"""
        self.model.eval()

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor([tokens], device=self.device)

        # Get model output with routing info
        _, routing_infos = self.model(input_ids, return_routing_info=True)

        # Decode tokens for display
        token_strs = [self.tokenizer.decode([t]).strip() for t in tokens]

        # Get POS tags (approximate alignment)
        pos_tags = self.get_pos_tags(text)

        # Collect per-layer neuron activations
        layer_data = []
        n_layers = len(routing_infos)

        for layer_idx, routing_info in enumerate(routing_infos):
            layer_result = {
                'layer': layer_idx,
                'Q': {}, 'K': {}, 'V': {}, 'M': {}
            }

            for comp in ['Q', 'K', 'V', 'M']:
                comp_key = f'compress_{comp}'
                if comp_key in routing_info:
                    data = routing_info[comp_key]
                    weights = data['weights'][0]  # [S, k]

                    if 'indices' in data:
                        indices = data['indices'][0]  # [S, k]
                    else:
                        k = min(8, weights.shape[-1])
                        _, indices = torch.topk(weights, k, dim=-1)

                    layer_result[comp] = {
                        'indices': indices.cpu().numpy(),
                        'weights': weights.cpu().numpy()
                    }

            layer_data.append(layer_result)

        return {
            'text': text,
            'tokens': token_strs,
            'token_ids': tokens,
            'pos_tags': pos_tags,
            'layer_data': layer_data,
            'n_layers': n_layers
        }

    def visualize_sentence(self, analysis: Dict, output_path: str, show_layers: List[int] = None):
        """Create token-neuron heatmap for a sentence"""
        if not HAS_MATPLOTLIB:
            print("matplotlib required for visualization")
            return

        tokens = analysis['tokens']
        n_tokens = len(tokens)
        layer_data = analysis['layer_data']
        n_layers = analysis['n_layers']

        if show_layers is None:
            show_layers = list(range(n_layers))

        # Create figure: one row per layer
        fig, axes = plt.subplots(len(show_layers), 4, figsize=(20, 3 * len(show_layers)))
        if len(show_layers) == 1:
            axes = axes.reshape(1, -1)

        comps = ['Q', 'K', 'V', 'M']

        for row_idx, layer_idx in enumerate(show_layers):
            layer = layer_data[layer_idx]

            for col_idx, comp in enumerate(comps):
                ax = axes[row_idx, col_idx]

                if comp in layer and layer[comp]:
                    indices = layer[comp]['indices']
                    weights = layer[comp]['weights']

                    # Create heatmap: tokens x top neurons
                    k = indices.shape[1]

                    # Get unique neurons used
                    unique_neurons = sorted(set(indices.flatten()))[:50]
                    neuron_to_idx = {n: i for i, n in enumerate(unique_neurons)}

                    heatmap = np.zeros((n_tokens, len(unique_neurons)))
                    for t in range(n_tokens):
                        for ki in range(k):
                            n_idx = indices[t, ki]
                            if n_idx in neuron_to_idx:
                                heatmap[t, neuron_to_idx[n_idx]] += weights[t, ki]

                    im = ax.imshow(heatmap.T, aspect='auto', cmap='YlOrRd')
                    ax.set_xticks(range(n_tokens))
                    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
                    ax.set_ylabel('Neuron')

                    if row_idx == 0:
                        ax.set_title(f'{comp}')
                    if col_idx == 0:
                        ax.set_ylabel(f'L{layer_idx}\nNeuron')
                else:
                    ax.axis('off')

        plt.suptitle(f'Token-Neuron Activation: "{analysis["text"][:50]}..."', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")

    def visualize_layer_progression(self, analysis: Dict, output_path: str, comp: str = 'Q'):
        """Visualize how neuron selection changes across layers"""
        if not HAS_MATPLOTLIB:
            return

        tokens = analysis['tokens']
        n_tokens = len(tokens)
        layer_data = analysis['layer_data']
        n_layers = analysis['n_layers']

        # Create 2x4 grid for 8 layers
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        for layer_idx in range(min(8, n_layers)):
            ax = axes[layer_idx // 4, layer_idx % 4]
            layer = layer_data[layer_idx]

            if comp in layer and layer[comp]:
                indices = layer[comp]['indices']
                weights = layer[comp]['weights']

                # Top-1 neuron for each token
                top1_neurons = indices[:, 0]
                top1_weights = weights[:, 0]

                # Create bar chart
                colors = plt.cm.tab20(top1_neurons % 20)
                bars = ax.bar(range(n_tokens), top1_weights, color=colors)

                ax.set_xticks(range(n_tokens))
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
                ax.set_title(f'Layer {layer_idx}')
                ax.set_ylabel('Weight')

                # Add neuron IDs as labels
                for i, (n, w) in enumerate(zip(top1_neurons, top1_weights)):
                    ax.text(i, w + 0.02, str(n), ha='center', fontsize=6)
            else:
                ax.axis('off')

        plt.suptitle(f'Layer Progression ({comp}): "{analysis["text"][:40]}..."', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")


# ============================================================
# 2. Ablation Experiments
# ============================================================

class AblationExperiment:
    """Actual ablation experiments with neuron removal"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        if HAS_SPACY:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                self.nlp = None
        else:
            self.nlp = None

    def get_token_pos(self, input_ids: torch.Tensor) -> List[List[str]]:
        """Get POS tags for batch of token sequences"""
        if self.nlp is None:
            return [['' for _ in range(seq.shape[0])] for seq in input_ids]

        batch_pos = []
        for seq in input_ids:
            text = self.tokenizer.decode(seq.tolist())
            doc = self.nlp(text)

            # Align tokens (approximate)
            pos_list = [token.pos_ for token in doc]
            # Pad/truncate to match sequence length
            if len(pos_list) < seq.shape[0]:
                pos_list.extend([''] * (seq.shape[0] - len(pos_list)))
            else:
                pos_list = pos_list[:seq.shape[0]]

            batch_pos.append(pos_list)

        return batch_pos

    @torch.no_grad()
    def compute_baseline_metrics(self, dataloader, max_batches: int = 50) -> Dict:
        """Compute baseline perplexity and per-POS accuracy"""
        self.model.eval()

        total_loss = 0.0
        n_tokens = 0
        pos_correct = defaultdict(int)
        pos_total = defaultdict(int)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Baseline", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Compute loss
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            loss, logits_tuple = self.model(input_ids, labels=labels)
            logits = logits_tuple[0] if isinstance(logits_tuple, tuple) else logits_tuple

            total_loss += loss.item() * B * S
            n_tokens += B * S

            # Per-POS accuracy
            predictions = logits[:, :-1].argmax(dim=-1)  # [B, S-1]
            targets = input_ids[:, 1:]  # [B, S-1]

            batch_pos = self.get_token_pos(input_ids)

            for b in range(B):
                for s in range(S - 1):
                    pos = batch_pos[b][s + 1] if s + 1 < len(batch_pos[b]) else ''
                    if pos:
                        pos_total[pos] += 1
                        if predictions[b, s] == targets[b, s]:
                            pos_correct[pos] += 1

        ppl = math.exp(total_loss / n_tokens)
        pos_accuracy = {pos: pos_correct[pos] / max(1, pos_total[pos])
                       for pos in pos_total}

        return {
            'ppl': ppl,
            'pos_accuracy': pos_accuracy,
            'pos_counts': dict(pos_total)
        }

    @torch.no_grad()
    def ablate_neuron(self, neuron_idx: int, dataloader, max_batches: int = 50) -> Dict:
        """
        Ablate a specific neuron and measure impact.
        Method: Mask out the neuron in router output (set weight to 0)
        """
        self.model.eval()

        # Store original forward function
        original_forwards = {}

        # Hook to mask neuron in routing
        def create_mask_hook(neuron_id):
            def hook(module, input, output):
                # output is routing weights [B, S, N] before top-k
                # Set weight of target neuron to -inf so it's never selected
                if hasattr(output, 'shape') and len(output.shape) == 3:
                    output[:, :, neuron_id] = -1e9
                return output
            return hook

        # Find router modules and add hooks
        hooks = []
        for name, module in self.model.named_modules():
            if 'router' in name.lower() and hasattr(module, 'forward'):
                # Try to find linear layer in router
                for child_name, child in module.named_children():
                    if isinstance(child, torch.nn.Linear):
                        hook = child.register_forward_hook(create_mask_hook(neuron_idx))
                        hooks.append(hook)

        # If no router hooks, try alternative: mask in shared_neurons
        if not hooks and hasattr(self.model, 'shared_neurons'):
            # Temporarily zero out neuron weights
            neurons = self.model.shared_neurons
            if hasattr(neurons, 'compress_neurons'):
                original_weight = neurons.compress_neurons.data[neuron_idx].clone()
                neurons.compress_neurons.data[neuron_idx] = 0

        # Compute metrics with ablation
        total_loss = 0.0
        n_tokens = 0
        pos_correct = defaultdict(int)
        pos_total = defaultdict(int)

        try:
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Ablate N{neuron_idx}", total=max_batches)):
                if batch_idx >= max_batches:
                    break

                input_ids = batch["input_ids"].to(self.device)
                B, S = input_ids.shape

                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = -100

                loss, logits_tuple = self.model(input_ids, labels=labels)
                logits = logits_tuple[0] if isinstance(logits_tuple, tuple) else logits_tuple

                total_loss += loss.item() * B * S
                n_tokens += B * S

                predictions = logits[:, :-1].argmax(dim=-1)
                targets = input_ids[:, 1:]

                batch_pos = self.get_token_pos(input_ids)

                for b in range(B):
                    for s in range(S - 1):
                        pos = batch_pos[b][s + 1] if s + 1 < len(batch_pos[b]) else ''
                        if pos:
                            pos_total[pos] += 1
                            if predictions[b, s] == targets[b, s]:
                                pos_correct[pos] += 1
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Restore neuron weights if modified
            if not hooks and hasattr(self.model, 'shared_neurons'):
                neurons = self.model.shared_neurons
                if hasattr(neurons, 'compress_neurons') and 'original_weight' in locals():
                    neurons.compress_neurons.data[neuron_idx] = original_weight

        ppl = math.exp(total_loss / n_tokens)
        pos_accuracy = {pos: pos_correct[pos] / max(1, pos_total[pos])
                       for pos in pos_total}

        return {
            'neuron_id': neuron_idx,
            'ppl': ppl,
            'pos_accuracy': pos_accuracy
        }

    def run_ablation_study(self, dataloader, ablation_targets: Dict[str, List[int]],
                          max_batches: int = 30) -> Dict:
        """Run full ablation study"""
        print("\n" + "="*60)
        print("ABLATION STUDY")
        print("="*60)

        # Compute baseline
        print("\nComputing baseline...")
        baseline = self.compute_baseline_metrics(dataloader, max_batches)
        print(f"  Baseline PPL: {baseline['ppl']:.2f}")

        results = {
            'baseline': baseline,
            'ablations': {},
            'causality_matrix': {}
        }

        # Run ablation for each target
        all_neurons = []
        for category, neurons in ablation_targets.items():
            all_neurons.extend(neurons)

        for category, neurons in ablation_targets.items():
            print(f"\n--- Ablating {category} neurons: {neurons} ---")
            results['ablations'][category] = {}

            for neuron_id in neurons:
                ablation_result = self.ablate_neuron(neuron_id, dataloader, max_batches)
                results['ablations'][category][neuron_id] = ablation_result

                # Compute delta from baseline
                ppl_delta = ablation_result['ppl'] - baseline['ppl']
                print(f"  Neuron {neuron_id}: PPL {ablation_result['ppl']:.2f} (Δ{ppl_delta:+.2f})")

                # Per-POS impact
                pos_impacts = {}
                for pos in baseline['pos_accuracy']:
                    if pos in ablation_result['pos_accuracy']:
                        delta = ablation_result['pos_accuracy'][pos] - baseline['pos_accuracy'][pos]
                        pos_impacts[pos] = delta * 100  # Convert to percentage

                results['causality_matrix'][neuron_id] = pos_impacts

        return results

    def visualize_causality_matrix(self, results: Dict, output_path: str):
        """Visualize neuron-POS causality matrix"""
        if not HAS_MATPLOTLIB:
            return

        matrix = results['causality_matrix']
        neurons = sorted(matrix.keys())

        # Get common POS tags
        all_pos = set()
        for impacts in matrix.values():
            all_pos.update(impacts.keys())
        pos_list = ['DET', 'ADP', 'AUX', 'VERB', 'NOUN', 'ADJ', 'ADV', 'PRON', 'PUNCT']
        pos_list = [p for p in pos_list if p in all_pos]

        # Build matrix
        data = np.zeros((len(neurons), len(pos_list)))
        for i, neuron in enumerate(neurons):
            for j, pos in enumerate(pos_list):
                if pos in matrix[neuron]:
                    data[i, j] = matrix[neuron][pos]

        fig, ax = plt.subplots(figsize=(12, max(6, len(neurons) * 0.4)))

        im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-20, vmax=20)

        ax.set_xticks(range(len(pos_list)))
        ax.set_xticklabels(pos_list)
        ax.set_yticks(range(len(neurons)))
        ax.set_yticklabels([f'N{n}' for n in neurons])

        # Add values
        for i in range(len(neurons)):
            for j in range(len(pos_list)):
                val = data[i, j]
                color = 'white' if abs(val) > 10 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                       fontsize=8, color=color)

        plt.colorbar(im, ax=ax, label='Accuracy Change (%)')
        ax.set_title('Neuron Ablation Impact on POS Accuracy')
        ax.set_xlabel('Part of Speech')
        ax.set_ylabel('Ablated Neuron')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")


# ============================================================
# 3. Semantic Neuron Analysis
# ============================================================

class SemanticAnalyzer:
    """Analyze semantic specialization of neurons"""

    # Semantic category definitions
    SEMANTIC_CATEGORIES = {
        'PERSON': ['he', 'she', 'man', 'woman', 'president', 'king', 'queen',
                   'doctor', 'teacher', 'father', 'mother', 'boy', 'girl', 'person'],
        'PLACE': ['city', 'country', 'building', 'street', 'house', 'room',
                  'world', 'place', 'town', 'village', 'area', 'region'],
        'TIME': ['year', 'day', 'month', 'week', 'hour', 'minute', 'time',
                 'yesterday', 'today', 'tomorrow', 'morning', 'night', 'century'],
        'NUMBER': ['one', 'two', 'three', 'four', 'five', 'ten', 'hundred',
                   'thousand', 'million', 'first', 'second', 'third'],
        'ANIMAL': ['cat', 'dog', 'bird', 'fish', 'horse', 'animal', 'lion',
                   'tiger', 'bear', 'wolf', 'elephant', 'mouse'],
        'FOOD': ['food', 'water', 'eat', 'drink', 'bread', 'meat', 'fruit',
                 'wine', 'beer', 'coffee', 'tea', 'sugar'],
        'ACTION': ['go', 'come', 'run', 'walk', 'move', 'take', 'give',
                   'make', 'do', 'see', 'know', 'think', 'say', 'tell'],
        'EMOTION': ['love', 'hate', 'fear', 'happy', 'sad', 'angry', 'hope',
                    'feel', 'want', 'like', 'enjoy', 'suffer'],
    }

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        if HAS_SPACY:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                self.nlp = None
        else:
            self.nlp = None

        # Build token ID sets for each category
        self.category_token_ids = {}
        for category, words in self.SEMANTIC_CATEGORIES.items():
            token_ids = set()
            for word in words:
                # Try different variations
                for variant in [word, word.capitalize(), ' ' + word, ' ' + word.capitalize()]:
                    ids = self.tokenizer.encode(variant, add_special_tokens=False)
                    token_ids.update(ids)
            self.category_token_ids[category] = token_ids

    @torch.no_grad()
    def analyze_semantic_neurons(self, dataloader, max_batches: int = 100) -> Dict:
        """Find neurons specialized for semantic categories"""
        print("\n" + "="*60)
        print("SEMANTIC NEURON ANALYSIS")
        print("="*60)

        self.model.eval()

        # Count neuron activations per semantic category
        n_neurons = self.model.shared_neurons.compress_neurons.shape[0]
        category_neuron_counts = {cat: np.zeros(n_neurons) for cat in self.SEMANTIC_CATEGORIES}
        neuron_total_counts = np.zeros(n_neurons)

        # NER neuron counts
        ner_neuron_counts = defaultdict(lambda: np.zeros(n_neurons))

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Semantic Analysis", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            # Only use first layer for simplicity
            routing_info = routing_infos[0]

            for comp_key in ['compress_Q', 'compress_K', 'compress_V', 'compress_M']:
                if comp_key not in routing_info:
                    continue

                data = routing_info[comp_key]
                weights = data['weights']  # [B, S, k]

                if 'indices' in data:
                    indices = data['indices']  # [B, S, k]
                else:
                    k = min(8, weights.shape[-1])
                    _, indices = torch.topk(weights, k, dim=-1)

                # GPU vectorized counting
                for b in range(B):
                    for s in range(S):
                        token_id = input_ids[b, s].item()
                        top_neurons = indices[b, s].cpu().numpy()
                        top_weights = weights[b, s].cpu().numpy()

                        # Update total counts
                        for ni, w in zip(top_neurons, top_weights):
                            neuron_total_counts[ni] += w

                        # Check semantic categories
                        for category, token_ids in self.category_token_ids.items():
                            if token_id in token_ids:
                                for ni, w in zip(top_neurons, top_weights):
                                    category_neuron_counts[category][ni] += w

            # NER analysis
            if self.nlp is not None:
                for b in range(B):
                    text = self.tokenizer.decode(input_ids[b].tolist())
                    doc = self.nlp(text)

                    for ent in doc.ents:
                        label = ent.label_
                        # Find corresponding tokens (approximate)
                        ent_tokens = self.tokenizer.encode(ent.text, add_special_tokens=False)

                        for token_id in ent_tokens:
                            # Find position in sequence
                            positions = (input_ids[b] == token_id).nonzero(as_tuple=True)[0]
                            for pos in positions:
                                if pos < S:
                                    for comp_key in ['compress_Q']:
                                        if comp_key in routing_info:
                                            data = routing_info[comp_key]
                                            if 'indices' in data:
                                                indices = data['indices'][b, pos].cpu().numpy()
                                            else:
                                                _, idx = torch.topk(data['weights'][b, pos], 8)
                                                indices = idx.cpu().numpy()
                                            for ni in indices:
                                                ner_neuron_counts[label][ni] += 1

        # Find top neurons for each category
        results = {
            'semantic_neurons': {},
            'ner_neurons': {}
        }

        print("\n--- Semantic Category Neurons ---")
        for category in self.SEMANTIC_CATEGORIES:
            counts = category_neuron_counts[category]
            # Normalize by total usage to find specialized neurons
            normalized = counts / (neuron_total_counts + 1e-10)
            top_neurons = np.argsort(normalized)[-10:][::-1]

            results['semantic_neurons'][category] = {
                'top_neurons': [(int(n), float(normalized[n]), float(counts[n]))
                               for n in top_neurons],
                'total_activations': float(counts.sum())
            }

            print(f"  {category}: top neurons {top_neurons[:5].tolist()}")

        print("\n--- NER Category Neurons ---")
        for label, counts in ner_neuron_counts.items():
            if counts.sum() > 0:
                top_neurons = np.argsort(counts)[-5:][::-1]
                results['ner_neurons'][label] = {
                    'top_neurons': [(int(n), float(counts[n])) for n in top_neurons],
                    'total_count': float(counts.sum())
                }
                print(f"  {label}: top neurons {top_neurons.tolist()}")

        return results

    def visualize_semantic_heatmap(self, results: Dict, output_path: str):
        """Visualize semantic category - neuron heatmap"""
        if not HAS_MATPLOTLIB:
            return

        categories = list(results['semantic_neurons'].keys())

        # Get unique neurons
        all_neurons = set()
        for cat_data in results['semantic_neurons'].values():
            for n, _, _ in cat_data['top_neurons'][:10]:
                all_neurons.add(n)
        neurons = sorted(all_neurons)

        # Build matrix
        data = np.zeros((len(categories), len(neurons)))
        for i, cat in enumerate(categories):
            for n, score, _ in results['semantic_neurons'][cat]['top_neurons']:
                if n in neurons:
                    j = neurons.index(n)
                    data[i, j] = score

        fig, ax = plt.subplots(figsize=(max(12, len(neurons) * 0.5), 8))

        im = ax.imshow(data, aspect='auto', cmap='YlOrRd')

        ax.set_xticks(range(len(neurons)))
        ax.set_xticklabels([f'N{n}' for n in neurons], rotation=45, ha='right')
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)

        plt.colorbar(im, ax=ax, label='Normalized Activation')
        ax.set_title('Semantic Category - Neuron Specialization')
        ax.set_xlabel('Neuron')
        ax.set_ylabel('Category')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")


# ============================================================
# 4. Neuron Catalog
# ============================================================

class NeuronCatalog:
    """Generate comprehensive catalog of all neurons"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.n_neurons = model.shared_neurons.compress_neurons.shape[0]

        if HAS_SPACY:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                self.nlp = None
        else:
            self.nlp = None

    @torch.no_grad()
    def build_catalog(self, dataloader, max_batches: int = 100,
                     ablation_results: Dict = None) -> Dict:
        """Build comprehensive catalog for all neurons"""
        print("\n" + "="*60)
        print("BUILDING NEURON CATALOG")
        print("="*60)

        self.model.eval()

        # Initialize tracking
        neuron_token_counts = [Counter() for _ in range(self.n_neurons)]
        neuron_pos_counts = [Counter() for _ in range(self.n_neurons)]
        neuron_layer_usage = np.zeros((self.n_neurons, 8))  # 8 layers
        neuron_cooccurrence = np.zeros((self.n_neurons, self.n_neurons))
        neuron_total_usage = np.zeros(self.n_neurons)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Building Catalog", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            B, S = input_ids.shape

            # Get POS tags
            if self.nlp:
                batch_pos = []
                for b in range(B):
                    text = self.tokenizer.decode(input_ids[b].tolist())
                    doc = self.nlp(text)
                    pos_list = [token.pos_ for token in doc]
                    if len(pos_list) < S:
                        pos_list.extend([''] * (S - len(pos_list)))
                    batch_pos.append(pos_list[:S])
            else:
                batch_pos = [['' for _ in range(S)] for _ in range(B)]

            _, routing_infos = self.model(input_ids, return_routing_info=True)

            for layer_idx, routing_info in enumerate(routing_infos):
                if layer_idx >= 8:
                    break

                for comp_key in ['compress_Q']:  # Focus on Q for catalog
                    if comp_key not in routing_info:
                        continue

                    data = routing_info[comp_key]
                    weights = data['weights']

                    if 'indices' in data:
                        indices = data['indices']
                    else:
                        k = min(8, weights.shape[-1])
                        _, indices = torch.topk(weights, k, dim=-1)

                    for b in range(B):
                        for s in range(S):
                            token_id = input_ids[b, s].item()
                            token_str = self.tokenizer.decode([token_id]).strip()
                            pos = batch_pos[b][s] if s < len(batch_pos[b]) else ''

                            top_neurons = indices[b, s].cpu().numpy()
                            top_weights = weights[b, s].cpu().numpy()

                            for ni, w in zip(top_neurons, top_weights):
                                neuron_token_counts[ni][token_str] += 1
                                if pos:
                                    neuron_pos_counts[ni][pos] += 1
                                neuron_layer_usage[ni, layer_idx] += w
                                neuron_total_usage[ni] += w

                            # Co-occurrence
                            for i, n1 in enumerate(top_neurons):
                                for n2 in top_neurons[i+1:]:
                                    neuron_cooccurrence[n1, n2] += 1
                                    neuron_cooccurrence[n2, n1] += 1

        # Build catalog entries
        catalog = {}

        print("\nGenerating neuron profiles...")
        for n in tqdm(range(self.n_neurons), desc="Profiling"):
            # Top tokens
            top_tokens = neuron_token_counts[n].most_common(20)

            # Primary POS
            pos_dist = dict(neuron_pos_counts[n])
            total_pos = sum(pos_dist.values())
            if total_pos > 0:
                pos_dist = {k: v / total_pos for k, v in pos_dist.items()}
                primary_pos = max(pos_dist, key=pos_dist.get) if pos_dist else 'UNK'
            else:
                primary_pos = 'UNK'

            # Layer usage
            layer_usage = neuron_layer_usage[n] / (neuron_layer_usage[n].sum() + 1e-10)

            # Co-occurring neurons
            cooccur = neuron_cooccurrence[n]
            co_neurons = np.argsort(cooccur)[-5:][::-1].tolist()

            # Semantic category
            semantic_cat = self._infer_semantic_category(top_tokens)

            # Ablation impact (if available)
            ablation_impact = {}
            if ablation_results and n in ablation_results.get('causality_matrix', {}):
                ablation_impact = ablation_results['causality_matrix'][n]

            # Role classification
            role = self._classify_role(primary_pos, pos_dist, top_tokens)

            catalog[n] = {
                'neuron_id': n,
                'top_tokens': [(t, c) for t, c in top_tokens],
                'primary_pos': primary_pos,
                'pos_distribution': pos_dist,
                'layer_usage': layer_usage.tolist(),
                'co_occurring_neurons': co_neurons,
                'semantic_category': semantic_cat,
                'ablation_impact': ablation_impact,
                'role': role,
                'total_usage': float(neuron_total_usage[n])
            }

        # Summary statistics
        role_counts = Counter(v['role'] for v in catalog.values())

        print("\n--- Neuron Role Distribution ---")
        for role, count in role_counts.most_common():
            print(f"  {role}: {count} neurons ({100*count/self.n_neurons:.1f}%)")

        return {
            'neurons': catalog,
            'role_distribution': dict(role_counts),
            'total_neurons': self.n_neurons
        }

    def _infer_semantic_category(self, top_tokens: List[Tuple[str, int]]) -> str:
        """Infer semantic category from top tokens"""
        categories = SemanticAnalyzer.SEMANTIC_CATEGORIES

        scores = {cat: 0 for cat in categories}
        for token, count in top_tokens[:10]:
            token_lower = token.lower().strip()
            for cat, words in categories.items():
                if token_lower in words:
                    scores[cat] += count

        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'general'

    def _classify_role(self, primary_pos: str, pos_dist: Dict,
                      top_tokens: List[Tuple[str, int]]) -> str:
        """Classify neuron role"""
        # Grammar neurons
        grammar_pos = {'DET', 'ADP', 'AUX', 'CCONJ', 'SCONJ', 'PUNCT', 'PART'}
        if primary_pos in grammar_pos and pos_dist.get(primary_pos, 0) > 0.5:
            return 'grammar'

        # Position-based (check if mostly used at certain positions)
        tokens = [t for t, _ in top_tokens[:10]]
        if all(t in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]'] for t in tokens if t):
            return 'position'

        # Semantic neurons (concrete content words)
        content_pos = {'NOUN', 'VERB', 'ADJ'}
        if primary_pos in content_pos and pos_dist.get(primary_pos, 0) > 0.4:
            return 'semantic'

        # Syntactic (mixed grammatical function)
        if pos_dist.get('VERB', 0) + pos_dist.get('NOUN', 0) > 0.5:
            return 'syntactic'

        return 'mixed'

    def visualize_catalog_summary(self, catalog: Dict, output_path: str):
        """Visualize catalog summary"""
        if not HAS_MATPLOTLIB:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Role distribution pie chart
        ax = axes[0, 0]
        roles = catalog['role_distribution']
        ax.pie(roles.values(), labels=roles.keys(), autopct='%1.1f%%')
        ax.set_title('Neuron Role Distribution')

        # 2. Top POS heatmap
        ax = axes[0, 1]
        neurons = catalog['neurons']
        pos_list = ['DET', 'ADP', 'AUX', 'VERB', 'NOUN', 'ADJ', 'ADV', 'PRON', 'PUNCT']

        # Sample 50 neurons for visibility
        sample_neurons = sorted(neurons.keys())[:50]
        pos_data = np.zeros((len(sample_neurons), len(pos_list)))

        for i, n in enumerate(sample_neurons):
            for j, pos in enumerate(pos_list):
                pos_data[i, j] = neurons[n]['pos_distribution'].get(pos, 0)

        im = ax.imshow(pos_data.T, aspect='auto', cmap='YlOrRd')
        ax.set_yticks(range(len(pos_list)))
        ax.set_yticklabels(pos_list)
        ax.set_xlabel('Neuron ID')
        ax.set_title('POS Distribution (first 50 neurons)')
        plt.colorbar(im, ax=ax)

        # 3. Layer usage heatmap
        ax = axes[1, 0]
        layer_data = np.array([neurons[n]['layer_usage'] for n in sample_neurons])
        im = ax.imshow(layer_data.T, aspect='auto', cmap='Blues')
        ax.set_yticks(range(8))
        ax.set_yticklabels([f'L{i}' for i in range(8)])
        ax.set_xlabel('Neuron ID')
        ax.set_title('Layer Usage (first 50 neurons)')
        plt.colorbar(im, ax=ax)

        # 4. Usage distribution
        ax = axes[1, 1]
        usages = [neurons[n]['total_usage'] for n in sorted(neurons.keys())]
        ax.bar(range(len(usages)), sorted(usages, reverse=True))
        ax.set_xlabel('Neuron Rank')
        ax.set_ylabel('Total Usage')
        ax.set_title('Neuron Usage Distribution')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")


# ============================================================
# 5. Advanced Visualizations
# ============================================================

class AdvancedVisualizer:
    """t-SNE, UMAP, and combined visualizations"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def visualize_neuron_embedding(self, catalog: Dict, output_path: str,
                                   method: str = 'pca'):
        """Visualize neuron embeddings with dimensionality reduction"""
        if not HAS_MATPLOTLIB:
            return

        print(f"\n--- Computing {method.upper()} embedding ---")

        # Get neuron weights
        neurons = self.model.shared_neurons.compress_neurons.data.cpu().numpy()
        N, D, R = neurons.shape
        neurons_flat = neurons.reshape(N, D * R)

        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2)
            embedding = reducer.fit_transform(neurons_flat)
        elif method == 'tsne' and HAS_SKLEARN:
            # PCA first for speed
            pca = PCA(n_components=50)
            neurons_pca = pca.fit_transform(neurons_flat)
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, N-1))
            embedding = tsne.fit_transform(neurons_pca)
        elif method == 'umap' and HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(neurons_flat)
        else:
            print(f"Method {method} not available, using PCA")
            reducer = PCA(n_components=2)
            embedding = reducer.fit_transform(neurons_flat)

        # Get roles for coloring
        role_to_color = {
            'grammar': 'red',
            'semantic': 'blue',
            'syntactic': 'green',
            'position': 'orange',
            'mixed': 'gray'
        }

        colors = []
        for n in range(N):
            role = catalog['neurons'][n]['role']
            colors.append(role_to_color.get(role, 'gray'))

        # Plot
        fig, ax = plt.subplots(figsize=(12, 12))

        for role, color in role_to_color.items():
            mask = [colors[i] == color for i in range(N)]
            points = embedding[mask]
            if len(points) > 0:
                ax.scatter(points[:, 0], points[:, 1], c=color, label=role, alpha=0.7, s=50)

        # Annotate some neurons
        for i in range(0, N, max(1, N // 30)):
            ax.annotate(str(i), (embedding[i, 0], embedding[i, 1]), fontsize=7)

        ax.legend()
        ax.set_title(f'Neuron Embedding ({method.upper()}) colored by Role')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")

    @torch.no_grad()
    def visualize_attention_neuron(self, text: str, output_path: str):
        """Combined attention pattern and neuron selection visualization"""
        if not HAS_MATPLOTLIB:
            return

        self.model.eval()

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor([tokens], device=self.device)
        token_strs = [self.tokenizer.decode([t]).strip() for t in tokens]

        # Get outputs with attention
        outputs = self.model(input_ids, return_routing_info=True, output_attentions=True)

        # This depends on model architecture - adjust as needed
        if len(outputs) >= 2:
            routing_infos = outputs[1]
        else:
            print("Cannot get routing info")
            return

        # Get attention patterns if available
        attentions = None
        if hasattr(self.model, 'get_attention_weights'):
            attentions = self.model.get_attention_weights()

        # Create visualization
        n_tokens = len(tokens)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Top row: Neuron selection per layer
        for layer_idx in range(min(4, len(routing_infos))):
            ax = axes[0, layer_idx]

            routing_info = routing_infos[layer_idx]
            if 'compress_Q' in routing_info:
                data = routing_info['compress_Q']
                if 'indices' in data:
                    indices = data['indices'][0].cpu().numpy()
                else:
                    _, indices = torch.topk(data['weights'][0], 8, dim=-1)
                    indices = indices.cpu().numpy()

                weights = data['weights'][0].cpu().numpy()

                # Heatmap of top-k neurons
                unique_neurons = sorted(set(indices.flatten()))[:30]
                neuron_to_idx = {n: i for i, n in enumerate(unique_neurons)}

                heatmap = np.zeros((n_tokens, len(unique_neurons)))
                for t in range(n_tokens):
                    for ki in range(indices.shape[1]):
                        n_idx = indices[t, ki]
                        if n_idx in neuron_to_idx:
                            heatmap[t, neuron_to_idx[n_idx]] += weights[t, ki]

                ax.imshow(heatmap.T, aspect='auto', cmap='Blues')
                ax.set_xticks(range(n_tokens))
                ax.set_xticklabels(token_strs, rotation=45, ha='right', fontsize=7)
                ax.set_title(f'Layer {layer_idx} Neurons')

        # Bottom row: Could show attention patterns if available
        for i in range(4):
            ax = axes[1, i]
            ax.text(0.5, 0.5, f'Attention L{i}\n(if available)',
                   ha='center', va='center', fontsize=12)
            ax.axis('off')

        plt.suptitle(f'Attention + Neuron: "{text[:50]}..."')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")


# ============================================================
# Main Runner
# ============================================================

class DAWNDeepAnalysis:
    """Main class to run all analyses"""

    def __init__(self, checkpoint_path: str, data_path: str, output_dir: str, device: str = 'cuda'):
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.device = device

        # Create output directories
        self.dirs = {
            'sentence': os.path.join(output_dir, 'sentence_visualizations'),
            'ablation': os.path.join(output_dir, 'ablation'),
            'semantic': os.path.join(output_dir, 'semantic'),
            'catalog': os.path.join(output_dir, 'catalog'),
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        # Load model and data
        self._load_model()
        self._load_data()

    def _load_model(self):
        """Load DAWN model from checkpoint"""
        print(f"Loading model from: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Detect model version
        config = checkpoint.get('config', {})

        # Try v10.1 first, then v10.0
        try:
            from models.model_v10_1 import DAWNModel, DAWNConfig
            print("Using model_v10_1")
        except ImportError:
            try:
                from models.model_v10 import DAWNModel, DAWNConfig
                print("Using model_v10")
            except ImportError:
                from models.model import DAWNModel, DAWNConfig
                print("Using base model")

        # Create config
        if isinstance(config, dict):
            model_config = DAWNConfig(**config)
        else:
            model_config = config

        # Create and load model
        self.model = DAWNModel(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params")

    def _load_data(self):
        """Load validation data"""
        print(f"Loading data from: {self.data_path}")

        with open(self.data_path, 'rb') as f:
            texts = pickle.load(f)

        # Create dataset and dataloader
        from torch.utils.data import Dataset, DataLoader

        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=128):
                self.encodings = []
                for text in texts[:5000]:  # Limit for speed
                    enc = tokenizer(text, truncation=True, max_length=max_length,
                                   padding='max_length', return_tensors='pt')
                    self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})

            def __len__(self):
                return len(self.encodings)

            def __getitem__(self, idx):
                return self.encodings[idx]

        dataset = TextDataset(texts, self.tokenizer)
        self.dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        print(f"Loaded {len(dataset)} samples")

    def run_all(self):
        """Run all analyses"""
        all_results = {}

        # 1. Sentence Visualization
        print("\n" + "="*60)
        print("1. SENTENCE VISUALIZATION")
        print("="*60)

        visualizer = SentenceVisualizer(self.model, self.tokenizer, self.device)

        test_sentences = [
            "The cat sat on the mat.",
            "She quickly ran to the store and bought milk.",
            "In 1990, the president signed the bill.",
            "The quick brown fox jumps over the lazy dog.",
            "Scientists discovered a new species in the Amazon.",
            "He said that she would come tomorrow.",
            "The book on the table is mine.",
            "They have been working here since 2010.",
            "If it rains, we will stay home.",
            "The company announced record profits yesterday.",
        ]

        for i, sentence in enumerate(test_sentences):
            print(f"\n  Analyzing: '{sentence[:40]}...'")
            analysis = visualizer.analyze_sentence(sentence)

            # Token-wise visualization
            visualizer.visualize_sentence(
                analysis,
                os.path.join(self.dirs['sentence'], f'sentence_{i+1:02d}_tokenwise.png'),
                show_layers=[0, 3, 7]
            )

            # Layer progression
            visualizer.visualize_layer_progression(
                analysis,
                os.path.join(self.dirs['sentence'], f'sentence_{i+1:02d}_layerwise.png')
            )

        # 2. Ablation Experiments
        print("\n" + "="*60)
        print("2. ABLATION EXPERIMENTS")
        print("="*60)

        ablation = AblationExperiment(self.model, self.tokenizer, self.device)

        ablation_targets = {
            'DET_neurons': [168, 200, 215],
            'ADP_neurons': [223, 165, 100],
            'AUX_neurons': [202, 154, 181],
            'PUNCT_neurons': [65, 74, 66],
        }

        ablation_results = ablation.run_ablation_study(self.dataloader, ablation_targets, max_batches=30)
        all_results['ablation'] = ablation_results

        # Save ablation results
        with open(os.path.join(self.dirs['ablation'], 'ablation_results.json'), 'w') as f:
            json.dump(ablation_results, f, indent=2, default=str)

        # Visualize causality matrix
        ablation.visualize_causality_matrix(
            ablation_results,
            os.path.join(self.dirs['ablation'], 'causality_matrix.png')
        )

        # 3. Semantic Analysis
        print("\n" + "="*60)
        print("3. SEMANTIC ANALYSIS")
        print("="*60)

        semantic = SemanticAnalyzer(self.model, self.tokenizer, self.device)
        semantic_results = semantic.analyze_semantic_neurons(self.dataloader, max_batches=50)
        all_results['semantic'] = semantic_results

        # Save semantic results
        with open(os.path.join(self.dirs['semantic'], 'semantic_neurons.json'), 'w') as f:
            json.dump(semantic_results, f, indent=2, default=str)

        # Visualize
        semantic.visualize_semantic_heatmap(
            semantic_results,
            os.path.join(self.dirs['semantic'], 'category_heatmap.png')
        )

        # 4. Neuron Catalog
        print("\n" + "="*60)
        print("4. NEURON CATALOG")
        print("="*60)

        catalog_builder = NeuronCatalog(self.model, self.tokenizer, self.device)
        catalog = catalog_builder.build_catalog(self.dataloader, max_batches=100,
                                                ablation_results=ablation_results)
        all_results['catalog'] = catalog

        # Save catalog
        with open(os.path.join(self.dirs['catalog'], 'neuron_catalog.json'), 'w') as f:
            json.dump(catalog, f, indent=2, default=str)

        # Visualize
        catalog_builder.visualize_catalog_summary(
            catalog,
            os.path.join(self.dirs['catalog'], 'neuron_roles_summary.png')
        )

        # 5. Advanced Visualizations
        print("\n" + "="*60)
        print("5. ADVANCED VISUALIZATIONS")
        print("="*60)

        adv_viz = AdvancedVisualizer(self.model, self.tokenizer, self.device)

        # PCA embedding
        adv_viz.visualize_neuron_embedding(
            catalog,
            os.path.join(self.dirs['catalog'], 'neuron_pca.png'),
            method='pca'
        )

        # Attention + Neuron combined
        adv_viz.visualize_attention_neuron(
            "The president signed the new economic bill yesterday.",
            os.path.join(self.dirs['sentence'], 'attention_neuron_combined.png')
        )

        # Generate report
        self._generate_report(all_results)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)

        return all_results

    def _generate_report(self, results: Dict):
        """Generate markdown report"""
        report_path = os.path.join(self.output_dir, 'report.md')

        with open(report_path, 'w') as f:
            f.write("# DAWN Deep Analysis Report\n\n")
            f.write(f"**Checkpoint:** `{self.checkpoint_path}`\n\n")
            f.write(f"**Data:** `{self.data_path}`\n\n")

            f.write("## 1. Sentence Visualization\n\n")
            f.write("Token-neuron mapping and layer progression visualizations generated.\n\n")
            f.write("See `sentence_visualizations/` directory.\n\n")

            f.write("## 2. Ablation Experiments\n\n")
            if 'ablation' in results:
                abl = results['ablation']
                f.write(f"**Baseline PPL:** {abl['baseline']['ppl']:.2f}\n\n")
                f.write("### Per-Neuron Impact\n\n")
                f.write("| Neuron | PPL | Delta |\n")
                f.write("|--------|-----|-------|\n")
                for cat, neurons in abl['ablations'].items():
                    for n, data in neurons.items():
                        delta = data['ppl'] - abl['baseline']['ppl']
                        f.write(f"| {n} ({cat}) | {data['ppl']:.2f} | {delta:+.2f} |\n")

            f.write("\n## 3. Semantic Analysis\n\n")
            if 'semantic' in results:
                f.write("### Semantic Category Neurons\n\n")
                for cat, data in results['semantic']['semantic_neurons'].items():
                    top = data['top_neurons'][:3]
                    neurons_str = ', '.join([f"N{n}" for n, _, _ in top])
                    f.write(f"- **{cat}**: {neurons_str}\n")

            f.write("\n## 4. Neuron Catalog\n\n")
            if 'catalog' in results:
                f.write("### Role Distribution\n\n")
                for role, count in results['catalog']['role_distribution'].items():
                    pct = 100 * count / results['catalog']['total_neurons']
                    f.write(f"- {role}: {count} neurons ({pct:.1f}%)\n")

            f.write("\n## 5. Visualizations\n\n")
            f.write("- `sentence_visualizations/`: Token-neuron heatmaps\n")
            f.write("- `ablation/causality_matrix.png`: Neuron-POS causality\n")
            f.write("- `semantic/category_heatmap.png`: Semantic specialization\n")
            f.write("- `catalog/neuron_roles_summary.png`: Role distribution\n")
            f.write("- `catalog/neuron_pca.png`: PCA embedding\n")

        print(f"  Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='DAWN Deep Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to validation data (pkl)')
    parser.add_argument('--output_dir', type=str, default='./dawn_analysis',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    analyzer = DAWNDeepAnalysis(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_dir=args.output_dir,
        device=args.device
    )

    analyzer.run_all()


if __name__ == '__main__':
    main()
