#!/usr/bin/env python3
"""
POS-based Neuron Analysis for DAWN
===================================
Analyze neuron specialization by Part-of-Speech tags.

Uses Universal Dependencies English Web Treebank to show that
different POS categories activate different neurons.

Usage:
    python scripts/analysis/pos_neuron_analysis.py \
        --checkpoint path/to/checkpoint \
        --layer 11 \
        --pool fv \
        --output pos_analysis/ \
        --bf16
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
from tqdm import tqdm

# Handle both module import and standalone execution
try:
    from .utils import (
        load_model, get_router,
        ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS,
        HAS_MATPLOTLIB, plt
    )
except ImportError:
    from scripts.analysis.utils import (
        load_model, get_router,
        ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS,
        HAS_MATPLOTLIB, plt
    )

if HAS_MATPLOTLIB:
    import seaborn as sns
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist

# Universal POS tags (UPOS)
UPOS_TAGS = [
    'ADJ',    # adjective
    'ADP',    # adposition
    'ADV',    # adverb
    'AUX',    # auxiliary
    'CCONJ',  # coordinating conjunction
    'DET',    # determiner
    'INTJ',   # interjection
    'NOUN',   # noun
    'NUM',    # numeral
    'PART',   # particle
    'PRON',   # pronoun
    'PROPN',  # proper noun
    'PUNCT',  # punctuation
    'SCONJ',  # subordinating conjunction
    'SYM',    # symbol
    'VERB',   # verb
    'X',      # other
]

# Simplified POS groups for cleaner visualization
POS_GROUPS = {
    'Content Words': ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'],
    'Function Words': ['DET', 'ADP', 'AUX', 'PRON', 'CCONJ', 'SCONJ', 'PART'],
    'Other': ['NUM', 'PUNCT', 'SYM', 'INTJ', 'X'],
}


class POSNeuronAnalyzer:
    """Analyze neuron activations by POS tags."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda',
        target_layer: int = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.target_layer = target_layer
        self.model.eval()

        # Storage for analysis
        self.pos_neuron_counts = defaultdict(lambda: defaultdict(int))
        self.pos_total_tokens = defaultdict(int)
        self.layer_pos_neurons = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def load_ud_dataset(self, split: str = 'train', max_sentences: int = None, data_path: str = None):
        """
        Load Universal Dependencies English Web Treebank.

        Uses conllu file parsing (HuggingFace datasets no longer supports UD).

        Args:
            split: 'train', 'dev', or 'test'
            max_sentences: Maximum sentences to load
            data_path: Path to local conllu file (optional)
        """
        # URL for UD English EWT
        ud_urls = {
            'train': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu',
            'dev': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu',
            'test': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu',
        }

        # Try to import conllu
        try:
            import conllu
        except ImportError:
            raise ImportError("Please install conllu: pip install conllu")

        # Load data
        if data_path and os.path.exists(data_path):
            print(f"Loading from local file: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = f.read()
        else:
            import urllib.request
            url = ud_urls.get(split, ud_urls['train'])
            print(f"Downloading UD English EWT ({split})...")
            print(f"URL: {url}")

            try:
                with urllib.request.urlopen(url) as response:
                    data = response.read().decode('utf-8')
            except Exception as e:
                print(f"Download failed: {e}")
                print("\nTrying alternative: NLTK treebank...")
                return self._load_nltk_treebank(max_sentences)

        # Parse conllu
        print("Parsing conllu data...")
        sentences = conllu.parse(data)

        if max_sentences:
            sentences = sentences[:max_sentences]

        # Convert to our format
        dataset = []
        for sent in sentences:
            tokens = [token['form'] for token in sent]
            upos = [token['upos'] for token in sent]
            dataset.append({'tokens': tokens, 'upos': upos})

        print(f"Loaded {len(dataset)} sentences")
        return dataset

    def _load_nltk_treebank(self, max_sentences: int = None):
        """Fallback: Load NLTK treebank with universal tagset."""
        try:
            import nltk
            nltk.download('treebank', quiet=True)
            nltk.download('universal_tagset', quiet=True)
            from nltk.corpus import treebank
        except ImportError:
            raise ImportError("Please install nltk: pip install nltk")

        print("Loading NLTK treebank...")

        # Map NLTK universal tags to UPOS
        nltk_to_upos = {
            'NOUN': 'NOUN', 'VERB': 'VERB', 'ADJ': 'ADJ', 'ADV': 'ADV',
            'ADP': 'ADP', 'DET': 'DET', 'PRON': 'PRON', 'NUM': 'NUM',
            'CONJ': 'CCONJ', 'PRT': 'PART', '.': 'PUNCT', 'X': 'X',
        }

        sentences = treebank.tagged_sents(tagset='universal')
        if max_sentences:
            sentences = sentences[:max_sentences]

        dataset = []
        for sent in sentences:
            tokens = [word for word, tag in sent]
            upos = [nltk_to_upos.get(tag, 'X') for word, tag in sent]
            dataset.append({'tokens': tokens, 'upos': upos})

        print(f"Loaded {len(dataset)} sentences from NLTK treebank")
        return dataset

    def get_pos_for_tokens(
        self,
        ud_tokens: List[str],
        ud_pos: List[str],
    ) -> Tuple[List[str], List[int]]:
        """
        Map DAWN tokenizer tokens to POS tags using character spans.

        Returns: (list of POS tags, list of token IDs)
        """
        # Build text and character spans for each UD token
        text = ""
        ud_char_spans = []  # [(start, end, pos), ...]

        for ud_token, pos in zip(ud_tokens, ud_pos):
            start = len(text)
            text += ud_token
            end = len(text)
            ud_char_spans.append((start, end, pos))
            text += " "  # space between tokens

        text = text.rstrip()  # remove trailing space

        # Try to use offset_mapping if tokenizer supports it
        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_tensors=None,
            )
            token_ids = encoding['input_ids']
            offset_mapping = encoding['offset_mapping']

            if not token_ids:
                return [], []

            # Map each token to POS using offset_mapping
            dawn_pos_tags = []
            for start, end in offset_mapping:
                # Find which UD token this span overlaps with
                assigned_pos = 'X'
                for ud_start, ud_end, pos in ud_char_spans:
                    # Check overlap
                    if start < ud_end and end > ud_start:
                        assigned_pos = pos
                        break
                dawn_pos_tags.append(assigned_pos)

            return dawn_pos_tags, token_ids

        except (TypeError, KeyError):
            # Fallback: tokenizer doesn't support offset_mapping
            # Use simple sequential mapping
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)

            if not token_ids:
                return [], []

            # Simple approach: assign each token to the UD token at same index
            # (works reasonably for 1:1 or close mappings)
            dawn_pos_tags = []
            ud_idx = 0
            decoded_so_far = ""

            for tid in token_ids:
                token_text = self.tokenizer.decode([tid])
                decoded_so_far += token_text

                # Find current position in original text
                # by counting how many UD tokens we've passed
                char_count = 0
                assigned_pos = 'X'
                for i, (ud_start, ud_end, pos) in enumerate(ud_char_spans):
                    char_count = ud_end + 1  # +1 for space
                    if len(decoded_so_far.strip()) <= char_count:
                        assigned_pos = pos
                        break

                dawn_pos_tags.append(assigned_pos)

            return dawn_pos_tags, token_ids

    def extract_routing_for_tokens(
        self,
        token_ids: List[int],
        pool_type: str = 'fv',
    ) -> Dict[int, List[int]]:
        """
        Get routing indices for each token position.

        Returns: {position: [neuron_indices]}
        """
        input_ids = torch.tensor([token_ids], device=self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, return_routing_info=True)

        if isinstance(outputs, tuple) and len(outputs) >= 2:
            routing_infos = outputs[1]
        else:
            return {}

        # Map pool_type to binary mask key (v18.x uses boolean masks for selection)
        mask_key_map = {
            'fv': 'fv_mask',
            'rv': 'rv_mask',
            'fqk_q': 'fqk_mask_Q',
            'fqk_k': 'fqk_mask_K',
            'rqk_q': 'rqk_mask_Q',
            'rqk_k': 'rqk_mask_K',
            'fknow': 'feature_know_mask',
            'rknow': 'restore_know_mask',
        }
        # Fallback to weights for older models
        weight_key_map = {
            'fv': 'fv_weights',
            'rv': 'rv_weights',
            'fqk_q': 'fqk_weights_Q',
            'fqk_k': 'fqk_weights_K',
            'rqk_q': 'rqk_weights_Q',
            'rqk_k': 'rqk_weights_K',
            'fknow': 'feature_know_w',
            'rknow': 'restore_know_w',
        }
        mask_key = mask_key_map.get(pool_type, 'fv_mask')
        weight_key = weight_key_map.get(pool_type, 'fv_weights')

        # Collect per-position neuron indices
        position_neurons = defaultdict(set)
        seq_len = input_ids.shape[1]

        # Debug: check structure on first call
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"\n[DEBUG] Routing info structure:")
            print(f"  Number of layers: {len(routing_infos)}")
            if routing_infos:
                layer0 = routing_infos[0]
                print(f"  Layer 0 keys: {list(layer0.keys())}")
                if 'attention' in layer0:
                    print(f"  attention keys: {list(layer0['attention'].keys())}")
                    attn = layer0['attention']
                    # Check for binary mask (preferred)
                    if mask_key in attn:
                        m = attn[mask_key]
                        print(f"  {mask_key} type: {type(m)}, shape: {m.shape if hasattr(m, 'shape') else 'N/A'}")
                        print(f"  {mask_key} dtype: {m.dtype if hasattr(m, 'dtype') else 'N/A'}")
                        print(f"  {mask_key} active: {m.sum().item() if hasattr(m, 'sum') else 'N/A'}")
                        print(f"  Using binary mask for POS analysis")
                    elif weight_key in attn:
                        w = attn[weight_key]
                        print(f"  {weight_key} type: {type(w)}, shape: {w.shape if hasattr(w, 'shape') else 'N/A'}")
                        print(f"  {weight_key} dtype: {w.dtype if hasattr(w, 'dtype') else 'N/A'}")
                        print(f"  Falling back to weights (no mask available)")
                    else:
                        print(f"  Neither {mask_key} nor {weight_key} found in attention!")
                if 'knowledge' in layer0:
                    print(f"  knowledge keys: {list(layer0['knowledge'].keys())}")

        for layer_idx, layer_info in enumerate(routing_infos):
            # Skip if not target layer
            if self.target_layer is not None and layer_idx != self.target_layer:
                continue

            # Get mask/weights from attention or knowledge
            attn = layer_info.get('attention', layer_info)
            know = layer_info.get('knowledge', {})

            # Prefer binary mask over weights
            mask = attn.get(mask_key)
            if mask is None:
                mask = know.get(mask_key)

            # Fallback to weights if mask not available
            use_mask = mask is not None
            if not use_mask:
                mask = attn.get(weight_key)
                if mask is None:
                    mask = know.get(weight_key)

            if mask is not None:
                # mask: [B, T, N] - boolean for mask, float for weights
                for pos in range(seq_len):
                    m = mask[0, pos]  # [N]
                    if use_mask:
                        # Binary mask: use nonzero directly
                        active_neurons = m.nonzero(as_tuple=True)[0].cpu().tolist()
                    else:
                        # Weights fallback: threshold at > 0
                        active_neurons = (m > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                    position_neurons[pos].update(active_neurons)

        return {pos: list(neurons) for pos, neurons in position_neurons.items()}

    def analyze_sentence(
        self,
        ud_tokens: List[str],
        ud_pos: List[str],
        pool_type: str = 'fv',
        debug: bool = False,
    ):
        """Analyze a single sentence and update statistics."""
        try:
            pos_tags, token_ids = self.get_pos_for_tokens(ud_tokens, ud_pos)
        except Exception as e:
            if debug:
                print(f"Alignment error: {e}")
            return

        if not token_ids:
            return

        if debug:
            text = ' '.join(ud_tokens)
            print(f"\nSentence: {text[:60]}...")
            for i, (tid, pos) in enumerate(zip(token_ids, pos_tags)):
                token_text = self.tokenizer.decode([tid])
                print(f"  [{i}] '{token_text}' -> {pos}")

        # Get routing for all positions
        position_neurons = self.extract_routing_for_tokens(token_ids, pool_type)

        # Update statistics
        for pos_idx, pos in enumerate(pos_tags):
            neurons = position_neurons.get(pos_idx, [])

            self.pos_total_tokens[pos] += 1

            for neuron in neurons:
                self.pos_neuron_counts[pos][neuron] += 1

    def analyze_dataset(
        self,
        dataset,
        pool_type: str = 'fv',
        max_sentences: int = None,
        debug: bool = False,
    ):
        """Analyze full dataset."""
        n_sentences = min(len(dataset), max_sentences) if max_sentences else len(dataset)

        print(f"\nAnalyzing {n_sentences} sentences...")
        print(f"Pool: {pool_type.upper()}")
        print(f"Layer: {self.target_layer if self.target_layer is not None else 'all'}")

        # Debug first 3 sentences
        if debug:
            print("\n[DEBUG] First 3 sentences alignment:")
            for i in range(min(3, n_sentences)):
                example = dataset[i]
                self.analyze_sentence(
                    example['tokens'], example['upos'], pool_type, debug=True
                )

        for i in tqdm(range(n_sentences), desc="Processing"):
            example = dataset[i]
            tokens = example['tokens']
            pos_tags = example['upos']

            try:
                self.analyze_sentence(tokens, pos_tags, pool_type, debug=False)
            except Exception as e:
                continue

            # Print progress stats every 500 sentences
            if (i + 1) % 500 == 0:
                total_tokens = sum(self.pos_total_tokens.values())
                total_neurons = sum(
                    len(neurons) for neurons in self.pos_neuron_counts.values()
                )
                tqdm.write(f"  Progress: {total_tokens} tokens, {total_neurons} neuron activations")

    def get_results(self) -> Dict:
        """Compile analysis results."""
        # Per-POS neuron frequency
        pos_neuron_freq = {}
        for pos in UPOS_TAGS:
            if self.pos_total_tokens[pos] > 0:
                neuron_counts = self.pos_neuron_counts[pos]
                total = self.pos_total_tokens[pos]

                # Normalize by token count
                freq = {
                    neuron: count / total
                    for neuron, count in neuron_counts.items()
                }
                pos_neuron_freq[pos] = freq

        # Top neurons per POS
        top_neurons_per_pos = {}
        for pos, freq in pos_neuron_freq.items():
            sorted_neurons = sorted(freq.items(), key=lambda x: -x[1])[:20]
            top_neurons_per_pos[pos] = sorted_neurons

        # Find POS-specific neurons (high in one POS, low in others)
        all_neurons = set()
        for freq in pos_neuron_freq.values():
            all_neurons.update(freq.keys())

        neuron_specificity = {}
        for neuron in all_neurons:
            scores = []
            for pos in UPOS_TAGS:
                if pos in pos_neuron_freq:
                    scores.append((pos, pos_neuron_freq[pos].get(neuron, 0)))

            if scores:
                scores.sort(key=lambda x: -x[1])
                top_pos, top_score = scores[0]
                if len(scores) > 1:
                    second_score = scores[1][1]
                    specificity = top_score / (second_score + 1e-6)
                else:
                    specificity = float('inf')

                if top_score > 0.1:  # At least 10% activation
                    neuron_specificity[neuron] = {
                        'top_pos': top_pos,
                        'top_score': top_score,
                        'specificity': min(specificity, 100),
                    }

        # POS overlap matrix
        overlap_matrix = {}
        for pos1 in UPOS_TAGS:
            if pos1 not in pos_neuron_freq:
                continue
            neurons1 = set(n for n, f in pos_neuron_freq[pos1].items() if f > 0.1)

            for pos2 in UPOS_TAGS:
                if pos2 not in pos_neuron_freq:
                    continue
                neurons2 = set(n for n, f in pos_neuron_freq[pos2].items() if f > 0.1)

                if neurons1 and neurons2:
                    overlap = len(neurons1 & neurons2)
                    union = len(neurons1 | neurons2)
                    jaccard = overlap / union if union > 0 else 0
                    overlap_matrix[f"{pos1}-{pos2}"] = jaccard

        return {
            'pos_token_counts': dict(self.pos_total_tokens),
            'pos_neuron_freq': {
                pos: {str(k): v for k, v in freq.items()}
                for pos, freq in pos_neuron_freq.items()
            },
            'top_neurons_per_pos': {
                pos: [(int(n), f) for n, f in neurons]
                for pos, neurons in top_neurons_per_pos.items()
            },
            'neuron_specificity': {
                str(k): v for k, v in sorted(
                    neuron_specificity.items(),
                    key=lambda x: -x[1]['specificity']
                )[:50]
            },
            'overlap_matrix': overlap_matrix,
            'total_neurons_seen': len(all_neurons),
        }


def plot_pos_heatmap(
    results: Dict,
    output_path: str = None,
    figsize: Tuple[int, int] = (16, 10),
):
    """Plot POS x Neuron activation heatmap."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return

    pos_neuron_freq = results['pos_neuron_freq']

    # Get all neurons and sort by total activation
    all_neurons = set()
    for freq in pos_neuron_freq.values():
        all_neurons.update(int(n) for n in freq.keys())

    # Filter to top neurons
    neuron_total = defaultdict(float)
    for freq in pos_neuron_freq.values():
        for n, f in freq.items():
            neuron_total[int(n)] += f

    top_neurons = sorted(neuron_total.keys(), key=lambda n: -neuron_total[n])[:100]

    # Build matrix
    pos_list = [p for p in UPOS_TAGS if p in pos_neuron_freq]
    matrix = np.zeros((len(pos_list), len(top_neurons)))

    for i, pos in enumerate(pos_list):
        for j, neuron in enumerate(top_neurons):
            matrix[i, j] = pos_neuron_freq[pos].get(str(neuron), 0)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix,
        xticklabels=[str(n) for n in top_neurons],
        yticklabels=pos_list,
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Activation Frequency'}
    )

    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('POS Tag')
    ax.set_title('Neuron Activation by Part-of-Speech')

    # Rotate x labels
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()


def plot_pos_clustering(
    results: Dict,
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """Plot POS clustering based on neuron activation similarity."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return

    pos_neuron_freq = results['pos_neuron_freq']

    # Get all neurons
    all_neurons = set()
    for freq in pos_neuron_freq.values():
        all_neurons.update(int(n) for n in freq.keys())
    all_neurons = sorted(all_neurons)

    # Build feature vectors
    pos_list = [p for p in UPOS_TAGS if p in pos_neuron_freq]
    vectors = np.zeros((len(pos_list), len(all_neurons)))

    for i, pos in enumerate(pos_list):
        for j, neuron in enumerate(all_neurons):
            vectors[i, j] = pos_neuron_freq[pos].get(str(neuron), 0)

    # Compute distances and cluster
    if len(pos_list) < 2:
        print("Not enough POS tags for clustering")
        return

    distances = pdist(vectors, metric='cosine')
    linkage = hierarchy.linkage(distances, method='ward')

    # Plot dendrogram
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Dendrogram
    hierarchy.dendrogram(
        linkage,
        labels=pos_list,
        orientation='left',
        ax=axes[0]
    )
    axes[0].set_title('POS Clustering by Neuron Patterns')
    axes[0].set_xlabel('Distance')

    # Similarity heatmap
    similarity = 1 - pdist(vectors, metric='cosine')
    sim_matrix = np.zeros((len(pos_list), len(pos_list)))
    idx = 0
    for i in range(len(pos_list)):
        for j in range(i + 1, len(pos_list)):
            sim_matrix[i, j] = similarity[idx]
            sim_matrix[j, i] = similarity[idx]
            idx += 1
        sim_matrix[i, i] = 1.0

    sns.heatmap(
        sim_matrix,
        xticklabels=pos_list,
        yticklabels=pos_list,
        cmap='RdYlGn',
        vmin=0, vmax=1,
        annot=True, fmt='.2f',
        ax=axes[1]
    )
    axes[1].set_title('POS Similarity (Cosine)')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()


def plot_top_neurons_by_pos(
    results: Dict,
    output_path: str = None,
    figsize: Tuple[int, int] = (14, 10),
):
    """Plot top neurons for each POS."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return

    top_neurons = results['top_neurons_per_pos']
    pos_list = [p for p in UPOS_TAGS if p in top_neurons]

    n_cols = 4
    n_rows = (len(pos_list) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, pos in enumerate(pos_list):
        neurons = top_neurons[pos][:10]
        if neurons:
            neuron_ids = [n[0] for n in neurons]
            freqs = [n[1] for n in neurons]

            axes[i].barh(range(len(neurons)), freqs, color='steelblue')
            axes[i].set_yticks(range(len(neurons)))
            axes[i].set_yticklabels([f'N{n}' for n in neuron_ids])
            axes[i].set_title(pos)
            axes[i].set_xlabel('Freq')
            axes[i].invert_yaxis()

    # Hide unused subplots
    for i in range(len(pos_list), len(axes)):
        axes[i].axis('off')

    plt.suptitle('Top 10 Neurons per POS Tag', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()


def plot_specificity(
    results: Dict,
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """Plot neuron specificity - neurons specialized for specific POS."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return

    specificity = results['neuron_specificity']
    if not specificity:
        print("No specific neurons found")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Top specific neurons
    items = list(specificity.items())[:20]
    neurons = [int(n) for n, _ in items]
    scores = [s['specificity'] for _, s in items]
    pos_tags = [s['top_pos'] for _, s in items]

    colors = plt.cm.tab20(np.linspace(0, 1, len(set(pos_tags))))
    pos_color_map = {pos: colors[i] for i, pos in enumerate(set(pos_tags))}

    axes[0].barh(
        range(len(neurons)),
        scores,
        color=[pos_color_map[p] for p in pos_tags]
    )
    axes[0].set_yticks(range(len(neurons)))
    axes[0].set_yticklabels([f'N{n} ({p})' for n, p in zip(neurons, pos_tags)])
    axes[0].set_xlabel('Specificity Score')
    axes[0].set_title('Most POS-Specific Neurons')
    axes[0].invert_yaxis()

    # POS-specific neuron counts
    pos_specific_counts = defaultdict(int)
    for _, s in specificity.items():
        pos_specific_counts[s['top_pos']] += 1

    pos_sorted = sorted(pos_specific_counts.items(), key=lambda x: -x[1])
    axes[1].barh(
        range(len(pos_sorted)),
        [c for _, c in pos_sorted],
        color='coral'
    )
    axes[1].set_yticks(range(len(pos_sorted)))
    axes[1].set_yticklabels([p for p, _ in pos_sorted])
    axes[1].set_xlabel('Number of Specific Neurons')
    axes[1].set_title('Specialized Neurons per POS')
    axes[1].invert_yaxis()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()


def print_results(results: Dict):
    """Print analysis results."""
    print("\n" + "=" * 70)
    print("POS NEURON ANALYSIS RESULTS")
    print("=" * 70)

    # Token counts
    print("\nToken counts by POS:")
    for pos, count in sorted(results['pos_token_counts'].items(), key=lambda x: -x[1]):
        print(f"  {pos:8s}: {count:6d}")

    print(f"\nTotal unique neurons: {results['total_neurons_seen']}")

    # Top neurons per POS
    print("\n" + "-" * 50)
    print("Top 5 neurons per POS:")
    print("-" * 50)
    for pos in UPOS_TAGS:
        if pos in results['top_neurons_per_pos']:
            neurons = results['top_neurons_per_pos'][pos][:5]
            neuron_str = ', '.join([f"N{n}({f:.2f})" for n, f in neurons])
            print(f"  {pos:8s}: {neuron_str}")

    # Specific neurons
    print("\n" + "-" * 50)
    print("Most POS-Specific Neurons:")
    print("-" * 50)
    for neuron, info in list(results['neuron_specificity'].items())[:10]:
        print(f"  Neuron {neuron:4s}: {info['top_pos']:8s} (score={info['top_score']:.2f}, specificity={info['specificity']:.1f}x)")


def main():
    parser = argparse.ArgumentParser(
        description='POS-based Neuron Analysis for DAWN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python scripts/analysis/pos_neuron_analysis.py --checkpoint checkpoint.pt

  # Specific layer and pool
  python scripts/analysis/pos_neuron_analysis.py --checkpoint checkpoint.pt --layer 11 --pool fv

  # With optimizations
  python scripts/analysis/pos_neuron_analysis.py --checkpoint checkpoint.pt --bf16 --max_sentences 1000
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file or directory')
    parser.add_argument('--output', type=str, default='pos_analysis',
                        help='Output directory (default: pos_analysis)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--pool', type=str, default='fv',
                        choices=['fv', 'rv', 'fqk_q', 'fqk_k', 'rqk_q', 'rqk_k', 'fknow', 'rknow'],
                        help='Pool type to analyze (default: fv)')
    parser.add_argument('--layer', type=int, default=None,
                        help='Specific layer to analyze (default: all)')
    parser.add_argument('--max_sentences', type=int, default=2000,
                        help='Maximum sentences to analyze (default: 2000)')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split (default: train)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to local conllu file (optional, downloads if not provided)')
    parser.add_argument('--bf16', action='store_true',
                        help='Use bfloat16 precision')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: show token alignment for first few sentences')
    args = parser.parse_args()

    # Device check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Find checkpoint
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        candidates = (
            list(ckpt_path.glob('*best*.pt')) +
            list(ckpt_path.glob('**/*best*.pt')) +
            list(ckpt_path.glob('*.pt')) +
            list(ckpt_path.glob('**/*.pt'))
        )
        candidates = [c for c in candidates if 'optimizer' not in c.name.lower()]
        if not candidates:
            print(f"No checkpoint found in {ckpt_path}")
            return
        best_candidates = [c for c in candidates if 'best' in c.name.lower()]
        ckpt_path = best_candidates[0] if best_candidates else candidates[0]
        print(f"Using checkpoint: {ckpt_path}")

    # Load model
    print(f"\n{'='*70}")
    print(f"Loading model from {ckpt_path}")
    print('='*70)
    model, tokenizer, config = load_model(str(ckpt_path), args.device)
    model = model.to(args.device)
    model.eval()

    # Enable debug mode for routing info (required for analysis)
    if hasattr(model, 'router') and hasattr(model.router, 'debug_mode'):
        model.router.debug_mode = True
        print("Enabled router debug_mode for analysis")

    # Apply optimizations
    if args.bf16:
        print("Using bfloat16 precision")
        model = model.to(torch.bfloat16)

    if args.compile:
        if hasattr(torch, 'compile'):
            print("Applying torch.compile...")
            model = torch.compile(model, mode='reduce-overhead')

    # Create analyzer
    analyzer = POSNeuronAnalyzer(
        model, tokenizer, args.device,
        target_layer=args.layer
    )

    # Load dataset
    dataset = analyzer.load_ud_dataset(args.split, args.max_sentences, args.data_path)

    # Run analysis
    analyzer.analyze_dataset(dataset, args.pool, args.max_sentences, debug=args.debug)

    # Get results
    results = analyzer.get_results()

    # Print results
    print_results(results)

    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, f'pos_analysis_{args.pool}_layer{args.layer or "all"}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Plotting
    if not args.no_plot and HAS_MATPLOTLIB:
        print("\nGenerating plots...")

        heatmap_path = os.path.join(args.output, 'pos_neuron_heatmap.png')
        plot_pos_heatmap(results, heatmap_path)
        print(f"  Saved: {heatmap_path}")

        cluster_path = os.path.join(args.output, 'pos_clustering.png')
        plot_pos_clustering(results, cluster_path)
        print(f"  Saved: {cluster_path}")

        top_path = os.path.join(args.output, 'top_neurons_by_pos.png')
        plot_top_neurons_by_pos(results, top_path)
        print(f"  Saved: {top_path}")

        spec_path = os.path.join(args.output, 'neuron_specificity.png')
        plot_specificity(results, spec_path)
        print(f"  Saved: {spec_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"  Sentences analyzed: {args.max_sentences}")
    print(f"  Pool: {args.pool.upper()}")
    print(f"  Layer: {args.layer if args.layer is not None else 'all'}")
    print(f"  Total unique neurons: {results['total_neurons_seen']}")
    print(f"  Output directory: {args.output}")


if __name__ == '__main__':
    main()
