"""
Semantic Analysis for DAWN
===========================
Analyze semantic properties of DAWN v17.1 routing.

This module validates the core claims of the DAWN paper:
1. Semantically similar inputs -> similar neuron routing paths
2. Neurons learn linguistic/semantic properties
3. Context-dependent dynamic routing

Includes:
- Semantic path similarity analysis
- POS-based routing analysis
- Context-dependent routing analysis
- Neuron-token activation heatmaps
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .utils import (
    ROUTING_KEYS,
    convert_to_serializable,
    get_batch_input_ids, get_routing_from_outputs,
    HAS_TQDM, tqdm
)

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


class SemanticAnalyzer:
    """Semantic analysis for DAWN routing patterns."""

    def __init__(self, model, router, tokenizer, device='cuda'):
        """
        Initialize analyzer.

        Args:
            model: DAWN model
            router: NeuronRouter instance
            tokenizer: Tokenizer instance
            device: Device for computation
        """
        self.model = model
        self.router = router
        self.tokenizer = tokenizer
        self.device = device

        # Load spaCy for POS tagging
        self.nlp = None
        if HAS_SPACY:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except Exception:
                pass

    def get_routing_path(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Extract routing path for a given text.

        Args:
            text: Input text

        Returns:
            Dictionary mapping routing key to [seq_len, n_neurons] tensor
        """
        enc = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = enc['input_ids'].to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, return_routing_info=True)

        routing_infos = get_routing_from_outputs(outputs)
        if routing_infos is None:
            return {}

        # Extract first layer's attention routing
        attn = routing_infos[0].get('attention', {})

        paths = {}
        for key, (display, pref_key, weight_key, pool) in ROUTING_KEYS.items():
            weights = attn.get(weight_key)
            if weights is not None:
                # [B, S, N] -> [S, N]
                if weights.dim() == 3:
                    paths[key] = weights[0].cpu()
                else:
                    paths[key] = weights.cpu()

        return paths

    def compute_path_similarity(self, path1: Dict, path2: Dict) -> Dict[str, Dict]:
        """
        Compute similarity between two routing paths.

        Args:
            path1: First routing path
            path2: Second routing path

        Returns:
            Dictionary with cosine and Jaccard similarity per routing key
        """
        results = {}

        for key in path1.keys():
            if key not in path2:
                continue

            p1, p2 = path1[key], path2[key]

            # Truncate to minimum length
            min_len = min(p1.shape[0], p2.shape[0])
            p1, p2 = p1[:min_len], p2[:min_len]

            # Cosine similarity (position-wise average)
            p1_norm = p1 / (p1.norm(dim=-1, keepdim=True) + 1e-8)
            p2_norm = p2 / (p2.norm(dim=-1, keepdim=True) + 1e-8)
            cosine = (p1_norm * p2_norm).sum(dim=-1).mean().item()

            # Jaccard similarity (top-8 neurons)
            k = min(8, p1.shape[-1])
            top1 = set(p1.sum(dim=0).topk(k)[1].tolist())
            top2 = set(p2.sum(dim=0).topk(k)[1].tolist())
            jaccard = len(top1 & top2) / len(top1 | top2) if (top1 | top2) else 0

            results[key] = {
                'cosine': cosine,
                'jaccard': jaccard,
            }

        return results

    def analyze_semantic_path_similarity(self, sentence_pairs: List[Tuple[str, str, str]]) -> Dict:
        """
        Compare routing similarity for semantically similar vs different sentence pairs.

        This validates the core claim: similar meaning -> similar routing.

        Args:
            sentence_pairs: List of (sent1, sent2, label) where label is 'similar' or 'different'

        Returns:
            Statistics for similar vs different pairs with interpretation
        """
        similar_sims = []
        different_sims = []

        for sent1, sent2, label in sentence_pairs:
            path1 = self.get_routing_path(sent1)
            path2 = self.get_routing_path(sent2)

            if not path1 or not path2:
                continue

            sim = self.compute_path_similarity(path1, path2)

            # Average across all routing keys
            avg_cosine = np.mean([v['cosine'] for v in sim.values()])
            avg_jaccard = np.mean([v['jaccard'] for v in sim.values()])

            if label == 'similar':
                similar_sims.append({'cosine': avg_cosine, 'jaccard': avg_jaccard})
            else:
                different_sims.append({'cosine': avg_cosine, 'jaccard': avg_jaccard})

        results = {
            'similar_pairs': {
                'count': len(similar_sims),
                'cosine_mean': np.mean([s['cosine'] for s in similar_sims]) if similar_sims else 0,
                'cosine_std': np.std([s['cosine'] for s in similar_sims]) if similar_sims else 0,
                'jaccard_mean': np.mean([s['jaccard'] for s in similar_sims]) if similar_sims else 0,
            },
            'different_pairs': {
                'count': len(different_sims),
                'cosine_mean': np.mean([s['cosine'] for s in different_sims]) if different_sims else 0,
                'cosine_std': np.std([s['cosine'] for s in different_sims]) if different_sims else 0,
                'jaccard_mean': np.mean([s['jaccard'] for s in different_sims]) if different_sims else 0,
            },
        }

        # Interpretation
        if similar_sims and different_sims:
            sim_cos = results['similar_pairs']['cosine_mean']
            diff_cos = results['different_pairs']['cosine_mean']
            gap = sim_cos - diff_cos

            if gap > 0.1:
                verdict = 'GOOD: Semantic similarity reflected in routing'
            elif gap > 0:
                verdict = 'WEAK: Routing has slight semantic correlation'
            else:
                verdict = 'BAD: Routing inversely correlated with semantics'

            results['interpretation'] = {
                'similarity_gap': gap,
                'verdict': verdict
            }

        return results

    def get_default_sentence_pairs(self) -> List[Tuple[str, str, str]]:
        """Get default test sentence pairs."""
        return [
            # Similar pairs (paraphrases)
            ("The cat sat on the mat.", "A feline rested on the rug.", "similar"),
            ("She bought a new car.", "She purchased a new vehicle.", "similar"),
            ("The weather is beautiful today.", "It's a lovely day outside.", "similar"),
            ("He runs every morning.", "He jogs each day at dawn.", "similar"),
            ("The book was interesting.", "The novel was captivating.", "similar"),
            ("I need to go to the store.", "I have to visit the shop.", "similar"),
            ("The children played in the park.", "Kids were playing at the playground.", "similar"),
            ("She cooked dinner for the family.", "She prepared a meal for her relatives.", "similar"),

            # Different pairs (unrelated)
            ("The cat sat on the mat.", "Stock prices rose sharply.", "different"),
            ("She bought a new car.", "The experiment failed completely.", "different"),
            ("The weather is beautiful today.", "Binary search has O(log n) complexity.", "different"),
            ("He runs every morning.", "The painting was sold at auction.", "different"),
            ("The book was interesting.", "Photosynthesis requires sunlight.", "different"),
            ("I need to go to the store.", "The treaty was signed in 1945.", "different"),
            ("The children played in the park.", "The server crashed unexpectedly.", "different"),
            ("She cooked dinner for the family.", "Quantum entanglement is mysterious.", "different"),
        ]

    def analyze_pos_routing(self, dataloader, max_batches: int = 50) -> Dict:
        """
        Analyze routing patterns by part-of-speech.

        Uses spaCy for POS tagging to see if different POS categories
        activate different neurons.

        Args:
            dataloader: DataLoader for input data
            max_batches: Maximum batches to process

        Returns:
            POS-wise routing statistics
        """
        if self.nlp is None:
            return {'error': 'spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm'}

        # POS -> routing weights
        pos_weights = defaultdict(list)
        pos_counts = defaultdict(int)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc='POS Analysis', total=max_batches)):
                if i >= max_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)

                outputs = self.model(input_ids, return_routing_info=True)
                routing_infos = get_routing_from_outputs(outputs)

                if routing_infos is None:
                    continue

                attn = routing_infos[0].get('attention', {})

                # Process each sequence
                for b in range(input_ids.shape[0]):
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[b].cpu().tolist())
                    text = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)

                    try:
                        doc = self.nlp(text)
                        spacy_tokens = [(t.text.lower(), t.pos_) for t in doc]
                    except Exception:
                        continue

                    # Match tokens to POS
                    spacy_idx = 0
                    for s, token in enumerate(tokens):
                        if token.startswith('[') or token.startswith('##'):
                            continue

                        token_clean = token.replace('##', '').lower()
                        while spacy_idx < len(spacy_tokens):
                            sp_token, sp_pos = spacy_tokens[spacy_idx]
                            if token_clean in sp_token or sp_token in token_clean:
                                # Match found - collect routing weights
                                for key, (_, _, weight_key, _) in ROUTING_KEYS.items():
                                    w = attn.get(weight_key)
                                    if w is not None and w.dim() >= 2:
                                        if w.dim() == 3 and s < w.shape[1]:
                                            pos_weights[f"{key}_{sp_pos}"].append(w[b, s].cpu())
                                        elif w.dim() == 2:
                                            pos_weights[f"{key}_{sp_pos}"].append(w[b].cpu())

                                pos_counts[sp_pos] += 1
                                spacy_idx += 1
                                break
                            spacy_idx += 1

        # Analyze results
        results = {
            'pos_counts': dict(pos_counts),
            'routing_by_pos': {},
        }

        for key_pos, weights in pos_weights.items():
            if len(weights) < 10:
                continue

            stacked = torch.stack(weights)
            mean_w = stacked.mean(dim=0)

            # Top activated neurons
            top_k = min(5, mean_w.shape[0])
            top_neurons = mean_w.topk(top_k)

            results['routing_by_pos'][key_pos] = {
                'count': len(weights),
                'mean_activation': float(mean_w.mean()),
                'top_neurons': [
                    (int(idx), float(val))
                    for idx, val in zip(top_neurons.indices.tolist(), top_neurons.values.tolist())
                ],
            }

        return results

    def analyze_context_dependent_routing(self, word_contexts: Dict[str, List[str]]) -> Dict:
        """
        Analyze if the same word has different routing in different contexts.

        This validates context-dependent dynamic routing.

        Args:
            word_contexts: Dictionary mapping word to list of sentences containing it

        Returns:
            Per-word routing variance statistics
        """
        results = {}

        for word, sentences in word_contexts.items():
            if len(sentences) < 2:
                continue

            word_routings = []

            for sent in sentences:
                # Find word position in tokenized sequence
                tokens = self.tokenizer.tokenize(sent.lower())
                word_lower = word.lower()

                word_positions = []
                for i, tok in enumerate(tokens):
                    if word_lower in tok or tok in word_lower:
                        word_positions.append(i + 1)  # +1 for [CLS]

                if not word_positions:
                    continue

                # Get routing path
                path = self.get_routing_path(sent)
                if not path:
                    continue

                # Extract routing at word position
                word_routing = {}
                for key, weights in path.items():
                    if weights.dim() >= 1:
                        pos = word_positions[0]
                        if pos < weights.shape[0]:
                            word_routing[key] = weights[pos]

                if word_routing:
                    word_routings.append(word_routing)

            if len(word_routings) < 2:
                continue

            # Compute routing variance across contexts
            variances = {}
            for key in word_routings[0].keys():
                key_routings = [wr[key] for wr in word_routings if key in wr]
                if len(key_routings) >= 2:
                    stacked = torch.stack(key_routings)
                    variance = stacked.var(dim=0).mean().item()
                    variances[key] = variance

            results[word] = {
                'n_contexts': len(word_routings),
                'routing_variance': variances,
                'avg_variance': np.mean(list(variances.values())) if variances else 0,
            }

        # Summary interpretation
        if results:
            avg_var = np.mean([r['avg_variance'] for r in results.values()])

            if avg_var > 0.1:
                interpretation = 'HIGH: Strong context-dependent routing'
            elif avg_var > 0.01:
                interpretation = 'MODERATE: Some context sensitivity'
            else:
                interpretation = 'LOW: Routing mostly context-independent'

            results['summary'] = {
                'overall_context_variance': avg_var,
                'interpretation': interpretation
            }

        return results

    def get_default_word_contexts(self) -> Dict[str, List[str]]:
        """Get default polysemous word contexts."""
        return {
            "bank": [
                "I deposited money at the bank.",
                "The river bank was covered with flowers.",
                "You can bank on his promise.",
            ],
            "bat": [
                "He swung the baseball bat.",
                "A bat flew out of the cave.",
            ],
            "light": [
                "Turn on the light please.",
                "The bag is very light.",
                "Light colors are better for summer.",
            ],
            "run": [
                "I run every morning.",
                "The program will run automatically.",
                "There's a run in her stocking.",
            ],
            "play": [
                "Children love to play outside.",
                "She will play the piano.",
                "We watched a play at the theater.",
            ],
        }

    def analyze_neuron_token_heatmap(self, dataloader, max_batches: int = 30,
                                      top_k_neurons: int = 20) -> Dict:
        """
        Generate neuron-token activation heatmap data.

        Shows which neurons activate for which token types.

        Args:
            dataloader: DataLoader for input data
            max_batches: Maximum batches to process
            top_k_neurons: Number of top neurons to include

        Returns:
            Per-neuron top token activations
        """
        neuron_tokens = defaultdict(lambda: defaultdict(float))

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc='Neuron-Token Heatmap', total=max_batches)):
                if i >= max_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)

                outputs = self.model(input_ids, return_routing_info=True)
                routing_infos = get_routing_from_outputs(outputs)

                if routing_infos is None:
                    continue

                attn = routing_infos[0].get('attention', {})

                # Use F-QK weights as representative
                weights = attn.get('fqk_weights_Q')
                if weights is None or weights.dim() != 3:
                    continue

                B, S, N = weights.shape

                for b in range(B):
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[b].cpu().tolist())
                    for s, token in enumerate(tokens):
                        if token.startswith('['):
                            continue

                        # Active neurons at this position
                        w = weights[b, s]
                        active_neurons = (w > 0).nonzero().squeeze(-1)

                        for neuron_id in active_neurons.tolist():
                            if isinstance(neuron_id, int):
                                neuron_tokens[neuron_id][token] += w[neuron_id].item()

        # Select top-k neurons by total activation
        neuron_total_activation = {n: sum(tokens.values()) for n, tokens in neuron_tokens.items()}
        top_neurons = sorted(neuron_total_activation.keys(),
                            key=lambda x: -neuron_total_activation[x])[:top_k_neurons]

        results = {}
        for neuron_id in top_neurons:
            token_counts = neuron_tokens[neuron_id]
            top_tokens = sorted(token_counts.items(), key=lambda x: -x[1])[:10]
            results[f'neuron_{neuron_id}'] = {
                'total_activation': neuron_total_activation[neuron_id],
                'top_tokens': top_tokens,
            }

        return results

    def run_all(self, dataloader=None, output_dir: str = './semantic_analysis') -> Dict:
        """
        Run all semantic analyses.

        Args:
            dataloader: DataLoader for input data (optional, required for some analyses)
            output_dir: Directory for outputs

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        # 1. Semantic Path Similarity
        print("\n[1/4] Analyzing Semantic Path Similarity...")
        pairs = self.get_default_sentence_pairs()
        results['path_similarity'] = self.analyze_semantic_path_similarity(pairs)

        # 2. Context-dependent Routing
        print("\n[2/4] Analyzing Context-dependent Routing...")
        word_contexts = self.get_default_word_contexts()
        results['context_routing'] = self.analyze_context_dependent_routing(word_contexts)

        # 3. POS Routing (requires dataloader)
        if dataloader is not None:
            print("\n[3/4] Analyzing POS Routing Patterns...")
            results['pos_routing'] = self.analyze_pos_routing(dataloader)

            print("\n[4/4] Generating Neuron-Token Heatmap...")
            results['neuron_heatmap'] = self.analyze_neuron_token_heatmap(dataloader)
        else:
            print("\n[3/4] Skipping POS analysis (no dataloader)")
            print("\n[4/4] Skipping heatmap (no dataloader)")

        # Save results
        import json
        output_path = os.path.join(output_dir, 'semantic_analysis.json')
        with open(output_path, 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results
