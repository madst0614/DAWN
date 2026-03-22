"""
Semantic Analysis - JAX Version
=================================
Analyze semantic properties of DAWN routing on JAX/TPU.

Validates core DAWN paper claims:
1. Semantically similar inputs → similar neuron routing paths
2. Context-dependent dynamic routing (polysemous words)
3. Per-routing-type and per-layer breakdown

NOTE: This is a JAX port of semantic.py.
No spaCy dependency — uses tokenizer-level analysis + hardcoded test data.
"""

import os
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .base_jax import BaseAnalyzerJAX
from .utils_jax import (
    ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS,
    WEIGHT_KEY_MAP,
    create_batches, JAXRoutingDataExtractor,
    HAS_JAX,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _calc_entropy_ratio(probs: np.ndarray) -> float:
    """Compute entropy ratio as percentage of maximum."""
    probs = np.maximum(probs, 0)
    total = probs.sum()
    if total <= 0:
        return 0.0
    probs = probs / total
    mask = probs > 0
    if not mask.any():
        return 0.0
    entropy = -np.sum(probs[mask] * np.log(probs[mask]))
    max_entropy = np.log(len(probs))
    return float(entropy / max_entropy * 100) if max_entropy > 0 else 0.0


class SemanticAnalyzerJAX(BaseAnalyzerJAX):
    """Semantic analysis for DAWN routing (JAX version)."""

    def __init__(self, model, params, config: Dict, tokenizer=None):
        super().__init__(model, params, config, tokenizer=tokenizer)

    def _extract_routing_path(self, token_ids: List[int]) -> Dict[str, np.ndarray]:
        """Extract routing path for a token sequence.

        Returns:
            Dict mapping routing key → weight vector (mean-pooled over positions)
        """
        input_ids = np.array([token_ids])
        routing = self.extractor.extract_routing(input_ids)

        path = {}
        attn = routing.get('attention', {})
        know = routing.get('knowledge', {})
        all_weights = {**attn, **know}

        all_keys = list(ROUTING_KEYS.keys()) + list(KNOWLEDGE_ROUTING_KEYS.keys())

        for key in all_keys:
            raw_key = WEIGHT_KEY_MAP.get(key, key)
            w = all_weights.get(raw_key)
            if w is None:
                continue
            # Mean-pool over batch and sequence: [B, S, N] → [N]
            if w.ndim == 3:
                path[key] = w.mean(axis=(0, 1))
            elif w.ndim == 2:
                path[key] = w.mean(axis=0)

        return path

    def compute_path_similarity(
        self, path1: Dict[str, np.ndarray], path2: Dict[str, np.ndarray]
    ) -> Dict:
        """Compute similarity between two routing paths.

        Returns per-routing-type cosine and jaccard, plus weighted overall.
        """
        per_type = {}
        all_cosines = []
        all_jaccards = []

        for key in path1:
            if key not in path2:
                continue

            p1, p2 = path1[key], path2[key]

            # Cosine similarity
            cosine = _cosine_similarity(p1, p2)

            # Jaccard (top-8 neurons)
            k = min(8, len(p1))
            top1 = set(np.argsort(p1)[-k:].tolist())
            top2 = set(np.argsort(p2)[-k:].tolist())
            jaccard = len(top1 & top2) / len(top1 | top2) if (top1 | top2) else 0

            per_type[key] = {'cosine': cosine, 'jaccard': jaccard}
            all_cosines.append(cosine)
            all_jaccards.append(jaccard)

        overall = {}
        if all_cosines:
            overall['cosine_mean'] = float(np.mean(all_cosines))
            overall['jaccard_mean'] = float(np.mean(all_jaccards))

        return {'per_routing_type': per_type, 'overall': overall}

    def analyze_semantic_path_similarity(
        self, sentence_pairs: List[Tuple[str, str, str]]
    ) -> Dict:
        """
        Compare routing similarity for similar vs different sentence pairs.

        Args:
            sentence_pairs: List of (sent1, sent2, label) where label is 'similar'/'different'

        Returns:
            Statistics comparing similar vs different pair routing similarities
        """
        if self.tokenizer is None:
            return {'error': 'tokenizer required'}

        # Extract paths for all unique sentences
        all_sentences = set()
        for s1, s2, _ in sentence_pairs:
            all_sentences.add(s1)
            all_sentences.add(s2)

        print(f"  Extracting routing for {len(all_sentences)} unique sentences...")
        sentence_paths = {}
        for sent in tqdm(list(all_sentences), desc='Routing paths'):
            ids = self.tokenizer.encode(sent, add_special_tokens=True)
            sentence_paths[sent] = self._extract_routing_path(ids)

        # Compute similarities
        similar_cosines = []
        different_cosines = []
        similar_jaccards = []
        different_jaccards = []
        routing_type_stats = defaultdict(lambda: {'similar': [], 'different': []})

        for sent1, sent2, label in sentence_pairs:
            p1, p2 = sentence_paths.get(sent1, {}), sentence_paths.get(sent2, {})
            if not p1 or not p2:
                continue

            sim = self.compute_path_similarity(p1, p2)
            overall = sim.get('overall', {})

            cos = overall.get('cosine_mean', 0)
            jac = overall.get('jaccard_mean', 0)

            if label == 'similar':
                similar_cosines.append(cos)
                similar_jaccards.append(jac)
            else:
                different_cosines.append(cos)
                different_jaccards.append(jac)

            for rtype, rdata in sim.get('per_routing_type', {}).items():
                routing_type_stats[rtype][label].append(rdata.get('cosine', 0))

        # Build results
        results = {}
        if similar_cosines:
            results['similar_pairs'] = {
                'count': len(similar_cosines),
                'cosine_mean': float(np.mean(similar_cosines)),
                'cosine_std': float(np.std(similar_cosines)),
                'jaccard_mean': float(np.mean(similar_jaccards)),
            }
        if different_cosines:
            results['different_pairs'] = {
                'count': len(different_cosines),
                'cosine_mean': float(np.mean(different_cosines)),
                'cosine_std': float(np.std(different_cosines)),
                'jaccard_mean': float(np.mean(different_jaccards)),
            }

        # Per routing type breakdown
        results['per_routing_type'] = {}
        attention_gaps = []
        knowledge_gaps = []

        for rtype, stats in routing_type_stats.items():
            sim_cos = np.mean(stats['similar']) if stats['similar'] else 0
            diff_cos = np.mean(stats['different']) if stats['different'] else 0
            gap = sim_cos - diff_cos

            results['per_routing_type'][rtype] = {
                'similar_cosine': float(sim_cos),
                'different_cosine': float(diff_cos),
                'gap': float(gap),
            }

            if rtype in ('fknow', 'rknow'):
                knowledge_gaps.append(gap)
            else:
                attention_gaps.append(gap)

        results['routing_type_summary'] = {
            'attention_avg_gap': float(np.mean(attention_gaps)) if attention_gaps else 0,
            'knowledge_avg_gap': float(np.mean(knowledge_gaps)) if knowledge_gaps else 0,
        }

        # Interpretation
        if similar_cosines and different_cosines:
            sim_mean = results['similar_pairs']['cosine_mean']
            diff_mean = results['different_pairs']['cosine_mean']
            gap = sim_mean - diff_mean

            if gap > 0.1:
                verdict = 'GOOD: Semantic similarity reflected in routing'
            elif gap > 0.05:
                verdict = 'MODERATE: Some semantic correlation in routing'
            elif gap > 0:
                verdict = 'WEAK: Routing has slight semantic correlation'
            else:
                verdict = 'BAD: Routing inversely correlated with semantics'

            results['interpretation'] = {
                'similarity_gap': float(gap),
                'verdict': verdict,
                'best_routing_type': max(
                    results['per_routing_type'].items(),
                    key=lambda x: x[1]['gap']
                )[0] if results['per_routing_type'] else None,
            }

        return results

    def analyze_context_dependent_routing(
        self, word_contexts: Dict[str, List[str]]
    ) -> Dict:
        """
        Analyze if the same word routes differently in different contexts.

        Validates DAWN's claim of context-dependent dynamic routing.

        Args:
            word_contexts: Dict mapping word → list of sentences containing it

        Returns:
            Per-word routing variance (higher = more context-dependent)
        """
        if self.tokenizer is None:
            return {'error': 'tokenizer required'}

        results = {}

        for word, sentences in word_contexts.items():
            if len(sentences) < 2:
                continue

            word_paths = []

            for sent in sentences:
                ids = self.tokenizer.encode(sent, add_special_tokens=True)
                tokens = self.tokenizer.convert_ids_to_tokens(ids)

                # Find word position in tokenized sequence
                word_lower = word.lower()
                word_positions = []
                for i, tok in enumerate(tokens):
                    cleaned = tok.lower().replace('##', '').replace('▁', '')
                    if word_lower in cleaned or cleaned in word_lower:
                        word_positions.append(i)

                if not word_positions:
                    continue

                # Get routing at word position
                input_ids = np.array([ids])
                routing = self.extractor.extract_routing(input_ids)

                attn = routing.get('attention', {})
                know = routing.get('knowledge', {})
                all_weights = {**attn, **know}

                all_keys = list(ROUTING_KEYS.keys()) + list(KNOWLEDGE_ROUTING_KEYS.keys())
                word_routing = {}

                pos = word_positions[0]
                for key in all_keys:
                    raw_key = WEIGHT_KEY_MAP.get(key, key)
                    w = all_weights.get(raw_key)
                    if w is None:
                        continue
                    if w.ndim == 3 and pos < w.shape[1]:
                        word_routing[key] = w[0, pos]  # [N]
                    elif w.ndim == 2:
                        word_routing[key] = w[0]  # [N]

                if word_routing:
                    word_paths.append(word_routing)

            if len(word_paths) < 2:
                continue

            # Compute variance across contexts
            variances = {}
            for key in word_paths[0]:
                key_vecs = [wp[key] for wp in word_paths if key in wp]
                if len(key_vecs) >= 2:
                    stacked = np.stack(key_vecs)
                    variance = float(stacked.var(axis=0).mean())
                    variances[key] = variance

            attn_vars = [v for k, v in variances.items() if k not in ('fknow', 'rknow')]
            know_vars = [v for k, v in variances.items() if k in ('fknow', 'rknow')]

            results[word] = {
                'n_contexts': len(word_paths),
                'routing_variance': variances,
                'avg_variance': float(np.mean(list(variances.values()))) if variances else 0,
                'attention_variance': float(np.mean(attn_vars)) if attn_vars else 0,
                'knowledge_variance': float(np.mean(know_vars)) if know_vars else 0,
            }

        # Summary
        if results:
            word_results = {k: v for k, v in results.items() if k != 'summary'}
            avg_var = np.mean([r['avg_variance'] for r in word_results.values()])
            attn_var = np.mean([r['attention_variance'] for r in word_results.values()])
            know_var = np.mean([r['knowledge_variance'] for r in word_results.values()])

            if avg_var > 0.1:
                interpretation = 'HIGH: Strong context-dependent routing'
            elif avg_var > 0.01:
                interpretation = 'MODERATE: Some context sensitivity'
            else:
                interpretation = 'LOW: Routing mostly context-independent'

            results['summary'] = {
                'overall_context_variance': float(avg_var),
                'attention_context_variance': float(attn_var),
                'knowledge_context_variance': float(know_var),
                'interpretation': interpretation,
                'more_context_sensitive': 'knowledge' if know_var > attn_var else 'attention',
            }

        return results

    def analyze_neuron_token_heatmap(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 30,
        batch_size: int = 32,
        seq_len: int = 512,
        top_k_neurons: int = 20,
    ) -> Dict:
        """
        Generate neuron-token activation heatmap data.

        Shows which neurons activate most for which tokens.

        Args:
            val_tokens: Flat validation token array
            n_batches: Number of batches
            batch_size: Batch size
            seq_len: Sequence length
            top_k_neurons: Number of top neurons per routing key

        Returns:
            Per-routing-key top neuron → top token mappings
        """
        if self.tokenizer is None:
            return {'error': 'tokenizer required'}

        batches = create_batches(val_tokens, batch_size, seq_len)[:n_batches]
        vocab_size = self.tokenizer.vocab_size

        all_keys = list(ROUTING_KEYS.keys()) + list(KNOWLEDGE_ROUTING_KEYS.keys())

        # {routing_key: np.array[vocab_size, N]}
        token_neuron_sums = {}
        neuron_sizes = {}

        for batch in tqdm(batches, desc='Neuron-Token Heatmap'):
            input_ids = np.array(batch)
            B, S = input_ids.shape

            routing = self.extractor.extract_routing(input_ids)
            attn = routing.get('attention', {})
            know = routing.get('knowledge', {})
            all_weights = {**attn, **know}

            for key in all_keys:
                raw_key = WEIGHT_KEY_MAP.get(key, key)
                w = all_weights.get(raw_key)
                if w is None:
                    continue

                if w.ndim == 2:
                    w = np.broadcast_to(w[:, np.newaxis, :], (B, S, w.shape[-1]))

                if w.ndim != 3:
                    continue

                N = w.shape[-1]

                if key not in token_neuron_sums:
                    token_neuron_sums[key] = np.zeros((vocab_size, N), dtype=np.float32)
                    neuron_sizes[key] = N

                # Accumulate: for each (b, s), add w[b,s,:] to sums[input_ids[b,s], :]
                flat_ids = input_ids.reshape(-1)
                flat_w = w.reshape(-1, N)

                # Skip special tokens (ids < 104 for BERT-style)
                valid = flat_ids >= 104
                if valid.any():
                    valid_ids = flat_ids[valid]
                    valid_w = flat_w[valid]
                    np.add.at(token_neuron_sums[key], valid_ids, valid_w)

        # Build results
        results = {}
        for key, sums in token_neuron_sums.items():
            N = neuron_sizes[key]
            neuron_totals = sums.sum(axis=0)
            top_neurons = np.argsort(neuron_totals)[-min(top_k_neurons, N):][::-1]

            neuron_results = {}
            for nid in top_neurons:
                activations = sums[:, nid]
                nonzero = np.nonzero(activations)[0]
                if len(nonzero) == 0:
                    continue

                top_k = min(10, len(nonzero))
                top_indices = nonzero[np.argsort(activations[nonzero])[-top_k:][::-1]]

                top_tokens = {}
                for tid in top_indices:
                    token = self.tokenizer.convert_ids_to_tokens([int(tid)])[0]
                    top_tokens[token] = float(activations[tid])

                neuron_results[int(nid)] = {
                    'total_activation': float(neuron_totals[nid]),
                    'top_tokens': top_tokens,
                }

            display = ROUTING_KEYS.get(key, KNOWLEDGE_ROUTING_KEYS.get(key, (key,)))[0]
            results[key] = {
                'display': display,
                'neurons': neuron_results,
            }

        return results

    def get_default_sentence_pairs(self) -> List[Tuple[str, str, str]]:
        """Default test sentence pairs (same as PyTorch version)."""
        return [
            ("The cat sat on the mat.", "A feline rested on the rug.", "similar"),
            ("She bought a new car.", "She purchased a new vehicle.", "similar"),
            ("The weather is beautiful today.", "It's a lovely day outside.", "similar"),
            ("He runs every morning.", "He jogs each day at dawn.", "similar"),
            ("The book was interesting.", "The novel was captivating.", "similar"),
            ("I need to go to the store.", "I have to visit the shop.", "similar"),
            ("The children played in the park.", "Kids were playing at the playground.", "similar"),
            ("She cooked dinner for the family.", "She prepared a meal for her relatives.", "similar"),

            ("The cat sat on the mat.", "Stock prices rose sharply.", "different"),
            ("She bought a new car.", "The experiment failed completely.", "different"),
            ("The weather is beautiful today.", "Binary search has O(log n) complexity.", "different"),
            ("He runs every morning.", "The painting was sold at auction.", "different"),
            ("The book was interesting.", "Photosynthesis requires sunlight.", "different"),
            ("I need to go to the store.", "The treaty was signed in 1945.", "different"),
            ("The children played in the park.", "The server crashed unexpectedly.", "different"),
            ("She cooked dinner for the family.", "Quantum entanglement is mysterious.", "different"),
        ]

    def get_default_word_contexts(self) -> Dict[str, List[str]]:
        """Default polysemous word contexts (same as PyTorch version)."""
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

    def run_all(
        self,
        val_tokens: np.ndarray = None,
        output_dir: str = './semantic_analysis',
        n_batches: int = 50,
    ) -> Dict:
        """Run all semantic analyses."""
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        # 1. Path similarity
        print("\n  [1/3] Semantic Path Similarity...")
        pairs = self.get_default_sentence_pairs()
        results['path_similarity'] = self.analyze_semantic_path_similarity(pairs)

        # 2. Context-dependent routing
        print("\n  [2/3] Context-dependent Routing...")
        word_contexts = self.get_default_word_contexts()
        results['context_routing'] = self.analyze_context_dependent_routing(word_contexts)

        # 3. Neuron-token heatmap
        if val_tokens is not None:
            print(f"\n  [3/3] Neuron-Token Heatmap ({n_batches} batches)...")
            results['neuron_heatmap'] = self.analyze_neuron_token_heatmap(
                val_tokens, n_batches=n_batches
            )
        else:
            print("\n  [3/3] Skipping heatmap (no val_tokens)")

        # Save
        import json
        output_path = os.path.join(output_dir, 'semantic_analysis.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results
