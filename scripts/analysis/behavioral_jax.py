"""
Behavioral Analysis - JAX Version
===================================
Analyze token-level behavioral patterns in DAWN v17.1 models on JAX/TPU.

Includes:
- Token trajectory analysis (routing entropy by position)
- Probing classifier for POS prediction from routing weights
- Trajectory visualization

NOTE: This is a JAX port of behavioral.py.
"""

import os
import gc
import numpy as np
from typing import Dict, Optional, List
from collections import defaultdict

from .base_jax import BaseAnalyzerJAX
from .utils_jax import (
    ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS,
    create_batches, JAXRoutingDataExtractor,
    HAS_JAX,
)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x


def _simple_pos_tag(token: str) -> str:
    """Simple rule-based POS tagger (no spaCy dependency)."""
    t = token.strip().lower().replace('##', '').replace('▁', '')
    if not t:
        return 'X'

    # Punctuation
    if all(c in '.,;:!?"\'-()[]{}' for c in t):
        return 'PUNCT'

    # Numbers
    if t.replace('.', '').replace(',', '').isdigit():
        return 'NUM'

    # Determiners
    if t in ('the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your',
             'his', 'her', 'its', 'our', 'their', 'some', 'any', 'no', 'every'):
        return 'DET'

    # Pronouns
    if t in ('i', 'me', 'you', 'he', 'him', 'she', 'her', 'it', 'we', 'us',
             'they', 'them', 'who', 'what', 'which', 'whom'):
        return 'PRON'

    # Prepositions
    if t in ('in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of',
             'about', 'into', 'through', 'during', 'before', 'after',
             'above', 'below', 'between', 'under', 'over'):
        return 'ADP'

    # Conjunctions
    if t in ('and', 'or', 'but', 'nor', 'yet', 'so', 'for', 'because',
             'although', 'while', 'if', 'when', 'that', 'whether'):
        return 'CONJ'

    # Auxiliaries
    if t in ('is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
             'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
             'shall', 'should', 'may', 'might', 'can', 'could', 'must'):
        return 'AUX'

    # Adverbs (common suffixes)
    if t.endswith('ly') and len(t) > 3:
        return 'ADV'

    # Adjectives (common suffixes)
    if any(t.endswith(s) for s in ('ful', 'ous', 'ive', 'able', 'ible', 'al', 'ial', 'ical')):
        return 'ADJ'

    # Verbs (common suffixes)
    if any(t.endswith(s) for s in ('ing', 'ed', 'ize', 'ise', 'ate')):
        return 'VERB'

    # Nouns (common suffixes)
    if any(t.endswith(s) for s in ('tion', 'sion', 'ment', 'ness', 'ity', 'ism', 'ist', 'er', 'or')):
        return 'NOUN'

    # Default
    return 'NOUN'


def _calc_entropy_ratio(weights: np.ndarray) -> float:
    """Compute entropy ratio of routing weights as percentage of max."""
    if weights.size == 0:
        return 0.0

    if weights.ndim > 1:
        probs = weights.mean(axis=tuple(range(weights.ndim - 1)))
    else:
        probs = weights

    probs = np.maximum(probs, 0)
    total = probs.sum()
    if total <= 0:
        return 0.0
    probs = probs / total

    # Filter zeros
    mask = probs > 0
    if not mask.any():
        return 0.0

    entropy = -np.sum(probs[mask] * np.log(probs[mask]))
    max_entropy = np.log(len(probs))
    return float(entropy / max_entropy * 100) if max_entropy > 0 else 0.0


class BehavioralAnalyzerJAX(BaseAnalyzerJAX):
    """Token-level behavioral analyzer (JAX version)."""

    def __init__(self, model, params, config: Dict, tokenizer=None):
        super().__init__(model, params, config, tokenizer=tokenizer)

    def analyze_token_trajectory(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 20,
        batch_size: int = 32,
        seq_len: int = 512,
        max_positions: int = 128,
    ) -> Dict:
        """
        Analyze routing entropy across sequence positions.

        Args:
            val_tokens: Flat validation token array
            n_batches: Number of batches
            batch_size: Batch size
            seq_len: Sequence length
            max_positions: Max positions to track

        Returns:
            Position-wise entropy statistics
        """
        batches = create_batches(val_tokens, batch_size, seq_len)[:n_batches]

        # {routing_key: {position: [entropy_values]}}
        position_routing = defaultdict(lambda: defaultdict(list))

        all_keys = list(ROUTING_KEYS.keys()) + list(KNOWLEDGE_ROUTING_KEYS.keys())

        for batch in tqdm(batches, desc='Trajectory'):
            input_ids = np.array(batch)
            routing_data = self.extractor.extract_routing(input_ids)

            attn = routing_data.get('attention', {})
            know = routing_data.get('knowledge', {})
            all_weights = {**attn, **know}

            for key in all_keys:
                # Map standard key to raw key
                from .utils_jax import WEIGHT_KEY_MAP
                raw_key = WEIGHT_KEY_MAP.get(key, key)
                w = all_weights.get(raw_key)
                if w is None:
                    continue

                if w.ndim == 3:  # [B, S, N]
                    for pos in range(min(w.shape[1], max_positions)):
                        ent = _calc_entropy_ratio(w[:, pos, :])
                        position_routing[key][pos].append(ent)
                elif w.ndim == 2:  # [B, N]
                    ent = _calc_entropy_ratio(w)
                    for pos in range(max_positions):
                        position_routing[key][pos].append(ent)

        # Build results
        results = {}
        for key, pos_data in position_routing.items():
            if not pos_data:
                continue

            pos_avg = {pos: float(np.mean(vals)) for pos, vals in pos_data.items()}
            early = [v for p, v in pos_avg.items() if p < 10]
            late = [v for p, v in pos_avg.items() if p >= 10]

            display = ROUTING_KEYS.get(key, KNOWLEDGE_ROUTING_KEYS.get(key, (key,)))[0]

            results[key] = {
                'display': display,
                'position_entropy': pos_avg,
                'early_avg': float(np.mean(early)) if early else 0,
                'late_avg': float(np.mean(late)) if late else 0,
            }

        return results

    def run_probing(
        self,
        val_tokens: np.ndarray,
        n_batches: int = 50,
        batch_size: int = 32,
        seq_len: int = 512,
        max_samples: int = 20000,
    ) -> Dict:
        """
        Probing classifier: predict POS from routing weights.

        Demonstrates that routing weights encode linguistic properties,
        validating DAWN's claim of meaningful neuron specialization.

        Args:
            val_tokens: Flat validation token array
            n_batches: Number of batches to process
            batch_size: Batch size
            seq_len: Sequence length
            max_samples: Max samples to collect

        Returns:
            Per-routing-key probing accuracy
        """
        if not HAS_SKLEARN:
            return {'error': 'sklearn not available'}

        if self.tokenizer is None:
            return {'error': 'tokenizer required for probing'}

        batches = create_batches(val_tokens, batch_size, seq_len)[:n_batches]

        # Collect {routing_key: [(weight_vector, pos_label), ...]}
        from .utils_jax import WEIGHT_KEY_MAP
        all_keys = list(ROUTING_KEYS.keys()) + list(KNOWLEDGE_ROUTING_KEYS.keys())
        samples = {key: {'X': [], 'y': []} for key in all_keys}
        total_collected = 0

        for batch in tqdm(batches, desc='Probing data'):
            if total_collected >= max_samples:
                break

            input_ids = np.array(batch)
            B, S = input_ids.shape

            routing_data = self.extractor.extract_routing(input_ids)
            attn = routing_data.get('attention', {})
            know = routing_data.get('knowledge', {})
            all_weights = {**attn, **know}

            # Get POS labels for all tokens in batch
            for b in range(B):
                if total_collected >= max_samples:
                    break

                token_ids = input_ids[b].tolist()
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
                pos_labels = [_simple_pos_tag(t) for t in tokens]

                for key in all_keys:
                    raw_key = WEIGHT_KEY_MAP.get(key, key)
                    w = all_weights.get(raw_key)
                    if w is None:
                        continue

                    for s in range(min(S, len(pos_labels))):
                        if pos_labels[s] == 'X':
                            continue

                        if w.ndim == 3:  # [B, S, N]
                            vec = w[b, s]
                        elif w.ndim == 2:  # [B, N]
                            vec = w[b]
                        else:
                            continue

                        samples[key]['X'].append(vec)
                        samples[key]['y'].append(pos_labels[s])

                total_collected += S

        # Train classifiers
        results = {'per_routing_key': {}}
        all_accuracies = []

        for key in all_keys:
            X_list = samples[key]['X']
            y_list = samples[key]['y']

            if len(X_list) < 100:
                continue

            X = np.array(X_list)
            y = np.array(y_list)

            n_classes = len(np.unique(y))
            if n_classes < 2:
                continue

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                solver = 'saga' if len(X_train) > 10000 else 'lbfgs'
                clf = LogisticRegression(max_iter=500, random_state=42, solver=solver, n_jobs=-1)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)

                display = ROUTING_KEYS.get(key, KNOWLEDGE_ROUTING_KEYS.get(key, (key,)))[0]

                results['per_routing_key'][key] = {
                    'display': display,
                    'accuracy': float(accuracy),
                    'n_samples': len(X),
                    'n_classes': n_classes,
                }
                all_accuracies.append(accuracy)

            except Exception as e:
                results['per_routing_key'][key] = {'error': str(e)}

            # Free memory
            del X_list, y_list
            samples[key] = {'X': [], 'y': []}
            gc.collect()

        if all_accuracies:
            results['overall'] = {
                'mean_accuracy': float(np.mean(all_accuracies)),
                'std_accuracy': float(np.std(all_accuracies)),
                'max_accuracy': float(np.max(all_accuracies)),
                'min_accuracy': float(np.min(all_accuracies)),
                'n_classifiers': len(all_accuracies),
            }

        return results

    def visualize_trajectory(self, trajectory_results: Dict, output_dir: str) -> Optional[str]:
        """Visualize entropy by position."""
        if not HAS_MATPLOTLIB:
            return None

        os.makedirs(output_dir, exist_ok=True)

        routing_keys = [k for k in trajectory_results
                        if k in ROUTING_KEYS or k in KNOWLEDGE_ROUTING_KEYS]
        if not routing_keys:
            return None

        n_cols = 3
        n_rows = (len(routing_keys) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for ax_idx, key in enumerate(routing_keys):
            if ax_idx >= len(axes):
                break

            data = trajectory_results[key]
            pos_ent = data.get('position_entropy', {})
            if not pos_ent:
                continue

            positions = sorted(pos_ent.keys())
            entropies = [pos_ent[p] for p in positions]

            axes[ax_idx].plot(positions, entropies, '-o', markersize=2)
            axes[ax_idx].set_xlabel('Position')
            axes[ax_idx].set_ylabel('Entropy (%)')
            axes[ax_idx].set_title(f'{data["display"]} Entropy by Position')
            axes[ax_idx].axhline(y=data['early_avg'], color='r', linestyle='--',
                                  alpha=0.5, label=f'Early: {data["early_avg"]:.1f}')
            axes[ax_idx].axhline(y=data['late_avg'], color='b', linestyle='--',
                                  alpha=0.5, label=f'Late: {data["late_avg"]:.1f}')
            axes[ax_idx].legend(fontsize=8)

        for i in range(len(routing_keys), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        path = os.path.join(output_dir, 'trajectory.png')
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def run_all(
        self,
        val_tokens: np.ndarray,
        output_dir: str = './behavioral_analysis',
        n_batches: int = 20,
    ) -> Dict:
        """Run all behavioral analyses."""
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        # Trajectory
        print("  Running trajectory analysis...")
        try:
            results['trajectory'] = self.analyze_token_trajectory(val_tokens, n_batches)
        except Exception as e:
            print(f"  ERROR in trajectory: {e}")
            results['trajectory'] = {'error': str(e)}

        # Probing
        print("  Running probing analysis...")
        try:
            results['probing'] = self.run_probing(val_tokens, n_batches * 2)
        except Exception as e:
            print(f"  ERROR in probing: {e}")
            results['probing'] = {'error': str(e)}

        # Visualization
        try:
            viz_path = self.visualize_trajectory(results.get('trajectory', {}), output_dir)
            if viz_path:
                results['trajectory_visualization'] = viz_path
        except Exception as e:
            print(f"  ERROR in visualization: {e}")

        # Save
        import json
        results_path = os.path.join(output_dir, 'behavioral_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results
