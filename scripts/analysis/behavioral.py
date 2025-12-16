"""
Behavioral Analysis
===================
Analyze token-level behavioral patterns in DAWN v17.1 models.

Includes:
- Single neuron analysis
- Token trajectory analysis (routing by position)
- Probing classifier for POS prediction
- Ablation studies
"""

import os
import numpy as np
import torch
from typing import Dict, Optional
from collections import defaultdict

from .utils import (
    NEURON_TYPES, ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS,
    calc_entropy_ratio, simple_pos_tag,
    get_batch_input_ids, get_routing_from_outputs,
    HAS_MATPLOTLIB, HAS_SKLEARN, HAS_TQDM, tqdm, plt
)

if HAS_SKLEARN:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score


class BehavioralAnalyzer:
    """Token-level behavioral analyzer."""

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

    def analyze_single_neuron(self, neuron_id: int, neuron_type: str) -> Dict:
        """
        Analyze a single neuron's properties.

        Args:
            neuron_id: Index of the neuron
            neuron_type: Type of neuron (e.g., 'feature_qk')

        Returns:
            Dictionary with neuron properties
        """
        results = {
            'neuron_type': neuron_type,
            'neuron_id': neuron_id,
        }

        # Get EMA and excitability
        type_info = NEURON_TYPES.get(neuron_type)
        if type_info:
            ema_attr = type_info[1]
            if hasattr(self.router, ema_attr):
                ema = getattr(self.router, ema_attr)
                if neuron_id < len(ema):
                    results['usage_ema'] = float(ema[neuron_id])
                    tau = self.router.tau
                    exc = max(0, min(1, 1.0 - ema[neuron_id].item() / tau))
                    results['excitability'] = float(exc)

        # Get embedding properties
        emb = self.router.neuron_emb.detach().cpu().numpy()

        offset = 0
        for name, (_, _, n_attr, _) in NEURON_TYPES.items():
            if hasattr(self.router, n_attr):
                n = getattr(self.router, n_attr)
                if name == neuron_type:
                    if neuron_id < n:
                        neuron_emb = emb[offset + neuron_id]
                        results['embedding_norm'] = float(np.linalg.norm(neuron_emb))
                        results['embedding_mean'] = float(neuron_emb.mean())
                        results['embedding_std'] = float(neuron_emb.std())
                    break
                offset += n

        return results

    def analyze_token_trajectory(self, dataloader, n_batches: int = 20, layer_idx: int = None) -> Dict:
        """
        Analyze how routing entropy changes across sequence positions for ALL layers.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process
            layer_idx: Specific layer to analyze (None = aggregate all layers)

        Returns:
            Dictionary with position-wise entropy statistics per layer
        """
        # {layer: {routing_key: {position: [entropy_values]}}}
        layer_position_routing = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Trajectory')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                outputs = self.model(input_ids, return_routing_info=True)
                routing_infos = get_routing_from_outputs(outputs)

                if routing_infos is None:
                    continue

                # Process ALL layers
                for lidx, layer_info in enumerate(routing_infos):
                    if layer_idx is not None and lidx != layer_idx:
                        continue

                    attn = layer_info.get('attention', {})

                    for key, (_, pref_key, _, _) in ROUTING_KEYS.items():
                        pref = attn.get(pref_key)
                        if pref is None:
                            continue

                        if pref.dim() == 3:  # [B, S, N] token-level
                            for pos in range(min(pref.shape[1], 128)):
                                ent = calc_entropy_ratio(pref[:, pos, :])
                                layer_position_routing[lidx][key][pos].append(ent)
                        elif pref.dim() == 2:  # [B, N] batch-level - same for all positions
                            ent = calc_entropy_ratio(pref)
                            for pos in range(128):
                                layer_position_routing[lidx][key][pos].append(ent)

        # Build per-layer results
        results = {'per_layer': {}}

        # Aggregate data for overall results
        aggregated_routing = defaultdict(lambda: defaultdict(list))

        for lidx, position_routing in layer_position_routing.items():
            layer_results = {}
            for key in ROUTING_KEYS.keys():
                if position_routing[key]:
                    pos_avg = {}
                    for pos, values in position_routing[key].items():
                        pos_avg[pos] = float(np.mean(values))
                        aggregated_routing[key][pos].extend(values)

                    early_positions = [v for p, v in pos_avg.items() if p < 10]
                    late_positions = [v for p, v in pos_avg.items() if p >= 10]

                    layer_results[key] = {
                        'display': ROUTING_KEYS[key][0],
                        'position_entropy': pos_avg,
                        'early_avg': float(np.mean(early_positions)) if early_positions else 0,
                        'late_avg': float(np.mean(late_positions)) if late_positions else 0,
                    }

            if layer_results:
                results['per_layer'][f'L{lidx}'] = layer_results

        # Aggregated results (backward compatibility)
        for key in ROUTING_KEYS.keys():
            if aggregated_routing[key]:
                pos_avg = {}
                for pos, values in aggregated_routing[key].items():
                    pos_avg[pos] = float(np.mean(values))

                early_positions = [v for p, v in pos_avg.items() if p < 10]
                late_positions = [v for p, v in pos_avg.items() if p >= 10]

                results[key] = {
                    'display': ROUTING_KEYS[key][0],
                    'position_entropy': pos_avg,
                    'early_avg': float(np.mean(early_positions)) if early_positions else 0,
                    'late_avg': float(np.mean(late_positions)) if late_positions else 0,
                }

        results['n_layers'] = len(layer_position_routing)
        return results

    def run_probing(self, dataloader, max_batches: int = 50, layer_idx: int = None) -> Dict:
        """
        Run probing classifier for POS prediction across ALL layers.

        Uses routing weights to predict part-of-speech tags.

        Args:
            dataloader: DataLoader for input data
            max_batches: Maximum batches to process
            layer_idx: Specific layer to analyze (None = all layers aggregated)

        Returns:
            Dictionary with probing accuracy per layer/routing key
        """
        if not HAS_SKLEARN:
            return {'error': 'sklearn not available'}

        # {layer_key: {routing_key: [features]}}
        X_data = defaultdict(lambda: defaultdict(list))
        y_labels = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc='Probing', total=max_batches)):
                if batch_idx >= max_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)

                # Get attention mask if available
                if isinstance(batch, dict):
                    attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(self.device)
                else:
                    attention_mask = torch.ones_like(input_ids)

                try:
                    outputs = self.model(input_ids, return_routing_info=True)
                    routing_infos = get_routing_from_outputs(outputs)
                    if routing_infos is None:
                        continue
                except Exception:
                    continue

                # Process ALL layers
                for lidx, layer_info in enumerate(routing_infos):
                    if layer_idx is not None and lidx != layer_idx:
                        continue

                    layer_key = f'L{lidx}'
                    attn = layer_info.get('attention', {})
                    knowledge = layer_info.get('knowledge', {})

                    # Collect attention routing weights
                    for key, (_, _, weight_key, _) in ROUTING_KEYS.items():
                        if weight_key in attn:
                            w = attn[weight_key]
                            if w.dim() == 3:  # [B, S, N] token-level
                                for b in range(w.shape[0]):
                                    for s in range(w.shape[1]):
                                        if attention_mask[b, s] == 0:
                                            continue
                                        X_data[layer_key][key].append(w[b, s].cpu().numpy())
                            elif w.dim() == 2:  # [B, N] batch-level
                                for b in range(w.shape[0]):
                                    for s in range(attention_mask.shape[1]):
                                        if attention_mask[b, s] == 0:
                                            continue
                                        X_data[layer_key][key].append(w[b].cpu().numpy())

                    # Collect knowledge routing weights
                    for key, (_, weight_key, _) in KNOWLEDGE_ROUTING_KEYS.items():
                        if weight_key in knowledge:
                            w = knowledge[weight_key]
                            if w.dim() == 3:  # [B, S, N] token-level
                                for b in range(w.shape[0]):
                                    for s in range(w.shape[1]):
                                        if attention_mask[b, s] == 0:
                                            continue
                                        X_data[layer_key][key].append(w[b, s].cpu().numpy())
                            elif w.dim() == 2:  # [B, N] batch-level
                                for b in range(w.shape[0]):
                                    for s in range(attention_mask.shape[1]):
                                        if attention_mask[b, s] == 0:
                                            continue
                                        X_data[layer_key][key].append(w[b].cpu().numpy())

                    # Collect POS labels for this layer
                    for b in range(input_ids.shape[0]):
                        for s in range(input_ids.shape[1]):
                            if attention_mask[b, s] == 0:
                                continue
                            token = self.tokenizer.decode([input_ids[b, s].item()])
                            pos = simple_pos_tag(token)
                            y_labels[layer_key].append(pos)

        # Train and evaluate classifiers per layer/routing key
        results = {'per_layer': {}}

        for layer_key in X_data.keys():
            results['per_layer'][layer_key] = {}

            for routing_key in X_data[layer_key].keys():
                X = np.array(X_data[layer_key][routing_key])
                y = np.array(y_labels[layer_key][:len(X)])

                if len(X) < 100 or len(np.unique(y)) < 2:
                    results['per_layer'][layer_key][routing_key] = {'error': 'Not enough data'}
                    continue

                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    clf = LogisticRegression(max_iter=1000, random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)

                    # Get display name
                    if routing_key in ROUTING_KEYS:
                        display = ROUTING_KEYS[routing_key][0]
                    elif routing_key in KNOWLEDGE_ROUTING_KEYS:
                        display = KNOWLEDGE_ROUTING_KEYS[routing_key][0]
                    else:
                        display = routing_key

                    results['per_layer'][layer_key][routing_key] = {
                        'display': f'{layer_key}/{display}',
                        'accuracy': float(accuracy),
                        'n_samples': len(X),
                        'n_classes': len(np.unique(y)),
                    }
                except Exception as e:
                    results['per_layer'][layer_key][routing_key] = {'error': str(e)}

        # Summary: best accuracy per routing type across layers
        results['summary'] = {}
        all_routing_keys = set()
        for layer_data in results['per_layer'].values():
            all_routing_keys.update(layer_data.keys())

        for rkey in all_routing_keys:
            accuracies = []
            for layer_key, layer_data in results['per_layer'].items():
                if rkey in layer_data and 'accuracy' in layer_data[rkey]:
                    accuracies.append(layer_data[rkey]['accuracy'])
            if accuracies:
                results['summary'][rkey] = {
                    'max_accuracy': max(accuracies),
                    'mean_accuracy': float(np.mean(accuracies)),
                }

        return results

    def run_ablation(self, dataloader, neuron_type: str, neuron_ids: list,
                     n_batches: int = 20) -> Dict:
        """
        Run ablation study by zeroing out specific neurons.

        Args:
            dataloader: DataLoader for input data
            neuron_type: Type of neurons to ablate
            neuron_ids: List of neuron IDs to ablate
            n_batches: Number of batches to process

        Returns:
            Dictionary with loss before and after ablation
        """
        import torch.nn.functional as F

        results = {
            'neuron_type': neuron_type,
            'ablated_neurons': neuron_ids,
        }

        # Compute baseline loss
        baseline_losses = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Baseline')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                outputs = self.model(input_ids)

                if isinstance(outputs, tuple):
                    logits = outputs[1] if len(outputs) > 1 else outputs[0]
                else:
                    logits = outputs

                # Compute CLM loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                baseline_losses.append(loss.item())

        results['baseline_loss'] = float(np.mean(baseline_losses))

        # Note: Actual ablation requires modifying the model forward pass
        # This is a placeholder for the ablation experiment
        results['note'] = 'Full ablation requires model modification. See model code for implementation.'

        return results

    def visualize_trajectory(self, trajectory_results: Dict, output_dir: str) -> Optional[str]:
        """
        Visualize token trajectory results.

        Args:
            trajectory_results: Results from analyze_token_trajectory()
            output_dir: Directory for output

        Returns:
            Path to visualization or None
        """
        if not HAS_MATPLOTLIB:
            return None

        os.makedirs(output_dir, exist_ok=True)

        n_keys = len([k for k in trajectory_results if k in ROUTING_KEYS])
        if n_keys == 0:
            return None

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        ax_idx = 0
        for key, data in trajectory_results.items():
            if key not in ROUTING_KEYS or ax_idx >= len(axes):
                continue

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
                                 alpha=0.5, label=f'Early avg: {data["early_avg"]:.1f}')
            axes[ax_idx].axhline(y=data['late_avg'], color='b', linestyle='--',
                                 alpha=0.5, label=f'Late avg: {data["late_avg"]:.1f}')
            axes[ax_idx].legend()
            ax_idx += 1

        for i in range(ax_idx, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        path = os.path.join(output_dir, 'trajectory.png')
        plt.savefig(path, dpi=150)
        plt.close()

        return path

    def run_all(self, dataloader, output_dir: str = './behavioral_analysis', n_batches: int = 20) -> Dict:
        """
        Run all behavioral analyses.

        Args:
            dataloader: DataLoader for input data
            output_dir: Directory for outputs
            n_batches: Number of batches to process

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'trajectory': self.analyze_token_trajectory(dataloader, n_batches),
            'probing': self.run_probing(dataloader, n_batches * 2),
        }

        # Visualization
        viz_path = self.visualize_trajectory(results['trajectory'], output_dir)
        if viz_path:
            results['trajectory_visualization'] = viz_path

        return results
