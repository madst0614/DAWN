"""
Routing Pattern Analysis
=========================
Analyze routing patterns in DAWN v17.1 models.

Includes:
- Routing entropy analysis
- Selection frequency analysis
- Selection diversity analysis
- Q/K overlap analysis
- Q/K usage pattern analysis (from analyze_dawn_qk.py)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from collections import Counter, defaultdict

from .utils import (
    NEURON_TYPES, ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS, QK_POOLS,
    calc_entropy_ratio, gini_coefficient,
    get_batch_input_ids, get_routing_from_outputs,
    HAS_MATPLOTLIB, HAS_TQDM, tqdm, plt
)


class RoutingAnalyzer:
    """Routing pattern analyzer for DAWN v17.1."""

    def __init__(self, model, router, device='cuda'):
        """
        Initialize analyzer.

        Args:
            model: DAWN model
            router: NeuronRouter instance
            device: Device for computation
        """
        self.model = model
        self.router = router
        self.device = device

    def _enable_pref_tensors(self):
        """Enable store_pref_tensors for detailed analysis (v18.2+)."""
        if hasattr(self.model, 'router') and hasattr(self.model.router, 'store_pref_tensors'):
            self.model.router.store_pref_tensors = True

    def _disable_pref_tensors(self):
        """Disable store_pref_tensors after analysis."""
        if hasattr(self.model, 'router') and hasattr(self.model.router, 'store_pref_tensors'):
            self.model.router.store_pref_tensors = False

    def analyze_entropy(self, dataloader, n_batches: int = 50) -> Dict:
        """
        Analyze routing entropy across batches.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with entropy statistics per routing key
        """
        entropy_data = {name: [] for name in ROUTING_KEYS.keys()}

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Entropy')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                outputs = self.model(input_ids, return_routing_info=True)
                routing_infos = get_routing_from_outputs(outputs)

                if routing_infos is None:
                    continue

                for layer_info in routing_infos:
                    attn = layer_info.get('attention', {})

                    for key, (display, pref_key, _, _) in ROUTING_KEYS.items():
                        pref = attn.get(pref_key)
                        if pref is not None:
                            ent = calc_entropy_ratio(pref)
                            entropy_data[key].append(ent)

        results = {}
        for key, (display, _, _, pool) in ROUTING_KEYS.items():
            if entropy_data[key]:
                results[key] = {
                    'display': display,
                    'pool': pool,
                    'mean_entropy': float(np.mean(entropy_data[key])),
                    'std_entropy': float(np.std(entropy_data[key])),
                    'min_entropy': float(np.min(entropy_data[key])),
                    'max_entropy': float(np.max(entropy_data[key])),
                }

        return results

    def analyze_selection_frequency(self, dataloader, n_batches: int = 50) -> Dict:
        """
        Analyze neuron selection frequency.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with selection frequency statistics
        """
        selection_counts = {name: Counter() for name in ROUTING_KEYS.keys()}

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Selection')):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)
                outputs = self.model(input_ids, return_routing_info=True)
                routing_infos = get_routing_from_outputs(outputs)

                if routing_infos is None:
                    continue

                for layer_info in routing_infos:
                    attn = layer_info.get('attention', {})

                    for key, (_, pref_key, _, _) in ROUTING_KEYS.items():
                        pref = attn.get(pref_key)
                        if pref is None:
                            continue

                        k = min(8, pref.shape[-1])
                        _, topk_idx = torch.topk(pref, k, dim=-1)
                        flat_idx = topk_idx.view(-1).cpu().numpy()

                        for idx in flat_idx:
                            selection_counts[key][int(idx)] += 1

        results = {}
        for key, (display, _, _, pool) in ROUTING_KEYS.items():
            counts = selection_counts[key]
            if not counts:
                continue

            total = sum(counts.values())
            top10 = counts.most_common(10)
            unique = len(counts)

            pool_info = NEURON_TYPES.get(pool, {})
            n_attr = pool_info[2] if pool_info else None
            n_total = getattr(self.router, n_attr, 0) if n_attr else 0

            results[key] = {
                'display': display,
                'pool': pool,
                'total_selections': total,
                'unique_selected': unique,
                'coverage': unique / n_total if n_total > 0 else 0,
                'top10': [(idx, cnt, cnt/total) for idx, cnt in top10],
                'concentration': sum(cnt for _, cnt in top10) / total if total > 0 else 0,
            }

        return results

    def analyze_selection_diversity(self, dataloader, n_batches: int = 100) -> Dict:
        """
        Analyze selection diversity across batches.

        Measures how many unique neurons are selected across the entire dataset
        vs per-batch selection.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with diversity metrics
        """
        # Use defaultdict for dynamic layer keys
        union_selected = defaultdict(set)
        per_batch_counts = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc='Selection Diversity', total=n_batches)):
                if i >= n_batches:
                    break

                input_ids = get_batch_input_ids(batch, self.device)

                try:
                    outputs = self.model(input_ids, return_routing_info=True)
                    routing_infos = get_routing_from_outputs(outputs)
                    if routing_infos is None:
                        continue
                except Exception:
                    continue

                # Process ALL layers
                for layer_idx, layer_info in enumerate(routing_infos):
                    attn = layer_info.get('attention', {})
                    knowledge = layer_info.get('knowledge', {})

                    # Attention routing
                    for key, (_, _, weight_key, _) in ROUTING_KEYS.items():
                        layer_key = f'L{layer_idx}/{key}'
                        if weight_key in attn:
                            weights = attn[weight_key]
                            if weights.dim() == 3:
                                selected = (weights > 0).any(dim=0).any(dim=0).cpu()
                                union_selected[layer_key].update(selected.nonzero().flatten().tolist())
                                per_batch_counts[layer_key].append(selected.sum().item())
                            elif weights.dim() == 2:
                                selected = (weights > 0).any(dim=0).cpu()
                                union_selected[layer_key].update(selected.nonzero().flatten().tolist())
                                per_batch_counts[layer_key].append(selected.sum().item())

                    # Knowledge routing
                    for key, (_, weight_key, _) in KNOWLEDGE_ROUTING_KEYS.items():
                        layer_key = f'L{layer_idx}/{key}'
                        if weight_key in knowledge:
                            weights = knowledge[weight_key]
                            if weights.dim() == 3:
                                selected = (weights > 0).any(dim=0).any(dim=0).cpu()
                                union_selected[layer_key].update(selected.nonzero().flatten().tolist())
                                per_batch_counts[layer_key].append(selected.sum().item())
                            elif weights.dim() == 2:
                                selected = (weights > 0).any(dim=0).cpu()
                                union_selected[layer_key].update(selected.nonzero().flatten().tolist())
                                per_batch_counts[layer_key].append(selected.sum().item())

        # Build results for all layer/key combinations
        results = {}

        for layer_key in union_selected.keys():
            # Parse layer_key: L{layer_idx}/{routing_key}
            parts = layer_key.split('/')
            if len(parts) != 2:
                continue

            layer_str, routing_key = parts
            layer_idx = int(layer_str[1:])

            # Get pool info
            if routing_key in ROUTING_KEYS:
                pool = ROUTING_KEYS[routing_key][3]
                display = f'{layer_str}/{ROUTING_KEYS[routing_key][0]}'
            elif routing_key in KNOWLEDGE_ROUTING_KEYS:
                pool = KNOWLEDGE_ROUTING_KEYS[routing_key][2]
                display = f'{layer_str}/{KNOWLEDGE_ROUTING_KEYS[routing_key][0]}'
            else:
                continue

            pool_info = NEURON_TYPES.get(pool, {})
            n_attr = pool_info[2] if pool_info else None
            n_total = getattr(self.router, n_attr, 0) if n_attr else 0

            union_count = len(union_selected[layer_key])
            batch_counts = per_batch_counts[layer_key]

            if len(batch_counts) > 0:
                per_batch_avg = np.mean(batch_counts)
                per_batch_std = np.std(batch_counts)
            else:
                per_batch_avg = 0
                per_batch_std = 0

            results[layer_key] = {
                'display': display,
                'layer': layer_idx,
                'pool': pool,
                'n_total': n_total,
                'per_batch_avg': float(per_batch_avg),
                'per_batch_std': float(per_batch_std),
                'union_count': union_count,
                'union_coverage': float(union_count / n_total) if n_total > 0 else 0,
                'diversity_ratio': float(union_count / per_batch_avg) if per_batch_avg > 0 else 0,
            }

        total_union = sum(len(union_selected[k]) for k in union_selected)
        n_batches_processed = 0
        if per_batch_counts:
            first_key = next(iter(per_batch_counts))
            n_batches_processed = len(per_batch_counts[first_key])

        results['summary'] = {
            'n_batches_processed': min(n_batches, n_batches_processed),
            'n_layers': len(set(k.split('/')[0] for k in union_selected.keys())),
            'total_keys_analyzed': len(union_selected),
            'interpretation': (
                'High diversity_ratio (>2) = many neurons selected differently per batch\n'
                'Low diversity_ratio (~1) = same neurons always selected'
            )
        }

        return results

    def analyze_qk_overlap(self, dataloader, n_batches: int = 50) -> Dict:
        """
        Analyze Q/K routing overlap (Jaccard similarity).
        Optimized with vectorized tensor operations.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with Q/K overlap statistics
        """
        overlap_data = {'fqk': [], 'rqk': []}

        def compute_jaccard_batch(q_prefs, k_prefs, topk=8):
            """Compute Jaccard similarity for batch using GPU tensors."""
            N = q_prefs.shape[-1]
            k = min(topk, N)

            # Flatten if 3D: [B, S, N] -> [B*S, N]
            if q_prefs.dim() == 3:
                B, S, _ = q_prefs.shape
                q_flat = q_prefs.view(-1, N)
                k_flat = k_prefs.view(-1, N)
            else:
                q_flat = q_prefs
                k_flat = k_prefs

            # Get top-k indices
            q_topk_idx = torch.topk(q_flat, k, dim=-1)[1]  # [B*S, k]
            k_topk_idx = torch.topk(k_flat, k, dim=-1)[1]  # [B*S, k]

            # Create one-hot masks
            batch_size = q_flat.shape[0]
            q_mask = torch.zeros(batch_size, N, device=q_prefs.device)
            k_mask = torch.zeros(batch_size, N, device=k_prefs.device)

            # Scatter to create binary masks
            q_mask.scatter_(1, q_topk_idx, 1.0)
            k_mask.scatter_(1, k_topk_idx, 1.0)

            # Compute intersection and union
            intersection = (q_mask * k_mask).sum(dim=-1)  # [B*S]
            union = ((q_mask + k_mask) > 0).float().sum(dim=-1)  # [B*S]

            # Jaccard = intersection / union
            jaccard = intersection / (union + 1e-8)

            return jaccard.cpu().tolist()

        self.model.eval()
        self._enable_pref_tensors()

        try:
            with torch.no_grad():
                for i, batch in enumerate(tqdm(dataloader, total=n_batches, desc='Q/K Overlap')):
                    if i >= n_batches:
                        break

                    input_ids = get_batch_input_ids(batch, self.device)
                    outputs = self.model(input_ids, return_routing_info=True)
                    routing_infos = get_routing_from_outputs(outputs)

                    if routing_infos is None:
                        continue

                    for layer_info in routing_infos:
                        attn = layer_info.get('attention', {})

                        # F-QK Q/K overlap - vectorized
                        fqk_q = attn.get('fqk_q_pref')
                        fqk_k = attn.get('fqk_k_pref')
                        if fqk_q is not None and fqk_k is not None:
                            overlap_data['fqk'].extend(compute_jaccard_batch(fqk_q, fqk_k))

                        # R-QK Q/K overlap - vectorized
                        rqk_q = attn.get('rqk_q_pref')
                        rqk_k = attn.get('rqk_k_pref')
                        if rqk_q is not None and rqk_k is not None:
                            overlap_data['rqk'].extend(compute_jaccard_batch(rqk_q, rqk_k))
        finally:
            self._disable_pref_tensors()

        results = {}
        for key in ['fqk', 'rqk']:
            if overlap_data[key]:
                mean_overlap = np.mean(overlap_data[key])
                results[key] = {
                    'mean_overlap': float(mean_overlap),
                    'std_overlap': float(np.std(overlap_data[key])),
                    'interpretation': (
                        'Q and K select similar neurons' if mean_overlap > 0.3
                        else 'Q and K select different neurons'
                    )
                }

        return results

    def analyze_qk_usage(self, dataloader, n_batches: int = 100, layer_idx: int = None) -> Dict:
        """
        Analyze per-neuron Q/K selection counts across ALL layers.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process
            layer_idx: Specific layer to analyze (None = aggregate all layers)

        Returns:
            Dictionary with Q/K usage statistics
        """
        results = {}
        self.model.eval()
        self._enable_pref_tensors()

        try:
            for pool_name, pool_info in QK_POOLS.items():
                n_neurons = getattr(self.router, pool_info['n_attr'], 0)
                if n_neurons == 0:
                    continue

                # Per-layer counts: {lidx: (q_counts, k_counts, overlaps)}
                layer_data = {}

                with torch.no_grad():
                    for i, batch in enumerate(tqdm(dataloader, desc=f'{pool_info["display"]} Q/K', total=n_batches)):
                        if i >= n_batches:
                            break

                        input_ids = get_batch_input_ids(batch, self.device)

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

                            attn = layer_info.get('attention', {})

                            # Get Q/K weights
                            w_q = attn.get(pool_info['q_weight'])
                            w_k = attn.get(pool_info['k_weight'])

                            if w_q is None or w_k is None:
                                w_q = attn.get(pool_info['q_pref'])
                                w_k = attn.get(pool_info['k_pref'])
                                if w_q is None or w_k is None:
                                    continue

                            # Initialize layer data if needed
                            if lidx not in layer_data:
                                layer_data[lidx] = (
                                    torch.zeros(n_neurons, device=self.device),
                                    torch.zeros(n_neurons, device=self.device),
                                    []
                                )

                            q_counts, k_counts, batch_overlaps = layer_data[lidx]

                            # Count selections
                            if w_q.dim() == 3:  # [B, S, N]
                                selected_q = (w_q > 0).float().sum(dim=[0, 1])
                                selected_k = (w_k > 0).float().sum(dim=[0, 1])
                            else:  # [B, N]
                                selected_q = (w_q > 0).float().sum(dim=0)
                                selected_k = (w_k > 0).float().sum(dim=0)

                            q_counts += selected_q
                            k_counts += selected_k

                            # Calculate batch overlap
                            if w_q.dim() >= 2:
                                overlap = ((w_q > 0) & (w_k > 0)).float()
                                active_q = (w_q > 0).float().sum(-1)
                                overlap_ratio = (overlap.sum(-1) / (active_q + 1e-8)).mean().item()
                                batch_overlaps.append(overlap_ratio)

                            layer_data[lidx] = (q_counts, k_counts, batch_overlaps)

                # Aggregate across all layers
                total_q = torch.zeros(n_neurons, device=self.device)
                total_k = torch.zeros(n_neurons, device=self.device)
                all_overlaps = []

                per_layer_results = {}
                for lidx, (q_counts, k_counts, batch_overlaps) in layer_data.items():
                    total_q += q_counts
                    total_k += k_counts
                    all_overlaps.extend(batch_overlaps)

                    # Per-layer stats
                    q_np = q_counts.cpu().numpy()
                    k_np = k_counts.cpu().numpy()
                    corr = float(np.corrcoef(q_np, k_np)[0, 1]) if q_np.sum() > 0 and k_np.sum() > 0 else 0.0

                    per_layer_results[f'L{lidx}'] = {
                        'correlation': corr,
                        'avg_overlap': float(np.mean(batch_overlaps)) if batch_overlaps else 0,
                        'q_total': int(q_np.sum()),
                        'k_total': int(k_np.sum()),
                    }

                # Aggregated statistics
                q_np = total_q.cpu().numpy()
                k_np = total_k.cpu().numpy()

                # Correlation
                if q_np.sum() > 0 and k_np.sum() > 0:
                    corr = float(np.corrcoef(q_np, k_np)[0, 1])
                else:
                    corr = 0.0

                # Specialization analysis
                threshold = (q_np.sum() + k_np.sum()) / (2 * len(q_np)) * 0.1
                q_only = int(((q_np > threshold) & (k_np < threshold)).sum())
                k_only = int(((k_np > threshold) & (q_np < threshold)).sum())
                shared = int(((q_np > threshold) & (k_np > threshold)).sum())
                inactive = int(((q_np < threshold) & (k_np < threshold)).sum())

                results[pool_name] = {
                    'display': pool_info['display'],
                    'n_neurons': n_neurons,
                    'n_layers': len(layer_data),
                    'q_counts': q_np.tolist(),
                    'k_counts': k_np.tolist(),
                    'correlation': corr,
                    'avg_overlap': float(np.mean(all_overlaps)) if all_overlaps else 0,
                    'std_overlap': float(np.std(all_overlaps)) if all_overlaps else 0,
                    'q_specialized': q_only,
                    'k_specialized': k_only,
                    'shared': shared,
                    'inactive': inactive,
                    'q_total': int(q_np.sum()),
                    'k_total': int(k_np.sum()),
                    'per_layer': per_layer_results,
                }

            results['n_batches'] = n_batches
        finally:
            self._disable_pref_tensors()

        return results

    def analyze_qk_entropy(self, dataloader, n_batches: int = 50) -> Dict:
        """
        Analyze Q/K routing entropy patterns across ALL layers.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to process

        Returns:
            Dictionary with entropy statistics per layer
        """
        # {pool_name: {layer_idx: {'q': [], 'k': []}}}
        layer_entropy = {pool: {} for pool in QK_POOLS.keys()}

        self.model.eval()
        self._enable_pref_tensors()

        try:
            with torch.no_grad():
                for i, batch in enumerate(tqdm(dataloader, desc='Q/K Entropy', total=n_batches)):
                    if i >= n_batches:
                        break

                    input_ids = get_batch_input_ids(batch, self.device)

                    try:
                        outputs = self.model(input_ids, return_routing_info=True)
                        routing_infos = get_routing_from_outputs(outputs)
                        if routing_infos is None:
                            continue
                    except Exception:
                        continue

                    # Process ALL layers
                    for layer_idx, layer_info in enumerate(routing_infos):
                        attn = layer_info.get('attention', {})

                        for pool_name, pool_info in QK_POOLS.items():
                            if layer_idx not in layer_entropy[pool_name]:
                                layer_entropy[pool_name][layer_idx] = {'q': [], 'k': []}

                            pref_q = attn.get(pool_info['q_pref'])
                            pref_k = attn.get(pool_info['k_pref'])

                            if pref_q is not None:
                                p = pref_q.mean(dim=0) if pref_q.dim() > 1 else pref_q
                                p = p.clamp(min=1e-8)
                                ent = -torch.sum(p * torch.log(p)).item()
                                max_ent = np.log(pref_q.shape[-1])
                                layer_entropy[pool_name][layer_idx]['q'].append(ent / max_ent * 100)

                            if pref_k is not None:
                                p = pref_k.mean(dim=0) if pref_k.dim() > 1 else pref_k
                                p = p.clamp(min=1e-8)
                                ent = -torch.sum(p * torch.log(p)).item()
                                max_ent = np.log(pref_k.shape[-1])
                                layer_entropy[pool_name][layer_idx]['k'].append(ent / max_ent * 100)
        finally:
            self._disable_pref_tensors()

        # Build results
        results = {}
        for pool_name, pool_info in QK_POOLS.items():
            pool_results = {}

            for layer_idx, ent_data in layer_entropy[pool_name].items():
                q_entropy = ent_data['q']
                k_entropy = ent_data['k']

                if q_entropy and k_entropy:
                    pool_results[f'L{layer_idx}'] = {
                        'q_entropy_mean': float(np.mean(q_entropy)),
                        'q_entropy_std': float(np.std(q_entropy)),
                        'k_entropy_mean': float(np.mean(k_entropy)),
                        'k_entropy_std': float(np.std(k_entropy)),
                        'entropy_diff': float(np.mean(q_entropy) - np.mean(k_entropy)),
                    }

            # Summary across layers
            all_q = [v['q_entropy_mean'] for v in pool_results.values()]
            all_k = [v['k_entropy_mean'] for v in pool_results.values()]

            if all_q and all_k:
                pool_results['summary'] = {
                    'q_entropy_avg': float(np.mean(all_q)),
                    'k_entropy_avg': float(np.mean(all_k)),
                    'entropy_diff_avg': float(np.mean(all_q) - np.mean(all_k)),
                }

            results[pool_name] = {
                'display': pool_info['display'],
                'per_layer': pool_results,
                # Keep backward compatibility
                'q_entropy_mean': float(np.mean(all_q)) if all_q else 0,
                'q_entropy_std': float(np.std(all_q)) if all_q else 0,
                'k_entropy_mean': float(np.mean(all_k)) if all_k else 0,
                'k_entropy_std': float(np.std(all_k)) if all_k else 0,
                'entropy_diff': float(np.mean(all_q) - np.mean(all_k)) if all_q and all_k else 0,
            }

        return results

    def visualize_qk_usage(self, usage_results: Dict, output_dir: str) -> Optional[str]:
        """
        Visualize Q/K usage patterns.

        Args:
            usage_results: Results from analyze_qk_usage()
            output_dir: Directory for output

        Returns:
            Path to visualization or None
        """
        if not HAS_MATPLOTLIB:
            return None

        os.makedirs(output_dir, exist_ok=True)

        n_pools = len([k for k in usage_results if k not in ['n_batches']])
        if n_pools == 0:
            return None

        fig, axes = plt.subplots(n_pools, 3, figsize=(18, 6 * n_pools))
        if n_pools == 1:
            axes = axes.reshape(1, -1)

        row = 0
        for pool_name, data in usage_results.items():
            if pool_name == 'n_batches':
                continue

            q_counts = np.array(data['q_counts'])
            k_counts = np.array(data['k_counts'])

            # 1. Scatter: Q vs K
            ax = axes[row, 0]
            ax.scatter(q_counts, k_counts, alpha=0.6, s=30, c=QK_POOLS[pool_name]['color'])
            max_val = max(q_counts.max(), k_counts.max())
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Q=K')
            ax.set_xlabel('Q Selection Count')
            ax.set_ylabel('K Selection Count')
            ax.set_title(f'{data["display"]}: Q vs K Usage\n(corr={data["correlation"]:.3f})')
            ax.legend()

            # 2. Bar: Specialization
            ax = axes[row, 1]
            categories = ['Q-only', 'K-only', 'Shared', 'Inactive']
            values = [data['q_specialized'], data['k_specialized'], data['shared'], data['inactive']]
            colors = ['blue', 'orange', 'green', 'gray']
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            ax.set_ylabel('Neuron Count')
            ax.set_title(f'{data["display"]}: Neuron Specialization')
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(val), ha='center', va='bottom')

            # 3. Histogram: Q/(Q+K) ratio
            ax = axes[row, 2]
            total = q_counts + k_counts + 1e-8
            q_ratio = q_counts / total
            active_mask = (q_counts + k_counts) > 0
            if active_mask.sum() > 0:
                ax.hist(q_ratio[active_mask], bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax.axvline(x=0.5, color='red', linestyle='--', label='Q=K')
            ax.set_xlabel('Q Ratio (Q / (Q+K))')
            ax.set_ylabel('Neuron Count')
            ax.set_title(f'{data["display"]}: Q/K Balance Distribution')
            ax.legend()

            row += 1

        plt.tight_layout()
        path = os.path.join(output_dir, 'qk_usage_analysis.png')
        plt.savefig(path, dpi=150)
        plt.close()

        return path

    def run_all(self, dataloader, output_dir: str = './routing_analysis', n_batches: int = 50) -> Dict:
        """
        Run all routing analyses.

        Args:
            dataloader: DataLoader for input data
            output_dir: Directory for outputs
            n_batches: Number of batches to process

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'entropy': self.analyze_entropy(dataloader, n_batches),
            'selection_frequency': self.analyze_selection_frequency(dataloader, n_batches),
            'selection_diversity': self.analyze_selection_diversity(dataloader, n_batches * 2),
            'qk_overlap': self.analyze_qk_overlap(dataloader, n_batches),
            'qk_usage': self.analyze_qk_usage(dataloader, n_batches * 2),
            'qk_entropy': self.analyze_qk_entropy(dataloader, n_batches),
        }

        # Visualizations
        viz_path = self.visualize_qk_usage(results['qk_usage'], output_dir)
        if viz_path:
            results['qk_visualization'] = viz_path

        return results
