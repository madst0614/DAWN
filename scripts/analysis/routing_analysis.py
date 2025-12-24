#!/usr/bin/env python3
"""
DAWN Routing Analysis for Generation
=====================================
Analyze token-level routing patterns during text generation.

Features:
1. Generate text while collecting per-token routing indices
2. Analyze common neurons across multiple runs
3. Visualize routing patterns with heatmaps

Usage:
    python scripts/analysis/routing_analysis.py \
        --checkpoint path/to/checkpoint \
        --prompt "The capital of France is" \
        --output routing_analysis/
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
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

from .utils import (
    load_model, get_router,
    ROUTING_KEYS, KNOWLEDGE_ROUTING_KEYS,
    HAS_MATPLOTLIB, plt
)

if HAS_MATPLOTLIB:
    import seaborn as sns


class GenerationRoutingAnalyzer:
    """Analyze routing patterns during text generation."""

    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize analyzer.

        Args:
            model: DAWN model
            tokenizer: Tokenizer for text encoding/decoding
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def generate_with_routing(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> Dict:
        """
        Generate text while collecting routing information.

        Args:
            prompt: Input prompt
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = greedy)

        Returns:
            Dictionary with tokens, routing indices, and generated text
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
        input_ids = input_ids.to(self.device)

        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]

        # Collect routing info per generated token
        routing_logs = {
            'tokens': [],
            'token_ids': [],
            # Feature routing (per token)
            'fv_indices': [],      # Feature V indices
            'fqk_q_indices': [],   # Feature QK (Q) indices
            'fqk_k_indices': [],   # Feature QK (K) indices
            # Restore routing (per token)
            'rv_indices': [],      # Restore V indices
            'rqk_q_indices': [],   # Restore QK (Q) indices
            'rqk_k_indices': [],   # Restore QK (K) indices
            # Knowledge routing (per token)
            'fknow_indices': [],   # Feature Knowledge indices
            'rknow_indices': [],   # Restore Knowledge indices
            # Weights for analysis
            'fv_weights': [],
            'rv_weights': [],
        }

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward with routing info
                outputs = self.model(generated, return_routing_info=True)

                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    logits, routing_infos = outputs[0], outputs[1]
                else:
                    logits = outputs
                    routing_infos = None

                # Get next token
                next_token_logits = logits[:, -1, :]

                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)

                # Decode token
                token_text = self.tokenizer.decode(next_token[0])
                routing_logs['tokens'].append(token_text)
                routing_logs['token_ids'].append(next_token.item())

                # Extract routing indices from last position
                if routing_infos is not None:
                    self._extract_routing_indices(routing_infos, routing_logs)

                generated = torch.cat([generated, next_token], dim=1)

        # Decode full generation
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        routing_logs['prompt'] = prompt
        routing_logs['generated_text'] = generated_text
        routing_logs['full_text'] = generated_text

        return routing_logs

    def _extract_routing_indices(self, routing_infos: List[Dict], routing_logs: Dict):
        """Extract routing indices from routing_infos for the last token."""
        # Aggregate across layers (using last layer or mean)
        fv_indices_all = []
        rv_indices_all = []
        fqk_q_indices_all = []
        fqk_k_indices_all = []
        rqk_q_indices_all = []
        rqk_k_indices_all = []
        fknow_indices_all = []
        rknow_indices_all = []

        fv_weights_last = None
        rv_weights_last = None

        for layer_info in routing_infos:
            attn = layer_info.get('attention', layer_info)
            know = layer_info.get('knowledge', {})

            # Feature V weights -> indices
            fv_w = attn.get('fv_weights')
            if fv_w is not None:
                # Get indices for last token position
                fv_last = fv_w[0, -1]  # [N_v]
                fv_idx = (fv_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                fv_indices_all.extend(fv_idx)
                fv_weights_last = fv_last.cpu()

            # Restore V weights -> indices
            rv_w = attn.get('rv_weights')
            if rv_w is not None:
                rv_last = rv_w[0, -1]
                rv_idx = (rv_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                rv_indices_all.extend(rv_idx)
                rv_weights_last = rv_last.cpu()

            # Feature QK (Q/K)
            fqk_q_w = attn.get('fqk_weights_Q')
            if fqk_q_w is not None:
                fqk_q_last = fqk_q_w[0, -1]
                fqk_q_idx = (fqk_q_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                fqk_q_indices_all.extend(fqk_q_idx)

            fqk_k_w = attn.get('fqk_weights_K')
            if fqk_k_w is not None:
                fqk_k_last = fqk_k_w[0, -1]
                fqk_k_idx = (fqk_k_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                fqk_k_indices_all.extend(fqk_k_idx)

            # Restore QK (Q/K)
            rqk_q_w = attn.get('rqk_weights_Q')
            if rqk_q_w is not None:
                rqk_q_last = rqk_q_w[0, -1]
                rqk_q_idx = (rqk_q_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                rqk_q_indices_all.extend(rqk_q_idx)

            rqk_k_w = attn.get('rqk_weights_K')
            if rqk_k_w is not None:
                rqk_k_last = rqk_k_w[0, -1]
                rqk_k_idx = (rqk_k_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                rqk_k_indices_all.extend(rqk_k_idx)

            # Knowledge
            fknow_w = know.get('feature_know_w')
            if fknow_w is not None:
                fknow_last = fknow_w[0, -1]
                fknow_idx = (fknow_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                fknow_indices_all.extend(fknow_idx)

            rknow_w = know.get('restore_know_w')
            if rknow_w is not None:
                rknow_last = rknow_w[0, -1]
                rknow_idx = (rknow_last > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                rknow_indices_all.extend(rknow_idx)

        # Store unique indices per token
        routing_logs['fv_indices'].append(list(set(fv_indices_all)))
        routing_logs['rv_indices'].append(list(set(rv_indices_all)))
        routing_logs['fqk_q_indices'].append(list(set(fqk_q_indices_all)))
        routing_logs['fqk_k_indices'].append(list(set(fqk_k_indices_all)))
        routing_logs['rqk_q_indices'].append(list(set(rqk_q_indices_all)))
        routing_logs['rqk_k_indices'].append(list(set(rqk_k_indices_all)))
        routing_logs['fknow_indices'].append(list(set(fknow_indices_all)))
        routing_logs['rknow_indices'].append(list(set(rknow_indices_all)))

        if fv_weights_last is not None:
            routing_logs['fv_weights'].append(fv_weights_last)
        if rv_weights_last is not None:
            routing_logs['rv_weights'].append(rv_weights_last)


def analyze_common_neurons(
    routing_logs_list: List[Dict],
    target_token: str = None,
    pool_type: str = 'fv',
) -> Dict:
    """
    Analyze common neurons across multiple generation runs.

    Args:
        routing_logs_list: List of routing logs from generate_with_routing()
        target_token: Specific token to analyze (e.g., "paris")
        pool_type: 'fv', 'rv', 'fqk_q', 'fqk_k', 'rqk_q', 'rqk_k', 'fknow', 'rknow'

    Returns:
        Dictionary with common neuron analysis
    """
    indices_key = f'{pool_type}_indices'
    all_indices = []
    token_to_indices = defaultdict(list)

    for routing_log in routing_logs_list:
        tokens = routing_log.get('tokens', [])
        indices_list = routing_log.get(indices_key, [])

        for i, (token, indices) in enumerate(zip(tokens, indices_list)):
            token_lower = token.strip().lower()
            all_indices.extend(indices)
            token_to_indices[token_lower].append(indices)

    # Overall frequency
    counter = Counter(all_indices)
    total_tokens = sum(len(routing_log.get('tokens', [])) for routing_log in routing_logs_list)

    results = {
        'pool_type': pool_type,
        'total_tokens_analyzed': total_tokens,
        'unique_neurons_used': len(counter),
        'top_neurons': counter.most_common(20),
    }

    # Target token analysis
    if target_token:
        target_lower = target_token.strip().lower()
        if target_lower in token_to_indices:
            target_indices_list = token_to_indices[target_lower]
            target_counter = Counter()
            for indices in target_indices_list:
                target_counter.update(indices)

            # Find neurons that appear in ALL occurrences
            n_occurrences = len(target_indices_list)
            common_neurons = [
                neuron for neuron, count in target_counter.items()
                if count == n_occurrences
            ]

            results['target_token'] = target_token
            results['target_occurrences'] = n_occurrences
            results['target_top_neurons'] = target_counter.most_common(10)
            results['target_common_neurons'] = common_neurons

    return results


def plot_routing_heatmap(
    routing_log: Dict,
    pool_type: str = 'fv',
    output_path: str = None,
    max_neurons: int = 50,
    figsize: Tuple[int, int] = (14, 8),
) -> Optional[str]:
    """
    Plot token x neuron routing heatmap.

    Args:
        routing_log: Routing log from generate_with_routing()
        pool_type: 'fv', 'rv', etc.
        output_path: Path to save figure
        max_neurons: Maximum number of neurons to display
        figsize: Figure size

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return None

    tokens = routing_log.get('tokens', [])
    weights_key = f'{pool_type}_weights'
    weights_list = routing_log.get(weights_key, [])

    if not weights_list:
        # Fall back to binary indices
        indices_key = f'{pool_type}_indices'
        indices_list = routing_log.get(indices_key, [])
        if not indices_list:
            print(f"No data for {pool_type}")
            return None

        # Find all unique neurons
        all_neurons = set()
        for indices in indices_list:
            all_neurons.update(indices)
        all_neurons = sorted(all_neurons)[:max_neurons]

        # Create binary matrix
        neuron_to_idx = {n: i for i, n in enumerate(all_neurons)}
        matrix = np.zeros((len(tokens), len(all_neurons)))
        for t_idx, indices in enumerate(indices_list):
            for n in indices:
                if n in neuron_to_idx:
                    matrix[t_idx, neuron_to_idx[n]] = 1

        neuron_labels = [str(n) for n in all_neurons]
    else:
        # Use actual weights
        weights_tensor = torch.stack(weights_list)  # [T, N]
        n_neurons = weights_tensor.shape[1]

        # Find most active neurons
        total_activation = weights_tensor.sum(dim=0)
        top_indices = total_activation.argsort(descending=True)[:max_neurons]

        matrix = weights_tensor[:, top_indices].numpy()
        neuron_labels = [str(idx.item()) for idx in top_indices]

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Truncate token labels
    token_labels = [t[:15] if len(t) > 15 else t for t in tokens]

    sns.heatmap(
        matrix,
        xticklabels=neuron_labels,
        yticklabels=token_labels,
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Routing Weight'}
    )

    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Generated Token')
    ax.set_title(f'Routing Heatmap ({pool_type.upper()})\n"{routing_log.get("prompt", "")[:50]}..."')

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def plot_routing_comparison(
    routing_logs_list: List[Dict],
    pool_type: str = 'fv',
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional[str]:
    """
    Compare routing patterns across multiple runs.

    Args:
        routing_logs_list: List of routing logs
        pool_type: Pool type to analyze
        output_path: Path to save figure

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return None

    indices_key = f'{pool_type}_indices'

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Neuron usage frequency
    all_neurons = Counter()
    for routing_log in routing_logs_list:
        for indices in routing_log.get(indices_key, []):
            all_neurons.update(indices)

    top_20 = all_neurons.most_common(20)
    if top_20:
        neurons, counts = zip(*top_20)
        axes[0].barh(range(len(neurons)), counts, color='steelblue')
        axes[0].set_yticks(range(len(neurons)))
        axes[0].set_yticklabels([f'N{n}' for n in neurons])
        axes[0].set_xlabel('Selection Count')
        axes[0].set_title(f'Top 20 Neurons ({pool_type.upper()})')
        axes[0].invert_yaxis()

    # 2. Neurons per token distribution
    neurons_per_token = []
    for routing_log in routing_logs_list:
        for indices in routing_log.get(indices_key, []):
            neurons_per_token.append(len(indices))

    if neurons_per_token:
        axes[1].hist(neurons_per_token, bins=30, color='coral', edgecolor='black')
        axes[1].axvline(np.mean(neurons_per_token), color='red', linestyle='--',
                        label=f'Mean: {np.mean(neurons_per_token):.1f}')
        axes[1].set_xlabel('Neurons per Token')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Routing Sparsity Distribution')
        axes[1].legend()

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def main():
    parser = argparse.ArgumentParser(description='DAWN Routing Analysis for Generation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--prompt', type=str, default="The capital of France is",
                        help='Prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=30,
                        help='Max tokens to generate')
    parser.add_argument('--output', type=str, default='routing_analysis',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--pool', type=str, default='fv',
                        choices=['fv', 'rv', 'fqk_q', 'fqk_k', 'rqk_q', 'rqk_k', 'fknow', 'rknow'],
                        help='Pool type to analyze')
    args = parser.parse_args()

    # Device check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, tokenizer, config = load_model(args.checkpoint, args.device)
    model = model.to(args.device)
    model.eval()

    # Create analyzer
    analyzer = GenerationRoutingAnalyzer(model, tokenizer, args.device)

    # Generate with routing
    print(f"\nGenerating with prompt: '{args.prompt}'")
    routing_log = analyzer.generate_with_routing(
        args.prompt,
        max_new_tokens=args.max_tokens,
    )

    # Print results
    print(f"\nGenerated text:")
    print(routing_log['full_text'])

    print(f"\n{'='*60}")
    print(f"Routing Analysis ({args.pool.upper()})")
    print('='*60)

    indices_key = f'{args.pool}_indices'
    for i, (token, indices) in enumerate(zip(routing_log['tokens'], routing_log.get(indices_key, []))):
        print(f"[{i:2d}] '{token:15s}' -> {len(indices):2d} neurons: {indices[:10]}{'...' if len(indices) > 10 else ''}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Plot heatmap
    heatmap_path = os.path.join(args.output, f'heatmap_{args.pool}.png')
    plot_routing_heatmap(routing_log, args.pool, heatmap_path)
    print(f"\nHeatmap saved to: {heatmap_path}")

    # Analyze common neurons
    common_analysis = analyze_common_neurons([routing_log], pool_type=args.pool)
    print(f"\nTop neurons: {common_analysis['top_neurons'][:10]}")

    # Save routing log
    import json

    # Convert tensors to lists for JSON serialization
    save_log = {k: v for k, v in routing_log.items() if not k.endswith('_weights')}
    log_path = os.path.join(args.output, 'routing_log.json')
    with open(log_path, 'w') as f:
        json.dump(save_log, f, indent=2)
    print(f"Routing log saved to: {log_path}")


if __name__ == '__main__':
    main()
