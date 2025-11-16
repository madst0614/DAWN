"""
Analysis tools for Simple Neuron Pool V1.

Provides visualization and analysis of neuron behavior.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_neuron_usage_by_layer(model, save_path: Optional[str] = None):
    """
    Plot neuron usage distribution for each layer.

    Args:
        model: SimpleTransformerWithNeuronPool model
        save_path: Path to save plot (optional)
    """
    n_layers = model.n_layers
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for layer_idx in range(n_layers):
        usage = model.neuron_pool.get_layer_neuron_usage(layer_idx)
        usage_np = usage.cpu().numpy()

        ax = axes[layer_idx]
        ax.hist(usage_np, bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f'Layer {layer_idx} Neuron Usage')
        ax.set_xlabel('Usage Count')
        ax.set_ylabel('# Neurons')
        ax.grid(True, alpha=0.3)

        # Add stats as text
        mean = usage_np.mean()
        std = usage_np.std()
        ax.text(
            0.95, 0.95,
            f'Mean: {mean:.1f}\nStd: {std:.1f}',
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved plot to {save_path}")
    else:
        plt.show()


def plot_neuron_weight_patterns(
    model,
    neuron_ids: list,
    save_path: Optional[str] = None
):
    """
    Visualize input/output weight patterns for specific neurons.

    Args:
        model: SimpleTransformerWithNeuronPool model
        neuron_ids: List of global neuron IDs to visualize
        save_path: Path to save plot (optional)
    """
    n_neurons = len(neuron_ids)
    fig, axes = plt.subplots(n_neurons, 2, figsize=(12, 3 * n_neurons))

    if n_neurons == 1:
        axes = axes.reshape(1, -1)

    for i, neuron_id in enumerate(neuron_ids):
        signature = model.neuron_pool.get_neuron_signature(neuron_id)

        layer = neuron_id // model.d_ff
        local_id = neuron_id % model.d_ff

        # Input pattern
        ax = axes[i, 0]
        input_pattern = signature['input_pattern'].numpy()
        ax.plot(input_pattern)
        ax.set_title(f'Neuron {neuron_id} (L{layer}, #{local_id}) - Input Weights')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)

        # Output pattern
        ax = axes[i, 1]
        output_pattern = signature['output_pattern'].numpy()
        ax.plot(output_pattern)
        ax.set_title(f'Neuron {neuron_id} (L{layer}, #{local_id}) - Output Weights')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved plot to {save_path}")
    else:
        plt.show()


def analyze_top_neurons(model, top_k: int = 10):
    """
    Analyze the most frequently used neurons.

    Args:
        model: SimpleTransformerWithNeuronPool model
        top_k: Number of top neurons to analyze
    """
    print("\n" + "="*70)
    print(f"TOP {top_k} MOST USED NEURONS")
    print("="*70)

    top_neurons = model.get_most_used_neurons(top_k)

    for i, neuron_info in enumerate(top_neurons, 1):
        print(f"\n{i}. Neuron {neuron_info['global_id']}:")
        print(f"   Layer: {neuron_info['layer']}")
        print(f"   Local ID: {neuron_info['local_id']}")
        print(f"   Usage count: {neuron_info['usage_count']:.0f}")

        # Get signature
        signature = model.neuron_pool.get_neuron_signature(neuron_info['global_id'])

        # Weight statistics
        input_pattern = signature['input_pattern']
        output_pattern = signature['output_pattern']

        print(f"   Input weights - norm: {input_pattern.norm():.3f}, "
              f"mean: {input_pattern.mean():.3f}, std: {input_pattern.std():.3f}")
        print(f"   Output weights - norm: {output_pattern.norm():.3f}, "
              f"mean: {output_pattern.mean():.3f}, std: {output_pattern.std():.3f}")

    print("="*70 + "\n")


def compare_layer_statistics(model):
    """
    Compare neuron usage statistics across layers.

    Args:
        model: SimpleTransformerWithNeuronPool model
    """
    stats = model.get_neuron_usage_stats()

    print("\n" + "="*70)
    print("LAYER-BY-LAYER COMPARISON")
    print("="*70)

    # Collect stats for comparison
    layer_means = []
    layer_stds = []
    layer_totals = []

    for layer_idx in range(model.n_layers):
        layer_stats = stats[f'layer_{layer_idx}']
        layer_means.append(layer_stats['mean_usage'])
        layer_stds.append(layer_stats['std_usage'])
        layer_totals.append(layer_stats['total_usage'])

    # Print table
    print(f"\n{'Layer':<10} {'Mean':<15} {'Std':<15} {'Total':<15}")
    print("-" * 70)
    for i in range(model.n_layers):
        print(f"{i:<10} {layer_means[i]:<15.1f} {layer_stds[i]:<15.1f} {layer_totals[i]:<15.0f}")

    # Find interesting patterns
    print("\n" + "="*70)
    print("INTERESTING PATTERNS")
    print("="*70)

    # Most/least used layer
    most_used_layer = np.argmax(layer_totals)
    least_used_layer = np.argmin(layer_totals)

    print(f"\nMost used layer: {most_used_layer} (total: {layer_totals[most_used_layer]:.0f})")
    print(f"Least used layer: {least_used_layer} (total: {layer_totals[least_used_layer]:.0f})")

    # Layer with highest variance
    highest_var_layer = np.argmax(layer_stds)
    print(f"\nHighest variance layer: {highest_var_layer} (std: {layer_stds[highest_var_layer]:.1f})")

    print("="*70 + "\n")


def neuron_activation_heatmap(
    model,
    input_ids: torch.Tensor,
    layer_idx: int,
    save_path: Optional[str] = None
):
    """
    Create heatmap of neuron activations for given input.

    Args:
        model: SimpleTransformerWithNeuronPool model
        input_ids: Input token IDs [batch, seq]
        layer_idx: Layer to visualize
        save_path: Path to save plot (optional)
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Forward pass up to target layer
    with torch.no_grad():
        batch, seq = input_ids.shape

        # Embeddings
        token_emb = model.token_embedding(input_ids)
        pos_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        pos_emb = model.pos_embedding(pos_ids)
        h = token_emb + pos_emb

        # Process layers up to target
        for i in range(layer_idx + 1):
            # Attention
            attn_out, _ = model.attentions[i](h, h, h)
            h = model.norms1[i](h + attn_out)

            # Get layer neurons
            start, end = model.neuron_pool.layer_ranges[i]
            W_in = model.neuron_pool.W_in[start:end]
            b_in = model.neuron_pool.b_in[start:end]

            # Compute activations before GELU
            activations = torch.einsum('nd,bsd->bsn', W_in, h) + b_in
            # [batch, seq, d_ff]

            if i == layer_idx:
                # This is our target layer
                target_activations = activations

            # Complete the layer
            activated = torch.nn.functional.gelu(activations)
            W_out = model.neuron_pool.W_out[start:end]
            b_out = model.neuron_pool.b_out[start:end]
            ffn_out = torch.einsum('nd,bsn->bsd', W_out, activated) + b_out
            h = model.norms2[i](h + ffn_out)

    # Plot heatmap for first sequence in batch
    activations_np = target_activations[0].cpu().numpy()  # [seq, d_ff]

    plt.figure(figsize=(12, 6))
    plt.imshow(activations_np.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    plt.colorbar(label='Activation')
    plt.xlabel('Sequence Position')
    plt.ylabel('Neuron ID')
    plt.title(f'Layer {layer_idx} Neuron Activations')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved heatmap to {save_path}")
    else:
        plt.show()


def full_analysis_report(model, sample_input_ids: Optional[torch.Tensor] = None):
    """
    Generate a comprehensive analysis report.

    Args:
        model: SimpleTransformerWithNeuronPool model
        sample_input_ids: Optional sample input for activation analysis
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE NEURON POOL ANALYSIS REPORT")
    print("="*70)

    # 1. Model info
    print("\nModel Configuration:")
    print(f"  Vocabulary size: {model.vocab_size}")
    print(f"  Model dimension: {model.d_model}")
    print(f"  Feed-forward dimension: {model.d_ff}")
    print(f"  Number of layers: {model.n_layers}")
    print(f"  Total neurons: {model.n_layers * model.d_ff}")

    # 2. Usage statistics
    model.visualize_neuron_usage()

    # 3. Layer comparison
    compare_layer_statistics(model)

    # 4. Top neurons
    analyze_top_neurons(model, top_k=10)

    print("\n" + "="*70)
    print("END OF REPORT")
    print("="*70 + "\n")


if __name__ == '__main__':
    # This would require a trained model
    print("Analysis tools for Simple Neuron Pool V1")
    print("Import these functions to analyze your trained model.")
