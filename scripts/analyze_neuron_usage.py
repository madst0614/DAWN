"""
Analyze individual neuron usage patterns in neuron-based model
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from src.models.sprout_neuron_based import NeuronBasedLanguageModel


def analyze_neuron_selection(model, tokenizer, device='cpu', top_k=512):
    """Analyze which neurons are selected for different inputs"""
    print("="*70)
    print("NEURON SELECTION ANALYSIS")
    print("="*70)

    model.eval()

    # Test inputs
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Artificial intelligence is transforming the world",
        "Python is a popular programming language",
        "The weather today is sunny and warm",
        "Machine learning requires large datasets",
        "Mathematics is the foundation of science",
        "I love reading books in my free time",
        "The capital of France is Paris"
    ]

    # Track neuron usage across all layers
    n_layers = len(model.layers)
    d_ff = model.layers[0].ffn.d_ff

    layer_usage = {i: torch.zeros(d_ff) for i in range(n_layers)}
    layer_scores = {i: [] for i in range(n_layers)}

    print(f"\nModel: {n_layers} layers, {d_ff} neurons per FFN")
    print(f"Sparsity: top_k={top_k} ({top_k/d_ff*100:.1f}%)")
    print(f"Test inputs: {len(test_texts)}")

    # Hook to capture router scores
    router_data = defaultdict(list)

    def make_hook(layer_idx):
        def hook(module, input, output):
            x = input[0]  # [batch, seq, d_model]
            batch, seq, d_model = x.shape
            x_flat = x.view(-1, d_model)

            # Router scores
            scores = x_flat @ module.router.W_router.T  # [batch*seq, d_ff]
            router_data[layer_idx].append(scores)

        return hook

    # Register hooks
    hooks = []
    for i, layer in enumerate(model.layers):
        hook = layer.ffn.register_forward_hook(make_hook(i))
        hooks.append(hook)

    # Process inputs
    with torch.no_grad():
        for text in test_texts:
            tokens = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=32,
                truncation=True
            )['input_ids'].to(device)

            # Forward pass with sparsity
            _ = model(tokens, top_k=top_k)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze router data
    print("\n" + "="*70)
    print("LAYER-WISE NEURON USAGE")
    print("="*70)

    for layer_idx in range(n_layers):
        scores_list = router_data[layer_idx]

        # Concatenate all scores
        all_scores = torch.cat(scores_list, dim=0)  # [total_positions, d_ff]

        # Get top-k selections
        _, top_indices = torch.topk(all_scores, top_k, dim=-1)

        # Count usage
        usage = torch.zeros(d_ff)
        for indices in top_indices:
            usage[indices] += 1

        usage_pct = usage / top_indices.shape[0] * 100

        # Statistics
        n_total = d_ff
        n_never = (usage == 0).sum().item()
        n_rare = ((usage > 0) & (usage < usage.mean())).sum().item()
        n_common = (usage >= usage.mean()).sum().item()

        mean_usage = usage.mean().item()
        max_usage = usage.max().item()
        min_usage = usage.min().item()
        std_usage = usage.std().item()

        print(f"\nLayer {layer_idx}:")
        print(f"  Never used: {n_never:4d} / {n_total} ({n_never/n_total*100:.1f}%)")
        print(f"  Rare (<avg): {n_rare:4d} / {n_total} ({n_rare/n_total*100:.1f}%)")
        print(f"  Common (≥avg): {n_common:4d} / {n_total} ({n_common/n_total*100:.1f}%)")
        print(f"  Mean usage: {mean_usage:.1f} ({mean_usage/top_indices.shape[0]*100:.1f}%)")
        print(f"  Max usage: {max_usage:.0f}, Min usage: {min_usage:.0f}, Std: {std_usage:.1f}")

        # Top 10 most used neurons
        top_10_usage, top_10_indices = torch.topk(usage, 10)
        print(f"  Top 10 neurons: {top_10_indices.tolist()}")
        print(f"  Top 10 usage: {[f'{u:.0f}' for u in top_10_usage.tolist()]}")

        layer_usage[layer_idx] = usage
        layer_scores[layer_idx] = all_scores

    return layer_usage, layer_scores


def analyze_neuron_similarity(model, layer_idx=0):
    """Analyze similarity between neurons"""
    print("\n" + "="*70)
    print(f"NEURON SIMILARITY ANALYSIS (Layer {layer_idx})")
    print("="*70)

    ffn = model.layers[layer_idx].ffn

    # Get all middle neuron weights
    weights = torch.stack([n.W_in for n in ffn.middle_neurons])  # [d_ff, d_model]

    # Compute pairwise cosine similarity
    normalized = F.normalize(weights, p=2, dim=1)
    similarity_matrix = normalized @ normalized.T  # [d_ff, d_ff]

    # Remove diagonal (self-similarity = 1.0)
    similarity_matrix.fill_diagonal_(0.0)

    # Statistics
    max_sim = similarity_matrix.max().item()
    mean_sim = similarity_matrix.mean().item()
    min_sim = similarity_matrix.min().item()

    print(f"\nNeuron weight similarity (cosine):")
    print(f"  Max: {max_sim:.4f}")
    print(f"  Mean: {mean_sim:.4f}")
    print(f"  Min: {min_sim:.4f}")

    # Find most similar pairs
    flat_sim = similarity_matrix.flatten()
    top_5_sim, top_5_idx = torch.topk(flat_sim, 5)

    d_ff = weights.shape[0]
    print(f"\nTop 5 most similar neuron pairs:")
    for sim, idx in zip(top_5_sim, top_5_idx):
        i = idx.item() // d_ff
        j = idx.item() % d_ff
        print(f"  Neuron {i:4d} ↔ {j:4d}: similarity = {sim:.4f}")

    # Clustering analysis
    threshold = 0.7
    n_similar_pairs = (similarity_matrix > threshold).sum().item() // 2
    print(f"\nNeuron pairs with similarity > {threshold}: {n_similar_pairs}")

    if mean_sim < 0.3:
        print("✅ Good diversity: neurons are learning different features")
    elif mean_sim < 0.5:
        print("⚠️  Moderate diversity: some redundancy")
    else:
        print("❌ Poor diversity: many neurons are redundant")

    return similarity_matrix


def analyze_routing_diversity(model, tokenizer, device='cpu', top_k=512):
    """Analyze how routing changes for different inputs"""
    print("\n" + "="*70)
    print("ROUTING DIVERSITY ANALYSIS")
    print("="*70)

    model.eval()

    # Different types of inputs
    test_groups = {
        "Science": [
            "Physics studies the nature of matter and energy",
            "Chemistry explores the composition of substances",
            "Biology examines living organisms and life processes"
        ],
        "Technology": [
            "Computers process information using binary code",
            "Software development requires programming skills",
            "Artificial intelligence mimics human cognition"
        ],
        "Arts": [
            "Painting expresses emotions through visual art",
            "Music creates harmony using different instruments",
            "Literature tells stories through written words"
        ]
    }

    # Collect routing patterns for each group
    layer_idx = 0  # Analyze first layer
    ffn = model.layers[layer_idx].ffn

    group_patterns = {}

    with torch.no_grad():
        for group_name, texts in test_groups.items():
            group_selections = []

            for text in texts:
                tokens = tokenizer(
                    text,
                    return_tensors='pt',
                    padding='max_length',
                    max_length=32,
                    truncation=True
                )['input_ids'].to(device)

                batch, seq = tokens.shape
                x = model.embed(tokens)  # [batch, seq, d_model]
                x_flat = x.view(-1, x.shape[-1])

                # Router scores
                scores = x_flat @ ffn.router.W_router.T  # [batch*seq, d_ff]
                _, top_indices = torch.topk(scores, top_k, dim=-1)

                # Convert to binary mask
                mask = torch.zeros_like(scores)
                mask.scatter_(-1, top_indices, 1.0)
                group_selections.append(mask.mean(dim=0))  # [d_ff]

            # Average pattern for group
            group_pattern = torch.stack(group_selections).mean(dim=0)
            group_patterns[group_name] = group_pattern

    # Compare patterns between groups
    print(f"\nLayer {layer_idx} routing patterns:")
    print(f"Sparsity: top_k={top_k}")

    groups = list(group_patterns.keys())
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            pattern_i = group_patterns[groups[i]]
            pattern_j = group_patterns[groups[j]]

            # Cosine similarity
            sim = F.cosine_similarity(
                pattern_i.unsqueeze(0),
                pattern_j.unsqueeze(0)
            ).item()

            # Overlap (Jaccard)
            overlap = ((pattern_i > 0) & (pattern_j > 0)).sum().item()
            union = ((pattern_i > 0) | (pattern_j > 0)).sum().item()
            jaccard = overlap / union if union > 0 else 0

            print(f"  {groups[i]:12s} ↔ {groups[j]:12s}: cosine={sim:.4f}, jaccard={jaccard:.4f}")

    if sim < 0.7:
        print("\n✅ Good routing diversity: different content uses different neurons")
    else:
        print("\n⚠️  Router needs training: patterns are too similar")


def plot_neuron_usage(layer_usage, save_path="neuron_usage.png"):
    """Plot neuron usage distribution"""
    print("\n" + "="*70)
    print("GENERATING USAGE PLOTS")
    print("="*70)

    n_layers = len(layer_usage)
    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()

    for layer_idx, usage in layer_usage.items():
        ax = axes[layer_idx]

        # Sort neurons by usage
        sorted_usage, _ = torch.sort(usage, descending=True)

        ax.bar(range(len(sorted_usage)), sorted_usage.numpy(), width=1.0)
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Neuron (sorted by usage)")
        ax.set_ylabel("Usage count")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Plot saved: {save_path}")


def main():
    print("="*70)
    print("NEURON-BASED MODEL ANALYSIS")
    print("="*70)

    # Model settings
    vocab_size = 30522
    d_model = 512
    d_ff = 2048
    n_layers = 6
    n_heads = 8
    top_k = 512  # 25% sparsity

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create model (or load from checkpoint)
    print("\nCreating model...")
    model = NeuronBasedLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layers,
        n_heads=n_heads
    ).to(device)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Run analyses
    print("\n")

    # 1. Neuron selection patterns
    layer_usage, layer_scores = analyze_neuron_selection(
        model, tokenizer, device, top_k=top_k
    )

    # 2. Neuron similarity
    similarity_matrix = analyze_neuron_similarity(model, layer_idx=0)

    # 3. Routing diversity
    analyze_routing_diversity(model, tokenizer, device, top_k=top_k)

    # 4. Plot usage
    # plot_neuron_usage(layer_usage, save_path="neuron_usage.png")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
