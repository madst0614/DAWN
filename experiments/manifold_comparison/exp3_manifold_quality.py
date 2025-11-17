"""
Experiment 3: Manifold Quality Visualization
목표: 매니폴드가 의미있는 구조를 형성하는지

분석 내용:
- 선택된 뉴런 조합들의 manifold 출력을 시각화
- t-SNE로 manifold 공간 구조 확인
- 같은 카테고리 입력들이 가까이 모이는지 확인
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import defaultdict
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.neuron_based_manifold import NeuronBasedLanguageModel as ManifoldModel


def get_manifold_output(model, input_ids, layer_idx=0):
    """Extract manifold output from the model"""
    model.eval()

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        # Embeddings
        batch_size, seq_len = input_ids.shape
        token_emb = model.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = model.position_embedding(positions)
        x = token_emb + pos_emb

        # Go through specified layer
        layer = model.layers[layer_idx]

        # Attention
        x = layer.norm1(x)
        x_attn, _ = layer.attention(x, x, x)
        x = x + layer.dropout(x_attn)

        # FFN with manifold extraction
        x_ffn = layer.norm2(x)

        # Get router scores and selected neurons
        x_flat = x_ffn.view(-1, model.d_model)
        scores = layer.ffn.router.compute_scores(x_flat)
        top_k = model.sparse_k if model.sparse_k is not None else model.d_ff
        _, top_indices = torch.topk(scores, top_k, dim=-1)

        # Get activations
        z = x_flat @ layer.ffn.W1.T
        a = torch.nn.functional.gelu(z)

        # Extract selected activations
        batch_seq = x_flat.shape[0]
        selected_activations = []
        for i in range(batch_seq):
            idx = top_indices[i]
            selected_activations.append(a[i, idx])

        selected_activations = torch.stack(selected_activations)

        # Get manifold embeddings (φ output)
        if hasattr(layer.ffn, 'manifold_mixer') and layer.ffn.manifold_mixer is not None:
            embeddings = layer.ffn.manifold_mixer.phi(selected_activations.unsqueeze(-1))
            # Average over neurons to get representation
            manifold_repr = embeddings.mean(dim=1)  # [batch_seq, d_hidden]
        else:
            # No manifold - use activation statistics
            manifold_repr = torch.cat([
                selected_activations.mean(dim=1, keepdim=True),
                selected_activations.std(dim=1, keepdim=True),
                selected_activations.max(dim=1, keepdim=True)[0],
                selected_activations.min(dim=1, keepdim=True)[0]
            ], dim=-1)

        return manifold_repr.cpu().numpy()


def collect_manifold_representations(model, dataset, categories, device):
    """Collect manifold representations for dataset"""

    representations = []
    labels = []

    for input_text, category in dataset:
        input_ids = torch.tensor([[hash(w) % model.vocab_size for w in input_text.split()]])
        manifold_repr = get_manifold_output(model, input_ids, layer_idx=0)

        # Average over sequence
        repr_avg = manifold_repr.mean(axis=0)

        representations.append(repr_avg)
        labels.append(category)

    return np.array(representations), labels


def visualize_manifold_structure(representations, labels, save_dir, name='Manifold'):
    """Visualize manifold structure using t-SNE"""

    print(f"\nRunning t-SNE for {name}...")

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(representations)-1))
    embeddings_2d = tsne.fit_transform(representations)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = np.array(labels) == label
        ax.scatter(embeddings_2d[mask, 0],
                   embeddings_2d[mask, 1],
                   c=[color],
                   label=label,
                   alpha=0.6,
                   s=100)

    ax.set_title(f'{name} Structure (t-SNE)')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'{name.lower()}_tsne.png', dpi=150)
    print(f"Saved to {save_dir / f'{name.lower()}_tsne.png'}")

    return embeddings_2d


def compute_cluster_quality(representations, labels):
    """Compute cluster quality metrics"""

    unique_labels = sorted(set(labels))

    # Intra-class distance (should be small)
    intra_distances = []
    for label in unique_labels:
        mask = np.array(labels) == label
        class_repr = representations[mask]

        if len(class_repr) > 1:
            centroid = class_repr.mean(axis=0)
            dists = np.linalg.norm(class_repr - centroid, axis=1)
            intra_distances.append(dists.mean())

    avg_intra = np.mean(intra_distances) if intra_distances else 0

    # Inter-class distance (should be large)
    centroids = []
    for label in unique_labels:
        mask = np.array(labels) == label
        centroid = representations[mask].mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)
    inter_distances = []

    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            inter_distances.append(dist)

    avg_inter = np.mean(inter_distances) if inter_distances else 0

    # Silhouette-like score: (inter - intra) / max(inter, intra)
    if max(avg_inter, avg_intra) > 0:
        quality_score = (avg_inter - avg_intra) / max(avg_inter, avg_intra)
    else:
        quality_score = 0

    return {
        'avg_intra_distance': avg_intra,
        'avg_inter_distance': avg_inter,
        'quality_score': quality_score
    }


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save directory
    save_dir = Path(__file__).parent / 'results' / 'exp3_manifold'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Model config
    config = {
        'vocab_size': 10000,
        'd_model': 256,
        'd_ff': 1024,
        'n_heads': 8,
        'n_layers': 4,
        'max_seq_len': 128,
        'dropout': 0.1,
        'sparse_k': 256
    }

    # Create manifold model
    print("Creating manifold model...")
    manifold_model = ManifoldModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        sparse_k=config['sparse_k'],
        use_manifold=True,
        manifold_d_hidden=64
    ).to(device)

    # Try to load trained model
    exp1_results_dir = Path(__file__).parent / 'results' / 'exp1_performance'
    manifold_ckpt = exp1_results_dir / 'Manifold_best.pt'

    if manifold_ckpt.exists():
        print(f"Loading manifold model from {manifold_ckpt}")
        ckpt = torch.load(manifold_ckpt, map_location=device)
        manifold_model.load_state_dict(ckpt['model_state_dict'])

    manifold_model.eval()

    # Create synthetic dataset with categories
    print("\nCreating categorized dataset...")

    categories_data = {
        'animals': [
            'cat dog pet animal',
            'bird fly wing',
            'fish water swim',
            'lion tiger wild',
            'elephant big mammal'
        ],
        'colors': [
            'red blue green',
            'yellow orange purple',
            'black white gray',
            'pink violet color',
            'brown beige hue'
        ],
        'numbers': [
            'one two three',
            'four five six',
            'seven eight nine',
            'ten eleven twelve',
            'zero count number'
        ],
        'vehicles': [
            'car drive road',
            'bus truck transport',
            'plane fly airport',
            'train rail station',
            'bike cycle pedal'
        ],
        'food': [
            'apple fruit eat',
            'bread bakery grain',
            'meat protein cook',
            'rice grain dish',
            'fish seafood meal'
        ]
    }

    # Flatten dataset
    dataset = []
    for category, texts in categories_data.items():
        for text in texts:
            dataset.append((text, category))

    # Collect representations
    print("Collecting manifold representations...")
    representations, labels = collect_manifold_representations(
        manifold_model, dataset, list(categories_data.keys()), device
    )

    # Visualize
    print("\nVisualizing manifold structure...")
    embeddings_2d = visualize_manifold_structure(
        representations, labels, save_dir, name='Manifold'
    )

    # Compute cluster quality
    print("\nComputing cluster quality...")
    quality_metrics = compute_cluster_quality(representations, labels)

    print("\n" + "="*80)
    print("CLUSTER QUALITY METRICS")
    print("="*80)
    print(f"Average Intra-class Distance: {quality_metrics['avg_intra_distance']:.4f}")
    print(f"Average Inter-class Distance: {quality_metrics['avg_inter_distance']:.4f}")
    print(f"Quality Score: {quality_metrics['quality_score']:.4f}")
    print("  (Higher is better: means clusters are well-separated)")
    print("="*80)

    # Save results
    results = {
        'config': config,
        'quality_metrics': quality_metrics,
        'num_samples': len(dataset),
        'num_categories': len(categories_data)
    }

    with open(save_dir / 'manifold_quality_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Experiment 3 completed!")
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
