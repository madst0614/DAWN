"""
Experiment 2: Neuron Semantics Analysis
목표: 뉴런들이 정말 compositional한 정보를 학습하는지

분석 내용:
- 비슷한 입력들이 비슷한 뉴런을 선택하는가?
- Manifold 버전이 더 일관성 있는 패턴을 보이는가?
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.neuron_based import NeuronBasedLanguageModel as BaselineModel
from models.neuron_based_manifold import NeuronBasedLanguageModel as ManifoldModel


def get_selected_neurons(model, input_ids, layer_idx=0):
    """Get selected neurons for a given input"""
    model.eval()

    with torch.no_grad():
        # Forward through embeddings
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        batch_size, seq_len = input_ids.shape
        token_emb = model.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = model.position_embedding(positions)
        x = token_emb + pos_emb

        # Forward through specified layer's FFN router
        layer = model.layers[layer_idx]

        # Get through attention first
        x = layer.norm1(x)
        x_attn, _ = layer.attention(x, x, x)
        x = x + layer.dropout(x_attn)

        # Get FFN input
        x_ffn = layer.norm2(x)

        # Get router scores
        x_flat = x_ffn.view(-1, model.d_model)
        scores = layer.ffn.router.compute_scores(x_flat)

        # Get top-k neurons
        top_k = model.sparse_k if model.sparse_k is not None else model.d_ff
        _, top_indices = torch.topk(scores, top_k, dim=-1)

        return top_indices.cpu().numpy()


def compute_neuron_overlap(neurons1, neurons2):
    """Compute overlap ratio between two neuron sets"""
    # neurons: [batch*seq, top_k]
    # Compute average overlap across all tokens
    overlaps = []

    for n1, n2 in zip(neurons1, neurons2):
        set1 = set(n1)
        set2 = set(n2)
        overlap = len(set1 & set2) / len(set1)
        overlaps.append(overlap)

    return np.mean(overlaps)


def analyze_semantic_pairs(model, tokenizer, pairs, model_name, device):
    """Analyze neuron overlap for semantic pairs"""

    results = []

    for pair_type, text1, text2 in pairs:
        # Tokenize
        input_ids1 = tokenizer.encode(text1, return_tensors='pt').to(device)
        input_ids2 = tokenizer.encode(text2, return_tensors='pt').to(device)

        # Get selected neurons
        neurons1 = get_selected_neurons(model, input_ids1, layer_idx=0)
        neurons2 = get_selected_neurons(model, input_ids2, layer_idx=0)

        # Compute overlap
        overlap = compute_neuron_overlap(neurons1, neurons2)

        results.append({
            'pair_type': pair_type,
            'text1': text1,
            'text2': text2,
            'overlap': overlap
        })

        print(f"{model_name} | {pair_type:15s} | {text1:20s} vs {text2:20s} | Overlap: {overlap:.2%}")

    return results


def analyze_neuron_consistency(model, test_inputs, model_name, device):
    """
    Analyze neuron selection consistency
    - Same input → should get same neurons
    - Similar inputs → should get similar neurons
    """

    # Test same input multiple times
    print(f"\n{model_name}: Testing consistency on same input...")

    input_ids = test_inputs[0].to(device)
    neurons_list = []

    for _ in range(5):
        neurons = get_selected_neurons(model, input_ids, layer_idx=0)
        neurons_list.append(neurons)

    # Compute pairwise overlap
    consistency_scores = []
    for i in range(len(neurons_list)):
        for j in range(i+1, len(neurons_list)):
            overlap = compute_neuron_overlap(neurons_list[i], neurons_list[j])
            consistency_scores.append(overlap)

    avg_consistency = np.mean(consistency_scores)
    print(f"  Average consistency: {avg_consistency:.2%}")

    return avg_consistency


def visualize_neuron_overlap_matrix(results_baseline, results_manifold, save_dir):
    """Visualize overlap patterns as heatmap"""

    # Group by pair type
    pair_types = ['similar', 'different', 'partial']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (results, name) in enumerate([(results_baseline, 'Baseline'),
                                             (results_manifold, 'Manifold')]):
        # Create overlap matrix
        overlaps_by_type = defaultdict(list)
        for r in results:
            overlaps_by_type[r['pair_type']].append(r['overlap'])

        # Prepare data for plotting
        data = []
        labels = []
        for pt in pair_types:
            if pt in overlaps_by_type:
                data.append(overlaps_by_type[pt])
                labels.append(pt.capitalize())

        # Box plot
        ax = axes[idx]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        ax.set_ylabel('Neuron Overlap Ratio')
        ax.set_title(f'{name} - Neuron Overlap by Pair Type')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(save_dir / 'neuron_overlap_comparison.png', dpi=150)
    print(f"\nVisualization saved to {save_dir / 'neuron_overlap_comparison.png'}")


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save directory
    save_dir = Path(__file__).parent / 'results' / 'exp2_semantics'
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

    # Load or create models
    print("Creating models...")

    baseline_model = BaselineModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        sparse_k=config['sparse_k']
    ).to(device)

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

    # Try to load trained models if available
    exp1_results_dir = Path(__file__).parent / 'results' / 'exp1_performance'
    baseline_ckpt = exp1_results_dir / 'Baseline_best.pt'
    manifold_ckpt = exp1_results_dir / 'Manifold_best.pt'

    if baseline_ckpt.exists():
        print(f"Loading baseline from {baseline_ckpt}")
        ckpt = torch.load(baseline_ckpt, map_location=device)
        baseline_model.load_state_dict(ckpt['model_state_dict'])

    if manifold_ckpt.exists():
        print(f"Loading manifold from {manifold_ckpt}")
        ckpt = torch.load(manifold_ckpt, map_location=device)
        manifold_model.load_state_dict(ckpt['model_state_dict'])

    baseline_model.eval()
    manifold_model.eval()

    # Create simple tokenizer (for demonstration)
    # In real use, use actual tokenizer
    class SimpleTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size

        def encode(self, text, return_tensors='pt'):
            # Simple hash-based encoding
            tokens = [hash(word) % self.vocab_size for word in text.split()]
            if return_tensors == 'pt':
                return torch.tensor([tokens])
            return tokens

    tokenizer = SimpleTokenizer(config['vocab_size'])

    # Define test pairs
    pairs = [
        # Similar pairs (should have high overlap)
        ('similar', 'cat dog animal', 'dog cat pet'),
        ('similar', 'red blue color', 'green yellow color'),
        ('similar', 'hello world greeting', 'hi world hello'),

        # Different pairs (should have low overlap)
        ('different', 'cat dog animal', 'number one two'),
        ('different', 'red blue color', 'car vehicle drive'),
        ('different', 'hello world greeting', 'math compute calculate'),

        # Partial similarity (medium overlap)
        ('partial', 'red ball toy', 'blue ball game'),
        ('partial', 'fast car vehicle', 'slow car road'),
        ('partial', 'big cat animal', 'small dog animal'),
    ]

    # Analyze baseline
    print("\n" + "="*80)
    print("BASELINE MODEL - Neuron Overlap Analysis")
    print("="*80)
    results_baseline = analyze_semantic_pairs(
        baseline_model, tokenizer, pairs, "Baseline", device
    )

    # Analyze manifold
    print("\n" + "="*80)
    print("MANIFOLD MODEL - Neuron Overlap Analysis")
    print("="*80)
    results_manifold = analyze_semantic_pairs(
        manifold_model, tokenizer, pairs, "Manifold", device
    )

    # Test consistency
    print("\n" + "="*80)
    print("CONSISTENCY TEST")
    print("="*80)

    test_inputs = [
        tokenizer.encode("hello world test", return_tensors='pt'),
        tokenizer.encode("another test example", return_tensors='pt'),
    ]

    consistency_baseline = analyze_neuron_consistency(
        baseline_model, test_inputs, "Baseline", device
    )
    consistency_manifold = analyze_neuron_consistency(
        manifold_model, test_inputs, "Manifold", device
    )

    # Visualize results
    visualize_neuron_overlap_matrix(results_baseline, results_manifold, save_dir)

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for model_name, results in [('Baseline', results_baseline),
                                  ('Manifold', results_manifold)]:
        print(f"\n{model_name}:")
        for pair_type in ['similar', 'different', 'partial']:
            overlaps = [r['overlap'] for r in results if r['pair_type'] == pair_type]
            if overlaps:
                print(f"  {pair_type.capitalize():10s}: "
                      f"mean={np.mean(overlaps):.2%}, "
                      f"std={np.std(overlaps):.2%}")

    # Save results
    all_results = {
        'baseline': results_baseline,
        'manifold': results_manifold,
        'consistency': {
            'baseline': consistency_baseline,
            'manifold': consistency_manifold
        },
        'config': config
    }

    with open(save_dir / 'semantics_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj

        json.dump(all_results, f, indent=2, default=convert)

    print(f"\n✓ Experiment 2 completed!")
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
