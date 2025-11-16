"""
Analyze neuron specialization in SPROUT (Neuron Pool).

Analyzes:
- Which neurons specialize for which token types (nouns, verbs, etc.)
- Neuron usage patterns
- Degree of specialization

Usage:
    python scripts/analyze_neuron_specialization.py \\
        --checkpoint ./checkpoints/sprout_neuron_pool_best.pt \\
        --num_samples 1000
"""

import os
import sys
import argparse
import torch
import numpy as np
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sprout import SPROUT_MLM


def get_pos_tag_simple(token, tokenizer):
    """
    Simple POS tagging based on token patterns.

    Returns: 'noun', 'verb', 'adj', 'adv', 'func', 'special', 'unknown'
    """
    token_str = tokenizer.convert_ids_to_tokens([token])[0]

    # Special tokens
    if token_str in ['[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]']:
        return 'special'

    # Common functional words
    func_words = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                  'can', 'could', 'may', 'might', 'should', 'must',
                  'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from']
    if token_str in func_words:
        return 'func'

    # Common verb endings
    if token_str.endswith(('ing', 'ed', 'es', 'en')):
        return 'verb'

    # Common adjective/adverb endings
    if token_str.endswith(('ly', 'ous', 'ive', 'ful', 'less', 'able')):
        if token_str.endswith('ly'):
            return 'adv'
        return 'adj'

    # Common noun endings
    if token_str.endswith(('tion', 'ness', 'ment', 'ity', 'ism', 'er', 'or', 'ist')):
        return 'noun'

    # Capitalized likely noun (simplified)
    if token_str[0].isupper():
        return 'noun'

    return 'unknown'


def analyze_neuron_specialization(model, dataloader, tokenizer, device, num_batches=50):
    """
    Analyze which neurons specialize for which token types.

    Returns:
        Dictionary mapping neuron indices to token type statistics
    """
    model.eval()

    # Track: neuron -> token_type -> count
    neuron_token_types = defaultdict(lambda: defaultdict(int))

    # Track: token_type -> neuron -> count
    token_type_neurons = defaultdict(lambda: defaultdict(int))

    print(f"\n{'='*70}")
    print("ANALYZING NEURON SPECIALIZATION")
    print(f"{'='*70}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing", total=num_batches)):
            if batch_idx >= num_batches:
                break

            input_ids = batch['input_ids'].to(device)
            batch_size, seq_len = input_ids.shape

            # Get routing information
            outputs = model(input_ids=input_ids, return_routing=True)
            routing_info = outputs['routing_info']

            # Analyze each layer
            for layer_idx, layer_routing in enumerate(routing_info):
                indices = layer_routing['indices']  # [batch, seq, k]

                # For each token
                for b in range(batch_size):
                    for s in range(seq_len):
                        token_id = input_ids[b, s].item()

                        # Skip padding
                        if token_id == tokenizer.pad_token_id:
                            continue

                        # Get token type
                        token_type = get_pos_tag_simple(token_id, tokenizer)

                        # Get selected neurons for this token
                        selected_neurons = indices[b, s].cpu().tolist()

                        # Record usage
                        for neuron_id in selected_neurons:
                            neuron_token_types[(layer_idx, neuron_id)][token_type] += 1
                            token_type_neurons[token_type][(layer_idx, neuron_id)] += 1

    return neuron_token_types, token_type_neurons


def compute_specialization_score(neuron_type_counts):
    """
    Compute specialization score for a neuron.

    Returns value between 0 (not specialized) and 1 (highly specialized).
    Uses entropy-based measure.
    """
    total = sum(neuron_type_counts.values())
    if total == 0:
        return 0.0

    # Compute distribution
    probs = np.array([count / total for count in neuron_type_counts.values()])

    # Entropy (lower = more specialized)
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log(len(neuron_type_counts))
    if max_entropy == 0:
        return 0.0

    # Specialization = 1 - normalized_entropy
    specialization = 1 - (entropy / max_entropy)

    return specialization


def print_analysis(neuron_token_types, token_type_neurons, pool_size, num_layers):
    """Print detailed analysis."""
    print(f"\n{'='*70}")
    print("SPECIALIZATION ANALYSIS")
    print(f"{'='*70}")

    # Overall statistics
    token_types = ['noun', 'verb', 'adj', 'adv', 'func', 'special']

    for token_type in token_types:
        neurons = token_type_neurons[token_type]
        if not neurons:
            continue

        print(f"\n{token_type.upper()}:")

        # Top neurons for this token type
        top_neurons = sorted(neurons.items(), key=lambda x: x[1], reverse=True)[:10]

        for (layer, neuron_id), count in top_neurons:
            # Compute specialization
            type_counts = neuron_token_types[(layer, neuron_id)]
            spec_score = compute_specialization_score(type_counts)

            total_usage = sum(type_counts.values())
            pct = 100 * count / total_usage

            print(f"  Layer {layer}, Neuron {neuron_id:4d}: {count:5d} uses ({pct:5.1f}%), spec={spec_score:.3f}")

    # Overall specialization distribution
    print(f"\n{'='*70}")
    print("OVERALL SPECIALIZATION DISTRIBUTION")
    print(f"{'='*70}")

    specialization_scores = []
    for (layer, neuron_id), type_counts in neuron_token_types.items():
        spec_score = compute_specialization_score(type_counts)
        specialization_scores.append(spec_score)

    if specialization_scores:
        spec_scores = np.array(specialization_scores)
        print(f"Mean specialization: {spec_scores.mean():.3f}")
        print(f"Std specialization: {spec_scores.std():.3f}")
        print(f"Min specialization: {spec_scores.min():.3f}")
        print(f"Max specialization: {spec_scores.max():.3f}")

        # Histogram
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, _ = np.histogram(spec_scores, bins=bins)
        print(f"\nSpecialization distribution:")
        for i in range(len(hist)):
            low, high = bins[i], bins[i+1]
            count = hist[i]
            pct = 100 * count / len(spec_scores)
            print(f"  [{low:.1f}, {high:.1f}): {count:5d} ({pct:5.1f}%)")


def plot_specialization(neuron_token_types, save_path=None):
    """Plot specialization heatmap."""
    # Extract data for plotting
    token_types = ['noun', 'verb', 'adj', 'adv', 'func', 'special', 'unknown']

    # Collect specialization scores
    scores_by_type = {tt: [] for tt in token_types}

    for (layer, neuron_id), type_counts in neuron_token_types.items():
        spec_score = compute_specialization_score(type_counts)

        # Find dominant type
        if type_counts:
            dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
            scores_by_type[dominant_type].append(spec_score)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = list(range(len(token_types)))
    data = [scores_by_type[tt] for tt in token_types]

    bp = ax.boxplot(data, positions=positions, labels=token_types, patch_artist=True)

    # Styling
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_xlabel('Token Type')
    ax.set_ylabel('Specialization Score')
    ax.set_title('Neuron Specialization by Token Type')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze neuron specialization")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to analyze')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_batches', type=int, default=50,
                        help='Number of batches to analyze')
    parser.add_argument('--save_plot', type=str,
                        help='Path to save plot')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"🔧 Using device: {device}")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load checkpoint
    print(f"📥 Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    checkpoint_args = checkpoint['args']

    model = SPROUT_MLM(
        vocab_size=len(tokenizer),
        d_model=checkpoint_args.get('d_model', 256),
        pool_size=checkpoint_args.get('pool_size', 4096),
        k=checkpoint_args.get('k', 128),
        n_steps=checkpoint_args.get('n_steps', 3),
        n_heads=checkpoint_args.get('n_heads', 4),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"✅ Loaded checkpoint (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")

    # Load data
    print(f"\n🔄 Loading sample data...")
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train[:1%]",
        trust_remote_code=False
    )

    # Simple tokenization for analysis
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    dataloader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Analyze
    neuron_token_types, token_type_neurons = analyze_neuron_specialization(
        model,
        dataloader,
        tokenizer,
        device,
        num_batches=args.num_batches
    )

    # Print analysis
    print_analysis(
        neuron_token_types,
        token_type_neurons,
        checkpoint_args.get('pool_size', 4096),
        checkpoint_args.get('n_steps', 3)
    )

    # Plot
    try:
        plot_specialization(neuron_token_types, save_path=args.save_plot)
    except Exception as e:
        print(f"⚠️  Could not create plot: {e}")


if __name__ == '__main__':
    main()
