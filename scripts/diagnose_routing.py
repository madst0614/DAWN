"""
Diagnostic script to analyze SPROUT routing behavior.

Analyzes:
- Compatibility score distributions
- Routing decisions (create vs route)
- Gate strength patterns
- Path usage statistics

Usage:
    python scripts/diagnose_routing.py --num_samples 100
"""

import os
import sys
import argparse
import torch
import numpy as np
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sprout import SproutLanguageModel


def collect_routing_stats(model, dataloader, num_batches=10, device='cpu'):
    """
    Collect detailed routing statistics.

    Returns:
        Dictionary with compatibility scores, gate strengths, routing decisions
    """
    stats = {
        'compatibilities': [],
        'gate_strengths': [],
        'routing_decisions': [],  # 'create' or 'route'
        'depths': [],
        'path_lengths': [],
    }

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting stats")):
            if batch_idx >= num_batches:
                break

            input_ids = batch['input_ids'].to(device)

            # Forward with detailed logging
            outputs = model(input_ids=input_ids, return_path_log=True)
            path_log = outputs.get('path_log', [])

            for log in path_log:
                # Compatibility scores
                if 'compatibility' in log:
                    compat = log['compatibility']
                    if isinstance(compat, torch.Tensor):
                        compat = compat.item()
                    stats['compatibilities'].append(compat)

                # Routing decisions
                action = log.get('action', 'unknown')
                stats['routing_decisions'].append(action)

                # Depth
                depth = log.get('depth', 0)
                stats['depths'].append(depth)

            # Path lengths
            stats['path_lengths'].append(len(path_log))

    return stats


def print_routing_analysis(stats):
    """Print detailed routing analysis."""
    print("\n" + "="*70)
    print("ROUTING BEHAVIOR ANALYSIS")
    print("="*70)

    # Compatibility distribution
    if stats['compatibilities']:
        compats = np.array(stats['compatibilities'])
        print(f"\n📊 Compatibility Scores:")
        print(f"   Mean: {compats.mean():.4f}")
        print(f"   Std:  {compats.std():.4f}")
        print(f"   Min:  {compats.min():.4f}")
        print(f"   Max:  {compats.max():.4f}")
        print(f"   Median: {np.median(compats):.4f}")
        print(f"\n   Percentiles:")
        print(f"   10%: {np.percentile(compats, 10):.4f}")
        print(f"   25%: {np.percentile(compats, 25):.4f}")
        print(f"   50%: {np.percentile(compats, 50):.4f}")
        print(f"   75%: {np.percentile(compats, 75):.4f}")
        print(f"   90%: {np.percentile(compats, 90):.4f}")

    # Routing decisions
    if stats['routing_decisions']:
        decisions = stats['routing_decisions']
        unique, counts = np.unique(decisions, return_counts=True)
        print(f"\n🔀 Routing Decisions:")
        for action, count in zip(unique, counts):
            pct = 100 * count / len(decisions)
            print(f"   {action}: {count} ({pct:.1f}%)")

    # Depth distribution
    if stats['depths']:
        depths = np.array(stats['depths'])
        print(f"\n📏 Depth Distribution:")
        unique, counts = np.unique(depths, return_counts=True)
        for depth, count in zip(unique, counts):
            pct = 100 * count / len(depths)
            print(f"   Depth {depth}: {count} ({pct:.1f}%)")

    # Path lengths
    if stats['path_lengths']:
        lengths = np.array(stats['path_lengths'])
        print(f"\n🛤️  Path Lengths:")
        print(f"   Mean: {lengths.mean():.2f}")
        print(f"   Std:  {lengths.std():.2f}")
        print(f"   Min:  {lengths.min()}")
        print(f"   Max:  {lengths.max()}")

    print("="*70 + "\n")


def plot_distributions(stats, save_path=None):
    """Plot routing statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Compatibility histogram
    if stats['compatibilities']:
        ax = axes[0, 0]
        ax.hist(stats['compatibilities'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0.5, color='red', linestyle='--', label='threshold=0.5')
        ax.axvline(0.8, color='orange', linestyle='--', label='threshold=0.8')
        ax.set_xlabel('Compatibility Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Compatibility Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Routing decisions
    if stats['routing_decisions']:
        ax = axes[0, 1]
        decisions = stats['routing_decisions']
        unique, counts = np.unique(decisions, return_counts=True)
        ax.bar(unique, counts, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Action')
        ax.set_ylabel('Count')
        ax.set_title('Routing Decisions')
        ax.grid(True, alpha=0.3)

    # Depth distribution
    if stats['depths']:
        ax = axes[1, 0]
        depths = stats['depths']
        unique, counts = np.unique(depths, return_counts=True)
        ax.bar(unique, counts, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Depth')
        ax.set_ylabel('Count')
        ax.set_title('Depth Distribution')
        ax.grid(True, alpha=0.3)

    # Path length histogram
    if stats['path_lengths']:
        ax = axes[1, 1]
        ax.hist(stats['path_lengths'], bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Path Length')
        ax.set_ylabel('Frequency')
        ax.set_title('Path Length Distribution')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved plot to {save_path}")
    else:
        plt.show()


def analyze_node_usage(model):
    """Analyze individual node usage patterns."""
    print("\n" + "="*70)
    print("NODE USAGE ANALYSIS")
    print("="*70)

    def analyze_node(node, prefix="", depth=0):
        """Recursively analyze node usage."""
        usage = node.usage_count
        num_children = len(node.child_nodes)

        print(f"{prefix}Node {node.node_id} (depth={depth})")
        print(f"{prefix}  Usage: {usage}")
        print(f"{prefix}  Children: {num_children}")

        if hasattr(node, 'router') and node.router is not None:
            print(f"{prefix}  Has router: Yes")

        # Recurse to children
        for i, child in enumerate(node.child_nodes):
            child_prefix = prefix + "  "
            if i == len(node.child_nodes) - 1:
                child_prefix = prefix + "  └─ "
            else:
                child_prefix = prefix + "  ├─ "
            analyze_node(child, child_prefix, depth + 1)

    analyze_node(model.sprout.root)
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Diagnose SPROUT routing behavior")
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint (optional)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to analyze')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to analyze')
    parser.add_argument('--save_plot', type=str, help='Path to save plot')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Model config (if not loading checkpoint)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--max_nodes', type=int, default=20)
    parser.add_argument('--compatibility_threshold', type=float, default=0.5)

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"🔧 Using device: {device}")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load or create model
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"📥 Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        model = SproutLanguageModel(
            vocab_size=len(tokenizer),
            hidden_dim=checkpoint['model_info']['hidden_dim'],
            max_depth=checkpoint['model_info']['max_depth'],
            max_nodes=checkpoint['model_info'].get('max_nodes', 20),
            compatibility_threshold=checkpoint['model_info'].get('compatibility_threshold', 0.5)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print(f"✅ Loaded checkpoint (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
    else:
        print("🔨 Creating new model")
        model = SproutLanguageModel(
            vocab_size=len(tokenizer),
            hidden_dim=args.hidden_dim,
            max_depth=args.max_depth,
            max_nodes=args.max_nodes,
            compatibility_threshold=args.compatibility_threshold
        ).to(device)

    # Show model info
    print(f"\n📊 Model Configuration:")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Hidden dim: {model.hidden_dim}")
    print(f"   Max depth: {model.sprout.root.max_depth}")
    print(f"   Max nodes: {model.sprout.max_nodes}")
    print(f"   Compatibility threshold: {model.sprout.root.compatibility_threshold}")
    print(f"   Total nodes: {model.sprout.count_total_nodes()}")

    # Load sample data
    print(f"\n🔄 Loading sample data...")
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train[:1%]",
        trust_remote_code=False
    )

    # Tokenize
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
        shuffle=True
    )

    # Collect statistics
    print(f"\n🔍 Analyzing routing behavior...")
    stats = collect_routing_stats(model, dataloader, num_batches=args.num_batches, device=device)

    # Print analysis
    print_routing_analysis(stats)

    # Analyze node usage
    analyze_node_usage(model)

    # Visualize structure
    print("\n🌲 Model Structure:")
    model.visualize_structure()

    # Plot distributions
    try:
        plot_distributions(stats, save_path=args.save_plot)
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")


if __name__ == '__main__':
    main()
