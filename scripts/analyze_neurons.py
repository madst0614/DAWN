"""
DAWN Neuron Activation Analysis for GLUE Tasks

Usage:
    python analyze_neurons.py --task sst2 --results_dir glue_results
    python analyze_neurons.py --task all --results_dir glue_results
    python analyze_neurons.py --compare --results_dir glue_results

Features:
    1. Per-task neuron usage distribution
    2. Cross-task neuron comparison
    3. Layer-wise activation patterns
    4. Heatmap visualizations
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available, plotting disabled")


# ============================================================
# Neuron Analysis Functions
# ============================================================

def load_neuron_activations(neuron_path):
    """Load saved neuron activations from file"""
    if not os.path.exists(neuron_path):
        print(f"File not found: {neuron_path}")
        return None

    activations = torch.load(neuron_path, map_location='cpu')
    return activations


def aggregate_activations(activations):
    """
    Aggregate neuron activations across batches

    Returns:
        layer_activations: dict with 'attention' and 'memory' keys
                          each containing [n_layers, n_compress] mean activations
    """
    if activations is None or len(activations) == 0:
        return None

    # Structure: list of batches, each batch is list of layers
    # Each layer has 'attention' and 'memory' with [B, n_compress] tensors

    n_layers = len(activations[0])
    attn_activations = [[] for _ in range(n_layers)]
    mem_activations = [[] for _ in range(n_layers)]

    for batch in activations:
        for layer_idx, layer_info in enumerate(batch):
            attn_activations[layer_idx].append(layer_info['attention'])
            mem_activations[layer_idx].append(layer_info['memory'])

    # Concatenate and compute mean
    layer_attn_mean = []
    layer_mem_mean = []

    for layer_idx in range(n_layers):
        attn = torch.cat(attn_activations[layer_idx], dim=0)  # [total_samples, n_compress]
        mem = torch.cat(mem_activations[layer_idx], dim=0)

        layer_attn_mean.append(attn.mean(dim=0).numpy())  # [n_compress]
        layer_mem_mean.append(mem.mean(dim=0).numpy())

    return {
        'attention': np.stack(layer_attn_mean),  # [n_layers, n_compress]
        'memory': np.stack(layer_mem_mean),
    }


def compute_neuron_statistics(aggregated_activations):
    """Compute statistics about neuron usage"""
    if aggregated_activations is None:
        return None

    stats = {}

    for module in ['attention', 'memory']:
        activations = aggregated_activations[module]  # [n_layers, n_compress]
        n_layers, n_compress = activations.shape

        # Per-layer statistics
        layer_stats = []
        for layer_idx in range(n_layers):
            layer_act = activations[layer_idx]
            layer_stats.append({
                'mean': float(layer_act.mean()),
                'std': float(layer_act.std()),
                'max': float(layer_act.max()),
                'min': float(layer_act.min()),
                'entropy': float(-np.sum(layer_act * np.log(layer_act + 1e-10))),
                'top_neurons': np.argsort(layer_act)[-5:][::-1].tolist(),
            })

        # Global statistics (across layers)
        global_act = activations.mean(axis=0)  # [n_compress]
        top_neurons_global = np.argsort(global_act)[-10:][::-1].tolist()
        bottom_neurons_global = np.argsort(global_act)[:10].tolist()

        stats[module] = {
            'layer_stats': layer_stats,
            'global_mean': float(global_act.mean()),
            'global_std': float(global_act.std()),
            'top_neurons': top_neurons_global,
            'bottom_neurons': bottom_neurons_global,
            'activation_matrix': activations.tolist(),
        }

    return stats


def compare_task_neurons(task_stats_dict):
    """
    Compare neuron usage across different tasks

    Args:
        task_stats_dict: dict mapping task_name -> stats
    """
    if len(task_stats_dict) < 2:
        print("Need at least 2 tasks for comparison")
        return None

    comparison = {
        'attention': {},
        'memory': {},
    }

    for module in ['attention', 'memory']:
        # Collect top neurons from each task
        task_top_neurons = {}
        task_activations = {}

        for task_name, stats in task_stats_dict.items():
            if stats is None or module not in stats:
                continue
            task_top_neurons[task_name] = set(stats[module]['top_neurons'])
            task_activations[task_name] = np.array(stats[module]['activation_matrix'])

        # Find shared vs task-specific neurons
        all_top = set()
        for neurons in task_top_neurons.values():
            all_top.update(neurons)

        shared_neurons = all_top.copy()
        for neurons in task_top_neurons.values():
            shared_neurons.intersection_update(neurons)

        task_specific = {}
        for task_name, neurons in task_top_neurons.items():
            task_specific[task_name] = list(neurons - shared_neurons)

        # Compute correlation between task activation patterns
        task_names = list(task_activations.keys())
        correlation_matrix = np.zeros((len(task_names), len(task_names)))

        for i, task1 in enumerate(task_names):
            for j, task2 in enumerate(task_names):
                act1 = task_activations[task1].flatten()
                act2 = task_activations[task2].flatten()
                correlation_matrix[i, j] = np.corrcoef(act1, act2)[0, 1]

        comparison[module] = {
            'shared_neurons': list(shared_neurons),
            'task_specific_neurons': task_specific,
            'correlation_matrix': correlation_matrix.tolist(),
            'task_names': task_names,
        }

    return comparison


# ============================================================
# Visualization Functions
# ============================================================

def plot_neuron_heatmap(stats, task_name, output_dir, module='attention'):
    """Plot neuron activation heatmap for a task"""
    if not HAS_PLOTTING:
        print("Plotting not available")
        return

    if stats is None or module not in stats:
        print(f"No stats for {module}")
        return

    activations = np.array(stats[module]['activation_matrix'])  # [n_layers, n_compress]

    plt.figure(figsize=(14, 6))
    sns.heatmap(
        activations,
        cmap='YlOrRd',
        xticklabels=10,
        yticklabels=True,
        cbar_kws={'label': 'Activation Weight'}
    )
    plt.xlabel('Neuron Index')
    plt.ylabel('Layer')
    plt.title(f'{task_name.upper()} - {module.capitalize()} Neuron Activations')

    save_path = os.path.join(output_dir, f'{task_name}_{module}_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap: {save_path}")


def plot_top_neurons_bar(stats, task_name, output_dir, module='attention'):
    """Plot top neuron usage as bar chart"""
    if not HAS_PLOTTING:
        return

    if stats is None or module not in stats:
        return

    activations = np.array(stats[module]['activation_matrix'])
    global_act = activations.mean(axis=0)
    n_compress = len(global_act)

    # Sort by activation
    sorted_idx = np.argsort(global_act)[::-1]

    plt.figure(figsize=(12, 5))
    plt.bar(range(n_compress), global_act[sorted_idx], color='steelblue', alpha=0.7)
    plt.xlabel('Neuron (sorted by activation)')
    plt.ylabel('Mean Activation Weight')
    plt.title(f'{task_name.upper()} - {module.capitalize()} Neuron Usage Distribution')

    # Highlight top 10
    for i in range(min(10, n_compress)):
        plt.bar(i, global_act[sorted_idx[i]], color='coral')

    save_path = os.path.join(output_dir, f'{task_name}_{module}_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution: {save_path}")


def plot_task_comparison(comparison, output_dir, module='attention'):
    """Plot cross-task comparison"""
    if not HAS_PLOTTING:
        return

    if comparison is None or module not in comparison:
        return

    comp = comparison[module]
    task_names = comp['task_names']
    corr_matrix = np.array(comp['correlation_matrix'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        xticklabels=[t.upper() for t in task_names],
        yticklabels=[t.upper() for t in task_names],
        vmin=-1, vmax=1,
        center=0,
    )
    plt.title(f'{module.capitalize()} Neuron Pattern Correlation Across Tasks')

    save_path = os.path.join(output_dir, f'task_correlation_{module}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation: {save_path}")


def plot_shared_vs_specific(comparison, output_dir, module='attention'):
    """Plot shared vs task-specific neurons"""
    if not HAS_PLOTTING:
        return

    if comparison is None or module not in comparison:
        return

    comp = comparison[module]
    shared = comp['shared_neurons']
    specific = comp['task_specific_neurons']

    # Count
    task_names = list(specific.keys())
    shared_count = len(shared)
    specific_counts = [len(specific[t]) for t in task_names]

    plt.figure(figsize=(10, 5))

    x = np.arange(len(task_names) + 1)
    labels = ['Shared'] + [t.upper() for t in task_names]
    values = [shared_count] + specific_counts
    colors = ['green'] + ['steelblue'] * len(task_names)

    plt.bar(x, values, color=colors, alpha=0.7)
    plt.xticks(x, labels)
    plt.ylabel('Number of Top Neurons')
    plt.title(f'{module.capitalize()} - Shared vs Task-Specific Top Neurons')

    for i, v in enumerate(values):
        plt.text(i, v + 0.1, str(v), ha='center')

    save_path = os.path.join(output_dir, f'shared_vs_specific_{module}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved shared/specific: {save_path}")


# ============================================================
# Main Analysis Function
# ============================================================

def analyze_task(task_name, results_dir, output_dir):
    """Analyze neuron activations for a single task"""
    print(f"\n{'='*60}")
    print(f"Analyzing {task_name.upper()}")
    print(f"{'='*60}")

    neuron_path = os.path.join(results_dir, f'{task_name}_neurons.pt')

    # Load activations
    activations = load_neuron_activations(neuron_path)
    if activations is None:
        print(f"No neuron data found for {task_name}")
        print(f"Run: python train_glue.py --task {task_name} --collect_neurons ...")
        return None

    # Aggregate
    aggregated = aggregate_activations(activations)
    if aggregated is None:
        print(f"Failed to aggregate activations for {task_name}")
        return None

    # Compute statistics
    stats = compute_neuron_statistics(aggregated)

    # Print summary
    for module in ['attention', 'memory']:
        print(f"\n{module.upper()} Module:")
        print(f"  Global mean activation: {stats[module]['global_mean']:.4f}")
        print(f"  Global std: {stats[module]['global_std']:.4f}")
        print(f"  Top 10 neurons: {stats[module]['top_neurons']}")
        print(f"  Bottom 10 neurons: {stats[module]['bottom_neurons']}")

        print(f"\n  Layer-wise top neurons:")
        for i, layer_stat in enumerate(stats[module]['layer_stats']):
            print(f"    Layer {i}: {layer_stat['top_neurons']} (entropy={layer_stat['entropy']:.3f})")

    # Plot
    os.makedirs(output_dir, exist_ok=True)
    for module in ['attention', 'memory']:
        plot_neuron_heatmap(stats, task_name, output_dir, module)
        plot_top_neurons_bar(stats, task_name, output_dir, module)

    # Save stats
    stats_path = os.path.join(output_dir, f'{task_name}_neuron_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {stats_path}")

    return stats


def compare_tasks(results_dir, output_dir, tasks=None):
    """Compare neuron usage across tasks"""
    print(f"\n{'='*60}")
    print("Cross-Task Neuron Comparison")
    print(f"{'='*60}")

    # Find all available tasks
    if tasks is None:
        tasks = []
        for f in os.listdir(results_dir):
            if f.endswith('_neurons.pt'):
                task = f.replace('_neurons.pt', '')
                tasks.append(task)

    if len(tasks) < 2:
        print("Need at least 2 tasks with neuron data for comparison")
        print(f"Found: {tasks}")
        return

    print(f"Comparing tasks: {', '.join(tasks)}")

    # Load and analyze each task
    task_stats = {}
    for task in tasks:
        neuron_path = os.path.join(results_dir, f'{task}_neurons.pt')
        activations = load_neuron_activations(neuron_path)
        if activations is not None:
            aggregated = aggregate_activations(activations)
            stats = compute_neuron_statistics(aggregated)
            task_stats[task] = stats

    if len(task_stats) < 2:
        print("Not enough tasks with valid data")
        return

    # Compare
    comparison = compare_task_neurons(task_stats)

    # Print summary
    for module in ['attention', 'memory']:
        print(f"\n{module.upper()} Module Comparison:")
        comp = comparison[module]
        print(f"  Shared top neurons (all tasks): {comp['shared_neurons']}")
        print(f"  Task-specific top neurons:")
        for task, neurons in comp['task_specific_neurons'].items():
            print(f"    {task.upper()}: {neurons}")

    # Plot comparisons
    os.makedirs(output_dir, exist_ok=True)
    for module in ['attention', 'memory']:
        plot_task_comparison(comparison, output_dir, module)
        plot_shared_vs_specific(comparison, output_dir, module)

    # Save comparison
    comparison_path = os.path.join(output_dir, 'task_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to: {comparison_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN Neuron Analysis')
    parser.add_argument('--task', type=str, default=None,
                        help='Task to analyze (or "all")')
    parser.add_argument('--compare', action='store_true',
                        help='Compare neuron usage across tasks')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing GLUE results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for analysis')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'analysis')

    os.makedirs(args.output_dir, exist_ok=True)

    if args.compare:
        compare_tasks(args.results_dir, args.output_dir)
    elif args.task:
        if args.task.lower() == 'all':
            # Analyze all available tasks
            for f in os.listdir(args.results_dir):
                if f.endswith('_neurons.pt'):
                    task = f.replace('_neurons.pt', '')
                    analyze_task(task, args.results_dir, args.output_dir)
            # Then compare
            compare_tasks(args.results_dir, args.output_dir)
        else:
            analyze_task(args.task, args.results_dir, args.output_dir)
    else:
        print("Please specify --task or --compare")
        parser.print_help()


if __name__ == '__main__':
    main()
