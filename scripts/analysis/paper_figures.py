"""
Paper Figure Generator
=======================
Generate publication-ready figures for DAWN paper.

Provides a unified interface to run all analyses and generate
figures in a consistent format.
"""

import os
import json
import numpy as np
from typing import Dict, Optional

from .utils import (
    load_model, get_router, get_neurons, create_dataloader,
    convert_to_serializable, save_results,
    HAS_MATPLOTLIB
)

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt
    import matplotlib


class PaperFigureGenerator:
    """Generate paper-ready figures from DAWN analysis."""

    def __init__(self, checkpoint_path: str, val_data_path: Optional[str] = None,
                 device: str = 'cuda'):
        """
        Initialize generator.

        Args:
            checkpoint_path: Path to model checkpoint
            val_data_path: Path to validation data (optional)
            device: Device for computation
        """
        print("Loading model...")
        self.model, self.tokenizer, self.config = load_model(checkpoint_path, device)
        self.router = get_router(self.model)
        self.neurons = get_neurons(self.model)
        self.device = device

        self.dataloader = None
        if val_data_path:
            print(f"Loading validation data from: {val_data_path}")
            self.dataloader = create_dataloader(val_data_path, self.tokenizer)

        # Import analyzers
        from .neuron_health import NeuronHealthAnalyzer
        from .routing import RoutingAnalyzer
        from .embedding import EmbeddingAnalyzer
        from .weight import WeightAnalyzer
        from .behavioral import BehavioralAnalyzer
        from .semantic import SemanticAnalyzer
        from .coselection import CoselectionAnalyzer

        # Initialize analyzers
        self.health = NeuronHealthAnalyzer(self.router)
        self.routing = RoutingAnalyzer(self.model, self.router, device)
        self.embedding = EmbeddingAnalyzer(self.router, device)
        self.weight = WeightAnalyzer(self.neurons, device)
        self.behavioral = BehavioralAnalyzer(self.model, self.router, self.tokenizer, device)
        self.semantic = SemanticAnalyzer(self.model, self.router, self.tokenizer, device)
        self.coselection = CoselectionAnalyzer(self.model, self.router, self.neurons, device)

    def generate_all(self, output_dir: str = './paper_figures', n_batches: int = 50):
        """
        Generate all paper figures.

        Args:
            output_dir: Directory for outputs
            n_batches: Number of batches for data-dependent analyses
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*60, flush=True)
        print("GENERATING PAPER FIGURES", flush=True)
        print("="*60, flush=True)

        all_results = {}

        # 1. Neuron Utilization
        print("\n[1/8] Neuron Health & Utilization...", flush=True)
        health_dir = os.path.join(output_dir, 'health')
        all_results['health'] = self.health.run_all(health_dir)

        # 2. Usage Histogram
        print("\n[2/8] Usage Histogram...", flush=True)
        self.table_neuron_utilization(all_results['health'], output_dir)

        # 3. Embedding Visualization
        print("\n[3/8] Embedding Analysis...", flush=True)
        emb_dir = os.path.join(output_dir, 'embedding')
        all_results['embedding'] = self.embedding.run_all(emb_dir)

        # 4. Weight Analysis
        print("\n[4/8] Weight Analysis...", flush=True)
        weight_dir = os.path.join(output_dir, 'weight')
        all_results['weight'] = self.weight.run_all(weight_dir)

        # 5. Semantic Analysis
        print("\n[5/8] Semantic Analysis...", flush=True)
        sem_dir = os.path.join(output_dir, 'semantic')
        all_results['semantic'] = self.semantic.run_all(self.dataloader, sem_dir, max_batches=n_batches)

        # Data-dependent analyses
        if self.dataloader is not None:
            # 6. Routing Analysis
            print("\n[6/8] Routing Analysis...", flush=True)
            routing_dir = os.path.join(output_dir, 'routing')
            all_results['routing'] = self.routing.run_all(self.dataloader, routing_dir, n_batches)

            # 7. Behavioral Analysis
            print("\n[7/8] Behavioral Analysis...", flush=True)
            behav_dir = os.path.join(output_dir, 'behavioral')
            all_results['behavioral'] = self.behavioral.run_all(
                self.dataloader, behav_dir, n_batches
            )
            print("  Behavioral analysis complete.", flush=True)

            # 8. Co-selection Analysis
            print("\n[8/8] Co-selection Analysis...", flush=True)
            cosel_dir = os.path.join(output_dir, 'coselection')
            all_results['coselection'] = self.coselection.run_all(
                self.dataloader, cosel_dir, 'all', n_batches
            )
            print("  Co-selection analysis complete.", flush=True)
        else:
            print("\n[6-8/8] Skipping data-dependent analyses (no dataloader)", flush=True)

        # Generate summary figure
        print("\n--- Generating Summary Figure ---", flush=True)
        self.figure_summary(all_results, output_dir)

        # Save all results
        results_path = os.path.join(output_dir, 'all_results.json')
        save_results(all_results, results_path)
        print(f"\nResults saved to: {results_path}", flush=True)

        print("\n" + "="*60, flush=True)
        print(f"All figures saved to: {output_dir}", flush=True)
        print("="*60, flush=True)

        return all_results

    def table_neuron_utilization(self, health_results: Dict, output_dir: str):
        """
        Generate Table 1: Neuron Utilization Statistics.

        Args:
            health_results: Results from NeuronHealthAnalyzer
            output_dir: Directory for output
        """
        ema_dist = health_results.get('ema_distribution', {})

        # Create CSV
        csv_path = os.path.join(output_dir, 'table1_neuron_utilization.csv')
        with open(csv_path, 'w') as f:
            f.write("Pool,Total,Active,Dead,Active%,Gini,Mean EMA,Std EMA\n")
            for name, data in ema_dist.items():
                if isinstance(data, dict) and 'display' in data:
                    stats = data.get('stats', {})
                    f.write(f"{data['display']},{data['total']},{data['active']},"
                           f"{data['dead']},{data['active_ratio']*100:.1f}%,"
                           f"{data['gini']:.3f},{stats.get('mean', 0):.4f},"
                           f"{stats.get('std', 0):.4f}\n")

        print(f"  Table 1 saved: {csv_path}")

        # Create LaTeX table
        latex_path = os.path.join(output_dir, 'table1_neuron_utilization.tex')
        with open(latex_path, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Neuron Utilization Statistics}\n")
            f.write("\\begin{tabular}{lrrrrrrr}\n")
            f.write("\\toprule\n")
            f.write("Pool & Total & Active & Dead & Active\\% & Gini & Mean EMA & Std EMA \\\\\n")
            f.write("\\midrule\n")
            for name, data in ema_dist.items():
                if isinstance(data, dict) and 'display' in data:
                    stats = data.get('stats', {})
                    f.write(f"{data['display']} & {data['total']} & {data['active']} & "
                           f"{data['dead']} & {data['active_ratio']*100:.1f}\\% & "
                           f"{data['gini']:.3f} & {stats.get('mean', 0):.4f} & "
                           f"{stats.get('std', 0):.4f} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:neuron-utilization}\n")
            f.write("\\end{table}\n")

        print(f"  LaTeX table saved: {latex_path}")

    def figure_summary(self, results: Dict, output_dir: str):
        """
        Generate summary figure combining key metrics.

        Args:
            results: Combined results from all analyses
            output_dir: Directory for output
        """
        if not HAS_MATPLOTLIB:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Active Neuron Ratios
        ax = axes[0, 0]
        ema_dist = results.get('health', {}).get('ema_distribution', {})
        names = []
        ratios = []
        for name, data in ema_dist.items():
            if isinstance(data, dict) and 'display' in data:
                names.append(data['display'])
                ratios.append(data['active_ratio'] * 100)
        if names:
            ax.bar(names, ratios, color='steelblue', alpha=0.7)
            ax.set_ylabel('Active Neurons (%)')
            ax.set_title('Neuron Utilization by Pool')
            ax.set_ylim(0, 100)
            ax.tick_params(axis='x', rotation=45)

        # 2. Diversity Score
        ax = axes[0, 1]
        diversity = results.get('health', {}).get('diversity', {})
        div_names = []
        div_scores = []
        for name, data in diversity.items():
            if isinstance(data, dict) and 'normalized_entropy' in data:
                div_names.append(data['display'])
                div_scores.append(data['normalized_entropy'] * 100)
        if div_names:
            ax.bar(div_names, div_scores, color='green', alpha=0.7)
            ax.set_ylabel('Normalized Entropy (%)')
            ax.set_title('Neuron Diversity')
            ax.set_ylim(0, 100)
            ax.tick_params(axis='x', rotation=45)

        # 3. Semantic Path Similarity
        ax = axes[0, 2]
        semantic = results.get('semantic', {}).get('path_similarity', {})
        if 'similar_pairs' in semantic and 'different_pairs' in semantic:
            sim_cos = semantic['similar_pairs'].get('cosine_mean', 0)
            diff_cos = semantic['different_pairs'].get('cosine_mean', 0)
            ax.bar(['Similar', 'Different'], [sim_cos, diff_cos],
                  color=['green', 'red'], alpha=0.7)
            ax.set_ylabel('Cosine Similarity')
            ax.set_title('Semantic Path Similarity')
            ax.set_ylim(0, 1)

        # 4. Q/K Correlation
        ax = axes[1, 0]
        qk_usage = results.get('routing', {}).get('qk_usage', {})
        qk_names = []
        qk_corrs = []
        for name, data in qk_usage.items():
            if isinstance(data, dict) and 'correlation' in data:
                qk_names.append(data['display'])
                qk_corrs.append(data['correlation'])
        if qk_names:
            ax.bar(qk_names, qk_corrs, color='purple', alpha=0.7)
            ax.set_ylabel('Q/K Correlation')
            ax.set_title('Q/K Usage Correlation')
            ax.set_ylim(-1, 1)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # 5. Context Variance
        ax = axes[1, 1]
        context = results.get('semantic', {}).get('context_routing', {})
        if 'summary' in context:
            var = context['summary'].get('overall_context_variance', 0)
            ax.bar(['Context\nVariance'], [var], color='orange', alpha=0.7)
            ax.set_ylabel('Routing Variance')
            ax.set_title('Context-Dependent Routing')

        # 6. Probing Accuracy
        ax = axes[1, 2]
        probing = results.get('behavioral', {}).get('probing', {})
        probe_names = []
        probe_accs = []
        for name, data in probing.items():
            if isinstance(data, dict) and 'accuracy' in data:
                probe_names.append(name[:5])
                probe_accs.append(data['accuracy'] * 100)
        if probe_names:
            ax.bar(probe_names, probe_accs, color='cyan', alpha=0.7)
            ax.set_ylabel('POS Prediction Accuracy (%)')
            ax.set_title('Probing Classifier')
            ax.set_ylim(0, 100)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        path = os.path.join(output_dir, 'summary_figure.png')
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"  Summary figure saved: {path}")

    def run_quick(self, output_dir: str = './quick_analysis'):
        """
        Run quick analysis (no data-dependent analyses).

        Args:
            output_dir: Directory for outputs
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        print("\n--- Quick Analysis ---")

        # Health
        print("1. Neuron Health...")
        results['health'] = self.health.run_all(os.path.join(output_dir, 'health'))

        # Embedding
        print("2. Embedding...")
        results['embedding'] = self.embedding.run_all(os.path.join(output_dir, 'embedding'))

        # Weight
        print("3. Weight SVD...")
        results['weight'] = self.weight.run_all(os.path.join(output_dir, 'weight'))

        # Semantic (no dataloader needed for path similarity)
        print("4. Semantic Path Similarity...")
        results['semantic'] = self.semantic.run_all(None, os.path.join(output_dir, 'semantic'), max_batches=50)

        # Save
        save_results(results, os.path.join(output_dir, 'quick_results.json'))
        print(f"\nQuick analysis saved to: {output_dir}")

        return results
