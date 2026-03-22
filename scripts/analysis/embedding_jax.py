"""
Embedding Analysis - JAX Version
=================================
Analyze neuron embeddings in DAWN models.

JAX/Flax compatible version for TPU analysis.

Includes:
- Intra-type similarity analysis
- Cross-type similarity analysis
- Clustering analysis
- Visualization (requires matplotlib)

NOTE: This is a JAX port of embedding.py (271 lines).
"""

import os
import numpy as np
from typing import Dict, Optional

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.decomposition import PCA
    HAS_PCA = True
except ImportError:
    HAS_PCA = False

from .base_jax import BaseAnalyzerJAX
from .utils_jax import (
    EMBEDDING_POOLS_V18,
    get_neuron_embeddings_jax,
)


class EmbeddingAnalyzerJAX(BaseAnalyzerJAX):
    """Neuron embedding analyzer (JAX version)."""

    def __init__(self, model, params, config: Dict):
        """
        Initialize analyzer.

        Args:
            model: DAWN JAX model class
            params: FrozenDict of model parameters
            config: Model configuration dict
        """
        super().__init__(model, params, config)

    def get_embeddings_by_type(self) -> Dict[str, np.ndarray]:
        """
        Extract embeddings grouped by embedding pool.

        Returns:
            Dictionary mapping pool name to embedding array
        """
        emb = get_neuron_embeddings_jax(self.params)
        if emb is None:
            return {}

        # Use embedding pools (6 unique pools)
        pools = self.get_embedding_pools()
        result = {}
        offset = 0

        # Pool sizes from config
        pool_sizes = {
            'feature_qk': self.n_feature_qk,
            'feature_v': self.n_feature_v,
            'restore_qk': self.n_restore_qk,
            'restore_v': self.n_restore_v,
            'feature_know': self.n_feature_know,
            'restore_know': self.n_restore_know,
        }

        for name, (display, n_attr, _) in pools.items():
            n = pool_sizes.get(name, 0)
            if n > 0 and offset + n <= len(emb):
                result[name] = emb[offset:offset + n]
                offset += n

        return result

    def analyze_similarity(self, output_dir: Optional[str] = None) -> Dict:
        """
        Analyze intra-type similarity using cosine similarity.

        Args:
            output_dir: Directory for visualization output

        Returns:
            Dictionary with similarity statistics
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        embeddings = self.get_embeddings_by_type()
        if not embeddings:
            return {'error': 'No embeddings found'}

        results = {}
        pools = self.get_embedding_pools()

        for name, emb in embeddings.items():
            if len(emb) < 2:
                continue

            # Normalize embeddings
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
            emb_norm = emb / norms

            # Compute similarity matrix
            sim_matrix = np.dot(emb_norm, emb_norm.T)

            # Extract off-diagonal elements
            n = sim_matrix.shape[0]
            mask = ~np.eye(n, dtype=bool)
            off_diag = sim_matrix[mask]

            display = pools[name][0] if name in pools else name.upper()
            results[name] = {
                'display': display,
                'n_neurons': n,
                'avg_similarity': float(off_diag.mean()),
                'max_similarity': float(off_diag.max()),
                'min_similarity': float(off_diag.min()),
                'std_similarity': float(off_diag.std()),
            }

        # Visualization
        if output_dir and HAS_MATPLOTLIB:
            os.makedirs(output_dir, exist_ok=True)
            try:
                pool_names = list(results.keys())
                n = len(pool_names)
                if n > 0:
                    # Cross-pool similarity matrix
                    sim_matrix = np.zeros((n, n))
                    for i, p1 in enumerate(pool_names):
                        for j, p2 in enumerate(pool_names):
                            if i == j:
                                sim_matrix[i, j] = results[p1]['avg_similarity']
                            else:
                                e1, e2 = embeddings[p1], embeddings[p2]
                                n1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-8)
                                n2 = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-8)
                                c1, c2 = n1.mean(axis=0), n2.mean(axis=0)
                                sim_matrix[i, j] = float(np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8))

                    display_names = [results[p]['display'] for p in pool_names]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
                    ax.set_xticks(range(n))
                    ax.set_yticks(range(n))
                    ax.set_xticklabels(display_names, rotation=45, ha='right')
                    ax.set_yticklabels(display_names)
                    plt.colorbar(im, label='Cosine Similarity')
                    ax.set_title('Neuron Embedding Similarity')
                    plt.tight_layout()
                    path = os.path.join(output_dir, 'similarity_heatmap.png')
                    plt.savefig(path, dpi=150)
                    plt.close()
                    results['visualization'] = path
            except Exception as e:
                print(f"  Visualization error: {e}")

        return results

    def analyze_cross_type_similarity(self) -> Dict:
        """
        Analyze similarity between neuron types using centroids.

        Returns:
            Dictionary with pairwise centroid similarities
        """
        if not HAS_JAX:
            return {'error': 'JAX not available'}

        embeddings = self.get_embeddings_by_type()
        if not embeddings:
            return {'error': 'No embeddings found'}

        pools = self.get_embedding_pools()

        # Compute centroids
        centroids = {}
        for name, emb in embeddings.items():
            centroids[name] = emb.mean(axis=0)

        # Compute pairwise similarities
        results = {}
        names = list(centroids.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                c1, c2 = centroids[n1], centroids[n2]
                # Cosine similarity
                sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
                d1 = pools[n1][0] if n1 in pools else n1.upper()
                d2 = pools[n2][0] if n2 in pools else n2.upper()
                key = f"{d1}-{d2}"
                results[key] = float(sim)

        return results

    def analyze_clustering(self, n_clusters: int = 5, output_dir: Optional[str] = None) -> Dict:
        """
        Perform clustering analysis on neuron embeddings.

        Args:
            n_clusters: Number of clusters
            output_dir: Directory for visualization output

        Returns:
            Dictionary with clustering results
        """
        if not HAS_SKLEARN:
            return {'error': 'sklearn not available'}

        if not HAS_JAX:
            return {'error': 'JAX not available'}

        emb_full = get_neuron_embeddings_jax(self.params)
        if emb_full is None:
            return {'error': 'No embeddings found'}

        pools = self.get_embedding_pools()

        # Pool sizes from config
        pool_sizes = {
            'feature_qk': self.n_feature_qk,
            'feature_v': self.n_feature_v,
            'restore_qk': self.n_restore_qk,
            'restore_v': self.n_restore_v,
            'feature_know': self.n_feature_know,
            'restore_know': self.n_restore_know,
        }

        # Build boundaries for each pool
        boundaries = {}
        offset = 0
        for name, (display, n_attr, _) in pools.items():
            n = pool_sizes.get(name, 0)
            if n > 0:
                boundaries[name] = (offset, offset + n, display)
                offset += n

        results = {}

        # Cluster each pool
        for name, (start, end, display) in boundaries.items():
            if end > len(emb_full):
                continue

            pool_emb = emb_full[start:end]
            n_neurons = pool_emb.shape[0]

            if n_neurons < n_clusters:
                results[name] = {'error': f'Not enough neurons for {n_clusters} clusters'}
                continue

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pool_emb)

            cluster_stats = []
            for c in range(n_clusters):
                cluster_mask = labels == c
                cluster_size = int(cluster_mask.sum())
                cluster_stats.append({
                    'cluster_id': c,
                    'size': cluster_size,
                })

            results[name] = {
                'display': display,
                'n_clusters': n_clusters,
                'clusters': sorted(cluster_stats, key=lambda x: -x['size']),
                'labels': labels.tolist(),
            }

        # Visualization
        if output_dir and HAS_MATPLOTLIB and HAS_PCA:
            os.makedirs(output_dir, exist_ok=True)
            try:
                # PCA projection of each pool's clusters
                n_pools = len(results)
                if n_pools > 0:
                    cols = min(3, n_pools)
                    rows = (n_pools + cols - 1) // cols
                    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                    if rows == 1 and cols == 1:
                        axes = np.array([axes])
                    axes = axes.flatten()

                    ax_idx = 0
                    for name, (start, end, display) in boundaries.items():
                        if name not in results or 'labels' not in results[name]:
                            continue
                        if end > len(emb_full) or ax_idx >= len(axes):
                            continue

                        pool_emb = emb_full[start:end]
                        labels = np.array(results[name]['labels'])

                        pca = PCA(n_components=2)
                        coords = pca.fit_transform(pool_emb)

                        scatter = axes[ax_idx].scatter(
                            coords[:, 0], coords[:, 1],
                            c=labels, cmap='tab10', s=5, alpha=0.6
                        )
                        axes[ax_idx].set_title(f'{display} ({n_clusters} clusters)')
                        axes[ax_idx].set_xlabel('PC1')
                        axes[ax_idx].set_ylabel('PC2')
                        ax_idx += 1

                    for i in range(ax_idx, len(axes)):
                        axes[i].axis('off')

                    plt.tight_layout()
                    path = os.path.join(output_dir, 'clustering.png')
                    plt.savefig(path, dpi=150)
                    plt.close()
                    results['visualization'] = path
            except Exception as e:
                print(f"  Clustering visualization error: {e}")

        return results

    def visualize(self, output_dir: str) -> Optional[str]:
        """
        Generate PCA visualization of all embeddings colored by pool.

        Args:
            output_dir: Directory for output

        Returns:
            Path to visualization or None
        """
        if not HAS_MATPLOTLIB or not HAS_PCA:
            return None

        emb = get_neuron_embeddings_jax(self.params)
        if emb is None:
            return None

        os.makedirs(output_dir, exist_ok=True)

        pools = self.get_embedding_pools()
        pool_sizes = {
            'feature_qk': self.n_feature_qk,
            'feature_v': self.n_feature_v,
            'restore_qk': self.n_restore_qk,
            'restore_v': self.n_restore_v,
            'feature_know': self.n_feature_know,
            'restore_know': self.n_restore_know,
        }

        # Build labels and colors
        labels = []
        colors_list = []
        pool_colors = {
            'feature_qk': '#e41a1c', 'feature_v': '#ff7f00',
            'restore_qk': '#377eb8', 'restore_v': '#4daf4a',
            'feature_know': '#984ea3', 'restore_know': '#00bfc4',
        }

        offset = 0
        for name, (display, _, _) in pools.items():
            n = pool_sizes.get(name, 0)
            if n > 0 and offset + n <= len(emb):
                labels.extend([display] * n)
                colors_list.extend([pool_colors.get(name, '#999999')] * n)
                offset += n

        if len(labels) != len(emb[:offset]):
            return None

        try:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(emb[:offset])

            fig, ax = plt.subplots(figsize=(10, 8))

            unique_labels = list(dict.fromkeys(labels))
            for label in unique_labels:
                mask = np.array([l == label for l in labels])
                color = colors_list[labels.index(label)]
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=color, s=5, alpha=0.5, label=label)

            ax.legend(markerscale=3, fontsize=9)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax.set_title('DAWN Neuron Embeddings (PCA)')
            plt.tight_layout()

            path = os.path.join(output_dir, 'dawn_embeddings.png')
            plt.savefig(path, dpi=150)
            plt.close()
            return path
        except Exception as e:
            print(f"  Embedding visualization error: {e}")
            return None

    def run_all(self, output_dir: str = './embedding_analysis', n_clusters: int = 5) -> Dict:
        """
        Run all embedding analyses.

        Args:
            output_dir: Directory for outputs
            n_clusters: Number of clusters for clustering analysis

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'similarity': self.analyze_similarity(output_dir),
            'cross_type_similarity': self.analyze_cross_type_similarity(),
            'clustering': self.analyze_clustering(n_clusters, output_dir),
        }

        # Main visualization
        viz_path = self.visualize(output_dir)
        if viz_path:
            results['visualization'] = viz_path

        return results


class NeuronEmbeddingAnalyzerJAX(BaseAnalyzerJAX):
    """Extended neuron embedding analyzer (JAX version)."""

    def __init__(self, model, params, config: Dict):
        super().__init__(model, params, config)
        self.tokenizer = None

    def run_full_analysis(
        self,
        val_tokens: np.ndarray,
        output_dir: str = './neuron_embedding',
        n_batches: int = 50,
        k_range: tuple = (5, 20),
        tokenizer=None
    ) -> Dict:
        """
        Run full neuron embedding analysis.

        Args:
            val_tokens: Validation tokens
            output_dir: Directory for outputs
            n_batches: Number of batches
            k_range: Range of k for optimal k analysis
            tokenizer: Tokenizer instance

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        # Run basic embedding analysis
        basic_analyzer = EmbeddingAnalyzerJAX(self.model, self.params, self.config)

        return {
            'pool_distribution': basic_analyzer.analyze_similarity(output_dir),
            'clustering': basic_analyzer.analyze_clustering(output_dir=output_dir),
            'cross_type_similarity': basic_analyzer.analyze_cross_type_similarity(),
        }
