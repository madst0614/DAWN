"""
Embedding Analysis
==================
Analyze neuron embeddings in DAWN v17.1 models.

Includes:
- Intra-type similarity analysis
- Cross-type similarity analysis
- Clustering analysis
- t-SNE/PCA visualization
"""

import os
import numpy as np
import torch
from typing import Dict, Optional

from .utils import (
    NEURON_TYPES, NEURON_TYPES_V18, EMBEDDING_POOLS_V18,
    HAS_MATPLOTLIB, HAS_SKLEARN, plt, sns
)

if HAS_SKLEARN:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans


class EmbeddingAnalyzer:
    """Neuron embedding analyzer."""

    def __init__(self, router, device='cuda'):
        """
        Initialize analyzer.

        Args:
            router: NeuronRouter instance
            device: Device for computation
        """
        self.router = router
        self.device = device
        # v18.x detection: Q/K separated EMA
        self.is_v18 = hasattr(router, 'usage_ema_feature_q')

    def _get_neuron_types(self):
        """Get appropriate NEURON_TYPES for model version (for EMA tracking)."""
        return NEURON_TYPES_V18 if self.is_v18 else NEURON_TYPES

    def _get_embedding_pools(self):
        """Get embedding pool boundaries (6 unique pools, Q/K share same pool)."""
        if self.is_v18:
            return EMBEDDING_POOLS_V18
        # For non-v18, use NEURON_TYPES (no Q/K separation)
        return {k: (v[0], v[2], v[3]) for k, v in NEURON_TYPES.items()}

    def get_embeddings_by_type(self, as_tensor: bool = False) -> Dict[str, np.ndarray]:
        """
        Extract embeddings grouped by embedding pool (not EMA type).

        Args:
            as_tensor: Return torch tensors on GPU instead of numpy arrays

        Returns:
            Dictionary mapping pool name to embedding array/tensor
        """
        emb = self.router.neuron_emb.detach()
        if not as_tensor:
            emb = emb.cpu().numpy()

        # Use embedding pools (6 unique pools) not neuron types (8 with Q/K separate)
        pools = self._get_embedding_pools()
        result = {}
        offset = 0
        for name, (display, n_attr, _) in pools.items():
            if hasattr(self.router, n_attr):
                n = getattr(self.router, n_attr)
                result[name] = emb[offset:offset + n]
                offset += n

        return result

    def analyze_similarity(self, output_dir: Optional[str] = None) -> Dict:
        """
        Analyze intra-type similarity using cosine similarity.
        Optimized with GPU tensor operations.

        Args:
            output_dir: Directory for visualization output

        Returns:
            Dictionary with similarity statistics
        """
        # Get embeddings as GPU tensors
        embeddings_gpu = self.get_embeddings_by_type(as_tensor=True)
        results = {}

        for name, emb in embeddings_gpu.items():
            if len(emb) < 2:
                continue

            # Ensure on GPU
            emb = emb.to(self.device)

            # Normalize and compute similarity matrix on GPU
            emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
            sim_matrix = torch.mm(emb_norm, emb_norm.t())

            # Extract off-diagonal elements
            n = sim_matrix.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool, device=self.device)
            off_diag = sim_matrix[mask]

            neuron_types = self._get_neuron_types()
            display = neuron_types[name][0]
            results[name] = {
                'display': display,
                'n_neurons': n,
                'avg_similarity': float(off_diag.mean().item()),
                'max_similarity': float(off_diag.max().item()),
                'min_similarity': float(off_diag.min().item()),
                'std_similarity': float(off_diag.std().item()),
            }

        # Visualization (needs numpy)
        if HAS_MATPLOTLIB and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            embeddings_np = self.get_embeddings_by_type(as_tensor=False)
            self._visualize_similarity(embeddings_np, output_dir)
            results['visualization'] = os.path.join(output_dir, 'similarity_heatmap.png')

        return results

    def _visualize_similarity(self, embeddings: Dict, output_dir: str):
        """Generate similarity heatmap visualization."""
        n_types = len(embeddings)
        fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4))
        if n_types == 1:
            axes = [axes]

        neuron_types = self._get_neuron_types()
        for ax, (name, emb) in zip(axes, embeddings.items()):
            emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
            sim_matrix = emb_norm @ emb_norm.T
            sns.heatmap(sim_matrix, ax=ax, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title(f'{neuron_types[name][0]} Similarity')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'similarity_heatmap.png'), dpi=150)
        plt.close()

    def analyze_cross_type_similarity(self) -> Dict:
        """
        Analyze similarity between neuron types using centroids.
        Optimized with GPU tensor operations.

        Returns:
            Dictionary with pairwise centroid similarities
        """
        embeddings = self.get_embeddings_by_type(as_tensor=True)

        # Compute centroids on GPU
        centroids = {}
        for name, emb in embeddings.items():
            emb = emb.to(self.device)
            centroids[name] = emb.mean(dim=0)

        # Compute pairwise similarities on GPU
        results = {}
        names = list(centroids.keys())
        pools = self._get_embedding_pools()
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                c1, c2 = centroids[n1], centroids[n2]
                sim = torch.dot(c1, c2) / (c1.norm() * c2.norm() + 1e-8)
                key = f"{pools[n1][0]}-{pools[n2][0]}"
                results[key] = float(sim.item())

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

        results = {}
        emb = self.router.neuron_emb.detach().cpu().numpy()

        # Use embedding pools (6 unique pools) for correct boundaries
        pools = self._get_embedding_pools()

        # Map pool names to EMA attributes (for v18, QK pools use combined Q+K)
        ema_mapping = {
            'feature_qk': ('usage_ema_feature_q', 'usage_ema_feature_k'),  # tuple for combined
            'feature_v': 'usage_ema_feature_v',
            'restore_qk': ('usage_ema_restore_q', 'usage_ema_restore_k'),
            'restore_v': 'usage_ema_restore_v',
            'feature_know': 'usage_ema_feature_know',
            'restore_know': 'usage_ema_restore_know',
        }

        # Build boundaries for each pool
        boundaries = {}
        offset = 0
        for name, (display, n_attr, _) in pools.items():
            if hasattr(self.router, n_attr):
                n = getattr(self.router, n_attr)
                ema_info = ema_mapping.get(name)
                boundaries[name] = (offset, offset + n, ema_info, display)
                offset += n

        # Cluster each pool
        for name, (start, end, ema_info, display) in boundaries.items():
            pool_emb = emb[start:end]
            n_neurons = pool_emb.shape[0]

            if n_neurons < n_clusters:
                results[name] = {'error': f'Not enough neurons for {n_clusters} clusters'}
                continue

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pool_emb)

            # Get EMA (for QK pools, use max of Q and K)
            ema = None
            if ema_info is not None:
                if isinstance(ema_info, tuple):
                    # QK pool: combine Q and K EMA
                    ema_q = getattr(self.router, ema_info[0], None)
                    ema_k = getattr(self.router, ema_info[1], None)
                    if ema_q is not None and ema_k is not None:
                        ema = torch.maximum(ema_q, ema_k).cpu().numpy()
                else:
                    ema_attr = getattr(self.router, ema_info, None)
                    if ema_attr is not None:
                        ema = ema_attr.cpu().numpy()

            cluster_stats = []
            for c in range(n_clusters):
                cluster_mask = labels == c
                cluster_size = cluster_mask.sum()

                if ema is not None:
                    cluster_ema = ema[cluster_mask]
                    cluster_stats.append({
                        'cluster_id': c,
                        'size': int(cluster_size),
                        'avg_usage': float(cluster_ema.mean()),
                        'max_usage': float(cluster_ema.max()),
                        'min_usage': float(cluster_ema.min()),
                        'active_count': int((cluster_ema > 0.01).sum()),
                    })
                else:
                    cluster_stats.append({
                        'cluster_id': c,
                        'size': int(cluster_size),
                    })

            results[name] = {
                'display': display,
                'n_clusters': n_clusters,
                'clusters': sorted(cluster_stats, key=lambda x: -x.get('avg_usage', 0)),
                'labels': labels.tolist(),
            }

        # Visualization
        if HAS_MATPLOTLIB and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._visualize_clustering(emb, boundaries, results, output_dir)
            results['visualization'] = os.path.join(output_dir, 'clustering.png')

        return results

    def _visualize_clustering(self, emb: np.ndarray, boundaries: Dict, results: Dict, output_dir: str):
        """Generate clustering visualization."""
        n_types = len([k for k in results if 'error' not in results.get(k, {})])
        if n_types == 0:
            return

        fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5))
        if n_types == 1:
            axes = [axes]

        ax_idx = 0
        for name, (start, end, _, display) in boundaries.items():
            if name not in results or 'error' in results[name]:
                continue

            pool_emb = emb[start:end]
            pca = PCA(n_components=2)
            emb_2d = pca.fit_transform(pool_emb)

            labels = results[name]['labels']
            axes[ax_idx].scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
            axes[ax_idx].set_title(f'{display} Clusters')
            ax_idx += 1

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clustering.png'), dpi=150)
        plt.close()

    def visualize(self, output_dir: str) -> Optional[str]:
        """
        Generate t-SNE/PCA visualization of all embeddings.

        Args:
            output_dir: Directory for output

        Returns:
            Path to visualization or None
        """
        if not HAS_MATPLOTLIB or not HAS_SKLEARN:
            return None

        os.makedirs(output_dir, exist_ok=True)

        emb = self.router.neuron_emb.detach().cpu().numpy()

        # Use embedding pools (6 unique pools) for correct boundaries
        pools = self._get_embedding_pools()
        # Build labels and color map
        labels = []
        colors_map = {}
        for name, (display, n_attr, color) in pools.items():
            if hasattr(self.router, n_attr):
                n = getattr(self.router, n_attr)
                labels.extend([display] * n)
                colors_map[display] = color

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb)-1))
        emb_tsne = tsne.fit_transform(emb)

        # PCA
        pca = PCA(n_components=2)
        emb_pca = pca.fit_transform(emb)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, data, title in [(axes[0], emb_tsne, 't-SNE'), (axes[1], emb_pca, 'PCA')]:
            for t in set(labels):
                mask = np.array([l == t for l in labels])
                ax.scatter(data[mask, 0], data[mask, 1],
                          c=colors_map.get(t, 'gray'), label=t, alpha=0.6, s=20)
            ax.set_title(f'DAWN Neuron Embeddings ({title})')
            ax.legend()

        plt.tight_layout()
        path = os.path.join(output_dir, 'dawn_embeddings.png')
        plt.savefig(path, dpi=150)
        plt.close()

        return path

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
