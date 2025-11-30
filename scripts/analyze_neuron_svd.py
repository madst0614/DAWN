"""
DAWN v10.0 Neuron SVD/PCA Analysis
===================================

학습된 뉴런의 저차원 표현 가능성 분석

분석 항목:
1. CompressNeurons의 주성분 분석 - 뉴런 간 유사성
2. ExpandNeurons의 주성분 분석
3. 저차원으로 얼마나 설명 가능한지 (variance explained)
4. 뉴런 클러스터링 가능성
5. Effective dimensionality

Usage:
    python analyze_neuron_svd.py --checkpoint <path>
    python analyze_neuron_svd.py --checkpoint <path> --output_dir ./svd_analysis
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            IN_NOTEBOOK = True
            get_ipython().run_line_magic('matplotlib', 'inline')
        else:
            IN_NOTEBOOK = False
            matplotlib.use('Agg')
    except (ImportError, AttributeError):
        IN_NOTEBOOK = False
        matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    IN_NOTEBOOK = False
    print("Warning: matplotlib not available")

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, using torch SVD instead")


def get_underlying_model(model):
    """Get the underlying model from torch.compile wrapper"""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


class NeuronSVDAnalyzer:
    """뉴런 SVD/PCA 분석기"""

    def __init__(self, model, device='cuda'):
        self.model = get_underlying_model(model)
        self.device = device

        # Extract config
        self.n_compress = self.model.n_compress
        self.n_expand = self.model.n_expand
        self.n_knowledge = self.model.n_knowledge
        self.d_model = self.model.d_model
        self.rank = self.model.rank

        print(f"\n{'='*60}")
        print(f"Neuron SVD/PCA Analyzer")
        print(f"{'='*60}")
        print(f"n_compress: {self.n_compress}, n_expand: {self.n_expand}")
        print(f"d_model: {self.d_model}, rank: {self.rank}")
        print(f"n_knowledge: {self.n_knowledge}")

    def extract_neurons(self):
        """모델에서 뉴런 추출"""
        shared = self.model.shared_neurons

        neurons = {
            'compress': shared.compress_neurons.data.cpu(),  # [N, D, R]
            'expand': shared.expand_neurons.data.cpu(),      # [N, R, D]
            'knowledge_K': shared.knowledge_K.data.cpu(),    # [N, R]
            'knowledge_V': shared.knowledge_V.data.cpu(),    # [N, D]
        }

        print(f"\nExtracted neurons:")
        print(f"  compress_neurons: {neurons['compress'].shape}")
        print(f"  expand_neurons: {neurons['expand'].shape}")
        print(f"  knowledge_K: {neurons['knowledge_K'].shape}")
        print(f"  knowledge_V: {neurons['knowledge_V'].shape}")

        return neurons

    def analyze_pca(self, neurons_flat, name, k_values=[4, 8, 16, 32, 64]):
        """
        PCA 분석 수행

        Args:
            neurons_flat: [N, features] - flattened neurons
            name: 뉴런 이름
            k_values: 확인할 주성분 개수

        Returns:
            results dict
        """
        print(f"\n--- {name} PCA Analysis ---")
        print(f"Shape: {neurons_flat.shape}")

        N, F = neurons_flat.shape
        max_components = min(N, F)

        if HAS_SKLEARN:
            # sklearn PCA
            pca = PCA()
            pca.fit(neurons_flat)
            explained_ratio = pca.explained_variance_ratio_
            singular_values = pca.singular_values_
            components = pca.components_
        else:
            # Torch SVD fallback
            # Center the data
            mean = neurons_flat.mean(dim=0, keepdim=True)
            centered = neurons_flat - mean

            U, S, Vh = torch.linalg.svd(torch.tensor(centered), full_matrices=False)

            total_var = (S ** 2).sum()
            explained_ratio = ((S ** 2) / total_var).numpy()
            singular_values = S.numpy()
            components = Vh.numpy()

        cumsum = np.cumsum(explained_ratio)

        # Report
        print(f"\nVariance explained by K components:")
        results = {'k_explained': {}, 'cumsum': cumsum.tolist()[:100]}

        for k in k_values:
            if k <= len(cumsum):
                pct = cumsum[k-1] * 100
                results['k_explained'][k] = pct
                status = "excellent" if pct > 95 else ("good" if pct > 90 else ("moderate" if pct > 80 else "low"))
                print(f"  K={k:3d}: {pct:6.2f}% - {status}")

        # Effective dimensionality (entropy-based)
        # H = -sum(p * log(p)), effective_dim = exp(H)
        p = explained_ratio + 1e-10
        entropy = -np.sum(p * np.log(p))
        eff_dim = np.exp(entropy)
        results['effective_dim'] = float(eff_dim)
        results['max_dim'] = max_components
        print(f"\nEffective dimensionality: {eff_dim:.1f}/{max_components} ({eff_dim/max_components*100:.1f}%)")

        # Find K for 90%, 95%, 99% explained
        thresholds = [0.90, 0.95, 0.99]
        results['k_for_threshold'] = {}
        for thresh in thresholds:
            k_needed = np.searchsorted(cumsum, thresh) + 1
            results['k_for_threshold'][f"{int(thresh*100)}%"] = int(k_needed)
            print(f"  K for {int(thresh*100)}% variance: {k_needed}")

        # Singular value analysis
        results['singular_values'] = singular_values[:50].tolist()
        results['sv_ratio_1_to_10'] = float(singular_values[0] / (singular_values[9] + 1e-10)) if len(singular_values) > 9 else None

        return results, explained_ratio, singular_values, components

    def analyze_compress_neurons(self, neurons):
        """CompressNeurons 분석"""
        print(f"\n{'='*60}")
        print("COMPRESS NEURONS ANALYSIS")
        print(f"{'='*60}")

        compress = neurons['compress'].numpy()  # [N, D, R]
        N, D, R = compress.shape

        results = {}

        # 1. Flatten 전체 분석: 각 뉴런을 D*R 벡터로
        neurons_flat = compress.reshape(N, D * R)
        results['full'], full_ratio, full_sv, full_comp = self.analyze_pca(
            neurons_flat, "CompressNeurons (full D*R)", [4, 8, 16, 32, 64, 128]
        )

        # 2. Row-wise 분석: 각 뉴런의 각 행(d_model 차원)을 rank 벡터로
        # 총 N*D개의 rank 차원 벡터
        rows_flat = compress.reshape(N * D, R)
        results['rows'], rows_ratio, rows_sv, rows_comp = self.analyze_pca(
            rows_flat, "CompressNeurons (rows, D→R)", [4, 8, 16, 32]
        )

        # 3. Column-wise 분석: 각 뉴런의 각 열(rank 차원)을 d_model 벡터로
        # 총 N*R개의 d_model 차원 벡터
        cols_flat = compress.transpose(0, 2, 1).reshape(N * R, D)
        results['cols'], cols_ratio, cols_sv, cols_comp = self.analyze_pca(
            cols_flat, "CompressNeurons (cols, R→D)", [4, 8, 16, 32, 64]
        )

        # 4. 뉴런 간 유사도 분석
        neurons_norm = neurons_flat / (np.linalg.norm(neurons_flat, axis=1, keepdims=True) + 1e-10)
        sim_matrix = neurons_norm @ neurons_norm.T

        # 자기 자신 제외
        mask = ~np.eye(N, dtype=bool)
        off_diag = sim_matrix[mask]

        results['similarity'] = {
            'mean': float(np.mean(off_diag)),
            'std': float(np.std(off_diag)),
            'max': float(np.max(off_diag)),
            'min': float(np.min(off_diag)),
        }

        print(f"\nNeuron similarity (cosine):")
        print(f"  Mean: {results['similarity']['mean']:.4f}")
        print(f"  Std:  {results['similarity']['std']:.4f}")
        print(f"  Max:  {results['similarity']['max']:.4f}")

        return results, sim_matrix, full_ratio, full_sv

    def analyze_expand_neurons(self, neurons):
        """ExpandNeurons 분석"""
        print(f"\n{'='*60}")
        print("EXPAND NEURONS ANALYSIS")
        print(f"{'='*60}")

        expand = neurons['expand'].numpy()  # [N, R, D]
        N, R, D = expand.shape

        results = {}

        # 1. Flatten 전체 분석
        neurons_flat = expand.reshape(N, R * D)
        results['full'], full_ratio, full_sv, full_comp = self.analyze_pca(
            neurons_flat, "ExpandNeurons (full R*D)", [4, 8, 16, 32]
        )

        # 2. Row-wise 분석: R→D
        rows_flat = expand.reshape(N * R, D)
        results['rows'], rows_ratio, rows_sv, rows_comp = self.analyze_pca(
            rows_flat, "ExpandNeurons (rows, R→D)", [4, 8, 16, 32, 64]
        )

        # 3. Column-wise 분석: D→R
        cols_flat = expand.transpose(0, 2, 1).reshape(N * D, R)
        results['cols'], cols_ratio, cols_sv, cols_comp = self.analyze_pca(
            cols_flat, "ExpandNeurons (cols, D→R)", [4, 8, 16, 32]
        )

        # 4. 뉴런 간 유사도
        neurons_norm = neurons_flat / (np.linalg.norm(neurons_flat, axis=1, keepdims=True) + 1e-10)
        sim_matrix = neurons_norm @ neurons_norm.T
        mask = ~np.eye(N, dtype=bool)
        off_diag = sim_matrix[mask]

        results['similarity'] = {
            'mean': float(np.mean(off_diag)),
            'std': float(np.std(off_diag)),
            'max': float(np.max(off_diag)),
            'min': float(np.min(off_diag)),
        }

        print(f"\nNeuron similarity (cosine):")
        print(f"  Mean: {results['similarity']['mean']:.4f}")
        print(f"  Std:  {results['similarity']['std']:.4f}")

        return results, sim_matrix, full_ratio, full_sv

    def analyze_knowledge_neurons(self, neurons):
        """KnowledgeNeurons 분석"""
        print(f"\n{'='*60}")
        print("KNOWLEDGE NEURONS ANALYSIS")
        print(f"{'='*60}")

        K = neurons['knowledge_K'].numpy()  # [N, R]
        V = neurons['knowledge_V'].numpy()  # [N, D]

        results = {}

        # K 분석
        results['K'], K_ratio, K_sv, K_comp = self.analyze_pca(
            K, "Knowledge K (query)", [4, 8, 16, 32]
        )

        # V 분석
        results['V'], V_ratio, V_sv, V_comp = self.analyze_pca(
            V, "Knowledge V (value)", [4, 8, 16, 32, 64]
        )

        # K-V alignment
        # 각 knowledge neuron의 K와 V가 얼마나 aligned인지
        # K를 V의 공간으로 project해서 비교 (간접적)

        # K 간 유사도
        K_norm = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-10)
        K_sim = K_norm @ K_norm.T
        mask = ~np.eye(K.shape[0], dtype=bool)

        results['K_similarity'] = {
            'mean': float(np.mean(K_sim[mask])),
            'std': float(np.std(K_sim[mask])),
            'max': float(np.max(K_sim[mask])),
        }

        print(f"\nKnowledge K similarity:")
        print(f"  Mean: {results['K_similarity']['mean']:.4f}")

        return results, K_ratio, V_ratio

    def analyze_clustering(self, neurons):
        """뉴런 클러스터링 분석"""
        if not HAS_SKLEARN:
            print("\nsklearn not available, skipping clustering")
            return {}

        print(f"\n{'='*60}")
        print("CLUSTERING ANALYSIS")
        print(f"{'='*60}")

        results = {}

        # Compress neurons clustering
        compress = neurons['compress'].numpy()
        N, D, R = compress.shape
        compress_flat = compress.reshape(N, D * R)

        # Try different K values
        k_values = [2, 4, 8, 16]
        print("\nCompressNeurons clustering:")

        for k in k_values:
            if k >= N:
                continue
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(compress_flat)
            inertia = kmeans.inertia_

            # Cluster sizes
            sizes = np.bincount(labels)
            print(f"  K={k}: inertia={inertia:.2f}, sizes={sizes.tolist()}")

            results[f'compress_k{k}'] = {
                'inertia': float(inertia),
                'sizes': sizes.tolist(),
            }

        return results

    def run_all(self):
        """모든 분석 실행"""
        neurons = self.extract_neurons()

        all_results = {}

        # 1. Compress neurons
        compress_results, compress_sim, compress_ratio, compress_sv = self.analyze_compress_neurons(neurons)
        all_results['compress'] = compress_results

        # 2. Expand neurons
        expand_results, expand_sim, expand_ratio, expand_sv = self.analyze_expand_neurons(neurons)
        all_results['expand'] = expand_results

        # 3. Knowledge neurons
        knowledge_results, K_ratio, V_ratio = self.analyze_knowledge_neurons(neurons)
        all_results['knowledge'] = knowledge_results

        # 4. Clustering
        clustering_results = self.analyze_clustering(neurons)
        all_results['clustering'] = clustering_results

        # Store for visualization
        self._viz_data = {
            'compress_ratio': compress_ratio,
            'compress_sv': compress_sv,
            'compress_sim': compress_sim,
            'expand_ratio': expand_ratio,
            'expand_sv': expand_sv,
            'expand_sim': expand_sim,
            'K_ratio': K_ratio,
            'V_ratio': V_ratio,
        }

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        print("\n[CompressNeurons]")
        print(f"  Effective dim: {compress_results['full']['effective_dim']:.1f}/{self.n_compress}")
        print(f"  K for 90%: {compress_results['full']['k_for_threshold']['90%']}")
        print(f"  K for 95%: {compress_results['full']['k_for_threshold']['95%']}")
        print(f"  Similarity: {compress_results['similarity']['mean']:.4f}")

        print("\n[ExpandNeurons]")
        print(f"  Effective dim: {expand_results['full']['effective_dim']:.1f}/{self.n_expand}")
        print(f"  K for 90%: {expand_results['full']['k_for_threshold']['90%']}")
        print(f"  Similarity: {expand_results['similarity']['mean']:.4f}")

        print("\n[KnowledgeNeurons]")
        print(f"  K effective dim: {knowledge_results['K']['effective_dim']:.1f}")
        print(f"  V effective dim: {knowledge_results['V']['effective_dim']:.1f}")

        # Recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")

        # Compress
        k_90 = compress_results['full']['k_for_threshold']['90%']
        if k_90 < self.n_compress * 0.3:
            print(f"  - CompressNeurons: 90%를 {k_90}개로 설명 가능")
            print(f"    → n_compress를 {k_90}~{int(k_90*1.5)}로 줄여도 될 수 있음")
        else:
            print(f"  - CompressNeurons: 뉴런들이 잘 분산되어 있음")

        # Similarity
        if compress_results['similarity']['mean'] > 0.5:
            print(f"  - CompressNeurons 유사도가 높음 ({compress_results['similarity']['mean']:.3f})")
            print(f"    → diversity loss 강화 또는 n_compress 줄이기 권장")

        if expand_results['similarity']['mean'] > 0.5:
            print(f"  - ExpandNeurons 유사도가 높음 ({expand_results['similarity']['mean']:.3f})")
            print(f"    → n_expand 줄이기 권장")

        return all_results

    def visualize(self, output_path):
        """시각화"""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        if not hasattr(self, '_viz_data'):
            print("Run run_all() first")
            return

        data = self._viz_data
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))

        # 1. Compress - cumulative variance
        ax = axes[0, 0]
        cumsum = np.cumsum(data['compress_ratio'])
        ax.plot(cumsum[:min(100, len(cumsum))], 'b-', linewidth=2)
        ax.axhline(y=0.9, color='r', linestyle='--', label='90%')
        ax.axhline(y=0.95, color='g', linestyle='--', label='95%')
        ax.set_xlabel('Component')
        ax.set_ylabel('Cumulative Variance')
        ax.set_title('CompressNeurons - Cumulative Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Compress - individual variance
        ax = axes[0, 1]
        ax.bar(range(min(50, len(data['compress_ratio']))),
               data['compress_ratio'][:50], color='steelblue')
        ax.set_xlabel('Component')
        ax.set_ylabel('Variance Ratio')
        ax.set_title('CompressNeurons - Individual Variance')

        # 3. Compress - singular values
        ax = axes[0, 2]
        sv = data['compress_sv'][:50]
        ax.semilogy(sv, 'b-', linewidth=2)
        ax.set_xlabel('Component')
        ax.set_ylabel('Singular Value (log)')
        ax.set_title('CompressNeurons - Singular Values')
        ax.grid(True, alpha=0.3)

        # 4. Expand - cumulative variance
        ax = axes[1, 0]
        cumsum = np.cumsum(data['expand_ratio'])
        ax.plot(cumsum[:min(100, len(cumsum))], 'b-', linewidth=2)
        ax.axhline(y=0.9, color='r', linestyle='--', label='90%')
        ax.axhline(y=0.95, color='g', linestyle='--', label='95%')
        ax.set_xlabel('Component')
        ax.set_ylabel('Cumulative Variance')
        ax.set_title('ExpandNeurons - Cumulative Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Expand - individual variance
        ax = axes[1, 1]
        ax.bar(range(min(50, len(data['expand_ratio']))),
               data['expand_ratio'][:50], color='coral')
        ax.set_xlabel('Component')
        ax.set_ylabel('Variance Ratio')
        ax.set_title('ExpandNeurons - Individual Variance')

        # 6. Compress similarity matrix
        ax = axes[1, 2]
        im = ax.imshow(data['compress_sim'], cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title('CompressNeurons Similarity')
        plt.colorbar(im, ax=ax)

        # 7. Knowledge K - cumulative
        ax = axes[2, 0]
        cumsum = np.cumsum(data['K_ratio'])
        ax.plot(cumsum[:min(50, len(cumsum))], 'g-', linewidth=2, label='K')
        cumsum_v = np.cumsum(data['V_ratio'])
        ax.plot(cumsum_v[:min(50, len(cumsum_v))], 'b-', linewidth=2, label='V')
        ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Component')
        ax.set_ylabel('Cumulative Variance')
        ax.set_title('Knowledge K & V - Cumulative')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 8. Expand similarity matrix
        ax = axes[2, 1]
        im = ax.imshow(data['expand_sim'], cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title('ExpandNeurons Similarity')
        plt.colorbar(im, ax=ax)

        # 9. Summary text
        ax = axes[2, 2]
        ax.axis('off')

        compress_k90 = np.searchsorted(np.cumsum(data['compress_ratio']), 0.9) + 1
        expand_k90 = np.searchsorted(np.cumsum(data['expand_ratio']), 0.9) + 1

        text = f"""
SUMMARY

CompressNeurons ({self.n_compress}):
  - K for 90%: {compress_k90}
  - Mean similarity: {np.mean(data['compress_sim'][~np.eye(self.n_compress, dtype=bool)]):.3f}

ExpandNeurons ({self.n_expand}):
  - K for 90%: {expand_k90}
  - Mean similarity: {np.mean(data['expand_sim'][~np.eye(self.n_expand, dtype=bool)]):.3f}

Knowledge ({self.n_knowledge}):
  - K k90: {np.searchsorted(np.cumsum(data['K_ratio']), 0.9) + 1}
  - V k90: {np.searchsorted(np.cumsum(data['V_ratio']), 0.9) + 1}
"""
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close()

        print(f"\nSaved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='DAWN Neuron SVD/PCA Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='./svd_analysis',
                        help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        best = checkpoint_path / 'best_model.pt'
        if best.exists():
            checkpoint_path = best
        else:
            pt_files = list(checkpoint_path.glob('*.pt'))
            if pt_files:
                checkpoint_path = max(pt_files, key=lambda p: p.stat().st_mtime)

    print(f"\nLoading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('model_config', checkpoint.get('config', {}))
    version = config.get('model_version', '10.0')

    print(f"Model version: {version}")

    # Load model
    if version.startswith('10'):
        from models.model_v10 import DAWN
    else:
        print(f"Unsupported version: {version}")
        return

    model = DAWN(
        vocab_size=config.get('vocab_size', 30522),
        d_model=config.get('d_model', 320),
        n_layers=config.get('n_layers', 4),
        n_heads=config.get('n_heads', 4),
        rank=config.get('rank', 64),
        max_seq_len=config.get('max_seq_len', 128),
        n_compress=config.get('n_compress', 224),
        n_expand=config.get('n_expand', 56),
        n_knowledge=config.get('n_knowledge', 80),
        knowledge_k=config.get('knowledge_k', 10),
        dropout=config.get('dropout', 0.1),
    )

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Analyze
    analyzer = NeuronSVDAnalyzer(model, device)
    results = analyzer.run_all()

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, 'svd_analysis.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Visualize
    if HAS_MATPLOTLIB:
        viz_path = os.path.join(args.output_dir, 'neuron_pca_analysis.png')
        analyzer.visualize(viz_path)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
