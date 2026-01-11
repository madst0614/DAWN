"""
DAWN Visualizers
================
Visualization functions for DAWN analysis results.

Each module corresponds to a specific paper figure or analysis type.
"""

from .qk_specialization import plot_qk_specialization, plot_qk_usage
from .neuron_health import plot_dead_neurons, plot_usage_histogram, plot_qk_ema_overlap
from .embedding import plot_similarity_heatmap, plot_clustering, plot_embedding_space

__all__ = [
    # Q/K Specialization (Figure 3)
    'plot_qk_specialization',
    'plot_qk_usage',
    # Neuron Health (Figure 4)
    'plot_dead_neurons',
    'plot_usage_histogram',
    'plot_qk_ema_overlap',
    # Embedding Structure (Figure 5)
    'plot_similarity_heatmap',
    'plot_clustering',
    'plot_embedding_space',
]
