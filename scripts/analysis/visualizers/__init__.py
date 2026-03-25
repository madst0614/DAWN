"""
DAWN Visualizers
================
Visualization functions for DAWN analysis results.

Each module corresponds to a specific paper figure or analysis type.

NOTE: Modules that transitively depend on torch (e.g. pos_neurons → utils)
are lazy-imported so this package works in JAX-only (TPU) environments.
"""

from .style import PAPER_STYLE, apply_paper_style
from .qk_specialization import plot_qk_specialization, plot_qk_pool, plot_qk_usage
from .neuron_health import (
    plot_dead_neurons, plot_usage_histogram,
    plot_activation_histogram, plot_diversity_summary
)
from .embedding import plot_similarity_heatmap, plot_clustering, plot_embedding_space
from .layer_contribution import plot_routing_stats, plot_layer_contribution
from .training_dynamics import (
    plot_training_dynamics, plot_training_from_logs,
    plot_training_from_checkpoints, find_training_log, parse_training_log
)

# Lazy imports for modules that depend on torch (via utils.py)
# These will fail gracefully in JAX-only environments
def __getattr__(name):
    _pos_neurons_names = {
        'plot_pos_heatmap', 'plot_pos_clustering',
        'plot_top_neurons_by_pos', 'plot_pos_specificity',
        'plot_pos_specialization_from_features',
        'plot_pos_selectivity_from_json',
        'plot_pos_selectivity_heatmap',
    }
    _factual_names = {
        'plot_factual_heatmap', 'plot_factual_comparison',
    }

    if name in _pos_neurons_names:
        from .pos_neurons import (
            plot_pos_heatmap, plot_pos_clustering,
            plot_top_neurons_by_pos, plot_pos_specificity,
            plot_pos_specialization_from_features,
            plot_pos_selectivity_from_json,
            plot_pos_selectivity_heatmap,
        )
        return locals()[name]
    elif name in _factual_names:
        from .factual_heatmap import plot_factual_heatmap, plot_factual_comparison
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Q/K Specialization (Figure 3)
    'plot_qk_specialization',
    'plot_qk_pool',
    'plot_qk_usage',
    # POS Neurons (Figure 4)
    'plot_pos_heatmap',
    'plot_pos_clustering',
    'plot_top_neurons_by_pos',
    'plot_pos_specificity',
    'plot_pos_specialization_from_features',
    'plot_pos_selectivity_from_json',
    'plot_pos_selectivity_heatmap',
    # Neuron Health (Figure 6a)
    'plot_dead_neurons',
    'plot_usage_histogram',
    'plot_activation_histogram',
    'plot_diversity_summary',
    # Routing Stats (Figure 7)
    'plot_routing_stats',
    'plot_layer_contribution',
    # Factual Knowledge (Figure 7)
    'plot_factual_heatmap',
    'plot_factual_comparison',
    # Embedding Structure
    'plot_similarity_heatmap',
    'plot_clustering',
    'plot_embedding_space',
    # Training Dynamics (Figure 6)
    'plot_training_dynamics',
    'plot_training_from_logs',
    'plot_training_from_checkpoints',
    'find_training_log',
    'parse_training_log',
]
