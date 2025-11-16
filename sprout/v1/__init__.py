"""
SPROUT V1 - Simple Neuron Pool

Reinterprets traditional Transformer FFN as a unified neuron pool.
Each layer uses a fixed subset of neurons (no dynamic routing).

This is mathematically identical to standard Transformer - just provides
a different view that enables analysis of neuron behavior.

Usage:
    from sprout.v1 import SimpleTransformerWithNeuronPool

    model = SimpleTransformerWithNeuronPool(
        vocab_size=30000,
        d_model=256,
        d_ff=1024,
        n_layers=6
    )
"""

from .simple_neuron_pool import SimpleNeuronPool, test_equivalence_to_ffn
from .simple_transformer import SimpleTransformerWithNeuronPool, test_simple_transformer
from .analyze import (
    plot_neuron_usage_by_layer,
    plot_neuron_weight_patterns,
    analyze_top_neurons,
    compare_layer_statistics,
    neuron_activation_heatmap,
    full_analysis_report,
)

__all__ = [
    # Core
    "SimpleNeuronPool",
    "SimpleTransformerWithNeuronPool",
    # Tests
    "test_equivalence_to_ffn",
    "test_simple_transformer",
    # Analysis
    "plot_neuron_usage_by_layer",
    "plot_neuron_weight_patterns",
    "analyze_top_neurons",
    "compare_layer_statistics",
    "neuron_activation_heatmap",
    "full_analysis_report",
]
