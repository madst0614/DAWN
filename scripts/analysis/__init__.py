"""
DAWN Analysis Package
======================
Comprehensive analysis toolkit for DAWN v17.1 models.

This package provides modular analysis tools for:
- Neuron health and utilization
- Routing patterns and entropy
- Embedding analysis and visualization
- Weight matrix analysis
- Behavioral analysis
- Semantic analysis (NEW)
- Co-selection pattern analysis
- Paper figure generation

Usage:
    # Quick analysis (no data required)
    from scripts.analysis import PaperFigureGenerator
    gen = PaperFigureGenerator('checkpoints/dawn_v17_1_best.pt')
    gen.run_quick('./quick_analysis')

    # Full analysis with data
    gen = PaperFigureGenerator(
        'checkpoints/dawn_v17_1_best.pt',
        'data/val.parquet'
    )
    gen.generate_all('./paper_figures')

    # Individual analyzers
    from scripts.analysis import load_model, get_router, NeuronHealthAnalyzer
    model, tokenizer, config = load_model('checkpoint.pt')
    router = get_router(model)
    health = NeuronHealthAnalyzer(router)
    results = health.run_all('./health_output')
"""

from .utils import (
    # Model loading
    load_model,
    get_router,
    get_neurons,
    get_shared_neurons,  # Alias
    create_dataloader,

    # Constants
    NEURON_TYPES,
    ROUTING_KEYS,
    KNOWLEDGE_ROUTING_KEYS,
    ALL_ROUTING_KEYS,
    NEURON_ATTRS,
    COSELECTION_PAIRS,
    QK_POOLS,

    # Utilities
    gini_coefficient,
    calc_entropy,
    calc_entropy_ratio,
    convert_to_serializable,
    save_results,
    simple_pos_tag,

    # Flags
    HAS_MATPLOTLIB,
    HAS_SKLEARN,
    HAS_TQDM,
)

from .neuron_health import NeuronHealthAnalyzer
from .routing import RoutingAnalyzer
from .embedding import EmbeddingAnalyzer
from .weight import WeightAnalyzer
from .behavioral import BehavioralAnalyzer
from .semantic import SemanticAnalyzer
from .coselection import CoselectionAnalyzer
from .paper_figures import PaperFigureGenerator
from .routing_analysis import (
    GenerationRoutingAnalyzer,
    analyze_common_neurons,
    analyze_token_neurons,
    plot_routing_heatmap,
    plot_routing_comparison,
)
from .pos_neuron_analysis import (
    POSNeuronAnalyzer,
    plot_pos_heatmap,
    plot_pos_clustering,
    plot_top_neurons_by_pos,
    plot_specificity,
)


__all__ = [
    # Model loading
    'load_model',
    'get_router',
    'get_neurons',
    'get_shared_neurons',
    'create_dataloader',

    # Constants
    'NEURON_TYPES',
    'ROUTING_KEYS',
    'KNOWLEDGE_ROUTING_KEYS',
    'ALL_ROUTING_KEYS',
    'NEURON_ATTRS',
    'COSELECTION_PAIRS',
    'QK_POOLS',

    # Utilities
    'gini_coefficient',
    'calc_entropy',
    'calc_entropy_ratio',
    'convert_to_serializable',
    'save_results',
    'simple_pos_tag',

    # Flags
    'HAS_MATPLOTLIB',
    'HAS_SKLEARN',
    'HAS_TQDM',

    # Analyzers
    'NeuronHealthAnalyzer',
    'RoutingAnalyzer',
    'EmbeddingAnalyzer',
    'WeightAnalyzer',
    'BehavioralAnalyzer',
    'SemanticAnalyzer',
    'CoselectionAnalyzer',
    'PaperFigureGenerator',

    # Routing analysis
    'GenerationRoutingAnalyzer',
    'analyze_common_neurons',
    'analyze_token_neurons',
    'plot_routing_heatmap',
    'plot_routing_comparison',

    # POS neuron analysis
    'POSNeuronAnalyzer',
    'plot_pos_heatmap',
    'plot_pos_clustering',
    'plot_top_neurons_by_pos',
    'plot_specificity',
]

__version__ = '1.0.0'
