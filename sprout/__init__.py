"""
SPROUT: Self-organizing Progressive Routing with Organic Unified Trees

Three implementations:
1. Tree-based SPROUT (original): Dynamic tree structures with compatibility routing
2. Neuron Pool SPROUT: Unified neuron pool with hard routing
3. Contextual SPROUT (new): Firing pattern-based with context combination
"""

# Tree-based SPROUT (original)
from .model import SPROUT
from .node import Node
from .router import Router
from .language_model import SproutLanguageModel

# Neuron Pool SPROUT
from .neuron_pool import NeuronPool
from .hard_router import HardRouter
from .sprout_layer import SPROUTLayer
from .sprout_mlm import SPROUT_MLM

# Contextual SPROUT (firing patterns)
from .learned_neuron_pool import LearnedNeuronPool, ContextualNeuronPool
from .context_router import ContextRouter, AdaptiveContextRouter
from .contextual_sprout_layer import ContextualSPROUTLayer, AdvancedContextualSPROUTLayer
from .contextual_sprout_mlm import ContextualSPROUT_MLM

__version__ = "0.3.0"

__all__ = [
    # Tree-based
    "SPROUT",
    "Node",
    "Router",
    "SproutLanguageModel",
    # Neuron Pool
    "NeuronPool",
    "HardRouter",
    "SPROUTLayer",
    "SPROUT_MLM",
    # Contextual (Firing Patterns)
    "LearnedNeuronPool",
    "ContextualNeuronPool",
    "ContextRouter",
    "AdaptiveContextRouter",
    "ContextualSPROUTLayer",
    "AdvancedContextualSPROUTLayer",
    "ContextualSPROUT_MLM",
]
