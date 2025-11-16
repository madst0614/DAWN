"""
SPROUT: Self-organizing Progressive Routing with Organic Unified Trees

Two implementations:
1. Tree-based SPROUT (original): Dynamic tree structures with compatibility routing
2. Neuron Pool SPROUT (new): Unified neuron pool with hard routing
"""

# Tree-based SPROUT (original)
from .model import SPROUT
from .node import Node
from .router import Router
from .language_model import SproutLanguageModel

# Neuron Pool SPROUT (new)
from .neuron_pool import NeuronPool
from .hard_router import HardRouter
from .sprout_layer import SPROUTLayer
from .sprout_mlm import SPROUT_MLM

__version__ = "0.2.0"

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
]
