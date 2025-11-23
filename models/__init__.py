"""
DAWN: Dynamic Architecture With Neurons

Neural network models for DAWN.
v4.5: Pattern-specific up projection + Cross-neuron gating
"""

from .model import (
    # Core components
    NeuronRouter,
    InteractionFFN,
    Layer,
    # Main model
    DAWN,
    DAWNLanguageModel,
    # Utilities
    create_model,
)

__all__ = [
    # Core components
    "NeuronRouter",
    "InteractionFFN",
    "Layer",
    # Main model
    "DAWN",
    "DAWNLanguageModel",
    # Utilities
    "create_model",
]
