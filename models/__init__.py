"""
DAWN: Dynamic Architecture With Neurons

Neural network models for DAWN.
Context-based Neuron Router with Pattern FFN.
"""

from .model import (
    # Core components
    NeuronRouter,
    PatternFFN,
    Layer,
    # Main model
    DAWN,
    DAWNLanguageModel,
    DAWNTrainer,
    # Utilities
    create_model,
    # Backward compatibility aliases
    DynamicNeuronTransformer,
    InputNeurons,
    ProcessNeurons,
)

__all__ = [
    # Core components
    "NeuronRouter",
    "PatternFFN",
    "Layer",
    # Main model & trainer
    "DAWN",
    "DAWNLanguageModel",
    "DAWNTrainer",
    # Utilities
    "create_model",
    # Backward compatibility
    "DynamicNeuronTransformer",
    "InputNeurons",
    "ProcessNeurons",
]
