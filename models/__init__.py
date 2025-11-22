"""
DAWN: Dynamic Architecture With Neurons

Neural network models for DAWN.
Dynamic Neuron Transformer architecture with QK-based retrieval.
"""

from .model import (
    # Dynamic Neuron Transformer components
    NeuronPool,
    PatternFFN,
    NeuronAttention,
    DynamicNeuronLayer,
    DynamicNeuronTransformer,
    # Main model
    DAWN,
    DAWNLanguageModel,
    DAWNTrainer,
    # Utilities
    create_model,
    # Backward compatibility aliases
    InputNeurons,
    ProcessNeurons,
)

__all__ = [
    # Dynamic Neuron Transformer
    "NeuronPool",
    "PatternFFN",
    "NeuronAttention",
    "DynamicNeuronLayer",
    "DynamicNeuronTransformer",
    # Main model & trainer
    "DAWN",
    "DAWNLanguageModel",
    "DAWNTrainer",
    # Utilities
    "create_model",
    # Backward compatibility
    "InputNeurons",
    "ProcessNeurons",
]
