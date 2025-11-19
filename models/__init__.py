"""
DAWN: Dynamic Architecture With Neurons

Neural network models for DAWN.
"""

from .model import (
    # New class names
    DynamicRouter,
    InputNeurons,
    ProcessNeurons,
    DAWNBlock,
    DAWNLayer,
    DAWNLanguageModel,
    # Backward compatibility aliases
    GlobalRouter,
    HierarchicalDynamicFFN,
    TransformerLayerWithHierarchicalFFN,
    HierarchicalLanguageModel,
)

__all__ = [
    # New class names
    "DynamicRouter",
    "InputNeurons",
    "ProcessNeurons",
    "DAWNBlock",
    "DAWNLayer",
    "DAWNLanguageModel",
    # Backward compatibility aliases
    "GlobalRouter",
    "HierarchicalDynamicFFN",
    "TransformerLayerWithHierarchicalFFN",
    "HierarchicalLanguageModel",
]
