"""
DAWN: Dynamic Architecture With Neurons

Neural network models for DAWN.
New simplified architecture with InputNeurons and ProcessNeurons.
"""

from .model import (
    # Core components
    BalancedInputNeurons,
    InputNeurons,  # Alias for backward compatibility
    LateralConnections,
    LowRankProcessNeurons,
    ProcessNeurons,  # Alias for backward compatibility
    DAWNLayer,
    DAWN,
    DAWNLanguageModel,
    DAWNTrainer,
    # Utilities
    create_model,
)

__all__ = [
    "BalancedInputNeurons",
    "InputNeurons",
    "LateralConnections",
    "LowRankProcessNeurons",
    "ProcessNeurons",
    "DAWNLayer",
    "DAWN",
    "DAWNLanguageModel",
    "DAWNTrainer",
    "create_model",
]
