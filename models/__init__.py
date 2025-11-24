"""
DAWN Models Module

v6.0: Orthogonal Basis FFN with Token-level Dynamic FFN
"""

from .model import (
    DAWN,
    DAWNLanguageModel,
    NeuronRouter,
    OrthogonalBasisFFN,
    DAWNLayer,
    Layer,  # Backward compatibility
    create_model,
    count_parameters,
)

__all__ = [
    'DAWN',
    'DAWNLanguageModel',
    'NeuronRouter',
    'OrthogonalBasisFFN',
    'DAWNLayer',
    'Layer',  # Backward compatibility
    'create_model',
    'count_parameters',
]

__version__ = "6.0"
