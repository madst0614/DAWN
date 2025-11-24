"""
DAWN Models Module

v7.0: Fixed Orthogonal Basis (default)
v6.0: Orthogonal Basis FFN with Token-level Dynamic FFN (legacy)
"""

# v7.0 (default)
from .model_v7 import (
    DAWN,
    DAWNLanguageModel,
    SimpleRouter,
    RecipeFFN,
    FixedOrthogonalBasis,
    DAWNLayer,
    create_model,
    count_parameters,
)

# v6.0 compatibility imports
from . import model as model_v6

__all__ = [
    'DAWN',
    'DAWNLanguageModel',
    'SimpleRouter',
    'RecipeFFN',
    'FixedOrthogonalBasis',
    'DAWNLayer',
    'create_model',
    'count_parameters',
    'model_v6',  # Access v6.0 via models.model_v6
]

__version__ = "7.0"
