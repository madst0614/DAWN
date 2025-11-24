"""
DAWN Models Module

v7.1: Symmetric Basis FFN (W_down 제거) - DEFAULT
v7.0: Fixed Orthogonal Basis
v6.0: Orthogonal Basis FFN with Token-level Dynamic FFN (legacy)
"""

# v7.1 (default) - Symmetric Basis FFN
from .model_v71 import (
    DAWN,
    DAWNLanguageModel,
    SimpleRouter,
    SymmetricBasisFFN,
    FixedOrthogonalBasis,
    DAWNLayer,
    create_model,
    count_parameters,
)

# v7.0 and v6.0 compatibility imports
from . import model_v7 as model_v70
from . import model as model_v6

# Baseline Vanilla Transformer
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from baseline_transformer import VanillaTransformer
except ImportError:
    VanillaTransformer = None

__all__ = [
    'DAWN',
    'DAWNLanguageModel',
    'SimpleRouter',
    'SymmetricBasisFFN',
    'FixedOrthogonalBasis',
    'DAWNLayer',
    'create_model',
    'count_parameters',
    'model_v70',  # Access v7.0 via models.model_v70
    'model_v6',   # Access v6.0 via models.model_v6
    'VanillaTransformer',  # Baseline model
]

__version__ = "7.1"


# Helper function to create model based on version
def create_model_by_version(version, config):
    """Create DAWN model by version string

    Args:
        version: "7.1", "7.0", "6.0", or "baseline"
        config: Model configuration dict

    Returns:
        DAWN or VanillaTransformer model instance
    """
    version = str(version)

    if version in ["7.1", "7", "71"]:
        from .model_v71 import DAWN as DAWN_v71
        return DAWN_v71(**config)
    elif version in ["7.0", "70"]:
        from .model_v7 import DAWN as DAWN_v70
        return DAWN_v70(**config)
    elif version in ["6.0", "6", "60"]:
        from .model import DAWN as DAWN_v60
        return DAWN_v60(**config)
    elif version == "baseline":
        if VanillaTransformer is None:
            raise ImportError("VanillaTransformer not available. "
                            "Make sure baseline_transformer.py is in the project root.")
        return VanillaTransformer(**config)
    else:
        raise ValueError(f"Unknown model version: {version}. "
                        f"Supported versions: 7.1, 7.0, 6.0, baseline")
