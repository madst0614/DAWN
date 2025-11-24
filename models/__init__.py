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
]

__version__ = "7.1"


# Helper function to create model based on version
def create_model_by_version(version, config):
    """Create DAWN model by version string

    Args:
        version: "7.1", "7.0", or "6.0"
        config: Model configuration dict

    Returns:
        DAWN model instance
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
    else:
        raise ValueError(f"Unknown model version: {version}. "
                        f"Supported versions: 7.1, 7.0, 6.0")
