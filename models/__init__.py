"""
DAWN Models Module

v8.2: v8.1 + F.normalize Householder (apply_householder 개선)
v8.1: v8.0 + Expanded KnowledgeNeurons (n_knowledge=128, knowledge_k=16)
v8.0: SharedNeurons + NeuronMemory (FFN 대체)
v7.9: NeuronCircuit with Householder Transformations
v7.8: Independent Neuron Projections (No Basis Mixing)
v7.7: QK/VO Basis Separation with Symmetric O Projection
v7.6: Independent O Basis + Recipe Diversity
v7.5: Dynamic Q/K/V Generation (v8 design)
v7.4: TT Weighted Karcher Mean
v7.2: Standard FFN + Neuron Routing (병목 탐색 실험)
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

# v8, v7.9, v7.8, v7.7, v7.6, v7.5, v7.4, v7.2, v7.0 and v6.0 compatibility imports
from . import model_v8 as model_v8
from . import model_v79 as model_v79
from . import model_v78 as model_v78
from . import model_v77 as model_v77
from . import model_v76 as model_v76
from . import model_v75 as model_v75
from . import model_v74 as model_v74
from . import model_v72 as model_v72
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
    'model_v8',   # Access v8.0 via models.model_v8
    'model_v79',  # Access v7.9 via models.model_v79
    'model_v78',  # Access v7.8 via models.model_v78
    'model_v77',  # Access v7.7 via models.model_v77
    'model_v76',  # Access v7.6 via models.model_v76
    'model_v75',  # Access v7.5 via models.model_v75
    'model_v74',  # Access v7.4 via models.model_v74
    'model_v72',  # Access v7.2 via models.model_v72
    'model_v70',  # Access v7.0 via models.model_v70
    'model_v6',   # Access v6.0 via models.model_v6
    'VanillaTransformer',  # Baseline model
]

__version__ = "7.1"


# Helper function to create model based on version
def create_model_by_version(version, config):
    """Create DAWN model by version string

    Args:
        version: "8.2", "8.1", "8.0", "7.9", "7.8", "7.7", "7.6", "7.5", "7.4", "7.2", "7.1", "7.0", "6.0", or "baseline"
        config: Model configuration dict

    Returns:
        DAWN or VanillaTransformer model instance
    """
    version = str(version)

    if version in ["8.2", "82", "8.1", "81", "8.0", "8", "80"]:
        from .model_v8 import DAWN as DAWN_v8
        return DAWN_v8(**config)
    elif version in ["7.9", "79"]:
        from .model_v79 import DAWN as DAWN_v79
        return DAWN_v79(**config)
    elif version in ["7.8", "78"]:
        from .model_v78 import DAWN as DAWN_v78
        return DAWN_v78(**config)
    elif version in ["7.7", "77"]:
        from .model_v77 import DAWN as DAWN_v77
        return DAWN_v77(**config)
    elif version in ["7.6", "76"]:
        from .model_v76 import DAWN as DAWN_v76
        return DAWN_v76(**config)
    elif version in ["7.5", "75"]:
        from .model_v75 import DAWN as DAWN_v75
        return DAWN_v75(**config)
    elif version in ["7.4", "74"]:
        from .model_v74 import DAWN as DAWN_v74
        return DAWN_v74(**config)
    elif version in ["7.2", "72"]:
        from .model_v72 import DAWN as DAWN_v72
        return DAWN_v72(**config)
    elif version in ["7.1", "7", "71"]:
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
                        f"Supported versions: 8.2, 8.1, 8.0, 7.9, 7.8, 7.7, 7.6, 7.5, 7.4, 7.2, 7.1, 7.0, 6.0, baseline")
