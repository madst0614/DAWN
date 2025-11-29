"""
DAWN Models Module

v9.0: CompressNeurons + ExpandNeurons + ReflectionNeurons
v8.x: SharedNeurons + NeuronMemory (QK/V/O/M 분리)
baseline: Vanilla Transformer
"""

# v9.0 (current) - CompressNeurons + ExpandNeurons + ReflectionNeurons
from . import model_v9 as model_v9
from . import model_v8 as model_v8

# Version registry
from .version_registry import (
    VERSION_REGISTRY,
    normalize_version,
    get_version_info,
    get_required_params,
    get_optional_params,
    build_model_kwargs,
    print_version_info,
    list_versions,
    get_all_versions_info,
)

# Baseline Vanilla Transformer
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from baseline_transformer import VanillaTransformer
except ImportError:
    VanillaTransformer = None

__all__ = [
    # Models
    'model_v9',
    'model_v8',
    'VanillaTransformer',
    # Version utilities
    'VERSION_REGISTRY',
    'normalize_version',
    'get_version_info',
    'get_required_params',
    'get_optional_params',
    'build_model_kwargs',
    'print_version_info',
    'list_versions',
    'get_all_versions_info',
    'create_model_by_version',
]

__version__ = "9.0"


# Helper function to create model based on version
def create_model_by_version(version, config):
    """Create DAWN model by version string

    Args:
        version: "9.0", "8.0", "8.1", "8.2", "8.3", or "baseline"
        config: Model configuration dict

    Returns:
        DAWN or VanillaTransformer model instance
    """
    version = normalize_version(version)

    if version == "9.0":
        from .model_v9 import DAWN as DAWN_v9
        return DAWN_v9(**config)
    elif version in ["8.0", "8.1", "8.2", "8.3"]:
        from .model_v8 import DAWN as DAWN_v8
        return DAWN_v8(**config)
    elif version == "baseline":
        if VanillaTransformer is None:
            raise ImportError("VanillaTransformer not available. "
                            "Make sure baseline_transformer.py is in the project root.")
        return VanillaTransformer(**config)
    else:
        raise ValueError(f"Unknown model version: {version}. "
                        f"Supported versions: {list_versions()}")
