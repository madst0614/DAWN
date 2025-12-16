"""
DAWN Models Module

v17.1: Q/K Shared Pool + Knowledge Feature-Restore
- Attention: Q/K shared pool (Feature_QK, Feature_V, Restore_QK, Restore_V)
- Knowledge: Feature-Restore pattern (Feature_Know, Restore_Know)
- 6 neuron pools total for efficient routing

baseline: Vanilla Transformer for fair comparison
"""

# v17.1 - Q/K Shared Pool + Knowledge Feature-Restore
from .model_v17_1 import DAWN as DAWN_v17_1

# Default DAWN is v17.1
DAWN = DAWN_v17_1

# Baseline for comparison
from .baseline_transformer import VanillaTransformer

# Version registry
from .version_registry import (
    VERSION_REGISTRY,
    normalize_version,
    get_version_info,
    get_required_params,
    get_optional_params,
    build_model_kwargs,
    build_args_config,
    load_model_params_to_args,
    print_version_info,
    list_versions,
    get_all_versions_info,
    get_routing_log_info,
)

__all__ = [
    # Models
    'DAWN',
    'DAWN_v17_1',
    'VanillaTransformer',
    # Version utilities
    'VERSION_REGISTRY',
    'normalize_version',
    'get_version_info',
    'get_required_params',
    'get_optional_params',
    'build_model_kwargs',
    'build_args_config',
    'load_model_params_to_args',
    'print_version_info',
    'list_versions',
    'get_all_versions_info',
    'get_routing_log_info',
    'create_model_by_version',
]

__version__ = "17.1"


def create_model_by_version(version, config):
    """Create DAWN model by version string

    Args:
        version: "17.1" or "baseline"
        config: Model configuration dict

    Returns:
        Model instance (DAWN or VanillaTransformer)
    """
    if version == "baseline":
        return VanillaTransformer(**config)

    version = normalize_version(version)

    if version == "17.1":
        return DAWN_v17_1(**config)
    else:
        raise ValueError(f"Unknown model version: {version}. "
                        f"Supported versions: 17.1, baseline")
