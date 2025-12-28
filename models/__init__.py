"""
DAWN Models Module

v18.0: Fixed Threshold Multi-Path Routing
- Fixed threshold + masked softmax for neuron selection
- Minimum/maximum neuron guarantees (path_min_k, path_max_k * max_paths)
- Multi-path (1~max_paths) parallel processing with path-wise Q,K,V aggregation
- rank=16, max_paths=4, fixed_tau=0.0, path_min_k=8, path_max_k=16

v17.1: Q/K Separate Pool + Knowledge Feature-Restore (default)
- Attention: Q/K separate pools (Feature_Q/K/V, Restore_Q/K/V)
- Knowledge: Feature-Restore pattern (Feature_Know, Restore_Know)
- 8 neuron pools for fine-grained routing

v17.2: Feature QK Unified + Restore Q/K Separate
- Feature stage: Q/K share single routing
- Restore stage: Q/K have separate routing
- Knowledge: Feature-Restore pattern

baseline: Vanilla Transformer for fair comparison
"""

# v18.0 - Fixed Threshold Multi-Path Routing
from .model_v18 import DAWN as DAWN_v18

# v17.1 - Q/K Separate Pool + Knowledge Feature-Restore (default)
from .model_v17_1 import DAWN as DAWN_v17_1

# v17.2 - Feature QK Unified + Restore Q/K Separate
from .model_v17_2 import DAWN as DAWN_v17_2

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
    'DAWN_v18',
    'DAWN_v17_2',
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
        version: "18.0", "17.2", "17.1", or "baseline"
        config: Model configuration dict

    Returns:
        Model instance (DAWN or VanillaTransformer)
    """
    if version == "baseline":
        return VanillaTransformer(**config)

    version = normalize_version(version)

    if version == "18.0":
        return DAWN_v18(**config)
    elif version == "17.2":
        return DAWN_v17_2(**config)
    elif version == "17.1":
        return DAWN_v17_1(**config)
    else:
        raise ValueError(f"Unknown model version: {version}. "
                        f"Supported versions: 18.0, 17.2, 17.1, baseline")
