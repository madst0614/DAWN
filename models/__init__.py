"""
DAWN Models Module

v10.0: Simplified Compress/Expand Architecture
- CompressNeurons: Q/K/V/M 통합 [n_compress, d_model, rank]
- ExpandNeurons: O 통합 [n_expand, rank, d_model]
- KnowledgeNeurons: [n_knowledge, rank] + [n_knowledge, d_model]
- Soft routing (no Householder)

baseline: Vanilla Transformer for fair comparison
"""

# v10.0 - stable version
from .model_v10 import DAWN as DAWN_v10

# Default DAWN is the latest version
DAWN = DAWN_v10

# Baseline for comparison
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline_transformer import VanillaTransformer

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

__all__ = [
    # Models
    'DAWN',
    'DAWN_v10',
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

__version__ = "10.0"


def create_model_by_version(version, config):
    """Create DAWN model by version string

    Args:
        version: "10.0", "10", or "baseline"
        config: Model configuration dict

    Returns:
        Model instance (DAWN or VanillaTransformer)
    """
    if version == "baseline":
        return VanillaTransformer(**config)

    version = normalize_version(version)

    if version == "10.0":
        return DAWN_v10(**config)
    else:
        raise ValueError(f"Unknown model version: {version}. "
                        f"Supported versions: 10.0, baseline")
