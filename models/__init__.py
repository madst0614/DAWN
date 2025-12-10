"""
DAWN Models Module

v16.0: Split Feature R/V (Rank Matrix)
- Feature neurons split into R (QK compression) and V pools
- Each neuron = rank matrix [n × d_model × rank]
- expand_Q/K/V linear layers for reconstruction

v16.1: Split Feature R/V + Langevin Excitability
- Same as v16.0 + adaptive dead neuron recovery
- Langevin dynamics: dw = -α*w + β*dead_ratio

v17.0: Full Vector Neurons + Full Soft Selection
- ALL neurons are vectors [n × d_model] (no rank matrices)
- 5 separate routing: feature_r, feature_v, relational_q, relational_k, value
- Full soft selection (train & inference)

baseline: Vanilla Transformer for fair comparison
"""

# v16.0 - Split Feature R/V (Rank Matrix)
from .model_v16 import DAWN as DAWN_v16

# v16.1 - Split Feature R/V + Langevin Excitability
from .model_v16_1 import DAWN as DAWN_v16_1

# v17.0 - Full Vector Neurons with Full Soft Selection
from .model_v17 import DAWN as DAWN_v17

# Default DAWN is v17 (latest)
DAWN = DAWN_v17

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
    'DAWN_v16',
    'DAWN_v16_1',
    'DAWN_v17',
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

__version__ = "17.0"


def create_model_by_version(version, config):
    """Create DAWN model by version string

    Args:
        version: "16.0", "17.0", or "baseline"
        config: Model configuration dict

    Returns:
        Model instance (DAWN or VanillaTransformer)
    """
    if version == "baseline":
        return VanillaTransformer(**config)

    version = normalize_version(version)

    if version == "16.0":
        return DAWN_v16(**config)
    elif version == "16.1":
        return DAWN_v16_1(**config)
    elif version == "17.0":
        return DAWN_v17(**config)
    else:
        raise ValueError(f"Unknown model version: {version}. "
                        f"Supported versions: 16.0, 16.1, 17.0, baseline")
