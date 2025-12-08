"""
DAWN Models Module

v10.0: Simplified Compress/Expand Architecture
- CompressNeurons: Q/K/V/M 통합 [n_compress, d_model, rank]
- ExpandNeurons: O 통합 [n_expand, rank, d_model]
- KnowledgeNeurons: [n_knowledge, rank] + [n_knowledge, d_model]
- Soft routing (no Householder)
- 3 compressors for Q/K/V

v13.0: Final Architecture
- Selective SSM + Context Enhancement
- Top-k Sparse Routing
- FlashAttention + Gradient Checkpointing

v13.1: Separate QK/V Expand Pools
- Q/K share expand pool, V has separate pool
- 5 routers: compress, Q, K, V, memory

v13.2: Unified Neuron Router
- All neurons in same embedding space
- Single router with type-specific slicing
- Starvation bonus with decay

v14.0: FRTK Architecture
- Feature-Relational-Transfer-Knowledge naming
- Synaptic Activation Regulation (SAR) replaces starvation weight
- LR-based adaptive bounds (tight early, loose late)

v15.0: Direct Knowledge Projection
- NeuronMemory bypasses Feature neurons
- x → proj_k → Q (direct 128-dim projection)
- Simpler memory access, independent of router

v16.0: Split Feature QK/V Vector Neurons
- Feature neurons split into QK and V pools
- Each neuron = single axis vector (n_feature_qk × d_model)
- expand_Q/K/V linear layers for reconstruction
- 41% parameter reduction from v15

v17.0: Full Vector Neurons
- ALL neurons are vectors (no rank matrices)
- 5 separate routing: feature_qk, feature_v, relational_q, relational_k, value
- Excitability completely removed
- 82% parameter reduction from v15

baseline: Vanilla Transformer for fair comparison
"""

# v10.0 - soft routing
from .model_v10 import DAWN as DAWN_v10

# v13.0 - Final Architecture (Selective SSM + Context + Top-k)
from .model_v13 import DAWN as DAWN_v13

# v13.1 - Separate QK/V Expand Pools
from .model_v13_1 import DAWN as DAWN_v13_1

# v13.2 - Unified Neuron Router
from .model_v13_2 import DAWN as DAWN_v13_2

# v14.0 - FRTK Architecture with SAR
from .model_v14 import DAWN as DAWN_v14

# v15.0 - Direct Knowledge Projection
from .model_v15 import DAWN as DAWN_v15

# v16.0 - Split Feature QK/V Vector Neurons
from .model_v16 import DAWN as DAWN_v16

# v17.0 - Full Vector Neurons with Excitability
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
    'DAWN_v10',
    'DAWN_v13',
    'DAWN_v13_1',
    'DAWN_v13_2',
    'DAWN_v14',
    'DAWN_v15',
    'DAWN_v16',
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
        version: "10.0", "13.0", "13.1", "13.2", "14.0", "15.0", "16.0", "17.0", or "baseline"
        config: Model configuration dict

    Returns:
        Model instance (DAWN or VanillaTransformer)
    """
    if version == "baseline":
        return VanillaTransformer(**config)

    version = normalize_version(version)

    if version == "10.0":
        return DAWN_v10(**config)
    elif version == "13.0":
        return DAWN_v13(**config)
    elif version == "13.1":
        return DAWN_v13_1(**config)
    elif version == "13.2":
        return DAWN_v13_2(**config)
    elif version == "14.0":
        return DAWN_v14(**config)
    elif version == "15.0":
        return DAWN_v15(**config)
    elif version == "16.0":
        return DAWN_v16(**config)
    elif version == "17.0":
        return DAWN_v17(**config)
    else:
        raise ValueError(f"Unknown model version: {version}. "
                        f"Supported versions: 10.0, 13.0, 13.1, 13.2, 14.0, 15.0, 16.0, 17.0, baseline")
