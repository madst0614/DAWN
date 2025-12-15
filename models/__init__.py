"""
DAWN Models Module

v16.0: Split Feature R/V (Rank Matrix)
- Feature neurons split into R (QK compression) and V pools
- Each neuron = rank matrix [n × d_model × rank]
- expand_Q/K/V linear layers for reconstruction

v16.1: Split Feature R/V + Langevin Excitability
- Same as v16.0 + adaptive dead neuron recovery
- Langevin dynamics: dw = -α*w + β*dead_ratio

v16.2: Full Q/K Projection Separation
- Feature_R과 Relational 모두 Q/K projection 분리
- proj_FR_Q, proj_FR_K: Feature_R Q/K용 별도 projection
- proj_rel_Q, proj_rel_K: Relational Q/K용 별도 projection

v16.3: Complete Q/K/V Pool Separation
- 완전한 풀 분리: FQ, FK, FV (압축) + RQ, RK, RV (복원)
- Q path: x → FQ → RQ → Q
- K path: x → FK → RK → K
- V path: x → FV → RV → V

v16.4: Shared Pool + Separate Routing (v16.3 optimized)
- Q/K 공유 풀 + 분리 routing
- Feature_QK, Feature_V (압축) + Restore_QK, Restore_V (복원)
- proj_all 통합 + matmul/bmm 최적화

v17: v16.3 + Knowledge Feature-Restore (Q/K/V 완전 분리)
- Attention: Q/K/V 완전 분리 (v16.3 동일)
- Knowledge: Feature-Restore pattern
- coarse/fine 제거 -> top_k_knowledge

v17.1: v16.4 + Knowledge Feature-Restore (Q/K 공유 풀)
- Attention: Q/K 공유 풀 (v16.4 동일)
- Knowledge: Feature-Restore pattern

baseline: Vanilla Transformer for fair comparison
"""

# v16.0 - Split Feature R/V (Rank Matrix)
from .model_v16 import DAWN as DAWN_v16

# v16.1 - Split Feature R/V + Langevin Excitability
from .model_v16_1 import DAWN as DAWN_v16_1

# v16.2 - Full Q/K Projection Separation
from .model_v16_2 import DAWN as DAWN_v16_2

# v16.3 - Complete Q/K/V Pool Separation
from .model_v16_3 import DAWN as DAWN_v16_3

# v16.4 - Shared Pool + Separate Routing (v16.3 optimized)
from .model_v16_4 import DAWN as DAWN_v16_4

# v17 - v16.3 + Knowledge Feature-Restore (Q/K/V 완전 분리)
from .model_v17 import DAWN as DAWN_v17

# v17.1 - v16.4 + Knowledge Feature-Restore (Q/K 공유 풀)
from .model_v17_1 import DAWN as DAWN_v17_1

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
    'DAWN_v16',
    'DAWN_v16_1',
    'DAWN_v16_2',
    'DAWN_v16_3',
    'DAWN_v16_4',
    'DAWN_v17',
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

__version__ = "17"


def create_model_by_version(version, config):
    """Create DAWN model by version string

    Args:
        version: "16.0", "16.1", "16.2", "16.3", "16.4", "17", "17.1", or "baseline"
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
    elif version == "16.2":
        return DAWN_v16_2(**config)
    elif version == "16.3":
        return DAWN_v16_3(**config)
    elif version == "16.4":
        return DAWN_v16_4(**config)
    elif version == "17":
        return DAWN_v17(**config)
    elif version == "17.1":
        return DAWN_v17_1(**config)
    else:
        raise ValueError(f"Unknown model version: {version}. "
                        f"Supported versions: 16.0, 16.1, 16.2, 16.3, 16.4, 17, 17.1, baseline")
