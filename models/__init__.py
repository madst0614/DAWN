"""
DAWN Models Module

v10.0: Simplified Compress/Expand Architecture
- CompressNeurons: Q/K/V/M 통합 [n_compress, d_model, rank]
- ExpandNeurons: O 통합 [n_expand, rank, d_model]
- KnowledgeNeurons: [n_knowledge, rank] + [n_knowledge, d_model]
- Soft routing (no Householder)
- 3 compressors for Q/K/V

v11.0: Unified Compression Architecture
- 1 compressor + expand_Q/K/V (instead of 3 compressors)
- Attention in d_model space (not rank space)
- d_head = d_model // n_heads
- 라우팅 연산 3배 감소

v12.0: SSM-guided Shared QKV Architecture
- SSM으로 토큰 중요도 계산
- 중요도 × 토큰별 뉴런 선호 → 레이어별 뉴런 가중치
- 공유 compress → Q/K/V
- d_model Attention

v12.1: SSM-guided Shared Neurons (v10 based)
- SSM으로 토큰 중요도 계산
- compress/expand 둘 다 뉴런 라우팅
- rank Attention (v10 style)

baseline: Vanilla Transformer for fair comparison
"""

# v10.0 - soft routing
from .model_v10 import DAWN as DAWN_v10

# v11.0 - d_model attention
from .model_v11 import DAWN as DAWN_v11

# v12.0 - SSM-guided shared QKV (d_model attention)
from .model_v12 import DAWN as DAWN_v12

# v12.1 - SSM-guided shared neurons (v10 based, rank attention)
from .model_v12_1 import DAWN as DAWN_v12_1

# Default DAWN is the latest version
DAWN = DAWN_v12_1

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
    'DAWN_v11',
    'DAWN_v12',
    'DAWN_v12_1',
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

__version__ = "12.1"


def create_model_by_version(version, config):
    """Create DAWN model by version string

    Args:
        version: "10.0", "11.0", "12.0", "12.1", or "baseline"
        config: Model configuration dict

    Returns:
        Model instance (DAWN or VanillaTransformer)
    """
    if version == "baseline":
        return VanillaTransformer(**config)

    version = normalize_version(version)

    if version == "10.0":
        return DAWN_v10(**config)
    elif version == "11.0":
        return DAWN_v11(**config)
    elif version == "12.0":
        return DAWN_v12(**config)
    elif version == "12.1":
        return DAWN_v12_1(**config)
    else:
        raise ValueError(f"Unknown model version: {version}. "
                        f"Supported versions: 10.0, 11.0, 12.0, 12.1, baseline")
