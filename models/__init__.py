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

v12.2: SSM-guided Dynamic Compress/Expand
- SSM으로 토큰 중요도 계산
- 공유 neuron_weights로 compress & expand_Q/K/V 동적 생성
- expand_neurons_Q/K/V as nn.Parameter (n_compress개)
- d_model Attention

v12.3: SSM-guided Shared Expand Pool
- expand_neurons_pool 1개 (공용 풀)
- expand_router_Q/K/V 3개로 분리 (다른 가중치, 같은 풀)
- 파라미터 절약: 3 pools → 1 pool (~0.48M)
- d_model Attention

v12.5: Global SSM + Global Router
- Global SSM: 24 -> 1 (모델 레벨에서 한 번 계산)
- Global Router: 60 -> 5 (compress, expand_Q/K/V, memory)
- SSM 문맥 강화: importance + context 출력
- context는 x에 더해서 문맥 강화

v12.6: No SSM Ablation
- SSM 제거, 단순 projection으로 importance 계산
- context 강화 제거
- Global Router만 유지 (ablation study)

v12.7: SSM without Context
- SSM 유지 (importance 계산)
- context 강화만 제거
- Ablation: v12.5 vs v12.7 = context 효과

v12.8: Top-k Sparse Mixing
- Soft mixing → Top-k sparse mixing
- Switch Transformer style load balance loss
- compress: top_k_compress (16), expand: top_k_expand (8)
- FlashAttention maintained

v13.0: Final Architecture
- Selective SSM + Context Enhancement
- Top-k Sparse Routing
- FlashAttention + Gradient Checkpointing

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

# v12.2 - SSM-guided dynamic compress/expand (shared neuron_weights, d_model attention)
from .model_v12_2 import DAWN as DAWN_v12_2

# v12.3 - SSM-guided shared expand pool (n_expand for Q/K/V, separate routers)
from .model_v12_3 import DAWN as DAWN_v12_3

# v12.4 - config-based dynamic O experiments (experimental)
from .model_v12_4 import DAWN as DAWN_v12_4

# v12.5 - Global SSM + Global Router
from .model_v12_5 import DAWN as DAWN_v12_5

# v12.6 - No SSM Ablation (simple projection for importance)
from .model_v12_6 import DAWN as DAWN_v12_6

# v12.7 - SSM without Context (SSM preserved, context removed)
from .model_v12_7 import DAWN as DAWN_v12_7

# v12.8 - Top-k Sparse Mixing with Switch-style load balance
from .model_v12_8 import DAWN as DAWN_v12_8

# v13.0 - Final Architecture (Selective SSM + Context + Top-k)
from .model_v13 import DAWN as DAWN_v13

# v13.1 - Separate QK/V Expand Pools
from .model_v13_1 import DAWN as DAWN_v13_1

# Default DAWN is v12.3 (stable)
DAWN = DAWN_v12_3

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
    'DAWN_v12_2',
    'DAWN_v12_3',
    'DAWN_v12_4',
    'DAWN_v12_5',
    'DAWN_v12_6',
    'DAWN_v12_7',
    'DAWN_v12_8',
    'DAWN_v13',
    'DAWN_v13_1',
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

__version__ = "12.3"


def create_model_by_version(version, config):
    """Create DAWN model by version string

    Args:
        version: "10.0", "11.0", "12.0", "12.1", "12.2", or "baseline"
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
    elif version == "12.2":
        return DAWN_v12_2(**config)
    elif version == "12.3":
        return DAWN_v12_3(**config)
    elif version == "12.4":
        return DAWN_v12_4(**config)
    elif version == "12.5":
        return DAWN_v12_5(**config)
    elif version == "12.6":
        return DAWN_v12_6(**config)
    elif version == "12.7":
        return DAWN_v12_7(**config)
    elif version == "12.8":
        return DAWN_v12_8(**config)
    elif version == "13.0":
        return DAWN_v13(**config)
    elif version == "13.1":
        return DAWN_v13_1(**config)
    else:
        raise ValueError(f"Unknown model version: {version}. "
                        f"Supported versions: 10.0, 11.0, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 13.0, 13.1, baseline")
