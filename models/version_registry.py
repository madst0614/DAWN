"""
DAWN Model Version Registry - Single Source of Truth

Version History:
  v16.0: Split Feature R/V (rank matrix) - Feature_R/V separate compression
  v16.1: Split Feature R/V + Langevin Excitability (adaptive dead neuron recovery)
  v16.2: Full Q/K Projection Separation - Q/K routing paths separated
  v16.3: Complete Q/K/V Pool Separation - FQ/FK/FV, RQ/RK/RV all independent
  v16.4: Shared Pool + Separate Routing - v16.3 optimized, Q/K shared pool with separate routing
  v17: v16.3 + Knowledge Feature-Restore 분리 라우팅 (8개 독립 풀: FQ/FK/FV/RQ/RK/RV + Feature_Know/Restore_Know)

================================================================================
HOW TO ADD A NEW VERSION (e.g., v16.5)
================================================================================

1. VERSION_REGISTRY 엔트리 추가 (이 파일)
   ─────────────────────────────────────────
   "16.5": {
       "description": "Short description",
       "aliases": ["165"],
       "module": "model_v16_5",
       "required_params": [
           "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
           "your_required_param1", "your_required_param2", ...
           "rank",
       ],
       "optional_params": {
           "dropout": 0.1,
           "your_optional_param": default_value,
           ...
       },
       "display_info": lambda args: [
           f"DAWN v16.5: Description",
           f"  param1: {args.get('param1')}",
           ...
       ],
   }

2. 모델 파일 생성: models/model_v16_5.py
   ─────────────────────────────────────────
   - class DAWN(nn.Module) with __version__ = "16.5"
   - __init__에서 VERSION_REGISTRY의 required_params + optional_params 받기

3. models/__init__.py 업데이트
   ─────────────────────────────────────────
   - import 추가: from .model_v16_5 import DAWN as DAWN_v16_5
   - __all__에 'DAWN_v16_5' 추가
   - create_model_by_version()에 elif 추가
   - (선택) 기본 DAWN = DAWN_v16_5로 변경

4. Config 파일 생성 (선택): configs/train_config_v16_5_*.yaml
   ─────────────────────────────────────────
   model:
     model_version: "16.5"
     your_required_param1: value
     your_optional_param: value
     ...

5. scripts/train.py 업데이트 (메이저 버전 변경 시 필수!)
   ─────────────────────────────────────────
   a) needs_routing_info() - 200 step마다 콘솔 로그 출력 조건
      - is_v17_model() 같은 버전 체크 함수 추가
      - needs_routing_info()에 or is_vXX_model(model) 추가

   b) _get_router_log_lines() - Usage EMA 로깅
      - router의 usage_ema_* 속성명이 변경되면 elif 브랜치 추가
      - 예: usage_ema_fq (v16.3) → usage_ema_feature_q (v17)

   c) get_routing_log_info() (이 파일 하단) - Entropy/Variance 로깅
      - routing_info의 *_pref 키가 변경되면 elif 브랜치 추가
      - 예: fq_pref (v16.3) → feature_q_pref (v17)

================================================================================
train.py 파라미터 로딩은 자동!
  - build_args_config(): args에서 모든 버전 파라미터 자동 추출
  - load_model_params_to_args(): YAML/checkpoint에서 args로 자동 로딩
  - build_model_kwargs(): 버전별 필요 파라미터만 필터링
================================================================================
"""

from typing import Dict, Any, List
import torch


VERSION_REGISTRY = {
    "16.0": {
        "description": "Split Feature R/V (rank matrix)",
        "aliases": ["16", "160"],
        "module": "model_v16",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_r", "n_feature_v", "n_relational", "n_value", "n_knowledge",
            "rank",  # single rank for all matrices
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_feature_r": 8,
            "top_k_feature_v": 8,
            "top_k_relational": 4,
            "top_k_value": 6,
            "d_space": 64,
            "coarse_k": 20,
            "fine_k": 10,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "excitability_tau": 1.5,
            "excitability_ema_alpha": 0.01,
            "excitability_decay_rate": 0.99995,
        },
        "display_info": lambda args: [
            f"DAWN v16: Split Feature R/V (rank matrix)",
            f"  rank={args.get('rank', args.get('basis_rank'))}",
            f"  Feature_R: {args.get('n_feature_r')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (top-k={args.get('top_k_feature_r', 8)})",
            f"  Feature_V: {args.get('n_feature_v')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (top-k={args.get('top_k_feature_v', 8)})",
            f"  Relational: {args.get('n_relational')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (Q/K expansion, top-k={args.get('top_k_relational', 4)})",
            f"  Value: {args.get('n_value')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (V expansion, top-k={args.get('top_k_value', 6)})",
            f"  Unified Router: d_space={args.get('d_space', 64)} + SAR",
            f"  Selective SSM: state_dim={args.get('state_dim', 64)}",
            f"  Architecture: Mamba SSM → Context → Unified Router → Rank Matrix Expand → FlashAttn",
            f"  Memory: 2-stage (x→router→coarse, x→encoder→fine)",
            f"  KnowledgeNeurons (K):",
            f"    - K: {args.get('n_knowledge')} × {args.get('knowledge_rank', 128)}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - coarse_k: {args.get('coarse_k', 20)} → fine_k: {args.get('fine_k', 10)}",
        ],
    },
    "16.1": {
        "description": "Split Feature R/V + Langevin Excitability",
        "aliases": ["161"],
        "module": "model_v16_1",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_r", "n_feature_v", "n_relational", "n_value", "n_knowledge",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_feature_r": 8,
            "top_k_feature_v": 8,
            "top_k_relational": 4,
            "top_k_value": 6,
            "d_space": 64,
            "coarse_k": 20,
            "fine_k": 10,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "excitability_tau": 1.5,
            "excitability_ema_alpha": 0.01,
            "langevin_alpha": 0.0003,
            "langevin_beta": 0.0006,
        },
        "display_info": lambda args: [
            f"DAWN v16.1: Split Feature R/V + Langevin Excitability",
            f"  rank={args.get('rank', args.get('basis_rank'))}",
            f"  Feature_R: {args.get('n_feature_r')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (top-k={args.get('top_k_feature_r', 8)})",
            f"  Feature_V: {args.get('n_feature_v')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (top-k={args.get('top_k_feature_v', 8)})",
            f"  Relational: {args.get('n_relational')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (Q/K expansion, top-k={args.get('top_k_relational', 4)})",
            f"  Value: {args.get('n_value')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (V expansion, top-k={args.get('top_k_value', 6)})",
            f"  Unified Router: d_space={args.get('d_space', 64)}",
            f"  Langevin: α={args.get('langevin_alpha', 0.0003)}, β={args.get('langevin_beta', 0.0006)}",
            f"  Selective SSM: state_dim={args.get('state_dim', 64)}",
        ],
    },
    "16.2": {
        "description": "Full Q/K Projection Separation",
        "aliases": ["162"],
        "module": "model_v16_2",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_r", "n_feature_v", "n_relational", "n_value", "n_knowledge",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_feature_r": 8,
            "top_k_feature_v": 8,
            "top_k_relational": 4,
            "top_k_value": 6,
            "d_space": 64,
            "coarse_k": 20,
            "fine_k": 10,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "excitability_tau": 1.5,
            "excitability_ema_alpha": 0.01,
            "excitability_decay_rate": 0.99995,
        },
        "display_info": lambda args: [
            f"DAWN v16.2: Full Q/K Projection Separation",
            f"  rank={args.get('rank', args.get('basis_rank'))}",
            f"  Feature_R: {args.get('n_feature_r')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (Q/K separate, top-k={args.get('top_k_feature_r', 8)})",
            f"  Feature_V: {args.get('n_feature_v')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (top-k={args.get('top_k_feature_v', 8)})",
            f"  Relational: {args.get('n_relational')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (Q/K separate, top-k={args.get('top_k_relational', 4)})",
            f"  Value: {args.get('n_value')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (top-k={args.get('top_k_value', 6)})",
            f"  Unified Router: d_space={args.get('d_space', 64)} + SAR",
            f"  Selective SSM: state_dim={args.get('state_dim', 64)}",
        ],
    },
    "16.3": {
        "description": "Complete Q/K/V Pool Separation",
        "aliases": ["163"],
        "module": "model_v16_3",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_fq", "n_fk", "n_fv", "n_rq", "n_rk", "n_rv", "n_knowledge",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_fq": 8,
            "top_k_fk": 8,
            "top_k_fv": 3,
            "top_k_rq": 8,
            "top_k_rk": 8,
            "top_k_rv": 3,
            "d_space": 64,
            "coarse_k": 40,
            "fine_k": 15,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "excitability_tau": 1.5,
            "excitability_ema_alpha": 0.01,
            "excitability_decay_rate": 0.99995,
        },
        "display_info": lambda args: [
            f"DAWN v16.3: Complete Q/K/V Pool Separation",
            f"  rank={args.get('rank', args.get('basis_rank'))}",
            f"  FQ: {args.get('n_fq')} × {args.get('d_model')} × {args.get('rank')} (top-k={args.get('top_k_fq', 8)})",
            f"  FK: {args.get('n_fk')} × {args.get('d_model')} × {args.get('rank')} (top-k={args.get('top_k_fk', 8)})",
            f"  FV: {args.get('n_fv')} × {args.get('d_model')} × {args.get('rank')} (top-k={args.get('top_k_fv', 8)})",
            f"  RQ: {args.get('n_rq')} × {args.get('rank')} × {args.get('d_model')} (top-k={args.get('top_k_rq', 8)})",
            f"  RK: {args.get('n_rk')} × {args.get('rank')} × {args.get('d_model')} (top-k={args.get('top_k_rk', 8)})",
            f"  RV: {args.get('n_rv')} × {args.get('rank')} × {args.get('d_model')} (top-k={args.get('top_k_rv', 8)})",
            f"  Knowledge: {args.get('n_knowledge')} (coarse={args.get('coarse_k', 40)} → fine={args.get('fine_k', 15)})",
            f"  Unified Router: d_space={args.get('d_space', 64)} + SAR",
            f"  Selective SSM: state_dim={args.get('state_dim', 64)}",
        ],
    },
    "16.4": {
        "description": "Shared Pool + Separate Routing (v16.3 optimized)",
        "aliases": ["164"],
        "module": "model_v16_4",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_qk", "n_feature_v", "n_restore_qk", "n_restore_v", "n_knowledge",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_feature_qk": 8,
            "top_k_feature_v": 3,
            "top_k_restore_qk": 8,
            "top_k_restore_v": 3,
            "d_space": 64,
            "coarse_k": 16,
            "fine_k": 8,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "router_dropout": 0.1,
            "token_routing": False,
            "use_ssm_context": True,
            "excitability_tau": 1.5,
            "excitability_ema_alpha": 0.01,
            "excitability_decay_rate": 0.99995,
        },
        "display_info": lambda args: [
            f"DAWN v16.4: Shared Pool + Separate Routing",
            f"  rank={args.get('rank', args.get('basis_rank'))}",
            f"  Feature_QK: {args.get('n_feature_qk')} × {args.get('d_model')} × {args.get('rank')} (Q/K shared, top-k={args.get('top_k_feature_qk', 8)})",
            f"  Feature_V: {args.get('n_feature_v')} × {args.get('d_model')} × {args.get('rank')} (top-k={args.get('top_k_feature_v', 3)})",
            f"  Restore_QK: {args.get('n_restore_qk')} × {args.get('rank')} × {args.get('d_model')} (Q/K shared, top-k={args.get('top_k_restore_qk', 8)})",
            f"  Restore_V: {args.get('n_restore_v')} × {args.get('rank')} × {args.get('d_model')} (top-k={args.get('top_k_restore_v', 3)})",
            f"  Knowledge: {args.get('n_knowledge')} (coarse={args.get('coarse_k', 16)} → fine={args.get('fine_k', 8)})",
            f"  Unified Router: d_space={args.get('d_space', 64)} + proj_all optimized",
            f"  Selective SSM: state_dim={args.get('state_dim', 64)}",
        ],
    },
    "17": {
        "description": "v16.3 + Knowledge Feature-Restore 분리 라우팅",
        "aliases": ["17.0", "170"],
        "module": "model_v17",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_q", "n_feature_k", "n_feature_v", "n_restore_q", "n_restore_k", "n_restore_v",
            "n_feature_know", "n_restore_know",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_feature_q": 8,
            "top_k_feature_k": 8,
            "top_k_feature_v": 3,
            "top_k_restore_q": 8,
            "top_k_restore_k": 8,
            "top_k_restore_v": 3,
            "top_k_feature_know": 4,
            "top_k_restore_know": 4,
            "d_space": 64,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "router_dropout": 0.1,
            "token_routing": False,
            "use_ssm_context": True,
            "excitability_tau": 1.5,
            "excitability_ema_alpha": 0.01,
            "excitability_decay_rate": 0.99995,
        },
        "display_info": lambda args: [
            f"DAWN v17: Q/K/V Separated + Knowledge Feature-Restore 분리",
            f"  rank={args.get('rank', args.get('basis_rank'))}, knowledge_rank={args.get('knowledge_rank', 128)}",
            f"  Feature_Q: {args.get('n_feature_q')} (k={args.get('top_k_feature_q', 8)}), Feature_K: {args.get('n_feature_k')} (k={args.get('top_k_feature_k', 8)}), Feature_V: {args.get('n_feature_v')} (k={args.get('top_k_feature_v', 3)})",
            f"  Restore_Q: {args.get('n_restore_q')} (k={args.get('top_k_restore_q', 8)}), Restore_K: {args.get('n_restore_k')} (k={args.get('top_k_restore_k', 8)}), Restore_V: {args.get('n_restore_v')} (k={args.get('top_k_restore_v', 3)})",
            f"  Feature_Know: {args.get('n_feature_know')} (k={args.get('top_k_feature_know', 4)}), Restore_Know: {args.get('n_restore_know')} (k={args.get('top_k_restore_know', 4)})",
        ],
    },

    "17.1": {
        "description": "v16.4 + Knowledge Feature-Restore (Q/K 공유 풀)",
        "aliases": ["171"],
        "module": "model_v17_1",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature_qk", "n_feature_v", "n_restore_qk", "n_restore_v",
            "n_feature_know", "n_restore_know",
            "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_feature_qk": 8,
            "top_k_feature_v": 3,
            "top_k_restore_qk": 8,
            "top_k_restore_v": 3,
            "top_k_feature_know": 4,
            "top_k_restore_know": 4,
            "d_space": 64,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
            "router_dropout": 0.1,
            "token_routing": False,
            "knowledge_token_routing": False,
            "use_ssm_context": True,
            "excitability_tau": 1.5,
            "excitability_ema_alpha": 0.01,
            "excitability_decay_rate": 0.99995,
        },
        "display_info": lambda args: [
            f"DAWN v17.1: Q/K Shared + Knowledge Feature-Restore",
            f"  rank={args.get('rank', args.get('basis_rank'))}, knowledge_rank={args.get('knowledge_rank', 128)}",
            f"  Feature_QK: {args.get('n_feature_qk')} (top-k={args.get('top_k_feature_qk', 8)}) - Q/K shared pool",
            f"  Feature_V: {args.get('n_feature_v')} (top-k={args.get('top_k_feature_v', 3)})",
            f"  Restore_QK: {args.get('n_restore_qk')} (top-k={args.get('top_k_restore_qk', 8)}) - Q/K shared pool",
            f"  Restore_V: {args.get('n_restore_v')} (top-k={args.get('top_k_restore_v', 3)})",
            f"  Feature_Know: {args.get('n_feature_know')} (k={args.get('top_k_feature_know', 4)}), Restore_Know: {args.get('n_restore_know')} (k={args.get('top_k_restore_know', 4)})",
        ],
    },

    # Baseline Transformer for fair comparison
    "baseline": {
        "description": "Vanilla Transformer Baseline",
        "aliases": ["vanilla", "base"],
        "module": "baseline_transformer",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
        ],
        "optional_params": {
            "d_ff": None,  # defaults to 4 * d_model if not specified
            "dropout": 0.1,
        },
        "display_info": lambda args: [
            f"Vanilla Transformer Baseline",
            f"  d_model={args.get('d_model')}, n_layers={args.get('n_layers')}, n_heads={args.get('n_heads')}",
            f"  d_ff={args.get('d_ff', args.get('d_model', 256) * 4)}",
        ],
    },
}


def normalize_version(version: str) -> str:
    """Normalize version string to canonical form."""
    version = str(version)

    if version in VERSION_REGISTRY:
        return version

    for canonical, info in VERSION_REGISTRY.items():
        if version in info.get('aliases', []):
            return canonical

    raise ValueError(f"Unknown version: {version}. Supported: {', '.join(VERSION_REGISTRY.keys())}")


def get_version_info(version: str) -> Dict[str, Any]:
    """Get version info from registry."""
    canonical = normalize_version(version)
    return VERSION_REGISTRY[canonical]


def get_required_params(version: str) -> List[str]:
    """Get list of required parameters for a version."""
    info = get_version_info(version)
    return info.get('required_params', [])


def get_optional_params(version: str) -> Dict[str, Any]:
    """Get optional parameters with defaults for a version."""
    info = get_version_info(version)
    return info.get('optional_params', {})


def build_model_kwargs(version: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Build model kwargs from config."""
    version = normalize_version(version)
    info = VERSION_REGISTRY[version]

    kwargs = {}

    for param in info.get('required_params', []):
        if param in config:
            kwargs[param] = config[param]

    for param, default in info.get('optional_params', {}).items():
        kwargs[param] = config.get(param, default)

    return kwargs


def load_model_params_to_args(args, config: Dict[str, Any]) -> None:
    """Load model params from config dict to args object using VERSION_REGISTRY.

    Updates args in-place with values from config, using VERSION_REGISTRY defaults.
    This is the inverse of build_args_config - used for loading from YAML or checkpoint.

    Args:
        args: Namespace object to update
        config: Config dict (from YAML or checkpoint)
    """
    # Collect all param names and defaults from all versions
    all_params = {}

    for version_info in VERSION_REGISTRY.values():
        for param in version_info.get('required_params', []):
            if param not in all_params:
                all_params[param] = None
        for param, default in version_info.get('optional_params', {}).items():
            if param not in all_params:
                all_params[param] = default

    # Special mappings (config param name -> args attribute name)
    special_mappings = {
        'rank': 'basis_rank',  # config['rank'] -> args.basis_rank
    }

    for param, default in all_params.items():
        # Skip vocab_size (set separately)
        if param in ('vocab_size',):
            continue

        # Get args attribute name
        args_attr = special_mappings.get(param, param)

        # Get value from config with fallback to current args value, then default
        current_value = getattr(args, args_attr, default)
        value = config.get(param, current_value)

        # Set on args
        setattr(args, args_attr, value)

        # Also set 'rank' if we set 'basis_rank' (keep them in sync)
        if args_attr == 'basis_rank':
            setattr(args, 'rank', value)


def build_args_config(args, vocab_size: int) -> Dict[str, Any]:
    """Build config dict from args object dynamically using VERSION_REGISTRY.

    Extracts all params defined in any version's required_params and optional_params.
    This is the single source of truth - no need to hardcode params in train.py.

    Args:
        args: Namespace object with model config attributes
        vocab_size: Vocabulary size (usually from tokenizer)

    Returns:
        Config dict ready for build_model_kwargs()
    """
    # Collect all param names and defaults from all versions
    all_params = {}  # param_name -> default_value

    for version_info in VERSION_REGISTRY.values():
        # Required params (no default, will use None)
        for param in version_info.get('required_params', []):
            if param not in all_params:
                all_params[param] = None

        # Optional params (with defaults)
        for param, default in version_info.get('optional_params', {}).items():
            if param not in all_params:
                all_params[param] = default

    # Build config from args
    config = {'vocab_size': vocab_size}

    # Special mappings (args attribute name -> config param name)
    special_mappings = {
        'basis_rank': 'rank',  # args.basis_rank -> config['rank']
    }

    for param, default in all_params.items():
        # Check special mappings first
        args_attr = None
        for args_name, config_name in special_mappings.items():
            if config_name == param:
                args_attr = args_name
                break

        if args_attr is None:
            args_attr = param

        # Get value from args with default
        if hasattr(args, args_attr):
            value = getattr(args, args_attr)
            # Handle None values for optional params
            if value is None and default is not None:
                value = default
            config[param] = value
        elif hasattr(args, param):
            value = getattr(args, param)
            if value is None and default is not None:
                value = default
            config[param] = value
        elif default is not None:
            config[param] = default

    return config


def print_version_info(version: str, args: Dict[str, Any]) -> None:
    """Print version-specific architecture information."""
    version = normalize_version(version)
    info = VERSION_REGISTRY.get(version, {})

    print(f"Model version: {version}")
    print(f"Description: {info.get('description', 'N/A')}")

    display_fn = info.get('display_info')
    if display_fn:
        lines = display_fn(args)
        for line in lines:
            print(line)


def list_versions() -> List[str]:
    """Get list of all available versions."""
    return list(VERSION_REGISTRY.keys())


def get_all_versions_info() -> str:
    """Get formatted string of all versions and descriptions."""
    lines = ["Available DAWN versions:"]
    for version, info in VERSION_REGISTRY.items():
        aliases = info.get('aliases', [])
        alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
        lines.append(f"  v{version}: {info['description']}{alias_str}")
    return "\n".join(lines)


def get_routing_log_info(routing_infos, calc_entropy_fn, calc_var_fn) -> Dict[str, Any]:
    """
    Extract routing log info based on routing_info structure.
    Auto-detects version from routing_info keys.
    Computes AVERAGE entropy across all layers for better global view.

    Args:
        routing_infos: List of routing_info dicts from all layers, or single dict (legacy)
        calc_entropy_fn: Function to calculate entropy ratio (pref -> float)
        calc_var_fn: Function to calculate token variance (pref -> float)

    Returns:
        Dict with keys: 'ent_str', 'var_str', 'overlap_str', 'version'
    """
    # Support both list and single dict (backward compatibility)
    if isinstance(routing_infos, dict):
        routing_infos = [routing_infos]

    # Use first layer to detect version
    attn0 = routing_infos[0].get('attention', routing_infos[0])

    # v16.4 / v17.1: Shared Pool + Separate Routing (fqk_q_pref exists)
    if attn0.get('fqk_q_pref') is not None:
        # Collect entropy from all layers
        all_ents = {k: [] for k in ['FQK_Q', 'FQK_K', 'FV', 'RQK_Q', 'RQK_K', 'RV']}
        all_vars = {k: [] for k in ['FQK_Q', 'FQK_K', 'FV', 'RQK_Q', 'RQK_K', 'RV']}

        for layer_info in routing_infos:
            attn = layer_info.get('attention', layer_info)
            prefs = {
                'FQK_Q': attn.get('fqk_q_pref'),
                'FQK_K': attn.get('fqk_k_pref'),
                'FV': attn.get('fv_pref'),
                'RQK_Q': attn.get('rqk_q_pref'),
                'RQK_K': attn.get('rqk_k_pref'),
                'RV': attn.get('rv_pref'),
            }
            for k, v in prefs.items():
                if v is not None:
                    all_ents[k].append(calc_entropy_fn(v))
                    all_vars[k].append(calc_var_fn(v))

        # Average across layers
        ents = {k: sum(v)/len(v) if v else 0.0 for k, v in all_ents.items()}
        vars_ = {k: sum(v)/len(v) if v else 0.0 for k, v in all_vars.items()}

        ent_str = f"Ent FQK_Q/K/FV/RQK_Q/K/RV:{ents['FQK_Q']:.0f}/{ents['FQK_K']:.0f}/{ents['FV']:.0f}/{ents['RQK_Q']:.0f}/{ents['RQK_K']:.0f}/{ents['RV']:.0f}"
        var_str = f"TokVar:{vars_['FQK_Q']:.4f}/{vars_['FQK_K']:.4f}/{vars_['FV']:.4f}/{vars_['RQK_Q']:.4f}/{vars_['RQK_K']:.4f}/{vars_['RV']:.4f}"

        # Q/K overlap ratio (use first layer for simplicity)
        def calc_overlap(w_Q, w_K):
            if w_Q is None or w_K is None:
                return 0.0
            overlap = ((w_Q > 0) & (w_K > 0)).float()
            active_Q = (w_Q > 0).float().sum(-1)
            return (overlap.sum(-1) / (active_Q + 1e-8)).mean().item()

        w_FQK_Q = attn0.get('fqk_weights_Q')
        w_FQK_K = attn0.get('fqk_weights_K')
        w_RQK_Q = attn0.get('rqk_weights_Q')
        w_RQK_K = attn0.get('rqk_weights_K')
        overlap_FQK = calc_overlap(w_FQK_Q, w_FQK_K)
        overlap_RQK = calc_overlap(w_RQK_Q, w_RQK_K)
        overlap_str = f"Q/K Overlap FQK/RQK:{overlap_FQK:.2f}/{overlap_RQK:.2f}"

        return {'ent_str': ent_str, 'var_str': var_str, 'overlap_str': overlap_str, 'version': '16.4'}

    # v17: Complete pool separation with new naming (feature_q_pref exists)
    elif attn0.get('feature_q_pref') is not None:
        # Collect entropy from all layers
        all_ents = {k: [] for k in ['FQ', 'FK', 'FV', 'RQ', 'RK', 'RV']}
        all_vars = {k: [] for k in ['FQ', 'FK', 'FV', 'RQ', 'RK', 'RV']}

        for layer_info in routing_infos:
            attn = layer_info.get('attention', layer_info)
            prefs = {
                'FQ': attn.get('feature_q_pref'),
                'FK': attn.get('feature_k_pref'),
                'FV': attn.get('feature_v_pref'),
                'RQ': attn.get('restore_q_pref'),
                'RK': attn.get('restore_k_pref'),
                'RV': attn.get('restore_v_pref'),
            }
            for k, v in prefs.items():
                if v is not None:
                    all_ents[k].append(calc_entropy_fn(v))
                    all_vars[k].append(calc_var_fn(v))

        # Average across layers
        ents = {k: sum(v)/len(v) if v else 0.0 for k, v in all_ents.items()}
        vars_ = {k: sum(v)/len(v) if v else 0.0 for k, v in all_vars.items()}

        ent_str = f"Ent FQ/FK/FV/RQ/RK/RV:{ents['FQ']:.0f}/{ents['FK']:.0f}/{ents['FV']:.0f}/{ents['RQ']:.0f}/{ents['RK']:.0f}/{ents['RV']:.0f}"
        var_str = f"TokVar:{vars_['FQ']:.4f}/{vars_['FK']:.4f}/{vars_['FV']:.4f}/{vars_['RQ']:.4f}/{vars_['RK']:.4f}/{vars_['RV']:.4f}"
        return {'ent_str': ent_str, 'var_str': var_str, 'overlap_str': None, 'version': '17'}

    # v16.3: Complete pool separation (fq_pref exists - legacy naming)
    elif attn0.get('fq_pref') is not None:
        # Collect entropy from all layers
        all_ents = {k: [] for k in ['FQ', 'FK', 'FV', 'RQ', 'RK', 'RV']}
        all_vars = {k: [] for k in ['FQ', 'FK', 'FV', 'RQ', 'RK', 'RV']}

        for layer_info in routing_infos:
            attn = layer_info.get('attention', layer_info)
            prefs = {
                'FQ': attn.get('fq_pref'),
                'FK': attn.get('fk_pref'),
                'FV': attn.get('fv_pref'),
                'RQ': attn.get('rq_pref'),
                'RK': attn.get('rk_pref'),
                'RV': attn.get('rv_pref'),
            }
            for k, v in prefs.items():
                if v is not None:
                    all_ents[k].append(calc_entropy_fn(v))
                    all_vars[k].append(calc_var_fn(v))

        # Average across layers
        ents = {k: sum(v)/len(v) if v else 0.0 for k, v in all_ents.items()}
        vars_ = {k: sum(v)/len(v) if v else 0.0 for k, v in all_vars.items()}

        ent_str = f"Ent FQ/FK/FV/RQ/RK/RV:{ents['FQ']:.0f}/{ents['FK']:.0f}/{ents['FV']:.0f}/{ents['RQ']:.0f}/{ents['RK']:.0f}/{ents['RV']:.0f}"
        var_str = f"TokVar:{vars_['FQ']:.4f}/{vars_['FK']:.4f}/{vars_['FV']:.4f}/{vars_['RQ']:.4f}/{vars_['RK']:.4f}/{vars_['RV']:.4f}"
        return {'ent_str': ent_str, 'var_str': var_str, 'overlap_str': None, 'version': '16.3'}

    # v16.2: Q/K projection separation (feature_r_q_pref exists)
    elif attn0.get('feature_r_q_pref') is not None:
        # Collect entropy from all layers
        all_ents = {k: [] for k in ['FRQ', 'FRK', 'FV', 'RQ', 'RK', 'V']}
        all_vars = {k: [] for k in ['FRQ', 'FRK', 'FV', 'RQ', 'RK', 'V']}

        for layer_info in routing_infos:
            attn = layer_info.get('attention', layer_info)
            prefs = {
                'FRQ': attn.get('feature_r_q_pref'),
                'FRK': attn.get('feature_r_k_pref'),
                'FV': attn.get('feature_v_pref'),
                'RQ': attn.get('relational_q_pref'),
                'RK': attn.get('relational_k_pref'),
                'V': attn.get('value_pref'),
            }
            for k, v in prefs.items():
                if v is not None:
                    all_ents[k].append(calc_entropy_fn(v))
                    all_vars[k].append(calc_var_fn(v))

        # Average across layers
        ents = {k: sum(v)/len(v) if v else 0.0 for k, v in all_ents.items()}
        vars_ = {k: sum(v)/len(v) if v else 0.0 for k, v in all_vars.items()}

        ent_str = f"Ent FRQ/FRK/FV/RQ/RK/V:{ents['FRQ']:.0f}/{ents['FRK']:.0f}/{ents['FV']:.0f}/{ents['RQ']:.0f}/{ents['RK']:.0f}/{ents['V']:.0f}"
        var_str = f"TokVar:{vars_['FRQ']:.4f}/{vars_['FRK']:.4f}/{vars_['FV']:.4f}/{vars_['RQ']:.4f}/{vars_['RK']:.4f}/{vars_['V']:.4f}"

        # Q/K overlap ratio (use first layer)
        def calc_overlap(w_Q, w_K):
            if w_Q is None or w_K is None:
                return 0.0
            overlap = ((w_Q > 0) & (w_K > 0)).float()
            active_Q = (w_Q > 0).float().sum(-1)
            return (overlap.sum(-1) / (active_Q + 1e-8)).mean().item()

        w_FRQ = attn0.get('feature_r_weights_Q')
        w_FRK = attn0.get('feature_r_weights_K')
        w_RQ = attn0.get('relational_weights_Q')
        w_RK = attn0.get('relational_weights_K')
        overlap_FR = calc_overlap(w_FRQ, w_FRK)
        overlap_R = calc_overlap(w_RQ, w_RK)
        overlap_str = f"Q/K Overlap FR/R:{overlap_FR:.2f}/{overlap_R:.2f}"

        return {'ent_str': ent_str, 'var_str': var_str, 'overlap_str': overlap_str, 'version': '16.2'}

    # v16.0/16.1: Shared Q/K (feature_r_pref exists)
    elif attn0.get('feature_r_pref') is not None:
        # Collect entropy from all layers
        all_ents = {k: [] for k in ['FR', 'FV', 'RQ', 'RK', 'V']}
        all_vars = {k: [] for k in ['FR', 'FV', 'RQ', 'RK', 'V']}

        for layer_info in routing_infos:
            attn = layer_info.get('attention', layer_info)
            prefs = {
                'FR': attn.get('feature_r_pref'),
                'FV': attn.get('feature_v_pref'),
                'RQ': attn.get('relational_q_pref'),
                'RK': attn.get('relational_k_pref'),
                'V': attn.get('value_pref'),
            }
            for k, v in prefs.items():
                if v is not None:
                    all_ents[k].append(calc_entropy_fn(v))
                    all_vars[k].append(calc_var_fn(v))

        # Average across layers
        ents = {k: sum(v)/len(v) if v else 0.0 for k, v in all_ents.items()}
        vars_ = {k: sum(v)/len(v) if v else 0.0 for k, v in all_vars.items()}

        ent_str = f"Ent FR/FV/RQ/RK/V:{ents['FR']:.0f}/{ents['FV']:.0f}/{ents['RQ']:.0f}/{ents['RK']:.0f}/{ents['V']:.0f}"
        var_str = f"TokVar:{vars_['FR']:.4f}/{vars_['FV']:.4f}/{vars_['RQ']:.4f}/{vars_['RK']:.4f}/{vars_['V']:.4f}"
        return {'ent_str': ent_str, 'var_str': var_str, 'overlap_str': None, 'version': '16.0'}

    # Unknown format
    else:
        return {'ent_str': "Ent: N/A", 'var_str': "TokVar: N/A", 'overlap_str': None, 'version': 'unknown'}
