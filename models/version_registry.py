"""
DAWN Model Version Registry

Centralized version information and model creation utilities.

=============================================================================
새 버전 추가 체크리스트 (New Version Checklist)
=============================================================================
1. models/model_vXX.py 생성
   - DAWN 클래스 작성 (__version__ = "X.X")
   - SharedNeurons, Compressor, Expander 등 구현

2. models/version_registry.py (이 파일)
   - VERSION_REGISTRY에 새 버전 항목 추가
   - required_params, optional_params 정의
   - display_info 람다 함수 (선택)

3. models/__init__.py
   - import 추가: from . import model_vXX as model_vXX
   - __all__에 추가
   - create_model_by_version()에 분기 추가

4. utils/checkpoint.py
   - VERSION_PARAM_CHANGES에 새 버전의 added/removed 파라미터 추가

5. scripts/train.py (선택)
   - model_kwargs에 새 버전용 파라미터 처리 추가
   - 버전별 print 정보 추가

6. 테스트
   - python -c "from models.model_vXX import DAWN; print(DAWN.__version__)"
=============================================================================
"""

from typing import Dict, Any, List, Optional, Callable


# =============================================================================
# Version Registry
# =============================================================================
# Each version defines:
#   - description: Short description
#   - aliases: Alternative version strings
#   - required_params: Parameters needed for this version
#   - optional_params: Optional parameters with defaults
#   - module: Module name to import from

VERSION_REGISTRY = {
    "9.1": {
        "description": "v9.0 + hard selection + gated reflection",
        "aliases": ["91"],
        "module": "model_v91",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_compress", "n_expand", "n_reflect", "reflect_k",
            "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
        },
        "display_info": lambda args: [
            f"SharedNeurons (v9.1): rank={args.get('rank', args.get('basis_rank'))}",
            f"  CompressNeurons: {args.get('n_compress')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (hard selection)",
            f"  ExpandNeurons: {args.get('n_expand')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (hard selection)",
            f"  ReflectionNeurons (gated):",
            f"    - reflect_d: {args.get('n_reflect')} × {args.get('d_model')}",
            f"    - reflect_r: {args.get('n_reflect')} × {args.get('rank', args.get('basis_rank'))}",
            f"    - Reflect top-k: {args.get('reflect_k')}",
            f"  KnowledgeNeurons:",
            f"    - K: {args.get('n_knowledge')} × {args.get('rank', args.get('basis_rank'))}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - Knowledge top-k: {args.get('knowledge_k')}",
        ],
    },
    "9.0": {
        "description": "CompressNeurons + ExpandNeurons + ReflectionNeurons",
        "aliases": ["9", "90"],
        "module": "model_v9",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_compress", "n_expand", "n_reflect", "reflect_k",
            "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
        },
        "display_info": lambda args: [
            f"SharedNeurons (v9.0): rank={args.get('rank', args.get('basis_rank'))}",
            f"  CompressNeurons: {args.get('n_compress')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))}",
            f"  ExpandNeurons: {args.get('n_expand')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')}",
            f"  ReflectionNeurons:",
            f"    - reflect_d: {args.get('n_reflect')} × {args.get('d_model')}",
            f"    - reflect_r: {args.get('n_reflect')} × {args.get('rank', args.get('basis_rank'))}",
            f"    - Reflect top-k: {args.get('reflect_k')}",
            f"  KnowledgeNeurons:",
            f"    - K: {args.get('n_knowledge')} × {args.get('rank', args.get('basis_rank'))}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - Knowledge top-k: {args.get('knowledge_k')}",
        ],
    },
    "8.3": {
        "description": "SharedNeurons + NeuronMemory (QK/V/O/M 분리)",
        "aliases": ["83"],
        "module": "model_v8",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_input", "n_process", "n_output", "process_k",
            "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
        },
    },
    "8.2": {
        "description": "SharedNeurons + NeuronMemory (QK/V/O 분리)",
        "aliases": ["82"],
        "module": "model_v8",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_input", "n_process", "n_output", "process_k",
            "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
        },
    },
    "8.1": {
        "description": "SharedNeurons + NeuronMemory (QK/VO 분리)",
        "aliases": ["81"],
        "module": "model_v8",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_input", "n_process", "n_output", "process_k",
            "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
        },
    },
    "8.0": {
        "description": "SharedNeurons + NeuronMemory",
        "aliases": ["8", "80"],
        "module": "model_v8",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_input", "n_process", "n_output", "process_k",
            "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
        },
    },
    "baseline": {
        "description": "Vanilla Transformer (no DAWN)",
        "aliases": [],
        "module": "baseline_transformer",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
        ],
        "optional_params": {
            "dropout": 0.1,
        },
    },
}


def normalize_version(version: str) -> str:
    """
    Normalize version string to canonical form.

    Args:
        version: Version string (e.g., "9", "90", "9.0")

    Returns:
        Canonical version string (e.g., "9.0")
    """
    version = str(version)

    # Check direct match first
    if version in VERSION_REGISTRY:
        return version

    # Check aliases
    for canonical, info in VERSION_REGISTRY.items():
        if version in info.get('aliases', []):
            return canonical

    raise ValueError(f"Unknown version: {version}. Available: {list(VERSION_REGISTRY.keys())}")


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
    """
    Build model kwargs from config, filtering to what the version needs.

    Args:
        version: Model version
        config: Full configuration dict

    Returns:
        Filtered kwargs dict for model creation
    """
    version = normalize_version(version)
    info = VERSION_REGISTRY[version]

    kwargs = {}

    # Add required params
    for param in info.get('required_params', []):
        if param in config:
            kwargs[param] = config[param]

    # Add optional params with defaults
    for param, default in info.get('optional_params', {}).items():
        kwargs[param] = config.get(param, default)

    return kwargs


def print_version_info(version: str, args: Dict[str, Any]) -> None:
    """
    Print version-specific architecture information.

    Args:
        version: Model version
        args: Arguments/config dict
    """
    version = normalize_version(version)
    info = VERSION_REGISTRY.get(version, {})

    print(f"Model version: {version}")
    print(f"Description: {info.get('description', 'N/A')}")

    # Use custom display function if available
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
