"""
DAWN Model Version Registry

v10.0: Simplified Compress/Expand Architecture (Soft Routing, 3 compressors for Q/K/V)
v11.0: Unified Compression (1 compressor + expand_Q/K/V, d_model attention)

To add a new version:
1. Add entry to VERSION_REGISTRY below
2. Create model file in models/model_vX_Y.py
3. Import and register in models/__init__.py
4. Add version handling in scripts/train.py (around line 1415)
5. Create config in configs/train_config_vX_Y.yaml
"""

from typing import Dict, Any, List


VERSION_REGISTRY = {
    "10.0": {
        "description": "Simplified Compress/Expand (No Householder, Q/K/V/M 통합)",
        "aliases": ["10", "100"],
        "module": "model_v10",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_compress", "n_expand", "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
        },
        "display_info": lambda args: [
            f"SharedNeurons (v10.0): rank={args.get('rank', args.get('basis_rank'))}",
            f"  CompressNeurons: {args.get('n_compress')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (Q/K/V/M shared)",
            f"  ExpandNeurons: {args.get('n_expand')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (O shared)",
            f"  KnowledgeNeurons:",
            f"    - K: {args.get('n_knowledge')} × {args.get('rank', args.get('basis_rank'))}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - top-k: {args.get('knowledge_k')}",
        ],
    },
    "11.0": {
        "description": "Unified Compression (1 compressor + expand_Q/K/V, d_model attention)",
        "aliases": ["11", "110"],
        "module": "model_v11",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_compress", "n_expand", "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
        },
        "display_info": lambda args: [
            f"SharedNeurons (v11.0): rank={args.get('rank', args.get('basis_rank'))}",
            f"  CompressNeurons: {args.get('n_compress')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (unified)",
            f"  ExpandNeurons: {args.get('n_expand')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')}",
            f"  Architecture: 1 compressor + expand_Q/K/V/O",
            f"  Attention: d_model space (d_head={args.get('d_model')}//{args.get('n_heads')})",
            f"  KnowledgeNeurons:",
            f"    - K: {args.get('n_knowledge')} × {args.get('rank', args.get('basis_rank'))}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - top-k: {args.get('knowledge_k')}",
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
