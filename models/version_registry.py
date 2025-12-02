"""
DAWN Model Version Registry

v10.0: Simplified Compress/Expand Architecture (Soft Routing, 3 compressors for Q/K/V)
v11.0: Unified Compression (1 compressor + expand_Q/K/V, d_model attention)
v12.0: SSM-guided Shared QKV (SSM → importance → shared compress → Q/K/V, d_model attention)
v12.1: SSM-guided Shared Neurons (v10 based, rank attention, neuron compress/expand)
v12.2: SSM-guided Dynamic Compress/Expand (shared neuron_weights, d_model attention)
v12.3: SSM-guided Shared Expand Pool (n_expand for Q/K/V, separate routers)

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
        "description": "d_model Attention (compress→expand→d_model attention)",
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
            f"  CompressNeurons: {args.get('n_compress')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (Q/K/V shared)",
            f"  ExpandNeurons: {args.get('n_expand')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')}",
            f"  Architecture: compressor_Q/K/V → expand_Q/K/V → d_model attention",
            f"  Attention: d_model space (d_head={args.get('d_model')}//{args.get('n_heads')})",
            f"  KnowledgeNeurons:",
            f"    - K: {args.get('n_knowledge')} × {args.get('knowledge_rank', args.get('rank', args.get('basis_rank')))}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - top-k: {args.get('knowledge_k')}",
        ],
    },
    "12.0": {
        "description": "SSM-guided Shared QKV (SSM → importance → shared compress → Q/K/V)",
        "aliases": ["12", "120"],
        "module": "model_v12",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_compress", "n_expand", "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
        },
        "display_info": lambda args: [
            f"SharedNeurons (v12.0): rank={args.get('rank', args.get('basis_rank'))}",
            f"  CompressNeurons: {args.get('n_compress')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (shared via SSM)",
            f"  ExpandNeurons: {args.get('n_expand')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')}",
            f"  SSM: state_dim={args.get('state_dim', 64)}",
            f"  Architecture: SSM → importance × neuron_pref → shared compress → Q/K/V",
            f"  Attention: d_model space (d_head={args.get('d_model')}//{args.get('n_heads')})",
            f"  KnowledgeNeurons:",
            f"    - K: {args.get('n_knowledge')} × {args.get('knowledge_rank', args.get('rank', args.get('basis_rank')))}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - top-k: {args.get('knowledge_k')}",
        ],
    },
    "12.1": {
        "description": "SSM-guided Shared Neurons (v10 based, rank attention, neuron compress/expand)",
        "aliases": ["121"],
        "module": "model_v12_1",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_compress", "n_expand", "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
        },
        "display_info": lambda args: [
            f"SharedNeurons (v12.1): rank={args.get('rank', args.get('basis_rank'))} (v10 based)",
            f"  CompressNeurons: {args.get('n_compress')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (SSM shared)",
            f"  ExpandNeurons: {args.get('n_expand')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (SSM shared)",
            f"  SSM: state_dim={args.get('state_dim', 64)}",
            f"  Architecture: SSM → importance × pref → shared compress/expand",
            f"  Attention: rank space (d_head={args.get('rank', args.get('basis_rank'))}//{args.get('n_heads')})",
            f"  KnowledgeNeurons:",
            f"    - K: {args.get('n_knowledge')} × {args.get('knowledge_rank', args.get('rank', args.get('basis_rank')))}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - top-k: {args.get('knowledge_k')}",
        ],
    },
    "12.2": {
        "description": "SSM-guided Dynamic Compress/Expand (shared neuron_weights, d_model attention)",
        "aliases": ["122"],
        "module": "model_v12_2",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_compress", "n_expand", "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
        },
        "display_info": lambda args: [
            f"SharedNeurons (v12.2): rank={args.get('rank', args.get('basis_rank'))} (dynamic expand)",
            f"  CompressNeurons: {args.get('n_compress')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (SSM shared)",
            f"  ExpandNeurons_Q/K/V: {args.get('n_compress')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (per-layer)",
            f"  SSM: state_dim={args.get('state_dim', 64)}",
            f"  Architecture: SSM → importance × pref → shared neuron_weights → compress & expand_Q/K/V",
            f"  Attention: d_model space (d_head={args.get('d_model')}//{args.get('n_heads')})",
            f"  KnowledgeNeurons:",
            f"    - K: {args.get('n_knowledge')} × {args.get('knowledge_rank', args.get('rank', args.get('basis_rank')))}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - top-k: {args.get('knowledge_k')}",
        ],
    },
    "12.3": {
        "description": "SSM-guided Shared Expand Pool (1 pool, 3 routers for Q/K/V)",
        "aliases": ["123"],
        "module": "model_v12_3",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_compress", "n_expand", "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
        },
        "display_info": lambda args: [
            f"SharedNeurons (v12.3): rank={args.get('rank', args.get('basis_rank'))} (shared expand pool)",
            f"  CompressNeurons: {args.get('n_compress')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (SSM shared)",
            f"  expand_neurons_pool: {args.get('n_expand')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (1 shared pool for Q/K/V)",
            f"  SSM: state_dim={args.get('state_dim', 64)}",
            f"  Architecture: SSM → compress_router + expand_router_Q/K/V → shared pool",
            f"  Key: 다른 가중치, 같은 풀 → Q/K/V 생성",
            f"  Attention: d_model space (d_head={args.get('d_model')}//{args.get('n_heads')})",
            f"  Parameter savings: 1 pool vs 3 pools (~0.48M saved)",
            f"  KnowledgeNeurons:",
            f"    - K: {args.get('n_knowledge')} × {args.get('knowledge_rank', args.get('rank', args.get('basis_rank')))}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - top-k: {args.get('knowledge_k')}",
        ],
    },
    "12.4": {
        "description": "Config-based Dynamic O Experiments (dynamic_O, low_rank_O options)",
        "aliases": ["124"],
        "module": "model_v12_4",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_compress", "n_expand", "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "dynamic_O": False,
            "n_O_expand": 12,
            "low_rank_O": False,
            "O_rank": 64,
        },
        "display_info": lambda args: [
            f"SharedNeurons (v12.4): rank={args.get('rank', args.get('basis_rank'))} (configurable O)",
            f"  Config: dynamic_O={args.get('dynamic_O', False)}, low_rank_O={args.get('low_rank_O', False)}, n_heads={args.get('n_heads')}",
            f"  CompressNeurons: {args.get('n_compress')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (SSM shared)",
            f"  expand_neurons_pool: {args.get('n_expand')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (QKV pool)",
            f"  O_compress_pool: {args.get('n_O_expand', 12)} × {args.get('d_model')} × {args.get('O_rank', 64)}" if args.get('dynamic_O', False) and args.get('low_rank_O', False) else (f"  O_pool: {args.get('n_O_expand', 12)} × {args.get('d_model')} × {args.get('d_model')}" if args.get('dynamic_O', False) else "  O projection: None (direct output)"),
            f"  O_expand_pool: {args.get('n_O_expand', 12)} × {args.get('O_rank', 64)} × {args.get('d_model')}" if args.get('dynamic_O', False) and args.get('low_rank_O', False) else "",
            f"  SSM: state_dim={args.get('state_dim', 64)}",
            f"  Architecture: SSM → Q/K/V expand → {'low-rank O' if args.get('low_rank_O', False) else ('full-rank O' if args.get('dynamic_O', False) else 'no O proj')}",
            f"  Attention: d_model space (d_head={args.get('d_model')}//{args.get('n_heads')})",
            f"  KnowledgeNeurons:",
            f"    - K: {args.get('n_knowledge')} × {args.get('knowledge_rank', args.get('rank', args.get('basis_rank')))}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - top-k: {args.get('knowledge_k')}",
        ],
    },
    "12.5": {
        "description": "Global SSM + Global Router (24→1 SSM, 60→5 routers)",
        "aliases": ["125"],
        "module": "model_v12_5",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_compress", "n_expand", "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
        },
        "display_info": lambda args: [
            f"SharedNeurons (v12.5): rank={args.get('rank', args.get('basis_rank'))} (global SSM + routers)",
            f"  CompressNeurons: {args.get('n_compress')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} (shared)",
            f"  expand_neurons_pool: {args.get('n_expand')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (QKV pool)",
            f"  Global SSM: 1 (model level) → importance + context",
            f"  Global Routers: 5 (compress, expand_Q/K/V, memory)",
            f"  Context Enhancement: SSM context added to x",
            f"  SSM: state_dim={args.get('state_dim', 64)}",
            f"  Architecture: Global SSM → importance + context → Global Routers (per-layer x)",
            f"  Attention: d_model space (d_head={args.get('d_model')}//{args.get('n_heads')})",
            f"  Parameter savings: SSM 24→1, Routers 60→5",
            f"  KnowledgeNeurons:",
            f"    - K: {args.get('n_knowledge')} × {args.get('knowledge_rank', args.get('rank', args.get('basis_rank')))}",
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
