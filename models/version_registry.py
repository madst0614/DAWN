"""
DAWN Model Version Registry

v16.0: Split Feature R/V (rank matrix) - Feature_R/V separate compression
v16.1: Split Feature R/V + Langevin Excitability (adaptive dead neuron recovery)

To add a new version:
1. Add entry to VERSION_REGISTRY below (with display_info lambda)
2. Create model file in models/model_vX_Y.py
3. Update models/__init__.py
4. Update scripts/train.py if router/routing_info structure changed
5. Create config in configs/train_config_vX_Y.yaml
"""

from typing import Dict, Any, List


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
