"""
DAWN Model Version Registry

Centralized version information and model creation utilities.
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
