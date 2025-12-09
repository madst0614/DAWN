"""
DAWN Model Version Registry

v10.0: Simplified Compress/Expand Architecture (Soft Routing, 3 compressors for Q/K/V)
v13.0: Final Architecture (Selective SSM + Context + Top-k + FlashAttention)
v13.1: Separate QK/V Expand Pools (Q/K share, V separate)
v13.2: Unified Neuron Router (all neurons in same embedding space)
v14.0: FRVK Architecture (Feature-Relational-Value-Knowledge) with SAR
v15.0: 2-Stage Hierarchical Knowledge Retrieval (x→router→coarse, x→proj_q→fine)
v16.0: Split Feature R/V (rank matrix, v15-based) - Feature_R/V separate compression
v17.0: Full Vector Neurons + Full Soft Selection (vector-based, train & inference both use soft)

To add a new version:
1. Add entry to VERSION_REGISTRY below (with display_info lambda)
2. Create model file in models/model_vX_Y.py
   - Add get_model_info() method returning list of log lines (v14+)
3. Update models/__init__.py:
   - Add import statement
   - Add to __all__ list
   - Add elif in create_model_by_version()
4. Update scripts/train.py (if router/routing_info structure changed):
   - _get_router_log_lines(): Add hasattr() branch for new router attributes
   - Training loop: Update routing_info parsing if keys changed
   - model_kwargs section: Add version to appropriate branch
   - Example: v14 added usage_ema_feature, v15 removed memory routing
5. Create config in configs/train_config_vX_Y.yaml
6. Add version description to this docstring
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
    "13.0": {
        "description": "Final Architecture (Selective SSM + Context + Top-k + FlashAttention)",
        "aliases": ["13", "130"],
        "module": "model_v13",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_compress", "n_expand", "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_compress": 8,
            "top_k_expand": 4,
            "gradient_checkpointing": False,
        },
        "display_info": lambda args: [
            f"DAWN v13.0 Final: rank={args.get('rank', args.get('basis_rank'))}",
            f"  CompressNeurons: {args.get('n_compress')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))}",
            f"  expand_neurons_pool: {args.get('n_expand')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')}",
            f"  Selective SSM: state_dim={args.get('state_dim', 64)} [A + W_delta + W_B + context + importance]",
            f"  Top-k Compress: {args.get('top_k_compress', 8)}/{args.get('n_compress')}",
            f"  Top-k Expand: {args.get('top_k_expand', 4)}/{args.get('n_expand')}",
            f"  Context Enhancement: enabled",
            f"  FlashAttention: enabled",
            f"  Gradient Checkpointing: {args.get('gradient_checkpointing', False)}",
            f"  Architecture: Selective SSM → Context → Top-k Routers → FlashAttn",
            f"  Attention: d_model space (d_head={args.get('d_model')}//{args.get('n_heads')})",
            f"  KnowledgeNeurons:",
            f"    - K: {args.get('n_knowledge')} × {args.get('knowledge_rank', args.get('rank', args.get('basis_rank')))}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - top-k: {args.get('knowledge_k')}",
        ],
    },
    "13.1": {
        "description": "Separate QK/V Expand Pools (Q/K share, V separate)",
        "aliases": ["131"],
        "module": "model_v13_1",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_compress", "n_expand_QK", "n_expand_V", "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_compress": 8,
            "top_k_QK": 4,
            "top_k_V": 6,
            "gradient_checkpointing": False,
        },
        "display_info": lambda args: [
            f"DAWN v13.1: rank={args.get('rank', args.get('basis_rank'))} (QK/V separated)",
            f"  CompressNeurons: {args.get('n_compress')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))}",
            f"  expand_neurons_QK: {args.get('n_expand_QK')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (Q/K shared)",
            f"  expand_neurons_V: {args.get('n_expand_V')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (V separate)",
            f"  Selective SSM: state_dim={args.get('state_dim', 64)}",
            f"  Top-k Compress: {args.get('top_k_compress', 8)}/{args.get('n_compress')}",
            f"  Top-k QK: {args.get('top_k_QK', 4)}/{args.get('n_expand_QK')}",
            f"  Top-k V: {args.get('top_k_V', 6)}/{args.get('n_expand_V')}",
            f"  Architecture: Selective SSM → Context → QK/V Routers → FlashAttn",
            f"  Attention: d_model space (d_head={args.get('d_model')}//{args.get('n_heads')})",
            f"  KnowledgeNeurons:",
            f"    - K: {args.get('n_knowledge')} × {args.get('knowledge_rank', args.get('rank', args.get('basis_rank')))}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - top-k: {args.get('knowledge_k')}",
        ],
    },
    "13.2": {
        "description": "Unified Neuron Router (all neurons in same embedding space)",
        "aliases": ["132"],
        "module": "model_v13_2",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_compress", "n_expand_QK", "n_expand_V", "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_compress": 8,
            "top_k_QK": 4,
            "top_k_V": 6,
            "d_space": 64,
            "gradient_checkpointing": False,
        },
        "display_info": lambda args: [
            f"DAWN v13.2: rank={args.get('rank', args.get('basis_rank'))} (Unified Router)",
            f"  CompressNeurons: {args.get('n_compress')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))}",
            f"  expand_neurons_QK: {args.get('n_expand_QK')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (Q/K pool)",
            f"  expand_neurons_V: {args.get('n_expand_V')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (V pool)",
            f"  Unified Router: d_space={args.get('d_space', 64)}",
            f"  Selective SSM: state_dim={args.get('state_dim', 64)}",
            f"  Top-k Compress: {args.get('top_k_compress', 8)}/{args.get('n_compress')}",
            f"  Top-k QK: {args.get('top_k_QK', 4)}/{args.get('n_expand_QK')}",
            f"  Top-k V: {args.get('top_k_V', 6)}/{args.get('n_expand_V')}",
            f"  Architecture: Selective SSM → Context → Unified Router → FlashAttn",
            f"  Attention: d_model space (d_head={args.get('d_model')}//{args.get('n_heads')})",
            f"  KnowledgeNeurons:",
            f"    - K: {args.get('n_knowledge')} × {args.get('knowledge_rank', args.get('rank', args.get('basis_rank')))}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - top-k: {args.get('knowledge_k')}",
        ],
    },
    "14.0": {
        "description": "FRVK Architecture (Feature-Relational-Value-Knowledge) with SAR",
        "aliases": ["14", "140"],
        "module": "model_v14",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature", "n_relational", "n_value", "n_knowledge", "knowledge_k", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_feature": 8,
            "top_k_relational": 4,
            "top_k_value": 6,
            "d_space": 64,
            "gradient_checkpointing": False,
        },
        "display_info": lambda args: [
            f"DAWN v14: rank={args.get('rank', args.get('basis_rank'))} (FRVK Architecture)",
            f"  FeatureNeurons (F): {args.get('n_feature')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))}",
            f"  RelationalNeurons (R): {args.get('n_relational')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (Q/K pool)",
            f"  ValueNeurons (V): {args.get('n_value')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (V pool)",
            f"  Unified Router: d_space={args.get('d_space', 64)} + SAR (Synaptic Activation Regulation)",
            f"  Selective SSM: state_dim={args.get('state_dim', 64)}",
            f"  Top-k Feature: {args.get('top_k_feature', 8)}/{args.get('n_feature')}",
            f"  Top-k Relational: {args.get('top_k_relational', 4)}/{args.get('n_relational')}",
            f"  Top-k Value: {args.get('top_k_value', 6)}/{args.get('n_value')}",
            f"  Architecture: Mamba SSM → Context → Unified Router (SAR) → FlashAttn",
            f"  Attention: d_model space (d_head={args.get('d_model')}//{args.get('n_heads')})",
            f"  KnowledgeNeurons (K):",
            f"    - K: {args.get('n_knowledge')} × {args.get('knowledge_rank', args.get('rank', args.get('basis_rank')))}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - top-k: {args.get('knowledge_k')}",
        ],
    },
    "15.0": {
        "description": "2-Stage Hierarchical Knowledge Retrieval (x→router→coarse, x→proj_q→fine)",
        "aliases": ["15", "150"],
        "module": "model_v15",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature", "n_relational", "n_value", "n_knowledge", "rank",
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_feature": 8,
            "top_k_relational": 4,
            "top_k_value": 6,
            "d_space": 64,
            "coarse_k": 20,
            "fine_k": 10,
            "knowledge_rank": 128,
            "gradient_checkpointing": False,
        },
        "display_info": lambda args: [
            f"DAWN v15: rank={args.get('rank', args.get('basis_rank'))} (2-Stage Knowledge Retrieval)",
            f"  FeatureNeurons (F): {args.get('n_feature')} × {args.get('d_model')} × {args.get('rank', args.get('basis_rank'))} [Attn only]",
            f"  RelationalNeurons (R): {args.get('n_relational')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (Q/K pool)",
            f"  ValueNeurons (V): {args.get('n_value')} × {args.get('rank', args.get('basis_rank'))} × {args.get('d_model')} (V pool)",
            f"  Unified Router: d_space={args.get('d_space', 64)} + SAR + Knowledge",
            f"  Selective SSM: state_dim={args.get('state_dim', 64)}",
            f"  Top-k Feature: {args.get('top_k_feature', 8)}/{args.get('n_feature')}",
            f"  Top-k Relational: {args.get('top_k_relational', 4)}/{args.get('n_relational')}",
            f"  Top-k Value: {args.get('top_k_value', 6)}/{args.get('n_value')}",
            f"  Architecture: Mamba SSM → Context → Unified Router (SAR) → FlashAttn",
            f"  Memory: 2-stage (x→router→coarse, x→proj_q→fine)",
            f"  KnowledgeNeurons (K):",
            f"    - K: {args.get('n_knowledge')} × {args.get('knowledge_rank', 128)}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - coarse_k: {args.get('coarse_k', 20)} → fine_k: {args.get('fine_k', 10)}",
        ],
    },
    "16.0": {
        "description": "Split Feature QK/V (rank matrix, v15-based)",
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
    "17.0": {
        "description": "Full Vector Neurons + Full Soft Selection",
        "aliases": ["17", "170"],
        "module": "model_v17",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
            "n_feature", "n_relational", "n_value", "n_knowledge",
            # Note: NO rank - all neurons are vectors [n, d_model]
        ],
        "optional_params": {
            "dropout": 0.1,
            "state_dim": 64,
            "top_k_feature": 64,      # kept for compatibility, not used
            "top_k_relational": 64,   # kept for compatibility, not used
            "top_k_value": 32,        # kept for compatibility, not used
            "d_space": 64,
            "coarse_k": 20,
            "fine_k": 10,
            "knowledge_rank": 128,
            "temperature": 1.0,       # soft selection sharpness
            "router_dropout": 0.1,
            "gradient_checkpointing": False,
            "use_ssm_context": True,
        },
        "display_info": lambda args: [
            f"DAWN v17: Full Vector Neurons + Full Soft Selection",
            f"  temperature={args.get('temperature', 1.0)}",
            f"  Selection: FULL SOFT (train & inference, all neurons via softmax)",
            f"  Feature: {args.get('n_feature')} × {args.get('d_model')} (SHARED QK/V)",
            f"  Relational: {args.get('n_relational')} × {args.get('d_model')} (SHARED Q/K)",
            f"  Value: {args.get('n_value')} × {args.get('d_model')}",
            f"  Unified Router: d_space={args.get('d_space', 64)} + Excitability (SAR)",
            f"  Selective SSM: state_dim={args.get('state_dim', 64)}",
            f"  Architecture: Mamba SSM → Context → Full Soft Router → Vector Neurons → FlashAttn",
            f"  Memory: 2-stage (x→router→coarse, x→encoder→fine)",
            f"  KnowledgeNeurons (K):",
            f"    - K: {args.get('n_knowledge')} × {args.get('knowledge_rank', 128)}",
            f"    - V: {args.get('n_knowledge')} × {args.get('d_model')}",
            f"    - coarse_k: {args.get('coarse_k', 20)} → fine_k: {args.get('fine_k', 10)}",
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
