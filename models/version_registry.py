"""
DAWN Model Version Registry

v16.0: Split Feature R/V (rank matrix) - Feature_R/V separate compression
v16.1: Split Feature R/V + Langevin Excitability (adaptive dead neuron recovery)
v16.2: Full Q/K Projection Separation - Q/K routing paths separated
v16.3: Complete Q/K/V Pool Separation - FQ/FK/FV, RQ/RK/RV all independent
v16.4: Shared Pool + Separate Routing - v16.3 optimized, Q/K shared pool with separate routing

To add a new version:
1. Add entry to VERSION_REGISTRY below (with display_info lambda)
2. Create model file in models/model_vX_Y.py
3. Update models/__init__.py
4. Create config in configs/train_config_vX_Y.yaml
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


def get_routing_log_info(routing_info: Dict[str, Any], calc_entropy_fn, calc_var_fn) -> Dict[str, Any]:
    """
    Extract routing log info based on routing_info structure.
    Auto-detects version from routing_info keys.

    Args:
        routing_info: Layer 0 routing info from model forward
        calc_entropy_fn: Function to calculate entropy ratio (pref -> float)
        calc_var_fn: Function to calculate token variance (pref -> float)

    Returns:
        Dict with keys: 'ent_str', 'var_str', 'overlap_str', 'version'
    """
    attn = routing_info.get('attention', routing_info)

    # v16.4: Shared Pool + Separate Routing (fqk_q_pref exists)
    if attn.get('fqk_q_pref') is not None:
        prefs = {
            'FQK_Q': attn.get('fqk_q_pref'),
            'FQK_K': attn.get('fqk_k_pref'),
            'FV': attn.get('fv_pref'),
            'RQK_Q': attn.get('rqk_q_pref'),
            'RQK_K': attn.get('rqk_k_pref'),
            'RV': attn.get('rv_pref'),
        }
        ents = {k: calc_entropy_fn(v) for k, v in prefs.items()}
        vars_ = {k: calc_var_fn(v) for k, v in prefs.items()}

        ent_str = f"Ent FQK_Q/K/FV/RQK_Q/K/RV:{ents['FQK_Q']:.0f}/{ents['FQK_K']:.0f}/{ents['FV']:.0f}/{ents['RQK_Q']:.0f}/{ents['RQK_K']:.0f}/{ents['RV']:.0f}"
        var_str = f"TokVar:{vars_['FQK_Q']:.4f}/{vars_['FQK_K']:.4f}/{vars_['FV']:.4f}/{vars_['RQK_Q']:.4f}/{vars_['RQK_K']:.4f}/{vars_['RV']:.4f}"

        # Q/K overlap ratio for shared pools
        def calc_overlap(w_Q, w_K):
            if w_Q is None or w_K is None:
                return 0.0
            overlap = ((w_Q > 0) & (w_K > 0)).float()
            active_Q = (w_Q > 0).float().sum(-1)
            return (overlap.sum(-1) / (active_Q + 1e-8)).mean().item()

        w_FQK_Q = attn.get('fqk_weights_Q')
        w_FQK_K = attn.get('fqk_weights_K')
        w_RQK_Q = attn.get('rqk_weights_Q')
        w_RQK_K = attn.get('rqk_weights_K')
        overlap_FQK = calc_overlap(w_FQK_Q, w_FQK_K)
        overlap_RQK = calc_overlap(w_RQK_Q, w_RQK_K)
        overlap_str = f"Q/K Overlap FQK/RQK:{overlap_FQK:.2f}/{overlap_RQK:.2f}"

        return {'ent_str': ent_str, 'var_str': var_str, 'overlap_str': overlap_str, 'version': '16.4'}

    # v16.3: Complete pool separation (fq_pref exists)
    elif attn.get('fq_pref') is not None:
        prefs = {
            'FQ': attn.get('fq_pref'),
            'FK': attn.get('fk_pref'),
            'FV': attn.get('fv_pref'),
            'RQ': attn.get('rq_pref'),
            'RK': attn.get('rk_pref'),
            'RV': attn.get('rv_pref'),
        }
        ents = {k: calc_entropy_fn(v) for k, v in prefs.items()}
        vars_ = {k: calc_var_fn(v) for k, v in prefs.items()}

        ent_str = f"Ent FQ/FK/FV/RQ/RK/RV:{ents['FQ']:.0f}/{ents['FK']:.0f}/{ents['FV']:.0f}/{ents['RQ']:.0f}/{ents['RK']:.0f}/{ents['RV']:.0f}"
        var_str = f"TokVar:{vars_['FQ']:.4f}/{vars_['FK']:.4f}/{vars_['FV']:.4f}/{vars_['RQ']:.4f}/{vars_['RK']:.4f}/{vars_['RV']:.4f}"
        return {'ent_str': ent_str, 'var_str': var_str, 'overlap_str': None, 'version': '16.3'}

    # v16.2: Q/K projection separation (feature_r_q_pref exists)
    elif attn.get('feature_r_q_pref') is not None:
        prefs = {
            'FRQ': attn.get('feature_r_q_pref'),
            'FRK': attn.get('feature_r_k_pref'),
            'FV': attn.get('feature_v_pref'),
            'RQ': attn.get('relational_q_pref'),
            'RK': attn.get('relational_k_pref'),
            'V': attn.get('value_pref'),
        }
        ents = {k: calc_entropy_fn(v) for k, v in prefs.items()}
        vars_ = {k: calc_var_fn(v) for k, v in prefs.items()}

        ent_str = f"Ent FRQ/FRK/FV/RQ/RK/V:{ents['FRQ']:.0f}/{ents['FRK']:.0f}/{ents['FV']:.0f}/{ents['RQ']:.0f}/{ents['RK']:.0f}/{ents['V']:.0f}"
        var_str = f"TokVar:{vars_['FRQ']:.4f}/{vars_['FRK']:.4f}/{vars_['FV']:.4f}/{vars_['RQ']:.4f}/{vars_['RK']:.4f}/{vars_['V']:.4f}"

        # Q/K overlap ratio
        def calc_overlap(w_Q, w_K):
            if w_Q is None or w_K is None:
                return 0.0
            overlap = ((w_Q > 0) & (w_K > 0)).float()
            active_Q = (w_Q > 0).float().sum(-1)
            return (overlap.sum(-1) / (active_Q + 1e-8)).mean().item()

        w_FRQ = attn.get('feature_r_weights_Q')
        w_FRK = attn.get('feature_r_weights_K')
        w_RQ = attn.get('relational_weights_Q')
        w_RK = attn.get('relational_weights_K')
        overlap_FR = calc_overlap(w_FRQ, w_FRK)
        overlap_R = calc_overlap(w_RQ, w_RK)
        overlap_str = f"Q/K Overlap FR/R:{overlap_FR:.2f}/{overlap_R:.2f}"

        return {'ent_str': ent_str, 'var_str': var_str, 'overlap_str': overlap_str, 'version': '16.2'}

    # v16.0/16.1: Shared Q/K (feature_r_pref exists)
    elif attn.get('feature_r_pref') is not None:
        prefs = {
            'FR': attn.get('feature_r_pref'),
            'FV': attn.get('feature_v_pref'),
            'RQ': attn.get('relational_q_pref'),
            'RK': attn.get('relational_k_pref'),
            'V': attn.get('value_pref'),
        }
        ents = {k: calc_entropy_fn(v) for k, v in prefs.items()}
        vars_ = {k: calc_var_fn(v) for k, v in prefs.items()}

        ent_str = f"Ent FR/FV/RQ/RK/V:{ents['FR']:.0f}/{ents['FV']:.0f}/{ents['RQ']:.0f}/{ents['RK']:.0f}/{ents['V']:.0f}"
        var_str = f"TokVar:{vars_['FR']:.4f}/{vars_['FV']:.4f}/{vars_['RQ']:.4f}/{vars_['RK']:.4f}/{vars_['V']:.4f}"
        return {'ent_str': ent_str, 'var_str': var_str, 'overlap_str': None, 'version': '16.0'}

    # Unknown format
    else:
        return {'ent_str': "Ent: N/A", 'var_str': "TokVar: N/A", 'overlap_str': None, 'version': 'unknown'}
