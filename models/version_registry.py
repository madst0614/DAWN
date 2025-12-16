"""
DAWN Model Version Registry - Single Source of Truth

Supported Versions:
  v17.2: Feature QK Unified + Restore Q/K Separate (recommended)
  v17.1: Q/K Shared Pool + Knowledge Feature-Restore
  baseline: Vanilla Transformer for comparison
"""

from typing import Dict, Any, List
import torch


VERSION_REGISTRY = {
    "17.2": {
        "description": "Feature QK Unified + Restore Q/K Separate",
        "aliases": ["172"],
        "module": "model_v17_2",
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
            f"DAWN v17.2: Feature QK Unified + Restore Q/K Separate",
            f"  rank={args.get('rank', args.get('basis_rank'))}, knowledge_rank={args.get('knowledge_rank', 128)}",
            f"  F-QK: {args.get('n_feature_qk')} (top-k={args.get('top_k_feature_qk', 8)}) - unified Q/K",
            f"  F-V: {args.get('n_feature_v')} (top-k={args.get('top_k_feature_v', 3)})",
            f"  R-QK: {args.get('n_restore_qk')} (top-k={args.get('top_k_restore_qk', 8)}) - separate Q/K",
            f"  R-V: {args.get('n_restore_v')} (top-k={args.get('top_k_restore_v', 3)})",
            f"  F-Know: {args.get('n_feature_know')} (k={args.get('top_k_feature_know', 4)}), R-Know: {args.get('n_restore_know')} (k={args.get('top_k_restore_know', 4)})",
        ],
    },

    "17.1": {
        "description": "Q/K Shared Pool + Knowledge Feature-Restore",
        "aliases": ["17", "171", "17.0"],
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
            f"  F-QK: {args.get('n_feature_qk')} (top-k={args.get('top_k_feature_qk', 8)}) - Q/K shared pool",
            f"  F-V: {args.get('n_feature_v')} (top-k={args.get('top_k_feature_v', 3)})",
            f"  R-QK: {args.get('n_restore_qk')} (top-k={args.get('top_k_restore_qk', 8)}) - Q/K shared pool",
            f"  R-V: {args.get('n_restore_v')} (top-k={args.get('top_k_restore_v', 3)})",
            f"  F-Know: {args.get('n_feature_know')} (k={args.get('top_k_feature_know', 4)}), R-Know: {args.get('n_restore_know')} (k={args.get('top_k_restore_know', 4)})",
        ],
    },

    "baseline": {
        "description": "Vanilla Transformer Baseline",
        "aliases": ["vanilla", "base"],
        "module": "baseline_transformer",
        "required_params": [
            "d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len",
        ],
        "optional_params": {
            "d_ff": None,
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
    """Load model params from config dict to args object."""
    all_params = {}

    for version_info in VERSION_REGISTRY.values():
        for param in version_info.get('required_params', []):
            if param not in all_params:
                all_params[param] = None
        for param, default in version_info.get('optional_params', {}).items():
            if param not in all_params:
                all_params[param] = default

    special_mappings = {
        'rank': 'basis_rank',
    }

    for param, default in all_params.items():
        if param in ('vocab_size',):
            continue

        args_attr = special_mappings.get(param, param)
        current_value = getattr(args, args_attr, default)
        value = config.get(param, current_value)
        setattr(args, args_attr, value)

        if args_attr == 'basis_rank':
            setattr(args, 'rank', value)


def build_args_config(args, vocab_size: int) -> Dict[str, Any]:
    """Build config dict from args object."""
    all_params = {}

    for version_info in VERSION_REGISTRY.values():
        for param in version_info.get('required_params', []):
            if param not in all_params:
                all_params[param] = None
        for param, default in version_info.get('optional_params', {}).items():
            if param not in all_params:
                all_params[param] = default

    config = {'vocab_size': vocab_size}

    special_mappings = {
        'basis_rank': 'rank',
    }

    for param, default in all_params.items():
        args_attr = None
        for args_name, config_name in special_mappings.items():
            if config_name == param:
                args_attr = args_name
                break

        if args_attr is None:
            args_attr = param

        if hasattr(args, args_attr):
            value = getattr(args, args_attr)
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
    Extract routing log info for v17.1/v17.2.
    Computes AVERAGE entropy across all layers.
    """
    if isinstance(routing_infos, dict):
        routing_infos = [routing_infos]

    attn0 = routing_infos[0].get('attention', routing_infos[0])

    # v17.2: Feature QK Unified (fqk_pref exists, fqk_q_pref does not)
    if attn0.get('fqk_pref') is not None:
        all_ents = {k: [] for k in ['FQK', 'FV', 'RQK_Q', 'RQK_K', 'RV']}
        all_vars = {k: [] for k in ['FQK', 'FV', 'RQK_Q', 'RQK_K', 'RV']}

        for layer_info in routing_infos:
            attn = layer_info.get('attention', layer_info)
            prefs = {
                'FQK': attn.get('fqk_pref'),
                'FV': attn.get('fv_pref'),
                'RQK_Q': attn.get('rqk_q_pref'),
                'RQK_K': attn.get('rqk_k_pref'),
                'RV': attn.get('rv_pref'),
            }
            for k, v in prefs.items():
                if v is not None:
                    all_ents[k].append(calc_entropy_fn(v))
                    all_vars[k].append(calc_var_fn(v))

        ents = {k: sum(v)/len(v) if v else 0.0 for k, v in all_ents.items()}
        vars_ = {k: sum(v)/len(v) if v else 0.0 for k, v in all_vars.items()}

        ent_str = f"Ent F-QK/F-V/R-QK_Q/K/R-V:{ents['FQK']:.0f}/{ents['FV']:.0f}/{ents['RQK_Q']:.0f}/{ents['RQK_K']:.0f}/{ents['RV']:.0f}"
        var_str = f"TokVar:{vars_['FQK']:.4f}/{vars_['FV']:.4f}/{vars_['RQK_Q']:.4f}/{vars_['RQK_K']:.4f}/{vars_['RV']:.4f}"

        def calc_overlap(w_Q, w_K):
            if w_Q is None or w_K is None:
                return 0.0
            overlap = ((w_Q > 0) & (w_K > 0)).float()
            active_Q = (w_Q > 0).float().sum(-1)
            return (overlap.sum(-1) / (active_Q + 1e-8)).mean().item()

        w_RQK_Q = attn0.get('rqk_weights_Q')
        w_RQK_K = attn0.get('rqk_weights_K')
        overlap_RQK = calc_overlap(w_RQK_Q, w_RQK_K)
        overlap_str = f"Q/K Overlap R-QK:{overlap_RQK:.2f}"

        return {'ent_str': ent_str, 'var_str': var_str, 'overlap_str': overlap_str, 'version': '17.2'}

    # v17.1: Shared Pool + Separate Routing (fqk_q_pref exists)
    elif attn0.get('fqk_q_pref') is not None:
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

        ents = {k: sum(v)/len(v) if v else 0.0 for k, v in all_ents.items()}
        vars_ = {k: sum(v)/len(v) if v else 0.0 for k, v in all_vars.items()}

        ent_str = f"Ent F-QK_Q/K/F-V/R-QK_Q/K/R-V:{ents['FQK_Q']:.0f}/{ents['FQK_K']:.0f}/{ents['FV']:.0f}/{ents['RQK_Q']:.0f}/{ents['RQK_K']:.0f}/{ents['RV']:.0f}"
        var_str = f"TokVar:{vars_['FQK_Q']:.4f}/{vars_['FQK_K']:.4f}/{vars_['FV']:.4f}/{vars_['RQK_Q']:.4f}/{vars_['RQK_K']:.4f}/{vars_['RV']:.4f}"

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
        overlap_str = f"Q/K Overlap F-QK/R-QK:{overlap_FQK:.2f}/{overlap_RQK:.2f}"

        return {'ent_str': ent_str, 'var_str': var_str, 'overlap_str': overlap_str, 'version': '17.1'}

    else:
        return {'ent_str': "Ent: N/A", 'var_str': "TokVar: N/A", 'overlap_str': None, 'version': 'unknown'}
