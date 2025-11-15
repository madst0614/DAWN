"""
DAWN Configuration Adapter

Converts RuntimeConfig to models package format.
"""

from typing import Dict, Any
from .registry import RuntimeConfig


def to_models_config(runtime_config: RuntimeConfig) -> Dict[str, Any]:
    """
    Convert RuntimeConfig to models package format

    Args:
        runtime_config: RuntimeConfig instance

    Returns:
        Config dict compatible with models.DAWN
    """

    # Base config from RuntimeConfig.to_dict()
    config = runtime_config.to_dict()

    # Add models specific configs

    # === Delta Module Config ===
    config['delta_module'] = {
        'num_blocks': runtime_config.refiner.num_blocks,
        'refiner': {
            'attention_dropout': runtime_config.refiner.attention_dropout,
            'ffn_dropout': runtime_config.refiner.ffn_dropout,
            'zero_init_final_layer': runtime_config.refiner.zero_init_final_layer,
            'init_std': runtime_config.refiner.init_std,
            'gate': {
                'temperature_scale': runtime_config.refiner.temperature_scale,
                'init_std': runtime_config.refiner.init_std,
            },
        },
    }

    # === Peer Context Config (Phase 2) ===
    if runtime_config.phase == 2 and runtime_config.peer_prediction:
        config['peer_context'] = {
            'num_heads': runtime_config.model.num_heads,
            'projection_rank': runtime_config.peer_prediction.aspect_dim or (runtime_config.model.hidden_size // 2),
            'dropout': runtime_config.model.dropout,
            'init_std': runtime_config.refiner.init_std,
        }

    # === Integration Config ===
    config['integration'] = {
        'use_context_modulation': True,
        'context_modulation': {
            'hidden_dim': runtime_config.model.hidden_size,
        },
        'dropout': runtime_config.model.dropout,
        'init_std': runtime_config.refiner.init_std,
    }

    # === Expert Integrator Config (Phase 2) ===
    if runtime_config.phase == 2:
        config['expert_integrator'] = {
            'base_expert': runtime_config.expert_names[0],  # Use first expert as base
            'expert_context': {
                'num_heads': runtime_config.model.num_heads,
                'projection_rank': runtime_config.model.hidden_size // 2,
                'dropout': runtime_config.model.dropout,
                'init_std': runtime_config.refiner.init_std,
            },
            'delta_module': {
                'num_blocks': 3,  # Lighter than expert-level
                'num_heads': runtime_config.model.num_heads,
                'intermediate_size': runtime_config.model.hidden_size * 4,
                'dropout': runtime_config.model.dropout,
                'refiner': {
                    'attention_dropout': runtime_config.model.dropout,
                    'ffn_dropout': runtime_config.model.dropout,
                    'zero_init_final_layer': True,
                    'init_std': runtime_config.refiner.init_std,
                    'gate': {
                        'temperature_scale': runtime_config.refiner.temperature_scale,
                        'init_std': runtime_config.refiner.init_std,
                    },
                },
            },
            'integration': {
                'use_context_modulation': True,
                'dropout': runtime_config.model.dropout,
                'init_std': runtime_config.refiner.init_std,
            },
            'gate': {
                'temperature_scale': runtime_config.refiner.temperature_scale,
                'init_std': runtime_config.refiner.init_std,
            },
        }

    # Initialize standard (default init_std)
    config['init_std'] = runtime_config.refiner.init_std

    return config


def get_models_args(runtime_config: RuntimeConfig) -> Dict[str, Any]:
    """
    Get constructor arguments for models.DAWN

    Args:
        runtime_config: RuntimeConfig instance

    Returns:
        Dict with 'config', 'enable_peer_prediction', 'active_experts'
    """
    return {
        'config': to_models_config(runtime_config),
        'enable_peer_prediction': runtime_config.phase == 2,
        'active_experts': runtime_config.expert_names,
    }
