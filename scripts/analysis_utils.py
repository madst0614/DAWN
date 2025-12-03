"""
DAWN Analysis Utilities - Version Agnostic

v7.9, v8.0, v10.x, v12.x, v13.x 모두 지원하는 유틸리티 함수들
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def detect_model_version(checkpoint):
    """체크포인트에서 모델 버전 감지"""
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    # Explicit version
    if 'model_version' in config:
        return str(config['model_version'])

    # Check state dict keys
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    keys = list(state_dict.keys())
    keys_str = ' '.join(keys)

    # v13: has context_proj in SSM
    if 'context_proj' in keys_str and 'global_ssm' in keys_str:
        return "13.0"

    # v12.7/v12.8: has A_log in SSM (Mamba-style)
    if 'A_log' in keys_str and 'global_ssm' in keys_str:
        return "12.7"

    # v12.5/v12.6: has global_ssm with old-style A parameter
    if 'global_ssm' in keys_str:
        return "12.5"

    # v12.3/v12.4: has expand_neurons_pool
    if 'expand_neurons_pool' in keys_str:
        return "12.3"

    # v12.0-v12.2: has SSM but not global_ssm
    if 'ssm' in keys_str.lower():
        return "12.0"

    # v10.x: has shared_neurons without SSM
    if any('shared_neurons' in k for k in keys):
        return "10.0"

    # v8.0: has knowledge_K/V
    if any('knowledge_K' in k for k in keys):
        return "8.0"

    # Default to 7.9
    return "7.9"


def load_model(checkpoint_path, device='cuda'):
    """버전에 맞는 모델 자동 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    version = detect_model_version(checkpoint)
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    print(f"Detected model version: {version}")

    # Use create_model_by_version for v10.x and later
    if version.startswith('10') or version.startswith('12') or version.startswith('13'):
        from models import create_model_by_version

        model_kwargs = {
            'vocab_size': config.get('vocab_size', 30522),
            'd_model': config.get('d_model', 320),
            'n_layers': config.get('n_layers', 4),
            'n_heads': config.get('n_heads', 4),
            'rank': config.get('rank', 64),
            'max_seq_len': config.get('max_seq_len', 128),
            'n_compress': config.get('n_compress', 48),
            'n_expand': config.get('n_expand', 12),
            'n_knowledge': config.get('n_knowledge', 80),
            'knowledge_k': config.get('knowledge_k', 10),
            'dropout': config.get('dropout', 0.1),
        }

        # Add version-specific params
        if version.startswith('12') or version.startswith('13'):
            model_kwargs['state_dim'] = config.get('state_dim', 64)
            if 'knowledge_rank' in config:
                model_kwargs['knowledge_rank'] = config['knowledge_rank']

        # v12.7, v12.8, v13 have top-k sparse routing
        if version in ['12.7', '12.8', '13.0'] or version.startswith('13'):
            model_kwargs['top_k_compress'] = config.get('top_k_compress', 8)
            model_kwargs['top_k_expand'] = config.get('top_k_expand', 4)

        model = create_model_by_version(version, model_kwargs)

    elif version == "8.0":
        from models.model_v8 import DAWN
        model = DAWN(
            vocab_size=config.get('vocab_size', 30522),
            d_model=config.get('d_model', 256),
            n_layers=config.get('n_layers', 4),
            n_heads=config.get('n_heads', 4),
            rank=config.get('rank', config.get('basis_rank', 64)),
            max_seq_len=config.get('max_seq_len', 128),
            n_input=config.get('n_input', 8),
            n_process=config.get('n_process', 32),
            n_output=config.get('n_output', 8),
            process_k=config.get('process_k', 3),
            n_knowledge=config.get('n_knowledge', 64),
            knowledge_k=config.get('knowledge_k', 8),
            dropout=config.get('dropout', 0.1),
        )
    else:  # 7.9
        from models.model_v79 import DAWN
        model = DAWN(
            vocab_size=config.get('vocab_size', 30522),
            d_model=config.get('d_model', 256),
            n_layers=config.get('n_layers', 4),
            n_heads=config.get('n_heads', 4),
            d_ff=config.get('d_ff', 1024),
            max_seq_len=config.get('max_seq_len', 128),
            rank=config.get('rank', config.get('basis_rank', 64)),
            n_input=config.get('n_input', 8),
            n_process=config.get('n_process', 32),
            n_output=config.get('n_output', 8),
            process_k=config.get('process_k', 3),
            dropout=config.get('dropout', 0.1),
            use_soft_selection=config.get('use_soft_selection', True),
        )

    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, version, config


def get_underlying_model(model):
    """torch.compile wrapper 제거"""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def get_routing_info_compat(routing_info, version):
    """
    버전별 routing_info 통일

    Returns:
        compress_weights: [B, n_compress] - compress neuron weights (v12+)
        expand_weights_Q/K/V: [B, n_expand] - expand neuron weights (v12+)
        memory_weights: [B, n_compress] - memory neuron weights (v12+)
        knowledge_indices: [B, S, k] - knowledge retrieval indices
        knowledge_weights: [B, S, k] - knowledge retrieval weights
        # Legacy (v7.9, v8.0):
        process_indices: [B, S, k] - process neuron indices
        input_weights: [B, S, n_input] or None
        output_weights: [B, S, n_output] or None
    """
    # v12.x, v13.x format
    if version.startswith('12') or version.startswith('13'):
        attn_routing = routing_info.get('attention', {})
        mem_routing = routing_info.get('memory', {})

        # v12.7/v13: use dense weights for analysis
        compress_weights = attn_routing.get('compress_weights_dense', attn_routing.get('compress_weights'))
        expand_weights_Q = attn_routing.get('expand_weights_Q_dense', attn_routing.get('expand_weights_Q'))
        expand_weights_K = attn_routing.get('expand_weights_K_dense', attn_routing.get('expand_weights_K'))
        expand_weights_V = attn_routing.get('expand_weights_V_dense', attn_routing.get('expand_weights_V'))
        memory_weights = mem_routing.get('memory_weights_dense', mem_routing.get('memory_weights', mem_routing.get('neuron_weights')))

        return {
            'compress_weights': compress_weights,
            'expand_weights_Q': expand_weights_Q,
            'expand_weights_K': expand_weights_K,
            'expand_weights_V': expand_weights_V,
            'memory_weights': memory_weights,
            'knowledge_indices': mem_routing.get('knowledge_indices'),
            'knowledge_weights': mem_routing.get('knowledge_weights'),
            # Top-k indices (v12.7/v13)
            'compress_topk_idx': attn_routing.get('compress_topk_idx'),
            'expand_topk_idx_Q': attn_routing.get('expand_topk_idx_Q'),
            'expand_topk_idx_K': attn_routing.get('expand_topk_idx_K'),
            'expand_topk_idx_V': attn_routing.get('expand_topk_idx_V'),
            'memory_topk_idx': mem_routing.get('memory_topk_idx'),
            # Importance
            'importance': routing_info.get('importance'),
        }
    elif version.startswith('10'):
        attn_routing = routing_info.get('attention', {})
        mem_routing = routing_info.get('memory', {})

        return {
            'Q_weights': attn_routing.get('Q', {}).get('weights'),
            'K_weights': attn_routing.get('K', {}).get('weights'),
            'V_weights': attn_routing.get('V', {}).get('weights'),
            'O_weights': attn_routing.get('O', {}).get('weights'),
            'memory_weights': mem_routing.get('M', {}).get('weights'),
            'knowledge_indices': mem_routing.get('knowledge_indices'),
            'knowledge_weights': mem_routing.get('knowledge_weights'),
        }
    elif version == "8.0":
        attn_routing = routing_info.get('attention', {})
        mem_routing = routing_info.get('memory', {})

        return {
            'process_indices': attn_routing.get('process_indices'),
            'input_weights': attn_routing.get('input_weights'),
            'output_weights': attn_routing.get('output_weights'),
            'memory_indices': mem_routing.get('top_indices'),  # knowledge retrieval indices
            'memory_weights': mem_routing.get('top_weights'),
        }
    else:  # 7.9
        routing_down = routing_info.get('routing_down', {})
        routing_up = routing_info.get('routing_up', {})

        return {
            'process_indices': routing_down.get('process_indices'),
            'input_weights': routing_down.get('input_weights'),
            'output_weights': routing_up.get('output_weights'),
            'memory_indices': None,
            'memory_weights': None,
        }


def get_neurons(model, version):
    """
    모델에서 뉴런 가져오기

    Returns (v12.x, v13.x):
        compress_neurons: [n_compress, d_model, rank]
        expand_neurons_pool: [n_expand, rank, d_model] (for Q/K/V)
        knowledge_K: [n_knowledge, d_model]
        knowledge_V: [n_knowledge, d_model]

    Returns (v10.x, v8.0):
        input_neurons: [n_input, d_model, rank]
        process_neurons: [n_process, rank]
        output_neurons: [n_output, rank, d_model]
        knowledge_K: [n_knowledge, rank] or None
        knowledge_V: [n_knowledge, d_model] or None
    """
    model = get_underlying_model(model)

    # v12.x, v13.x
    if version.startswith('12') or version.startswith('13'):
        shared = model.shared_neurons
        result = {
            'compress_neurons': shared.compress_neurons.data,
            'knowledge_K': shared.knowledge_K.data if hasattr(shared, 'knowledge_K') else None,
            'knowledge_V': shared.knowledge_V.data if hasattr(shared, 'knowledge_V') else None,
        }
        if hasattr(shared, 'expand_neurons_pool'):
            result['expand_neurons_pool'] = shared.expand_neurons_pool.data
        elif hasattr(shared, 'expand_neurons'):
            result['expand_neurons'] = shared.expand_neurons.data
        return result

    elif version.startswith('10'):
        shared = model.shared_neurons
        result = {
            'compress_neurons': shared.compress_neurons.data if hasattr(shared, 'compress_neurons') else None,
            'expand_neurons': shared.expand_neurons.data if hasattr(shared, 'expand_neurons') else None,
            'knowledge_K': shared.knowledge_K.data if hasattr(shared, 'knowledge_K') else None,
            'knowledge_V': shared.knowledge_V.data if hasattr(shared, 'knowledge_V') else None,
        }
        return result

    elif version == "8.0":
        shared = model.shared_neurons
        return {
            'input_neurons': shared.input_neurons.data,
            'process_neurons': shared.process_neurons.data,
            'output_neurons': shared.output_neurons.data,
            'knowledge_K': shared.knowledge_K.data,
            'knowledge_V': shared.knowledge_V.data,
        }
    else:  # 7.9
        # Get from first layer (all layers have their own neurons)
        layer = model.layers[0]
        qkv = layer.qkv_circuit
        return {
            'input_neurons': qkv.circuit_Q.input_neurons.data,
            'process_neurons': qkv.circuit_Q.process_neurons.data,
            'output_neurons': qkv.circuit_O.output_neurons.data,
            'knowledge_K': None,
            'knowledge_V': None,
        }


def get_layer_neurons(model, layer_idx, version):
    """
    특정 레이어의 뉴런 가져오기

    v12.x, v13.x, v10.x, v8: 모든 레이어가 shared_neurons 공유
    v7.9: 각 레이어별 독립 뉴런
    """
    model = get_underlying_model(model)

    # v12.x, v13.x, v10.x - all share neurons
    if version.startswith('12') or version.startswith('13') or version.startswith('10'):
        return get_neurons(model, version)

    elif version == "8.0":
        # v8은 모든 레이어가 같은 뉴런 공유
        shared = model.shared_neurons
        return {
            'input_neurons': shared.input_neurons.data,
            'process_neurons': shared.process_neurons.data,
            'output_neurons': shared.output_neurons.data,
        }
    else:  # 7.9
        layer = model.layers[layer_idx]
        qkv = layer.qkv_circuit
        return {
            'input_neurons': qkv.circuit_Q.input_neurons.data,
            'process_neurons': qkv.circuit_Q.process_neurons.data,
            'output_neurons': qkv.circuit_O.output_neurons.data,
        }


def has_ffn(version):
    """버전이 FFN을 가지고 있는지"""
    return version == "7.9"


def has_memory(version):
    """버전이 NeuronMemory를 가지고 있는지"""
    return version.startswith('12') or version.startswith('13') or version == "8.0"


def has_shared_neurons(version):
    """버전이 SharedNeurons를 가지고 있는지"""
    return version.startswith('10') or version.startswith('12') or version.startswith('13') or version == "8.0"


def has_ssm(version):
    """버전이 SSM을 가지고 있는지"""
    return version.startswith('12') or version.startswith('13')


def has_topk_routing(version):
    """버전이 Top-k sparse routing을 가지고 있는지"""
    return version in ['12.7', '12.8', '13.0'] or version.startswith('13')


class VersionAdapter:
    """분석 스크립트용 버전 어댑터"""

    def __init__(self, model, version):
        self.model = get_underlying_model(model)
        self.version = version
        self.n_layers = len(self.model.layers)

        # v12.x, v13.x, v10.x
        if version.startswith('12') or version.startswith('13') or version.startswith('10'):
            self.n_compress = getattr(self.model, 'n_compress', None)
            self.n_expand = getattr(self.model, 'n_expand', None)
            self.n_knowledge = getattr(self.model, 'n_knowledge', None)
            self.knowledge_k = getattr(self.model, 'knowledge_k', None)
            self.top_k_compress = getattr(self.model, 'top_k_compress', None)
            self.top_k_expand = getattr(self.model, 'top_k_expand', None)
            # Legacy compatibility
            self.n_process = self.n_compress
            self.process_k = self.top_k_compress or 8
        elif version == "8.0":
            self.n_process = self.model.n_process
            self.process_k = self.model.process_k
            self.n_knowledge = self.model.n_knowledge
            self.knowledge_k = self.model.knowledge_k
        else:  # 7.9
            self.n_process = self.model.n_process
            self.process_k = self.model.process_k
            self.n_knowledge = None
            self.knowledge_k = None

    def forward_with_routing(self, input_ids):
        """Forward pass returning routing info"""
        return self.model(input_ids, return_routing_info=True)

    def get_compress_weights(self, routing_info):
        """Get compress weights from routing_info (v12+)"""
        compat = get_routing_info_compat(routing_info, self.version)
        return compat.get('compress_weights')

    def get_expand_weights(self, routing_info, comp='Q'):
        """Get expand weights from routing_info (v12+)"""
        compat = get_routing_info_compat(routing_info, self.version)
        return compat.get(f'expand_weights_{comp}')

    def get_process_indices(self, routing_info):
        """Get process indices from routing_info (legacy)"""
        compat = get_routing_info_compat(routing_info, self.version)
        return compat.get('process_indices')

    def get_memory_indices(self, routing_info):
        """Get memory/knowledge indices"""
        compat = get_routing_info_compat(routing_info, self.version)
        return compat.get('knowledge_indices') or compat.get('memory_indices')

    def get_neurons(self):
        """Get all neurons"""
        return get_neurons(self.model, self.version)

    def get_auxiliary_losses(self):
        """Get auxiliary losses"""
        return self.model.get_auxiliary_losses()
