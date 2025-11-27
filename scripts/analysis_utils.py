"""
DAWN Analysis Utilities - Version Agnostic

v7.9와 v8.0 모두 지원하는 유틸리티 함수들
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

    # Check for v8 specific params
    if 'n_knowledge' in config or 'knowledge_k' in config:
        return "8.0"

    # Check state dict keys
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any('shared_neurons' in k for k in state_dict.keys()):
        return "8.0"
    if any('knowledge_K' in k for k in state_dict.keys()):
        return "8.0"

    # Default to 7.9
    return "7.9"


def load_model(checkpoint_path, device='cuda'):
    """버전에 맞는 모델 자동 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    version = detect_model_version(checkpoint)
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    print(f"Detected model version: {version}")

    if version == "8.0":
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

    v7.9 structure:
        routing_info = {
            'routing_down': {'process_indices': ..., 'input_weights': ...},
            'routing_up': {'process_indices': ..., 'output_weights': ...},
        }

    v8.0 structure:
        routing_info = {
            'attention': {
                'routing_Q': {'process_indices': ...},
                'routing_K': {'process_indices': ...},
                'routing_V': {'process_indices': ...},
                'routing_O': {'process_indices': ...},
                'neuron_indices': ...,
            },
            'memory': {'top_indices': ..., 'top_weights': ...},
            'neuron_indices': ...,
        }

    Returns:
        process_indices: [B, S, k] - process neuron indices
        input_weights: [B, S, n_input] or None
        output_weights: [B, S, n_output] or None
        memory_indices: [B, S, knowledge_k] or None (v8 only)
    """
    if version == "8.0":
        attn_routing = routing_info.get('attention', {})
        mem_routing = routing_info.get('memory', {})

        # Get process_indices from attention routing
        # Use neuron_indices (which is routing_Q['process_indices'])
        process_indices = attn_routing.get('neuron_indices')
        if process_indices is None:
            # Fallback: try routing_Q directly
            routing_Q = attn_routing.get('routing_Q', {})
            process_indices = routing_Q.get('process_indices')

        # Also try top-level neuron_indices as last fallback
        if process_indices is None:
            process_indices = routing_info.get('neuron_indices')

        return {
            'process_indices': process_indices,
            'input_weights': attn_routing.get('routing_Q', {}).get('input_weights'),
            'output_weights': attn_routing.get('routing_O', {}).get('output_weights'),
            'memory_indices': mem_routing.get('top_indices'),
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

    Returns:
        input_neurons: [n_input, d_model, rank]
        process_neurons: [n_process, rank]
        output_neurons: [n_output, rank, d_model]
        knowledge_K: [n_knowledge, rank] or None (v8 only)
        knowledge_V: [n_knowledge, d_model] or None (v8 only)
    """
    model = get_underlying_model(model)

    if version == "8.0":
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

    v8: 모든 레이어가 shared_neurons 공유
    v7.9: 각 레이어별 독립 뉴런
    """
    model = get_underlying_model(model)

    if version == "8.0":
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
    return version == "8.0"


def has_shared_neurons(version):
    """버전이 SharedNeurons를 가지고 있는지"""
    return version == "8.0"


class VersionAdapter:
    """분석 스크립트용 버전 어댑터"""

    def __init__(self, model, version):
        self.model = get_underlying_model(model)
        self.version = version
        self.n_layers = len(self.model.layers)
        self.n_process = self.model.n_process
        self.process_k = self.model.process_k

        # Version specific
        if version == "8.0":
            self.n_knowledge = self.model.n_knowledge
            self.knowledge_k = self.model.knowledge_k
        else:
            self.n_knowledge = None
            self.knowledge_k = None

    def forward_with_routing(self, input_ids):
        """Forward pass returning routing info"""
        return self.model(input_ids, return_routing_info=True)

    def get_process_indices(self, routing_info):
        """Get process indices from routing_info"""
        compat = get_routing_info_compat(routing_info, self.version)
        return compat['process_indices']

    def get_memory_indices(self, routing_info):
        """Get memory indices (v8 only)"""
        if self.version != "8.0":
            return None
        compat = get_routing_info_compat(routing_info, self.version)
        return compat['memory_indices']

    def get_neurons(self):
        """Get all neurons"""
        return get_neurons(self.model, self.version)

    def get_auxiliary_losses(self):
        """Get auxiliary losses"""
        return self.model.get_auxiliary_losses()
