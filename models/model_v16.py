"""
DAWN v16: Split Feature QK/V Vector Neurons

Core Changes from v15:
- Feature neurons split into QK and V separate vector pools
- feature_qk_neurons: (n_feature_qk, d_model) - axis vectors for Q/K
- feature_v_neurons: (n_feature_v, d_model) - axis vectors for V
- expand_Q/K/V linear layers replace relational/value neuron pools
- 41% parameter reduction vs v15

Architecture (FRVK v16):
- Feature QK Neurons: x → top-k axis selection → h_qk (Q/K compression)
- Feature V Neurons: x → top-k axis selection → h_v (V compression)
- Expand layers: h_qk → Q/K, h_v → V via linear projections
- Relational Neurons: Still used for Q/K pattern modulation (optional)
- Value Neurons: Still used for V pattern modulation (optional)
- Knowledge Neurons (K): 2-stage retrieval (router coarse → x fine)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Mamba selective scan import (fallback included)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed, using slow for-loop SSM")


class UnifiedNeuronRouter(nn.Module):
    """
    v16: Unified neuron embedding space with split Feature QK/V

    Token-neuron matching in shared embedding space.
    Excitability (refractory-inspired) for balanced usage.

    Neuron types:
    - feature_qk: axis vectors for Q/K compression
    - feature_v: axis vectors for V compression
    - relational: Q/K pattern generation
    - value: V pattern generation
    - knowledge: memory retrieval
    """
    def __init__(self, d_model, n_feature_qk, n_feature_v, n_relational, n_value, n_knowledge,
                 d_space=64, dropout=0.1):
        super().__init__()
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_relational = n_relational
        self.n_value = n_value
        self.n_knowledge = n_knowledge
        self.d_space = d_space

        total_neurons = n_feature_qk + n_feature_v + n_relational + n_value + n_knowledge
        self.total_neurons = total_neurons

        # Index boundaries
        self.feature_qk_end = n_feature_qk
        self.feature_v_end = n_feature_qk + n_feature_v
        self.relational_end = n_feature_qk + n_feature_v + n_relational
        self.value_end = n_feature_qk + n_feature_v + n_relational + n_value
        # knowledge is value_end ~ total_neurons

        # Shared projection
        self.proj = nn.Linear(d_model, d_space)
        self.dropout = nn.Dropout(dropout)

        # Unified neuron embeddings [total_neurons, d_space]
        self.neuron_emb = nn.Parameter(torch.randn(total_neurons, d_space) * 0.02)

        # Type-specific usage tracking
        self.register_buffer('usage_ema_feature_qk', torch.zeros(n_feature_qk))
        self.register_buffer('usage_ema_feature_v', torch.zeros(n_feature_v))
        self.register_buffer('usage_ema_relational', torch.zeros(n_relational))
        self.register_buffer('usage_ema_value', torch.zeros(n_value))
        self.register_buffer('usage_ema_knowledge', torch.zeros(n_knowledge))

        # Excitability: tau (recovery time constant) + decaying weight
        self.tau = 1.5
        self.excitability_weight = 1.0

    def decay_excitability(self, decay_rate=0.9997):
        """Decay excitability_weight each step."""
        self.excitability_weight *= decay_rate

    def get_excitability(self, usage_ema):
        """
        Neuronal excitability based on usage.
        - usage_ema ≈ 0 → excitability = 1.0 (fully ready)
        - usage_ema ≈ tau → excitability ≈ 0.0 (refractory)
        """
        return torch.clamp(1.0 - usage_ema / self.tau, min=0.0, max=1.0)

    def get_logits(self, x, neuron_type):
        """
        x: [B, S, d_model]
        neuron_type: 'feature_qk', 'feature_v', 'relational_Q', 'relational_K', 'value', 'knowledge'
        """
        h_proj = self.proj(x)  # [B, S, d_space]
        h_proj = self.dropout(h_proj)

        # Normalize neuron embeddings
        neuron_emb_norm = F.normalize(self.neuron_emb, dim=-1)

        # Dot product with all neurons
        all_logits = torch.einsum('bsd,nd->bsn', h_proj, neuron_emb_norm)

        # Type-specific slicing + Excitability
        if neuron_type == 'feature_qk':
            logits = all_logits[..., :self.feature_qk_end]
            if self.training:
                excitability = self.get_excitability(self.usage_ema_feature_qk)
                logits = logits + excitability * self.excitability_weight

        elif neuron_type == 'feature_v':
            logits = all_logits[..., self.feature_qk_end:self.feature_v_end]
            if self.training:
                excitability = self.get_excitability(self.usage_ema_feature_v)
                logits = logits + excitability * self.excitability_weight

        elif neuron_type in ['relational_Q', 'relational_K']:
            logits = all_logits[..., self.feature_v_end:self.relational_end]
            if self.training:
                excitability = self.get_excitability(self.usage_ema_relational)
                logits = logits + excitability * self.excitability_weight

        elif neuron_type == 'value':
            logits = all_logits[..., self.relational_end:self.value_end]
            if self.training:
                excitability = self.get_excitability(self.usage_ema_value)
                logits = logits + excitability * self.excitability_weight

        elif neuron_type == 'knowledge':
            logits = all_logits[..., self.value_end:]
            if self.training:
                excitability = self.get_excitability(self.usage_ema_knowledge)
                logits = logits + excitability * self.excitability_weight

        return logits

    def update_usage(self, weights, neuron_type, attention_mask=None):
        """Update usage statistics for selected neurons"""
        if not self.training:
            return

        if weights.dim() == 3:
            active = (weights > 0).float()
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                active = active * mask
                count = mask.sum() + 1e-8
                usage = active.sum(dim=[0, 1]) / count
            else:
                usage = active.mean(dim=[0, 1])
        else:
            usage = (weights > 0).float().mean(dim=0)

        if neuron_type == 'feature_qk':
            self.usage_ema_feature_qk = 0.99 * self.usage_ema_feature_qk + 0.01 * usage
        elif neuron_type == 'feature_v':
            self.usage_ema_feature_v = 0.99 * self.usage_ema_feature_v + 0.01 * usage
        elif neuron_type == 'relational':
            self.usage_ema_relational = 0.99 * self.usage_ema_relational + 0.01 * usage
        elif neuron_type == 'value':
            self.usage_ema_value = 0.99 * self.usage_ema_value + 0.01 * usage
        elif neuron_type == 'knowledge':
            self.usage_ema_knowledge = 0.99 * self.usage_ema_knowledge + 0.01 * usage


class SharedNeurons(nn.Module):
    """
    v16: Shared neurons with split Feature QK/V vector pools

    Feature QK: (n_feature_qk, d_model) - each neuron is an axis vector
    Feature V: (n_feature_v, d_model) - each neuron is an axis vector
    Relational: (n_relational, rank, d_model) - Q/K pattern generation
    Value: (n_value, rank, d_model) - V pattern generation
    Knowledge K/V: for memory retrieval
    """
    def __init__(
        self,
        d_model: int,
        rank_qk: int,
        rank_v: int,
        n_feature_qk: int,
        n_feature_v: int,
        n_relational: int,
        n_value: int,
        n_knowledge: int,
        knowledge_rank: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank_qk = rank_qk
        self.rank_v = rank_v
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else 128
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_relational = n_relational
        self.n_value = n_value
        self.n_knowledge = n_knowledge

        # v16: Vector neurons (each neuron = one axis)
        # Feature QK: [n_feature_qk, d_model]
        self.feature_qk_neurons = nn.Parameter(torch.randn(n_feature_qk, d_model) * 0.02)

        # Feature V: [n_feature_v, d_model]
        self.feature_v_neurons = nn.Parameter(torch.randn(n_feature_v, d_model) * 0.02)

        # Relational neurons: [n_relational, rank_qk, d_model] - for Q/K pattern modulation
        self.relational_neurons = nn.Parameter(torch.zeros(n_relational, rank_qk, d_model))

        # Value neurons: [n_value, rank_v, d_model] - for V pattern modulation
        self.value_neurons = nn.Parameter(torch.zeros(n_value, rank_v, d_model))

        # Knowledge neurons
        self.knowledge_neurons_K = nn.Parameter(torch.zeros(n_knowledge, self.knowledge_rank))
        self.knowledge_neurons_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_parameters()

    def _init_parameters(self):
        # Feature neurons: normalize each axis vector
        nn.init.normal_(self.feature_qk_neurons, std=0.02)
        nn.init.normal_(self.feature_v_neurons, std=0.02)

        # Relational/Value neurons: orthogonal init
        for i in range(self.n_relational):
            nn.init.orthogonal_(self.relational_neurons.data[i])
        for i in range(self.n_value):
            nn.init.orthogonal_(self.value_neurons.data[i])

        nn.init.normal_(self.knowledge_neurons_K, std=0.02)
        nn.init.normal_(self.knowledge_neurons_V, std=0.02)


class GlobalSSM(nn.Module):
    """
    Selective SSM + Context Enhancement with Parallel Scan
    (Same as v15)
    """
    def __init__(self, d_model: int, state_dim: int, return_context: bool = True):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.return_context = return_context

        self.A_log = nn.Parameter(torch.randn(d_model, state_dim) * 0.1)
        self.W_delta = nn.Linear(d_model, d_model, bias=False)
        self.W_B = nn.Linear(d_model, state_dim, bias=False)
        self.W_C = nn.Linear(d_model, state_dim, bias=False)

        self.ssm_norm = nn.LayerNorm(d_model)
        self.context_proj = nn.Linear(d_model, d_model, bias=False)
        self.context_scale = nn.Parameter(torch.tensor(0.1))
        self.importance_proj = nn.Linear(d_model, d_model, bias=False)
        self.importance_temperature = 0.5

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_delta.weight, std=0.02)
        nn.init.normal_(self.W_B.weight, std=0.02)
        nn.init.normal_(self.W_C.weight, std=0.02)
        nn.init.normal_(self.context_proj.weight, std=0.02)
        nn.init.normal_(self.importance_proj.weight, std=0.02)

    def forward(self, x, attention_mask=None):
        B, S, D = x.shape

        delta = F.softplus(self.W_delta(x))
        B_sel = self.W_B(x)
        C_sel = self.W_C(x)
        A = -torch.exp(self.A_log)

        if MAMBA_AVAILABLE:
            dtype = x.dtype
            x_mamba = x.transpose(1, 2).contiguous()
            delta_mamba = delta.transpose(1, 2).contiguous()
            B_mamba = B_sel.transpose(1, 2).contiguous().to(dtype)
            C_mamba = C_sel.transpose(1, 2).contiguous().to(dtype)
            A = A.to(dtype)

            y = selective_scan_fn(
                x_mamba, delta_mamba, A, B_mamba, C_mamba,
                D=None, z=None, delta_bias=None,
                delta_softplus=False, return_last_state=False
            )
            ssm_out = y.transpose(1, 2).contiguous()
        else:
            ssm_out = self._slow_forward(x, delta, A, B_sel, C_sel)

        ssm_out = self.ssm_norm(ssm_out)

        h_final = ssm_out[:, -1, :]
        h_proj = self.importance_proj(h_final)
        raw_importance = torch.einsum('bsd,bd->bs', x, h_proj)

        if attention_mask is not None:
            masked_importance = raw_importance.masked_fill(attention_mask == 0, float('-inf'))
            importance = F.softmax(masked_importance / self.importance_temperature, dim=-1)
        else:
            importance = F.softmax(raw_importance / self.importance_temperature, dim=-1)

        if self.return_context:
            context = self.context_proj(ssm_out) * self.context_scale
        else:
            context = None

        return importance, context, raw_importance

    def _slow_forward(self, x, delta, A, B_sel, C_sel):
        B, S, D = x.shape
        N = self.state_dim

        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(S):
            delta_t = delta[:, t, :, None]
            A_exp = A[None, :, :]
            decay = torch.exp(delta_t * A_exp)

            B_t = B_sel[:, t, None, :]
            x_t = x[:, t, :, None]

            h = h * decay + (delta_t * x_t) * B_t
            C_t = C_sel[:, t, :]
            y_t = torch.einsum('bdn,bn->bd', h, C_t)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class GlobalRouters(nn.Module):
    """
    v16: Unified routing with split Feature QK/V

    Separate routing for:
    - feature_qk: select QK axis vectors
    - feature_v: select V axis vectors
    - relational Q/K: select Q/K pattern neurons
    - value: select V pattern neurons
    """
    def __init__(self, d_model: int, n_feature_qk: int, n_feature_v: int,
                 n_relational: int, n_value: int, n_knowledge: int,
                 top_k_feature_qk: int = 64, top_k_feature_v: int = 32,
                 top_k_relational: int = 16, top_k_value: int = 3,
                 d_space: int = 64, router_dropout: float = 0.1, token_routing: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_relational = n_relational
        self.n_value = n_value
        self.n_knowledge = n_knowledge
        self.top_k_feature_qk = top_k_feature_qk
        self.top_k_feature_v = top_k_feature_v
        self.top_k_relational = top_k_relational
        self.top_k_value = top_k_value
        self.token_routing = token_routing

        self.neuron_router = UnifiedNeuronRouter(
            d_model, n_feature_qk, n_feature_v, n_relational, n_value, n_knowledge,
            d_space=d_space, dropout=router_dropout
        )

    def get_attention_weights(self, x, importance, attention_mask=None):
        """
        Compute attention routing weights with top-k selection.

        Returns:
            idx_qk: [B, top_k_feature_qk] - sorted indices for QK neurons
            idx_v: [B, top_k_feature_v] - sorted indices for V neurons
            relational_weights_Q: [B, n_relational]
            relational_weights_K: [B, n_relational]
            value_weights: [B, n_value]
            routing_info: dict
            aux_loss: scalar
        """
        # Feature QK routing
        logits_qk = self.neuron_router.get_logits(x, 'feature_qk')  # [B, S, n_feature_qk]
        pref_qk = F.softmax(logits_qk, dim=-1)
        weights_qk = torch.einsum('bs,bsn->bn', importance, pref_qk)  # [B, n_feature_qk]

        # Top-k + sort by index
        _, idx_qk = torch.topk(weights_qk, self.top_k_feature_qk, dim=-1)  # [B, top_k]
        idx_qk = idx_qk.sort(dim=-1).values  # Sort for consistent ordering

        # Feature V routing
        logits_v = self.neuron_router.get_logits(x, 'feature_v')  # [B, S, n_feature_v]
        pref_v = F.softmax(logits_v, dim=-1)
        weights_v = torch.einsum('bs,bsn->bn', importance, pref_v)  # [B, n_feature_v]

        _, idx_v = torch.topk(weights_v, self.top_k_feature_v, dim=-1)  # [B, top_k]
        idx_v = idx_v.sort(dim=-1).values

        # Relational Q/K routing (same as v15)
        relational_logits_Q = self.neuron_router.get_logits(x, 'relational_Q')
        relational_logits_K = self.neuron_router.get_logits(x, 'relational_K')
        relational_pref_Q = F.softmax(relational_logits_Q, dim=-1)
        relational_pref_K = F.softmax(relational_logits_K, dim=-1)

        relational_weights_Q_dense = torch.einsum('bs,bsn->bn', importance, relational_pref_Q)
        relational_weights_K_dense = torch.einsum('bs,bsn->bn', importance, relational_pref_K)

        relational_weights_Q, _ = self._topk_sparsify(relational_weights_Q_dense, self.top_k_relational)
        relational_weights_K, _ = self._topk_sparsify(relational_weights_K_dense, self.top_k_relational)

        # Value routing
        value_logits = self.neuron_router.get_logits(x, 'value')
        value_pref = F.softmax(value_logits, dim=-1)
        value_weights_dense = torch.einsum('bs,bsn->bn', importance, value_pref)
        value_weights, _ = self._topk_sparsify(value_weights_dense, self.top_k_value)

        # Compute aux_loss (load balance)
        aux_loss = 0.0
        if self.training:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                count = mask.sum() + 1e-8
                usage_QK = (pref_qk * mask).sum(dim=(0, 1)) / count
                usage_V = (pref_v * mask).sum(dim=(0, 1)) / count
                usage_R_Q = (relational_pref_Q * mask).sum(dim=(0, 1)) / count
                usage_R_K = (relational_pref_K * mask).sum(dim=(0, 1)) / count
                usage_Val = (value_pref * mask).sum(dim=(0, 1)) / count
            else:
                usage_QK = pref_qk.mean(dim=(0, 1))
                usage_V = pref_v.mean(dim=(0, 1))
                usage_R_Q = relational_pref_Q.mean(dim=(0, 1))
                usage_R_K = relational_pref_K.mean(dim=(0, 1))
                usage_Val = value_pref.mean(dim=(0, 1))

            target_QK = 1.0 / self.n_feature_qk
            aux_loss = aux_loss + ((usage_QK - target_QK) ** 2).sum() * self.n_feature_qk

            target_V = 1.0 / self.n_feature_v
            aux_loss = aux_loss + ((usage_V - target_V) ** 2).sum() * self.n_feature_v

            target_R = 1.0 / self.n_relational
            aux_loss = aux_loss + ((usage_R_Q - target_R) ** 2).sum() * self.n_relational
            aux_loss = aux_loss + ((usage_R_K - target_R) ** 2).sum() * self.n_relational

            target_Val = 1.0 / self.n_value
            aux_loss = aux_loss + ((usage_Val - target_Val) ** 2).sum() * self.n_value

        routing_info = {
            'idx_qk': idx_qk.detach(),
            'idx_v': idx_v.detach(),
            'relational_weights_Q': relational_weights_Q.detach(),
            'relational_weights_K': relational_weights_K.detach(),
            'value_weights': value_weights.detach(),
            'feature_qk_pref': pref_qk.detach(),
            'feature_v_pref': pref_v.detach(),
        }

        # Update usage statistics
        if self.training:
            # Create indicator for selected QK/V indices
            qk_indicator = torch.zeros(weights_qk.shape[0], self.n_feature_qk, device=x.device)
            qk_indicator.scatter_(1, idx_qk, 1.0)
            self.neuron_router.update_usage(qk_indicator, 'feature_qk', attention_mask)

            v_indicator = torch.zeros(weights_v.shape[0], self.n_feature_v, device=x.device)
            v_indicator.scatter_(1, idx_v, 1.0)
            self.neuron_router.update_usage(v_indicator, 'feature_v', attention_mask)

            self.neuron_router.update_usage(value_weights, 'value', attention_mask)

            relational_used = ((relational_weights_Q > 0) | (relational_weights_K > 0)).float()
            self.neuron_router.update_usage(relational_used, 'relational', attention_mask)

        return idx_qk, idx_v, relational_weights_Q, relational_weights_K, value_weights, routing_info, aux_loss

    def _topk_sparsify(self, weights, k):
        """Apply top-k sparsification and renormalize"""
        topk_vals, topk_idx = torch.topk(weights, k, dim=-1)
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(-1, topk_idx, topk_vals)
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-8)
        return sparse_weights, topk_idx


class NeuronCircuit(nn.Module):
    """
    v16: Attention circuit with split Feature QK/V vector neurons

    Flow:
    1. x → feature_qk_neurons[idx_qk] → h_qk (compression to rank_qk)
    2. x → feature_v_neurons[idx_v] → h_v (compression to rank_v)
    3. h_qk → expand_Q/K → Q, K (with optional relational modulation)
    4. h_v → expand_V → V (with optional value modulation)
    5. Multi-head attention
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank_qk: int,
        rank_v: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank_qk = rank_qk
        self.rank_v = rank_v

        # Expand matrices: h_qk/h_v → Q/K/V
        self.expand_Q = nn.Linear(rank_qk, d_model, bias=False)
        self.expand_K = nn.Linear(rank_qk, d_model, bias=False)
        self.expand_V = nn.Linear(rank_v, d_model, bias=False)
        self.expand_O = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, idx_qk, idx_v, relational_weights_Q, relational_weights_K, value_weights, attention_mask=None):
        """
        Args:
            x: [B, S, D]
            idx_qk: [B, top_k_qk] - indices of selected QK feature neurons
            idx_v: [B, top_k_v] - indices of selected V feature neurons
            relational_weights_Q: [B, n_relational]
            relational_weights_K: [B, n_relational]
            value_weights: [B, n_value]
            attention_mask: [B, S] optional
        """
        B, S, D = x.shape

        # 1. Feature QK: select axis vectors and compress
        # selected_qk: [B, top_k_qk, d_model]
        selected_qk = self.shared_neurons.feature_qk_neurons[idx_qk]  # [B, top_k_qk, D]
        W_qk = selected_qk.transpose(-1, -2)  # [B, D, top_k_qk]
        h_qk = torch.einsum('bsd,bdk->bsk', x, W_qk)  # [B, S, top_k_qk]

        # 2. Feature V: select axis vectors and compress
        selected_v = self.shared_neurons.feature_v_neurons[idx_v]  # [B, top_k_v, D]
        W_v = selected_v.transpose(-1, -2)  # [B, D, top_k_v]
        h_v = torch.einsum('bsd,bdk->bsk', x, W_v)  # [B, S, top_k_v]

        # 3. Generate Q/K from h_qk with relational modulation
        # Relational neurons: [n_relational, rank_qk, d_model]
        pool_R = self.shared_neurons.relational_neurons
        # Weighted combination: [B, rank_qk, d_model]
        shared_relational_Q = torch.einsum('bn,nrd->brd', relational_weights_Q, pool_R)
        shared_relational_K = torch.einsum('bn,nrd->brd', relational_weights_K, pool_R)

        # h_qk: [B, S, rank_qk], shared_relational: [B, rank_qk, d_model]
        # First apply base expand, then add relational pattern
        Q_base = self.expand_Q(h_qk)  # [B, S, D]
        K_base = self.expand_K(h_qk)  # [B, S, D]

        # Add relational modulation
        Q_rel = torch.einsum('bsr,brd->bsd', h_qk, shared_relational_Q)  # [B, S, D]
        K_rel = torch.einsum('bsr,brd->bsd', h_qk, shared_relational_K)  # [B, S, D]

        Q = Q_base + Q_rel
        K = K_base + K_rel

        # 4. Generate V from h_v with value modulation
        pool_V = self.shared_neurons.value_neurons  # [n_value, rank_v, d_model]
        shared_value = torch.einsum('bn,nrd->brd', value_weights, pool_V)  # [B, rank_v, D]

        V_base = self.expand_V(h_v)  # [B, S, D]
        V_mod = torch.einsum('bsr,brd->bsd', h_v, shared_value)  # [B, S, D]
        V = V_base + V_mod

        # 5. Multi-head Attention with FlashAttention
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        if attention_mask is not None:
            causal_mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
            pad_mask = (attention_mask == 0).view(B, 1, 1, S)
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) | pad_mask
            attn_out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=~combined_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                Q, K, V,
                is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)

        # 6. Output projection
        output = self.expand_O(attn_out)
        output = self.out_dropout(output)

        return output, None


class NeuronMemory(nn.Module):
    """
    v16: 2-stage hierarchical knowledge retrieval (same as v15)
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_knowledge: int,
        knowledge_rank: int = None,
        coarse_k: int = 20,
        fine_k: int = 10,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_knowledge = n_knowledge
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else 128
        self.coarse_k = coarse_k
        self.fine_k = fine_k

    def forward(self, x, router, knowledge_encoder, attention_mask=None):
        B, S, D = x.shape

        # Stage 1: Coarse candidate selection via router
        k_logits = router.get_logits(x, 'knowledge')
        coarse_scores, candidate_idx = torch.topk(k_logits, self.coarse_k, dim=-1)

        if self.training:
            coarse_indicator = torch.zeros(B, S, self.n_knowledge, device=x.device)
            coarse_indicator.scatter_(-1, candidate_idx, 1.0)
            router.update_usage(coarse_indicator, 'knowledge', attention_mask)

        # Stage 2: Fine matching within candidates
        query = knowledge_encoder(x)

        K_all = self.shared_neurons.knowledge_neurons_K
        V_all = self.shared_neurons.knowledge_neurons_V

        candidate_idx_k = candidate_idx.unsqueeze(-1).expand(B, S, self.coarse_k, self.knowledge_rank)
        K_candidates = K_all.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1).gather(2, candidate_idx_k)

        fine_scores = torch.einsum('bsd,bscd->bsc', query, K_candidates) / math.sqrt(self.knowledge_rank)

        fine_topk_scores, fine_topk_local_idx = torch.topk(fine_scores, self.fine_k, dim=-1)
        fine_weights = F.softmax(fine_topk_scores, dim=-1)

        fine_global_idx = candidate_idx.gather(-1, fine_topk_local_idx)

        fine_idx_v = fine_global_idx.unsqueeze(-1).expand(B, S, self.fine_k, self.d_model)
        V_expanded = V_all.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        selected_V = V_expanded.gather(2, fine_idx_v)

        output = (selected_V * fine_weights.unsqueeze(-1)).sum(dim=2)

        info = {
            'coarse_indices': candidate_idx,
            'coarse_scores': coarse_scores,
            'fine_indices': fine_global_idx,
            'fine_weights': fine_weights,
        }

        return output, info


class DAWNBlock(nn.Module):
    """DAWN v16 block with split Feature QK/V"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank_qk: int,
        rank_v: int,
        n_knowledge: int,
        knowledge_rank: int = None,
        coarse_k: int = 20,
        fine_k: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = NeuronCircuit(shared_neurons, d_model, n_heads, rank_qk, rank_v, dropout)
        self.memory = NeuronMemory(
            shared_neurons, d_model, n_knowledge,
            knowledge_rank=knowledge_rank, coarse_k=coarse_k, fine_k=fine_k
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, importance, global_routers: GlobalRouters, knowledge_encoder, attention_mask=None):
        normed_x = self.norm1(x)
        idx_qk, idx_v, relational_Q, relational_K, value_w, attn_routing, attn_aux_loss = \
            global_routers.get_attention_weights(normed_x, importance, attention_mask)

        attn_out, _ = self.attn(normed_x, idx_qk, idx_v, relational_Q, relational_K, value_w, attention_mask)
        x = x + attn_out

        normed_x2 = self.norm2(x)
        mem_out, knowledge_info = self.memory(normed_x2, global_routers.neuron_router, knowledge_encoder, attention_mask)
        x = x + self.dropout(mem_out)

        attn_out_norm = attn_out.norm(dim=-1).mean().detach()
        mem_out_norm = mem_out.norm(dim=-1).mean().detach()

        routing_info = {
            'attention': {**attn_routing, 'idx_qk': idx_qk.detach(), 'idx_v': idx_v.detach()},
            'memory': {**{k: v.detach() for k, v in knowledge_info.items()}},
            'attn_out_norm': attn_out_norm,
            'mem_out_norm': mem_out_norm,
        }

        return x, routing_info, attn_aux_loss


class DAWN(nn.Module):
    """
    DAWN v16: Split Feature QK/V Vector Neurons

    Core Changes from v15:
    - Feature neurons split: feature_qk_neurons + feature_v_neurons
    - Each feature neuron is a single axis vector (not rank matrix)
    - expand_Q/K/V linear layers for reconstruction
    - 41% parameter reduction

    Architecture (FRVK v16):
    - Feature QK: (n_feature_qk, d_model) axis vectors for Q/K
    - Feature V: (n_feature_v, d_model) axis vectors for V
    - Relational: Q/K pattern modulation
    - Value: V pattern modulation
    - Knowledge: 2-stage retrieval
    """
    __version__ = "16.0"

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 384,
        n_layers: int = 12,
        n_heads: int = 4,
        rank_qk: int = 64,
        rank_v: int = 32,
        max_seq_len: int = 512,
        n_feature_qk: int = 512,
        n_feature_v: int = 256,
        n_relational: int = 160,
        n_value: int = 12,
        n_knowledge: int = 256,
        coarse_k: int = 16,
        fine_k: int = 8,
        knowledge_rank: int = None,
        state_dim: int = 64,
        top_k_feature_qk: int = 64,
        top_k_feature_v: int = 32,
        top_k_relational: int = 16,
        top_k_value: int = 3,
        d_space: int = 64,
        dropout: float = 0.1,
        router_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        token_routing: bool = False,
        use_ssm_context: bool = True,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.rank_qk = rank_qk
        self.rank_v = rank_v
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else 128
        self.max_seq_len = max_seq_len
        self.state_dim = state_dim
        self.top_k_feature_qk = top_k_feature_qk
        self.top_k_feature_v = top_k_feature_v
        self.top_k_relational = top_k_relational
        self.top_k_value = top_k_value
        self.d_space = d_space
        self.token_routing = token_routing
        self.router_dropout = router_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.use_ssm_context = use_ssm_context

        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_relational = n_relational
        self.n_value = n_value
        self.n_knowledge = n_knowledge
        self.coarse_k = coarse_k
        self.fine_k = fine_k

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.shared_neurons = SharedNeurons(
            d_model=d_model, rank_qk=rank_qk, rank_v=rank_v,
            n_feature_qk=n_feature_qk, n_feature_v=n_feature_v,
            n_relational=n_relational, n_value=n_value,
            n_knowledge=n_knowledge, knowledge_rank=self.knowledge_rank,
        )

        self.global_ssm = GlobalSSM(d_model, state_dim, return_context=use_ssm_context)

        self.global_routers = GlobalRouters(
            d_model, n_feature_qk, n_feature_v, n_relational, n_value, n_knowledge,
            top_k_feature_qk, top_k_feature_v, top_k_relational, top_k_value,
            d_space=d_space, router_dropout=router_dropout, token_routing=token_routing
        )

        self.knowledge_encoder = nn.Linear(d_model, self.knowledge_rank, bias=False)

        self.layers = nn.ModuleList([
            DAWNBlock(
                shared_neurons=self.shared_neurons, d_model=d_model, n_heads=n_heads,
                rank_qk=rank_qk, rank_v=rank_v, n_knowledge=n_knowledge,
                knowledge_rank=self.knowledge_rank, coarse_k=coarse_k, fine_k=fine_k, dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.aux_loss = 0.0

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, labels=None, attention_mask=None, return_routing_info=False):
        B, S = input_ids.shape
        device = input_ids.device

        self.aux_loss = 0.0

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        routing_infos = []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                importance, context, raw_importance = checkpoint(
                    self.global_ssm, x, attention_mask, use_reentrant=False
                )
            else:
                importance, context, raw_importance = self.global_ssm(x, attention_mask)
            if context is not None:
                x = x + context

            if self.gradient_checkpointing and self.training:
                x, routing_info, layer_aux_loss = checkpoint(
                    layer, x, importance, self.global_routers, self.knowledge_encoder, attention_mask,
                    use_reentrant=False
                )
            else:
                x, routing_info, layer_aux_loss = layer(x, importance, self.global_routers, self.knowledge_encoder, attention_mask)

            self.aux_loss = self.aux_loss + layer_aux_loss

            if return_routing_info:
                routing_info['importance'] = importance.detach()
                routing_info['raw_importance'] = raw_importance.detach()
                routing_infos.append(routing_info)

        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().long()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), ignore_index=-100)
            if return_routing_info:
                return loss, logits, routing_infos
            return loss, logits

        if return_routing_info:
            return logits, routing_infos
        return logits

    def orthogonality_loss(self):
        """Orthogonality loss for relational/value neurons"""
        I_qk = torch.eye(self.rank_qk, device=self.shared_neurons.relational_neurons.device).unsqueeze(0)
        I_v = torch.eye(self.rank_v, device=self.shared_neurons.value_neurons.device).unsqueeze(0)

        # Relational: [n_relational, rank_qk, d_model] -> W @ W^T
        W_r = self.shared_neurons.relational_neurons
        WWt_r = torch.bmm(W_r, W_r.transpose(1, 2))
        loss_r = ((WWt_r - I_qk) ** 2).mean()

        # Value: [n_value, rank_v, d_model] -> W @ W^T
        W_v = self.shared_neurons.value_neurons
        WWt_v = torch.bmm(W_v, W_v.transpose(1, 2))
        loss_v = ((WWt_v - I_v) ** 2).mean()

        return (loss_r + loss_v) / 2

    def knowledge_diversity_loss(self):
        K = self.shared_neurons.knowledge_neurons_K
        K_norm = F.normalize(K, dim=-1)
        sim = K_norm @ K_norm.T
        mask = ~torch.eye(self.n_knowledge, dtype=torch.bool, device=K.device)
        return sim[mask].abs().mean()

    def get_auxiliary_losses(self):
        return {
            'orth_total': self.orthogonality_loss(),
            'knowledge_div': self.knowledge_diversity_loss(),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_by_component(self):
        feature_qk = self.shared_neurons.feature_qk_neurons.numel()
        feature_v = self.shared_neurons.feature_v_neurons.numel()
        relational = self.shared_neurons.relational_neurons.numel()
        value = self.shared_neurons.value_neurons.numel()
        knowledge = self.shared_neurons.knowledge_neurons_K.numel() + self.shared_neurons.knowledge_neurons_V.numel()
        embed = self.token_emb.weight.numel() + self.pos_emb.weight.numel()

        ssm_total = (
            self.global_ssm.A_log.numel() +
            self.global_ssm.W_delta.weight.numel() +
            self.global_ssm.W_B.weight.numel() +
            self.global_ssm.W_C.weight.numel() +
            self.global_ssm.context_proj.weight.numel() +
            self.global_ssm.importance_proj.weight.numel()
        )

        routers = sum(p.numel() for p in self.global_routers.neuron_router.parameters())

        # Expand matrices (per layer)
        expand_per_layer = (
            self.layers[0].attn.expand_Q.weight.numel() +
            self.layers[0].attn.expand_K.weight.numel() +
            self.layers[0].attn.expand_V.weight.numel() +
            self.layers[0].attn.expand_O.weight.numel()
        )
        expand_total = expand_per_layer * self.n_layers

        norms = sum(p.numel() for n, p in self.named_parameters() if 'norm' in n)

        print(f"=== DAWN v16 Parameter Breakdown (Split Feature QK/V) ===")
        print(f"Feature QK Neurons:    {feature_qk:,} ({feature_qk/1e3:.1f}K) [{self.n_feature_qk} × {self.d_model}]")
        print(f"Feature V Neurons:     {feature_v:,} ({feature_v/1e3:.1f}K) [{self.n_feature_v} × {self.d_model}]")
        print(f"RelationalNeurons (R): {relational:,} ({relational/1e6:.2f}M)")
        print(f"ValueNeurons (V):      {value:,} ({value/1e6:.2f}M)")
        print(f"Expand Q/K/V/O:        {expand_total:,} ({expand_total/1e3:.1f}K)")
        print(f"KnowledgeNeurons (K):  {knowledge:,} ({knowledge/1e3:.1f}K)")
        print(f"Embeddings:            {embed:,} ({embed/1e6:.2f}M)")
        print(f"Mamba SSM:             {ssm_total:,} ({ssm_total/1e3:.1f}K)")
        print(f"Unified Router:        {routers:,} ({routers/1e3:.1f}K) [d_space={self.d_space}]")
        print(f"LayerNorms:            {norms:,} ({norms/1e3:.1f}K)")
        print(f"---")
        print(f"Top-k Feature QK: {self.top_k_feature_qk}/{self.n_feature_qk}")
        print(f"Top-k Feature V:  {self.top_k_feature_v}/{self.n_feature_v}")
        print(f"Top-k Relational: {self.top_k_relational}/{self.n_relational}")
        print(f"Top-k Value:      {self.top_k_value}/{self.n_value}")
        print(f"Mamba Available: {MAMBA_AVAILABLE}")
        print(f"---")
        print(f"Total:                 {self.count_parameters():,} ({self.count_parameters()/1e6:.2f}M)")

        return {
            'feature_qk': feature_qk, 'feature_v': feature_v,
            'relational': relational, 'value': value,
            'expand': expand_total, 'knowledge': knowledge,
            'embeddings': embed, 'ssm': ssm_total,
            'routers': routers, 'norms': norms,
        }

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'rank_qk': self.rank_qk, 'rank_v': self.rank_v,
            'knowledge_rank': self.knowledge_rank,
            'max_seq_len': self.max_seq_len,
            'n_feature_qk': self.n_feature_qk, 'n_feature_v': self.n_feature_v,
            'n_relational': self.n_relational, 'n_value': self.n_value,
            'n_knowledge': self.n_knowledge,
            'coarse_k': self.coarse_k, 'fine_k': self.fine_k,
            'state_dim': self.state_dim,
            'top_k_feature_qk': self.top_k_feature_qk,
            'top_k_feature_v': self.top_k_feature_v,
            'top_k_relational': self.top_k_relational,
            'top_k_value': self.top_k_value,
            'd_space': self.d_space,
            'router_dropout': self.router_dropout,
            'gradient_checkpointing': self.gradient_checkpointing,
            'token_routing': self.token_routing,
        }

    def get_model_info(self):
        """Return model architecture info for logging"""
        return [
            f"DAWN v{self.__version__}: Split Feature QK/V Vector Neurons",
            f"  rank_qk={self.rank_qk}, rank_v={self.rank_v}, knowledge_rank={self.knowledge_rank}",
            f"  Feature QK Neurons: {self.n_feature_qk} × {self.d_model} (top-k={self.top_k_feature_qk})",
            f"  Feature V Neurons: {self.n_feature_v} × {self.d_model} (top-k={self.top_k_feature_v})",
            f"  Relational (R): {self.n_relational} × {self.rank_qk} × {self.d_model} (Q/K patterns)",
            f"  Value (V): {self.n_value} × {self.rank_v} × {self.d_model} (V patterns)",
            f"  KnowledgeNeurons (K): {self.n_knowledge} (coarse={self.coarse_k} → fine={self.fine_k})",
            f"  Expand: Q({self.rank_qk}→{self.d_model}), K({self.rank_qk}→{self.d_model}), V({self.rank_v}→{self.d_model})",
            f"  Router: d_space={self.d_space}, Excitability (SAR)",
        ]

    def load_state_dict(self, state_dict, strict=True):
        """Backward compatible load"""
        remapped = {}
        for key, value in state_dict.items():
            new_key = key
            # v15 → v16 compatibility: feature_neurons → feature_qk_neurons
            if 'feature_neurons' in key and 'feature_qk' not in key and 'feature_v' not in key:
                # Skip old feature_neurons, they have different shape
                continue
            if 'transfer_neurons' in key:
                new_key = key.replace('transfer_neurons', 'value_neurons')
            if 'usage_ema_feature' in key and 'usage_ema_feature_qk' not in key and 'usage_ema_feature_v' not in key:
                # Old feature usage, skip
                continue
            remapped[new_key] = value

        return super().load_state_dict(remapped, strict=strict)
