"""
DAWN v17: Full Vector Neurons

Core Changes from v16:
- ALL neurons are vectors (no rank matrices)
- 5 separate routing types: feature_qk, feature_v, relational_q, relational_k, value
- Relational Q/K neurons: [n_relational_q/k, d_model] - expansion vectors
- Value neurons: [n_value, d_model] - expansion vectors
- Excitability (SAR) for balanced neuron usage
- 82% parameter reduction vs v15

Architecture (FRVK v17):
- Feature QK: x → top-k selection → h_qk (compression, 320→64)
- Feature V: x → top-k selection → h_v (compression, 320→32)
- Relational Q: h_qk → top-k selection → Q (expansion, 64→320)
- Relational K: h_qk → top-k selection → K (expansion, 64→320)
- Value: h_v → top-k selection → V (expansion, 32→320)
- Knowledge: 2-stage retrieval (same as v16)
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
    v17: Unified neuron router with 5 separate types + Excitability

    Neuron types:
    - feature_qk: compression axis vectors for Q/K
    - feature_v: compression axis vectors for V
    - relational_q: expansion vectors for Q
    - relational_k: expansion vectors for K
    - value: expansion vectors for V
    - knowledge: memory retrieval

    Excitability (refractory-inspired):
    - Less-used neurons get higher excitability bonus
    - Encourages balanced neuron usage during training
    """
    def __init__(self, d_model, n_feature_qk, n_feature_v, n_relational_q, n_relational_k,
                 n_value, n_knowledge, d_space=64, dropout=0.1):
        super().__init__()
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_relational_q = n_relational_q
        self.n_relational_k = n_relational_k
        self.n_value = n_value
        self.n_knowledge = n_knowledge
        self.d_space = d_space

        total_neurons = n_feature_qk + n_feature_v + n_relational_q + n_relational_k + n_value + n_knowledge
        self.total_neurons = total_neurons

        # Index boundaries
        self.feature_qk_end = n_feature_qk
        self.feature_v_end = n_feature_qk + n_feature_v
        self.relational_q_end = n_feature_qk + n_feature_v + n_relational_q
        self.relational_k_end = n_feature_qk + n_feature_v + n_relational_q + n_relational_k
        self.value_end = n_feature_qk + n_feature_v + n_relational_q + n_relational_k + n_value
        # knowledge is value_end ~ total_neurons

        # Shared projection
        self.proj = nn.Linear(d_model, d_space)
        self.dropout = nn.Dropout(dropout)

        # Unified neuron embeddings [total_neurons, d_space]
        self.neuron_emb = nn.Parameter(torch.randn(total_neurons, d_space) * 0.02)

        # Usage tracking for excitability
        self.register_buffer('usage_ema_feature_qk', torch.zeros(n_feature_qk))
        self.register_buffer('usage_ema_feature_v', torch.zeros(n_feature_v))
        self.register_buffer('usage_ema_relational_q', torch.zeros(n_relational_q))
        self.register_buffer('usage_ema_relational_k', torch.zeros(n_relational_k))
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

    @torch.compiler.disable
    def update_usage(self, indices, neuron_type, n_neurons):
        """Update usage EMA for selected neurons (index-based)

        Excluded from torch.compile() to ensure buffer updates work correctly.
        """
        if not self.training:
            return

        with torch.no_grad():
            # Vectorized counting: flatten indices and use single scatter_add_
            indices = indices.detach()
            B = indices.shape[0]
            flat_indices = indices.view(-1)  # [B * top_k]
            usage = torch.zeros(n_neurons, device=indices.device)
            ones = torch.ones_like(flat_indices, dtype=usage.dtype)
            usage.scatter_add_(0, flat_indices, ones)
            usage = usage / B  # Normalize by batch size

            # In-place update: ema = 0.99 * ema + 0.01 * usage
            if neuron_type == 'feature_qk':
                self.usage_ema_feature_qk.mul_(0.99).add_(usage, alpha=0.01)
            elif neuron_type == 'feature_v':
                self.usage_ema_feature_v.mul_(0.99).add_(usage, alpha=0.01)
            elif neuron_type == 'relational_q':
                self.usage_ema_relational_q.mul_(0.99).add_(usage, alpha=0.01)
            elif neuron_type == 'relational_k':
                self.usage_ema_relational_k.mul_(0.99).add_(usage, alpha=0.01)
            elif neuron_type == 'value':
                self.usage_ema_value.mul_(0.99).add_(usage, alpha=0.01)
            elif neuron_type == 'knowledge':
                self.usage_ema_knowledge.mul_(0.99).add_(usage, alpha=0.01)

    def get_logits(self, x, neuron_type):
        """
        x: [B, S, d_model] or [B, S, rank] for expansion types
        neuron_type: 'feature_qk', 'feature_v', 'relational_q', 'relational_k', 'value', 'knowledge'
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

        elif neuron_type == 'relational_q':
            logits = all_logits[..., self.feature_v_end:self.relational_q_end]
            if self.training:
                excitability = self.get_excitability(self.usage_ema_relational_q)
                logits = logits + excitability * self.excitability_weight

        elif neuron_type == 'relational_k':
            logits = all_logits[..., self.relational_q_end:self.relational_k_end]
            if self.training:
                excitability = self.get_excitability(self.usage_ema_relational_k)
                logits = logits + excitability * self.excitability_weight

        elif neuron_type == 'value':
            logits = all_logits[..., self.relational_k_end:self.value_end]
            if self.training:
                excitability = self.get_excitability(self.usage_ema_value)
                logits = logits + excitability * self.excitability_weight

        elif neuron_type == 'knowledge':
            logits = all_logits[..., self.value_end:]
            if self.training:
                excitability = self.get_excitability(self.usage_ema_knowledge)
                logits = logits + excitability * self.excitability_weight

        return logits


class SharedNeurons(nn.Module):
    """
    v17: ALL neurons are vectors (no rank matrices)

    Compression neurons:
    - feature_qk_neurons: [n_feature_qk, d_model]
    - feature_v_neurons: [n_feature_v, d_model]

    Expansion neurons:
    - relational_q_neurons: [n_relational_q, d_model]
    - relational_k_neurons: [n_relational_k, d_model]
    - value_neurons: [n_value, d_model]

    Knowledge neurons:
    - knowledge_neurons_K: [n_knowledge, knowledge_rank]
    - knowledge_neurons_V: [n_knowledge, d_model]
    """
    def __init__(
        self,
        d_model: int,
        n_feature_qk: int,
        n_feature_v: int,
        n_relational_q: int,
        n_relational_k: int,
        n_value: int,
        n_knowledge: int,
        knowledge_rank: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else 128
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_relational_q = n_relational_q
        self.n_relational_k = n_relational_k
        self.n_value = n_value
        self.n_knowledge = n_knowledge

        # Compression neurons (input → hidden)
        self.feature_qk_neurons = nn.Parameter(torch.randn(n_feature_qk, d_model) * 0.02)
        self.feature_v_neurons = nn.Parameter(torch.randn(n_feature_v, d_model) * 0.02)

        # Expansion neurons (hidden → output)
        self.relational_q_neurons = nn.Parameter(torch.randn(n_relational_q, d_model) * 0.02)
        self.relational_k_neurons = nn.Parameter(torch.randn(n_relational_k, d_model) * 0.02)
        self.value_neurons = nn.Parameter(torch.randn(n_value, d_model) * 0.02)

        # Knowledge neurons
        self.knowledge_neurons_K = nn.Parameter(torch.zeros(n_knowledge, self.knowledge_rank))
        self.knowledge_neurons_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_parameters()

    def _init_parameters(self):
        # All vector neurons: normal init
        nn.init.normal_(self.feature_qk_neurons, std=0.02)
        nn.init.normal_(self.feature_v_neurons, std=0.02)
        nn.init.normal_(self.relational_q_neurons, std=0.02)
        nn.init.normal_(self.relational_k_neurons, std=0.02)
        nn.init.normal_(self.value_neurons, std=0.02)

        nn.init.normal_(self.knowledge_neurons_K, std=0.02)
        nn.init.normal_(self.knowledge_neurons_V, std=0.02)


class GlobalSSM(nn.Module):
    """
    Selective SSM + Context Enhancement with Parallel Scan
    (Same as v16)
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
    v17: 5-type routing (all vectors)

    Routes:
    - feature_qk: [B, S, n_feature_qk] → top_k_feature_qk indices
    - feature_v: [B, S, n_feature_v] → top_k_feature_v indices
    - relational_q: [B, S, n_relational_q] → top_k_relational indices
    - relational_k: [B, S, n_relational_k] → top_k_relational indices
    - value: [B, S, n_value] → top_k_value indices
    """
    def __init__(self, d_model: int, n_feature_qk: int, n_feature_v: int,
                 n_relational_q: int, n_relational_k: int, n_value: int, n_knowledge: int,
                 top_k_feature_qk: int = 64, top_k_feature_v: int = 32,
                 top_k_relational: int = 64, top_k_value: int = 32,
                 d_space: int = 64, router_dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_relational_q = n_relational_q
        self.n_relational_k = n_relational_k
        self.n_value = n_value
        self.n_knowledge = n_knowledge
        self.top_k_feature_qk = top_k_feature_qk
        self.top_k_feature_v = top_k_feature_v
        self.top_k_relational = top_k_relational
        self.top_k_value = top_k_value

        self.neuron_router = UnifiedNeuronRouter(
            d_model, n_feature_qk, n_feature_v, n_relational_q, n_relational_k,
            n_value, n_knowledge, d_space=d_space, dropout=router_dropout
        )

    def get_attention_weights(self, x, importance, attention_mask=None):
        """
        Compute attention routing weights with top-k selection.

        Returns:
            idx_qk: [B, top_k_feature_qk] - sorted indices for Feature QK
            idx_v: [B, top_k_feature_v] - sorted indices for Feature V
            idx_q: [B, top_k_relational] - sorted indices for Relational Q
            idx_k: [B, top_k_relational] - sorted indices for Relational K
            idx_v2: [B, top_k_value] - sorted indices for Value
            routing_info: dict
            aux_loss: scalar
        """
        # Feature QK routing
        logits_qk = self.neuron_router.get_logits(x, 'feature_qk')
        pref_qk = F.softmax(logits_qk, dim=-1)
        weights_qk = torch.einsum('bs,bsn->bn', importance, pref_qk)

        _, idx_qk = torch.topk(weights_qk, self.top_k_feature_qk, dim=-1)
        idx_qk = idx_qk.sort(dim=-1).values
        # Update usage EMA
        self.neuron_router.update_usage(idx_qk, 'feature_qk', self.n_feature_qk)

        # Feature V routing
        logits_v = self.neuron_router.get_logits(x, 'feature_v')
        pref_v = F.softmax(logits_v, dim=-1)
        weights_v = torch.einsum('bs,bsn->bn', importance, pref_v)

        _, idx_v = torch.topk(weights_v, self.top_k_feature_v, dim=-1)
        idx_v = idx_v.sort(dim=-1).values
        # Update usage EMA
        self.neuron_router.update_usage(idx_v, 'feature_v', self.n_feature_v)

        # Relational Q routing
        logits_rel_q = self.neuron_router.get_logits(x, 'relational_q')
        pref_rel_q = F.softmax(logits_rel_q, dim=-1)
        weights_rel_q = torch.einsum('bs,bsn->bn', importance, pref_rel_q)

        _, idx_q = torch.topk(weights_rel_q, self.top_k_relational, dim=-1)
        idx_q = idx_q.sort(dim=-1).values
        # Update usage EMA
        self.neuron_router.update_usage(idx_q, 'relational_q', self.n_relational_q)

        # Relational K routing
        logits_rel_k = self.neuron_router.get_logits(x, 'relational_k')
        pref_rel_k = F.softmax(logits_rel_k, dim=-1)
        weights_rel_k = torch.einsum('bs,bsn->bn', importance, pref_rel_k)

        _, idx_k = torch.topk(weights_rel_k, self.top_k_relational, dim=-1)
        idx_k = idx_k.sort(dim=-1).values
        # Update usage EMA
        self.neuron_router.update_usage(idx_k, 'relational_k', self.n_relational_k)

        # Value routing
        logits_val = self.neuron_router.get_logits(x, 'value')
        pref_val = F.softmax(logits_val, dim=-1)
        weights_val = torch.einsum('bs,bsn->bn', importance, pref_val)

        _, idx_v2 = torch.topk(weights_val, self.top_k_value, dim=-1)
        idx_v2 = idx_v2.sort(dim=-1).values
        # Update usage EMA
        self.neuron_router.update_usage(idx_v2, 'value', self.n_value)

        # Compute aux_loss (load balance)
        aux_loss = 0.0
        if self.training:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                count = mask.sum() + 1e-8
                usage_fqk = (pref_qk * mask).sum(dim=(0, 1)) / count
                usage_fv = (pref_v * mask).sum(dim=(0, 1)) / count
                usage_rq = (pref_rel_q * mask).sum(dim=(0, 1)) / count
                usage_rk = (pref_rel_k * mask).sum(dim=(0, 1)) / count
                usage_val = (pref_val * mask).sum(dim=(0, 1)) / count
            else:
                usage_fqk = pref_qk.mean(dim=(0, 1))
                usage_fv = pref_v.mean(dim=(0, 1))
                usage_rq = pref_rel_q.mean(dim=(0, 1))
                usage_rk = pref_rel_k.mean(dim=(0, 1))
                usage_val = pref_val.mean(dim=(0, 1))

            target_fqk = 1.0 / self.n_feature_qk
            aux_loss = aux_loss + ((usage_fqk - target_fqk) ** 2).sum() * self.n_feature_qk

            target_fv = 1.0 / self.n_feature_v
            aux_loss = aux_loss + ((usage_fv - target_fv) ** 2).sum() * self.n_feature_v

            target_rq = 1.0 / self.n_relational_q
            aux_loss = aux_loss + ((usage_rq - target_rq) ** 2).sum() * self.n_relational_q

            target_rk = 1.0 / self.n_relational_k
            aux_loss = aux_loss + ((usage_rk - target_rk) ** 2).sum() * self.n_relational_k

            target_val = 1.0 / self.n_value
            aux_loss = aux_loss + ((usage_val - target_val) ** 2).sum() * self.n_value

        routing_info = {
            'idx_qk': idx_qk.detach(),
            'idx_v': idx_v.detach(),
            'idx_q': idx_q.detach(),
            'idx_k': idx_k.detach(),
            'idx_v2': idx_v2.detach(),
            'feature_qk_pref': pref_qk.detach(),
            'feature_v_pref': pref_v.detach(),
            'relational_q_pref': pref_rel_q.detach(),
            'relational_k_pref': pref_rel_k.detach(),
            'value_pref': pref_val.detach(),
        }

        return idx_qk, idx_v, idx_q, idx_k, idx_v2, routing_info, aux_loss


class NeuronCircuit(nn.Module):
    """
    v17: Attention circuit with ALL vector neurons

    Flow:
    1. x → feature_qk_neurons[idx_qk].T → h_qk (compression: 320→64)
    2. x → feature_v_neurons[idx_v].T → h_v (compression: 320→32)
    3. h_qk @ relational_q_neurons[idx_q] → Q (expansion: 64→320)
    4. h_qk @ relational_k_neurons[idx_k] → K (expansion: 64→320)
    5. h_v @ value_neurons[idx_v2] → V (expansion: 32→320)
    6. Multi-head attention
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Output projection (kept from v16)
        self.expand_O = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, idx_qk, idx_v, idx_q, idx_k, idx_v2, attention_mask=None):
        """
        Args:
            x: [B, S, D]
            idx_qk: [B, top_k_qk] - Feature QK indices
            idx_v: [B, top_k_v] - Feature V indices
            idx_q: [B, top_k_rel] - Relational Q indices
            idx_k: [B, top_k_rel] - Relational K indices
            idx_v2: [B, top_k_val] - Value indices
            attention_mask: [B, S] optional
        """
        B, S, D = x.shape

        # 1. Feature QK compression: x → h_qk
        # selected_qk: [B, top_k_qk, d_model]
        selected_qk = self.shared_neurons.feature_qk_neurons[idx_qk]
        W_qk = selected_qk.transpose(-1, -2)  # [B, D, top_k_qk]
        h_qk = torch.einsum('bsd,bdk->bsk', x, W_qk)  # [B, S, top_k_qk]

        # 2. Feature V compression: x → h_v
        selected_v = self.shared_neurons.feature_v_neurons[idx_v]  # [B, top_k_v, D]
        W_v = selected_v.transpose(-1, -2)  # [B, D, top_k_v]
        h_v = torch.einsum('bsd,bdk->bsk', x, W_v)  # [B, S, top_k_v]

        # 3. Relational Q expansion: h_qk → Q
        # selected_rel_q: [B, top_k_rel, d_model]
        selected_rel_q = self.shared_neurons.relational_q_neurons[idx_q]
        Q = torch.einsum('bsr,brd->bsd', h_qk, selected_rel_q)  # [B, S, D]

        # 4. Relational K expansion: h_qk → K
        selected_rel_k = self.shared_neurons.relational_k_neurons[idx_k]
        K = torch.einsum('bsr,brd->bsd', h_qk, selected_rel_k)  # [B, S, D]

        # 5. Value expansion: h_v → V
        selected_val = self.shared_neurons.value_neurons[idx_v2]  # [B, top_k_val, D]
        V = torch.einsum('bsr,brd->bsd', h_v, selected_val)  # [B, S, D]

        # 6. Multi-head Attention
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

        # 7. Output projection
        output = self.expand_O(attn_out)
        output = self.out_dropout(output)

        return output, None


class NeuronMemory(nn.Module):
    """
    v17: 2-stage hierarchical knowledge retrieval (same as v16)
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
    """DAWN v17 block with full vector neurons"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        n_knowledge: int,
        knowledge_rank: int = None,
        coarse_k: int = 20,
        fine_k: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = NeuronCircuit(shared_neurons, d_model, n_heads, dropout)
        self.memory = NeuronMemory(
            shared_neurons, d_model, n_knowledge,
            knowledge_rank=knowledge_rank, coarse_k=coarse_k, fine_k=fine_k
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, importance, global_routers: GlobalRouters, knowledge_encoder, attention_mask=None):
        normed_x = self.norm1(x)
        idx_qk, idx_v, idx_q, idx_k, idx_v2, attn_routing, attn_aux_loss = \
            global_routers.get_attention_weights(normed_x, importance, attention_mask)

        attn_out, _ = self.attn(normed_x, idx_qk, idx_v, idx_q, idx_k, idx_v2, attention_mask)
        x = x + attn_out

        normed_x2 = self.norm2(x)
        mem_out, knowledge_info = self.memory(normed_x2, global_routers.neuron_router, knowledge_encoder, attention_mask)
        x = x + self.dropout(mem_out)

        attn_out_norm = attn_out.norm(dim=-1).mean().detach()
        mem_out_norm = mem_out.norm(dim=-1).mean().detach()

        routing_info = {
            'attention': {**attn_routing},
            'memory': {**{k: v.detach() for k, v in knowledge_info.items()}},
            'attn_out_norm': attn_out_norm,
            'mem_out_norm': mem_out_norm,
        }

        return x, routing_info, attn_aux_loss


class DAWN(nn.Module):
    """
    DAWN v17: Full Vector Neurons

    Core Changes from v16:
    - ALL neurons are vectors (no rank matrices)
    - 5 separate routing: feature_qk, feature_v, relational_q, relational_k, value
    - Excitability (SAR) for balanced neuron usage
    - 82% parameter reduction vs v15

    Architecture:
    - Feature QK: [n_feature_qk, d_model] compression vectors
    - Feature V: [n_feature_v, d_model] compression vectors
    - Relational Q: [n_relational_q, d_model] expansion vectors
    - Relational K: [n_relational_k, d_model] expansion vectors
    - Value: [n_value, d_model] expansion vectors
    - Knowledge: 2-stage retrieval
    """
    __version__ = "17.0"

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 320,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 512,
        # Compression neurons
        n_feature_qk: int = 128,
        n_feature_v: int = 64,
        top_k_feature_qk: int = 64,
        top_k_feature_v: int = 32,
        # Expansion neurons
        n_relational_q: int = 256,
        n_relational_k: int = 256,
        n_value: int = 128,
        top_k_relational: int = 64,
        top_k_value: int = 32,
        # Knowledge
        n_knowledge: int = 80,
        coarse_k: int = 20,
        fine_k: int = 10,
        knowledge_rank: int = 128,
        # Other
        state_dim: int = 64,
        d_space: int = 64,
        dropout: float = 0.1,
        router_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        use_ssm_context: bool = True,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.knowledge_rank = knowledge_rank
        self.max_seq_len = max_seq_len
        self.state_dim = state_dim
        self.d_space = d_space
        self.router_dropout = router_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.use_ssm_context = use_ssm_context

        # Compression neuron counts
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.top_k_feature_qk = top_k_feature_qk
        self.top_k_feature_v = top_k_feature_v

        # Expansion neuron counts
        self.n_relational_q = n_relational_q
        self.n_relational_k = n_relational_k
        self.n_value = n_value
        self.top_k_relational = top_k_relational
        self.top_k_value = top_k_value

        # Knowledge
        self.n_knowledge = n_knowledge
        self.coarse_k = coarse_k
        self.fine_k = fine_k

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.shared_neurons = SharedNeurons(
            d_model=d_model,
            n_feature_qk=n_feature_qk, n_feature_v=n_feature_v,
            n_relational_q=n_relational_q, n_relational_k=n_relational_k,
            n_value=n_value, n_knowledge=n_knowledge,
            knowledge_rank=knowledge_rank,
        )

        self.global_ssm = GlobalSSM(d_model, state_dim, return_context=use_ssm_context)

        self.global_routers = GlobalRouters(
            d_model, n_feature_qk, n_feature_v, n_relational_q, n_relational_k,
            n_value, n_knowledge,
            top_k_feature_qk, top_k_feature_v, top_k_relational, top_k_value,
            d_space=d_space, router_dropout=router_dropout
        )

        self.knowledge_encoder = nn.Linear(d_model, knowledge_rank, bias=False)

        self.layers = nn.ModuleList([
            DAWNBlock(
                shared_neurons=self.shared_neurons, d_model=d_model, n_heads=n_heads,
                n_knowledge=n_knowledge, knowledge_rank=knowledge_rank,
                coarse_k=coarse_k, fine_k=fine_k, dropout=dropout,
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

    def knowledge_diversity_loss(self):
        K = self.shared_neurons.knowledge_neurons_K
        K_norm = F.normalize(K, dim=-1)
        sim = K_norm @ K_norm.T
        mask = ~torch.eye(self.n_knowledge, dtype=torch.bool, device=K.device)
        return sim[mask].abs().mean()

    def get_auxiliary_losses(self):
        return {
            'knowledge_div': self.knowledge_diversity_loss(),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_by_component(self):
        feature_qk = self.shared_neurons.feature_qk_neurons.numel()
        feature_v = self.shared_neurons.feature_v_neurons.numel()
        relational_q = self.shared_neurons.relational_q_neurons.numel()
        relational_k = self.shared_neurons.relational_k_neurons.numel()
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

        # expand_O per layer
        expand_per_layer = self.layers[0].attn.expand_O.weight.numel()
        expand_total = expand_per_layer * self.n_layers

        norms = sum(p.numel() for n, p in self.named_parameters() if 'norm' in n)

        print(f"=== DAWN v17 Parameter Breakdown (Full Vector Neurons) ===")
        print(f"Feature QK Neurons:     {feature_qk:,} ({feature_qk/1e3:.1f}K) [{self.n_feature_qk} × {self.d_model}]")
        print(f"Feature V Neurons:      {feature_v:,} ({feature_v/1e3:.1f}K) [{self.n_feature_v} × {self.d_model}]")
        print(f"Relational Q Neurons:   {relational_q:,} ({relational_q/1e3:.1f}K) [{self.n_relational_q} × {self.d_model}]")
        print(f"Relational K Neurons:   {relational_k:,} ({relational_k/1e3:.1f}K) [{self.n_relational_k} × {self.d_model}]")
        print(f"Value Neurons:          {value:,} ({value/1e3:.1f}K) [{self.n_value} × {self.d_model}]")
        print(f"Expand O (per layer):   {expand_total:,} ({expand_total/1e3:.1f}K)")
        print(f"Knowledge Neurons (K):  {knowledge:,} ({knowledge/1e3:.1f}K)")
        print(f"Embeddings:             {embed:,} ({embed/1e6:.2f}M)")
        print(f"Mamba SSM:              {ssm_total:,} ({ssm_total/1e3:.1f}K)")
        print(f"Unified Router:         {routers:,} ({routers/1e3:.1f}K) [d_space={self.d_space}]")
        print(f"LayerNorms:             {norms:,} ({norms/1e3:.1f}K)")
        print(f"---")
        print(f"Top-k Feature QK: {self.top_k_feature_qk}/{self.n_feature_qk}")
        print(f"Top-k Feature V:  {self.top_k_feature_v}/{self.n_feature_v}")
        print(f"Top-k Relational: {self.top_k_relational}/{self.n_relational_q}")
        print(f"Top-k Value:      {self.top_k_value}/{self.n_value}")
        print(f"Mamba Available: {MAMBA_AVAILABLE}")
        print(f"---")
        print(f"Total:                  {self.count_parameters():,} ({self.count_parameters()/1e6:.2f}M)")

        return {
            'feature_qk': feature_qk, 'feature_v': feature_v,
            'relational_q': relational_q, 'relational_k': relational_k,
            'value': value, 'expand': expand_total, 'knowledge': knowledge,
            'embeddings': embed, 'ssm': ssm_total,
            'routers': routers, 'norms': norms,
        }

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'knowledge_rank': self.knowledge_rank,
            'max_seq_len': self.max_seq_len,
            'n_feature_qk': self.n_feature_qk, 'n_feature_v': self.n_feature_v,
            'top_k_feature_qk': self.top_k_feature_qk, 'top_k_feature_v': self.top_k_feature_v,
            'n_relational_q': self.n_relational_q, 'n_relational_k': self.n_relational_k,
            'n_value': self.n_value,
            'top_k_relational': self.top_k_relational, 'top_k_value': self.top_k_value,
            'n_knowledge': self.n_knowledge,
            'coarse_k': self.coarse_k, 'fine_k': self.fine_k,
            'state_dim': self.state_dim,
            'd_space': self.d_space,
            'router_dropout': self.router_dropout,
            'gradient_checkpointing': self.gradient_checkpointing,
        }

    def get_model_info(self):
        """Return model architecture info for logging"""
        return [
            f"DAWN v{self.__version__}: Full Vector Neurons",
            f"  d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  Compression:",
            f"    Feature QK: {self.n_feature_qk} × {self.d_model} (top-k={self.top_k_feature_qk})",
            f"    Feature V: {self.n_feature_v} × {self.d_model} (top-k={self.top_k_feature_v})",
            f"  Expansion:",
            f"    Relational Q: {self.n_relational_q} × {self.d_model} (top-k={self.top_k_relational})",
            f"    Relational K: {self.n_relational_k} × {self.d_model} (top-k={self.top_k_relational})",
            f"    Value: {self.n_value} × {self.d_model} (top-k={self.top_k_value})",
            f"  Knowledge: {self.n_knowledge} (coarse={self.coarse_k} → fine={self.fine_k})",
            f"  Router: d_space={self.d_space}, Excitability (SAR)",
        ]
