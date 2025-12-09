"""
DAWN v17: Full Vector Neurons + Full Soft Selection

Core Changes from v16:
- ALL neurons are vectors (no rank matrices)
- 3 shared pools with separate routing heads:
  * Feature: compression neurons (SHARED for QK/V routing)
  * Relational: expansion neurons (SHARED for Q/K routing)
  * Value: expansion neurons
- Full Soft Selection (both training AND inference):
  * All neurons participate via softmax
  * Gradient flows to all neurons (training)
  * No top-k sparsification
- Temperature parameter for controlling selection sharpness

Architecture (FRV v17):
- Feature QK: x → soft selection → h_qk (shared pool, QK routing)
- Feature V:  x → soft selection → h_v (shared pool, V routing)
- Relational Q: h_qk @ soft-weighted neurons → Q (shared pool, Q routing)
- Relational K: h_qk @ soft-weighted neurons → K (shared pool, K routing)
- Value: h_v @ soft-weighted neurons → V
- Knowledge: 2-stage retrieval (same as v16)

Full soft selection:
  weights = softmax(logits / temp) → full weighted sum over ALL neurons
  output = neurons @ soft_weights (gradient flows to all neurons)
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
    v17: Unified neuron router with shared pools + Excitability

    Neuron types:
    - feature: compression axis vectors (SHARED for QK/V, routing differs)
    - relational: expansion vectors for Q/K (SHARED, routing differs)
    - value: expansion vectors for V
    - knowledge: memory retrieval

    Excitability (refractory-inspired):
    - Less-used neurons get higher excitability bonus
    - Encourages balanced neuron usage during training
    """
    def __init__(self, d_model, n_feature, n_relational,
                 n_value, n_knowledge, d_space=64, dropout=0.1):
        super().__init__()
        self.n_feature = n_feature  # Shared for QK/V
        self.n_relational = n_relational  # Shared for Q/K
        self.n_value = n_value
        self.n_knowledge = n_knowledge
        self.d_space = d_space

        total_neurons = n_feature + n_relational + n_value + n_knowledge
        self.total_neurons = total_neurons

        # Index boundaries
        self.feature_end = n_feature
        self.relational_end = n_feature + n_relational
        self.value_end = n_feature + n_relational + n_value
        # knowledge is value_end ~ total_neurons

        # Shared projection
        self.proj = nn.Linear(d_model, d_space)
        self.dropout = nn.Dropout(dropout)

        # Unified neuron embeddings [total_neurons, d_space]
        self.neuron_emb = nn.Parameter(torch.randn(total_neurons, d_space) * 0.02)

        # Separate routing heads for shared pools
        # Feature V routing (shares same pool as Feature QK, different selection)
        self.neuron_emb_feature_v = nn.Parameter(torch.randn(n_feature, d_space) * 0.02)
        # Relational K routing (shares same pool as Relational Q, different selection)
        self.neuron_emb_relational_k = nn.Parameter(torch.randn(n_relational, d_space) * 0.02)

        # Usage tracking for excitability
        self.register_buffer('usage_ema_feature', torch.zeros(n_feature))  # Shared for QK/V
        self.register_buffer('usage_ema_relational', torch.zeros(n_relational))  # Shared for Q/K
        self.register_buffer('usage_ema_value', torch.zeros(n_value))
        self.register_buffer('usage_ema_knowledge', torch.zeros(n_knowledge))

        # Excitability: tau (recovery time constant) + decaying weight
        self.tau = 1.5
        self.excitability_weight = 0.0  # Disabled by default

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
            if neuron_type == 'feature':
                self.usage_ema_feature.mul_(0.99).add_(usage, alpha=0.01)
            elif neuron_type == 'relational':
                self.usage_ema_relational.mul_(0.99).add_(usage, alpha=0.01)
            elif neuron_type == 'value':
                self.usage_ema_value.mul_(0.99).add_(usage, alpha=0.01)
            elif neuron_type == 'knowledge':
                self.usage_ema_knowledge.mul_(0.99).add_(usage, alpha=0.01)

    @torch.compiler.disable
    def update_usage_soft(self, weights, neuron_type):
        """Update usage EMA for soft selection (weight-based)

        weights: [B, n_neurons] - soft selection weights for all neurons
        """
        if not self.training:
            return

        with torch.no_grad():
            # Average weights across batch
            usage = weights.detach().mean(dim=0)  # [n_neurons]

            # In-place update: ema = 0.99 * ema + 0.01 * usage
            if neuron_type == 'feature':
                self.usage_ema_feature.mul_(0.99).add_(usage, alpha=0.01)
            elif neuron_type == 'relational':
                self.usage_ema_relational.mul_(0.99).add_(usage, alpha=0.01)
            elif neuron_type == 'value':
                self.usage_ema_value.mul_(0.99).add_(usage, alpha=0.01)
            elif neuron_type == 'knowledge':
                self.usage_ema_knowledge.mul_(0.99).add_(usage, alpha=0.01)

    def get_logits(self, x, neuron_type):
        """
        x: [B, S, d_model] or [B, S, rank] for expansion types
        neuron_type: 'feature_qk', 'feature_v', 'relational_q', 'relational_k', 'value', 'knowledge'

        Shared pools with separate routing heads:
        - feature_qk and feature_v: SHARED feature pool, SEPARATE routing heads
        - relational_q and relational_k: SHARED relational pool, SEPARATE routing heads
        """
        h_proj = self.proj(x)  # [B, S, d_space]
        h_proj = self.dropout(h_proj)

        # Type-specific slicing + Excitability
        if neuron_type == 'feature_qk':
            # QK routing: uses unified neuron_emb for feature section
            neuron_emb_norm = F.normalize(self.neuron_emb[:self.feature_end], dim=-1)
            logits = torch.einsum('bsd,nd->bsn', h_proj, neuron_emb_norm)
            if self.training:
                excitability = self.get_excitability(self.usage_ema_feature)
                logits = logits + excitability * self.excitability_weight

        elif neuron_type == 'feature_v':
            # V routing: uses SEPARATE neuron_emb_feature_v for different selection (same pool)
            neuron_emb_norm = F.normalize(self.neuron_emb_feature_v, dim=-1)
            logits = torch.einsum('bsd,nd->bsn', h_proj, neuron_emb_norm)
            if self.training:
                excitability = self.get_excitability(self.usage_ema_feature)
                logits = logits + excitability * self.excitability_weight

        elif neuron_type == 'relational_q':
            # Q routing: uses unified neuron_emb for relational section
            neuron_emb_norm = F.normalize(self.neuron_emb[self.feature_end:self.relational_end], dim=-1)
            logits = torch.einsum('bsd,nd->bsn', h_proj, neuron_emb_norm)
            if self.training:
                excitability = self.get_excitability(self.usage_ema_relational)
                logits = logits + excitability * self.excitability_weight

        elif neuron_type == 'relational_k':
            # K routing: uses SEPARATE neuron_emb_relational_k for different selection
            neuron_emb_norm = F.normalize(self.neuron_emb_relational_k, dim=-1)
            logits = torch.einsum('bsd,nd->bsn', h_proj, neuron_emb_norm)
            if self.training:
                excitability = self.get_excitability(self.usage_ema_relational)
                logits = logits + excitability * self.excitability_weight

        elif neuron_type == 'value':
            neuron_emb_norm = F.normalize(self.neuron_emb[self.relational_end:self.value_end], dim=-1)
            logits = torch.einsum('bsd,nd->bsn', h_proj, neuron_emb_norm)
            if self.training:
                excitability = self.get_excitability(self.usage_ema_value)
                logits = logits + excitability * self.excitability_weight

        elif neuron_type == 'knowledge':
            neuron_emb_norm = F.normalize(self.neuron_emb[self.value_end:], dim=-1)
            logits = torch.einsum('bsd,nd->bsn', h_proj, neuron_emb_norm)
            if self.training:
                excitability = self.get_excitability(self.usage_ema_knowledge)
                logits = logits + excitability * self.excitability_weight

        return logits


class SharedNeurons(nn.Module):
    """
    v17: ALL neurons are vectors (no rank matrices)

    Compression neurons:
    - feature_neurons: [n_feature, d_model] (SHARED for QK/V, routing differs)

    Expansion neurons:
    - relational_neurons: [n_relational, d_model] (SHARED for Q/K)
    - value_neurons: [n_value, d_model]

    Knowledge neurons:
    - knowledge_neurons_K: [n_knowledge, knowledge_rank]
    - knowledge_neurons_V: [n_knowledge, d_model]
    """
    def __init__(
        self,
        d_model: int,
        n_feature: int,
        n_relational: int,
        n_value: int,
        n_knowledge: int,
        knowledge_rank: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else 128
        self.n_feature = n_feature  # Shared for QK/V
        self.n_relational = n_relational  # Shared for Q/K
        self.n_value = n_value
        self.n_knowledge = n_knowledge

        # Compression neurons (input → hidden) - SHARED for QK/V
        self.feature_neurons = nn.Parameter(torch.randn(n_feature, d_model) * 0.02)

        # Expansion neurons (hidden → output)
        self.relational_neurons = nn.Parameter(torch.randn(n_relational, d_model) * 0.02)  # Shared Q/K
        self.value_neurons = nn.Parameter(torch.randn(n_value, d_model) * 0.02)

        # Knowledge neurons
        self.knowledge_neurons_K = nn.Parameter(torch.zeros(n_knowledge, self.knowledge_rank))
        self.knowledge_neurons_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_parameters()

    def _init_parameters(self):
        # All vector neurons: normal init
        nn.init.normal_(self.feature_neurons, std=0.02)
        nn.init.normal_(self.relational_neurons, std=0.02)
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
    v17: 4-type routing with Soft/Hard Selection

    Soft/Hard Selection:
    - Training: soft selection (ALL neurons via softmax, gradient flow to all)
    - Inference: top-k hard selection (sparse, efficient)

    Routes:
    - feature_qk: [B, n_feature] soft weights OR [B, top_k] indices (SHARED pool, QK routing)
    - feature_v: [B, n_feature] soft weights OR [B, top_k] indices (SHARED pool, V routing)
    - relational: [B, n_relational] soft weights OR [B, top_k] indices (SHARED pool, Q/K routing)
    - value: [B, n_value] soft weights OR [B, top_k] indices
    """
    def __init__(self, d_model: int, n_feature: int,
                 n_relational: int, n_value: int, n_knowledge: int,
                 top_k_feature: int = 64,
                 top_k_relational: int = 64, top_k_value: int = 32,
                 d_space: int = 64, router_dropout: float = 0.1,
                 temperature: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.n_feature = n_feature  # Shared for QK/V
        self.n_relational = n_relational  # Shared for Q/K
        self.n_value = n_value
        self.n_knowledge = n_knowledge
        self.top_k_feature = top_k_feature  # Used only during inference
        self.top_k_relational = top_k_relational
        self.top_k_value = top_k_value
        self.temperature = temperature  # Controls softmax sharpness

        self.neuron_router = UnifiedNeuronRouter(
            d_model, n_feature, n_relational,
            n_value, n_knowledge, d_space=d_space, dropout=router_dropout
        )

    def get_attention_weights(self, x, importance, attention_mask=None):
        """
        Compute attention routing weights with Full Soft Selection.

        Both training and inference use soft selection:
        - Returns soft weights for ALL neurons [B, n_neurons]
        - Gradient flows to all neurons (training)
        - Full softmax over all neurons (no top-k sparsification)

        Returns:
            selection_data: dict containing soft weights for all neurons
            routing_info: dict with routing statistics
            aux_loss: scalar
        """
        # Feature QK routing (shared pool, QK routing head)
        logits_qk = self.neuron_router.get_logits(x, 'feature_qk')  # [B, S, n_feature]
        pref_qk = F.softmax(logits_qk, dim=-1)
        weights_qk = torch.einsum('bs,bsn->bn', importance, pref_qk)  # [B, n_feature]

        # Feature V routing (shared pool, V routing head)
        logits_v = self.neuron_router.get_logits(x, 'feature_v')
        pref_v = F.softmax(logits_v, dim=-1)
        weights_v = torch.einsum('bs,bsn->bn', importance, pref_v)  # [B, n_feature]

        # Relational Q routing (shared pool, Q routing head)
        logits_rel_q = self.neuron_router.get_logits(x, 'relational_q')
        pref_rel_q = F.softmax(logits_rel_q, dim=-1)
        weights_rel_q = torch.einsum('bs,bsn->bn', importance, pref_rel_q)  # [B, n_relational]

        # Relational K routing (shared pool, K routing head)
        logits_rel_k = self.neuron_router.get_logits(x, 'relational_k')
        pref_rel_k = F.softmax(logits_rel_k, dim=-1)
        weights_rel_k = torch.einsum('bs,bsn->bn', importance, pref_rel_k)  # [B, n_relational]

        # Value routing
        logits_val = self.neuron_router.get_logits(x, 'value')
        pref_val = F.softmax(logits_val, dim=-1)
        weights_val = torch.einsum('bs,bsn->bn', importance, pref_val)  # [B, n_value]

        # SOFT SELECTION: all neurons participate via softmax (both train and inference)
        soft_qk = F.softmax(weights_qk / self.temperature, dim=-1)    # [B, n_feature]
        soft_v = F.softmax(weights_v / self.temperature, dim=-1)      # [B, n_feature]
        soft_q = F.softmax(weights_rel_q / self.temperature, dim=-1)  # [B, n_relational]
        soft_k = F.softmax(weights_rel_k / self.temperature, dim=-1)  # [B, n_relational]
        soft_v2 = F.softmax(weights_val / self.temperature, dim=-1)   # [B, n_value]

        # Update usage EMA (soft selection) - only during training
        if self.training:
            self.neuron_router.update_usage_soft(soft_qk, 'feature')
            self.neuron_router.update_usage_soft(soft_v, 'feature')
            self.neuron_router.update_usage_soft(soft_q, 'relational')
            self.neuron_router.update_usage_soft(soft_k, 'relational')
            self.neuron_router.update_usage_soft(soft_v2, 'value')

        selection_data = {
            'mode': 'soft',
            'soft_qk': soft_qk,    # [B, n_feature]
            'soft_v': soft_v,      # [B, n_feature]
            'soft_q': soft_q,      # [B, n_relational]
            'soft_k': soft_k,      # [B, n_relational]
            'soft_v2': soft_v2,    # [B, n_value]
        }

        # Compute aux_loss (load balance) - only during training
        aux_loss = 0.0
        if self.training:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                count = mask.sum() + 1e-8
                usage_fqk = (pref_qk * mask).sum(dim=(0, 1)) / count
                usage_fv = (pref_v * mask).sum(dim=(0, 1)) / count
                usage_rel_q = (pref_rel_q * mask).sum(dim=(0, 1)) / count
                usage_rel_k = (pref_rel_k * mask).sum(dim=(0, 1)) / count
                usage_val = (pref_val * mask).sum(dim=(0, 1)) / count
            else:
                usage_fqk = pref_qk.mean(dim=(0, 1))
                usage_fv = pref_v.mean(dim=(0, 1))
                usage_rel_q = pref_rel_q.mean(dim=(0, 1))
                usage_rel_k = pref_rel_k.mean(dim=(0, 1))
                usage_val = pref_val.mean(dim=(0, 1))

            # Shared feature pool: both QK and V contribute to load balance
            target_f = 1.0 / self.n_feature
            aux_loss = aux_loss + ((usage_fqk - target_f) ** 2).sum() * self.n_feature
            aux_loss = aux_loss + ((usage_fv - target_f) ** 2).sum() * self.n_feature

            # Shared relational pool: both Q and K contribute to load balance
            target_rel = 1.0 / self.n_relational
            aux_loss = aux_loss + ((usage_rel_q - target_rel) ** 2).sum() * self.n_relational
            aux_loss = aux_loss + ((usage_rel_k - target_rel) ** 2).sum() * self.n_relational

            target_val = 1.0 / self.n_value
            aux_loss = aux_loss + ((usage_val - target_val) ** 2).sum() * self.n_value

        routing_info = {
            'mode': 'soft',
            'feature_qk_pref': pref_qk.detach(),
            'feature_v_pref': pref_v.detach(),
            'relational_q_pref': pref_rel_q.detach(),
            'relational_k_pref': pref_rel_k.detach(),
            'value_pref': pref_val.detach(),
        }

        return selection_data, routing_info, aux_loss


class NeuronCircuit(nn.Module):
    """
    v17: Attention circuit with Full Soft Selection

    Always uses soft selection (both training and inference):
    - Weighted sum over ALL neurons: output = sum(soft_weights * neurons)
    - Gradient flows to all neurons (training)
    - Full softmax over all neurons (no top-k sparsification)

    Flow:
    1. x → feature_neurons @ soft_qk → h_qk [B, S, n_feature]
    2. x → feature_neurons @ soft_v → h_v [B, S, n_feature]
    3. h_qk → proj_to_relational → h_qk_rel [B, S, n_relational]
    4. h_v → proj_to_value → h_v_val [B, S, n_value]
    5. h_qk_rel @ relational_neurons @ soft_q → Q
    6. h_qk_rel @ relational_neurons @ soft_k → K
    7. h_v_val @ value_neurons @ soft_v2 → V
    8. Multi-head attention
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

        n_feature = shared_neurons.n_feature
        n_relational = shared_neurons.n_relational
        n_value = shared_neurons.n_value

        # Projection from feature hidden dim to expansion dims
        # h_qk [B, S, n_feature] → [B, S, n_relational] for Q/K expansion
        self.proj_to_relational = nn.Linear(n_feature, n_relational, bias=False)
        # h_v [B, S, n_feature] → [B, S, n_value] for V expansion
        self.proj_to_value = nn.Linear(n_feature, n_value, bias=False)

        # Output projection
        self.expand_O = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, selection_data, attention_mask=None):
        """
        Args:
            x: [B, S, D]
            selection_data: dict with soft selection weights
            attention_mask: [B, S] optional
        """
        B, S, D = x.shape

        # SOFT SELECTION: weighted sum over all neurons (both train and inference)
        soft_qk = selection_data['soft_qk']  # [B, n_feature]
        soft_v = selection_data['soft_v']    # [B, n_feature]
        soft_q = selection_data['soft_q']    # [B, n_relational]
        soft_k = selection_data['soft_k']    # [B, n_relational]
        soft_v2 = selection_data['soft_v2']  # [B, n_value]

        # 1. Feature QK compression: x @ (soft_qk * neurons).T → h_qk
        # neurons: [n_feature, D], soft_qk: [B, n_feature]
        # weighted_neurons_qk: [B, n_feature, D]
        weighted_neurons_qk = self.shared_neurons.feature_neurons.unsqueeze(0) * soft_qk.unsqueeze(-1)
        # W_qk: [B, D, n_feature] (effectively weighted projection)
        W_qk = weighted_neurons_qk.transpose(-1, -2)
        h_qk = torch.einsum('bsd,bdk->bsk', x, W_qk)  # [B, S, n_feature]

        # 2. Feature V compression: x @ (soft_v * neurons).T → h_v
        weighted_neurons_v = self.shared_neurons.feature_neurons.unsqueeze(0) * soft_v.unsqueeze(-1)
        W_v = weighted_neurons_v.transpose(-1, -2)
        h_v = torch.einsum('bsd,bdk->bsk', x, W_v)  # [B, S, n_feature]

        # 3. Project h_qk from n_feature → n_relational for Q/K expansion
        h_qk_rel = self.proj_to_relational(h_qk)  # [B, S, n_relational]

        # 4. Project h_v from n_feature → n_value for V expansion
        h_v_val = self.proj_to_value(h_v)  # [B, S, n_value]

        # 5. Relational Q expansion: h_qk_rel @ (soft_q * neurons) → Q
        weighted_rel_q = self.shared_neurons.relational_neurons.unsqueeze(0) * soft_q.unsqueeze(-1)
        Q = torch.einsum('bsr,brd->bsd', h_qk_rel, weighted_rel_q)  # [B, S, D]

        # 6. Relational K expansion: h_qk_rel @ (soft_k * neurons) → K
        weighted_rel_k = self.shared_neurons.relational_neurons.unsqueeze(0) * soft_k.unsqueeze(-1)
        K = torch.einsum('bsr,brd->bsd', h_qk_rel, weighted_rel_k)  # [B, S, D]

        # 7. Value expansion: h_v_val @ (soft_v2 * neurons) → V
        weighted_val = self.shared_neurons.value_neurons.unsqueeze(0) * soft_v2.unsqueeze(-1)
        V = torch.einsum('bsr,brd->bsd', h_v_val, weighted_val)  # [B, S, D]

        # 8. Multi-head Attention
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

    @torch.compiler.disable
    def _update_knowledge_usage(self, router, candidate_idx, B, S, attention_mask=None):
        """Update knowledge neuron usage EMA (index-based)"""
        with torch.no_grad():
            # candidate_idx: [B, S, coarse_k]
            flat_indices = candidate_idx.detach().view(-1)  # [B * S * coarse_k]
            usage = torch.zeros(self.n_knowledge, device=candidate_idx.device)
            ones = torch.ones_like(flat_indices, dtype=usage.dtype)
            usage.scatter_add_(0, flat_indices, ones)

            # Normalize by B*S (each position selects coarse_k neurons)
            if attention_mask is not None:
                valid_positions = attention_mask.sum().item()
            else:
                valid_positions = B * S
            usage = usage / max(valid_positions, 1)

            # In-place EMA update
            router.usage_ema_knowledge.mul_(0.99).add_(usage, alpha=0.01)

    def forward(self, x, router, knowledge_encoder, attention_mask=None):
        B, S, D = x.shape

        # Stage 1: Coarse candidate selection via router
        k_logits = router.get_logits(x, 'knowledge')
        coarse_scores, candidate_idx = torch.topk(k_logits, self.coarse_k, dim=-1)

        # Update knowledge usage (index-based, [B, S, coarse_k] -> flatten)
        if self.training:
            self._update_knowledge_usage(router, candidate_idx, B, S, attention_mask)

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
    """DAWN v17 block with Soft/Hard Selection"""
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
        selection_data, attn_routing, attn_aux_loss = \
            global_routers.get_attention_weights(normed_x, importance, attention_mask)

        attn_out, _ = self.attn(normed_x, selection_data, attention_mask)
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
    DAWN v17: Full Vector Neurons + Full Soft Selection

    Core Changes from v16:
    - ALL neurons are vectors (no rank matrices)
    - Full Soft Selection (both training AND inference):
      * All neurons participate via softmax
      * Gradient flows to all neurons (training)
      * No top-k sparsification
    - Temperature parameter for controlling selection sharpness
    - Shared pools: feature (QK/V), relational (Q/K) with separate routing heads

    Architecture:
    - Feature: [n_feature, d_model] SHARED pool (QK/V have separate routing heads)
    - Relational: [n_relational, d_model] SHARED pool (Q/K have separate routing heads)
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
        n_feature: int = 768,  # Shared for QK/V
        top_k_feature: int = 64,  # Inference only
        # Expansion neurons
        n_relational: int = 256,  # Shared for Q/K
        n_value: int = 128,
        top_k_relational: int = 64,  # Inference only
        top_k_value: int = 32,  # Inference only
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
        temperature: float = 1.0,  # Soft selection sharpness
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
        self.temperature = temperature
        self.gradient_checkpointing = gradient_checkpointing
        self.use_ssm_context = use_ssm_context

        # Compression neuron counts (shared pool for QK/V)
        self.n_feature = n_feature
        self.top_k_feature = top_k_feature

        # Expansion neuron counts
        self.n_relational = n_relational  # Shared for Q/K
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
            n_feature=n_feature,
            n_relational=n_relational,
            n_value=n_value, n_knowledge=n_knowledge,
            knowledge_rank=knowledge_rank,
        )

        self.global_ssm = GlobalSSM(d_model, state_dim, return_context=use_ssm_context)

        self.global_routers = GlobalRouters(
            d_model, n_feature, n_relational,
            n_value, n_knowledge,
            top_k_feature, top_k_relational, top_k_value,
            d_space=d_space, router_dropout=router_dropout,
            temperature=temperature
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
        feature = self.shared_neurons.feature_neurons.numel()
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

        # expand_O per layer
        expand_per_layer = self.layers[0].attn.expand_O.weight.numel()
        expand_total = expand_per_layer * self.n_layers

        norms = sum(p.numel() for n, p in self.named_parameters() if 'norm' in n)

        print(f"=== DAWN v17 Parameter Breakdown (Vector Neurons + Full Soft Selection) ===")
        print(f"Feature Neurons:        {feature:,} ({feature/1e3:.1f}K) [{self.n_feature} × {self.d_model}] (SHARED QK/V)")
        print(f"Relational Neurons:     {relational:,} ({relational/1e3:.1f}K) [{self.n_relational} × {self.d_model}] (SHARED Q/K)")
        print(f"Value Neurons:          {value:,} ({value/1e3:.1f}K) [{self.n_value} × {self.d_model}]")
        print(f"Expand O (per layer):   {expand_total:,} ({expand_total/1e3:.1f}K)")
        print(f"Knowledge Neurons (K):  {knowledge:,} ({knowledge/1e3:.1f}K)")
        print(f"Embeddings:             {embed:,} ({embed/1e6:.2f}M)")
        print(f"Mamba SSM:              {ssm_total:,} ({ssm_total/1e3:.1f}K)")
        print(f"Unified Router:         {routers:,} ({routers/1e3:.1f}K) [d_space={self.d_space}]")
        print(f"LayerNorms:             {norms:,} ({norms/1e3:.1f}K)")
        print(f"---")
        print(f"Selection Mode: FULL SOFT (train & inference)")
        print(f"Temperature:      {self.temperature}")
        print(f"Mamba Available: {MAMBA_AVAILABLE}")
        print(f"---")
        print(f"Total:                  {self.count_parameters():,} ({self.count_parameters()/1e6:.2f}M)")

        return {
            'feature': feature,
            'relational': relational,
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
            'n_feature': self.n_feature,  # Shared for QK/V
            'top_k_feature': self.top_k_feature,  # Inference only
            'n_relational': self.n_relational,  # Shared for Q/K
            'n_value': self.n_value,
            'top_k_relational': self.top_k_relational,  # Inference only
            'top_k_value': self.top_k_value,  # Inference only
            'n_knowledge': self.n_knowledge,
            'coarse_k': self.coarse_k, 'fine_k': self.fine_k,
            'state_dim': self.state_dim,
            'd_space': self.d_space,
            'router_dropout': self.router_dropout,
            'temperature': self.temperature,
            'gradient_checkpointing': self.gradient_checkpointing,
        }

    def get_model_info(self):
        """Return model architecture info for logging"""
        return [
            f"DAWN v{self.__version__}: Vector Neurons + Full Soft Selection",
            f"  d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  Compression:",
            f"    Feature: {self.n_feature} × {self.d_model} [SHARED QK/V]",
            f"  Expansion:",
            f"    Relational: {self.n_relational} × {self.d_model} [SHARED Q/K]",
            f"    Value: {self.n_value} × {self.d_model}",
            f"  Knowledge: {self.n_knowledge} (coarse={self.coarse_k} → fine={self.fine_k})",
            f"  Selection: FULL SOFT (train & inference, temp={self.temperature})",
            f"  Router: d_space={self.d_space}, Excitability (SAR)",
        ]
