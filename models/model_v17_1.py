"""
DAWN v17.1: Unified Neuron Architecture (v16.4 기반, Q/K 공유)

Attention과 Knowledge 모두 Feature-Restore 뉴런 패턴 사용:
- AttentionCircuit: Feature_QK/V -> Restore_QK/V (Q/K 공유 풀)
- KnowledgeCircuit: Feature_Know -> Restore_Know (새로운 구조)

Key changes from v16.4:
- NeuronCircuit -> AttentionCircuit (rename)
- NeuronMemory -> KnowledgeCircuit (Feature-Restore pattern)
- knowledge_encoder 제거
- coarse_k, fine_k 제거 -> top_k_knowledge 추가
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed, using slow for-loop SSM")


class UnifiedNeuronRouter(nn.Module):
    """
    v17: 6개 attention projection + knowledge projection
    """
    def __init__(self, d_model, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v, n_knowledge,
                 d_space=64, dropout=0.1, excitability_tau=1.5, excitability_ema_alpha=0.01,
                 excitability_decay_rate=0.99995):
        super().__init__()
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_knowledge = n_knowledge
        self.d_space = d_space
        self.ema_alpha = excitability_ema_alpha

        total_neurons = n_feature_qk + n_feature_v + n_restore_qk + n_restore_v + n_knowledge
        self.total_neurons = total_neurons

        # Index boundaries
        self.feature_qk_end = n_feature_qk
        self.feature_v_end = n_feature_qk + n_feature_v
        self.restore_qk_end = n_feature_qk + n_feature_v + n_restore_qk
        self.restore_v_end = n_feature_qk + n_feature_v + n_restore_qk + n_restore_v

        # 6 attention projections + 1 knowledge projection
        self.proj_all = nn.Linear(d_model, d_space * 6)  # fqk_Q, fqk_K, fv, rqk_Q, rqk_K, rv
        self.proj_knowledge = nn.Linear(d_model, d_space)
        self.dropout = nn.Dropout(dropout)

        # Unified neuron embeddings
        self.neuron_emb = nn.Parameter(torch.randn(total_neurons, d_space) * 0.02)

        # Usage tracking
        self.register_buffer('usage_ema_feature_qk', torch.zeros(n_feature_qk))
        self.register_buffer('usage_ema_feature_v', torch.zeros(n_feature_v))
        self.register_buffer('usage_ema_restore_qk', torch.zeros(n_restore_qk))
        self.register_buffer('usage_ema_restore_v', torch.zeros(n_restore_v))
        self.register_buffer('usage_ema_knowledge', torch.zeros(n_knowledge))

        # Excitability
        self.tau = excitability_tau
        self.decay_rate = excitability_decay_rate
        self.register_buffer('excitability_weight', torch.tensor(1.0))

    def decay_excitability(self):
        self.excitability_weight.mul_(self.decay_rate)

    def get_excitability(self, usage_ema):
        return torch.clamp(1.0 - usage_ema / self.tau, min=0.0, max=1.0)

    def get_logits(self, x, neuron_type):
        """Knowledge logits only"""
        if neuron_type != 'knowledge':
            raise ValueError(f"Use get_all_logits for {neuron_type}")

        emb_norm = F.normalize(self.neuron_emb, dim=-1)
        h_proj = self.dropout(self.proj_knowledge(x))
        knowledge_emb = emb_norm[self.restore_v_end:]

        logits = torch.einsum('bsd,nd->bsn', h_proj, knowledge_emb)
        if self.training:
            excitability = self.get_excitability(self.usage_ema_knowledge)
            logits = logits + excitability * self.excitability_weight
        return logits

    def get_all_logits(self, x):
        """6 attention logits at once"""
        emb_norm = F.normalize(self.neuron_emb, dim=-1)

        all_proj = self.dropout(self.proj_all(x))
        h_fqk_Q, h_fqk_K, h_fv, h_rqk_Q, h_rqk_K, h_rv = all_proj.chunk(6, dim=-1)

        fqk_emb = emb_norm[:self.feature_qk_end]
        fv_emb = emb_norm[self.feature_qk_end:self.feature_v_end]
        rqk_emb = emb_norm[self.feature_v_end:self.restore_qk_end]
        rv_emb = emb_norm[self.restore_qk_end:self.restore_v_end]

        logits_fqk_Q = torch.einsum('bsd,nd->bsn', h_fqk_Q, fqk_emb)
        logits_fqk_K = torch.einsum('bsd,nd->bsn', h_fqk_K, fqk_emb)
        logits_fv = torch.einsum('bsd,nd->bsn', h_fv, fv_emb)
        logits_rqk_Q = torch.einsum('bsd,nd->bsn', h_rqk_Q, rqk_emb)
        logits_rqk_K = torch.einsum('bsd,nd->bsn', h_rqk_K, rqk_emb)
        logits_rv = torch.einsum('bsd,nd->bsn', h_rv, rv_emb)

        if self.training:
            w = self.excitability_weight
            exc_fqk = self.get_excitability(self.usage_ema_feature_qk) * w
            exc_fv = self.get_excitability(self.usage_ema_feature_v) * w
            exc_rqk = self.get_excitability(self.usage_ema_restore_qk) * w
            exc_rv = self.get_excitability(self.usage_ema_restore_v) * w

            logits_fqk_Q = logits_fqk_Q + exc_fqk
            logits_fqk_K = logits_fqk_K + exc_fqk
            logits_fv = logits_fv + exc_fv
            logits_rqk_Q = logits_rqk_Q + exc_rqk
            logits_rqk_K = logits_rqk_K + exc_rqk
            logits_rv = logits_rv + exc_rv

        return logits_fqk_Q, logits_fqk_K, logits_fv, logits_rqk_Q, logits_rqk_K, logits_rv

    def update_usage(self, weights, neuron_type, attention_mask=None):
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

        decay = 1 - self.ema_alpha
        if neuron_type == 'feature_qk':
            self.usage_ema_feature_qk.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'feature_v':
            self.usage_ema_feature_v.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'restore_qk':
            self.usage_ema_restore_qk.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'restore_v':
            self.usage_ema_restore_v.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'knowledge':
            self.usage_ema_knowledge.mul_(decay).add_(usage, alpha=self.ema_alpha)


class SharedNeurons(nn.Module):
    """
    v17: Attention + Knowledge 모두 Feature-Restore 패턴
    """
    def __init__(
        self,
        d_model: int,
        rank: int,
        n_feature_qk: int,
        n_feature_v: int,
        n_restore_qk: int,
        n_restore_v: int,
        n_knowledge: int,
        knowledge_rank: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.knowledge_rank = knowledge_rank
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_knowledge = n_knowledge

        # Attention neurons (contiguous memory)
        self.f_neurons = nn.Parameter(torch.zeros(n_feature_qk + n_feature_v, d_model, rank))
        self.r_neurons = nn.Parameter(torch.zeros(n_restore_qk + n_restore_v, rank, d_model))

        # Knowledge neurons: Feature-Restore pattern (NEW in v17)
        self.feature_know = nn.Parameter(torch.zeros(n_knowledge, d_model, knowledge_rank))
        self.restore_know = nn.Parameter(torch.zeros(n_knowledge, knowledge_rank, d_model))

        self._init_parameters()

    @property
    def feature_qk_neurons(self):
        return self.f_neurons[:self.n_feature_qk]

    @property
    def feature_v_neurons(self):
        return self.f_neurons[self.n_feature_qk:]

    @property
    def restore_qk_neurons(self):
        return self.r_neurons[:self.n_restore_qk]

    @property
    def restore_v_neurons(self):
        return self.r_neurons[self.n_restore_qk:]

    def _init_parameters(self):
        # Attention neurons
        for i in range(self.n_feature_qk + self.n_feature_v):
            nn.init.orthogonal_(self.f_neurons.data[i])
        for i in range(self.n_restore_qk + self.n_restore_v):
            nn.init.orthogonal_(self.r_neurons.data[i])
        # Knowledge neurons (orthogonal init)
        for i in range(self.n_knowledge):
            nn.init.orthogonal_(self.feature_know.data[i])
            nn.init.orthogonal_(self.restore_know.data[i])


class GlobalSSM(nn.Module):
    """Selective SSM (same as v16.4)"""
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
    """v17 routing with shared pools"""
    def __init__(self, d_model: int, n_feature_qk: int, n_feature_v: int,
                 n_restore_qk: int, n_restore_v: int, n_knowledge: int,
                 top_k_feature_qk: int = 8, top_k_feature_v: int = 8,
                 top_k_restore_qk: int = 8, top_k_restore_v: int = 8,
                 top_k_knowledge: int = 4,
                 d_space: int = 64, router_dropout: float = 0.1, token_routing: bool = False,
                 excitability_tau: float = 1.5, excitability_ema_alpha: float = 0.01,
                 excitability_decay_rate: float = 0.99995):
        super().__init__()
        self.d_model = d_model
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_knowledge = n_knowledge
        self.top_k_feature_qk = top_k_feature_qk
        self.top_k_feature_v = top_k_feature_v
        self.top_k_restore_qk = top_k_restore_qk
        self.top_k_restore_v = top_k_restore_v
        self.top_k_knowledge = top_k_knowledge
        self.token_routing = token_routing

        self.neuron_router = UnifiedNeuronRouter(
            d_model, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v, n_knowledge,
            d_space=d_space, dropout=router_dropout,
            excitability_tau=excitability_tau, excitability_ema_alpha=excitability_ema_alpha,
            excitability_decay_rate=excitability_decay_rate
        )

    def _topk_sparsify(self, weights, k):
        topk_vals, topk_idx = torch.topk(weights, k, dim=-1)
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(-1, topk_idx, topk_vals)
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-8)
        return sparse_weights, topk_idx

    def get_attention_weights(self, x, importance, attention_mask=None):
        (fqk_logits_Q, fqk_logits_K, fv_logits,
         rqk_logits_Q, rqk_logits_K, rv_logits) = self.neuron_router.get_all_logits(x)

        fqk_pref_Q = F.softmax(fqk_logits_Q, dim=-1)
        fqk_pref_K = F.softmax(fqk_logits_K, dim=-1)
        fv_pref = F.softmax(fv_logits, dim=-1)
        rqk_pref_Q = F.softmax(rqk_logits_Q, dim=-1)
        rqk_pref_K = F.softmax(rqk_logits_K, dim=-1)
        rv_pref = F.softmax(rv_logits, dim=-1)

        aux_loss = 0.0
        if self.training:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                count = mask.sum() + 1e-8
                usage_fqk_Q = (fqk_pref_Q * mask).sum(dim=(0, 1)) / count
                usage_fqk_K = (fqk_pref_K * mask).sum(dim=(0, 1)) / count
                usage_fv = (fv_pref * mask).sum(dim=(0, 1)) / count
                usage_rqk_Q = (rqk_pref_Q * mask).sum(dim=(0, 1)) / count
                usage_rqk_K = (rqk_pref_K * mask).sum(dim=(0, 1)) / count
                usage_rv = (rv_pref * mask).sum(dim=(0, 1)) / count
            else:
                usage_fqk_Q = fqk_pref_Q.mean(dim=(0, 1))
                usage_fqk_K = fqk_pref_K.mean(dim=(0, 1))
                usage_fv = fv_pref.mean(dim=(0, 1))
                usage_rqk_Q = rqk_pref_Q.mean(dim=(0, 1))
                usage_rqk_K = rqk_pref_K.mean(dim=(0, 1))
                usage_rv = rv_pref.mean(dim=(0, 1))

            target_fqk = 1.0 / self.n_feature_qk
            target_fv = 1.0 / self.n_feature_v
            target_rqk = 1.0 / self.n_restore_qk
            target_rv = 1.0 / self.n_restore_v

            aux_loss += ((usage_fqk_Q - target_fqk) ** 2).sum() * self.n_feature_qk
            aux_loss += ((usage_fqk_K - target_fqk) ** 2).sum() * self.n_feature_qk
            aux_loss += ((usage_fv - target_fv) ** 2).sum() * self.n_feature_v
            aux_loss += ((usage_rqk_Q - target_rqk) ** 2).sum() * self.n_restore_qk
            aux_loss += ((usage_rqk_K - target_rqk) ** 2).sum() * self.n_restore_qk
            aux_loss += ((usage_rv - target_rv) ** 2).sum() * self.n_restore_v

        if self.token_routing:
            fqk_weights_Q = fqk_pref_Q
            fqk_weights_K = fqk_pref_K
            fv_weights = fv_pref
            rqk_weights_Q = rqk_pref_Q
            rqk_weights_K = rqk_pref_K
            rv_weights = rv_pref

            routing_info = {
                'fqk_weights_Q': fqk_weights_Q.detach(),
                'fqk_weights_K': fqk_weights_K.detach(),
                'fv_weights': fv_weights.detach(),
                'rqk_weights_Q': rqk_weights_Q.detach(),
                'rqk_weights_K': rqk_weights_K.detach(),
                'rv_weights': rv_weights.detach(),
                'token_routing': True,
            }
        else:
            # Batch-level routing
            fqk_weights_Q_dense = torch.einsum('bs,bsn->bn', importance, fqk_pref_Q)
            fqk_weights_K_dense = torch.einsum('bs,bsn->bn', importance, fqk_pref_K)
            fv_weights_dense = torch.einsum('bs,bsn->bn', importance, fv_pref)
            rqk_weights_Q_dense = torch.einsum('bs,bsn->bn', importance, rqk_pref_Q)
            rqk_weights_K_dense = torch.einsum('bs,bsn->bn', importance, rqk_pref_K)
            rv_weights_dense = torch.einsum('bs,bsn->bn', importance, rv_pref)

            fqk_weights_Q, _ = self._topk_sparsify(fqk_weights_Q_dense, self.top_k_feature_qk)
            fqk_weights_K, _ = self._topk_sparsify(fqk_weights_K_dense, self.top_k_feature_qk)
            fv_weights, _ = self._topk_sparsify(fv_weights_dense, self.top_k_feature_v)
            rqk_weights_Q, _ = self._topk_sparsify(rqk_weights_Q_dense, self.top_k_restore_qk)
            rqk_weights_K, _ = self._topk_sparsify(rqk_weights_K_dense, self.top_k_restore_qk)
            rv_weights, _ = self._topk_sparsify(rv_weights_dense, self.top_k_restore_v)

            routing_info = {
                'fqk_weights_Q': fqk_weights_Q.detach(),
                'fqk_weights_K': fqk_weights_K.detach(),
                'fv_weights': fv_weights.detach(),
                'rqk_weights_Q': rqk_weights_Q.detach(),
                'rqk_weights_K': rqk_weights_K.detach(),
                'rv_weights': rv_weights.detach(),
                'fqk_q_pref': fqk_pref_Q.detach(),
                'fqk_k_pref': fqk_pref_K.detach(),
                'fv_pref': fv_pref.detach(),
                'rqk_q_pref': rqk_pref_Q.detach(),
                'rqk_k_pref': rqk_pref_K.detach(),
                'rv_pref': rv_pref.detach(),
                'token_routing': False,
            }

        # Update usage
        if self.training:
            fqk_used = ((fqk_weights_Q > 0) | (fqk_weights_K > 0)).float()
            self.neuron_router.update_usage(fqk_used, 'feature_qk', attention_mask)
            self.neuron_router.update_usage(fv_weights, 'feature_v', attention_mask)
            rqk_used = ((rqk_weights_Q > 0) | (rqk_weights_K > 0)).float()
            self.neuron_router.update_usage(rqk_used, 'restore_qk', attention_mask)
            self.neuron_router.update_usage(rv_weights, 'restore_v', attention_mask)

        return fqk_weights_Q, fqk_weights_K, fv_weights, rqk_weights_Q, rqk_weights_K, rv_weights, routing_info, aux_loss

    def get_knowledge_weights(self, x, importance, attention_mask=None):
        """Get knowledge neuron weights using batch-level routing"""
        k_logits = self.neuron_router.get_logits(x, 'knowledge')
        k_pref = F.softmax(k_logits, dim=-1)

        # Batch-level routing
        k_weights_dense = torch.einsum('bs,bsn->bn', importance, k_pref)
        k_weights, topk_idx = self._topk_sparsify(k_weights_dense, self.top_k_knowledge)

        # Update usage
        if self.training:
            self.neuron_router.update_usage(k_weights, 'knowledge', attention_mask)

        return k_weights, topk_idx


class AttentionCircuit(nn.Module):
    """v17 attention circuit (renamed from NeuronCircuit)"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank = rank

        self.expand_O = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, fqk_weights_Q, fqk_weights_K, fv_weights, rqk_weights_Q, rqk_weights_K, rv_weights, attention_mask=None):
        B, S, D = x.shape
        R = self.rank
        token_routing = fqk_weights_Q.dim() == 3

        if token_routing:
            # Token-level routing
            shared_fqk_Q = torch.einsum('bsn,ndr->bsdr', fqk_weights_Q, self.shared_neurons.feature_qk_neurons)
            shared_fqk_K = torch.einsum('bsn,ndr->bsdr', fqk_weights_K, self.shared_neurons.feature_qk_neurons)
            shared_fv = torch.einsum('bsn,ndr->bsdr', fv_weights, self.shared_neurons.feature_v_neurons)

            h_q = torch.einsum('bsd,bsdr->bsr', x, shared_fqk_Q)
            h_k = torch.einsum('bsd,bsdr->bsr', x, shared_fqk_K)
            h_v = torch.einsum('bsd,bsdr->bsr', x, shared_fv)

            shared_rqk_Q = torch.einsum('bsn,nrd->bsrd', rqk_weights_Q, self.shared_neurons.restore_qk_neurons)
            shared_rqk_K = torch.einsum('bsn,nrd->bsrd', rqk_weights_K, self.shared_neurons.restore_qk_neurons)
            shared_rv = torch.einsum('bsn,nrd->bsrd', rv_weights, self.shared_neurons.restore_v_neurons)

            Q = torch.einsum('bsr,bsrd->bsd', h_q, shared_rqk_Q)
            K = torch.einsum('bsr,bsrd->bsd', h_k, shared_rqk_K)
            V = torch.einsum('bsr,bsrd->bsd', h_v, shared_rv)
        else:
            # Batch-level routing - matmul optimized
            fqk_flat = self.shared_neurons.feature_qk_neurons.view(-1, D * R)
            shared_fqk_Q = (fqk_weights_Q @ fqk_flat).view(B, D, R)
            shared_fqk_K = (fqk_weights_K @ fqk_flat).view(B, D, R)

            fv_flat = self.shared_neurons.feature_v_neurons.view(-1, D * R)
            shared_fv = (fv_weights @ fv_flat).view(B, D, R)

            h_q = torch.bmm(x, shared_fqk_Q)
            h_k = torch.bmm(x, shared_fqk_K)
            h_v = torch.bmm(x, shared_fv)

            rqk_flat = self.shared_neurons.restore_qk_neurons.view(-1, R * D)
            shared_rqk_Q = (rqk_weights_Q @ rqk_flat).view(B, R, D)
            shared_rqk_K = (rqk_weights_K @ rqk_flat).view(B, R, D)

            rv_flat = self.shared_neurons.restore_v_neurons.view(-1, R * D)
            shared_rv = (rv_weights @ rv_flat).view(B, R, D)

            Q = torch.bmm(h_q, shared_rqk_Q)
            K = torch.bmm(h_k, shared_rqk_K)
            V = torch.bmm(h_v, shared_rv)

        # Multi-head Attention
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

        output = self.expand_O(attn_out)
        output = self.out_dropout(output)

        return output, None


class KnowledgeCircuit(nn.Module):
    """
    v17 Knowledge Circuit: Feature-Restore pattern (NEW)

    x -> router selects knowledge neurons (top_k)
      -> Feature_Know extracts info (d_model -> knowledge_rank)
      -> Restore_Know restores knowledge (knowledge_rank -> d_model)
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_knowledge: int,
        knowledge_rank: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_knowledge = n_knowledge
        self.knowledge_rank = knowledge_rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, k_weights, attention_mask=None):
        """
        Args:
            x: [B, S, D]
            k_weights: [B, n_knowledge] sparse weights from router
        """
        B, S, D = x.shape
        R = self.knowledge_rank

        # Feature-Restore pattern (matmul optimized)
        # Feature: [n_knowledge, d_model, knowledge_rank] -> [B, d_model, knowledge_rank]
        feature_flat = self.shared_neurons.feature_know.view(-1, D * R)
        shared_feature = (k_weights @ feature_flat).view(B, D, R)

        # Compress: x @ Feature -> h
        h = torch.bmm(x, shared_feature)  # [B, S, R]

        # Restore: [n_knowledge, knowledge_rank, d_model] -> [B, knowledge_rank, d_model]
        restore_flat = self.shared_neurons.restore_know.view(-1, R * D)
        shared_restore = (k_weights @ restore_flat).view(B, R, D)

        # Expand: h @ Restore -> output
        output = torch.bmm(h, shared_restore)  # [B, S, D]
        output = self.dropout(output)

        return output


class DAWNBlock(nn.Module):
    """DAWN v17 block"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_knowledge: int,
        knowledge_rank: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = AttentionCircuit(shared_neurons, d_model, n_heads, rank, dropout)
        self.knowledge = KnowledgeCircuit(shared_neurons, d_model, n_knowledge, knowledge_rank, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, importance, global_routers: GlobalRouters, attention_mask=None):
        # Attention
        normed_x = self.norm1(x)
        fqk_w_Q, fqk_w_K, fv_w, rqk_w_Q, rqk_w_K, rv_w, attn_routing, attn_aux_loss = \
            global_routers.get_attention_weights(normed_x, importance, attention_mask)

        attn_out, _ = self.attn(normed_x, fqk_w_Q, fqk_w_K, fv_w, rqk_w_Q, rqk_w_K, rv_w, attention_mask)
        x = x + attn_out

        # Knowledge
        normed_x2 = self.norm2(x)
        k_weights, k_topk_idx = global_routers.get_knowledge_weights(normed_x2, importance, attention_mask)
        know_out = self.knowledge(normed_x2, k_weights, attention_mask)
        x = x + know_out

        routing_info = {
            'attention': {**attn_routing},
            'knowledge': {
                'weights': k_weights.detach(),
                'topk_indices': k_topk_idx.detach(),
            },
            'attn_out_norm': attn_out.norm(dim=-1).mean().detach(),
            'know_out_norm': know_out.norm(dim=-1).mean().detach(),
        }

        return x, routing_info, attn_aux_loss


class DAWN(nn.Module):
    """
    DAWN v17: Unified Neuron Architecture

    Both Attention and Knowledge use Feature-Restore pattern:
    - AttentionCircuit: Feature_QK/V -> Restore_QK/V
    - KnowledgeCircuit: Feature_Know -> Restore_Know
    """
    __version__ = "17.1"

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 384,
        n_layers: int = 12,
        n_heads: int = 6,
        rank: int = 64,
        max_seq_len: int = 512,
        # Attention Feature
        n_feature_qk: int = 96,
        n_feature_v: int = 24,
        top_k_feature_qk: int = 8,
        top_k_feature_v: int = 3,
        # Attention Restore
        n_restore_qk: int = 96,
        n_restore_v: int = 24,
        top_k_restore_qk: int = 8,
        top_k_restore_v: int = 3,
        # Knowledge (Feature-Restore)
        n_knowledge: int = 24,
        knowledge_rank: int = 128,
        top_k_knowledge: int = 4,
        # Other
        state_dim: int = 64,
        d_space: int = 64,
        dropout: float = 0.1,
        router_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        token_routing: bool = False,
        use_ssm_context: bool = True,
        # Excitability
        excitability_tau: float = 1.5,
        excitability_ema_alpha: float = 0.01,
        excitability_decay_rate: float = 0.99995,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.rank = rank
        self.knowledge_rank = knowledge_rank
        self.max_seq_len = max_seq_len
        self.state_dim = state_dim
        self.d_space = d_space
        self.token_routing = token_routing
        self.router_dropout = router_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.use_ssm_context = use_ssm_context
        self.excitability_tau = excitability_tau
        self.excitability_ema_alpha = excitability_ema_alpha

        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.top_k_feature_qk = top_k_feature_qk
        self.top_k_feature_v = top_k_feature_v

        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.top_k_restore_qk = top_k_restore_qk
        self.top_k_restore_v = top_k_restore_v

        self.n_knowledge = n_knowledge
        self.top_k_knowledge = top_k_knowledge

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.shared_neurons = SharedNeurons(
            d_model=d_model, rank=rank,
            n_feature_qk=n_feature_qk, n_feature_v=n_feature_v,
            n_restore_qk=n_restore_qk, n_restore_v=n_restore_v,
            n_knowledge=n_knowledge, knowledge_rank=knowledge_rank,
        )

        self.global_ssm = GlobalSSM(d_model, state_dim, return_context=use_ssm_context)

        self.global_routers = GlobalRouters(
            d_model, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v, n_knowledge,
            top_k_feature_qk, top_k_feature_v, top_k_restore_qk, top_k_restore_v,
            top_k_knowledge=top_k_knowledge,
            d_space=d_space, router_dropout=router_dropout, token_routing=token_routing,
            excitability_tau=excitability_tau, excitability_ema_alpha=excitability_ema_alpha,
            excitability_decay_rate=excitability_decay_rate
        )

        self.layers = nn.ModuleList([
            DAWNBlock(
                shared_neurons=self.shared_neurons, d_model=d_model, n_heads=n_heads,
                rank=rank, n_knowledge=n_knowledge, knowledge_rank=knowledge_rank,
                dropout=dropout,
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
                    layer, x, importance, self.global_routers, attention_mask,
                    use_reentrant=False
                )
            else:
                x, routing_info, layer_aux_loss = layer(x, importance, self.global_routers, attention_mask)

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
        # Attention orthogonality
        I = torch.eye(self.rank, device=self.shared_neurons.f_neurons.device).unsqueeze(0)

        W_fqk = self.shared_neurons.feature_qk_neurons
        WtW_fqk = torch.bmm(W_fqk.transpose(1, 2), W_fqk)
        loss_fqk = ((WtW_fqk - I) ** 2).mean()

        W_fv = self.shared_neurons.feature_v_neurons
        WtW_fv = torch.bmm(W_fv.transpose(1, 2), W_fv)
        loss_fv = ((WtW_fv - I) ** 2).mean()

        W_rqk = self.shared_neurons.restore_qk_neurons
        WWt_rqk = torch.bmm(W_rqk, W_rqk.transpose(1, 2))
        loss_rqk = ((WWt_rqk - I) ** 2).mean()

        W_rv = self.shared_neurons.restore_v_neurons
        WWt_rv = torch.bmm(W_rv, W_rv.transpose(1, 2))
        loss_rv = ((WWt_rv - I) ** 2).mean()

        # Knowledge orthogonality (NEW in v17)
        I_know = torch.eye(self.knowledge_rank, device=self.shared_neurons.feature_know.device).unsqueeze(0)

        W_fknow = self.shared_neurons.feature_know
        WtW_fknow = torch.bmm(W_fknow.transpose(1, 2), W_fknow)
        loss_fknow = ((WtW_fknow - I_know) ** 2).mean()

        W_rknow = self.shared_neurons.restore_know
        WWt_rknow = torch.bmm(W_rknow, W_rknow.transpose(1, 2))
        loss_rknow = ((WWt_rknow - I_know) ** 2).mean()

        # 6 terms average
        return (loss_fqk + loss_fv + loss_rqk + loss_rv + loss_fknow + loss_rknow) / 6

    def knowledge_diversity_loss(self):
        # Feature_Know neuron diversity
        F = self.shared_neurons.feature_know  # [n_knowledge, d_model, rank]
        F_flat = F.view(self.n_knowledge, -1)  # [n_knowledge, d_model * rank]
        F_norm = F.normalize(F_flat, dim=-1)
        sim = F_norm @ F_norm.T
        mask = ~torch.eye(self.n_knowledge, dtype=torch.bool, device=F.device)
        return sim[mask].abs().mean()

    def get_auxiliary_losses(self):
        return {
            'orth_total': self.orthogonality_loss(),
            'knowledge_div': self.knowledge_diversity_loss(),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'rank': self.rank, 'knowledge_rank': self.knowledge_rank,
            'max_seq_len': self.max_seq_len,
            'n_feature_qk': self.n_feature_qk, 'n_feature_v': self.n_feature_v,
            'top_k_feature_qk': self.top_k_feature_qk, 'top_k_feature_v': self.top_k_feature_v,
            'n_restore_qk': self.n_restore_qk, 'n_restore_v': self.n_restore_v,
            'top_k_restore_qk': self.top_k_restore_qk, 'top_k_restore_v': self.top_k_restore_v,
            'n_knowledge': self.n_knowledge, 'top_k_knowledge': self.top_k_knowledge,
            'state_dim': self.state_dim, 'd_space': self.d_space,
            'router_dropout': self.router_dropout,
            'gradient_checkpointing': self.gradient_checkpointing,
            'token_routing': self.token_routing,
        }

    def get_model_info(self):
        return [
            f"DAWN v{self.__version__}: Unified Neuron Architecture",
            f"  rank={self.rank}, knowledge_rank={self.knowledge_rank}",
            f"  Feature_QK: {self.n_feature_qk} (top-k={self.top_k_feature_qk}) - Q/K shared pool",
            f"  Feature_V: {self.n_feature_v} (top-k={self.top_k_feature_v})",
            f"  Restore_QK: {self.n_restore_qk} (top-k={self.top_k_restore_qk}) - Q/K shared pool",
            f"  Restore_V: {self.n_restore_v} (top-k={self.top_k_restore_v})",
            f"  Knowledge: {self.n_knowledge} neurons (top-k={self.top_k_knowledge}) - Feature-Restore pattern",
            f"  Router: d_space={self.d_space}, proj_all(6) + proj_knowledge(1)",
        ]
