"""
DAWN v17: v16.3 + Knowledge Feature-Restore (Q/K/V 완전 분리)

v16.3 기반 + Knowledge도 Feature-Restore 패턴:
- Attention: Q/K/V 완전 분리 (v16.3 그대로)
  - FQ, FK, FV (압축) + RQ, RK, RV (복원)
- Knowledge: Feature-Restore 패턴 (새 구조)
  - feature_know: [n_knowledge, d_model, knowledge_rank]
  - restore_know: [n_knowledge, knowledge_rank, d_model]

Key changes from v16.3:
- NeuronMemory -> KnowledgeCircuit (Feature-Restore pattern)
- knowledge_encoder 제거
- coarse_k, fine_k 제거 -> top_k_knowledge 추가
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Mamba selective scan import (fallback 포함)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed, using slow for-loop SSM")


class UnifiedNeuronRouter(nn.Module):
    """
    v17: Complete Pool Separation Router (same as v16.3) + Knowledge routing

    6개의 독립적인 뉴런 풀을 라우팅:
    - FQ, FK, FV: 압축 뉴런
    - RQ, RK, RV: 복원 뉴런
    - Knowledge: Feature-Restore 패턴
    """
    def __init__(self, d_model, n_feature_q, n_feature_k, n_feature_v, n_restore_q, n_restore_k, n_restore_v, n_knowledge,
                 d_space=64, dropout=0.1, excitability_tau=1.5, excitability_ema_alpha=0.01,
                 excitability_decay_rate=0.99995):
        super().__init__()
        self.n_feature_q = n_feature_q
        self.n_feature_k = n_feature_k
        self.n_feature_v = n_feature_v
        self.n_restore_q = n_restore_q
        self.n_restore_k = n_restore_k
        self.n_restore_v = n_restore_v
        self.n_knowledge = n_knowledge
        self.d_space = d_space
        self.ema_alpha = excitability_ema_alpha

        total_neurons = n_feature_q + n_feature_k + n_feature_v + n_restore_q + n_restore_k + n_restore_v + n_knowledge
        self.total_neurons = total_neurons

        # 인덱스 경계
        self.feature_q_end = n_feature_q
        self.feature_k_end = n_feature_q + n_feature_k
        self.feature_v_end = n_feature_q + n_feature_k + n_feature_v
        self.restore_q_end = n_feature_q + n_feature_k + n_feature_v + n_restore_q
        self.restore_k_end = n_feature_q + n_feature_k + n_feature_v + n_restore_q + n_restore_k
        self.restore_v_end = n_feature_q + n_feature_k + n_feature_v + n_restore_q + n_restore_k + n_restore_v
        # knowledge는 restore_v_end ~ total_neurons

        # 통합 projection (6개) + knowledge 별도
        self.proj_all = nn.Linear(d_model, d_space * 6)  # feature_q, feature_k, feature_v, restore_q, restore_k, restore_v
        self.proj_knowledge = nn.Linear(d_model, d_space)
        self.dropout = nn.Dropout(dropout)

        # 통합 뉴런 임베딩 [total_neurons, d_space]
        self.neuron_emb = nn.Parameter(torch.randn(total_neurons, d_space) * 0.02)

        # 타입별 usage 추적
        self.register_buffer('usage_ema_feature_q', torch.zeros(n_feature_q))
        self.register_buffer('usage_ema_feature_k', torch.zeros(n_feature_k))
        self.register_buffer('usage_ema_feature_v', torch.zeros(n_feature_v))
        self.register_buffer('usage_ema_restore_q', torch.zeros(n_restore_q))
        self.register_buffer('usage_ema_restore_k', torch.zeros(n_restore_k))
        self.register_buffer('usage_ema_restore_v', torch.zeros(n_restore_v))
        self.register_buffer('usage_ema_knowledge', torch.zeros(n_knowledge))

        # Excitability: tau (recovery time constant) + decaying weight
        self.tau = excitability_tau
        self.decay_rate = excitability_decay_rate
        self.register_buffer('excitability_weight', torch.tensor(1.0))

    def decay_excitability(self):
        """Decay excitability_weight each step."""
        self.excitability_weight.mul_(self.decay_rate)

    def get_excitability(self, usage_ema):
        """Neuronal excitability based on usage."""
        return torch.clamp(1.0 - usage_ema / self.tau, min=0.0, max=1.0)

    def get_logits(self, x, neuron_type):
        """
        x: [B, S, d_model]
        neuron_type: 'knowledge' only (feature_q~restore_v는 get_all_logits 사용)
        """
        if neuron_type != 'knowledge':
            raise ValueError(f"Use get_all_logits for {neuron_type}. get_logits is for 'knowledge' only.")

        emb_norm = F.normalize(self.neuron_emb, dim=-1)
        h_proj = self.dropout(self.proj_knowledge(x))
        emb = emb_norm[self.restore_v_end:]

        logits = torch.einsum('bsd,nd->bsn', h_proj, emb)
        if self.training:
            excitability = self.get_excitability(self.usage_ema_knowledge)
            logits = logits + excitability * self.excitability_weight
        return logits

    def get_all_logits(self, x):
        """6개 풀 logits를 효율적으로 계산"""
        B, S, D = x.shape

        # 6개 projection 한번에 → chunk
        all_proj = self.dropout(self.proj_all(x))  # [B, S, d_space * 6]
        h_feature_q, h_feature_k, h_feature_v, h_restore_q, h_restore_k, h_restore_v = all_proj.chunk(6, dim=-1)

        # 뉴런 임베딩 정규화 (한 번만)
        emb_norm = F.normalize(self.neuron_emb[:self.restore_v_end], dim=-1)  # knowledge 제외

        # 각 타입별 emb 슬라이스 (view라 비용 없음)
        emb_feature_q = emb_norm[:self.feature_q_end]
        emb_feature_k = emb_norm[self.feature_q_end:self.feature_k_end]
        emb_feature_v = emb_norm[self.feature_k_end:self.feature_v_end]
        emb_restore_q = emb_norm[self.feature_v_end:self.restore_q_end]
        emb_restore_k = emb_norm[self.restore_q_end:self.restore_k_end]
        emb_restore_v = emb_norm[self.restore_k_end:self.restore_v_end]

        # 각각 자기 영역만 연산
        logits_feature_q = torch.einsum('bsd,nd->bsn', h_feature_q, emb_feature_q)
        logits_feature_k = torch.einsum('bsd,nd->bsn', h_feature_k, emb_feature_k)
        logits_feature_v = torch.einsum('bsd,nd->bsn', h_feature_v, emb_feature_v)
        logits_restore_q = torch.einsum('bsd,nd->bsn', h_restore_q, emb_restore_q)
        logits_restore_k = torch.einsum('bsd,nd->bsn', h_restore_k, emb_restore_k)
        logits_restore_v = torch.einsum('bsd,nd->bsn', h_restore_v, emb_restore_v)

        # excitability 추가
        if self.training:
            w = self.excitability_weight
            logits_feature_q = logits_feature_q + self.get_excitability(self.usage_ema_feature_q) * w
            logits_feature_k = logits_feature_k + self.get_excitability(self.usage_ema_feature_k) * w
            logits_feature_v = logits_feature_v + self.get_excitability(self.usage_ema_feature_v) * w
            logits_restore_q = logits_restore_q + self.get_excitability(self.usage_ema_restore_q) * w
            logits_restore_k = logits_restore_k + self.get_excitability(self.usage_ema_restore_k) * w
            logits_restore_v = logits_restore_v + self.get_excitability(self.usage_ema_restore_v) * w

        return logits_feature_q, logits_feature_k, logits_feature_v, logits_restore_q, logits_restore_k, logits_restore_v

    def update_usage(self, weights, neuron_type, attention_mask=None):
        """top-k 후 선택된 뉴런 사용량 업데이트"""
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

        # In-place update to avoid memory leak from buffer reassignment
        decay = 1 - self.ema_alpha
        if neuron_type == 'feature_q':
            self.usage_ema_feature_q.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'feature_k':
            self.usage_ema_feature_k.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'feature_v':
            self.usage_ema_feature_v.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'restore_q':
            self.usage_ema_restore_q.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'restore_k':
            self.usage_ema_restore_k.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'restore_v':
            self.usage_ema_restore_v.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'knowledge':
            self.usage_ema_knowledge.mul_(decay).add_(usage, alpha=self.ema_alpha)


class SharedNeurons(nn.Module):
    """
    v17: Complete Pool Separation + Knowledge Feature-Restore

    Attention neurons (same as v16.3):
    - FQ: [n_feature_q, d_model, rank] - Q 압축
    - FK: [n_feature_k, d_model, rank] - K 압축
    - FV: [n_feature_v, d_model, rank] - V 압축
    - RQ: [n_restore_q, rank, d_model] - Q 복원
    - RK: [n_restore_k, rank, d_model] - K 복원
    - RV: [n_restore_v, rank, d_model] - V 복원

    Knowledge neurons (NEW - Feature-Restore pattern):
    - feature_know: [n_knowledge, d_model, knowledge_rank] - 압축
    - restore_know: [n_knowledge, knowledge_rank, d_model] - 복원
    """
    def __init__(
        self,
        d_model: int,
        rank: int,
        n_feature_q: int,
        n_feature_k: int,
        n_feature_v: int,
        n_restore_q: int,
        n_restore_k: int,
        n_restore_v: int,
        n_knowledge: int,
        knowledge_rank: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else 128
        self.n_feature_q = n_feature_q
        self.n_feature_k = n_feature_k
        self.n_feature_v = n_feature_v
        self.n_restore_q = n_restore_q
        self.n_restore_k = n_restore_k
        self.n_restore_v = n_restore_v
        self.n_knowledge = n_knowledge

        # Contiguous memory for GPU cache efficiency
        # F group: [n_feature_q + n_feature_k + n_feature_v, d_model, rank]
        self.f_neurons = nn.Parameter(torch.zeros(n_feature_q + n_feature_k + n_feature_v, d_model, rank))
        # R group: [n_restore_q + n_restore_k + n_restore_v, rank, d_model]
        self.r_neurons = nn.Parameter(torch.zeros(n_restore_q + n_restore_k + n_restore_v, rank, d_model))

        # Knowledge neurons: Feature-Restore pattern (NEW in v17)
        self.feature_know = nn.Parameter(torch.zeros(n_knowledge, d_model, self.knowledge_rank))
        self.restore_know = nn.Parameter(torch.zeros(n_knowledge, self.knowledge_rank, d_model))

        self._init_parameters()

    # Properties for sliced access (contiguous memory)
    @property
    def feature_q_neurons(self):
        return self.f_neurons[:self.n_feature_q]

    @property
    def feature_k_neurons(self):
        return self.f_neurons[self.n_feature_q:self.n_feature_q + self.n_feature_k]

    @property
    def feature_v_neurons(self):
        return self.f_neurons[self.n_feature_q + self.n_feature_k:]

    @property
    def restore_q_neurons(self):
        return self.r_neurons[:self.n_restore_q]

    @property
    def restore_k_neurons(self):
        return self.r_neurons[self.n_restore_q:self.n_restore_q + self.n_restore_k]

    @property
    def restore_v_neurons(self):
        return self.r_neurons[self.n_restore_q + self.n_restore_k:]

    def _init_parameters(self):
        # Attention neurons: orthogonal init
        for i in range(self.n_feature_q + self.n_feature_k + self.n_feature_v):
            nn.init.orthogonal_(self.f_neurons.data[i])
        for i in range(self.n_restore_q + self.n_restore_k + self.n_restore_v):
            nn.init.orthogonal_(self.r_neurons.data[i])

        # Knowledge neurons: orthogonal init
        for i in range(self.n_knowledge):
            nn.init.orthogonal_(self.feature_know.data[i])
            nn.init.orthogonal_(self.restore_know.data[i])


class GlobalSSM(nn.Module):
    """Global SSM for importance scoring - v16.3/v17.1과 동일"""
    def __init__(self, d_model, state_dim=64, return_context=True):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.return_context = return_context

        # v16.3/v17.1과 동일
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
    v17: Global routing for all neuron types

    Attention: 6개 독립 풀 (v16.3 동일)
    Knowledge: top_k selection -> Feature-Restore
    """
    def __init__(self, d_model, n_feature_q, n_feature_k, n_feature_v, n_restore_q, n_restore_k, n_restore_v, n_knowledge,
                 top_k_feature_q=8, top_k_feature_k=8, top_k_feature_v=3, top_k_restore_q=8, top_k_restore_k=8, top_k_restore_v=3,
                 top_k_knowledge=4,
                 d_space=64, router_dropout=0.1, token_routing=False,
                 excitability_tau=1.5, excitability_ema_alpha=0.01, excitability_decay_rate=0.99995):
        super().__init__()
        self.top_k_feature_q = top_k_feature_q
        self.top_k_feature_k = top_k_feature_k
        self.top_k_feature_v = top_k_feature_v
        self.top_k_restore_q = top_k_restore_q
        self.top_k_restore_k = top_k_restore_k
        self.top_k_restore_v = top_k_restore_v
        self.top_k_knowledge = top_k_knowledge
        self.token_routing = token_routing

        self.neuron_router = UnifiedNeuronRouter(
            d_model, n_feature_q, n_feature_k, n_feature_v, n_restore_q, n_restore_k, n_restore_v, n_knowledge,
            d_space=d_space, dropout=router_dropout,
            excitability_tau=excitability_tau, excitability_ema_alpha=excitability_ema_alpha,
            excitability_decay_rate=excitability_decay_rate
        )

    def _topk_sparsify(self, weights, k):
        """Top-k sparsification with renormalization"""
        topk_vals, topk_idx = torch.topk(weights, k, dim=-1)
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(-1, topk_idx, topk_vals)
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-8)
        return sparse_weights, topk_idx

    def get_attention_weights(self, x, importance, attention_mask=None):
        """Get all 6 attention weights"""
        logits_feature_q, logits_feature_k, logits_feature_v, logits_restore_q, logits_restore_k, logits_restore_v = \
            self.neuron_router.get_all_logits(x)

        pref_feature_q = F.softmax(logits_feature_q, dim=-1)
        pref_feature_k = F.softmax(logits_feature_k, dim=-1)
        pref_feature_v = F.softmax(logits_feature_v, dim=-1)
        pref_restore_q = F.softmax(logits_restore_q, dim=-1)
        pref_restore_k = F.softmax(logits_restore_k, dim=-1)
        pref_restore_v = F.softmax(logits_restore_v, dim=-1)

        # Compute aux_loss (load balancing)
        aux_loss = 0.0
        if self.training:
            n_fq = self.neuron_router.n_feature_q
            n_fk = self.neuron_router.n_feature_k
            n_fv = self.neuron_router.n_feature_v
            n_rq = self.neuron_router.n_restore_q
            n_rk = self.neuron_router.n_restore_k
            n_rv = self.neuron_router.n_restore_v

            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                count = mask.sum() + 1e-8
                usage_fq = (pref_feature_q * mask).sum(dim=(0, 1)) / count
                usage_fk = (pref_feature_k * mask).sum(dim=(0, 1)) / count
                usage_fv = (pref_feature_v * mask).sum(dim=(0, 1)) / count
                usage_rq = (pref_restore_q * mask).sum(dim=(0, 1)) / count
                usage_rk = (pref_restore_k * mask).sum(dim=(0, 1)) / count
                usage_rv = (pref_restore_v * mask).sum(dim=(0, 1)) / count
            else:
                usage_fq = pref_feature_q.mean(dim=(0, 1))
                usage_fk = pref_feature_k.mean(dim=(0, 1))
                usage_fv = pref_feature_v.mean(dim=(0, 1))
                usage_rq = pref_restore_q.mean(dim=(0, 1))
                usage_rk = pref_restore_k.mean(dim=(0, 1))
                usage_rv = pref_restore_v.mean(dim=(0, 1))

            target_fq = 1.0 / n_fq
            target_fk = 1.0 / n_fk
            target_fv = 1.0 / n_fv
            target_rq = 1.0 / n_rq
            target_rk = 1.0 / n_rk
            target_rv = 1.0 / n_rv

            aux_loss += ((usage_fq - target_fq) ** 2).sum() * n_fq
            aux_loss += ((usage_fk - target_fk) ** 2).sum() * n_fk
            aux_loss += ((usage_fv - target_fv) ** 2).sum() * n_fv
            aux_loss += ((usage_rq - target_rq) ** 2).sum() * n_rq
            aux_loss += ((usage_rk - target_rk) ** 2).sum() * n_rk
            aux_loss += ((usage_rv - target_rv) ** 2).sum() * n_rv

        if self.token_routing:
            # Token-level routing
            feature_q_w, _ = self._topk_sparsify(pref_feature_q, self.top_k_feature_q)
            feature_k_w, _ = self._topk_sparsify(pref_feature_k, self.top_k_feature_k)
            feature_v_w, _ = self._topk_sparsify(pref_feature_v, self.top_k_feature_v)
            restore_q_w, _ = self._topk_sparsify(pref_restore_q, self.top_k_restore_q)
            restore_k_w, _ = self._topk_sparsify(pref_restore_k, self.top_k_restore_k)
            restore_v_w, _ = self._topk_sparsify(pref_restore_v, self.top_k_restore_v)
        else:
            # Batch-level routing (default)
            # importance-weighted aggregation
            feature_q_w_dense = torch.einsum('bs,bsn->bn', importance, pref_feature_q)
            feature_k_w_dense = torch.einsum('bs,bsn->bn', importance, pref_feature_k)
            feature_v_w_dense = torch.einsum('bs,bsn->bn', importance, pref_feature_v)
            restore_q_w_dense = torch.einsum('bs,bsn->bn', importance, pref_restore_q)
            restore_k_w_dense = torch.einsum('bs,bsn->bn', importance, pref_restore_k)
            restore_v_w_dense = torch.einsum('bs,bsn->bn', importance, pref_restore_v)

            feature_q_w, _ = self._topk_sparsify(feature_q_w_dense, self.top_k_feature_q)
            feature_k_w, _ = self._topk_sparsify(feature_k_w_dense, self.top_k_feature_k)
            feature_v_w, _ = self._topk_sparsify(feature_v_w_dense, self.top_k_feature_v)
            restore_q_w, _ = self._topk_sparsify(restore_q_w_dense, self.top_k_restore_q)
            restore_k_w, _ = self._topk_sparsify(restore_k_w_dense, self.top_k_restore_k)
            restore_v_w, _ = self._topk_sparsify(restore_v_w_dense, self.top_k_restore_v)

        # Update usage
        if self.training:
            self.neuron_router.update_usage(feature_q_w, 'feature_q', attention_mask)
            self.neuron_router.update_usage(feature_k_w, 'feature_k', attention_mask)
            self.neuron_router.update_usage(feature_v_w, 'feature_v', attention_mask)
            self.neuron_router.update_usage(restore_q_w, 'restore_q', attention_mask)
            self.neuron_router.update_usage(restore_k_w, 'restore_k', attention_mask)
            self.neuron_router.update_usage(restore_v_w, 'restore_v', attention_mask)

        # Entropy calculation for logging
        def entropy_pct(pref):
            p = pref.mean(dim=0) if pref.dim() == 3 else pref.mean(dim=0)
            p = p + 1e-8
            p = p / p.sum()
            max_ent = math.log(len(p))
            ent = -(p * p.log()).sum()
            return (ent / max_ent * 100).item() if max_ent > 0 else 0

        routing_info = {
            'feature_q_pref': pref_feature_q.detach(), 'feature_k_pref': pref_feature_k.detach(), 'feature_v_pref': pref_feature_v.detach(),
            'restore_q_pref': pref_restore_q.detach(), 'restore_k_pref': pref_restore_k.detach(), 'restore_v_pref': pref_restore_v.detach(),
            'ent_feature_q': entropy_pct(pref_feature_q), 'ent_feature_k': entropy_pct(pref_feature_k), 'ent_feature_v': entropy_pct(pref_feature_v),
            'ent_restore_q': entropy_pct(pref_restore_q), 'ent_restore_k': entropy_pct(pref_restore_k), 'ent_restore_v': entropy_pct(pref_restore_v),
        }

        return feature_q_w, feature_k_w, feature_v_w, restore_q_w, restore_k_w, restore_v_w, routing_info, aux_loss

    def get_knowledge_weights(self, x, importance, attention_mask=None):
        """Knowledge neuron routing (top_k selection) - NEW in v17"""
        k_logits = self.neuron_router.get_logits(x, 'knowledge')
        k_pref = F.softmax(k_logits, dim=-1)

        # Batch-level routing
        k_weights_dense = torch.einsum('bs,bsn->bn', importance, k_pref)
        k_weights, topk_idx = self._topk_sparsify(k_weights_dense, self.top_k_knowledge)

        if self.training:
            self.neuron_router.update_usage(k_weights, 'knowledge', attention_mask)

        return k_weights, topk_idx


class AttentionCircuit(nn.Module):
    """
    v17: Attention circuit with Complete Q/K/V Pool Separation (same as v16.3)

    Flow:
    1. x → FQ neurons → h_q [Q compression]
    2. x → FK neurons → h_k [K compression]
    3. x → FV neurons → h_v [V compression]
    4. h_q → RQ neurons → Q [Q restoration]
    5. h_k → RK neurons → K [K restoration]
    6. h_v → RV neurons → V [V restoration]
    7. Multi-head attention
    """
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

    def forward(self, x, feature_q_w, feature_k_w, feature_v_w, restore_q_w, restore_k_w, restore_v_w, attention_mask=None):
        B, S, D = x.shape
        token_routing = feature_q_w.dim() == 3

        if token_routing:
            # Token-level routing [B, S, n_neurons]
            shared_feature_q = torch.einsum('bsn,ndr->bsdr', feature_q_w, self.shared_neurons.feature_q_neurons)
            shared_feature_k = torch.einsum('bsn,ndr->bsdr', feature_k_w, self.shared_neurons.feature_k_neurons)
            shared_feature_v = torch.einsum('bsn,ndr->bsdr', feature_v_w, self.shared_neurons.feature_v_neurons)

            h_q = torch.einsum('bsd,bsdr->bsr', x, shared_feature_q)
            h_k = torch.einsum('bsd,bsdr->bsr', x, shared_feature_k)
            h_v = torch.einsum('bsd,bsdr->bsr', x, shared_feature_v)

            shared_restore_q = torch.einsum('bsn,nrd->bsrd', restore_q_w, self.shared_neurons.restore_q_neurons)
            shared_restore_k = torch.einsum('bsn,nrd->bsrd', restore_k_w, self.shared_neurons.restore_k_neurons)
            shared_restore_v = torch.einsum('bsn,nrd->bsrd', restore_v_w, self.shared_neurons.restore_v_neurons)

            Q = torch.einsum('bsr,bsrd->bsd', h_q, shared_restore_q)
            K = torch.einsum('bsr,bsrd->bsd', h_k, shared_restore_k)
            V = torch.einsum('bsr,bsrd->bsd', h_v, shared_restore_v)
        else:
            # Batch-level routing [B, n_neurons] - matmul optimized
            feature_q_flat = self.shared_neurons.feature_q_neurons.view(-1, D * self.rank)
            feature_k_flat = self.shared_neurons.feature_k_neurons.view(-1, D * self.rank)
            feature_v_flat = self.shared_neurons.feature_v_neurons.view(-1, D * self.rank)

            shared_feature_q = (feature_q_w @ feature_q_flat).view(B, D, self.rank)
            shared_feature_k = (feature_k_w @ feature_k_flat).view(B, D, self.rank)
            shared_feature_v = (feature_v_w @ feature_v_flat).view(B, D, self.rank)

            h_q = torch.bmm(x, shared_feature_q)
            h_k = torch.bmm(x, shared_feature_k)
            h_v = torch.bmm(x, shared_feature_v)

            restore_q_flat = self.shared_neurons.restore_q_neurons.view(-1, self.rank * D)
            restore_k_flat = self.shared_neurons.restore_k_neurons.view(-1, self.rank * D)
            restore_v_flat = self.shared_neurons.restore_v_neurons.view(-1, self.rank * D)

            shared_restore_q = (restore_q_w @ restore_q_flat).view(B, self.rank, D)
            shared_restore_k = (restore_k_w @ restore_k_flat).view(B, self.rank, D)
            shared_restore_v = (restore_v_w @ restore_v_flat).view(B, self.rank, D)

            Q = torch.bmm(h_q, shared_restore_q)
            K = torch.bmm(h_k, shared_restore_k)
            V = torch.bmm(h_v, shared_restore_v)

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

    Flow:
    x -> router selects neurons (top_k)
      -> feature_know (d_model -> knowledge_rank) [compression]
      -> restore_know (knowledge_rank -> d_model) [restoration]
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
        x: [B, S, D]
        k_weights: [B, n_knowledge] - sparse weights from router
        """
        B, S, D = x.shape
        R = self.knowledge_rank

        # Feature-Restore (batch-level, matmul optimized)
        # feature_know: [n_knowledge, d_model, knowledge_rank]
        feature_flat = self.shared_neurons.feature_know.view(-1, D * R)
        shared_feature = (k_weights @ feature_flat).view(B, D, R)

        h = torch.bmm(x, shared_feature)  # [B, S, R]

        # restore_know: [n_knowledge, knowledge_rank, d_model]
        restore_flat = self.shared_neurons.restore_know.view(-1, R * D)
        shared_restore = (k_weights @ restore_flat).view(B, R, D)

        output = torch.bmm(h, shared_restore)  # [B, S, D]
        output = self.dropout(output)

        return output


class DAWNBlock(nn.Module):
    """DAWN v17 block: Complete Q/K/V Separation + Knowledge Feature-Restore"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_knowledge: int,
        knowledge_rank: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = AttentionCircuit(shared_neurons, d_model, n_heads, rank, dropout)
        self.knowledge = KnowledgeCircuit(
            shared_neurons, d_model, n_knowledge,
            knowledge_rank=knowledge_rank if knowledge_rank else 128,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, importance, global_routers: GlobalRouters, attention_mask=None):
        normed_x = self.norm1(x)
        feature_q_w, feature_k_w, feature_v_w, restore_q_w, restore_k_w, restore_v_w, attn_routing, attn_aux_loss = \
            global_routers.get_attention_weights(normed_x, importance, attention_mask)

        attn_out, _ = self.attn(normed_x, feature_q_w, feature_k_w, feature_v_w, restore_q_w, restore_k_w, restore_v_w, attention_mask)
        x = x + attn_out

        normed_x2 = self.norm2(x)
        k_weights, k_topk_idx = global_routers.get_knowledge_weights(normed_x2, importance, attention_mask)
        know_out = self.knowledge(normed_x2, k_weights, attention_mask)
        x = x + know_out

        attn_out_norm = attn_out.norm(dim=-1).mean().detach()
        know_out_norm = know_out.norm(dim=-1).mean().detach()

        routing_info = {
            'attention': {**attn_routing},
            'knowledge': {
                'weights': k_weights.detach(),
                'topk_indices': k_topk_idx.detach(),
            },
            'attn_out_norm': attn_out_norm,
            'know_out_norm': know_out_norm,
        }

        return x, routing_info, attn_aux_loss


class DAWN(nn.Module):
    """
    DAWN v17: v16.3 + Knowledge Feature-Restore

    Attention: Q/K/V 완전 분리 (v16.3 동일)
    - FQ/FK/FV: 각각 독립적인 압축 뉴런 풀
    - RQ/RK/RV: 각각 독립적인 복원 뉴런 풀

    Knowledge: Feature-Restore pattern (NEW)
    - feature_know: [n_knowledge, d_model, knowledge_rank]
    - restore_know: [n_knowledge, knowledge_rank, d_model]
    """
    __version__ = "17"

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 320,
        n_layers: int = 4,
        n_heads: int = 4,
        rank: int = 64,
        max_seq_len: int = 128,
        # Q pool
        n_feature_q: int = 32,
        n_restore_q: int = 32,
        top_k_feature_q: int = 8,
        top_k_restore_q: int = 8,
        # K pool
        n_feature_k: int = 32,
        n_restore_k: int = 32,
        top_k_feature_k: int = 8,
        top_k_restore_k: int = 8,
        # V pool
        n_feature_v: int = 24,
        n_restore_v: int = 24,
        top_k_feature_v: int = 3,
        top_k_restore_v: int = 3,
        # Knowledge (NEW - no coarse_k, fine_k)
        n_knowledge: int = 24,
        top_k_knowledge: int = 4,
        knowledge_rank: int = None,
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
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else 128
        self.max_seq_len = max_seq_len
        self.state_dim = state_dim
        self.d_space = d_space
        self.token_routing = token_routing
        self.router_dropout = router_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.use_ssm_context = use_ssm_context
        self.excitability_tau = excitability_tau
        self.excitability_ema_alpha = excitability_ema_alpha

        # Q pool
        self.n_feature_q = n_feature_q
        self.n_restore_q = n_restore_q
        self.top_k_feature_q = top_k_feature_q
        self.top_k_restore_q = top_k_restore_q

        # K pool
        self.n_feature_k = n_feature_k
        self.n_restore_k = n_restore_k
        self.top_k_feature_k = top_k_feature_k
        self.top_k_restore_k = top_k_restore_k

        # V pool
        self.n_feature_v = n_feature_v
        self.n_restore_v = n_restore_v
        self.top_k_feature_v = top_k_feature_v
        self.top_k_restore_v = top_k_restore_v

        # Knowledge (NEW)
        self.n_knowledge = n_knowledge
        self.top_k_knowledge = top_k_knowledge

        # v15 compat
        self.n_feature = n_feature_q
        self.n_neurons = n_feature_q
        self.basis_rank = rank

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.shared_neurons = SharedNeurons(
            d_model=d_model, rank=rank,
            n_feature_q=n_feature_q, n_feature_k=n_feature_k, n_feature_v=n_feature_v,
            n_restore_q=n_restore_q, n_restore_k=n_restore_k, n_restore_v=n_restore_v,
            n_knowledge=n_knowledge, knowledge_rank=self.knowledge_rank,
        )

        self.global_ssm = GlobalSSM(d_model, state_dim, return_context=use_ssm_context)

        self.global_routers = GlobalRouters(
            d_model, n_feature_q, n_feature_k, n_feature_v, n_restore_q, n_restore_k, n_restore_v, n_knowledge,
            top_k_feature_q, top_k_feature_k, top_k_feature_v, top_k_restore_q, top_k_restore_k, top_k_restore_v,
            top_k_knowledge=top_k_knowledge,
            d_space=d_space, router_dropout=router_dropout, token_routing=token_routing,
            excitability_tau=excitability_tau, excitability_ema_alpha=excitability_ema_alpha,
            excitability_decay_rate=excitability_decay_rate
        )

        # NOTE: knowledge_encoder removed in v17 (Feature-Restore pattern instead)

        self.layers = nn.ModuleList([
            DAWNBlock(
                shared_neurons=self.shared_neurons, d_model=d_model, n_heads=n_heads,
                rank=rank, n_knowledge=n_knowledge, knowledge_rank=self.knowledge_rank,
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
        """Orthogonality loss for all 8 neuron types (6 attention + 2 knowledge)"""
        I = torch.eye(self.rank, device=self.shared_neurons.feature_q_neurons.device).unsqueeze(0)

        # Attention neurons (6개) - same as v16.3
        W_feature_q = self.shared_neurons.feature_q_neurons
        WtW_feature_q = torch.bmm(W_feature_q.transpose(1, 2), W_feature_q)
        loss_feature_q = ((WtW_feature_q - I) ** 2).mean()

        W_feature_k = self.shared_neurons.feature_k_neurons
        WtW_feature_k = torch.bmm(W_feature_k.transpose(1, 2), W_feature_k)
        loss_feature_k = ((WtW_feature_k - I) ** 2).mean()

        W_feature_v = self.shared_neurons.feature_v_neurons
        WtW_feature_v = torch.bmm(W_feature_v.transpose(1, 2), W_feature_v)
        loss_feature_v = ((WtW_feature_v - I) ** 2).mean()

        W_restore_q = self.shared_neurons.restore_q_neurons
        WWt_restore_q = torch.bmm(W_restore_q, W_restore_q.transpose(1, 2))
        loss_restore_q = ((WWt_restore_q - I) ** 2).mean()

        W_restore_k = self.shared_neurons.restore_k_neurons
        WWt_restore_k = torch.bmm(W_restore_k, W_restore_k.transpose(1, 2))
        loss_restore_k = ((WWt_restore_k - I) ** 2).mean()

        W_restore_v = self.shared_neurons.restore_v_neurons
        WWt_restore_v = torch.bmm(W_restore_v, W_restore_v.transpose(1, 2))
        loss_restore_v = ((WWt_restore_v - I) ** 2).mean()

        # Knowledge neurons (2개) - NEW in v17
        I_know = torch.eye(self.knowledge_rank, device=self.shared_neurons.feature_know.device).unsqueeze(0)

        W_feature_know = self.shared_neurons.feature_know
        WtW_feature_know = torch.bmm(W_feature_know.transpose(1, 2), W_feature_know)
        loss_feature_know = ((WtW_feature_know - I_know) ** 2).mean()

        W_restore_know = self.shared_neurons.restore_know
        WWt_restore_know = torch.bmm(W_restore_know, W_restore_know.transpose(1, 2))
        loss_restore_know = ((WWt_restore_know - I_know) ** 2).mean()

        # 8개 평균
        return (loss_feature_q + loss_feature_k + loss_feature_v + loss_restore_q + loss_restore_k + loss_restore_v + loss_feature_know + loss_restore_know) / 8

    def knowledge_diversity_loss(self):
        """Diversity loss for knowledge neurons - uses feature_know"""
        F_know = self.shared_neurons.feature_know  # [n_knowledge, d_model, knowledge_rank]
        F_flat = F_know.view(self.n_knowledge, -1)
        F_norm = F.normalize(F_flat, dim=-1)
        sim = F_norm @ F_norm.T
        mask = ~torch.eye(self.n_knowledge, dtype=torch.bool, device=F_know.device)
        return sim[mask].abs().mean()

    def get_auxiliary_losses(self):
        return {
            'orth_total': self.orthogonality_loss(),
            'knowledge_div': self.knowledge_diversity_loss(),
        }

    def count_by_component(self):
        """Return detailed parameter counts by component"""
        embedding = self.token_emb.weight.numel() + self.pos_emb.weight.numel()

        # Attention neurons
        f_neurons = self.shared_neurons.f_neurons.numel()
        r_neurons = self.shared_neurons.r_neurons.numel()

        # Knowledge neurons (NEW)
        feature_know = self.shared_neurons.feature_know.numel()
        restore_know = self.shared_neurons.restore_know.numel()

        # Per-layer: expand_O, norms
        expand_o = self.layers[0].attn.expand_O.weight.numel() * self.n_layers
        norms = sum(p.numel() for layer in self.layers
                   for name, p in layer.named_parameters() if 'norm' in name)

        # Router
        router = sum(p.numel() for p in self.global_routers.parameters())

        # SSM
        ssm = sum(p.numel() for p in self.global_ssm.parameters())

        # Print breakdown
        print("\n=== DAWN v17 Parameter Breakdown ===")
        print(f"embedding:             {embedding:,} ({embedding/1e6:.2f}M)")
        print(f"f_neurons (FQ/FK/FV):  {f_neurons:,} ({f_neurons/1e3:.1f}K)")
        print(f"r_neurons (RQ/RK/RV):  {r_neurons:,} ({r_neurons/1e3:.1f}K)")
        print(f"feature_know:          {feature_know:,} ({feature_know/1e3:.1f}K)")
        print(f"restore_know:          {restore_know:,} ({restore_know/1e3:.1f}K)")
        print(f"expand_O:              {expand_o:,} ({expand_o/1e3:.1f}K)")
        print(f"norms:                 {norms:,} ({norms/1e3:.1f}K)")
        print(f"router:                {router:,} ({router/1e3:.1f}K)")
        print(f"ssm:                   {ssm:,} ({ssm/1e3:.1f}K)")
        print("=" * 40)

        return {
            'embedding': embedding, 'f_neurons': f_neurons, 'r_neurons': r_neurons,
            'feature_know': feature_know, 'restore_know': restore_know,
            'expand_o': expand_o, 'norms': norms, 'router': router, 'ssm': ssm,
        }

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'rank': self.rank, 'knowledge_rank': self.knowledge_rank,
            'max_seq_len': self.max_seq_len,
            'n_feature_q': self.n_feature_q, 'n_feature_k': self.n_feature_k, 'n_feature_v': self.n_feature_v,
            'n_restore_q': self.n_restore_q, 'n_restore_k': self.n_restore_k, 'n_restore_v': self.n_restore_v,
            'top_k_feature_q': self.top_k_feature_q, 'top_k_feature_k': self.top_k_feature_k, 'top_k_feature_v': self.top_k_feature_v,
            'top_k_restore_q': self.top_k_restore_q, 'top_k_restore_k': self.top_k_restore_k, 'top_k_restore_v': self.top_k_restore_v,
            'n_knowledge': self.n_knowledge,
            'top_k_knowledge': self.top_k_knowledge,
            'state_dim': self.state_dim, 'd_space': self.d_space,
            'router_dropout': self.router_dropout,
            'gradient_checkpointing': self.gradient_checkpointing,
            'token_routing': self.token_routing,
        }

    def get_model_info(self):
        """Return model architecture info for logging"""
        return [
            f"DAWN v{self.__version__}: Q/K/V Separated + Knowledge Feature-Restore",
            f"  rank={self.rank}, knowledge_rank={self.knowledge_rank}",
            f"  FQ: {self.n_feature_q} × {self.d_model} × {self.rank} (top-k={self.top_k_feature_q})",
            f"  FK: {self.n_feature_k} × {self.d_model} × {self.rank} (top-k={self.top_k_feature_k})",
            f"  FV: {self.n_feature_v} × {self.d_model} × {self.rank} (top-k={self.top_k_feature_v})",
            f"  RQ: {self.n_restore_q} × {self.rank} × {self.d_model} (top-k={self.top_k_restore_q})",
            f"  RK: {self.n_restore_k} × {self.rank} × {self.d_model} (top-k={self.top_k_restore_k})",
            f"  RV: {self.n_restore_v} × {self.rank} × {self.d_model} (top-k={self.top_k_restore_v})",
            f"  Knowledge: {self.n_knowledge} (top-k={self.top_k_knowledge}) - Feature-Restore pattern",
            f"  Unified Router: d_space={self.d_space}",
        ]
