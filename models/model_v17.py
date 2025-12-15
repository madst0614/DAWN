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
    def __init__(self, d_model, n_fq, n_fk, n_fv, n_rq, n_rk, n_rv, n_knowledge,
                 d_space=64, dropout=0.1, excitability_tau=1.5, excitability_ema_alpha=0.01,
                 excitability_decay_rate=0.99995):
        super().__init__()
        self.n_fq = n_fq
        self.n_fk = n_fk
        self.n_fv = n_fv
        self.n_rq = n_rq
        self.n_rk = n_rk
        self.n_rv = n_rv
        self.n_knowledge = n_knowledge
        self.d_space = d_space
        self.ema_alpha = excitability_ema_alpha

        total_neurons = n_fq + n_fk + n_fv + n_rq + n_rk + n_rv + n_knowledge
        self.total_neurons = total_neurons

        # 인덱스 경계
        self.fq_end = n_fq
        self.fk_end = n_fq + n_fk
        self.fv_end = n_fq + n_fk + n_fv
        self.rq_end = n_fq + n_fk + n_fv + n_rq
        self.rk_end = n_fq + n_fk + n_fv + n_rq + n_rk
        self.rv_end = n_fq + n_fk + n_fv + n_rq + n_rk + n_rv
        # knowledge는 rv_end ~ total_neurons

        # 통합 projection (6개) + knowledge 별도
        self.proj_all = nn.Linear(d_model, d_space * 6)  # fq, fk, fv, rq, rk, rv
        self.proj_knowledge = nn.Linear(d_model, d_space)
        self.dropout = nn.Dropout(dropout)

        # 통합 뉴런 임베딩 [total_neurons, d_space]
        self.neuron_emb = nn.Parameter(torch.randn(total_neurons, d_space) * 0.02)

        # 타입별 usage 추적
        self.register_buffer('usage_ema_fq', torch.zeros(n_fq))
        self.register_buffer('usage_ema_fk', torch.zeros(n_fk))
        self.register_buffer('usage_ema_fv', torch.zeros(n_fv))
        self.register_buffer('usage_ema_rq', torch.zeros(n_rq))
        self.register_buffer('usage_ema_rk', torch.zeros(n_rk))
        self.register_buffer('usage_ema_rv', torch.zeros(n_rv))
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
        neuron_type: 'knowledge' only (fq~rv는 get_all_logits 사용)
        """
        if neuron_type != 'knowledge':
            raise ValueError(f"Use get_all_logits for {neuron_type}. get_logits is for 'knowledge' only.")

        emb_norm = F.normalize(self.neuron_emb, dim=-1)
        h_proj = self.dropout(self.proj_knowledge(x))
        emb = emb_norm[self.rv_end:]

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
        h_fq, h_fk, h_fv, h_rq, h_rk, h_rv = all_proj.chunk(6, dim=-1)

        # 뉴런 임베딩 정규화 (한 번만)
        emb_norm = F.normalize(self.neuron_emb[:self.rv_end], dim=-1)  # knowledge 제외

        # 각 타입별 emb 슬라이스 (view라 비용 없음)
        emb_fq = emb_norm[:self.fq_end]
        emb_fk = emb_norm[self.fq_end:self.fk_end]
        emb_fv = emb_norm[self.fk_end:self.fv_end]
        emb_rq = emb_norm[self.fv_end:self.rq_end]
        emb_rk = emb_norm[self.rq_end:self.rk_end]
        emb_rv = emb_norm[self.rk_end:self.rv_end]

        # 각각 자기 영역만 연산
        logits_fq = torch.einsum('bsd,nd->bsn', h_fq, emb_fq)
        logits_fk = torch.einsum('bsd,nd->bsn', h_fk, emb_fk)
        logits_fv = torch.einsum('bsd,nd->bsn', h_fv, emb_fv)
        logits_rq = torch.einsum('bsd,nd->bsn', h_rq, emb_rq)
        logits_rk = torch.einsum('bsd,nd->bsn', h_rk, emb_rk)
        logits_rv = torch.einsum('bsd,nd->bsn', h_rv, emb_rv)

        # excitability 추가
        if self.training:
            w = self.excitability_weight
            logits_fq = logits_fq + self.get_excitability(self.usage_ema_fq) * w
            logits_fk = logits_fk + self.get_excitability(self.usage_ema_fk) * w
            logits_fv = logits_fv + self.get_excitability(self.usage_ema_fv) * w
            logits_rq = logits_rq + self.get_excitability(self.usage_ema_rq) * w
            logits_rk = logits_rk + self.get_excitability(self.usage_ema_rk) * w
            logits_rv = logits_rv + self.get_excitability(self.usage_ema_rv) * w

        return logits_fq, logits_fk, logits_fv, logits_rq, logits_rk, logits_rv

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
        if neuron_type == 'fq':
            self.usage_ema_fq.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'fk':
            self.usage_ema_fk.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'fv':
            self.usage_ema_fv.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'rq':
            self.usage_ema_rq.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'rk':
            self.usage_ema_rk.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'rv':
            self.usage_ema_rv.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'knowledge':
            self.usage_ema_knowledge.mul_(decay).add_(usage, alpha=self.ema_alpha)


class SharedNeurons(nn.Module):
    """
    v17: Complete Pool Separation + Knowledge Feature-Restore

    Attention neurons (same as v16.3):
    - FQ: [n_fq, d_model, rank] - Q 압축
    - FK: [n_fk, d_model, rank] - K 압축
    - FV: [n_fv, d_model, rank] - V 압축
    - RQ: [n_rq, rank, d_model] - Q 복원
    - RK: [n_rk, rank, d_model] - K 복원
    - RV: [n_rv, rank, d_model] - V 복원

    Knowledge neurons (NEW - Feature-Restore pattern):
    - feature_know: [n_knowledge, d_model, knowledge_rank] - 압축
    - restore_know: [n_knowledge, knowledge_rank, d_model] - 복원
    """
    def __init__(
        self,
        d_model: int,
        rank: int,
        n_fq: int,
        n_fk: int,
        n_fv: int,
        n_rq: int,
        n_rk: int,
        n_rv: int,
        n_knowledge: int,
        knowledge_rank: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else 128
        self.n_fq = n_fq
        self.n_fk = n_fk
        self.n_fv = n_fv
        self.n_rq = n_rq
        self.n_rk = n_rk
        self.n_rv = n_rv
        self.n_knowledge = n_knowledge

        # Contiguous memory for GPU cache efficiency
        # F group: [n_fq + n_fk + n_fv, d_model, rank]
        self.f_neurons = nn.Parameter(torch.zeros(n_fq + n_fk + n_fv, d_model, rank))
        # R group: [n_rq + n_rk + n_rv, rank, d_model]
        self.r_neurons = nn.Parameter(torch.zeros(n_rq + n_rk + n_rv, rank, d_model))

        # Knowledge neurons: Feature-Restore pattern (NEW in v17)
        self.feature_know = nn.Parameter(torch.zeros(n_knowledge, d_model, self.knowledge_rank))
        self.restore_know = nn.Parameter(torch.zeros(n_knowledge, self.knowledge_rank, d_model))

        self._init_parameters()

    # Properties for sliced access (contiguous memory)
    @property
    def fq_neurons(self):
        return self.f_neurons[:self.n_fq]

    @property
    def fk_neurons(self):
        return self.f_neurons[self.n_fq:self.n_fq + self.n_fk]

    @property
    def fv_neurons(self):
        return self.f_neurons[self.n_fq + self.n_fk:]

    @property
    def rq_neurons(self):
        return self.r_neurons[:self.n_rq]

    @property
    def rk_neurons(self):
        return self.r_neurons[self.n_rq:self.n_rq + self.n_rk]

    @property
    def rv_neurons(self):
        return self.r_neurons[self.n_rq + self.n_rk:]

    def _init_parameters(self):
        # Attention neurons: orthogonal init
        for i in range(self.n_fq + self.n_fk + self.n_fv):
            nn.init.orthogonal_(self.f_neurons.data[i])
        for i in range(self.n_rq + self.n_rk + self.n_rv):
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
    def __init__(self, d_model, n_fq, n_fk, n_fv, n_rq, n_rk, n_rv, n_knowledge,
                 top_k_fq=8, top_k_fk=8, top_k_fv=3, top_k_rq=8, top_k_rk=8, top_k_rv=3,
                 top_k_knowledge=4,
                 d_space=64, router_dropout=0.1, token_routing=False,
                 excitability_tau=1.5, excitability_ema_alpha=0.01, excitability_decay_rate=0.99995):
        super().__init__()
        self.top_k_fq = top_k_fq
        self.top_k_fk = top_k_fk
        self.top_k_fv = top_k_fv
        self.top_k_rq = top_k_rq
        self.top_k_rk = top_k_rk
        self.top_k_rv = top_k_rv
        self.top_k_knowledge = top_k_knowledge
        self.token_routing = token_routing

        self.neuron_router = UnifiedNeuronRouter(
            d_model, n_fq, n_fk, n_fv, n_rq, n_rk, n_rv, n_knowledge,
            d_space=d_space, dropout=router_dropout,
            excitability_tau=excitability_tau, excitability_ema_alpha=excitability_ema_alpha,
            excitability_decay_rate=excitability_decay_rate
        )

    def _topk_sparsify(self, weights, k):
        """Top-k sparsification with straight-through gradient"""
        topk_val, topk_idx = torch.topk(weights, k, dim=-1)
        sparse = torch.zeros_like(weights)
        sparse.scatter_(-1, topk_idx, topk_val)
        sparse = sparse - weights.detach() + weights  # STE
        return sparse, topk_idx

    def get_attention_weights(self, x, importance, attention_mask=None):
        """Get all 6 attention weights"""
        logits_fq, logits_fk, logits_fv, logits_rq, logits_rk, logits_rv = \
            self.neuron_router.get_all_logits(x)

        if self.token_routing:
            # Token-level routing
            pref_fq = F.softmax(logits_fq, dim=-1)
            pref_fk = F.softmax(logits_fk, dim=-1)
            pref_fv = F.softmax(logits_fv, dim=-1)
            pref_rq = F.softmax(logits_rq, dim=-1)
            pref_rk = F.softmax(logits_rk, dim=-1)
            pref_rv = F.softmax(logits_rv, dim=-1)

            fq_w, _ = self._topk_sparsify(pref_fq, self.top_k_fq)
            fk_w, _ = self._topk_sparsify(pref_fk, self.top_k_fk)
            fv_w, _ = self._topk_sparsify(pref_fv, self.top_k_fv)
            rq_w, _ = self._topk_sparsify(pref_rq, self.top_k_rq)
            rk_w, _ = self._topk_sparsify(pref_rk, self.top_k_rk)
            rv_w, _ = self._topk_sparsify(pref_rv, self.top_k_rv)
        else:
            # Batch-level routing (default)
            pref_fq = F.softmax(logits_fq, dim=-1)
            pref_fk = F.softmax(logits_fk, dim=-1)
            pref_fv = F.softmax(logits_fv, dim=-1)
            pref_rq = F.softmax(logits_rq, dim=-1)
            pref_rk = F.softmax(logits_rk, dim=-1)
            pref_rv = F.softmax(logits_rv, dim=-1)

            # importance-weighted aggregation
            fq_w_dense = torch.einsum('bs,bsn->bn', importance, pref_fq)
            fk_w_dense = torch.einsum('bs,bsn->bn', importance, pref_fk)
            fv_w_dense = torch.einsum('bs,bsn->bn', importance, pref_fv)
            rq_w_dense = torch.einsum('bs,bsn->bn', importance, pref_rq)
            rk_w_dense = torch.einsum('bs,bsn->bn', importance, pref_rk)
            rv_w_dense = torch.einsum('bs,bsn->bn', importance, pref_rv)

            fq_w, _ = self._topk_sparsify(fq_w_dense, self.top_k_fq)
            fk_w, _ = self._topk_sparsify(fk_w_dense, self.top_k_fk)
            fv_w, _ = self._topk_sparsify(fv_w_dense, self.top_k_fv)
            rq_w, _ = self._topk_sparsify(rq_w_dense, self.top_k_rq)
            rk_w, _ = self._topk_sparsify(rk_w_dense, self.top_k_rk)
            rv_w, _ = self._topk_sparsify(rv_w_dense, self.top_k_rv)

        # Update usage
        if self.training:
            self.neuron_router.update_usage(fq_w, 'fq', attention_mask)
            self.neuron_router.update_usage(fk_w, 'fk', attention_mask)
            self.neuron_router.update_usage(fv_w, 'fv', attention_mask)
            self.neuron_router.update_usage(rq_w, 'rq', attention_mask)
            self.neuron_router.update_usage(rk_w, 'rk', attention_mask)
            self.neuron_router.update_usage(rv_w, 'rv', attention_mask)

        # Entropy calculation for logging
        def entropy_pct(pref):
            p = pref.mean(dim=0) if pref.dim() == 3 else pref.mean(dim=0)
            p = p + 1e-8
            p = p / p.sum()
            max_ent = math.log(len(p))
            ent = -(p * p.log()).sum()
            return (ent / max_ent * 100).item() if max_ent > 0 else 0

        routing_info = {
            'fq_pref': pref_fq.detach(), 'fk_pref': pref_fk.detach(), 'fv_pref': pref_fv.detach(),
            'rq_pref': pref_rq.detach(), 'rk_pref': pref_rk.detach(), 'rv_pref': pref_rv.detach(),
            'ent_fq': entropy_pct(pref_fq), 'ent_fk': entropy_pct(pref_fk), 'ent_fv': entropy_pct(pref_fv),
            'ent_rq': entropy_pct(pref_rq), 'ent_rk': entropy_pct(pref_rk), 'ent_rv': entropy_pct(pref_rv),
        }

        aux_loss = 0.0
        return fq_w, fk_w, fv_w, rq_w, rk_w, rv_w, routing_info, aux_loss

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


class NeuronCircuit(nn.Module):
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

    def forward(self, x, fq_w, fk_w, fv_w, rq_w, rk_w, rv_w, attention_mask=None):
        B, S, D = x.shape
        token_routing = fq_w.dim() == 3

        if token_routing:
            # Token-level routing [B, S, n_neurons]
            shared_fq = torch.einsum('bsn,ndr->bsdr', fq_w, self.shared_neurons.fq_neurons)
            shared_fk = torch.einsum('bsn,ndr->bsdr', fk_w, self.shared_neurons.fk_neurons)
            shared_fv = torch.einsum('bsn,ndr->bsdr', fv_w, self.shared_neurons.fv_neurons)

            h_q = torch.einsum('bsd,bsdr->bsr', x, shared_fq)
            h_k = torch.einsum('bsd,bsdr->bsr', x, shared_fk)
            h_v = torch.einsum('bsd,bsdr->bsr', x, shared_fv)

            shared_rq = torch.einsum('bsn,nrd->bsrd', rq_w, self.shared_neurons.rq_neurons)
            shared_rk = torch.einsum('bsn,nrd->bsrd', rk_w, self.shared_neurons.rk_neurons)
            shared_rv = torch.einsum('bsn,nrd->bsrd', rv_w, self.shared_neurons.rv_neurons)

            Q = torch.einsum('bsr,bsrd->bsd', h_q, shared_rq)
            K = torch.einsum('bsr,bsrd->bsd', h_k, shared_rk)
            V = torch.einsum('bsr,bsrd->bsd', h_v, shared_rv)
        else:
            # Batch-level routing [B, n_neurons] - matmul optimized
            fq_flat = self.shared_neurons.fq_neurons.view(-1, D * self.rank)
            fk_flat = self.shared_neurons.fk_neurons.view(-1, D * self.rank)
            fv_flat = self.shared_neurons.fv_neurons.view(-1, D * self.rank)

            shared_fq = (fq_w @ fq_flat).view(B, D, self.rank)
            shared_fk = (fk_w @ fk_flat).view(B, D, self.rank)
            shared_fv = (fv_w @ fv_flat).view(B, D, self.rank)

            h_q = torch.bmm(x, shared_fq)
            h_k = torch.bmm(x, shared_fk)
            h_v = torch.bmm(x, shared_fv)

            rq_flat = self.shared_neurons.rq_neurons.view(-1, self.rank * D)
            rk_flat = self.shared_neurons.rk_neurons.view(-1, self.rank * D)
            rv_flat = self.shared_neurons.rv_neurons.view(-1, self.rank * D)

            shared_rq = (rq_w @ rq_flat).view(B, self.rank, D)
            shared_rk = (rk_w @ rk_flat).view(B, self.rank, D)
            shared_rv = (rv_w @ rv_flat).view(B, self.rank, D)

            Q = torch.bmm(h_q, shared_rq)
            K = torch.bmm(h_k, shared_rk)
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

        self.attn = NeuronCircuit(shared_neurons, d_model, n_heads, rank, dropout)
        self.knowledge = KnowledgeCircuit(
            shared_neurons, d_model, n_knowledge,
            knowledge_rank=knowledge_rank if knowledge_rank else 128,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, importance, global_routers: GlobalRouters, attention_mask=None):
        normed_x = self.norm1(x)
        fq_w, fk_w, fv_w, rq_w, rk_w, rv_w, attn_routing, attn_aux_loss = \
            global_routers.get_attention_weights(normed_x, importance, attention_mask)

        attn_out, _ = self.attn(normed_x, fq_w, fk_w, fv_w, rq_w, rk_w, rv_w, attention_mask)
        x = x + attn_out

        normed_x2 = self.norm2(x)
        k_weights, k_topk_idx = global_routers.get_knowledge_weights(normed_x2, importance, attention_mask)
        know_out = self.knowledge(normed_x2, k_weights, attention_mask)
        x = x + self.dropout(know_out)

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
        n_fq: int = 32,
        n_rq: int = 32,
        top_k_fq: int = 8,
        top_k_rq: int = 8,
        # K pool
        n_fk: int = 32,
        n_rk: int = 32,
        top_k_fk: int = 8,
        top_k_rk: int = 8,
        # V pool
        n_fv: int = 24,
        n_rv: int = 24,
        top_k_fv: int = 3,
        top_k_rv: int = 3,
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
        self.n_fq = n_fq
        self.n_rq = n_rq
        self.top_k_fq = top_k_fq
        self.top_k_rq = top_k_rq

        # K pool
        self.n_fk = n_fk
        self.n_rk = n_rk
        self.top_k_fk = top_k_fk
        self.top_k_rk = top_k_rk

        # V pool
        self.n_fv = n_fv
        self.n_rv = n_rv
        self.top_k_fv = top_k_fv
        self.top_k_rv = top_k_rv

        # Knowledge (NEW)
        self.n_knowledge = n_knowledge
        self.top_k_knowledge = top_k_knowledge

        # v15 compat
        self.n_feature = n_fq
        self.n_neurons = n_fq
        self.basis_rank = rank

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.shared_neurons = SharedNeurons(
            d_model=d_model, rank=rank,
            n_fq=n_fq, n_fk=n_fk, n_fv=n_fv,
            n_rq=n_rq, n_rk=n_rk, n_rv=n_rv,
            n_knowledge=n_knowledge, knowledge_rank=self.knowledge_rank,
        )

        self.global_ssm = GlobalSSM(d_model, state_dim, return_context=use_ssm_context)

        self.global_routers = GlobalRouters(
            d_model, n_fq, n_fk, n_fv, n_rq, n_rk, n_rv, n_knowledge,
            top_k_fq, top_k_fk, top_k_fv, top_k_rq, top_k_rk, top_k_rv,
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
        I = torch.eye(self.rank, device=self.shared_neurons.fq_neurons.device).unsqueeze(0)

        # Attention neurons (6개) - same as v16.3
        W_fq = self.shared_neurons.fq_neurons
        WtW_fq = torch.bmm(W_fq.transpose(1, 2), W_fq)
        loss_fq = ((WtW_fq - I) ** 2).mean()

        W_fk = self.shared_neurons.fk_neurons
        WtW_fk = torch.bmm(W_fk.transpose(1, 2), W_fk)
        loss_fk = ((WtW_fk - I) ** 2).mean()

        W_fv = self.shared_neurons.fv_neurons
        WtW_fv = torch.bmm(W_fv.transpose(1, 2), W_fv)
        loss_fv = ((WtW_fv - I) ** 2).mean()

        W_rq = self.shared_neurons.rq_neurons
        WWt_rq = torch.bmm(W_rq, W_rq.transpose(1, 2))
        loss_rq = ((WWt_rq - I) ** 2).mean()

        W_rk = self.shared_neurons.rk_neurons
        WWt_rk = torch.bmm(W_rk, W_rk.transpose(1, 2))
        loss_rk = ((WWt_rk - I) ** 2).mean()

        W_rv = self.shared_neurons.rv_neurons
        WWt_rv = torch.bmm(W_rv, W_rv.transpose(1, 2))
        loss_rv = ((WWt_rv - I) ** 2).mean()

        # Knowledge neurons (2개) - NEW in v17
        I_know = torch.eye(self.knowledge_rank, device=self.shared_neurons.feature_know.device).unsqueeze(0)

        W_fknow = self.shared_neurons.feature_know
        WtW_fknow = torch.bmm(W_fknow.transpose(1, 2), W_fknow)
        loss_fknow = ((WtW_fknow - I_know) ** 2).mean()

        W_rknow = self.shared_neurons.restore_know
        WWt_rknow = torch.bmm(W_rknow, W_rknow.transpose(1, 2))
        loss_rknow = ((WWt_rknow - I_know) ** 2).mean()

        # 8개 평균
        return (loss_fq + loss_fk + loss_fv + loss_rq + loss_rk + loss_rv + loss_fknow + loss_rknow) / 8

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
            'n_fq': self.n_fq, 'n_fk': self.n_fk, 'n_fv': self.n_fv,
            'n_rq': self.n_rq, 'n_rk': self.n_rk, 'n_rv': self.n_rv,
            'top_k_fq': self.top_k_fq, 'top_k_fk': self.top_k_fk, 'top_k_fv': self.top_k_fv,
            'top_k_rq': self.top_k_rq, 'top_k_rk': self.top_k_rk, 'top_k_rv': self.top_k_rv,
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
            f"  FQ: {self.n_fq} × {self.d_model} × {self.rank} (top-k={self.top_k_fq})",
            f"  FK: {self.n_fk} × {self.d_model} × {self.rank} (top-k={self.top_k_fk})",
            f"  FV: {self.n_fv} × {self.d_model} × {self.rank} (top-k={self.top_k_fv})",
            f"  RQ: {self.n_rq} × {self.rank} × {self.d_model} (top-k={self.top_k_rq})",
            f"  RK: {self.n_rk} × {self.rank} × {self.d_model} (top-k={self.top_k_rk})",
            f"  RV: {self.n_rv} × {self.rank} × {self.d_model} (top-k={self.top_k_rv})",
            f"  Knowledge: {self.n_knowledge} (top-k={self.top_k_knowledge}) - Feature-Restore pattern",
            f"  Unified Router: d_space={self.d_space}",
        ]
