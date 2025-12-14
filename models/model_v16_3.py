"""
DAWN v16_3: Complete Q/K/V Pool Separation

v16.2 기반 + 완전한 풀 분리:
- FQ: Q 압축 전용 뉴런 풀
- FK: K 압축 전용 뉴런 풀
- FV: V 압축 전용 뉴런 풀
- RQ: Q 복원 전용 뉴런 풀
- RK: K 복원 전용 뉴런 풀
- RV: V 복원 전용 뉴런 풀

Terminology:
- F (Feature): 압축 뉴런 (d_model → rank)
- R (Restore): 복원 뉴런 (rank → d_model)
- Q/K/V: 용도

Routing paths:
- Q path: x → FQ → h_q → RQ → Q
- K path: x → FK → h_k → RK → K
- V path: x → FV → h_v → RV → V
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
    v16.3: Complete Pool Separation Router

    6개의 독립적인 뉴런 풀을 라우팅:
    - FQ, FK, FV: 압축 뉴런
    - RQ, RK, RV: 복원 뉴런
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
        """6개 풀 logits를 한 번에 계산 (stack + single einsum)"""
        B, S, D = x.shape

        # 6개 projection 한번에 → chunk
        all_proj = self.dropout(self.proj_all(x))  # [B, S, d_space * 6]
        h_fq, h_fk, h_fv, h_rq, h_rk, h_rv = all_proj.chunk(6, dim=-1)

        # stack으로 묶음
        h_stack = torch.stack([h_fq, h_fk, h_fv, h_rq, h_rk, h_rv], dim=2)  # [B, S, 6, d_space]

        # 뉴런 임베딩 정규화 (한 번만)
        emb_norm = F.normalize(self.neuron_emb, dim=-1)  # [total_neurons, d_space]

        # 한번에 전체 연산
        all_logits = torch.einsum('bshd,nd->bshn', h_stack, emb_norm)  # [B, S, 6, total_neurons]

        # 각 타입별 자기 영역만 슬라이싱
        logits_fq = all_logits[:, :, 0, :self.fq_end]
        logits_fk = all_logits[:, :, 1, self.fq_end:self.fk_end]
        logits_fv = all_logits[:, :, 2, self.fk_end:self.fv_end]
        logits_rq = all_logits[:, :, 3, self.fv_end:self.rq_end]
        logits_rk = all_logits[:, :, 4, self.rq_end:self.rk_end]
        logits_rv = all_logits[:, :, 5, self.rk_end:self.rv_end]

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
    v16.3: Complete Pool Separation

    6개의 독립적인 뉴런 풀:
    - FQ: [n_fq, d_model, rank] - Q 압축
    - FK: [n_fk, d_model, rank] - K 압축
    - FV: [n_fv, d_model, rank] - V 압축
    - RQ: [n_rq, rank, d_model] - Q 복원
    - RK: [n_rk, rank, d_model] - K 복원
    - RV: [n_rv, rank, d_model] - V 복원
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

        # Knowledge neurons
        self.knowledge_neurons_K = nn.Parameter(torch.zeros(n_knowledge, self.knowledge_rank))
        self.knowledge_neurons_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

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
        # F group initialization
        for i in range(self.n_fq + self.n_fk + self.n_fv):
            nn.init.orthogonal_(self.f_neurons.data[i])
        # R group initialization
        for i in range(self.n_rq + self.n_rk + self.n_rv):
            nn.init.orthogonal_(self.r_neurons.data[i])
        nn.init.normal_(self.knowledge_neurons_K, std=0.02)
        nn.init.normal_(self.knowledge_neurons_V, std=0.02)


class GlobalSSM(nn.Module):
    """
    Selective SSM + Context Enhancement with Parallel Scan
    (Same as v15/v16)
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
    v16.3: Complete Pool Separation Routing

    6개의 독립적인 뉴런 풀을 라우팅:
    - FQ, FK, FV: 압축 뉴런
    - RQ, RK, RV: 복원 뉴런
    """
    def __init__(self, d_model: int, n_fq: int, n_fk: int, n_fv: int,
                 n_rq: int, n_rk: int, n_rv: int, n_knowledge: int,
                 top_k_fq: int = 8, top_k_fk: int = 8, top_k_fv: int = 8,
                 top_k_rq: int = 8, top_k_rk: int = 8, top_k_rv: int = 8,
                 d_space: int = 64, router_dropout: float = 0.1, token_routing: bool = False,
                 excitability_tau: float = 1.5, excitability_ema_alpha: float = 0.01,
                 excitability_decay_rate: float = 0.99995):
        super().__init__()
        self.d_model = d_model
        self.n_fq = n_fq
        self.n_fk = n_fk
        self.n_fv = n_fv
        self.n_rq = n_rq
        self.n_rk = n_rk
        self.n_rv = n_rv
        self.n_knowledge = n_knowledge
        self.top_k_fq = top_k_fq
        self.top_k_fk = top_k_fk
        self.top_k_fv = top_k_fv
        self.top_k_rq = top_k_rq
        self.top_k_rk = top_k_rk
        self.top_k_rv = top_k_rv
        self.token_routing = token_routing

        self.neuron_router = UnifiedNeuronRouter(
            d_model, n_fq, n_fk, n_fv, n_rq, n_rk, n_rv, n_knowledge,
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
        """
        Returns: fq_weights, fk_weights, fv_weights, rq_weights, rk_weights, rv_weights, routing_info, aux_loss
        """
        # Get all logits at once (stack + single einsum)
        fq_logits, fk_logits, fv_logits, rq_logits, rk_logits, rv_logits = self.neuron_router.get_all_logits(x)

        fq_pref = F.softmax(fq_logits, dim=-1)
        fk_pref = F.softmax(fk_logits, dim=-1)
        fv_pref = F.softmax(fv_logits, dim=-1)
        rq_pref = F.softmax(rq_logits, dim=-1)
        rk_pref = F.softmax(rk_logits, dim=-1)
        rv_pref = F.softmax(rv_logits, dim=-1)

        # Compute aux_loss (load balancing)
        aux_loss = 0.0
        if self.training:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                count = mask.sum() + 1e-8
                usage_fq = (fq_pref * mask).sum(dim=(0, 1)) / count
                usage_fk = (fk_pref * mask).sum(dim=(0, 1)) / count
                usage_fv = (fv_pref * mask).sum(dim=(0, 1)) / count
                usage_rq = (rq_pref * mask).sum(dim=(0, 1)) / count
                usage_rk = (rk_pref * mask).sum(dim=(0, 1)) / count
                usage_rv = (rv_pref * mask).sum(dim=(0, 1)) / count
            else:
                usage_fq = fq_pref.mean(dim=(0, 1))
                usage_fk = fk_pref.mean(dim=(0, 1))
                usage_fv = fv_pref.mean(dim=(0, 1))
                usage_rq = rq_pref.mean(dim=(0, 1))
                usage_rk = rk_pref.mean(dim=(0, 1))
                usage_rv = rv_pref.mean(dim=(0, 1))

            target_fq = 1.0 / self.n_fq
            target_fk = 1.0 / self.n_fk
            target_fv = 1.0 / self.n_fv
            target_rq = 1.0 / self.n_rq
            target_rk = 1.0 / self.n_rk
            target_rv = 1.0 / self.n_rv

            aux_loss += ((usage_fq - target_fq) ** 2).sum() * self.n_fq
            aux_loss += ((usage_fk - target_fk) ** 2).sum() * self.n_fk
            aux_loss += ((usage_fv - target_fv) ** 2).sum() * self.n_fv
            aux_loss += ((usage_rq - target_rq) ** 2).sum() * self.n_rq
            aux_loss += ((usage_rk - target_rk) ** 2).sum() * self.n_rk
            aux_loss += ((usage_rv - target_rv) ** 2).sum() * self.n_rv

        if self.token_routing:
            fq_weights = fq_pref
            fk_weights = fk_pref
            fv_weights = fv_pref
            rq_weights = rq_pref
            rk_weights = rk_pref
            rv_weights = rv_pref

            routing_info = {
                'fq_weights': fq_weights.detach(),
                'fk_weights': fk_weights.detach(),
                'fv_weights': fv_weights.detach(),
                'rq_weights': rq_weights.detach(),
                'rk_weights': rk_weights.detach(),
                'rv_weights': rv_weights.detach(),
                'token_routing': True,
            }
        else:
            # Batch-level routing
            fq_weights_dense = torch.einsum('bs,bsn->bn', importance, fq_pref)
            fk_weights_dense = torch.einsum('bs,bsn->bn', importance, fk_pref)
            fv_weights_dense = torch.einsum('bs,bsn->bn', importance, fv_pref)
            rq_weights_dense = torch.einsum('bs,bsn->bn', importance, rq_pref)
            rk_weights_dense = torch.einsum('bs,bsn->bn', importance, rk_pref)
            rv_weights_dense = torch.einsum('bs,bsn->bn', importance, rv_pref)

            fq_weights, _ = self._topk_sparsify(fq_weights_dense, self.top_k_fq)
            fk_weights, _ = self._topk_sparsify(fk_weights_dense, self.top_k_fk)
            fv_weights, _ = self._topk_sparsify(fv_weights_dense, self.top_k_fv)
            rq_weights, _ = self._topk_sparsify(rq_weights_dense, self.top_k_rq)
            rk_weights, _ = self._topk_sparsify(rk_weights_dense, self.top_k_rk)
            rv_weights, _ = self._topk_sparsify(rv_weights_dense, self.top_k_rv)

            routing_info = {
                'fq_weights': fq_weights.detach(),
                'fk_weights': fk_weights.detach(),
                'fv_weights': fv_weights.detach(),
                'rq_weights': rq_weights.detach(),
                'rk_weights': rk_weights.detach(),
                'rv_weights': rv_weights.detach(),
                # v16.3: separate prefs for logging
                'fq_pref': fq_pref.detach(),
                'fk_pref': fk_pref.detach(),
                'fv_pref': fv_pref.detach(),
                'rq_pref': rq_pref.detach(),
                'rk_pref': rk_pref.detach(),
                'rv_pref': rv_pref.detach(),
                'token_routing': False,
            }

        # Update usage statistics
        if self.training:
            self.neuron_router.update_usage(fq_weights, 'fq', attention_mask)
            self.neuron_router.update_usage(fk_weights, 'fk', attention_mask)
            self.neuron_router.update_usage(fv_weights, 'fv', attention_mask)
            self.neuron_router.update_usage(rq_weights, 'rq', attention_mask)
            self.neuron_router.update_usage(rk_weights, 'rk', attention_mask)
            self.neuron_router.update_usage(rv_weights, 'rv', attention_mask)

        return fq_weights, fk_weights, fv_weights, rq_weights, rk_weights, rv_weights, routing_info, aux_loss


class NeuronCircuit(nn.Module):
    """
    v16.3: Attention circuit with Complete Pool Separation

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

    def forward(self, x, fq_weights, fk_weights, fv_weights, rq_weights, rk_weights, rv_weights, attention_mask=None):
        """
        Args:
            x: [B, S, D]
            fq_weights: [B, N_FQ] - Q compression weights
            fk_weights: [B, N_FK] - K compression weights
            fv_weights: [B, N_FV] - V compression weights
            rq_weights: [B, N_RQ] - Q restoration weights
            rk_weights: [B, N_RK] - K restoration weights
            rv_weights: [B, N_RV] - V restoration weights
        """
        B, S, D = x.shape
        token_routing = fq_weights.dim() == 3

        if token_routing:
            # Token-level routing (keep einsum for now)
            shared_fq = torch.einsum('bsn,ndr->bsdr', fq_weights, self.shared_neurons.fq_neurons)
            shared_fk = torch.einsum('bsn,ndr->bsdr', fk_weights, self.shared_neurons.fk_neurons)
            shared_fv = torch.einsum('bsn,ndr->bsdr', fv_weights, self.shared_neurons.fv_neurons)

            h_q = torch.einsum('bsd,bsdr->bsr', x, shared_fq)
            h_k = torch.einsum('bsd,bsdr->bsr', x, shared_fk)
            h_v = torch.einsum('bsd,bsdr->bsr', x, shared_fv)

            shared_rq = torch.einsum('bsn,nrd->bsrd', rq_weights, self.shared_neurons.rq_neurons)
            shared_rk = torch.einsum('bsn,nrd->bsrd', rk_weights, self.shared_neurons.rk_neurons)
            shared_rv = torch.einsum('bsn,nrd->bsrd', rv_weights, self.shared_neurons.rv_neurons)

            Q = torch.einsum('bsr,bsrd->bsd', h_q, shared_rq)
            K = torch.einsum('bsr,bsrd->bsd', h_k, shared_rk)
            V = torch.einsum('bsr,bsrd->bsd', h_v, shared_rv)
        else:
            # Batch-level routing - optimized with matmul
            R = self.rank

            # Feature neurons: einsum 'bn,ndr->bdr' → matmul + reshape
            shared_fq = (fq_weights @ self.shared_neurons.fq_neurons.view(-1, D*R)).view(B, D, R)
            shared_fk = (fk_weights @ self.shared_neurons.fk_neurons.view(-1, D*R)).view(B, D, R)
            shared_fv = (fv_weights @ self.shared_neurons.fv_neurons.view(-1, D*R)).view(B, D, R)

            # Compression: einsum 'bsd,bdr->bsr' → bmm
            h_q = torch.bmm(x, shared_fq)  # [B, S, R]
            h_k = torch.bmm(x, shared_fk)
            h_v = torch.bmm(x, shared_fv)

            # Restore neurons: einsum 'bn,nrd->brd' → matmul + reshape
            shared_rq = (rq_weights @ self.shared_neurons.rq_neurons.view(-1, R*D)).view(B, R, D)
            shared_rk = (rk_weights @ self.shared_neurons.rk_neurons.view(-1, R*D)).view(B, R, D)
            shared_rv = (rv_weights @ self.shared_neurons.rv_neurons.view(-1, R*D)).view(B, R, D)

            # Restoration: einsum 'bsr,brd->bsd' → bmm
            Q = torch.bmm(h_q, shared_rq)  # [B, S, D]
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


class NeuronMemory(nn.Module):
    """
    v16.3: 2-stage hierarchical knowledge retrieval (same as v15/v16)
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

        # Update knowledge usage statistics
        if self.training:
            coarse_indicator = torch.zeros(B, S, self.n_knowledge, device=x.device)
            coarse_indicator.scatter_(-1, candidate_idx, 1.0)
            router.update_usage(coarse_indicator, 'knowledge', attention_mask)
            del coarse_indicator

        # Stage 2: Fine matching within candidates
        query = knowledge_encoder(x)

        K_all = self.shared_neurons.knowledge_neurons_K
        V_all = self.shared_neurons.knowledge_neurons_V

        # Memory-optimized: direct indexing instead of expand+gather
        K_candidates = K_all[candidate_idx]  # [B, S, coarse_k, knowledge_rank]

        fine_scores = torch.einsum('bsd,bscd->bsc', query, K_candidates) / math.sqrt(self.knowledge_rank)

        fine_topk_scores, fine_topk_local_idx = torch.topk(fine_scores, self.fine_k, dim=-1)
        fine_weights = F.softmax(fine_topk_scores, dim=-1)

        fine_global_idx = candidate_idx.gather(-1, fine_topk_local_idx)

        # Memory-optimized: direct indexing instead of expand+gather
        selected_V = V_all[fine_global_idx]  # [B, S, fine_k, d_model]

        output = (selected_V * fine_weights.unsqueeze(-1)).sum(dim=2)

        info = {
            'coarse_indices': candidate_idx.detach(),
            'coarse_scores': coarse_scores.detach(),
            'fine_indices': fine_global_idx.detach(),
            'fine_weights': fine_weights.detach(),
        }

        return output, info


class DAWNBlock(nn.Module):
    """DAWN v16.3 block with Complete Pool Separation"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_knowledge: int,
        knowledge_rank: int = None,
        coarse_k: int = 20,
        fine_k: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = NeuronCircuit(shared_neurons, d_model, n_heads, rank, dropout)
        self.memory = NeuronMemory(
            shared_neurons, d_model, n_knowledge,
            knowledge_rank=knowledge_rank, coarse_k=coarse_k, fine_k=fine_k
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, importance, global_routers: GlobalRouters, knowledge_encoder, attention_mask=None):
        normed_x = self.norm1(x)
        fq_w, fk_w, fv_w, rq_w, rk_w, rv_w, attn_routing, attn_aux_loss = \
            global_routers.get_attention_weights(normed_x, importance, attention_mask)

        attn_out, _ = self.attn(normed_x, fq_w, fk_w, fv_w, rq_w, rk_w, rv_w, attention_mask)
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
    DAWN v16.3: Complete Q/K/V Pool Separation

    v16.2 기반 + 완전한 풀 분리:
    - FQ/FK/FV: 각각 독립적인 압축 뉴런 풀
    - RQ/RK/RV: 각각 독립적인 복원 뉴런 풀

    Routing paths:
    - Q path: x → FQ → h_q → RQ → Q
    - K path: x → FK → h_k → RK → K
    - V path: x → FV → h_v → RV → V
    """
    __version__ = "16.3"

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
        # Knowledge
        n_knowledge: int = 256,
        coarse_k: int = 40,
        fine_k: int = 15,
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

        # Knowledge
        self.n_knowledge = n_knowledge
        self.coarse_k = coarse_k
        self.fine_k = fine_k

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
            d_space=d_space, router_dropout=router_dropout, token_routing=token_routing,
            excitability_tau=excitability_tau, excitability_ema_alpha=excitability_ema_alpha,
            excitability_decay_rate=excitability_decay_rate
        )

        self.knowledge_encoder = nn.Linear(d_model, self.knowledge_rank, bias=False)

        self.layers = nn.ModuleList([
            DAWNBlock(
                shared_neurons=self.shared_neurons, d_model=d_model, n_heads=n_heads,
                rank=rank, n_knowledge=n_knowledge, knowledge_rank=self.knowledge_rank,
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

    def orthogonality_loss(self):
        I = torch.eye(self.rank, device=self.shared_neurons.fq_neurons.device).unsqueeze(0)

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

        return (loss_fq + loss_fk + loss_fv + loss_rq + loss_rk + loss_rv) / 6

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
        fq = self.shared_neurons.fq_neurons.numel()
        fk = self.shared_neurons.fk_neurons.numel()
        fv = self.shared_neurons.fv_neurons.numel()
        rq = self.shared_neurons.rq_neurons.numel()
        rk = self.shared_neurons.rk_neurons.numel()
        rv = self.shared_neurons.rv_neurons.numel()
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

        expand_o = self.layers[0].attn.expand_O.weight.numel() * self.n_layers
        norms = sum(p.numel() for n, p in self.named_parameters() if 'norm' in n)

        print(f"=== DAWN v16.3 Parameter Breakdown (Complete Pool Separation) ===")
        print(f"FQ Neurons:            {fq:,} ({fq/1e6:.2f}M) [{self.n_fq} × {self.d_model} × {self.rank}]")
        print(f"FK Neurons:            {fk:,} ({fk/1e6:.2f}M) [{self.n_fk} × {self.d_model} × {self.rank}]")
        print(f"FV Neurons:            {fv:,} ({fv/1e6:.2f}M) [{self.n_fv} × {self.d_model} × {self.rank}]")
        print(f"RQ Neurons:            {rq:,} ({rq/1e6:.2f}M) [{self.n_rq} × {self.rank} × {self.d_model}]")
        print(f"RK Neurons:            {rk:,} ({rk/1e6:.2f}M) [{self.n_rk} × {self.rank} × {self.d_model}]")
        print(f"RV Neurons:            {rv:,} ({rv/1e6:.2f}M) [{self.n_rv} × {self.rank} × {self.d_model}]")
        print(f"expand_O:              {expand_o:,} ({expand_o/1e3:.1f}K)")
        print(f"Knowledge Neurons:     {knowledge:,} ({knowledge/1e3:.1f}K)")
        print(f"Embeddings:            {embed:,} ({embed/1e6:.2f}M)")
        print(f"Mamba SSM:             {ssm_total:,} ({ssm_total/1e3:.1f}K)")
        print(f"Unified Router:        {routers:,} ({routers/1e3:.1f}K) [d_space={self.d_space}]")
        print(f"LayerNorms:            {norms:,} ({norms/1e3:.1f}K)")
        print(f"---")
        print(f"Top-k FQ: {self.top_k_fq}/{self.n_fq}, FK: {self.top_k_fk}/{self.n_fk}, FV: {self.top_k_fv}/{self.n_fv}")
        print(f"Top-k RQ: {self.top_k_rq}/{self.n_rq}, RK: {self.top_k_rk}/{self.n_rk}, RV: {self.top_k_rv}/{self.n_rv}")
        print(f"Mamba Available: {MAMBA_AVAILABLE}")
        print(f"---")
        print(f"Total:                 {self.count_parameters():,} ({self.count_parameters()/1e6:.2f}M)")

        return {
            'fq': fq, 'fk': fk, 'fv': fv,
            'rq': rq, 'rk': rk, 'rv': rv,
            'expand_o': expand_o, 'knowledge': knowledge,
            'embeddings': embed, 'ssm': ssm_total,
            'routers': routers, 'norms': norms,
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
            'coarse_k': self.coarse_k, 'fine_k': self.fine_k,
            'state_dim': self.state_dim, 'd_space': self.d_space,
            'router_dropout': self.router_dropout,
            'gradient_checkpointing': self.gradient_checkpointing,
            'token_routing': self.token_routing,
        }

    def get_model_info(self):
        """Return model architecture info for logging"""
        return [
            f"DAWN v{self.__version__}: Complete Q/K/V Pool Separation",
            f"  rank={self.rank}, knowledge_rank={self.knowledge_rank}",
            f"  FQ: {self.n_fq} × {self.d_model} × {self.rank} (top-k={self.top_k_fq})",
            f"  FK: {self.n_fk} × {self.d_model} × {self.rank} (top-k={self.top_k_fk})",
            f"  FV: {self.n_fv} × {self.d_model} × {self.rank} (top-k={self.top_k_fv})",
            f"  RQ: {self.n_rq} × {self.rank} × {self.d_model} (top-k={self.top_k_rq})",
            f"  RK: {self.n_rk} × {self.rank} × {self.d_model} (top-k={self.top_k_rk})",
            f"  RV: {self.n_rv} × {self.rank} × {self.d_model} (top-k={self.top_k_rv})",
            f"  Knowledge: {self.n_knowledge} (coarse={self.coarse_k} → fine={self.fine_k})",
            f"  Router: d_space={self.d_space}, Excitability (SAR)",
        ]
