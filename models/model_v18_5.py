"""
DAWN v18.5: Context-Aware Restore Routing

Key Concept:
- Based on v18.4 with context-aware restore routing
- Feature routing: done on input x (same as v18.4)
- Restore routing: done on [h_proj, neuron_context] where:
  - h_proj: intermediate representation h projected to d_space
  - neuron_context: weighted sum of feature neuron embeddings

Changes from v18.4:
- Restore routing now sees feature processing results (h + activated neurons)
- Feature/Restore routing separated in GlobalRouters
- Circuit classes call router for restore routing after feature processing
- Path-parallel batched restore routing (no for loops)

Architecture:
- UnifiedNeuronRouter: Context-aware restore projections added
- GlobalRouters: Feature-only routing + context-aware restore routing
- AttentionCircuit: h projection + restore context generation + router callback
- KnowledgeCircuit: Same pattern as AttentionCircuit
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
    v18.5: Context-aware restore routing

    Feature routing: 3 attention projections (fqk_Q, fqk_K, fv) + 1 knowledge projection
    Restore routing: Context-based projections (from h_proj + neuron_context)
    """
    def __init__(self, d_model, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
                 n_feature_know, n_restore_know,
                 d_space=64, dropout=0.1, fixed_tau=0.0, **kwargs):
        super().__init__()
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.d_space = d_space
        self.fixed_tau = fixed_tau
        self.ema_alpha = kwargs.get('excitability_ema_alpha', 0.01)

        # 6 attention + 2 knowledge pools
        total_neurons = (n_feature_qk + n_feature_v + n_restore_qk + n_restore_v +
                        n_feature_know + n_restore_know)
        self.total_neurons = total_neurons

        # Index boundaries
        self.feature_qk_end = n_feature_qk
        self.feature_v_end = n_feature_qk + n_feature_v
        self.restore_qk_end = n_feature_qk + n_feature_v + n_restore_qk
        self.restore_v_end = n_feature_qk + n_feature_v + n_restore_qk + n_restore_v
        self.feature_know_end = self.restore_v_end + n_feature_know
        # restore_know: feature_know_end ~ total_neurons

        # v18.5: Feature-only projections (restore moved to context-based)
        self.proj_feature = nn.Linear(d_model, d_space * 3)  # fqk_Q, fqk_K, fv
        self.proj_feature_know = nn.Linear(d_model, d_space)
        self.dropout = nn.Dropout(dropout)

        # LayerNorm for feature projections
        self.norm_fqk_Q = nn.LayerNorm(d_space)
        self.norm_fqk_K = nn.LayerNorm(d_space)
        self.norm_fv = nn.LayerNorm(d_space)
        self.norm_feature_know = nn.LayerNorm(d_space)

        # v18.5: Context-aware restore routing projections (from 2*d_space context)
        self.proj_restore_qk_from_ctx = nn.Linear(2 * d_space, d_space)
        self.proj_restore_v_from_ctx = nn.Linear(2 * d_space, d_space)
        self.proj_restore_know_from_ctx = nn.Linear(2 * d_space, d_space)

        self.norm_restore_qk_ctx = nn.LayerNorm(d_space)
        self.norm_restore_v_ctx = nn.LayerNorm(d_space)
        self.norm_restore_know_ctx = nn.LayerNorm(d_space)

        # Unified neuron embeddings (std=0.02 is standard transformer initialization)
        self.neuron_emb = nn.Parameter(torch.randn(total_neurons, d_space) * 0.02)

        # Usage tracking (for logging)
        self.register_buffer('usage_ema_feature_q', torch.zeros(n_feature_qk))
        self.register_buffer('usage_ema_feature_k', torch.zeros(n_feature_qk))
        self.register_buffer('usage_ema_feature_v', torch.zeros(n_feature_v))
        self.register_buffer('usage_ema_restore_q', torch.zeros(n_restore_qk))
        self.register_buffer('usage_ema_restore_k', torch.zeros(n_restore_qk))
        self.register_buffer('usage_ema_restore_v', torch.zeros(n_restore_v))
        self.register_buffer('usage_ema_feature_know', torch.zeros(n_feature_know))
        self.register_buffer('usage_ema_restore_know', torch.zeros(n_restore_know))

    def get_thresholds(self, x):
        """
        Return fixed tau for all pools (no learnable threshold)
        x: [B, S, d_model]
        Returns: dict of scalar thresholds
        """
        return {
            'fqk': self.fixed_tau,
            'fv': self.fixed_tau,
            'rqk': self.fixed_tau,
            'rv': self.fixed_tau,
            'feature_know': self.fixed_tau,
            'restore_know': self.fixed_tau,
        }

    def get_feature_knowledge_logits(self, x):
        """
        v18.5: Return feature knowledge logits only (restore is context-based)
        x: [B, S, d_model]
        """
        emb_norm = F.normalize(self.neuron_emb, dim=-1)

        h_feature_know = self.norm_feature_know(self.dropout(self.proj_feature_know(x)))
        emb_feature_know = emb_norm[self.restore_v_end:self.feature_know_end]
        logits_feature_know = torch.einsum('bsd,nd->bsn', h_feature_know, emb_feature_know)

        return logits_feature_know

    def get_feature_attention_logits(self, x):
        """
        v18.5: Return 3 feature attention logits only (restore is context-based)
        x: [B, S, d_model]
        """
        emb_norm = F.normalize(self.neuron_emb, dim=-1)

        all_proj = self.dropout(self.proj_feature(x))
        h_fqk_Q, h_fqk_K, h_fv = all_proj.chunk(3, dim=-1)

        # Apply LayerNorm to each projection
        h_fqk_Q = self.norm_fqk_Q(h_fqk_Q)
        h_fqk_K = self.norm_fqk_K(h_fqk_K)
        h_fv = self.norm_fv(h_fv)

        fqk_emb = emb_norm[:self.feature_qk_end]
        fv_emb = emb_norm[self.feature_qk_end:self.feature_v_end]

        logits_fqk_Q = torch.einsum('bsd,nd->bsn', h_fqk_Q, fqk_emb)
        logits_fqk_K = torch.einsum('bsd,nd->bsn', h_fqk_K, fqk_emb)
        logits_fv = torch.einsum('bsd,nd->bsn', h_fv, fv_emb)

        return logits_fqk_Q, logits_fqk_K, logits_fv

    def get_restore_qk_logits_from_context(self, ctx):
        """
        v18.5: Context-aware restore QK routing
        ctx: [P, B, S, 2*d_space] or [P*B, S, 2*d_space]
        Returns: [P, B, S, N_rqk] or [P*B, S, N_rqk]
        """
        original_shape = ctx.shape
        if ctx.dim() == 4:
            P, B, S, _ = ctx.shape
            ctx = ctx.view(P * B, S, -1)

        emb_norm = F.normalize(self.neuron_emb, dim=-1)
        rqk_emb = emb_norm[self.feature_v_end:self.restore_qk_end]

        h = self.norm_restore_qk_ctx(self.dropout(self.proj_restore_qk_from_ctx(ctx)))
        logits = torch.einsum('bsd,nd->bsn', h, rqk_emb)

        if len(original_shape) == 4:
            logits = logits.view(P, B, S, -1)

        return logits

    def get_restore_v_logits_from_context(self, ctx):
        """
        v18.5: Context-aware restore V routing
        ctx: [P, B, S, 2*d_space] or [P*B, S, 2*d_space]
        Returns: [P, B, S, N_rv] or [P*B, S, N_rv]
        """
        original_shape = ctx.shape
        if ctx.dim() == 4:
            P, B, S, _ = ctx.shape
            ctx = ctx.view(P * B, S, -1)

        emb_norm = F.normalize(self.neuron_emb, dim=-1)
        rv_emb = emb_norm[self.restore_qk_end:self.restore_v_end]

        h = self.norm_restore_v_ctx(self.dropout(self.proj_restore_v_from_ctx(ctx)))
        logits = torch.einsum('bsd,nd->bsn', h, rv_emb)

        if len(original_shape) == 4:
            logits = logits.view(P, B, S, -1)

        return logits

    def get_restore_know_logits_from_context(self, ctx):
        """
        v18.5: Context-aware restore knowledge routing
        ctx: [P, B, S, 2*d_space] or [P*B, S, 2*d_space]
        Returns: [P, B, S, N_restore_know] or [P*B, S, N_restore_know]
        """
        original_shape = ctx.shape
        if ctx.dim() == 4:
            P, B, S, _ = ctx.shape
            ctx = ctx.view(P * B, S, -1)

        emb_norm = F.normalize(self.neuron_emb, dim=-1)
        restore_know_emb = emb_norm[self.feature_know_end:]

        h = self.norm_restore_know_ctx(self.dropout(self.proj_restore_know_from_ctx(ctx)))
        logits = torch.einsum('bsd,nd->bsn', h, restore_know_emb)

        if len(original_shape) == 4:
            logits = logits.view(P, B, S, -1)

        return logits

    def get_fqk_emb(self):
        """Return normalized feature QK neuron embeddings"""
        return F.normalize(self.neuron_emb[:self.feature_qk_end], dim=-1)

    def get_fv_emb(self):
        """Return normalized feature V neuron embeddings"""
        return F.normalize(self.neuron_emb[self.feature_qk_end:self.feature_v_end], dim=-1)

    def get_feature_know_emb(self):
        """Return normalized feature knowledge neuron embeddings"""
        return F.normalize(self.neuron_emb[self.restore_v_end:self.feature_know_end], dim=-1)

    def update_usage(self, weights, neuron_type, attention_mask=None):
        """
        Update usage EMA for neuron tracking.
        v18 uses soft weights (sigmoid outputs 0~1), so we use the actual weight values
        as usage intensity, not binary active/inactive.
        """
        if not self.training:
            return

        if weights.dim() == 3:
            # Use actual soft weight values (not binary) for v18
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                weights_masked = weights * mask
                count = mask.sum() + 1e-8
                usage = weights_masked.sum(dim=[0, 1]) / count
            else:
                usage = weights.mean(dim=[0, 1])
        else:
            usage = weights.mean(dim=0)

        # Detach to prevent memory leak from computation graph retention
        usage = usage.detach()

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
        elif neuron_type == 'feature_know':
            self.usage_ema_feature_know.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'restore_know':
            self.usage_ema_restore_know.mul_(decay).add_(usage, alpha=self.ema_alpha)


class SharedNeurons(nn.Module):
    """
    v18.0: Both Attention + Knowledge use Feature-Restore pattern
    Same structure as v17.1, rank reduced to 16
    """
    def __init__(
        self,
        d_model: int,
        rank: int,
        n_feature_qk: int,
        n_feature_v: int,
        n_restore_qk: int,
        n_restore_v: int,
        n_feature_know: int,
        n_restore_know: int,
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
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know

        # Attention neurons (contiguous memory)
        self.f_neurons = nn.Parameter(torch.zeros(n_feature_qk + n_feature_v, d_model, rank))
        self.r_neurons = nn.Parameter(torch.zeros(n_restore_qk + n_restore_v, rank, d_model))

        # Knowledge neurons: Feature-Restore pattern (separate routing)
        self.feature_know = nn.Parameter(torch.zeros(n_feature_know, d_model, knowledge_rank))
        self.restore_know = nn.Parameter(torch.zeros(n_restore_know, knowledge_rank, d_model))

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
        for i in range(self.n_feature_know):
            nn.init.orthogonal_(self.feature_know.data[i])
        for i in range(self.n_restore_know):
            nn.init.orthogonal_(self.restore_know.data[i])


class GlobalSSM(nn.Module):
    """Selective SSM (same as v17.1)"""
    def __init__(self, d_model: int, state_dim: int, return_context: bool = True):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.return_context = return_context

        # A_log initialization (small values for stable SSM dynamics)
        self.A_log = nn.Parameter(torch.randn(d_model, state_dim) * 0.1)
        self.W_delta = nn.Linear(d_model, d_model, bias=False)
        self.W_B = nn.Linear(d_model, state_dim, bias=False)
        self.W_C = nn.Linear(d_model, state_dim, bias=False)

        self.ssm_norm = nn.LayerNorm(d_model)
        self.context_proj = nn.Linear(d_model, d_model, bias=False)
        # Initial context scale (small to avoid disrupting early training)
        self.context_scale = nn.Parameter(torch.tensor(0.1))
        self.importance_proj = nn.Linear(d_model, d_model, bias=False)
        # Temperature for importance softmax (lower = sharper distribution)
        self.importance_temperature = 0.5

        self._init_weights()

    def _init_weights(self):
        # std=0.02 is standard transformer initialization (GPT-2, BERT)
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

        # Position-wise importance (no future leakage)
        h_proj = self.importance_proj(ssm_out)  # [B, S, D]
        raw_importance = (x * h_proj).sum(dim=-1)  # [B, S]

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
    v18.5: Context-aware restore routing

    Key changes from v18.4:
    1. Feature routing: same as v18.4 (on input x)
    2. Restore routing: context-based (on [h_proj, neuron_context])
    3. Separated feature/restore routing methods
    4. Batched top-k selection for path-aware processing
    """
    def __init__(self, d_model: int, n_feature_qk: int, n_feature_v: int,
                 n_restore_qk: int, n_restore_v: int,
                 n_feature_know: int, n_restore_know: int,
                 rank: int = 16,
                 max_paths: int = 4,
                 fixed_tau: float = 0.0,
                 path_max_k: int = 16,
                 d_space: int = 64, router_dropout: float = 0.1,
                 attention_token_routing: bool = False,
                 knowledge_token_routing: bool = False,
                 learnable_tau: bool = True,
                 tau_reg_weight: float = 0.0,
                 **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_space = d_space
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.rank = rank
        self.max_paths = max_paths
        self.fixed_tau = fixed_tau
        self.path_max_k = path_max_k
        self.attention_token_routing = attention_token_routing
        self.knowledge_token_routing = knowledge_token_routing
        self.learnable_tau = learnable_tau
        self.tau_reg_weight = tau_reg_weight
        self.inference_hard_mask = False
        # ============================================================
        # MODE FLAGS
        # ============================================================
        self.debug_mode = False
        self.store_pref_tensors = False
        self.store_path_weights = False

        # v18.5: Learnable tau for feature routing only (6 pools)
        # Restore tau is computed from context scores
        if learnable_tau:
            # Feature tau projection: [fq, fk, fv, feature_know]
            self.tau_proj_feature = nn.Linear(d_model, 4)
            nn.init.zeros_(self.tau_proj_feature.weight)
            nn.init.constant_(self.tau_proj_feature.bias, -0.5)

        self.neuron_router = UnifiedNeuronRouter(
            d_model, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
            n_feature_know, n_restore_know,
            d_space=d_space, dropout=router_dropout, fixed_tau=fixed_tau, **kwargs
        )

    def get_tau_feature(self, x):
        """
        v18.5: Compute tau for feature routing only.

        Args:
            x: [B, S, d_model] input tensor

        Returns:
            [B, S, 4] tau values for feature pools, or None if not learnable
            Order: [fq, fk, fv, feature_know]
        """
        if self.learnable_tau:
            return self.tau_proj_feature(x)  # [B, S, 4]
        return None

    def get_tau_reg_loss(self):
        """DEPRECATED: tau_reg computed inside routing functions"""
        return 0.0

    def get_all_tau_offset_values(self):
        """
        v18.5: Returns tau_offset values for feature pools only.
        Restore tau is context-dependent and not logged here.
        """
        if self.learnable_tau:
            bias = self.tau_proj_feature.bias.detach()
            return {
                'fq': bias[0].item(),
                'fk': bias[1].item(),
                'fv': bias[2].item(),
                'feature_know': bias[3].item(),
                # Restore tau is context-dependent
                'rq': 0.0,
                'rk': 0.0,
                'rv': 0.0,
                'restore_know': 0.0,
            }
        else:
            return {
                'fq': 0.0, 'fk': 0.0, 'fv': 0.0,
                'rq': 0.0, 'rk': 0.0, 'rv': 0.0,
                'feature_know': 0.0, 'restore_know': 0.0,
            }

    def _topk_select_and_chunk(self, scores, tau, path_max_k, max_paths):
        """
        Optimized routing: top-k selection → tau threshold → exp gate → chunking

        v18.4: Relative tau (calculated from full scores by caller)
        - tau = score_mean + tau_offset * score_std (relative to full neuron pool)
        - gate = score - tau (positive) or 1e-8 * exp(score - tau) (negative, for gradient)
        - exp_gate = exp(gate) - 1 (amplifies differences, 0 when gate=0)
        - gate_strength = tanh(max(exp_gate))
        - scaled_weights = (exp_gate / sum) * gate_strength

        Args:
            scores: [B, S, N] neuron scores
            tau: scalar or [B, S, 1] tensor (threshold value)
            path_max_k: neurons per path
            max_paths: maximum number of paths

        Returns:
            path_weights_list: list of [B, S, N] weights (length = max_paths)
            weights: [B, S, N] sparse weights for aux_loss
            mask: [B, S, N] boolean mask for statistics
            gate: [B, S, k] gate values (with gradient flow)
            scaled_weights: [B, S, k] normalized exp gate weights
        """
        B, S, N = scores.shape
        k = min(path_max_k * max_paths, N)

        # 1. Top-k selection (sorted by descending scores)
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=-1, sorted=True)

        # 2. Threshold mask for top-k neurons (tau already computed by caller)
        topk_mask = (topk_scores > tau)

        # 4. Gate with gradient flow for dead neurons
        raw_gate = topk_scores - tau
        gate = torch.where(
            raw_gate > 0,
            raw_gate,
            1e-8 * torch.exp(raw_gate)  # 음수여도 작은 값 + gradient 흐름
        )

        # 5. Exponential scaling for sparsity + normalization
        exp_gate = torch.exp(gate) - 1  # 차이 극대화, gate=0이면 0
        exp_gate_sum = exp_gate.sum(dim=-1, keepdim=True)

        # 비율 계산
        ratio_weights = torch.where(
            exp_gate_sum > 1e-8,
            exp_gate / (exp_gate_sum + 1e-8),
            exp_gate * 1e-8  # gradient 유지
        )

        # gate_strength: max exp_gate의 tanh로 전체 강도 조절
        gate_strength = torch.tanh(exp_gate.max(dim=-1, keepdim=True).values)
        scaled_weights = ratio_weights * gate_strength

        # 6. Chunk to paths (already sorted by topk)
        out_dtype = scaled_weights.dtype
        path_weights_list = []
        for p in range(max_paths):
            start_idx = p * path_max_k
            end_idx = min((p + 1) * path_max_k, k)

            if start_idx >= k:
                path_weights_list.append(torch.zeros(B, S, N, device=scores.device, dtype=out_dtype))
                continue

            path_weights = torch.zeros(B, S, N, device=scores.device, dtype=out_dtype)
            path_indices = topk_indices[:, :, start_idx:end_idx]
            path_w = scaled_weights[:, :, start_idx:end_idx]

            path_weights.scatter_(dim=-1, index=path_indices, src=path_w)
            path_weights_list.append(path_weights)

        # 7. Create sparse full weights for aux_loss (use scaled_weights)
        weights = torch.zeros(B, S, N, device=scores.device, dtype=out_dtype)
        weights.scatter_(dim=-1, index=topk_indices, src=scaled_weights)

        # 8. Create full mask for statistics
        mask = torch.zeros(B, S, N, device=scores.device, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_indices, src=topk_mask)

        return path_weights_list, weights, mask, gate, scaled_weights

    def _topk_select_batched(self, scores, tau, path_max_k):
        """
        v18.5: Path-batched top-k selection for restore routing.

        Args:
            scores: [P, B, S, N] neuron scores
            tau: [P, B, S, 1] threshold values
            path_max_k: neurons per path

        Returns:
            weights: [P, B, S, N] sparse weights
            mask: [P, B, S, N] boolean mask
            aux_loss: scalar auxiliary loss
        """
        P, B, S, N = scores.shape
        k = min(path_max_k, N)

        # Flatten for batched processing
        scores_flat = scores.view(P * B, S, N)
        tau_flat = tau.view(P * B, S, 1)

        # Top-k
        topk_scores, topk_indices = torch.topk(scores_flat, k=k, dim=-1, sorted=True)

        # Gate
        raw_gate = topk_scores - tau_flat
        gate = torch.where(
            raw_gate > 0,
            raw_gate,
            1e-8 * torch.exp(raw_gate)
        )

        # Normalize
        exp_gate = torch.exp(gate) - 1
        exp_gate_sum = exp_gate.sum(dim=-1, keepdim=True)
        ratio = torch.where(
            exp_gate_sum > 1e-8,
            exp_gate / (exp_gate_sum + 1e-8),
            exp_gate * 1e-8
        )
        gate_strength = torch.tanh(exp_gate.max(dim=-1, keepdim=True).values)
        scaled_weights = ratio * gate_strength

        # Scatter
        weights_flat = torch.zeros(P * B, S, N, device=scores.device, dtype=scores.dtype)
        weights_flat.scatter_(dim=-1, index=topk_indices, src=scaled_weights)

        # Mask
        topk_mask = (topk_scores > tau_flat)
        mask_flat = torch.zeros(P * B, S, N, device=scores.device, dtype=torch.bool)
        mask_flat.scatter_(dim=-1, index=topk_indices, src=topk_mask)

        weights = weights_flat.view(P, B, S, N)
        mask = mask_flat.view(P, B, S, N)

        return weights, mask

    def get_feature_attention_weights(self, x, attention_mask=None, tau_feature=None):
        """
        v18.5: Feature-only attention routing (restore is context-based).

        Args:
            x: [B, S, D] input
            tau_feature: [B, S, 4] pre-computed tau values (optional)
                         Order: [fq, fk, fv, feature_know]

        Returns:
            feature_weights: dict with 'fqk_Q', 'fqk_K', 'fv' (each list of [B, S, N])
            routing_info: dict with routing statistics
            aux_loss: auxiliary loss for load balancing
        """
        fqk_logits_Q, fqk_logits_K, fv_logits = self.neuron_router.get_feature_attention_logits(x)

        # Get tau
        if self.learnable_tau:
            if tau_feature is None:
                tau_feature = self.tau_proj_feature(x)
            tau_offset_fq = tau_feature[..., 0:1]
            tau_offset_fk = tau_feature[..., 1:2]
            tau_offset_fv = tau_feature[..., 2:3]

            tau_fq = fqk_logits_Q.mean(dim=-1, keepdim=True) + tau_offset_fq * (fqk_logits_Q.std(dim=-1, keepdim=True) + 1e-8)
            tau_fk = fqk_logits_K.mean(dim=-1, keepdim=True) + tau_offset_fk * (fqk_logits_K.std(dim=-1, keepdim=True) + 1e-8)
            tau_fv = fv_logits.mean(dim=-1, keepdim=True) + tau_offset_fv * (fv_logits.std(dim=-1, keepdim=True) + 1e-8)
        else:
            tau_offset_fq = tau_offset_fk = tau_offset_fv = 0.0
            tau_fq = fqk_logits_Q.mean(dim=-1, keepdim=True)
            tau_fk = fqk_logits_K.mean(dim=-1, keepdim=True)
            tau_fv = fv_logits.mean(dim=-1, keepdim=True)

        # Top-k selection
        fqk_paths_Q, fqk_weights_Q, fqk_mask_Q, fqk_gate_Q, _ = self._topk_select_and_chunk(
            fqk_logits_Q, tau_fq, self.path_max_k, self.max_paths)
        fqk_paths_K, fqk_weights_K, fqk_mask_K, fqk_gate_K, _ = self._topk_select_and_chunk(
            fqk_logits_K, tau_fk, self.path_max_k, self.max_paths)
        fv_paths, fv_weights, fv_mask, fv_gate, _ = self._topk_select_and_chunk(
            fv_logits, tau_fv, self.path_max_k, self.max_paths)

        # Aux loss (feature only)
        aux_loss = 0.0
        if self.training:
            fqk_pref_Q = F.softmax(fqk_logits_Q, dim=-1)
            fqk_pref_K = F.softmax(fqk_logits_K, dim=-1)
            fv_pref = F.softmax(fv_logits, dim=-1)

            if attention_mask is not None:
                seq_mask = attention_mask.unsqueeze(-1).float()
                count = seq_mask.sum() + 1e-8
                usage_fqk_Q = (fqk_pref_Q * seq_mask).sum(dim=(0, 1)) / count
                usage_fqk_K = (fqk_pref_K * seq_mask).sum(dim=(0, 1)) / count
                usage_fv = (fv_pref * seq_mask).sum(dim=(0, 1)) / count
            else:
                usage_fqk_Q = fqk_pref_Q.mean(dim=(0, 1))
                usage_fqk_K = fqk_pref_K.mean(dim=(0, 1))
                usage_fv = fv_pref.mean(dim=(0, 1))

            target_fqk = 1.0 / self.n_feature_qk
            target_fv = 1.0 / self.n_feature_v

            aux_loss += ((usage_fqk_Q - target_fqk) ** 2).sum() * self.n_feature_qk
            aux_loss += ((usage_fqk_K - target_fqk) ** 2).sum() * self.n_feature_qk
            aux_loss += ((usage_fv - target_fv) ** 2).sum() * self.n_feature_v

        # Tau regularization
        if self.training and self.learnable_tau and self.tau_reg_weight > 0:
            tau_reg = F.relu(tau_offset_fq).mean()
            tau_reg += F.relu(tau_offset_fk).mean()
            tau_reg += F.relu(tau_offset_fv).mean()
            aux_loss += tau_reg * self.tau_reg_weight

        feature_weights = {
            'fqk_Q': fqk_paths_Q,
            'fqk_K': fqk_paths_K,
            'fv': fv_paths,
        }

        # Debug info
        if self.debug_mode:
            with torch.no_grad():
                routing_info = {
                    'top_k': self.path_max_k * self.max_paths,
                    'selected_fqk_Q': fqk_mask_Q.float().sum(dim=-1).mean().item(),
                    'selected_fqk_K': fqk_mask_K.float().sum(dim=-1).mean().item(),
                    'selected_fv': fv_mask.float().sum(dim=-1).mean().item(),
                    'tau_offset_fq': tau_offset_fq.mean().item() if self.learnable_tau else 0.0,
                    'tau_offset_fk': tau_offset_fk.mean().item() if self.learnable_tau else 0.0,
                    'tau_offset_fv': tau_offset_fv.mean().item() if self.learnable_tau else 0.0,
                    'gstr_fq': torch.tanh(fqk_gate_Q.max(dim=-1).values).mean().item(),
                    'gstr_fk': torch.tanh(fqk_gate_K.max(dim=-1).values).mean().item(),
                    'gstr_fv': torch.tanh(fv_gate.max(dim=-1).values).mean().item(),
                    'overlap_fqk': ((fqk_mask_Q & fqk_mask_K).float().sum(dim=-1) /
                                   torch.maximum(fqk_mask_Q.float().sum(dim=-1),
                                                fqk_mask_K.float().sum(dim=-1)).clamp(min=1)).mean().item(),
                }
        else:
            routing_info = {}

        if self.store_pref_tensors:
            routing_info['fqk_q_pref'] = fqk_logits_Q
            routing_info['fqk_k_pref'] = fqk_logits_K
            routing_info['fv_pref'] = fv_logits
            routing_info['fqk_weights_Q'] = fqk_weights_Q
            routing_info['fqk_weights_K'] = fqk_weights_K
            routing_info['fv_weights'] = fv_weights

        # Update usage
        if self.training:
            self.neuron_router.update_usage(fqk_mask_Q.float(), 'feature_q', attention_mask)
            self.neuron_router.update_usage(fqk_mask_K.float(), 'feature_k', attention_mask)
            self.neuron_router.update_usage(fv_mask.float(), 'feature_v', attention_mask)

        return feature_weights, routing_info, aux_loss

    def get_restore_attention_weights(self, ctx, pool_type, attention_mask=None):
        """
        v18.5: Context-aware restore attention routing.

        Args:
            ctx: [P, B, S, 2*d_space] context (h_proj + neuron_context)
            pool_type: 'qk' or 'v'
            attention_mask: [B, S] optional mask

        Returns:
            weights: [P, B, S, N] sparse weights
            aux_loss: auxiliary loss
            routing_info: dict with routing stats (only when debug_mode=True)
        """
        P, B, S, _ = ctx.shape

        if pool_type == 'qk':
            logits = self.neuron_router.get_restore_qk_logits_from_context(ctx)  # [P, B, S, N_rqk]
            n_neurons = self.n_restore_qk
        else:
            logits = self.neuron_router.get_restore_v_logits_from_context(ctx)   # [P, B, S, N_rv]
            n_neurons = self.n_restore_v

        # Compute tau from context scores (mean-based, no learnable offset for restore)
        tau = logits.mean(dim=-1, keepdim=True)

        # Batched top-k selection
        weights, mask = self._topk_select_batched(logits, tau, self.path_max_k)

        # Aux loss
        aux_loss = 0.0
        if self.training:
            pref = F.softmax(logits, dim=-1)
            usage = pref.mean(dim=(0, 1, 2))  # Average over P, B, S
            target = 1.0 / n_neurons
            aux_loss = ((usage - target) ** 2).sum() * n_neurons

        # Update usage
        if self.training:
            neuron_type = 'restore_q' if pool_type == 'qk' else 'restore_v'
            # Average mask over paths for usage tracking
            mask_avg = mask.float().mean(dim=0)  # [B, S, N]
            self.neuron_router.update_usage(mask_avg, neuron_type, attention_mask)

        # Debug info
        routing_info = {}
        if self.debug_mode:
            with torch.no_grad():
                selected = mask.float().sum(dim=-1).mean().item()
                routing_info[f'selected_r{pool_type}'] = selected

        return weights, aux_loss, routing_info

    def get_feature_knowledge_weights(self, x, attention_mask=None, tau_feature=None):
        """
        v18.5: Feature-only knowledge routing (restore is context-based).

        Args:
            x: [B, S, D] input
            tau_feature: [B, S, 4] pre-computed tau values (optional)

        Returns:
            feature_paths: list of [B, S, N] weights
            routing_info: dict with routing statistics
            aux_loss: auxiliary loss
        """
        logits_f = self.neuron_router.get_feature_knowledge_logits(x)

        # Get tau
        if self.learnable_tau:
            if tau_feature is None:
                tau_feature = self.tau_proj_feature(x)
            tau_offset_f = tau_feature[..., 3:4]
            tau_f = logits_f.mean(dim=-1, keepdim=True) + tau_offset_f * (logits_f.std(dim=-1, keepdim=True) + 1e-8)
        else:
            tau_offset_f = 0.0
            tau_f = logits_f.mean(dim=-1, keepdim=True)

        # Top-k selection
        f_paths, f_weights, f_mask, f_gate, _ = self._topk_select_and_chunk(
            logits_f, tau_f, self.path_max_k, self.max_paths)

        # Aux loss
        aux_loss = 0.0
        if self.training:
            f_pref = F.softmax(logits_f, dim=-1)

            if attention_mask is not None:
                seq_mask = attention_mask.unsqueeze(-1).float()
                count = seq_mask.sum() + 1e-8
                usage_f = (f_pref * seq_mask).sum(dim=(0, 1)) / count
            else:
                usage_f = f_pref.mean(dim=(0, 1))

            target_f = 1.0 / self.n_feature_know
            aux_loss += ((usage_f - target_f) ** 2).sum() * self.n_feature_know

        # Tau regularization
        if self.training and self.learnable_tau and self.tau_reg_weight > 0:
            tau_reg = F.relu(tau_offset_f).mean()
            aux_loss += tau_reg * self.tau_reg_weight

        # Update usage
        if self.training:
            self.neuron_router.update_usage(f_mask.float(), 'feature_know', attention_mask)

        # Debug info
        if self.debug_mode:
            with torch.no_grad():
                know_info = {
                    'top_k': self.path_max_k * self.max_paths,
                    'selected_feature': f_mask.float().sum(dim=-1).mean().item(),
                    'tau_offset_feature': tau_offset_f.mean().item() if self.learnable_tau else 0.0,
                    'gstr_feature': torch.tanh((torch.exp(f_gate) - 1).max(dim=-1).values).mean().item(),
                }
        else:
            know_info = {}

        return f_paths, know_info, aux_loss

    def get_restore_knowledge_weights(self, ctx, attention_mask=None):
        """
        v18.5: Context-aware restore knowledge routing.

        Args:
            ctx: [P, B, S, 2*d_space] context (h_proj + neuron_context)
            attention_mask: [B, S] optional mask

        Returns:
            weights: [P, B, S, N] sparse weights
            aux_loss: auxiliary loss
            routing_info: dict with routing stats (only when debug_mode=True)
        """
        P, B, S, _ = ctx.shape

        logits = self.neuron_router.get_restore_know_logits_from_context(ctx)  # [P, B, S, N_r]

        # Compute tau from context scores
        tau = logits.mean(dim=-1, keepdim=True)

        # Batched top-k selection
        weights, mask = self._topk_select_batched(logits, tau, self.path_max_k)

        # Aux loss
        aux_loss = 0.0
        if self.training:
            pref = F.softmax(logits, dim=-1)
            usage = pref.mean(dim=(0, 1, 2))
            target = 1.0 / self.n_restore_know
            aux_loss = ((usage - target) ** 2).sum() * self.n_restore_know

        # Update usage
        if self.training:
            mask_avg = mask.float().mean(dim=0)
            self.neuron_router.update_usage(mask_avg, 'restore_know', attention_mask)

        # Debug info
        routing_info = {}
        if self.debug_mode:
            with torch.no_grad():
                selected = mask.float().sum(dim=-1).mean().item()
                routing_info['selected_restore'] = selected

        return weights, aux_loss, routing_info


class AttentionCircuit(nn.Module):
    """
    v18.5: Multi-path attention circuit with context-aware restore routing.

    Feature processing: same as v18.4
    Restore routing: context-based (h_proj + neuron_context) -> router callback
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        d_space: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank = rank
        self.d_space = d_space

        self.expand_O = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # v18.5: h -> d_space projection for restore context
        self.proj_h_qk = nn.Linear(rank, d_space)
        self.proj_h_v = nn.Linear(rank, d_space)

    def forward(self, x, feature_weights, router, attention_mask=None):
        """
        v18.5: Feature processing + context-aware restore routing.

        Args:
            x: [B, S, D] input
            feature_weights: dict with 'fqk_Q', 'fqk_K', 'fv' (each list of [B, S, N])
            router: GlobalRouters instance for restore routing callback

        Returns:
            output: [B, S, D]
            restore_aux_loss: auxiliary loss from restore routing
        """
        B, S, D = x.shape
        P = len(feature_weights['fqk_Q'])  # max_paths

        # Stack feature weights: [P, B, S, N]
        fqk_w_Q_stacked = torch.stack(feature_weights['fqk_Q'], dim=0)
        fqk_w_K_stacked = torch.stack(feature_weights['fqk_K'], dim=0)
        fv_w_stacked = torch.stack(feature_weights['fv'], dim=0)

        # ===== Feature 처리 =====
        f_qk = self.shared_neurons.feature_qk_neurons  # [N_fqk, D, R]
        f_v = self.shared_neurons.feature_v_neurons    # [N_fv, D, R]

        all_h_qk = torch.einsum('bsd,ndr->bsnr', x, f_qk)  # [B, S, N_fqk, R]
        all_h_v = torch.einsum('bsd,ndr->bsnr', x, f_v)    # [B, S, N_fv, R]

        h_q_all = torch.einsum('bsnr,pbsn->pbsr', all_h_qk, fqk_w_Q_stacked)  # [P, B, S, R]
        h_k_all = torch.einsum('bsnr,pbsn->pbsr', all_h_qk, fqk_w_K_stacked)
        h_v_all = torch.einsum('bsnr,pbsn->pbsr', all_h_v, fv_w_stacked)

        # ===== Restore context 생성 =====
        fqk_emb = router.neuron_router.get_fqk_emb()  # [N_fqk, d_space]
        fv_emb = router.neuron_router.get_fv_emb()    # [N_fv, d_space]

        # h projection (Q/K 합쳐서 효율적 처리)
        h_qk_cat = torch.cat([h_q_all, h_k_all], dim=0)  # [2P, B, S, R]
        h_qk_proj = self.proj_h_qk(h_qk_cat.view(2*P*B, S, -1)).view(2*P, B, S, self.d_space)
        h_q_proj, h_k_proj = h_qk_proj[:P], h_qk_proj[P:]  # 각 [P, B, S, d_space]

        h_v_proj = self.proj_h_v(h_v_all.view(P*B, S, -1)).view(P, B, S, self.d_space)

        # neuron context: weighted sum of feature embeddings
        ctx_q = torch.einsum('pbsn,nd->pbsd', fqk_w_Q_stacked, fqk_emb)  # [P, B, S, d_space]
        ctx_k = torch.einsum('pbsn,nd->pbsd', fqk_w_K_stacked, fqk_emb)
        ctx_v = torch.einsum('pbsn,nd->pbsd', fv_w_stacked, fv_emb)

        # concat: [h_proj, neuron_context]
        rqk_Q_input = torch.cat([h_q_proj, ctx_q], dim=-1)  # [P, B, S, 2*d_space]
        rqk_K_input = torch.cat([h_k_proj, ctx_k], dim=-1)
        rv_input = torch.cat([h_v_proj, ctx_v], dim=-1)

        # ===== Restore 라우팅 (router callback) =====
        rqk_Q_weights, aux_rqk_Q, info_rqk_Q = router.get_restore_attention_weights(rqk_Q_input, 'qk', attention_mask)
        rqk_K_weights, aux_rqk_K, info_rqk_K = router.get_restore_attention_weights(rqk_K_input, 'qk', attention_mask)
        rv_weights, aux_rv, info_rv = router.get_restore_attention_weights(rv_input, 'v', attention_mask)

        restore_aux_loss = aux_rqk_Q + aux_rqk_K + aux_rv

        # Merge restore routing info
        restore_info = {}
        if info_rqk_Q:
            restore_info['selected_rqk_Q'] = info_rqk_Q.get('selected_rqk', 0)
        if info_rqk_K:
            restore_info['selected_rqk_K'] = info_rqk_K.get('selected_rqk', 0)
        if info_rv:
            restore_info['selected_rv'] = info_rv.get('selected_rv', 0)

        # ===== Restore 처리 =====
        r_qk = self.shared_neurons.restore_qk_neurons  # [N_rqk, R, D]
        r_v = self.shared_neurons.restore_v_neurons    # [N_rv, R, D]

        Q_all = torch.einsum('pbsr,nrd,pbsn->pbsd', h_q_all, r_qk, rqk_Q_weights)  # [P, B, S, D]
        K_all = torch.einsum('pbsr,nrd,pbsn->pbsd', h_k_all, r_qk, rqk_K_weights)
        V_all = torch.einsum('pbsr,nrd,pbsn->pbsd', h_v_all, r_v, rv_weights)

        # Sum over paths
        Q_total = Q_all.sum(dim=0)  # [B, S, D]
        K_total = K_all.sum(dim=0)
        V_total = V_all.sum(dim=0)

        # Q norm for dead routing detection
        q_norm = Q_total.norm(dim=-1, keepdim=True)

        # Multi-head attention
        Q = Q_total.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K_total.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V_total.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            is_causal=True,
            dropout_p=dropout_p,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        # Q≈0 토큰은 attn 출력 억제
        scale = torch.where(
            q_norm > 1e-6,
            torch.ones_like(q_norm),
            q_norm * 1e-6
        )
        attn_out = attn_out * scale

        output = self.expand_O(attn_out)
        output = self.out_dropout(output)

        return output, restore_aux_loss, restore_info


class KnowledgeCircuit(nn.Module):
    """
    v18.5: Multi-path Knowledge Circuit with context-aware restore routing.

    Feature processing: same as v18.4
    Restore routing: context-based (h_proj + neuron_context) -> router callback
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_feature_know: int,
        n_restore_know: int,
        knowledge_rank: int,
        d_space: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.knowledge_rank = knowledge_rank
        self.d_space = d_space
        self.dropout = nn.Dropout(dropout)

        # v18.5: h -> d_space projection for restore context
        self.proj_h_know = nn.Linear(knowledge_rank, d_space)

    def forward(self, x, feature_paths, router, attention_mask=None):
        """
        v18.5: Feature processing + context-aware restore routing.

        Args:
            x: [B, S, D] input
            feature_paths: list of [B, S, N] weights
            router: GlobalRouters instance for restore routing callback

        Returns:
            output: [B, S, D]
            restore_aux_loss: auxiliary loss from restore routing
        """
        B, S, D = x.shape
        P = len(feature_paths)

        # Stack feature weights: [P, B, S, N]
        feature_stacked = torch.stack(feature_paths, dim=0)

        # ===== Feature 처리 =====
        f_know = self.shared_neurons.feature_know  # [N_f, D, R]
        all_h = torch.einsum('bsd,ndr->bsnr', x, f_know)  # [B, S, N_f, R]
        h_all = torch.einsum('bsnr,pbsn->pbsr', all_h, feature_stacked)  # [P, B, S, R]

        # ===== Restore context 생성 =====
        feature_emb = router.neuron_router.get_feature_know_emb()  # [N_f, d_space]

        h_proj = self.proj_h_know(h_all.view(P*B, S, -1)).view(P, B, S, self.d_space)
        neuron_ctx = torch.einsum('pbsn,nd->pbsd', feature_stacked, feature_emb)  # [P, B, S, d_space]
        restore_input = torch.cat([h_proj, neuron_ctx], dim=-1)  # [P, B, S, 2*d_space]

        # ===== Restore 라우팅 (router callback) =====
        restore_weights, restore_aux_loss, restore_info = router.get_restore_knowledge_weights(restore_input, attention_mask)

        # ===== Restore 처리 =====
        r_know = self.shared_neurons.restore_know  # [N_r, R, D]
        output_all = torch.einsum('pbsr,nrd,pbsn->pbsd', h_all, r_know, restore_weights)  # [P, B, S, D]

        # Sum over paths
        output = output_all.sum(dim=0)  # [B, S, D]

        return self.dropout(output), restore_aux_loss, restore_info


class DAWNBlock(nn.Module):
    """DAWN v18.5 block: context-aware restore routing"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_feature_know: int,
        n_restore_know: int,
        knowledge_rank: int = 128,
        d_space: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_space = d_space

        self.attn = AttentionCircuit(
            shared_neurons, d_model, n_heads, rank,
            d_space=d_space, dropout=dropout
        )
        self.knowledge = KnowledgeCircuit(
            shared_neurons, d_model, n_feature_know, n_restore_know,
            knowledge_rank=knowledge_rank, d_space=d_space, dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, importance, global_routers: GlobalRouters, attention_mask=None):
        # ===== Attention =====
        normed_x_attn = self.norm1(x)
        tau_feature_attn = global_routers.get_tau_feature(normed_x_attn)

        # v18.5: Feature routing only (restore is context-based inside circuit)
        feature_weights, attn_info, feature_aux_loss = global_routers.get_feature_attention_weights(
            normed_x_attn, attention_mask, tau_feature=tau_feature_attn
        )

        # Circuit handles restore routing internally via router callback
        attn_out, restore_aux_loss, attn_restore_info = self.attn(normed_x_attn, feature_weights, global_routers, attention_mask)
        x = x + attn_out

        attn_aux_loss = feature_aux_loss + restore_aux_loss

        # Merge restore info into attention info
        if attn_restore_info:
            attn_info.update(attn_restore_info)

        # ===== Knowledge =====
        normed_x_know = self.norm2(x)
        tau_feature_know = global_routers.get_tau_feature(normed_x_know)

        # v18.5: Feature routing only
        know_feature_paths, know_info, know_feature_aux_loss = global_routers.get_feature_knowledge_weights(
            normed_x_know, attention_mask, tau_feature=tau_feature_know
        )

        # Circuit handles restore routing internally via router callback
        know_out, know_restore_aux_loss, know_restore_info = self.knowledge(normed_x_know, know_feature_paths, global_routers, attention_mask)
        x = x + know_out

        know_aux_loss = know_feature_aux_loss + know_restore_aux_loss

        # Merge restore info into knowledge info
        if know_restore_info:
            know_info.update(know_restore_info)

        # Routing info
        routing_info = {
            'attention': attn_info,
            'knowledge': know_info,
        }
        # Store path_weights for analysis (when enabled)
        if getattr(global_routers, 'store_path_weights', False):
            routing_info['path_weights'] = {
                'fv': feature_weights['fv'],
                'fqk_Q': feature_weights['fqk_Q'],
                'fqk_K': feature_weights['fqk_K'],
                'feature_know': know_feature_paths,
                # Note: restore weights computed inside circuits, not stored here
            }
        # Norms only in debug mode (avoid GPU sync overhead)
        if global_routers.debug_mode:
            with torch.no_grad():
                routing_info['attn_out_norm'] = attn_out.norm(dim=-1).mean().item()
                routing_info['know_out_norm'] = know_out.norm(dim=-1).mean().item()

        return x, routing_info, attn_aux_loss + know_aux_loss


class DAWN(nn.Module):
    """
    DAWN v18.5: Context-Aware Restore Routing

    Key Features:
    - Based on v18.4 with context-aware restore routing
    - Feature routing: done on input x (same as v18.4)
    - Restore routing: done on [h_proj, neuron_context] where:
      - h_proj: intermediate representation h projected to d_space
      - neuron_context: weighted sum of feature neuron embeddings

    Architecture:
    - UnifiedNeuronRouter: Context-aware restore projections added
    - GlobalRouters: Feature-only routing + context-aware restore routing
    - AttentionCircuit: h projection + restore context generation + router callback
    - KnowledgeCircuit: Same pattern as AttentionCircuit
    """
    __version__ = "18.5"

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 320,
        n_layers: int = 4,
        n_heads: int = 8,
        rank: int = 16,  # v18: reduced from 64
        max_seq_len: int = 512,
        state_dim: int = 64,
        d_space: int = 64,
        # v18: Multi-path parameters
        max_paths: int = 4,
        fixed_tau: float = 0.0,
        path_max_k: int = 16,
        learnable_tau: bool = True,
        # Attention - shared Q/K pool
        n_feature_qk: int = 56,
        n_feature_v: int = 24,
        n_restore_qk: int = 56,
        n_restore_v: int = 24,
        # Knowledge - Feature/Restore separation
        n_feature_know: int = 24,
        n_restore_know: int = 24,
        knowledge_rank: int = 128,
        # Others
        dropout: float = 0.1,
        attention_token_routing: bool = False,
        knowledge_token_routing: bool = False,
        router_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        use_ssm_context: bool = True,
        tau_reg_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        # Validation checks
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.rank = rank
        self.knowledge_rank = knowledge_rank
        self.max_seq_len = max_seq_len
        self.state_dim = state_dim
        self.d_space = d_space
        self.dropout_rate = dropout
        self.attention_token_routing = attention_token_routing
        self.knowledge_token_routing = knowledge_token_routing
        self.router_dropout = router_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.use_ssm_context = use_ssm_context

        # v18 specific
        self.max_paths = max_paths
        self.fixed_tau = fixed_tau
        self.path_max_k = path_max_k
        self.learnable_tau = learnable_tau
        self.tau_reg_weight = tau_reg_weight

        # Neuron counts
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know

        # v15 compat
        self.n_feature = n_feature_qk
        self.n_neurons = n_feature_qk
        self.basis_rank = rank

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.global_ssm = GlobalSSM(d_model, state_dim, return_context=use_ssm_context)

        self.shared_neurons = SharedNeurons(
            d_model=d_model, rank=rank,
            n_feature_qk=n_feature_qk, n_feature_v=n_feature_v,
            n_restore_qk=n_restore_qk, n_restore_v=n_restore_v,
            n_feature_know=n_feature_know, n_restore_know=n_restore_know,
            knowledge_rank=knowledge_rank,
        )

        self.router = GlobalRouters(
            d_model=d_model,
            n_feature_qk=n_feature_qk, n_feature_v=n_feature_v,
            n_restore_qk=n_restore_qk, n_restore_v=n_restore_v,
            n_feature_know=n_feature_know, n_restore_know=n_restore_know,
            rank=rank,
            max_paths=max_paths,
            fixed_tau=fixed_tau,
            path_max_k=path_max_k,
            d_space=d_space, router_dropout=router_dropout,
            attention_token_routing=attention_token_routing,
            knowledge_token_routing=knowledge_token_routing,
            learnable_tau=learnable_tau,
            tau_reg_weight=tau_reg_weight,
        )

        self.layers = nn.ModuleList([
            DAWNBlock(
                shared_neurons=self.shared_neurons, d_model=d_model, n_heads=n_heads,
                rank=rank, n_feature_know=n_feature_know, n_restore_know=n_restore_know,
                knowledge_rank=knowledge_rank,
                d_space=d_space,  # v18.5: pass d_space for context-aware routing
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
                # Skip tau_proj_feature (has its own initialization)
                if hasattr(self, 'router') and hasattr(self.router, 'tau_proj_feature'):
                    if module is self.router.tau_proj_feature:
                        continue
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, attention_mask=None, return_routing_info=False, return_path_weights=False):
        B, S = input_ids.shape
        device = input_ids.device

        # Enable path_weights storage for analysis
        if return_path_weights:
            self.router.store_path_weights = True

        # Validate sequence length
        if S > self.max_seq_len:
            raise ValueError(f"Sequence length {S} exceeds max_seq_len {self.max_seq_len}")

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(positions))

        # SSM only skipped when BOTH attention and knowledge use token routing
        if self.attention_token_routing and self.knowledge_token_routing:
            importance = None
            context = None
        else:
            importance, context, raw_importance = self.global_ssm(x, attention_mask)
            if context is not None:
                if attention_mask is not None:
                    context = context * attention_mask.unsqueeze(-1)
                x = x + context

        routing_infos = []
        total_aux_loss = 0.0

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x, routing_info, aux_loss = checkpoint(
                    layer, x, importance, self.router, attention_mask,
                    use_reentrant=False
                )
            else:
                x, routing_info, aux_loss = layer(x, importance, self.router, attention_mask)

            routing_infos.append(routing_info)
            total_aux_loss += aux_loss

        self.aux_loss = total_aux_loss
        x = self.norm(x)
        logits = self.lm_head(x)

        # Reset path_weights flag after forward
        if return_path_weights:
            self.router.store_path_weights = False

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().long()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), ignore_index=-100)
            if return_routing_info or return_path_weights:
                return loss, logits, routing_infos
            return loss, logits

        if return_routing_info or return_path_weights:
            return logits, routing_infos
        return logits

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'rank': self.rank, 'knowledge_rank': self.knowledge_rank,
            'max_seq_len': self.max_seq_len,
            'max_paths': self.max_paths,
            'fixed_tau': self.fixed_tau,
            'path_max_k': self.path_max_k,
            'learnable_tau': self.learnable_tau,
            'tau_reg_weight': self.tau_reg_weight,
            'n_feature_qk': self.n_feature_qk, 'n_feature_v': self.n_feature_v,
            'n_restore_qk': self.n_restore_qk, 'n_restore_v': self.n_restore_v,
            'n_feature_know': self.n_feature_know, 'n_restore_know': self.n_restore_know,
            'state_dim': self.state_dim, 'd_space': self.d_space,
            'knowledge_token_routing': self.knowledge_token_routing,
        }

    def get_model_info(self):
        """Return model architecture info for logging"""
        return [
            f"DAWN v{self.__version__}: Context-Aware Restore Routing",
            f"  d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  rank={self.rank}, knowledge_rank={self.knowledge_rank}",
            f"  max_paths={self.max_paths}, fixed_tau={self.fixed_tau}, path_max_k={self.path_max_k}",
            f"  max_seq_len={self.max_seq_len}, state_dim={self.state_dim}, dropout={self.dropout_rate}",
            f"",
            f"  [Attention - Context-Aware Restore] (learnable_tau={self.learnable_tau})",
            f"  Feature_QK: {self.n_feature_qk} × {self.d_model} × {self.rank}",
            f"  Feature_V: {self.n_feature_v} × {self.d_model} × {self.rank}",
            f"  Restore_QK: {self.n_restore_qk} × {self.rank} × {self.d_model} (context-routed)",
            f"  Restore_V: {self.n_restore_v} × {self.rank} × {self.d_model} (context-routed)",
            f"",
            f"  [Knowledge - Context-Aware Restore] (learnable_tau={self.learnable_tau})",
            f"  Feature_Know: {self.n_feature_know} × {self.d_model} × {self.knowledge_rank}",
            f"  Restore_Know: {self.n_restore_know} × {self.knowledge_rank} × {self.d_model} (context-routed)",
            f"",
            f"  [Router - Feature routing + Context-aware restore routing]",
            f"  d_space={self.d_space}, router_dropout={self.router_dropout}",
            f"  attention_token_routing={self.attention_token_routing}, knowledge_token_routing={self.knowledge_token_routing}",
            f"  use_ssm_context={self.use_ssm_context}, tau_reg_weight={self.tau_reg_weight}",
            f"",
            f"  [Other]",
            f"  gradient_checkpointing={self.gradient_checkpointing}",
        ]

    def knowledge_diversity_loss(self):
        feat_know = self.shared_neurons.feature_know
        rest_know = self.shared_neurons.restore_know

        feat_flat = feat_know.view(feat_know.size(0), -1)
        feat_norm = F.normalize(feat_flat, dim=-1)
        feat_sim = torch.matmul(feat_norm, feat_norm.T)
        mask_f = ~torch.eye(feat_sim.size(0), dtype=torch.bool, device=feat_sim.device)
        feat_loss = feat_sim[mask_f].abs().mean()

        rest_flat = rest_know.view(rest_know.size(0), -1)
        rest_norm = F.normalize(rest_flat, dim=-1)
        rest_sim = torch.matmul(rest_norm, rest_norm.T)
        mask_r = ~torch.eye(rest_sim.size(0), dtype=torch.bool, device=rest_sim.device)
        rest_loss = rest_sim[mask_r].abs().mean()

        return (feat_loss + rest_loss) / 2

    def orthogonality_loss(self):
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

        I_know = torch.eye(self.knowledge_rank, device=self.shared_neurons.feature_know.device).unsqueeze(0)

        W_fknow = self.shared_neurons.feature_know
        WtW_fknow = torch.bmm(W_fknow.transpose(1, 2), W_fknow)
        loss_fknow = ((WtW_fknow - I_know) ** 2).mean()

        W_rknow = self.shared_neurons.restore_know
        WWt_rknow = torch.bmm(W_rknow, W_rknow.transpose(1, 2))
        loss_rknow = ((WWt_rknow - I_know) ** 2).mean()

        return (loss_fqk + loss_fv + loss_rqk + loss_rv + loss_fknow + loss_rknow) / 6

    def get_auxiliary_losses(self):
        return {
            'orth_total': self.orthogonality_loss(),
            'knowledge_div': self.knowledge_diversity_loss(),
        }

    def __repr__(self):
        params = sum(p.numel() for p in self.parameters()) / 1e6
        attn_neurons = self.n_feature_qk + self.n_feature_v + self.n_restore_qk + self.n_restore_v
        know_neurons = self.n_feature_know + self.n_restore_know
        return (
            f"DAWN v{self.__version__}: Context-Aware Restore Routing\n"
            f"  Params: {params:.1f}M\n"
            f"  rank={self.rank}, max_paths={self.max_paths}, path_max_k={self.path_max_k}\n"
            f"  Attention: Feature_QK={self.n_feature_qk}, Feature_V={self.n_feature_v}\n"
            f"            Restore_QK={self.n_restore_qk}, Restore_V={self.n_restore_v} (context-routed)\n"
            f"  Knowledge: Feature={self.n_feature_know}, Restore={self.n_restore_know} (context-routed)\n"
            f"  Total neurons: {attn_neurons} (attn) + {know_neurons} (know) = {attn_neurons + know_neurons}"
        )


# Alias for version registry
DAWN_v18_5 = DAWN
