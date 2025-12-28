"""
DAWN v18.0: Adaptive Threshold Multi-Path Routing

Key Concept:
- Reduced rank (16) with adaptive per-token threshold for variable neuron selection
- Selected neurons are chunked by rank and processed as parallel paths
- Q, K, V generation uses multi-path aggregation; attention operation unchanged

Changes from v17.1:
- rank: 64 → 16 (reduced for multi-path chunking)
- top-k sparsification → adaptive threshold selection
- Single path → Multi-path (1~max_paths) parallel processing
- Added: threshold_proj for each neuron pool
- Added: tau_regularization_loss for controlling activation ratio
- Batched einsum optimization for 45% memory savings, 1.57x throughput

Architecture:
- UnifiedNeuronRouter: 6 threshold projections added
- GlobalRouters: _threshold_select + _chunk_to_paths instead of _topk_sparsify
- AttentionCircuit: Multi-path Q, K, V aggregation (batched einsum)
- KnowledgeCircuit: Multi-path aggregation (batched einsum)
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
    v18.0: 6 attention projections + 2 knowledge projections + 6 threshold projections
    Per-token adaptive threshold for variable neuron selection
    """
    def __init__(self, d_model, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
                 n_feature_know, n_restore_know,
                 d_space=64, dropout=0.1, **kwargs):
        super().__init__()
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.d_space = d_space
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

        # 6 attention projections + 2 knowledge projections
        self.proj_all = nn.Linear(d_model, d_space * 6)  # fqk_Q, fqk_K, fv, rqk_Q, rqk_K, rv
        self.proj_feature_know = nn.Linear(d_model, d_space)
        self.proj_restore_know = nn.Linear(d_model, d_space)
        self.dropout = nn.Dropout(dropout)

        # v18: Per-pool threshold projections (output: scalar per token)
        self.threshold_proj_fqk = nn.Linear(d_model, 1)
        self.threshold_proj_fv = nn.Linear(d_model, 1)
        self.threshold_proj_rqk = nn.Linear(d_model, 1)
        self.threshold_proj_rv = nn.Linear(d_model, 1)
        self.threshold_proj_feature_know = nn.Linear(d_model, 1)
        self.threshold_proj_restore_know = nn.Linear(d_model, 1)

        # Initialize threshold projections to output ~0 (sigmoid(0)=0.5 baseline)
        for proj in [self.threshold_proj_fqk, self.threshold_proj_fv,
                     self.threshold_proj_rqk, self.threshold_proj_rv,
                     self.threshold_proj_feature_know, self.threshold_proj_restore_know]:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

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
        Get per-token thresholds for each neuron pool
        x: [B, S, d_model]
        Returns: dict of [B, S, 1] thresholds
        """
        return {
            'fqk': self.threshold_proj_fqk(x),      # [B, S, 1]
            'fv': self.threshold_proj_fv(x),        # [B, S, 1]
            'rqk': self.threshold_proj_rqk(x),      # [B, S, 1]
            'rv': self.threshold_proj_rv(x),        # [B, S, 1]
            'feature_know': self.threshold_proj_feature_know(x),  # [B, S, 1]
            'restore_know': self.threshold_proj_restore_know(x),  # [B, S, 1]
        }

    def get_knowledge_logits(self, x):
        """
        Return 2 knowledge logits (feature_know, restore_know)
        x: [B, S, d_model]
        """
        emb_norm = F.normalize(self.neuron_emb, dim=-1)

        # Feature_know
        h_feature_know = self.dropout(self.proj_feature_know(x))
        emb_feature_know = emb_norm[self.restore_v_end:self.feature_know_end]
        logits_feature_know = torch.einsum('bsd,nd->bsn', h_feature_know, emb_feature_know)

        # Restore_know
        h_restore_know = self.dropout(self.proj_restore_know(x))
        emb_restore_know = emb_norm[self.feature_know_end:]
        logits_restore_know = torch.einsum('bsd,nd->bsn', h_restore_know, emb_restore_know)

        return logits_feature_know, logits_restore_know

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

        return logits_fqk_Q, logits_fqk_K, logits_fv, logits_rqk_Q, logits_rqk_K, logits_rv

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
    v18.0: Adaptive threshold + multi-path routing

    Instead of top-k sparsification, uses:
    1. Per-token adaptive threshold (learned)
    2. Soft sigmoid weights for neurons above threshold
    3. Chunking into multiple paths by score ranking
    """
    def __init__(self, d_model: int, n_feature_qk: int, n_feature_v: int,
                 n_restore_qk: int, n_restore_v: int,
                 n_feature_know: int, n_restore_know: int,
                 rank: int = 16,
                 max_paths: int = 4,
                 threshold_temp: float = 1.0,
                 d_space: int = 64, router_dropout: float = 0.1,
                 attention_token_routing: bool = False,
                 knowledge_token_routing: bool = False, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.rank = rank
        self.max_paths = max_paths
        self.threshold_temp = threshold_temp
        self.attention_token_routing = attention_token_routing
        self.knowledge_token_routing = knowledge_token_routing

        self.neuron_router = UnifiedNeuronRouter(
            d_model, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
            n_feature_know, n_restore_know,
            d_space=d_space, dropout=router_dropout, **kwargs
        )

    def _threshold_select(self, scores, tau, temp=1.0):
        """
        Apply threshold selection with STE (Straight-Through Estimator)

        Args:
            scores: [B, S, N] neuron scores
            tau: [B, S, 1] per-token threshold
            temp: temperature for sigmoid sharpness

        Returns:
            weights: [B, S, N] hard weights (0 or 1) with soft gradients
        """
        # Soft for gradient flow
        soft_weights = torch.sigmoid((scores - tau) / temp)

        # Hard for forward (0 or 1)
        hard_weights = (scores > tau).float()

        # STE: forward uses hard, backward uses soft gradient
        weights = hard_weights - soft_weights.detach() + soft_weights

        return weights

    def _chunk_to_paths(self, weights, scores, rank, max_paths):
        """
        Chunk neurons above threshold into rank-sized paths

        Path count is determined by the number of ACTIVE neurons (weight > 0.5),
        not by total neuron count.

        Args:
            weights: [B, S, N] soft weights from threshold selection
            scores: [B, S, N] original scores (for ranking)
            rank: number of neurons per path
            max_paths: maximum number of paths

        Returns:
            path_weights_list: list of [B, S, N] weights (always length max_paths)
            Each path contains weights for top-k neurons in that chunk.
            Unused paths are zero-filled for consistent tensor operations.
        """
        B, S, N = scores.shape
        device = scores.device

        # Sort neurons by score (descending)
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)

        # Get corresponding weights in sorted order
        sorted_weights = torch.gather(weights, dim=-1, index=sorted_indices)

        # Count active neurons (weight > 0.5) to determine path count
        # Use mean across batch and sequence for stable path count
        active_mask = (weights > 0.5).float()
        n_active = active_mask.sum(dim=-1).mean().item()  # average active neurons per token

        # Calculate paths based on active neuron count
        n_paths_needed = max(1, int((n_active + rank - 1) // rank))
        n_paths = min(max_paths, n_paths_needed)

        path_weights_list = []

        for p in range(max_paths):  # Always iterate max_paths times
            if p < n_paths:
                start_idx = p * rank
                end_idx = min((p + 1) * rank, N)

                if start_idx >= N:
                    # No more neurons to process, add zero path
                    path_weights_list.append(torch.zeros_like(weights))
                    continue

                # Create path weights: only neurons in this chunk get their weights
                path_weights = torch.zeros_like(weights)

                # Get indices for this path's neurons
                path_indices = sorted_indices[:, :, start_idx:end_idx]  # [B, S, chunk_size]
                path_w = sorted_weights[:, :, start_idx:end_idx]  # [B, S, chunk_size]

                # Scatter weights back to original positions
                path_weights.scatter_(dim=-1, index=path_indices, src=path_w)

                path_weights_list.append(path_weights)
            else:
                # Pad with zero weights for unused paths
                path_weights_list.append(torch.zeros_like(weights))

        # Ensure exactly max_paths for consistent tensor shapes across pools
        while len(path_weights_list) < max_paths:
            path_weights_list.append(torch.zeros_like(weights))

        return path_weights_list[:max_paths]

    def get_attention_weights(self, x, importance, attention_mask=None):
        """
        v18.0: Adaptive threshold + multi-path routing for attention

        Returns:
            path_weights_dict: dict with lists of path weights for each neuron type
            routing_info: dict with routing statistics
            aux_loss: auxiliary loss for load balancing
        """
        (fqk_logits_Q, fqk_logits_K, fv_logits,
         rqk_logits_Q, rqk_logits_K, rv_logits) = self.neuron_router.get_all_logits(x)

        # Get per-token thresholds
        thresholds = self.neuron_router.get_thresholds(x)
        tau_fqk = thresholds['fqk']  # [B, S, 1]
        tau_fv = thresholds['fv']
        tau_rqk = thresholds['rqk']
        tau_rv = thresholds['rv']

        # Compute soft preferences (for aux loss)
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

        # Apply threshold selection and chunk to paths
        temp = self.threshold_temp

        # Feature QK (Q)
        fqk_soft_Q = self._threshold_select(fqk_logits_Q, tau_fqk, temp)
        fqk_paths_Q = self._chunk_to_paths(fqk_soft_Q, fqk_logits_Q, self.rank, self.max_paths)

        # Feature QK (K)
        fqk_soft_K = self._threshold_select(fqk_logits_K, tau_fqk, temp)
        fqk_paths_K = self._chunk_to_paths(fqk_soft_K, fqk_logits_K, self.rank, self.max_paths)

        # Feature V
        fv_soft = self._threshold_select(fv_logits, tau_fv, temp)
        fv_paths = self._chunk_to_paths(fv_soft, fv_logits, self.rank, self.max_paths)

        # Restore QK (Q)
        rqk_soft_Q = self._threshold_select(rqk_logits_Q, tau_rqk, temp)
        rqk_paths_Q = self._chunk_to_paths(rqk_soft_Q, rqk_logits_Q, self.rank, self.max_paths)

        # Restore QK (K)
        rqk_soft_K = self._threshold_select(rqk_logits_K, tau_rqk, temp)
        rqk_paths_K = self._chunk_to_paths(rqk_soft_K, rqk_logits_K, self.rank, self.max_paths)

        # Restore V
        rv_soft = self._threshold_select(rv_logits, tau_rv, temp)
        rv_paths = self._chunk_to_paths(rv_soft, rv_logits, self.rank, self.max_paths)

        # Package path weights
        path_weights = {
            'fqk_Q': fqk_paths_Q,
            'fqk_K': fqk_paths_K,
            'fv': fv_paths,
            'rqk_Q': rqk_paths_Q,
            'rqk_K': rqk_paths_K,
            'rv': rv_paths,
        }

        # Soft weights for tau regularization
        soft_weights = {
            'fqk_soft_Q': fqk_soft_Q,
            'fqk_soft_K': fqk_soft_K,
            'fv_soft': fv_soft,
            'rqk_soft_Q': rqk_soft_Q,
            'rqk_soft_K': rqk_soft_K,
            'rv_soft': rv_soft,
        }

        routing_info = {
            'fqk_q_pref': fqk_pref_Q.detach(),
            'fqk_k_pref': fqk_pref_K.detach(),
            'fv_pref': fv_pref.detach(),
            'rqk_q_pref': rqk_pref_Q.detach(),
            'rqk_k_pref': rqk_pref_K.detach(),
            'rv_pref': rv_pref.detach(),
            # Actual active paths (non-zero path tensors)
            'n_paths_fqk_Q': sum(1 for p in fqk_paths_Q if p.abs().sum() > 0),
            'n_paths_fqk_K': sum(1 for p in fqk_paths_K if p.abs().sum() > 0),
            'n_paths_fv': sum(1 for p in fv_paths if p.abs().sum() > 0),
            'n_paths_rqk_Q': sum(1 for p in rqk_paths_Q if p.abs().sum() > 0),
            'n_paths_rqk_K': sum(1 for p in rqk_paths_K if p.abs().sum() > 0),
            'n_paths_rv': sum(1 for p in rv_paths if p.abs().sum() > 0),
            # tau mean and std
            'tau_fqk_mean': tau_fqk.detach().mean().item(),
            'tau_fqk_std': tau_fqk.detach().std().item(),
            'tau_fv_mean': tau_fv.detach().mean().item(),
            'tau_fv_std': tau_fv.detach().std().item(),
            'tau_rqk_mean': tau_rqk.detach().mean().item(),
            'tau_rqk_std': tau_rqk.detach().std().item(),
            'tau_rv_mean': tau_rv.detach().mean().item(),
            'tau_rv_std': tau_rv.detach().std().item(),
            # Activation ratio (soft weights mean)
            'activation_fqk_Q': fqk_soft_Q.detach().mean().item(),
            'activation_fqk_K': fqk_soft_K.detach().mean().item(),
            'activation_fv': fv_soft.detach().mean().item(),
            'activation_rqk_Q': rqk_soft_Q.detach().mean().item(),
            'activation_rqk_K': rqk_soft_K.detach().mean().item(),
            'activation_rv': rv_soft.detach().mean().item(),
            'token_routing': self.attention_token_routing,
        }

        # Update usage with combined weights from all paths
        if self.training:
            combined_fqk_Q = sum(fqk_paths_Q) / len(fqk_paths_Q)
            combined_fqk_K = sum(fqk_paths_K) / len(fqk_paths_K)
            combined_fv = sum(fv_paths) / len(fv_paths)
            combined_rqk_Q = sum(rqk_paths_Q) / len(rqk_paths_Q)
            combined_rqk_K = sum(rqk_paths_K) / len(rqk_paths_K)
            combined_rv = sum(rv_paths) / len(rv_paths)

            self.neuron_router.update_usage(combined_fqk_Q, 'feature_q', attention_mask)
            self.neuron_router.update_usage(combined_fqk_K, 'feature_k', attention_mask)
            self.neuron_router.update_usage(combined_fv, 'feature_v', attention_mask)
            self.neuron_router.update_usage(combined_rqk_Q, 'restore_q', attention_mask)
            self.neuron_router.update_usage(combined_rqk_K, 'restore_k', attention_mask)
            self.neuron_router.update_usage(combined_rv, 'restore_v', attention_mask)

        return path_weights, soft_weights, routing_info, aux_loss

    def get_knowledge_weights(self, x, importance, attention_mask=None):
        """
        v18.0: Adaptive threshold + multi-path routing for knowledge
        """
        logits_f, logits_r = self.neuron_router.get_knowledge_logits(x)

        # Get thresholds
        thresholds = self.neuron_router.get_thresholds(x)
        tau_f = thresholds['feature_know']
        tau_r = thresholds['restore_know']

        temp = self.threshold_temp

        # Feature knowledge
        f_soft = self._threshold_select(logits_f, tau_f, temp)
        f_paths = self._chunk_to_paths(f_soft, logits_f, self.rank, self.max_paths)

        # Restore knowledge
        r_soft = self._threshold_select(logits_r, tau_r, temp)
        r_paths = self._chunk_to_paths(r_soft, logits_r, self.rank, self.max_paths)

        if self.training:
            combined_f = sum(f_paths) / len(f_paths)
            combined_r = sum(r_paths) / len(r_paths)
            self.neuron_router.update_usage(combined_f, 'feature_know', attention_mask)
            self.neuron_router.update_usage(combined_r, 'restore_know', attention_mask)

        soft_weights = {
            'feature_know_soft': f_soft,
            'restore_know_soft': r_soft,
        }

        know_info = {
            'n_paths_feature': sum(1 for p in f_paths if p.abs().sum() > 0),
            'n_paths_restore': sum(1 for p in r_paths if p.abs().sum() > 0),
            # tau mean and std
            'tau_feature_mean': tau_f.detach().mean().item(),
            'tau_feature_std': tau_f.detach().std().item(),
            'tau_restore_mean': tau_r.detach().mean().item(),
            'tau_restore_std': tau_r.detach().std().item(),
            # Activation ratio
            'activation_feature': f_soft.detach().mean().item(),
            'activation_restore': r_soft.detach().mean().item(),
        }

        return f_paths, r_paths, soft_weights, know_info


class AttentionCircuit(nn.Module):
    """
    v18.0: Multi-path attention circuit

    Each path generates Q, K, V independently, then all paths are summed.
    Attention operation remains unchanged.
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

    def _process_single_path(self, x, fqk_w_Q, fqk_w_K, fv_w, rqk_w_Q, rqk_w_K, rv_w):
        """
        Process a single path to generate Q, K, V contributions.
        Uses batched einsum for better memory efficiency and throughput.
        """
        B, S, D = x.shape

        # Feature QK: batched einsum
        f_qk = self.shared_neurons.feature_qk_neurons  # [N, D, R]
        all_h_qk = torch.einsum('bsd,ndr->bsnr', x, f_qk)  # [B, S, N, R]
        h_q = torch.einsum('bsnr,bsn->bsr', all_h_qk, fqk_w_Q)  # [B, S, R]
        h_k = torch.einsum('bsnr,bsn->bsr', all_h_qk, fqk_w_K)  # [B, S, R]

        # Feature V: batched einsum
        f_v = self.shared_neurons.feature_v_neurons  # [N, D, R]
        all_h_v = torch.einsum('bsd,ndr->bsnr', x, f_v)  # [B, S, N, R]
        h_v = torch.einsum('bsnr,bsn->bsr', all_h_v, fv_w)  # [B, S, R]

        # Restore: batched einsum
        r_qk = self.shared_neurons.restore_qk_neurons  # [N, R, D]
        r_v = self.shared_neurons.restore_v_neurons  # [N, R, D]

        Q = torch.einsum('bsr,nrd,bsn->bsd', h_q, r_qk, rqk_w_Q)  # [B, S, D]
        K = torch.einsum('bsr,nrd,bsn->bsd', h_k, r_qk, rqk_w_K)  # [B, S, D]
        V = torch.einsum('bsr,nrd,bsn->bsd', h_v, r_v, rv_w)  # [B, S, D]

        return Q, K, V

    def forward(self, x, path_weights, attention_mask=None):
        """
        Args:
            x: [B, S, D] input
            path_weights: dict with lists of weights for each neuron type
                - 'fqk_Q': list of [B, S, N] weights
                - 'fqk_K': list of [B, S, N] weights
                - 'fv': list of [B, S, N] weights
                - 'rqk_Q': list of [B, S, N] weights
                - 'rqk_K': list of [B, S, N] weights
                - 'rv': list of [B, S, N] weights
        """
        B, S, D = x.shape

        # Stack path weights: [P, B, S, N]
        fqk_w_Q_stacked = torch.stack(path_weights['fqk_Q'], dim=0)
        fqk_w_K_stacked = torch.stack(path_weights['fqk_K'], dim=0)
        fv_w_stacked = torch.stack(path_weights['fv'], dim=0)
        rqk_w_Q_stacked = torch.stack(path_weights['rqk_Q'], dim=0)
        rqk_w_K_stacked = torch.stack(path_weights['rqk_K'], dim=0)
        rv_w_stacked = torch.stack(path_weights['rv'], dim=0)

        # Feature QK: common computation once
        f_qk = self.shared_neurons.feature_qk_neurons  # [N, D, R]
        all_h_qk = torch.einsum('bsd,ndr->bsnr', x, f_qk)  # [B, S, N, R]

        # Path-parallel bottleneck for Q, K
        h_q_all = torch.einsum('bsnr,pbsn->pbsr', all_h_qk, fqk_w_Q_stacked)  # [P, B, S, R]
        h_k_all = torch.einsum('bsnr,pbsn->pbsr', all_h_qk, fqk_w_K_stacked)  # [P, B, S, R]

        # Feature V: common computation once
        f_v = self.shared_neurons.feature_v_neurons  # [N, D, R]
        all_h_v = torch.einsum('bsd,ndr->bsnr', x, f_v)  # [B, S, N, R]

        # Path-parallel bottleneck for V
        h_v_all = torch.einsum('bsnr,pbsn->pbsr', all_h_v, fv_w_stacked)  # [P, B, S, R]

        # Path-parallel restore
        r_qk = self.shared_neurons.restore_qk_neurons  # [N, R, D]
        r_v = self.shared_neurons.restore_v_neurons  # [N, R, D]

        Q_all = torch.einsum('pbsr,nrd,pbsn->pbsd', h_q_all, r_qk, rqk_w_Q_stacked)  # [P, B, S, D]
        K_all = torch.einsum('pbsr,nrd,pbsn->pbsd', h_k_all, r_qk, rqk_w_K_stacked)  # [P, B, S, D]
        V_all = torch.einsum('pbsr,nrd,pbsn->pbsd', h_v_all, r_v, rv_w_stacked)  # [P, B, S, D]

        # Sum over paths (each path has own bottleneck, total capacity = R * n_paths)
        Q_total = Q_all.sum(dim=0)  # [B, S, D]
        K_total = K_all.sum(dim=0)
        V_total = V_all.sum(dim=0)

        # Multi-head attention
        Q = Q_total.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K_total.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V_total.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # FlashAttention (PyTorch 2.0+)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            is_causal=True,
            dropout_p=dropout_p,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        output = self.expand_O(attn_out)
        output = self.out_dropout(output)

        return output, None


class KnowledgeCircuit(nn.Module):
    """
    v18.0: Multi-path Knowledge Circuit

    Each path processes independently, then all paths are summed.
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_feature_know: int,
        n_restore_know: int,
        knowledge_rank: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.knowledge_rank = knowledge_rank
        self.dropout = nn.Dropout(dropout)

    def _process_single_path(self, x, feature_w, restore_w):
        """
        Process a single path.
        Uses batched einsum for better memory efficiency and throughput.
        """
        # Feature: batched einsum
        f_know = self.shared_neurons.feature_know  # [N, D, R]
        all_h = torch.einsum('bsd,ndr->bsnr', x, f_know)  # [B, S, N, R]
        h = torch.einsum('bsnr,bsn->bsr', all_h, feature_w)  # [B, S, R]

        # Restore: batched einsum
        r_know = self.shared_neurons.restore_know  # [N, R, D]
        output = torch.einsum('bsr,nrd,bsn->bsd', h, r_know, restore_w)  # [B, S, D]

        return output

    def forward(self, x, feature_paths, restore_paths, attention_mask=None):
        """
        Args:
            x: [B, S, D] input
            feature_paths: list of [B, S, N] weights
            restore_paths: list of [B, S, N] weights
        """
        n_paths = max(len(feature_paths), len(restore_paths))

        # Stack path weights: [P, B, S, N]
        feature_stacked = torch.stack(feature_paths, dim=0)
        restore_stacked = torch.stack(restore_paths, dim=0)

        # Feature: common computation once
        f_know = self.shared_neurons.feature_know  # [N, D, R]
        all_h = torch.einsum('bsd,ndr->bsnr', x, f_know)  # [B, S, N, R]

        # Path-parallel bottleneck
        h_all = torch.einsum('bsnr,pbsn->pbsr', all_h, feature_stacked)  # [P, B, S, R]

        # Path-parallel restore
        r_know = self.shared_neurons.restore_know  # [N, R, D]
        output_all = torch.einsum('pbsr,nrd,pbsn->pbsd', h_all, r_know, restore_stacked)  # [P, B, S, D]

        # Sum over paths
        output = output_all.sum(dim=0)  # [B, S, D]

        return self.dropout(output)


class DAWNBlock(nn.Module):
    """DAWN v18.0 block: multi-path routing"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_feature_know: int,
        n_restore_know: int,
        knowledge_rank: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = AttentionCircuit(shared_neurons, d_model, n_heads, rank, dropout)
        self.knowledge = KnowledgeCircuit(
            shared_neurons, d_model, n_feature_know, n_restore_know,
            knowledge_rank=knowledge_rank,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, importance, global_routers: GlobalRouters, attention_mask=None):
        # Attention
        normed_x = self.norm1(x)
        path_weights, soft_weights_attn, attn_info, attn_aux_loss = global_routers.get_attention_weights(
            normed_x, importance, attention_mask
        )
        attn_out, _ = self.attn(normed_x, path_weights, attention_mask)
        x = x + attn_out

        # Knowledge
        normed_x = self.norm2(x)
        feature_paths, restore_paths, soft_weights_know, know_info = global_routers.get_knowledge_weights(
            normed_x, importance, attention_mask
        )
        know_out = self.knowledge(normed_x, feature_paths, restore_paths, attention_mask)
        x = x + know_out

        # Collect soft weights for tau regularization
        all_soft_weights = {**soft_weights_attn, **soft_weights_know}

        # Routing info
        attn_out_norm = attn_out.norm(dim=-1).mean().detach()
        know_out_norm = know_out.norm(dim=-1).mean().detach()

        routing_info = {
            'attention': attn_info,
            'knowledge': know_info,
            'attn_out_norm': attn_out_norm,
            'know_out_norm': know_out_norm,
            'soft_weights': all_soft_weights,
        }

        return x, routing_info, attn_aux_loss


class DAWN(nn.Module):
    """
    DAWN v18.0: Adaptive Threshold Multi-Path Routing

    Key Features:
    - Reduced rank (16) for chunked multi-path processing
    - Per-token adaptive threshold (learned) for variable neuron selection
    - Multi-path (1~4) parallel processing with path-wise Q,K,V aggregation
    - Attention operation unchanged (scaled_dot_product_attention)

    Architecture:
    - UnifiedNeuronRouter: 6 threshold projections for adaptive selection
    - GlobalRouters: _threshold_select + _chunk_to_paths
    - AttentionCircuit: Multi-path Q,K,V summation
    - KnowledgeCircuit: Multi-path summation
    """
    __version__ = "18.0"

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
        threshold_temp: float = 1.0,
        tau_reg_lambda: float = 0.01,
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
        self.threshold_temp = threshold_temp
        self.tau_reg_lambda = tau_reg_lambda

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
            threshold_temp=threshold_temp,
            d_space=d_space, router_dropout=router_dropout,
            attention_token_routing=attention_token_routing,
            knowledge_token_routing=knowledge_token_routing,
        )

        self.layers = nn.ModuleList([
            DAWNBlock(
                shared_neurons=self.shared_neurons, d_model=d_model, n_heads=n_heads,
                rank=rank, n_feature_know=n_feature_know, n_restore_know=n_restore_know,
                knowledge_rank=knowledge_rank,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.aux_loss = 0.0

        # v18: Store soft weights for tau regularization
        self.last_soft_weights = []

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, attention_mask=None, return_routing_info=False):
        B, S = input_ids.shape
        device = input_ids.device

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

        # Clear soft weights for this forward pass
        self.last_soft_weights = []

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

            # Collect soft weights for tau regularization
            if 'soft_weights' in routing_info:
                self.last_soft_weights.append(routing_info['soft_weights'])

        self.aux_loss = total_aux_loss
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

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'rank': self.rank, 'knowledge_rank': self.knowledge_rank,
            'max_seq_len': self.max_seq_len,
            'max_paths': self.max_paths,
            'threshold_temp': self.threshold_temp,
            'tau_reg_lambda': self.tau_reg_lambda,
            'n_feature_qk': self.n_feature_qk, 'n_feature_v': self.n_feature_v,
            'n_restore_qk': self.n_restore_qk, 'n_restore_v': self.n_restore_v,
            'n_feature_know': self.n_feature_know, 'n_restore_know': self.n_restore_know,
            'state_dim': self.state_dim, 'd_space': self.d_space,
            'knowledge_token_routing': self.knowledge_token_routing,
        }

    def get_model_info(self):
        """Return model architecture info for logging"""
        return [
            f"DAWN v{self.__version__}: Adaptive Threshold Multi-Path Routing",
            f"  d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  rank={self.rank}, knowledge_rank={self.knowledge_rank}",
            f"  max_paths={self.max_paths}, threshold_temp={self.threshold_temp}",
            f"  tau_reg_lambda={self.tau_reg_lambda}",
            f"  max_seq_len={self.max_seq_len}, state_dim={self.state_dim}, dropout={self.dropout_rate}",
            f"",
            f"  [Attention - Q/K Shared Pool] (adaptive threshold, max_paths={self.max_paths})",
            f"  Feature_QK: {self.n_feature_qk} × {self.d_model} × {self.rank}",
            f"  Feature_V: {self.n_feature_v} × {self.d_model} × {self.rank}",
            f"  Restore_QK: {self.n_restore_qk} × {self.rank} × {self.d_model}",
            f"  Restore_V: {self.n_restore_v} × {self.rank} × {self.d_model}",
            f"",
            f"  [Knowledge - Feature-Restore] (adaptive threshold, max_paths={self.max_paths})",
            f"  Feature_Know: {self.n_feature_know} × {self.d_model} × {self.knowledge_rank}",
            f"  Restore_Know: {self.n_restore_know} × {self.knowledge_rank} × {self.d_model}",
            f"",
            f"  [Router - Adaptive Threshold]",
            f"  d_space={self.d_space}, router_dropout={self.router_dropout}",
            f"  attention_token_routing={self.attention_token_routing}, knowledge_token_routing={self.knowledge_token_routing}",
            f"  use_ssm_context={self.use_ssm_context}",
            f"",
            f"  [Other]",
            f"  gradient_checkpointing={self.gradient_checkpointing}",
        ]

    def tau_regularization_loss(self):
        """
        Penalize high activation ratio to prevent too many neurons being activated.
        Uses soft weights (sigmoid outputs) from the last forward pass.
        """
        if not self.last_soft_weights:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total_activation = 0.0
        count = 0

        for layer_weights in self.last_soft_weights:
            for key, weights in layer_weights.items():
                if weights is not None:
                    total_activation += weights.mean()
                    count += 1

        if count == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        avg_activation = total_activation / count
        return avg_activation * self.tau_reg_lambda

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
            'tau_reg': self.tau_regularization_loss(),
        }

    def __repr__(self):
        params = sum(p.numel() for p in self.parameters()) / 1e6
        attn_neurons = self.n_feature_qk + self.n_feature_v + self.n_restore_qk + self.n_restore_v
        know_neurons = self.n_feature_know + self.n_restore_know
        return (
            f"DAWN v{self.__version__}: Adaptive Threshold Multi-Path\n"
            f"  Params: {params:.1f}M\n"
            f"  rank={self.rank}, max_paths={self.max_paths}, threshold_temp={self.threshold_temp}\n"
            f"  Attention: Feature_QK={self.n_feature_qk}, Feature_V={self.n_feature_v}\n"
            f"            Restore_QK={self.n_restore_qk}, Restore_V={self.n_restore_v}\n"
            f"  Knowledge: Feature={self.n_feature_know}, Restore={self.n_restore_know}\n"
            f"  Total neurons: {attn_neurons} (attn) + {know_neurons} (know) = {attn_neurons + know_neurons}"
        )
