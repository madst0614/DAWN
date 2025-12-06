"""
DAWN v13.2: Unified Neuron Router

Changes from v13.1:
- 5 separate routers → 1 UnifiedNeuronRouter
- All neurons (compress, expand_QK, expand_V) in same d_space embedding
- Token projection → dot product with neuron embeddings
- Starvation bonus with decay for early exploration
- Usage EMA tracking per neuron type

Architecture:
- Pool: QK shared, V separate (same as v13.1)
- Router: 1 unified router with shared projection
- Routing: token → proj → dot product → type-specific slicing
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
    통합 뉴런 임베딩 공간에서 토큰-뉴런 매칭

    모든 뉴런(compress, expand_QK, expand_V)이 같은 공간에 존재.
    토큰이 투영되어 가까운 뉴런 선택.
    """
    def __init__(self, d_model, n_compress, n_expand_QK, n_expand_V,
                 d_space=64, dropout=0.1):
        super().__init__()
        self.n_compress = n_compress
        self.n_expand_QK = n_expand_QK
        self.n_expand_V = n_expand_V
        self.d_space = d_space

        total_neurons = n_compress + n_expand_QK + n_expand_V
        self.total_neurons = total_neurons

        # 인덱스 경계
        self.compress_end = n_compress
        self.expand_QK_end = n_compress + n_expand_QK
        # expand_V는 expand_QK_end ~ total_neurons

        # 공유 projection
        self.proj = nn.Linear(d_model, d_space)
        self.dropout = nn.Dropout(dropout)

        # 통합 뉴런 임베딩 [total_neurons, d_space]
        self.neuron_emb = nn.Parameter(torch.randn(total_neurons, d_space) * 0.02)

        # 타입별 usage 추적
        self.register_buffer('usage_ema_compress', torch.zeros(n_compress))
        self.register_buffer('usage_ema_expand_QK', torch.zeros(n_expand_QK))
        self.register_buffer('usage_ema_expand_V', torch.zeros(n_expand_V))

    def get_logits(self, x, neuron_type, starvation_weight=0.0):
        """
        x: [B, S, d_model]
        neuron_type: 'compress', 'expand_Q', 'expand_K', 'expand_V', 'memory'
        """
        h_proj = self.proj(x)  # [B, S, d_space]
        h_proj = self.dropout(h_proj)

        # Normalize neuron embeddings only (preserve token magnitude for routing signal)
        neuron_emb_norm = F.normalize(self.neuron_emb, dim=-1)

        # 전체 뉴런과 내적
        all_logits = torch.einsum('bsd,nd->bsn', h_proj, neuron_emb_norm)

        # 타입별 슬라이싱 + starvation
        if neuron_type in ['compress', 'memory']:
            logits = all_logits[..., :self.compress_end]
            if self.training and starvation_weight > 0:
                bonus = (1 - self.usage_ema_compress) * starvation_weight
                logits = logits + bonus

        elif neuron_type in ['expand_Q', 'expand_K']:
            logits = all_logits[..., self.compress_end:self.expand_QK_end]
            if self.training and starvation_weight > 0:
                bonus = (1 - self.usage_ema_expand_QK) * starvation_weight
                logits = logits + bonus

        elif neuron_type == 'expand_V':
            logits = all_logits[..., self.expand_QK_end:]
            if self.training and starvation_weight > 0:
                bonus = (1 - self.usage_ema_expand_V) * starvation_weight
                logits = logits + bonus

        return logits

    def update_usage(self, weights, neuron_type):
        """top-k 후 선택된 뉴런 사용량 업데이트"""
        if not self.training:
            return

        # weights: [B, N] or [B, S, N]
        if weights.dim() == 3:
            usage = (weights > 0).float().mean(dim=[0, 1])
        else:
            usage = (weights > 0).float().mean(dim=0)

        if neuron_type in ['compress', 'memory']:
            self.usage_ema_compress = 0.99 * self.usage_ema_compress + 0.01 * usage
        elif neuron_type in ['expand_Q', 'expand_K']:
            self.usage_ema_expand_QK = 0.99 * self.usage_ema_expand_QK + 0.01 * usage
        elif neuron_type == 'expand_V':
            self.usage_ema_expand_V = 0.99 * self.usage_ema_expand_V + 0.01 * usage

class SharedNeurons(nn.Module):
    """v13.1: Shared neurons with separate QK/V expand pools"""
    def __init__(
        self,
        d_model: int,
        rank: int,
        n_compress: int,
        n_expand_QK: int,
        n_expand_V: int,
        n_knowledge: int,
        knowledge_rank: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else rank
        self.n_compress = n_compress
        self.n_expand_QK = n_expand_QK
        self.n_expand_V = n_expand_V
        self.n_knowledge = n_knowledge

        # Compress pool: [n_compress, d_model, rank]
        self.compress_neurons = nn.Parameter(torch.zeros(n_compress, d_model, rank))

        # Separate expand pools: QK shared, V separate
        self.expand_neurons_QK = nn.Parameter(torch.zeros(n_expand_QK, rank, d_model))
        self.expand_neurons_V = nn.Parameter(torch.zeros(n_expand_V, rank, d_model))

        self.knowledge_K = nn.Parameter(torch.zeros(n_knowledge, self.knowledge_rank))
        self.knowledge_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_parameters()

    def _init_parameters(self):
        for i in range(self.n_compress):
            nn.init.orthogonal_(self.compress_neurons.data[i])
        for i in range(self.n_expand_QK):
            nn.init.orthogonal_(self.expand_neurons_QK.data[i])
        for i in range(self.n_expand_V):
            nn.init.orthogonal_(self.expand_neurons_V.data[i])
        nn.init.normal_(self.knowledge_K, std=0.02)
        nn.init.normal_(self.knowledge_V, std=0.02)


class GlobalSSM(nn.Module):
    """
    Selective SSM + Context Enhancement with Parallel Scan

    Mamba 라이브러리 사용 시 O(S) → O(log S) 병렬화

    - Selective: 토큰별 delta, B_t로 중요 정보만 h_final에 남김
    - Context: states를 x에 더해서 병목 보완 (optional)

    Args:
        return_context: if False, skips context computation for memory savings
    """
    def __init__(self, d_model: int, state_dim: int, return_context: bool = True):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.return_context = return_context

        # Mamba-style parameters
        # A: [d_model, state_dim] - 채널별 decay
        self.A_log = nn.Parameter(torch.randn(d_model, state_dim) * 0.1)

        # Input projections
        self.W_delta = nn.Linear(d_model, d_model, bias=False)  # delta per channel
        self.W_B = nn.Linear(d_model, state_dim, bias=False)
        self.W_C = nn.Linear(d_model, state_dim, bias=False)  # output projection from state

        # SSM output normalization (값 폭발 방지)
        self.ssm_norm = nn.LayerNorm(d_model)

        # Context projection
        self.context_proj = nn.Linear(d_model, d_model, bias=False)
        self.context_scale = nn.Parameter(torch.tensor(0.1))

        # Importance projection
        self.importance_proj = nn.Linear(d_model, d_model, bias=False)

        # Temperature for importance softmax (lower = sharper distribution)
        self.importance_temperature = 0.5

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_delta.weight, std=0.02)
        nn.init.normal_(self.W_B.weight, std=0.02)
        nn.init.normal_(self.W_C.weight, std=0.02)
        nn.init.normal_(self.context_proj.weight, std=0.02)
        nn.init.normal_(self.importance_proj.weight, std=0.02)

    def forward(self, x):
        """
        Args:
            x: [B, S, d_model]
        Returns:
            importance: [B, S]
            context: [B, S, d_model]
        """
        B, S, D = x.shape

        # Compute selective parameters
        delta = F.softplus(self.W_delta(x))  # [B, S, d_model]
        B_sel = self.W_B(x)  # [B, S, state_dim]
        C_sel = self.W_C(x)  # [B, S, state_dim]

        # A: negative for decay
        A = -torch.exp(self.A_log)  # [d_model, state_dim]

        if MAMBA_AVAILABLE:
            # 모든 텐서를 x와 같은 dtype으로
            dtype = x.dtype

            x_mamba = x.transpose(1, 2).contiguous()  # [B, D, S]
            delta_mamba = delta.transpose(1, 2).contiguous()  # [B, D, S]
            B_mamba = B_sel.transpose(1, 2).contiguous().to(dtype)  # [B, N, S]
            C_mamba = C_sel.transpose(1, 2).contiguous().to(dtype)  # [B, N, S]
            A = A.to(dtype)

            y = selective_scan_fn(
                x_mamba,
                delta_mamba,
                A,
                B_mamba,
                C_mamba,
                D=None,
                z=None,
                delta_bias=None,
                delta_softplus=False,
                return_last_state=False
            )
            ssm_out = y.transpose(1, 2).contiguous()  # [B, S, D]
        else:
            # Fallback: slow for-loop
            ssm_out = self._slow_forward(x, delta, A, B_sel, C_sel)

        # Normalize SSM output (값 폭발 방지)
        ssm_out = self.ssm_norm(ssm_out)

        # Importance from final state
        h_final = ssm_out[:, -1, :]  # [B, d_model]
        h_proj = self.importance_proj(h_final)  # [B, d_model]
        raw_importance = torch.einsum('bsd,bd->bs', x, h_proj)  # [B, S] - before softmax
        importance = F.softmax(raw_importance / self.importance_temperature, dim=-1)

        # Context enhancement (optional, for memory savings)
        if self.return_context:
            context = self.context_proj(ssm_out) * self.context_scale  # [B, S, d_model]
        else:
            context = None

        return importance, context, raw_importance

    def _slow_forward(self, x, delta, A, B_sel, C_sel):
        """Fallback slow implementation"""
        B, S, D = x.shape
        N = self.state_dim

        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(S):
            # h = h * exp(delta * A) + delta * B * x
            delta_t = delta[:, t, :, None]  # [B, D, 1]
            A_exp = A[None, :, :]  # [1, D, N]
            decay = torch.exp(delta_t * A_exp)  # [B, D, N]

            B_t = B_sel[:, t, None, :]  # [B, 1, N]
            x_t = x[:, t, :, None]  # [B, D, 1]

            h = h * decay + (delta_t * x_t) * B_t  # [B, D, N]

            # Output: y = h @ C
            C_t = C_sel[:, t, :]  # [B, N]
            y_t = torch.einsum('bdn,bn->bd', h, C_t)  # [B, D]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # [B, S, D]


class GlobalRouters(nn.Module):
    """
    v13.2: Unified neuron space routing

    1 UnifiedNeuronRouter for all neuron types:
    - All neurons in same d_space embedding
    - Type-specific slicing for compress/expand_QK/expand_V
    - Starvation bonus with decay
    """
    def __init__(self, d_model: int, n_compress: int, n_expand_QK: int, n_expand_V: int,
                 top_k_compress: int = 8, top_k_QK: int = 4, top_k_V: int = 6,
                 d_space: int = 64, router_dropout: float = 0.1, token_routing: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_compress = n_compress
        self.n_expand_QK = n_expand_QK
        self.n_expand_V = n_expand_V
        self.top_k_compress = top_k_compress
        self.top_k_QK = top_k_QK
        self.top_k_V = top_k_V
        self.token_routing = token_routing

        # Unified router (replaces 5 separate routers)
        self.neuron_router = UnifiedNeuronRouter(
            d_model, n_compress, n_expand_QK, n_expand_V,
            d_space=d_space, dropout=router_dropout
        )

    def _topk_sparsify(self, weights, k):
        """Apply top-k sparsification and renormalize"""
        topk_vals, topk_idx = torch.topk(weights, k, dim=-1)
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(-1, topk_idx, topk_vals)
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-8)
        return sparse_weights, topk_idx

    def get_attention_weights(self, x, importance, starvation_weight=0.0):
        """
        Compute attention routing weights with Top-k sparsification

        Returns: compress_weights, expand_weights_Q, expand_weights_K, expand_weights_V, routing_info, aux_loss
        """
        # Get logits from unified router
        compress_logits = self.neuron_router.get_logits(x, 'compress', starvation_weight)
        expand_logits_Q = self.neuron_router.get_logits(x, 'expand_Q', starvation_weight)
        expand_logits_K = self.neuron_router.get_logits(x, 'expand_K', starvation_weight)
        expand_logits_V = self.neuron_router.get_logits(x, 'expand_V', starvation_weight)

        compress_pref = F.softmax(compress_logits, dim=-1)
        expand_pref_Q = F.softmax(expand_logits_Q, dim=-1)
        expand_pref_K = F.softmax(expand_logits_K, dim=-1)
        expand_pref_V = F.softmax(expand_logits_V, dim=-1)

        # Compute aux_loss (load balance) directly here
        aux_loss = 0.0
        if self.training:
            # Compress
            usage_C = compress_pref.mean(dim=(0, 1))
            target_C = 1.0 / self.n_compress
            aux_loss = aux_loss + ((usage_C - target_C) ** 2).sum() * self.n_compress

            # Q expand
            usage_Q = expand_pref_Q.mean(dim=(0, 1))
            target_QK = 1.0 / self.n_expand_QK
            aux_loss = aux_loss + ((usage_Q - target_QK) ** 2).sum() * self.n_expand_QK

            # K expand
            usage_K = expand_pref_K.mean(dim=(0, 1))
            aux_loss = aux_loss + ((usage_K - target_QK) ** 2).sum() * self.n_expand_QK

            # V expand
            usage_V = expand_pref_V.mean(dim=(0, 1))
            target_V = 1.0 / self.n_expand_V
            aux_loss = aux_loss + ((usage_V - target_V) ** 2).sum() * self.n_expand_V

        if self.token_routing:
            # Token-level routing: use per-token weights directly [B, S, N]
            compress_weights = compress_pref
            expand_weights_Q = expand_pref_Q
            expand_weights_K = expand_pref_K
            expand_weights_V = expand_pref_V

            routing_info = {
                'compress_weights': compress_weights.detach(),
                'expand_weights_Q': expand_weights_Q.detach(),
                'expand_weights_K': expand_weights_K.detach(),
                'expand_weights_V': expand_weights_V.detach(),
                'token_routing': True,
            }
        else:
            # Batch-level routing: aggregate by importance [B, N]
            compress_weights_dense = torch.einsum('bs,bsn->bn', importance, compress_pref)
            expand_weights_Q_dense = torch.einsum('bs,bsn->bn', importance, expand_pref_Q)
            expand_weights_K_dense = torch.einsum('bs,bsn->bn', importance, expand_pref_K)
            expand_weights_V_dense = torch.einsum('bs,bsn->bn', importance, expand_pref_V)

            # Top-k sparsification
            compress_weights, compress_topk_idx = self._topk_sparsify(compress_weights_dense, self.top_k_compress)
            expand_weights_Q, expand_topk_idx_Q = self._topk_sparsify(expand_weights_Q_dense, self.top_k_QK)
            expand_weights_K, expand_topk_idx_K = self._topk_sparsify(expand_weights_K_dense, self.top_k_QK)
            expand_weights_V, expand_topk_idx_V = self._topk_sparsify(expand_weights_V_dense, self.top_k_V)

            routing_info = {
                # Sparse weights (for forward)
                'compress_weights': compress_weights.detach(),
                'expand_weights_Q': expand_weights_Q.detach(),
                'expand_weights_K': expand_weights_K.detach(),
                'expand_weights_V': expand_weights_V.detach(),
                # Token-level preferences (for monitoring)
                'compress_pref': compress_pref.detach(),
                'expand_pref_Q': expand_pref_Q.detach(),
                'expand_pref_K': expand_pref_K.detach(),
                'expand_pref_V': expand_pref_V.detach(),
                'token_routing': False,
            }

        # Update usage statistics
        if self.training:
            self.neuron_router.update_usage(compress_weights, 'compress')
            self.neuron_router.update_usage(expand_weights_Q, 'expand_Q')
            self.neuron_router.update_usage(expand_weights_K, 'expand_K')
            self.neuron_router.update_usage(expand_weights_V, 'expand_V')

        return compress_weights, expand_weights_Q, expand_weights_K, expand_weights_V, routing_info, aux_loss

    def get_memory_weights(self, x, importance, starvation_weight=0.0):
        """Compute memory routing weights with Top-k sparsification

        Returns: memory_weights, routing_info, aux_loss
        """
        memory_logits = self.neuron_router.get_logits(x, 'memory', starvation_weight)
        memory_pref = F.softmax(memory_logits, dim=-1)

        # Compute aux_loss (load balance) directly here
        aux_loss = 0.0
        if self.training:
            usage_M = memory_pref.mean(dim=(0, 1))
            target_M = 1.0 / self.n_compress
            aux_loss = ((usage_M - target_M) ** 2).sum() * self.n_compress

        if self.token_routing:
            # Token-level routing: use per-token weights directly [B, S, N]
            memory_weights = memory_pref
            routing_info = {
                'memory_weights': memory_weights.detach(),
                'token_routing': True,
            }
        else:
            # Batch-level routing: aggregate by importance [B, N]
            memory_weights_dense = torch.einsum('bs,bsn->bn', importance, memory_pref)
            memory_weights, memory_topk_idx = self._topk_sparsify(memory_weights_dense, self.top_k_compress)

            routing_info = {
                'memory_weights': memory_weights.detach(),
                'token_routing': False,
            }

        # Update usage statistics
        if self.training:
            self.neuron_router.update_usage(memory_weights, 'memory')

        return memory_weights, routing_info, aux_loss


class NeuronCircuit(nn.Module):
    """v13.1: Attention circuit with separate QK/V expand pools"""
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

    def forward(self, x, compress_weights, expand_weights_Q, expand_weights_K, expand_weights_V):
        """
        Args:
            x: [B, S, D]
            compress_weights: [B, N] or [B, S, N]
            expand_weights_Q: [B, N_QK] or [B, S, N_QK] - Q router weights (uses QK pool)
            expand_weights_K: [B, N_QK] or [B, S, N_QK] - K router weights (uses QK pool)
            expand_weights_V: [B, N_V] or [B, S, N_V] - V router weights (uses V pool)
        """
        B, S, D = x.shape
        token_routing = compress_weights.dim() == 3  # [B, S, N] vs [B, N]

        if token_routing:
            # Token-level routing: different matrix per token
            # 1. Per-token compress matrix [B, S, D, R]
            shared_compress = torch.einsum('bsn,ndr->bsdr', compress_weights,
                                            self.shared_neurons.compress_neurons)
            # 2. Compress: [B, S, D] @ [B, S, D, R] -> [B, S, R]
            h = torch.einsum('bsd,bsdr->bsr', x, shared_compress)

            # 3. Per-token expand matrices - Q/K use QK pool, V uses V pool
            pool_QK = self.shared_neurons.expand_neurons_QK
            pool_V = self.shared_neurons.expand_neurons_V
            shared_expand_Q = torch.einsum('bsn,nrd->bsrd', expand_weights_Q, pool_QK)
            shared_expand_K = torch.einsum('bsn,nrd->bsrd', expand_weights_K, pool_QK)
            shared_expand_V = torch.einsum('bsn,nrd->bsrd', expand_weights_V, pool_V)

            # 4. Generate Q/K/V: Q and K use different weights, V from V pool
            Q = torch.einsum('bsr,bsrd->bsd', h, shared_expand_Q)
            K = torch.einsum('bsr,bsrd->bsd', h, shared_expand_K)
            V = torch.einsum('bsr,bsrd->bsd', h, shared_expand_V)
        else:
            # Batch-level routing: same matrix for all tokens
            # 1. Shared compress matrix [B, D, R]
            shared_compress = torch.einsum('bn,ndr->bdr', compress_weights,
                                            self.shared_neurons.compress_neurons)
            # 2. Compress: [B, S, D] @ [B, D, R] -> [B, S, R]
            h = torch.einsum('bsd,bdr->bsr', x, shared_compress)

            # 3. Dynamic expand matrices - Q/K use QK pool, V uses V pool
            pool_QK = self.shared_neurons.expand_neurons_QK
            pool_V = self.shared_neurons.expand_neurons_V
            shared_expand_Q = torch.einsum('bn,nrd->brd', expand_weights_Q, pool_QK)
            shared_expand_K = torch.einsum('bn,nrd->brd', expand_weights_K, pool_QK)
            shared_expand_V = torch.einsum('bn,nrd->brd', expand_weights_V, pool_V)

            # 4. Generate Q/K/V: Q and K use different weights, V from V pool
            Q = torch.einsum('bsr,brd->bsd', h, shared_expand_Q)
            K = torch.einsum('bsr,brd->bsd', h, shared_expand_K)
            V = torch.einsum('bsr,brd->bsd', h, shared_expand_V)

        # 5. Multi-head Attention with FlashAttention
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # FlashAttention via PyTorch 2.0+
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)

        # 6. Output projection
        output = self.expand_O(attn_out)
        output = self.out_dropout(output)

        return output, None  # FlashAttention doesn't return attention weights


class NeuronMemory(nn.Module):
    """v13: Memory using global routing"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        knowledge_k: int = 8,
        knowledge_rank: int = None,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else rank
        self.knowledge_k = knowledge_k

        if self.knowledge_rank != rank:
            self.query_proj = nn.Linear(rank, self.knowledge_rank, bias=False)
        else:
            self.query_proj = None

    def forward(self, x, memory_weights):
        B, S, D = x.shape
        token_routing = memory_weights.dim() == 3  # [B, S, N] vs [B, N]

        if token_routing:
            # Token-level routing: different matrix per token
            shared_compress = torch.einsum('bsn,ndr->bsdr', memory_weights,
                                            self.shared_neurons.compress_neurons)
            Q = torch.einsum('bsd,bsdr->bsr', x, shared_compress)
        else:
            # Batch-level routing: same matrix for all tokens
            shared_compress = torch.einsum('bn,ndr->bdr', memory_weights,
                                            self.shared_neurons.compress_neurons)
            Q = torch.einsum('bsd,bdr->bsr', x, shared_compress)

        if self.query_proj is not None:
            Q = self.query_proj(Q)

        K = self.shared_neurons.knowledge_K
        V = self.shared_neurons.knowledge_V

        scores = Q @ K.T / math.sqrt(self.knowledge_rank)
        topk_scores, topk_idx = torch.topk(scores, self.knowledge_k, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        idx_expanded = topk_idx.unsqueeze(-1).expand(B, S, self.knowledge_k, self.d_model)
        V_expanded = V.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        selected_V = V_expanded.gather(2, idx_expanded)

        output = (selected_V * weights.unsqueeze(-1)).sum(dim=2)

        return output, {'knowledge_indices': topk_idx, 'knowledge_weights': weights}


class DAWNBlock(nn.Module):
    """DAWN v13 block"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        knowledge_k: int,
        knowledge_rank: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = NeuronCircuit(shared_neurons, d_model, n_heads, rank, dropout)
        self.memory = NeuronMemory(shared_neurons, d_model, rank, knowledge_k, knowledge_rank)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, importance, global_routers: GlobalRouters, starvation_weight=0.0):
        normed_x = self.norm1(x)
        compress_w, expand_Q, expand_K, expand_V, attn_routing, attn_aux_loss = \
            global_routers.get_attention_weights(normed_x, importance, starvation_weight)

        attn_out, _ = self.attn(normed_x, compress_w, expand_Q, expand_K, expand_V)
        x = x + attn_out

        normed_x2 = self.norm2(x)
        memory_w, mem_routing, mem_aux_loss = global_routers.get_memory_weights(normed_x2, importance, starvation_weight)

        mem_out, knowledge_info = self.memory(normed_x2, memory_w)
        x = x + self.dropout(mem_out)

        # Output norms for attn/mem balance monitoring
        attn_out_norm = attn_out.norm(dim=-1).mean().detach()
        mem_out_norm = mem_out.norm(dim=-1).mean().detach()

        routing_info = {
            'attention': {**attn_routing, 'neuron_weights': compress_w.detach()},
            'memory': {**mem_routing, **knowledge_info, 'neuron_weights': memory_w.detach()},
            'attn_out_norm': attn_out_norm,
            'mem_out_norm': mem_out_norm,
        }

        aux_loss = attn_aux_loss + mem_aux_loss
        return x, routing_info, aux_loss


class DAWN(nn.Module):
    """
    DAWN v13.2: Unified Neuron Router

    Changes from v13.1:
    - 5 separate routers → 1 UnifiedNeuronRouter
    - All neurons in same d_space embedding
    - Starvation bonus with decay for exploration
    - Usage EMA tracking per neuron type

    Architecture:
    - Pool: QK shared, V separate (same as v13.1)
    - Router: 1 unified with shared projection
    - Routing: token → proj → dot product → type-specific slicing
    """
    __version__ = "13.2"

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 320,
        n_layers: int = 4,
        n_heads: int = 4,
        rank: int = 64,
        max_seq_len: int = 128,
        n_compress: int = 48,
        n_expand_QK: int = 12,
        n_expand_V: int = 12,
        n_knowledge: int = 80,
        knowledge_k: int = 10,
        knowledge_rank: int = None,
        state_dim: int = 64,
        top_k_compress: int = 8,
        top_k_QK: int = 4,
        top_k_V: int = 6,
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
        self.rank = rank
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else rank
        self.max_seq_len = max_seq_len
        self.state_dim = state_dim
        self.top_k_compress = top_k_compress
        self.top_k_QK = top_k_QK
        self.top_k_V = top_k_V
        self.d_space = d_space
        self.token_routing = token_routing
        self.router_dropout = router_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.use_ssm_context = use_ssm_context

        self.n_compress = n_compress
        self.n_expand_QK = n_expand_QK
        self.n_expand_V = n_expand_V
        self.n_knowledge = n_knowledge
        self.knowledge_k = knowledge_k

        self.n_neurons = n_compress
        self.basis_rank = rank

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.shared_neurons = SharedNeurons(
            d_model=d_model, rank=rank, n_compress=n_compress,
            n_expand_QK=n_expand_QK, n_expand_V=n_expand_V,
            n_knowledge=n_knowledge, knowledge_rank=self.knowledge_rank,
        )

        # Selective SSM with context (optional for memory savings)
        self.global_ssm = GlobalSSM(d_model, state_dim, return_context=use_ssm_context)

        # Unified Neuron Router (replaces 5 separate routers)
        self.global_routers = GlobalRouters(
            d_model, n_compress, n_expand_QK, n_expand_V,
            top_k_compress, top_k_QK, top_k_V,
            d_space=d_space, router_dropout=router_dropout, token_routing=token_routing
        )

        self.layers = nn.ModuleList([
            DAWNBlock(
                shared_neurons=self.shared_neurons, d_model=d_model, n_heads=n_heads,
                rank=rank, knowledge_k=knowledge_k, knowledge_rank=self.knowledge_rank, dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # Auxiliary loss accumulator (set during forward)
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

    def forward(self, input_ids, labels=None, return_routing_info=False,
                step=None, total_steps=None):
        """
        Args:
            input_ids: [B, S] token ids
            labels: [B, S] labels for loss calculation
            return_routing_info: whether to return routing info
            step: current training step (for starvation decay)
            total_steps: total training steps (for starvation decay)
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Reset aux_loss accumulator
        self.aux_loss = 0.0

        # Calculate starvation weight (exponential decay with floor for continued exploration)
        if step is not None and total_steps is not None and total_steps > 0:
            starvation_weight = max(0.05, math.exp(-3.0 * step / total_steps))
        else:
            starvation_weight = 0.0  # default (inference or not provided)

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        routing_infos = []
        for layer in self.layers:
            # Selective SSM: importance + context (recalculated per layer)
            if self.gradient_checkpointing and self.training:
                importance, context, raw_importance = checkpoint(
                    self.global_ssm, x, use_reentrant=False
                )
            else:
                importance, context, raw_importance = self.global_ssm(x)
            if context is not None:
                x = x + context

            if self.gradient_checkpointing and self.training:
                x, routing_info, layer_aux_loss = checkpoint(
                    layer, x, importance, self.global_routers, starvation_weight,
                    use_reentrant=False
                )
            else:
                x, routing_info, layer_aux_loss = layer(x, importance, self.global_routers, starvation_weight)

            # Accumulate aux_loss
            self.aux_loss = self.aux_loss + layer_aux_loss

            if return_routing_info:
                routing_info['importance'] = importance.detach()
                routing_info['raw_importance'] = raw_importance.detach()
                routing_infos.append(routing_info)

        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100)
            if return_routing_info:
                return loss, logits, routing_infos
            return loss, logits

        if return_routing_info:
            return logits, routing_infos
        return logits

    def orthogonality_loss(self):
        # Compress: [n_compress, d_model, rank] -> W^T @ W = [n_compress, rank, rank]
        W_c = self.shared_neurons.compress_neurons
        WtW = torch.bmm(W_c.transpose(1, 2), W_c)  # [n_compress, rank, rank]
        I = torch.eye(self.rank, device=W_c.device).unsqueeze(0)
        loss_c = ((WtW - I) ** 2).mean()

        # Expand QK: [n_expand_QK, rank, d_model] -> W @ W^T
        W_e_QK = self.shared_neurons.expand_neurons_QK
        WWt_QK = torch.bmm(W_e_QK, W_e_QK.transpose(1, 2))
        loss_e_QK = ((WWt_QK - I) ** 2).mean()

        # Expand V: [n_expand_V, rank, d_model] -> W @ W^T
        W_e_V = self.shared_neurons.expand_neurons_V
        WWt_V = torch.bmm(W_e_V, W_e_V.transpose(1, 2))
        loss_e_V = ((WWt_V - I) ** 2).mean()

        return (loss_c + loss_e_QK + loss_e_V) / 3

    def knowledge_diversity_loss(self):
        K = self.shared_neurons.knowledge_K
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
        compress = self.shared_neurons.compress_neurons.numel()
        expand_QK = self.shared_neurons.expand_neurons_QK.numel()
        expand_V = self.shared_neurons.expand_neurons_V.numel()
        expand_pool = expand_QK + expand_V  # Combined for display
        knowledge = self.shared_neurons.knowledge_K.numel() + self.shared_neurons.knowledge_V.numel()
        embed = self.token_emb.weight.numel() + self.pos_emb.weight.numel()

        # Mamba-style SSM parameters
        ssm_total = (
            self.global_ssm.A_log.numel() +
            self.global_ssm.W_delta.weight.numel() +
            self.global_ssm.W_B.weight.numel() +
            self.global_ssm.W_C.weight.numel() +
            self.global_ssm.context_proj.weight.numel() +
            self.global_ssm.importance_proj.weight.numel()
        )

        # Unified router parameters
        routers = sum(p.numel() for p in self.global_routers.neuron_router.parameters())

        expand_o = self.layers[0].attn.expand_O.weight.numel() * self.n_layers
        norms = sum(p.numel() for n, p in self.named_parameters() if 'norm' in n)

        print(f"=== DAWN v13.2 Parameter Breakdown (Unified Router) ===")
        print(f"CompressNeurons:   {compress:,} ({compress/1e6:.2f}M)")
        print(f"ExpandPool QK:     {expand_QK:,} ({expand_QK/1e6:.2f}M)")
        print(f"ExpandPool V:      {expand_V:,} ({expand_V/1e6:.2f}M)")
        print(f"expand_O:          {expand_o:,} ({expand_o/1e3:.1f}K)")
        print(f"KnowledgeNeurons:  {knowledge:,} ({knowledge/1e3:.1f}K)")
        print(f"Embeddings:        {embed:,} ({embed/1e6:.2f}M)")
        print(f"Mamba SSM:         {ssm_total:,} ({ssm_total/1e3:.1f}K)")
        print(f"Unified Router:    {routers:,} ({routers/1e3:.1f}K) [d_space={self.d_space}]")
        print(f"LayerNorms:        {norms:,} ({norms/1e3:.1f}K)")
        print(f"---")
        print(f"Top-k Compress: {self.top_k_compress}/{self.n_compress}")
        print(f"Top-k QK:       {self.top_k_QK}/{self.n_expand_QK}")
        print(f"Top-k V:        {self.top_k_V}/{self.n_expand_V}")
        print(f"Mamba Available: {MAMBA_AVAILABLE}")
        print(f"Architecture: Mamba SSM → Context → Unified Router → FlashAttn")
        print(f"---")
        print(f"Total:             {self.count_parameters():,} ({self.count_parameters()/1e6:.2f}M)")

        return {
            'compress': compress, 'expand_QK': expand_QK, 'expand_V': expand_V,
            'expand_pool': expand_pool, 'expand_o': expand_o,
            'knowledge': knowledge, 'embeddings': embed, 'ssm': ssm_total,
            'routers': routers, 'norms': norms,
        }

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'rank': self.rank, 'knowledge_rank': self.knowledge_rank,
            'max_seq_len': self.max_seq_len, 'n_compress': self.n_compress,
            'n_expand_QK': self.n_expand_QK, 'n_expand_V': self.n_expand_V,
            'n_knowledge': self.n_knowledge, 'knowledge_k': self.knowledge_k,
            'state_dim': self.state_dim,
            'top_k_compress': self.top_k_compress,
            'top_k_QK': self.top_k_QK, 'top_k_V': self.top_k_V,
            'd_space': self.d_space,
            'router_dropout': self.router_dropout,
            'gradient_checkpointing': self.gradient_checkpointing,
            'token_routing': self.token_routing,
        }
