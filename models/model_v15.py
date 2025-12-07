"""
DAWN v15: 2-Stage Hierarchical Knowledge Retrieval

Changes from v14:
- NeuronMemory: 2-stage hierarchical retrieval
  * Stage 1: Router-based coarse candidate selection (x → router → top-coarse_k)
  * Stage 2: h-based fine matching within candidates (h → proj_q → similarity)
- Knowledge neurons integrated into unified router space
- knowledge_rank: 64 → 128 (larger matching space)

Architecture (FRVK):
- Feature Neurons (F): x → low-rank projection (compress) [Attention only]
- Relational Neurons (R): Q/K generation for attention patterns
- Value Neurons (V): V generation for value transfer
- Knowledge Neurons (K): 2-stage retrieval (router coarse → h fine)
- Unified router with Excitability for balanced usage
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

    모든 뉴런(feature, relational, value, knowledge)이 같은 공간에 존재.
    토큰이 투영되어 가까운 뉴런 선택.
    Excitability (refractory-inspired)로 균형 잡힌 사용 유도.
    """
    def __init__(self, d_model, n_feature, n_relational, n_value, n_knowledge,
                 d_space=64, dropout=0.1):
        super().__init__()
        self.n_feature = n_feature
        self.n_relational = n_relational
        self.n_value = n_value
        self.n_knowledge = n_knowledge
        self.d_space = d_space

        total_neurons = n_feature + n_relational + n_value + n_knowledge
        self.total_neurons = total_neurons

        # 인덱스 경계
        self.feature_end = n_feature
        self.relational_end = n_feature + n_relational
        self.value_end = n_feature + n_relational + n_value
        # knowledge는 value_end ~ total_neurons

        # 공유 projection
        self.proj = nn.Linear(d_model, d_space)
        self.dropout = nn.Dropout(dropout)

        # 통합 뉴런 임베딩 [total_neurons, d_space]
        self.neuron_emb = nn.Parameter(torch.randn(total_neurons, d_space) * 0.02)

        # 타입별 usage 추적
        self.register_buffer('usage_ema_feature', torch.zeros(n_feature))
        self.register_buffer('usage_ema_relational', torch.zeros(n_relational))
        self.register_buffer('usage_ema_value', torch.zeros(n_value))
        self.register_buffer('usage_ema_knowledge', torch.zeros(n_knowledge))

        # Excitability: tau (recovery time constant) + decaying weight
        self.tau = 1.0
        self.excitability_weight = 1.0  # Decays over training (like starvation)

    def decay_excitability(self, decay_rate=0.9995):
        """Decay excitability_weight each step. Call from training loop."""
        self.excitability_weight *= decay_rate

    def get_excitability(self, usage_ema):
        """
        Neuronal excitability based on usage.
        Mirrors biological refractory dynamics.

        - usage_ema ≈ 0 → excitability = 1.0 (fully ready)
        - usage_ema ≈ tau → excitability ≈ 0.0 (refractory)

        Unlike floor/ceiling approaches, this just provides opportunity -
        gradient decides the rest.
        """
        return torch.clamp(1.0 - usage_ema / self.tau, min=0.0, max=1.0)

    def get_logits(self, x, neuron_type):
        """
        x: [B, S, d_model]
        neuron_type: 'feature', 'relational_Q', 'relational_K', 'value', 'knowledge'
        """
        h_proj = self.proj(x)  # [B, S, d_space]
        h_proj = self.dropout(h_proj)

        # Normalize neuron embeddings only (preserve token magnitude for routing signal)
        neuron_emb_norm = F.normalize(self.neuron_emb, dim=-1)

        # 전체 뉴런과 내적
        all_logits = torch.einsum('bsd,nd->bsn', h_proj, neuron_emb_norm)

        # 타입별 슬라이싱 + Excitability (refractory-inspired)
        if neuron_type == 'feature':
            logits = all_logits[..., :self.feature_end]
            if self.training:
                excitability = self.get_excitability(self.usage_ema_feature)
                logits = logits + excitability * self.excitability_weight

        elif neuron_type in ['relational_Q', 'relational_K']:
            logits = all_logits[..., self.feature_end:self.relational_end]
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

    def update_usage(self, weights, neuron_type):
        """top-k 후 선택된 뉴런 사용량 업데이트"""
        if not self.training:
            return

        # weights: [B, N] or [B, S, N]
        if weights.dim() == 3:
            usage = (weights > 0).float().mean(dim=[0, 1])
        else:
            usage = (weights > 0).float().mean(dim=0)

        if neuron_type == 'feature':
            self.usage_ema_feature = 0.99 * self.usage_ema_feature + 0.01 * usage
        elif neuron_type == 'relational':
            self.usage_ema_relational = 0.99 * self.usage_ema_relational + 0.01 * usage
        elif neuron_type == 'value':
            self.usage_ema_value = 0.99 * self.usage_ema_value + 0.01 * usage
        elif neuron_type == 'knowledge':
            self.usage_ema_knowledge = 0.99 * self.usage_ema_knowledge + 0.01 * usage

class SharedNeurons(nn.Module):
    """v14: Shared neurons with FRVK naming"""
    def __init__(
        self,
        d_model: int,
        rank: int,
        n_feature: int,
        n_relational: int,
        n_value: int,
        n_knowledge: int,
        knowledge_rank: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else rank
        self.n_feature = n_feature
        self.n_relational = n_relational
        self.n_value = n_value
        self.n_knowledge = n_knowledge

        # Feature pool: [n_feature, d_model, rank]
        self.feature_neurons = nn.Parameter(torch.zeros(n_feature, d_model, rank))

        # Separate expand pools: Relational shared, Value separate
        self.relational_neurons = nn.Parameter(torch.zeros(n_relational, rank, d_model))
        self.value_neurons = nn.Parameter(torch.zeros(n_value, rank, d_model))

        self.knowledge_neurons_K = nn.Parameter(torch.zeros(n_knowledge, self.knowledge_rank))
        self.knowledge_neurons_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_parameters()

    def _init_parameters(self):
        for i in range(self.n_feature):
            nn.init.orthogonal_(self.feature_neurons.data[i])
        for i in range(self.n_relational):
            nn.init.orthogonal_(self.relational_neurons.data[i])
        for i in range(self.n_value):
            nn.init.orthogonal_(self.value_neurons.data[i])
        nn.init.normal_(self.knowledge_neurons_K, std=0.02)
        nn.init.normal_(self.knowledge_neurons_V, std=0.02)


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
    v15: Unified neuron space routing with FRVK naming + Knowledge

    1 UnifiedNeuronRouter for all neuron types:
    - All neurons in same d_space embedding
    - Type-specific slicing for feature/relational/value/knowledge
    - Excitability (refractory-inspired) for balanced usage
    """
    def __init__(self, d_model: int, n_feature: int, n_relational: int, n_value: int,
                 n_knowledge: int,
                 top_k_feature: int = 8, top_k_relational: int = 4, top_k_value: int = 6,
                 d_space: int = 64, router_dropout: float = 0.1, token_routing: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_feature = n_feature
        self.n_relational = n_relational
        self.n_value = n_value
        self.n_knowledge = n_knowledge
        self.top_k_feature = top_k_feature
        self.top_k_relational = top_k_relational
        self.top_k_value = top_k_value
        self.token_routing = token_routing

        # Unified router (replaces 5 separate routers)
        self.neuron_router = UnifiedNeuronRouter(
            d_model, n_feature, n_relational, n_value, n_knowledge,
            d_space=d_space, dropout=router_dropout
        )

    def _topk_sparsify(self, weights, k):
        """Apply top-k sparsification and renormalize"""
        topk_vals, topk_idx = torch.topk(weights, k, dim=-1)
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(-1, topk_idx, topk_vals)
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-8)
        return sparse_weights, topk_idx

    def get_attention_weights(self, x, importance):
        """
        Compute attention routing weights with Top-k sparsification

        Returns: feature_weights, relational_weights_Q, relational_weights_K, value_weights, routing_info, aux_loss
        """
        # Get logits from unified router (Excitability applied internally)
        feature_logits = self.neuron_router.get_logits(x, 'feature')
        relational_logits_Q = self.neuron_router.get_logits(x, 'relational_Q')
        relational_logits_K = self.neuron_router.get_logits(x, 'relational_K')
        value_logits = self.neuron_router.get_logits(x, 'value')

        feature_pref = F.softmax(feature_logits, dim=-1)
        relational_pref_Q = F.softmax(relational_logits_Q, dim=-1)
        relational_pref_K = F.softmax(relational_logits_K, dim=-1)
        value_pref = F.softmax(value_logits, dim=-1)

        # Compute aux_loss (load balance) directly here
        aux_loss = 0.0
        if self.training:
            # Feature
            usage_F = feature_pref.mean(dim=(0, 1))
            target_F = 1.0 / self.n_feature
            aux_loss = aux_loss + ((usage_F - target_F) ** 2).sum() * self.n_feature

            # Relational Q
            usage_Q = relational_pref_Q.mean(dim=(0, 1))
            target_R = 1.0 / self.n_relational
            aux_loss = aux_loss + ((usage_Q - target_R) ** 2).sum() * self.n_relational

            # Relational K
            usage_K = relational_pref_K.mean(dim=(0, 1))
            aux_loss = aux_loss + ((usage_K - target_R) ** 2).sum() * self.n_relational

            # Value
            usage_V = value_pref.mean(dim=(0, 1))
            target_V = 1.0 / self.n_value
            aux_loss = aux_loss + ((usage_V - target_V) ** 2).sum() * self.n_value

        if self.token_routing:
            # Token-level routing: use per-token weights directly [B, S, N]
            feature_weights = feature_pref
            relational_weights_Q = relational_pref_Q
            relational_weights_K = relational_pref_K
            value_weights = value_pref

            routing_info = {
                'feature_weights': feature_weights.detach(),
                'relational_weights_Q': relational_weights_Q.detach(),
                'relational_weights_K': relational_weights_K.detach(),
                'value_weights': value_weights.detach(),
                'token_routing': True,
            }
        else:
            # Batch-level routing: aggregate by importance [B, N]
            feature_weights_dense = torch.einsum('bs,bsn->bn', importance, feature_pref)
            relational_weights_Q_dense = torch.einsum('bs,bsn->bn', importance, relational_pref_Q)
            relational_weights_K_dense = torch.einsum('bs,bsn->bn', importance, relational_pref_K)
            value_weights_dense = torch.einsum('bs,bsn->bn', importance, value_pref)

            # Top-k sparsification
            feature_weights, feature_topk_idx = self._topk_sparsify(feature_weights_dense, self.top_k_feature)
            relational_weights_Q, relational_topk_idx_Q = self._topk_sparsify(relational_weights_Q_dense, self.top_k_relational)
            relational_weights_K, relational_topk_idx_K = self._topk_sparsify(relational_weights_K_dense, self.top_k_relational)
            value_weights, value_topk_idx = self._topk_sparsify(value_weights_dense, self.top_k_value)

            routing_info = {
                # Sparse weights (for forward)
                'feature_weights': feature_weights.detach(),
                'relational_weights_Q': relational_weights_Q.detach(),
                'relational_weights_K': relational_weights_K.detach(),
                'value_weights': value_weights.detach(),
                # Token-level preferences (for monitoring)
                'feature_pref': feature_pref.detach(),
                'relational_pref_Q': relational_pref_Q.detach(),
                'relational_pref_K': relational_pref_K.detach(),
                'value_pref': value_pref.detach(),
                'token_routing': False,
            }

        # Update usage statistics
        if self.training:
            self.neuron_router.update_usage(feature_weights, 'feature')
            self.neuron_router.update_usage(value_weights, 'value')

            # Relational: Q OR K에서 선택되면 사용된 것으로 카운트
            relational_used = ((relational_weights_Q > 0) | (relational_weights_K > 0)).float()
            self.neuron_router.update_usage(relational_used, 'relational')

        return feature_weights, relational_weights_Q, relational_weights_K, value_weights, routing_info, aux_loss


class NeuronCircuit(nn.Module):
    """v14: Attention circuit with FRVK naming"""
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

    def forward(self, x, feature_weights, relational_weights_Q, relational_weights_K, value_weights):
        """
        Args:
            x: [B, S, D]
            feature_weights: [B, N] or [B, S, N]
            relational_weights_Q: [B, N_R] or [B, S, N_R] - Q router weights (uses Relational pool)
            relational_weights_K: [B, N_R] or [B, S, N_R] - K router weights (uses Relational pool)
            value_weights: [B, N_V] or [B, S, N_V] - V router weights (uses Value pool)
        """
        B, S, D = x.shape
        token_routing = feature_weights.dim() == 3  # [B, S, N] vs [B, N]

        if token_routing:
            # Token-level routing: different matrix per token
            # 1. Per-token feature matrix [B, S, D, R]
            shared_feature = torch.einsum('bsn,ndr->bsdr', feature_weights,
                                            self.shared_neurons.feature_neurons)
            # 2. Feature: [B, S, D] @ [B, S, D, R] -> [B, S, R]
            h = torch.einsum('bsd,bsdr->bsr', x, shared_feature)

            # 3. Per-token expand matrices - Q/K use Relational pool, V uses Value pool
            pool_R = self.shared_neurons.relational_neurons
            pool_V = self.shared_neurons.value_neurons
            shared_relational_Q = torch.einsum('bsn,nrd->bsrd', relational_weights_Q, pool_R)
            shared_relational_K = torch.einsum('bsn,nrd->bsrd', relational_weights_K, pool_R)
            shared_value = torch.einsum('bsn,nrd->bsrd', value_weights, pool_V)

            # 4. Generate Q/K/V: Q and K use different weights, V from Value pool
            Q = torch.einsum('bsr,bsrd->bsd', h, shared_relational_Q)
            K = torch.einsum('bsr,bsrd->bsd', h, shared_relational_K)
            V = torch.einsum('bsr,bsrd->bsd', h, shared_value)
        else:
            # Batch-level routing: same matrix for all tokens
            # 1. Shared feature matrix [B, D, R]
            shared_feature = torch.einsum('bn,ndr->bdr', feature_weights,
                                            self.shared_neurons.feature_neurons)
            # 2. Feature: [B, S, D] @ [B, D, R] -> [B, S, R]
            h = torch.einsum('bsd,bdr->bsr', x, shared_feature)

            # 3. Dynamic expand matrices - Q/K use Relational pool, V uses Value pool
            pool_R = self.shared_neurons.relational_neurons
            pool_V = self.shared_neurons.value_neurons
            shared_relational_Q = torch.einsum('bn,nrd->brd', relational_weights_Q, pool_R)
            shared_relational_K = torch.einsum('bn,nrd->brd', relational_weights_K, pool_R)
            shared_value = torch.einsum('bn,nrd->brd', value_weights, pool_V)

            # 4. Generate Q/K/V: Q and K use different weights, V from Value pool
            Q = torch.einsum('bsr,brd->bsd', h, shared_relational_Q)
            K = torch.einsum('bsr,brd->bsd', h, shared_relational_K)
            V = torch.einsum('bsr,brd->bsd', h, shared_value)

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
    """
    v15: 2-stage hierarchical knowledge retrieval

    Stage 1: Router-based coarse candidate selection (x → router → top-coarse_k)
             Uses unified router's knowledge logits for initial filtering
    Stage 2: h-based fine matching within candidates (h → proj_q → similarity → top-fine_k)
             Uses compressed representation for precise selection
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_knowledge: int,
        knowledge_rank: int = None,
        coarse_k: int = 20,
        fine_k: int = 10,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.n_knowledge = n_knowledge
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else rank
        self.coarse_k = coarse_k
        self.fine_k = fine_k

        # Stage 2: h → query projection for fine matching
        self.proj_q = nn.Linear(rank, self.knowledge_rank, bias=False)

    def forward(self, x, h, router):
        """
        v15: 2-stage hierarchical knowledge retrieval

        Args:
            x: [B, S, D] input embeddings (for Stage 1 coarse selection)
            h: [B, S, rank] compressed representation (for Stage 2 fine matching)
            router: UnifiedNeuronRouter (provides knowledge logits)

        Returns:
            output: [B, S, D] memory output
            info: dict with coarse/fine indices and weights
        """
        B, S, D = x.shape

        # ===== Stage 1: Coarse candidate selection via router =====
        # Router's knowledge logits include Excitability during training
        k_logits = router.get_logits(x, 'knowledge')  # [B, S, n_knowledge]
        coarse_scores, candidate_idx = torch.topk(k_logits, self.coarse_k, dim=-1)  # [B, S, coarse_k]

        # Update knowledge usage statistics
        if self.training:
            # Create sparse indicator for selected candidates
            coarse_indicator = torch.zeros(B, S, self.n_knowledge, device=x.device)
            coarse_indicator.scatter_(-1, candidate_idx, 1.0)
            router.update_usage(coarse_indicator, 'knowledge')

        # ===== Stage 2: Fine matching within candidates =====
        # h → query for fine-grained similarity
        query = self.proj_q(h)  # [B, S, knowledge_rank]

        # Get candidate keys from shared neurons
        K_all = self.shared_neurons.knowledge_neurons_K  # [n_knowledge, knowledge_rank]
        V_all = self.shared_neurons.knowledge_neurons_V  # [n_knowledge, d_model]

        # Gather candidate keys: [B, S, coarse_k, knowledge_rank]
        candidate_idx_k = candidate_idx.unsqueeze(-1).expand(B, S, self.coarse_k, self.knowledge_rank)
        K_candidates = K_all.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1).gather(2, candidate_idx_k)

        # Similarity within candidates
        # query: [B, S, knowledge_rank] → [B, S, 1, knowledge_rank]
        # K_candidates: [B, S, coarse_k, knowledge_rank]
        fine_scores = torch.einsum('bsd,bscd->bsc', query, K_candidates) / math.sqrt(self.knowledge_rank)
        # fine_scores: [B, S, coarse_k]

        # Select top-fine_k within coarse candidates
        fine_topk_scores, fine_topk_local_idx = torch.topk(fine_scores, self.fine_k, dim=-1)  # [B, S, fine_k]
        fine_weights = F.softmax(fine_topk_scores, dim=-1)  # [B, S, fine_k]

        # Map local indices back to global knowledge indices
        fine_global_idx = candidate_idx.gather(-1, fine_topk_local_idx)  # [B, S, fine_k]

        # Gather selected values
        fine_idx_v = fine_global_idx.unsqueeze(-1).expand(B, S, self.fine_k, self.d_model)
        V_expanded = V_all.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        selected_V = V_expanded.gather(2, fine_idx_v)  # [B, S, fine_k, d_model]

        # Weighted sum of selected values
        output = (selected_V * fine_weights.unsqueeze(-1)).sum(dim=2)  # [B, S, D]

        info = {
            'coarse_indices': candidate_idx,
            'coarse_scores': coarse_scores,
            'fine_indices': fine_global_idx,
            'fine_weights': fine_weights,
        }

        return output, info


class DAWNBlock(nn.Module):
    """DAWN v15 block with 2-stage knowledge retrieval"""
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
        self.shared_neurons = shared_neurons
        self.rank = rank

        self.attn = NeuronCircuit(shared_neurons, d_model, n_heads, rank, dropout)
        self.memory = NeuronMemory(
            shared_neurons, d_model, rank, n_knowledge,
            knowledge_rank=knowledge_rank, coarse_k=coarse_k, fine_k=fine_k
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, importance, global_routers: GlobalRouters):
        normed_x = self.norm1(x)
        feature_w, relational_Q, relational_K, value_w, attn_routing, attn_aux_loss = \
            global_routers.get_attention_weights(normed_x, importance)

        attn_out, _ = self.attn(normed_x, feature_w, relational_Q, relational_K, value_w)
        x = x + attn_out

        normed_x2 = self.norm2(x)

        # v15: 2-stage knowledge retrieval
        # Compute h (compressed representation) for Stage 2 fine matching
        # Reuse feature_w from attention routing for consistency
        token_routing = feature_w.dim() == 3
        if token_routing:
            shared_feature = torch.einsum('bsn,ndr->bsdr', feature_w,
                                          self.shared_neurons.feature_neurons)
            h = torch.einsum('bsd,bsdr->bsr', normed_x2, shared_feature)
        else:
            shared_feature = torch.einsum('bn,ndr->bdr', feature_w,
                                          self.shared_neurons.feature_neurons)
            h = torch.einsum('bsd,bdr->bsr', normed_x2, shared_feature)

        # Memory with 2-stage: x for coarse (router), h for fine (proj_q)
        mem_out, knowledge_info = self.memory(normed_x2, h, global_routers.neuron_router)
        x = x + self.dropout(mem_out)

        # Output norms for attn/mem balance monitoring
        attn_out_norm = attn_out.norm(dim=-1).mean().detach()
        mem_out_norm = mem_out.norm(dim=-1).mean().detach()

        routing_info = {
            'attention': {**attn_routing, 'neuron_weights': feature_w.detach()},
            'memory': {**{k: v.detach() for k, v in knowledge_info.items()}},
            'attn_out_norm': attn_out_norm,
            'mem_out_norm': mem_out_norm,
        }

        return x, routing_info, attn_aux_loss


class DAWN(nn.Module):
    """
    DAWN v15: 2-Stage Hierarchical Knowledge Retrieval

    Changes from v14:
    - NeuronMemory: 2-stage hierarchical retrieval
      * Stage 1: Router-based coarse candidate selection (x → router → top-coarse_k)
      * Stage 2: h-based fine matching within candidates (h → proj_q → similarity → top-fine_k)
    - Knowledge neurons integrated into unified router space
    - knowledge_rank: 64 → 128 (larger matching space)

    Architecture (FRVK):
    - Feature Neurons (F): input compression [Attention only]
    - Relational Neurons (R): Q/K generation
    - Value Neurons (V): V generation
    - Knowledge Neurons (K): 2-stage retrieval (router coarse → h fine)
    """
    __version__ = "15.0"

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 320,
        n_layers: int = 4,
        n_heads: int = 4,
        rank: int = 64,
        max_seq_len: int = 128,
        n_feature: int = 48,
        n_relational: int = 12,
        n_value: int = 12,
        n_knowledge: int = 80,
        coarse_k: int = 20,
        fine_k: int = 10,
        knowledge_rank: int = None,
        state_dim: int = 64,
        top_k_feature: int = 8,
        top_k_relational: int = 4,
        top_k_value: int = 6,
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
        self.top_k_feature = top_k_feature
        self.top_k_relational = top_k_relational
        self.top_k_value = top_k_value
        self.d_space = d_space
        self.token_routing = token_routing
        self.router_dropout = router_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.use_ssm_context = use_ssm_context

        self.n_feature = n_feature
        self.n_relational = n_relational
        self.n_value = n_value
        self.n_knowledge = n_knowledge
        self.coarse_k = coarse_k
        self.fine_k = fine_k

        self.n_neurons = n_feature
        self.basis_rank = rank

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.shared_neurons = SharedNeurons(
            d_model=d_model, rank=rank, n_feature=n_feature,
            n_relational=n_relational, n_value=n_value,
            n_knowledge=n_knowledge, knowledge_rank=self.knowledge_rank,
        )

        # Selective SSM with context (optional for memory savings)
        self.global_ssm = GlobalSSM(d_model, state_dim, return_context=use_ssm_context)

        # Unified Neuron Router (replaces 5 separate routers)
        self.global_routers = GlobalRouters(
            d_model, n_feature, n_relational, n_value, n_knowledge,
            top_k_feature, top_k_relational, top_k_value,
            d_space=d_space, router_dropout=router_dropout, token_routing=token_routing
        )

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

    def forward(self, input_ids, labels=None, return_routing_info=False):
        """
        Args:
            input_ids: [B, S] token ids
            labels: [B, S] labels for loss calculation
            return_routing_info: whether to return routing info
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Reset aux_loss accumulator
        self.aux_loss = 0.0

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
                    layer, x, importance, self.global_routers,
                    use_reentrant=False
                )
            else:
                x, routing_info, layer_aux_loss = layer(x, importance, self.global_routers)

            # Accumulate aux_loss
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
        # Feature: [n_feature, d_model, rank] -> W^T @ W = [n_feature, rank, rank]
        W_f = self.shared_neurons.feature_neurons
        WtW = torch.bmm(W_f.transpose(1, 2), W_f)  # [n_feature, rank, rank]
        I = torch.eye(self.rank, device=W_f.device).unsqueeze(0)
        loss_f = ((WtW - I) ** 2).mean()

        # Relational: [n_relational, rank, d_model] -> W @ W^T
        W_r = self.shared_neurons.relational_neurons
        WWt_r = torch.bmm(W_r, W_r.transpose(1, 2))
        loss_r = ((WWt_r - I) ** 2).mean()

        # Value: [n_value, rank, d_model] -> W @ W^T
        W_v = self.shared_neurons.value_neurons
        WWt_v = torch.bmm(W_v, W_v.transpose(1, 2))
        loss_v = ((WWt_v - I) ** 2).mean()

        return (loss_f + loss_r + loss_v) / 3

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
        feature = self.shared_neurons.feature_neurons.numel()
        relational = self.shared_neurons.relational_neurons.numel()
        value = self.shared_neurons.value_neurons.numel()
        knowledge = self.shared_neurons.knowledge_neurons_K.numel() + self.shared_neurons.knowledge_neurons_V.numel()
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

        print(f"=== DAWN v15 Parameter Breakdown (FRVK Architecture) ===")
        print(f"FeatureNeurons (F):    {feature:,} ({feature/1e6:.2f}M)")
        print(f"RelationalNeurons (R): {relational:,} ({relational/1e6:.2f}M)")
        print(f"ValueNeurons (V):      {value:,} ({value/1e6:.2f}M)")
        print(f"expand_O:              {expand_o:,} ({expand_o/1e3:.1f}K)")
        print(f"KnowledgeNeurons (K):  {knowledge:,} ({knowledge/1e3:.1f}K)")
        print(f"Embeddings:            {embed:,} ({embed/1e6:.2f}M)")
        print(f"Mamba SSM:             {ssm_total:,} ({ssm_total/1e3:.1f}K)")
        print(f"Unified Router:        {routers:,} ({routers/1e3:.1f}K) [d_space={self.d_space}]")
        print(f"LayerNorms:            {norms:,} ({norms/1e3:.1f}K)")
        print(f"---")
        print(f"Top-k Feature:    {self.top_k_feature}/{self.n_feature}")
        print(f"Top-k Relational: {self.top_k_relational}/{self.n_relational}")
        print(f"Top-k Value:      {self.top_k_value}/{self.n_value}")
        print(f"Mamba Available: {MAMBA_AVAILABLE}")
        print(f"Architecture: Mamba SSM → Context → Unified Router (Excitability) → FlashAttn")
        print(f"---")
        print(f"Total:                 {self.count_parameters():,} ({self.count_parameters()/1e6:.2f}M)")

        return {
            'feature': feature, 'relational': relational, 'value': value,
            'expand_o': expand_o,
            'knowledge': knowledge, 'embeddings': embed, 'ssm': ssm_total,
            'routers': routers, 'norms': norms,
        }

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'rank': self.rank, 'knowledge_rank': self.knowledge_rank,
            'max_seq_len': self.max_seq_len, 'n_feature': self.n_feature,
            'n_relational': self.n_relational, 'n_value': self.n_value,
            'n_knowledge': self.n_knowledge,
            'coarse_k': self.coarse_k, 'fine_k': self.fine_k,
            'state_dim': self.state_dim,
            'top_k_feature': self.top_k_feature,
            'top_k_relational': self.top_k_relational, 'top_k_value': self.top_k_value,
            'd_space': self.d_space,
            'router_dropout': self.router_dropout,
            'gradient_checkpointing': self.gradient_checkpointing,
            'token_routing': self.token_routing,
        }

    def get_model_info(self):
        """Return model architecture info for logging (used by train.py)"""
        return [
            f"DAWN v{self.__version__}: 2-Stage Hierarchical Knowledge Retrieval",
            f"  rank={self.rank}, knowledge_rank={self.knowledge_rank}",
            f"  FeatureNeurons (F): {self.n_feature} × {self.d_model} × {self.rank} [Attn only]",
            f"  RelationalNeurons (R): {self.n_relational} × {self.rank} × {self.d_model} (Q/K pool)",
            f"  ValueNeurons (V): {self.n_value} × {self.rank} × {self.d_model} (V pool)",
            f"  Top-k: F={self.top_k_feature}, R={self.top_k_relational}, V={self.top_k_value}",
            f"  KnowledgeNeurons (K): {self.n_knowledge} (coarse={self.coarse_k} → fine={self.fine_k})",
            f"  Memory: 2-stage (x→router→coarse_k, h→proj_q→fine_k)",
            f"  Router: d_space={self.d_space}, Excitability (SAR), Knowledge integrated",
        ]

    def load_state_dict(self, state_dict, strict=True):
        """Backward compatible load: remap old parameter names"""
        remapped = {}
        for key, value in state_dict.items():
            new_key = key
            # T → V rename (v14 backward compat)
            if 'transfer_neurons' in key:
                new_key = key.replace('transfer_neurons', 'value_neurons')
            if 'n_transfer' in key:
                new_key = key.replace('n_transfer', 'n_value')
            if 'usage_ema_transfer' in key:
                new_key = key.replace('usage_ema_transfer', 'usage_ema_value')
            # proj_k → proj_q rename (old v15 compat)
            if 'memory.proj_k' in key:
                new_key = key.replace('memory.proj_k', 'memory.proj_q')
            remapped[new_key] = value

        return super().load_state_dict(remapped, strict=strict)
