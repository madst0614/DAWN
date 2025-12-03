"""
DAWN v12.7: Selective SSM for Context-aware Importance

Selective SSM mechanism:
- 기존: 고정 A, B → 모든 토큰 동일하게 decay
- 변경: 토큰별 delta, B_t → 중요 토큰은 오래 남고, 불필요한 토큰은 빨리 사라짐

vs v12.5:
- SSM 유지 (importance 계산)
- context_proj 제거
- x = x + context 제거

vs v12.6:
- v12.6: SSM 완전 제거
- v12.7: Selective SSM 유지, context만 제거

Ablation hierarchy:
- v12.5: SSM + context
- v12.7: Selective SSM only (no context)
- v12.6: no SSM (simple projection)
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


class SharedNeurons(nn.Module):
    """v12.7: Shared neurons (same as v12.5)"""
    def __init__(
        self,
        d_model: int,
        rank: int,
        n_compress: int,
        n_expand: int,
        n_knowledge: int,
        knowledge_rank: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else rank
        self.n_compress = n_compress
        self.n_expand = n_expand
        self.n_knowledge = n_knowledge

        # Compress pool: [n_compress, d_model, rank]
        self.compress_neurons = nn.Parameter(torch.zeros(n_compress, d_model, rank))

        # Shared expand pool for Q/K/V
        self.expand_neurons_pool = nn.Parameter(torch.zeros(n_expand, rank, d_model))

        self.knowledge_K = nn.Parameter(torch.zeros(n_knowledge, self.knowledge_rank))
        self.knowledge_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_parameters()

    def _init_parameters(self):
        for i in range(self.n_compress):
            nn.init.orthogonal_(self.compress_neurons.data[i])
        for i in range(self.n_expand):
            nn.init.orthogonal_(self.expand_neurons_pool.data[i])
        nn.init.normal_(self.knowledge_K, std=0.02)
        nn.init.normal_(self.knowledge_V, std=0.02)


class GlobalSSM(nn.Module):
    """
    Selective SSM with Parallel Scan (No Context - Ablation)

    Mamba 라이브러리 사용 시 O(S) → O(log S) 병렬화
    v13과 동일하지만 context 없음 (ablation)

    - Selective: 토큰별 delta, B_t로 중요 정보만 h_final에 남김
    - Context: ❌ (v12.7 ablation)
    """
    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # Mamba-style parameters
        # A: [d_model, state_dim] - 채널별 decay
        self.A_log = nn.Parameter(torch.randn(d_model, state_dim) * 0.1)

        # Input projections
        self.W_delta = nn.Linear(d_model, d_model, bias=False)  # delta per channel
        self.W_B = nn.Linear(d_model, state_dim, bias=False)
        self.W_C = nn.Linear(d_model, state_dim, bias=False)  # output projection from state

        # Importance projection (no context_proj in v12.7)
        self.importance_proj = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_delta.weight, std=0.02)
        nn.init.normal_(self.W_B.weight, std=0.02)
        nn.init.normal_(self.W_C.weight, std=0.02)
        nn.init.normal_(self.importance_proj.weight, std=0.02)

    def forward(self, x):
        """
        Args:
            x: [B, S, d_model]
        Returns:
            importance: [B, S] (no context in v12.7)
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

        # Importance from final state (no context in v12.7)
        h_final = ssm_out[:, -1, :]  # [B, d_model]
        h_proj = self.importance_proj(h_final)  # [B, d_model]
        importance = torch.einsum('bsd,bd->bs', x, h_proj)
        importance = F.softmax(importance, dim=-1)

        return importance

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
    """v12.7: Global routers with Top-k sparse selection"""
    def __init__(self, d_model: int, n_compress: int, n_expand: int,
                 top_k_compress: int = 8, top_k_expand: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_compress = n_compress
        self.n_expand = n_expand
        self.top_k_compress = top_k_compress
        self.top_k_expand = top_k_expand

        # Attention routers
        self.compress_router = nn.Linear(d_model, n_compress, bias=False)
        self.expand_router_Q = nn.Linear(d_model, n_expand, bias=False)
        self.expand_router_K = nn.Linear(d_model, n_expand, bias=False)
        self.expand_router_V = nn.Linear(d_model, n_expand, bias=False)

        # Memory router
        self.memory_router = nn.Linear(d_model, n_compress, bias=False)

    def _topk_sparsify(self, weights, k):
        """Apply top-k sparsification and renormalize"""
        topk_vals, topk_idx = torch.topk(weights, k, dim=-1)
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(-1, topk_idx, topk_vals)
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-8)
        return sparse_weights, topk_idx

    def get_attention_weights(self, x, importance):
        # Compress routing
        compress_pref = F.softmax(self.compress_router(x), dim=-1)
        compress_weights_dense = torch.einsum('bs,bsn->bn', importance, compress_pref)

        # Top-k sparsification
        compress_weights, compress_topk_idx = self._topk_sparsify(compress_weights_dense, self.top_k_compress)

        # Expand routing
        expand_pref_Q = F.softmax(self.expand_router_Q(x), dim=-1)
        expand_pref_K = F.softmax(self.expand_router_K(x), dim=-1)
        expand_pref_V = F.softmax(self.expand_router_V(x), dim=-1)

        expand_weights_Q_dense = torch.einsum('bs,bsn->bn', importance, expand_pref_Q)
        expand_weights_K_dense = torch.einsum('bs,bsn->bn', importance, expand_pref_K)
        expand_weights_V_dense = torch.einsum('bs,bsn->bn', importance, expand_pref_V)

        expand_weights_Q, expand_topk_idx_Q = self._topk_sparsify(expand_weights_Q_dense, self.top_k_expand)
        expand_weights_K, expand_topk_idx_K = self._topk_sparsify(expand_weights_K_dense, self.top_k_expand)
        expand_weights_V, expand_topk_idx_V = self._topk_sparsify(expand_weights_V_dense, self.top_k_expand)

        routing_info = {
            # Sparse weights (for forward)
            'compress_weights': compress_weights.detach(),
            'expand_weights_Q': expand_weights_Q.detach(),
            'expand_weights_K': expand_weights_K.detach(),
            'expand_weights_V': expand_weights_V.detach(),
            # Dense weights (for load balance loss)
            'compress_weights_dense': compress_weights_dense.detach(),
            'expand_weights_Q_dense': expand_weights_Q_dense.detach(),
            'expand_weights_K_dense': expand_weights_K_dense.detach(),
            'expand_weights_V_dense': expand_weights_V_dense.detach(),
            # Top-k indices
            'compress_topk_idx': compress_topk_idx.detach(),
            'expand_topk_idx_Q': expand_topk_idx_Q.detach(),
            'expand_topk_idx_K': expand_topk_idx_K.detach(),
            'expand_topk_idx_V': expand_topk_idx_V.detach(),
            # Token-level preferences (for analysis)
            'compress_pref': compress_pref.detach(),
            'expand_pref_Q': expand_pref_Q.detach(),
            'expand_pref_K': expand_pref_K.detach(),
            'expand_pref_V': expand_pref_V.detach(),
        }

        return compress_weights, expand_weights_Q, expand_weights_K, expand_weights_V, routing_info

    def get_memory_weights(self, x, importance):
        memory_pref = F.softmax(self.memory_router(x), dim=-1)
        memory_weights_dense = torch.einsum('bs,bsn->bn', importance, memory_pref)
        memory_weights, memory_topk_idx = self._topk_sparsify(memory_weights_dense, self.top_k_compress)

        routing_info = {
            'memory_weights': memory_weights.detach(),
            'memory_weights_dense': memory_weights_dense.detach(),
            'memory_topk_idx': memory_topk_idx.detach(),
        }

        return memory_weights, routing_info


class NeuronCircuit(nn.Module):
    """v12.7: Attention circuit (same as v12.5)"""
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

    def forward(self, x, compress_weights, expand_weights_Q, expand_weights_K, expand_weights_V, mask=None):
        B, S, D = x.shape

        shared_compress = torch.einsum('bn,ndr->bdr', compress_weights,
                                        self.shared_neurons.compress_neurons)
        h = torch.einsum('bsd,bdr->bsr', x, shared_compress)

        pool = self.shared_neurons.expand_neurons_pool
        shared_expand_Q = torch.einsum('bn,nrd->brd', expand_weights_Q, pool)
        shared_expand_K = torch.einsum('bn,nrd->brd', expand_weights_K, pool)
        shared_expand_V = torch.einsum('bn,nrd->brd', expand_weights_V, pool)

        Q = torch.einsum('bsr,brd->bsd', h, shared_expand_Q)
        K = torch.einsum('bsr,brd->bsd', h, shared_expand_K)
        V = torch.einsum('bsr,brd->bsd', h, shared_expand_V)

        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # FlashAttention via PyTorch 2.0+ scaled_dot_product_attention
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)

        output = self.expand_O(attn_out)
        output = self.out_dropout(output)

        # FlashAttention doesn't return attention weights
        return output, None


class NeuronMemory(nn.Module):
    """v12.7: Memory (same as v12.5)"""
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
    """DAWN v12.7 block"""
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

    def forward(self, x, importance, global_routers: GlobalRouters, mask=None):
        normed_x = self.norm1(x)
        compress_w, expand_Q, expand_K, expand_V, attn_routing = \
            global_routers.get_attention_weights(normed_x, importance)

        attn_out, attn_weights = self.attn(
            normed_x, compress_w, expand_Q, expand_K, expand_V, mask
        )
        x = x + attn_out

        normed_x2 = self.norm2(x)
        memory_w, mem_routing = global_routers.get_memory_weights(normed_x2, importance)

        mem_out, knowledge_info = self.memory(normed_x2, memory_w)
        x = x + self.dropout(mem_out)

        routing_info = {
            'attention': {**attn_routing, 'neuron_weights': compress_w.detach()},  # FlashAttn: no attn_weights
            'memory': {**mem_routing, **knowledge_info, 'neuron_weights': memory_w.detach()},
        }
        return x, routing_info


class DAWN(nn.Module):
    """
    DAWN v12.7: Global SSM without Context Enhancement

    Ablation: SSM 유지, context 강화만 제거

    vs v12.5: context 제거
    vs v12.6: SSM 유지 (v12.6은 SSM 완전 제거)
    """
    __version__ = "12.7"

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 320,
        n_layers: int = 4,
        n_heads: int = 4,
        rank: int = 64,
        max_seq_len: int = 128,
        n_compress: int = 48,
        n_expand: int = 12,
        n_knowledge: int = 80,
        knowledge_k: int = 10,
        knowledge_rank: int = None,
        state_dim: int = 64,
        top_k_compress: int = 8,
        top_k_expand: int = 4,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
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
        self.top_k_expand = top_k_expand
        self.gradient_checkpointing = gradient_checkpointing

        self.n_compress = n_compress
        self.n_expand = n_expand
        self.n_knowledge = n_knowledge
        self.knowledge_k = knowledge_k

        self.n_neurons = n_compress
        self.basis_rank = rank

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.shared_neurons = SharedNeurons(
            d_model=d_model, rank=rank, n_compress=n_compress,
            n_expand=n_expand, n_knowledge=n_knowledge, knowledge_rank=self.knowledge_rank,
        )

        # Global SSM (v12.7: no context_proj)
        self.global_ssm = GlobalSSM(d_model, state_dim)

        # Top-k Global Routers
        self.global_routers = GlobalRouters(
            d_model, n_compress, n_expand, top_k_compress, top_k_expand
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
        B, S = input_ids.shape
        device = input_ids.device

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
        mask = ~mask.unsqueeze(0).unsqueeze(0)

        routing_infos = []
        for layer in self.layers:
            # v12.7: SSM returns importance only (no context), recalculated per layer
            importance = self.global_ssm(x)  # [B, S]

            if self.gradient_checkpointing and self.training:
                x, routing_info = checkpoint(
                    layer, x, importance, self.global_routers, mask,
                    use_reentrant=False
                )
            else:
                x, routing_info = layer(x, importance, self.global_routers, mask)
            if return_routing_info:
                routing_info['importance'] = importance.detach()
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
        loss = 0.0
        for i in range(self.n_compress):
            W = self.shared_neurons.compress_neurons[i]
            WtW = W.T @ W
            I = torch.eye(self.rank, device=W.device)
            loss += ((WtW - I) ** 2).mean()

        for i in range(self.n_expand):
            W = self.shared_neurons.expand_neurons_pool[i]
            WWt = W @ W.T
            I = torch.eye(self.rank, device=W.device)
            loss += ((WWt - I) ** 2).mean()

        return loss / (self.n_compress + self.n_expand)

    def routing_entropy_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def knowledge_diversity_loss(self):
        K = self.shared_neurons.knowledge_K
        K_norm = F.normalize(K, dim=-1)
        sim = K_norm @ K_norm.T
        mask = ~torch.eye(self.n_knowledge, dtype=torch.bool, device=K.device)
        return sim[mask].abs().mean()

    def load_balance_loss(self, routing_infos):
        """Load balance on dense (pre-top-k) weights to prevent dead neurons"""
        loss = 0.0
        count = 0

        for layer_info in routing_infos:
            # Use dense weights (before top-k)
            compress_w = layer_info['attention']['compress_weights_dense']
            target_compress = 1.0 / self.n_compress
            loss += ((compress_w.mean(dim=0) - target_compress) ** 2).sum() * self.n_compress
            count += 1

            target_expand = 1.0 / self.n_expand
            for key in ['expand_weights_Q_dense', 'expand_weights_K_dense', 'expand_weights_V_dense']:
                expand_w = layer_info['attention'][key]
                loss += ((expand_w.mean(dim=0) - target_expand) ** 2).sum() * self.n_expand
                count += 1

            mem_w = layer_info['memory']['memory_weights_dense']
            loss += ((mem_w.mean(dim=0) - target_compress) ** 2).sum() * self.n_compress
            count += 1

        return loss / (count + 1e-10)

    def get_auxiliary_losses(self):
        return {
            'orth_total': self.orthogonality_loss(),
            'knowledge_div': self.knowledge_diversity_loss(),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_by_component(self):
        compress = self.shared_neurons.compress_neurons.numel()
        expand_pool = self.shared_neurons.expand_neurons_pool.numel()
        knowledge = self.shared_neurons.knowledge_K.numel() + self.shared_neurons.knowledge_V.numel()
        embed = self.token_emb.weight.numel() + self.pos_emb.weight.numel()

        # Mamba-style SSM (v12.7: no context)
        ssm_total = (
            self.global_ssm.A_log.numel() +
            self.global_ssm.W_delta.weight.numel() +
            self.global_ssm.W_B.weight.numel() +
            self.global_ssm.W_C.weight.numel() +
            self.global_ssm.importance_proj.weight.numel()
        )

        routers = (
            self.global_routers.compress_router.weight.numel() +
            self.global_routers.expand_router_Q.weight.numel() +
            self.global_routers.expand_router_K.weight.numel() +
            self.global_routers.expand_router_V.weight.numel() +
            self.global_routers.memory_router.weight.numel()
        )

        expand_o = self.layers[0].attn.expand_O.weight.numel() * self.n_layers
        norms = sum(p.numel() for n, p in self.named_parameters() if 'norm' in n)

        print(f"=== DAWN v12.7 Parameter Breakdown (Mamba SSM + Top-k, No Context) ===")
        print(f"CompressNeurons:   {compress:,} ({compress/1e6:.2f}M)")
        print(f"ExpandPool (QKV):  {expand_pool:,} ({expand_pool/1e6:.2f}M)")
        print(f"expand_O:          {expand_o:,} ({expand_o/1e3:.1f}K)")
        print(f"KnowledgeNeurons:  {knowledge:,} ({knowledge/1e3:.1f}K)")
        print(f"Embeddings:        {embed:,} ({embed/1e6:.2f}M)")
        print(f"Mamba SSM:         {ssm_total:,} ({ssm_total/1e3:.1f}K) [A_log + W_delta + W_B + W_C + importance]")
        print(f"Global Routers:    {routers:,} ({routers/1e3:.1f}K)")
        print(f"LayerNorms:        {norms:,} ({norms/1e3:.1f}K)")
        print(f"---")
        print(f"Top-k Compress: {self.top_k_compress}/{self.n_compress}")
        print(f"Top-k Expand:   {self.top_k_expand}/{self.n_expand}")
        print(f"Mamba Available: {MAMBA_AVAILABLE}")
        print(f"Architecture: Mamba SSM → Top-k Routers → FlashAttn")
        print(f"Context: ❌ (ablation)")
        print(f"---")
        print(f"Total:             {self.count_parameters():,} ({self.count_parameters()/1e6:.2f}M)")

        return {
            'compress': compress, 'expand_pool': expand_pool, 'expand_o': expand_o,
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
            'n_expand': self.n_expand, 'n_knowledge': self.n_knowledge,
            'knowledge_k': self.knowledge_k, 'state_dim': self.state_dim,
            'top_k_compress': self.top_k_compress, 'top_k_expand': self.top_k_expand,
            'gradient_checkpointing': self.gradient_checkpointing,
        }
