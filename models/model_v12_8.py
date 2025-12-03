"""
DAWN v12.8: Top-k Sparse Mixing with Switch-style Load Balancing

Changes from v12.7:
- Soft mixing → Top-k sparse mixing
- Full neuron weighted sum → Top-k neuron gather + weighted sum
- Switch Transformer style load balance auxiliary loss

Top-k Router logic:
- compress: n_compress (288) → top_k_compress (16)
- expand Q/K/V: n_expand (72) → top_k_expand (8)
- memory: n_compress (288) → top_k_compress (16)

Load balance loss (Switch style):
    f = frequency of each neuron being selected (one-hot counts)
    P = mean routing probability per neuron
    aux_loss = n_neurons * sum(f * P)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SharedNeurons(nn.Module):
    """v12.8: Shared neurons (same as v12.7)"""
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
    """v12.8: Global SSM (same as v12.7)"""
    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
        self.importance_proj = nn.Linear(state_dim, d_model, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        device = x.device

        h = torch.zeros(B, self.state_dim, device=device)
        for t in range(S):
            h = h @ self.A + x[:, t] @ self.B

        h_proj = self.importance_proj(h)
        importance = torch.einsum('bsd,bd->bs', x, h_proj)
        importance = F.softmax(importance, dim=-1)

        return importance


class GlobalRouters(nn.Module):
    """
    v12.8: Global routers with Top-k sparse selection

    Changes from v12.7:
    - Add top_k_compress, top_k_expand parameters
    - Return top-k indices and renormalized weights
    - Track routing probabilities for load balance loss
    """
    def __init__(self, d_model: int, n_compress: int, n_expand: int,
                 top_k_compress: int = 16, top_k_expand: int = 8):
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

    def _topk_routing(self, weights, k, n_total):
        """
        Top-k selection with load balance info

        Args:
            weights: [B, n_total] routing weights (already normalized)
            k: number of top neurons to select
            n_total: total number of neurons

        Returns:
            topk_weights: [B, k] renormalized weights
            topk_idx: [B, k] indices
            load_balance_info: dict with 'router_probs' and 'selected_mask'
        """
        # Top-k selection
        topk_weights, topk_idx = torch.topk(weights, k, dim=-1)

        # Renormalize
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # For load balance loss: track which neurons were selected
        # selected_mask: [B, n_total] one-hot style mask
        selected_mask = torch.zeros_like(weights)
        selected_mask.scatter_(1, topk_idx, 1.0)

        load_balance_info = {
            'router_probs': weights.detach(),  # P: routing probabilities
            'selected_mask': selected_mask.detach(),  # f: selection frequency
        }

        return topk_weights, topk_idx, load_balance_info

    def get_attention_weights(self, x, importance):
        """
        Get Top-k attention routing weights

        Returns:
            compress_topk_weights: [B, k_compress]
            compress_topk_idx: [B, k_compress]
            expand_topk_weights_Q/K/V: [B, k_expand]
            expand_topk_idx_Q/K/V: [B, k_expand]
            routing_info: dict with load balance info
        """
        B = x.shape[0]

        # Compress routing
        compress_pref = F.softmax(self.compress_router(x), dim=-1)  # [B, S, n_compress]
        compress_weights = torch.einsum('bs,bsn->bn', importance, compress_pref)
        compress_weights = compress_weights / (compress_weights.sum(dim=-1, keepdim=True) + 1e-8)

        compress_topk_w, compress_topk_idx, compress_lb = self._topk_routing(
            compress_weights, self.top_k_compress, self.n_compress
        )

        # Expand routing - Q/K/V each
        expand_pref_Q = F.softmax(self.expand_router_Q(x), dim=-1)
        expand_pref_K = F.softmax(self.expand_router_K(x), dim=-1)
        expand_pref_V = F.softmax(self.expand_router_V(x), dim=-1)

        expand_weights_Q = torch.einsum('bs,bsn->bn', importance, expand_pref_Q)
        expand_weights_K = torch.einsum('bs,bsn->bn', importance, expand_pref_K)
        expand_weights_V = torch.einsum('bs,bsn->bn', importance, expand_pref_V)

        expand_weights_Q = expand_weights_Q / (expand_weights_Q.sum(dim=-1, keepdim=True) + 1e-8)
        expand_weights_K = expand_weights_K / (expand_weights_K.sum(dim=-1, keepdim=True) + 1e-8)
        expand_weights_V = expand_weights_V / (expand_weights_V.sum(dim=-1, keepdim=True) + 1e-8)

        expand_topk_w_Q, expand_topk_idx_Q, expand_lb_Q = self._topk_routing(
            expand_weights_Q, self.top_k_expand, self.n_expand
        )
        expand_topk_w_K, expand_topk_idx_K, expand_lb_K = self._topk_routing(
            expand_weights_K, self.top_k_expand, self.n_expand
        )
        expand_topk_w_V, expand_topk_idx_V, expand_lb_V = self._topk_routing(
            expand_weights_V, self.top_k_expand, self.n_expand
        )

        routing_info = {
            'compress_weights': compress_weights.detach(),
            'compress_topk_idx': compress_topk_idx.detach(),
            'compress_lb': compress_lb,
            'expand_weights_Q': expand_weights_Q.detach(),
            'expand_weights_K': expand_weights_K.detach(),
            'expand_weights_V': expand_weights_V.detach(),
            'expand_topk_idx_Q': expand_topk_idx_Q.detach(),
            'expand_topk_idx_K': expand_topk_idx_K.detach(),
            'expand_topk_idx_V': expand_topk_idx_V.detach(),
            'expand_lb_Q': expand_lb_Q,
            'expand_lb_K': expand_lb_K,
            'expand_lb_V': expand_lb_V,
        }

        return (compress_topk_w, compress_topk_idx,
                expand_topk_w_Q, expand_topk_idx_Q,
                expand_topk_w_K, expand_topk_idx_K,
                expand_topk_w_V, expand_topk_idx_V,
                routing_info)

    def get_memory_weights(self, x, importance):
        """Get Top-k memory routing weights"""
        memory_pref = F.softmax(self.memory_router(x), dim=-1)
        memory_weights = torch.einsum('bs,bsn->bn', importance, memory_pref)
        memory_weights = memory_weights / (memory_weights.sum(dim=-1, keepdim=True) + 1e-8)

        memory_topk_w, memory_topk_idx, memory_lb = self._topk_routing(
            memory_weights, self.top_k_compress, self.n_compress
        )

        routing_info = {
            'memory_weights': memory_weights.detach(),
            'memory_topk_idx': memory_topk_idx.detach(),
            'memory_lb': memory_lb,
        }

        return memory_topk_w, memory_topk_idx, routing_info


class NeuronCircuit(nn.Module):
    """
    v12.8: Attention circuit with Top-k sparse neuron selection

    Changes from v12.7:
    - Use gather instead of full einsum for neuron combination
    - Only combine top-k neurons
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        top_k_compress: int = 16,
        top_k_expand: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank = rank
        self.top_k_compress = top_k_compress
        self.top_k_expand = top_k_expand

        self.expand_O = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x,
                compress_topk_w, compress_topk_idx,
                expand_topk_w_Q, expand_topk_idx_Q,
                expand_topk_w_K, expand_topk_idx_K,
                expand_topk_w_V, expand_topk_idx_V,
                mask=None):
        B, S, D = x.shape

        # Gather top-k compress neurons: [B, k, d_model, rank]
        compress_neurons = self.shared_neurons.compress_neurons  # [n_compress, d_model, rank]
        selected_compress = compress_neurons[compress_topk_idx]  # [B, k, d_model, rank]

        # Weighted sum of top-k compress neurons
        # compress_topk_w: [B, k] -> [B, k, 1, 1]
        shared_compress = torch.einsum('bk,bkdr->bdr', compress_topk_w, selected_compress)

        # x @ compress -> h
        h = torch.einsum('bsd,bdr->bsr', x, shared_compress)

        # Gather top-k expand neurons for Q/K/V
        expand_pool = self.shared_neurons.expand_neurons_pool  # [n_expand, rank, d_model]

        selected_expand_Q = expand_pool[expand_topk_idx_Q]  # [B, k, rank, d_model]
        selected_expand_K = expand_pool[expand_topk_idx_K]
        selected_expand_V = expand_pool[expand_topk_idx_V]

        # Weighted sum of top-k expand neurons
        shared_expand_Q = torch.einsum('bk,bkrd->brd', expand_topk_w_Q, selected_expand_Q)
        shared_expand_K = torch.einsum('bk,bkrd->brd', expand_topk_w_K, selected_expand_K)
        shared_expand_V = torch.einsum('bk,bkrd->brd', expand_topk_w_V, selected_expand_V)

        # h @ expand -> Q/K/V
        Q = torch.einsum('bsr,brd->bsd', h, shared_expand_Q)
        K = torch.einsum('bsr,brd->bsd', h, shared_expand_K)
        V = torch.einsum('bsr,brd->bsd', h, shared_expand_V)

        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # FlashAttention
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
    v12.8: Memory with Top-k sparse compress selection
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        knowledge_k: int = 8,
        knowledge_rank: int = None,
        top_k_compress: int = 16,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else rank
        self.knowledge_k = knowledge_k
        self.top_k_compress = top_k_compress

        if self.knowledge_rank != rank:
            self.query_proj = nn.Linear(rank, self.knowledge_rank, bias=False)
        else:
            self.query_proj = None

    def forward(self, x, memory_topk_w, memory_topk_idx):
        B, S, D = x.shape

        # Gather top-k compress neurons
        compress_neurons = self.shared_neurons.compress_neurons
        selected_compress = compress_neurons[memory_topk_idx]  # [B, k, d_model, rank]

        # Weighted sum
        shared_compress = torch.einsum('bk,bkdr->bdr', memory_topk_w, selected_compress)
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
    """DAWN v12.8 block with Top-k sparse routing"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        knowledge_k: int,
        knowledge_rank: int = None,
        top_k_compress: int = 16,
        top_k_expand: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = NeuronCircuit(
            shared_neurons, d_model, n_heads, rank,
            top_k_compress, top_k_expand, dropout
        )
        self.memory = NeuronMemory(
            shared_neurons, d_model, rank, knowledge_k, knowledge_rank, top_k_compress
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, importance, global_routers: GlobalRouters, mask=None):
        normed_x = self.norm1(x)

        # Get Top-k routing
        (compress_topk_w, compress_topk_idx,
         expand_topk_w_Q, expand_topk_idx_Q,
         expand_topk_w_K, expand_topk_idx_K,
         expand_topk_w_V, expand_topk_idx_V,
         attn_routing) = global_routers.get_attention_weights(normed_x, importance)

        attn_out, _ = self.attn(
            normed_x,
            compress_topk_w, compress_topk_idx,
            expand_topk_w_Q, expand_topk_idx_Q,
            expand_topk_w_K, expand_topk_idx_K,
            expand_topk_w_V, expand_topk_idx_V,
            mask
        )
        x = x + attn_out

        normed_x2 = self.norm2(x)
        memory_topk_w, memory_topk_idx, mem_routing = global_routers.get_memory_weights(normed_x2, importance)

        mem_out, knowledge_info = self.memory(normed_x2, memory_topk_w, memory_topk_idx)
        x = x + self.dropout(mem_out)

        routing_info = {
            'attention': {
                **attn_routing,
                'compress_topk_w': compress_topk_w.detach(),
                'expand_topk_w_Q': expand_topk_w_Q.detach(),
            },
            'memory': {
                **mem_routing,
                **knowledge_info,
                'memory_topk_w': memory_topk_w.detach(),
            },
        }
        return x, routing_info


class DAWN(nn.Module):
    """
    DAWN v12.8: Top-k Sparse Mixing with Switch-style Load Balancing

    Changes from v12.7:
    - Soft mixing → Top-k sparse mixing
    - Switch Transformer style load balance loss
    - FlashAttention maintained
    """
    __version__ = "12.8"

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
        top_k_compress: int = 16,
        top_k_expand: int = 8,
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

        # Global SSM
        self.global_ssm = GlobalSSM(d_model, state_dim)

        # Global Routers with Top-k
        self.global_routers = GlobalRouters(
            d_model, n_compress, n_expand, top_k_compress, top_k_expand
        )

        self.layers = nn.ModuleList([
            DAWNBlock(
                shared_neurons=self.shared_neurons, d_model=d_model, n_heads=n_heads,
                rank=rank, knowledge_k=knowledge_k, knowledge_rank=self.knowledge_rank,
                top_k_compress=top_k_compress, top_k_expand=top_k_expand, dropout=dropout,
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

        importance = self.global_ssm(x)

        mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
        mask = ~mask.unsqueeze(0).unsqueeze(0)

        routing_infos = []
        for layer in self.layers:
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

    def switch_load_balance_loss(self, routing_infos):
        """
        Switch Transformer style load balance loss

        L_aux = n * sum(f_i * P_i)

        where:
        - f_i = fraction of samples selecting neuron i (selection frequency)
        - P_i = mean routing probability for neuron i
        - n = number of neurons

        This encourages uniform neuron utilization.
        """
        loss = 0.0
        count = 0

        for layer_info in routing_infos:
            # Compress load balance
            compress_lb = layer_info['attention']['compress_lb']
            f_compress = compress_lb['selected_mask'].mean(dim=0)  # [n_compress]
            P_compress = compress_lb['router_probs'].mean(dim=0)   # [n_compress]
            loss += self.n_compress * (f_compress * P_compress).sum()
            count += 1

            # Expand Q/K/V load balance
            for key in ['expand_lb_Q', 'expand_lb_K', 'expand_lb_V']:
                expand_lb = layer_info['attention'][key]
                f_expand = expand_lb['selected_mask'].mean(dim=0)
                P_expand = expand_lb['router_probs'].mean(dim=0)
                loss += self.n_expand * (f_expand * P_expand).sum()
                count += 1

            # Memory load balance
            memory_lb = layer_info['memory']['memory_lb']
            f_memory = memory_lb['selected_mask'].mean(dim=0)
            P_memory = memory_lb['router_probs'].mean(dim=0)
            loss += self.n_compress * (f_memory * P_memory).sum()
            count += 1

        return loss / (count + 1e-10)

    def load_balance_loss(self, routing_infos):
        """Alias for switch_load_balance_loss for compatibility"""
        return self.switch_load_balance_loss(routing_infos)

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

        ssm_total = (
            self.global_ssm.A.numel() +
            self.global_ssm.B.numel() +
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

        print(f"=== DAWN v12.8 Parameter Breakdown (Top-k Sparse) ===")
        print(f"CompressNeurons:   {compress:,} ({compress/1e6:.2f}M)")
        print(f"ExpandPool (QKV):  {expand_pool:,} ({expand_pool/1e6:.2f}M)")
        print(f"expand_O:          {expand_o:,} ({expand_o/1e3:.1f}K)")
        print(f"KnowledgeNeurons:  {knowledge:,} ({knowledge/1e3:.1f}K)")
        print(f"Embeddings:        {embed:,} ({embed/1e6:.2f}M)")
        print(f"Global SSM:        {ssm_total:,} ({ssm_total/1e3:.1f}K)")
        print(f"Global Routers:    {routers:,} ({routers/1e3:.1f}K)")
        print(f"LayerNorms:        {norms:,} ({norms/1e3:.1f}K)")
        print(f"---")
        print(f"Top-k Compress: {self.top_k_compress}/{self.n_compress}")
        print(f"Top-k Expand:   {self.top_k_expand}/{self.n_expand}")
        print(f"Architecture: Global SSM → Top-k Routers → FlashAttn")
        print(f"Load Balance: Switch Transformer style")
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
