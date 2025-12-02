"""
DAWN v12.4: Config-based Dynamic O Experiments

v12.4a: dynamic_O=True, low_rank_O=False
- O_pool: [n_O_expand, d_model, d_model]
- expand_router_O로 동적 O 행렬 생성
- n_heads=4

v12.4b: dynamic_O=False
- expand_O 제거 (attention output 직접 사용)
- n_heads=1

v12.4c: dynamic_O=True, low_rank_O=True
- O_compress_pool: [n_O_expand, d_model, O_rank]
- O_expand_pool: [n_O_expand, O_rank, d_model]
- 하나의 expand_router_O로 두 풀 공유
- n_heads=4

Base: v12.3 (SSM-guided Shared Expand Pool)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedNeurons(nn.Module):
    """v12.4: Shared neurons with configurable O"""
    def __init__(
        self,
        d_model: int,
        rank: int,
        n_compress: int,
        n_expand: int,
        n_knowledge: int,
        knowledge_rank: int = None,
        dynamic_O: bool = False,
        n_O_expand: int = 12,
        low_rank_O: bool = False,
        O_rank: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else rank
        self.n_compress = n_compress
        self.n_expand = n_expand
        self.n_knowledge = n_knowledge
        self.dynamic_O = dynamic_O
        self.n_O_expand = n_O_expand
        self.low_rank_O = low_rank_O
        self.O_rank = O_rank

        # Compress pool: [n_compress, d_model, rank]
        self.compress_neurons = nn.Parameter(torch.zeros(n_compress, d_model, rank))

        # Expand pool: [n_expand, rank, d_model] - general expand
        self.expand_neurons = nn.Parameter(torch.zeros(n_expand, rank, d_model))

        # Q/K/V shared expand pool
        self.expand_neurons_pool = nn.Parameter(torch.zeros(n_expand, rank, d_model))

        # O pool (only if dynamic_O)
        if dynamic_O:
            if low_rank_O:
                # Low-rank O: compress then expand
                # O_compress_pool: [n_O_expand, d_model, O_rank]
                # O_expand_pool: [n_O_expand, O_rank, d_model]
                self.O_compress_pool = nn.Parameter(torch.zeros(n_O_expand, d_model, O_rank))
                self.O_expand_pool = nn.Parameter(torch.zeros(n_O_expand, O_rank, d_model))
                self.register_parameter('O_pool', None)
            else:
                # Full-rank O: [n_O_expand, d_model, d_model]
                self.O_pool = nn.Parameter(torch.zeros(n_O_expand, d_model, d_model))
                self.register_parameter('O_compress_pool', None)
                self.register_parameter('O_expand_pool', None)
        else:
            self.register_parameter('O_pool', None)
            self.register_parameter('O_compress_pool', None)
            self.register_parameter('O_expand_pool', None)

        self.knowledge_K = nn.Parameter(torch.zeros(n_knowledge, self.knowledge_rank))
        self.knowledge_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_parameters()

    def _init_parameters(self):
        for i in range(self.n_compress):
            nn.init.orthogonal_(self.compress_neurons.data[i])
        for i in range(self.n_expand):
            nn.init.orthogonal_(self.expand_neurons.data[i])
            nn.init.orthogonal_(self.expand_neurons_pool.data[i])
        if self.dynamic_O:
            if self.low_rank_O:
                for i in range(self.n_O_expand):
                    nn.init.orthogonal_(self.O_compress_pool.data[i])
                    nn.init.orthogonal_(self.O_expand_pool.data[i])
            elif self.O_pool is not None:
                for i in range(self.n_O_expand):
                    nn.init.orthogonal_(self.O_pool.data[i])
        nn.init.normal_(self.knowledge_K, std=0.02)
        nn.init.normal_(self.knowledge_V, std=0.02)


class SSM(nn.Module):
    """Simple SSM for context state tracking"""
    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
        self.importance_proj = nn.Linear(state_dim, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, S, d_model]
        Returns:
            importance: [B, S] token importance
            h: [B, state_dim] final state
        """
        B, S, D = x.shape
        device = x.device

        h = torch.zeros(B, self.state_dim, device=device)

        for t in range(S):
            h = h @ self.A + x[:, t] @ self.B

        h_final = h
        h_proj = self.importance_proj(h_final)
        importance = torch.einsum('bsd,bd->bs', x, h_proj)
        importance = F.softmax(importance, dim=-1)

        return importance, h_final


class NeuronCircuit(nn.Module):
    """
    v12.4: SSM-guided with configurable O projection

    Options:
    - dynamic_O=True, low_rank_O=False: O from O_pool [n_O_expand, d_model, d_model]
    - dynamic_O=True, low_rank_O=True: O from O_compress_pool & O_expand_pool (low-rank)
    - dynamic_O=False: no O projection (direct output)
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_compress: int,
        n_expand: int,
        state_dim: int = 64,
        dropout: float = 0.1,
        dynamic_O: bool = False,
        n_O_expand: int = 12,
        low_rank_O: bool = False,
        O_rank: int = 64,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank = rank
        self.n_compress = n_compress
        self.n_expand = n_expand
        self.dynamic_O = dynamic_O
        self.n_O_expand = n_O_expand
        self.low_rank_O = low_rank_O
        self.O_rank = O_rank

        # SSM
        self.ssm = SSM(d_model, state_dim)

        # Router: compress
        self.compress_router = nn.Linear(d_model, n_compress, bias=False)

        # Router: expand Q/K/V
        self.expand_router_Q = nn.Linear(d_model, n_expand, bias=False)
        self.expand_router_K = nn.Linear(d_model, n_expand, bias=False)
        self.expand_router_V = nn.Linear(d_model, n_expand, bias=False)

        # Router: expand O (only if dynamic_O)
        # Same router for both full-rank and low-rank O
        if dynamic_O:
            self.expand_router_O = nn.Linear(d_model, n_O_expand, bias=False)
        else:
            self.expand_router_O = None

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, S, d_model]
            mask: [B, 1, S, S] causal mask
        Returns:
            output: [B, S, d_model]
            routing_info: dict
        """
        B, S, D = x.shape

        # 1. SSM -> token importance
        importance, ssm_state = self.ssm(x)  # [B, S], [B, state_dim]

        # 2. Compress routing
        compress_pref = F.softmax(self.compress_router(x), dim=-1)  # [B, S, n_compress]
        compress_weights = torch.einsum('bs,bsn->bn', importance, compress_pref)
        compress_weights = compress_weights / (compress_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 3. Expand routing - Q/K/V
        expand_pref_Q = F.softmax(self.expand_router_Q(x), dim=-1)
        expand_pref_K = F.softmax(self.expand_router_K(x), dim=-1)
        expand_pref_V = F.softmax(self.expand_router_V(x), dim=-1)

        expand_weights_Q = torch.einsum('bs,bsn->bn', importance, expand_pref_Q)
        expand_weights_K = torch.einsum('bs,bsn->bn', importance, expand_pref_K)
        expand_weights_V = torch.einsum('bs,bsn->bn', importance, expand_pref_V)

        expand_weights_Q = expand_weights_Q / (expand_weights_Q.sum(dim=-1, keepdim=True) + 1e-8)
        expand_weights_K = expand_weights_K / (expand_weights_K.sum(dim=-1, keepdim=True) + 1e-8)
        expand_weights_V = expand_weights_V / (expand_weights_V.sum(dim=-1, keepdim=True) + 1e-8)

        # 4. Shared compress matrix
        shared_compress = torch.einsum('bn,ndr->bdr', compress_weights,
                                        self.shared_neurons.compress_neurons)

        # 5. Compress
        h = torch.einsum('bsd,bdr->bsr', x, shared_compress)  # [B, S, rank]

        # 6. Dynamic expand matrices
        pool = self.shared_neurons.expand_neurons_pool

        shared_expand_Q = torch.einsum('bn,nrd->brd', expand_weights_Q, pool)
        shared_expand_K = torch.einsum('bn,nrd->brd', expand_weights_K, pool)
        shared_expand_V = torch.einsum('bn,nrd->brd', expand_weights_V, pool)

        # 7. Q/K/V reconstruction
        Q = torch.einsum('bsr,brd->bsd', h, shared_expand_Q)
        K = torch.einsum('bsr,brd->bsd', h, shared_expand_K)
        V = torch.einsum('bsr,brd->bsd', h, shared_expand_V)

        # 8. Multi-head Attention
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        attn_out = torch.matmul(attn, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)

        # 9. Output projection (configurable)
        if self.dynamic_O:
            # Dynamic O routing (same for both full-rank and low-rank)
            expand_pref_O = F.softmax(self.expand_router_O(x), dim=-1)
            expand_weights_O = torch.einsum('bs,bsn->bn', importance, expand_pref_O)
            expand_weights_O = expand_weights_O / (expand_weights_O.sum(dim=-1, keepdim=True) + 1e-8)

            if self.low_rank_O:
                # Low-rank O: (attn_out @ O_compress) @ O_expand
                # O_compress_pool: [n_O_expand, d_model, O_rank]
                # O_expand_pool: [n_O_expand, O_rank, d_model]
                shared_O_compress = torch.einsum('bn,ndr->bdr', expand_weights_O,
                                                  self.shared_neurons.O_compress_pool)
                shared_O_expand = torch.einsum('bn,nrd->brd', expand_weights_O,
                                                self.shared_neurons.O_expand_pool)
                # attn_out: [B, S, d_model] -> [B, S, O_rank] -> [B, S, d_model]
                h_o = torch.einsum('bsd,bdr->bsr', attn_out, shared_O_compress)
                output = torch.einsum('bsr,brd->bsd', h_o, shared_O_expand)
            else:
                # Full-rank O: O_pool [n_O_expand, d_model, d_model]
                shared_O = torch.einsum('bn,nde->bde', expand_weights_O, self.shared_neurons.O_pool)
                output = torch.einsum('bsd,bde->bse', attn_out, shared_O)
        else:
            # No O projection - direct output
            output = attn_out

        output = self.out_dropout(output)

        routing_info = {
            'importance': importance.detach(),
            'neuron_weights': compress_weights.detach(),
            'compress_weights': compress_weights.detach(),
            'expand_weights_Q': expand_weights_Q.detach(),
            'expand_weights_K': expand_weights_K.detach(),
            'expand_weights_V': expand_weights_V.detach(),
            'ssm_state': ssm_state.detach(),
            'attn_weights': attn.detach(),
        }

        if self.dynamic_O:
            routing_info['expand_weights_O'] = expand_weights_O.detach()

        return output, routing_info


class NeuronMemory(nn.Module):
    """v12.4: SSM-guided Knowledge Retrieval"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_compress: int,
        knowledge_k: int = 8,
        knowledge_rank: int = None,
        state_dim: int = 64,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else rank
        self.knowledge_k = knowledge_k
        self.n_compress = n_compress

        self.ssm = SSM(d_model, state_dim)
        self.router = nn.Linear(d_model, n_compress, bias=False)

        if self.knowledge_rank != rank:
            self.query_proj = nn.Linear(rank, self.knowledge_rank, bias=False)
        else:
            self.query_proj = None

    def forward(self, x):
        B, S, D = x.shape

        importance, _ = self.ssm(x)
        token_neuron_pref = F.softmax(self.router(x), dim=-1)
        neuron_weights = torch.einsum('bs,bsn->bn', importance, token_neuron_pref)
        neuron_weights = neuron_weights / (neuron_weights.sum(dim=-1, keepdim=True) + 1e-8)

        shared_compress = torch.einsum('bn,ndr->bdr', neuron_weights,
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

        routing_info = {
            'importance': importance.detach(),
            'neuron_weights': neuron_weights.detach(),
            'knowledge_indices': topk_idx,
            'knowledge_weights': weights,
        }
        return output, routing_info


class DAWNBlock(nn.Module):
    """DAWN v12.4 block"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_compress: int,
        n_expand: int,
        knowledge_k: int,
        knowledge_rank: int = None,
        state_dim: int = 64,
        dropout: float = 0.1,
        dynamic_O: bool = False,
        n_O_expand: int = 12,
        low_rank_O: bool = False,
        O_rank: int = 64,
    ):
        super().__init__()

        self.attn = NeuronCircuit(
            shared_neurons, d_model, n_heads, rank, n_compress, n_expand,
            state_dim, dropout, dynamic_O, n_O_expand, low_rank_O, O_rank
        )
        self.memory = NeuronMemory(
            shared_neurons, d_model, rank, n_compress, knowledge_k, knowledge_rank, state_dim
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_routing = self.attn(self.norm1(x), mask)
        x = x + attn_out

        mem_out, mem_routing = self.memory(self.norm2(x))
        x = x + self.dropout(mem_out)

        routing_info = {
            'attention': attn_routing,
            'memory': mem_routing,
        }
        return x, routing_info


class DAWN(nn.Module):
    """
    DAWN v12.4: Config-based Dynamic O Experiments

    v12.4a: dynamic_O=True, low_rank_O=False, n_heads=4
    - O_pool: [n_O_expand, d_model, d_model]
    - expand_router_O로 동적 O 행렬 생성

    v12.4b: dynamic_O=False, n_heads=1
    - expand_O 제거 (attention output 직접 사용)

    v12.4c: dynamic_O=True, low_rank_O=True, n_heads=4
    - O_compress_pool: [n_O_expand, d_model, O_rank]
    - O_expand_pool: [n_O_expand, O_rank, d_model]
    - 하나의 router로 저랭크 O 동적 생성
    """
    __version__ = "12.4"

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
        dropout: float = 0.1,
        # v12.4 specific
        dynamic_O: bool = False,
        n_O_expand: int = 12,
        low_rank_O: bool = False,
        O_rank: int = 64,
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

        self.n_compress = n_compress
        self.n_expand = n_expand
        self.n_knowledge = n_knowledge
        self.knowledge_k = knowledge_k

        # v12.4 specific
        self.dynamic_O = dynamic_O
        self.n_O_expand = n_O_expand
        self.low_rank_O = low_rank_O
        self.O_rank = O_rank

        self.n_neurons = n_compress
        self.basis_rank = rank

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # SharedNeurons
        self.shared_neurons = SharedNeurons(
            d_model=d_model,
            rank=rank,
            n_compress=n_compress,
            n_expand=n_expand,
            n_knowledge=n_knowledge,
            knowledge_rank=self.knowledge_rank,
            dynamic_O=dynamic_O,
            n_O_expand=n_O_expand,
            low_rank_O=low_rank_O,
            O_rank=O_rank,
        )

        # Layers
        self.layers = nn.ModuleList([
            DAWNBlock(
                shared_neurons=self.shared_neurons,
                d_model=d_model,
                n_heads=n_heads,
                rank=rank,
                n_compress=n_compress,
                n_expand=n_expand,
                knowledge_k=knowledge_k,
                knowledge_rank=self.knowledge_rank,
                state_dim=state_dim,
                dropout=dropout,
                dynamic_O=dynamic_O,
                n_O_expand=n_O_expand,
                low_rank_O=low_rank_O,
                O_rank=O_rank,
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
            x, routing_info = layer(x, mask)
            if return_routing_info:
                routing_infos.append(routing_info)

        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            if return_routing_info:
                return loss, logits, routing_infos
            return loss, logits

        if return_routing_info:
            return logits, routing_infos
        return logits

    def orthogonality_loss(self):
        loss = 0.0
        # SharedNeurons compress
        for i in range(self.n_compress):
            W = self.shared_neurons.compress_neurons[i]
            WtW = W.T @ W
            I = torch.eye(self.rank, device=W.device)
            loss += ((WtW - I) ** 2).mean()

        # SharedNeurons expand
        for i in range(self.n_expand):
            W = self.shared_neurons.expand_neurons[i]
            WWt = W @ W.T
            I = torch.eye(self.rank, device=W.device)
            loss += ((WWt - I) ** 2).mean()

        # expand_neurons_pool
        for i in range(self.n_expand):
            W = self.shared_neurons.expand_neurons_pool[i]
            WWt = W @ W.T
            I = torch.eye(self.rank, device=W.device)
            loss += ((WWt - I) ** 2).mean()

        # O_pool (if dynamic_O)
        if self.dynamic_O:
            if self.low_rank_O:
                # Low-rank O: O_compress_pool and O_expand_pool
                for i in range(self.n_O_expand):
                    W_c = self.shared_neurons.O_compress_pool[i]
                    WtW = W_c.T @ W_c
                    I = torch.eye(self.O_rank, device=W_c.device)
                    loss += ((WtW - I) ** 2).mean()

                    W_e = self.shared_neurons.O_expand_pool[i]
                    WWt = W_e @ W_e.T
                    I = torch.eye(self.O_rank, device=W_e.device)
                    loss += ((WWt - I) ** 2).mean()
                total_count = self.n_compress + self.n_expand + self.n_expand + self.n_O_expand * 2
            elif self.shared_neurons.O_pool is not None:
                for i in range(self.n_O_expand):
                    W = self.shared_neurons.O_pool[i]
                    WtW = W.T @ W
                    I = torch.eye(self.d_model, device=W.device)
                    loss += ((WtW - I) ** 2).mean()
                total_count = self.n_compress + self.n_expand + self.n_expand + self.n_O_expand
            else:
                total_count = self.n_compress + self.n_expand + self.n_expand
        else:
            total_count = self.n_compress + self.n_expand + self.n_expand

        return loss / total_count

    def routing_entropy_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def knowledge_diversity_loss(self):
        K = self.shared_neurons.knowledge_K
        K_norm = F.normalize(K, dim=-1)
        sim = K_norm @ K_norm.T
        mask = ~torch.eye(self.n_knowledge, dtype=torch.bool, device=K.device)
        return sim[mask].abs().mean()

    def load_balance_loss(self, routing_infos):
        """Neuron weight balance loss"""
        loss = 0.0
        count = 0

        for layer_info in routing_infos:
            # Compress neuron weights
            compress_w = layer_info['attention']['compress_weights']
            target_compress = 1.0 / self.n_compress
            loss += ((compress_w.mean(dim=0) - target_compress) ** 2).sum() * self.n_compress
            count += 1

            # Expand neuron weights (Q/K/V)
            target_expand = 1.0 / self.n_expand
            for key in ['expand_weights_Q', 'expand_weights_K', 'expand_weights_V']:
                expand_w = layer_info['attention'][key]
                loss += ((expand_w.mean(dim=0) - target_expand) ** 2).sum() * self.n_expand
                count += 1

            # O neuron weights (if dynamic_O)
            if self.dynamic_O and 'expand_weights_O' in layer_info['attention']:
                o_w = layer_info['attention']['expand_weights_O']
                target_o = 1.0 / self.n_O_expand
                loss += ((o_w.mean(dim=0) - target_o) ** 2).sum() * self.n_O_expand
                count += 1

            # Memory neuron weights
            mem_neuron_w = layer_info['memory']['neuron_weights']
            loss += ((mem_neuron_w.mean(dim=0) - target_compress) ** 2).sum() * self.n_compress
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
        expand = self.shared_neurons.expand_neurons.numel()
        expand_pool = self.shared_neurons.expand_neurons_pool.numel()

        if self.dynamic_O:
            if self.low_rank_O:
                o_compress_pool = self.shared_neurons.O_compress_pool.numel()
                o_expand_pool = self.shared_neurons.O_expand_pool.numel()
                o_pool = o_compress_pool + o_expand_pool
            elif self.shared_neurons.O_pool is not None:
                o_pool = self.shared_neurons.O_pool.numel()
            else:
                o_pool = 0
        else:
            o_pool = 0

        knowledge = (self.shared_neurons.knowledge_K.numel() +
                    self.shared_neurons.knowledge_V.numel())
        embed = self.token_emb.weight.numel() + self.pos_emb.weight.numel()

        # SSM parameters per layer
        ssm_per_layer = (
            self.layers[0].attn.ssm.A.numel() +
            self.layers[0].attn.ssm.B.numel() +
            self.layers[0].attn.ssm.importance_proj.weight.numel() +
            self.layers[0].memory.ssm.A.numel() +
            self.layers[0].memory.ssm.B.numel() +
            self.layers[0].memory.ssm.importance_proj.weight.numel()
        )
        ssm_total = ssm_per_layer * self.n_layers

        router_per_layer = (
            self.layers[0].attn.compress_router.weight.numel() +
            self.layers[0].attn.expand_router_Q.weight.numel() +
            self.layers[0].attn.expand_router_K.weight.numel() +
            self.layers[0].attn.expand_router_V.weight.numel() +
            self.layers[0].memory.router.weight.numel()
        )
        if self.dynamic_O and self.layers[0].attn.expand_router_O is not None:
            router_per_layer += self.layers[0].attn.expand_router_O.weight.numel()
        routers = router_per_layer * self.n_layers

        norms = sum(p.numel() for n, p in self.named_parameters() if 'norm' in n)

        if self.dynamic_O:
            if self.low_rank_O:
                o_type = f"low-rank O (rank={self.O_rank})"
            else:
                o_type = "full-rank O_pool"
        else:
            o_type = "no O projection"

        print(f"=== DAWN v12.4 Parameter Breakdown ===")
        print(f"Config: dynamic_O={self.dynamic_O}, low_rank_O={self.low_rank_O}, n_heads={self.n_heads}")
        print(f"CompressNeurons:   {compress:,} ({compress/1e6:.2f}M) [{self.n_compress} neurons, shared]")
        print(f"ExpandNeurons:     {expand:,} ({expand/1e6:.2f}M) [{self.n_expand} neurons, shared]")
        print(f"expand_pool (QKV): {expand_pool:,} ({expand_pool/1e6:.2f}M) [{self.n_expand} neurons, 1 shared pool]")
        if self.dynamic_O:
            if self.low_rank_O:
                print(f"O_compress_pool:   {o_compress_pool:,} ({o_compress_pool/1e6:.2f}M) [{self.n_O_expand} × {self.d_model} × {self.O_rank}]")
                print(f"O_expand_pool:     {o_expand_pool:,} ({o_expand_pool/1e6:.2f}M) [{self.n_O_expand} × {self.O_rank} × {self.d_model}]")
            else:
                print(f"O_pool:            {o_pool:,} ({o_pool/1e6:.2f}M) [{self.n_O_expand} × {self.d_model} × {self.d_model}]")
        else:
            print(f"O projection:      None ({o_type})")
        print(f"KnowledgeNeurons:  {knowledge:,} ({knowledge/1e3:.1f}K)")
        print(f"Embeddings:        {embed:,} ({embed/1e6:.2f}M)")
        print(f"SSM:               {ssm_total:,} ({ssm_total/1e3:.1f}K)")
        print(f"Routers:           {routers:,} ({routers/1e3:.1f}K) [compress + Q/K/V expand + memory" + (" + O" if self.dynamic_O else "") + "]")
        print(f"LayerNorms:        {norms:,} ({norms/1e3:.1f}K)")
        print(f"---")
        print(f"Architecture: SSM -> compress + expand_Q/K/V -> {o_type}")
        print(f"Attention in d_model space (d_head={self.d_model // self.n_heads})")
        print(f"---")
        print(f"Total:             {self.count_parameters():,} ({self.count_parameters()/1e6:.2f}M)")

        return {
            'compress': compress,
            'expand': expand,
            'expand_pool': expand_pool,
            'o_pool': o_pool,
            'knowledge': knowledge,
            'embeddings': embed,
            'ssm': ssm_total,
            'routers': routers,
            'norms': norms,
        }

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'rank': self.rank,
            'knowledge_rank': self.knowledge_rank,
            'max_seq_len': self.max_seq_len,
            'n_compress': self.n_compress,
            'n_expand': self.n_expand,
            'n_knowledge': self.n_knowledge,
            'knowledge_k': self.knowledge_k,
            'state_dim': self.state_dim,
            'dynamic_O': self.dynamic_O,
            'n_O_expand': self.n_O_expand,
            'low_rank_O': self.low_rank_O,
            'O_rank': self.O_rank,
        }
