"""
DAWN v12.2: SSM-guided Dynamic Compress/Expand

Key features:
- SSM → neuron_weights (공유)
- compress: neuron_weights × compress_neurons
- expand_Q/K/V: neuron_weights × expand_neurons_Q/K/V
- 같은 뉴런 가중치로 압축-복원 일관성
- Attention in d_model space

vs v12.0:
- v12.0: expand_Q/K/V = nn.Linear (고정)
- v12.2: expand_neurons_Q/K/V = nn.Parameter (동적 뉴런 풀)

vs v12.1:
- v12.1: Q/K/V 각각 다른 router, rank attention
- v12.2: 하나의 router로 compress/expand 모두, d_model attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedNeurons(nn.Module):
    """v12.2: Same as v10.0"""
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

        self.compress_neurons = nn.Parameter(torch.zeros(n_compress, d_model, rank))
        self.expand_neurons = nn.Parameter(torch.zeros(n_expand, rank, d_model))
        self.knowledge_K = nn.Parameter(torch.zeros(n_knowledge, self.knowledge_rank))
        self.knowledge_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_parameters()

    def _init_parameters(self):
        for i in range(self.n_compress):
            nn.init.orthogonal_(self.compress_neurons.data[i])
        for i in range(self.n_expand):
            nn.init.orthogonal_(self.expand_neurons.data[i])
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
            importance: [B, S] 토큰별 중요도
            h: [B, state_dim] 최종 상태
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
    v12.2: SSM-guided Dynamic Compress/Expand

    SSM → neuron_weights (공유)
    compress: neuron_weights × compress_neurons
    expand_Q/K/V: neuron_weights × expand_neurons_Q/K/V

    같은 뉴런 가중치로 압축-복원 일관성
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
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # d_model attention
        self.rank = rank
        self.n_compress = n_compress
        self.n_expand = n_expand

        # SSM
        self.ssm = SSM(d_model, state_dim)

        # Router: 토큰별 뉴런 선호도 (하나로 공유)
        self.router = nn.Linear(d_model, n_compress, bias=False)

        # Expand neurons for Q/K/V (각각 다른 뉴런 풀)
        self.expand_neurons_Q = nn.Parameter(torch.zeros(n_compress, rank, d_model))
        self.expand_neurons_K = nn.Parameter(torch.zeros(n_compress, rank, d_model))
        self.expand_neurons_V = nn.Parameter(torch.zeros(n_compress, rank, d_model))

        # Output projection
        self.expand_O = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self._init_expand_neurons()

    def _init_expand_neurons(self):
        for i in range(self.n_compress):
            nn.init.orthogonal_(self.expand_neurons_Q.data[i])
            nn.init.orthogonal_(self.expand_neurons_K.data[i])
            nn.init.orthogonal_(self.expand_neurons_V.data[i])

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

        # 1. SSM → 토큰 중요도
        importance, ssm_state = self.ssm(x)  # [B, S], [B, state_dim]

        # 2. 토큰별 뉴런 선호도
        token_neuron_pref = F.softmax(self.router(x), dim=-1)  # [B, S, n_compress]

        # 3. 중요도 × 뉴런선호 → 뉴런 가중치 (공유)
        neuron_weights = torch.einsum('bs,bsn->bn', importance, token_neuron_pref)
        neuron_weights = neuron_weights / (neuron_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 4. 공유 compress 행렬 생성
        shared_compress = torch.einsum('bn,ndr->bdr', neuron_weights,
                                        self.shared_neurons.compress_neurons)

        # 5. 압축
        h = torch.einsum('bsd,bdr->bsr', x, shared_compress)  # [B, S, rank]

        # 6. 동적 expand 행렬 생성 (같은 neuron_weights 사용)
        shared_expand_Q = torch.einsum('bn,nrd->brd', neuron_weights, self.expand_neurons_Q)
        shared_expand_K = torch.einsum('bn,nrd->brd', neuron_weights, self.expand_neurons_K)
        shared_expand_V = torch.einsum('bn,nrd->brd', neuron_weights, self.expand_neurons_V)

        # 7. Q/K/V 복원 (같은 뉴런 관점으로)
        Q = torch.einsum('bsr,brd->bsd', h, shared_expand_Q)  # [B, S, d_model]
        K = torch.einsum('bsr,brd->bsd', h, shared_expand_K)
        V = torch.einsum('bsr,brd->bsd', h, shared_expand_V)

        # 8. Multi-head Attention in d_model space
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

        # 9. Output projection
        output = self.expand_O(attn_out)
        output = self.out_dropout(output)

        routing_info = {
            'importance': importance.detach(),
            'neuron_weights': neuron_weights.detach(),
            'ssm_state': ssm_state.detach(),
            'attn_weights': attn.detach(),
        }
        return output, routing_info


class NeuronMemory(nn.Module):
    """v12.2: SSM-guided Knowledge Retrieval"""
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

        # 1. SSM → 중요도
        importance, _ = self.ssm(x)

        # 2. 토큰별 뉴런 선호
        token_neuron_pref = F.softmax(self.router(x), dim=-1)

        # 3. 뉴런 가중치
        neuron_weights = torch.einsum('bs,bsn->bn', importance, token_neuron_pref)
        neuron_weights = neuron_weights / (neuron_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 4. 공유 compress
        shared_compress = torch.einsum('bn,ndr->bdr', neuron_weights,
                                        self.shared_neurons.compress_neurons)

        # 5. Query 생성
        Q = torch.einsum('bsd,bdr->bsr', x, shared_compress)  # [B, S, rank]

        if self.query_proj is not None:
            Q = self.query_proj(Q)

        # 6. Knowledge lookup
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
    """DAWN v12.2 block"""
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
    ):
        super().__init__()

        self.attn = NeuronCircuit(
            shared_neurons, d_model, n_heads, rank, n_compress, n_expand, state_dim, dropout
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
    DAWN v12.2: SSM-guided Dynamic Compress/Expand

    Key features:
    - SSM → neuron_weights (공유)
    - compress: neuron_weights × compress_neurons
    - expand_Q/K/V: neuron_weights × expand_neurons_Q/K/V
    - 같은 뉴런 가중치로 압축-복원 일관성

    vs v12.0:
    - v12.0: expand_Q/K/V = nn.Linear (고정)
    - v12.2: expand_neurons_Q/K/V = nn.Parameter (동적 뉴런 풀)

    vs v12.1:
    - v12.1: Q/K/V 각각 다른 router, rank attention
    - v12.2: 하나의 router로 compress/expand 모두, d_model attention
    """
    __version__ = "12.2"

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

        # SharedNeurons expand (not used in v12.2 NeuronCircuit but still exists)
        for i in range(self.n_expand):
            W = self.shared_neurons.expand_neurons[i]
            WWt = W @ W.T
            I = torch.eye(self.rank, device=W.device)
            loss += ((WWt - I) ** 2).mean()

        # Per-layer expand_neurons_Q/K/V
        for layer in self.layers:
            for i in range(self.n_compress):
                for neurons in [layer.attn.expand_neurons_Q, layer.attn.expand_neurons_K, layer.attn.expand_neurons_V]:
                    W = neurons[i]
                    WWt = W @ W.T
                    I = torch.eye(self.rank, device=W.device)
                    loss += ((WWt - I) ** 2).mean()

        total_count = self.n_compress + self.n_expand + self.n_layers * 3 * self.n_compress
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
        """뉴런 가중치 균형 loss"""
        loss = 0.0
        count = 0

        for layer_info in routing_infos:
            # Attention 뉴런 가중치
            neuron_w = layer_info['attention']['neuron_weights']  # [B, n_compress]
            target = 1.0 / self.n_compress
            loss += ((neuron_w.mean(dim=0) - target) ** 2).sum() * self.n_compress
            count += 1

            # Memory 뉴런 가중치
            mem_neuron_w = layer_info['memory']['neuron_weights']
            loss += ((mem_neuron_w.mean(dim=0) - target) ** 2).sum() * self.n_compress
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
            self.layers[0].attn.router.weight.numel() +
            self.layers[0].memory.router.weight.numel()
        )
        routers = router_per_layer * self.n_layers

        # expand_neurons_Q/K/V per layer
        expand_qkv_per_layer = (
            self.layers[0].attn.expand_neurons_Q.numel() +
            self.layers[0].attn.expand_neurons_K.numel() +
            self.layers[0].attn.expand_neurons_V.numel()
        )
        expand_qkv = expand_qkv_per_layer * self.n_layers

        # expand_O per layer
        expand_o_per_layer = self.layers[0].attn.expand_O.weight.numel()
        expand_o = expand_o_per_layer * self.n_layers

        norms = sum(p.numel() for n, p in self.named_parameters() if 'norm' in n)

        print(f"=== DAWN v12.2 Parameter Breakdown ===")
        print(f"CompressNeurons: {compress:,} ({compress/1e6:.2f}M) [shared]")
        print(f"ExpandNeurons:   {expand:,} ({expand/1e6:.2f}M) [shared, unused in attn]")
        print(f"expand_Q/K/V:    {expand_qkv:,} ({expand_qkv/1e6:.2f}M) [per-layer]")
        print(f"expand_O:        {expand_o:,} ({expand_o/1e3:.1f}K) [per-layer]")
        print(f"KnowledgeNeurons: {knowledge:,} ({knowledge/1e3:.1f}K)")
        print(f"Embeddings:      {embed:,} ({embed/1e6:.2f}M)")
        print(f"SSM:             {ssm_total:,} ({ssm_total/1e3:.1f}K)")
        print(f"Routers:         {routers:,} ({routers/1e3:.1f}K)")
        print(f"LayerNorms:      {norms:,} ({norms/1e3:.1f}K)")
        print(f"---")
        print(f"Architecture: SSM → neuron_weights → dynamic compress/expand")
        print(f"Attention in d_model space (d_head={self.d_model // self.n_heads})")
        print(f"---")
        print(f"Total:           {self.count_parameters():,} ({self.count_parameters()/1e6:.2f}M)")

        return {
            'compress': compress,
            'expand': expand,
            'expand_qkv': expand_qkv,
            'expand_o': expand_o,
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
        }
