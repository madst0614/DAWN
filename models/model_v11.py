"""
DAWN v11.0: d_model Attention Architecture

Key changes from v10:
- Attention: rank space → d_model space
- d_head: rank // n_heads → d_model // n_heads
- Q/K/V: compress → expand to d_model

Architecture:
    x (d_model)
    → compressor_Q/K/V: router → softmax → weighted compress_neurons → h (rank)
    → expand_Q/K/V: h (rank) → Q/K/V (d_model)
    → Attention in d_model space (d_head = d_model // n_heads)
    → expand_O: attn_out (d_model) → output (d_model)

v10 vs v11:
- v10: Attention in rank space (d_head = rank // n_heads)
- v11: Attention in d_model space (d_head = d_model // n_heads)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedNeurons(nn.Module):
    """
    v11.0: SharedNeurons (same as v10.0)

    - CompressNeurons: [n_compress, d_model, rank] - unified compression
    - ExpandNeurons: [n_expand, rank, d_model] - O shared
    - KnowledgeNeurons: [n_knowledge, knowledge_rank] + [n_knowledge, d_model]
    """
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

        # CompressNeurons: d_model → rank
        self.compress_neurons = nn.Parameter(torch.zeros(n_compress, d_model, rank))

        # ExpandNeurons: rank → d_model
        self.expand_neurons = nn.Parameter(torch.zeros(n_expand, rank, d_model))

        # KnowledgeNeurons
        self.knowledge_K = nn.Parameter(torch.zeros(n_knowledge, self.knowledge_rank))
        self.knowledge_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_parameters()

    def _init_parameters(self):
        # CompressNeurons: orthogonal init
        for i in range(self.n_compress):
            nn.init.orthogonal_(self.compress_neurons.data[i])

        # ExpandNeurons: orthogonal init
        for i in range(self.n_expand):
            nn.init.orthogonal_(self.expand_neurons.data[i])

        # KnowledgeNeurons
        nn.init.normal_(self.knowledge_K, std=0.02)
        nn.init.normal_(self.knowledge_V, std=0.02)


class Compressor(nn.Module):
    """
    d_model → rank compression with Soft Routing

    Flow:
    1. Router → scores
    2. Softmax over all neurons
    3. Weighted sum of all projections
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_compress: int,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.n_compress = n_compress

        # Independent router
        self.router = nn.Linear(d_model, n_compress, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, S, d_model]
        Returns:
            output: [B, S, rank]
            routing_info: dict with weights
        """
        # 1. Router → weights
        scores = self.router(x)  # [B, S, n_compress]
        weights = F.softmax(scores, dim=-1)  # [B, S, n_compress]

        # 2. Project with all neurons
        neurons = self.shared_neurons.compress_neurons  # [n_compress, d_model, rank]
        all_proj = torch.einsum('bsd,ndr->bsnr', x, neurons)  # [B, S, n_compress, rank]

        # 3. Weighted sum
        output = (all_proj * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, rank]

        routing_info = {'weights': weights}
        return output, routing_info


class Expander(nn.Module):
    """
    rank → d_model expansion with Soft Routing

    Flow:
    1. Router → scores
    2. Softmax over all neurons
    3. Weighted sum of all projections
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_expand: int,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.n_expand = n_expand

        # Independent router
        self.router = nn.Linear(rank, n_expand, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, S, rank]
        Returns:
            output: [B, S, d_model]
            routing_info: dict with weights
        """
        # 1. Router → weights
        scores = self.router(x)  # [B, S, n_expand]
        weights = F.softmax(scores, dim=-1)  # [B, S, n_expand]

        # 2. Project with all neurons
        neurons = self.shared_neurons.expand_neurons  # [n_expand, rank, d_model]
        all_proj = torch.einsum('bsr,nrd->bsnd', x, neurons)  # [B, S, n_expand, d_model]

        # 3. Weighted sum
        output = (all_proj * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, d_model]

        routing_info = {'weights': weights}
        return output, routing_info


class NeuronCircuit(nn.Module):
    """
    v11.0 Attention Layer

    Architecture:
    1. x → compressor_Q/K/V → h_Q/K/V (rank) - 각각 다른 압축!
    2. h_Q/K/V → expand_Q/K/V → Q/K/V (d_model)
    3. Attention in d_model space
    4. attn_out → expand_O → output
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_compress: int,
        n_expand: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # d_model based
        self.rank = rank

        # Separate compressors for Q/K/V (각각 다른 라우팅)
        self.compressor_Q = Compressor(shared_neurons, d_model, rank, n_compress)
        self.compressor_K = Compressor(shared_neurons, d_model, rank, n_compress)
        self.compressor_V = Compressor(shared_neurons, d_model, rank, n_compress)

        # Expand from rank to d_model for Q/K/V
        self.expand_Q = nn.Linear(rank, d_model, bias=False)
        self.expand_K = nn.Linear(rank, d_model, bias=False)
        self.expand_V = nn.Linear(rank, d_model, bias=False)

        # Output projection (d_model → d_model)
        self.expand_O = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, S, d_model]
            mask: [B, 1, S, S] causal mask
        Returns:
            output: [B, S, d_model]
            routing_info: dict with routing info
        """
        B, S, D = x.shape

        # 1. Compress Q/K/V separately (각각 다른 뉴런 조합)
        h_Q, q_info = self.compressor_Q(x)  # [B, S, rank]
        h_K, k_info = self.compressor_K(x)  # [B, S, rank]
        h_V, v_info = self.compressor_V(x)  # [B, S, rank]

        # 2. Expand to d_model
        Q = self.expand_Q(h_Q)  # [B, S, d_model]
        K = self.expand_K(h_K)  # [B, S, d_model]
        V = self.expand_V(h_V)  # [B, S, d_model]

        # 3. Reshape for multi-head attention (in d_model space)
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, S, d_head]
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # 4. Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # 5. Apply attention to V
        attn_out = torch.matmul(attn, V)  # [B, H, S, d_head]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)  # [B, S, d_model]

        # 6. Output projection
        output = self.expand_O(attn_out)  # [B, S, d_model]
        output = self.out_dropout(output)

        routing_info = {
            'Q': q_info,
            'K': k_info,
            'V': v_info,
            'attn_weights': attn.detach(),  # [B, H, S, S]
        }
        return output, routing_info


class NeuronMemory(nn.Module):
    """
    v11.0 Knowledge Retrieval

    Query compression → Knowledge lookup → Output
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_compress: int,
        knowledge_k: int = 8,
        knowledge_rank: int = None,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.knowledge_rank = knowledge_rank if knowledge_rank is not None else rank
        self.knowledge_k = knowledge_k

        # Query Compressor
        self.query_compressor = Compressor(shared_neurons, d_model, rank, n_compress)

        # Query projection if knowledge_rank differs from rank
        if self.knowledge_rank != rank:
            self.query_proj = nn.Linear(rank, self.knowledge_rank, bias=False)
        else:
            self.query_proj = None

    def forward(self, x):
        """
        Args:
            x: [B, S, d_model]
        Returns:
            output: [B, S, d_model]
            routing_info: dict
        """
        B, S, D = x.shape

        # Query compression
        Q, q_info = self.query_compressor(x)  # [B, S, rank]

        # Project to knowledge_rank if needed
        if self.query_proj is not None:
            Q = self.query_proj(Q)  # [B, S, knowledge_rank]

        # Knowledge lookup
        K = self.shared_neurons.knowledge_K  # [n_knowledge, knowledge_rank]
        V = self.shared_neurons.knowledge_V  # [n_knowledge, d_model]

        scores = Q @ K.T / math.sqrt(self.knowledge_rank)  # [B, S, n_knowledge]
        topk_scores, topk_idx = torch.topk(scores, self.knowledge_k, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)  # [B, S, k]

        # Gather selected V
        idx_expanded = topk_idx.unsqueeze(-1).expand(B, S, self.knowledge_k, self.d_model)
        V_expanded = V.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        selected_V = V_expanded.gather(2, idx_expanded)  # [B, S, k, d_model]

        # Weighted sum
        output = (selected_V * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, d_model]

        routing_info = {
            'M': q_info,
            'knowledge_indices': topk_idx,
            'knowledge_weights': weights,
        }
        return output, routing_info


class DAWNBlock(nn.Module):
    """Single DAWN v11.0 block: Attention + FFN(Memory) + LayerNorms"""
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
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = NeuronCircuit(
            shared_neurons, d_model, n_heads, rank, n_compress, n_expand, dropout
        )
        self.memory = NeuronMemory(
            shared_neurons, d_model, rank, n_compress, knowledge_k, knowledge_rank
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Returns:
            x: [B, S, d_model]
            routing_info: dict with attention and memory routing
        """
        # Attention
        attn_out, attn_routing = self.attn(self.norm1(x), mask)
        x = x + attn_out  # dropout already in NeuronCircuit

        # Memory (FFN replacement)
        mem_out, mem_routing = self.memory(self.norm2(x))
        x = x + self.dropout(mem_out)

        routing_info = {
            'attention': attn_routing,
            'memory': mem_routing,
        }
        return x, routing_info


class DAWN(nn.Module):
    """
    DAWN v11.0: d_model Attention Architecture

    Key changes from v10:
    - Attention in d_model space (not rank space)
    - d_head = d_model // n_heads (not rank // n_heads)
    - Q/K/V: compress → expand to d_model

    v10 vs v11:
    - v10: Attention in rank space (d_head = rank // n_heads)
    - v11: Attention in d_model space (d_head = d_model // n_heads)
    """
    __version__ = "11.0"

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 320,
        n_layers: int = 4,
        n_heads: int = 4,
        rank: int = 80,
        max_seq_len: int = 128,
        n_compress: int = 64,
        n_expand: int = 64,
        n_knowledge: int = 80,
        knowledge_k: int = 10,
        knowledge_rank: int = None,
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

        # Config storage
        self.n_compress = n_compress
        self.n_expand = n_expand
        self.n_knowledge = n_knowledge
        self.knowledge_k = knowledge_k

        # train.py compatibility
        self.n_neurons = n_compress  # For load balance loss
        self.basis_rank = rank  # For analysis scripts

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # SharedNeurons (shared across all layers)
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
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
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

        # Embeddings
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
        mask = ~mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        # Layers
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
        """CompressNeurons/ExpandNeurons orthogonality"""
        loss = 0.0

        # compress_neurons: [d_model, rank] → W.T @ W ≈ I
        for i in range(self.n_compress):
            W = self.shared_neurons.compress_neurons[i]
            WtW = W.T @ W
            I = torch.eye(self.rank, device=W.device)
            loss += ((WtW - I) ** 2).mean()

        # expand_neurons: [rank, d_model] → W @ W.T ≈ I
        for i in range(self.n_expand):
            W = self.shared_neurons.expand_neurons[i]
            WWt = W @ W.T
            I = torch.eye(self.rank, device=W.device)
            loss += ((WWt - I) ** 2).mean()

        return loss / (self.n_compress + self.n_expand)

    def routing_entropy_loss(self):
        """Placeholder for routing entropy"""
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def knowledge_diversity_loss(self):
        """Knowledge K vectors diversity"""
        K = self.shared_neurons.knowledge_K
        K_norm = F.normalize(K, dim=-1)
        sim = K_norm @ K_norm.T
        mask = ~torch.eye(self.n_knowledge, dtype=torch.bool, device=K.device)
        return sim[mask].abs().mean()

    def load_balance_loss(self, routing_infos):
        """
        Load balance loss for soft routing

        Args:
            routing_infos: forward에서 반환된 layer별 routing 정보
        Returns:
            load balance loss (lower = more balanced)
        """
        loss = 0.0
        count = 0

        for layer_info in routing_infos:
            # Attention Q/K/V compressors
            for comp in ['Q', 'K', 'V']:
                weights = layer_info['attention'][comp]['weights']  # [B, S, n_compress]
                usage = weights.mean(dim=(0, 1))  # [n_compress]
                target = 1.0 / self.n_compress
                loss += ((usage - target) ** 2).sum() * self.n_compress
                count += 1

            # Memory M compressor
            m_weights = layer_info['memory']['M']['weights']  # [B, S, n_compress]
            m_usage = m_weights.mean(dim=(0, 1))
            target = 1.0 / self.n_compress
            loss += ((m_usage - target) ** 2).sum() * self.n_compress
            count += 1

        return loss / (count + 1e-10)

    def get_auxiliary_losses(self):
        """train.py compatibility"""
        return {
            'orth_total': self.orthogonality_loss(),
            'knowledge_div': self.knowledge_diversity_loss(),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_by_component(self):
        """Component-wise parameter count"""
        compress = self.shared_neurons.compress_neurons.numel()
        expand = self.shared_neurons.expand_neurons.numel()
        knowledge = (self.shared_neurons.knowledge_K.numel() +
                    self.shared_neurons.knowledge_V.numel())

        embed = self.token_emb.weight.numel() + self.pos_emb.weight.numel()

        # Routers: per layer (3 compressors Q/K/V + 1 memory compressor)
        router_per_layer = (
            self.layers[0].attn.compressor_Q.router.weight.numel() +
            self.layers[0].attn.compressor_K.router.weight.numel() +
            self.layers[0].attn.compressor_V.router.weight.numel() +
            self.layers[0].memory.query_compressor.router.weight.numel()
        )
        routers = router_per_layer * self.n_layers

        # expand_Q/K/V/O per layer
        expand_qkvo_per_layer = (
            self.layers[0].attn.expand_Q.weight.numel() +
            self.layers[0].attn.expand_K.weight.numel() +
            self.layers[0].attn.expand_V.weight.numel() +
            self.layers[0].attn.expand_O.weight.numel()
        )
        expand_qkvo = expand_qkvo_per_layer * self.n_layers

        # LayerNorms
        norms = sum(p.numel() for n, p in self.named_parameters() if 'norm' in n)

        print(f"=== DAWN v11.0 Parameter Breakdown ===")
        print(f"CompressNeurons: {compress:,} ({compress/1e6:.2f}M)")
        print(f"ExpandNeurons:   {expand:,} ({expand/1e6:.2f}M)")
        print(f"KnowledgeNeurons: {knowledge:,} ({knowledge/1e3:.1f}K)")
        print(f"Embeddings:      {embed:,} ({embed/1e6:.2f}M)")
        print(f"Routers:         {routers:,} ({routers/1e3:.1f}K)")
        print(f"Expand Q/K/V/O:  {expand_qkvo:,} ({expand_qkvo/1e6:.2f}M)")
        print(f"LayerNorms:      {norms:,} ({norms/1e3:.1f}K)")
        print(f"---")
        print(f"Architecture: compressor_Q/K/V + expand_Q/K/V/O")
        print(f"Attention in d_model space (d_head={self.d_model // self.n_heads})")
        print(f"---")
        print(f"Total:           {self.count_parameters():,} ({self.count_parameters()/1e6:.2f}M)")

        return {
            'compress': compress,
            'expand': expand,
            'knowledge': knowledge,
            'embeddings': embed,
            'routers': routers,
            'expand_qkvo': expand_qkvo,
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
        }
