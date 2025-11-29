"""
DAWN v10.0: Simplified Compress/Expand Architecture

Key changes from v8:
- CompressNeurons: Q/K/V/M 통합 (하나의 풀 공유)
- ExpandNeurons: O 통합
- ProcessNeurons/Householder 완전 제거
- 라우터만 독립 (Compressor/Expander별)

Architecture:
    x (d_model)
    → Compressor: router → softmax → weighted compress_neurons → (rank)
    → Expander: router → softmax → weighted expand_neurons → (d_model)

Simple. No Householder. Just soft routing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedNeurons(nn.Module):
    """
    v10.0: 단순화된 SharedNeurons

    - CompressNeurons: [n_compress, d_model, rank] - Q/K/V/M 공유
    - ExpandNeurons: [n_expand, rank, d_model] - O 공유
    - KnowledgeNeurons: [n_knowledge, rank] + [n_knowledge, d_model]
    """
    def __init__(
        self,
        d_model: int,
        rank: int,
        n_compress: int,
        n_expand: int,
        n_knowledge: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.n_compress = n_compress
        self.n_expand = n_expand
        self.n_knowledge = n_knowledge

        # CompressNeurons: d_model → rank (Q/K/V/M 공유)
        self.compress_neurons = nn.Parameter(torch.zeros(n_compress, d_model, rank))

        # ExpandNeurons: rank → d_model (O 공유)
        self.expand_neurons = nn.Parameter(torch.zeros(n_expand, rank, d_model))

        # KnowledgeNeurons
        self.knowledge_K = nn.Parameter(torch.zeros(n_knowledge, rank))
        self.knowledge_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_parameters()

    def _init_parameters(self):
        # CompressNeurons: 직교 초기화
        for i in range(self.n_compress):
            nn.init.orthogonal_(self.compress_neurons.data[i])

        # ExpandNeurons: 직교 초기화
        for i in range(self.n_expand):
            nn.init.orthogonal_(self.expand_neurons.data[i])

        # KnowledgeNeurons
        nn.init.normal_(self.knowledge_K, std=0.02)
        nn.init.normal_(self.knowledge_V, std=0.02)


class Compressor(nn.Module):
    """
    d_model → rank 압축

    흐름:
    1. Router → softmax weights
    2. Weighted sum of compress_neurons projections
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

        # 독립 라우터
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

        # 2. Project with all neurons: [B, S, d_model] @ [n_compress, d_model, rank]
        neurons = self.shared_neurons.compress_neurons  # [n_compress, d_model, rank]
        all_proj = torch.einsum('bsd,ndr->bsnr', x, neurons)  # [B, S, n_compress, rank]

        # 3. Weighted sum
        output = (all_proj * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, rank]

        routing_info = {'weights': weights}
        return output, routing_info


class Expander(nn.Module):
    """
    rank → d_model 확장

    흐름:
    1. Router → softmax weights
    2. Weighted sum of expand_neurons projections
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

        # 독립 라우터
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

        # 2. Project with all neurons: [B, S, rank] @ [n_expand, rank, d_model]
        neurons = self.shared_neurons.expand_neurons  # [n_expand, rank, d_model]
        all_proj = torch.einsum('bsr,nrd->bsnd', x, neurons)  # [B, S, n_expand, d_model]

        # 3. Weighted sum
        output = (all_proj * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, d_model]

        routing_info = {'weights': weights}
        return output, routing_info


class NeuronCircuit(nn.Module):
    """
    v10.0 Attention Layer

    Q/K/V 각각 독립 Compressor (같은 compress_neurons 공유)
    O는 Expander
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
        self.d_head = rank // n_heads
        self.rank = rank

        # Q/K/V: 각각 독립 Compressor (compress_neurons 공유)
        self.compressor_Q = Compressor(shared_neurons, d_model, rank, n_compress)
        self.compressor_K = Compressor(shared_neurons, d_model, rank, n_compress)
        self.compressor_V = Compressor(shared_neurons, d_model, rank, n_compress)

        # O: Expander
        self.expander_O = Expander(shared_neurons, d_model, rank, n_expand)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, S, d_model]
            mask: [B, 1, S, S] causal mask
        Returns:
            output: [B, S, d_model]
        """
        B, S, D = x.shape

        # Q/K/V compression
        Q, q_info = self.compressor_Q(x)  # [B, S, rank]
        K, k_info = self.compressor_K(x)  # [B, S, rank]
        V, v_info = self.compressor_V(x)  # [B, S, rank]

        # Reshape for multi-head attention
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, S, d_head]
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to V
        attn_out = torch.matmul(attn, V)  # [B, H, S, d_head]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.rank)  # [B, S, rank]

        # Output expansion
        output, o_info = self.expander_O(attn_out)  # [B, S, d_model]

        return output


class NeuronMemory(nn.Module):
    """
    v10.0 Knowledge Retrieval

    Query compression → Knowledge lookup → Output
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_compress: int,
        knowledge_k: int = 8,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.knowledge_k = knowledge_k

        # Query Compressor
        self.query_compressor = Compressor(shared_neurons, d_model, rank, n_compress)

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

        # Knowledge lookup
        K = self.shared_neurons.knowledge_K  # [n_knowledge, rank]
        V = self.shared_neurons.knowledge_V  # [n_knowledge, d_model]

        scores = Q @ K.T / math.sqrt(self.rank)  # [B, S, n_knowledge]
        topk_scores, topk_idx = torch.topk(scores, self.knowledge_k, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)  # [B, S, k]

        # Gather selected V
        idx_expanded = topk_idx.unsqueeze(-1).expand(B, S, self.knowledge_k, self.d_model)
        V_expanded = V.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        selected_V = V_expanded.gather(2, idx_expanded)  # [B, S, k, d_model]

        # Weighted sum
        output = (selected_V * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, d_model]

        routing_info = {
            'query_routing': q_info,
            'knowledge_indices': topk_idx,
            'knowledge_weights': weights,
        }
        return output, routing_info


class DAWNBlock(nn.Module):
    """Single DAWN v10.0 block: Attention + FFN(Memory) + LayerNorms"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_compress: int,
        n_expand: int,
        knowledge_k: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = NeuronCircuit(
            shared_neurons, d_model, n_heads, rank, n_compress, n_expand, dropout
        )
        self.memory = NeuronMemory(
            shared_neurons, d_model, rank, n_compress, knowledge_k
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention
        x = x + self.dropout(self.attn(self.norm1(x), mask))

        # Memory (FFN replacement)
        mem_out, mem_info = self.memory(self.norm2(x))
        x = x + self.dropout(mem_out)

        return x


class DAWN(nn.Module):
    """
    DAWN v10.0: Simplified Compress/Expand Architecture

    - CompressNeurons: Q/K/V/M 통합
    - ExpandNeurons: O 통합
    - No Householder, just soft routing
    """
    __version__ = "10.0"

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
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.rank = rank
        self.max_seq_len = max_seq_len

        # Config 저장
        self.n_compress = n_compress
        self.n_expand = n_expand
        self.n_knowledge = n_knowledge
        self.knowledge_k = knowledge_k

        # train.py 호환용
        self.n_neurons = n_compress  # For load balance loss (if needed)
        self.basis_rank = rank  # For analysis scripts

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # SharedNeurons (전체 레이어 공유)
        self.shared_neurons = SharedNeurons(
            d_model=d_model,
            rank=rank,
            n_compress=n_compress,
            n_expand=n_expand,
            n_knowledge=n_knowledge,
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

    def forward(self, input_ids, labels=None):
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
        mask = ~mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        # Layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            return loss, logits

        return logits

    def orthogonality_loss(self):
        """CompressNeurons/ExpandNeurons 직교성 유지"""
        loss = 0.0

        # compress_neurons: 각 [d_model, rank] → W.T @ W ≈ I
        for i in range(self.n_compress):
            W = self.shared_neurons.compress_neurons[i]
            WtW = W.T @ W
            I = torch.eye(self.rank, device=W.device)
            loss += ((WtW - I) ** 2).mean()

        # expand_neurons: 각 [rank, d_model] → W @ W.T ≈ I
        for i in range(self.n_expand):
            W = self.shared_neurons.expand_neurons[i]
            WWt = W @ W.T
            I = torch.eye(self.rank, device=W.device)
            loss += ((WWt - I) ** 2).mean()

        return loss / (self.n_compress + self.n_expand)

    def knowledge_diversity_loss(self):
        """Knowledge K vectors 다양성"""
        K = self.shared_neurons.knowledge_K
        K_norm = F.normalize(K, dim=-1)
        sim = K_norm @ K_norm.T
        mask = ~torch.eye(self.n_knowledge, dtype=torch.bool, device=K.device)
        return sim[mask].abs().mean()

    def get_auxiliary_losses(self):
        """train.py 호환"""
        return {
            'orth_total': self.orthogonality_loss(),
            'knowledge_div': self.knowledge_diversity_loss(),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'rank': self.rank,
            'max_seq_len': self.max_seq_len,
            'n_compress': self.n_compress,
            'n_expand': self.n_expand,
            'n_knowledge': self.n_knowledge,
            'knowledge_k': self.knowledge_k,
        }
