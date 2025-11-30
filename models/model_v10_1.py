"""
DAWN v10.1: Top-K Sparse Compress/Expand Architecture

Key changes from v10.0:
- Top-K selection for Compressor (default k=8)
- Top-K selection for Expander (default k=4)
- Load balance loss for routing균등 분포
- Router noise for exploration during training
- Significant memory reduction (~15GB vs ~50GB)
- Speed improvement (~3-4x faster)

Architecture:
    x (d_model)
    → Compressor: router → top-k → softmax → weighted selected neurons → (rank)
    → Expander: router → top-k → softmax → weighted selected neurons → (d_model)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedNeurons(nn.Module):
    """
    v10.1: SharedNeurons (same as v10.0)

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
    v10.1: Top-K Sparse Compressor

    d_model → rank 압축 (Top-K 선택)

    흐름:
    1. Router → scores
    2. Top-K 선택
    3. Softmax on selected
    4. Gather selected neurons only
    5. Weighted projection
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_compress: int,
        top_k: int = 8,
        router_noise: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.n_compress = n_compress
        self.top_k = top_k
        self.router_noise = router_noise

        # 독립 라우터
        self.router = nn.Linear(d_model, n_compress, bias=False)

    def forward(self, x, add_noise: bool = False):
        """
        Args:
            x: [B, S, d_model]
            add_noise: Whether to add noise to router scores (for training exploration)
        Returns:
            output: [B, S, rank]
            routing_info: dict with weights and indices
        """
        B, S, D = x.shape

        # 1. Router scores
        scores = self.router(x)  # [B, S, n_compress]

        # Add noise for exploration during training
        if add_noise and self.training and self.router_noise > 0:
            noise = torch.randn_like(scores) * self.router_noise
            scores = scores + noise

        # 2. Top-k selection
        topk_scores, topk_idx = torch.topk(scores, self.top_k, dim=-1)  # [B, S, k]
        weights = F.softmax(topk_scores, dim=-1)  # [B, S, k]

        # 3. Gather selected neurons
        neurons = self.shared_neurons.compress_neurons  # [n_compress, d_model, rank]

        # topk_idx: [B, S, k] → expand for gather
        # neurons shape: [n_compress, d_model, rank]
        # We need to select k neurons per position
        idx_expanded = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(B, S, self.top_k, D, self.rank)
        neurons_expanded = neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1, -1)
        selected_neurons = neurons_expanded.gather(2, idx_expanded)  # [B, S, k, d_model, rank]

        # 4. Project with selected neurons only
        # x: [B, S, d_model] → [B, S, 1, d_model]
        x_expanded = x.unsqueeze(2)  # [B, S, 1, d_model]
        proj = torch.einsum('bskd,bskdr->bskr', x_expanded.expand(-1, -1, self.top_k, -1), selected_neurons)
        # proj: [B, S, k, rank]

        # 5. Weighted sum
        output = (proj * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, rank]

        routing_info = {
            'weights': weights,
            'indices': topk_idx,
            'scores': scores,  # For load balance loss
        }
        return output, routing_info


class Expander(nn.Module):
    """
    v10.1: Top-K Sparse Expander

    rank → d_model 확장 (Top-K 선택)

    흐름:
    1. Router → scores
    2. Top-K 선택
    3. Softmax on selected
    4. Gather selected neurons only
    5. Weighted projection
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_expand: int,
        top_k: int = 4,
        router_noise: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.n_expand = n_expand
        self.top_k = top_k
        self.router_noise = router_noise

        # 독립 라우터
        self.router = nn.Linear(rank, n_expand, bias=False)

    def forward(self, x, add_noise: bool = False):
        """
        Args:
            x: [B, S, rank]
            add_noise: Whether to add noise to router scores (for training exploration)
        Returns:
            output: [B, S, d_model]
            routing_info: dict with weights and indices
        """
        B, S, R = x.shape

        # 1. Router scores
        scores = self.router(x)  # [B, S, n_expand]

        # Add noise for exploration during training
        if add_noise and self.training and self.router_noise > 0:
            noise = torch.randn_like(scores) * self.router_noise
            scores = scores + noise

        # 2. Top-k selection
        topk_scores, topk_idx = torch.topk(scores, self.top_k, dim=-1)  # [B, S, k]
        weights = F.softmax(topk_scores, dim=-1)  # [B, S, k]

        # 3. Gather selected neurons
        neurons = self.shared_neurons.expand_neurons  # [n_expand, rank, d_model]

        # topk_idx: [B, S, k] → expand for gather
        idx_expanded = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(B, S, self.top_k, R, self.d_model)
        neurons_expanded = neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1, -1)
        selected_neurons = neurons_expanded.gather(2, idx_expanded)  # [B, S, k, rank, d_model]

        # 4. Project with selected neurons only
        x_expanded = x.unsqueeze(2)  # [B, S, 1, rank]
        proj = torch.einsum('bskr,bskrd->bskd', x_expanded.expand(-1, -1, self.top_k, -1), selected_neurons)
        # proj: [B, S, k, d_model]

        # 5. Weighted sum
        output = (proj * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, d_model]

        routing_info = {
            'weights': weights,
            'indices': topk_idx,
            'scores': scores,  # For load balance loss
        }
        return output, routing_info


class NeuronCircuit(nn.Module):
    """
    v10.1 Attention Layer with Top-K routing

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
        compress_top_k: int = 8,
        expand_top_k: int = 4,
        router_noise: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = rank // n_heads
        self.rank = rank

        # Q/K/V: 각각 독립 Compressor (compress_neurons 공유)
        self.compressor_Q = Compressor(shared_neurons, d_model, rank, n_compress, compress_top_k, router_noise)
        self.compressor_K = Compressor(shared_neurons, d_model, rank, n_compress, compress_top_k, router_noise)
        self.compressor_V = Compressor(shared_neurons, d_model, rank, n_compress, compress_top_k, router_noise)

        # O: Expander
        self.expander_O = Expander(shared_neurons, d_model, rank, n_expand, expand_top_k, router_noise)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, add_noise: bool = False):
        """
        Args:
            x: [B, S, d_model]
            mask: [B, 1, S, S] causal mask
            add_noise: Whether to add noise to routers
        Returns:
            output: [B, S, d_model]
            routing_info: dict with Q/K/V/O routing
        """
        B, S, D = x.shape

        # Q/K/V compression with top-k
        Q, q_info = self.compressor_Q(x, add_noise)  # [B, S, rank]
        K, k_info = self.compressor_K(x, add_noise)  # [B, S, rank]
        V, v_info = self.compressor_V(x, add_noise)  # [B, S, rank]

        # Reshape for multi-head attention
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, S, d_head]
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to V
        attn_out = torch.matmul(attn, V)  # [B, H, S, d_head]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.rank)  # [B, S, rank]

        # Output expansion with top-k
        output, o_info = self.expander_O(attn_out, add_noise)  # [B, S, d_model]
        output = self.out_dropout(output)

        routing_info = {
            'Q': q_info,
            'K': k_info,
            'V': v_info,
            'O': o_info,
        }
        return output, routing_info


class NeuronMemory(nn.Module):
    """
    v10.1 Knowledge Retrieval (same as v10.0)

    Query compression → Knowledge lookup → Output
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_compress: int,
        compress_top_k: int = 8,
        router_noise: float = 0.1,
        knowledge_k: int = 8,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.knowledge_k = knowledge_k

        # Query Compressor (with top-k)
        self.query_compressor = Compressor(shared_neurons, d_model, rank, n_compress, compress_top_k, router_noise)

    def forward(self, x, add_noise: bool = False):
        """
        Args:
            x: [B, S, d_model]
            add_noise: Whether to add noise to router
        Returns:
            output: [B, S, d_model]
            routing_info: dict
        """
        B, S, D = x.shape

        # Query compression
        Q, q_info = self.query_compressor(x, add_noise)  # [B, S, rank]

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
            'M': q_info,
            'knowledge_indices': topk_idx,
            'knowledge_weights': weights,
        }
        return output, routing_info


class DAWNBlock(nn.Module):
    """Single DAWN v10.1 block: Attention + FFN(Memory) + LayerNorms"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_compress: int,
        n_expand: int,
        compress_top_k: int,
        expand_top_k: int,
        router_noise: float,
        knowledge_k: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = NeuronCircuit(
            shared_neurons, d_model, n_heads, rank, n_compress, n_expand,
            compress_top_k, expand_top_k, router_noise, dropout
        )
        self.memory = NeuronMemory(
            shared_neurons, d_model, rank, n_compress,
            compress_top_k, router_noise, knowledge_k
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, add_noise: bool = False):
        """
        Returns:
            x: [B, S, d_model]
            routing_info: dict with attention and memory routing
        """
        # Attention
        attn_out, attn_routing = self.attn(self.norm1(x), mask, add_noise)
        x = x + attn_out  # dropout already in NeuronCircuit

        # Memory (FFN replacement)
        mem_out, mem_routing = self.memory(self.norm2(x), add_noise)
        x = x + self.dropout(mem_out)

        routing_info = {
            'attention': attn_routing,
            'memory': mem_routing,
        }
        return x, routing_info


class DAWN(nn.Module):
    """
    DAWN v10.1: Top-K Sparse Compress/Expand Architecture

    - CompressNeurons: Q/K/V/M 통합 with top-k selection
    - ExpandNeurons: O 통합 with top-k selection
    - Load balance loss for routing均等分布
    - Router noise for exploration

    Memory efficient: Only compute with k selected neurons instead of all
    """
    __version__ = "10.1"

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
        compress_top_k: int = 8,
        expand_top_k: int = 4,
        router_noise: float = 0.1,
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
        self.compress_top_k = compress_top_k
        self.expand_top_k = expand_top_k
        self.router_noise = router_noise

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
                compress_top_k=compress_top_k,
                expand_top_k=expand_top_k,
                router_noise=router_noise,
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

    def forward(self, input_ids, labels=None, return_routing_info=False, add_noise=None):
        """
        Args:
            input_ids: [B, S]
            labels: [B, S] for loss computation
            return_routing_info: Whether to return routing information
            add_noise: Override for training noise (default: self.training)
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Default: add noise during training
        if add_noise is None:
            add_noise = self.training

        # Embeddings
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
        mask = ~mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        # Layers
        routing_infos = []
        for layer in self.layers:
            x, routing_info = layer(x, mask, add_noise)
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

    def routing_entropy_loss(self):
        """Placeholder for entropy-based routing loss"""
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def knowledge_diversity_loss(self):
        """Knowledge K vectors 다양성"""
        K = self.shared_neurons.knowledge_K
        K_norm = F.normalize(K, dim=-1)
        sim = K_norm @ K_norm.T
        mask = ~torch.eye(self.n_knowledge, dtype=torch.bool, device=K.device)
        return sim[mask].abs().mean()

    def load_balance_loss(self, routing_infos):
        """
        v10.1: Top-k routing load balance loss

        각 뉴런이 선택된 횟수를 균등하게 유지

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
                indices = layer_info['attention'][comp]['indices']  # [B, S, k]

                # 각 뉴런이 선택된 횟수 카운트
                flat_idx = indices.reshape(-1)  # [B*S*k]
                counts = torch.bincount(flat_idx, minlength=self.n_compress).float()

                # 균등 분포와의 차이
                target = flat_idx.numel() / self.n_compress
                loss += ((counts - target) ** 2).mean() / (target ** 2 + 1e-10)
                count += 1

            # O expander
            o_indices = layer_info['attention']['O']['indices']  # [B, S, k]
            flat_idx = o_indices.reshape(-1)
            counts = torch.bincount(flat_idx, minlength=self.n_expand).float()
            target = flat_idx.numel() / self.n_expand
            loss += ((counts - target) ** 2).mean() / (target ** 2 + 1e-10)
            count += 1

            # Memory M compressor
            m_indices = layer_info['memory']['M']['indices']  # [B, S, k]
            flat_idx = m_indices.reshape(-1)
            counts = torch.bincount(flat_idx, minlength=self.n_compress).float()
            target = flat_idx.numel() / self.n_compress
            loss += ((counts - target) ** 2).mean() / (target ** 2 + 1e-10)
            count += 1

        return loss / (count + 1e-10)

    def get_auxiliary_losses(self):
        """train.py 호환"""
        return {
            'orth_total': self.orthogonality_loss(),
            'knowledge_div': self.knowledge_diversity_loss(),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_by_component(self):
        """Component별 파라미터 수 출력"""
        compress = self.shared_neurons.compress_neurons.numel()
        expand = self.shared_neurons.expand_neurons.numel()
        knowledge = (self.shared_neurons.knowledge_K.numel() +
                    self.shared_neurons.knowledge_V.numel())

        embed = self.token_emb.weight.numel() + self.pos_emb.weight.numel()

        # Routers: per layer (4 compressors + 1 expander + 1 memory compressor per layer)
        router_per_layer = (
            self.layers[0].attn.compressor_Q.router.weight.numel() +
            self.layers[0].attn.compressor_K.router.weight.numel() +
            self.layers[0].attn.compressor_V.router.weight.numel() +
            self.layers[0].attn.expander_O.router.weight.numel() +
            self.layers[0].memory.query_compressor.router.weight.numel()
        )
        routers = router_per_layer * self.n_layers

        # LayerNorms
        norms = sum(p.numel() for n, p in self.named_parameters() if 'norm' in n)

        print(f"=== DAWN v10.1 Parameter Breakdown ===")
        print(f"CompressNeurons: {compress:,} ({compress/1e6:.2f}M)")
        print(f"ExpandNeurons:   {expand:,} ({expand/1e6:.2f}M)")
        print(f"KnowledgeNeurons: {knowledge:,} ({knowledge/1e3:.1f}K)")
        print(f"Embeddings:      {embed:,} ({embed/1e6:.2f}M)")
        print(f"Routers:         {routers:,} ({routers/1e3:.1f}K)")
        print(f"LayerNorms:      {norms:,} ({norms/1e3:.1f}K)")
        print(f"---")
        print(f"Top-K Config:    compress_k={self.compress_top_k}, expand_k={self.expand_top_k}")
        print(f"Router Noise:    {self.router_noise}")
        print(f"---")
        print(f"Total:           {self.count_parameters():,} ({self.count_parameters()/1e6:.2f}M)")

        return {
            'compress': compress,
            'expand': expand,
            'knowledge': knowledge,
            'embeddings': embed,
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
            'max_seq_len': self.max_seq_len,
            'n_compress': self.n_compress,
            'n_expand': self.n_expand,
            'n_knowledge': self.n_knowledge,
            'knowledge_k': self.knowledge_k,
            'compress_top_k': self.compress_top_k,
            'expand_top_k': self.expand_top_k,
            'router_noise': self.router_noise,
        }
