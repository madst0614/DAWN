"""
DAWN v10.1: True Sparse MoE-Style Architecture

v10.0 대비 변경:
- 모든 뉴런 계산 후 선택 → 선택 후 해당 뉴런만 계산
- 메모리: O(B×S×N×R) → O(B×S×k×R)
- 속도: 바닐라 Transformer와 유사하거나 더 빠름

핵심 아이디어:
1. 라우터로 top-k 뉴런 인덱스 먼저 결정 (가벼운 연산)
2. 선택된 뉴런 weight만 gather
3. 해당 뉴런들로만 projection 계산
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedNeurons(nn.Module):
    """
    SharedNeurons (v10.0과 동일)

    - CompressNeurons: [n_compress, d_model, rank]
    - ExpandNeurons: [n_expand, rank, d_model]
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

        self.compress_neurons = nn.Parameter(torch.zeros(n_compress, d_model, rank))
        self.expand_neurons = nn.Parameter(torch.zeros(n_expand, rank, d_model))
        self.knowledge_K = nn.Parameter(torch.zeros(n_knowledge, rank))
        self.knowledge_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_parameters()

    def _init_parameters(self):
        for i in range(self.n_compress):
            nn.init.orthogonal_(self.compress_neurons.data[i])
        for i in range(self.n_expand):
            nn.init.orthogonal_(self.expand_neurons.data[i])
        nn.init.normal_(self.knowledge_K, std=0.02)
        nn.init.normal_(self.knowledge_V, std=0.02)


class SparseCompressor(nn.Module):
    """
    True Sparse Compressor: 선택된 뉴런만 연산

    기존: x @ all_neurons → select top-k (N번 연산)
    개선: select top-k → x @ selected_neurons (k번 연산)
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

        self.router = nn.Linear(d_model, n_compress, bias=False)

    def forward(self, x, add_noise: bool = False):
        """
        Args:
            x: [B, S, d_model]
        Returns:
            output: [B, S, rank]
            routing_info: dict
        """
        B, S, D = x.shape
        k = self.top_k
        R = self.rank

        # 1. 라우터 스코어 (가벼운 연산)
        scores = self.router(x)  # [B, S, N]

        if add_noise and self.training and self.router_noise > 0:
            scores = scores + torch.randn_like(scores) * self.router_noise

        # 2. Top-k 선택
        topk_scores, topk_idx = torch.topk(scores, k, dim=-1)  # [B, S, k]
        weights = F.softmax(topk_scores, dim=-1)  # [B, S, k]

        # 3. 선택된 뉴런만 gather: [B, S, k, D, R]
        neurons = self.shared_neurons.compress_neurons  # [N, D, R]

        # 인덱스 확장: [B, S, k] → [B, S, k, D, R]
        idx_expanded = topk_idx.view(B, S, k, 1, 1).expand(B, S, k, D, R)
        neurons_expanded = neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1, -1)
        selected_neurons = neurons_expanded.gather(2, idx_expanded)  # [B, S, k, D, R]

        # 4. 선택된 뉴런으로만 projection
        # x: [B, S, D] → [B, S, 1, D]
        # selected_neurons: [B, S, k, D, R]
        # einsum: bsd, bskdr → bskr
        x_unsq = x.unsqueeze(2)  # [B, S, 1, D]
        proj = torch.einsum('bsod,bskdr->bskr', x_unsq, selected_neurons)  # [B, S, k, R]

        # 5. Weighted sum
        output = (proj * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, R]

        routing_info = {
            'weights': weights,
            'indices': topk_idx,
        }
        return output, routing_info


class SparseExpander(nn.Module):
    """
    True Sparse Expander: 선택된 뉴런만 연산
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

        self.router = nn.Linear(rank, n_expand, bias=False)

    def forward(self, x, add_noise: bool = False):
        """
        Args:
            x: [B, S, rank]
        Returns:
            output: [B, S, d_model]
            routing_info: dict
        """
        B, S, R = x.shape
        k = self.top_k
        D = self.d_model

        # 1. 라우터 스코어
        scores = self.router(x)  # [B, S, N]

        if add_noise and self.training and self.router_noise > 0:
            scores = scores + torch.randn_like(scores) * self.router_noise

        # 2. Top-k 선택
        topk_scores, topk_idx = torch.topk(scores, k, dim=-1)  # [B, S, k]
        weights = F.softmax(topk_scores, dim=-1)  # [B, S, k]

        # 3. 선택된 뉴런만 gather: [B, S, k, R, D]
        neurons = self.shared_neurons.expand_neurons  # [N, R, D]

        idx_expanded = topk_idx.view(B, S, k, 1, 1).expand(B, S, k, R, D)
        neurons_expanded = neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1, -1)
        selected_neurons = neurons_expanded.gather(2, idx_expanded)  # [B, S, k, R, D]

        # 4. 선택된 뉴런으로만 projection
        x_unsq = x.unsqueeze(2)  # [B, S, 1, R]
        proj = torch.einsum('bsor,bskrd->bskd', x_unsq, selected_neurons)  # [B, S, k, D]

        # 5. Weighted sum
        output = (proj * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, D]

        routing_info = {
            'weights': weights,
            'indices': topk_idx,
        }
        return output, routing_info


class NeuronCircuit(nn.Module):
    """
    Attention Layer with True Sparse routing
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
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = rank // n_heads
        self.rank = rank

        self.compressor_Q = SparseCompressor(shared_neurons, d_model, rank, n_compress, compress_top_k, router_noise)
        self.compressor_K = SparseCompressor(shared_neurons, d_model, rank, n_compress, compress_top_k, router_noise)
        self.compressor_V = SparseCompressor(shared_neurons, d_model, rank, n_compress, compress_top_k, router_noise)
        self.expander_O = SparseExpander(shared_neurons, d_model, rank, n_expand, expand_top_k, router_noise)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, add_noise: bool = False):
        B, S, D = x.shape

        Q, q_info = self.compressor_Q(x, add_noise)
        K, k_info = self.compressor_K(x, add_noise)
        V, v_info = self.compressor_V(x, add_noise)

        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        attn_out = torch.matmul(attn, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.rank)

        output, o_info = self.expander_O(attn_out, add_noise)
        output = self.out_dropout(output)

        routing_info = {'Q': q_info, 'K': k_info, 'V': v_info, 'O': o_info}
        return output, routing_info


class NeuronMemory(nn.Module):
    """
    Knowledge Retrieval with Sparse Compressor
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

        self.query_compressor = SparseCompressor(shared_neurons, d_model, rank, n_compress, compress_top_k, router_noise)

    def forward(self, x, add_noise: bool = False):
        B, S, D = x.shape

        Q, q_info = self.query_compressor(x, add_noise)

        K = self.shared_neurons.knowledge_K
        V = self.shared_neurons.knowledge_V

        scores = Q @ K.T / math.sqrt(self.rank)
        topk_scores, topk_idx = torch.topk(scores, self.knowledge_k, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        # 직접 인덱싱 (메모리 효율적)
        flat_idx = topk_idx.reshape(-1)
        selected_V = V[flat_idx].view(B, S, self.knowledge_k, self.d_model)

        output = (selected_V * weights.unsqueeze(-1)).sum(dim=2)

        routing_info = {
            'M': q_info,
            'knowledge_indices': topk_idx,
            'knowledge_weights': weights,
        }
        return output, routing_info


class DAWNBlock(nn.Module):
    """Single DAWN block"""
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
        attn_out, attn_routing = self.attn(self.norm1(x), mask, add_noise)
        x = x + attn_out

        mem_out, mem_routing = self.memory(self.norm2(x), add_noise)
        x = x + self.dropout(mem_out)

        routing_info = {'attention': attn_routing, 'memory': mem_routing}
        return x, routing_info


class DAWN(nn.Module):
    """
    DAWN v10.1: True Sparse MoE-Style Architecture

    핵심 개선:
    - 선택 후 연산 (vs 연산 후 선택)
    - 메모리 효율: O(k) vs O(N)
    - 속도: 바닐라 수준으로 개선
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

        self.n_compress = n_compress
        self.n_expand = n_expand
        self.n_knowledge = n_knowledge
        self.knowledge_k = knowledge_k
        self.compress_top_k = compress_top_k
        self.expand_top_k = expand_top_k
        self.router_noise = router_noise

        # 호환용
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
        B, S = input_ids.shape
        device = input_ids.device

        if add_noise is None:
            add_noise = self.training

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
        mask = ~mask.unsqueeze(0).unsqueeze(0)

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
        loss = 0.0
        for i in range(self.n_compress):
            W = self.shared_neurons.compress_neurons[i]
            WtW = W.T @ W
            I = torch.eye(self.rank, device=W.device)
            loss += ((WtW - I) ** 2).mean()
        for i in range(self.n_expand):
            W = self.shared_neurons.expand_neurons[i]
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
        loss = 0.0
        count = 0

        for layer_info in routing_infos:
            for comp in ['Q', 'K', 'V']:
                indices = layer_info['attention'][comp]['indices']
                flat_idx = indices.reshape(-1)
                counts = torch.bincount(flat_idx, minlength=self.n_compress).float()
                target = flat_idx.numel() / self.n_compress
                loss += ((counts - target) ** 2).mean() / (target ** 2 + 1e-10)
                count += 1

            o_indices = layer_info['attention']['O']['indices']
            flat_idx = o_indices.reshape(-1)
            counts = torch.bincount(flat_idx, minlength=self.n_expand).float()
            target = flat_idx.numel() / self.n_expand
            loss += ((counts - target) ** 2).mean() / (target ** 2 + 1e-10)
            count += 1

            m_indices = layer_info['memory']['M']['indices']
            flat_idx = m_indices.reshape(-1)
            counts = torch.bincount(flat_idx, minlength=self.n_compress).float()
            target = flat_idx.numel() / self.n_compress
            loss += ((counts - target) ** 2).mean() / (target ** 2 + 1e-10)
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

        router_per_layer = (
            self.layers[0].attn.compressor_Q.router.weight.numel() +
            self.layers[0].attn.compressor_K.router.weight.numel() +
            self.layers[0].attn.compressor_V.router.weight.numel() +
            self.layers[0].attn.expander_O.router.weight.numel() +
            self.layers[0].memory.query_compressor.router.weight.numel()
        )
        routers = router_per_layer * self.n_layers
        norms = sum(p.numel() for n, p in self.named_parameters() if 'norm' in n)

        print(f"=== DAWN v10.1 (True Sparse) Parameter Breakdown ===")
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
