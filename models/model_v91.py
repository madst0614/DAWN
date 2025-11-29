"""
DAWN v9.1 - Dynamic Architecture With Neurons

v9.0 → v9.1 변경:
1. Compress/Expand: soft weighted sum (v8 스타일, 배치 병렬화 가능)
   - weights = softmax(scores)
   - all_proj = einsum(x, neurons)
   - output = (all_proj * weights).sum()

2. Reflection: hard → gated
   - v9.0: x = apply_reflection(x, v)  # 100% 적용
   - v9.1: gate = sigmoid(score), x = gate * reflected + (1 - gate) * x

3. ReflectionNeurons: separate pools
   - reflect_d: [n_reflect_d, d_model]
   - reflect_r: [n_reflect_r, rank]

구조:
    SharedNeurons
    ├── CompressNeurons: [n_compress, d_model, rank]   # soft weighted sum
    ├── ExpandNeurons: [n_expand, rank, d_model]       # soft weighted sum
    │
    ├── ReflectionNeurons (gated application, separate pools)
    │   ├── reflect_d: [n_reflect_d, d_model]   # d_model space reflections
    │   └── reflect_r: [n_reflect_r, rank]       # rank space reflections
    │
    └── KnowledgeNeurons
        ├── knowledge_K: [n_knowledge, rank]
        └── knowledge_V: [n_knowledge, d_model]

변환 흐름:
    [Compressor: d_model → rank]
    x [d_model]
    → reflect_d gated 적용 (관점 변환, 부분 적용)
    → compress_neuron soft 선택 (weighted sum)
    → output [rank]

    [Expander: rank → d_model]
    x [rank]
    → reflect_r gated 적용
    → expand_neuron soft 선택 (weighted sum)
    → reflect_d gated 적용
    → output [d_model]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SharedNeurons(nn.Module):
    """
    공유 뉴런 풀 - 모든 레이어에서 참조

    v9.1:
    - CompressNeurons/ExpandNeurons: soft weighted sum (배치 병렬화)
    - ReflectionNeurons: gated application (sigmoid gate), separate pools
    - KnowledgeNeurons: knowledge_K/V
    """
    def __init__(
        self,
        d_model: int,
        rank: int,
        n_compress: int,
        n_expand: int,
        n_reflect_d: int,
        n_reflect_r: int,
        n_knowledge: int,
        # Legacy params (ignored)
        n_process: int = None,
        n_input: int = None,
        n_output: int = None,
        n_reflect: int = None,  # Legacy: use n_reflect_d/n_reflect_r instead
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.n_compress = n_compress
        self.n_expand = n_expand
        self.n_reflect_d = n_reflect_d
        self.n_reflect_r = n_reflect_r
        self.n_knowledge = n_knowledge

        # CompressNeurons (다중 압축 행렬)
        self.compress_neurons = nn.Parameter(torch.zeros(n_compress, d_model, rank))

        # ExpandNeurons (다중 확장 행렬)
        self.expand_neurons = nn.Parameter(torch.zeros(n_expand, rank, d_model))

        # ReflectionNeurons (분리된 풀)
        self.reflect_d = nn.Parameter(torch.zeros(n_reflect_d, d_model))
        self.reflect_r = nn.Parameter(torch.zeros(n_reflect_r, rank))

        # KnowledgeNeurons
        self.knowledge_K = nn.Parameter(torch.zeros(n_knowledge, rank))
        self.knowledge_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_weights()

    def _init_weights(self):
        """뉴런 초기화"""
        # compress_neurons: 각각 직교 초기화
        for i in range(self.n_compress):
            if self.d_model >= self.rank:
                q, _ = torch.linalg.qr(torch.randn(self.d_model, self.rank))
                self.compress_neurons.data[i] = q
            else:
                q, _ = torch.linalg.qr(torch.randn(self.rank, self.d_model))
                self.compress_neurons.data[i] = q.T

        # expand_neurons: 각각 직교 초기화
        for i in range(self.n_expand):
            if self.rank >= self.d_model:
                q, _ = torch.linalg.qr(torch.randn(self.rank, self.d_model))
                self.expand_neurons.data[i] = q
            else:
                q, _ = torch.linalg.qr(torch.randn(self.d_model, self.rank))
                self.expand_neurons.data[i] = q.T

        # reflect_d: 단위 벡터 초기화
        for i in range(self.n_reflect_d):
            v = torch.randn(self.d_model)
            self.reflect_d.data[i] = v / (v.norm() + 1e-8)

        # reflect_r: 단위 벡터 초기화
        for i in range(self.n_reflect_r):
            v = torch.randn(self.rank)
            self.reflect_r.data[i] = v / (v.norm() + 1e-8)

        # knowledge_K: 정규분포 + 정규화
        nn.init.normal_(self.knowledge_K, std=0.02)
        with torch.no_grad():
            self.knowledge_K.data = F.normalize(self.knowledge_K.data, dim=-1)

        # knowledge_V: 정규분포
        nn.init.normal_(self.knowledge_V, std=0.02)

    def apply_reflection(self, x, v):
        """
        Householder 반사: x - 2 * v * (v.T @ x) / ||v||²

        Args:
            x: [B, S, dim]
            v: [B, S, dim]

        Returns:
            reflected x: [B, S, dim]
        """
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + 1e-8
        v_normalized = v / v_norm_sq.sqrt()
        vTx = (x * v_normalized).sum(dim=-1, keepdim=True)
        return x - 2 * v_normalized * vTx

    def apply_gated_reflection(self, x, v, gate):
        """
        Gated Householder 반사: gate * reflected + (1 - gate) * x

        Args:
            x: [B, S, dim]
            v: [B, S, dim]
            gate: [B, S, 1] sigmoid gate (0~1)

        Returns:
            gated reflected x: [B, S, dim]
        """
        reflected = self.apply_reflection(x, v)
        return gate * reflected + (1 - gate) * x


class Compressor(nn.Module):
    """
    d_model → rank 압축 (v9.1)

    1. reflect_d gated 적용 (관점 변환, 부분 적용)
    2. compress_neuron soft 선택 (weighted sum, 배치 병렬화)
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_compress: int,
        n_reflect_d: int,
        reflect_k: int,
        neuron_type: str = 'q',
        # Legacy params
        n_process: int = None,
        process_k: int = None,
        n_input: int = None,
        n_reflect: int = None,  # Legacy
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.n_compress = n_compress
        self.n_reflect_d = n_reflect_d
        self.reflect_k = reflect_k
        self.neuron_type = neuron_type

        # 독립 라우터 (타입별/레이어별)
        self.router_d = nn.Linear(d_model, n_reflect_d, bias=False)
        self.router_compress = nn.Linear(d_model, n_compress, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, S, d_model]

        Returns:
            output: [B, S, rank]
            routing_info: dict
        """
        B, S, D = x.shape
        sn = self.shared_neurons

        # 1. d_model 공간 반사 (gated, top-k)
        scores_d = self.router_d(x)  # [B, S, n_reflect]
        topk_scores, indices_d = torch.topk(scores_d, self.reflect_k, dim=-1)  # [B, S, k]

        for i in range(self.reflect_k):
            idx = indices_d[:, :, i]  # [B, S]
            v = sn.reflect_d[idx]  # [B, S, d_model]
            gate = torch.sigmoid(topk_scores[:, :, i:i+1])  # [B, S, 1]
            x = sn.apply_gated_reflection(x, v, gate)

        # 2. compress_neuron soft 선택 (weighted sum)
        scores_c = self.router_compress(x)  # [B, S, n_compress]
        weights_c = F.softmax(scores_c, dim=-1)  # [B, S, n_compress]

        # Soft selection: weighted sum of all projections
        # all_proj: [B, S, n_compress, rank]
        all_proj = torch.einsum('bsd,ndr->bsnr', x, sn.compress_neurons)
        # output: [B, S, rank]
        x = (all_proj * weights_c.unsqueeze(-1)).sum(dim=2)

        routing_info = {
            'indices_d': indices_d,
            'gates_d': torch.sigmoid(topk_scores),
            'weights_compress': weights_c,
        }

        return x, routing_info


class Expander(nn.Module):
    """
    rank → d_model 확장 (v9.1)

    1. reflect_r gated 적용 (잠재 공간 변환)
    2. expand_neuron soft 선택 (weighted sum, 배치 병렬화)
    3. reflect_d gated 적용 (출력 관점 변환)
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_expand: int,
        n_reflect_d: int,
        n_reflect_r: int,
        reflect_k: int,
        neuron_type: str = 'o',
        # Legacy params
        n_process: int = None,
        process_k: int = None,
        n_output: int = None,
        n_reflect: int = None,  # Legacy
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.n_expand = n_expand
        self.n_reflect_d = n_reflect_d
        self.n_reflect_r = n_reflect_r
        self.reflect_k = reflect_k
        self.neuron_type = neuron_type

        # 독립 라우터
        self.router_r = nn.Linear(rank, n_reflect_r, bias=False)
        self.router_expand = nn.Linear(rank, n_expand, bias=False)
        self.router_d = nn.Linear(d_model, n_reflect_d, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, S, rank]

        Returns:
            output: [B, S, d_model]
            routing_info: dict
        """
        B, S, R = x.shape
        sn = self.shared_neurons

        # 1. rank 공간 반사 (gated, top-k)
        scores_r = self.router_r(x)  # [B, S, n_reflect]
        topk_scores_r, indices_r = torch.topk(scores_r, self.reflect_k, dim=-1)  # [B, S, k]

        for i in range(self.reflect_k):
            idx = indices_r[:, :, i]  # [B, S]
            v = sn.reflect_r[idx]  # [B, S, rank]
            gate = torch.sigmoid(topk_scores_r[:, :, i:i+1])  # [B, S, 1]
            x = sn.apply_gated_reflection(x, v, gate)

        # 2. expand_neuron soft 선택 (weighted sum)
        scores_e = self.router_expand(x)  # [B, S, n_expand]
        weights_e = F.softmax(scores_e, dim=-1)  # [B, S, n_expand]

        # Soft selection: weighted sum of all projections
        # all_proj: [B, S, n_expand, d_model]
        all_proj = torch.einsum('bsr,nrd->bsnd', x, sn.expand_neurons)
        # output: [B, S, d_model]
        x = (all_proj * weights_e.unsqueeze(-1)).sum(dim=2)

        # 3. d_model 공간 반사 (gated, top-k)
        scores_d = self.router_d(x)  # [B, S, n_reflect]
        topk_scores_d, indices_d = torch.topk(scores_d, self.reflect_k, dim=-1)  # [B, S, k]

        for i in range(self.reflect_k):
            idx = indices_d[:, :, i]  # [B, S]
            v = sn.reflect_d[idx]  # [B, S, d_model]
            gate = torch.sigmoid(topk_scores_d[:, :, i:i+1])  # [B, S, 1]
            x = sn.apply_gated_reflection(x, v, gate)

        routing_info = {
            'indices_r': indices_r,
            'gates_r': torch.sigmoid(topk_scores_r),
            'weights_expand': weights_e,
            'indices_d': indices_d,
            'gates_d': torch.sigmoid(topk_scores_d),
        }

        return x, routing_info


class NeuronAttention(nn.Module):
    """
    Attention 대체 - Compressor/Expander 기반 (v9.1)
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_compress: int,
        n_expand: int,
        n_reflect_d: int,
        n_reflect_r: int,
        reflect_k: int,
        dropout: float = 0.1,
        # Legacy params
        n_process: int = None,
        process_k: int = None,
        n_input: int = None,
        n_output: int = None,
        n_reflect: int = None,  # Legacy
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = rank // n_heads
        self.rank = rank

        # Q/K/V/O - 같은 풀 공유, 라우터만 독립
        self.compressor_Q = Compressor(shared_neurons, d_model, rank, n_compress, n_reflect_d, reflect_k, neuron_type='q')
        self.compressor_K = Compressor(shared_neurons, d_model, rank, n_compress, n_reflect_d, reflect_k, neuron_type='k')
        self.compressor_V = Compressor(shared_neurons, d_model, rank, n_compress, n_reflect_d, reflect_k, neuron_type='v')
        self.expander_O = Expander(shared_neurons, d_model, rank, n_expand, n_reflect_d, n_reflect_r, reflect_k, neuron_type='o')

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, D = x.shape

        # Compress to Q, K, V
        Q, routing_Q = self.compressor_Q(x)
        K, routing_K = self.compressor_K(x)
        V, routing_V = self.compressor_V(x)

        # Multi-head reshape
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # Attention
        attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = attn_weights @ V

        # Concat heads
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.rank)

        # Expand back
        out, routing_O = self.expander_O(attn_out)

        routing_info = {
            'routing_Q': routing_Q,
            'routing_K': routing_K,
            'routing_V': routing_V,
            'routing_O': routing_O,
            'neuron_indices': routing_Q['indices_d'],
        }

        return out, routing_info


class NeuronMemory(nn.Module):
    """
    FFN 대체 - 내적 기반 지식 조회 (v9.1)
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_compress: int,
        n_reflect_d: int,
        reflect_k: int,
        knowledge_k: int = 8,
        # Legacy params
        n_process: int = None,
        process_k: int = None,
        n_input: int = None,
        n_reflect: int = None,  # Legacy
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.knowledge_k = knowledge_k

        self.query_compressor = Compressor(
            shared_neurons, d_model, rank, n_compress, n_reflect_d, reflect_k, neuron_type='m'
        )

    def forward(self, x):
        B, S, D = x.shape

        Q, query_routing = self.query_compressor(x)

        K = self.shared_neurons.knowledge_K
        V = self.shared_neurons.knowledge_V

        scores = Q @ K.T / math.sqrt(self.rank)
        topk_scores, topk_idx = torch.topk(scores, self.knowledge_k, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        V_selected = V[topk_idx]
        out = (V_selected * weights.unsqueeze(-1)).sum(dim=2)

        routing_info = {
            'knowledge_indices': topk_idx,
            'knowledge_weights': weights,
            'query_routing': query_routing,
        }

        return out, routing_info


class NeuronCircuit(nn.Module):
    """
    Transformer Block - NeuronAttention + NeuronMemory (v9.1)
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_compress: int,
        n_expand: int,
        n_reflect_d: int,
        n_reflect_r: int,
        reflect_k: int,
        knowledge_k: int,
        dropout: float = 0.1,
        # Legacy params
        n_process: int = None,
        process_k: int = None,
        n_input: int = None,
        n_output: int = None,
        n_reflect: int = None,  # Legacy
    ):
        super().__init__()

        self.attention = NeuronAttention(
            shared_neurons=shared_neurons,
            d_model=d_model,
            n_heads=n_heads,
            rank=rank,
            n_compress=n_compress,
            n_expand=n_expand,
            n_reflect_d=n_reflect_d,
            n_reflect_r=n_reflect_r,
            reflect_k=reflect_k,
            dropout=dropout,
        )

        self.memory = NeuronMemory(
            shared_neurons=shared_neurons,
            d_model=d_model,
            rank=rank,
            n_compress=n_compress,
            n_reflect_d=n_reflect_d,
            reflect_k=reflect_k,
            knowledge_k=knowledge_k,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_routing = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        mem_out, mem_routing = self.memory(self.norm2(x))
        x = x + self.dropout(mem_out)

        routing_info = {
            'attention': attn_routing,
            'memory': mem_routing,
            'neuron_indices': attn_routing['neuron_indices'],
        }

        return x, routing_info


class DAWN(nn.Module):
    """
    DAWN v9.1 - Dynamic Architecture With Neurons

    v9.1 특징:
    - Compress/Expand: soft weighted sum (v8 스타일, 배치 병렬화 가능)
    - Reflection: gated (sigmoid gate)
    - ReflectionNeurons: separate pools (n_reflect_d, n_reflect_r)
    """
    __version__ = "9.1"

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        rank: int = 64,
        max_seq_len: int = 128,
        n_compress: int = 4,
        n_expand: int = 4,
        n_reflect_d: int = 64,
        n_reflect_r: int = 64,
        reflect_k: int = 3,
        n_knowledge: int = 64,
        knowledge_k: int = 8,
        dropout: float = 0.1,
        # Legacy params (mapped)
        n_process: int = None,
        process_k: int = None,
        n_input: int = None,
        n_output: int = None,
        n_reflect: int = None,  # Legacy: use n_reflect_d/n_reflect_r
        **kwargs
    ):
        super().__init__()

        # Legacy param mapping
        if n_reflect is not None:
            # Legacy: single n_reflect -> split equally
            if n_reflect_d == 64 and n_reflect_r == 64:
                n_reflect_d = n_reflect
                n_reflect_r = n_reflect
        if reflect_k is None and process_k is not None:
            reflect_k = process_k

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.rank = rank
        self.max_seq_len = max_seq_len

        self.n_compress = n_compress
        self.n_expand = n_expand
        self.n_reflect_d = n_reflect_d
        self.n_reflect_r = n_reflect_r
        self.reflect_k = reflect_k
        self.n_knowledge = n_knowledge
        self.knowledge_k = knowledge_k

        # Legacy compatibility
        self.n_reflect = n_reflect_d  # For backward compat
        self.n_neurons = n_reflect_d
        self.n_process = n_reflect_d
        self.process_k = reflect_k
        self.basis_rank = rank

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.shared_neurons = SharedNeurons(
            d_model=d_model,
            rank=rank,
            n_compress=n_compress,
            n_expand=n_expand,
            n_reflect_d=n_reflect_d,
            n_reflect_r=n_reflect_r,
            n_knowledge=n_knowledge,
        )

        self.layers = nn.ModuleList([
            NeuronCircuit(
                shared_neurons=self.shared_neurons,
                d_model=d_model,
                n_heads=n_heads,
                rank=rank,
                n_compress=n_compress,
                n_expand=n_expand,
                n_reflect_d=n_reflect_d,
                n_reflect_r=n_reflect_r,
                reflect_k=reflect_k,
                knowledge_k=knowledge_k,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        )

        self.dropout = nn.Dropout(dropout)
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
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None, return_routing_info=False, return_losses=False):
        B, S = input_ids.shape
        device = input_ids.device

        pos = torch.arange(S, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        mask = self.causal_mask[:, :, :S, :S]

        routing_infos = []
        for layer in self.layers:
            x, routing_info = layer(x, mask)
            if return_routing_info:
                routing_infos.append(routing_info)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

        if return_losses:
            aux_losses = self.get_auxiliary_losses()
            if return_routing_info:
                return (logits, aux_losses, routing_infos)
            return (logits, aux_losses)

        if return_routing_info:
            if labels is not None:
                return (loss, logits, routing_infos)
            return (logits, routing_infos)

        return (loss, logits) if labels is not None else logits

    def orthogonality_loss(self):
        """Compress/ExpandNeurons 직교성 유지"""
        loss = 0.0

        # compress_neurons: 각 [d_model, rank] -> W.T @ W ≈ I
        for i in range(self.n_compress):
            W = self.shared_neurons.compress_neurons[i]
            WtW = W.T @ W
            I = torch.eye(self.rank, device=W.device)
            loss += ((WtW - I) ** 2).mean()

        # expand_neurons: 각 [rank, d_model] -> W @ W.T ≈ I
        for i in range(self.n_expand):
            W = self.shared_neurons.expand_neurons[i]
            WWt = W @ W.T
            I = torch.eye(self.rank, device=W.device)
            loss += ((WWt - I) ** 2).mean()

        return loss / (self.n_compress + self.n_expand)

    def reflection_norm_loss(self):
        """ReflectionNeurons ||v|| ≈ 1 유지"""
        loss = 0.0

        # reflect_d
        norms = self.shared_neurons.reflect_d.norm(dim=-1)
        loss += ((norms - 1.0) ** 2).mean()

        # reflect_r
        norms = self.shared_neurons.reflect_r.norm(dim=-1)
        loss += ((norms - 1.0) ** 2).mean()

        return loss / 2

    def knowledge_diversity_loss(self):
        K = self.shared_neurons.knowledge_K
        K_norm = F.normalize(K, dim=-1)
        sim = K_norm @ K_norm.T
        mask = ~torch.eye(self.n_knowledge, dtype=torch.bool, device=K.device)
        return sim[mask].abs().mean()

    def get_auxiliary_losses(self):
        return {
            'orth_total': self.orthogonality_loss(),
            'process_norm': self.reflection_norm_loss(),
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
            'n_reflect_d': self.n_reflect_d,
            'n_reflect_r': self.n_reflect_r,
            'reflect_k': self.reflect_k,
            'n_knowledge': self.n_knowledge,
            'knowledge_k': self.knowledge_k,
        }
