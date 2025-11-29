"""
DAWN v9.0 - Dynamic Architecture With Neurons

v8.1 → v9.0 주요 변경:
- Input/Output neurons 단순화: 4개(q/k/v/m) → 1개로 통합
- Compressor/Expander에서 라우터 제거, 단순 행렬곱 사용
- 모든 다양성은 Process Householder 조합에서 발생

구조:
    SharedNeurons (공유 뉴런 풀 - 레이어 간 공유)
    ├── TransformNeurons (변환용)
    │     ├── input_neuron:       [d_model, rank] — 공유 압축 행렬
    │     ├── process_neurons_q:  [n_process, rank] — Q용
    │     ├── process_neurons_k:  [n_process, rank] — K용
    │     ├── process_neurons_v:  [n_process, rank] — V용
    │     ├── process_neurons_o:  [n_process, rank] — O용
    │     ├── process_neurons_m:  [n_process, rank] — Memory Query용
    │     └── output_neuron:      [rank, d_model] — 공유 확장 행렬
    │
    └── KnowledgeNeurons (지식용)
          ├── knowledge_K: [n_knowledge, rank]
          └── knowledge_V: [n_knowledge, d_model]

    NeuronCircuit (Transformer Block)
    ├── NeuronAttention (Attention 대체)
    │     ├── Compressor_Q (neuron_type='q')
    │     ├── Compressor_K (neuron_type='k')
    │     ├── Compressor_V (neuron_type='v')
    │     ├── Expander_O (neuron_type='o')
    │     └── Multi-Head Attention 연산
    │
    └── NeuronMemory (FFN 대체)
          ├── query_compressor (neuron_type='m')
          └── 내적 기반 지식 조회

파라미터 효율:
- v8.1 Input: n_input * d_model * rank * 4 = 8 * 256 * 64 * 4 = 524,288
- v9.0 Input: d_model * rank = 256 * 64 = 16,384
- v8.1 Output: n_output * rank * d_model = 8 * 64 * 256 = 131,072
- v9.0 Output: rank * d_model = 64 * 256 = 16,384
- 총 절감: ~655K → ~33K (Input/Output만)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SharedNeurons(nn.Module):
    """
    공유 뉴런 풀 - 모든 레이어에서 참조

    v9.0 변경:
    - input_neurons_q/k/v/m → input_neuron (단일 공유 행렬)
    - output_neurons_o → output_neuron (단일 공유 행렬)
    - get_input_neurons(), get_output_neurons() 삭제

    TransformNeurons:
        - input_neuron: 공유 압축 행렬 [d_model, rank]
        - process_neurons_q: Q용 Householder 변환 [n_process, rank]
        - process_neurons_k: K용 Householder 변환 [n_process, rank]
        - process_neurons_v: V용 Householder 변환 [n_process, rank]
        - process_neurons_o: O용 Householder 변환 [n_process, rank]
        - process_neurons_m: Memory Query용 Householder 변환 [n_process, rank]
        - output_neuron: 공유 확장 행렬 [rank, d_model]

    KnowledgeNeurons:
        - knowledge_K: 지식 검색 키 [n_knowledge, rank]
        - knowledge_V: 지식 내용 값 [n_knowledge, d_model]
    """
    def __init__(
        self,
        d_model: int,
        rank: int,
        n_process: int,
        n_knowledge: int,
        # Legacy params (ignored in v9)
        n_input: int = None,
        n_output: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.n_process = n_process
        self.n_knowledge = n_knowledge

        # TransformNeurons - Input (단일 공유 행렬)
        self.input_neuron = nn.Parameter(torch.zeros(d_model, rank))

        # TransformNeurons - Process (Q/K/V/O/M 분리 유지)
        self.process_neurons_q = nn.Parameter(torch.zeros(n_process, rank))
        self.process_neurons_k = nn.Parameter(torch.zeros(n_process, rank))
        self.process_neurons_v = nn.Parameter(torch.zeros(n_process, rank))
        self.process_neurons_o = nn.Parameter(torch.zeros(n_process, rank))
        self.process_neurons_m = nn.Parameter(torch.zeros(n_process, rank))

        # TransformNeurons - Output (단일 공유 행렬)
        self.output_neuron = nn.Parameter(torch.zeros(rank, d_model))

        # KnowledgeNeurons
        self.knowledge_K = nn.Parameter(torch.zeros(n_knowledge, rank))
        self.knowledge_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_weights()

    def _init_weights(self):
        """뉴런 초기화"""
        # input_neuron: 직교 초기화 (QR)
        if self.d_model >= self.rank:
            q, _ = torch.linalg.qr(torch.randn(self.d_model, self.rank))
            self.input_neuron.data = q
        else:
            q, _ = torch.linalg.qr(torch.randn(self.rank, self.d_model))
            self.input_neuron.data = q.T

        # process_neurons_q: 단위 벡터 초기화
        for i in range(self.n_process):
            v = torch.randn(self.rank)
            self.process_neurons_q.data[i] = v / (v.norm() + 1e-8)

        # process_neurons_k: 단위 벡터 초기화
        for i in range(self.n_process):
            v = torch.randn(self.rank)
            self.process_neurons_k.data[i] = v / (v.norm() + 1e-8)

        # process_neurons_v: 단위 벡터 초기화
        for i in range(self.n_process):
            v = torch.randn(self.rank)
            self.process_neurons_v.data[i] = v / (v.norm() + 1e-8)

        # process_neurons_o: 단위 벡터 초기화
        for i in range(self.n_process):
            v = torch.randn(self.rank)
            self.process_neurons_o.data[i] = v / (v.norm() + 1e-8)

        # process_neurons_m: 단위 벡터 초기화
        for i in range(self.n_process):
            v = torch.randn(self.rank)
            self.process_neurons_m.data[i] = v / (v.norm() + 1e-8)

        # output_neuron: 직교 초기화 (QR)
        if self.rank >= self.d_model:
            q, _ = torch.linalg.qr(torch.randn(self.rank, self.d_model))
            self.output_neuron.data = q
        else:
            q, _ = torch.linalg.qr(torch.randn(self.d_model, self.rank))
            self.output_neuron.data = q.T  # [rank, d_model]

        # knowledge_K: 정규분포 + 정규화
        nn.init.normal_(self.knowledge_K, std=0.02)
        with torch.no_grad():
            self.knowledge_K.data = F.normalize(self.knowledge_K.data, dim=-1)

        # knowledge_V: 정규분포
        nn.init.normal_(self.knowledge_V, std=0.02)

    def get_process_neurons(self, neuron_type: str):
        """neuron_type에 따라 적절한 process neurons 반환"""
        if neuron_type == 'q':
            return self.process_neurons_q
        elif neuron_type == 'k':
            return self.process_neurons_k
        elif neuron_type == 'v':
            return self.process_neurons_v
        elif neuron_type == 'o':
            return self.process_neurons_o
        elif neuron_type == 'm':
            return self.process_neurons_m
        else:
            raise ValueError(f"Unknown neuron_type: {neuron_type}. Use 'q', 'k', 'v', 'o', or 'm'.")

    def apply_householder(self, x, v):
        """
        Householder 변환: H @ x = x - 2 * v * (v.T @ x) / ||v||²

        Args:
            x: [B, S, rank] or [B, S, d_model]
            v: [B, S, rank] or [B, S, d_model]

        Returns:
            H @ x: same shape as x
        """
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + 1e-8
        v_normalized = v / v_norm_sq.sqrt()
        vTx = (x * v_normalized).sum(dim=-1, keepdim=True)
        return x - 2 * v_normalized * vTx


class Compressor(nn.Module):
    """
    d_model → rank 압축 (Q/K/V용)

    v9.0 변경:
    - input_router 삭제
    - 단순 x @ input_neuron으로 압축 (라우터 없음)
    - 다양성은 Process Householder에서만 발생

    흐름:
    1. x @ input_neuron → d_model → rank 압축
    2. Process neuron selection (top-k) → Householder 순차 적용
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_process: int,
        process_k: int,
        neuron_type: str = 'q',  # 'q', 'k', 'v', 'm'
        # Legacy params (ignored in v9)
        n_input: int = None,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.n_process = n_process
        self.process_k = process_k
        self.neuron_type = neuron_type

        # process_router만 유지 (레이어/모듈별 독립)
        self.process_router = nn.Linear(rank, n_process, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, S, d_model]

        Returns:
            output: [B, S, rank]
            routing_info: dict with process_indices
        """
        B, S, D = x.shape

        # 1. 단순 압축: d_model → rank (라우터 없음)
        # input_neuron: [d_model, rank]
        x_compressed = x @ self.shared_neurons.input_neuron  # [B, S, rank]

        # 2. Process neuron selection (top-k) - 압축 후 결정
        process_scores = self.process_router(x_compressed)  # [B, S, n_process]
        _, process_indices = torch.topk(process_scores, self.process_k, dim=-1)  # [B, S, k]

        # 3. Householder 순차 적용 (neuron_type에 따라 다른 pool 사용)
        process_neurons = self.shared_neurons.get_process_neurons(self.neuron_type)
        idx_expanded = process_indices.unsqueeze(-1).expand(B, S, self.process_k, self.rank)
        selected_v = process_neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        selected_v = selected_v.gather(2, idx_expanded)  # [B, S, k, rank]

        for i in range(self.process_k):
            v = selected_v[:, :, i, :]  # [B, S, rank]
            x_compressed = self.shared_neurons.apply_householder(x_compressed, v)

        routing_info = {
            'process_indices': process_indices,
        }

        return x_compressed, routing_info


class Expander(nn.Module):
    """
    rank → d_model 복원 (O projection용)

    v9.0 변경:
    - output_router 삭제
    - 단순 x @ output_neuron으로 확장 (라우터 없음)
    - 다양성은 Process Householder에서만 발생

    흐름:
    1. Process neuron selection (top-k) → Householder 순차 적용
    2. x @ output_neuron → rank → d_model 복원
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_process: int,
        process_k: int,
        neuron_type: str = 'o',  # 'o'
        # Legacy params (ignored in v9)
        n_output: int = None,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.n_process = n_process
        self.process_k = process_k
        self.neuron_type = neuron_type

        # process_router만 유지 (레이어/모듈별 독립)
        self.process_router = nn.Linear(rank, n_process, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, S, rank]

        Returns:
            output: [B, S, d_model]
            routing_info: dict with process_indices
        """
        B, S, _ = x.shape

        # 1. Process neuron selection (top-k)
        process_scores = self.process_router(x)  # [B, S, n_process]
        _, process_indices = torch.topk(process_scores, self.process_k, dim=-1)  # [B, S, k]

        # 2. Householder 순차 적용 (neuron_type에 따라 다른 pool 사용)
        process_neurons = self.shared_neurons.get_process_neurons(self.neuron_type)
        idx_expanded = process_indices.unsqueeze(-1).expand(B, S, self.process_k, self.rank)
        selected_v = process_neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        selected_v = selected_v.gather(2, idx_expanded)  # [B, S, k, rank]

        for i in range(self.process_k):
            v = selected_v[:, :, i, :]  # [B, S, rank]
            x = self.shared_neurons.apply_householder(x, v)

        # 3. 단순 확장: rank → d_model (라우터 없음)
        # output_neuron: [rank, d_model]
        x_expanded = x @ self.shared_neurons.output_neuron  # [B, S, d_model]

        routing_info = {
            'process_indices': process_indices,
        }

        return x_expanded, routing_info


class NeuronAttention(nn.Module):
    """
    Attention 대체 - Compressor/Expander 기반

    구조:
        x → Compressor_Q (q) → Q
        x → Compressor_K (k) → K
        x → Compressor_V (v) → V
        attention(Q, K, V) → attn_out
        attn_out → Expander_O (o) → output

    v9.0: Q/K/V/O는 같은 input/output neuron 공유, Process만 분리
        - Q는 process_neurons_q 사용
        - K는 process_neurons_k 사용
        - V는 process_neurons_v 사용
        - O는 process_neurons_o 사용
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_process: int,
        process_k: int,
        dropout: float = 0.1,
        # Legacy params (ignored in v9)
        n_input: int = None,
        n_output: int = None,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = rank // n_heads
        self.rank = rank

        # Q는 'q' process neuron pool 사용
        self.compressor_Q = Compressor(shared_neurons, d_model, rank, n_process, process_k, neuron_type='q')

        # K는 'k' process neuron pool 사용
        self.compressor_K = Compressor(shared_neurons, d_model, rank, n_process, process_k, neuron_type='k')

        # V는 'v' process neuron pool 사용
        self.compressor_V = Compressor(shared_neurons, d_model, rank, n_process, process_k, neuron_type='v')

        # O는 'o' process neuron pool 사용 (Expander)
        self.expander_O = Expander(shared_neurons, d_model, rank, n_process, process_k, neuron_type='o')

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, S, d_model]
            mask: [B, 1, S, S] causal mask

        Returns:
            output: [B, S, d_model]
            routing_info: dict with Q/K/V/O routing info
        """
        B, S, D = x.shape

        # Compress to Q, K, V
        Q, routing_Q = self.compressor_Q(x)  # [B, S, rank]
        K, routing_K = self.compressor_K(x)
        V, routing_V = self.compressor_V(x)

        # Multi-head reshape
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, S, d_head]
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # Attention
        attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = attn_weights @ V  # [B, H, S, d_head]

        # Concat heads
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.rank)  # [B, S, rank]

        # Expand back
        out, routing_O = self.expander_O(attn_out)  # [B, S, d_model]

        routing_info = {
            'routing_Q': routing_Q,
            'routing_K': routing_K,
            'routing_V': routing_V,
            'routing_O': routing_O,
            # For train.py load balance loss compatibility
            'neuron_indices': routing_Q['process_indices'],  # [B, S, k]
        }

        return out, routing_info


class NeuronMemory(nn.Module):
    """
    FFN 대체 - 내적 기반 지식 조회

    흐름:
        x → query_compressor → Q [B, S, rank]
        Q @ KnowledgeNeurons.K.T / sqrt(rank) → scores [B, S, n_knowledge]
        top-k 선택 → indices [B, S, k]
        softmax(topk_scores) → weights [B, S, k]
        KnowledgeNeurons.V[indices] → selected_V [B, S, k, d_model]
        (selected_V * weights.unsqueeze(-1)).sum(dim=2) → output [B, S, d_model]
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_process: int,
        process_k: int,
        knowledge_k: int = 8,
        # Legacy params (ignored in v9)
        n_input: int = None,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.knowledge_k = knowledge_k

        # Query 생성용 Compressor (neuron_type='m' for Memory)
        self.query_compressor = Compressor(
            shared_neurons, d_model, rank, n_process, process_k, neuron_type='m'
        )

    def forward(self, x):
        """
        Args:
            x: [B, S, d_model]

        Returns:
            output: [B, S, d_model]
            routing_info: dict with knowledge routing info
        """
        B, S, D = x.shape

        # Query 생성 (Compressor 사용)
        Q, query_routing = self.query_compressor(x)  # [B, S, rank]

        # Knowledge 검색 (내적 기반)
        K = self.shared_neurons.knowledge_K  # [n_knowledge, rank]
        V = self.shared_neurons.knowledge_V  # [n_knowledge, d_model]

        scores = Q @ K.T / math.sqrt(self.rank)  # [B, S, n_knowledge]
        topk_scores, topk_idx = torch.topk(scores, self.knowledge_k, dim=-1)  # [B, S, k]
        weights = F.softmax(topk_scores, dim=-1)  # [B, S, k]

        # 지식 조회
        V_selected = V[topk_idx]  # [B, S, k, d_model]
        out = (V_selected * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, d_model]

        routing_info = {
            'knowledge_indices': topk_idx,
            'knowledge_weights': weights,
            'query_routing': query_routing,
        }

        return out, routing_info


class NeuronCircuit(nn.Module):
    """
    Transformer Block - NeuronAttention + NeuronMemory

    구조:
        x → LayerNorm → NeuronAttention → + x (residual)
        x → LayerNorm → NeuronMemory → + x (residual)
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_process: int,
        process_k: int,
        knowledge_k: int,
        dropout: float = 0.1,
        # Legacy params (ignored in v9)
        n_input: int = None,
        n_output: int = None,
    ):
        super().__init__()

        self.attention = NeuronAttention(
            shared_neurons=shared_neurons,
            d_model=d_model,
            n_heads=n_heads,
            rank=rank,
            n_process=n_process,
            process_k=process_k,
            dropout=dropout,
        )

        self.memory = NeuronMemory(
            shared_neurons=shared_neurons,
            d_model=d_model,
            rank=rank,
            n_process=n_process,
            process_k=process_k,
            knowledge_k=knowledge_k,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, S, d_model]
            mask: [B, 1, S, S] causal mask

        Returns:
            output: [B, S, d_model]
            routing_info: dict with attention and memory routing info
        """
        # Attention
        attn_out, attn_routing = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # Memory
        mem_out, mem_routing = self.memory(self.norm2(x))
        x = x + self.dropout(mem_out)

        routing_info = {
            'attention': attn_routing,
            'memory': mem_routing,
            # For train.py compatibility
            'neuron_indices': attn_routing['neuron_indices'],
        }

        return x, routing_info


class DAWN(nn.Module):
    """
    DAWN v9.0 - Dynamic Architecture With Neurons

    v9.0 핵심 변경:
    - input_neurons_q/k/v/m → input_neuron (단일 공유)
    - output_neurons_o → output_neuron (단일 공유)
    - Compressor/Expander에서 라우터 제거
    - 모든 다양성은 Process Householder 조합에서 발생

    파라미터 효율:
    - v8.1: ~4.2M (Input/Output neurons)
    - v9.0: ~66K (Input/Output neurons)
    """
    __version__ = "9.0"

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        rank: int = 64,
        max_seq_len: int = 128,
        n_process: int = 32,
        process_k: int = 3,
        n_knowledge: int = 64,
        knowledge_k: int = 8,
        dropout: float = 0.1,
        # Legacy params (kept for compatibility)
        n_input: int = None,
        n_output: int = None,
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
        self.n_process = n_process
        self.process_k = process_k
        self.n_knowledge = n_knowledge
        self.knowledge_k = knowledge_k

        # train.py 호환용
        self.n_neurons = n_process  # For load balance loss
        self.basis_rank = rank  # For analysis scripts

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # SharedNeurons (레이어 간 공유!)
        self.shared_neurons = SharedNeurons(
            d_model=d_model,
            rank=rank,
            n_process=n_process,
            n_knowledge=n_knowledge,
        )

        # Layers (모두 같은 SharedNeurons 참조)
        self.layers = nn.ModuleList([
            NeuronCircuit(
                shared_neurons=self.shared_neurons,
                d_model=d_model,
                n_heads=n_heads,
                rank=rank,
                n_process=n_process,
                process_k=process_k,
                knowledge_k=knowledge_k,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        )

        self.dropout = nn.Dropout(dropout)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """Linear, Embedding, LayerNorm 초기화"""
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
        """
        Args:
            input_ids: [B, S]
            labels: [B, S] (optional)
            return_routing_info: bool
            return_losses: bool

        Returns:
            다양한 조합의 반환값 (train.py 호환)
        """
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

        # train.py 호환 반환 형식
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
        """
        input/output neurons 직교성 유지

        Input: W.T @ W ≈ I (rank × rank)
        Output: W @ W.T ≈ I (rank × rank)
        """
        loss = 0.0

        # Input neuron: [d_model, rank]
        # W.T @ W should be identity (rank × rank)
        inp = self.shared_neurons.input_neuron
        WtW = inp.T @ inp  # [rank, rank]
        I = torch.eye(self.rank, device=inp.device)
        loss += ((WtW - I) ** 2).mean()

        # Output neuron: [rank, d_model]
        # W @ W.T should be identity (rank × rank)
        out = self.shared_neurons.output_neuron
        WWt = out @ out.T  # [rank, rank]
        loss += ((WWt - I) ** 2).mean()

        return loss / 2

    def process_norm_loss(self):
        """process neurons ||v|| ≈ 1 유지 (Q/K/V/O/M 다섯 풀 모두)"""
        # Q pool
        norms_q = self.shared_neurons.process_neurons_q.norm(dim=-1)
        loss_q = ((norms_q - 1.0) ** 2).mean()

        # K pool
        norms_k = self.shared_neurons.process_neurons_k.norm(dim=-1)
        loss_k = ((norms_k - 1.0) ** 2).mean()

        # V pool
        norms_v = self.shared_neurons.process_neurons_v.norm(dim=-1)
        loss_v = ((norms_v - 1.0) ** 2).mean()

        # O pool
        norms_o = self.shared_neurons.process_neurons_o.norm(dim=-1)
        loss_o = ((norms_o - 1.0) ** 2).mean()

        # M pool (Memory Query)
        norms_m = self.shared_neurons.process_neurons_m.norm(dim=-1)
        loss_m = ((norms_m - 1.0) ** 2).mean()

        return (loss_q + loss_k + loss_v + loss_o + loss_m) / 5

    def knowledge_diversity_loss(self):
        """
        knowledge K 다양성 유지

        지식 키 벡터들 간의 유사도를 낮춰서 다양한 지식을 저장하도록 함
        """
        K = self.shared_neurons.knowledge_K  # [n_knowledge, rank]
        K_norm = F.normalize(K, dim=-1)
        sim = K_norm @ K_norm.T  # [n_knowledge, n_knowledge]

        # 대각선 제외한 유사도 낮추기
        mask = ~torch.eye(self.n_knowledge, dtype=torch.bool, device=K.device)
        return sim[mask].abs().mean()

    def get_auxiliary_losses(self):
        """train.py 호환 auxiliary losses"""
        return {
            'orth_total': self.orthogonality_loss(),
            'process_norm': self.process_norm_loss(),
            'knowledge_div': self.knowledge_diversity_loss(),
        }

    def count_parameters(self):
        """총 학습 가능한 파라미터 수"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self):
        """모델 설정 반환"""
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'rank': self.rank,
            'max_seq_len': self.max_seq_len,
            'n_process': self.n_process,
            'process_k': self.process_k,
            'n_knowledge': self.n_knowledge,
            'knowledge_k': self.knowledge_k,
        }
