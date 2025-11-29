"""
DAWN v9.0 - Dynamic Architecture With Neurons

핵심 아이디어: base input/output 1개 + Householder로 동적 변형

구조:
    SharedNeurons (공유 뉴런 풀 - 레이어 간 공유)
    ├── TransformNeurons
    │     ├── base_input:          [d_model, rank] — 기본 압축 행렬
    │     ├── base_output:         [rank, d_model] — 기본 확장 행렬
    │     ├── input_householder:   [n_process, d_model] — Input 변형용 통합 풀
    │     ├── process_householder: [n_process, rank] — 64차원 변환용 통합 풀
    │     └── output_householder:  [n_process, d_model] — Output 변형용 통합 풀
    │
    └── KnowledgeNeurons
          ├── knowledge_K: [n_knowledge, rank]
          └── knowledge_V: [n_knowledge, d_model]

파라미터:
- base_input: 256 × 64 = 16K
- base_output: 64 × 256 = 16K
- input_householder: 32 × 256 = 8K
- process_householder: 32 × 64 = 2K
- output_householder: 32 × 256 = 8K
- 총: ~50K (vs v8.1의 655K, 92% 절감)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SharedNeurons(nn.Module):
    """
    공유 뉴런 풀 - 모든 레이어에서 참조

    v9.0:
    - base_input/output: 기본 변환 행렬 (1개씩)
    - input/process/output_householder: 통합 Householder 풀
    - 각 Compressor/Expander가 독립 라우터로 같은 풀에서 다른 조합 선택
    """
    def __init__(
        self,
        d_model: int,
        rank: int,
        n_process: int,
        n_knowledge: int,
        # Legacy params (ignored)
        n_input: int = None,
        n_output: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.n_process = n_process
        self.n_knowledge = n_knowledge

        # Base matrices (1개씩)
        self.base_input = nn.Parameter(torch.zeros(d_model, rank))
        self.base_output = nn.Parameter(torch.zeros(rank, d_model))

        # Unified Householder pools
        self.input_householder = nn.Parameter(torch.zeros(n_process, d_model))
        self.process_householder = nn.Parameter(torch.zeros(n_process, rank))
        self.output_householder = nn.Parameter(torch.zeros(n_process, d_model))

        # KnowledgeNeurons
        self.knowledge_K = nn.Parameter(torch.zeros(n_knowledge, rank))
        self.knowledge_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_weights()

    def _init_weights(self):
        """뉴런 초기화"""
        # base_input: 직교 초기화 (QR)
        if self.d_model >= self.rank:
            q, _ = torch.linalg.qr(torch.randn(self.d_model, self.rank))
            self.base_input.data = q
        else:
            q, _ = torch.linalg.qr(torch.randn(self.rank, self.d_model))
            self.base_input.data = q.T

        # base_output: 직교 초기화 (QR)
        if self.rank >= self.d_model:
            q, _ = torch.linalg.qr(torch.randn(self.rank, self.d_model))
            self.base_output.data = q
        else:
            q, _ = torch.linalg.qr(torch.randn(self.d_model, self.rank))
            self.base_output.data = q.T

        # input_householder: 단위 벡터 초기화
        for i in range(self.n_process):
            v = torch.randn(self.d_model)
            self.input_householder.data[i] = v / (v.norm() + 1e-8)

        # process_householder: 단위 벡터 초기화
        for i in range(self.n_process):
            v = torch.randn(self.rank)
            self.process_householder.data[i] = v / (v.norm() + 1e-8)

        # output_householder: 단위 벡터 초기화
        for i in range(self.n_process):
            v = torch.randn(self.d_model)
            self.output_householder.data[i] = v / (v.norm() + 1e-8)

        # knowledge_K: 정규분포 + 정규화
        nn.init.normal_(self.knowledge_K, std=0.02)
        with torch.no_grad():
            self.knowledge_K.data = F.normalize(self.knowledge_K.data, dim=-1)

        # knowledge_V: 정규분포
        nn.init.normal_(self.knowledge_V, std=0.02)

    def apply_householder_to_vector(self, x, v):
        """
        Householder 변환 (벡터에 적용): H @ x = x - 2 * v * (v.T @ x) / ||v||²

        Args:
            x: [B, S, dim]
            v: [B, S, dim]

        Returns:
            H @ x: [B, S, dim]
        """
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + 1e-8
        v_normalized = v / v_norm_sq.sqrt()
        vTx = (x * v_normalized).sum(dim=-1, keepdim=True)
        return x - 2 * v_normalized * vTx

    def apply_householder_to_matrix(self, W, v):
        """
        Householder 변환 (행렬에 적용): H @ W where H = I - 2*vv^T/||v||^2

        Args:
            W: [dim1, dim2] base matrix
            v: [B, S, dim1] householder vectors

        Returns:
            H @ W: [B, S, dim1, dim2]
        """
        # v 정규화
        v_norm_sq = (v * v).sum(-1, keepdim=True) + 1e-8
        v_normalized = v / v_norm_sq.sqrt()  # [B, S, dim1]

        # v^T @ W: [B, S, dim2]
        vTW = torch.einsum('bsd,dr->bsr', v_normalized, W)

        # H @ W = W - 2 * v * (v^T @ W)
        # W: [dim1, dim2] -> [1, 1, dim1, dim2]
        # v_normalized: [B, S, dim1] -> [B, S, dim1, 1]
        # vTW: [B, S, dim2] -> [B, S, 1, dim2]
        dynamic_W = W.unsqueeze(0).unsqueeze(0) - 2 * v_normalized.unsqueeze(-1) * vTW.unsqueeze(-2)

        return dynamic_W  # [B, S, dim1, dim2]


class Compressor(nn.Module):
    """
    d_model → rank 압축

    v9.0:
    1. Input Householder 선택 → base_input 동적 변형
    2. 동적 압축: x @ dynamic_input
    3. Process Householder 선택 → rank 공간에서 변환
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_process: int,
        process_k: int,
        neuron_type: str = 'q',
        n_input: int = None,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.n_process = n_process
        self.process_k = process_k
        self.neuron_type = neuron_type

        # 독립 라우터 (같은 풀에서 다른 조합 선택)
        self.input_router = nn.Linear(d_model, n_process, bias=False)
        self.process_router = nn.Linear(rank, n_process, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, S, d_model]

        Returns:
            output: [B, S, rank]
            routing_info: dict
        """
        B, S, D = x.shape

        # 1. Input Householder 선택 (soft selection)
        input_scores = self.input_router(x)  # [B, S, n_process]
        input_weights = F.softmax(input_scores, dim=-1)  # [B, S, n_process]

        # 가중 평균으로 Householder 벡터 생성
        v_input = input_weights @ self.shared_neurons.input_householder  # [B, S, d_model]

        # 2. base_input을 Householder로 변형
        dynamic_input = self.shared_neurons.apply_householder_to_matrix(
            self.shared_neurons.base_input, v_input
        )  # [B, S, d_model, rank]

        # 3. 동적 압축
        x_compressed = torch.einsum('bsd,bsdr->bsr', x, dynamic_input)  # [B, S, rank]

        # 4. Process Householder 선택 (top-k)
        process_scores = self.process_router(x_compressed)  # [B, S, n_process]
        _, process_indices = torch.topk(process_scores, self.process_k, dim=-1)  # [B, S, k]

        # 5. Householder 순차 적용
        process_neurons = self.shared_neurons.process_householder
        idx_expanded = process_indices.unsqueeze(-1).expand(B, S, self.process_k, self.rank)
        selected_v = process_neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        selected_v = selected_v.gather(2, idx_expanded)  # [B, S, k, rank]

        for i in range(self.process_k):
            v = selected_v[:, :, i, :]  # [B, S, rank]
            x_compressed = self.shared_neurons.apply_householder_to_vector(x_compressed, v)

        routing_info = {
            'input_weights': input_weights,
            'process_indices': process_indices,
        }

        return x_compressed, routing_info


class Expander(nn.Module):
    """
    rank → d_model 복원

    v9.0:
    1. Process Householder 선택 → rank 공간에서 변환
    2. Output Householder 선택 → base_output 동적 변형
    3. 동적 확장: x @ dynamic_output
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_process: int,
        process_k: int,
        neuron_type: str = 'o',
        n_output: int = None,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.n_process = n_process
        self.process_k = process_k
        self.neuron_type = neuron_type

        # 독립 라우터
        self.process_router = nn.Linear(rank, n_process, bias=False)
        self.output_router = nn.Linear(rank, n_process, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, S, rank]

        Returns:
            output: [B, S, d_model]
            routing_info: dict
        """
        B, S, _ = x.shape

        # 1. Process Householder 선택 (top-k)
        process_scores = self.process_router(x)  # [B, S, n_process]
        _, process_indices = torch.topk(process_scores, self.process_k, dim=-1)  # [B, S, k]

        # 2. Householder 순차 적용
        process_neurons = self.shared_neurons.process_householder
        idx_expanded = process_indices.unsqueeze(-1).expand(B, S, self.process_k, self.rank)
        selected_v = process_neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        selected_v = selected_v.gather(2, idx_expanded)  # [B, S, k, rank]

        for i in range(self.process_k):
            v = selected_v[:, :, i, :]  # [B, S, rank]
            x = self.shared_neurons.apply_householder_to_vector(x, v)

        # 3. Output Householder 선택 (soft selection)
        output_scores = self.output_router(x)  # [B, S, n_process]
        output_weights = F.softmax(output_scores, dim=-1)  # [B, S, n_process]

        # 가중 평균으로 Householder 벡터 생성
        v_output = output_weights @ self.shared_neurons.output_householder  # [B, S, d_model]

        # 4. base_output을 Householder로 변형
        # base_output: [rank, d_model] -> apply H from left
        # H @ base_output.T = (base_output @ H.T).T
        # 더 간단하게: base_output.T를 변형 후 다시 transpose
        dynamic_output = self.shared_neurons.apply_householder_to_matrix(
            self.shared_neurons.base_output.T, v_output
        )  # [B, S, d_model, rank]
        dynamic_output = dynamic_output.transpose(-1, -2)  # [B, S, rank, d_model]

        # 5. 동적 확장
        x_expanded = torch.einsum('bsr,bsrd->bsd', x, dynamic_output)  # [B, S, d_model]

        routing_info = {
            'process_indices': process_indices,
            'output_weights': output_weights,
        }

        return x_expanded, routing_info


class NeuronAttention(nn.Module):
    """
    Attention 대체 - Compressor/Expander 기반
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
        n_input: int = None,
        n_output: int = None,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = rank // n_heads
        self.rank = rank

        # Q/K/V/O - 각각 독립 라우터, 같은 Householder 풀 공유
        self.compressor_Q = Compressor(shared_neurons, d_model, rank, n_process, process_k, neuron_type='q')
        self.compressor_K = Compressor(shared_neurons, d_model, rank, n_process, process_k, neuron_type='k')
        self.compressor_V = Compressor(shared_neurons, d_model, rank, n_process, process_k, neuron_type='v')
        self.expander_O = Expander(shared_neurons, d_model, rank, n_process, process_k, neuron_type='o')

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
            'neuron_indices': routing_Q['process_indices'],
        }

        return out, routing_info


class NeuronMemory(nn.Module):
    """
    FFN 대체 - 내적 기반 지식 조회
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        rank: int,
        n_process: int,
        process_k: int,
        knowledge_k: int = 8,
        n_input: int = None,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.rank = rank
        self.knowledge_k = knowledge_k

        self.query_compressor = Compressor(
            shared_neurons, d_model, rank, n_process, process_k, neuron_type='m'
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
    Transformer Block - NeuronAttention + NeuronMemory
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
    DAWN v9.0 - Dynamic Architecture With Neurons

    핵심: base input/output + Householder로 동적 변형
    - 통합 Householder 풀 (input/process/output)
    - 각 Compressor/Expander가 독립 라우터로 같은 풀에서 다른 조합 선택
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

        self.n_process = n_process
        self.process_k = process_k
        self.n_knowledge = n_knowledge
        self.knowledge_k = knowledge_k

        self.n_neurons = n_process
        self.basis_rank = rank

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.shared_neurons = SharedNeurons(
            d_model=d_model,
            rank=rank,
            n_process=n_process,
            n_knowledge=n_knowledge,
        )

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
        """base input/output 직교성 유지"""
        loss = 0.0

        # base_input: [d_model, rank] -> W.T @ W ≈ I
        inp = self.shared_neurons.base_input
        WtW = inp.T @ inp
        I = torch.eye(self.rank, device=inp.device)
        loss += ((WtW - I) ** 2).mean()

        # base_output: [rank, d_model] -> W @ W.T ≈ I
        out = self.shared_neurons.base_output
        WWt = out @ out.T
        loss += ((WWt - I) ** 2).mean()

        return loss / 2

    def householder_norm_loss(self):
        """Householder vectors ||v|| ≈ 1 유지"""
        loss = 0.0

        # input_householder
        norms = self.shared_neurons.input_householder.norm(dim=-1)
        loss += ((norms - 1.0) ** 2).mean()

        # process_householder
        norms = self.shared_neurons.process_householder.norm(dim=-1)
        loss += ((norms - 1.0) ** 2).mean()

        # output_householder
        norms = self.shared_neurons.output_householder.norm(dim=-1)
        loss += ((norms - 1.0) ** 2).mean()

        return loss / 3

    def knowledge_diversity_loss(self):
        K = self.shared_neurons.knowledge_K
        K_norm = F.normalize(K, dim=-1)
        sim = K_norm @ K_norm.T
        mask = ~torch.eye(self.n_knowledge, dtype=torch.bool, device=K.device)
        return sim[mask].abs().mean()

    def get_auxiliary_losses(self):
        return {
            'orth_total': self.orthogonality_loss(),
            'process_norm': self.householder_norm_loss(),
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
            'n_process': self.n_process,
            'process_k': self.process_k,
            'n_knowledge': self.n_knowledge,
            'knowledge_k': self.knowledge_k,
        }
