"""
DAWN v7.9 - NeuronCircuit with Householder Transformations

핵심 아이디어:
- Basis weighted sum이 rank 붕괴 일으킴 (v7.7 문제)
- 대신 Householder 행렬 순차 곱으로 직교성 유지
- Input/Process/Output 블록 분리

구조:
    InputNeuron: [n_input, d_model, rank] - 256차원 → 64차원 압축
    ProcessNeuron: [n_process, rank] - Householder 반사 벡터
    OutputNeuron: [n_output, rank, d_model] - 64차원 → 256차원 복원

v7.8 대비 변경점:
- NeuronBank → NeuronCircuit (Input/Process/Output 분리)
- Householder 변환으로 직교성 자동 유지
- 파라미터 감소: ~264K (v7.7 basis 540K보다 적음)

조합 수: n_input × C(n_process, k) × n_output = 8 × C(32,3) × 8 = 317K 조합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NeuronCircuit(nn.Module):
    """
    NeuronCircuit: Input/Process/Output 분리 구조

    - InputNeuron: 차원 압축 (d_model → rank)
    - ProcessNeuron: Householder 변환 벡터들
    - OutputNeuron: 차원 복원 (rank → d_model)
    """
    def __init__(
        self,
        d_model: int = 256,
        rank: int = 64,
        n_input: int = 8,
        n_process: int = 32,
        n_output: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.n_input = n_input
        self.n_process = n_process
        self.n_output = n_output

        # InputNeuron: [n_input, d_model, rank] - 차원 압축
        self.input_neurons = nn.Parameter(torch.zeros(n_input, d_model, rank))

        # ProcessNeuron: [n_process, rank] - Householder 반사 벡터
        self.process_neurons = nn.Parameter(torch.zeros(n_process, rank))

        # OutputNeuron: [n_output, rank, d_model] - 차원 복원
        self.output_neurons = nn.Parameter(torch.zeros(n_output, rank, d_model))

        # 초기화
        self._init_weights()

    def _init_weights(self):
        """직교 초기화"""
        # InputNeuron: 직교 행렬로 초기화
        for i in range(self.n_input):
            q, _ = torch.linalg.qr(torch.randn(self.d_model, self.rank))
            self.input_neurons.data[i] = q

        # ProcessNeuron: 단위 벡터로 초기화 (랜덤 방향)
        for i in range(self.n_process):
            v = torch.randn(self.rank)
            v = v / (v.norm() + 1e-8)
            self.process_neurons.data[i] = v

        # OutputNeuron: 직교 행렬로 초기화
        for i in range(self.n_output):
            q, _ = torch.linalg.qr(torch.randn(self.d_model, self.rank))
            self.output_neurons.data[i] = q.T  # [rank, d_model]

    def apply_householder(self, x, v):
        """
        Householder 변환 적용

        H = I - 2 * v @ v.T / ||v||²
        H @ x = x - 2 * v * (v.T @ x) / ||v||²

        Args:
            x: [..., rank]
            v: [rank] or [..., rank]

        Returns:
            H @ x: [..., rank]
        """
        # v 정규화
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + 1e-8  # [..., 1]
        v_normalized = v / v_norm_sq.sqrt()  # [..., rank]

        # v.T @ x
        vTx = (x * v_normalized).sum(dim=-1, keepdim=True)  # [..., 1]

        # H @ x = x - 2 * v * (v.T @ x)
        return x - 2 * v_normalized * vTx

    def forward_input(self, x, input_idx=None, input_weights=None):
        """
        Input projection: d_model → rank

        Args:
            x: [B, S, d_model]
            input_idx: [B, S] - hard selection (1개 선택)
            input_weights: [B, S, n_input] - soft selection (가중 평균)

        Returns:
            projected: [B, S, rank]
        """
        B, S, D = x.shape

        if input_weights is not None:
            # Soft selection: 모든 input neuron으로 projection 후 가중 평균
            # x @ input_neurons: [B, S, n_input, rank]
            all_proj = torch.einsum('bsd,ndr->bsnr', x, self.input_neurons)
            # 가중 평균
            weights = input_weights.unsqueeze(-1)  # [B, S, n_input, 1]
            return (all_proj * weights).sum(dim=2)  # [B, S, rank]
        else:
            # Hard selection
            # 선택된 input neuron만 gather
            # input_neurons[input_idx]: [B, S, D, rank]
            selected = self.input_neurons[input_idx]  # [B, S, D, rank]
            return torch.einsum('bsd,bsdr->bsr', x, selected)  # [B, S, rank]

    def forward_process(self, x, process_indices):
        """
        Process: Householder 변환 순차 적용

        Args:
            x: [B, S, rank]
            process_indices: [B, S, k] - 선택된 process neuron indices

        Returns:
            transformed: [B, S, rank]
        """
        B, S, _ = x.shape
        k = process_indices.shape[-1]

        # 선택된 Householder 벡터들 gather
        # process_neurons: [n_process, rank]
        # process_indices: [B, S, k]
        # 결과: [B, S, k, rank]
        idx_expanded = process_indices.unsqueeze(-1).expand(B, S, k, self.rank)
        selected_v = self.process_neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        selected_v = selected_v.gather(2, idx_expanded)  # [B, S, k, rank]

        # 순차적으로 Householder 변환 적용
        for i in range(k):
            v = selected_v[:, :, i, :]  # [B, S, rank]
            x = self.apply_householder(x, v)

        return x

    def forward_output(self, x, output_idx=None, output_weights=None):
        """
        Output projection: rank → d_model

        Args:
            x: [B, S, rank]
            output_idx: [B, S] - hard selection
            output_weights: [B, S, n_output] - soft selection

        Returns:
            projected: [B, S, d_model]
        """
        B, S, _ = x.shape

        if output_weights is not None:
            # Soft selection
            # x @ output_neurons: [B, S, n_output, d_model]
            all_proj = torch.einsum('bsr,nrd->bsnd', x, self.output_neurons)
            weights = output_weights.unsqueeze(-1)  # [B, S, n_output, 1]
            return (all_proj * weights).sum(dim=2)  # [B, S, d_model]
        else:
            # Hard selection
            selected = self.output_neurons[output_idx]  # [B, S, rank, d_model]
            return torch.einsum('bsr,bsrd->bsd', x, selected)  # [B, S, d_model]

    def forward(self, x, input_idx=None, input_weights=None,
                process_indices=None, output_idx=None, output_weights=None):
        """
        Full forward pass

        Args:
            x: [B, S, d_model]
            input_idx/input_weights: Input neuron selection
            process_indices: [B, S, k] Process neuron indices
            output_idx/output_weights: Output neuron selection

        Returns:
            output: [B, S, d_model]
        """
        # 1. Input projection
        x = self.forward_input(x, input_idx, input_weights)

        # 2. Process (Householder transforms)
        if process_indices is not None:
            x = self.forward_process(x, process_indices)

        # 3. Output projection
        x = self.forward_output(x, output_idx, output_weights)

        return x

    def orthogonality_loss(self):
        """
        Input/Output Neuron의 직교성 유지 loss

        W @ W.T ≈ I 유지
        """
        loss = 0.0

        # InputNeuron: [n_input, D, rank]
        # W.T @ W: [n_input, rank, rank]
        WtW_in = torch.bmm(
            self.input_neurons.transpose(-1, -2),
            self.input_neurons
        )  # [n_input, rank, rank]
        I = torch.eye(self.rank, device=self.input_neurons.device).unsqueeze(0)
        loss += ((WtW_in - I) ** 2).mean()

        # OutputNeuron: [n_output, rank, D]
        WtW_out = torch.bmm(
            self.output_neurons,
            self.output_neurons.transpose(-1, -2)
        )  # [n_output, rank, rank]
        loss += ((WtW_out - I) ** 2).mean()

        return loss / 2

    def process_norm_loss(self):
        """
        ProcessNeuron 벡터의 norm 유지 loss

        ||v|| ≈ 1 유지 (Householder 안정성)
        """
        norms = self.process_neurons.norm(dim=-1)  # [n_process]
        return ((norms - 1.0) ** 2).mean()


class CircuitRouter(nn.Module):
    """
    NeuronCircuit용 라우터

    입력 x로부터 Input/Process/Output neuron 선택
    """
    def __init__(
        self,
        d_model: int,
        n_input: int = 8,
        n_process: int = 32,
        n_output: int = 8,
        process_k: int = 3,
        use_soft_selection: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_input = n_input
        self.n_process = n_process
        self.n_output = n_output
        self.process_k = process_k
        self.use_soft_selection = use_soft_selection

        # 라우터 projections
        self.input_router = nn.Linear(d_model, n_input, bias=False)
        self.process_router = nn.Linear(d_model, n_process, bias=False)
        self.output_router = nn.Linear(d_model, n_output, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, S, d_model]

        Returns:
            routing_info: dict containing:
                - input_idx or input_weights
                - process_indices
                - output_idx or output_weights
        """
        B, S, D = x.shape

        # Input routing
        input_scores = self.input_router(x)  # [B, S, n_input]
        if self.use_soft_selection:
            input_weights = F.softmax(input_scores, dim=-1)
        else:
            input_idx = input_scores.argmax(dim=-1)  # [B, S]

        # Process routing (always top-k selection)
        process_scores = self.process_router(x)  # [B, S, n_process]
        topk_scores, process_indices = torch.topk(
            process_scores, self.process_k, dim=-1
        )  # [B, S, k]
        process_weights = F.softmax(topk_scores, dim=-1)

        # Output routing
        output_scores = self.output_router(x)  # [B, S, n_output]
        if self.use_soft_selection:
            output_weights = F.softmax(output_scores, dim=-1)
        else:
            output_idx = output_scores.argmax(dim=-1)  # [B, S]

        routing_info = {
            'process_indices': process_indices,
            'process_weights': process_weights,
        }

        if self.use_soft_selection:
            routing_info['input_weights'] = input_weights
            routing_info['output_weights'] = output_weights
        else:
            routing_info['input_idx'] = input_idx
            routing_info['output_idx'] = output_idx

        return routing_info


class NeuronCircuitQKV(nn.Module):
    """
    NeuronCircuit 기반 동적 Q/K/V/O 생성

    각 Q/K/V/O에 대해 별도의 NeuronCircuit 사용
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rank: int,
        n_input: int = 8,
        n_process: int = 32,
        n_output: int = 8,
        process_k: int = 3,
        dropout: float = 0.1,
        use_soft_selection: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = rank // n_heads
        self.rank = rank

        # Q/K/V/O 각각에 대한 NeuronCircuit
        self.circuit_Q = NeuronCircuit(d_model, rank, n_input, n_process, n_output)
        self.circuit_K = NeuronCircuit(d_model, rank, n_input, n_process, n_output)
        self.circuit_V = NeuronCircuit(d_model, rank, n_input, n_process, n_output)
        self.circuit_O = NeuronCircuit(rank, d_model, n_input, n_process, n_output)

        # 공유 라우터 (Q/K/V 동일한 routing 사용)
        self.router = CircuitRouter(
            d_model, n_input, n_process, n_output, process_k, use_soft_selection
        )

        # O projection용 별도 라우터 (rank → d_model)
        self.router_O = CircuitRouter(
            rank, n_input, n_process, n_output, process_k, use_soft_selection
        )

        self.dropout = nn.Dropout(dropout)

        # 분석 스크립트 호환용
        self.basis_rank = rank

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, S, D]
            mask: [B, 1, S, S] causal mask

        Returns:
            attn_out: [B, S, D]
            routing_info: dict
        """
        B, S, D = x.shape

        # 1. 라우팅 - Input/Process/Output neuron 선택
        routing_info = self.router(x)

        # Selection 방식 확인
        use_soft = 'input_weights' in routing_info

        if use_soft:
            input_weights = routing_info['input_weights']
            output_weights = routing_info['output_weights']
            input_idx = None
            output_idx = None
        else:
            input_weights = None
            output_weights = None
            input_idx = routing_info['input_idx']
            output_idx = routing_info['output_idx']

        process_indices = routing_info['process_indices']

        # 2. Q/K/V 생성 via NeuronCircuit
        Q = self.circuit_Q(
            x, input_idx, input_weights, process_indices, output_idx, output_weights
        )  # [B, S, rank]
        K = self.circuit_K(
            x, input_idx, input_weights, process_indices, output_idx, output_weights
        )
        V = self.circuit_V(
            x, input_idx, input_weights, process_indices, output_idx, output_weights
        )

        # 3. Multi-head reshape
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # 4. Attention
        attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = attn_weights @ V  # [B, H, S, d_head]

        # 5. Concat
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.rank)  # [B, S, rank]

        # 6. O projection via NeuronCircuit (rank → d_model)
        routing_info_O = self.router_O(attn_out)
        if 'input_weights' in routing_info_O:
            attn_out = self.circuit_O(
                attn_out,
                input_weights=routing_info_O['input_weights'],
                process_indices=routing_info_O['process_indices'],
                output_weights=routing_info_O['output_weights'],
            )
        else:
            attn_out = self.circuit_O(
                attn_out,
                input_idx=routing_info_O['input_idx'],
                process_indices=routing_info_O['process_indices'],
                output_idx=routing_info_O['output_idx'],
            )

        routing_info['routing_O'] = routing_info_O

        return attn_out, routing_info


class TransformerBlock(nn.Module):
    """
    Transformer Block: NeuronCircuitQKV + FFN
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rank: int,
        n_input: int = 8,
        n_process: int = 32,
        n_output: int = 8,
        process_k: int = 3,
        dropout: float = 0.1,
        use_soft_selection: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        # Attention with NeuronCircuit Q/K/V
        self.qkv_circuit = NeuronCircuitQKV(
            d_model=d_model,
            n_heads=n_heads,
            rank=rank,
            n_input=n_input,
            n_process=n_process,
            n_output=n_output,
            process_k=process_k,
            dropout=dropout,
            use_soft_selection=use_soft_selection,
        )

        # Standard FFN
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, S, D]
            mask: [B, 1, S, S]

        Returns:
            x: [B, S, D]
            routing_info: dict
        """
        # Attention with residual
        residual = x
        x_norm = self.norm1(x)
        attn_out, routing_info = self.qkv_circuit(x_norm, mask)
        x = residual + self.dropout(attn_out)

        # FFN with residual
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.w_down(F.gelu(self.w_up(x_norm)))
        x = residual + self.dropout(ffn_out)

        return x, routing_info


class DAWN(nn.Module):
    """
    DAWN v7.9 - NeuronCircuit with Householder Transformations

    핵심 변경:
    - NeuronBank → NeuronCircuit (Input/Process/Output 분리)
    - Householder 변환으로 직교성 자동 유지
    - rank 붕괴 방지
    """
    __version__ = "7.9"

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 128,
        rank: int = 64,
        n_input: int = 8,
        n_process: int = 32,
        n_output: int = 8,
        process_k: int = 3,
        dropout: float = 0.1,
        use_soft_selection: bool = True,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.n_input = n_input
        self.n_process = n_process
        self.n_output = n_output
        self.process_k = process_k
        self.use_soft_selection = use_soft_selection

        # 분석 스크립트 호환용
        self.basis_rank = rank
        self.n_neurons = n_input  # 대략적인 호환성

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                rank=rank,
                n_input=n_input,
                n_process=n_process,
                n_output=n_output,
                process_k=process_k,
                dropout=dropout,
                use_soft_selection=use_soft_selection,
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

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

        # Handle return_losses (for train.py compatibility)
        if return_losses:
            losses = self.get_auxiliary_losses()
            if return_routing_info:
                return (logits, losses, routing_infos)
            return (logits, losses)

        if return_routing_info:
            return (loss, logits, routing_infos) if labels is not None else (logits, routing_infos)
        else:
            return (loss, logits) if labels is not None else logits

    def orthogonality_loss(self):
        """Input/Output Neuron의 직교성 유지"""
        total_loss = 0.0
        count = 0

        for layer in self.layers:
            qkv = layer.qkv_circuit
            for circuit in [qkv.circuit_Q, qkv.circuit_K, qkv.circuit_V, qkv.circuit_O]:
                total_loss += circuit.orthogonality_loss()
                count += 1

        return total_loss / count if count > 0 else 0.0

    def process_norm_loss(self):
        """ProcessNeuron 벡터의 norm 유지"""
        total_loss = 0.0
        count = 0

        for layer in self.layers:
            qkv = layer.qkv_circuit
            for circuit in [qkv.circuit_Q, qkv.circuit_K, qkv.circuit_V, qkv.circuit_O]:
                total_loss += circuit.process_norm_loss()
                count += 1

        return total_loss / count if count > 0 else 0.0

    def load_balance_loss(self):
        """
        라우터 load balance loss

        모든 neuron이 균등하게 사용되도록 유도
        """
        # 이 loss는 forward 시에 routing_info를 사용해야 함
        # 여기서는 간단한 구현으로 0 반환
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def get_auxiliary_losses(self):
        """모든 보조 loss 반환"""
        return {
            'orthogonality': self.orthogonality_loss(),
            'process_norm': self.process_norm_loss(),
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
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'rank': self.rank,
            'n_input': self.n_input,
            'n_process': self.n_process,
            'n_output': self.n_output,
            'process_k': self.process_k,
            'use_soft_selection': self.use_soft_selection,
        }
