"""
DAWN v7.9 - NeuronCircuit with Householder Transformations

핵심 아이디어:
- Basis weighted sum이 rank 붕괴 일으킴 (v7.7 문제)
- 대신 Householder 행렬 순차 곱으로 직교성 유지
- Input/Process/Output 블록 분리

구조:
    NeuronCircuitDown (Q/K/V용): d_model → rank 압축
        - InputNeuron: [n_input, d_model, rank]
        - ProcessNeuron: [n_process, rank] (Householder on rank space)

    NeuronCircuitUp (O용): rank → d_model 복원
        - OutputNeuron: [n_output, rank, d_model]
        - ProcessNeuron: [n_process, d_model] (Householder on d_model space)

v7.8 대비 변경점:
- NeuronBank → NeuronCircuitDown/Up (명확한 방향성)
- Householder 변환으로 직교성 자동 유지
- 불필요한 파라미터 제거 (Down에서 OutputNeuron, Up에서 InputNeuron)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NeuronCircuitDown(nn.Module):
    """
    NeuronCircuitDown: d_model → rank 압축용 (Q/K/V)

    - InputNeuron: 차원 압축 (d_model → rank)
    - ProcessNeuron: Householder 변환 (rank space)
    """
    def __init__(
        self,
        d_model: int = 256,
        rank: int = 64,
        n_input: int = 8,
        n_process: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.n_input = n_input
        self.n_process = n_process

        # InputNeuron: [n_input, d_model, rank] - 차원 압축
        self.input_neurons = nn.Parameter(torch.zeros(n_input, d_model, rank))

        # ProcessNeuron: [n_process, rank] - Householder 반사 벡터 (rank space)
        self.process_neurons = nn.Parameter(torch.zeros(n_process, rank))

        self._init_weights()

    def _init_weights(self):
        """직교 초기화"""
        # InputNeuron: 직교 행렬로 초기화 [n_input, d_model, rank]
        for i in range(self.n_input):
            if self.d_model >= self.rank:
                q, _ = torch.linalg.qr(torch.randn(self.d_model, self.rank))
                self.input_neurons.data[i] = q
            else:
                q, _ = torch.linalg.qr(torch.randn(self.rank, self.d_model))
                self.input_neurons.data[i] = q.T

        # ProcessNeuron: 단위 벡터로 초기화
        for i in range(self.n_process):
            v = torch.randn(self.rank)
            v = v / (v.norm() + 1e-8)
            self.process_neurons.data[i] = v

    def apply_householder(self, x, v):
        """Householder 변환: H @ x = x - 2 * v * (v.T @ x) / ||v||²"""
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + 1e-8
        v_normalized = v / v_norm_sq.sqrt()
        vTx = (x * v_normalized).sum(dim=-1, keepdim=True)
        return x - 2 * v_normalized * vTx

    def forward(self, x, input_idx=None, input_weights=None, process_indices=None):
        """
        Args:
            x: [B, S, d_model]
            input_idx: [B, S] - hard selection
            input_weights: [B, S, n_input] - soft selection
            process_indices: [B, S, k] - Householder indices

        Returns:
            output: [B, S, rank]
        """
        B, S, D = x.shape

        # 1. Input projection: d_model → rank
        if input_weights is not None:
            all_proj = torch.einsum('bsd,ndr->bsnr', x, self.input_neurons)
            weights = input_weights.unsqueeze(-1)
            x = (all_proj * weights).sum(dim=2)
        else:
            selected = self.input_neurons[input_idx]
            x = torch.einsum('bsd,bsdr->bsr', x, selected)

        # 2. Process: Householder transforms on rank space
        if process_indices is not None:
            k = process_indices.shape[-1]
            idx_expanded = process_indices.unsqueeze(-1).expand(B, S, k, self.rank)
            selected_v = self.process_neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
            selected_v = selected_v.gather(2, idx_expanded)

            for i in range(k):
                v = selected_v[:, :, i, :]
                x = self.apply_householder(x, v)

        return x

    def orthogonality_loss(self):
        """InputNeuron 직교성 유지"""
        WtW = torch.bmm(
            self.input_neurons.transpose(-1, -2),
            self.input_neurons
        )
        I = torch.eye(self.rank, device=self.input_neurons.device).unsqueeze(0)
        return ((WtW - I) ** 2).mean()

    def process_norm_loss(self):
        """ProcessNeuron ||v|| ≈ 1 유지"""
        norms = self.process_neurons.norm(dim=-1)
        return ((norms - 1.0) ** 2).mean()


class NeuronCircuitUp(nn.Module):
    """
    NeuronCircuitUp: rank → d_model 복원용 (O projection)

    순서: Householder(rank) → Output projection
    - ProcessNeuron: Householder 변환 (rank space) - 파라미터 효율적
    - OutputNeuron: 차원 복원 (rank → d_model)

    Down과 대칭 구조:
    - Down: Input(d_model→rank) → Householder(rank)
    - Up:   Householder(rank) → Output(rank→d_model)
    """
    def __init__(
        self,
        rank: int = 64,
        d_model: int = 256,
        n_output: int = 8,
        n_process: int = 32,
    ):
        super().__init__()
        self.rank = rank
        self.d_model = d_model
        self.n_output = n_output
        self.n_process = n_process

        # ProcessNeuron: [n_process, rank] - Householder 반사 벡터 (rank space)
        # 파라미터 효율: [32, 64] vs [32, 256]
        self.process_neurons = nn.Parameter(torch.zeros(n_process, rank))

        # OutputNeuron: [n_output, rank, d_model] - 차원 복원
        self.output_neurons = nn.Parameter(torch.zeros(n_output, rank, d_model))

        self._init_weights()

    def _init_weights(self):
        """직교 초기화"""
        # ProcessNeuron: 단위 벡터로 초기화 (rank space)
        for i in range(self.n_process):
            v = torch.randn(self.rank)
            v = v / (v.norm() + 1e-8)
            self.process_neurons.data[i] = v

        # OutputNeuron: [n_output, rank, d_model]
        for i in range(self.n_output):
            if self.rank >= self.d_model:
                q, _ = torch.linalg.qr(torch.randn(self.rank, self.d_model))
                self.output_neurons.data[i] = q
            else:
                q, _ = torch.linalg.qr(torch.randn(self.d_model, self.rank))
                self.output_neurons.data[i] = q.T

    def apply_householder(self, x, v):
        """Householder 변환: H @ x = x - 2 * v * (v.T @ x) / ||v||²"""
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + 1e-8
        v_normalized = v / v_norm_sq.sqrt()
        vTx = (x * v_normalized).sum(dim=-1, keepdim=True)
        return x - 2 * v_normalized * vTx

    def forward(self, x, output_idx=None, output_weights=None, process_indices=None):
        """
        Args:
            x: [B, S, rank]
            output_idx: [B, S] - hard selection
            output_weights: [B, S, n_output] - soft selection
            process_indices: [B, S, k] - Householder indices

        Returns:
            output: [B, S, d_model]
        """
        B, S, _ = x.shape

        # 1. Process: Householder transforms on rank space (FIRST)
        if process_indices is not None:
            k = process_indices.shape[-1]
            idx_expanded = process_indices.unsqueeze(-1).expand(B, S, k, self.rank)
            selected_v = self.process_neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
            selected_v = selected_v.gather(2, idx_expanded)

            for i in range(k):
                v = selected_v[:, :, i, :]
                x = self.apply_householder(x, v)

        # 2. Output projection: rank → d_model (SECOND)
        if output_weights is not None:
            all_proj = torch.einsum('bsr,nrd->bsnd', x, self.output_neurons)
            weights = output_weights.unsqueeze(-1)
            x = (all_proj * weights).sum(dim=2)
        else:
            selected = self.output_neurons[output_idx]
            x = torch.einsum('bsr,bsrd->bsd', x, selected)

        return x

    def orthogonality_loss(self):
        """OutputNeuron 직교성 유지"""
        WtW = torch.bmm(
            self.output_neurons,
            self.output_neurons.transpose(-1, -2)
        )
        I = torch.eye(self.rank, device=self.output_neurons.device).unsqueeze(0)
        return ((WtW - I) ** 2).mean()

    def process_norm_loss(self):
        """ProcessNeuron ||v|| ≈ 1 유지"""
        norms = self.process_neurons.norm(dim=-1)
        return ((norms - 1.0) ** 2).mean()


class CircuitRouterDown(nn.Module):
    """Down circuit용 라우터 (Input + Process 선택)"""
    def __init__(
        self,
        d_model: int,
        n_input: int = 8,
        n_process: int = 32,
        process_k: int = 3,
        use_soft_selection: bool = True,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_process = n_process
        self.process_k = process_k
        self.use_soft_selection = use_soft_selection

        self.input_router = nn.Linear(d_model, n_input, bias=False)
        self.process_router = nn.Linear(d_model, n_process, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, S, d_model]
        Returns:
            routing_info: dict with input_weights/idx, process_indices
        """
        input_scores = self.input_router(x)
        process_scores = self.process_router(x)

        _, process_indices = torch.topk(process_scores, self.process_k, dim=-1)

        routing_info = {'process_indices': process_indices}

        if self.use_soft_selection:
            routing_info['input_weights'] = F.softmax(input_scores, dim=-1)
        else:
            routing_info['input_idx'] = input_scores.argmax(dim=-1)

        return routing_info


class CircuitRouterUp(nn.Module):
    """Up circuit용 라우터 (Output + Process 선택)"""
    def __init__(
        self,
        rank: int,
        n_output: int = 8,
        n_process: int = 32,
        process_k: int = 3,
        use_soft_selection: bool = True,
    ):
        super().__init__()
        self.n_output = n_output
        self.n_process = n_process
        self.process_k = process_k
        self.use_soft_selection = use_soft_selection

        self.output_router = nn.Linear(rank, n_output, bias=False)
        self.process_router = nn.Linear(rank, n_process, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, S, rank]
        Returns:
            routing_info: dict with output_weights/idx, process_indices
        """
        output_scores = self.output_router(x)
        process_scores = self.process_router(x)

        _, process_indices = torch.topk(process_scores, self.process_k, dim=-1)

        routing_info = {'process_indices': process_indices}

        if self.use_soft_selection:
            routing_info['output_weights'] = F.softmax(output_scores, dim=-1)
        else:
            routing_info['output_idx'] = output_scores.argmax(dim=-1)

        return routing_info


class NeuronCircuitQKV(nn.Module):
    """
    NeuronCircuit 기반 동적 Q/K/V/O 생성

    Q/K/V: NeuronCircuitDown (d_model → rank)
    O: NeuronCircuitUp (rank → d_model)
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

        # Q/K/V: d_model → rank
        self.circuit_Q = NeuronCircuitDown(d_model, rank, n_input, n_process)
        self.circuit_K = NeuronCircuitDown(d_model, rank, n_input, n_process)
        self.circuit_V = NeuronCircuitDown(d_model, rank, n_input, n_process)

        # O: rank → d_model
        self.circuit_O = NeuronCircuitUp(rank, d_model, n_output, n_process)

        # 라우터
        self.router_down = CircuitRouterDown(d_model, n_input, n_process, process_k, use_soft_selection)
        self.router_up = CircuitRouterUp(rank, n_output, n_process, process_k, use_soft_selection)

        self.dropout = nn.Dropout(dropout)
        self.basis_rank = rank  # 분석 스크립트 호환용

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

        # 1. Down routing
        routing_down = self.router_down(x)
        input_weights = routing_down.get('input_weights')
        input_idx = routing_down.get('input_idx')
        process_indices_down = routing_down['process_indices']

        # 2. Q/K/V: d_model → rank
        Q = self.circuit_Q(x, input_idx, input_weights, process_indices_down)
        K = self.circuit_K(x, input_idx, input_weights, process_indices_down)
        V = self.circuit_V(x, input_idx, input_weights, process_indices_down)

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

        # 6. Up routing & O projection: rank → d_model
        routing_up = self.router_up(attn_out)
        output_weights = routing_up.get('output_weights')
        output_idx = routing_up.get('output_idx')
        process_indices_up = routing_up['process_indices']

        attn_out = self.circuit_O(attn_out, output_idx, output_weights, process_indices_up)

        routing_info = {
            'routing_down': routing_down,
            'routing_up': routing_up,
            # For compatibility with train.py load balance loss
            'neuron_indices': process_indices_down,  # [B, S, k]
        }

        return attn_out, routing_info


class TransformerBlock(nn.Module):
    """Transformer Block: NeuronCircuitQKV + FFN"""
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

        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x_norm = self.norm1(x)
        attn_out, routing_info = self.qkv_circuit(x_norm, mask)
        x = residual + self.dropout(attn_out)

        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.w_down(F.gelu(self.w_up(x_norm)))
        x = residual + self.dropout(ffn_out)

        return x, routing_info


class DAWN(nn.Module):
    """DAWN v7.9 - NeuronCircuit with Householder Transformations"""
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
        self.n_neurons = n_input

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
        """Input/Output Neuron 직교성 유지"""
        total_loss = 0.0
        count = 0

        for layer in self.layers:
            qkv = layer.qkv_circuit
            for circuit in [qkv.circuit_Q, qkv.circuit_K, qkv.circuit_V]:
                total_loss += circuit.orthogonality_loss()
                count += 1
            total_loss += qkv.circuit_O.orthogonality_loss()
            count += 1

        return total_loss / count if count > 0 else 0.0

    def process_norm_loss(self):
        """ProcessNeuron ||v|| ≈ 1 유지"""
        total_loss = 0.0
        count = 0

        for layer in self.layers:
            qkv = layer.qkv_circuit
            for circuit in [qkv.circuit_Q, qkv.circuit_K, qkv.circuit_V]:
                total_loss += circuit.process_norm_loss()
                count += 1
            total_loss += qkv.circuit_O.process_norm_loss()
            count += 1

        return total_loss / count if count > 0 else 0.0

    def get_auxiliary_losses(self):
        return {
            'orth_total': self.orthogonality_loss(),
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
