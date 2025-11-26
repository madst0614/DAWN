"""
DAWN v7.5 - Dynamic Q/K/V/O with Neuron-Based Routing

핵심 아이디어:
- 라우터가 x만 보고 뉴런 선택 (깔끔한 설계)
- 선택된 뉴런의 recipe_Q/K/V/O로 동적 W_Q/K/V/O 생성
- Q, K, V, O 모두 동적으로 생성 (완전한 동적 attention)
- 표준 Attention 사용
- basis_emb 제거, context score 제거

구조:
    입력 x → 라우터(x) → 뉴런 선택 → recipe_Q/K/V/O 조합
    → W_Q/K/V 동적 생성 → Q/K/V 계산 → Attention
    → W_O 동적 생성 (basis transpose) → 차원 복원 → FFN

v8 설계를 v7.5로 구현 + O projection도 동적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SharedBasis(nn.Module):
    """
    공유 Basis (모든 layer가 사용)
    학습 가능 - 직교성 Loss로 유지
    """
    def __init__(self, n_basis: int, d_model: int, basis_rank: int):
        super().__init__()
        self.n_basis = n_basis
        self.d_model = d_model
        self.basis_rank = basis_rank

        # Basis 초기화 (직교)
        basis = torch.zeros(n_basis, d_model, basis_rank)
        for i in range(n_basis):
            q, r = torch.linalg.qr(torch.randn(d_model, basis_rank))
            basis[i] = q

        self.basis = nn.Parameter(basis)  # [n_basis, D, rank] - 학습 가능

    def forward(self):
        return self.basis

    def orthogonality_loss(self):
        """
        Basis 간 직교성 유지를 위한 loss
        각 basis를 펼쳐서 내적 행렬이 단위행렬이 되도록 강제
        """
        # basis: [n_basis, d_model, basis_rank]
        # 각 basis를 펼쳐서 [n_basis, d_model * basis_rank]
        B = self.basis.view(self.n_basis, -1)  # [32, 256*64]

        # 내적 행렬 (Gram matrix)
        gram = B @ B.T  # [32, 32]

        # 단위행렬과의 차이
        I = torch.eye(self.n_basis, device=gram.device)
        return ((gram - I) ** 2).mean()


class NeuronBasedQKV(nn.Module):
    """
    뉴런 기반 동적 Q/K/V/O 생성

    각 뉴런이 Q/K/V/O 만드는 방법(recipe)을 학습
    라우터가 x 보고 뉴런 선택
    선택된 뉴런들의 recipe 조합해서 토큰별 동적 W_Q/K/V/O 생성

    W_Q/K/V: basis [n_basis, D, rank]로 생성
    W_O: basis transpose [n_basis, rank, D]로 생성 (차원 복원)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_neurons: int,
        neuron_k: int,  # Top-K 선택
        n_basis: int,
        basis_rank: int,
        shared_basis: SharedBasis,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = basis_rank // n_heads
        self.n_neurons = n_neurons
        self.k = neuron_k
        self.n_basis = n_basis
        self.basis_rank = basis_rank
        self.shared_basis = shared_basis

        # 라우터: x만 보고 뉴런 선택
        self.W_router = nn.Linear(d_model, n_neurons, bias=False)

        # 뉴런별 Q/K/V/O recipe (학습됨) - 초기화 분리로 다양성 확보
        self.neuron_recipe_Q = nn.Parameter(torch.randn(n_neurons, n_basis) * 0.5)
        self.neuron_recipe_K = nn.Parameter(torch.randn(n_neurons, n_basis) * 0.5 + 0.1)
        self.neuron_recipe_V = nn.Parameter(torch.randn(n_neurons, n_basis) * 0.5 - 0.1)
        self.neuron_recipe_O = nn.Parameter(torch.randn(n_neurons, n_basis) * 0.5 + 0.05)

        # Output projection은 동적으로 생성 (recipe_O 사용)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, S, D]
            mask: [B, 1, S, S] causal mask

        Returns:
            attn_out: [B, S, D]
            routing_info: dict (for analysis)
        """
        B, S, D = x.shape

        # 1. 라우팅: x만 보고 뉴런 선택
        scores = self.W_router(x)  # [B, S, n_neurons]
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)  # [B, S, k]
        weights = F.softmax(topk_scores, dim=-1)  # [B, S, k]

        # 2. Recipe 가져오기
        recipe_Q = self.neuron_recipe_Q[topk_idx]  # [B, S, k, n_basis]
        recipe_K = self.neuron_recipe_K[topk_idx]
        recipe_V = self.neuron_recipe_V[topk_idx]

        # 3. 가중 평균으로 토큰별 recipe
        token_recipe_Q = (recipe_Q * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, n_basis]
        token_recipe_K = (recipe_K * weights.unsqueeze(-1)).sum(dim=2)
        token_recipe_V = (recipe_V * weights.unsqueeze(-1)).sum(dim=2)

        # 4. Softmax로 비율화
        token_recipe_Q = F.softmax(token_recipe_Q, dim=-1)
        token_recipe_K = F.softmax(token_recipe_K, dim=-1)
        token_recipe_V = F.softmax(token_recipe_V, dim=-1)

        # 5. 동적 변환 행렬 생성
        basis = self.shared_basis()  # [n_basis, D, rank]

        # einsum: 토큰별 recipe와 basis 조합
        W_Q = torch.einsum('bsn,ndr->bsdr', token_recipe_Q, basis)  # [B, S, D, rank]
        W_K = torch.einsum('bsn,ndr->bsdr', token_recipe_K, basis)
        W_V = torch.einsum('bsn,ndr->bsdr', token_recipe_V, basis)

        # 6. Q, K, V 생성
        Q = torch.einsum('bsd,bsdr->bsr', x, W_Q)  # [B, S, rank]
        K = torch.einsum('bsd,bsdr->bsr', x, W_K)
        V = torch.einsum('bsd,bsdr->bsr', x, W_V)

        # 7. Multi-head reshape
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, S, d_head]
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # 8. Attention
        attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)  # [B, H, S, S]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, S, S]
        attn_weights = self.dropout(attn_weights)

        attn_out = attn_weights @ V  # [B, H, S, d_head]

        # 9. Concat
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.basis_rank)  # [B, S, rank]

        # 10. 동적 W_O 생성 및 복원 (rank → d_model)
        recipe_O = self.neuron_recipe_O[topk_idx]  # [B, S, k, n_basis]
        token_recipe_O = (recipe_O * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, n_basis]
        token_recipe_O = F.softmax(token_recipe_O, dim=-1)

        # basis를 transpose해서 [n_basis, rank, D] 생성
        basis_up = self.shared_basis().transpose(-1, -2)  # [n_basis, rank, D]
        W_O = torch.einsum('bsn,nrd->bsrd', token_recipe_O, basis_up)  # [B, S, rank, D]
        attn_out = torch.einsum('bsr,bsrd->bsd', attn_out, W_O)  # [B, S, D]

        # Routing info for analysis
        routing_info = {
            'neuron_indices': topk_idx,  # [B, S, k]
            'neuron_weights': weights,   # [B, S, k]
        }

        return attn_out, routing_info


class TransformerBlock(nn.Module):
    """
    Transformer Block: NeuronBasedQKV + FFN
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_neurons: int,
        neuron_k: int,
        n_basis: int,
        basis_rank: int,
        shared_basis: SharedBasis,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        # Attention with dynamic Q/K/V
        self.qkv_dynamic = NeuronBasedQKV(
            d_model=d_model,
            n_heads=n_heads,
            n_neurons=n_neurons,
            neuron_k=neuron_k,
            n_basis=n_basis,
            basis_rank=basis_rank,
            shared_basis=shared_basis,
            dropout=dropout
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
        attn_out, routing_info = self.qkv_dynamic(x_norm, mask)
        x = residual + self.dropout(attn_out)

        # FFN with residual
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.w_down(F.gelu(self.w_up(x_norm)))
        x = residual + self.dropout(ffn_out)

        return x, routing_info


class DAWN(nn.Module):
    """
    DAWN v7.5 - Dynamic Q/K/V Generation

    모든 Q/K/V를 뉴런 기반으로 동적 생성
    """
    __version__ = "7.5"

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 128,
        n_neurons: int = 128,
        neuron_k: int = 8,
        n_basis: int = 32,
        basis_rank: int = 64,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.n_neurons = n_neurons
        self.neuron_k = neuron_k
        self.n_basis = n_basis
        self.basis_rank = basis_rank

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Shared Basis (고정, 학습 안 함)
        self.shared_basis = SharedBasis(n_basis, d_model, basis_rank)

        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                n_neurons=n_neurons,
                neuron_k=neuron_k,
                n_basis=n_basis,
                basis_rank=basis_rank,
                shared_basis=self.shared_basis,
                dropout=dropout
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
        """Initialize weights"""
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

    def forward(self, input_ids, labels=None, return_routing_info=False):
        """
        Args:
            input_ids: [B, S]
            labels: [B, S] (optional, for training)
            return_routing_info: bool (for analysis)

        Returns:
            loss or logits, optionally with routing_info
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos = torch.arange(S, device=device).unsqueeze(0)  # [1, S]
        x = self.token_emb(input_ids) + self.pos_emb(pos)  # [B, S, D]
        x = self.dropout(x)

        # Causal mask
        mask = self.causal_mask[:, :, :S, :S]  # [1, 1, S, S]

        # Transformer layers
        routing_infos = []
        for layer in self.layers:
            x, routing_info = layer(x, mask)
            if return_routing_info:
                routing_infos.append(routing_info)

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)  # [B, S, vocab_size]

        # Loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

        if return_routing_info:
            return (loss, logits, routing_infos) if labels is not None else (logits, routing_infos)
        else:
            return (loss, logits) if labels is not None else logits

    def orthogonality_loss(self):
        """
        Basis 직교성 유지 loss
        모든 레이어가 같은 SharedBasis를 공유하므로 한 번만 계산
        """
        return self.shared_basis.orthogonality_loss()

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self):
        """Get model configuration"""
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'n_neurons': self.n_neurons,
            'neuron_k': self.neuron_k,
            'n_basis': self.n_basis,
            'basis_rank': self.basis_rank,
        }
