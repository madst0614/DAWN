"""
DAWN v7.8 - Independent Neuron Projections (No Basis Mixing)

핵심 변경:
- Basis 제거: recipe @ basis 혼합 없음
- 뉴런별 독립 W_Q/K/V/O: 각 뉴런이 완전한 projection 행렬 소유
- 직교 초기화: 각 뉴런의 W에 orthogonal init

v7.7 대비 변경점:
- SharedBasis 제거 → NeuronBank (뉴런별 독립 W)
- recipe mixing 제거 → 뉴런 W의 가중 평균
- 파라미터 증가: ~11M → ~15M (n_neurons=64로 조절)

구조:
    입력 x → 라우터(x) → 뉴런 선택 (top-k)
    → 선택된 뉴런의 W_Q/K/V 가져오기
    → 가중 평균으로 W_mixed 생성
    → Q/K/V projection → Attention
    → W_O 가중 평균 → 차원 복원 → FFN

장점:
- Condition number 폭발 방지 (basis mixing 없음)
- 뉴런별 독립적인 표현력
- 직교 초기화 유지 가능
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NeuronBank(nn.Module):
    """
    뉴런별 독립 W_Q/K/V/O 저장소

    각 뉴런이 완전한 projection 행렬 소유
    Basis mixing 없이 뉴런 단위로만 혼합
    """
    def __init__(self, n_neurons: int, d_model: int, rank: int):
        super().__init__()
        self.n_neurons = n_neurons
        self.d_model = d_model
        self.rank = rank

        # 뉴런별 독립 projection matrices
        # W_Q, W_K, W_V: [n_neurons, d_model, rank] - 차원 축소
        # W_O: [n_neurons, rank, d_model] - 차원 복원
        self.W_Q = nn.Parameter(torch.zeros(n_neurons, d_model, rank))
        self.W_K = nn.Parameter(torch.zeros(n_neurons, d_model, rank))
        self.W_V = nn.Parameter(torch.zeros(n_neurons, d_model, rank))
        self.W_O = nn.Parameter(torch.zeros(n_neurons, rank, d_model))

        # 직교 초기화
        self._orthogonal_init()

    def _orthogonal_init(self):
        """각 뉴런의 W를 직교 행렬로 초기화"""
        for i in range(self.n_neurons):
            # QR 분해로 직교 행렬 생성
            q, _ = torch.linalg.qr(torch.randn(self.d_model, self.rank))
            self.W_Q.data[i] = q

            q, _ = torch.linalg.qr(torch.randn(self.d_model, self.rank))
            self.W_K.data[i] = q

            q, _ = torch.linalg.qr(torch.randn(self.d_model, self.rank))
            self.W_V.data[i] = q

            # W_O는 rank → d_model이므로 transpose 형태
            q, _ = torch.linalg.qr(torch.randn(self.d_model, self.rank))
            self.W_O.data[i] = q.T  # [rank, d_model]

    def get_W_Q(self, indices):
        """선택된 뉴런의 W_Q 반환: [B, S, k, D, rank]"""
        return self.W_Q[indices]

    def get_W_K(self, indices):
        """선택된 뉴런의 W_K 반환: [B, S, k, D, rank]"""
        return self.W_K[indices]

    def get_W_V(self, indices):
        """선택된 뉴런의 W_V 반환: [B, S, k, D, rank]"""
        return self.W_V[indices]

    def get_W_O(self, indices):
        """선택된 뉴런의 W_O 반환: [B, S, k, rank, D]"""
        return self.W_O[indices]

    def orthogonality_loss(self):
        """
        각 뉴런 W의 직교성 유지 loss

        W @ W.T ≈ I 유지
        """
        loss = 0.0

        for W in [self.W_Q, self.W_K, self.W_V]:
            # W: [n_neurons, D, rank]
            # W @ W.T: [n_neurons, D, D] - 너무 큼
            # 대신 W.T @ W: [n_neurons, rank, rank] 사용
            WtW = torch.bmm(W.transpose(-1, -2), W)  # [n_neurons, rank, rank]
            I = torch.eye(self.rank, device=W.device).unsqueeze(0)
            loss += ((WtW - I) ** 2).mean()

        # W_O: [n_neurons, rank, D]
        WtW_O = torch.bmm(self.W_O, self.W_O.transpose(-1, -2))  # [n_neurons, rank, rank]
        I = torch.eye(self.rank, device=self.W_O.device).unsqueeze(0)
        loss += ((WtW_O - I) ** 2).mean()

        return loss / 4


class NeuronBasedQKV(nn.Module):
    """
    뉴런 기반 동적 Q/K/V/O 생성 (v7.8)

    Basis mixing 없이 뉴런 W의 가중 평균만 사용
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_neurons: int,
        neuron_k: int,
        rank: int,
        neuron_bank: NeuronBank,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = rank // n_heads
        self.n_neurons = n_neurons
        self.k = neuron_k
        self.rank = rank
        self.neuron_bank = neuron_bank

        # 라우터
        self.W_router = nn.Linear(d_model, n_neurons, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 분석 스크립트 호환용 속성
        self.basis_rank = rank

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, S, D]
            mask: [B, 1, S, S] causal mask

        Returns:
            attn_out: [B, S, D]
            routing_info: dict

        Memory-optimized: Project all neurons first, then gather selected.
        Before: W[topk_idx] creates [B, S, k, D, rank] = huge
        After: x @ W creates [B, S, n_neurons, rank] = much smaller
        """
        B, S, D = x.shape

        # 1. 라우팅 - 뉴런 선택
        scores = self.W_router(x)  # [B, S, n_neurons]
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)  # [B, S, k]
        weights = F.softmax(topk_scores, dim=-1)  # [B, S, k]

        # 2. 전체 뉴런 projection 먼저 계산 (메모리 효율적)
        # W_Q: [n_neurons, D, rank] -> all_Q: [B, S, n_neurons, rank]
        all_Q = torch.einsum('bsd,ndr->bsnr', x, self.neuron_bank.W_Q)
        all_K = torch.einsum('bsd,ndr->bsnr', x, self.neuron_bank.W_K)
        all_V = torch.einsum('bsd,ndr->bsnr', x, self.neuron_bank.W_V)

        # 3. 선택된 뉴런만 gather
        # topk_idx: [B, S, k] -> expand to [B, S, k, rank]
        idx_expanded = topk_idx.unsqueeze(-1).expand(B, S, self.k, self.rank)
        Q_selected = all_Q.gather(2, idx_expanded)  # [B, S, k, rank]
        K_selected = all_K.gather(2, idx_expanded)
        V_selected = all_V.gather(2, idx_expanded)

        # 4. 가중 평균으로 혼합
        weights_expanded = weights.unsqueeze(-1)  # [B, S, k, 1]
        Q = (Q_selected * weights_expanded).sum(dim=2)  # [B, S, rank]
        K = (K_selected * weights_expanded).sum(dim=2)
        V = (V_selected * weights_expanded).sum(dim=2)

        # 5. Multi-head reshape
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # 6. Attention
        attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = attn_weights @ V  # [B, H, S, d_head]

        # 7. Concat
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.rank)  # [B, S, rank]

        # 8. O projection - 전체 계산 후 gather
        # W_O: [n_neurons, rank, D] -> all_O: [B, S, n_neurons, D]
        all_O = torch.einsum('bsr,nrd->bsnd', attn_out, self.neuron_bank.W_O)
        idx_expanded_d = topk_idx.unsqueeze(-1).expand(B, S, self.k, D)
        O_selected = all_O.gather(2, idx_expanded_d)  # [B, S, k, D]
        attn_out = (O_selected * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, D]

        routing_info = {
            'neuron_indices': topk_idx,
            'neuron_weights': weights,
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
        rank: int,
        neuron_bank: NeuronBank,
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
            rank=rank,
            neuron_bank=neuron_bank,
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
    DAWN v7.8 - Independent Neuron Projections

    Basis mixing 제거, 뉴런별 독립 W_Q/K/V/O
    """
    __version__ = "7.8"

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 128,
        n_neurons: int = 64,  # 파라미터 조절을 위해 64로 감소
        neuron_k: int = 8,
        rank: int = 64,  # basis_rank 대신 rank
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
        self.rank = rank

        # 분석 스크립트 호환용
        self.n_basis = n_neurons  # v7.8에서는 n_neurons와 동일
        self.basis_rank = rank

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Neuron Bank (모든 layer가 공유)
        self.neuron_bank = NeuronBank(n_neurons, d_model, rank)

        # 분석 스크립트 호환용 alias
        self.shared_basis = self.neuron_bank

        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                n_neurons=n_neurons,
                neuron_k=neuron_k,
                rank=rank,
                neuron_bank=self.neuron_bank,
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

        if return_routing_info:
            return (loss, logits, routing_infos) if labels is not None else (logits, routing_infos)
        else:
            return (loss, logits) if labels is not None else logits

    def orthogonality_loss(self):
        """뉴런 W의 직교성 유지"""
        return self.neuron_bank.orthogonality_loss()

    def neuron_diversity_loss(self):
        """
        뉴런 간 W 다양성 강제

        각 뉴런이 서로 다른 projection을 학습하도록 유도
        """
        total_loss = 0.0

        for W in [self.neuron_bank.W_Q, self.neuron_bank.W_K,
                  self.neuron_bank.W_V, self.neuron_bank.W_O]:

            # W: [n_neurons, D, rank] or [n_neurons, rank, D]
            W_flat = W.view(self.n_neurons, -1)  # [n_neurons, D*rank]

            # 뉴런 간 코사인 유사도
            W_norm = F.normalize(W_flat, dim=-1)
            sim_matrix = W_norm @ W_norm.T  # [n_neurons, n_neurons]

            # 대각선 제외
            mask = ~torch.eye(self.n_neurons, dtype=torch.bool, device=W.device)

            # 유사도 낮추기
            total_loss += sim_matrix[mask].mean()

        return total_loss / 4

    def get_auxiliary_losses(self):
        """모든 보조 loss 반환"""
        return {
            'orthogonality': self.orthogonality_loss(),
            'diversity': self.neuron_diversity_loss(),
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
            'n_neurons': self.n_neurons,
            'neuron_k': self.neuron_k,
            'rank': self.rank,
        }
