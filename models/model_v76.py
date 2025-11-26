"""
DAWN v7.6 - Independent O Basis + Recipe Diversity

핵심 변경:
- 압축용(down)과 복원용(up) basis 분리하여 독립 학습
- Q/K/V는 basis_down 사용, O는 basis_up 사용
- Recipe Diversity Loss 추가: 뉴런 간 다양성 강제

v7.5 대비 변경점:
- SharedBasis: basis_down / basis_up 분리
- NeuronBasedQKV: get_basis_down() / get_basis_up() 사용
- DAWN: recipe_diversity_loss() 추가
- 예상 파라미터 증가: ~18% (basis 2배)

구조:
    입력 x → 라우터(x) → 뉴런 선택 → recipe_Q/K/V/O 조합
    → W_Q/K/V (basis_down) → Q/K/V 계산 → Attention
    → W_O (basis_up, 독립 학습) → 차원 복원 → FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SharedBasis(nn.Module):
    """
    공유 Basis (모든 layer가 사용)
    압축용(down)과 복원용(up) 분리하여 독립 학습
    """
    def __init__(self, n_basis: int, d_model: int, basis_rank: int):
        super().__init__()
        self.n_basis = n_basis
        self.d_model = d_model
        self.basis_rank = basis_rank

        # 압축용 Basis: D → rank
        # Q, K, V 생성에 사용
        basis_down = torch.zeros(n_basis, d_model, basis_rank)
        for i in range(n_basis):
            q, _ = torch.linalg.qr(torch.randn(d_model, basis_rank))
            basis_down[i] = q
        self.basis_down = nn.Parameter(basis_down)

        # 복원용 Basis: rank → D
        # O projection에 사용, 독립적으로 학습
        basis_up = torch.zeros(n_basis, basis_rank, d_model)
        for i in range(n_basis):
            q, _ = torch.linalg.qr(torch.randn(basis_rank, d_model))
            basis_up[i] = q
        self.basis_up = nn.Parameter(basis_up)

    def get_basis_down(self):
        """압축용 basis 반환 (Q, K, V용)"""
        return self.basis_down

    def get_basis_up(self):
        """복원용 basis 반환 (O용)"""
        return self.basis_up

    def forward(self):
        """기존 호환성 위해 압축용 반환"""
        return self.basis_down

    def orthogonality_loss(self):
        """
        양쪽 Basis 직교성 유지
        """
        loss = 0.0

        # 압축용 basis 직교성
        B_down = self.basis_down.view(self.n_basis, -1)
        gram_down = B_down @ B_down.T
        I = torch.eye(self.n_basis, device=gram_down.device)
        loss += ((gram_down - I) ** 2).mean()

        # 복원용 basis 직교성
        B_up = self.basis_up.view(self.n_basis, -1)
        gram_up = B_up @ B_up.T
        loss += ((gram_up - I) ** 2).mean()

        return loss / 2


class NeuronBasedQKV(nn.Module):
    """
    뉴런 기반 동적 Q/K/V/O 생성

    압축(Q/K/V)과 복원(O)이 독립된 basis 사용
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_neurons: int,
        neuron_k: int,
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

        # 라우터
        self.W_router = nn.Linear(d_model, n_neurons, bias=False)

        # 뉴런별 Recipe (Q/K/V/O 각각 독립)
        self.neuron_recipe_Q = nn.Parameter(torch.randn(n_neurons, n_basis) * 0.5)
        self.neuron_recipe_K = nn.Parameter(torch.randn(n_neurons, n_basis) * 0.5 + 0.1)
        self.neuron_recipe_V = nn.Parameter(torch.randn(n_neurons, n_basis) * 0.5 - 0.1)
        self.neuron_recipe_O = nn.Parameter(torch.randn(n_neurons, n_basis) * 0.5 + 0.05)

        self.dropout = nn.Dropout(dropout)

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

        # 1. 라우팅
        scores = self.W_router(x)
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        # 2. Q/K/V Recipe 가져오기
        recipe_Q = self.neuron_recipe_Q[topk_idx]
        recipe_K = self.neuron_recipe_K[topk_idx]
        recipe_V = self.neuron_recipe_V[topk_idx]

        # 3. 가중 평균으로 토큰별 recipe
        token_recipe_Q = (recipe_Q * weights.unsqueeze(-1)).sum(dim=2)
        token_recipe_K = (recipe_K * weights.unsqueeze(-1)).sum(dim=2)
        token_recipe_V = (recipe_V * weights.unsqueeze(-1)).sum(dim=2)

        # 4. Softmax
        token_recipe_Q = F.softmax(token_recipe_Q, dim=-1)
        token_recipe_K = F.softmax(token_recipe_K, dim=-1)
        token_recipe_V = F.softmax(token_recipe_V, dim=-1)

        # 5. 압축용 basis로 W_Q/K/V 생성
        basis_down = self.shared_basis.get_basis_down()  # [n_basis, D, rank]

        W_Q = torch.einsum('bsn,ndr->bsdr', token_recipe_Q, basis_down)
        W_K = torch.einsum('bsn,ndr->bsdr', token_recipe_K, basis_down)
        W_V = torch.einsum('bsn,ndr->bsdr', token_recipe_V, basis_down)

        # 6. Q, K, V 생성
        Q = torch.einsum('bsd,bsdr->bsr', x, W_Q)
        K = torch.einsum('bsd,bsdr->bsr', x, W_K)
        V = torch.einsum('bsd,bsdr->bsr', x, W_V)

        # 7. Multi-head reshape
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # 8. Attention
        attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = attn_weights @ V

        # 9. Concat
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.basis_rank)

        # 10. O projection - 독립 학습된 복원용 basis 사용
        recipe_O = self.neuron_recipe_O[topk_idx]
        token_recipe_O = (recipe_O * weights.unsqueeze(-1)).sum(dim=2)
        token_recipe_O = F.softmax(token_recipe_O, dim=-1)

        # 핵심 변경: transpose 대신 독립 basis_up 사용
        basis_up = self.shared_basis.get_basis_up()  # [n_basis, rank, D]
        W_O = torch.einsum('bsn,nrd->bsrd', token_recipe_O, basis_up)
        attn_out = torch.einsum('bsr,bsrd->bsd', attn_out, W_O)

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
    DAWN v7.6 - Independent O Basis + Recipe Diversity
    """
    __version__ = "7.6"

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

        # Shared Basis (압축/복원 분리)
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
        """Basis 직교성 유지 (압축/복원 둘 다)"""
        return self.shared_basis.orthogonality_loss()

    def recipe_diversity_loss(self):
        """
        뉴런 간 recipe 다양성 강제

        각 뉴런이 서로 다른 basis 조합을 사용하도록 유도
        유사도가 낮을수록 좋음
        """
        total_loss = 0.0
        count = 0

        for layer in self.layers:
            qkv = layer.qkv_dynamic

            for recipe in [qkv.neuron_recipe_Q, qkv.neuron_recipe_K,
                           qkv.neuron_recipe_V, qkv.neuron_recipe_O]:

                # Softmax로 정규화된 recipe
                recipe_norm = F.softmax(recipe, dim=-1)  # [n_neurons, n_basis]

                # 뉴런 간 코사인 유사도 행렬
                # 정규화된 recipe끼리 내적 = 코사인 유사도
                recipe_normalized = F.normalize(recipe_norm, dim=-1)
                sim_matrix = recipe_normalized @ recipe_normalized.T  # [n_neurons, n_neurons]

                # 대각선 제외 (자기 자신과의 유사도 = 1)
                mask = ~torch.eye(self.n_neurons, dtype=torch.bool, device=recipe.device)

                # 평균 유사도를 낮추는 게 목표
                total_loss += sim_matrix[mask].mean()
                count += 1

        return total_loss / count

    def get_auxiliary_losses(self):
        """모든 보조 loss 반환"""
        return {
            'orthogonality': self.orthogonality_loss(),
            'diversity': self.recipe_diversity_loss(),
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
            'n_basis': self.n_basis,
            'basis_rank': self.basis_rank,
        }
