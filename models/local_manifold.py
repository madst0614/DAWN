"""
SPROUT Local Manifold Architecture

핵심 아이디어:
각 뉴런이 두 가지를 소유:
1. neuron_vecs[i]: 뉴런의 의미 정보 (d_neuron 차원)
2. neuron_out_dirs[i]: 뉴런의 출력 방향 (d_model 차원)

과정:
1. Router로 국소 뉴런 선택
2. 선택된 뉴런들의 neuron_vecs로 Self-Attention (상호작용)
3. Attention 출력 → 스칼라 가중치로 변환
4. 각 뉴런의 기여: weight[i] * neuron_out_dirs[i]
5. 합산하여 최종 출력

기존 FFN과의 대응:
- 기존: activation[i] * W2[:, i]
- 지금: weight[i] * neuron_out_dirs[i]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils.checkpoint import checkpoint


# ============================================================
# Router
# ============================================================

class Router(nn.Module):
    """학습 가능한 라우터"""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        use_mlp: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_mlp = use_mlp

        if use_mlp:
            self.W_router_1 = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
            self.W_router_2 = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)
            self.norm = nn.LayerNorm(d_model)
        else:
            self.W_router = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)

    def compute_scores(self, x: torch.Tensor) -> torch.Tensor:
        """라우터 점수 계산"""
        if self.use_mlp:
            x_norm = self.norm(x)
            h = F.gelu(x_norm @ self.W_router_1.T)
            scores = h @ self.W_router_2.T
        else:
            scores = x @ self.W_router.T

        return scores


# ============================================================
# Local Manifold FFN Layer
# ============================================================

class LocalManifoldFFNLayer(nn.Module):
    """
    국소 매니폴드 FFN

    각 뉴런이 소유:
    - neuron_vecs: 의미 정보
    - neuron_out_dirs: 출력 방향

    과정:
    1. 국소 뉴런 선택
    2. 뉴런 간 상호작용 (Self-Attention)
    3. 가중치 계산
    4. 각 뉴런의 기여 합산
    """
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        d_neuron: int = 128,     # 뉴런 의미 벡터 차원
        n_attn_heads: int = 4,   # 뉴런 간 Attention heads
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_neuron = d_neuron
        self.n_attn_heads = n_attn_heads

        # ✨ 각 뉴런이 소유하는 정보
        self.neuron_vecs = nn.Embedding(d_ff, d_neuron)      # 의미
        self.neuron_out_dirs = nn.Embedding(d_ff, d_model)   # 출력 방향

        # Router
        self.router = Router(d_model, d_ff, use_mlp=True)

        # ✨ 뉴런 간 상호작용 (Self-Attention)
        self.neuron_attn = nn.MultiheadAttention(
            embed_dim=d_neuron,
            num_heads=n_attn_heads,
            dropout=0.1,
            batch_first=True
        )

        # ✨ Attention 출력 → 스칼라 가중치
        self.weight_proj = nn.Sequential(
            nn.Linear(d_neuron, d_neuron // 2),
            nn.GELU(),
            nn.Linear(d_neuron // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        nn.init.xavier_normal_(self.neuron_vecs.weight)
        nn.init.xavier_normal_(self.neuron_out_dirs.weight)

    def forward(
        self,
        x: torch.Tensor,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, d_model]
            top_k: 선택할 뉴런 개수 (None이면 dense)

        Returns:
            output: [batch, seq, d_model]
        """
        batch, seq, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [B*S, d_model]

        if top_k is None or top_k >= self.d_ff:
            # Dense: 모든 뉴런 사용
            return self._forward_dense(x_flat, batch, seq)
        else:
            # Sparse: top_k 뉴런만 사용
            return self._forward_sparse(x_flat, top_k, batch, seq)

    def _forward_dense(
        self,
        x_flat: torch.Tensor,
        batch: int,
        seq: int
    ) -> torch.Tensor:
        """Dense 연산: 모든 뉴런 사용"""
        batch_seq = x_flat.shape[0]

        # 1. 모든 뉴런의 정보 가져오기
        neuron_infos = self.neuron_vecs.weight.unsqueeze(0).expand(batch_seq, -1, -1)
        # [B*S, d_ff, d_neuron]

        neuron_out_dirs = self.neuron_out_dirs.weight.unsqueeze(0).expand(batch_seq, -1, -1)
        # [B*S, d_ff, d_model]

        # 2. ✨ 뉴런 간 상호작용 (Self-Attention)
        attended_infos, _ = self.neuron_attn(
            neuron_infos,  # query
            neuron_infos,  # key
            neuron_infos   # value
        )
        # [B*S, d_ff, d_neuron]

        # 3. ✨ 가중치 계산 (attended → scalar)
        neuron_weights = self.weight_proj(attended_infos)
        # [B*S, d_ff, 1]

        # 4. ✨ 각 뉴런의 기여 계산
        contributions = neuron_weights * neuron_out_dirs
        # [B*S, d_ff, d_model]

        # 5. 합산
        output = contributions.sum(dim=1)
        # [B*S, d_model]

        return output.view(batch, seq, self.d_model)

    def _forward_sparse(
        self,
        x_flat: torch.Tensor,
        top_k: int,
        batch: int,
        seq: int
    ) -> torch.Tensor:
        """Sparse 연산: top_k 뉴런만 사용"""
        batch_seq = x_flat.shape[0]

        # 1. Router로 뉴런 선택
        router_scores = self.router.compute_scores(x_flat)
        _, selected_indices = torch.topk(router_scores, top_k, dim=-1)
        # [B*S, top_k]

        # 2. ✨ 선택된 뉴런들의 정보 가져오기
        neuron_infos = self.neuron_vecs(selected_indices)
        # [B*S, top_k, d_neuron]

        neuron_out_dirs = self.neuron_out_dirs(selected_indices)
        # [B*S, top_k, d_model]

        # 3. ✨ 뉴런 간 상호작용 (Self-Attention)
        attended_infos, _ = self.neuron_attn(
            neuron_infos,  # query
            neuron_infos,  # key
            neuron_infos   # value
        )
        # [B*S, top_k, d_neuron]

        # 4. ✨ 가중치 계산 (attended → scalar)
        neuron_weights = self.weight_proj(attended_infos)
        # [B*S, top_k, 1]

        # 5. ✨ 각 뉴런의 기여 계산
        contributions = neuron_weights * neuron_out_dirs
        # [B*S, top_k, d_model]

        # 6. 합산
        output = contributions.sum(dim=1)
        # [B*S, d_model]

        return output.view(batch, seq, self.d_model)

    def compute_router_loss(
        self,
        x: torch.Tensor,
        top_k: int,
        load_balance_weight: float = 0.01
    ) -> torch.Tensor:
        """라우터 학습을 위한 보조 loss"""
        batch, seq, _ = x.shape
        x_flat = x.view(-1, self.d_model)

        # Router scores
        router_scores = self.router.compute_scores(x_flat)

        # Load Balancing Loss
        _, selected_indices = torch.topk(router_scores, top_k, dim=-1)
        neuron_usage = torch.zeros(self.d_ff, device=x.device)
        ones = torch.ones_like(selected_indices, dtype=torch.float)
        neuron_usage.scatter_add_(0, selected_indices.flatten(), ones.flatten())

        total_selections = neuron_usage.sum()
        expected_usage = total_selections / self.d_ff
        load_balance_loss = ((neuron_usage - expected_usage).pow(2).mean())

        return load_balance_weight * load_balance_loss


# ============================================================
# Local Manifold Language Model
# ============================================================

class LocalManifoldLanguageModel(nn.Module):
    """국소 매니폴드 기반 Language Model"""
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 512,
        d_ff: int = 2048,
        n_heads: int = 8,
        n_layers: int = 6,
        max_len: int = 512,
        max_seq_len: int = None,
        dropout: float = 0.1,
        sparse_k: Optional[int] = None,
        gradient_checkpointing: bool = False,
        d_neuron: int = 128,
        n_attn_heads: int = 4,
    ):
        if max_seq_len is not None:
            max_len = max_seq_len
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.sparse_k = sparse_k
        self.gradient_checkpointing = gradient_checkpointing

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerLayerWithLocalManifoldFFN(
                d_model,
                d_ff,
                n_heads,
                dropout,
                use_checkpoint=gradient_checkpointing,
                d_neuron=d_neuron,
                n_attn_heads=n_attn_heads
            )
            for _ in range(n_layers)
        ])

        # Output head
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.output_projection.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None
    ) -> dict:
        """
        Args:
            input_ids: [batch, seq]
            attention_mask: [batch, seq]
            labels: [batch, seq]
            top_k: FFN에서 사용할 뉴런 수

        Returns:
            dict with 'logits', 'loss'
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if top_k is None:
            top_k = self.sparse_k

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask, top_k=top_k)

        x = self.norm(x)
        logits = self.output_projection(x)

        # Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        return {
            'logits': logits,
            'loss': loss
        }


# ============================================================
# Transformer Layer
# ============================================================

class TransformerLayerWithLocalManifoldFFN(nn.Module):
    """Transformer layer with Local Manifold FFN"""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        use_checkpoint: bool = False,
        d_neuron: int = 128,
        n_attn_heads: int = 4
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # ✨ Local Manifold FFN
        self.ffn = LocalManifoldFFNLayer(
            d_model,
            d_ff,
            d_neuron=d_neuron,
            n_attn_heads=n_attn_heads
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint

    def _attention_block(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = self.dropout(x)
        return x

    def _ffn_block(self, x: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
        x = self.norm2(x)
        x = self.ffn(x, top_k=top_k)
        x = self.dropout(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        # Attention
        if self.use_checkpoint and self.training:
            attn_out = checkpoint(self._attention_block, x, attention_mask, use_reentrant=False)
        else:
            attn_out = self._attention_block(x, attention_mask)
        x = x + attn_out

        # FFN
        if self.use_checkpoint and self.training:
            ffn_out = checkpoint(self._ffn_block, x, top_k, use_reentrant=False)
        else:
            ffn_out = self._ffn_block(x, top_k)
        x = x + ffn_out

        return x
