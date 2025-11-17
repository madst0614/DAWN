"""
SPROUT Local Manifold Architecture

핵심 아이디어:
각 뉴런이 세 가지를 소유:
1. neuron_patterns[i]: 입력 패턴 (W_in, d_model 차원) - "어떤 입력에 반응하는가"
2. neuron_vecs[i]: 의미 정보 (d_neuron 차원) - "이 뉴런이 무엇을 의미하는가"
3. neuron_out_dirs[i]: 출력 방향 (W_out, d_model 차원) - "어디로 출력하는가"

과정:
1. Router로 국소 뉴런 선택
2. Base activation: GELU(x @ neuron_patterns[i]) - 입력 매칭
3. Semantic interaction: Self-Attention on neuron_vecs - 뉴런 간 상호작용
4. Semantic modulation: attended_semantics → weight - 맥락 기반 조정
5. Final activation: base_activation * semantic_weight
6. 각 뉴런의 기여: final_activation[i] * neuron_out_dirs[i]
7. 합산하여 최종 출력

기존 FFN과의 대응:
- 기존: GELU(x @ W1[:, i]) * W2[:, i]
- 지금: (GELU(x @ pattern[i]) * semantic_weight[i]) * out_dir[i]
       = base_activation * modulation * output_direction
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
    - neuron_patterns: 입력 패턴 (어떤 입력에 반응)
    - neuron_vecs: 의미 정보 (뉴런이 무엇을 의미)
    - neuron_out_dirs: 출력 방향 (어디로 출력)

    과정:
    1. Router로 국소 뉴런 선택
    2. Base activation: 입력 패턴 매칭
    3. Semantic interaction: Self-Attention으로 의미 상호작용
    4. Semantic modulation: 상호작용 결과 → 가중치
    5. Final activation: base × modulation
    6. 각 뉴런의 기여: activation × output_direction
    7. 합산
    """
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        d_neuron: int = 128,     # 뉴런 의미 벡터 차원
        n_attn_heads: int = 8,   # 뉴런 간 Attention heads
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_neuron = d_neuron
        self.n_attn_heads = n_attn_heads

        # ✨ 각 뉴런이 소유하는 정보
        self.neuron_patterns = nn.Embedding(d_ff, d_model)   # 입력 패턴 (W_in)
        self.neuron_vecs = nn.Embedding(d_ff, d_neuron)      # 의미 정보
        self.neuron_out_dirs = nn.Embedding(d_ff, d_model)   # 출력 방향 (W_out)

        # Router (입력 패턴으로 초기화)
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
        nn.init.xavier_normal_(self.neuron_patterns.weight)
        nn.init.xavier_normal_(self.neuron_vecs.weight)
        nn.init.xavier_normal_(self.neuron_out_dirs.weight)

        # Router를 입력 패턴으로 초기화
        if hasattr(self.router, 'W_router_2'):
            self.router.W_router_2.data = self.neuron_patterns.weight.data.clone()
        elif hasattr(self.router, 'W_router'):
            self.router.W_router.data = self.neuron_patterns.weight.data.clone()

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
        """Dense 연산: 모든 뉴런 사용 (chunked for memory efficiency)"""
        batch_seq = x_flat.shape[0]

        # 1. 모든 뉴런의 정보 가져오기
        neuron_patterns = self.neuron_patterns.weight  # [d_ff, d_model]
        neuron_semantics = self.neuron_vecs.weight     # [d_ff, d_neuron]
        neuron_out_dirs = self.neuron_out_dirs.weight  # [d_ff, d_model]

        # ✨ Chunking으로 메모리 절약
        chunk_size = batch  # 한 번에 1 batch만 처리

        outputs = []
        for chunk_start in range(0, batch_seq, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_seq)
            x_chunk = x_flat[chunk_start:chunk_end]  # [chunk, d_model]

            # 2. ✨ Base activation (입력 패턴 매칭)
            base_activations = F.gelu(x_chunk @ neuron_patterns.T)
            # [chunk, d_ff]

            # 3. ✨ 의미 정보로 상호작용 (Self-Attention)
            # Expand semantics for chunk
            chunk_len = x_chunk.shape[0]
            semantics_expanded = neuron_semantics.unsqueeze(0).expand(chunk_len, -1, -1)
            # [chunk, d_ff, d_neuron]

            attended_semantics, _ = self.neuron_attn(
                semantics_expanded,
                semantics_expanded,
                semantics_expanded
            )
            # [chunk, d_ff, d_neuron]

            # 4. ✨ Semantic modulation (attended → weight)
            semantic_weights = self.weight_proj(attended_semantics)
            # [chunk, d_ff, 1]

            semantic_weights = torch.sigmoid(semantic_weights)

            # 5. ✨ Final activation = base × semantic modulation
            final_activations = base_activations.unsqueeze(-1) * semantic_weights
            # [chunk, d_ff, 1]

            # 6. ✨ 각 뉴런의 기여
            out_dirs_expanded = neuron_out_dirs.unsqueeze(0).expand(chunk_len, -1, -1)
            contributions = final_activations * out_dirs_expanded
            # [chunk, d_ff, d_model]

            # 7. 합산
            output_chunk = contributions.sum(dim=1)
            # [chunk, d_model]

            outputs.append(output_chunk)

        # Concatenate all chunks
        output = torch.cat(outputs, dim=0)  # [B*S, d_model]

        return output.view(batch, seq, self.d_model)

    def _forward_sparse(
        self,
        x_flat: torch.Tensor,
        top_k: int,
        batch: int,
        seq: int
    ) -> torch.Tensor:
        """Sparse 연산: top_k 뉴런만 사용 (chunked for memory efficiency)"""
        batch_seq = x_flat.shape[0]

        # ✨ Chunking으로 메모리 절약
        # [B*S, top_k, top_k] attention은 메모리 집약적
        # Chunk size: 시퀀스 단위로 처리 (batch 단위)
        chunk_size = batch  # 한 번에 1 batch만 처리

        outputs = []
        for chunk_start in range(0, batch_seq, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_seq)
            x_chunk = x_flat[chunk_start:chunk_end]  # [chunk, d_model]

            # 1. Router로 뉴런 선택
            router_scores = self.router.compute_scores(x_chunk)
            _, selected_indices = torch.topk(router_scores, top_k, dim=-1)
            # [chunk, top_k]

            # 2. ✨ 선택된 뉴런들의 정보 가져오기
            neuron_patterns = self.neuron_patterns(selected_indices)   # [chunk, top_k, d_model]
            neuron_semantics = self.neuron_vecs(selected_indices)      # [chunk, top_k, d_neuron]
            neuron_out_dirs = self.neuron_out_dirs(selected_indices)   # [chunk, top_k, d_model]

            # 3. ✨ Base activation (입력 패턴 매칭)
            base_activations = torch.bmm(
                x_chunk.unsqueeze(1),           # [chunk, 1, d_model]
                neuron_patterns.transpose(1, 2) # [chunk, d_model, top_k]
            ).squeeze(1)                        # [chunk, top_k]

            base_activations = F.gelu(base_activations)

            # 4. ✨ 의미 정보로 상호작용 (Self-Attention)
            # chunk 크기만큼만 attention 수행 → 메모리 절약
            attended_semantics, _ = self.neuron_attn(
                neuron_semantics,
                neuron_semantics,
                neuron_semantics
            )
            # [chunk, top_k, d_neuron]

            # 5. ✨ Semantic modulation (attended → weight)
            semantic_weights = self.weight_proj(attended_semantics)
            # [chunk, top_k, 1]

            semantic_weights = torch.sigmoid(semantic_weights)

            # 6. ✨ Final activation = base × semantic modulation
            final_activations = base_activations.unsqueeze(-1) * semantic_weights
            # [chunk, top_k, 1]

            # 7. ✨ 각 뉴런의 기여
            contributions = final_activations * neuron_out_dirs
            # [chunk, top_k, d_model]

            # 8. 합산
            output_chunk = contributions.sum(dim=1)
            # [chunk, d_model]

            outputs.append(output_chunk)

        # Concatenate all chunks
        output = torch.cat(outputs, dim=0)  # [B*S, d_model]

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
