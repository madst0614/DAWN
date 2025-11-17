"""
SPROUT DeepSets-Based Architecture

각 뉴런이 학습 가능한 정보 벡터를 가지고
선택된 뉴런들의 정보를 DeepSets로 조합

핵심 아이디어:
- neuron_vecs[i]: i번째 뉴런의 학습 가능한 정보
- φ: 뉴런 정보 + activation → 중간 표현
- Σ: Sum aggregation (permutation invariant)
- ρ: 집계된 정보 → 최종 출력
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils.checkpoint import checkpoint


# ============================================================
# Router (from baseline)
# ============================================================

class Router(nn.Module):
    """학습 가능한 라우터"""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        init_from_W1: torch.Tensor = None,
        use_mlp: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_mlp = use_mlp

        if use_mlp:
            self.W_router_1 = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
            self.W_router_2 = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)

            if init_from_W1 is not None:
                self.W_router_2.data = init_from_W1.clone()

            self.norm = nn.LayerNorm(d_model)
        else:
            if init_from_W1 is not None:
                self.W_router = nn.Parameter(init_from_W1.clone())
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
# DeepSets FFN Layer
# ============================================================

class DeepSetsFFNLayer(nn.Module):
    """
    DeepSets 기반 FFN

    각 뉴런이 학습 가능한 정보 벡터를 가지고
    선택된 뉴런들의 정보를 조합하여 출력 생성

    F({neurons}) = ρ(Σ φ(neuron_vec[i], activation[i]))
    """
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        d_neuron: int = 128,   # 각 뉴런 정보 벡터 크기
        d_hidden: int = 256,   # φ 출력 크기
        use_context: bool = False  # 맥락 사용 여부
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_neuron = d_neuron
        self.d_hidden = d_hidden
        self.use_context = use_context

        # ✨ 각 뉴런의 학습 가능한 정보 벡터
        self.neuron_vecs = nn.Parameter(
            torch.randn(d_ff, d_neuron) * 0.02
        )

        # 입력 패턴 (activation 계산용)
        self.W_in = nn.Parameter(
            torch.randn(d_ff, d_model) * 0.02
        )

        # Router
        self.router = Router(d_model, d_ff, init_from_W1=self.W_in.data)

        # ✨ φ: 뉴런 정보 + activation → 중간 표현
        if use_context:
            # 맥락 포함: 뉴런 정보 + activation + 입력 + 전체 맥락
            phi_input_dim = d_neuron + 1 + d_model + d_neuron
        else:
            # 기본: 뉴런 정보 + activation
            phi_input_dim = d_neuron + 1

        self.phi = nn.Sequential(
            nn.Linear(phi_input_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden)
        )

        # ✨ ρ: 집계된 정보 → 최종 출력
        self.rho = nn.Sequential(
            nn.LayerNorm(d_hidden),  # ✨ Sum aggregation 안정화
            nn.Linear(d_hidden, d_hidden * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden * 2, d_model)
        )

    def forward(
        self,
        x: torch.Tensor,
        top_k: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, d_model]
            top_k: 선택할 뉴런 개수
            chunk_size: 청크 크기 (현재 미사용)

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
        """Dense 연산: 모든 뉴런 사용 (vectorized)"""
        batch_seq = x_flat.shape[0]

        # 모든 뉴런의 activation 계산
        activations = F.gelu(x_flat @ self.W_in.T)  # [B*S, d_ff]

        # 벡터화된 φ 입력 구성
        if self.use_context:
            # 전체 맥락
            context = self.neuron_vecs.mean(dim=0)  # [d_neuron]

            # Expand for batch processing
            neuron_vecs_expanded = self.neuron_vecs.unsqueeze(0).expand(batch_seq, -1, -1)  # [B*S, d_ff, d_neuron]
            activations_expanded = activations.unsqueeze(-1)  # [B*S, d_ff, 1]
            x_expanded = x_flat.unsqueeze(1).expand(-1, self.d_ff, -1)  # [B*S, d_ff, d_model]
            context_expanded = context.unsqueeze(0).unsqueeze(0).expand(batch_seq, self.d_ff, -1)  # [B*S, d_ff, d_neuron]

            phi_input = torch.cat([
                neuron_vecs_expanded,
                activations_expanded,
                x_expanded,
                context_expanded
            ], dim=-1)  # [B*S, d_ff, d_neuron + 1 + d_model + d_neuron]
        else:
            # 기본 버전 - 벡터화
            neuron_vecs_expanded = self.neuron_vecs.unsqueeze(0).expand(batch_seq, -1, -1)  # [B*S, d_ff, d_neuron]
            activations_expanded = activations.unsqueeze(-1)  # [B*S, d_ff, 1]

            phi_input = torch.cat([
                neuron_vecs_expanded,
                activations_expanded
            ], dim=-1)  # [B*S, d_ff, d_neuron + 1]

        # ✨ φ: 각 뉴런 독립적으로 변환 (2D reshape for efficiency)
        phi_input_2d = phi_input.view(-1, phi_input.size(-1))  # [B*S * d_ff, input_dim]
        transformed_2d = self.phi(phi_input_2d)  # [B*S * d_ff, d_hidden]
        transformed = transformed_2d.view(batch_seq, self.d_ff, -1)  # [B*S, d_ff, d_hidden]

        # ✨ Σ: Sum aggregation (permutation invariant!)
        aggregated = transformed.sum(dim=1)  # [B*S, d_hidden]

        # ✨ ρ: 최종 변환
        output = self.rho(aggregated)  # [B*S, d_model]

        return output.view(batch, seq, self.d_model)

    def _forward_sparse(
        self,
        x_flat: torch.Tensor,
        top_k: int,
        batch: int,
        seq: int
    ) -> torch.Tensor:
        """Sparse 연산: top_k 뉴런만 사용 (chunked processing)"""
        batch_seq = x_flat.shape[0]

        # Chunk size for memory efficiency
        chunk_size = 32  # Process 32 tokens at a time (balanced memory/speed)

        outputs = []

        for chunk_start in range(0, batch_seq, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_seq)
            x_chunk = x_flat[chunk_start:chunk_end]
            chunk_len = x_chunk.shape[0]

            # 1. Router로 뉴런 선택
            router_scores = self.router.compute_scores(x_chunk)
            _, selected_indices = torch.topk(router_scores, top_k, dim=-1)
            # [chunk_len, top_k]

            # 2. 선택된 뉴런의 activation 계산
            W_in_selected = self.W_in[selected_indices]  # [chunk_len, top_k, d_model]

            # Batched dot product
            activations = torch.bmm(
                x_chunk.unsqueeze(1),  # [chunk_len, 1, d_model]
                W_in_selected.transpose(1, 2)  # [chunk_len, d_model, top_k]
            ).squeeze(1)  # [chunk_len, top_k]

            activations = F.gelu(activations)

            # 3. 선택된 뉴런 정보 가져오기
            neuron_vecs_selected = self.neuron_vecs[selected_indices]
            # [chunk_len, top_k, d_neuron]

            # 4. φ 입력 구성
            if self.use_context:
                # 전체 맥락 계산
                context = neuron_vecs_selected.mean(dim=1)  # [chunk_len, d_neuron]

                # 확장
                x_expanded = x_chunk.unsqueeze(1).expand(-1, top_k, -1)
                context_expanded = context.unsqueeze(1).expand(-1, top_k, -1)

                phi_input = torch.cat([
                    neuron_vecs_selected,      # [chunk_len, top_k, d_neuron]
                    activations.unsqueeze(-1), # [chunk_len, top_k, 1]
                    x_expanded,                # [chunk_len, top_k, d_model]
                    context_expanded           # [chunk_len, top_k, d_neuron]
                ], dim=-1)  # [chunk_len, top_k, d_neuron + 1 + d_model + d_neuron]
            else:
                phi_input = torch.cat([
                    neuron_vecs_selected,      # [chunk_len, top_k, d_neuron]
                    activations.unsqueeze(-1)  # [chunk_len, top_k, 1]
                ], dim=-1)  # [chunk_len, top_k, d_neuron + 1]

            # 5. ✨ φ: 각 뉴런 독립적으로 변환
            # Reshape to 2D for memory efficiency
            phi_input_2d = phi_input.view(-1, phi_input.size(-1))  # [chunk_len * top_k, input_dim]
            transformed_2d = self.phi(phi_input_2d)  # [chunk_len * top_k, d_hidden]
            transformed = transformed_2d.view(chunk_len, top_k, -1)  # [chunk_len, top_k, d_hidden]

            # 6. ✨ Σ: Sum aggregation (permutation invariant!)
            aggregated = transformed.sum(dim=1)  # [chunk_len, d_hidden]

            # 7. ✨ ρ: 최종 변환
            chunk_output = self.rho(aggregated)  # [chunk_len, d_model]

            outputs.append(chunk_output)

            # Clean up intermediate tensors to free memory
            del phi_input, phi_input_2d, transformed_2d, transformed, aggregated
            del neuron_vecs_selected, activations, W_in_selected, router_scores, selected_indices
            if self.use_context:
                del context, x_expanded, context_expanded

            # Periodically clear CUDA cache
            if chunk_start % 64 == 0:
                torch.cuda.empty_cache()

        # Concatenate all chunks
        output = torch.cat(outputs, dim=0)  # [B*S, d_model]

        return output.view(batch, seq, self.d_model)

    def compute_router_loss(
        self,
        x: torch.Tensor,
        top_k: int,
        load_balance_weight: float = 0.01
    ) -> torch.Tensor:
        """라우터 품질 loss 계산 (baseline과 동일)"""
        batch, seq, _ = x.shape
        x_flat = x.view(-1, self.d_model)

        # Oracle scores
        with torch.no_grad():
            z_oracle = x_flat @ self.W_in.T
            a_oracle = F.gelu(z_oracle)
            oracle_scores = a_oracle.abs()

        # Router scores
        router_scores = self.router.compute_scores(x_flat)

        # Cross Entropy Loss
        _, oracle_indices = torch.topk(oracle_scores, top_k, dim=-1)
        oracle_mask = torch.zeros_like(router_scores)
        oracle_mask.scatter_(-1, oracle_indices, 1.0)

        router_scores_norm = F.log_softmax(router_scores, dim=-1)
        ce_loss = -(oracle_mask * router_scores_norm).sum() / oracle_mask.sum()

        # Ranking Loss
        oracle_scores_norm = (oracle_scores - oracle_scores.mean(dim=-1, keepdim=True)) / (oracle_scores.std(dim=-1, keepdim=True) + 1e-6)
        router_scores_norm_mse = (router_scores - router_scores.mean(dim=-1, keepdim=True)) / (router_scores.std(dim=-1, keepdim=True) + 1e-6)
        ranking_loss = ((oracle_scores_norm - router_scores_norm_mse) * oracle_mask).pow(2).sum() / oracle_mask.sum()

        # Load Balancing
        _, router_indices = torch.topk(router_scores, top_k, dim=-1)
        neuron_usage = torch.zeros(self.d_ff, device=x.device)
        ones = torch.ones_like(router_indices, dtype=torch.float)
        neuron_usage.scatter_add_(0, router_indices.flatten(), ones.flatten())

        total_selections = neuron_usage.sum()
        expected_usage = total_selections / self.d_ff
        load_balance_loss = ((neuron_usage - expected_usage).pow(2).mean())

        total_loss = ce_loss + 0.1 * ranking_loss + load_balance_weight * load_balance_loss

        return total_loss


# ============================================================
# DeepSets Language Model
# ============================================================

class DeepSetsLanguageModel(nn.Module):
    """DeepSets 기반 Language Model"""
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
        d_neuron: int = 128,       # 뉴런 정보 벡터 크기
        d_hidden: int = 256,       # φ 출력 크기
        use_context: bool = False  # 맥락 사용 여부
    ):
        if max_seq_len is not None:
            max_len = max_seq_len
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.sparse_k = sparse_k
        self.gradient_checkpointing = gradient_checkpointing
        self.use_context = use_context

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerLayerWithDeepSetsFFN(
                d_model,
                d_ff,
                n_heads,
                dropout,
                use_checkpoint=gradient_checkpointing,
                d_neuron=d_neuron,
                d_hidden=d_hidden,
                use_context=use_context
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

class TransformerLayerWithDeepSetsFFN(nn.Module):
    """Transformer layer with DeepSets FFN"""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        use_checkpoint: bool = False,
        d_neuron: int = 128,
        d_hidden: int = 256,
        use_context: bool = False
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # ✨ DeepSets FFN
        self.ffn = DeepSetsFFNLayer(
            d_model,
            d_ff,
            d_neuron=d_neuron,
            d_hidden=d_hidden,
            use_context=use_context
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
