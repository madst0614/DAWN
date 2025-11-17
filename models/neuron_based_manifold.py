"""
SPROUT Neuron-Based Architecture with Manifold Mixing

뉴런 기반 동적 FFN + DeepSets 매니폴드 형성
- 선택된 뉴런들이 협력하여 국소 매니폴드 생성
- 단일 함수로 어떤 뉴런 조합이든 안정적으로 처리
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
    """
    학습 가능한 라우터 - 입력과 컨텍스트에 따라 최적의 뉴런 선택

    2-layer MLP 구조로 비선형 패턴 학습 가능
    W1로 초기화하여 학습 초기부터 뉴런 특성 반영
    컨텍스트 반영으로 동적 라우팅 지원
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        init_from_W1: torch.Tensor = None,
        use_mlp: bool = True,
        context_weight: float = 0.3
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_mlp = use_mlp
        self.context_weight = context_weight

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

    def select_batch(self, x: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        배치 처리로 뉴런 선택

        Args:
            x: [batch*seq, d_model]
            top_k: 선택할 뉴런 개수

        Returns:
            mask: [batch*seq, d_ff] - Binary mask (1=선택, 0=제외)
        """
        if self.use_mlp:
            x_norm = self.norm(x)
            h = F.gelu(x_norm @ self.W_router_1.T)
            scores = h @ self.W_router_2.T
        else:
            scores = x @ self.W_router.T

        _, top_indices = torch.topk(scores, top_k, dim=-1)

        mask = torch.zeros_like(scores)
        mask.scatter_(-1, top_indices, 1.0)

        return mask

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
# Manifold Mixing Layer
# ============================================================

class DeepSetsManifoldMixing(nn.Module):
    """
    DeepSets 기반 매니폴드 형성

    핵심 아이디어:
    - 선택된 뉴런들 = 독립적 정보 단위
    - φ: 각 뉴런의 "의미" 추출
    - Aggregate: 의미들 조합 (순서 무관)
    - ρ: 조합된 의미를 매니폴드 좌표로 변환

    F(뉴런들) = ρ(mean(φ(각 뉴런)))
    → 어떤 뉴런 조합이든 안정적으로 처리!
    """
    def __init__(self, d_hidden: int = 64):
        super().__init__()
        self.d_hidden = d_hidden

        # φ: 각 뉴런의 activation을 의미 벡터로
        # [1] → [d_hidden]
        self.phi = nn.Sequential(
            nn.Linear(1, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, d_hidden),
            nn.LayerNorm(d_hidden)
        )

        # ρ: 조합된 의미를 매니폴드 좌표로
        # [d_hidden] → [1] (각 뉴런 출력)
        self.rho = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden * 2, 1)
        )

    def forward(self, selected_activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            selected_activations: [batch*seq, top_k] - 선택된 뉴런들의 activation

        Returns:
            manifold_activations: [batch*seq, top_k] - 매니폴드 공간의 좌표
        """
        batch_seq, top_k = selected_activations.shape

        # 1. φ: 각 뉴런을 독립적으로 임베딩
        # [B*S, top_k] → [B*S, top_k, 1] → [B*S, top_k, d_hidden]
        embeddings = self.phi(selected_activations.unsqueeze(-1))

        # 2. Aggregate: 의미들을 조합 (permutation invariant!)
        # [B*S, top_k, d_hidden] → [B*S, d_hidden]
        # mean을 써서 뉴런 개수에 robust
        combined = embeddings.mean(dim=1)

        # 3. ρ: 조합된 의미를 각 뉴런의 새로운 activation으로
        # 각 뉴런이 "전체 문맥"을 반영한 값을 가지게 됨
        # [B*S, d_hidden] → [B*S, top_k, d_hidden] (broadcast)
        context = combined.unsqueeze(1).expand(-1, top_k, -1)

        # 각 뉴런 임베딩 + 전체 문맥 결합
        contextualized = embeddings + context  # [B*S, top_k, d_hidden]

        # 최종 activation
        # [B*S, top_k, d_hidden] → [B*S, top_k, 1] → [B*S, top_k]
        manifold_activations = self.rho(contextualized).squeeze(-1)

        return manifold_activations


# ============================================================
# Dynamic FFN Layer with Manifold
# ============================================================

class DynamicFFNLayer(nn.Module):
    """
    뉴런 기반 동적 FFN + 매니폴드 형성

    기존: 뉴런 선택 → 독립적 activation → 출력
    개선: 뉴런 선택 → 협력적 매니폴드 형성 → 출력
    """
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        use_manifold: bool = True,
        manifold_d_hidden: int = 64
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_manifold = use_manifold

        # W1, W2를 직접 파라미터로
        self.W1 = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)
        self.W2 = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)

        # 라우터
        self.router = Router(d_model, d_ff, init_from_W1=self.W1.data)

        # ✨ 매니폴드 mixing layer
        if use_manifold:
            self.manifold_mixer = DeepSetsManifoldMixing(d_hidden=manifold_d_hidden)
        else:
            self.manifold_mixer = None

    def forward(
        self,
        x: torch.Tensor,
        top_k: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, d_model]
            top_k: 사용할 중간 뉴런 개수
            chunk_size: 청크 크기

        Returns:
            output: [batch, seq, d_model]
        """
        batch, seq, d_model = x.shape
        total_tokens = batch * seq

        # 청크 크기 자동 결정
        if chunk_size is None:
            if self.d_ff >= 8192:
                chunk_size = min(2048, total_tokens)
            elif self.d_ff >= 4096:
                chunk_size = min(4096, total_tokens)
            else:
                chunk_size = total_tokens

        x_flat = x.view(-1, d_model)

        # 청크 처리
        if chunk_size < total_tokens:
            return self._forward_chunked(x_flat, top_k, chunk_size, batch, seq)

        # 단일 배치 처리
        if top_k is not None and top_k < self.d_ff:
            output = self._forward_sparse(x_flat, top_k, batch, seq)
        else:
            # Dense 연산
            z = x_flat @ self.W1.T
            a = F.gelu(z)
            output = a @ self.W2.T

        return output.view(batch, seq, d_model)

    def _forward_sparse(
        self,
        x_flat: torch.Tensor,
        top_k: int,
        batch: int,
        seq: int
    ) -> torch.Tensor:
        """
        희소 연산 + 매니폴드 형성
        """
        # 라우터로 뉴런 선택
        scores = self.router.compute_scores(x_flat)
        _, top_indices = torch.topk(scores, top_k, dim=-1)

        # 전체 계산
        z = x_flat @ self.W1.T  # [batch*seq, d_ff]

        # 마스킹
        mask = torch.zeros_like(z)
        mask.scatter_(-1, top_indices, 1.0)
        z_masked = z * mask

        # 기본 activation
        a = F.gelu(z_masked)

        # ✨ 매니폴드 형성 (핵심!)
        if self.manifold_mixer is not None and self.training:
            # 선택된 뉴런들의 activation 추출
            # [batch*seq, d_ff] → [batch*seq, top_k]
            batch_seq = x_flat.shape[0]
            selected_activations = []

            for i in range(batch_seq):
                idx = top_indices[i]  # [top_k]
                selected_activations.append(a[i, idx])

            selected_activations = torch.stack(selected_activations)  # [B*S, top_k]

            # 매니폴드 mixing: 뉴런들이 협력하여 표현 생성
            manifold_output = self.manifold_mixer(selected_activations)  # [B*S, top_k]

            # 다시 sparse 공간으로
            a_manifold = torch.zeros_like(a)
            for i in range(batch_seq):
                idx = top_indices[i]
                a_manifold[i, idx] = manifold_output[i]

            a = a_manifold

        # 출력 투영
        output = a @ self.W2.T

        return output

    def _forward_chunked(
        self,
        x_flat: torch.Tensor,
        top_k: Optional[int],
        chunk_size: int,
        batch: int,
        seq: int
    ) -> torch.Tensor:
        """청크 단위 처리"""
        total_tokens = x_flat.shape[0]
        outputs = []

        for i in range(0, total_tokens, chunk_size):
            end_idx = min(i + chunk_size, total_tokens)
            x_chunk = x_flat[i:end_idx]

            if top_k is not None and top_k < self.d_ff:
                # 희소 연산 (청크 단위)
                chunk_batch_size = end_idx - i
                chunk_output = self._forward_sparse(
                    x_chunk,
                    top_k,
                    chunk_batch_size,
                    1
                )
            else:
                # Dense 연산
                z = x_chunk @ self.W1.T
                a = F.gelu(z)
                chunk_output = a @ self.W2.T

            outputs.append(chunk_output)

        output = torch.cat(outputs, dim=0)
        return output.view(batch, seq, self.d_model)

    def compute_router_loss(
        self,
        x: torch.Tensor,
        top_k: int,
        load_balance_weight: float = 0.01
    ) -> torch.Tensor:
        """라우터 품질 loss 계산"""
        batch, seq, _ = x.shape
        x_flat = x.view(-1, self.d_model)

        # Oracle scores
        with torch.no_grad():
            z_oracle = x_flat @ self.W1.T
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
# Neuron-Based Language Model
# ============================================================

class NeuronBasedLanguageModel(nn.Module):
    """
    뉴런 기반 Language Model + 매니폴드 형성
    """
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
        use_manifold: bool = True,  # ✨ 매니폴드 사용 여부
        manifold_d_hidden: int = 64
    ):
        if max_seq_len is not None:
            max_len = max_seq_len
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.sparse_k = sparse_k
        self.gradient_checkpointing = gradient_checkpointing
        self.use_manifold = use_manifold

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerLayerWithDynamicFFN(
                d_model,
                d_ff,
                n_heads,
                dropout,
                use_checkpoint=gradient_checkpointing,
                use_manifold=use_manifold,
                manifold_d_hidden=manifold_d_hidden
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

class TransformerLayerWithDynamicFFN(nn.Module):
    """Transformer layer with dynamic FFN + manifold"""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        use_checkpoint: bool = False,
        use_manifold: bool = True,
        manifold_d_hidden: int = 64
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # ✨ Dynamic FFN with manifold
        self.ffn = DynamicFFNLayer(
            d_model,
            d_ff,
            use_manifold=use_manifold,
            manifold_d_hidden=manifold_d_hidden
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
