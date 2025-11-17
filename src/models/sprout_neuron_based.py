"""
SPROUT Neuron-Based Architecture

뉴런 기반 동적 FFN을 사용하는 Transformer 모델
- W1, W2를 직접 Parameter로 관리 (빠른 계산)
- 학습 가능한 Router로 뉴런 선택
- 표준 FFN과 완벽히 동등 (모든 뉴런 사용시)
- Runtime sparsity 조절 가능
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
        self.context_weight = context_weight  # 컨텍스트 반영 비율

        if use_mlp:
            # 2-layer MLP router for better context sensitivity
            # Hidden dim = d_model (same as input)
            self.W_router_1 = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
            self.W_router_2 = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)

            # Initialize W_router_2 from W1 if provided
            if init_from_W1 is not None:
                self.W_router_2.data = init_from_W1.clone()

            # Layer norm for stability
            self.norm = nn.LayerNorm(d_model)
        else:
            # Simple linear router (original)
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
        # 라우터 점수
        if self.use_mlp:
            # 2-layer MLP with GELU activation
            x_norm = self.norm(x)
            h = F.gelu(x_norm @ self.W_router_1.T)  # [batch*seq, d_model]
            scores = h @ self.W_router_2.T  # [batch*seq, d_ff]
        else:
            # Simple linear
            scores = x @ self.W_router.T  # [batch*seq, d_ff]

        # Top-k 선택
        _, top_indices = torch.topk(scores, top_k, dim=-1)

        # Binary mask
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, top_indices, 1.0)

        return mask

    def compute_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        라우터 점수 계산 (마스크 없이)

        Args:
            x: [batch*seq, d_model]

        Returns:
            scores: [batch*seq, d_ff]
        """
        if self.use_mlp:
            x_norm = self.norm(x)
            h = F.gelu(x_norm @ self.W_router_1.T)
            scores = h @ self.W_router_2.T
        else:
            scores = x @ self.W_router.T

        return scores


# ============================================================
# Dynamic FFN Layer
# ============================================================

class DynamicFFNLayer(nn.Module):
    """
    뉴런 기반 동적 FFN

    기존 FFN과 완벽히 동등하지만, 일부 뉴런만 선택 가능

    최적화: W1, W2를 직접 파라미터로 관리 (매번 stack하지 않음)
    """
    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # W1, W2를 직접 파라미터로 (훨씬 빠름!)
        self.W1 = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)
        self.W2 = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)

        # 라우터 (학습 가능) - W1로 초기화하여 뉴런 특성 반영
        self.router = Router(d_model, d_ff, init_from_W1=self.W1.data)

    @property
    def middle_neurons(self):
        """개별 middle neuron 접근 (backward compatibility)"""
        class NeuronView:
            def __init__(self, weight_matrix):
                self.weight_matrix = weight_matrix

            def __len__(self):
                return self.weight_matrix.shape[0]

            def __getitem__(self, idx):
                class Neuron:
                    def __init__(self, w):
                        self.W_in = w
                return Neuron(self.weight_matrix[idx])

        return NeuronView(self.W1)

    @property
    def output_neurons(self):
        """개별 output neuron 접근 (backward compatibility)"""
        class NeuronView:
            def __init__(self, weight_matrix):
                self.weight_matrix = weight_matrix

            def __len__(self):
                return self.weight_matrix.shape[0]

            def __getitem__(self, idx):
                class Neuron:
                    def __init__(self, w):
                        self.W = w
                return Neuron(self.weight_matrix[idx])

        return NeuronView(self.W2)

    def forward(self, x: torch.Tensor, top_k: Optional[int] = None, chunk_size: Optional[int] = None) -> torch.Tensor:
        """
        메모리 효율적인 배치 처리 구현

        top_k < d_ff인 경우: 선택된 뉴런만 실제 계산 (진짜 희소 연산)
        top_k >= d_ff 또는 None: 전체 계산 (Dense FFN)

        Args:
            x: [batch, seq, d_model]
            top_k: 사용할 중간 뉴런 개수 (None이면 전부)
            chunk_size: 청크 크기 (None이면 자동 결정, 메모리 절약용)

        Returns:
            output: [batch, seq, d_model]
        """
        batch, seq, d_model = x.shape
        total_tokens = batch * seq

        # 자동 청크 크기 결정: 메모리 효율성과 성능 균형
        # d_ff가 클수록, 청크 크기를 작게 설정
        if chunk_size is None:
            # 휴리스틱: d_ff에 따라 청크 크기 자동 조정
            if self.d_ff >= 8192:
                chunk_size = min(2048, total_tokens)  # 큰 d_ff: 작은 청크
            elif self.d_ff >= 4096:
                chunk_size = min(4096, total_tokens)  # 중간 d_ff: 중간 청크
            else:
                chunk_size = total_tokens  # 작은 d_ff: 청크 없음

        # Flatten
        x_flat = x.view(-1, d_model)  # [batch*seq, d_model]

        # 청크 처리가 필요한 경우
        if chunk_size < total_tokens:
            return self._forward_chunked(x_flat, top_k, chunk_size, batch, seq)

        # 단일 배치 처리
        if top_k is not None and top_k < self.d_ff:
            # ===== 희소 연산: 전략 선택 =====
            sparsity_ratio = top_k / self.d_ff

            # 전략 1: 매우 sparse (< 10%) → 진짜 sparse 계산 (빠름!)
            if sparsity_ratio < 0.10:
                output = self._forward_true_sparse(x_flat, top_k, batch, seq)

            # 전략 2: 중간 sparse (10% ~ 90%) → 마스킹 방식 (안전)
            else:
                # 라우터로 top-k 선택
                scores = self.router.compute_scores(x_flat)
                _, top_indices = torch.topk(scores, top_k, dim=-1)

                # 전체 계산 후 마스킹
                z = x_flat @ self.W1.T  # [batch*seq, d_ff]
                mask = torch.zeros_like(z)
                mask.scatter_(-1, top_indices, 1.0)
                z.mul_(mask)  # In-place
                del mask

                # 활성화 및 출력
                a = F.gelu(z)
                output = a @ self.W2.T

        else:
            # ===== Dense 연산: 전체 뉴런 사용 =====
            z = x_flat @ self.W1.T  # [batch*seq, d_ff]
            a = F.gelu(z)  # [batch*seq, d_ff]
            output = a @ self.W2.T  # [batch*seq, d_model]

        return output.view(batch, seq, d_model)

    def _forward_true_sparse(
        self,
        x_flat: torch.Tensor,
        top_k: int,
        batch: int,
        seq: int
    ) -> torch.Tensor:
        """
        진짜 sparse 계산: 선택된 뉴런만 실제 계산

        메모리 안전을 위해 작은 청크로 처리
        매우 sparse한 경우 (top_k < 10% d_ff)에만 사용

        Args:
            x_flat: [total_tokens, d_model]
            top_k: 선택할 뉴런 개수
            batch, seq: 원본 형태

        Returns:
            output: [batch, seq, d_model]
        """
        total_tokens = x_flat.shape[0]

        # 라우터로 top-k 선택
        scores = self.router.compute_scores(x_flat)
        _, top_indices = torch.topk(scores, top_k, dim=-1)  # [total_tokens, top_k]

        # 메모리 안전을 위한 청크 크기
        # 휴리스틱: selected_W1이 2GB를 넘지 않도록
        # selected_W1 size = chunk_size * top_k * d_model * 4 bytes
        max_memory_gb = 2.0
        max_elements = (max_memory_gb * 1024**3) / 4  # float32
        chunk_size = int(max_elements / (top_k * self.d_model))
        chunk_size = max(1, min(chunk_size, total_tokens))

        # 출력 텐서 초기화
        z_sparse = torch.zeros(total_tokens, self.d_ff, device=x_flat.device, dtype=x_flat.dtype)

        # 청크 단위로 진짜 sparse 계산
        for i in range(0, total_tokens, chunk_size):
            end_idx = min(i + chunk_size, total_tokens)
            x_chunk = x_flat[i:end_idx]  # [chunk, d_model]
            indices_chunk = top_indices[i:end_idx]  # [chunk, top_k]

            # 각 토큰별로 선택된 뉴런의 가중치 gather
            # 배치 처리: advanced indexing 사용
            batch_indices = torch.arange(indices_chunk.shape[0], device=x_flat.device).unsqueeze(1).expand_as(indices_chunk)

            # 선택된 가중치 수집: [chunk, top_k, d_model]
            selected_W1 = self.W1[indices_chunk]  # Broadcasting으로 안전

            # 선택된 뉴런만 계산: [chunk, top_k]
            z_selected = torch.bmm(
                x_chunk.unsqueeze(1),  # [chunk, 1, d_model]
                selected_W1.transpose(1, 2)  # [chunk, d_model, top_k]
            ).squeeze(1)  # [chunk, top_k]

            # 희소 텐서에 할당
            z_sparse[i:end_idx].scatter_(1, indices_chunk, z_selected)

        # 활성화 및 출력 (dense)
        a = F.gelu(z_sparse)
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
        """
        청크 단위로 처리하여 메모리 사용량 감소

        Args:
            x_flat: [total_tokens, d_model]
            top_k: 사용할 뉴런 개수
            chunk_size: 청크 크기
            batch, seq: 원본 형태

        Returns:
            output: [batch, seq, d_model]
        """
        total_tokens = x_flat.shape[0]
        outputs = []

        for i in range(0, total_tokens, chunk_size):
            end_idx = min(i + chunk_size, total_tokens)
            x_chunk = x_flat[i:end_idx]

            if top_k is not None and top_k < self.d_ff:
                # 희소 연산
                z = x_chunk @ self.W1.T

                # 라우터로 top-k 선택
                scores = self.router.compute_scores(x_chunk)
                _, top_indices = torch.topk(scores, top_k, dim=-1)

                # 마스크 생성 및 적용
                mask = torch.zeros_like(z)
                mask.scatter_(-1, top_indices, 1.0)
                z.mul_(mask)
                del mask

                # 활성화 및 출력
                a = F.gelu(z)
                out_chunk = a @ self.W2.T
            else:
                # Dense 연산
                z = x_chunk @ self.W1.T
                a = F.gelu(z)
                out_chunk = a @ self.W2.T

            outputs.append(out_chunk)

        # 청크 결과 결합
        output = torch.cat(outputs, dim=0)
        return output.view(batch, seq, self.d_model)

    def get_W1(self) -> torch.Tensor:
        """W1 행렬 추출 [d_ff, d_model]"""
        return self.W1

    def get_W2(self) -> torch.Tensor:
        """W2 행렬 추출 [d_model, d_ff]"""
        return self.W2

    def compute_router_loss(
        self,
        x: torch.Tensor,
        top_k: int,
        load_balance_weight: float = 0.01
    ) -> torch.Tensor:
        """
        개선된 라우터 품질 loss 계산

        Oracle (실제 activation 기반 선택)과 Learned router의 차이를 최소화
        + Load balancing + Ranking consistency

        Args:
            x: [batch, seq, d_model]
            top_k: 선택할 뉴런 개수
            load_balance_weight: Load balancing loss 가중치

        Returns:
            loss: 스칼라 텐서
        """
        batch, seq, _ = x.shape
        x_flat = x.view(-1, self.d_model)

        # 1. Oracle scores: 실제 activation 크기
        with torch.no_grad():
            z_oracle = x_flat @ self.W1.T  # [batch*seq, d_ff]
            a_oracle = F.gelu(z_oracle)
            oracle_scores = a_oracle.abs()  # 실제 activation 크기

        # 2. Router scores: 학습된 라우터
        router_scores = self.router.compute_scores(x_flat)  # [batch*seq, d_ff]

        # ===== Loss 1: Top-k Overlap (Cross Entropy) =====
        _, oracle_indices = torch.topk(oracle_scores, top_k, dim=-1)
        oracle_mask = torch.zeros_like(router_scores)
        oracle_mask.scatter_(-1, oracle_indices, 1.0)

        router_scores_norm = F.log_softmax(router_scores, dim=-1)
        ce_loss = -(oracle_mask * router_scores_norm).sum() / oracle_mask.sum()

        # ===== Loss 2: Ranking Consistency (순위 일치) =====
        # Oracle과 Router의 상위 뉴런 순위가 일치하도록
        # Spearman correlation 근사: MSE on normalized scores
        oracle_scores_norm = (oracle_scores - oracle_scores.mean(dim=-1, keepdim=True)) / (oracle_scores.std(dim=-1, keepdim=True) + 1e-6)
        router_scores_norm_mse = (router_scores - router_scores.mean(dim=-1, keepdim=True)) / (router_scores.std(dim=-1, keepdim=True) + 1e-6)

        # Top-k 영역에서만 MSE 계산 (중요한 뉴런만)
        ranking_loss = ((oracle_scores_norm - router_scores_norm_mse) * oracle_mask).pow(2).sum() / oracle_mask.sum()

        # ===== Loss 3: Load Balancing (뉴런 사용 다양성) =====
        # 모든 뉴런이 골고루 선택되도록
        _, router_indices = torch.topk(router_scores, top_k, dim=-1)  # [batch*seq, top_k]

        # 각 뉴런의 선택 빈도 계산
        neuron_usage = torch.zeros(self.d_ff, device=x.device)
        ones = torch.ones_like(router_indices, dtype=torch.float)
        neuron_usage.scatter_add_(0, router_indices.flatten(), ones.flatten())

        # 균등 분포에서의 차이 (엔트로피 최대화)
        total_selections = neuron_usage.sum()
        expected_usage = total_selections / self.d_ff
        load_balance_loss = ((neuron_usage - expected_usage).pow(2).mean())

        # ===== Total Loss =====
        total_loss = ce_loss + 0.1 * ranking_loss + load_balance_weight * load_balance_loss

        return total_loss

    def analyze_neuron_usage(self, x: torch.Tensor, top_k: int) -> dict:
        """
        뉴런 사용 패턴 분석

        Args:
            x: [batch, seq, d_model]
            top_k: 선택할 뉴런 개수

        Returns:
            stats: 통계 정보
        """
        batch, seq, _ = x.shape
        x_flat = x.view(-1, self.d_model)

        # 라우터 점수
        scores = self.router.compute_scores(x_flat)  # [batch*seq, d_ff]

        # 선택된 뉴런
        _, top_indices = torch.topk(scores, top_k, dim=-1)  # [batch*seq, top_k]

        # 사용 빈도 (벡터화)
        usage_count = torch.zeros(self.d_ff, device=x.device)
        flattened_indices = top_indices.flatten()
        ones = torch.ones_like(flattened_indices, dtype=torch.float)
        usage_count.scatter_add_(0, flattened_indices, ones)

        usage_freq = usage_count / (batch * seq)

        return {
            'usage_frequency': usage_freq.cpu(),
            'unique_neurons_used': (usage_count > 0).sum().item(),
            'max_usage': usage_freq.max().item(),
            'min_usage': usage_freq.min().item(),
            'mean_usage': usage_freq.mean().item()
        }


# ============================================================
# Neuron-Based Language Model
# ============================================================

class NeuronBasedLanguageModel(nn.Module):
    """
    뉴런 기반 Language Model

    구조:
    - Token Embedding + Position Embedding
    - Transformer Layers with DynamicFFN
    - Output Projection
    """
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 512,
        d_ff: int = 2048,
        n_heads: int = 8,
        n_layers: int = 6,
        max_len: int = 512,
        max_seq_len: int = None,  # Alias for max_len
        dropout: float = 0.1,
        sparse_k: Optional[int] = None,  # None이면 dense
        gradient_checkpointing: bool = False  # 메모리 절약 옵션
    ):
        # Handle max_seq_len alias
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

        # Transformer Encoder Layers
        self.layers = nn.ModuleList([
            TransformerLayerWithDynamicFFN(
                d_model, d_ff, n_heads, dropout, use_checkpoint=gradient_checkpointing
            )
            for _ in range(n_layers)
        ])

        # Output head (LM head)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def enable_gradient_checkpointing(self):
        """메모리 절약을 위한 gradient checkpointing 활성화"""
        self.gradient_checkpointing = True
        for layer in self.layers:
            layer.use_checkpoint = True

    def disable_gradient_checkpointing(self):
        """Gradient checkpointing 비활성화"""
        self.gradient_checkpointing = False
        for layer in self.layers:
            layer.use_checkpoint = False

    def _init_weights(self):
        """Initialize weights"""
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
            labels: [batch, seq] for MLM
            top_k: 각 FFN에서 사용할 뉴런 수 (None이면 self.sparse_k 사용)

        Returns:
            dict with 'logits', 'loss' (if labels provided)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if top_k is None:
            top_k = self.sparse_k

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb  # [batch, seq, d_model]

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask, top_k=top_k)

        x = self.norm(x)

        # Output projection
        logits = self.output_projection(x)  # [batch, seq, vocab_size]

        # Loss (MLM)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        return {
            'logits': logits,
            'loss': loss
        }


# ============================================================
# Transformer Layer with Dynamic FFN
# ============================================================

class TransformerLayerWithDynamicFFN(nn.Module):
    """
    Transformer layer with dynamic FFN

    = Attention + Dynamic FFN
    """
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float, use_checkpoint: bool = False):
        super().__init__()

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Dynamic FFN
        self.ffn = DynamicFFNLayer(d_model, d_ff)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Gradient checkpointing
        self.use_checkpoint = use_checkpoint

    def _attention_block(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Attention block for checkpointing"""
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = self.dropout(x)
        return x

    def _ffn_block(self, x: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
        """FFN block for checkpointing"""
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
        """
        Args:
            x: [batch, seq, d_model]
            attention_mask: [batch, seq]
            top_k: FFN에서 사용할 뉴런 수
        """
        # Attention block
        if self.use_checkpoint and self.training:
            attn_out = checkpoint(self._attention_block, x, attention_mask, use_reentrant=False)
        else:
            attn_out = self._attention_block(x, attention_mask)
        x = x + attn_out

        # FFN block
        if self.use_checkpoint and self.training:
            ffn_out = checkpoint(self._ffn_block, x, top_k, use_reentrant=False)
        else:
            ffn_out = self._ffn_block(x, top_k)
        x = x + ffn_out

        return x
