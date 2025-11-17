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


# ============================================================
# Router
# ============================================================

class Router(nn.Module):
    """
    학습 가능한 라우터 - 입력에 따라 최적의 뉴런 선택

    2-layer MLP 구조로 비선형 패턴 학습 가능
    W1로 초기화하여 학습 초기부터 뉴런 특성 반영
    """
    def __init__(self, d_model: int, d_ff: int, init_from_W1: torch.Tensor = None, use_mlp: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_mlp = use_mlp

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

    def forward(self, x: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        """
        효율적인 배치 처리 구현

        top_k < d_ff인 경우: 선택된 뉴런만 실제 계산 (진짜 희소 연산)
        top_k >= d_ff 또는 None: 전체 계산 (Dense FFN)

        Args:
            x: [batch, seq, d_model]
            top_k: 사용할 중간 뉴런 개수 (None이면 전부)

        Returns:
            output: [batch, seq, d_model]
        """
        batch, seq, d_model = x.shape

        # Flatten
        x_flat = x.view(-1, d_model)  # [batch*seq, d_model]

        if top_k is not None and top_k < self.d_ff:
            # ===== 희소 연산: 선택된 뉴런만 계산 =====

            # 1. 라우터로 top-k 뉴런 선택
            scores = self.router.compute_scores(x_flat)  # [batch*seq, d_ff]
            _, top_indices = torch.topk(scores, top_k, dim=-1)  # [batch*seq, top_k]

            # 2. 선택된 W1 가중치 gather
            # W1: [d_ff, d_model] → selected_W1: [batch*seq, top_k, d_model]
            selected_W1 = self.W1[top_indices]

            # 3. 선택된 뉴런만 계산 (z = x @ W1.T)
            # x_flat: [batch*seq, d_model]
            # selected_W1: [batch*seq, top_k, d_model]
            # → z_sparse: [batch*seq, top_k]
            z_sparse = torch.einsum('bd,bkd->bk', x_flat, selected_W1)

            # 4. GELU 활성화
            a_sparse = F.gelu(z_sparse)  # [batch*seq, top_k]

            # 5. 선택된 W2 가중치 gather
            # W2: [d_model, d_ff] → W2.T: [d_ff, d_model]
            # selected_W2_T: [batch*seq, top_k, d_model]
            selected_W2_T = self.W2.T[top_indices]

            # 6. 출력 계산 (output = a @ W2.T)
            # a_sparse: [batch*seq, top_k]
            # selected_W2_T: [batch*seq, top_k, d_model]
            # → output: [batch*seq, d_model]
            output = torch.einsum('bk,bkd->bd', a_sparse, selected_W2_T)

        else:
            # ===== Dense 연산: 전체 뉴런 사용 =====
            z = x_flat @ self.W1.T  # [batch*seq, d_ff]
            a = F.gelu(z)  # [batch*seq, d_ff]
            output = a @ self.W2.T  # [batch*seq, d_model]

        return output.view(batch, seq, d_model)

    def get_W1(self) -> torch.Tensor:
        """W1 행렬 추출 [d_ff, d_model]"""
        return self.W1

    def get_W2(self) -> torch.Tensor:
        """W2 행렬 추출 [d_model, d_ff]"""
        return self.W2

    def compute_router_loss(self, x: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        라우터 품질 loss 계산

        Oracle (실제 activation 기반 선택)과 Learned router의 차이를 최소화

        Args:
            x: [batch, seq, d_model]
            top_k: 선택할 뉴런 개수

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

        # 3. Top-k overlap 최대화
        # Oracle이 선택할 뉴런을 Router도 높은 점수로 예측하도록
        _, oracle_indices = torch.topk(oracle_scores, top_k, dim=-1)

        # Oracle이 선택한 뉴런에 대해 router가 높은 점수를 주도록
        oracle_mask = torch.zeros_like(router_scores)
        oracle_mask.scatter_(-1, oracle_indices, 1.0)

        # Router가 oracle 선택을 따라가도록 유도
        # Positive: oracle이 선택한 것 → 높은 점수
        # Negative: oracle이 선택 안 한 것 → 낮은 점수
        router_scores_norm = F.log_softmax(router_scores, dim=-1)
        loss = -(oracle_mask * router_scores_norm).sum() / oracle_mask.sum()

        return loss

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
        sparse_k: Optional[int] = None  # None이면 dense
    ):
        # Handle max_seq_len alias
        if max_seq_len is not None:
            max_len = max_seq_len
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.sparse_k = sparse_k

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # Transformer Encoder Layers
        self.layers = nn.ModuleList([
            TransformerLayerWithDynamicFFN(
                d_model, d_ff, n_heads, dropout
            )
            for _ in range(n_layers)
        ])

        # Output head (LM head)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

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
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float):
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
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = self.dropout(x)
        x = residual + x

        # FFN block
        residual = x
        x = self.norm2(x)
        x = self.ffn(x, top_k=top_k)
        x = self.dropout(x)
        x = residual + x

        return x
