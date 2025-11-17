"""
SPROUT Neuron-Based Architecture

완전히 새로운 접근:
- FFN을 뉴런 레벨로 분해
- 각 뉴런이 독립적인 모듈
- 동적으로 뉴런 선택
- 표준 FFN과 완벽히 동등 (모든 뉴런 사용시)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================
# 1. 뉴런 클래스들
# ============================================================

class MiddleNeuron(nn.Module):
    """
    중간 뉴런 (W1의 한 행에 해당)

    입력 받아서 활성화 전 값 출력
    """
    def __init__(self, neuron_id: int, d_model: int = 512):
        super().__init__()
        self.neuron_id = neuron_id
        # W1의 neuron_id번째 행
        self.W_in = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [d_model] 또는 [batch*seq, d_model]
        returns: scalar 또는 [batch*seq]
        """
        if x.dim() == 1:
            return torch.dot(self.W_in, x)
        else:
            return x @ self.W_in  # [batch*seq]


class OutputNeuron(nn.Module):
    """
    출력 뉴런 (W2의 한 행에 해당)

    중간 뉴런들의 활성화 받아서 출력
    """
    def __init__(self, neuron_id: int, n_middle: int = 2048):
        super().__init__()
        self.neuron_id = neuron_id
        # W2의 neuron_id번째 행
        self.W = nn.Parameter(torch.randn(n_middle) * 0.02)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        activations: [d_ff] 희소 벡터 또는 [batch*seq, d_ff]
        returns: scalar 또는 [batch*seq]
        """
        if activations.dim() == 1:
            return torch.dot(self.W, activations)
        else:
            return activations @ self.W  # [batch*seq]


# ============================================================
# 2. 라우터
# ============================================================

class Router(nn.Module):
    """
    입력 계층: 어떤 중간 뉴런 활성화할지 결정

    학습 가능한 라우터 - 입력 보고 최적의 뉴런 선택
    W1로 초기화하여 학습 초기부터 뉴런 특성을 반영
    """
    def __init__(self, d_model: int, d_ff: int, init_from_W1: torch.Tensor = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # 라우터 가중치: W1로 초기화 (뉴런 시그니처 = 뉴런 가중치)
        if init_from_W1 is not None:
            self.W_router = nn.Parameter(init_from_W1.clone())
        else:
            # Fallback: 랜덤 초기화
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
        # 각 중간 뉴런과의 유사도
        scores = x @ self.W_router.T  # [batch*seq, d_ff]

        # 각 샘플마다 상위 k개 선택
        _, top_indices = torch.topk(scores, top_k, dim=-1)

        # Binary mask 생성
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, top_indices, 1.0)

        return mask


# ============================================================
# 3. 동적 FFN 레이어
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

        Args:
            x: [batch, seq, d_model]
            top_k: 사용할 중간 뉴런 개수 (None이면 전부)

        Returns:
            output: [batch, seq, d_model]
        """
        batch, seq, d_model = x.shape

        # Flatten
        x_flat = x.view(-1, d_model)  # [batch*seq, d_model]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. 중간 계층 (Sparse 가능)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # W1, W2는 이미 파라미터로 존재 (매번 재구성 불필요!)

        z = x_flat @ self.W1.T  # [batch*seq, d_ff]

        if top_k is not None and top_k < self.d_ff:
            # 희소화: 라우터로 상위 k개만 선택
            mask = self.router.select_batch(x_flat, top_k)
            z = z * mask

        # 활성화
        a = F.gelu(z)  # [batch*seq, d_ff]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2. 출력 계층
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        output = a @ self.W2.T  # [batch*seq, d_model]

        return output.view(batch, seq, d_model)

    def get_W1(self) -> torch.Tensor:
        """W1 행렬 추출 [d_ff, d_model]"""
        return self.W1

    def get_W2(self) -> torch.Tensor:
        """W2 행렬 추출 [d_model, d_ff]"""
        return self.W2

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
        scores = x_flat @ self.router.W_router.T  # [batch*seq, d_ff]

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
# 4. 전체 Language Model
# ============================================================

class NeuronBasedLanguageModel(nn.Module):
    """
    뉴런 기반 Language Model

    구조:
    - Token Embedding
    - Transformer Encoder (attention)
    - Dynamic FFN (뉴런 기반)
    - Output Head
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

    def analyze_neuron_usage(self, dataloader, top_k: int, num_batches: int = 10):
        """
        전체 모델의 뉴런 사용 패턴 분석

        Args:
            dataloader: 데이터 로더
            top_k: 사용할 뉴런 수
            num_batches: 분석할 배치 수

        Returns:
            layer별 통계
        """
        self.eval()

        layer_stats = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids']

                # Forward through embeddings
                token_emb = self.token_embedding(input_ids)
                positions = torch.arange(input_ids.shape[1], device=input_ids.device)
                positions = positions.unsqueeze(0).expand(input_ids.shape[0], -1)
                pos_emb = self.position_embedding(positions)
                x = token_emb + pos_emb

                # Each layer
                for layer_idx, layer in enumerate(self.layers):
                    # Attention
                    x = layer.attention(layer.norm1(x))

                    # FFN neuron analysis
                    stats = layer.ffn.analyze_neuron_usage(x, top_k)

                    if batch_idx == 0:
                        layer_stats.append([stats])
                    else:
                        layer_stats[layer_idx].append(stats)

                    # Complete layer
                    x = layer(x, top_k=top_k)

        # Aggregate stats
        aggregated = []
        for layer_idx, stats_list in enumerate(layer_stats):
            # Average usage frequencies
            all_freqs = torch.stack([s['usage_frequency'] for s in stats_list])
            avg_freq = all_freqs.mean(dim=0)

            aggregated.append({
                'layer': layer_idx,
                'avg_usage_frequency': avg_freq,
                'unique_neurons_used': (avg_freq > 0).sum().item(),
                'max_usage': avg_freq.max().item(),
                'min_usage': avg_freq.min().item()
            })

        return aggregated


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


# ============================================================
# 5. Helper Functions
# ============================================================

def create_neuron_based_model(
    vocab_size: int = 30000,
    d_model: int = 512,
    d_ff: int = 2048,
    n_heads: int = 8,
    n_layers: int = 6,
    sparse_k: Optional[int] = None,
    **kwargs
) -> NeuronBasedLanguageModel:
    """
    Create neuron-based language model

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        d_ff: FFN hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        sparse_k: Number of neurons to use in FFN (None = dense)

    Returns:
        NeuronBasedLanguageModel
    """
    return NeuronBasedLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_layers=n_layers,
        sparse_k=sparse_k,
        **kwargs
    )


if __name__ == "__main__":
    # 간단한 테스트
    print("=" * 70)
    print("Neuron-Based Language Model Test")
    print("=" * 70)

    # Model
    model = create_neuron_based_model(
        vocab_size=1000,
        d_model=256,
        d_ff=1024,
        n_heads=4,
        n_layers=2,
        sparse_k=512  # 50% sparse
    )

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Test input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")

    # Forward
    outputs = model(input_ids, top_k=512)
    logits = outputs['logits']

    print(f"Output shape: {logits.shape}")
    print(f"\nTest passed! ✅")
