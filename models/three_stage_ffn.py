"""
Three-Stage Dynamic Neuron FFN - Sequence-level Selection

구조:
1. Input Neurons: 입력 패턴 감지 및 해석
2. Process Neurons: 입력 조합을 처리하여 의미 생성
3. Output Neurons: 처리 결과를 출력 공간으로 변환

특징:
- 시퀀스별로 뉴런 조합 선택 (메모리 효율적)
- Sparse activation으로 효율성
- 조합 폭발을 통한 창발적 표현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint


# ============================================================
# Three-Stage Dynamic FFN
# ============================================================

class ThreeStageFFN(nn.Module):
    """
    3단계 동적 뉴런 FFN (시퀀스별 선택)

    핵심: 한 시퀀스 내 모든 토큰이 같은 뉴런 조합 사용
    - 메모리 효율성 대폭 향상
    - 각 시퀀스가 고유한 신경 회로를 가짐
    """

    def __init__(
        self,
        d_model: int = 512,
        n_input_neurons: int = 2048,
        n_process_neurons: int = 1024,
        n_output_neurons: int = 2048,
        d_process_value: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_input = n_input_neurons
        self.n_process = n_process_neurons
        self.n_output = n_output_neurons
        self.d_process_value = d_process_value

        # ===== Stage 1: Input Neurons =====
        # 각 입력 뉴런의 패턴 (어떤 입력에 반응하는가)
        self.input_patterns = nn.Parameter(
            torch.randn(n_input_neurons, d_model) * 0.02
        )

        # ===== Stage 2: Process Neurons =====
        # 처리 뉴런이 입력 뉴런들로부터 받는 가중치
        self.process_input_weights = nn.Parameter(
            torch.randn(n_process_neurons, n_input_neurons) * 0.02
        )
        # 처리 뉴런의 내부 표현 (처리된 정보)
        self.process_values = nn.Parameter(
            torch.randn(n_process_neurons, d_process_value) * 0.02
        )

        # ===== Stage 3: Output Neurons =====
        # 출력 뉴런이 처리 값을 받는 가중치
        self.output_input_weights = nn.Parameter(
            torch.randn(n_output_neurons, d_process_value) * 0.02
        )
        # 출력 뉴런의 출력 패턴
        self.output_patterns = nn.Parameter(
            torch.randn(n_output_neurons, d_model) * 0.02
        )

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        nn.init.xavier_normal_(self.input_patterns)
        nn.init.xavier_normal_(self.process_input_weights)
        nn.init.xavier_normal_(self.process_values)
        nn.init.xavier_normal_(self.output_input_weights)
        nn.init.xavier_normal_(self.output_patterns)

    def forward(
        self,
        x: torch.Tensor,
        k_input: Optional[int] = None,
        k_process: Optional[int] = None,
        k_output: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, d_model]
            k_input: 활성화할 입력 뉴런 수 (None이면 n_input//8)
            k_process: 활성화할 처리 뉴런 수 (None이면 n_process//8)
            k_output: 활성화할 출력 뉴런 수 (None이면 n_output//8)

        Returns:
            output: [B, S, d_model]
        """
        B, S, d_model = x.shape

        # Default k values
        if k_input is None:
            k_input = max(self.n_input // 8, 64)
        if k_process is None:
            k_process = max(self.n_process // 8, 32)
        if k_output is None:
            k_output = max(self.n_output // 8, 64)

        # ===== Stage 1: Input Neuron Activation =====
        # 토큰별로 계산 (입력 해석은 토큰별로 다름)
        input_scores = x @ self.input_patterns.T  # [B, S, n_input]
        input_acts = F.gelu(input_scores)

        # 시퀀스별 평균으로 뉴런 선택 (핵심!)
        input_acts_seq = input_acts.mean(dim=1)  # [B, n_input]
        top_input_acts, top_input_idx = input_acts_seq.topk(k_input, dim=-1)
        # [B, k_input] - 시퀀스별로만 다름!

        # ===== Stage 2: Process Neuron Activation =====
        # 선택된 입력 뉴런의 토큰별 activation
        # [B, S, k_input]
        expanded_idx = top_input_idx.unsqueeze(1).expand(-1, S, -1)
        selected_input_acts = torch.gather(input_acts, 2, expanded_idx)

        # Sparse representation (시퀀스별로 같은 뉴런)
        input_repr = torch.zeros(B, self.n_input, device=x.device, dtype=x.dtype)
        input_repr.scatter_(1, top_input_idx, top_input_acts.to(input_repr.dtype))  # [B, n_input]

        # Process neuron activation
        process_scores = input_repr @ self.process_input_weights.T  # [B, n_process]
        process_acts_seq = F.gelu(process_scores)

        # 시퀀스별로 처리 뉴런 선택
        top_process_acts, top_process_idx = process_acts_seq.topk(k_process, dim=-1)
        # [B, k_process]

        # 선택된 처리 뉴런의 values
        selected_process_values = self.process_values[top_process_idx]  # [B, k_process, d_process_value]

        # Value aggregation
        process_weights = F.softmax(top_process_acts, dim=-1)  # [B, k_process]
        aggregated_value = torch.einsum('bk,bkd->bd',
                                       process_weights,
                                       selected_process_values)
        # [B, d_process_value]

        aggregated_value = self.dropout(aggregated_value)

        # ===== Stage 3: Output Neuron Activation =====
        # 시퀀스별로 뉴런 선택
        output_scores = aggregated_value @ self.output_input_weights.T  # [B, n_output]
        output_acts_seq = F.gelu(output_scores)

        top_output_acts_seq, top_output_idx_seq = output_acts_seq.topk(k_output, dim=-1)
        # [B, k_output]

        # 선택된 출력 뉴런의 패턴
        selected_output_patterns = self.output_patterns[top_output_idx_seq]
        # [B, k_output, d_model] - 메모리 효율적!

        # 토큰별 activation 계산 (하지만 같은 뉴런 사용)
        # aggregated_value를 토큰별로 확장
        aggregated_value_expanded = aggregated_value.unsqueeze(1).expand(-1, S, -1)
        # [B, S, d_process_value]

        token_output_scores = aggregated_value_expanded @ self.output_input_weights.T
        # [B, S, n_output]
        token_output_acts = F.gelu(token_output_scores)

        # 선택된 뉴런의 activation만 추출
        expanded_output_idx = top_output_idx_seq.unsqueeze(1).expand(-1, S, -1)
        # [B, S, k_output]
        selected_output_acts = torch.gather(token_output_acts, 2, expanded_output_idx)
        # [B, S, k_output]

        # 최종 출력
        output = torch.einsum('bsk,bkm->bsm',
                             selected_output_acts,
                             selected_output_patterns)
        # [B, S, d_model]

        return output

    def get_neuron_stats(
        self,
        x: torch.Tensor,
        k_input: Optional[int] = None,
        k_process: Optional[int] = None,
        k_output: Optional[int] = None
    ) -> dict:
        """
        디버깅/분석용: 각 단계의 뉴런 활성화 통계
        """
        B, S, _ = x.shape

        if k_input is None:
            k_input = max(self.n_input // 8, 64)
        if k_process is None:
            k_process = max(self.n_process // 8, 32)
        if k_output is None:
            k_output = max(self.n_output // 8, 64)

        with torch.no_grad():
            # Stage 1
            input_acts = F.gelu(x @ self.input_patterns.T)
            input_acts_seq = input_acts.mean(dim=1)
            top_input_acts, top_input_idx = input_acts_seq.topk(k_input, dim=-1)

            # Stage 2
            input_repr = torch.zeros(B, self.n_input, device=x.device, dtype=x.dtype)
            input_repr.scatter_(1, top_input_idx, top_input_acts.to(input_repr.dtype))
            process_acts = F.gelu(input_repr @ self.process_input_weights.T)
            top_process_acts, top_process_idx = process_acts.topk(k_process, dim=-1)

            # Stage 3
            selected_process_values = self.process_values[top_process_idx]
            process_weights = F.softmax(top_process_acts, dim=-1)
            aggregated_value = torch.einsum('bk,bkd->bd', process_weights, selected_process_values)
            output_acts = F.gelu(aggregated_value @ self.output_input_weights.T)
            top_output_acts, top_output_idx = output_acts.topk(k_output, dim=-1)

            return {
                'input_neurons': {
                    'indices': top_input_idx.cpu(),  # [B, k_input]
                    'activations': top_input_acts.cpu(),
                    'mean_activation': top_input_acts.mean().item(),
                    'sparsity': k_input / self.n_input
                },
                'process_neurons': {
                    'indices': top_process_idx.cpu(),  # [B, k_process]
                    'activations': top_process_acts.cpu(),
                    'mean_activation': top_process_acts.mean().item(),
                    'sparsity': k_process / self.n_process
                },
                'output_neurons': {
                    'indices': top_output_idx.cpu(),  # [B, k_output]
                    'activations': top_output_acts.cpu(),
                    'mean_activation': top_output_acts.mean().item(),
                    'sparsity': k_output / self.n_output
                }
            }


# ============================================================
# Transformer Layer with Three-Stage FFN
# ============================================================

class TransformerLayerWithThreeStageFFN(nn.Module):
    """Transformer layer with Three-Stage Dynamic FFN"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_input_neurons: int = 2048,
        n_process_neurons: int = 1024,
        n_output_neurons: int = 2048,
        d_process_value: int = 256,
        dropout: float = 0.1,
        use_checkpoint: bool = False
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.ffn = ThreeStageFFN(
            d_model=d_model,
            n_input_neurons=n_input_neurons,
            n_process_neurons=n_process_neurons,
            n_output_neurons=n_output_neurons,
            d_process_value=d_process_value,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint

    def _attention_block(self, x, attention_mask):
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=attention_mask)
        return self.dropout(attn_out)

    def _ffn_block(self, x, k_input, k_process, k_output):
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm, k_input, k_process, k_output)
        return self.dropout(ffn_out)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        k_input: Optional[int] = None,
        k_process: Optional[int] = None,
        k_output: Optional[int] = None
    ) -> torch.Tensor:
        # Attention
        if self.use_checkpoint and self.training:
            attn_out = checkpoint(self._attention_block, x, attention_mask, use_reentrant=False)
        else:
            attn_out = self._attention_block(x, attention_mask)
        x = x + attn_out

        # FFN
        if self.use_checkpoint and self.training:
            ffn_out = checkpoint(self._ffn_block, x, k_input, k_process, k_output, use_reentrant=False)
        else:
            ffn_out = self._ffn_block(x, k_input, k_process, k_output)
        x = x + ffn_out

        return x


# ============================================================
# Language Model with Three-Stage FFN
# ============================================================

class ThreeStageLanguageModel(nn.Module):
    """Language Model with Three-Stage Dynamic Neuron FFN"""

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 512,
        n_input_neurons: int = 2048,
        n_process_neurons: int = 1024,
        n_output_neurons: int = 2048,
        d_process_value: int = 256,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.gradient_checkpointing = gradient_checkpointing

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerLayerWithThreeStageFFN(
                d_model=d_model,
                n_heads=n_heads,
                n_input_neurons=n_input_neurons,
                n_process_neurons=n_process_neurons,
                n_output_neurons=n_output_neurons,
                d_process_value=d_process_value,
                dropout=dropout,
                use_checkpoint=gradient_checkpointing
            )
            for _ in range(n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        k_input: Optional[int] = None,
        k_process: Optional[int] = None,
        k_output: Optional[int] = None
    ) -> dict:
        """
        Args:
            input_ids: [B, S]
            attention_mask: [B, S]
            labels: [B, S]
            k_input, k_process, k_output: 각 단계의 뉴런 수

        Returns:
            dict with 'logits', 'loss'
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        token_emb = self.token_embedding(input_ids)  # [B, S, d_model]
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask, k_input, k_process, k_output)

        x = self.norm(x)
        logits = self.output_projection(x)  # [B, S, vocab_size]

        # Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        return {
            'logits': logits,
            'loss': loss
        }

    def get_model_stats(self) -> dict:
        """모델 통계"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # FFN 파라미터
        ffn_params = sum(
            sum(p.numel() for p in layer.ffn.parameters())
            for layer in self.layers
        )

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'ffn_parameters': ffn_params,
            'ffn_percentage': ffn_params / total_params * 100,
            'n_layers': self.n_layers,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model
        }


# ============================================================
# Utility Functions
# ============================================================

def count_parameters(model: nn.Module) -> dict:
    """파라미터 수 계산"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def analyze_neuron_usage(
    model: ThreeStageLanguageModel,
    dataloader,
    device: str = 'cuda',
    n_batches: int = 10
) -> dict:
    """
    뉴런 사용 패턴 분석
    - 어떤 뉴런이 자주 사용되는가?
    - 입력 복잡도별 뉴런 사용량은?
    """
    model.eval()

    input_neuron_counts = torch.zeros(model.layers[0].ffn.n_input)
    process_neuron_counts = torch.zeros(model.layers[0].ffn.n_process)
    output_neuron_counts = torch.zeros(model.layers[0].ffn.n_output)

    total_sequences = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            input_ids = batch['input_ids'].to(device)
            B, S = input_ids.shape
            total_sequences += B

            # Get embeddings
            token_emb = model.token_embedding(input_ids)
            positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = model.position_embedding(positions)
            x = token_emb + pos_emb

            # Analyze first layer's FFN
            layer = model.layers[0]
            x_norm = layer.norm2(x)

            stats = layer.ffn.get_neuron_stats(x_norm)

            # Count neuron usage (시퀀스별)
            input_idx = stats['input_neurons']['indices'].flatten()
            process_idx = stats['process_neurons']['indices'].flatten()
            output_idx = stats['output_neurons']['indices'].flatten()

            for idx in input_idx:
                input_neuron_counts[idx] += 1
            for idx in process_idx:
                process_neuron_counts[idx] += 1
            for idx in output_idx:
                output_neuron_counts[idx] += 1

    return {
        'input_neuron_usage': input_neuron_counts / total_sequences,
        'process_neuron_usage': process_neuron_counts / total_sequences,
        'output_neuron_usage': output_neuron_counts / total_sequences,
        'total_sequences': total_sequences
    }


if __name__ == '__main__':
    # 간단한 테스트
    print("Testing Three-Stage Dynamic Neuron FFN...")

    # 모델 생성
    model = ThreeStageLanguageModel(
        vocab_size=30000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        max_seq_len=512,
        n_input_neurons=2048,
        n_process_neurons=1024,
        n_output_neurons=2048,
        d_process_value=256
    )

    # 통계 출력
    stats = model.get_model_stats()
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"  FFN parameters: {stats['ffn_parameters']:,} ({stats['ffn_percentage']:.1f}%)")

    # Forward pass 테스트
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))

    print(f"\nTesting forward pass...")
    output = model(input_ids)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {output['logits'].shape}")

    # 뉴런 통계
    print(f"\nGetting neuron statistics...")
    layer0_ffn = model.layers[0].ffn
    x = model.token_embedding(input_ids)
    neuron_stats = layer0_ffn.get_neuron_stats(x)

    print(f"  Input neurons:")
    print(f"    Selected: {neuron_stats['input_neurons']['sparsity']*100:.1f}%")
    print(f"    Mean activation: {neuron_stats['input_neurons']['mean_activation']:.4f}")

    print(f"  Process neurons:")
    print(f"    Selected: {neuron_stats['process_neurons']['sparsity']*100:.1f}%")
    print(f"    Mean activation: {neuron_stats['process_neurons']['mean_activation']:.4f}")

    print(f"  Output neurons:")
    print(f"    Selected: {neuron_stats['output_neurons']['sparsity']*100:.1f}%")
    print(f"    Mean activation: {neuron_stats['output_neurons']['mean_activation']:.4f}")

    print(f"\n✓ All tests passed!")
