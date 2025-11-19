"""
DAWN: Dynamic Architecture With Neurons

Attention-Guided Routing with Information Preservation

핵심 구조:
- GlobalRouter: Attention 기반 뉴런 선택 (정보 보존)
- HierarchicalDynamicFFN: Input → Process 2단계 뉴런 구조
- Standalone DAWN: Transformer 불필요
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ============================================================
# Attention-Guided Global Router
# ============================================================

class GlobalRouter(nn.Module):
    """
    Attention 정보를 최대한 보존하면서 뉴런 선택

    핵심:
    1. Multi-head Attention으로 맥락 파악
    2. Attended 표현 유지 (정보 손실 없음)
    3. Attention weights로 position 중요도 파악
    4. 세 가지 관점 결합: content, importance, pattern
    """

    def __init__(
        self,
        d_model: int = 512,
        n_input_neurons: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_input = n_input_neurons
        self.n_heads = n_heads

        # Multi-head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Content-based Scoring
        self.content_to_neuron = nn.Linear(d_model, n_input_neurons)

        # Routing 통계
        self.input_neuron_counts = None
        self.last_routing_scores = None

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.content_to_neuron.weight)
        if self.content_to_neuron.bias is not None:
            nn.init.zeros_(self.content_to_neuron.bias)

    def forward(
        self,
        x: torch.Tensor,
        k_input: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S, d_model] - position embedded 입력
            k_input: 선택할 뉴런 개수
            attention_mask: [B, S] - padding mask

        Returns:
            input_idx: [B, k_input] - 선택된 뉴런 인덱스
            routing_weights: [B, n_input] - Soft routing weights
            attended: [B, S, d_model] - Attention된 표현 (정보 보존!)
        """
        B, S, d_model = x.shape

        # Step 1: Multi-head Attention
        attended, attn_weights = self.attention(
            x, x, x,
            key_padding_mask=attention_mask,
            need_weights=True,
            average_attn_weights=False
        )
        # attended: [B, S, d_model]
        # attn_weights: [B, n_heads, S, S]

        # Step 2: Content-based Neuron Affinity
        neuron_affinity = self.content_to_neuron(attended)
        # [B, S, n_input]

        # Step 3: Position Importance from Attention
        position_importance = attn_weights.mean(dim=1).sum(dim=2)
        # [B, S]

        position_importance = position_importance / (
            position_importance.sum(dim=1, keepdim=True) + 1e-8
        )
        # [B, S]

        # Step 4: Combine with Three Strategies

        # Attention-weighted average
        weighted_scores = (
            neuron_affinity * position_importance.unsqueeze(-1)
        ).sum(dim=1)
        # [B, n_input]

        # Max pooling
        max_scores, _ = neuron_affinity.max(dim=1)
        # [B, n_input]

        # Mean pooling
        mean_scores = neuron_affinity.mean(dim=1)
        # [B, n_input]

        # Combine strategies
        final_scores = (
            0.5 * weighted_scores +
            0.3 * max_scores +
            0.2 * mean_scores
        )
        # [B, n_input]

        # Step 5: Neuron Selection
        routing_probs = F.softmax(final_scores, dim=-1)

        # Top-k selection
        _, input_idx = final_scores.topk(k_input, dim=-1)
        # [B, k_input]

        # Straight-through estimator
        one_hot = torch.zeros_like(routing_probs)
        one_hot.scatter_(1, input_idx, 1.0)
        routing_weights = (one_hot - routing_probs).detach() + routing_probs
        # [B, n_input]

        # 통계 저장
        if self.training:
            if self.input_neuron_counts is None:
                self.input_neuron_counts = torch.zeros(
                    self.n_input, device=x.device, dtype=torch.float32
                )
            self.input_neuron_counts += routing_weights.sum(dim=0).detach()
            self.last_routing_scores = routing_weights.detach()

        return input_idx, routing_weights, attended


# ============================================================
# Input Neurons with Self-Attention
# ============================================================

class InputNeurons(nn.Module):
    """
    뉴런 공간에서 Self-Attention

    핵심:
    1. 토큰을 뉴런 공간으로 변환
    2. 뉴런끼리 Self-Attention
    3. Residual로 원본 보존
    """

    def __init__(
        self,
        d_model: int = 512,
        n_input_neurons: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_input = n_input_neurons

        # 뉴런 패턴
        self.input_patterns = nn.Parameter(
            torch.empty(n_input_neurons, d_model)
        )

        # 뉴런 공간에서 Self-Attention
        self.neuron_attention = nn.MultiheadAttention(
            embed_dim=n_input_neurons,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(n_input_neurons)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.input_patterns, gain=math.sqrt(2.0))

    def forward(self, attended: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attended: [B, S, d_model] - Router의 attended 출력

        Returns:
            input_acts: [B, S, n_input] - 뉴런 activation
        """
        # Step 1: 토큰을 뉴런 공간으로 변환
        local_acts = F.gelu(attended @ self.input_patterns.T)
        # [B, S, n_input]

        # Step 2: 뉴런 공간에서 Self-Attention
        attn_out, _ = self.neuron_attention(
            local_acts, local_acts, local_acts
        )
        # [B, S, n_input]

        # Step 3: Residual + Norm
        input_acts = self.norm(local_acts + self.dropout(attn_out))
        # [B, S, n_input]

        return input_acts


# ============================================================
# Process Neurons
# ============================================================

class ProcessNeurons(nn.Module):
    """
    선택된 입력 뉴런들을 조합하여 출력 생성
    """

    def __init__(
        self,
        d_model: int = 512,
        n_input_neurons: int = 64,
        n_process_neurons: int = 128
    ):
        super().__init__()

        self.n_input = n_input_neurons
        self.n_process = n_process_neurons
        self.d_model = d_model

        # Process weights
        self.process_weights = nn.Parameter(
            torch.empty(n_process_neurons, n_input_neurons)
        )

        # Process outputs
        self.process_outputs = nn.Parameter(
            torch.empty(n_process_neurons, d_model)
        )

        # 통계
        self.process_neuron_counts = None

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.process_weights, gain=math.sqrt(2.0))
        nn.init.orthogonal_(self.process_outputs, gain=math.sqrt(2.0))

    def forward(
        self,
        selected_input_acts: torch.Tensor,
        input_idx: torch.Tensor,
        k_process: int
    ) -> torch.Tensor:
        """
        Args:
            selected_input_acts: [B, S, k_input]
            input_idx: [B, k_input]
            k_process: 선택할 process 뉴런 수

        Returns:
            output: [B, S, d_model]
        """
        B, S, k_input = selected_input_acts.shape

        # Process weights 수집
        process_weights_expanded = self.process_weights.unsqueeze(0).expand(
            B, -1, -1
        )  # [B, n_process, n_input]

        input_idx_expanded = input_idx.unsqueeze(1).expand(
            -1, self.n_process, -1
        )  # [B, n_process, k_input]

        selected_process_weights = torch.gather(
            process_weights_expanded, 2, input_idx_expanded
        )  # [B, n_process, k_input]

        # Process activations
        process_acts = F.gelu(
            torch.bmm(
                selected_input_acts,
                selected_process_weights.transpose(1, 2)
            )
        )  # [B, S, n_process]

        # Process neuron 선택
        process_scores = process_acts.mean(dim=1)  # [B, n_process]
        _, process_idx = process_scores.topk(k_process, dim=-1)
        # [B, k_process]

        # 선택된 process neurons
        expanded_process_idx = process_idx.unsqueeze(1).expand(-1, S, -1)
        selected_process_acts = torch.gather(
            process_acts, 2, expanded_process_idx
        )  # [B, S, k_process]

        # Output 가중치
        expanded_process_idx_for_output = process_idx.unsqueeze(2).expand(
            -1, -1, self.d_model
        )
        selected_process_outputs = torch.gather(
            self.process_outputs.unsqueeze(0).expand(B, -1, -1),
            1,
            expanded_process_idx_for_output
        )  # [B, k_process, d_model]

        # 최종 출력
        output = torch.bmm(selected_process_acts, selected_process_outputs)
        # [B, S, d_model]

        # 통계 저장
        if self.training:
            if self.process_neuron_counts is None:
                self.process_neuron_counts = torch.zeros(
                    self.n_process, device=output.device, dtype=torch.float32
                )
            ones = torch.ones_like(process_idx, dtype=torch.float32)
            self.process_neuron_counts.scatter_add_(
                0, process_idx.flatten(), ones.flatten()
            )

        return output


# ============================================================
# Hierarchical Dynamic FFN (Complete DAWN Block)
# ============================================================

class HierarchicalDynamicFFN(nn.Module):
    """
    완전한 DAWN Block - Attention-Guided Routing

    구조:
    1. GlobalRouter (Attention 기반, 정보 보존)
    2. InputNeurons (뉴런 Self-Attention)
    3. ProcessNeurons (조합 및 출력)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_input_neurons: int = 64,
        n_process_neurons: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_input = n_input_neurons
        self.n_process = n_process_neurons

        # Global Router
        self.global_router = GlobalRouter(
            d_model=d_model,
            n_input_neurons=n_input_neurons,
            n_heads=n_heads,
            dropout=dropout
        )

        # Input Neurons
        self.input_neurons = InputNeurons(
            d_model=d_model,
            n_input_neurons=n_input_neurons,
            n_heads=4,
            dropout=dropout
        )

        # Process Neurons
        self.process_neurons = ProcessNeurons(
            d_model=d_model,
            n_input_neurons=n_input_neurons,
            n_process_neurons=n_process_neurons
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        k_input: Optional[int] = None,
        k_process: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, d_model]
            k_input: 선택할 입력 뉴런 수
            k_process: 선택할 처리 뉴런 수
            attention_mask: [B, S] padding mask

        Returns:
            output: [B, S, d_model]
        """
        B, S, d_model = x.shape

        # Default k values
        if k_input is None:
            k_input = self.n_input // 2
        if k_process is None:
            k_process = self.n_process // 2

        # Stage 1: Global Router
        input_idx, routing_weights, attended = self.global_router(
            x, k_input, attention_mask
        )
        # input_idx: [B, k_input]
        # routing_weights: [B, n_input]
        # attended: [B, S, d_model] ← 정보 보존!

        # Stage 2: Input Neurons
        input_acts = self.input_neurons(attended)
        # [B, S, n_input]

        # Routing 적용
        weighted_acts = input_acts * routing_weights.unsqueeze(1)
        # [B, S, n_input]

        # 선택된 뉴런만
        expanded_idx = input_idx.unsqueeze(1).expand(-1, S, -1)
        selected_input_acts = torch.gather(weighted_acts, 2, expanded_idx)
        # [B, S, k_input]

        # Stage 3: Process Neurons
        output = self.process_neurons(
            selected_input_acts, input_idx, k_process
        )
        # [B, S, d_model]

        return self.dropout(output)

    def get_load_balance_loss(self) -> torch.Tensor:
        """Load balancing loss 계산"""
        if self.global_router.input_neuron_counts is None or not self.training:
            device = self.input_neurons.input_patterns.device
            return torch.tensor(0.0, device=device)

        counts = self.global_router.input_neuron_counts
        device = counts.device

        if counts.sum() == 0:
            return torch.tensor(0.0, device=device)

        # 정규화
        usage_probs = counts / (counts.sum() + 1e-8)

        # 목표: 균등 분포
        target_prob = 1.0 / self.n_input
        target = torch.full_like(usage_probs, target_prob)

        # KL divergence
        usage_probs = usage_probs + 1e-8
        target = target + 1e-8

        kl_loss = F.kl_div(
            usage_probs.log(),
            target,
            reduction='sum'
        ) / self.n_input

        # Routing entropy
        if self.global_router.last_routing_scores is not None:
            avg_probs = self.global_router.last_routing_scores.mean(dim=0) + 1e-8
            entropy = -(avg_probs * avg_probs.log()).sum()
            max_entropy = torch.log(
                torch.tensor(float(self.n_input), device=device)
            )
            normalized_entropy = entropy / max_entropy
            entropy_loss = torch.clamp(1.0 - normalized_entropy, min=0.0, max=1.0)
        else:
            entropy_loss = torch.tensor(0.0, device=device)

        return kl_loss + entropy_loss

    def reset_routing_counts(self):
        """Routing 통계 초기화"""
        self.global_router.input_neuron_counts = None
        self.global_router.last_routing_scores = None
        self.process_neurons.process_neuron_counts = None

    def get_routing_entropy(self) -> float:
        """현재 routing의 entropy 계산"""
        if self.global_router.last_routing_scores is None:
            return 0.0

        avg_probs = self.global_router.last_routing_scores.mean(dim=0) + 1e-8
        avg_probs = avg_probs / avg_probs.sum()
        entropy = -(avg_probs * avg_probs.log()).sum()
        return entropy.item()

    def get_usage_statistics(self) -> dict:
        """뉴런 사용 통계 반환"""
        if self.global_router.input_neuron_counts is None:
            return {
                'total': self.n_input,
                'dead_count': self.n_input,
                'usage_counts': None
            }

        counts = self.global_router.input_neuron_counts.cpu()
        dead_count = (counts == 0).sum().item()

        # Gini coefficient
        sorted_counts = counts.sort()[0]
        n = len(counts)
        cumsum = torch.cumsum(sorted_counts, dim=0)
        total = counts.sum()
        if total > 0:
            gini = (2 * cumsum.sum() / (n * total) - (n + 1) / n).item()
        else:
            gini = 1.0

        # Top-k concentration
        top_k = min(10, self.n_input)
        top_counts = counts.topk(top_k)[0]
        concentration = (top_counts.sum() / total).item() if total > 0 else 0

        return {
            'total': self.n_input,
            'dead_count': dead_count,
            'dead_ratio': dead_count / self.n_input,
            'gini': gini,
            'top10_concentration': concentration,
            'usage_counts': counts.numpy()
        }


# ============================================================
# Transformer Layer with DAWN
# ============================================================

class TransformerLayerWithHierarchicalFFN(nn.Module):
    """
    DAWN Block만으로 구성된 Layer
    Transformer의 Attention + FFN을 대체
    """

    def __init__(
        self,
        d_model: int = 512,
        n_input_neurons: int = 64,
        n_process_neurons: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.ffn = HierarchicalDynamicFFN(
            d_model=d_model,
            n_input_neurons=n_input_neurons,
            n_process_neurons=n_process_neurons,
            n_heads=n_heads,
            dropout=dropout
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        k_input: Optional[int] = None,
        k_process: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, d_model]

        Returns:
            output: [B, S, d_model]
        """
        # DAWN processing
        ffn_out = self.ffn(x, k_input, k_process, attention_mask)

        # Residual + Norm
        output = self.norm(x + ffn_out)

        return output


# ============================================================
# Complete Language Model
# ============================================================

class HierarchicalLanguageModel(nn.Module):
    """
    DAWN 기반 언어 모델
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 512,
        n_layers: int = 12,
        n_input_neurons: int = 64,
        n_process_neurons: int = 128,
        n_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        **kwargs  # 호환성을 위한 추가 인자 무시
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # DAWN Layers
        self.layers = nn.ModuleList([
            TransformerLayerWithHierarchicalFFN(
                d_model=d_model,
                n_input_neurons=n_input_neurons,
                n_process_neurons=n_process_neurons,
                n_heads=n_heads,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

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
        k_process: Optional[int] = None
    ) -> dict:
        """
        Args:
            input_ids: [B, S]
            attention_mask: [B, S]
            labels: [B, S]

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embedding(positions)

        x = self.dropout(token_emb + pos_emb)

        # DAWN Layers
        for layer in self.layers:
            x = layer(x, attention_mask, k_input, k_process)

        x = self.norm(x)
        logits = self.output_projection(x)

        # Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )

        return {
            'logits': logits,
            'loss': loss
        }

    def get_model_stats(self) -> dict:
        """모델 통계"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        ffn_params = sum(
            sum(p.numel() for p in layer.ffn.parameters())
            for layer in self.layers
        )

        router_params = sum(
            sum(p.numel() for p in layer.ffn.global_router.parameters())
            for layer in self.layers
        )

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'ffn_parameters': ffn_params,
            'router_parameters': router_params,
            'ffn_percentage': ffn_params / total_params * 100,
            'router_percentage': router_params / total_params * 100,
            'n_layers': self.n_layers,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model
        }


# ============================================================
# Test & Demo
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("DAWN: Dynamic Architecture With Neurons")
    print("Attention-Guided Routing with Information Preservation")
    print("=" * 60)
    print()

    # 모델 생성
    model = HierarchicalLanguageModel(
        vocab_size=30000,
        d_model=512,
        n_layers=6,
        n_input_neurons=64,
        n_process_neurons=128,
        n_heads=8,
        max_seq_len=512,
        dropout=0.1
    )

    # 파라미터 통계
    stats = model.get_model_stats()

    print("Model Statistics:")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"  FFN parameters: {stats['ffn_parameters']:,} ({stats['ffn_percentage']:.1f}%)")
    print(f"  Router parameters: {stats['router_parameters']:,} ({stats['router_percentage']:.1f}%)")
    print()

    # Forward pass 테스트
    batch_size = 4
    seq_len = 128

    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    labels = torch.randint(0, 30000, (batch_size, seq_len))

    print("Testing forward pass...")
    output = model(input_ids, labels=labels)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")
    print()

    print("✓ All tests passed!")
    print()
    print("Key Features:")
    print("  1. Attention-Guided Router - 정보 보존")
    print("  2. Input Neurons with Self-Attention")
    print("  3. Process Neurons - 효율적 조합")
    print("  4. Standalone DAWN - Transformer 불필요")
