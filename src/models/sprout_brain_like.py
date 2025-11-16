"""
SPROUT Brain-Like Architecture

완전히 새로운 뇌 기반 패러다임:
1. 고정된 전역 뉴런 풀
2. 입력에 따라 sparse 활성화
3. 활성 뉴런들이 협력 처리
4. 토큰별이 아니라 전체 패턴
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class NeuronState:
    """
    각 뉴런의 현재 상태

    activation: 각 뉴런의 현재 활성화 강도 (스칼라) [n_neurons]
    hidden_state: 각 뉴런의 내부 상태 (벡터) [n_neurons, d_state]
    """
    activation: torch.Tensor  # [n_neurons]
    hidden_state: torch.Tensor  # [n_neurons, d_state]

    @classmethod
    def create(cls, n_neurons: int, d_state: int, device: torch.device):
        """새로운 뉴런 상태 생성"""
        return cls(
            activation=torch.zeros(n_neurons, device=device),
            hidden_state=torch.zeros(n_neurons, d_state, device=device)
        )


class GlobalNeuronPool(nn.Module):
    """
    고정된 전역 뉴런 풀

    뉴런들은 항상 존재하며, 각각 고유한 "정체성"을 가짐
    입력에 따라 일부만 sparse하게 활성화됨
    """
    def __init__(self, n_neurons: int = 4096, d_state: int = 256):
        super().__init__()

        self.n_neurons = n_neurons
        self.d_state = d_state

        # 각 뉴런의 "특성" - 뉴런의 정체성을 정의
        # 뉴런_0: [0.5, 0.8, -0.3, ..., 0.2]  # "명사" 특성
        # 뉴런_234: [0.1, 0.9, 0.2, ..., 0.7]  # "복수" 특성
        self.neuron_signatures = nn.Parameter(
            torch.randn(n_neurons, d_state) / math.sqrt(d_state)
        )

        # 뉴런 간 연결 가중치 (희소하게 유지)
        # 어떤 뉴런들이 서로 강하게 연결되어 있는지
        self.connection_strength = nn.Parameter(
            torch.randn(n_neurons, n_neurons) * 0.01
        )

    def get_signatures(self, indices: torch.Tensor) -> torch.Tensor:
        """특정 뉴런들의 signature 가져오기"""
        return self.neuron_signatures[indices]

    def get_connections(self, indices: torch.Tensor) -> torch.Tensor:
        """특정 뉴런들 간의 연결 강도"""
        # [k, k] 서브매트릭스
        return self.connection_strength[indices][:, indices]


class InputToActivation(nn.Module):
    """
    토큰 시퀀스 → 뉴런 활성화 패턴

    전체 시퀀스를 하나의 sparse 뉴런 활성화 패턴으로 변환
    """
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 256,
        n_neurons: int = 4096,
        n_heads: int = 4,
        n_layers: int = 2,
        top_k: int = 128
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_neurons = n_neurons
        self.top_k = top_k

        # 토큰 임베딩
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)  # max seq len

        # 시퀀스 인코더 (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # 시퀀스 → 단일 벡터 (attention pooling)
        self.pooling_query = nn.Parameter(torch.randn(1, d_model))
        self.pooling_attention = nn.MultiheadAttention(
            d_model, num_heads=n_heads, batch_first=True
        )

        # 단일 벡터 → 뉴런 활성화 패턴
        self.to_activation = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, n_neurons)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [batch, seq_len]
        returns: [batch, n_neurons] sparse activation pattern
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # 1. 토큰 임베딩
        token_emb = self.token_embedding(tokens)  # [batch, seq, d_model]

        # 2. Position 임베딩
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        embeddings = token_emb + pos_emb

        # 3. 시퀀스 인코딩
        encoded = self.sequence_encoder(embeddings)  # [batch, seq, d_model]

        # 4. Attention pooling으로 단일 벡터로 통합
        query = self.pooling_query.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 1, d_model]
        pooled, _ = self.pooling_attention(
            query, encoded, encoded
        )  # [batch, 1, d_model]
        pooled = pooled.squeeze(1)  # [batch, d_model]

        # 5. 뉴런 활성화 패턴 생성
        activation_logits = self.to_activation(pooled)  # [batch, n_neurons]

        # 6. Sparse 활성화 (top-k)
        activation = self.sparse_activation(activation_logits, k=self.top_k)

        return activation

    def sparse_activation(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Top-k sparse activation
        상위 k개만 활성화, 나머지는 0
        """
        batch_size = logits.shape[0]

        # Top-k 선택
        topk_values, topk_indices = torch.topk(logits, k, dim=-1)

        # Sigmoid로 [0, 1] 범위로
        topk_activations = torch.sigmoid(topk_values)

        # Sparse 텐서 생성
        sparse_activation = torch.zeros_like(logits)
        sparse_activation.scatter_(1, topk_indices, topk_activations)

        return sparse_activation


class NeuronInteraction(nn.Module):
    """
    활성 뉴런들끼리 정보 교환 및 상태 업데이트 (BATCH 처리)

    뉴런들이 서로를 보고, 메시지를 주고받으며,
    자신의 활성화 강도와 내부 상태를 업데이트

    **중요: 배치 전체를 한 번에 처리!**
    """
    def __init__(
        self,
        n_neurons: int = 4096,
        d_state: int = 256,
        n_heads: int = 4
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.d_state = d_state
        self.n_heads = n_heads

        # 메시지 전달용 attention (batch 처리)
        self.message_attention = nn.MultiheadAttention(
            d_state,
            num_heads=n_heads,
            batch_first=True
        )

        # 상태 업데이트 (Linear로 변경 - 배치 처리 가능)
        self.state_update = nn.Sequential(
            nn.Linear(d_state * 2, d_state),  # [old_state, message] concat
            nn.GELU(),
            nn.Linear(d_state, d_state)
        )

        # 활성화 강도 업데이트
        self.activation_update = nn.Sequential(
            nn.Linear(d_state * 2, d_state),
            nn.GELU(),
            nn.Linear(d_state, 1),
            nn.Sigmoid()
        )

        # Layer normalization
        self.norm = nn.LayerNorm(d_state)

    def forward(
        self,
        activation: torch.Tensor,  # [batch, n_neurons]
        hidden_state: torch.Tensor,  # [batch, n_neurons, d_state]
        sparsity_k: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        뉴런 상태 업데이트 (BATCH 처리!)

        Args:
            activation: [batch, n_neurons]
            hidden_state: [batch, n_neurons, d_state]
            sparsity_k: 최대 활성 뉴런 수

        Returns:
            new_activation: [batch, n_neurons]
            new_hidden_state: [batch, n_neurons, d_state]
        """
        batch_size = activation.shape[0]

        # 1. 활성 마스크
        active_mask = activation > 0.01  # [batch, n_neurons]

        # 2. 메시지 계산 (전체 뉴런, masked attention)
        # Attention은 모든 뉴런에 대해 계산하지만,
        # 비활성 뉴런은 마스크로 제외
        messages = self.compute_messages(
            hidden_state,  # [batch, n_neurons, d_state]
            activation,    # [batch, n_neurons]
            active_mask    # [batch, n_neurons]
        )  # [batch, n_neurons, d_state]

        # 3. 상태 업데이트
        new_hidden = self.update_states(
            hidden_state,  # [batch, n_neurons, d_state]
            messages       # [batch, n_neurons, d_state]
        )  # [batch, n_neurons, d_state]

        # 4. 활성화 강도 업데이트
        new_activation = self.update_activations(
            hidden_state,  # [batch, n_neurons, d_state]
            new_hidden,    # [batch, n_neurons, d_state]
            activation     # [batch, n_neurons]
        )  # [batch, n_neurons]

        # 5. 희소성 유지 (top-k)
        new_activation, new_hidden = self.maintain_sparsity(
            new_activation,
            new_hidden,
            k=sparsity_k
        )

        return new_activation, new_hidden

    def compute_messages(
        self,
        states: torch.Tensor,      # [batch, n_neurons, d_state]
        activations: torch.Tensor, # [batch, n_neurons]
        active_mask: torch.Tensor  # [batch, n_neurons]
    ) -> torch.Tensor:
        """
        활성 뉴런들끼리 메시지 교환 (진짜 SPARSE!)

        **메모리 효율:** 4096×4096 attention 대신 128×128만 계산!
        """
        batch_size, n_neurons, d_state = states.shape

        # 배치 전체에 대해 활성 뉴런 모으기
        # 각 샘플마다 다른 뉴런이 활성화될 수 있음

        # 방법: 각 배치별로 top-k 뉴런만 추출
        k = active_mask.sum(dim=-1).max().item()  # 최대 활성 뉴런 수
        k = min(k, 256)  # 최대 256개로 제한

        if k == 0:
            return torch.zeros_like(states)

        # Top-k 활성 뉴런의 인덱스 (배치별로)
        topk_values, topk_indices = torch.topk(activations, k=k, dim=-1)
        # topk_indices: [batch, k]

        # 활성 뉴런 상태 추출
        batch_indices = torch.arange(batch_size, device=states.device).unsqueeze(1).expand(-1, k)
        active_states = states[batch_indices, topk_indices]  # [batch, k, d_state]

        # Attention on active neurons only! (k×k instead of n_neurons×n_neurons)
        messages_sparse, _ = self.message_attention(
            active_states,  # Q: [batch, k, d_state] (k ≈ 128)
            active_states,  # K
            active_states   # V
        )  # [batch, k, d_state]

        # 활성화 강도로 가중
        active_weights = topk_values.unsqueeze(-1)  # [batch, k, 1]
        messages_sparse = messages_sparse * active_weights

        # 원래 크기로 복원 (sparse → dense)
        # Mixed precision 대응: messages_sparse의 dtype에 맞춰 생성
        messages = torch.zeros(batch_size, n_neurons, d_state,
                              dtype=messages_sparse.dtype,
                              device=states.device)
        messages[batch_indices, topk_indices] = messages_sparse

        return messages

    def update_states(
        self,
        old_states: torch.Tensor,  # [batch, n_neurons, d_state]
        messages: torch.Tensor     # [batch, n_neurons, d_state]
    ) -> torch.Tensor:
        """상태 업데이트 (배치 처리)"""
        # Concat old state + message
        combined = torch.cat([old_states, messages], dim=-1)  # [batch, n_neurons, d_state*2]

        # Update
        new_states = self.state_update(combined)  # [batch, n_neurons, d_state]
        new_states = self.norm(new_states)

        return new_states

    def update_activations(
        self,
        old_states: torch.Tensor,  # [batch, n_neurons, d_state]
        new_states: torch.Tensor,  # [batch, n_neurons, d_state]
        old_activations: torch.Tensor  # [batch, n_neurons]
    ) -> torch.Tensor:
        """활성화 강도 업데이트 (배치 처리)"""
        # Concat
        combined = torch.cat([old_states, new_states], dim=-1)  # [batch, n_neurons, d_state*2]

        # Delta
        delta = self.activation_update(combined).squeeze(-1)  # [batch, n_neurons]

        # Update (관성 유지)
        new_activations = 0.7 * old_activations + 0.3 * delta

        # Clamp
        new_activations = torch.clamp(new_activations, 0.0, 1.0)

        return new_activations

    def maintain_sparsity(
        self,
        activation: torch.Tensor,    # [batch, n_neurons]
        hidden_state: torch.Tensor,  # [batch, n_neurons, d_state]
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        희소성 유지: 배치 전체에 대해 top-k만 유지
        """
        batch_size, n_neurons = activation.shape

        # Top-k 선택 (배치별로)
        topk_values, topk_indices = torch.topk(activation, min(k, n_neurons), dim=-1)
        # topk_values: [batch, k]
        # topk_indices: [batch, k]

        # Sparse 텐서 생성 (dtype 유지)
        new_activation = torch.zeros_like(activation)
        new_hidden = torch.zeros_like(hidden_state)

        # 배치별로 scatter (mixed precision 대응)
        for b in range(batch_size):
            new_activation[b].scatter_(0, topk_indices[b], topk_values[b].to(new_activation.dtype))
            new_hidden[b, topk_indices[b]] = hidden_state[b, topk_indices[b]].to(new_hidden.dtype)

        return new_activation, new_hidden


class OutputDecoder(nn.Module):
    """
    뉴런 활성화 패턴 → 예측 (BATCH 처리)

    최종 뉴런 패턴을 출력 공간으로 디코딩
    """
    def __init__(
        self,
        n_neurons: int = 4096,
        d_state: int = 256,
        vocab_size: int = 30000
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.d_state = d_state
        self.vocab_size = vocab_size

        # 활성화 패턴 → dense 표현
        self.to_dense = nn.Sequential(
            nn.Linear(n_neurons, d_state * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_state * 2, d_state)
        )

        # Dense → output logits
        self.to_logits = nn.Linear(d_state, vocab_size)

        # Layer norm
        self.norm = nn.LayerNorm(d_state)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        """
        activation: [batch, n_neurons]
        returns: [batch, vocab_size] logits
        """
        # Dense 표현으로 변환
        dense = self.to_dense(activation)  # [batch, d_state]
        dense = self.norm(dense)

        # 예측 logits
        logits = self.to_logits(dense)  # [batch, vocab_size]

        return logits


class SPROUT_BrainLike(nn.Module):
    """
    뇌처럼 작동하는 SPROUT

    핵심 아이디어:
    1. 전역 뉴런 풀 (4096개 뉴런이 항상 존재)
    2. 입력에 따라 sparse 활성화 (128-256개만)
    3. 반복적 상호작용으로 패턴 정제 (5 steps)
    4. 전체 시퀀스를 하나의 패턴으로 표현

    기존 Transformer와의 차이:
    - Transformer: [batch, seq, d_model] - 토큰별 독립 처리
    - SPROUT: [n_neurons] sparse pattern - 전체를 하나의 패턴으로
    """
    def __init__(
        self,
        vocab_size: int = 30000,
        n_neurons: int = 4096,
        d_state: int = 256,
        n_heads: int = 4,
        encoder_layers: int = 2,
        n_interaction_steps: int = 5,
        initial_sparsity: int = 128,
        final_sparsity: int = 256
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_neurons = n_neurons
        self.d_state = d_state
        self.n_interaction_steps = n_interaction_steps
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity

        # 전역 뉴런 풀
        self.neuron_pool = GlobalNeuronPool(n_neurons, d_state)

        # 입력 인코더
        self.input_encoder = InputToActivation(
            vocab_size=vocab_size,
            d_model=d_state,
            n_neurons=n_neurons,
            n_heads=n_heads,
            n_layers=encoder_layers,
            top_k=initial_sparsity
        )

        # 뉴런 상호작용
        self.interaction = NeuronInteraction(
            n_neurons=n_neurons,
            d_state=d_state,
            n_heads=n_heads
        )

        # 출력 디코더
        self.output_decoder = OutputDecoder(
            n_neurons=n_neurons,
            d_state=d_state,
            vocab_size=vocab_size
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_activation_history: bool = False
    ) -> torch.Tensor:
        """
        tokens: [batch, seq_len]
        returns: [batch, vocab_size] logits

        **중요: 배치 전체를 한 번에 처리!**
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        # 1. 초기 활성화 패턴 생성 (배치 처리!)
        activation = self.input_encoder(tokens)  # [batch, n_neurons]

        # 2. 뉴런 hidden state 초기화 (배치 처리!)
        hidden_state = torch.zeros(batch_size, self.n_neurons, self.d_state, device=device)

        # 활성 뉴런에 signature 할당
        active_mask = activation > 0.01  # [batch, n_neurons]
        # 모든 뉴런에 signature를 브로드캐스트
        neuron_sigs = self.neuron_pool.neuron_signatures.unsqueeze(0).expand(batch_size, -1, -1)
        # [batch, n_neurons, d_state]
        hidden_state = neuron_sigs * active_mask.unsqueeze(-1)  # 비활성 뉴런은 0

        # 3. 반복적 상호작용 (배치 처리!)
        history = [activation.detach().cpu()] if return_activation_history else None

        for step in range(self.n_interaction_steps):
            activation, hidden_state = self.interaction(
                activation,
                hidden_state,
                sparsity_k=self.final_sparsity
            )

            if return_activation_history:
                history.append(activation.detach().cpu())

        # 4. 출력 디코딩 (배치 처리!)
        logits = self.output_decoder(activation)  # [batch, vocab_size]

        if return_activation_history:
            return logits, history
        else:
            return logits

    def analyze_activation(self, tokens: torch.Tensor) -> dict:
        """
        활성화 패턴 분석

        각 step별로 어떤 뉴런이 활성화되는지 추적
        """
        with torch.no_grad():
            logits, history = self.forward(tokens, return_activation_history=True)
            # history: list of [batch, n_neurons]

            # 첫 번째 샘플만 분석
            sample_history = [h[0] for h in history]  # [n_steps] of [n_neurons]

            analysis = {
                'n_steps': len(sample_history),
                'initial_active': (sample_history[0] > 0.01).sum().item(),
                'final_active': (sample_history[-1] > 0.01).sum().item(),
                'activation_history': sample_history,
                'top_neurons_per_step': []
            }

            for step, activation in enumerate(sample_history):
                top_values, top_indices = torch.topk(activation, k=10)
                analysis['top_neurons_per_step'].append({
                    'step': step,
                    'indices': top_indices.tolist(),
                    'values': top_values.tolist()
                })

            return analysis


def create_brain_like_sprout(
    vocab_size: int = 30000,
    n_neurons: int = 4096,
    d_state: int = 256,
    **kwargs
) -> SPROUT_BrainLike:
    """
    Brain-like SPROUT 모델 생성

    기본 설정:
    - 4096개 뉴런
    - 초기 128개 활성화
    - 5 step 반복 처리
    - 최종 256개까지 확장 가능
    """
    return SPROUT_BrainLike(
        vocab_size=vocab_size,
        n_neurons=n_neurons,
        d_state=d_state,
        **kwargs
    )


if __name__ == "__main__":
    # 간단한 테스트
    print("=" * 60)
    print("SPROUT Brain-Like Architecture Test")
    print("=" * 60)

    # 모델 생성
    model = create_brain_like_sprout(
        vocab_size=1000,
        n_neurons=512,
        d_state=128,
        n_interaction_steps=3
    )

    print(f"\n모델 파라미터:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total_params:,}")

    # 테스트 입력
    batch_size = 2
    seq_len = 10
    tokens = torch.randint(0, 1000, (batch_size, seq_len))

    print(f"\n입력 shape: {tokens.shape}")

    # Forward pass
    logits = model(tokens)
    print(f"출력 shape: {logits.shape}")

    # 활성화 분석
    print("\n" + "=" * 60)
    print("활성화 패턴 분석")
    print("=" * 60)

    analysis = model.analyze_activation(tokens[:1])

    print(f"\n초기 활성 뉴런: {analysis['initial_active']}")
    print(f"최종 활성 뉴런: {analysis['final_active']}")

    print("\n각 step별 Top 5 뉴런:")
    for step_info in analysis['top_neurons_per_step']:
        step = step_info['step']
        indices = step_info['indices'][:5]
        values = step_info['values'][:5]
        print(f"  Step {step}:")
        for idx, val in zip(indices, values):
            print(f"    뉴런_{idx}: {val:.3f}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
