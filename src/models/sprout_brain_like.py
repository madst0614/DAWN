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
    활성 뉴런들끼리 정보 교환 및 상태 업데이트

    뉴런들이 서로를 보고, 메시지를 주고받으며,
    자신의 활성화 강도와 내부 상태를 업데이트
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

        # 메시지 전달용 attention
        self.message_attention = nn.MultiheadAttention(
            d_state,
            num_heads=n_heads,
            batch_first=True
        )

        # 상태 업데이트 (GRU 스타일)
        self.state_update = nn.GRUCell(d_state, d_state)

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
        neuron_state: NeuronState,
        sparsity_k: int = 256
    ) -> NeuronState:
        """
        뉴런 상태 업데이트

        1. 활성 뉴런들끼리 메시지 교환
        2. 각 뉴런의 내부 상태 업데이트
        3. 활성화 강도 조정
        4. 새로운 뉴런 활성화 가능
        """
        device = neuron_state.activation.device

        # 현재 활성 뉴런 찾기
        active_mask = neuron_state.activation > 0.01
        active_indices = active_mask.nonzero(as_tuple=True)[0]

        if len(active_indices) == 0:
            return neuron_state

        # 활성 뉴런들의 데이터
        active_states = neuron_state.hidden_state[active_indices]  # [k, d_state]
        active_activations = neuron_state.activation[active_indices]  # [k]

        # 1. 메시지 계산 (attention)
        messages = self.compute_messages(
            active_states,
            active_activations
        )  # [k, d_state]

        # 2. 상태 업데이트
        new_states = self.update_states(
            active_states,
            messages
        )  # [k, d_state]

        # 3. 활성화 강도 업데이트
        new_activations = self.update_activations(
            active_states,
            new_states,
            active_activations
        )  # [k]

        # 4. 전역 상태에 반영
        new_neuron_state = NeuronState(
            activation=torch.zeros_like(neuron_state.activation),
            hidden_state=torch.zeros_like(neuron_state.hidden_state)
        )

        new_neuron_state.activation[active_indices] = new_activations
        new_neuron_state.hidden_state[active_indices] = new_states

        # 5. 희소성 유지 (top-k만 유지)
        new_neuron_state = self.maintain_sparsity(new_neuron_state, k=sparsity_k)

        return new_neuron_state

    def compute_messages(
        self,
        states: torch.Tensor,
        activations: torch.Tensor
    ) -> torch.Tensor:
        """
        활성 뉴런들끼리 메시지 교환

        Attention으로 구현:
        - Query, Key, Value = 뉴런 상태
        - Attention weight는 활성화 강도로 조정
        """
        # [k, d_state] → [1, k, d_state] (batch dimension 추가)
        states_batched = states.unsqueeze(0)

        # Self-attention
        messages, attn_weights = self.message_attention(
            states_batched,
            states_batched,
            states_batched
        )

        # [1, k, d_state] → [k, d_state]
        messages = messages.squeeze(0)

        # 활성화 강도로 가중
        activation_weights = activations.unsqueeze(-1)  # [k, 1]
        messages = messages * activation_weights

        return messages

    def update_states(
        self,
        old_states: torch.Tensor,
        messages: torch.Tensor
    ) -> torch.Tensor:
        """GRU 스타일 상태 업데이트"""
        # GRUCell: (input, hidden) → new_hidden
        new_states = self.state_update(messages, old_states)
        new_states = self.norm(new_states)

        return new_states

    def update_activations(
        self,
        old_states: torch.Tensor,
        new_states: torch.Tensor,
        old_activations: torch.Tensor
    ) -> torch.Tensor:
        """
        활성화 강도 업데이트

        상태 변화를 보고 활성화를 강화하거나 약화
        """
        # 이전 상태와 새 상태를 concat
        combined = torch.cat([old_states, new_states], dim=-1)  # [k, d_state*2]

        # 델타 계산
        delta = self.activation_update(combined).squeeze(-1)  # [k]

        # 업데이트 (약간의 관성 유지)
        new_activations = 0.7 * old_activations + 0.3 * delta

        # [0, 1] 범위로 클램프
        new_activations = torch.clamp(new_activations, 0.0, 1.0)

        return new_activations

    def maintain_sparsity(self, neuron_state: NeuronState, k: int) -> NeuronState:
        """
        희소성 유지: 상위 k개만 유지, 나머지 제거
        """
        activations = neuron_state.activation

        # Top-k 선택
        topk_values, topk_indices = torch.topk(activations, min(k, len(activations)))

        # 새 sparse 상태
        sparse_state = NeuronState(
            activation=torch.zeros_like(activations),
            hidden_state=torch.zeros_like(neuron_state.hidden_state)
        )

        sparse_state.activation[topk_indices] = topk_values
        sparse_state.hidden_state[topk_indices] = neuron_state.hidden_state[topk_indices]

        return sparse_state


class OutputDecoder(nn.Module):
    """
    뉴런 활성화 패턴 → 예측

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

    def forward(self, neuron_state: NeuronState) -> torch.Tensor:
        """
        neuron_state: NeuronState
        returns: [vocab_size] logits
        """
        # 활성화 패턴
        activation = neuron_state.activation  # [n_neurons]

        # Dense 표현으로 변환
        dense = self.to_dense(activation)  # [d_state]
        dense = self.norm(dense)

        # 예측 logits
        logits = self.to_logits(dense)  # [vocab_size]

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

        또는 return_activation_history=True면:
        returns: (logits, activation_history)
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        # 1. 초기 활성화 패턴 생성
        initial_activation = self.input_encoder(tokens)  # [batch, n_neurons]

        # Batch 처리
        all_logits = []
        all_histories = [] if return_activation_history else None

        for b in range(batch_size):
            # 각 샘플 독립 처리
            activation = initial_activation[b]  # [n_neurons]

            # 2. 뉴런 상태 초기화
            state = NeuronState.create(self.n_neurons, self.d_state, device)
            state.activation = activation

            # 활성 뉴런의 hidden state 초기화
            active_indices = (activation > 0.01).nonzero(as_tuple=True)[0]
            if len(active_indices) > 0:
                state.hidden_state[active_indices] = \
                    self.neuron_pool.neuron_signatures[active_indices]

            # 3. 반복적 상호작용
            history = [state.activation.detach().cpu()] if return_activation_history else None

            for step in range(self.n_interaction_steps):
                state = self.interaction(state, sparsity_k=self.final_sparsity)

                if return_activation_history:
                    history.append(state.activation.detach().cpu())

            # 4. 출력 디코딩
            logits = self.output_decoder(state)  # [vocab_size]

            all_logits.append(logits)
            if return_activation_history:
                all_histories.append(history)

        # Stack batch
        logits = torch.stack(all_logits)  # [batch, vocab_size]

        if return_activation_history:
            return logits, all_histories
        else:
            return logits

    def analyze_activation(self, tokens: torch.Tensor) -> dict:
        """
        활성화 패턴 분석

        각 step별로 어떤 뉴런이 활성화되는지 추적
        """
        with torch.no_grad():
            logits, histories = self.forward(tokens, return_activation_history=True)

            # 첫 번째 샘플만 분석
            history = histories[0]

            analysis = {
                'n_steps': len(history),
                'initial_active': (history[0] > 0.01).sum().item(),
                'final_active': (history[-1] > 0.01).sum().item(),
                'activation_history': history,
                'top_neurons_per_step': []
            }

            for step, activation in enumerate(history):
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
