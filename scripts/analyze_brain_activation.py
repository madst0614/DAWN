"""
뇌 기반 SPROUT 활성화 패턴 분석 도구

뉴런 활성화 패턴을 시각화하고 분석
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from src.models.sprout_brain_like import create_brain_like_sprout


def visualize_activation_evolution(
    activation_history: List[torch.Tensor],
    save_path: str = None
):
    """
    활성화 패턴의 시간에 따른 변화 시각화

    Args:
        activation_history: 각 step의 뉴런 활성화 [n_steps, n_neurons]
        save_path: 저장 경로 (None이면 표시만)
    """
    n_steps = len(activation_history)
    n_neurons = activation_history[0].shape[0]

    # Numpy로 변환
    if isinstance(activation_history[0], torch.Tensor):
        history_np = [act.cpu().numpy() for act in activation_history]
    else:
        history_np = activation_history

    # Figure 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 히트맵: 시간에 따른 활성 뉴런
    ax = axes[0, 0]
    active_neurons_per_step = []
    for act in history_np:
        # 활성 뉴런 인덱스
        active_idx = np.where(act > 0.01)[0]
        active_neurons_per_step.append(active_idx)

    # 전체 활성 뉴런 수집
    all_active = sorted(set(np.concatenate([list(idx) for idx in active_neurons_per_step])))

    if len(all_active) > 0:
        # 히트맵 데이터 생성
        heatmap = np.zeros((len(all_active), n_steps))
        for step, act in enumerate(history_np):
            for i, neuron_idx in enumerate(all_active):
                heatmap[i, step] = act[neuron_idx]

        im = ax.imshow(heatmap, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_xlabel('Processing Step')
        ax.set_ylabel('Neuron Index')
        ax.set_title('Activation Heatmap Over Time')
        plt.colorbar(im, ax=ax)

        # Y축 레이블 (일부만 표시)
        if len(all_active) > 20:
            step_size = len(all_active) // 10
            yticks = list(range(0, len(all_active), step_size))
            yticklabels = [str(all_active[i]) for i in yticks]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
        else:
            ax.set_yticks(range(len(all_active)))
            ax.set_yticklabels([str(idx) for idx in all_active])

    # 2. 활성 뉴런 수 변화
    ax = axes[0, 1]
    n_active = [np.sum(act > 0.01) for act in history_np]
    ax.plot(range(n_steps), n_active, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Processing Step')
    ax.set_ylabel('Number of Active Neurons')
    ax.set_title('Active Neurons Over Time')
    ax.grid(True, alpha=0.3)

    # 3. 활성화 강도 분포
    ax = axes[1, 0]
    for step in [0, n_steps // 2, n_steps - 1]:
        act = history_np[step]
        active_values = act[act > 0.01]
        if len(active_values) > 0:
            ax.hist(active_values, bins=20, alpha=0.5, label=f'Step {step}')
    ax.set_xlabel('Activation Strength')
    ax.set_ylabel('Count')
    ax.set_title('Activation Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Top-10 뉴런 추적
    ax = axes[1, 1]
    # 최종 step의 top-10 뉴런
    final_act = history_np[-1]
    top_indices = np.argsort(final_act)[-10:][::-1]

    for neuron_idx in top_indices:
        values = [act[neuron_idx] for act in history_np]
        ax.plot(range(n_steps), values, 'o-', label=f'Neuron {neuron_idx}', alpha=0.7)

    ax.set_xlabel('Processing Step')
    ax.set_ylabel('Activation Strength')
    ax.set_title('Top-10 Neurons Trajectory')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig


def analyze_neuron_specialization(
    model,
    test_inputs: List[torch.Tensor],
    test_labels: List[str]
):
    """
    뉴런들이 어떤 입력에 반응하는지 분석

    특정 뉴런이 특정 패턴(예: 복수형, 동물 등)에 특화되었는지 확인
    """
    n_neurons = model.n_neurons

    # 각 뉴런이 각 입력에 대해 얼마나 활성화되는지
    neuron_responses = np.zeros((n_neurons, len(test_inputs)))

    with torch.no_grad():
        for i, tokens in enumerate(test_inputs):
            analysis = model.analyze_activation(tokens.unsqueeze(0))
            final_activation = analysis['activation_history'][-1].numpy()
            neuron_responses[:, i] = final_activation

    # 각 뉴런의 "특화도" 계산
    # 특정 입력에만 강하게 반응하는가?
    specialization_scores = np.std(neuron_responses, axis=1)

    # 가장 특화된 뉴런들
    top_specialized = np.argsort(specialization_scores)[-20:][::-1]

    print("=" * 60)
    print("뉴런 특화도 분석")
    print("=" * 60)

    for neuron_idx in top_specialized:
        responses = neuron_responses[neuron_idx]
        # 가장 강하게 반응하는 입력
        top_input_idx = np.argmax(responses)

        print(f"\n뉴런_{neuron_idx}:")
        print(f"  특화도: {specialization_scores[neuron_idx]:.3f}")
        print(f"  가장 강한 반응: '{test_labels[top_input_idx]}' ({responses[top_input_idx]:.3f})")

        # 반응 패턴
        print(f"  반응 패턴:")
        for j, label in enumerate(test_labels):
            if responses[j] > 0.1:
                print(f"    {label}: {responses[j]:.3f}")


def compare_activation_patterns(
    model,
    text_pairs: List[tuple],
):
    """
    두 입력의 활성화 패턴 비교

    예: "cat" vs "cats" - 어떤 뉴런이 다르게 활성화되나?
    """
    from transformers import AutoTokenizer

    # 간단한 토크나이저 (실제로는 모델에 맞는 것 사용)
    # 여기서는 단순히 문자 코드 사용
    def simple_tokenize(text, max_len=10):
        tokens = [ord(c) % 1000 for c in text[:max_len]]
        tokens = tokens + [0] * (max_len - len(tokens))  # padding
        return torch.tensor(tokens).unsqueeze(0)

    print("=" * 60)
    print("활성화 패턴 비교")
    print("=" * 60)

    for text1, text2 in text_pairs:
        print(f"\n비교: '{text1}' vs '{text2}'")

        tokens1 = simple_tokenize(text1)
        tokens2 = simple_tokenize(text2)

        with torch.no_grad():
            analysis1 = model.analyze_activation(tokens1)
            analysis2 = model.analyze_activation(tokens2)

        act1 = analysis1['activation_history'][-1].numpy()
        act2 = analysis2['activation_history'][-1].numpy()

        # 공통 활성 뉴런
        active1 = set(np.where(act1 > 0.1)[0])
        active2 = set(np.where(act2 > 0.1)[0])

        common = active1 & active2
        only1 = active1 - active2
        only2 = active2 - active1

        print(f"  공통 활성 뉴런: {len(common)}개")
        print(f"  '{text1}' 고유: {len(only1)}개")
        print(f"  '{text2}' 고유: {len(only2)}개")

        if len(only1) > 0:
            print(f"    '{text1}' 고유 뉴런: {sorted(list(only1))[:10]}")
        if len(only2) > 0:
            print(f"    '{text2}' 고유 뉴런: {sorted(list(only2))[:10]}")


def main():
    """메인 분석 루틴"""

    print("=" * 60)
    print("SPROUT Brain-Like 활성화 패턴 분석")
    print("=" * 60)

    # 모델 생성
    model = create_brain_like_sprout(
        vocab_size=1000,
        n_neurons=512,
        d_state=128,
        n_interaction_steps=5
    )

    model.eval()

    # 테스트 입력
    print("\n1. 단일 입력 분석")
    print("-" * 60)

    test_input = torch.randint(0, 1000, (1, 15))
    print(f"입력 shape: {test_input.shape}")

    analysis = model.analyze_activation(test_input)

    print(f"\n초기 활성 뉴런: {analysis['initial_active']}개")
    print(f"최종 활성 뉴런: {analysis['final_active']}개")
    print(f"처리 steps: {analysis['n_steps']}")

    print("\nTop-5 뉴런 (각 step):")
    for step_info in analysis['top_neurons_per_step']:
        step = step_info['step']
        indices = step_info['indices'][:5]
        values = step_info['values'][:5]

        print(f"  Step {step}:")
        for idx, val in zip(indices, values):
            print(f"    뉴런_{idx:3d}: {val:.4f}")

    # 시각화
    print("\n2. 활성화 패턴 시각화")
    print("-" * 60)

    history = analysis['activation_history']
    fig = visualize_activation_evolution(history, save_path='activation_evolution.png')

    # 패턴 비교
    print("\n3. 다양한 입력 비교")
    print("-" * 60)

    # 다양한 길이의 입력
    inputs = [
        torch.randint(0, 1000, (1, 5)),   # 짧은 시퀀스
        torch.randint(0, 1000, (1, 10)),  # 중간
        torch.randint(0, 1000, (1, 20)),  # 긴 시퀀스
    ]
    labels = ["짧은 (5)", "중간 (10)", "긴 (20)"]

    for inp, label in zip(inputs, labels):
        analysis = model.analyze_activation(inp)
        print(f"\n{label}:")
        print(f"  초기 활성: {analysis['initial_active']}개")
        print(f"  최종 활성: {analysis['final_active']}개")

        # 활성화 진화
        history = analysis['activation_history']
        n_active_per_step = [(h > 0.01).sum().item() for h in history]
        print(f"  활성 뉴런 수 변화: {n_active_per_step}")

    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
