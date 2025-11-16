"""
Brain-Like SPROUT 간단 테스트

Mixed precision으로 실제 작동 확인
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.models.sprout_brain_like import create_brain_like_sprout


def simple_test():
    """간단한 forward/backward 테스트"""

    print("=" * 70)
    print("BRAIN-LIKE SPROUT SIMPLE TEST")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # 모델 생성
    print("\n1. Creating model...")
    model = create_brain_like_sprout(
        vocab_size=1000,
        n_neurons=512,
        d_state=128,
        n_interaction_steps=5,
        initial_sparsity=64,
        final_sparsity=128
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # 테스트 입력
    print("\n2. Forward pass...")
    batch_size = 4
    seq_len = 20
    tokens = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Mixed precision
    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        logits = model(tokens)

    print(f"   Input shape: {tokens.shape}")
    print(f"   Output shape: {logits.shape}")

    # 활성화 분석
    print("\n3. Activation analysis...")
    with torch.no_grad():
        analysis = model.analyze_activation(tokens[:1])

    print(f"   Initial active neurons: {analysis['initial_active']}")
    print(f"   Final active neurons: {analysis['final_active']}")

    print("\n   Top 5 neurons per step:")
    for step_info in analysis['top_neurons_per_step']:
        step = step_info['step']
        indices = step_info['indices'][:5]
        values = step_info['values'][:5]
        print(f"     Step {step}: neurons {indices.tolist()} (max: {values[0]:.3f})")

    # Backward
    print("\n4. Backward pass...")
    target = torch.randint(0, 1000, (batch_size,), device=device)

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    if scaler:
        with torch.cuda.amp.autocast():
            loss = nn.functional.cross_entropy(logits, target)
        scaler.scale(loss).backward()
    else:
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()

    print(f"   Loss: {loss.item():.4f}")

    # Gradient 확인
    n_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_params = sum(1 for p in model.parameters())
    print(f"   Gradients: {n_grads}/{n_params} parameters")

    print("\n" + "=" * 70)
    print("SUCCESS! Model works correctly! ✅")
    print("=" * 70)

    # 메모리 사용량 (CUDA인 경우)
    if device.type == 'cuda':
        print(f"\nMemory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


if __name__ == "__main__":
    simple_test()
