# SPROUT Brain-Like Architecture

## 완전 재설계: 뇌처럼 작동하는 신경망

기존 Transformer의 토큰별 처리 방식을 벗어나, **뇌의 작동 방식**을 모방한 완전히 새로운 아키텍처입니다.

---

## 🧠 핵심 아이디어

### 기존 Transformer
```
[batch, seq_len, d_model]
↓
각 토큰이 독립적인 표현을 가짐
토큰별로 병렬 처리
```

### Brain-Like SPROUT
```
[n_neurons] sparse activation pattern
↓
전체 시퀀스를 하나의 뉴런 활성화 패턴으로
뉴런들이 협력하며 반복적으로 처리
```

---

## 📋 4가지 핵심 원칙

1. **고정된 전역 뉴런 풀**
   - 4096개의 뉴런이 항상 존재
   - 각 뉴런은 고유한 "정체성" (signature)을 가짐

2. **입력에 따라 Sparse 활성화**
   - 입력마다 128-256개 뉴런만 활성화
   - 대부분의 뉴런은 비활성 상태

3. **활성 뉴런들이 협력 처리**
   - 활성화된 뉴런들끼리 메시지 교환
   - 서로 영향을 주고받으며 패턴 정제

4. **토큰별이 아니라 전체 패턴**
   - 전체 시퀀스를 하나의 활성화 패턴으로 표현
   - 반복적 처리로 패턴 수렴

---

## 🏗️ 아키텍처 구조

### 1. Global Neuron Pool
```python
class GlobalNeuronPool:
    # 4096개 뉴런의 "정체성"
    neuron_signatures: [4096, 256]

    # 뉴런 간 연결 강도
    connection_strength: [4096, 4096]
```

각 뉴런은 256차원 벡터로 표현되는 고유한 특성을 가짐:
```
뉴런_0:    [0.5, 0.8, -0.3, ...]  # "명사" 특성
뉴런_234:  [0.1, 0.9, 0.2, ...]   # "복수형" 특성
뉴런_1567: [0.8, 0.1, 0.9, ...]   # "동물" 특성
```

### 2. Input → Activation Pattern

```python
class InputToActivation:
    # 토큰 시퀀스를 뉴런 활성화 패턴으로 변환

    입력: ["The", "cats", "are"]
    ↓
    토큰 임베딩 + Transformer 인코더
    ↓
    Attention pooling으로 단일 벡터
    ↓
    4096차원 활성화 패턴
    ↓
    Top-k Sparse (128개만 활성화)
```

**결과:**
```
activation[0] = 0.0
activation[15] = 0.9   # 뉴런_15 강하게 활성화
activation[50] = 0.0
activation[234] = 0.8  # 뉴런_234 활성화
...
(총 128개만 non-zero)
```

### 3. Neuron Interaction (핵심!)

```python
class NeuronInteraction:
    # 활성 뉴런들끼리 상호작용

    for step in range(5):
        1. 활성 뉴런들끼리 메시지 교환 (Attention)
        2. 각 뉴런의 내부 상태 업데이트 (GRU)
        3. 활성화 강도 조정
        4. 새로운 뉴런 활성화 가능
```

**예시: "The cats are" 처리**

```
t=0 (초기):
  활성 뉴런: [15, 234, 567, ...]  (128개)
  강도: [0.9, 0.8, 0.7, ...]

t=1 (1차 상호작용):
  뉴런_15 → 뉴런_234 확인: "명사 + 복수, 관련 높음"
  뉴런_234 → 뉴런_15: "나도 관련 있어, 강화!"
  뉴런_1567 새로 활성화: "cats = 동물!"

  활성 뉴런: [15, 234, 567, 1567, ...]  (140개)
  강도: [0.95, 0.85, 0.65, 0.3, ...]

t=2-4:
  계속 정제...
  약한 뉴런 제거, 강한 뉴런 더 강화

t=5 (수렴):
  안정된 최종 패턴
  활성 뉴런: [15, 234, 1567, 2345, ...]  (130개)
  강도: [0.99, 0.95, 0.90, 0.85, ...]
```

### 4. Output Decoder

```python
class OutputDecoder:
    # 최종 뉴런 패턴 → 예측

    뉴런 활성화 패턴 [4096]
    ↓
    Dense 표현 [256]
    ↓
    Logits [vocab_size]
```

---

## 💻 사용법

### 1. 모델 생성

```python
from src.models.sprout_brain_like import create_brain_like_sprout

model = create_brain_like_sprout(
    vocab_size=30000,
    n_neurons=4096,          # 전역 뉴런 수
    d_state=256,             # 뉴런 상태 차원
    n_interaction_steps=5,   # 반복 처리 횟수
    initial_sparsity=128,    # 초기 활성 뉴런 수
    final_sparsity=256       # 최대 활성 뉴런 수
)
```

### 2. Forward Pass

```python
# 입력: 토큰 시퀀스
tokens = torch.tensor([[101, 2003, 1996, ...]])  # [batch, seq_len]

# 출력: 예측 logits
logits = model(tokens)  # [batch, vocab_size]
```

### 3. 활성화 패턴 분석

```python
# 뉴런 활성화 패턴 추적
analysis = model.analyze_activation(tokens)

print(f"초기 활성 뉴런: {analysis['initial_active']}개")
print(f"최종 활성 뉴런: {analysis['final_active']}개")

# 각 step별 Top 뉴런
for step_info in analysis['top_neurons_per_step']:
    print(f"Step {step_info['step']}:")
    for idx, val in zip(step_info['indices'][:5], step_info['values'][:5]):
        print(f"  뉴런_{idx}: {val:.3f}")
```

### 4. 학습

```bash
# 기본 학습
python scripts/train_brain_like.py \
    --num_epochs 3 \
    --batch_size 32 \
    --n_neurons 4096 \
    --n_interaction_steps 5

# 활성화 패턴 분석 포함
python scripts/train_brain_like.py \
    --num_epochs 3 \
    --batch_size 32 \
    --analyze_activation
```

### 5. 시각화

```python
from scripts.analyze_brain_activation import visualize_activation_evolution

# 활성화 패턴 진화 시각화
history = analysis['activation_history']
visualize_activation_evolution(history, save_path='activation.png')
```

---

## 📊 분석 도구

### `scripts/analyze_brain_activation.py`

다양한 분석 기능 제공:

1. **활성화 패턴 히트맵**
   - 시간에 따른 뉴런 활성화 변화

2. **활성 뉴런 수 추적**
   - 각 step별 활성 뉴런 수

3. **활성화 강도 분포**
   - 활성화 값의 히스토그램

4. **Top 뉴런 추적**
   - 주요 뉴런들의 궤적

```bash
python scripts/analyze_brain_activation.py
```

---

## 🔬 핵심 특징

### Sparse Activation
- **효율성**: 4096개 중 128-256개만 활성화
- **선택성**: 입력에 따라 다른 뉴런 조합
- **유연성**: 동적으로 뉴런 선택

### Iterative Processing
- **정제**: 5번의 반복으로 패턴 수렴
- **협력**: 뉴런들이 서로 강화/억제
- **안정성**: 최종 패턴은 안정적

### Global Neuron Pool
- **재사용**: 모든 입력이 같은 뉴런 풀 사용
- **특화**: 각 뉴런이 특정 패턴에 특화
- **학습**: 뉴런 특성이 학습됨

---

## 📈 기존 방식과 비교

| 특성 | Transformer | Brain-Like SPROUT |
|------|-------------|-------------------|
| 표현 | [batch, seq, d_model] | [n_neurons] sparse |
| 처리 | 토큰별 병렬 | 전체 패턴 반복 |
| 활성화 | Dense (모든 차원) | Sparse (일부만) |
| 뉴런 | 없음 | 고정된 전역 풀 |
| 상호작용 | Attention만 | Message passing + GRU |

---

## 🎯 장점

1. **생물학적 타당성**
   - 뇌의 작동 방식과 유사
   - Sparse activation은 실제 뉴런처럼

2. **효율성**
   - 4096개 중 일부만 활성화
   - 계산량 절감 가능

3. **해석 가능성**
   - 어떤 뉴런이 활성화되었는지 추적
   - 뉴런의 특화도 분석 가능

4. **유연성**
   - 입력 길이에 관계없이 동일한 패턴 크기
   - 다양한 task에 적용 가능

---

## 📝 파일 구조

```
sprout/
├── src/models/
│   └── sprout_brain_like.py       # 메인 모델
├── scripts/
│   ├── train_brain_like.py        # 학습 스크립트
│   └── analyze_brain_activation.py # 분석 도구
└── BRAIN_LIKE_ARCHITECTURE.md     # 이 문서
```

---

## 🚀 Quick Start

```bash
# 1. 간단한 테스트
python src/models/sprout_brain_like.py

# 2. 학습
python scripts/train_brain_like.py --debug_mode

# 3. 분석
python scripts/analyze_brain_activation.py
```

---

## 🔍 향후 개선 방향

1. **학습 최적화**
   - Sparse 연산 최적화
   - Batch 처리 효율화

2. **아키텍처 개선**
   - 뉴런 수 동적 조정
   - 계층적 뉴런 풀

3. **분석 도구 확장**
   - 뉴런 특화도 시각화
   - 뉴런 간 연결 분석

4. **다양한 Task**
   - Classification
   - Generation
   - Question Answering

---

## 📚 참고

이 아키텍처는 다음 개념들에서 영감을 받았습니다:

- **Sparse Coding**: 뇌의 sparse representation
- **Attractor Networks**: 반복적 패턴 수렴
- **Hebbian Learning**: 뉴런 간 연결 강화
- **Predictive Coding**: 예측 기반 처리

---

## 🤝 기여

질문이나 제안사항은 이슈로 남겨주세요!
