# Manifold Mixing Experiments

뉴런 기반 동적 FFN + DeepSets 매니폴드 형성 실험

## 개요

이 폴더는 Baseline (표준 뉴런 선택)과 Manifold (DeepSets 기반 협력적 뉴런 조합) 버전을 비교하는 실험들을 포함합니다.

## 모델 버전

- **Baseline**: `models/neuron_based.py` - 표준 동적 뉴런 선택
- **Manifold**: `models/neuron_based_manifold.py` - DeepSets 매니폴드 형성

## 실험 목록

### Quick Validation (빠른 검증)
```bash
python quick_validation.py
```

작은 모델로 빠른 검증:
- 기본 forward/backward 작동 확인
- 간단한 학습/평가
- 다양한 sparsity level 테스트

**추천**: 먼저 이것을 실행하여 모든 것이 작동하는지 확인하세요.

---

### Experiment 1: Basic Performance Comparison
```bash
python exp1_performance_comparison.py
```

**목표**: 매니폴드가 성능을 해치지 않는지 확인

**측정 지표**:
- Training loss curve
- Validation perplexity
- 수렴 속도
- 최종 성능

**기대 결과**:
- 비슷하거나 약간 더 좋은 perplexity
- 학습이 안정적

---

### Experiment 2: Neuron Semantics Analysis
```bash
python exp2_neuron_semantics.py
```

**목표**: 뉴런들이 정말 compositional한 정보를 학습하는지

**분석 내용**:
- 비슷한 입력들이 비슷한 뉴런을 선택하는가?
- Manifold 버전이 더 일관성 있는 패턴을 보이는가?

**기대 결과**:
- 비슷한 입력 → 높은 overlap (>50%)
- 다른 입력 → 낮은 overlap (<20%)
- Manifold 버전이 더 일관성 있는 패턴

---

### Experiment 3: Manifold Quality Visualization
```bash
python exp3_manifold_quality.py
```

**목표**: 매니폴드가 의미있는 구조를 형성하는지

**분석 내용**:
- 선택된 뉴런 조합들의 manifold 출력을 시각화
- t-SNE로 manifold 공간 구조 확인
- 같은 카테고리 입력들이 가까이 모이는지 확인

**기대 결과**:
- 같은 카테고리 입력들이 manifold 공간에서 가까움
- 명확한 클러스터 형성

---

### Experiment 4: Combination Generalization
```bash
python exp4_combination_generalization.py
```

**목표**: 안 본 뉴런 조합도 잘 처리하는가?

**분석 내용**:
- 학습 중 뉴런 조합 빈도 추적
- 테스트에서 드문 조합 찾기
- 드문 조합에서의 성능 vs 흔한 조합에서의 성능

**기대 결과**:
- Manifold 버전: rare vs common 차이 작음
- Baseline: rare에서 성능 하락

---

### Experiment 5: Sparsity vs Performance
```bash
python exp5_sparsity_vs_performance.py
```

**목표**: 다양한 top_k에서 매니폴드 효과

**분석 내용**:
- 다양한 sparsity level에서 성능 측정
- Manifold가 매우 sparse할 때 더 유리한지 확인

**기대 결과**:
- 매우 sparse할 때 (top_k=32) manifold가 더 유리
- Dense할 때는 비슷

---

### Experiment 6: Learning Stability
```bash
python exp6_learning_stability.py
```

**목표**: 매니폴드가 학습을 불안정하게 만들지 않는가?

**측정 지표**:
- Gradient norm
- Loss variance
- Neuron usage entropy (다양성)

**기대 결과**:
- Gradient norm 안정적
- 뉴런 사용이 더 균등 (높은 entropy)

---

## 전체 실험 실행

모든 실험을 순서대로 실행:

```bash
# 1. Quick validation
python quick_validation.py

# 2. Performance comparison (모델 학습)
python exp1_performance_comparison.py

# 3-6. 분석 실험들 (exp1의 학습된 모델 사용)
python exp2_neuron_semantics.py
python exp3_manifold_quality.py
python exp4_combination_generalization.py
python exp5_sparsity_vs_performance.py

# 7. Stability monitoring (새로 학습)
python exp6_learning_stability.py
```

**참고**:
- 실험 2-5는 실험 1에서 학습된 모델을 로드합니다
- 실험 6은 학습 과정을 모니터링하므로 별도로 실행됩니다

## 결과 위치

결과는 `results/` 폴더에 저장됩니다:

```
results/
├── exp1_performance/
│   ├── Baseline_best.pt
│   ├── Manifold_best.pt
│   ├── Baseline_results.json
│   ├── Manifold_results.json
│   └── performance_comparison.png
├── exp2_semantics/
│   ├── semantics_results.json
│   └── neuron_overlap_comparison.png
├── exp3_manifold/
│   ├── manifold_quality_results.json
│   └── manifold_tsne.png
├── exp4_generalization/
│   └── generalization_results.json
├── exp5_sparsity/
│   ├── sparsity_results.json
│   └── sparsity_comparison.png
└── exp6_stability/
    ├── stability_results.json
    └── stability_comparison.png
```

## 핵심 질문

이 실험들은 다음 질문들에 답합니다:

1. **성능**: Manifold가 성능을 해치는가? (Exp 1)
2. **의미**: 뉴런들이 의미있는 정보를 학습하는가? (Exp 2)
3. **구조**: Manifold가 의미있는 구조를 형성하는가? (Exp 3)
4. **일반화**: 안 본 조합도 잘 처리하는가? (Exp 4)
5. **효율성**: Sparse할 때 더 좋은가? (Exp 5)
6. **안정성**: 학습이 안정적인가? (Exp 6)

## 요구사항

```bash
pip install torch numpy matplotlib seaborn scikit-learn tqdm
```

## 커스터마이징

각 실험 스크립트의 `config` 딕셔너리를 수정하여 실험 설정을 변경할 수 있습니다:

```python
config = {
    'vocab_size': 10000,
    'd_model': 256,
    'd_ff': 1024,
    'n_heads': 8,
    'n_layers': 4,
    # ... 등등
}
```
