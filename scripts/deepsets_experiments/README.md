# DeepSets FFN Experiments

DeepSets 기반 FFN 실험 - Baseline vs DeepSets 비교

## 개요

각 뉴런이 **학습 가능한 정보 벡터**를 가지고, 선택된 뉴런들의 정보를 DeepSets로 조합하여 출력을 생성합니다.

## 모델 버전

- **Baseline**: `models/neuron_based.py` - 표준 동적 뉴런 선택
- **DeepSets**: `models/neuron_based_deepsets.py` - 학습 가능한 뉴런 정보 벡터 기반

### DeepSets 핵심 아이디어

```python
# 각 뉴런이 학습 가능한 정보 가짐
neuron_vecs[i] ∈ R^d_neuron

# DeepSets 구조
φ: (neuron_vec, activation) → hidden
Σ: Sum aggregation (permutation invariant)
ρ: hidden → output

# F({neurons}) = ρ(Σ φ(neuron_vec[i], act[i]))
```

**바로 시작:**
```bash
# 1. Quick test
python quick_validation.py

# 2. Sanity check
python exp_deepsets_sanity.py

# 3. Full training
python train_deepsets.py --config small
python train_deepsets.py --config medium
python train_deepsets.py --config large
```
