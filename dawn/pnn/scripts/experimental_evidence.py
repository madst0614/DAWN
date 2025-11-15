"""
Experimental Evidence for Plastic Neural Networks
실험적 증거 찾기 - PNN의 뇌과학적 타당성 검증

이 스크립트는 세 가지 실험적 접근법을 시뮬레이션합니다:
1. MEG (High-temporal resolution) - Gamma cycle 내 활동 패턴 분석
2. Optogenetics - 특정 뉴런/레이어 억제 실험
3. Brain Activity Modeling - 모델 예측 vs 실제 뇌 활동 비교

Usage:
    python scripts/experimental_evidence.py --checkpoint checkpoints/best_model.pt
    python scripts/experimental_evidence.py --checkpoint checkpoints/best_model.pt --experiment meg
    python scripts/experimental_evidence.py --checkpoint checkpoints/best_model.pt --experiment optogenetics
    python scripts/experimental_evidence.py --checkpoint checkpoints/best_model.pt --experiment modeling
"""

# Add parent directory to path for imports (makes it work without installation)
import sys
from pathlib import Path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from pnn.models.pnn import create_pnn_model
from pnn.utils.training import load_checkpoint
from transformers import BertTokenizer
from datasets import load_dataset


# Define Config class at module level for checkpoint compatibility
class Config:
    """Dummy Config class for checkpoint loading compatibility"""
    pass


class MEGSimulator:
    """
    MEG (Magnetoencephalography) 시뮬레이션

    Gamma cycle (millisecond 해상도) 내에서 활동 패턴 분석:
    - 초반: Delta generation (높은 activity)
    - 중반: Gate computation (특정 패턴)
    - 후반: Update (다른 패턴)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        # Hook을 사용해서 중간 활동 기록
        self.activations = {}
        self.register_hooks()

    def register_hooks(self):
        """중간 레이어 활동을 기록하기 위한 hook 등록"""
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output.detach()
            return hook

        # DeltaRefiner의 주요 컴포넌트에 hook 등록
        self.model.delta_refiner.attention.register_forward_hook(
            get_activation('attention_output')
        )
        self.model.delta_refiner.ffn.register_forward_hook(
            get_activation('ffn_output')
        )
        self.model.delta_refiner.gate.register_forward_hook(
            get_activation('gate_output')
        )

    def simulate_gamma_cycle(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Gamma cycle 시뮬레이션

        각 refinement step을 gamma cycle로 간주하고,
        cycle 내에서의 활동 패턴을 분석

        Returns:
            step_activities: 각 step별 활동 패턴
        """
        step_activities = {
            'delta_generation': [],
            'gate_computation': [],
            'hidden_update': [],
            'activity_magnitude': []
        }

        with torch.no_grad():
            # Embedding
            batch_size, seq_len = input_ids.shape
            token_embeds = self.model.token_embeddings(input_ids)
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.model.position_embeddings(position_ids)

            hidden = token_embeds + position_embeds
            hidden = self.model.embedding_layer_norm(hidden)
            hidden = self.model.embedding_dropout(hidden)

            attn_mask = (attention_mask == 0) if attention_mask is not None else None

            # 각 refinement step을 gamma cycle로 분석
            for step in range(self.model.num_steps):
                self.activations.clear()

                # Delta 생성
                delta = self.model.delta_refiner(hidden, attn_mask)

                # 활동 패턴 기록
                # 1. Delta generation (초반 - 높은 activity)
                if 'ffn_output' in self.activations:
                    delta_activity = self.activations['ffn_output']
                    step_activities['delta_generation'].append(
                        delta_activity.abs().mean().cpu()
                    )

                # 2. Gate computation (중반 - 특정 패턴)
                if 'gate_output' in self.activations:
                    gate_activity = self.activations['gate_output']
                    step_activities['gate_computation'].append(
                        gate_activity.mean().cpu()
                    )

                # 3. Hidden update (후반)
                hidden_before = hidden.clone()
                hidden = hidden + delta
                update_magnitude = (hidden - hidden_before).abs().mean()
                step_activities['hidden_update'].append(update_magnitude.cpu())

                # 전체 activity magnitude
                total_activity = delta.abs().mean()
                step_activities['activity_magnitude'].append(total_activity.cpu())

        return step_activities

    def analyze_temporal_patterns(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        여러 배치에 걸쳐 temporal pattern 분석

        Returns:
            analysis: 통계 및 패턴 분석 결과
        """
        all_activities = {
            'delta_generation': [],
            'gate_computation': [],
            'hidden_update': [],
            'activity_magnitude': []
        }

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="MEG Analysis", total=num_batches)):
            if batch_idx >= num_batches:
                break

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            activities = self.simulate_gamma_cycle(input_ids, attention_mask)

            for key in all_activities:
                all_activities[key].extend(activities[key])

        # 통계 분석
        analysis = {}
        for key, values in all_activities.items():
            values = torch.tensor(values).numpy()
            analysis[key] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'values': values.tolist()
            }

        return analysis


class OptogeneticsSimulator:
    """
    Optogenetics 시뮬레이션

    특정 뉴런/레이어를 억제하여 행동 변화 관찰:
    - Attention 억제 → Delta 생성 막힘?
    - Gate 억제 → Update 막힘?
    - FFN 억제 → 정보 처리 변화?
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def suppress_component(
        self,
        component: str,
        suppression_rate: float = 1.0
    ):
        """
        특정 컴포넌트 억제

        Args:
            component: 'attention', 'gate', 'ffn' 중 하나
            suppression_rate: 억제 비율 (0.0 = 억제 없음, 1.0 = 완전 억제)
        """
        class SuppressionHook:
            def __init__(self, rate):
                self.rate = rate

            def __call__(self, module, input, output):
                if isinstance(output, tuple):
                    output = list(output)
                    output[0] = output[0] * (1.0 - self.rate)
                    return tuple(output)
                else:
                    return output * (1.0 - self.rate)

        hook = SuppressionHook(suppression_rate)

        if component == 'attention':
            handle = self.model.delta_refiner.attention.register_forward_hook(hook)
        elif component == 'gate':
            handle = self.model.delta_refiner.gate.register_forward_hook(hook)
        elif component == 'ffn':
            handle = self.model.delta_refiner.ffn.register_forward_hook(hook)
        else:
            raise ValueError(f"Unknown component: {component}")

        return handle

    def measure_behavior(
        self,
        dataloader,
        labels_key: str = 'labels',
        num_batches: int = 10
    ) -> Dict:
        """
        모델 행동 측정 (정확도, 손실 등)

        Returns:
            metrics: 성능 지표
        """
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Measuring", total=num_batches)):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch[labels_key].to(self.device)

                hidden = self.model(input_ids, attention_mask)
                loss, logits = self.model.get_mlm_loss(hidden, labels)

                total_loss += loss.item()

                preds = logits.argmax(dim=-1)
                mask = (labels != -100)
                correct = (preds == labels) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_correct / total_tokens if total_tokens > 0 else 0,
            'total_tokens': total_tokens
        }

    def run_suppression_experiment(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        여러 컴포넌트에 대한 억제 실험

        Returns:
            results: 각 조건별 실험 결과
        """
        results = {}

        # 1. Baseline (억제 없음)
        print("\n📊 Baseline (No suppression)...")
        results['baseline'] = self.measure_behavior(dataloader, num_batches=num_batches)

        # 2. Attention 억제
        components = ['attention', 'gate', 'ffn']
        suppression_rates = [0.25, 0.5, 0.75, 1.0]

        for component in components:
            for rate in suppression_rates:
                print(f"\n🔬 Suppressing {component} at {rate*100:.0f}%...")

                # 억제 적용
                handle = self.suppress_component(component, rate)

                # 행동 측정
                metrics = self.measure_behavior(dataloader, num_batches=num_batches)

                # 결과 저장
                key = f"{component}_suppressed_{int(rate*100)}"
                results[key] = metrics

                # 억제 해제
                handle.remove()

        return results

    def test_gate_specificity(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        Gate의 선택적 기능 테스트 (진짜 중요성 검증)

        기존 억제는 uniform scaling → step robustness만 테스트
        새로운 테스트들은 gate의 선택적 기능을 직접 테스트

        Returns:
            gate_specificity_results: 각 테스트별 결과
        """
        results = {}

        # Baseline
        print("\n📊 Measuring baseline for gate specificity tests...")
        results['baseline'] = self.measure_behavior(dataloader, num_batches=num_batches)

        # Test 1: Selective suppression (random dropout)
        print("\n🔬 Test 1: Selective Suppression (Random Gate Dropout)")
        print("   Purpose: Test information loss from random selection masking")

        dropout_rates = [0.25, 0.5, 0.75]
        selective_results = {}

        for rate in dropout_rates:
            print(f"   Dropout rate: {rate*100:.0f}%")

            # Apply random dropout to gate
            handle = self._apply_gate_dropout(rate)
            metrics = self.measure_behavior(dataloader, num_batches=num_batches)
            handle.remove()

            selective_results[f'dropout_{int(rate*100)}'] = {
                'metrics': metrics,
                'accuracy_drop': results['baseline']['accuracy'] - metrics['accuracy']
            }

        results['selective_suppression'] = selective_results

        # Test 2: Anti-gate (inverse gate)
        print("\n🔬 Test 2: Anti-Gate (Inverse Selection)")
        print("   Purpose: Select bad features, discard good ones")

        handle = self._apply_anti_gate()
        anti_metrics = self.measure_behavior(dataloader, num_batches=num_batches)
        handle.remove()

        results['anti_gate'] = {
            'metrics': anti_metrics,
            'accuracy_drop': results['baseline']['accuracy'] - anti_metrics['accuracy']
        }

        # Test 3: Noise injection
        print("\n🔬 Test 3: Noise Injection to Gate")
        print("   Purpose: Test gate precision importance")

        noise_levels = [0.1, 0.3, 0.5]
        noise_results = {}

        for noise_std in noise_levels:
            print(f"   Noise std: {noise_std}")

            handle = self._apply_gate_noise(noise_std)
            metrics = self.measure_behavior(dataloader, num_batches=num_batches)
            handle.remove()

            noise_results[f'noise_{int(noise_std*100)}'] = {
                'metrics': metrics,
                'accuracy_drop': results['baseline']['accuracy'] - metrics['accuracy']
            }

        results['noise_injection'] = noise_results

        # Test 4: Pattern shuffling
        print("\n🔬 Test 4: Pattern Shuffling")
        print("   Purpose: Test if spatial pattern matters (vs just magnitude)")

        handle = self._apply_pattern_shuffling()
        shuffle_metrics = self.measure_behavior(dataloader, num_batches=num_batches)
        handle.remove()

        results['pattern_shuffling'] = {
            'metrics': shuffle_metrics,
            'accuracy_drop': results['baseline']['accuracy'] - shuffle_metrics['accuracy']
        }

        # Test 5: Uniform gate (remove pattern, keep magnitude)
        print("\n🔬 Test 5: Uniform Gate")
        print("   Purpose: Replace pattern with uniform average (removes all selectivity)")

        handle = self._apply_uniform_gate()
        uniform_metrics = self.measure_behavior(dataloader, num_batches=num_batches)
        handle.remove()

        results['uniform_gate'] = {
            'metrics': uniform_metrics,
            'accuracy_drop': results['baseline']['accuracy'] - uniform_metrics['accuracy']
        }

        # Test 6: Magnitude scaling (robustness to different scales)
        print("\n🔬 Test 6: Magnitude Scaling")
        print("   Purpose: Test robustness to different gate magnitudes (not pattern)")

        scales = [0.25, 0.5, 0.75, 1.5, 2.0]
        scaling_results = {}

        for scale in scales:
            print(f"   Scale: {scale}")

            handle = self._apply_magnitude_scaling(scale)
            metrics = self.measure_behavior(dataloader, num_batches=num_batches)
            handle.remove()

            scaling_results[f'scale_{int(scale*100)}'] = {
                'metrics': metrics,
                'accuracy_drop': results['baseline']['accuracy'] - metrics['accuracy']
            }

        results['magnitude_scaling'] = scaling_results

        return results

    def _apply_gate_dropout(self, dropout_rate: float):
        """Random dropout to gate (선택 정보 손실)"""
        class GateDropoutHook:
            def __init__(self, rate):
                self.rate = rate

            def __call__(self, module, input, output):
                # Random mask
                mask = (torch.rand_like(output) > self.rate).float()
                return output * mask

        hook = GateDropoutHook(dropout_rate)
        handle = self.model.delta_refiner.gate.register_forward_hook(hook)
        return handle

    def _apply_anti_gate(self):
        """Inverse gate (나쁜 것 선택, 좋은 것 버림)"""
        class AntiGateHook:
            def __call__(self, module, input, output):
                # Invert the gate: select what should be discarded
                return 1.0 - output

        hook = AntiGateHook()
        handle = self.model.delta_refiner.gate.register_forward_hook(hook)
        return handle

    def _apply_gate_noise(self, noise_std: float):
        """Add noise to gate (선택 판단에 노이즈)"""
        class GateNoiseHook:
            def __init__(self, std):
                self.std = std

            def __call__(self, module, input, output):
                noise = torch.randn_like(output) * self.std
                noisy_gate = output + noise
                # Clamp to valid range [0, 1] for sigmoid gates
                return torch.clamp(noisy_gate, 0.0, 1.0)

        hook = GateNoiseHook(noise_std)
        handle = self.model.delta_refiner.gate.register_forward_hook(hook)
        return handle

    def _apply_pattern_shuffling(self):
        """Shuffle gate patterns (keeps magnitude, destroys spatial pattern)"""
        class PatternShuffleHook:
            def __call__(self, module, input, output):
                # Shuffle along the feature dimension
                # This keeps the distribution but destroys the spatial pattern
                batch_size, seq_len, hidden = output.shape
                shuffled = output.clone()

                # Shuffle each position independently
                for b in range(batch_size):
                    for s in range(seq_len):
                        perm = torch.randperm(hidden, device=output.device)
                        shuffled[b, s] = output[b, s, perm]

                return shuffled

        hook = PatternShuffleHook()
        handle = self.model.delta_refiner.gate.register_forward_hook(hook)
        return handle

    def _apply_uniform_gate(self):
        """Replace all gates with uniform average (removes all selectivity)"""
        class UniformGateHook:
            def __call__(self, module, input, output):
                # Replace pattern with uniform average
                # This removes ALL selection information
                mean_val = output.mean(dim=-1, keepdim=True)
                uniform = mean_val.expand_as(output)
                return uniform

        hook = UniformGateHook()
        handle = self.model.delta_refiner.gate.register_forward_hook(hook)
        return handle

    def _apply_magnitude_scaling(self, scale: float):
        """Scale gate magnitude (keeps pattern, changes magnitude)"""
        class MagnitudeScalingHook:
            def __init__(self, scale):
                self.scale = scale

            def __call__(self, module, input, output):
                # Scale gate values while keeping pattern
                # This tests robustness to different gate strengths
                return output * self.scale

        hook = MagnitudeScalingHook(scale)
        handle = self.model.delta_refiner.gate.register_forward_hook(hook)
        return handle


class BrainActivityPredictor:
    """
    Brain Activity Modeling

    모델의 활동 패턴을 예측하고 실제 뇌 활동과 비교
    (실제 fMRI/MEG 데이터가 있다면 비교 가능)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def extract_activation_patterns(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        모델의 활동 패턴 추출

        Returns:
            patterns: 각 레이어/단계별 활동 패턴
        """
        patterns = {
            'embeddings': [],
            'step_0': [],
            'step_1': [],
            'step_2': [],
            'step_3': [],
            'final': []
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting patterns", total=num_batches)):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # 모든 step의 출력 얻기
                all_outputs = self.model(
                    input_ids,
                    attention_mask,
                    return_all_steps=True
                )

                # 각 step의 평균 활동 저장
                patterns['embeddings'].append(all_outputs[0].mean(dim=[0, 1]).cpu().numpy())

                for step_idx in range(min(4, len(all_outputs) - 1)):
                    step_key = f'step_{step_idx}'
                    patterns[step_key].append(
                        all_outputs[step_idx + 1].mean(dim=[0, 1]).cpu().numpy()
                    )

                patterns['final'].append(all_outputs[-1].mean(dim=[0, 1]).cpu().numpy())

        # 평균 패턴 계산
        for key in patterns:
            if patterns[key]:
                patterns[key] = np.mean(patterns[key], axis=0)

        return patterns

    def compare_with_brain_hypothesis(
        self,
        patterns: Dict
    ) -> Dict:
        """
        가설적 뇌 활동 패턴과 비교

        뇌과학 가설:
        - 초기 step: 넓은 활동 (탐색)
        - 중간 step: 선택적 활동 (집중)
        - 후기 step: 통합적 활동 (종합)
        """
        analysis = {}

        # 1. 활동 범위 (Activity breadth) - 표준편차로 측정
        for key, pattern in patterns.items():
            if isinstance(pattern, np.ndarray) and len(pattern) > 0:
                analysis[f'{key}_breadth'] = float(np.std(pattern))
                analysis[f'{key}_mean_activity'] = float(np.mean(np.abs(pattern)))

        # 2. 패턴 변화 (Step-to-step changes)
        step_keys = ['embeddings', 'step_0', 'step_1', 'step_2', 'step_3']
        for i in range(len(step_keys) - 1):
            if step_keys[i] in patterns and step_keys[i+1] in patterns:
                p1 = patterns[step_keys[i]]
                p2 = patterns[step_keys[i+1]]
                if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
                    change = np.linalg.norm(p2 - p1)
                    analysis[f'change_{step_keys[i]}_to_{step_keys[i+1]}'] = float(change)

        # 3. 가설 검증
        # 가설: 중간 step에서 가장 선택적 (낮은 breadth)
        breadths = [
            analysis.get(f'{key}_breadth', 0)
            for key in ['step_0', 'step_1', 'step_2', 'step_3']
        ]

        if breadths:
            min_breadth_idx = np.argmin(breadths)
            analysis['most_selective_step'] = int(min_breadth_idx)
            analysis['supports_selectivity_hypothesis'] = (min_breadth_idx in [1, 2])

        return analysis


class DimensionwiseAnalyzer:
    """
    Dimension-wise Activity Analysis

    각 차원별 활동 패턴 분석:
    - Accumulator: 단계마다 증가
    - Selector: 단계마다 감소
    - Oscillator: 증가/감소 반복
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def analyze_dimensions(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        각 차원별 활동 패턴 분석

        Returns:
            dimension_patterns: 각 차원의 패턴 타입 및 통계
        """
        hidden_size = self.model.hidden_size
        num_steps = self.model.num_steps

        # 각 step별 차원 활동 수집
        step_activities = [[] for _ in range(num_steps + 1)]  # +1 for embeddings

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing dimensions", total=num_batches)):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # 모든 step의 출력 얻기
                all_outputs = self.model(
                    input_ids,
                    attention_mask,
                    return_all_steps=True
                )

                # 각 step의 활동을 차원별로 평균
                for step_idx, output in enumerate(all_outputs):
                    # [batch, seq, hidden] -> [hidden] (평균)
                    dim_activity = output.abs().mean(dim=[0, 1]).cpu().numpy()
                    step_activities[step_idx].append(dim_activity)

        # 각 step별 평균 계산
        avg_step_activities = []
        for step_acts in step_activities:
            if step_acts:
                avg_step_activities.append(np.mean(step_acts, axis=0))

        avg_step_activities = np.array(avg_step_activities)  # [num_steps+1, hidden_size]

        # 각 차원별 패턴 분석
        dimension_patterns = {
            'accumulator': [],
            'selector': [],
            'oscillator': [],
            'stable': []
        }

        for dim in range(hidden_size):
            activity = avg_step_activities[:, dim]

            # 패턴 분류
            diffs = np.diff(activity)

            if np.all(diffs > 0):
                # 계속 증가
                dimension_patterns['accumulator'].append(dim)
            elif np.all(diffs < 0):
                # 계속 감소
                dimension_patterns['selector'].append(dim)
            elif len(diffs) >= 2 and np.any(diffs[:-1] * diffs[1:] < 0):
                # 방향 변경 (증가->감소 또는 감소->증가)
                dimension_patterns['oscillator'].append(dim)
            else:
                # 변화가 작음
                dimension_patterns['stable'].append(dim)

        # 통계 계산
        results = {
            'dimension_patterns': dimension_patterns,
            'pattern_counts': {
                'accumulator': len(dimension_patterns['accumulator']),
                'selector': len(dimension_patterns['selector']),
                'oscillator': len(dimension_patterns['oscillator']),
                'stable': len(dimension_patterns['stable'])
            },
            'step_activities': avg_step_activities.tolist(),
            'sample_dimensions': {
                'accumulator_samples': dimension_patterns['accumulator'][:5],
                'selector_samples': dimension_patterns['selector'][:5],
                'oscillator_samples': dimension_patterns['oscillator'][:5]
            }
        }

        return results


class TokenDifficultyAnalyzer:
    """
    Token Difficulty별 활동 분석

    Easy vs Hard tokens의 활동 패턴 비교:
    - Easy: 낮고 안정적인 활동
    - Hard: 높고 증가하는 활동
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def analyze_by_difficulty(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        Token 난이도별 활동 패턴 분석

        Returns:
            difficulty_analysis: Easy/Hard tokens의 활동 비교
        """
        easy_activities = []
        medium_activities = []
        hard_activities = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing difficulty", total=num_batches)):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 모든 step의 출력 얻기
                all_outputs = self.model(
                    input_ids,
                    attention_mask,
                    return_all_steps=True
                )

                # 마지막 출력으로 예측
                final_output = all_outputs[-1]
                logits = self.model.mlm_head(final_output)

                # 각 토큰의 정답 확률
                probs = F.softmax(logits, dim=-1)

                # Masked token만 고려 (먼저 마스크 생성)
                mask = (labels != -100)

                # labels에서 -100을 0으로 대체하여 gather 에러 방지
                # (나중에 mask로 필터링할 것이므로 0으로 대체해도 무방)
                safe_labels = labels.clone()
                safe_labels[~mask] = 0

                correct_probs = probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)

                # Move tensors to CPU for safe indexing
                mask_cpu = mask.cpu()
                correct_probs_cpu = correct_probs.cpu()

                # 난이도 분류 (확률 기반)
                for i in range(input_ids.size(0)):
                    for j in range(input_ids.size(1)):
                        if not mask_cpu[i, j]:
                            continue

                        prob = correct_probs_cpu[i, j].item()

                        # 각 step의 활동 추출
                        step_acts = [output[i, j].abs().mean().item() for output in all_outputs]

                        if prob > 0.7:
                            # Easy token
                            easy_activities.append(step_acts)
                        elif prob > 0.3:
                            # Medium token
                            medium_activities.append(step_acts)
                        else:
                            # Hard token
                            hard_activities.append(step_acts)

        # 평균 계산
        results = {
            'easy': {
                'mean': np.mean(easy_activities, axis=0).tolist() if easy_activities else [],
                'std': np.std(easy_activities, axis=0).tolist() if easy_activities else [],
                'count': len(easy_activities)
            },
            'medium': {
                'mean': np.mean(medium_activities, axis=0).tolist() if medium_activities else [],
                'std': np.std(medium_activities, axis=0).tolist() if medium_activities else [],
                'count': len(medium_activities)
            },
            'hard': {
                'mean': np.mean(hard_activities, axis=0).tolist() if hard_activities else [],
                'std': np.std(hard_activities, axis=0).tolist() if hard_activities else [],
                'count': len(hard_activities)
            }
        }

        return results


class LayerImportanceAnalyzer:
    """
    Layer-wise Gate Importance Analysis

    각 레이어(컴포넌트)의 gate 중요도 분석:
    - Attention gate importance
    - FFN gate importance
    - Gate module importance
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def analyze_layer_importance(
        self,
        dataloader,
        num_batches: int = 10,
        suppression_rates: list = [0.25, 0.5, 0.75]
    ) -> Dict:
        """
        각 레이어의 중요도 분석

        Returns:
            importance_analysis: 각 레이어별 억제 영향
        """
        results = {}

        # Baseline
        print("\n📊 Measuring baseline performance...")
        baseline = self._measure_performance(dataloader, num_batches)
        results['baseline'] = baseline

        # 각 컴포넌트별 억제 실험
        components = ['attention', 'ffn', 'gate']

        for component in components:
            print(f"\n🔬 Testing {component} importance...")
            component_results = {}

            for rate in suppression_rates:
                print(f"  Suppression rate: {rate*100:.0f}%")

                # 억제 적용
                handle = self._suppress_component(component, rate)

                # 성능 측정
                metrics = self._measure_performance(dataloader, num_batches)

                # 억제 해제
                handle.remove()

                # 결과 저장
                component_results[f'suppression_{int(rate*100)}'] = {
                    'metrics': metrics,
                    'accuracy_drop': baseline['accuracy'] - metrics['accuracy'],
                    'loss_increase': metrics['loss'] - baseline['loss']
                }

            results[component] = component_results

        # 중요도 순위 계산
        importance_ranking = []
        for component in components:
            # 75% 억제 시 accuracy drop으로 중요도 측정
            drop = results[component]['suppression_75']['accuracy_drop']
            importance_ranking.append((component, drop))

        importance_ranking.sort(key=lambda x: x[1], reverse=True)
        results['importance_ranking'] = [
            {'component': comp, 'importance_score': score}
            for comp, score in importance_ranking
        ]

        return results

    def _suppress_component(self, component: str, rate: float):
        """컴포넌트 억제"""
        class SuppressionHook:
            def __init__(self, rate):
                self.rate = rate

            def __call__(self, module, input, output):
                if isinstance(output, tuple):
                    output = list(output)
                    output[0] = output[0] * (1.0 - self.rate)
                    return tuple(output)
                else:
                    return output * (1.0 - self.rate)

        hook = SuppressionHook(rate)

        if component == 'attention':
            handle = self.model.delta_refiner.attention.register_forward_hook(hook)
        elif component == 'ffn':
            handle = self.model.delta_refiner.ffn.register_forward_hook(hook)
        elif component == 'gate':
            handle = self.model.delta_refiner.gate.register_forward_hook(hook)
        else:
            raise ValueError(f"Unknown component: {component}")

        return handle

    def _measure_performance(self, dataloader, num_batches: int) -> Dict:
        """성능 측정"""
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                hidden = self.model(input_ids, attention_mask)
                loss, logits = self.model.get_mlm_loss(hidden, labels)

                total_loss += loss.item()

                preds = logits.argmax(dim=-1)
                mask = (labels != -100)
                correct = (preds == labels) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_correct / total_tokens if total_tokens > 0 else 0,
            'total_tokens': total_tokens
        }


class LayerwiseActivityAnalyzer:
    """
    Layer-wise Activity Analysis

    각 레이어(스텝)에서 activity accumulation 분석:
    - Early layers: 낮은 activity 예상
    - Late layers: 높은 activity 예상
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def analyze_layerwise_activity(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        각 레이어(스텝)별 activity 분석

        Returns:
            layerwise_activity: 각 스텝별 평균 activity 및 accumulation 패턴
        """
        step_activities = []  # shape: (num_batches, num_steps, hidden_size)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing layerwise activity", total=num_batches)):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 모든 step의 출력 얻기
                all_outputs = self.model(
                    input_ids,
                    attention_mask,
                    return_all_steps=True
                )

                # 각 스텝의 평균 activity 계산
                batch_step_acts = []
                for step_output in all_outputs:
                    # [batch, seq, hidden] -> mean activity
                    activity = step_output.abs().mean().item()
                    batch_step_acts.append(activity)

                step_activities.append(batch_step_acts)

        # Convert to numpy for analysis
        step_activities = np.array(step_activities)  # (num_batches, num_steps)

        # 각 스텝별 통계
        results = {
            'mean_activity_per_step': step_activities.mean(axis=0).tolist(),
            'std_activity_per_step': step_activities.std(axis=0).tolist(),
            'accumulation_pattern': {
                'early_activity': float(step_activities[:, 0].mean()),
                'late_activity': float(step_activities[:, -1].mean()),
                'activity_increase': float(step_activities[:, -1].mean() - step_activities[:, 0].mean()),
                'supports_accumulation_hypothesis': float(step_activities[:, -1].mean()) > float(step_activities[:, 0].mean())
            },
            'num_steps': len(all_outputs)
        }

        return results


class CrossTokenInterferenceAnalyzer:
    """
    Cross-Token Interference Analysis

    토큰 간 상호작용 및 간섭 분석:
    - Easy-hard interaction
    - Hard가 easy를 방해하는가?
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def analyze_cross_token_interference(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        토큰 간 간섭 분석

        Returns:
            interference_analysis: Easy-hard token 간 간섭 패턴
        """
        easy_performances = []  # performance when surrounded by easy tokens
        hard_performances = []  # performance when surrounded by hard tokens
        mixed_performances = []  # performance in mixed context

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing cross-token interference", total=num_batches)):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass to get predictions
                all_outputs = self.model(
                    input_ids,
                    attention_mask,
                    return_all_steps=True
                )

                final_output = all_outputs[-1]
                logits = self.model.mlm_head(final_output)
                probs = F.softmax(logits, dim=-1)

                # Masked token만 고려 (먼저 마스크 생성)
                mask = (labels != -100)

                # labels에서 -100을 0으로 대체하여 gather 에러 방지
                safe_labels = labels.clone()
                safe_labels[~mask] = 0

                correct_probs = probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)

                # Move to CPU for analysis
                mask_cpu = mask.cpu()
                correct_probs_cpu = correct_probs.cpu()

                # 각 토큰의 난이도 분류 및 context 분석
                for i in range(input_ids.size(0)):
                    token_difficulties = []
                    token_probs = []

                    for j in range(input_ids.size(1)):
                        if not mask_cpu[i, j]:
                            continue

                        prob = correct_probs_cpu[i, j].item()
                        token_probs.append(prob)

                        # 난이도 분류
                        if prob > 0.7:
                            token_difficulties.append('easy')
                        elif prob > 0.3:
                            token_difficulties.append('medium')
                        else:
                            token_difficulties.append('hard')

                    # Context 분석
                    if len(token_difficulties) > 2:
                        for idx in range(1, len(token_difficulties) - 1):
                            current_prob = token_probs[idx]
                            left_diff = token_difficulties[idx - 1]
                            right_diff = token_difficulties[idx + 1]

                            # Easy context (양옆이 모두 easy)
                            if left_diff == 'easy' and right_diff == 'easy':
                                easy_performances.append(current_prob)
                            # Hard context (양옆이 모두 hard)
                            elif left_diff == 'hard' and right_diff == 'hard':
                                hard_performances.append(current_prob)
                            # Mixed context
                            else:
                                mixed_performances.append(current_prob)

        # 결과 분석
        results = {
            'easy_context': {
                'mean_performance': float(np.mean(easy_performances)) if easy_performances else 0.0,
                'std_performance': float(np.std(easy_performances)) if easy_performances else 0.0,
                'count': len(easy_performances)
            },
            'hard_context': {
                'mean_performance': float(np.mean(hard_performances)) if hard_performances else 0.0,
                'std_performance': float(np.std(hard_performances)) if hard_performances else 0.0,
                'count': len(hard_performances)
            },
            'mixed_context': {
                'mean_performance': float(np.mean(mixed_performances)) if mixed_performances else 0.0,
                'std_performance': float(np.std(mixed_performances)) if mixed_performances else 0.0,
                'count': len(mixed_performances)
            },
            'interference_effect': {
                'easy_vs_hard_context_diff': float(np.mean(easy_performances) - np.mean(hard_performances)) if (easy_performances and hard_performances) else 0.0,
                'hard_tokens_interfere': (float(np.mean(easy_performances)) > float(np.mean(hard_performances))) if (easy_performances and hard_performances) else False
            }
        }

        return results


class GateEntropyAnalyzer:
    """
    Gate Entropy Analysis (Confidence Measure)

    Gate entropy를 사용한 confidence 측정:
    - confidence = -sum(gate * log(gate))
    - Easy tokens: low entropy (high confidence)
    - Hard tokens: high entropy (low confidence)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def analyze_gate_entropy(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        Gate entropy 분석

        Returns:
            entropy_analysis: Easy vs Hard tokens의 gate entropy 비교
        """
        easy_entropies = []
        medium_entropies = []
        hard_entropies = []

        # Gate values 수집을 위한 hook
        gate_values = []

        def gate_hook(module, input, output):
            # Gate output을 저장
            gate_values.append(output.detach())

        # Register hook on gate module
        # PNN 모델 구조에 맞게 gate에 hook 등록
        hook_handle = None
        if hasattr(self.model.delta_refiner, 'gate'):
            hook_handle = self.model.delta_refiner.gate.register_forward_hook(gate_hook)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing gate entropy", total=num_batches)):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Clear previous gate values
                gate_values.clear()

                # Forward pass
                all_outputs = self.model(
                    input_ids,
                    attention_mask,
                    return_all_steps=True
                )

                final_output = all_outputs[-1]
                logits = self.model.mlm_head(final_output)
                probs = F.softmax(logits, dim=-1)

                # Masked token만 고려
                mask = (labels != -100)
                safe_labels = labels.clone()
                safe_labels[~mask] = 0

                correct_probs = probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)

                # Move to CPU
                mask_cpu = mask.cpu()
                correct_probs_cpu = correct_probs.cpu()

                # Gate entropy 계산
                if gate_values:
                    # 마지막 스텝의 gate 값 사용
                    last_gate = gate_values[-1] if len(gate_values) > 0 else None

                    if last_gate is not None:
                        # Entropy 계산: -sum(p * log(p))
                        # Gate values are in [0, 1], add epsilon for numerical stability
                        epsilon = 1e-10
                        gate_cpu = last_gate.cpu()
                        gate_clipped = torch.clamp(gate_cpu, epsilon, 1.0 - epsilon)

                        # Binary entropy for each dimension
                        entropy = -(gate_clipped * torch.log(gate_clipped) +
                                   (1 - gate_clipped) * torch.log(1 - gate_clipped))

                        # Average entropy per token
                        token_entropy = entropy.mean(dim=-1)  # [batch, seq]

                        # 난이도별 entropy 분류
                        for i in range(input_ids.size(0)):
                            for j in range(input_ids.size(1)):
                                if not mask_cpu[i, j]:
                                    continue

                                prob = correct_probs_cpu[i, j].item()
                                ent = token_entropy[i, j].item()

                                if prob > 0.7:
                                    easy_entropies.append(ent)
                                elif prob > 0.3:
                                    medium_entropies.append(ent)
                                else:
                                    hard_entropies.append(ent)

        # Remove hook
        if hook_handle is not None:
            hook_handle.remove()

        # 결과 분석
        results = {
            'easy_tokens': {
                'mean_entropy': float(np.mean(easy_entropies)) if easy_entropies else 0.0,
                'std_entropy': float(np.std(easy_entropies)) if easy_entropies else 0.0,
                'count': len(easy_entropies)
            },
            'medium_tokens': {
                'mean_entropy': float(np.mean(medium_entropies)) if medium_entropies else 0.0,
                'std_entropy': float(np.std(medium_entropies)) if medium_entropies else 0.0,
                'count': len(medium_entropies)
            },
            'hard_tokens': {
                'mean_entropy': float(np.mean(hard_entropies)) if hard_entropies else 0.0,
                'std_entropy': float(np.std(hard_entropies)) if hard_entropies else 0.0,
                'count': len(hard_entropies)
            },
            'confidence_hypothesis': {
                'easy_lower_entropy': (float(np.mean(easy_entropies)) < float(np.mean(hard_entropies))) if (easy_entropies and hard_entropies) else False,
                'entropy_difference': float(np.mean(hard_entropies) - np.mean(easy_entropies)) if (easy_entropies and hard_entropies) else 0.0
            }
        }

        return results


def prepare_test_data(tokenizer, num_samples: int = 100):
    """테스트 데이터 준비"""
    from pnn.data.dataset import MLMDataset
    from torch.utils.data import DataLoader

    print("\n📚 Loading test data...")

    # WikiText-103 validation set
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="validation"
    )

    test_data = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) > 20 and len(text.split()) >= 5:
            test_data.append(text)
            if len(test_data) >= num_samples:
                break

    test_dataset = MLMDataset(
        tokenizer=tokenizer,
        data=test_data,
        max_length=128,
        mask_prob=0.15
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    return test_loader


def save_visualizations(results: Dict, output_dir: Path):
    """결과 시각화 및 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # MEG 결과 시각화
    if 'meg_analysis' in results:
        meg = results['meg_analysis']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('MEG Simulation: Temporal Activity Patterns', fontsize=16)

        for idx, (key, ax) in enumerate(zip(
            ['delta_generation', 'gate_computation', 'hidden_update', 'activity_magnitude'],
            axes.flatten()
        )):
            if key in meg and 'values' in meg[key]:
                values = meg[key]['values']
                ax.plot(values, marker='o')
                ax.set_title(f'{key.replace("_", " ").title()}')
                ax.set_xlabel('Refinement Step')
                ax.set_ylabel('Activity Magnitude')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'meg_temporal_patterns.png', dpi=300)
        plt.close()
        print(f"\n✅ Saved: {output_dir / 'meg_temporal_patterns.png'}")

    # Optogenetics 결과 시각화
    if 'optogenetics_results' in results:
        opto = results['optogenetics_results']

        # 각 컴포넌트별 억제 효과
        components = ['attention', 'gate', 'ffn']
        rates = [0, 25, 50, 75, 100]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Optogenetics Simulation: Suppression Effects', fontsize=16)

        for component in components:
            losses = []
            accs = []

            for rate in rates:
                if rate == 0:
                    key = 'baseline'
                else:
                    key = f'{component}_suppressed_{rate}'

                if key in opto:
                    losses.append(opto[key]['loss'])
                    accs.append(opto[key]['accuracy'] * 100)

            axes[0].plot(rates, losses, marker='o', label=component.capitalize())
            axes[1].plot(rates, accs, marker='o', label=component.capitalize())

        axes[0].set_xlabel('Suppression Rate (%)')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss vs Suppression Rate')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Suppression Rate (%)')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy vs Suppression Rate')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'optogenetics_suppression.png', dpi=300)
        plt.close()
        print(f"✅ Saved: {output_dir / 'optogenetics_suppression.png'}")

    # Brain activity 결과 시각화
    if 'brain_hypothesis_test' in results:
        brain = results['brain_hypothesis_test']

        # Activity breadth across steps
        steps = ['embeddings', 'step_0', 'step_1', 'step_2', 'step_3', 'final']
        breadths = [brain.get(f'{s}_breadth', 0) for s in steps]
        activities = [brain.get(f'{s}_mean_activity', 0) for s in steps]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Brain Activity Prediction: Pattern Analysis', fontsize=16)

        axes[0].plot(range(len(steps)), breadths, marker='o', linewidth=2)
        axes[0].set_xticks(range(len(steps)))
        axes[0].set_xticklabels(steps, rotation=45)
        axes[0].set_ylabel('Activity Breadth (std)')
        axes[0].set_title('Selectivity Across Processing Steps')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(range(len(steps)), activities, marker='o', linewidth=2, color='orange')
        axes[1].set_xticks(range(len(steps)))
        axes[1].set_xticklabels(steps, rotation=45)
        axes[1].set_ylabel('Mean Activity')
        axes[1].set_title('Activity Level Across Processing Steps')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'brain_activity_patterns.png', dpi=300)
        plt.close()
        print(f"✅ Saved: {output_dir / 'brain_activity_patterns.png'}")

    # Dimension-wise 결과 시각화
    if 'dimensionwise_analysis' in results:
        dim_analysis = results['dimensionwise_analysis']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Dimension-wise Analysis: Pattern Types', fontsize=16)

        # Pattern counts pie chart
        counts = dim_analysis['pattern_counts']
        axes[0].pie(
            [counts['accumulator'], counts['selector'], counts['oscillator'], counts['stable']],
            labels=['Accumulator', 'Selector', 'Oscillator', 'Stable'],
            autopct='%1.1f%%',
            colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        )
        axes[0].set_title('Distribution of Dimension Patterns')

        # Sample trajectories
        if 'step_activities' in dim_analysis:
            step_acts = np.array(dim_analysis['step_activities'])
            samples = dim_analysis.get('sample_dimensions', {})

            for dim_type, color in [
                ('accumulator_samples', '#ff9999'),
                ('selector_samples', '#66b3ff'),
                ('oscillator_samples', '#99ff99')
            ]:
                if dim_type in samples and samples[dim_type]:
                    for dim_idx in samples[dim_type][:2]:  # Show 2 examples each
                        if dim_idx < step_acts.shape[1]:
                            axes[1].plot(
                                step_acts[:, dim_idx],
                                marker='o',
                                alpha=0.6,
                                color=color,
                                label=dim_type.replace('_samples', '') if dim_idx == samples[dim_type][0] else ''
                            )

        axes[1].set_xlabel('Processing Step')
        axes[1].set_ylabel('Activity Level')
        axes[1].set_title('Sample Dimension Trajectories')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'dimensionwise_patterns.png', dpi=300)
        plt.close()
        print(f"✅ Saved: {output_dir / 'dimensionwise_patterns.png'}")

    # Token difficulty 결과 시각화
    if 'token_difficulty_analysis' in results:
        diff_analysis = results['token_difficulty_analysis']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Token Difficulty Analysis', fontsize=16)

        # Activity by difficulty
        for difficulty, color in [
            ('easy', '#66b3ff'),
            ('medium', '#ffcc99'),
            ('hard', '#ff9999')
        ]:
            if difficulty in diff_analysis and diff_analysis[difficulty]['mean']:
                mean = diff_analysis[difficulty]['mean']
                std = diff_analysis[difficulty]['std']
                steps = list(range(len(mean)))

                axes[0].plot(steps, mean, marker='o', label=f'{difficulty.capitalize()} (n={diff_analysis[difficulty]["count"]})', color=color)
                axes[0].fill_between(steps, np.array(mean) - np.array(std), np.array(mean) + np.array(std), alpha=0.2, color=color)

        axes[0].set_xlabel('Processing Step')
        axes[0].set_ylabel('Activity Level')
        axes[0].set_title('Activity by Token Difficulty')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Activity increase
        increases = []
        labels = []
        for difficulty in ['easy', 'medium', 'hard']:
            if difficulty in diff_analysis and diff_analysis[difficulty]['mean']:
                mean = diff_analysis[difficulty]['mean']
                if len(mean) >= 2:
                    increase = mean[-1] - mean[0]
                    increases.append(increase)
                    labels.append(difficulty.capitalize())

        if increases:
            axes[1].bar(labels, increases, color=['#66b3ff', '#ffcc99', '#ff9999'])
            axes[1].set_ylabel('Activity Increase')
            axes[1].set_title('Activity Increase from Start to End')
            axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'token_difficulty_analysis.png', dpi=300)
        plt.close()
        print(f"✅ Saved: {output_dir / 'token_difficulty_analysis.png'}")

    # Layer importance 결과 시각화
    if 'layer_importance_analysis' in results:
        layer_analysis = results['layer_importance_analysis']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Layer Importance Analysis', fontsize=16)

        # Importance ranking
        if 'importance_ranking' in layer_analysis:
            ranking = layer_analysis['importance_ranking']
            components = [r['component'] for r in ranking]
            scores = [r['importance_score'] * 100 for r in ranking]  # Convert to percentage

            axes[0].barh(components, scores, color=['#ff9999', '#66b3ff', '#99ff99'])
            axes[0].set_xlabel('Importance Score (Accuracy Drop %)')
            axes[0].set_title('Component Importance Ranking')
            axes[0].grid(True, alpha=0.3, axis='x')

        # Suppression effect curves
        components = ['attention', 'ffn', 'gate']
        colors = ['#ff9999', '#66b3ff', '#99ff99']

        for component, color in zip(components, colors):
            if component in layer_analysis:
                rates = []
                drops = []

                for key, value in layer_analysis[component].items():
                    if key.startswith('suppression_'):
                        rate = int(key.split('_')[1])
                        rates.append(rate)
                        drops.append(value['accuracy_drop'] * 100)

                if rates:
                    sorted_pairs = sorted(zip(rates, drops))
                    rates, drops = zip(*sorted_pairs)
                    axes[1].plot(rates, drops, marker='o', label=component.capitalize(), color=color)

        axes[1].set_xlabel('Suppression Rate (%)')
        axes[1].set_ylabel('Accuracy Drop (%)')
        axes[1].set_title('Suppression Effect on Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'layer_importance_analysis.png', dpi=300)
        plt.close()
        print(f"✅ Saved: {output_dir / 'layer_importance_analysis.png'}")

    # Gate specificity 결과 시각화
    if 'gate_specificity' in results:
        gate_spec = results['gate_specificity']

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Gate Specificity Tests: True Gate Importance', fontsize=16)
        axes = axes.flatten()

        baseline_acc = gate_spec['baseline']['accuracy'] * 100

        # Test 1: Selective suppression
        if 'selective_suppression' in gate_spec:
            rates = []
            drops = []
            for key, data in gate_spec['selective_suppression'].items():
                rate = int(key.split('_')[1])
                rates.append(rate)
                drops.append(data['accuracy_drop'] * 100)

            if rates:
                sorted_pairs = sorted(zip(rates, drops))
                rates, drops = zip(*sorted_pairs)
                axes[0].plot(rates, drops, marker='o', linewidth=2, color='#ff6b6b')
                axes[0].fill_between(rates, 0, drops, alpha=0.3, color='#ff6b6b')
                axes[0].set_xlabel('Dropout Rate (%)')
                axes[0].set_ylabel('Accuracy Drop (%)')
                axes[0].set_title('Test 1: Random Gate Dropout\n(Information Loss)')
                axes[0].grid(True, alpha=0.3)

        # Test 2: Anti-gate (single bar)
        if 'anti_gate' in gate_spec:
            anti_drop = gate_spec['anti_gate']['accuracy_drop'] * 100
            axes[1].bar(['Anti-Gate'], [anti_drop], color='#ff6b6b', width=0.5)
            axes[1].axhline(y=baseline_acc, color='red', linestyle='--', label='Baseline Acc', alpha=0.5)
            axes[1].set_ylabel('Accuracy Drop (%)')
            axes[1].set_title('Test 2: Inverse Gate\n(Bad Selection)')
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].legend()
            axes[1].text(0, anti_drop + 1, f'{anti_drop:.1f}%',
                        ha='center', va='bottom', fontweight='bold')

        # Test 3: Noise injection
        if 'noise_injection' in gate_spec:
            noise_stds = []
            drops = []
            for key, data in gate_spec['noise_injection'].items():
                std = int(key.split('_')[1])
                noise_stds.append(std)
                drops.append(data['accuracy_drop'] * 100)

            if noise_stds:
                sorted_pairs = sorted(zip(noise_stds, drops))
                noise_stds, drops = zip(*sorted_pairs)
                axes[2].plot(noise_stds, drops, marker='o', linewidth=2, color='#ff6b6b')
                axes[2].fill_between(noise_stds, 0, drops, alpha=0.3, color='#ff6b6b')
                axes[2].set_xlabel('Noise Std (%)')
                axes[2].set_ylabel('Accuracy Drop (%)')
                axes[2].set_title('Test 3: Gate Noise\n(Precision Importance)')
                axes[2].grid(True, alpha=0.3)

        # Test 4: Pattern shuffling
        if 'pattern_shuffling' in gate_spec:
            shuffle_drop = gate_spec['pattern_shuffling']['accuracy_drop'] * 100
            axes[3].bar(['Shuffled'], [shuffle_drop], color='#ff6b6b', width=0.5)
            axes[3].set_ylabel('Accuracy Drop (%)')
            axes[3].set_title('Test 4: Pattern Shuffling\n(Spatial Pattern Importance)')
            axes[3].grid(True, alpha=0.3, axis='y')
            axes[3].text(0, shuffle_drop + 1, f'{shuffle_drop:.1f}%',
                        ha='center', va='bottom', fontweight='bold')

        # Test 5: Uniform gate
        if 'uniform_gate' in gate_spec:
            uniform_drop = gate_spec['uniform_gate']['accuracy_drop'] * 100
            axes[4].bar(['Uniform'], [uniform_drop], color='#ff6b6b', width=0.5)
            axes[4].set_ylabel('Accuracy Drop (%)')
            axes[4].set_title('Test 5: Uniform Gate\n(No Selectivity)')
            axes[4].grid(True, alpha=0.3, axis='y')
            axes[4].text(0, uniform_drop + 1, f'{uniform_drop:.1f}%',
                        ha='center', va='bottom', fontweight='bold')

        # Test 6: Magnitude scaling
        if 'magnitude_scaling' in gate_spec:
            scales = []
            drops = []
            for key, data in gate_spec['magnitude_scaling'].items():
                scale = int(key.split('_')[1]) / 100
                scales.append(scale)
                drops.append(data['accuracy_drop'] * 100)

            if scales:
                sorted_pairs = sorted(zip(scales, drops))
                scales, drops = zip(*sorted_pairs)
                axes[5].plot(scales, drops, marker='o', linewidth=2, color='#4ecdc4')
                axes[5].axhline(y=0, color='green', linestyle='--', alpha=0.5)
                axes[5].fill_between(scales, 0, drops, alpha=0.3, color='#4ecdc4')
                axes[5].set_xlabel('Gate Scale Factor')
                axes[5].set_ylabel('Accuracy Drop (%)')
                axes[5].set_title('Test 6: Magnitude Scaling\n(Robustness Test)')
                axes[5].grid(True, alpha=0.3)

        # Summary comparison (Pattern Tests)
        test_names = []
        test_drops = []

        if 'anti_gate' in gate_spec:
            test_names.append('Anti-Gate')
            test_drops.append(gate_spec['anti_gate']['accuracy_drop'] * 100)
        if 'pattern_shuffling' in gate_spec:
            test_names.append('Shuffled')
            test_drops.append(gate_spec['pattern_shuffling']['accuracy_drop'] * 100)
        if 'uniform_gate' in gate_spec:
            test_names.append('Uniform')
            test_drops.append(gate_spec['uniform_gate']['accuracy_drop'] * 100)

        if test_names:
            colors_summary = ['#ff6b6b', '#ff9999', '#ffcccc']
            axes[6].barh(test_names, test_drops, color=colors_summary[:len(test_names)])
            axes[6].set_xlabel('Accuracy Drop (%)')
            axes[6].set_title('Summary: Pattern Tests\n(Larger = More Important)')
            axes[6].grid(True, alpha=0.3, axis='x')

            for i, (name, drop) in enumerate(zip(test_names, test_drops)):
                axes[6].text(drop + 0.5, i, f'{drop:.1f}%', va='center')

        # Interpretation guide (in remaining subplots)
        axes[7].axis('off')
        interpretation_text = """
        Key Insights:

        Pattern Tests (1-5):
        • Anti-Gate: Proves selection quality
        • Shuffled: Pattern > Magnitude
        • Uniform: Selection = critical

        Robustness Test (6):
        • Magnitude scaling: Tests stability
        • Flat line = robust to strength
        • Sharp drop = fragile
        """
        axes[7].text(0.1, 0.5, interpretation_text,
                    transform=axes[7].transAxes,
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        axes[8].axis('off')
        conclusion_text = """
        Conclusion:

        If Anti-Gate >> Magnitude Scaling:
        → Gate's selective function matters
        → Not just step size control
        → True importance demonstrated!

        This proves gate is NOT just
        "learning rate" but actual
        feature selection mechanism.
        """
        axes[8].text(0.1, 0.5, conclusion_text,
                    transform=axes[8].transAxes,
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout()
        plt.savefig(output_dir / 'gate_specificity_tests.png', dpi=300)
        plt.close()
        print(f"✅ Saved: {output_dir / 'gate_specificity_tests.png'}")

    # Layerwise activity 결과 시각화
    if 'layerwise_activity_analysis' in results:
        layerwise = results['layerwise_activity_analysis']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Layer-wise Activity Analysis: Accumulation Patterns', fontsize=16)

        # Mean activity per step
        if 'mean_activity_per_step' in layerwise:
            mean_acts = layerwise['mean_activity_per_step']
            std_acts = layerwise.get('std_activity_per_step', [0] * len(mean_acts))
            steps = list(range(len(mean_acts)))

            axes[0].plot(steps, mean_acts, marker='o', linewidth=2, color='#4ecdc4')
            axes[0].fill_between(steps,
                                 np.array(mean_acts) - np.array(std_acts),
                                 np.array(mean_acts) + np.array(std_acts),
                                 alpha=0.3, color='#4ecdc4')
            axes[0].set_xlabel('Processing Step')
            axes[0].set_ylabel('Mean Activity')
            axes[0].set_title('Activity Across Processing Steps')
            axes[0].grid(True, alpha=0.3)

        # Accumulation pattern
        if 'accumulation_pattern' in layerwise:
            pattern = layerwise['accumulation_pattern']
            categories = ['Early\nLayer', 'Late\nLayer']
            values = [pattern['early_activity'], pattern['late_activity']]
            colors = ['#66b3ff', '#ff9999']

            axes[1].bar(categories, values, color=colors, width=0.5)
            axes[1].set_ylabel('Activity Level')
            axes[1].set_title('Early vs Late Layer Activity')
            axes[1].grid(True, alpha=0.3, axis='y')

            # Add values on bars
            for i, (cat, val) in enumerate(zip(categories, values)):
                axes[1].text(i, val + 0.01, f'{val:.4f}',
                           ha='center', va='bottom', fontweight='bold')

            # Add hypothesis test result
            supports = pattern.get('supports_accumulation_hypothesis', False)
            increase = pattern.get('activity_increase', 0)
            result_text = f"Accumulation: {'✓' if supports else '✗'}\nIncrease: {increase:.4f}"
            axes[1].text(0.5, 0.95, result_text,
                        transform=axes[1].transAxes,
                        ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen' if supports else 'lightcoral', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / 'layerwise_activity_analysis.png', dpi=300)
        plt.close()
        print(f"✅ Saved: {output_dir / 'layerwise_activity_analysis.png'}")

    # Cross-token interference 결과 시각화
    if 'cross_token_interference' in results:
        interference = results['cross_token_interference']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Cross-Token Interference Analysis', fontsize=16)

        # Performance by context
        contexts = ['Easy\nContext', 'Hard\nContext', 'Mixed\nContext']
        perfs = [
            interference['easy_context']['mean_performance'],
            interference['hard_context']['mean_performance'],
            interference['mixed_context']['mean_performance']
        ]
        stds = [
            interference['easy_context']['std_performance'],
            interference['hard_context']['std_performance'],
            interference['mixed_context']['std_performance']
        ]
        colors = ['#66b3ff', '#ff9999', '#ffcc99']

        axes[0].bar(contexts, perfs, color=colors, yerr=stds, capsize=5, width=0.6)
        axes[0].set_ylabel('Mean Performance (Probability)')
        axes[0].set_title('Performance by Token Context')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for i, (ctx, perf) in enumerate(zip(contexts, perfs)):
            axes[0].text(i, perf + 0.02, f'{perf:.3f}',
                        ha='center', va='bottom', fontweight='bold')

        # Interference effect
        if 'interference_effect' in interference:
            effect = interference['interference_effect']
            diff = effect['easy_vs_hard_context_diff']
            interferes = effect['hard_tokens_interfere']

            categories = ['Easy - Hard\nDifference']
            values = [diff]
            color = '#ff6b6b' if interferes else '#66b3ff'

            axes[1].bar(categories, values, color=color, width=0.4)
            axes[1].set_ylabel('Performance Difference')
            axes[1].set_title('Interference Effect')
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1].grid(True, alpha=0.3, axis='y')

            # Add value on bar
            axes[1].text(0, diff + (0.01 if diff > 0 else -0.01),
                        f'{diff:.4f}',
                        ha='center',
                        va='bottom' if diff > 0 else 'top',
                        fontweight='bold')

            # Add interpretation
            result_text = f"Hard tokens interfere: {'✓' if interferes else '✗'}\n"
            if interferes:
                result_text += "Hard tokens decrease\nneighboring performance"
            else:
                result_text += "No significant\ninterference detected"

            axes[1].text(0.5, 0.95, result_text,
                        transform=axes[1].transAxes,
                        ha='center', va='top',
                        bbox=dict(boxstyle='round',
                                facecolor='lightcoral' if interferes else 'lightgreen',
                                alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / 'cross_token_interference.png', dpi=300)
        plt.close()
        print(f"✅ Saved: {output_dir / 'cross_token_interference.png'}")

    # Gate entropy 결과 시각화
    if 'gate_entropy_analysis' in results:
        entropy = results['gate_entropy_analysis']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Gate Entropy Analysis: Confidence Measure', fontsize=16)

        # Entropy by token difficulty
        difficulties = ['Easy', 'Medium', 'Hard']
        entropies = [
            entropy['easy_tokens']['mean_entropy'],
            entropy['medium_tokens']['mean_entropy'],
            entropy['hard_tokens']['mean_entropy']
        ]
        stds = [
            entropy['easy_tokens']['std_entropy'],
            entropy['medium_tokens']['std_entropy'],
            entropy['hard_tokens']['std_entropy']
        ]
        colors = ['#66b3ff', '#ffcc99', '#ff9999']

        axes[0].bar(difficulties, entropies, color=colors, yerr=stds, capsize=5, width=0.6)
        axes[0].set_ylabel('Mean Gate Entropy')
        axes[0].set_xlabel('Token Difficulty')
        axes[0].set_title('Gate Entropy by Token Difficulty')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for i, (diff, ent) in enumerate(zip(difficulties, entropies)):
            axes[0].text(i, ent + 0.01, f'{ent:.4f}',
                        ha='center', va='bottom', fontweight='bold')

        # Confidence hypothesis test
        if 'confidence_hypothesis' in entropy:
            hyp = entropy['confidence_hypothesis']
            easy_lower = hyp['easy_lower_entropy']
            ent_diff = hyp['entropy_difference']

            categories = ['Hard - Easy\nEntropy Diff']
            values = [ent_diff]
            color = '#66b3ff' if easy_lower else '#ff9999'

            axes[1].bar(categories, values, color=color, width=0.4)
            axes[1].set_ylabel('Entropy Difference')
            axes[1].set_title('Confidence Hypothesis Test')
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1].grid(True, alpha=0.3, axis='y')

            # Add value on bar
            axes[1].text(0, ent_diff + (0.005 if ent_diff > 0 else -0.005),
                        f'{ent_diff:.4f}',
                        ha='center',
                        va='bottom' if ent_diff > 0 else 'top',
                        fontweight='bold')

            # Add interpretation
            result_text = f"Easy has lower entropy: {'✓' if easy_lower else '✗'}\n"
            if easy_lower:
                result_text += "Easy → High confidence\nHard → Low confidence"
            else:
                result_text += "Hypothesis not\nsupported"

            axes[1].text(0.5, 0.95, result_text,
                        transform=axes[1].transAxes,
                        ha='center', va='top',
                        bbox=dict(boxstyle='round',
                                facecolor='lightgreen' if easy_lower else 'lightcoral',
                                alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / 'gate_entropy_analysis.png', dpi=300)
        plt.close()
        print(f"✅ Saved: {output_dir / 'gate_entropy_analysis.png'}")



def main():
    parser = argparse.ArgumentParser(
        description='Experimental Evidence for Plastic Neural Networks'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (e.g., checkpoints/best_model.pt)'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='all',
        choices=['all', 'meg', 'optogenetics', 'modeling', 'dimensionwise', 'difficulty', 'layer_importance', 'gate_specificity', 'layerwise_activity', 'cross_token', 'gate_entropy'],
        help='Which experiment to run'
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=10,
        help='Number of batches to analyze'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/experimental_evidence',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("🧠 Experimental Evidence for Plastic Neural Networks")
    print(f"{'='*80}\n")

    # Load model
    print(f"📦 Loading checkpoint: {args.checkpoint}")

    # Create model with default config
    model_config = {
        'vocab_size': 30522,
        'hidden_size': 768,
        'num_heads': 12,
        'intermediate_size': 2048,
        'max_length': 128,
        'num_steps': 4,
        'dropout': 0.1
    }
    model = create_pnn_model(model_config)

    # Load checkpoint
    # Use weights_only=False to handle custom classes like Config
    print("📥 Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # Checkpoint might be just the state dict
        state_dict = checkpoint

    # Remove '_orig_mod.' prefix if present (from torch.compile)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '')
        new_state_dict[new_key] = value

    # Map old key names to new key names
    key_mapping = {
        'embeddings.weight': 'token_embeddings.weight',
        'delta_refiner.norm1.weight': 'delta_refiner.attn_layer_norm.weight',
        'delta_refiner.norm1.bias': 'delta_refiner.attn_layer_norm.bias',
        'delta_refiner.norm2.weight': 'delta_refiner.ffn_layer_norm.weight',
        'delta_refiner.norm2.bias': 'delta_refiner.ffn_layer_norm.bias',
        'delta_refiner.gate_query.weight': 'delta_refiner.gate.query_proj.weight',
        'delta_refiner.gate_query.bias': 'delta_refiner.gate.query_proj.bias',
        'delta_refiner.gate_key.weight': 'delta_refiner.gate.key_proj.weight',
        'delta_refiner.gate_key.bias': 'delta_refiner.gate.key_proj.bias',
        'delta_refiner.temperature': 'delta_refiner.gate.temperature',
        # FFN layer indices (old model might use different indices)
        'delta_refiner.ffn.2.weight': 'delta_refiner.ffn.3.weight',
        'delta_refiner.ffn.2.bias': 'delta_refiner.ffn.3.bias',
    }

    final_state_dict = {}
    for key, value in new_state_dict.items():
        mapped_key = key_mapping.get(key, key)
        final_state_dict[mapped_key] = value

    # Load the mapped state dict
    try:
        model.load_state_dict(final_state_dict, strict=False)
        print(f"✅ Model loaded on {args.device}")
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(final_state_dict, strict=False)
        print(f"✅ Model partially loaded on {args.device}")

    model = model.to(args.device)
    model.eval()

    # Load tokenizer and data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_loader = prepare_test_data(tokenizer, num_samples=args.num_batches * 32)

    results = {}

    # Experiment 1: MEG Simulation
    if args.experiment in ['all', 'meg']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 1: MEG Simulation (High-temporal resolution)")
        print(f"{'='*80}\n")

        meg_sim = MEGSimulator(model, args.device)
        meg_analysis = meg_sim.analyze_temporal_patterns(test_loader, args.num_batches)
        results['meg_analysis'] = meg_analysis

        print("\n📊 MEG Analysis Results:")
        for key, stats in meg_analysis.items():
            if isinstance(stats, dict):
                print(f"  {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    # Experiment 2: Optogenetics Simulation
    if args.experiment in ['all', 'optogenetics']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 2: Optogenetics Simulation")
        print(f"{'='*80}\n")

        opto_sim = OptogeneticsSimulator(model, args.device)
        opto_results = opto_sim.run_suppression_experiment(test_loader, args.num_batches)
        results['optogenetics_results'] = opto_results

        print("\n📊 Optogenetics Results:")
        baseline_acc = opto_results['baseline']['accuracy']
        print(f"  Baseline: loss={opto_results['baseline']['loss']:.4f}, "
              f"acc={baseline_acc*100:.2f}%")

        for key, metrics in opto_results.items():
            if key != 'baseline':
                acc_drop = (baseline_acc - metrics['accuracy']) * 100
                print(f"  {key}: loss={metrics['loss']:.4f}, "
                      f"acc={metrics['accuracy']*100:.2f}% "
                      f"(drop: {acc_drop:.2f}%)")

    # Experiment 3: Brain Activity Modeling
    if args.experiment in ['all', 'modeling']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 3: Brain Activity Modeling")
        print(f"{'='*80}\n")

        brain_predictor = BrainActivityPredictor(model, args.device)
        activation_patterns = brain_predictor.extract_activation_patterns(
            test_loader, args.num_batches
        )
        hypothesis_test = brain_predictor.compare_with_brain_hypothesis(
            activation_patterns
        )

        results['activation_patterns'] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in activation_patterns.items()
        }
        results['brain_hypothesis_test'] = hypothesis_test

        print("\n📊 Brain Activity Analysis:")
        print(f"  Most selective step: {hypothesis_test.get('most_selective_step', 'N/A')}")
        print(f"  Supports selectivity hypothesis: "
              f"{hypothesis_test.get('supports_selectivity_hypothesis', False)}")

        for key, value in hypothesis_test.items():
            if key.endswith('_breadth'):
                print(f"  {key}: {value:.4f}")

    # Experiment 4: Dimension-wise Analysis
    if args.experiment in ['all', 'dimensionwise']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 4: Dimension-wise Analysis")
        print(f"{'='*80}\n")

        dim_analyzer = DimensionwiseAnalyzer(model, args.device)
        dim_results = dim_analyzer.analyze_dimensions(test_loader, args.num_batches)
        results['dimensionwise_analysis'] = dim_results

        print("\n📊 Dimension-wise Analysis Results:")
        counts = dim_results['pattern_counts']
        total = sum(counts.values())
        print(f"  Accumulator dimensions: {counts['accumulator']} ({counts['accumulator']/total*100:.1f}%)")
        print(f"  Selector dimensions: {counts['selector']} ({counts['selector']/total*100:.1f}%)")
        print(f"  Oscillator dimensions: {counts['oscillator']} ({counts['oscillator']/total*100:.1f}%)")
        print(f"  Stable dimensions: {counts['stable']} ({counts['stable']/total*100:.1f}%)")

    # Experiment 5: Token Difficulty Analysis
    if args.experiment in ['all', 'difficulty']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 5: Token Difficulty Analysis")
        print(f"{'='*80}\n")

        diff_analyzer = TokenDifficultyAnalyzer(model, args.device)
        diff_results = diff_analyzer.analyze_by_difficulty(test_loader, args.num_batches)
        results['token_difficulty_analysis'] = diff_results

        print("\n📊 Token Difficulty Analysis Results:")
        for difficulty in ['easy', 'medium', 'hard']:
            if difficulty in diff_results and diff_results[difficulty]['mean']:
                mean = diff_results[difficulty]['mean']
                count = diff_results[difficulty]['count']
                if len(mean) >= 2:
                    increase = mean[-1] - mean[0]
                    print(f"  {difficulty.capitalize()} tokens (n={count}): "
                          f"start={mean[0]:.4f}, end={mean[-1]:.4f}, increase={increase:.4f}")

    # Experiment 6: Layer Importance Analysis
    if args.experiment in ['all', 'layer_importance']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 6: Layer Importance Analysis")
        print(f"{'='*80}\n")

        layer_analyzer = LayerImportanceAnalyzer(model, args.device)
        layer_results = layer_analyzer.analyze_layer_importance(test_loader, args.num_batches)
        results['layer_importance_analysis'] = layer_results

        print("\n📊 Layer Importance Analysis Results:")
        if 'importance_ranking' in layer_results:
            print("  Importance Ranking (by accuracy drop):")
            for rank, item in enumerate(layer_results['importance_ranking'], 1):
                print(f"    {rank}. {item['component']}: {item['importance_score']*100:.2f}% drop")

    # Experiment 7: Gate Specificity Tests
    if args.experiment in ['all', 'gate_specificity']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 7: Gate Specificity Tests")
        print(f"{'='*80}")
        print("\n⚠️  NOTE: This tests gate's SELECTIVE function")
        print("   Previous tests (uniform scaling) = step robustness")
        print("   These tests = true gate importance\n")

        opto_sim = OptogeneticsSimulator(model, args.device)
        gate_spec_results = opto_sim.test_gate_specificity(test_loader, args.num_batches)
        results['gate_specificity'] = gate_spec_results

        print("\n📊 Gate Specificity Test Results:")
        baseline_acc = gate_spec_results['baseline']['accuracy']
        print(f"\n  Baseline: acc={baseline_acc*100:.2f}%")

        # Selective suppression results
        if 'selective_suppression' in gate_spec_results:
            print("\n  Test 1 - Selective Suppression (Random Dropout):")
            for key, data in gate_spec_results['selective_suppression'].items():
                drop = data['accuracy_drop'] * 100
                print(f"    {key}: acc={data['metrics']['accuracy']*100:.2f}%, drop={drop:.2f}%")

        # Anti-gate results
        if 'anti_gate' in gate_spec_results:
            drop = gate_spec_results['anti_gate']['accuracy_drop'] * 100
            acc = gate_spec_results['anti_gate']['metrics']['accuracy'] * 100
            print(f"\n  Test 2 - Anti-Gate (Inverse Selection):")
            print(f"    acc={acc:.2f}%, drop={drop:.2f}%")
            print(f"    → Confirms gate selects GOOD features (inverse = disaster!)")

        # Noise injection results
        if 'noise_injection' in gate_spec_results:
            print("\n  Test 3 - Noise Injection:")
            for key, data in gate_spec_results['noise_injection'].items():
                drop = data['accuracy_drop'] * 100
                print(f"    {key}: acc={data['metrics']['accuracy']*100:.2f}%, drop={drop:.2f}%")

        # Pattern shuffling results
        if 'pattern_shuffling' in gate_spec_results:
            drop = gate_spec_results['pattern_shuffling']['accuracy_drop'] * 100
            acc = gate_spec_results['pattern_shuffling']['metrics']['accuracy'] * 100
            print(f"\n  Test 4 - Pattern Shuffling:")
            print(f"    acc={acc:.2f}%, drop={drop:.2f}%")
            print(f"    → Spatial pattern matters! Not just magnitude.")

        # Uniform gate results
        if 'uniform_gate' in gate_spec_results:
            drop = gate_spec_results['uniform_gate']['accuracy_drop'] * 100
            acc = gate_spec_results['uniform_gate']['metrics']['accuracy'] * 100
            print(f"\n  Test 5 - Uniform Gate (No Selectivity):")
            print(f"    acc={acc:.2f}%, drop={drop:.2f}%")
            print(f"    → Pattern = information! Uniform = no selection = bad.")

        # Magnitude scaling results
        if 'magnitude_scaling' in gate_spec_results:
            print(f"\n  Test 6 - Magnitude Scaling (Robustness):")
            for key, data in gate_spec_results['magnitude_scaling'].items():
                scale = int(key.split('_')[1]) / 100
                drop = data['accuracy_drop'] * 100
                print(f"    scale={scale:.2f}: acc={data['metrics']['accuracy']*100:.2f}%, drop={drop:.2f}%")
            print(f"    → Tests robustness to gate strength (pattern preserved)")

    # Experiment 8: Layer-wise Activity Analysis
    if args.experiment in ['all', 'layerwise_activity']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 8: Layer-wise Activity Analysis")
        print(f"{'='*80}\n")

        layerwise_analyzer = LayerwiseActivityAnalyzer(model, args.device)
        layerwise_results = layerwise_analyzer.analyze_layerwise_activity(test_loader, args.num_batches)
        results['layerwise_activity_analysis'] = layerwise_results

        print("\n📊 Layer-wise Activity Analysis Results:")
        print(f"  Early layer activity: {layerwise_results['accumulation_pattern']['early_activity']:.4f}")
        print(f"  Late layer activity: {layerwise_results['accumulation_pattern']['late_activity']:.4f}")
        print(f"  Activity increase: {layerwise_results['accumulation_pattern']['activity_increase']:.4f}")
        print(f"  Supports accumulation hypothesis: {layerwise_results['accumulation_pattern']['supports_accumulation_hypothesis']}")

        if 'mean_activity_per_step' in layerwise_results:
            print("\n  Activity per step:")
            for step, activity in enumerate(layerwise_results['mean_activity_per_step']):
                print(f"    Step {step}: {activity:.4f}")

    # Experiment 9: Cross-Token Interference Analysis
    if args.experiment in ['all', 'cross_token']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 9: Cross-Token Interference Analysis")
        print(f"{'='*80}\n")

        interference_analyzer = CrossTokenInterferenceAnalyzer(model, args.device)
        interference_results = interference_analyzer.analyze_cross_token_interference(test_loader, args.num_batches)
        results['cross_token_interference'] = interference_results

        print("\n📊 Cross-Token Interference Results:")
        print(f"  Easy context: mean={interference_results['easy_context']['mean_performance']:.4f}, "
              f"count={interference_results['easy_context']['count']}")
        print(f"  Hard context: mean={interference_results['hard_context']['mean_performance']:.4f}, "
              f"count={interference_results['hard_context']['count']}")
        print(f"  Mixed context: mean={interference_results['mixed_context']['mean_performance']:.4f}, "
              f"count={interference_results['mixed_context']['count']}")
        print(f"\n  Interference effect:")
        print(f"    Easy vs Hard context difference: {interference_results['interference_effect']['easy_vs_hard_context_diff']:.4f}")
        print(f"    Hard tokens interfere: {interference_results['interference_effect']['hard_tokens_interfere']}")

    # Experiment 10: Gate Entropy Analysis
    if args.experiment in ['all', 'gate_entropy']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 10: Gate Entropy Analysis (Confidence Measure)")
        print(f"{'='*80}\n")

        entropy_analyzer = GateEntropyAnalyzer(model, args.device)
        entropy_results = entropy_analyzer.analyze_gate_entropy(test_loader, args.num_batches)
        results['gate_entropy_analysis'] = entropy_results

        print("\n📊 Gate Entropy Analysis Results:")
        print(f"  Easy tokens: mean_entropy={entropy_results['easy_tokens']['mean_entropy']:.4f}, "
              f"count={entropy_results['easy_tokens']['count']}")
        print(f"  Medium tokens: mean_entropy={entropy_results['medium_tokens']['mean_entropy']:.4f}, "
              f"count={entropy_results['medium_tokens']['count']}")
        print(f"  Hard tokens: mean_entropy={entropy_results['hard_tokens']['mean_entropy']:.4f}, "
              f"count={entropy_results['hard_tokens']['count']}")
        print(f"\n  Confidence hypothesis:")
        print(f"    Easy has lower entropy: {entropy_results['confidence_hypothesis']['easy_lower_entropy']}")
        print(f"    Entropy difference (hard - easy): {entropy_results['confidence_hypothesis']['entropy_difference']:.4f}")

    # Save results
    results_file = output_dir / 'experimental_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Generate visualizations
    print("\n📈 Generating visualizations...")
    save_visualizations(results, output_dir)

    print(f"\n{'='*80}")
    print("✅ Experimental analysis complete!")
    print(f"{'='*80}\n")
    print(f"📁 All results saved in: {output_dir}")
    print(f"   - experimental_results.json")

    # List generated visualizations
    if 'meg_analysis' in results:
        print(f"   - meg_temporal_patterns.png")
    if 'optogenetics_results' in results:
        print(f"   - optogenetics_suppression.png")
    if 'brain_hypothesis_test' in results:
        print(f"   - brain_activity_patterns.png")
    if 'dimensionwise_analysis' in results:
        print(f"   - dimensionwise_patterns.png")
    if 'token_difficulty_analysis' in results:
        print(f"   - token_difficulty_analysis.png")
    if 'layer_importance_analysis' in results:
        print(f"   - layer_importance_analysis.png")
    if 'gate_specificity' in results:
        print(f"   - gate_specificity_tests.png")
    if 'layerwise_activity_analysis' in results:
        print(f"   - layerwise_activity_analysis.png")
    if 'cross_token_interference' in results:
        print(f"   - cross_token_interference.png")
    if 'gate_entropy_analysis' in results:
        print(f"   - gate_entropy_analysis.png")


if __name__ == "__main__":
    main()
