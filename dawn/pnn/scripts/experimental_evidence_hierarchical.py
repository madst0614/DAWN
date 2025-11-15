"""
Experimental Evidence for Hierarchical PNN
============================================

Hierarchical PNN (DeltaRefinerHierarchical) 전용 실험 스크립트

실험 종류:
1. MEG Simulation - Gamma cycle 내 활동 패턴
2. Block Analysis - 각 hierarchical block의 기여도
3. Gate Analysis - Mini-gate들의 선택성 분석
4. Mountain Effect - Mountain-shaped FFN의 효과 검증

Usage:
    python scripts/experimental_evidence_hierarchical.py \
        --checkpoint /content/drive/MyDrive/pnn/hierarchical/best_model.pt \
        --intermediate_size "640,896,1024,896,640" \
        --num_blocks 5
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from pnn.models.pnn import create_pnn_model
from transformers import BertTokenizer
from datasets import load_dataset


class HierarchicalMEGSimulator:
    """
    Hierarchical PNN용 MEG 시뮬레이션
    각 block의 temporal dynamics 분석
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        # Hierarchical refiner인지 확인
        if not hasattr(model.delta_refiner, 'blocks'):
            raise ValueError("This simulator requires a hierarchical PNN model")

        self.num_blocks = len(model.delta_refiner.blocks)
        self.activations = {}
        self.register_hooks()

    def register_hooks(self):
        """각 block의 활동 기록"""
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output.detach()
            return hook

        # 각 block의 attention, ffn에 hook 등록
        for i, block in enumerate(self.model.delta_refiner.blocks):
            block['attention'].register_forward_hook(
                get_activation(f'block_{i}_attention')
            )
            block['ffn'].register_forward_hook(
                get_activation(f'block_{i}_ffn')
            )

        # Mini-gates에 hook 등록
        for i, gate in enumerate(self.model.delta_refiner.mini_gates):
            gate.register_forward_hook(
                get_activation(f'mini_gate_{i}')
            )

    def analyze_temporal_patterns(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10
    ) -> Dict:
        """
        Temporal pattern 분석
        각 refinement step에서 block들의 활동 패턴 측정
        """
        results = {
            'step_block_activities': [],  # [batches, steps, blocks]
            'step_gate_values': [],       # [batches, steps, blocks]
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="MEG Analysis")):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

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

                # 각 step별 분석
                step_activities = []
                step_gates = []

                for step in range(self.model.num_steps):
                    self.activations.clear()

                    # Delta refiner forward
                    delta = self.model.delta_refiner(hidden, attn_mask)

                    # Block별 활동 수집
                    block_activities = []
                    gate_values = []

                    for i in range(self.num_blocks):
                        # FFN 활동
                        ffn_key = f'block_{i}_ffn'
                        if ffn_key in self.activations:
                            activity = self.activations[ffn_key].abs().mean().cpu().item()
                            block_activities.append(activity)

                        # Gate 값
                        gate_key = f'mini_gate_{i}'
                        if gate_key in self.activations:
                            gate_val = self.activations[gate_key].mean().cpu().item()
                            gate_values.append(gate_val)

                    step_activities.append(block_activities)
                    step_gates.append(gate_values)

                    hidden = hidden + delta

                results['step_block_activities'].append(step_activities)
                results['step_gate_values'].append(step_gates)

        # 평균 계산
        step_block_act = np.array(results['step_block_activities'])  # [batches, steps, blocks]
        step_gate_val = np.array(results['step_gate_values'])

        results['mean_block_activities'] = step_block_act.mean(axis=0)  # [steps, blocks]
        results['mean_gate_values'] = step_gate_val.mean(axis=0)

        return results


class BlockContributionAnalyzer:
    """
    각 hierarchical block의 기여도 분석
    - Block ablation (특정 block 제거)
    - Mountain effect 검증 (중간 block이 중요한지)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        if not hasattr(model.delta_refiner, 'blocks'):
            raise ValueError("This analyzer requires a hierarchical PNN model")

        self.num_blocks = len(model.delta_refiner.blocks)

    def analyze_block_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10
    ) -> Dict:
        """
        각 block을 제거했을 때 성능 변화 측정
        mini_gate를 0으로 만들어서 block 기여도 제거
        """
        results = {
            'baseline': {'loss': 0.0, 'accuracy': 0.0},
            'ablations': {}
        }

        # Baseline 성능
        print("  Measuring baseline performance...")
        baseline_metrics = self._evaluate(dataloader, num_batches)
        results['baseline'] = baseline_metrics

        # 각 block ablation
        for block_idx in range(self.num_blocks):
            print(f"  Ablating block {block_idx}...")

            # mini_gate를 비활성화 (항상 0 출력)
            # Hook을 사용해서 gate 출력을 0으로 변경
            def zero_gate_hook(module, input, output):
                return torch.zeros_like(output)

            handle = self.model.delta_refiner.mini_gates[block_idx].register_forward_hook(zero_gate_hook)

            # 평가
            ablation_metrics = self._evaluate(dataloader, num_batches)
            results['ablations'][f'block_{block_idx}'] = ablation_metrics

            # Hook 제거 (복구)
            handle.remove()

        # 기여도 계산 (baseline - ablation)
        results['contributions'] = {}
        for block_name, metrics in results['ablations'].items():
            loss_increase = metrics['loss'] - baseline_metrics['loss']
            acc_decrease = baseline_metrics['accuracy'] - metrics['accuracy']
            results['contributions'][block_name] = {
                'loss_increase': loss_increase,
                'accuracy_decrease': acc_decrease
            }

        return results

    def _evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int
    ) -> Dict:
        """모델 평가"""
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

                # Forward pass
                hidden = self.model(input_ids, attention_mask)
                loss, logits = self.model.get_mlm_loss(hidden, labels)

                # Accuracy
                predictions = logits.argmax(dim=-1)
                mask = labels != -100
                correct = (predictions == labels) & mask

                total_loss += loss.item()
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        return {'loss': avg_loss, 'accuracy': accuracy}


class GateSpecificityAnalyzer:
    """
    Mini-gate들의 선택성 분석
    - 각 gate가 얼마나 선택적인지 (entropy 측정)
    - Block별 gate pattern 차이
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        if not hasattr(model.delta_refiner, 'mini_gates'):
            raise ValueError("This analyzer requires a hierarchical PNN model")

        self.num_blocks = len(model.delta_refiner.mini_gates)
        self.gate_values = {i: [] for i in range(self.num_blocks)}

    def analyze_gate_patterns(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10
    ) -> Dict:
        """Gate pattern 분석"""

        # Hook 등록
        def get_gate_hook(gate_idx):
            def hook(module, input, output):
                self.gate_values[gate_idx].append(output.detach().cpu())
            return hook

        handles = []
        for i, gate in enumerate(self.model.delta_refiner.mini_gates):
            handle = gate.register_forward_hook(get_gate_hook(i))
            handles.append(handle)

        # Data 수집
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Gate Analysis")):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                _ = self.model(input_ids, attention_mask)

        # Hook 제거
        for handle in handles:
            handle.remove()

        # 분석
        results = {
            'gate_statistics': {},
            'gate_entropy': {},
            'gate_sparsity': {}
        }

        for gate_idx in range(self.num_blocks):
            gate_vals = torch.cat(self.gate_values[gate_idx], dim=0)  # [total_tokens, hidden_size]

            # Statistics
            results['gate_statistics'][f'gate_{gate_idx}'] = {
                'mean': gate_vals.mean().item(),
                'std': gate_vals.std().item(),
                'min': gate_vals.min().item(),
                'max': gate_vals.max().item()
            }

            # Entropy (per dimension)
            # Higher entropy = less selective
            probs = gate_vals.mean(dim=0)  # Average across tokens
            probs = torch.clamp(probs, 1e-10, 1.0)
            entropy = -(probs * torch.log(probs) + (1-probs) * torch.log(1-probs)).mean()
            results['gate_entropy'][f'gate_{gate_idx}'] = entropy.item()

            # Sparsity (얼마나 많은 dimension이 거의 0인지)
            threshold = 0.1
            sparsity = (gate_vals.mean(dim=0) < threshold).float().mean()
            results['gate_sparsity'][f'gate_{gate_idx}'] = sparsity.item()

        return results


class OptogeneticsSimulator:
    """
    Optogenetics 시뮬레이션 - Hierarchical PNN용
    특정 block/gate를 억제하여 행동 변화 관찰
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        if not hasattr(model.delta_refiner, 'blocks'):
            raise ValueError("This simulator requires a hierarchical PNN model")

        self.num_blocks = len(model.delta_refiner.blocks)

    def suppress_block(
        self,
        block_idx: int,
        suppression_rate: float = 1.0
    ):
        """특정 block의 mini_gate를 억제"""
        class SuppressionHook:
            def __init__(self, rate):
                self.rate = rate

            def __call__(self, module, input, output):
                return output * (1.0 - self.rate)

        hook = SuppressionHook(suppression_rate)
        handle = self.model.delta_refiner.mini_gates[block_idx].register_forward_hook(hook)
        return handle

    def measure_behavior(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """모델 행동 측정"""
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

    def run_suppression_experiment(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """여러 block에 대한 억제 실험"""
        results = {}

        # Baseline
        print("\n📊 Baseline (No suppression)...")
        results['baseline'] = self.measure_behavior(dataloader, num_batches=num_batches)

        # Block 억제 실험
        suppression_rates = [0.25, 0.5, 0.75, 1.0]

        for block_idx in range(self.num_blocks):
            for rate in suppression_rates:
                print(f"\n🔬 Suppressing block {block_idx} at {rate*100:.0f}%...")

                handle = self.suppress_block(block_idx, rate)
                metrics = self.measure_behavior(dataloader, num_batches=num_batches)

                key = f"block_{block_idx}_suppressed_{int(rate*100)}"
                results[key] = metrics

                handle.remove()

        return results

    def test_gate_specificity(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        Mini-gate의 선택적 기능 테스트 (진짜 중요성 검증)

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

            # Apply random dropout to mini_gates
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
        """Random dropout to mini_gates (선택 정보 손실)"""
        class GateDropoutHook:
            def __init__(self, rate):
                self.rate = rate

            def __call__(self, module, input, output):
                # Random mask
                mask = (torch.rand_like(output) > self.rate).float()
                return output * mask

        # Apply to all mini_gates
        hook = GateDropoutHook(dropout_rate)
        handles = []
        for gate in self.model.delta_refiner.mini_gates:
            handle = gate.register_forward_hook(hook)
            handles.append(handle)

        # Return a composite handle
        class CompositeHandle:
            def __init__(self, handles):
                self.handles = handles

            def remove(self):
                for h in self.handles:
                    h.remove()

        return CompositeHandle(handles)

    def _apply_anti_gate(self):
        """Inverse gate (나쁜 것 선택, 좋은 것 버림)"""
        class AntiGateHook:
            def __call__(self, module, input, output):
                # Invert the gate: select what should be discarded
                return 1.0 - output

        hook = AntiGateHook()
        handles = []
        for gate in self.model.delta_refiner.mini_gates:
            handle = gate.register_forward_hook(hook)
            handles.append(handle)

        class CompositeHandle:
            def __init__(self, handles):
                self.handles = handles

            def remove(self):
                for h in self.handles:
                    h.remove()

        return CompositeHandle(handles)

    def _apply_gate_noise(self, noise_std: float):
        """Add noise to mini_gates (선택 판단에 노이즈)"""
        class GateNoiseHook:
            def __init__(self, std):
                self.std = std

            def __call__(self, module, input, output):
                noise = torch.randn_like(output) * self.std
                noisy_gate = output + noise
                # Clamp to valid range [0, 1] for sigmoid gates
                return torch.clamp(noisy_gate, 0.0, 1.0)

        hook = GateNoiseHook(noise_std)
        handles = []
        for gate in self.model.delta_refiner.mini_gates:
            handle = gate.register_forward_hook(hook)
            handles.append(handle)

        class CompositeHandle:
            def __init__(self, handles):
                self.handles = handles

            def remove(self):
                for h in self.handles:
                    h.remove()

        return CompositeHandle(handles)

    def _apply_pattern_shuffling(self):
        """Shuffle gate patterns (keeps magnitude, destroys spatial pattern)"""
        class PatternShuffleHook:
            def __call__(self, module, input, output):
                # Shuffle along the feature dimension
                batch_size, seq_len, hidden = output.shape
                shuffled = output.clone()

                # Shuffle each position independently
                for b in range(batch_size):
                    for s in range(seq_len):
                        perm = torch.randperm(hidden, device=output.device)
                        shuffled[b, s] = output[b, s, perm]

                return shuffled

        hook = PatternShuffleHook()
        handles = []
        for gate in self.model.delta_refiner.mini_gates:
            handle = gate.register_forward_hook(hook)
            handles.append(handle)

        class CompositeHandle:
            def __init__(self, handles):
                self.handles = handles

            def remove(self):
                for h in self.handles:
                    h.remove()

        return CompositeHandle(handles)

    def _apply_uniform_gate(self):
        """Replace all gates with uniform average (removes all selectivity)"""
        class UniformGateHook:
            def __call__(self, module, input, output):
                # Replace pattern with uniform average
                mean_val = output.mean(dim=-1, keepdim=True)
                uniform = mean_val.expand_as(output)
                return uniform

        hook = UniformGateHook()
        handles = []
        for gate in self.model.delta_refiner.mini_gates:
            handle = gate.register_forward_hook(hook)
            handles.append(handle)

        class CompositeHandle:
            def __init__(self, handles):
                self.handles = handles

            def remove(self):
                for h in self.handles:
                    h.remove()

        return CompositeHandle(handles)

    def _apply_magnitude_scaling(self, scale: float):
        """Scale gate magnitude (keeps pattern, changes magnitude)"""
        class MagnitudeScalingHook:
            def __init__(self, scale):
                self.scale = scale

            def __call__(self, module, input, output):
                # Scale gate values while keeping pattern
                return output * self.scale

        hook = MagnitudeScalingHook(scale)
        handles = []
        for gate in self.model.delta_refiner.mini_gates:
            handle = gate.register_forward_hook(hook)
            handles.append(handle)

        class CompositeHandle:
            def __init__(self, handles):
                self.handles = handles

            def remove(self):
                for h in self.handles:
                    h.remove()

        return CompositeHandle(handles)


class DimensionwiseAnalyzer:
    """차원별 gate 활동 분석"""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        if not hasattr(model.delta_refiner, 'mini_gates'):
            raise ValueError("This analyzer requires a hierarchical PNN model")

        self.num_blocks = len(model.delta_refiner.mini_gates)

    def analyze_dimensionwise_gates(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """차원별 gate 활성화 패턴 분석"""
        gate_activations = {i: [] for i in range(self.num_blocks)}

        def get_gate_hook(gate_idx):
            def hook(module, input, output):
                gate_activations[gate_idx].append(output.detach().cpu())
            return hook

        handles = []
        for i, gate in enumerate(self.model.delta_refiner.mini_gates):
            handle = gate.register_forward_hook(get_gate_hook(i))
            handles.append(handle)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Dimension Analysis")):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                _ = self.model(input_ids, attention_mask)

        for handle in handles:
            handle.remove()

        results = {
            'per_dimension_stats': {},
            'top_dimensions': {},
        }

        for gate_idx in range(self.num_blocks):
            gate_vals = torch.cat(gate_activations[gate_idx], dim=0)  # [tokens, hidden_size]

            # Per-dimension statistics
            dim_means = gate_vals.mean(dim=0)  # [hidden_size]
            dim_stds = gate_vals.std(dim=0)

            results['per_dimension_stats'][f'gate_{gate_idx}'] = {
                'means': dim_means.tolist(),
                'stds': dim_stds.tolist(),
                'active_ratio': (dim_means > 0.5).float().mean().item()
            }

            # Top activated dimensions
            top_dims = torch.argsort(dim_means, descending=True)[:10].tolist()
            results['top_dimensions'][f'gate_{gate_idx}'] = top_dims

        return results


class TokenDifficultyAnalyzer:
    """토큰 난이도별 모델 행동 분석"""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def analyze_by_difficulty(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """토큰 난이도별로 loss와 gate 활동 분석"""
        token_losses = []
        token_predictions = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Difficulty Analysis")):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                hidden = self.model(input_ids, attention_mask)
                loss, logits = self.model.get_mlm_loss(hidden, labels)

                # Per-token loss
                token_loss = F.cross_entropy(
                    logits.view(-1, self.model.vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='none'
                ).view(labels.shape)

                mask = labels != -100
                token_losses.extend(token_loss[mask].cpu().tolist())

                preds = logits.argmax(dim=-1)
                correct = (preds == labels) & mask
                token_predictions.extend(correct[mask].cpu().tolist())

        # Difficulty bins
        token_losses = np.array(token_losses)
        token_predictions = np.array(token_predictions)

        # Sort by loss (difficulty)
        sorted_indices = np.argsort(token_losses)
        bin_size = len(sorted_indices) // 4

        results = {
            'easy': {
                'avg_loss': float(token_losses[sorted_indices[:bin_size]].mean()),
                'accuracy': float(token_predictions[sorted_indices[:bin_size]].mean())
            },
            'medium': {
                'avg_loss': float(token_losses[sorted_indices[bin_size:2*bin_size]].mean()),
                'accuracy': float(token_predictions[sorted_indices[bin_size:2*bin_size]].mean())
            },
            'hard': {
                'avg_loss': float(token_losses[sorted_indices[2*bin_size:3*bin_size]].mean()),
                'accuracy': float(token_predictions[sorted_indices[2*bin_size:3*bin_size]].mean())
            },
            'very_hard': {
                'avg_loss': float(token_losses[sorted_indices[3*bin_size:]].mean()),
                'accuracy': float(token_predictions[sorted_indices[3*bin_size:]].mean())
            }
        }

        return results


class CrossTokenInterferenceAnalyzer:
    """토큰 간 간섭 효과 분석"""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def analyze_interference(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """단일 토큰 vs 컨텍스트 내 토큰의 예측 차이 분석"""
        single_token_accs = []
        context_token_accs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Interference Analysis")):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Full context prediction
                hidden = self.model(input_ids, attention_mask)
                _, logits = self.model.get_mlm_loss(hidden, labels)

                preds = logits.argmax(dim=-1)
                mask = labels != -100
                correct_context = (preds == labels) & mask

                # Single token prediction (mask all other tokens)
                for seq_idx in range(input_ids.size(0)):
                    for pos_idx in range(input_ids.size(1)):
                        if not mask[seq_idx, pos_idx]:
                            continue

                        # Create single-token input
                        single_input = input_ids[seq_idx:seq_idx+1, pos_idx:pos_idx+1]
                        single_mask = attention_mask[seq_idx:seq_idx+1, pos_idx:pos_idx+1]
                        single_label = labels[seq_idx:seq_idx+1, pos_idx:pos_idx+1]

                        single_hidden = self.model(single_input, single_mask)
                        _, single_logits = self.model.get_mlm_loss(single_hidden, single_label)

                        single_pred = single_logits.argmax(dim=-1)
                        correct_single = (single_pred == single_label).item()

                        single_token_accs.append(correct_single)
                        context_token_accs.append(correct_context[seq_idx, pos_idx].item())

        results = {
            'single_token_accuracy': float(np.mean(single_token_accs)),
            'context_accuracy': float(np.mean(context_token_accs)),
            'interference_effect': float(np.mean(context_token_accs) - np.mean(single_token_accs))
        }

        return results


class BrainActivityPredictor:
    """
    Brain Activity Modeling for Hierarchical PNN

    모델의 활동 패턴을 예측하고 실제 뇌 활동과 비교
    계층적 구조에서 각 블록별 패턴 분석
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        self.num_blocks = len(model.delta_refiner.blocks)

    def extract_activation_patterns(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        모델의 활동 패턴 추출 (계층적 블록 단위)

        Returns:
            patterns: 각 블록/단계별 활동 패턴
        """
        # 패턴 수집용 딕셔너리
        patterns = {
            'embeddings': [],
            'final': []
        }
        # 각 블록별 패턴
        for i in range(self.num_blocks):
            patterns[f'block_{i}'] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting patterns", total=num_batches)):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # 각 블록별 활동 캡처
                activations = {}
                def get_activation(name):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            activations[name] = output[0].detach()
                        else:
                            activations[name] = output.detach()
                    return hook

                # 각 블록의 FFN에 훅 등록
                handles = []
                for i, block in enumerate(self.model.delta_refiner.blocks):
                    handle = block['ffn'].register_forward_hook(get_activation(f'block_{i}'))
                    handles.append(handle)

                # Forward pass with return_all_steps to get embeddings
                try:
                    all_outputs = self.model(input_ids, attention_mask, return_all_steps=True)
                    # all_outputs[0] is the initial representation (embeddings + position)
                    patterns['embeddings'].append(all_outputs[0].mean(dim=[0, 1]).cpu().numpy())
                    # all_outputs[-1] is the final output
                    patterns['final'].append(all_outputs[-1].mean(dim=[0, 1]).cpu().numpy())
                except TypeError:
                    # Fallback if return_all_steps not supported
                    hidden = self.model(input_ids, attention_mask)
                    patterns['final'].append(hidden.mean(dim=[0, 1]).cpu().numpy())

                # 블록별 활동 저장
                for i in range(self.num_blocks):
                    if f'block_{i}' in activations:
                        patterns[f'block_{i}'].append(
                            activations[f'block_{i}'].mean(dim=[0, 1]).cpu().numpy()
                        )

                # 훅 제거
                for handle in handles:
                    handle.remove()

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
        - 초기 블록: 넓은 활동 (탐색)
        - 중간 블록: 선택적 활동 (집중)
        - 후기 블록: 통합적 활동 (종합)
        """
        analysis = {}

        # 1. 활동 범위 (Activity breadth) - 표준편차로 측정
        for key, pattern in patterns.items():
            if isinstance(pattern, np.ndarray) and len(pattern) > 0:
                analysis[f'{key}_breadth'] = float(np.std(pattern))
                analysis[f'{key}_mean_activity'] = float(np.mean(np.abs(pattern)))

        # 2. 패턴 변화 (Block-to-block changes)
        block_keys = [f'block_{i}' for i in range(self.num_blocks)]
        for i in range(len(block_keys) - 1):
            if block_keys[i] in patterns and block_keys[i+1] in patterns:
                p1 = patterns[block_keys[i]]
                p2 = patterns[block_keys[i+1]]
                if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
                    change = np.linalg.norm(p2 - p1)
                    analysis[f'change_{block_keys[i]}_to_{block_keys[i+1]}'] = float(change)

        # 3. 가설 검증
        # 가설: 중간 블록에서 가장 선택적 (낮은 breadth)
        breadths = [
            analysis.get(f'block_{i}_breadth', 0)
            for i in range(self.num_blocks)
        ]

        if breadths:
            min_breadth_idx = np.argmin(breadths)
            analysis['most_selective_block'] = int(min_breadth_idx)
            # 중간 블록이 가장 선택적인지 확인
            middle_range = range(self.num_blocks // 3, 2 * self.num_blocks // 3 + 1)
            analysis['supports_selectivity_hypothesis'] = (min_breadth_idx in middle_range)

        return analysis


class LayerImportanceAnalyzer:
    """
    Layer-wise Gate Importance Analysis for Hierarchical PNN

    각 블록의 컴포넌트 중요도 분석:
    - Attention importance per block
    - FFN importance per block
    - Mini-gate importance between blocks
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        self.num_blocks = len(model.delta_refiner.blocks)

    def analyze_layer_importance(
        self,
        dataloader,
        num_batches: int = 10,
        suppression_rates: list = [0.25, 0.5, 0.75]
    ) -> Dict:
        """
        각 레이어의 중요도 분석

        Returns:
            importance_analysis: 각 컴포넌트별 억제 영향
        """
        results = {}

        # Baseline
        print("\n📊 Measuring baseline performance...")
        baseline = self._measure_performance(dataloader, num_batches)
        results['baseline'] = baseline

        # 각 블록의 컴포넌트별 억제 실험
        for block_idx in range(self.num_blocks):
            for component in ['attention', 'ffn']:
                comp_name = f'block_{block_idx}_{component}'
                print(f"\n🔬 Testing {comp_name} importance...")
                component_results = {}

                for rate in suppression_rates:
                    print(f"  Suppression rate: {rate*100:.0f}%")

                    # 억제 적용
                    handle = self._suppress_block_component(block_idx, component, rate)

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

                results[comp_name] = component_results

        # Mini-gates 중요도
        for gate_idx in range(len(self.model.delta_refiner.mini_gates)):
            gate_name = f'mini_gate_{gate_idx}'
            print(f"\n🔬 Testing {gate_name} importance...")
            gate_results = {}

            for rate in suppression_rates:
                print(f"  Suppression rate: {rate*100:.0f}%")

                handle = self._suppress_mini_gate(gate_idx, rate)
                metrics = self._measure_performance(dataloader, num_batches)
                handle.remove()

                gate_results[f'suppression_{int(rate*100)}'] = {
                    'metrics': metrics,
                    'accuracy_drop': baseline['accuracy'] - metrics['accuracy'],
                    'loss_increase': metrics['loss'] - baseline['loss']
                }

            results[gate_name] = gate_results

        # 중요도 순위 계산
        importance_ranking = []
        for key in results:
            if key != 'baseline' and 'suppression_75' in results[key]:
                drop = results[key]['suppression_75']['accuracy_drop']
                importance_ranking.append((key, drop))

        importance_ranking.sort(key=lambda x: x[1], reverse=True)
        results['importance_ranking'] = [
            {'component': comp, 'importance_score': score}
            for comp, score in importance_ranking
        ]

        return results

    def _suppress_block_component(self, block_idx: int, component: str, rate: float):
        """블록 컴포넌트 억제"""
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
        block = self.model.delta_refiner.blocks[block_idx]

        if component == 'attention':
            handle = block['attention'].register_forward_hook(hook)
        elif component == 'ffn':
            handle = block['ffn'].register_forward_hook(hook)
        else:
            raise ValueError(f"Unknown component: {component}")

        return handle

    def _suppress_mini_gate(self, gate_idx: int, rate: float):
        """Mini-gate 억제"""
        class SuppressionHook:
            def __init__(self, rate):
                self.rate = rate

            def __call__(self, module, input, output):
                return output * (1.0 - self.rate)

        hook = SuppressionHook(rate)
        handle = self.model.delta_refiner.mini_gates[gate_idx].register_forward_hook(hook)
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
    Layer-wise Activity Analysis for Hierarchical PNN

    각 블록에서 activity accumulation 분석:
    - Early blocks: 낮은 activity 예상
    - Late blocks: 높은 activity 예상
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        self.num_blocks = len(model.delta_refiner.blocks)

    def analyze_layerwise_activity(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        각 블록별 activity 분석

        Returns:
            layerwise_activity: 각 블록별 평균 activity 및 accumulation 패턴
        """
        block_activities = []  # shape: (num_batches, num_blocks)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing layerwise activity", total=num_batches)):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # 각 블록의 출력 캡처
                activations = {}
                def get_activation(name):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            activations[name] = output[0].detach()
                        else:
                            activations[name] = output.detach()
                    return hook

                # 각 블록의 FFN에 훅 등록
                handles = []
                for i, block in enumerate(self.model.delta_refiner.blocks):
                    handle = block['ffn'].register_forward_hook(get_activation(f'block_{i}'))
                    handles.append(handle)

                # Forward pass
                hidden = self.model(input_ids, attention_mask)

                # 각 블록의 평균 activity 계산
                batch_block_acts = []
                for i in range(self.num_blocks):
                    if f'block_{i}' in activations:
                        # [batch, seq, hidden] -> mean activity
                        activity = activations[f'block_{i}'].abs().mean().item()
                        batch_block_acts.append(activity)

                block_activities.append(batch_block_acts)

                # 훅 제거
                for handle in handles:
                    handle.remove()

        # Convert to numpy for analysis
        block_activities = np.array(block_activities)  # (num_batches, num_blocks)

        # 각 블록별 통계
        results = {
            'mean_activity_per_block': block_activities.mean(axis=0).tolist(),
            'std_activity_per_block': block_activities.std(axis=0).tolist(),
            'accumulation_pattern': {
                'early_activity': float(block_activities[:, 0].mean()),
                'late_activity': float(block_activities[:, -1].mean()),
                'activity_increase': float(block_activities[:, -1].mean() - block_activities[:, 0].mean()),
                'supports_accumulation_hypothesis': float(block_activities[:, -1].mean()) > float(block_activities[:, 0].mean())
            },
            'num_blocks': self.num_blocks
        }

        return results


class GateEntropyAnalyzer:
    """
    Gate Entropy Analysis for Hierarchical PNN

    Mini-gate entropy를 사용한 confidence 측정:
    - confidence = -sum(gate * log(gate))
    - Easy tokens: low entropy (high confidence)
    - Hard tokens: high entropy (low confidence)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        self.num_gates = len(model.delta_refiner.mini_gates)

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

        # Register hooks on all mini_gates
        hook_handles = []
        for gate in self.model.delta_refiner.mini_gates:
            handle = gate.register_forward_hook(gate_hook)
            hook_handles.append(handle)

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
                hidden = self.model(input_ids, attention_mask)
                _, logits = self.model.get_mlm_loss(hidden, labels)
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
                    # 마지막 gate 사용 (가장 높은 수준의 결정)
                    last_gate = gate_values[-1] if len(gate_values) > 0 else None

                    if last_gate is not None:
                        # Entropy 계산: -sum(p * log(p))
                        epsilon = 1e-10
                        gate_cpu = last_gate.cpu()
                        gate_clipped = torch.clamp(gate_cpu, epsilon, 1.0 - epsilon)

                        # Binary entropy for each dimension
                        entropy = -(gate_clipped * torch.log(gate_clipped) +
                                   (1 - gate_clipped) * torch.log(1 - gate_clipped))

                        # Average entropy per token
                        token_entropy = entropy.mean(dim=-1)  # [batch, seq]

                        # Classify tokens by difficulty
                        for b in range(mask_cpu.size(0)):
                            for s in range(mask_cpu.size(1)):
                                if not mask_cpu[b, s]:
                                    continue

                                prob = correct_probs_cpu[b, s].item()
                                ent = token_entropy[b, s].item()

                                if prob > 0.7:
                                    easy_entropies.append(ent)
                                elif prob > 0.3:
                                    medium_entropies.append(ent)
                                else:
                                    hard_entropies.append(ent)

        # Remove hooks
        for handle in hook_handles:
            handle.remove()

        # 결과 분석
        results = {
            'easy_tokens': {
                'mean_entropy': float(np.mean(easy_entropies)) if easy_entropies else 0,
                'std_entropy': float(np.std(easy_entropies)) if easy_entropies else 0,
                'count': len(easy_entropies)
            },
            'medium_tokens': {
                'mean_entropy': float(np.mean(medium_entropies)) if medium_entropies else 0,
                'std_entropy': float(np.std(medium_entropies)) if medium_entropies else 0,
                'count': len(medium_entropies)
            },
            'hard_tokens': {
                'mean_entropy': float(np.mean(hard_entropies)) if hard_entropies else 0,
                'std_entropy': float(np.std(hard_entropies)) if hard_entropies else 0,
                'count': len(hard_entropies)
            }
        }

        # 가설 검증: Easy tokens는 낮은 entropy (높은 confidence)
        if easy_entropies and hard_entropies:
            results['entropy_difference'] = float(np.mean(hard_entropies) - np.mean(easy_entropies))
            results['supports_confidence_hypothesis'] = np.mean(easy_entropies) < np.mean(hard_entropies)

        return results


def prepare_test_data(tokenizer, num_samples=320, max_length=128):
    """테스트 데이터 준비"""
    from pnn.data.dataset import MLMDataset

    print("📚 Loading test data...")
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="validation",
        streaming=True
    )

    samples = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) > 20:
            samples.append(text)
            if len(samples) >= num_samples:
                break

    mlm_dataset = MLMDataset(
        tokenizer,
        samples,
        max_length=max_length,
        mask_prob=0.15
    )

    dataloader = torch.utils.data.DataLoader(
        mlm_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    return dataloader


def visualize_results(results: Dict, output_dir: Path):
    """결과 시각화"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. MEG Analysis - Block activities over steps
    if 'meg_analysis' in results:
        meg_data = results['meg_analysis']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('MEG Simulation: Hierarchical Block Activity Patterns', fontsize=16)

        # Block activities
        if 'mean_block_activities' in meg_data:
            activities = meg_data['mean_block_activities']  # [steps, blocks]
            ax = axes[0, 0]
            for block_idx in range(activities.shape[1]):
                ax.plot(activities[:, block_idx], marker='o', label=f'Block {block_idx}', linewidth=2)
            ax.set_xlabel('Refinement Step')
            ax.set_ylabel('Activity Magnitude')
            ax.set_title('Block FFN Activity Across Steps')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Gate values
        if 'mean_gate_values' in meg_data:
            gates = meg_data['mean_gate_values']  # [steps, blocks]
            ax = axes[0, 1]
            for block_idx in range(gates.shape[1]):
                ax.plot(gates[:, block_idx], marker='s', label=f'Gate {block_idx}', linewidth=2)
            ax.set_xlabel('Refinement Step')
            ax.set_ylabel('Gate Value')
            ax.set_title('Mini-Gate Values Across Steps')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Average activity per step
        if 'mean_block_activities' in meg_data:
            activities = meg_data['mean_block_activities']
            ax = axes[1, 0]
            step_avg = activities.mean(axis=1)  # Average across blocks
            ax.plot(step_avg, marker='o', linewidth=2, color='purple')
            ax.fill_between(range(len(step_avg)), step_avg, alpha=0.3, color='purple')
            ax.set_xlabel('Refinement Step')
            ax.set_ylabel('Average Activity')
            ax.set_title('Overall Activity Magnitude')
            ax.grid(True, alpha=0.3)

        # Block-wise variance
        if 'mean_block_activities' in meg_data:
            activities = meg_data['mean_block_activities']
            ax = axes[1, 1]
            block_variance = activities.var(axis=0)  # Variance over steps for each block
            ax.bar(range(len(block_variance)), block_variance, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(block_variance))))
            ax.set_xlabel('Block Index')
            ax.set_ylabel('Activity Variance')
            ax.set_title('Block Activity Variance (Stability)')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'meg_temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_dir / 'meg_temporal_patterns.png'}")

    # 2. Block Contribution (Ablation Study)
    if 'block_analysis' in results:
        block_data = results['block_analysis']
        if 'contributions' in block_data:
            contributions = block_data['contributions']

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('Block Importance Analysis (Ablation Study)', fontsize=16)

            block_names = list(contributions.keys())
            acc_decreases = [contributions[b]['accuracy_decrease'] * 100 for b in block_names]
            loss_increases = [contributions[b]['loss_increase'] for b in block_names]

            # Accuracy impact
            bars = ax1.bar(range(len(block_names)), acc_decreases)
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(block_names)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            ax1.set_xlabel('Block')
            ax1.set_ylabel('Accuracy Decrease (%)')
            ax1.set_title('Accuracy Impact When Ablated')
            ax1.set_xticks(range(len(block_names)))
            ax1.set_xticklabels([f'B{i}' for i in range(len(block_names))], rotation=0)
            ax1.grid(True, alpha=0.3, axis='y')

            # Add values on bars
            for i, v in enumerate(acc_decreases):
                ax1.text(i, v + 0.1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

            # Loss impact
            bars = ax2.bar(range(len(block_names)), loss_increases, color='coral')
            ax2.set_xlabel('Block')
            ax2.set_ylabel('Loss Increase')
            ax2.set_title('Loss Impact When Ablated')
            ax2.set_xticks(range(len(block_names)))
            ax2.set_xticklabels([f'B{i}' for i in range(len(block_names))], rotation=0)
            ax2.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(output_dir / 'block_contributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Saved: {output_dir / 'block_contributions.png'}")

    # 3. Gate Analysis
    if 'gate_analysis' in results:
        gate_data = results['gate_analysis']
        if 'gate_statistics' in gate_data:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Mini-Gate Analysis', fontsize=16)

            stats = gate_data['gate_statistics']
            gate_names = list(stats.keys())
            means = [stats[g]['mean'] for g in gate_names]
            stds = [stats[g]['std'] for g in gate_names]

            # Mean values
            ax = axes[0, 0]
            ax.bar(range(len(gate_names)), means, color='skyblue')
            ax.set_xlabel('Gate')
            ax.set_ylabel('Mean Gate Value')
            ax.set_title('Average Gate Activation')
            ax.set_xticks(range(len(gate_names)))
            ax.set_xticklabels([f'G{i}' for i in range(len(gate_names))], rotation=0)
            ax.grid(True, alpha=0.3, axis='y')

            # Entropy
            ax = axes[0, 1]
            if 'gate_entropy' in gate_data:
                entropies = [gate_data['gate_entropy'][g] for g in gate_names]
                ax.bar(range(len(gate_names)), entropies, color='coral')
                ax.set_xlabel('Gate')
                ax.set_ylabel('Entropy')
                ax.set_title('Gate Selectivity (Lower = More Selective)')
                ax.set_xticks(range(len(gate_names)))
                ax.set_xticklabels([f'G{i}' for i in range(len(gate_names))], rotation=0)
                ax.grid(True, alpha=0.3, axis='y')

            # Sparsity
            ax = axes[1, 0]
            if 'gate_sparsity' in gate_data:
                sparsities = [gate_data['gate_sparsity'][g] * 100 for g in gate_names]
                ax.bar(range(len(gate_names)), sparsities, color='lightgreen')
                ax.set_xlabel('Gate')
                ax.set_ylabel('Sparsity (%)')
                ax.set_title('Gate Sparsity (dims < 0.1)')
                ax.set_xticks(range(len(gate_names)))
                ax.set_xticklabels([f'G{i}' for i in range(len(gate_names))], rotation=0)
                ax.grid(True, alpha=0.3, axis='y')

            # Mean vs Std scatter
            ax = axes[1, 1]
            ax.scatter(means, stds, s=100, alpha=0.6, c=range(len(gate_names)), cmap='viridis')
            for i, name in enumerate(gate_names):
                ax.annotate(f'G{i}', (means[i], stds[i]), fontsize=9, ha='right')
            ax.set_xlabel('Mean Gate Value')
            ax.set_ylabel('Std Gate Value')
            ax.set_title('Gate Mean vs Variability')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'gate_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Saved: {output_dir / 'gate_analysis.png'}")

    # 4. Optogenetics - Suppression Effects
    if 'optogenetics' in results:
        opto = results['optogenetics']
        baseline_loss = opto['baseline']['loss']
        baseline_acc = opto['baseline']['accuracy'] * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Optogenetics Simulation: Block Suppression Effects', fontsize=16)

        rates = [0, 25, 50, 75, 100]
        num_blocks = 0

        # Determine number of blocks
        for key in opto.keys():
            if key.startswith('block_'):
                block_idx = int(key.split('_')[1])
                num_blocks = max(num_blocks, block_idx + 1)

        # Loss plot
        ax = axes[0]
        for block_idx in range(num_blocks):
            losses = []
            for rate in rates:
                if rate == 0:
                    key = 'baseline'
                else:
                    key = f'block_{block_idx}_suppressed_{rate}'
                if key in opto:
                    losses.append(opto[key]['loss'])
            if losses:
                ax.plot(rates, losses, marker='o', label=f'Block {block_idx}', linewidth=2)

        ax.set_xlabel('Suppression Rate (%)')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Suppression Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy plot
        ax = axes[1]
        for block_idx in range(num_blocks):
            accs = []
            for rate in rates:
                if rate == 0:
                    key = 'baseline'
                else:
                    key = f'block_{block_idx}_suppressed_{rate}'
                if key in opto:
                    accs.append(opto[key]['accuracy'] * 100)
            if accs:
                ax.plot(rates, accs, marker='o', label=f'Block {block_idx}', linewidth=2)

        ax.set_xlabel('Suppression Rate (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy vs Suppression Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'optogenetics_suppression.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_dir / 'optogenetics_suppression.png'}")

    # 5. Dimensionwise Analysis
    if 'dimensionwise' in results:
        dim_data = results['dimensionwise']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Dimensionwise Gate Analysis', fontsize=16)

        # Active ratios
        ax = axes[0]
        if 'per_dimension_stats' in dim_data:
            gate_names = list(dim_data['per_dimension_stats'].keys())
            active_ratios = [dim_data['per_dimension_stats'][g]['active_ratio'] * 100 for g in gate_names]

            bars = ax.bar(range(len(gate_names)), active_ratios, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(gate_names))))
            ax.set_xlabel('Gate')
            ax.set_ylabel('Active Dimensions (%)')
            ax.set_title('Percentage of Active Dimensions (> 0.5)')
            ax.set_xticks(range(len(gate_names)))
            ax.set_xticklabels([f'G{i}' for i in range(len(gate_names))], rotation=0)
            ax.grid(True, alpha=0.3, axis='y')

            for i, v in enumerate(active_ratios):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

        # Top dimensions table
        ax = axes[1]
        if 'top_dimensions' in dim_data:
            gate_names = list(dim_data['top_dimensions'].keys())
            top_dims_data = []
            for g in gate_names:
                top_dims_data.append(dim_data['top_dimensions'][g][:5])  # Top 5

            ax.axis('off')

            # Create table with proper formatting
            table_data = [[f'Gate {i}'] + [f'{d}' for d in dims] for i, dims in enumerate(top_dims_data)]
            table = ax.table(
                cellText=table_data,
                colLabels=['', 'Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5'],
                cellLoc='center',
                loc='center',
                bbox=[0.1, 0.1, 0.8, 0.75]  # [left, bottom, width, height]
            )

            table.auto_set_font_size(False)
            table.set_fontsize(9)

            # Set column widths explicitly
            for i in range(len(table_data) + 1):
                table[(i, 0)].set_width(0.15)  # Gate column
                for j in range(1, 6):
                    table[(i, j)].set_width(0.15)  # Dimension columns

            # Style the table
            for i in range(len(table_data) + 1):
                for j in range(6):
                    cell = table[(i, j)]
                    cell.set_height(0.08)
                    if i == 0:  # Header row
                        cell.set_facecolor('#4472C4')
                        cell.set_text_props(weight='bold', color='white', fontsize=10)
                    else:
                        cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
                        cell.set_text_props(fontsize=9)
                    cell.set_edgecolor('#CCCCCC')
                    cell.set_linewidth(1)

            # Add title above table
            ax.text(0.5, 0.95, 'Top 5 Most Activated Dimensions per Gate',
                   ha='center', va='top', fontsize=13, fontweight='bold',
                   transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(output_dir / 'dimensionwise_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_dir / 'dimensionwise_patterns.png'}")

    # 6. Token Difficulty Analysis
    if 'difficulty' in results:
        diff_data = results['difficulty']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Token Difficulty Analysis', fontsize=16)

        categories = ['easy', 'medium', 'hard', 'very_hard']
        colors_diff = ['#66b3ff', '#99ccff', '#ffcc99', '#ff9999']

        # Loss by difficulty
        ax = axes[0]
        losses = [diff_data[cat]['avg_loss'] for cat in categories if cat in diff_data]
        bars = ax.bar(range(len(losses)), losses, color=colors_diff[:len(losses)])
        ax.set_xlabel('Difficulty')
        ax.set_ylabel('Average Loss')
        ax.set_title('Loss by Token Difficulty')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([c.capitalize() for c in categories], rotation=0)
        ax.grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(losses):
            ax.text(i, v + 0.05, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        # Accuracy by difficulty
        ax = axes[1]
        accs = [diff_data[cat]['accuracy'] * 100 for cat in categories if cat in diff_data]
        bars = ax.bar(range(len(accs)), accs, color=colors_diff[:len(accs)])
        ax.set_xlabel('Difficulty')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy by Token Difficulty')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([c.capitalize() for c in categories], rotation=0)
        ax.grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(accs):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'token_difficulty_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_dir / 'token_difficulty_analysis.png'}")

    # 7. Cross-Token Interference
    if 'cross_token' in results:
        interference_data = results['cross_token']

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Cross-Token Interference Analysis', fontsize=16)

        categories = ['Single Token', 'With Context']
        accuracies = [
            interference_data['single_token_accuracy'] * 100,
            interference_data['context_accuracy'] * 100
        ]
        colors_int = ['#ff9999', '#66b3ff']

        bars = ax.bar(categories, accuracies, color=colors_int, width=0.5)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Token Prediction Accuracy: Isolated vs Contextual')
        ax.grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(accuracies):
            ax.text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add interference effect annotation
        interference_effect = interference_data['interference_effect'] * 100
        ax.text(0.5, min(accuracies) - 5,
               f'Interference Effect: {interference_effect:+.2f}%\n' +
               ('(Context helps)' if interference_effect > 0 else '(Context hurts)'),
               ha='center', va='top', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / 'cross_token_interference.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_dir / 'cross_token_interference.png'}")

    # 8. Brain Activity Patterns
    if 'brain_hypothesis_test' in results:
        brain = results['brain_hypothesis_test']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Brain Activity Prediction: Pattern Analysis', fontsize=16)

        # Activity breadth across blocks
        block_keys = [key for key in brain.keys() if '_breadth' in key and 'block_' in key]
        block_keys = sorted(block_keys, key=lambda x: int(x.split('_')[1]))
        breadths = [brain[key] for key in block_keys]
        block_labels = [f"Block {key.split('_')[1]}" for key in block_keys]

        axes[0].plot(range(len(breadths)), breadths, marker='o', linewidth=2)
        axes[0].set_xticks(range(len(breadths)))
        axes[0].set_xticklabels(block_labels, rotation=45)
        axes[0].set_ylabel('Activity Breadth (std)')
        axes[0].set_title('Selectivity Across Processing Blocks')
        axes[0].grid(True, alpha=0.3)

        # Activity levels across blocks
        activity_keys = [key for key in brain.keys() if '_mean_activity' in key and 'block_' in key]
        activity_keys = sorted(activity_keys, key=lambda x: int(x.split('_')[1]))
        activities = [brain[key] for key in activity_keys]

        axes[1].plot(range(len(activities)), activities, marker='o', linewidth=2, color='orange')
        axes[1].set_xticks(range(len(activities)))
        axes[1].set_xticklabels(block_labels, rotation=45)
        axes[1].set_ylabel('Mean Activity')
        axes[1].set_title('Activity Level Across Processing Blocks')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'brain_activity_patterns.png', dpi=300)
        plt.close()
        print(f"  ✅ Saved: {output_dir / 'brain_activity_patterns.png'}")

    # 9. Layer Importance Analysis
    if 'layer_importance_analysis' in results:
        layer_analysis = results['layer_importance_analysis']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Layer Importance Analysis: Component Criticality', fontsize=16)

        # Importance ranking
        if 'importance_ranking' in layer_analysis:
            ranking = layer_analysis['importance_ranking'][:10]  # Top 10
            components = [r['component'] for r in ranking]
            scores = [r['importance_score'] * 100 for r in ranking]

            bars = axes[0].barh(range(len(components)), scores, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(components))))
            axes[0].set_yticks(range(len(components)))
            axes[0].set_yticklabels(components, fontsize=9)
            axes[0].set_xlabel('Accuracy Drop (%)')
            axes[0].set_title('Top 10 Most Critical Components')
            axes[0].grid(True, alpha=0.3, axis='x')
            axes[0].invert_yaxis()

            for i, v in enumerate(scores):
                axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

        # Suppression effect curves
        axes[1].set_xlabel('Suppression Rate (%)')
        axes[1].set_ylabel('Accuracy Drop (%)')
        axes[1].set_title('Suppression Effect on Accuracy')
        axes[1].grid(True, alpha=0.3)

        # Plot a few key components
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        plotted = 0
        for component in ['block_0_ffn', 'block_0_attention', 'mini_gate_0']:
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
                    color = colors[plotted % len(colors)]
                    axes[1].plot(rates, drops, marker='o', label=component, color=color)
                    plotted += 1

        if plotted > 0:
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'layer_importance_analysis.png', dpi=300)
        plt.close()
        print(f"  ✅ Saved: {output_dir / 'layer_importance_analysis.png'}")

    # 10. Gate Specificity Tests
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
                if key.startswith('dropout_'):
                    rate = int(key.split('_')[1])
                    rates.append(rate)
                    drops.append(data['accuracy_drop'] * 100)

            if rates:
                sorted_pairs = sorted(zip(rates, drops))
                rates, drops = zip(*sorted_pairs)
                axes[0].plot(rates, drops, marker='o', linewidth=2, color='#e74c3c')
                axes[0].set_xlabel('Dropout Rate (%)')
                axes[0].set_ylabel('Accuracy Drop (%)')
                axes[0].set_title('Test 1: Random Gate Dropout')
                axes[0].grid(True, alpha=0.3)

        # Test 2: Anti-gate
        if 'anti_gate' in gate_spec:
            drop = gate_spec['anti_gate']['accuracy_drop'] * 100
            axes[1].bar(['Baseline', 'Anti-Gate'], [0, drop], color=['#2ecc71', '#e74c3c'])
            axes[1].set_ylabel('Accuracy Drop (%)')
            axes[1].set_title('Test 2: Anti-Gate (Inverse Selection)')
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].text(1, drop + 1, f'{drop:.1f}%', ha='center', va='bottom')

        # Test 3: Noise injection
        if 'noise_injection' in gate_spec:
            noise_levels = []
            drops = []
            for key, data in gate_spec['noise_injection'].items():
                if key.startswith('noise_'):
                    level = int(key.split('_')[1])
                    noise_levels.append(level)
                    drops.append(data['accuracy_drop'] * 100)

            if noise_levels:
                sorted_pairs = sorted(zip(noise_levels, drops))
                noise_levels, drops = zip(*sorted_pairs)
                axes[2].plot(noise_levels, drops, marker='o', linewidth=2, color='#9b59b6')
                axes[2].set_xlabel('Noise Std (%)')
                axes[2].set_ylabel('Accuracy Drop (%)')
                axes[2].set_title('Test 3: Noise Injection')
                axes[2].grid(True, alpha=0.3)

        # Test 4: Pattern shuffling
        if 'pattern_shuffling' in gate_spec:
            drop = gate_spec['pattern_shuffling']['accuracy_drop'] * 100
            axes[3].bar(['Baseline', 'Shuffled'], [0, drop], color=['#2ecc71', '#e74c3c'])
            axes[3].set_ylabel('Accuracy Drop (%)')
            axes[3].set_title('Test 4: Pattern Shuffling')
            axes[3].grid(True, alpha=0.3, axis='y')
            axes[3].text(1, drop + 1, f'{drop:.1f}%', ha='center', va='bottom')

        # Test 5: Uniform gate
        if 'uniform_gate' in gate_spec:
            drop = gate_spec['uniform_gate']['accuracy_drop'] * 100
            axes[4].bar(['Baseline', 'Uniform'], [0, drop], color=['#2ecc71', '#e74c3c'])
            axes[4].set_ylabel('Accuracy Drop (%)')
            axes[4].set_title('Test 5: Uniform Gate')
            axes[4].grid(True, alpha=0.3, axis='y')
            axes[4].text(1, drop + 1, f'{drop:.1f}%', ha='center', va='bottom')

        # Test 6: Magnitude scaling
        if 'magnitude_scaling' in gate_spec:
            scales = []
            drops = []
            for key, data in gate_spec['magnitude_scaling'].items():
                if key.startswith('scale_'):
                    scale = int(key.split('_')[1])
                    scales.append(scale)
                    drops.append(data['accuracy_drop'] * 100)

            if scales:
                sorted_pairs = sorted(zip(scales, drops))
                scales, drops = zip(*sorted_pairs)
                axes[5].plot(scales, drops, marker='o', linewidth=2, color='#3498db')
                axes[5].set_xlabel('Scale (%)')
                axes[5].set_ylabel('Accuracy Drop (%)')
                axes[5].set_title('Test 6: Magnitude Scaling')
                axes[5].grid(True, alpha=0.3)
                axes[5].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Interpretation panels
        axes[6].axis('off')
        interpretation_text = """
        Key Tests:

        • Dropout: Random masking
        • Anti-Gate: Invert selection
        • Noise: Add randomness
        • Shuffle: Destroy pattern
        • Uniform: Remove selectivity
        • Magnitude: Test scaling
        """
        axes[6].text(0.1, 0.5, interpretation_text,
                    transform=axes[6].transAxes,
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        axes[7].axis('off')
        interpretation_text = """
        What This Shows:

        • Dropout/Noise: Robustness
        • Anti-Gate: Selection quality
        • Shuffle/Uniform: Pattern importance
        • Magnitude scaling: Stability
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
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        plt.tight_layout()
        plt.savefig(output_dir / 'gate_specificity_tests.png', dpi=300)
        plt.close()
        print(f"  ✅ Saved: {output_dir / 'gate_specificity_tests.png'}")

    # 11. Layerwise Activity Analysis
    if 'layerwise_activity_analysis' in results:
        layerwise = results['layerwise_activity_analysis']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Layer-wise Activity Analysis: Accumulation Patterns', fontsize=16)

        # Mean activity per block
        if 'mean_activity_per_block' in layerwise:
            mean_acts = layerwise['mean_activity_per_block']
            std_acts = layerwise.get('std_activity_per_block', [0] * len(mean_acts))
            blocks = list(range(len(mean_acts)))

            axes[0].plot(blocks, mean_acts, marker='o', linewidth=2, color='#4ecdc4')
            axes[0].fill_between(blocks,
                                 np.array(mean_acts) - np.array(std_acts),
                                 np.array(mean_acts) + np.array(std_acts),
                                 alpha=0.3, color='#4ecdc4')
            axes[0].set_xlabel('Block Index')
            axes[0].set_ylabel('Mean Activity')
            axes[0].set_title('Activity Progression Across Blocks')
            axes[0].grid(True, alpha=0.3)

        # Early vs Late activity
        if 'accumulation_pattern' in layerwise:
            pattern = layerwise['accumulation_pattern']
            categories = ['Early\nBlocks', 'Late\nBlocks']
            values = [pattern['early_activity'], pattern['late_activity']]
            colors = ['#66b3ff', '#ff9999']

            axes[1].bar(categories, values, color=colors, width=0.5)
            axes[1].set_ylabel('Activity Level')
            axes[1].set_title('Early vs Late Block Activity')
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
        print(f"  ✅ Saved: {output_dir / 'layerwise_activity_analysis.png'}")

    # 12. Gate Entropy Analysis
    if 'gate_entropy_analysis' in results:
        entropy = results['gate_entropy_analysis']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Gate Entropy Analysis: Confidence Measurement', fontsize=16)

        # Entropy by difficulty
        categories = ['Easy', 'Medium', 'Hard']
        if all(cat.lower() + '_tokens' in entropy for cat in categories):
            means = [entropy[cat.lower() + '_tokens']['mean_entropy'] for cat in categories]
            stds = [entropy[cat.lower() + '_tokens']['std_entropy'] for cat in categories]
            colors_entropy = ['#66b3ff', '#99ccff', '#ff9999']

            bars = axes[0].bar(categories, means, yerr=stds, color=colors_entropy, capsize=5)
            axes[0].set_ylabel('Mean Gate Entropy')
            axes[0].set_title('Gate Entropy by Token Difficulty')
            axes[0].grid(True, alpha=0.3, axis='y')

            for i, (m, s) in enumerate(zip(means, stds)):
                axes[0].text(i, m + s + 0.005, f'{m:.3f}',
                           ha='center', va='bottom', fontsize=9)

        # Hypothesis test
        if 'entropy_difference' in entropy and 'supports_confidence_hypothesis' in entropy:
            ent_diff = entropy['entropy_difference']
            easy_lower = entropy['supports_confidence_hypothesis']

            color = '#2ecc71' if easy_lower else '#e74c3c'

            axes[1].bar(categories[0], ent_diff, color=color, width=0.4)
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
        print(f"  ✅ Saved: {output_dir / 'gate_entropy_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Experimental Evidence for Hierarchical PNN'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to hierarchical PNN checkpoint'
    )
    parser.add_argument(
        '--intermediate_size',
        type=str,
        required=True,
        help='Comma-separated list of intermediate sizes (e.g., "640,896,1024,896,640")'
    )
    parser.add_argument(
        '--num_blocks',
        type=int,
        required=True,
        help='Number of hierarchical blocks'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='all',
        choices=['all', 'meg', 'blocks', 'gates', 'optogenetics', 'dimensionwise', 'difficulty', 'cross_token',
                 'modeling', 'layer_importance', 'gate_specificity', 'layerwise_activity', 'gate_entropy'],
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
        default='results/hierarchical_evidence',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("🧠 Experimental Evidence for Hierarchical PNN")
    print(f"{'='*80}\n")

    # Parse intermediate_size
    intermediate_sizes = [int(x.strip()) for x in args.intermediate_size.split(',')]
    assert len(intermediate_sizes) == args.num_blocks, \
        f"intermediate_size list length ({len(intermediate_sizes)}) must match num_blocks ({args.num_blocks})"

    print(f"📦 Loading checkpoint: {args.checkpoint}")
    print(f"   Intermediate sizes: {intermediate_sizes}")
    print(f"   Num blocks: {args.num_blocks}")

    # Create hierarchical model
    model_config = {
        'vocab_size': 30522,
        'hidden_size': 768,
        'num_heads': 12,
        'intermediate_size': intermediate_sizes,
        'max_length': 128,
        'num_steps': 4,
        'num_blocks': args.num_blocks,
        'use_checkpoint': False,
        'dropout': 0.1
    }

    model = create_pnn_model(model_config, model_type='pnn_hierarchical')

    # Load checkpoint
    print("📥 Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove _orig_mod. prefix if present
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '')
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(args.device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {total_params:,} parameters ({total_params/1e6:.1f}M)")

    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'best_acc' in checkpoint:
        print(f"   Best Acc: {checkpoint['best_acc']:.4f} ({checkpoint['best_acc']*100:.2f}%)")

    # Load data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_loader = prepare_test_data(tokenizer, num_samples=args.num_batches * 32)

    results = {}

    # Experiment 1: MEG Analysis
    if args.experiment in ['all', 'meg']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 1: MEG Simulation (Temporal Dynamics)")
        print(f"{'='*80}\n")

        meg_sim = HierarchicalMEGSimulator(model, args.device)
        meg_results = meg_sim.analyze_temporal_patterns(test_loader, args.num_batches)
        results['meg_analysis'] = meg_results

        print("\n📊 MEG Results:")
        activities = meg_results['mean_block_activities']
        print(f"   Block activities shape: {activities.shape} (steps × blocks)")
        print(f"   Activities by step:")
        for step_idx in range(activities.shape[0]):
            step_acts = activities[step_idx]
            print(f"     Step {step_idx}: {[f'{x:.4f}' for x in step_acts]}")

    # Experiment 2: Block Importance
    if args.experiment in ['all', 'blocks']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 2: Block Contribution Analysis")
        print(f"{'='*80}\n")

        block_analyzer = BlockContributionAnalyzer(model, args.device)
        block_results = block_analyzer.analyze_block_importance(test_loader, args.num_batches)
        results['block_analysis'] = block_results

        print("\n📊 Block Ablation Results:")
        baseline = block_results['baseline']
        print(f"   Baseline: loss={baseline['loss']:.4f}, acc={baseline['accuracy']*100:.2f}%")
        print(f"\n   Ablation impacts:")
        for block_name, contrib in block_results['contributions'].items():
            print(f"     {block_name}: Δacc={contrib['accuracy_decrease']*100:+.2f}%, "
                  f"Δloss={contrib['loss_increase']:+.4f}")

    # Experiment 3: Gate Analysis
    if args.experiment in ['all', 'gates']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 3: Gate Specificity Analysis")
        print(f"{'='*80}\n")

        gate_analyzer = GateSpecificityAnalyzer(model, args.device)
        gate_results = gate_analyzer.analyze_gate_patterns(test_loader, args.num_batches)
        results['gate_analysis'] = gate_results

        print("\n📊 Gate Analysis Results:")
        for gate_name, stats in gate_results['gate_statistics'].items():
            entropy = gate_results['gate_entropy'].get(gate_name, 0)
            sparsity = gate_results['gate_sparsity'].get(gate_name, 0)
            print(f"   {gate_name}:")
            print(f"     Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"     Entropy: {entropy:.4f} (lower = more selective)")
            print(f"     Sparsity: {sparsity*100:.1f}% (dims < 0.1)")

    # Experiment 4: Optogenetics
    if args.experiment in ['all', 'optogenetics']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 4: Optogenetics Simulation")
        print(f"{'='*80}\n")

        opto_sim = OptogeneticsSimulator(model, args.device)
        opto_results = opto_sim.run_suppression_experiment(test_loader, args.num_batches)
        results['optogenetics'] = opto_results

        print("\n📊 Optogenetics Results:")
        baseline = opto_results['baseline']
        print(f"   Baseline: loss={baseline['loss']:.4f}, acc={baseline['accuracy']*100:.2f}%")
        print(f"\n   Suppression impacts:")
        for key, metrics in opto_results.items():
            if key != 'baseline':
                acc_drop = (baseline['accuracy'] - metrics['accuracy']) * 100
                print(f"     {key}: acc={metrics['accuracy']*100:.2f}% (Δ{acc_drop:+.2f}%)")

    # Experiment 5: Dimensionwise Analysis
    if args.experiment in ['all', 'dimensionwise']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 5: Dimensionwise Gate Analysis")
        print(f"{'='*80}\n")

        dim_analyzer = DimensionwiseAnalyzer(model, args.device)
        dim_results = dim_analyzer.analyze_dimensionwise_gates(test_loader, args.num_batches)
        results['dimensionwise'] = dim_results

        print("\n📊 Dimensionwise Results:")
        for gate_name, stats in dim_results['per_dimension_stats'].items():
            print(f"   {gate_name}:")
            print(f"     Active ratio: {stats['active_ratio']*100:.1f}%")
            print(f"     Top dimensions: {dim_results['top_dimensions'][gate_name][:5]}")

    # Experiment 6: Token Difficulty
    if args.experiment in ['all', 'difficulty']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 6: Token Difficulty Analysis")
        print(f"{'='*80}\n")

        diff_analyzer = TokenDifficultyAnalyzer(model, args.device)
        diff_results = diff_analyzer.analyze_by_difficulty(test_loader, args.num_batches)
        results['difficulty'] = diff_results

        print("\n📊 Difficulty Results:")
        for category, stats in diff_results.items():
            print(f"   {category.upper()}: loss={stats['avg_loss']:.4f}, acc={stats['accuracy']*100:.2f}%")

    # Experiment 7: Cross-Token Interference
    if args.experiment in ['all', 'cross_token']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 7: Cross-Token Interference Analysis")
        print(f"{'='*80}\n")

        interference_analyzer = CrossTokenInterferenceAnalyzer(model, args.device)
        interference_results = interference_analyzer.analyze_interference(test_loader, args.num_batches)
        results['cross_token'] = interference_results

        print("\n📊 Interference Results:")
        print(f"   Single token accuracy: {interference_results['single_token_accuracy']*100:.2f}%")
        print(f"   Context accuracy: {interference_results['context_accuracy']*100:.2f}%")
        print(f"   Interference effect: {interference_results['interference_effect']*100:+.2f}%")

    # Experiment 8: Brain Activity Modeling
    if args.experiment in ['all', 'modeling']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 8: Brain Activity Modeling")
        print(f"{'='*80}\n")

        brain_predictor = BrainActivityPredictor(model, args.device)
        activation_patterns = brain_predictor.extract_activation_patterns(test_loader, args.num_batches)
        hypothesis_test = brain_predictor.compare_with_brain_hypothesis(activation_patterns)

        results['activation_patterns'] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in activation_patterns.items()
        }
        results['brain_hypothesis_test'] = hypothesis_test

        print("\n📊 Brain Activity Analysis:")
        print(f"  Most selective block: {hypothesis_test.get('most_selective_block', 'N/A')}")
        print(f"  Supports selectivity hypothesis: {hypothesis_test.get('supports_selectivity_hypothesis', False)}")

        for key, value in hypothesis_test.items():
            if key.endswith('_breadth'):
                print(f"  {key}: {value:.4f}")

    # Experiment 9: Layer Importance Analysis
    if args.experiment in ['all', 'layer_importance']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 9: Layer Importance Analysis")
        print(f"{'='*80}\n")

        layer_analyzer = LayerImportanceAnalyzer(model, args.device)
        layer_results = layer_analyzer.analyze_layer_importance(test_loader, args.num_batches)
        results['layer_importance_analysis'] = layer_results

        print("\n📊 Layer Importance Results:")
        baseline = layer_results['baseline']
        print(f"   Baseline: loss={baseline['loss']:.4f}, acc={baseline['accuracy']*100:.2f}%")

        if 'importance_ranking' in layer_results:
            print(f"\n   Top 5 Most Critical Components:")
            for i, comp_data in enumerate(layer_results['importance_ranking'][:5]):
                comp = comp_data['component']
                score = comp_data['importance_score']
                print(f"     {i+1}. {comp}: {score*100:.2f}% drop")

    # Experiment 10: Gate Specificity Tests
    if args.experiment in ['all', 'gate_specificity']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 10: Gate Specificity Tests")
        print(f"{'='*80}\n")

        opto_sim = OptogeneticsSimulator(model, args.device)
        gate_spec_results = opto_sim.test_gate_specificity(test_loader, args.num_batches)
        results['gate_specificity'] = gate_spec_results

        print("\n📊 Gate Specificity Results:")
        baseline = gate_spec_results['baseline']
        print(f"   Baseline: loss={baseline['loss']:.4f}, acc={baseline['accuracy']*100:.2f}%")

        if 'anti_gate' in gate_spec_results:
            anti_drop = gate_spec_results['anti_gate']['accuracy_drop'] * 100
            print(f"   Anti-Gate accuracy drop: {anti_drop:.2f}%")

        if 'uniform_gate' in gate_spec_results:
            uniform_drop = gate_spec_results['uniform_gate']['accuracy_drop'] * 100
            print(f"   Uniform Gate accuracy drop: {uniform_drop:.2f}%")

    # Experiment 11: Layerwise Activity Analysis
    if args.experiment in ['all', 'layerwise_activity']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 11: Layerwise Activity Analysis")
        print(f"{'='*80}\n")

        layerwise_analyzer = LayerwiseActivityAnalyzer(model, args.device)
        layerwise_results = layerwise_analyzer.analyze_layerwise_activity(test_loader, args.num_batches)
        results['layerwise_activity_analysis'] = layerwise_results

        print("\n📊 Layerwise Activity Results:")
        if 'mean_activity_per_block' in layerwise_results:
            mean_acts = layerwise_results['mean_activity_per_block']
            print(f"   Activity per block:")
            for i, act in enumerate(mean_acts):
                print(f"     Block {i}: {act:.4f}")

        if 'accumulation_pattern' in layerwise_results:
            pattern = layerwise_results['accumulation_pattern']
            print(f"\n   Accumulation Pattern:")
            print(f"     Early activity: {pattern['early_activity']:.4f}")
            print(f"     Late activity: {pattern['late_activity']:.4f}")
            print(f"     Supports hypothesis: {pattern['supports_accumulation_hypothesis']}")

    # Experiment 12: Gate Entropy Analysis
    if args.experiment in ['all', 'gate_entropy']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 12: Gate Entropy Analysis")
        print(f"{'='*80}\n")

        entropy_analyzer = GateEntropyAnalyzer(model, args.device)
        entropy_results = entropy_analyzer.analyze_gate_entropy(test_loader, args.num_batches)
        results['gate_entropy_analysis'] = entropy_results

        print("\n📊 Gate Entropy Results:")
        for category in ['easy_tokens', 'medium_tokens', 'hard_tokens']:
            if category in entropy_results:
                data = entropy_results[category]
                print(f"   {category.replace('_', ' ').title()}:")
                print(f"     Mean entropy: {data['mean_entropy']:.4f}")
                print(f"     Count: {data['count']}")

        if 'supports_confidence_hypothesis' in entropy_results:
            print(f"\n   Supports confidence hypothesis: {entropy_results['supports_confidence_hypothesis']}")
            if 'entropy_difference' in entropy_results:
                print(f"   Entropy difference (Hard - Easy): {entropy_results['entropy_difference']:.4f}")

    # Save results
    results_file = output_dir / 'results.json'

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            # Handle numpy scalar types (bool_, int64, float64, etc.)
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")

    # Visualize
    print("\n📊 Generating visualizations...")
    visualize_results(results, output_dir)

    print(f"\n{'='*80}")
    print("✅ Analysis Complete!")
    print(f"   Results: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
