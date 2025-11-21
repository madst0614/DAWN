"""
DAWN Checkpoint Comprehensive Analysis
체크포인트 상세 분석 스크립트

분석 항목:
1. 활성화 패턴 분석 (희소성, 뉴런 사용률)
2. 뉴런 특화도 분석 (Dead neurons, 균등 사용)
3. Attention Weights 분석 (거리, 패턴)
4. 레이어별 표현 변화 (Norm, 유사도)
5. 패턴 템플릿 분석 (학습된 패턴)
6. Rank 효율성 분석 (Low-rank 효과)
7. 학습 곡선 분석 (추세, 예측)
8. 토큰 예측 품질 (잘/못 맞추는 토큰)
9. 시각화 종합

Usage:
    python scripts/analyze_dawn.py --checkpoint path/to/checkpoint.pt --data path/to/data
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import re
from scipy import stats

from models.model import DAWN
from transformers import BertTokenizer
from utils.data import CacheLoader, TextDataset, collate_fn_dynamic_padding, apply_mlm_masking
from torch.utils.data import DataLoader
from functools import partial


# ============================================================
# 1. 활성화 패턴 분석
# ============================================================

def analyze_activation_patterns(model, dataloader, num_batches=10):
    """
    활성화 패턴 상세 분석
    - Sparsity (희소성)
    - 뉴런별 사용률
    - 레이어별 통계
    """
    all_layer_stats = []

    print("\n" + "="*70)
    print("1. ACTIVATION PATTERN ANALYSIS")
    print("="*70)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Analyzing activations")):
            if batch_idx >= num_batches:
                break

            input_ids = batch['input_ids'].cuda()
            logits, all_activations = model(input_ids, return_activations=True)

            for layer_idx, acts in enumerate(all_activations):
                input_acts = acts['input_activations']  # [B, S, N_in]
                relational_acts = acts['relational_activations']
                enriched_acts = acts['enriched_activations']
                process_acts = acts['process_activations']  # [B, S, N_proc]

                stats_dict = {
                    'layer': layer_idx,
                    'batch': batch_idx,

                    # InputNeurons
                    'input_mean': input_acts.mean().item(),
                    'input_std': input_acts.std().item(),
                    'input_sparsity_01': (input_acts < 0.1).float().mean().item(),
                    'input_sparsity_05': (input_acts < 0.5).float().mean().item(),
                    'input_active_05': (input_acts > 0.5).float().mean().item(),
                    'input_active_08': (input_acts > 0.8).float().mean().item(),

                    # Relational
                    'relational_mean': relational_acts.mean().item(),
                    'relational_std': relational_acts.std().item(),

                    # ProcessNeurons
                    'process_mean': process_acts.mean().item(),
                    'process_std': process_acts.std().item(),
                    'process_sparsity_01': (process_acts < 0.1).float().mean().item(),
                    'process_sparsity_05': (process_acts < 0.5).float().mean().item(),
                    'process_active_05': (process_acts > 0.5).float().mean().item(),
                    'process_active_08': (process_acts > 0.8).float().mean().item(),
                }

                all_layer_stats.append(stats_dict)

    # DataFrame으로 변환
    df = pd.DataFrame(all_layer_stats)

    # 레이어별 평균
    layer_summary = df.groupby('layer').mean()

    print("\nInputNeurons (per layer):")
    print(layer_summary[['input_mean', 'input_std', 'input_active_05', 'input_active_08']])

    print("\nProcessNeurons (per layer):")
    print(layer_summary[['process_mean', 'process_std', 'process_active_05', 'process_active_08']])

    print("\nSparsity (per layer):")
    print(layer_summary[['input_sparsity_01', 'process_sparsity_01']])

    return df, layer_summary


# ============================================================
# 2. 뉴런 특화도 분석
# ============================================================

def analyze_neuron_specialization(model, dataloader, num_batches=50):
    """
    뉴런별 활성화율 분석
    - 각 뉴런이 얼마나 자주 활성화?
    - Dead neurons?
    - 균등하게 사용?
    """
    print("\n" + "="*70)
    print("2. NEURON SPECIALIZATION ANALYSIS")
    print("="*70)

    num_layers = len(model.layers)
    num_input = 64
    num_process = 128

    # 뉴런별 활성화 누적
    input_neuron_acts = [torch.zeros(num_input).cuda() for _ in range(num_layers)]
    process_neuron_acts = [torch.zeros(num_process).cuda() for _ in range(num_layers)]

    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Analyzing neurons")):
            if batch_idx >= num_batches:
                break

            input_ids = batch['input_ids'].cuda()
            B, S = input_ids.shape
            total_tokens += B * S

            _, all_activations = model(input_ids, return_activations=True)

            for layer_idx, acts in enumerate(all_activations):
                # [B, S, N] → 토큰별 활성화 (> 0.5) 평균
                input_acts = acts['input_activations']
                process_acts = acts['process_activations']

                # 뉴런별 평균 활성화
                input_active = (input_acts > 0.5).float().mean(dim=[0, 1])  # [N_in]
                process_active = (process_acts > 0.5).float().mean(dim=[0, 1])  # [N_proc]

                input_neuron_acts[layer_idx] += input_active
                process_neuron_acts[layer_idx] += process_active

    # 평균
    for layer_idx in range(num_layers):
        input_neuron_acts[layer_idx] /= num_batches
        process_neuron_acts[layer_idx] /= num_batches

    # 분석
    for layer_idx in range(num_layers):
        input_rates = input_neuron_acts[layer_idx].cpu().numpy()
        process_rates = process_neuron_acts[layer_idx].cpu().numpy()

        print(f"\nLayer {layer_idx}:")
        print(f"  InputNeurons (64):")
        print(f"    Mean activation: {input_rates.mean():.4f}")
        print(f"    Std: {input_rates.std():.4f}")
        print(f"    Max: {input_rates.max():.4f}")
        print(f"    Min: {input_rates.min():.4f}")
        print(f"    Dead (< 0.01): {(input_rates < 0.01).sum()}/64")
        print(f"    Underused (< 0.1): {(input_rates < 0.1).sum()}/64")

        print(f"  ProcessNeurons (128):")
        print(f"    Mean activation: {process_rates.mean():.4f}")
        print(f"    Std: {process_rates.std():.4f}")
        print(f"    Max: {process_rates.max():.4f}")
        print(f"    Min: {process_rates.min():.4f}")
        print(f"    Dead (< 0.01): {(process_rates < 0.01).sum()}/128")
        print(f"    Underused (< 0.1): {(process_rates < 0.1).sum()}/128")

    return input_neuron_acts, process_neuron_acts


# ============================================================
# 3. Attention Weights 분석
# ============================================================

def analyze_attention_patterns(model, dataloader, num_samples=5):
    """
    Attention weights 시각화
    - 인접 토큰에 집중?
    - 장거리 의존성?
    - 레이어별 차이?
    """
    print("\n" + "="*70)
    print("3. ATTENTION PATTERN ANALYSIS")
    print("="*70)

    attention_patterns = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_samples, desc="Analyzing attention")):
            if batch_idx >= num_samples:
                break

            input_ids = batch['input_ids'][:1].cuda()  # 1개만
            _, all_activations = model(input_ids, return_activations=True)

            for layer_idx, acts in enumerate(all_activations):
                attn_weights = acts['attention_weights']  # [B=1, S, S]
                attn_weights = attn_weights[0].cpu().numpy()  # [S, S]

                attention_patterns.append({
                    'layer': layer_idx,
                    'sample': batch_idx,
                    'weights': attn_weights
                })

    # 평균 패턴 계산
    for layer_idx in range(6):
        layer_attn = [p['weights'] for p in attention_patterns if p['layer'] == layer_idx]

        if layer_attn:
            # 평균 attention 거리
            avg_attn = np.mean(layer_attn, axis=0)
            seq_len = avg_attn.shape[0]

            # 각 토큰이 평균적으로 몇 칸 떨어진 토큰을 보는가?
            distances = []
            for i in range(seq_len):
                if i > 0:  # causal이므로
                    attn_dist = avg_attn[i, :i] * np.arange(1, i+1)[::-1]
                    avg_distance = attn_dist.sum() if attn_dist.sum() > 0 else 0
                    distances.append(avg_distance)

            print(f"\nLayer {layer_idx}:")
            print(f"  Average attention distance: {np.mean(distances):.2f} tokens")
            print(f"  Max attention distance: {np.max(distances):.2f} tokens")

            # 인접 토큰 집중도 (1-2 토큰 거리)
            adjacent_focus = []
            for i in range(1, seq_len):
                if i >= 2:
                    adjacent = avg_attn[i, i-2:i].sum()
                    adjacent_focus.append(adjacent)
            print(f"  Adjacent focus (1-2 tokens): {np.mean(adjacent_focus):.4f}")

    return attention_patterns


# ============================================================
# 4. 레이어별 표현 변화
# ============================================================

def analyze_layer_representations(model, dataloader, num_samples=10):
    """
    레이어별 hidden state 분석
    - Norm 변화
    - 코사인 유사도
    - 정보 흐름
    """
    print("\n" + "="*70)
    print("4. LAYER REPRESENTATION ANALYSIS")
    print("="*70)

    layer_norms = []
    layer_similarities = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_samples, desc="Analyzing layers")):
            if batch_idx >= num_samples:
                break

            input_ids = batch['input_ids'].cuda()

            # 각 레이어 출력 저장
            layer_outputs = []
            x = model.token_embedding(input_ids)
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            pos_emb = model.position_embedding(positions)
            x = x + pos_emb
            x = model.embedding_dropout(x)

            layer_outputs.append(x.clone())

            for layer in model.layers:
                x, _ = layer(x)
                layer_outputs.append(x.clone())

            # 분석
            for i in range(len(layer_outputs)):
                norm = layer_outputs[i].norm(dim=-1).mean().item()
                layer_norms.append({'layer': i, 'norm': norm})

                if i > 0:
                    # 이전 레이어와 코사인 유사도
                    prev = layer_outputs[i-1].flatten(0, 1)  # [B*S, H]
                    curr = layer_outputs[i].flatten(0, 1)

                    cos_sim = F.cosine_similarity(prev, curr, dim=-1).mean().item()
                    layer_similarities.append({
                        'from_layer': i-1,
                        'to_layer': i,
                        'similarity': cos_sim
                    })

    df_norms = pd.DataFrame(layer_norms)
    df_sims = pd.DataFrame(layer_similarities)

    print("\nNorm per layer:")
    print(df_norms.groupby('layer').mean())

    print("\nCosine similarity (layer → layer+1):")
    print(df_sims.groupby(['from_layer', 'to_layer']).mean())

    return df_norms, df_sims


# ============================================================
# 5. 패턴 템플릿 분석
# ============================================================

def analyze_pattern_templates(model):
    """
    학습된 패턴 템플릿 분석
    - InputNeurons의 patterns
    - 뉴런 간 유사도
    - 클러스터링
    """
    print("\n" + "="*70)
    print("5. PATTERN TEMPLATE ANALYSIS")
    print("="*70)

    for layer_idx, layer in enumerate(model.layers):
        patterns = layer.input_neurons.patterns.data  # [64, 512]

        # 패턴 간 코사인 유사도
        patterns_norm = F.normalize(patterns, dim=-1)
        similarity = torch.matmul(patterns_norm, patterns_norm.t())  # [64, 64]

        # 대각선 제외
        similarity_off_diag = similarity.clone()
        similarity_off_diag.fill_diagonal_(0)

        print(f"\nLayer {layer_idx}:")
        print(f"  Pattern norm mean: {patterns.norm(dim=-1).mean():.4f}")
        print(f"  Pattern norm std: {patterns.norm(dim=-1).std():.4f}")
        print(f"  Inter-pattern similarity:")
        print(f"    Mean: {similarity_off_diag.mean():.4f}")
        print(f"    Max: {similarity_off_diag.max():.4f}")
        print(f"    Min: {similarity_off_diag.min():.4f}")

        # 유사한 패턴 쌍 찾기
        high_sim = (similarity_off_diag > 0.9).sum().item()
        print(f"  Highly similar pairs (> 0.9): {high_sim}/4032")


# ============================================================
# 6. Rank 효율성 분석
# ============================================================

def analyze_rank_efficiency(model):
    """
    Low-rank 분해의 효율성
    - Effective rank
    - 정보 손실
    """
    print("\n" + "="*70)
    print("6. RANK EFFICIENCY ANALYSIS")
    print("="*70)

    for layer_idx, layer in enumerate(model.layers):
        # InputNeurons adapt
        down = layer.input_neurons.neuron_adapt_down.data  # [64, 512, 16]
        up = layer.input_neurons.neuron_adapt_up.data  # [64, 16, 512]

        # 각 뉴런의 effective rank (샘플링)
        effective_ranks = []
        for n in range(0, 64, 8):  # 샘플링
            full_matrix = torch.matmul(down[n], up[n])  # [512, 512]
            U, S, V = torch.svd(full_matrix)

            # Effective rank (Shannon entropy)
            S_norm = S / S.sum()
            S_norm = S_norm[S_norm > 1e-10]
            entropy = -(S_norm * torch.log(S_norm)).sum()
            eff_rank = torch.exp(entropy).item()
            effective_ranks.append(eff_rank)

        print(f"\nLayer {layer_idx} InputNeurons:")
        print(f"  Nominal rank: 16")
        print(f"  Effective rank mean: {np.mean(effective_ranks):.2f}")
        print(f"  Effective rank std: {np.std(effective_ranks):.2f}")

        # ProcessNeurons (샘플링)
        down_proc = layer.process_neurons.down_proj.data  # [128, 512, 128]
        up_proc = layer.process_neurons.up_proj.data  # [128, 128, 512]

        effective_ranks_proc = []
        for n in range(0, 128, 16):  # 샘플링
            full_matrix = torch.matmul(down_proc[n], up_proc[n])
            U, S, V = torch.svd(full_matrix)
            S_norm = S / S.sum()
            S_norm = S_norm[S_norm > 1e-10]
            entropy = -(S_norm * torch.log(S_norm)).sum()
            eff_rank = torch.exp(entropy).item()
            effective_ranks_proc.append(eff_rank)

        print(f"  ProcessNeurons:")
        print(f"    Nominal rank: 128")
        print(f"    Effective rank mean: {np.mean(effective_ranks_proc):.2f}")
        print(f"    Effective rank std: {np.std(effective_ranks_proc):.2f}")


# ============================================================
# 7. 학습 곡선 분석
# ============================================================

def analyze_training_curves(log_path):
    """
    학습 로그 분석
    - Loss/Acc 추세
    - 예측
    - 과적합 여부
    """
    print("\n" + "="*70)
    print("7. TRAINING CURVE ANALYSIS")
    print("="*70)

    if not Path(log_path).exists():
        print(f"\n⚠️  Log file not found: {log_path}")
        return

    # 로그 파싱 (training_log.txt)
    epochs = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    with open(log_path, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Train Loss' in line:
                # 파싱
                match = re.search(r'Epoch (\d+)/\d+', line)
                if match:
                    epoch = int(match.group(1))
                    epochs.append(epoch)

                match = re.search(r'Train Loss: ([\d.]+)', line)
                if match:
                    train_losses.append(float(match.group(1)))

                match = re.search(r'Val Loss: ([\d.]+)', line)
                if match:
                    val_losses.append(float(match.group(1)))

                match = re.search(r'Train Acc: ([\d.]+)', line)
                if match:
                    train_accs.append(float(match.group(1)))

                match = re.search(r'Val Acc: ([\d.]+)', line)
                if match:
                    val_accs.append(float(match.group(1)))

    if not epochs:
        print("\n⚠️  No training data found in log")
        return

    # 추세
    slope_loss, _, _, _, _ = stats.linregress(epochs, val_losses)
    slope_acc, _, _, _, _ = stats.linregress(epochs, val_accs)

    print(f"\nCurrent epoch: {epochs[-1]}")
    print(f"Val Loss: {val_losses[-1]:.4f}")
    print(f"Val Acc: {val_accs[-1]:.4f}")

    print(f"\nTrends:")
    print(f"  Loss slope: {slope_loss:.6f} per epoch")
    print(f"  Acc slope: {slope_acc:.6f} per epoch")

    # Epoch 30 예측
    pred_loss_30 = val_losses[-1] + slope_loss * (30 - epochs[-1])
    pred_acc_30 = val_accs[-1] + slope_acc * (30 - epochs[-1])

    print(f"\nPredicted at epoch 30:")
    print(f"  Loss: {pred_loss_30:.4f}")
    print(f"  Acc: {pred_acc_30:.4f}")

    # 과적합
    gap = np.array(train_losses) - np.array(val_losses)
    print(f"\nOverfitting check:")
    print(f"  Train-Val gap: {gap[-1]:.4f}")
    print(f"  Gap trend: {gap[-1] - gap[0]:.4f}")
    if gap[-1] > 0.5:
        print("  ⚠️  Warning: Large train-val gap")
    else:
        print("  ✓ Healthy learning")


# ============================================================
# 8. 토큰 예측 품질
# ============================================================

def analyze_prediction_quality(model, dataloader, tokenizer, num_samples=100):
    """
    예측 품질 분석
    - 자주 맞추는 토큰?
    - 자주 틀리는 토큰?
    """
    print("\n" + "="*70)
    print("8. PREDICTION QUALITY ANALYSIS")
    print("="*70)

    correct_tokens = {}
    incorrect_tokens = {}
    total_per_token = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_samples, desc="Analyzing predictions")):
            if batch_idx >= num_samples:
                break

            input_ids = batch['input_ids'].cuda()

            # Apply MLM masking on the fly
            masked_input_ids, targets = apply_mlm_masking(input_ids.clone(), tokenizer)

            logits = model(masked_input_ids)
            preds = logits.argmax(dim=-1)

            # 마스킹된 위치만
            mask = targets != -100

            correct = (preds == targets) & mask

            for i in range(input_ids.shape[0]):
                for j in range(input_ids.shape[1]):
                    if mask[i, j]:
                        token_id = targets[i, j].item()

                        total_per_token[token_id] = total_per_token.get(token_id, 0) + 1

                        if correct[i, j]:
                            correct_tokens[token_id] = correct_tokens.get(token_id, 0) + 1
                        else:
                            incorrect_tokens[token_id] = incorrect_tokens.get(token_id, 0) + 1

    # 정확도 계산
    token_accuracies = {}
    for token_id, total in total_per_token.items():
        correct_count = correct_tokens.get(token_id, 0)
        token_accuracies[token_id] = correct_count / total

    # Top/Bottom 토큰
    sorted_tokens = sorted(token_accuracies.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 20 most accurate tokens:")
    for token_id, acc in sorted_tokens[:20]:
        token_str = tokenizer.decode([token_id])
        count = total_per_token[token_id]
        print(f"  '{token_str}': {acc:.4f} ({count} samples)")

    print("\nBottom 20 least accurate tokens:")
    for token_id, acc in sorted_tokens[-20:]:
        token_str = tokenizer.decode([token_id])
        count = total_per_token[token_id]
        print(f"  '{token_str}': {acc:.4f} ({count} samples)")

    return token_accuracies


# ============================================================
# 9. 시각화 종합
# ============================================================

def create_visualizations(input_acts, process_acts, attn_patterns, df_norms, output_path='dawn_analysis.png'):
    """
    종합 시각화
    """
    print("\n" + "="*70)
    print("9. CREATING VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 뉴런 활성화율 (Layer 0)
    ax = axes[0, 0]
    ax.bar(range(64), input_acts[0].cpu().numpy())
    ax.set_title('InputNeuron Activation (Layer 0)')
    ax.set_xlabel('Neuron ID')
    ax.set_ylabel('Activation Rate')

    # 2. ProcessNeuron 활성화율 (Layer 0)
    ax = axes[0, 1]
    ax.bar(range(128), process_acts[0].cpu().numpy())
    ax.set_title('ProcessNeuron Activation (Layer 0)')
    ax.set_xlabel('Neuron ID')
    ax.set_ylabel('Activation Rate')

    # 3. Attention heatmap (Layer 0, Sample 0)
    ax = axes[0, 2]
    attn = [p for p in attn_patterns if p['layer'] == 0 and p['sample'] == 0]
    if attn:
        attn_data = attn[0]['weights']
        size = min(50, attn_data.shape[0])
        sns.heatmap(attn_data[:size, :size], ax=ax, cmap='viridis')
        ax.set_title('Attention Weights (Layer 0)')

    # 4. 레이어별 평균 활성화
    ax = axes[1, 0]
    input_means = [input_acts[i].mean().item() for i in range(6)]
    process_means = [process_acts[i].mean().item() for i in range(6)]
    ax.plot(input_means, 'o-', label='InputNeurons')
    ax.plot(process_means, 's-', label='ProcessNeurons')
    ax.set_title('Mean Activation per Layer')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Activation')
    ax.legend()

    # 5. Norm per layer
    ax = axes[1, 1]
    layer_norms = df_norms.groupby('layer')['norm'].mean()
    ax.plot(layer_norms.values, 'o-')
    ax.set_title('Hidden State Norm per Layer')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Norm')

    # 6. Dead neurons
    ax = axes[1, 2]
    dead_input = [(input_acts[i] < 0.01).sum().item() for i in range(6)]
    dead_process = [(process_acts[i] < 0.01).sum().item() for i in range(6)]
    ax.plot(dead_input, 'o-', label='InputNeurons')
    ax.plot(dead_process, 's-', label='ProcessNeurons')
    ax.set_title('Dead Neurons per Layer')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Count')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization: {output_path}")


# ============================================================
# Main Analysis Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN Checkpoint Comprehensive Analysis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--data', type=str, default=None, help='Path to validation data (optional)')
    parser.add_argument('--num-batches', type=int, default=20, help='Number of batches for analysis')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for visualizations')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("DAWN COMPREHENSIVE CHECKPOINT ANALYSIS")
    print("="*70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Data: {args.data}")

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint)

    # Load model
    print("Loading model...")
    model = DAWN(
        vocab_size=checkpoint.get('vocab_size', 30522),
        hidden_dim=512,
        num_layers=6,
        num_input_neurons=64,
        num_process_neurons=128,
        adapt_rank=16,
        process_rank=128,
        max_seq_len=2048,
        dropout=0.1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    print("✓ Model loaded")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("✓ Tokenizer loaded")

    # Load data
    print("\nLoading validation data...")

    if args.data:
        # Load from provided path
        import pickle
        with open(args.data, 'rb') as f:
            val_texts = pickle.load(f)
        print(f"✓ Loaded {len(val_texts)} texts from {args.data}")
    else:
        # Use cached data
        val_texts = CacheLoader.load_validation_texts(dataset="wikitext")
        if val_texts is None:
            print("❌ Failed to load validation data from cache")
            print("   Please provide --data argument with path to validation data")
            return

    # Create dataset and dataloader
    val_dataset = TextDataset(val_texts, tokenizer, max_length=128)
    collate_fn = partial(collate_fn_dynamic_padding, tokenizer=tokenizer)
    dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    print(f"✓ Created dataloader with {len(val_dataset)} samples")

    # Run all analyses
    print("\nRunning comprehensive analysis...")

    # 1. 활성화 패턴 분석
    df_acts, summary_acts = analyze_activation_patterns(model, dataloader, num_batches=args.num_batches)

    # 2. 뉴런 특화도 분석
    input_acts, process_acts = analyze_neuron_specialization(model, dataloader, num_batches=args.num_batches)

    # 3. Attention 패턴
    attn_patterns = analyze_attention_patterns(model, dataloader, num_samples=10)

    # 4. 레이어 표현
    df_norms, df_sims = analyze_layer_representations(model, dataloader, num_samples=20)

    # 5. 패턴 템플릿
    analyze_pattern_templates(model)

    # 6. Rank 효율성
    analyze_rank_efficiency(model)

    # 7. 학습 곡선
    log_path = Path(args.checkpoint).parent / 'training_log.txt'
    analyze_training_curves(str(log_path))

    # 8. 예측 품질
    token_accs = analyze_prediction_quality(model, dataloader, tokenizer, num_samples=100)

    # 9. 시각화
    output_path = Path(args.output_dir) / 'dawn_analysis.png'
    create_visualizations(input_acts, process_acts, attn_patterns, df_norms, output_path=str(output_path))

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
