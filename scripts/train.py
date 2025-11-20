"""
DAWN (Dynamic Architecture With Neurons) Training Script

Usage:
    # 기본 학습 (configs/train_config.yaml 사용)
    python scripts/train.py

    # 커스텀 config 파일 사용
    python scripts/train.py --config configs/my_config.yaml
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
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import time
import numpy as np
import math

from models.model import HierarchicalLanguageModel, debug_logger
from utils.training import CheckpointManager, TrainingMonitor, count_parameters, format_time

# MLM 마스킹 설정
MLM_CONFIG = {
    "mask_prob": 0.15,
    "mask_token_ratio": 0.8,
    "random_token_ratio": 0.5
}


# ============================================================
# Routing Auxiliary Loss Functions
# ============================================================

def compute_routing_aux_loss(
    weights: torch.Tensor,
    indices: torch.Tensor,
    n_neurons: int
) -> dict:
    """
    Compute auxiliary losses for routing.

    Args:
        weights: Routing weights [batch, n_neurons]
        indices: Selected indices [batch, k]
        n_neurons: Total number of neurons

    Returns:
        Dictionary of auxiliary losses
    """
    batch_size = weights.shape[0]

    # 1. Entropy loss (encourage diversity)
    # Normalized to [0, 1]: 0 = max entropy (good), 1 = min entropy (bad)
    entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean()
    max_entropy = math.log(n_neurons)
    entropy_loss = 1.0 - (entropy / max_entropy)  # Low = good diversity

    # 2. Usage loss (encourage all neurons to be used)
    # Average usage across batch
    avg_usage = weights.mean(dim=0)  # [n_neurons]

    # Coefficient of variation (std / mean)
    usage_std = avg_usage.std()
    usage_mean = avg_usage.mean()
    cv = usage_std / (usage_mean + 1e-8)

    # We want low coefficient of variation (uniform usage)
    usage_loss = cv

    # 3. Weight variance loss (within selected neurons)
    # Encourage selected neurons to have meaningful weights
    selected_weights = torch.gather(weights, 1, indices)
    weight_variance = selected_weights.var(dim=-1).mean()
    # Normalize: high variance is good, so invert
    # Approximate max variance for k selected from uniform is ~0.25
    variance_loss = torch.clamp(0.25 - weight_variance, min=0.0) / 0.25  # [0, 1]

    # 4. Load balance loss (across batches)
    # Count how many times each neuron is selected
    neuron_usage = torch.zeros(n_neurons, device=weights.device)
    neuron_usage.scatter_add_(
        0,
        indices.flatten(),
        torch.ones_like(indices.flatten(), dtype=torch.float32)
    )

    # Normalize by batch size and k
    neuron_usage = neuron_usage / (batch_size * indices.shape[1])

    # Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    sorted_usage, _ = torch.sort(neuron_usage)
    n = len(sorted_usage)
    index = torch.arange(1, n + 1, device=weights.device, dtype=torch.float32)
    gini = (2 * (index * sorted_usage).sum()) / (n * sorted_usage.sum() + 1e-8) - (n + 1) / n

    load_balance_loss = gini

    return {
        'entropy': entropy_loss,
        'usage': usage_loss,
        'variance': variance_loss,
        'load_balance': load_balance_loss,
        'gini': gini.detach(),  # For monitoring
    }


def aggregate_aux_losses(
    aux_losses_list: list,
    weights: dict = None
) -> tuple:
    """
    Aggregate auxiliary losses from all layers.

    Args:
        aux_losses_list: List of aux loss dicts from each layer
        weights: Weights for each loss component

    Returns:
        total_aux_loss: Weighted sum
        aux_metrics: Individual loss values (for logging)
    """
    if weights is None:
        weights = {
            'entropy': 0.01,      # Diversity
            'usage': 0.01,        # Uniform usage
            'variance': 0.001,    # Meaningful weights
            'load_balance': 0.01, # Cross-batch balance
        }

    # Aggregate across layers
    total_aux = 0.0
    aux_metrics = {}

    for key in ['entropy', 'usage', 'variance', 'load_balance']:
        values = [aux[key] for aux in aux_losses_list]
        mean_value = sum(values) / len(values)

        total_aux = total_aux + weights[key] * mean_value
        aux_metrics[key] = mean_value.item()

    # Add Gini for monitoring
    gini_values = [aux['gini'] for aux in aux_losses_list]
    aux_metrics['gini'] = (sum(gini_values) / len(gini_values)).item()

    return total_aux, aux_metrics


# ============================================================
# Comprehensive Debugging Function
# ============================================================

def comprehensive_debug(
    step: int,
    model,
    input_ids,
    labels,
    logits,
    loss,
    aux_loss,
    optimizer,
    debug_first_n_steps: int = 10,
    log_file: str = None
):
    """
    종합 디버깅 함수 - 학습이 안 되는 원인을 찾기 위한 완전한 진단

    Args:
        step: 현재 스텝
        model: 모델
        input_ids: [B, S]
        labels: [B, S]
        logits: [B, S, V]
        loss: 스칼라
        aux_loss: 스칼라
        optimizer: 옵티마이저
        debug_first_n_steps: 처음 몇 스텝 디버깅할지
        log_file: 로그 파일 경로
    """

    if step > debug_first_n_steps:
        return

    # Redirect all output to file
    if log_file is None:
        return  # Skip debug output if no log file specified

    import sys
    from io import StringIO

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    print("\n" + "="*70)
    print(f"🔍 COMPREHENSIVE DEBUG - STEP {step}")
    print("="*70)

    B, S = input_ids.shape
    V = logits.shape[-1]
    device = input_ids.device

    # ============================================================
    # 1. 데이터 검증 (가장 중요!)
    # ============================================================
    print("\n📊 1. DATA VALIDATION")
    print("-" * 70)

    # 1.1 입력 통계
    print(f"\n[Input IDs]")
    print(f"  Shape: {input_ids.shape}")
    print(f"  Range: [{input_ids.min().item()}, {input_ids.max().item()}]")
    print(f"  Unique tokens: {input_ids.unique().numel()}")
    print(f"  First sequence (first 30): {input_ids[0, :30].tolist()}")

    # 1.2 라벨 통계 (핵심!)
    print(f"\n[Labels - CRITICAL]")
    print(f"  Shape: {labels.shape}")
    print(f"  Range: [{labels.min().item()}, {labels.max().item()}]")

    valid_mask = (labels != -100)
    masked_count = (labels == -100).sum().item()
    valid_count = valid_mask.sum().item()
    total_count = labels.numel()

    print(f"  Total tokens: {total_count}")
    print(f"  Masked (-100): {masked_count} ({masked_count/total_count*100:.2f}%)")
    print(f"  Valid tokens: {valid_count} ({valid_count/total_count*100:.2f}%)")
    print(f"  ⚠️  Masking ratio: {masked_count/total_count*100:.1f}%")

    if masked_count / total_count > 0.5:
        print(f"  🚨 WARNING: Over 50% masked! This is abnormal!")
    if masked_count / total_count > 0.9:
        print(f"  🚨🚨 CRITICAL: Over 90% masked! Training will fail!")

    # 첫 시퀀스의 라벨 패턴 분석
    print(f"\n[First Sequence Label Pattern]")
    first_labels = labels[0]
    first_valid = first_labels[first_labels != -100]

    print(f"  Valid positions (first 20): {valid_mask[0].nonzero().squeeze()[:20].tolist()}")
    print(f"  First 30 labels: {labels[0, :30].tolist()}")
    if len(first_valid) > 0:
        print(f"  Valid label values (first 20): {first_valid[:20].tolist()}")

    # 라벨과 입력 비교
    print(f"\n[Input vs Labels Alignment]")
    print(f"  Input [0, :10]:  {input_ids[0, :10].tolist()}")
    print(f"  Labels[0, :10]:  {labels[0, :10].tolist()}")
    print(f"  Input [0, -10:]: {input_ids[0, -10:].tolist()}")
    print(f"  Labels[0, -10:]: {labels[0, -10:].tolist()}")

    # ============================================================
    # 2. 모델 출력 검증
    # ============================================================
    print("\n📈 2. MODEL OUTPUT VALIDATION")
    print("-" * 70)

    # 2.1 Logits 통계
    print(f"\n[Logits]")
    print(f"  Shape: {logits.shape}")
    print(f"  Range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"  Mean: {logits.mean().item():.4f}")
    print(f"  Std: {logits.std().item():.4f}")

    # NaN/Inf 체크
    if torch.isnan(logits).any():
        print(f"  🚨 NaN detected in logits!")
        nan_count = torch.isnan(logits).sum().item()
        print(f"     NaN count: {nan_count}/{logits.numel()}")
    if torch.isinf(logits).any():
        print(f"  🚨 Inf detected in logits!")

    # 2.2 예측 분석
    print(f"\n[Predictions]")
    preds = logits.argmax(dim=-1)
    print(f"  Predicted tokens (first seq, first 30): {preds[0, :30].tolist()}")

    # 예측의 다양성
    unique_preds = preds.unique().numel()
    print(f"  Unique predictions: {unique_preds}/{V} ({unique_preds/V*100:.2f}%)")

    # 가장 자주 예측되는 토큰
    pred_counts = torch.bincount(preds.flatten(), minlength=V)
    top_preds = pred_counts.topk(10)
    print(f"  Top 10 predicted tokens:")
    for i, (count, token_id) in enumerate(zip(top_preds.values, top_preds.indices)):
        print(f"    #{i+1}: token {token_id.item()} (count: {count.item()})")

    # 2.3 정확도 계산 (valid token만)
    print(f"\n[Accuracy on Valid Tokens]")
    if valid_count > 0:
        correct = (preds == labels) & valid_mask
        acc = correct.sum().item() / valid_count
        print(f"  Overall: {acc*100:.2f}% ({correct.sum().item()}/{valid_count})")

        # 첫 시퀀스의 매칭
        if valid_mask[0].sum() > 0:
            first_preds = preds[0][valid_mask[0]]
            first_labels_valid = labels[0][valid_mask[0]]
            first_matches = (first_preds == first_labels_valid).sum().item()
            print(f"  First sequence: {first_matches}/{len(first_preds)} correct")
            print(f"    Predicted: {first_preds[:10].tolist()}")
            print(f"    Actual:    {first_labels_valid[:10].tolist()}")
    else:
        print(f"  ⚠️  No valid tokens to compute accuracy!")

    # ============================================================
    # 3. Loss 분석
    # ============================================================
    print("\n💰 3. LOSS ANALYSIS")
    print("-" * 70)

    print(f"\n[Main Loss]")
    print(f"  Value: {loss.item():.4f}")
    print(f"  Expected range: 0-10 (vocab ~30k)")

    # Per-token loss 계산
    if valid_count > 0:
        # Flatten and compute per-token loss
        logits_flat = logits.view(-1, V)
        labels_flat = labels.view(-1)

        per_token_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=-100,
            reduction='none'
        )

        # Valid token의 loss만
        valid_losses = per_token_loss[labels_flat != -100]

        print(f"\n[Per-Token Loss Statistics]")
        print(f"  Min: {valid_losses.min().item():.4f}")
        print(f"  Max: {valid_losses.max().item():.4f}")
        print(f"  Mean: {valid_losses.mean().item():.4f}")
        print(f"  Median: {valid_losses.median().item():.4f}")
        print(f"  Std: {valid_losses.std().item():.4f}")

        # Loss 분포
        print(f"\n[Loss Distribution]")
        print(f"  <1.0:  {(valid_losses < 1.0).sum().item()} tokens")
        print(f"  1-5:   {((valid_losses >= 1.0) & (valid_losses < 5.0)).sum().item()} tokens")
        print(f"  5-10:  {((valid_losses >= 5.0) & (valid_losses < 10.0)).sum().item()} tokens")
        print(f"  >10:   {(valid_losses >= 10.0).sum().item()} tokens")

        # 첫 10개 valid token의 loss
        first_seq_valid_idx = valid_mask[0].nonzero().squeeze()
        if len(first_seq_valid_idx) > 0:
            first_10_idx = first_seq_valid_idx[:10]
            print(f"\n[First Sequence - First 10 Valid Tokens]")
            for i, idx in enumerate(first_10_idx):
                idx_item = idx.item()
                token_id = labels[0, idx_item].item()
                pred_id = preds[0, idx_item].item()
                token_loss = per_token_loss[idx_item].item()
                match = "✓" if token_id == pred_id else "✗"
                print(f"    Pos {idx_item}: label={token_id}, pred={pred_id}, loss={token_loss:.3f} {match}")

    print(f"\n[Auxiliary Loss]")
    print(f"  Value: {aux_loss.item():.6f}")
    print(f"  Expected range: 0-2")

    # ============================================================
    # 4. 종합 진단
    # ============================================================
    print("\n🏥 4. DIAGNOSTIC SUMMARY")
    print("-" * 70)

    issues = []
    warnings = []

    # 체크리스트
    if masked_count / total_count > 0.9:
        issues.append("🚨 CRITICAL: >90% tokens masked - training impossible")
    elif masked_count / total_count > 0.5:
        warnings.append("⚠️  >50% tokens masked - training inefficient")

    if valid_count == 0:
        issues.append("🚨 CRITICAL: No valid tokens - cannot compute loss")

    if torch.isnan(logits).any():
        issues.append("🚨 CRITICAL: NaN in logits")

    if loss.item() > 12:
        warnings.append(f"⚠️  Very high loss: {loss.item():.2f}")

    if valid_count > 0:
        correct = (preds == labels) & valid_mask
        acc = correct.sum().item() / valid_count
        if acc < 0.001:
            warnings.append(f"⚠️  Near-zero accuracy: {acc*100:.3f}%")

    # 출력
    if issues:
        print("\n🚨 CRITICAL ISSUES:")
        for issue in issues:
            print(f"  {issue}")

    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")

    if not issues and not warnings:
        print("\n✅ No major issues detected")

    # 추천 액션
    print("\n💡 RECOMMENDED ACTIONS:")
    if masked_count / total_count > 0.5:
        print("  1. Check data preprocessing - fix masking ratio")
        print("     → Look for collate_fn or Dataset code")
        print("     → Should mask 0% (CLM) or ~15% (MLM)")

    if aux_loss.item() < 0.001:
        print("  2. Increase aux_loss_weight:")
        print("     → Currently too small, increase to 0.01 or 0.1")

    print("\n" + "="*70)
    print()

    # Restore stdout and write to file
    captured = sys.stdout.getvalue()
    sys.stdout = old_stdout

    if log_file:
        with open(log_file, 'a') as f:
            f.write(captured)


# ============================================================
# Deep Learning Analysis Function
# ============================================================

def deep_learning_analysis(model, x, labels, step, debug_first_n_steps=10, log_file=None):
    """
    Simplified learning analysis for new DAWN architecture.

    Args:
        model: DAWNLanguageModel
        x: Input token IDs [B, S]
        labels: Labels [B, S]
        step: Current step
        debug_first_n_steps: Debug first N steps only
        log_file: Log file path for redirecting output
    """
    if step > debug_first_n_steps:
        return

    if log_file is None:
        return

    import sys
    from io import StringIO
    from collections import Counter

    old_stdout = sys.stdout
    sys.stdout = StringIO()

    print(f"\n{'='*70}")
    print(f"🔬 DEEP LEARNING ANALYSIS - Step {step}")
    print(f"{'='*70}")

    with torch.no_grad():
        B, S = x.shape
        token_emb = model.token_embedding(x)
        positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = model.position_embedding(positions)
        x_emb = token_emb + pos_emb

        print(f"\n[Embedding Layer]")
        print(f"  Token emb norm: {token_emb.norm(dim=-1).mean():.4f}")
        print(f"  Pos emb norm: {pos_emb.norm(dim=-1).mean():.4f}")
        print(f"  Combined std: {x_emb.std():.4f}")

    # Forward and backward for gradient analysis
    model.zero_grad()
    output = model(x, labels=labels)
    loss = output['loss']
    loss.backward()

    print(f"\n[Loss]")
    print(f"  Value: {loss.item():.4f}")

    # Gradient analysis
    print(f"\n[Gradient Analysis]")
    for name, param in model.named_parameters():
        if param.grad is not None and 'router' in name:
            grad = param.grad
            print(f"  {name}: norm={grad.norm():.6f}, mean={grad.abs().mean():.6f}")

    # Logits analysis
    logits = output['logits']
    print(f"\n[Logits Distribution]")
    print(f"  Mean: {logits.mean():.4f}, Std: {logits.std():.4f}")
    print(f"  Range: [{logits.min():.4f}, {logits.max():.4f}]")

    # Prediction diversity
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        unique_preds = preds.unique().numel()
        print(f"\n[Prediction Diversity]")
        print(f"  Unique tokens: {unique_preds}/{model.vocab_size} ({unique_preds/model.vocab_size*100:.2f}%)")

    print(f"\n{'='*70}\n")

    captured = sys.stdout.getvalue()
    sys.stdout = old_stdout

    if log_file:
        with open(log_file, 'a') as f:
            f.write(captured)


# ============================================================
# MLM Masking Function
# ============================================================

def apply_mlm_masking(input_ids, tokenizer, config=None):
    """
    Apply MLM-style masking (80% [MASK], 10% random, 10% keep).
    Based on dawn/utils/data_utils.py MaskingStrategy.apply_mlm_masking

    Args:
        input_ids: [B, S] Token IDs to mask
        tokenizer: Tokenizer instance
        config: Optional config dict with mask_prob, mask_token_ratio, random_token_ratio

    Returns:
        Tuple of (masked_input_ids, labels)
    """
    if config is None:
        config = MLM_CONFIG

    labels = input_ids.clone()
    mask_prob = config.get("mask_prob", 0.15)
    device = input_ids.device

    probability_matrix = torch.full(labels.shape, mask_prob, device=device)

    # ✅ Exclude special tokens (CLS, SEP, PAD, etc.) - Dawn style
    # labels is [B, S], need to preserve batch dimension
    special_tokens_mask = []
    for seq in labels.tolist():  # Iterate over batch
        seq_mask = [
            tokenizer.get_special_tokens_mask([val], already_has_special_tokens=True)[0]
            for val in seq
        ]
        special_tokens_mask.append(seq_mask)
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # ✅ Exclude padding tokens (belt and suspenders)
    padding_mask = input_ids == tokenizer.pad_token_id
    probability_matrix.masked_fill_(padding_mask, value=0.0)

    # Sample masked positions
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # Apply masking strategy
    mask_ratio = config.get("mask_token_ratio", 0.8)
    random_ratio = config.get("random_token_ratio", 0.5)

    # 80% [MASK]
    indices_replaced = masked_indices & (torch.rand(labels.shape, device=device) < mask_ratio)
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% random (of remaining)
    indices_random = (
        masked_indices
        & ~indices_replaced
        & (torch.rand(labels.shape, device=device) < random_ratio)
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]

    # 10% keep original (implicit)
    return input_ids, labels


# ============================================================
# Dataset
# ============================================================

class TextDataset(Dataset):
    """Dataset for tokenized texts"""
    def __init__(self, texts, tokenizer, max_length=128):  # CHANGED: 512 → 128
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize (NO padding here - will be done dynamically in collate_fn)
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def collate_fn_dynamic_padding(batch, tokenizer):
    """
    Collate function with DYNAMIC padding (배치별 최대 길이만큼만 padding)

    큰 개선:
    - Before: 모든 시퀀스를 512로 padding → 90% padding!
    - After: 배치 내 최대 길이로만 padding → ~10-30% padding
    """
    # Find max length in this batch
    max_len = max(item['input_ids'].size(0) for item in batch)

    input_ids_list = []
    attention_mask_list = []

    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        seq_len = input_ids.size(0)

        # Pad to batch max length
        if seq_len < max_len:
            padding_len = max_len - seq_len
            input_ids = torch.cat([
                input_ids,
                torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_len, dtype=torch.long)
            ])

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list)
    }


def load_data(data_config, max_length=128, batch_size=128, tokenizer_path=None):
    """Load data from config paths with DYNAMIC padding"""
    from transformers import AutoTokenizer
    from functools import partial
    import pickle

    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load data from config paths
    base_dir = data_config['base_dir']
    train_path = os.path.join(base_dir, data_config['train_file'])
    val_path = os.path.join(base_dir, data_config['val_file'])

    print(f"Loading data from: {base_dir}")

    # Load train texts
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train data not found: {train_path}")
    with open(train_path, 'rb') as f:
        train_texts = pickle.load(f)

    # Load validation texts
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    with open(val_path, 'rb') as f:
        val_texts = pickle.load(f)

    print(f"Loaded {len(train_texts)} train texts, {len(val_texts)} val texts")

    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length)

    # Create collate function with tokenizer
    collate_fn = partial(collate_fn_dynamic_padding, tokenizer=tokenizer)

    # Create dataloaders with DYNAMIC padding
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn  # ← Dynamic padding!
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        collate_fn=collate_fn  # ← Dynamic padding!
    )

    return train_loader, val_loader, tokenizer


# ============================================================
# Training Functions
# ============================================================

def get_curriculum_k_values(epoch, total_epochs, n_input, n_process):
    """
    Curriculum learning: Dense → Sparse scheduling.

    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        n_input: Total input neurons
        n_process: Total process neurons

    Returns:
        k_input, k_process for this epoch
    """
    # Cosine schedule: start dense (75%), end sparse (25%)
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    sparsity = 0.5 * (1 + math.cos(math.pi * progress))

    k_input = int(n_input * (0.25 + 0.5 * sparsity))
    k_process = int(n_process * (0.25 + 0.5 * sparsity))

    return k_input, k_process


def update_temperature(model, epoch, total_epochs):
    """
    Temperature annealing: high (explore) → low (exploit).

    Args:
        model: DAWN model
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
    """
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    # Start: 2.0 (explore), End: 1.0 (mild exploit) - keep higher for diversity
    temperature = 2.0 * (1 - progress) + 1.0 * progress

    for layer in model.layers:
        layer.block.router.temperature.fill_(temperature)

    return temperature


def reset_routing_stats(model):
    """Placeholder - new architecture doesn't track routing stats."""
    pass


def print_diagnostic_metrics(model, epoch):
    """
    학습 진단 메트릭 출력 (simplified for new DAWN architecture)
    """
    print(f"\n{'='*60}")
    print(f"Diagnostic Metrics (Epoch {epoch})")
    print(f"{'='*60}")

    # Gradient norm
    grad_norm = 0.0
    grad_count = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
            grad_count += 1

    if grad_count > 0:
        grad_norm = grad_norm ** 0.5
        status = "✓ OK"
        if grad_norm < 0.01:
            status = "⚠ VANISHING"
        elif grad_norm > 100:
            status = "⚠ EXPLODING"
        print(f"Gradient Norm: {grad_norm:.4f} {status}")
        print(f"  Expected: 0.1 ~ 10 | < 0.01 = vanishing | > 100 = exploding")
    else:
        print(f"Gradient Norm: N/A (no gradients)")

    print(f"{'='*60}\n")


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args, scaler=None, tokenizer=None, log_file=None, debug_log_file=None):
    """Train for one epoch"""
    model.train()

    # Setup debug logger
    if debug_log_file:
        debug_logger.setup(debug_log_file, enabled=True)
        debug_logger.log("Train", f"\n{'='*60}\nEpoch {epoch} started\n{'='*60}")

    # Epoch 시작 시 routing 통계 초기화
    reset_routing_stats(model)

    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_valid_tokens = 0  # CRITICAL FIX: Track valid tokens only (labels != -100)
    total_gini = 0
    total_aux = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        # Update debug logger step
        global_step = (epoch - 1) * len(dataloader) + step
        debug_logger.set_step(global_step)

        input_ids = batch["input_ids"].to(device)

        # Apply MLM masking
        if tokenizer is not None:
            input_ids, labels = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)
        else:
            # Fallback: no masking
            labels = input_ids.clone()

        # Detailed debugging for first 10 steps of epoch 1
        debug_mode = (epoch == 1 and step < 10)

        # Capture debug output to file
        debug_output_buffer = None
        old_stdout = None
        if debug_mode and debug_log_file:
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            debug_output_buffer = StringIO()
            sys.stdout = debug_output_buffer

        if debug_mode:
            print(f"\n{'='*60}")
            print(f"Step {step + 1} Debugging")
            print(f"{'='*60}")

            # CRITICAL: Label analysis for first step
            if step == 0:
                print(f"\n[Label Analysis]")
                print(f"  Total tokens: {labels.numel()}")
                print(f"  Masked tokens (-100): {(labels == -100).sum().item()}")
                print(f"  Valid tokens: {(labels != -100).sum().item()}")
                print(f"  Masking ratio: {(labels == -100).sum().item() / labels.numel() * 100:.1f}%")

            print(f"\nBefore Forward:")
            print(f"  Input shape: {input_ids.shape}, range: [{input_ids.min()}, {input_ids.max()}]")
            print(f"  Model embedding norm: {model.token_embedding.weight.norm():.4f}")

        optimizer.zero_grad()

        # Mixed precision training
        # Aux weight configuration
        aux_weight = 0.01  # Overall aux loss weight

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    k_input=args.k_input,
                    k_process=args.k_process,
                    return_routing_info=True
                )
                loss = outputs['loss']
                logits = outputs['logits']

                # Compute new aux losses from routing info
                if 'routing_info' in outputs:
                    aux_losses_list = []
                    for layer_info in outputs['routing_info']:
                        layer_aux = compute_routing_aux_loss(
                            weights=layer_info['weights'],
                            indices=layer_info['indices'],
                            n_neurons=args.n_input
                        )
                        aux_losses_list.append(layer_aux)

                    aux_loss, aux_metrics = aggregate_aux_losses(aux_losses_list)
                    gini = aux_metrics.get('gini', 0.0)
                else:
                    # Fallback to legacy aux loss
                    model_aux = outputs['aux_loss']
                    aux_loss = model_aux['load_balance'] * 0.001 + model_aux['entropy'] * 0.1
                    gini = 0.0

                total_loss_combined = loss + aux_weight * aux_loss

            # NaN/Inf detection - STOP immediately
            if torch.isnan(total_loss_combined) or torch.isinf(total_loss_combined):
                nan_info = f"""
{'='*60}
NaN/Inf DETECTED - STOPPING TRAINING
{'='*60}
Epoch: {epoch}, Step: {step}
Loss: {loss.item() if not torch.isnan(loss) else 'NaN'}
Aux Loss: {aux_loss.item() if hasattr(aux_loss, 'item') and not torch.isnan(aux_loss) else aux_loss}
Total Loss: {total_loss_combined.item() if not torch.isnan(total_loss_combined) else 'NaN'}
Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]
Logits has NaN: {torch.isnan(logits).any().item()}
Logits has Inf: {torch.isinf(logits).any().item()}

Router weights check:
"""
                for name, param in model.named_parameters():
                    if 'router' in name:
                        has_nan = torch.isnan(param).any().item()
                        has_inf = torch.isinf(param).any().item()
                        nan_info += f"  {name}: NaN={has_nan}, Inf={has_inf}, norm={param.norm().item():.4f}\n"

                nan_info += f"{'='*60}\n"

                # Log to debug file
                if debug_log_file:
                    with open(debug_log_file, 'a') as f:
                        f.write(nan_info)

                raise RuntimeError(f"NaN/Inf detected at epoch {epoch}, step {step}. Check debug log for details.")

            if debug_mode:
                print(f"\nAfter Forward:")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                print(f"  Loss: {loss.item():.4f}, Aux Loss: {aux_loss.item() if hasattr(aux_loss, 'item') else aux_loss:.4f}")

            scaler.scale(total_loss_combined).backward()

            scaler.unscale_(optimizer)

            # Gradient clipping with verification
            grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Debug: log gradients (only if debug mode)
            if debug_log_file:
                debug_logger.log("Gradients", f"total_norm: {grad_norm_before:.4f}")
                if grad_norm_before > 10.0:
                    debug_logger.log("Gradients", f"⚠️ Gradient exploding! norm={grad_norm_before:.2f}")
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            g_norm = param.grad.norm().item()
                            if g_norm > 1.0:
                                has_nan = torch.isnan(param.grad).any().item()
                                has_inf = torch.isinf(param.grad).any().item()
                                debug_logger.log("Gradients", f"  {name}: norm={g_norm:.4f}, nan={has_nan}, inf={has_inf}")

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                k_input=args.k_input,
                k_process=args.k_process,
                return_routing_info=True
            )
            loss = outputs['loss']
            logits = outputs['logits']

            # Compute new aux losses from routing info
            if 'routing_info' in outputs:
                aux_losses_list = []
                for layer_info in outputs['routing_info']:
                    layer_aux = compute_routing_aux_loss(
                        weights=layer_info['weights'],
                        indices=layer_info['indices'],
                        n_neurons=args.n_input
                    )
                    aux_losses_list.append(layer_aux)

                aux_loss, aux_metrics = aggregate_aux_losses(aux_losses_list)
                gini = aux_metrics.get('gini', 0.0)
            else:
                # Fallback to legacy aux loss
                model_aux = outputs['aux_loss']
                aux_loss = model_aux['load_balance'] * 0.001 + model_aux['entropy'] * 0.1
                gini = 0.0

            total_loss_combined = loss + aux_weight * aux_loss

            # NaN/Inf detection - STOP immediately
            if torch.isnan(total_loss_combined) or torch.isinf(total_loss_combined):
                nan_info = f"""
{'='*60}
NaN/Inf DETECTED - STOPPING TRAINING
{'='*60}
Epoch: {epoch}, Step: {step}
Loss: {loss.item() if not torch.isnan(loss) else 'NaN'}
Aux Loss: {aux_loss.item() if hasattr(aux_loss, 'item') and not torch.isnan(aux_loss) else aux_loss}
Total Loss: {total_loss_combined.item() if not torch.isnan(total_loss_combined) else 'NaN'}
Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]
Logits has NaN: {torch.isnan(logits).any().item()}
Logits has Inf: {torch.isinf(logits).any().item()}

Router weights check:
"""
                for name, param in model.named_parameters():
                    if 'router' in name:
                        has_nan = torch.isnan(param).any().item()
                        has_inf = torch.isinf(param).any().item()
                        nan_info += f"  {name}: NaN={has_nan}, Inf={has_inf}, norm={param.norm().item():.4f}\n"

                nan_info += f"{'='*60}\n"

                # Log to debug file
                if debug_log_file:
                    with open(debug_log_file, 'a') as f:
                        f.write(nan_info)

                raise RuntimeError(f"NaN/Inf detected at epoch {epoch}, step {step}. Check debug log for details.")

            if debug_mode:
                print(f"\nAfter Forward:")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                print(f"  Loss: {loss.item():.4f}, Aux Loss: {aux_loss.item() if hasattr(aux_loss, 'item') else aux_loss:.4f}")

            total_loss_combined.backward()

            # Gradient clipping with verification
            grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Debug: log gradients (only if debug mode)
            if debug_log_file:
                debug_logger.log("Gradients", f"total_norm: {grad_norm_before:.4f}")
                if grad_norm_before > 10.0:
                    debug_logger.log("Gradients", f"⚠️ Gradient exploding! norm={grad_norm_before:.2f}")
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            g_norm = param.grad.norm().item()
                            if g_norm > 1.0:
                                has_nan = torch.isnan(param.grad).any().item()
                                has_inf = torch.isinf(param.grad).any().item()
                                debug_logger.log("Gradients", f"  {name}: norm={g_norm:.4f}, nan={has_nan}, inf={has_inf}")

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # ===== CRITICAL FIX: Accuracy calculation (only valid tokens) =====
        predictions = logits.argmax(dim=-1)  # [B, S]

        # Only count tokens that are not masked (-100)
        valid_mask = (labels != -100)  # [B, S]
        correct_predictions = (predictions == labels) & valid_mask

        correct = correct_predictions.sum().item()
        valid_tokens = valid_mask.sum().item()

        total_correct += correct
        total_valid_tokens += valid_tokens

        # Debug prediction analysis
        if debug_mode and step == 0:
            print(f"\n[Prediction Analysis - First Sequence]")
            seq_0_valid = valid_mask[0]
            seq_0_labels = labels[0][seq_0_valid]
            seq_0_preds = predictions[0][seq_0_valid]

            print(f"  Valid tokens in seq 0: {seq_0_valid.sum().item()}")
            print(f"  First 10 labels: {seq_0_labels[:10].cpu().tolist()}")
            print(f"  First 10 preds:  {seq_0_preds[:10].cpu().tolist()}")
            print(f"  Matches: {(seq_0_labels[:10] == seq_0_preds[:10]).sum().item()}/10")

        # Loss는 전체 토큰 수로 계산 (기존 유지)
        batch_size, seq_len = input_ids.shape
        num_tokens = batch_size * seq_len
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        aux_loss_val = aux_loss.item() if hasattr(aux_loss, 'item') else aux_loss

        # Accumulate routing metrics
        total_gini += gini
        total_aux += aux_loss_val
        num_batches += 1
        step_acc = correct / valid_tokens if valid_tokens > 0 else 0.0
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "aux": f"{aux_loss_val:.4f}",
            "gini": f"{gini:.3f}",
            "acc": f"{step_acc:.4f}"
        })

        # Debug: log step summary (every 10 steps to reduce log size)
        if debug_log_file and step % 10 == 0:
            debug_logger.log("Summary", f"loss={loss.item():.4f}, aux={aux_loss_val:.4f}, acc={step_acc:.4f}, grad_norm={grad_norm_before:.4f}")
            debug_logger.log("Summary", "-" * 40)

        # Log to file every step
        if log_file:
            with open(log_file, 'a') as f:
                acc_val = correct / valid_tokens if valid_tokens > 0 else 0.0
                f.write(f"epoch={epoch},step={step+1},loss={loss.item():.6f},"
                       f"aux_loss={aux_loss.item():.6f},weighted_aux={(aux_weight * aux_loss).item():.6f},"
                       f"acc={acc_val:.6f}\n")

        # Restore stdout and write debug output to file
        if debug_mode and debug_log_file and old_stdout is not None:
            captured = debug_output_buffer.getvalue()
            sys.stdout = old_stdout
            if captured:
                with open(debug_log_file, 'a') as f:
                    f.write(captured)

    avg_loss = total_loss / total_tokens
    avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
    avg_gini = total_gini / num_batches if num_batches > 0 else 0.0
    avg_aux = total_aux / num_batches if num_batches > 0 else 0.0

    routing_metrics = {
        'gini': avg_gini,
        'aux_loss': avg_aux
    }

    return avg_loss, avg_acc, routing_metrics


def evaluate(model, dataloader, device, args, tokenizer=None):
    """Evaluate model with MLM masking"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_valid_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)

            # Apply same MLM masking as training
            if tokenizer is not None:
                masked_input_ids, labels = apply_mlm_masking(input_ids, tokenizer)
            else:
                # Fallback: use all tokens (not recommended)
                masked_input_ids = input_ids
                labels = input_ids.clone()

            outputs = model(
                input_ids=masked_input_ids,
                labels=labels,
                k_input=args.k_input,
                k_process=args.k_process
            )
            loss = outputs['loss']
            logits = outputs['logits']

            # ===== CRITICAL FIX: Accuracy calculation (only valid tokens) =====
            predictions = logits.argmax(dim=-1)

            # Only count tokens that are not masked (-100)
            valid_mask = (labels != -100)
            correct_predictions = (predictions == labels) & valid_mask

            correct = correct_predictions.sum().item()
            valid_tokens = valid_mask.sum().item()

            total_correct += correct
            total_valid_tokens += valid_tokens

            batch_size, seq_len = input_ids.shape
            num_tokens = batch_size * seq_len
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
    return avg_loss, avg_acc


def load_config(config_path):
    """Load config from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train DAWN (Dynamic Architecture With Neurons)')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging to file')
    cli_args = parser.parse_args()

    # Load config
    config_path = Path(PROJECT_ROOT) / cli_args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)

    # Create args namespace from config
    class Args:
        pass
    args = Args()

    # Model
    args.d_model = cfg['model']['d_model']
    args.n_heads = cfg['model']['n_heads']
    args.n_layers = cfg['model']['n_layers']
    args.n_input = cfg['model']['n_input']
    args.n_process = cfg['model']['n_process']
    args.max_seq_len = cfg['model']['max_seq_len']
    args.dropout = cfg['model']['dropout']

    # Sparsity
    args.k_input = cfg['sparsity']['k_input']
    args.k_process = cfg['sparsity']['k_process']

    # Training
    args.batch_size = cfg['training']['batch_size']
    args.num_epochs = cfg['training']['num_epochs']
    args.lr = cfg['training']['lr']
    args.weight_decay = cfg['training']['weight_decay']
    args.warmup_epochs = cfg['training']['warmup_epochs']

    # Router
    args.router_lr_mult = cfg['router']['lr_multiplier']
    args.router_weight_decay = cfg['router']['weight_decay']

    # Other
    args.use_amp = cfg['use_amp']
    args.checkpoint_dir = cfg['checkpoint_dir']
    args.log_dir = cfg['log_dir']

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories with timestamp for each run
    base_checkpoint_dir = Path(args.checkpoint_dir)
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Find latest run folder and best checkpoint for auto-resume
    latest_best_checkpoint = None
    if not cli_args.resume:
        # Look for existing run folders
        run_folders = sorted([
            d for d in base_checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith('run_')
        ], reverse=True)

        if run_folders:
            latest_folder = run_folders[0]
            best_ckpt = latest_folder / 'best_model.pt'
            if best_ckpt.exists():
                latest_best_checkpoint = best_ckpt
                print(f"\nFound latest checkpoint: {latest_best_checkpoint}")

    # Create new run folder with Korean timestamp and random number
    import random
    from datetime import timezone, timedelta
    kst = timezone(timedelta(hours=9))
    timestamp = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
    random_suffix = random.randint(1000, 9999)
    run_name = f"run_{timestamp}_{random_suffix}"
    checkpoint_dir = base_checkpoint_dir / run_name
    log_dir = checkpoint_dir  # Same folder for simplicity
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run folder: {checkpoint_dir}")

    # Save config
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DAWN (Dynamic Architecture With Neurons) Training")
    print(f"{'='*60}")
    print(f"\nConfig file: {config_path}")
    print(f"\nModel: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"Neurons: n_input={args.n_input}, n_process={args.n_process}")
    print(f"Sparsity: k_input={args.k_input or 'auto'}, k_process={args.k_process or 'auto'}")
    print(f"Training: batch={args.batch_size}, epochs={args.num_epochs}, lr={args.lr}")

    # Load data
    print(f"\n{'='*60}")
    print("Loading data...")
    print(f"{'='*60}")
    train_loader, val_loader, tokenizer = load_data(
        data_config=cfg['data'],
        max_length=args.max_seq_len,
        batch_size=args.batch_size
    )

    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print(f"\n{'='*60}")
    print("Creating DAWN model...")
    print(f"{'='*60}")

    model = HierarchicalLanguageModel.from_config(cfg, vocab_size)
    model = model.to(device)

    # Model statistics
    stats = model.get_model_stats()
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"  Number of layers: {stats['n_layers']}")

    # Sparsity info (DAWN uses 50% by default)
    k_input_default = args.n_input // 2
    k_process_default = args.n_process // 2

    print(f"\nSparsity Configuration (default):")
    print(f"  Input neurons: {k_input_default}/{args.n_input} ({k_input_default/args.n_input*100:.1f}%)")
    print(f"  Process neurons: {k_process_default}/{args.n_process} ({k_process_default/args.n_process*100:.1f}%)")

    # Optimizer & Scheduler
    # Separate parameter groups: Router gets higher LR for faster learning
    router_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'router' in name or 'affinity_proj' in name:
            router_params.append(param)
        else:
            other_params.append(param)

    print(f"\nOptimizer parameter groups:")
    print(f"  Router params: {len(router_params)} tensors")
    print(f"  Other params: {len(other_params)} tensors")
    print(f"  Router LR: {args.lr * args.router_lr_mult:.2e} ({args.router_lr_mult}x base)")
    print(f"  Other LR: {args.lr:.2e}")

    optimizer = torch.optim.AdamW([
        {'params': router_params, 'lr': args.lr * args.router_lr_mult, 'weight_decay': args.router_weight_decay},
        {'params': other_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
    ],
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # Warmup + Cosine Annealing scheduler
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.num_epochs * len(train_loader)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=args.lr * 0.1
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
    if args.use_amp:
        print(f"\nUsing Automatic Mixed Precision (AMP)")

    # Resume from checkpoint if specified or auto-resume from latest best
    start_epoch = 1
    resume_checkpoint = None

    if cli_args.resume:
        resume_checkpoint = Path(cli_args.resume)
    elif latest_best_checkpoint:
        resume_checkpoint = latest_best_checkpoint

    if resume_checkpoint and resume_checkpoint.exists():
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"  Resumed from epoch {start_epoch - 1}")
        print(f"  Starting from epoch {start_epoch}")
    elif cli_args.resume:
        print(f"\nWarning: Checkpoint not found: {cli_args.resume}")
        print(f"Starting from scratch...")
    else:
        print(f"\nStarting fresh training (no previous checkpoint found)")

    # Checkpoint & Monitor
    ckpt_manager = CheckpointManager(str(checkpoint_dir), keep_best_n=3)
    monitor = TrainingMonitor(str(log_dir))

    # Training log file (append mode)
    training_log_file = checkpoint_dir / "training_log.txt"
    debug_log_file = checkpoint_dir / "debug_log.txt"

    # Write header to training log
    with open(training_log_file, 'w') as f:
        f.write("# Training Log\n")
        f.write("# Format: epoch,step,loss,aux_loss,weighted_aux,acc\n")

    # Write header to debug log
    with open(debug_log_file, 'w') as f:
        f.write("# Debug Log\n")
        f.write("# Contains detailed debugging information for first 10 steps of epoch 1\n\n")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"  Training log: {training_log_file}")
    print(f"  Debug log: {debug_log_file}")
    print(f"{'='*60}")
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start = time.time()

        # Curriculum learning: adjust k values (dense → sparse)
        k_input, k_process = get_curriculum_k_values(
            epoch, args.num_epochs, args.n_input, args.n_process
        )
        args.k_input = k_input
        args.k_process = k_process

        # Temperature annealing: update router temperature (explore → exploit)
        temperature = update_temperature(model, epoch, args.num_epochs)

        if epoch == 1 or epoch % 5 == 0:
            print(f"\nEpoch {epoch}: k_input={k_input}, k_process={k_process}, temp={temperature:.2f}")

        # Train
        train_loss, train_acc, routing_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args, scaler, tokenizer,
            log_file=str(training_log_file),
            debug_log_file=str(debug_log_file) if cli_args.debug else None
        )

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device, args, tokenizer)

        epoch_time = time.time() - epoch_start

        # Log
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'gini': routing_metrics['gini'],
            'aux_loss': routing_metrics['aux_loss'],
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        monitor.log_epoch(epoch, metrics)

        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Gini: {routing_metrics['gini']:.4f} | Aux Loss: {routing_metrics['aux_loss']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {format_time(epoch_time)}")

        # Print diagnostic metrics every 100 epochs (or first epoch)
        if epoch == 1 or epoch % 100 == 0:
            print_diagnostic_metrics(model, epoch)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  New best model! (val_loss: {best_val_loss:.4f})")

        ckpt_manager.save_checkpoint(
            model, optimizer, epoch, val_loss, metrics, is_best=is_best,
            scheduler=scheduler, scaler=scaler
        )

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")


if __name__ == '__main__':
    main()
