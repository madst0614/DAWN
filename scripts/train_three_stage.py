"""
Hierarchical Dynamic Neuron FFN Training Script

계층적 동적 뉴런 FFN 모델 학습

Usage:
    # 기본 학습
    python scripts/train_three_stage.py

    # 커스텀 설정
    python scripts/train_three_stage.py \
        --d_model 768 \
        --n_input_neurons 4096 \
        --n_process_neurons 2048 \
        --d_routing 512 \
        --batch_size 16 \
        --num_epochs 30 \
        --lr 3e-4

    # Mixed precision training
    python scripts/train_three_stage.py --use_amp

    # Gradient checkpointing (메모리 절약)
    python scripts/train_three_stage.py --gradient_checkpointing
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

from models.three_stage_ffn import HierarchicalLanguageModel
from utils.training import CheckpointManager, TrainingMonitor, count_parameters, format_time
from utils.data import CacheLoader

# MLM 마스킹 설정
MLM_CONFIG = {
    "mask_prob": 0.15,
    "mask_token_ratio": 0.8,
    "random_token_ratio": 0.5
}


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
    debug_first_n_steps: int = 10
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
    """

    if step > debug_first_n_steps:
        return

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


# ============================================================
# Deep Learning Analysis Function
# ============================================================

def deep_learning_analysis(model, x, labels, step, debug_first_n_steps=10):
    """
    학습 과정의 본질적 정보 추출 - 정보 흐름, gradient 흐름, 라우팅, weight 분포 등 심층 분석

    Args:
        model: HierarchicalLanguageModel
        x: Input token IDs [B, S]
        labels: Labels [B, S]
        step: Current step
        debug_first_n_steps: Debug first N steps only
    """
    if step > debug_first_n_steps:
        return

    import torch.nn.functional as F
    from collections import Counter

    print(f"\n{'='*70}")
    print(f"🔬 DEEP LEARNING ANALYSIS - Step {step}")
    print(f"{'='*70}")

    # ============================================================
    # 1. 정보 흐름 분석 (Information Flow)
    # ============================================================
    print("\n📊 1. INFORMATION FLOW ANALYSIS")
    print("-" * 70)

    with torch.no_grad():
        # Embedding 출력
        B, S = x.shape
        token_emb = model.token_embedding(x)
        positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = model.position_embedding(positions)
        x_emb = token_emb + pos_emb

        print(f"\n[Embedding Layer]")
        print(f"  Token emb norm: {token_emb.norm(dim=-1).mean():.4f}")
        print(f"  Pos emb norm: {pos_emb.norm(dim=-1).mean():.4f}")
        print(f"  Combined std: {x_emb.std():.4f}")
        print(f"  Combined range: [{x_emb.min():.4f}, {x_emb.max():.4f}]")

        # 각 레이어별 출력 분석
        x_layer = x_emb
        for i, layer in enumerate(model.layers):
            # Attention block (uses _attention_block method)
            attn_out = layer._attention_block(x_layer, None)
            print(f"\n[Layer {i} - Attention]")
            print(f"  Output norm: {attn_out.norm(dim=-1).mean():.4f}")
            print(f"  Output std: {attn_out.std():.4f}")
            print(f"  Signal strength: {attn_out.abs().mean():.4f}")

            x_layer = x_layer + attn_out

            # FFN block (uses _ffn_block method)
            ffn_out = layer._ffn_block(x_layer, None, None)
            print(f"\n[Layer {i} - FFN]")
            print(f"  Output norm: {ffn_out.norm(dim=-1).mean():.4f}")
            print(f"  Output std: {ffn_out.std():.4f}")
            print(f"  Signal strength: {ffn_out.abs().mean():.4f}")

            # FFN 내부 분석
            ffn = layer.ffn
            ffn_input = layer.norm2(x_layer)  # Get normalized input for analysis
            with torch.no_grad():
                # Input neuron activations
                input_acts = F.gelu(ffn_input @ ffn.input_patterns.T)
                print(f"  Input neurons:")
                print(f"    Activation mean: {input_acts.mean():.4f}")
                print(f"    Activation std: {input_acts.std():.4f}")
                print(f"    Dead neurons (act < 0.01): {(input_acts.abs() < 0.01).float().mean()*100:.1f}%")
                print(f"    Active neurons: {(input_acts.abs() > 0.1).float().mean()*100:.1f}%")

            x_layer = x_layer + ffn_out

            print(f"\n[Layer {i} - Residual Output]")
            print(f"  Output norm: {x_layer.norm(dim=-1).mean():.4f}")
            print(f"  Output std: {x_layer.std():.4f}")

    # ============================================================
    # 2. Gradient Flow 분석
    # ============================================================
    print(f"\n📈 2. GRADIENT FLOW ANALYSIS")
    print("-" * 70)

    # Forward pass with gradient tracking
    model.zero_grad()
    output = model(x, labels=labels)
    loss = output['loss']
    loss.backward()

    print(f"\n[Loss Value]")
    print(f"  Total loss: {loss.item():.4f}")

    # 각 레이어별 gradient 분석
    for i, layer in enumerate(model.layers):
        ffn = layer.ffn

        print(f"\n[Layer {i} - Gradient Magnitudes]")

        # Neuron keys (routing)
        if ffn.global_router.neuron_keys.grad is not None:
            grad = ffn.global_router.neuron_keys.grad
            print(f"  Router neuron_keys:")
            print(f"    Grad norm: {grad.norm():.6f}")
            print(f"    Grad mean (abs): {grad.abs().mean():.6f}")
            print(f"    Grad max: {grad.abs().max():.6f}")
            print(f"    Non-zero grads: {(grad.abs() > 1e-8).sum()}/{grad.numel()}")

        # Input patterns
        if ffn.input_patterns.grad is not None:
            grad = ffn.input_patterns.grad
            print(f"  Input patterns:")
            print(f"    Grad norm: {grad.norm():.6f}")
            print(f"    Grad mean (abs): {grad.abs().mean():.6f}")
            print(f"    Dead neurons (grad < 1e-6): {(grad.abs().mean(dim=1) < 1e-6).sum()}/{grad.shape[0]}")

        # Process weights
        if ffn.process_weights.grad is not None:
            grad = ffn.process_weights.grad
            print(f"  Process weights:")
            print(f"    Grad norm: {grad.norm():.6f}")
            print(f"    Grad mean (abs): {grad.abs().mean():.6f}")
            print(f"    Dead neurons (grad < 1e-6): {(grad.abs().mean(dim=1) < 1e-6).sum()}/{grad.shape[0]}")

    # ============================================================
    # 3. 학습 역학 분석 (Learning Dynamics)
    # ============================================================
    print(f"\n🎯 3. LEARNING DYNAMICS")
    print("-" * 70)

    with torch.no_grad():
        # Routing pattern 변화
        layer0_ffn = model.layers[0].ffn
        B, S = x.shape
        token_emb = model.token_embedding(x)
        positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = model.position_embedding(positions)
        x_emb = token_emb + pos_emb

        # Get routing info
        input_idx, routing_weights = layer0_ffn.global_router(x_emb, k_input=1024)

        print(f"\n[Routing Behavior]")
        print(f"  Selected neurons (first batch, first 10): {input_idx[0, :10].tolist()}")
        top_weights = routing_weights[0].topk(10)
        print(f"  Top 10 routing weights: {top_weights.values.tolist()}")
        print(f"  Top 10 neuron indices: {top_weights.indices.tolist()}")

        # Routing entropy
        routing_probs = routing_weights[0] / routing_weights[0].sum()
        entropy = -(routing_probs * torch.log(routing_probs + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(float(routing_weights.shape[1])))
        print(f"  Routing entropy: {entropy:.4f} (max: {max_entropy:.4f})")
        print(f"  Normalized entropy: {(entropy / max_entropy):.4f}")

        # Logits 분포
        logits = output['logits']
        print(f"\n[Logits Distribution]")
        print(f"  Mean: {logits.mean():.4f}")
        print(f"  Std: {logits.std():.4f}")
        print(f"  Max: {logits.max():.4f}")
        print(f"  Min: {logits.min():.4f}")

        # Softmax 후 확률 분포
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)
        print(f"\n[Prediction Confidence]")
        print(f"  Mean max probability: {max_probs.mean():.4f}")
        print(f"  Min max probability: {max_probs.min():.4f}")
        print(f"  Max max probability: {max_probs.max():.4f}")

        # Entropy of predictions
        entropy_pred = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        max_vocab_entropy = torch.log(torch.tensor(float(model.vocab_size)))
        print(f"\n[Prediction Entropy]")
        print(f"  Mean entropy: {entropy_pred.mean():.4f}")
        print(f"  Max possible entropy: {max_vocab_entropy:.4f}")
        print(f"  Normalized entropy: {(entropy_pred.mean() / max_vocab_entropy):.4f}")

    # ============================================================
    # 4. Weight 분포 분석
    # ============================================================
    print(f"\n⚖️  4. WEIGHT DISTRIBUTION ANALYSIS")
    print("-" * 70)

    for i, layer in enumerate(model.layers):
        ffn = layer.ffn

        print(f"\n[Layer {i}]")

        # Input patterns
        print(f"  Input patterns:")
        print(f"    Mean: {ffn.input_patterns.mean():.6f}")
        print(f"    Std: {ffn.input_patterns.std():.6f}")
        print(f"    Norm: {ffn.input_patterns.norm():.6f}")

        # Process weights
        print(f"  Process weights:")
        print(f"    Mean: {ffn.process_weights.mean():.6f}")
        print(f"    Std: {ffn.process_weights.std():.6f}")
        print(f"    Norm: {ffn.process_weights.norm():.6f}")

        # Neuron keys
        print(f"  Neuron keys:")
        print(f"    Mean: {ffn.global_router.neuron_keys.mean():.6f}")
        print(f"    Std: {ffn.global_router.neuron_keys.std():.6f}")
        print(f"    Norm: {ffn.global_router.neuron_keys.norm():.6f}")

    # ============================================================
    # 5. 학습 진전도 (Learning Progress)
    # ============================================================
    print(f"\n📉 5. LEARNING PROGRESS INDICATORS")
    print("-" * 70)

    with torch.no_grad():
        # Token prediction diversity
        _, preds = logits.max(dim=-1)
        unique_preds = preds.unique().numel()
        print(f"\n[Prediction Diversity]")
        print(f"  Unique tokens predicted: {unique_preds}/{model.vocab_size}")
        print(f"  Diversity ratio: {unique_preds/model.vocab_size*100:.2f}%")

        # Most common predictions
        pred_counts = Counter(preds.flatten().tolist())
        top_10_preds = pred_counts.most_common(10)
        print(f"\n[Top 10 Most Predicted Tokens]")
        for token, count in top_10_preds:
            pct = count/preds.numel()*100
            print(f"    Token {token}: {count} times ({pct:.2f}%)")

        # Label distribution (for comparison)
        if labels is not None:
            valid_labels = labels[labels != -100]
            if valid_labels.numel() > 0:
                label_counts = Counter(valid_labels.tolist())
                top_10_labels = label_counts.most_common(10)
                print(f"\n[Top 10 Most Frequent True Labels]")
                for token, count in top_10_labels:
                    pct = count/valid_labels.numel()*100
                    print(f"    Token {token}: {count} times ({pct:.2f}%)")

    print(f"\n{'='*70}\n")


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


def load_cached_data(tokenizer_path=None, max_length=128, batch_size=128):  # CHANGED: defaults
    """Load cached WikiText data with DYNAMIC padding"""
    from transformers import AutoTokenizer
    from functools import partial

    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load cached texts
    print("Loading cached WikiText data...")
    train_texts = CacheLoader.load_train_texts(dataset="wikitext")
    val_texts = CacheLoader.load_validation_texts(dataset="wikitext")

    if train_texts is None or val_texts is None:
        raise ValueError(
            "Cached data not found! "
            f"Expected at: {CacheLoader.CACHE_BASE_DIR}/{{train,validation}}/wikitext_5to1_texts.pkl"
        )

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

def compute_load_balance_loss(model):
    """
    모든 레이어의 load balancing loss를 계산합니다.

    Args:
        model: HierarchicalFFNModel

    Returns:
        total_loss: 전체 레이어의 평균 load balance loss
    """
    total_loss = 0
    count = 0

    for layer in model.layers:
        # 각 layer의 ffn에서 load balance loss 계산
        lb_loss = layer.ffn.get_load_balance_loss()
        total_loss = total_loss + lb_loss
        count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0)


def reset_routing_stats(model):
    """모든 레이어의 routing 통계를 초기화합니다."""
    for layer in model.layers:
        layer.ffn.reset_routing_counts()


def print_diagnostic_metrics(model, epoch):
    """
    학습 진단 메트릭 출력

    - Gradient norm (exploding/vanishing 체크)
    - Neuron 사용률 (input neurons)
    - Router entropy (다양성)
    - Process neuron 사용률
    """
    print(f"\n{'='*60}")
    print(f"Diagnostic Metrics (Epoch {epoch})")
    print(f"{'='*60}")

    # 1. Gradient norm
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

    # 첫 번째 레이어의 FFN 분석 (대표값)
    first_ffn = model.layers[0].ffn

    # 2. Input Neuron 사용률
    if first_ffn.input_neuron_counts is not None:
        active_input = (first_ffn.input_neuron_counts > 0).sum().item()
        total_input = first_ffn.n_input
        usage_pct = active_input / total_input * 100

        status = "✓ OK" if usage_pct > 50 else "⚠ LOW"
        print(f"\nInput Neurons (Layer 0): {active_input}/{total_input} ({usage_pct:.1f}%) {status}")
        print(f"  Expected: > 50% for good diversity")
    else:
        print(f"\nInput Neurons: N/A (no stats collected)")

    # 3. Router Entropy (다양성)
    neuron_keys = first_ffn.global_router.neuron_keys  # [n_input, d_routing]
    # 각 뉴런의 key를 확률 분포로 변환 (softmax over neurons)
    # 높은 entropy = 다양한 뉴런이 선택될 가능성
    routing_logits = neuron_keys.norm(dim=1)  # [n_input] - 각 뉴런의 key 크기
    routing_probs = F.softmax(routing_logits, dim=0)
    entropy = -(routing_probs * (routing_probs + 1e-10).log()).sum().item()
    max_entropy = torch.log(torch.tensor(float(first_ffn.n_input))).item()
    entropy_pct = entropy / max_entropy * 100

    status = "✓ OK" if entropy_pct > 50 else "⚠ LOW"
    print(f"\nRouter Entropy (Layer 0): {entropy_pct:.1f}% of max {status}")
    print(f"  Expected: > 50% for diverse routing")

    # 4. Process Neuron 사용률
    if first_ffn.process_neuron_counts is not None:
        active_process = (first_ffn.process_neuron_counts > 0).sum().item()
        total_process = first_ffn.n_process
        process_pct = active_process / total_process * 100

        status = "✓ OK" if process_pct > 50 else "⚠ LOW"
        print(f"\nProcess Neurons (Layer 0): {active_process}/{total_process} ({process_pct:.1f}%) {status}")
        print(f"  Expected: ~100% if k_process=n_process (no selection)")
    else:
        print(f"\nProcess Neurons: N/A (no stats collected)")

    print(f"{'='*60}\n")


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args, scaler=None, tokenizer=None):
    """Train for one epoch"""
    model.train()

    # Epoch 시작 시 routing 통계 초기화
    reset_routing_stats(model)

    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_valid_tokens = 0  # CRITICAL FIX: Track valid tokens only (labels != -100)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)

        # Apply MLM masking
        if tokenizer is not None:
            input_ids, labels = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)
        else:
            # Fallback: no masking
            labels = input_ids.clone()

        # Detailed debugging for first 10 steps of epoch 1
        debug_mode = (epoch == 1 and step < 10)

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
        # Dynamic aux weight: stronger in early epochs
        if epoch <= 5:
            aux_weight = 0.5  # Strong regularization initially (was 0.05)
        else:
            aux_weight = 0.1  # Moderate regularization later (was 0.01)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    k_input=args.k_input,
                    k_process=args.k_process
                )
                loss = outputs['loss']
                logits = outputs['logits']

                # Load balancing loss 추가
                aux_loss = compute_load_balance_loss(model)
                total_loss_combined = loss + aux_weight * aux_loss

            if debug_mode:
                print(f"\nAfter Forward:")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                print(f"  Loss: {loss.item():.4f}, Aux Loss: {aux_loss.item():.4f}")

                # Routing debug
                print(f"\n[Routing Debug - Step {step+1}]")
                first_ffn = model.layers[0].ffn

                if first_ffn.last_routing_scores is not None:
                    routing_weights = first_ffn.last_routing_scores  # [B, n_input]

                    # Top-k pattern analysis
                    topk_vals, topk_idx = routing_weights[0].topk(10)
                    print(f"  Top 10 routing weights: {topk_vals.cpu().numpy()}")

                    # Distribution stats
                    print(f"  Distribution: min={routing_weights.min():.6f}, max={routing_weights.max():.6f}, mean={routing_weights.mean():.6f}")

                    # Sparsity check
                    nonzero = (routing_weights[0] > 1e-6).sum()
                    print(f"  Non-zero weights: {nonzero}/{len(routing_weights[0])}")

                    # Usage stats
                    if first_ffn.input_neuron_counts is not None:
                        counts = first_ffn.input_neuron_counts
                        active_neurons = (counts > 0).sum()
                        print(f"  Active input neurons: {active_neurons}/{len(counts)}")
                else:
                    print(f"  No routing scores available")

            scaler.scale(total_loss_combined).backward()

            # 🔥 COMPREHENSIVE DEBUGGING (first 10 steps)
            comprehensive_debug(
                step=step + 1,
                model=model,
                input_ids=input_ids,
                labels=labels,
                logits=logits,
                loss=loss,
                aux_loss=aux_loss,
                optimizer=optimizer,
                debug_first_n_steps=10
            )

            # 🔬 DEEP LEARNING ANALYSIS (first 10 steps)
            deep_learning_analysis(
                model=model,
                x=input_ids,
                labels=labels,
                step=step + 1,
                debug_first_n_steps=10
            )

            if debug_mode:
                print(f"\n[Additional Debug Info - After Backward]")

                # CRITICAL: Check neuron_keys gradient specifically
                print(f"\n[Critical Gradient Check - neuron_keys]")
                neuron_keys = model.layers[0].ffn.global_router.neuron_keys
                if neuron_keys.grad is not None:
                    grad_norm = neuron_keys.grad.norm().item()
                    grad_mean = neuron_keys.grad.abs().mean().item()
                    nonzero = (neuron_keys.grad.abs() > 1e-10).sum().item()
                    total = neuron_keys.grad.numel()

                    print(f"  ✓ neuron_keys HAS gradient!")
                    print(f"    Shape: {neuron_keys.grad.shape}")
                    print(f"    Grad norm: {grad_norm:.6f}")
                    print(f"    Grad mean (abs): {grad_mean:.8f}")
                    print(f"    Non-zero grads: {nonzero}/{total} ({nonzero/total*100:.1f}%)")

                    if grad_norm < 1e-7:
                        print(f"    ⚠ WARNING: Gradient too small (vanishing)")
                    elif grad_norm > 100:
                        print(f"    ⚠ WARNING: Gradient too large (exploding)")
                    else:
                        print(f"    ✓ Gradient magnitude OK")
                else:
                    print(f"  ✗ neuron_keys has NO GRADIENT - PROBLEM!")

                # Check other parameters
                grad_issues = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm < 1e-7:
                            grad_issues.append(f"  ⚠ {name}: grad too small ({grad_norm:.2e})")
                        elif grad_norm > 100:
                            grad_issues.append(f"  ⚠ {name}: grad too large ({grad_norm:.2e})")
                    else:
                        # Skip neuron_keys since we already checked it above
                        if 'neuron_keys' not in name:
                            grad_issues.append(f"  ⚠ {name}: NO GRADIENT")

                if grad_issues:
                    print("\n  Other Gradient Issues:")
                    for issue in grad_issues[:10]:  # Show first 10 issues
                        print(issue)
                else:
                    print("\n  ✓ All other gradients OK")

            scaler.unscale_(optimizer)

            # Gradient clipping with verification
            grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if debug_mode:
                print(f"\n[Gradient Clipping]")
                print(f"  Grad norm before clipping: {grad_norm_before:.2f}")
                if grad_norm_before > 10.0:
                    print(f"  ⚠ WARNING: Gradient exploding! (>{10.0})")

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                k_input=args.k_input,
                k_process=args.k_process
            )
            loss = outputs['loss']
            logits = outputs['logits']

            # Load balancing loss 추가
            aux_loss = compute_load_balance_loss(model)
            total_loss_combined = loss + aux_weight * aux_loss

            if debug_mode:
                print(f"\nAfter Forward:")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                print(f"  Loss: {loss.item():.4f}, Aux Loss: {aux_loss.item():.4f}")

                # Routing debug
                print(f"\n[Routing Debug - Step {step+1}]")
                first_ffn = model.layers[0].ffn

                if first_ffn.last_routing_scores is not None:
                    routing_weights = first_ffn.last_routing_scores  # [B, n_input]

                    # Top-k pattern analysis
                    topk_vals, topk_idx = routing_weights[0].topk(10)
                    print(f"  Top 10 routing weights: {topk_vals.cpu().numpy()}")

                    # Distribution stats
                    print(f"  Distribution: min={routing_weights.min():.6f}, max={routing_weights.max():.6f}, mean={routing_weights.mean():.6f}")

                    # Sparsity check
                    nonzero = (routing_weights[0] > 1e-6).sum()
                    print(f"  Non-zero weights: {nonzero}/{len(routing_weights[0])}")

                    # Usage stats
                    if first_ffn.input_neuron_counts is not None:
                        counts = first_ffn.input_neuron_counts
                        active_neurons = (counts > 0).sum()
                        print(f"  Active input neurons: {active_neurons}/{len(counts)}")
                else:
                    print(f"  No routing scores available")

            total_loss_combined.backward()

            # 🔥 COMPREHENSIVE DEBUGGING (first 10 steps)
            comprehensive_debug(
                step=step + 1,
                model=model,
                input_ids=input_ids,
                labels=labels,
                logits=logits,
                loss=loss,
                aux_loss=aux_loss,
                optimizer=optimizer,
                debug_first_n_steps=10
            )

            # 🔬 DEEP LEARNING ANALYSIS (first 10 steps)
            deep_learning_analysis(
                model=model,
                x=input_ids,
                labels=labels,
                step=step + 1,
                debug_first_n_steps=10
            )

            if debug_mode:
                print(f"\n[Additional Debug Info - After Backward]")

                # CRITICAL: Check neuron_keys gradient specifically
                print(f"\n[Critical Gradient Check - neuron_keys]")
                neuron_keys = model.layers[0].ffn.global_router.neuron_keys
                if neuron_keys.grad is not None:
                    grad_norm = neuron_keys.grad.norm().item()
                    grad_mean = neuron_keys.grad.abs().mean().item()
                    nonzero = (neuron_keys.grad.abs() > 1e-10).sum().item()
                    total = neuron_keys.grad.numel()

                    print(f"  ✓ neuron_keys HAS gradient!")
                    print(f"    Shape: {neuron_keys.grad.shape}")
                    print(f"    Grad norm: {grad_norm:.6f}")
                    print(f"    Grad mean (abs): {grad_mean:.8f}")
                    print(f"    Non-zero grads: {nonzero}/{total} ({nonzero/total*100:.1f}%)")

                    if grad_norm < 1e-7:
                        print(f"    ⚠ WARNING: Gradient too small (vanishing)")
                    elif grad_norm > 100:
                        print(f"    ⚠ WARNING: Gradient too large (exploding)")
                    else:
                        print(f"    ✓ Gradient magnitude OK")
                else:
                    print(f"  ✗ neuron_keys has NO GRADIENT - PROBLEM!")

                # Check other parameters
                grad_issues = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm < 1e-7:
                            grad_issues.append(f"  ⚠ {name}: grad too small ({grad_norm:.2e})")
                        elif grad_norm > 100:
                            grad_issues.append(f"  ⚠ {name}: grad too large ({grad_norm:.2e})")
                    else:
                        # Skip neuron_keys since we already checked it above
                        if 'neuron_keys' not in name:
                            grad_issues.append(f"  ⚠ {name}: NO GRADIENT")

                if grad_issues:
                    print("\n  Other Gradient Issues:")
                    for issue in grad_issues[:10]:  # Show first 10 issues
                        print(issue)
                else:
                    print("\n  ✓ All other gradients OK")

            # Gradient clipping with verification
            grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if debug_mode:
                print(f"\n[Gradient Clipping]")
                print(f"  Grad norm before clipping: {grad_norm_before:.2f}")
                if grad_norm_before > 10.0:
                    print(f"  ⚠ WARNING: Gradient exploding! (>{10.0})")

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

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "aux": f"{aux_loss.item():.4f}",
            "w_aux": f"{(aux_weight * aux_loss).item():.5f}",
            "acc": f"{correct / valid_tokens:.4f}" if valid_tokens > 0 else "0.0000"
        })

    avg_loss = total_loss / total_tokens
    avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
    return avg_loss, avg_acc


def evaluate(model, dataloader, device, args):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_valid_tokens = 0  # CRITICAL FIX: Track valid tokens only

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()

            outputs = model(
                input_ids=input_ids,
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


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical Dynamic Neuron FFN')

    # Model architecture
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--max_seq_len', type=int, default=128,  # CHANGED: 512 → 128 (Scenario B)
                        help='Maximum sequence length')

    # Hierarchical FFN specific
    parser.add_argument('--n_input_neurons', type=int, default=2048,
                        help='Number of input neurons')
    parser.add_argument('--n_process_neurons', type=int, default=1024,
                        help='Number of process neurons')
    parser.add_argument('--d_routing', type=int, default=256,
                        help='Routing dimension for global router')

    # Sparsity control (runtime)
    parser.add_argument('--k_input', type=int, default=None,
                        help='Number of input neurons to activate (None = n_input//8)')
    parser.add_argument('--k_process', type=int, default=None,
                        help='Number of process neurons to activate (None = n_process//8)')

    # Training
    parser.add_argument('--batch_size', type=int, default=128,  # CHANGED: 32 → 128 (Scenario B)
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Optimization
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Use gradient checkpointing to save memory')

    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Log directory')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(args.checkpoint_dir) / "three_stage" / timestamp
    log_dir = Path(args.log_dir) / "three_stage" / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Hierarchical Dynamic Neuron FFN Training")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Load data
    print(f"\n{'='*60}")
    print("Loading cached WikiText data...")
    print(f"{'='*60}")
    train_loader, val_loader, tokenizer = load_cached_data(
        max_length=args.max_seq_len,
        batch_size=args.batch_size
    )

    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print(f"\n{'='*60}")
    print("Creating Hierarchical FFN model...")
    print(f"{'='*60}")

    model = HierarchicalLanguageModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        n_input_neurons=args.n_input_neurons,
        n_process_neurons=args.n_process_neurons,
        d_routing=args.d_routing,
        dropout=args.dropout,
        gradient_checkpointing=args.gradient_checkpointing
    )

    model = model.to(device)

    # Model statistics
    stats = model.get_model_stats()
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"  FFN parameters: {stats['ffn_parameters']:,} ({stats['ffn_percentage']:.1f}%)")
    print(f"  Router parameters: {stats['router_parameters']:,} ({stats['router_percentage']:.1f}%)")
    print(f"  Number of layers: {stats['n_layers']}")

    # Sparsity info
    # Start with less aggressive sparsity to verify architecture works
    # Then gradually increase sparsity if training succeeds
    if args.k_input is None:
        k_input_actual = args.n_input_neurons // 2  # 50% (was 12.5%)
    else:
        k_input_actual = args.k_input

    if args.k_process is None:
        k_process_actual = args.n_process_neurons  # 100% - no selection (was 12.5%)
    else:
        k_process_actual = args.k_process

    print(f"\nSparsity Configuration:")
    print(f"  Input neurons: {k_input_actual}/{args.n_input_neurons} ({k_input_actual/args.n_input_neurons*100:.1f}%)")
    print(f"  Process neurons: {k_process_actual}/{args.n_process_neurons} ({k_process_actual/args.n_process_neurons*100:.1f}%)")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )

    # Warmup + Cosine Annealing scheduler
    warmup_epochs = 2
    warmup_steps = warmup_epochs * len(train_loader)
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

    # Checkpoint & Monitor
    ckpt_manager = CheckpointManager(str(checkpoint_dir), keep_best_n=3)
    monitor = TrainingMonitor(str(log_dir))

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}")
    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args, scaler, tokenizer
        )

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device, args)

        epoch_time = time.time() - epoch_start

        # Log
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        monitor.log_epoch(epoch, metrics)

        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
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
            model, optimizer, epoch, val_loss, metrics, is_best=is_best
        )

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")


if __name__ == '__main__':
    main()
