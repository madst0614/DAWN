"""
DAWN (Dynamic Architecture With Neurons) 종합 분석 스크립트

학습된 DAWN 모델의 상세 분석:
- 레이어별 특성 분석
- 성능 breakdown
- 예측 분포 분석

Usage:
    python scripts/analyze_dawn.py --checkpoint path/to/checkpoint.pt
    python scripts/analyze_dawn.py --checkpoint path/to/checkpoint.pt --output results.json
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
import json
import argparse
from tqdm import tqdm
from collections import Counter, defaultdict
from datetime import datetime

from models.model import HierarchicalLanguageModel
from utils.training import CheckpointManager
from utils.data import apply_mlm_masking, compute_mlm_accuracy


# ============================================================
# Data Loading
# ============================================================

def load_data_from_config(config_path, batch_size=64):
    """Config에서 데이터 로드"""
    import yaml
    import pickle
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, Dataset

    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg['data']
    model_cfg = cfg['model']

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load validation texts
    val_path = os.path.join(data_cfg['base_dir'], data_cfg['val_file'])
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found: {val_path}")

    with open(val_path, 'rb') as f:
        val_texts = pickle.load(f)

    # Simple dataset
    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }

    val_dataset = TextDataset(val_texts, tokenizer, model_cfg['max_seq_len'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

    return val_loader, tokenizer, cfg


# ============================================================
# Model Loading
# ============================================================

def load_checkpoint(checkpoint_path, device='cuda'):
    """체크포인트에서 모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_dir = Path(checkpoint_path).parent

    # Config 로드 (config.json 파일에서)
    config_path = checkpoint_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from: {config_path}")
    else:
        # 기본값 사용
        print("Config file not found, using defaults")
        config = {
            'model': {
                'd_model': 512,
                'n_heads': 8,
                'n_layers': 6,
                'max_seq_len': 128,
                'n_input': 128,
                'n_process': 256,
                'dropout': 0.1
            }
        }

    # 가중치 로드
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # vocab_size를 state_dict에서 추론 (token_embedding.weight shape)
    vocab_size = 30522  # Default for bert-base-uncased
    if 'token_embedding.weight' in state_dict:
        vocab_size = state_dict['token_embedding.weight'].shape[0]
        print(f"Inferred vocab_size from state_dict: {vocab_size}")

    # config가 새 형식인지 확인
    if 'model' in config:
        model_cfg = config['model']
    else:
        # 구 형식 호환
        model_cfg = config

    # 모델 생성
    model = HierarchicalLanguageModel(
        vocab_size=vocab_size,
        d_model=model_cfg.get('d_model', 512),
        n_heads=model_cfg.get('n_heads', 8),
        n_layers=model_cfg.get('n_layers', 6),
        max_seq_len=model_cfg.get('max_seq_len', 128),
        n_input=model_cfg.get('n_input', 128),
        n_process=model_cfg.get('n_process', 256),
        dropout=model_cfg.get('dropout', 0.1)
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # 체크포인트 정보 출력
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        if 'val_acc' in metrics:
            print(f"Checkpoint val_acc: {metrics['val_acc']:.4f}")

    return model, config


# ============================================================
# Analysis Functions
# ============================================================

def analyze_routing_patterns(model, val_loader, device, max_batches=50):
    """Router 패턴 분석 - 각 레이어의 소프트 라우팅 통계"""
    print("\n📊 Analyzing Routing Patterns...")

    n_layers = len(model.layers)
    n_input = model.layers[0].block.n_input

    # Track soft weight accumulation per layer
    layer_weight_sums = [torch.zeros(n_input, device=device) for _ in range(n_layers)]
    layer_learned_params = [[] for _ in range(n_layers)]
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Routing analysis")):
        if batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)
        batch_size = input_ids.shape[0]
        total_samples += batch_size

        with torch.no_grad():
            # Get embeddings
            B, S = input_ids.shape
            token_emb = model.token_embedding(input_ids)
            positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = model.position_embedding(positions)
            x = model.dropout(token_emb + pos_emb)

            # Track routing through layers
            for layer_idx, layer in enumerate(model.layers):
                # Get router output (pure soft selection!)
                weights, context, selection_info = layer.block.router(x)

                # Accumulate soft weights
                layer_weight_sums[layer_idx] += weights.sum(dim=0)

                # Track learned parameters
                layer_learned_params[layer_idx].append({
                    'threshold': selection_info['learned_threshold'],
                    'steepness': selection_info['learned_steepness'],
                    'temperature': selection_info['temperature'],
                    'effective_k': selection_info['effective_k'],
                    'effective_k_ratio': selection_info['effective_k_ratio']
                })

                # Forward through layer for next iteration
                x, _ = layer(x)

    # Compute statistics
    results = {
        'n_layers': n_layers,
        'n_input': n_input,
        'total_samples': total_samples,
        'layers': {}
    }

    for layer_idx in range(n_layers):
        # Average soft weights
        avg_weights = (layer_weight_sums[layer_idx] / total_samples).cpu().numpy()

        # Average learned parameters
        params = layer_learned_params[layer_idx]
        avg_threshold = sum(p['threshold'] for p in params) / len(params)
        avg_steepness = sum(p['steepness'] for p in params) / len(params)
        avg_temperature = sum(p['temperature'] for p in params) / len(params)
        avg_effective_k = sum(p['effective_k'] for p in params) / len(params)
        avg_effective_k_ratio = sum(p['effective_k_ratio'] for p in params) / len(params)

        results['layers'][layer_idx] = {
            'mean_weight': float(avg_weights.mean()),
            'std_weight': float(avg_weights.std()),
            'min_weight': float(avg_weights.min()),
            'max_weight': float(avg_weights.max()),
            'low_weight_neurons': int((avg_weights < 0.001).sum()),
            'top_5_neurons': avg_weights.argsort()[-5:][::-1].tolist(),
            'learned_params': {
                'threshold': avg_threshold,
                'steepness': avg_steepness,
                'temperature': avg_temperature,
                'effective_k': avg_effective_k,
                'effective_k_ratio': avg_effective_k_ratio
            }
        }

    return results


def analyze_neuron_usage(model, val_loader, device, max_batches=50):
    """뉴런 사용 분포 분석 - Load balancing 확인"""
    print("\n🧠 Analyzing Neuron Usage...")

    n_layers = len(model.layers)
    n_input = model.layers[0].block.n_input

    # Accumulate routing weights per layer
    layer_weights = [torch.zeros(n_input, device=device) for _ in range(n_layers)]
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Usage analysis")):
        if batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)
        batch_size = input_ids.shape[0]
        total_samples += batch_size

        with torch.no_grad():
            B, S = input_ids.shape
            token_emb = model.token_embedding(input_ids)
            positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = model.position_embedding(positions)
            x = model.dropout(token_emb + pos_emb)

            for layer_idx, layer in enumerate(model.layers):
                weights, _, _ = layer.block.router(x)

                # Accumulate weights
                layer_weights[layer_idx] += weights.sum(dim=0)

                x, _ = layer(x)

    results = {'layers': {}}

    for layer_idx in range(n_layers):
        weights = layer_weights[layer_idx].cpu().numpy()
        weights = weights / total_samples  # Normalize

        # Compute Gini coefficient for load balance
        sorted_weights = np.sort(weights)
        n = len(sorted_weights)
        cumsum = np.cumsum(sorted_weights)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_weights))) / (n * np.sum(sorted_weights)) - (n + 1) / n

        results['layers'][layer_idx] = {
            'mean_weight': float(weights.mean()),
            'std_weight': float(weights.std()),
            'gini_coefficient': float(gini),
            'load_balance_score': float(1 - abs(gini))  # 1 = perfectly balanced
        }

    return results


def analyze_neuron_specialization(model, val_loader, tokenizer, device, max_batches=100, layer_idx=3):
    """
    뉴런 특화 분석 - PMI 기반 개선 버전

    개선점:
    - PMI (Pointwise Mutual Information) 사용으로 frequent token bias 제거
    - 모든 tokens 고려 (not just first 10)
    - 특정 layer 분석 (default: layer 3)
    - 더 많은 batches (100)

    Args:
        layer_idx: 분석할 layer (기본값: 3, middle layer)
    """
    print(f"\n💎 Analyzing Neuron Specialization (Layer {layer_idx}, PMI-based)...")

    n_input = model.layers[layer_idx].block.n_input

    # Global token frequency (전체 토큰 빈도)
    global_token_counts = defaultdict(int)

    # Per-neuron token co-occurrence
    neuron_token_counts = defaultdict(lambda: defaultdict(int))
    neuron_activation_counts = defaultdict(int)

    total_tokens = 0

    for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Specialization analysis (L{layer_idx})")):
        if batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)

        with torch.no_grad():
            B, S = input_ids.shape
            token_emb = model.token_embedding(input_ids)
            positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = model.position_embedding(positions)
            x = model.dropout(token_emb + pos_emb)

            # Forward through layers up to target layer
            for i in range(layer_idx):
                x, _ = model.layers[i](x)

            # Get routing for target layer
            layer = model.layers[layer_idx]
            weights, _, selection_info = layer.block.router(x)

            # Get top-k neurons based on soft weights (for analysis)
            k_eff = int(selection_info['effective_k'])
            _, top_indices = weights.topk(k_eff, dim=-1)

            # Track ALL tokens (not just first 10!)
            for b in range(B):
                for neuron_idx in top_indices[b].cpu().tolist():
                    neuron_activation_counts[neuron_idx] += 1

                    # Count ALL tokens in sequence
                    for pos in range(S):
                        token_id = input_ids[b, pos].item()
                        # Skip special tokens (PAD, CLS, SEP)
                        if token_id not in [0, 101, 102]:
                            neuron_token_counts[neuron_idx][token_id] += 1
                            global_token_counts[token_id] += 1
                            total_tokens += 1

    # Compute PMI for each neuron
    total_activations = sum(neuron_activation_counts.values())

    neuron_specializations = []

    for neuron_idx in range(n_input):
        if neuron_activation_counts[neuron_idx] < 10:
            continue  # Skip rarely activated neurons

        token_pmi_scores = {}

        for token_id, count in neuron_token_counts[neuron_idx].items():
            # P(token, neuron) = joint probability
            p_joint = count / total_tokens

            # P(token) = marginal probability
            p_token = global_token_counts[token_id] / total_tokens

            # P(neuron) = marginal probability
            p_neuron = neuron_activation_counts[neuron_idx] / total_activations

            # PMI = log(P(token, neuron) / (P(token) * P(neuron)))
            # Higher PMI = stronger association (not just frequency!)
            if p_token > 0 and p_neuron > 0 and p_joint > 0:
                pmi = np.log((p_joint / (p_token * p_neuron)) + 1e-10)
                token_pmi_scores[token_id] = pmi

        # Sort by PMI (not raw count!)
        top_tokens_pmi = sorted(token_pmi_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        if top_tokens_pmi:
            top_token_words = []
            for token_id, pmi_score in top_tokens_pmi:
                try:
                    word = tokenizer.decode([token_id])
                except:
                    word = f"[{token_id}]"
                top_token_words.append({
                    'token': word,
                    'pmi': float(pmi_score),
                    'raw_count': neuron_token_counts[neuron_idx][token_id]
                })

            # Compute specialization strength (avg PMI of top tokens)
            avg_pmi = np.mean([pmi for _, pmi in top_tokens_pmi[:5]])

            neuron_specializations.append({
                'neuron_idx': neuron_idx,
                'specialization_strength': float(avg_pmi),
                'top_tokens_pmi': top_token_words,
                'activation_count': neuron_activation_counts[neuron_idx],
                'unique_tokens': len(token_pmi_scores)
            })

    # Sort by specialization strength
    neuron_specializations.sort(key=lambda x: x['specialization_strength'], reverse=True)

    results = {
        'layer_idx': layer_idx,
        'total_analyzed': len(neuron_specializations),
        'avg_specialization': float(np.mean([n['specialization_strength'] for n in neuron_specializations])) if neuron_specializations else 0,
        'specialized_neurons': neuron_specializations[:20],  # Top 20 most specialized
        'total_tokens_analyzed': total_tokens
    }

    return results


def analyze_layer_differences(model, val_loader, device):
    """레이어별 특성 분석"""
    print("\n📈 Analyzing Layer Differences...")

    layer_outputs = defaultdict(list)

    for batch in tqdm(val_loader, desc="Layer analysis"):
        input_ids = batch['input_ids'].to(device)

        with torch.no_grad():
            # Embedding
            B, S = input_ids.shape
            token_emb = model.token_embedding(input_ids)
            positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = model.position_embedding(positions)
            x = model.dropout(token_emb + pos_emb)

            # 각 레이어 통과
            for layer_idx, layer in enumerate(model.layers):
                x, _ = layer(x)

                # 출력 통계
                layer_outputs[layer_idx].append({
                    'norm': x.norm().item(),
                    'mean': x.mean().item(),
                    'std': x.std().item()
                })

    # 평균 계산
    results = {'layers': {}}

    for layer_idx in range(len(model.layers)):
        outputs = layer_outputs[layer_idx]
        results['layers'][layer_idx] = {
            'avg_norm': np.mean([o['norm'] for o in outputs]),
            'avg_mean': np.mean([o['mean'] for o in outputs]),
            'avg_std': np.mean([o['std'] for o in outputs])
        }

    return results


def analyze_performance(model, val_loader, tokenizer, device):
    """성능 세부 분석 - MLM masking 적용"""
    print("\n🎯 Analyzing Performance...")

    all_losses = []
    all_corrects = []

    for batch in tqdm(val_loader, desc="Performance analysis"):
        input_ids = batch['input_ids'].to(device)

        # Apply MLM masking (CRITICAL FIX: 기존에는 labels = input_ids.clone()으로 순환 논리 발생)
        masked_input_ids, labels = apply_mlm_masking(input_ids.clone(), tokenizer)

        with torch.no_grad():
            outputs = model(masked_input_ids, labels=labels)
            logits = outputs['logits']

            # Per-token loss (only on masked tokens)
            loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            per_token_loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            # Accuracy using unified function
            num_correct, num_valid = compute_mlm_accuracy(logits, labels)

            # Collect per-token data (only valid tokens)
            valid_mask = (labels.view(-1) != -100)
            valid_losses = per_token_loss[valid_mask].cpu().tolist()

            # Get per-token correctness for valid tokens only
            preds = logits.argmax(dim=-1)
            correct = ((preds == labels) & (labels != -100)).view(-1)
            valid_corrects = correct[valid_mask].float().cpu().tolist()

            all_losses.extend(valid_losses)
            all_corrects.extend(valid_corrects)

    losses = np.array(all_losses)
    corrects = np.array(all_corrects)

    # Percentile 분석
    easy_threshold = np.percentile(losses, 25)
    hard_threshold = np.percentile(losses, 75)

    easy_mask = losses < easy_threshold
    hard_mask = losses > hard_threshold

    results = {
        'overall_loss': float(np.mean(losses)),
        'overall_acc': float(np.mean(corrects)),
        'easy_samples_acc': float(np.mean(corrects[easy_mask])) if easy_mask.sum() > 0 else 0,
        'hard_samples_acc': float(np.mean(corrects[hard_mask])) if hard_mask.sum() > 0 else 0,
        'loss_percentiles': {
            'p25': float(easy_threshold),
            'p50': float(np.percentile(losses, 50)),
            'p75': float(hard_threshold)
        },
        'total_valid_tokens': len(losses)
    }

    return results


def analyze_aux_loss_components(model, val_loader, device):
    """Aux loss 구성 요소 분석"""
    print("\n⚖️  Analyzing Loss Components...")

    total_main_loss = 0
    total_load_balance = 0
    total_entropy = 0
    n_batches = 0

    for batch in tqdm(val_loader, desc="Loss analysis"):
        input_ids = batch['input_ids'].to(device)
        labels = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            main_loss = outputs['loss']
            aux_loss = outputs['aux_loss']

            total_main_loss += main_loss.item()
            total_load_balance += aux_loss['load_balance'].item()
            total_entropy += aux_loss['entropy'].item()
            n_batches += 1

    avg_aux = (total_load_balance + total_entropy) / (2 * n_batches)
    avg_main = total_main_loss / n_batches

    results = {
        'avg_main_loss': avg_main,
        'avg_load_balance': total_load_balance / n_batches,
        'avg_entropy': total_entropy / n_batches,
        'avg_aux_loss': avg_aux,
        'aux_to_main_ratio': avg_aux / avg_main if avg_main > 0 else 0
    }

    return results


# ============================================================
# Advanced Analysis Functions
# ============================================================

def analyze_weight_matrices(model):
    """
    Weight matrix 자체의 구조 분석
    - Singular values (rank)
    - Condition number (stability)
    - Weight norm 분포
    """
    print("\n📐 Analyzing Weight Matrices...")

    results = {'layers': {}}

    for layer_idx, layer in enumerate(model.layers):
        layer_results = {}

        # Router patterns analysis
        if hasattr(layer.block.router, 'neuron_patterns'):
            patterns = layer.block.router.neuron_patterns.data

            # SVD 분석
            U, S, V = torch.svd(patterns)

            layer_results['router'] = {
                'singular_values_top5': S[:5].cpu().tolist(),
                'effective_rank': (S.sum() / S.max()).item() if S.max() > 0 else 0,
                'condition_number': (S.max() / (S.min() + 1e-8)).item(),
                'frobenius_norm': patterns.norm().item(),
            }

        # InputNeurons patterns
        if hasattr(layer.block, 'input_neurons') and hasattr(layer.block.input_neurons, 'patterns'):
            layer_results['input_pattern_norm'] = layer.block.input_neurons.patterns.norm().item()

        # ProcessNeurons weights
        if hasattr(layer.block, 'process_neurons') and hasattr(layer.block.process_neurons, 'combination_weights'):
            layer_results['process_weight_norm'] = layer.block.process_neurons.combination_weights.norm().item()

        results['layers'][layer_idx] = layer_results

    return results


def analyze_prediction_confidence(model, val_loader, tokenizer, device):
    """
    모델이 얼마나 확신하는지
    - Softmax entropy 분포
    - Calibration (confidence vs accuracy)
    """
    print("\n🎲 Analyzing Prediction Confidence...")

    confidences = []
    accuracies = []

    for batch in tqdm(val_loader, desc="Confidence analysis"):
        input_ids = batch['input_ids'].to(device)

        # Apply MLM masking
        masked_input_ids, labels = apply_mlm_masking(input_ids.clone(), tokenizer)

        with torch.no_grad():
            outputs = model(masked_input_ids, labels=labels)
            logits = outputs['logits']

            probs = F.softmax(logits, dim=-1)
            confidence, preds = probs.max(dim=-1)

            # Only count masked tokens
            valid_mask = (labels != -100)
            valid_confidence = confidence[valid_mask].cpu().tolist()
            valid_correct = ((preds == labels) & valid_mask)[valid_mask].cpu().tolist()

            confidences.extend(valid_confidence)
            accuracies.extend(valid_correct)

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    # Compute calibration bins
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    calibration = []

    for i in range(n_bins):
        bin_mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if bin_mask.sum() > 0:
            bin_acc = accuracies[bin_mask].mean()
            bin_conf = confidences[bin_mask].mean()
            calibration.append({
                'bin': i,
                'confidence': float(bin_conf),
                'accuracy': float(bin_acc),
                'count': int(bin_mask.sum())
            })

    # Compute ECE (Expected Calibration Error)
    ece = 0.0
    total = len(confidences)
    for cal in calibration:
        ece += (cal['count'] / total) * abs(cal['confidence'] - cal['accuracy'])

    correct_mask = np.array(accuracies) == 1
    wrong_mask = ~correct_mask

    results = {
        'avg_confidence': float(np.mean(confidences)),
        'confidence_when_correct': float(np.mean(confidences[correct_mask])) if correct_mask.sum() > 0 else 0,
        'confidence_when_wrong': float(np.mean(confidences[wrong_mask])) if wrong_mask.sum() > 0 else 0,
        'calibration_error': float(ece),
        'calibration_bins': calibration,
        'overconfident_ratio': float(((confidences > 0.8) & wrong_mask).sum() / max(1, wrong_mask.sum()))
    }

    return results


def analyze_error_patterns(model, val_loader, tokenizer, device, max_batches=50):
    """
    어떤 종류의 토큰을 틀리나?
    - Frequency별 accuracy (rare vs common)
    - Position별 accuracy (시작/중간/끝)
    """
    print("\n❌ Analyzing Error Patterns...")

    token_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    position_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Error analysis")):
        if batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)

        # Apply MLM masking
        masked_input_ids, labels = apply_mlm_masking(input_ids.clone(), tokenizer)

        with torch.no_grad():
            outputs = model(masked_input_ids, labels=labels)
            preds = outputs['logits'].argmax(dim=-1)

            for b in range(input_ids.size(0)):
                for pos in range(input_ids.size(1)):
                    if labels[b, pos].item() == -100:
                        continue

                    token_id = labels[b, pos].item()
                    is_correct = (preds[b, pos] == labels[b, pos]).item()

                    # Token stats
                    token_stats[token_id]['total'] += 1
                    if is_correct:
                        token_stats[token_id]['correct'] += 1

                    # Position stats (normalized to 0-9 bins)
                    seq_len = (labels[b] != -100).sum().item()
                    if seq_len > 0:
                        pos_bin = min(9, int(pos * 10 / input_ids.size(1)))
                        position_stats[pos_bin]['total'] += 1
                        if is_correct:
                            position_stats[pos_bin]['correct'] += 1

    # Compute token accuracy
    token_acc = []
    for token_id, stats in token_stats.items():
        if stats['total'] >= 5:  # Minimum samples
            acc = stats['correct'] / stats['total']
            token_acc.append((token_id, acc, stats['total']))

    token_acc.sort(key=lambda x: x[1])

    # Worst tokens
    worst_tokens = []
    for token_id, acc, count in token_acc[:10]:
        try:
            word = tokenizer.decode([token_id])
        except:
            word = f"[{token_id}]"
        worst_tokens.append({'token': word, 'accuracy': acc, 'count': count})

    # Best tokens
    best_tokens = []
    for token_id, acc, count in token_acc[-10:]:
        try:
            word = tokenizer.decode([token_id])
        except:
            word = f"[{token_id}]"
        best_tokens.append({'token': word, 'accuracy': acc, 'count': count})

    # Position accuracy
    position_accuracy = {}
    for pos_bin in range(10):
        stats = position_stats[pos_bin]
        if stats['total'] > 0:
            position_accuracy[pos_bin] = stats['correct'] / stats['total']
        else:
            position_accuracy[pos_bin] = 0.0

    results = {
        'worst_tokens': worst_tokens,
        'best_tokens': best_tokens,
        'position_accuracy': position_accuracy,
        'total_tokens_analyzed': sum(s['total'] for s in token_stats.values())
    }

    return results


def analyze_gradient_flow(model, val_loader, tokenizer, device, n_batches=10):
    """
    Gradient가 레이어별로 얼마나 잘 흐르는지
    - Gradient norm per layer
    """
    print("\n🌊 Analyzing Gradient Flow...")

    model.train()  # Enable gradient

    layer_gradients = defaultdict(list)
    router_gradients = defaultdict(list)

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Gradient analysis")):
        if batch_idx >= n_batches:
            break

        input_ids = batch['input_ids'].to(device)
        masked_input_ids, labels = apply_mlm_masking(input_ids.clone(), tokenizer)

        model.zero_grad()

        outputs = model(masked_input_ids, labels=labels)
        loss = outputs['loss']
        loss.backward()

        # Collect gradients per layer
        for layer_idx, layer in enumerate(model.layers):
            # Overall layer gradient
            layer_grad_norm = 0.0
            param_count = 0
            for param in layer.parameters():
                if param.grad is not None:
                    layer_grad_norm += param.grad.norm().item() ** 2
                    param_count += 1
            if param_count > 0:
                layer_grad_norm = (layer_grad_norm / param_count) ** 0.5
                layer_gradients[layer_idx].append(layer_grad_norm)

            # Router specific gradient
            if hasattr(layer.block.router, 'neuron_patterns') and layer.block.router.neuron_patterns.grad is not None:
                grad_norm = layer.block.router.neuron_patterns.grad.norm().item()
                router_gradients[layer_idx].append(grad_norm)

    model.eval()

    results = {
        'layer_gradient_norms': {
            idx: float(np.mean(grads)) for idx, grads in layer_gradients.items()
        },
        'router_gradient_norms': {
            idx: float(np.mean(grads)) for idx, grads in router_gradients.items()
        }
    }

    # Compute gradient ratio (early vs late)
    if len(layer_gradients) >= 2:
        early_grad = np.mean([np.mean(layer_gradients[i]) for i in range(len(layer_gradients)//2)])
        late_grad = np.mean([np.mean(layer_gradients[i]) for i in range(len(layer_gradients)//2, len(layer_gradients))])
        results['early_late_ratio'] = float(early_grad / (late_grad + 1e-8))

    return results


def analyze_top_neurons(model, val_loader, tokenizer, device, layer_idx=None, top_k=10, max_batches=30):
    """
    특정 레이어의 top-k neurons가 무엇을 하는지 상세 분석
    """
    # Find layer with highest Gini if not specified
    if layer_idx is None:
        # Default to middle layer
        layer_idx = len(model.layers) // 2

    print(f"\n🔬 Deep Dive: Layer {layer_idx} Top-{top_k} Neurons...")

    n_input = model.layers[layer_idx].block.n_input

    # Track neuron activations
    neuron_counts = torch.zeros(n_input, device=device)
    neuron_token_affinity = defaultdict(lambda: defaultdict(float))

    for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Layer {layer_idx} analysis")):
        if batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)

        with torch.no_grad():
            # Get embeddings and pass through layers up to target
            B, S = input_ids.shape
            token_emb = model.token_embedding(input_ids)
            positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = model.position_embedding(positions)
            x = model.dropout(token_emb + pos_emb)

            for i in range(layer_idx):
                x, _ = model.layers[i](x)

            # Get routing for target layer
            layer = model.layers[layer_idx]
            weights, _, selection_info = layer.block.router(x)

            # Accumulate soft weights
            neuron_counts += weights.sum(dim=0)

            # Get top neurons for token association tracking
            k_eff = int(selection_info['effective_k'])
            _, top_indices = weights.topk(k_eff, dim=-1)

            # Track token-neuron associations
            for b in range(B):
                for neuron_idx in top_indices[b].cpu().tolist():
                    for token_id in input_ids[b, :20].cpu().tolist():  # First 20 tokens
                        neuron_token_affinity[neuron_idx][token_id] += 1

    # Get top neurons
    top_neuron_indices = neuron_counts.topk(top_k).indices.cpu().tolist()

    top_neurons = []
    for neuron_idx in top_neuron_indices:
        token_counts = neuron_token_affinity[neuron_idx]
        if not token_counts:
            continue

        # Top tokens for this neuron
        top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_token_words = []
        for token_id, count in top_tokens:
            try:
                word = tokenizer.decode([token_id])
            except:
                word = f"[{token_id}]"
            top_token_words.append({'token': word, 'count': int(count)})

        top_neurons.append({
            'neuron_idx': neuron_idx,
            'activation_count': int(neuron_counts[neuron_idx].item()),
            'top_tokens': top_token_words
        })

    # Compute co-activation matrix for top neurons
    coactivation = np.zeros((top_k, top_k))
    # (Simplified - would need another pass for full co-activation)

    results = {
        'layer_idx': layer_idx,
        'top_neurons': top_neurons,
        'total_activations': int(neuron_counts.sum().item())
    }

    return results


def analyze_process_neurons(model, val_loader, device, max_batches=30):
    """
    Process neurons 분석
    - 어떤 process neurons이 자주 사용되나?
    """
    print("\n⚙️  Analyzing Process Neurons...")

    n_layers = len(model.layers)
    if not hasattr(model.layers[0].block, 'process_neurons'):
        print("  No process neurons found in model")
        return {'error': 'No process neurons'}

    n_process = model.layers[0].block.n_process

    # Track process neuron usage per layer
    layer_process_counts = [torch.zeros(n_process, device=device) for _ in range(n_layers)]
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Process neuron analysis")):
        if batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)
        B = input_ids.size(0)
        total_samples += B

        with torch.no_grad():
            # Forward pass with routing info
            outputs = model(input_ids, return_routing_info=True)

            if 'routing_info' in outputs:
                for layer_idx, layer_info in enumerate(outputs['routing_info']):
                    if 'process_indices' in layer_info:
                        indices = layer_info['process_indices']
                        k = indices.size(1)
                        for b in range(B):
                            layer_process_counts[layer_idx].scatter_add_(
                                0, indices[b], torch.ones(k, device=device)
                            )

    results = {'layers': {}}

    for layer_idx in range(n_layers):
        counts = layer_process_counts[layer_idx].cpu().numpy()
        if counts.sum() == 0:
            continue

        # Compute Gini for process neurons
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_counts))) / (n * np.sum(sorted_counts) + 1e-8) - (n + 1) / n

        # Top process neurons
        top_indices = counts.argsort()[-5:][::-1].tolist()

        results['layers'][layer_idx] = {
            'gini': float(gini),
            'top_process_neurons': top_indices,
            'usage_std': float(counts.std()),
            'unused_count': int((counts == 0).sum())
        }

    return results


# ============================================================
# Main Analysis
# ============================================================

def comprehensive_analysis(model, val_loader, tokenizer, device):
    """DAWN 모델 종합 분석"""
    print("=" * 60)
    print("DAWN Comprehensive Analysis")
    print("=" * 60)

    results = {
        'timestamp': datetime.now().isoformat(),
        'model_config': model.get_model_stats()
    }

    # 1. Routing Patterns
    routing_results = analyze_routing_patterns(model, val_loader, device)
    results['routing'] = routing_results

    print("\n📊 ROUTING STATISTICS (Soft Weights)")
    print("-" * 40)
    print(f"  Layers: {routing_results['n_layers']}, Input neurons: {routing_results['n_input']}")
    for layer_idx, stats in routing_results['layers'].items():
        lp = stats['learned_params']
        print(f"  Layer {layer_idx}:")
        print(f"    Soft weights: mean={stats['mean_weight']:.4f}±{stats['std_weight']:.4f}")
        print(f"    Low-weight neurons (<0.001): {stats['low_weight_neurons']}")
        print(f"    Learned params: threshold={lp['threshold']:.3f}, steepness={lp['steepness']:.2f}, "
              f"temp={lp['temperature']:.2f}")
        print(f"    Effective k: {lp['effective_k']:.1f} ({lp['effective_k_ratio']:.1%})")

    # 2. Neuron Usage (Load Balance)
    usage_results = analyze_neuron_usage(model, val_loader, device)
    results['usage'] = usage_results

    print("\n🧠 NEURON USAGE (Load Balance)")
    print("-" * 40)
    for layer_idx, stats in usage_results['layers'].items():
        print(f"  Layer {layer_idx}: balance_score={stats['load_balance_score']:.3f}, "
              f"gini={stats['gini_coefficient']:.3f}")

    # 3. Specialization (PMI-based)
    spec_results = analyze_neuron_specialization(
        model, val_loader, tokenizer, device,
        max_batches=100,  # More batches for better PMI estimates
        layer_idx=3  # Middle layer (can be adjusted)
    )
    results['specialization'] = spec_results

    print("\n💎 NEURON SPECIALIZATION (PMI-based)")
    print("-" * 40)
    print(f"  Layer: {spec_results['layer_idx']}")
    print(f"  Analyzed neurons: {spec_results['total_analyzed']}")
    print(f"  Avg specialization (PMI): {spec_results['avg_specialization']:.3f}")
    print(f"  Total tokens: {spec_results['total_tokens_analyzed']:,}")
    if spec_results['specialized_neurons']:
        print(f"  Top specialized neurons:")
        for neuron in spec_results['specialized_neurons'][:5]:
            tokens = [f"{t['token']}({t['pmi']:.2f})" for t in neuron['top_tokens_pmi'][:3]]
            print(f"    Neuron {neuron['neuron_idx']}: {', '.join(tokens)}")

    # 4. Layer Differences
    layer_results = analyze_layer_differences(model, val_loader, device)
    results['layers'] = layer_results

    print("\n📈 LAYER-WISE ANALYSIS")
    print("-" * 40)
    for layer_idx, stats in layer_results['layers'].items():
        print(f"  Layer {layer_idx}: norm={stats['avg_norm']:.2f}, "
              f"std={stats['avg_std']:.4f}")

    # 5. Performance
    perf_results = analyze_performance(model, val_loader, tokenizer, device)
    results['performance'] = perf_results

    print("\n🎯 PERFORMANCE BREAKDOWN")
    print("-" * 40)
    print(f"  Overall accuracy: {perf_results['overall_acc']*100:.2f}%")
    print(f"  Easy samples (top 25%): {perf_results['easy_samples_acc']*100:.2f}%")
    print(f"  Hard samples (bottom 25%): {perf_results['hard_samples_acc']*100:.2f}%")

    # 6. Aux Loss
    aux_results = analyze_aux_loss_components(model, val_loader, device)
    results['aux_loss'] = aux_results

    print("\n⚖️  AUX LOSS ANALYSIS")
    print("-" * 40)
    print(f"  Main loss: {aux_results['avg_main_loss']:.4f}")
    print(f"  Load balance loss: {aux_results['avg_load_balance']:.6f}")
    print(f"  Entropy loss: {aux_results['avg_entropy']:.6f}")
    print(f"  Total aux loss: {aux_results['avg_aux_loss']:.6f}")
    print(f"  Aux/Main ratio: {aux_results['aux_to_main_ratio']:.4f}")

    # 7. Weight Matrices
    weight_results = analyze_weight_matrices(model)
    results['weight_matrices'] = weight_results

    print("\n📐 WEIGHT MATRIX ANALYSIS")
    print("-" * 40)
    for layer_idx, stats in weight_results['layers'].items():
        if 'router' in stats:
            print(f"  Layer {layer_idx}: eff_rank={stats['router']['effective_rank']:.1f}, "
                  f"cond={stats['router']['condition_number']:.1f}, "
                  f"norm={stats['router']['frobenius_norm']:.2f}")

    # 8. Prediction Confidence
    conf_results = analyze_prediction_confidence(model, val_loader, tokenizer, device)
    results['confidence'] = conf_results

    print("\n🎲 PREDICTION CONFIDENCE")
    print("-" * 40)
    print(f"  Avg confidence: {conf_results['avg_confidence']:.3f}")
    print(f"  When correct: {conf_results['confidence_when_correct']:.3f}")
    print(f"  When wrong: {conf_results['confidence_when_wrong']:.3f}")
    print(f"  Calibration error (ECE): {conf_results['calibration_error']:.4f}")
    print(f"  Overconfident ratio: {conf_results['overconfident_ratio']:.3f}")

    # 9. Error Patterns
    error_results = analyze_error_patterns(model, val_loader, tokenizer, device)
    results['error_patterns'] = error_results

    print("\n❌ ERROR PATTERNS")
    print("-" * 40)
    print(f"  Total tokens analyzed: {error_results['total_tokens_analyzed']}")
    if error_results['worst_tokens']:
        print(f"  Worst tokens:")
        for t in error_results['worst_tokens'][:5]:
            print(f"    '{t['token']}': {t['accuracy']*100:.1f}% ({t['count']} samples)")
    print(f"  Position accuracy (start→end):")
    pos_accs = [error_results['position_accuracy'].get(i, 0) for i in range(10)]
    print(f"    {' '.join([f'{a*100:.0f}%' for a in pos_accs])}")

    # 10. Gradient Flow
    grad_results = analyze_gradient_flow(model, val_loader, tokenizer, device)
    results['gradient_flow'] = grad_results

    print("\n🌊 GRADIENT FLOW")
    print("-" * 40)
    for layer_idx, grad_norm in grad_results['layer_gradient_norms'].items():
        router_grad = grad_results['router_gradient_norms'].get(layer_idx, 0)
        print(f"  Layer {layer_idx}: grad={grad_norm:.4f}, router_grad={router_grad:.4f}")
    if 'early_late_ratio' in grad_results:
        print(f"  Early/Late ratio: {grad_results['early_late_ratio']:.2f}")

    # 11. Top Neurons Deep Dive
    top_results = analyze_top_neurons(model, val_loader, tokenizer, device)
    results['top_neurons'] = top_results

    print(f"\n🔬 TOP NEURONS (Layer {top_results['layer_idx']})")
    print("-" * 40)
    for neuron in top_results['top_neurons'][:5]:
        tokens = [t['token'] for t in neuron['top_tokens'][:3]]
        print(f"  Neuron {neuron['neuron_idx']}: {neuron['activation_count']} activations")
        print(f"    Top tokens: {tokens}")

    # 12. Process Neurons
    process_results = analyze_process_neurons(model, val_loader, device)
    results['process_neurons'] = process_results

    if 'error' not in process_results:
        print("\n⚙️  PROCESS NEURONS")
        print("-" * 40)
        for layer_idx, stats in process_results['layers'].items():
            print(f"  Layer {layer_idx}: gini={stats['gini']:.3f}, "
                  f"unused={stats['unused_count']}")

    print("\n" + "=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description='DAWN Model Analysis')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file for data loading')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for analysis')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, device)
    print(f"Model loaded successfully!")

    # Load data from config
    config_path = Path(PROJECT_ROOT) / args.config
    print(f"\nLoading validation data from config: {config_path}")
    val_loader, tokenizer, _ = load_data_from_config(
        config_path=config_path,
        batch_size=args.batch_size
    )
    print(f"Loaded {len(val_loader)} batches")

    # Run analysis
    results = comprehensive_analysis(model, val_loader, tokenizer, device)

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"dawn_analysis_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    results = convert_to_serializable(results)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Analysis complete! Results saved to: {output_path}")


if __name__ == '__main__':
    main()
