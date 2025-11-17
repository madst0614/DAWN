"""
Comprehensive Analysis for Neuron-Based SPROUT Model

Implements all 7 analysis categories:
1. Neuron usage patterns
2. Performance vs sparsity tradeoff
3. Router quality analysis
4. Neuron specialization
5. Layer-wise differences
6. Dynamic routing effects
7. Computation efficiency
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from collections import defaultdict
import numpy as np
import argparse
import time
from typing import Dict, List, Tuple

from src.models.sprout_neuron_based import NeuronBasedLanguageModel


# ============================================================
# 1. Neuron Usage Patterns (가장 중요!)
# ============================================================

def analyze_neuron_usage_patterns(
    model: nn.Module,
    tokenizer,
    device: str,
    num_samples: int = 1000,
    top_k: int = 768
) -> Dict:
    """
    뉴런 사용 패턴 분석 - 불균형 체크

    Returns:
        stats: 각 레이어별 사용 통계
    """
    print("\n" + "="*70)
    print("1️⃣  NEURON USAGE PATTERNS ANALYSIS")
    print("="*70)

    model.eval()

    # 다양한 테스트 문장
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Artificial intelligence is transforming the world",
        "Python is a popular programming language",
        "Machine learning requires large datasets",
        "Deep neural networks learn hierarchical representations",
        "Natural language processing enables computers to understand text",
        "The weather today is sunny and warm",
        "I love reading books in my free time",
        "Mathematics is the foundation of science",
        "Climate change is a global challenge",
    ] * 100  # 1000 samples

    test_texts = test_texts[:num_samples]

    n_layers = len(model.layers)
    d_ff = model.layers[0].ffn.d_ff

    # 각 레이어별 뉴런 사용 카운트
    layer_usage = {i: torch.zeros(d_ff, device=device) for i in range(n_layers)}

    print(f"\nAnalyzing {num_samples} samples with top_k={top_k} ({top_k/d_ff*100:.1f}%)")

    with torch.no_grad():
        for text in test_texts:
            tokens = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=32,
                truncation=True
            )['input_ids'].to(device)

            batch, seq = tokens.shape

            # Forward pass 수동으로 진행하며 라우팅 기록
            token_emb = model.token_embedding(tokens)
            positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
            pos_emb = model.position_embedding(positions)
            x = token_emb + pos_emb

            for layer_idx, layer in enumerate(model.layers):
                # Attention
                x_norm = layer.norm1(x)
                attn_out, _ = layer.attention(x_norm, x_norm, x_norm)
                x = x + layer.dropout(attn_out)

                # FFN - 라우터 점수 계산
                x_norm = layer.norm2(x)
                x_flat = x_norm.view(-1, x_norm.shape[-1])

                # Router scores
                router_scores = x_flat @ layer.ffn.router.W_router.T
                _, top_indices = torch.topk(router_scores, top_k, dim=-1)

                # 사용된 뉴런 기록
                for indices in top_indices:
                    layer_usage[layer_idx][indices] += 1

                # FFN 통과
                x_ffn = layer.ffn(x_norm, top_k=top_k)
                x = x + layer.dropout(x_ffn)

    # 분석 결과
    results = {}
    total_positions = num_samples * 32  # seq_len = 32

    print(f"\n{'Layer':<8} {'Never':<8} {'Rare(<10%)':<12} {'Common(≥10%)':<15} {'Mean':<10} {'Max':<10} {'Gini':<10}")
    print("-" * 85)

    for layer_idx in range(n_layers):
        usage = layer_usage[layer_idx]

        # 통계
        n_never = (usage == 0).sum().item()
        n_rare = ((usage > 0) & (usage < total_positions * 0.1)).sum().item()
        n_common = (usage >= total_positions * 0.1).sum().item()

        mean_usage = usage.mean().item()
        max_usage = usage.max().item()

        # Gini coefficient (불균형 지수, 0=완전 균등, 1=완전 불균등)
        usage_sorted = torch.sort(usage)[0]
        n = len(usage_sorted)
        index = torch.arange(1, n + 1, device=device, dtype=torch.float)
        gini = (2 * (index * usage_sorted).sum() / (n * usage_sorted.sum()) - (n + 1) / n).item()

        results[layer_idx] = {
            'never': n_never,
            'rare': n_rare,
            'common': n_common,
            'mean': mean_usage,
            'max': max_usage,
            'gini': gini,
            'usage_distribution': usage.cpu()
        }

        print(f"Layer {layer_idx:<2} {n_never:<8} {n_rare:<12} {n_common:<15} {mean_usage:<10.1f} {max_usage:<10.0f} {gini:<10.4f}")

    # 종합 평가
    print("\n📊 Usage Pattern Evaluation:")
    avg_gini = np.mean([r['gini'] for r in results.values()])
    avg_never = np.mean([r['never'] / d_ff for r in results.values()])

    if avg_gini < 0.3 and avg_never < 0.1:
        print(f"✅ EXCELLENT: Balanced usage (Gini={avg_gini:.3f}, Never={avg_never:.1%})")
    elif avg_gini < 0.5 and avg_never < 0.2:
        print(f"✔️  GOOD: Acceptable usage (Gini={avg_gini:.3f}, Never={avg_never:.1%})")
    else:
        print(f"⚠️  WARNING: Unbalanced usage (Gini={avg_gini:.3f}, Never={avg_never:.1%})")

    return results


# ============================================================
# 2. Performance vs Sparsity Tradeoff
# ============================================================

def analyze_performance_vs_sparsity(
    model: nn.Module,
    tokenizer,
    device: str,
    test_texts: List[str] = None
) -> Dict:
    """
    성능 vs 희소도 트레이드오프 분석
    """
    print("\n" + "="*70)
    print("2️⃣  PERFORMANCE VS SPARSITY TRADEOFF")
    print("="*70)

    model.eval()

    if test_texts is None:
        test_texts = [
            "The [MASK] is shining brightly in the sky",
            "I love to [MASK] books in my free time",
            "Python is a programming [MASK] for data science",
            "She went to the [MASK] to buy groceries",
            "The cat is [MASK] on the comfortable mat",
            "Artificial [MASK] is transforming technology",
            "The capital of France is [MASK]",
            "Machine [MASK] requires large datasets",
            "Deep [MASK] networks learn from data",
            "The weather today is very [MASK]",
        ]

    d_ff = model.layers[0].ffn.d_ff

    # 테스트할 희소도 레벨
    sparsity_levels = [
        (None, "Dense (100%)"),
        (int(0.75 * d_ff), "75%"),
        (int(0.50 * d_ff), "50%"),
        (int(0.25 * d_ff), "25%"),
        (int(0.10 * d_ff), "10%"),
        (int(0.05 * d_ff), "5%"),
    ]

    results = {}

    # Dense 출력 (ground truth)
    print("\nComputing dense outputs as ground truth...")
    dense_outputs = []
    dense_correct = 0

    with torch.no_grad():
        for text in test_texts:
            tokens = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=32,
                truncation=True
            ).to(device)

            input_ids = tokens['input_ids']
            outputs = model(input_ids, top_k=None)
            logits = outputs['logits']

            dense_outputs.append(logits)

            # MLM accuracy (if [MASK] exists)
            mask_pos = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
            if len(mask_pos[0]) > 0:
                mask_logits = logits[mask_pos[0][0], mask_pos[1][0]]
                pred_token = tokenizer.decode([mask_logits.argmax()])
                # Simple check (rough)
                dense_correct += 1

    print(f"Dense baseline: {len(test_texts)} samples")

    # 각 희소도 레벨 테스트
    print(f"\n{'Sparsity':<15} {'MSE Loss':<12} {'Cosine Sim':<12} {'Norm %':<12} {'Quality':<10}")
    print("-" * 70)

    for top_k, label in sparsity_levels:
        mse_losses = []
        cos_sims = []
        norm_ratios = []

        with torch.no_grad():
            for i, text in enumerate(test_texts):
                tokens = tokenizer(
                    text,
                    return_tensors='pt',
                    padding='max_length',
                    max_length=32,
                    truncation=True
                ).to(device)

                outputs = model(tokens['input_ids'], top_k=top_k)
                logits_sparse = outputs['logits']
                logits_dense = dense_outputs[i]

                # Metrics
                mse = F.mse_loss(logits_sparse, logits_dense).item()

                flat_sparse = logits_sparse.flatten()
                flat_dense = logits_dense.flatten()
                cos_sim = F.cosine_similarity(
                    flat_sparse.unsqueeze(0),
                    flat_dense.unsqueeze(0)
                ).item()

                norm_ratio = (logits_sparse.norm() / logits_dense.norm()).item()

                mse_losses.append(mse)
                cos_sims.append(cos_sim)
                norm_ratios.append(norm_ratio)

        avg_mse = np.mean(mse_losses)
        avg_cos = np.mean(cos_sims)
        avg_norm = np.mean(norm_ratios) * 100

        # Quality rating
        if avg_cos > 0.98 and avg_norm > 95:
            quality = "🌟 Excellent"
        elif avg_cos > 0.95 and avg_norm > 90:
            quality = "✅ Good"
        elif avg_cos > 0.90 and avg_norm > 80:
            quality = "✔️  OK"
        else:
            quality = "⚠️  Poor"

        results[label] = {
            'top_k': top_k,
            'mse': avg_mse,
            'cosine_sim': avg_cos,
            'norm_ratio': avg_norm,
            'quality': quality
        }

        print(f"{label:<15} {avg_mse:<12.6f} {avg_cos:<12.6f} {avg_norm:<12.1f} {quality:<10}")

    # 추천 희소도
    print("\n🎯 Recommended Sparsity:")
    for label, stats in results.items():
        if "Good" in stats['quality'] or "Excellent" in stats['quality']:
            sparsity_pct = (stats['top_k'] / d_ff * 100) if stats['top_k'] else 100
            print(f"  • {label}: Quality={stats['quality']}, Cosine={stats['cosine_sim']:.4f}")
            if stats['top_k'] and sparsity_pct < 60:
                print(f"    → Can use {sparsity_pct:.0f}% sparsity without significant degradation!")
                break

    return results


# ============================================================
# 3. Router Quality Analysis
# ============================================================

def analyze_router_quality(
    model: nn.Module,
    tokenizer,
    device: str,
    num_samples: int = 100,
    top_k: int = 768
) -> Dict:
    """
    라우터 품질 분석: Learned vs Random vs Oracle
    """
    print("\n" + "="*70)
    print("3️⃣  ROUTER QUALITY ANALYSIS")
    print("="*70)

    model.eval()

    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python programming language is widely used in data science",
    ] * 34
    test_texts = test_texts[:num_samples]

    layer_idx = 0  # Analyze first layer
    ffn = model.layers[layer_idx].ffn
    d_ff = ffn.d_ff

    print(f"\nAnalyzing Layer {layer_idx} with {num_samples} samples")
    print(f"Sparsity: top_k={top_k} ({top_k/d_ff*100:.1f}%)\n")

    results = {}

    # 1. Learned Router
    print("Testing Learned Router...")
    learned_mse = []

    with torch.no_grad():
        for text in test_texts:
            tokens = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=32,
                truncation=True
            )['input_ids'].to(device)

            batch, seq = tokens.shape
            token_emb = model.token_embedding(tokens)
            positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
            pos_emb = model.position_embedding(positions)
            x = token_emb + pos_emb  # [batch, seq, d_model]

            # Dense output (ground truth)
            out_dense = ffn(x, top_k=None)

            # Learned router output
            out_learned = ffn(x, top_k=top_k)

            mse = F.mse_loss(out_learned, out_dense).item()
            learned_mse.append(mse)

    results['learned'] = {
        'mse': np.mean(learned_mse),
        'std': np.std(learned_mse)
    }

    # 2. Random Router
    print("Testing Random Router...")
    random_mse = []

    # 임시로 랜덤 라우터 생성
    original_router = ffn.router.W_router.data.clone()
    ffn.router.W_router.data = torch.randn_like(ffn.router.W_router.data) * 0.02

    with torch.no_grad():
        for text in test_texts:
            tokens = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=32,
                truncation=True
            )['input_ids'].to(device)

            batch, seq = tokens.shape
            token_emb = model.token_embedding(tokens)
            positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
            pos_emb = model.position_embedding(positions)
            x = token_emb + pos_emb

            out_dense = ffn(x, top_k=None)
            out_random = ffn(x, top_k=top_k)

            mse = F.mse_loss(out_random, out_dense).item()
            random_mse.append(mse)

    results['random'] = {
        'mse': np.mean(random_mse),
        'std': np.std(random_mse)
    }

    # Restore original router
    ffn.router.W_router.data = original_router

    # 3. Oracle Router (최적 선택)
    print("Testing Oracle Router...")
    oracle_mse = []

    with torch.no_grad():
        for text in test_texts:
            tokens = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=32,
                truncation=True
            )['input_ids'].to(device)

            batch, seq = tokens.shape
            token_emb = model.token_embedding(tokens)
            positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
            pos_emb = model.position_embedding(positions)
            x = token_emb + pos_emb
            x_flat = x.view(-1, x.shape[-1])

            # Dense 계산
            z_dense = x_flat @ ffn.W1.T  # [batch*seq, d_ff]
            a_dense = F.gelu(z_dense)
            out_dense = (a_dense @ ffn.W2.T).view(batch, seq, -1)

            # Oracle: 실제 activation 크기로 top-k 선택
            activation_scores = a_dense.abs()  # 실제 activation 크기
            _, top_indices = torch.topk(activation_scores, top_k, dim=-1)

            mask = torch.zeros_like(z_dense)
            mask.scatter_(-1, top_indices, 1.0)

            z_oracle = z_dense * mask
            a_oracle = F.gelu(z_oracle)
            out_oracle = (a_oracle @ ffn.W2.T).view(batch, seq, -1)

            mse = F.mse_loss(out_oracle, out_dense).item()
            oracle_mse.append(mse)

    results['oracle'] = {
        'mse': np.mean(oracle_mse),
        'std': np.std(oracle_mse)
    }

    # 결과 출력
    print(f"\n{'Router Type':<15} {'MSE Loss':<15} {'Std Dev':<15} {'Quality':<15}")
    print("-" * 65)

    for router_type in ['oracle', 'learned', 'random']:
        mse = results[router_type]['mse']
        std = results[router_type]['std']

        if router_type == 'oracle':
            quality = "🌟 Upper bound"
        elif router_type == 'learned':
            # Compare to oracle
            gap = (mse - results['oracle']['mse']) / results['oracle']['mse']
            if gap < 0.1:
                quality = "✅ Excellent"
            elif gap < 0.3:
                quality = "✔️  Good"
            else:
                quality = "⚠️  Needs training"
        else:  # random
            quality = "❌ Baseline"

        results[router_type]['quality'] = quality
        print(f"{router_type.capitalize():<15} {mse:<15.6f} {std:<15.6f} {quality:<15}")

    # 학습 효과
    improvement = (results['random']['mse'] - results['learned']['mse']) / results['random']['mse'] * 100
    gap_to_oracle = (results['learned']['mse'] - results['oracle']['mse']) / results['oracle']['mse'] * 100

    print(f"\n📈 Router Learning Effect:")
    print(f"  • Improvement over random: {improvement:.1f}%")
    print(f"  • Gap to oracle: {gap_to_oracle:.1f}%")

    if improvement > 30 and gap_to_oracle < 20:
        print(f"  ✅ Router is well-trained!")
    elif improvement > 10:
        print(f"  ✔️  Router learned something useful")
    else:
        print(f"  ⚠️  Router needs more training")

    return results


# ============================================================
# 4. Neuron Specialization Analysis
# ============================================================

def analyze_neuron_specialization(
    model: nn.Module,
    tokenizer,
    device: str,
    layer_idx: int = 0
) -> Dict:
    """
    뉴런 특화도 분석: 각 뉴런이 어떤 입력에 반응하는지
    """
    print("\n" + "="*70)
    print("4️⃣  NEURON SPECIALIZATION ANALYSIS")
    print("="*70)

    model.eval()

    # 카테고리별 테스트 문장
    test_groups = {
        "Science": [
            "Physics studies matter and energy in the universe",
            "Chemistry explores molecular structures and reactions",
            "Biology examines living organisms and ecosystems",
        ],
        "Technology": [
            "Computers process information using binary code",
            "Software development requires programming skills",
            "Artificial intelligence mimics human cognition",
        ],
        "Arts": [
            "Painting expresses emotions through visual art",
            "Music creates harmony using different instruments",
            "Literature tells stories through written words",
        ],
        "Numbers": [
            "One plus one equals two in mathematics",
            "The year two thousand and twenty four",
            "Three hundred and sixty degrees in a circle",
        ],
    }

    ffn = model.layers[layer_idx].ffn
    d_ff = ffn.d_ff

    print(f"\nAnalyzing Layer {layer_idx} with {len(test_groups)} categories")

    # 각 카테고리별 뉴런 활성화 패턴
    category_activations = {}

    with torch.no_grad():
        for category, texts in test_groups.items():
            activations = []

            for text in texts:
                tokens = tokenizer(
                    text,
                    return_tensors='pt',
                    padding='max_length',
                    max_length=32,
                    truncation=True
                )['input_ids'].to(device)

                batch, seq = tokens.shape
                token_emb = model.token_embedding(tokens)
                positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
                pos_emb = model.position_embedding(positions)
                x = token_emb + pos_emb
                x_flat = x.view(-1, x.shape[-1])

                # FFN activation
                z = x_flat @ ffn.W1.T  # [batch*seq, d_ff]
                a = F.gelu(z)

                # Average activation per neuron
                avg_activation = a.mean(dim=0)  # [d_ff]
                activations.append(avg_activation)

            # Category average
            category_activations[category] = torch.stack(activations).mean(dim=0)

    # 각 뉴런의 특화도 계산
    print(f"\n{'Category':<15} {'Specialized':<15} {'Top Neurons':<30}")
    print("-" * 65)

    results = {}
    all_activations = torch.stack(list(category_activations.values()))  # [n_categories, d_ff]

    for cat_idx, (category, activation) in enumerate(category_activations.items()):
        # 이 카테고리에서만 강하게 활성화되는 뉴런
        # Specialization = activation_this / activation_others
        other_activations = all_activations[[i for i in range(len(all_activations)) if i != cat_idx]]
        max_other = other_activations.max(dim=0)[0]

        specialization = activation / (max_other + 1e-8)
        specialized_neurons = (specialization > 1.5).sum().item()

        # Top 5 specialized neurons
        top_5_spec, top_5_idx = torch.topk(specialization, 5)

        results[category] = {
            'specialized_count': specialized_neurons,
            'top_neurons': top_5_idx.tolist(),
            'specialization_scores': top_5_spec.tolist()
        }

        print(f"{category:<15} {specialized_neurons:<15} {top_5_idx.tolist()}")

    # 공통 뉴런 vs 특화 뉴런
    all_top_neurons = set()
    for res in results.values():
        all_top_neurons.update(res['top_neurons'])

    print(f"\n📊 Specialization Summary:")
    print(f"  • Total unique specialized neurons: {len(all_top_neurons)}")
    print(f"  • Average per category: {np.mean([r['specialized_count'] for r in results.values()]):.1f}")

    if len(all_top_neurons) > d_ff * 0.3:
        print(f"  ✅ Good diversity: neurons are specialized for different inputs")
    else:
        print(f"  ⚠️  Low diversity: many neurons overlap across categories")

    return results


# ============================================================
# 5. Layer-wise Differences
# ============================================================

def analyze_layer_differences(
    model: nn.Module,
    tokenizer,
    device: str,
    num_samples: int = 100,
    top_k: int = 768
) -> Dict:
    """
    레이어별 차이 분석
    """
    print("\n" + "="*70)
    print("5️⃣  LAYER-WISE DIFFERENCES ANALYSIS")
    print("="*70)

    model.eval()

    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming artificial intelligence",
    ] * 50
    test_texts = test_texts[:num_samples]

    n_layers = len(model.layers)
    d_ff = model.layers[0].ffn.d_ff

    print(f"\nAnalyzing {n_layers} layers with {num_samples} samples")

    # 각 레이어별 통계
    layer_stats = {}

    with torch.no_grad():
        for text in test_texts:
            tokens = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=32,
                truncation=True
            )['input_ids'].to(device)

            batch, seq = tokens.shape
            token_emb = model.token_embedding(tokens)
            positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
            pos_emb = model.position_embedding(positions)
            x = token_emb + pos_emb

            for layer_idx, layer in enumerate(model.layers):
                if layer_idx not in layer_stats:
                    layer_stats[layer_idx] = {
                        'router_entropy': [],
                        'activation_sparsity': [],
                        'output_norm': []
                    }

                # Attention
                x_norm = layer.norm1(x)
                attn_out, _ = layer.attention(x_norm, x_norm, x_norm)
                x = x + layer.dropout(attn_out)

                # FFN
                x_norm = layer.norm2(x)
                x_flat = x_norm.view(-1, x_norm.shape[-1])

                # Router scores
                router_scores = x_flat @ layer.ffn.router.W_router.T
                router_probs = F.softmax(router_scores, dim=-1)

                # Entropy (분산도)
                entropy = -(router_probs * (router_probs + 1e-8).log()).sum(dim=-1).mean().item()
                layer_stats[layer_idx]['router_entropy'].append(entropy)

                # Activation sparsity (실제 0인 비율)
                z = x_flat @ layer.ffn.W1.T
                a = F.gelu(z)
                sparsity = (a.abs() < 0.01).float().mean().item()
                layer_stats[layer_idx]['activation_sparsity'].append(sparsity)

                # Output norm
                x_ffn = layer.ffn(x_norm, top_k=top_k)
                norm = x_ffn.norm().item()
                layer_stats[layer_idx]['output_norm'].append(norm)

                x = x + layer.dropout(x_ffn)

    # 결과 출력
    print(f"\n{'Layer':<8} {'Entropy':<12} {'Act Sparsity':<15} {'Output Norm':<15} {'Pattern':<15}")
    print("-" * 75)

    results = {}
    for layer_idx in range(n_layers):
        entropy = np.mean(layer_stats[layer_idx]['router_entropy'])
        sparsity = np.mean(layer_stats[layer_idx]['activation_sparsity'])
        norm = np.mean(layer_stats[layer_idx]['output_norm'])

        # Pattern classification
        if layer_idx < n_layers // 3:
            pattern = "🔵 Early (syntax)"
        elif layer_idx < 2 * n_layers // 3:
            pattern = "🟢 Middle (semantic)"
        else:
            pattern = "🔴 Late (abstract)"

        results[layer_idx] = {
            'entropy': entropy,
            'sparsity': sparsity,
            'norm': norm,
            'pattern': pattern
        }

        print(f"Layer {layer_idx:<2} {entropy:<12.4f} {sparsity:<15.4f} {norm:<15.2f} {pattern:<15}")

    # 레이어 간 유사도
    print(f"\n🔗 Layer Similarity (Router Weight Cosine):")
    for i in range(n_layers - 1):
        w_i = model.layers[i].ffn.router.W_router.data
        w_j = model.layers[i + 1].ffn.router.W_router.data

        # Flatten and compute cosine
        sim = F.cosine_similarity(w_i.flatten().unsqueeze(0), w_j.flatten().unsqueeze(0)).item()
        print(f"  Layer {i} ↔ {i+1}: {sim:.4f}")

    return results


# ============================================================
# 6. Dynamic Routing Effects
# ============================================================

def analyze_dynamic_routing(
    model: nn.Module,
    tokenizer,
    device: str,
    top_k: int = 768
) -> Dict:
    """
    동적 라우팅 효과 분석: 컨텍스트에 따른 변화
    """
    print("\n" + "="*70)
    print("6️⃣  DYNAMIC ROUTING EFFECTS ANALYSIS")
    print("="*70)

    model.eval()

    # 같은 단어, 다른 컨텍스트
    test_pairs = [
        {
            "word": "bank",
            "context1": "I went to the bank to deposit money",
            "context2": "We sat by the river bank enjoying nature"
        },
        {
            "word": "light",
            "context1": "The light from the sun is bright",
            "context2": "This bag is very light to carry"
        },
        {
            "word": "play",
            "context1": "Children love to play in the park",
            "context2": "We watched a play at the theater"
        },
    ]

    layer_idx = 0
    ffn = model.layers[layer_idx].ffn

    print(f"\nAnalyzing Layer {layer_idx} routing for ambiguous words\n")

    results = {}

    for pair in test_pairs:
        word = pair['word']
        context1 = pair['context1']
        context2 = pair['context2']

        # Tokenize both contexts
        tokens1 = tokenizer(
            context1,
            return_tensors='pt',
            padding='max_length',
            max_length=32,
            truncation=True
        )['input_ids'].to(device)

        tokens2 = tokenizer(
            context2,
            return_tensors='pt',
            padding='max_length',
            max_length=32,
            truncation=True
        )['input_ids'].to(device)

        with torch.no_grad():
            # Context 1
            batch, seq = tokens1.shape
            token_emb1 = model.token_embedding(tokens1)
            positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
            pos_emb = model.position_embedding(positions)
            x1 = token_emb1 + pos_emb
            x1_flat = x1.view(-1, x1.shape[-1])

            scores1 = x1_flat @ ffn.router.W_router.T
            _, top_indices1 = torch.topk(scores1, top_k, dim=-1)

            # Context 2
            token_emb2 = model.token_embedding(tokens2)
            x2 = token_emb2 + pos_emb
            x2_flat = x2.view(-1, x2.shape[-1])

            scores2 = x2_flat @ ffn.router.W_router.T
            _, top_indices2 = torch.topk(scores2, top_k, dim=-1)

            # 선택된 뉴런 집합
            set1 = set(top_indices1.flatten().tolist())
            set2 = set(top_indices2.flatten().tolist())

            overlap = len(set1 & set2)
            union = len(set1 | set2)
            jaccard = overlap / union if union > 0 else 0

            unique1 = len(set1 - set2)
            unique2 = len(set2 - set1)

        results[word] = {
            'context1': context1,
            'context2': context2,
            'overlap': overlap,
            'jaccard': jaccard,
            'unique1': unique1,
            'unique2': unique2
        }

        print(f"Word: '{word}'")
        print(f"  Context 1: {context1}")
        print(f"  Context 2: {context2}")
        print(f"  Overlap: {overlap} neurons ({jaccard:.2%} Jaccard)")
        print(f"  Unique to context 1: {unique1}")
        print(f"  Unique to context 2: {unique2}")
        print()

    # 평가
    avg_jaccard = np.mean([r['jaccard'] for r in results.values()])

    print(f"📊 Dynamic Routing Evaluation:")
    print(f"  Average Jaccard similarity: {avg_jaccard:.2%}")

    if avg_jaccard < 0.7:
        print(f"  ✅ Strong context sensitivity: routing adapts to context")
    elif avg_jaccard < 0.85:
        print(f"  ✔️  Moderate context sensitivity")
    else:
        print(f"  ⚠️  Weak context sensitivity: routing is mostly static")

    return results


# ============================================================
# 7. Computation Efficiency
# ============================================================

def analyze_computation_efficiency(
    model: nn.Module,
    tokenizer,
    device: str,
    num_samples: int = 100
) -> Dict:
    """
    계산 효율성 분석: 실제 속도 향상
    """
    print("\n" + "="*70)
    print("7️⃣  COMPUTATION EFFICIENCY ANALYSIS")
    print("="*70)

    model.eval()

    # Test text
    test_text = "The quick brown fox jumps over the lazy dog"
    tokens = tokenizer(
        test_text,
        return_tensors='pt',
        padding='max_length',
        max_length=32,
        truncation=True
    )['input_ids'].to(device)

    d_ff = model.layers[0].ffn.d_ff

    # 테스트할 희소도 레벨
    sparsity_levels = [
        (None, "Dense (100%)"),
        (int(0.50 * d_ff), "50%"),
        (int(0.25 * d_ff), "25%"),
        (int(0.10 * d_ff), "10%"),
    ]

    results = {}

    print(f"\nBenchmarking with {num_samples} iterations per config\n")
    print(f"{'Sparsity':<15} {'Time (ms)':<15} {'Speedup':<12} {'FLOPs %':<12}")
    print("-" * 60)

    baseline_time = None

    for top_k, label in sparsity_levels:
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(tokens, top_k=top_k)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(num_samples):
                _ = model(tokens, top_k=top_k)

        if device == 'cuda':
            torch.cuda.synchronize()

        elapsed = (time.time() - start) * 1000 / num_samples  # ms per sample

        if baseline_time is None:
            baseline_time = elapsed

        speedup = baseline_time / elapsed

        # FLOPs 추정
        if top_k is None:
            flops_pct = 100.0
        else:
            flops_pct = top_k / d_ff * 100

        results[label] = {
            'top_k': top_k,
            'time_ms': elapsed,
            'speedup': speedup,
            'flops_pct': flops_pct
        }

        print(f"{label:<15} {elapsed:<15.2f} {speedup:<12.2f}x {flops_pct:<12.1f}%")

    # 메모리 사용량 (if CUDA)
    if device == 'cuda':
        print(f"\n💾 GPU Memory Usage:")
        for top_k, label in sparsity_levels:
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                _ = model(tokens, top_k=top_k)

            peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            results[label]['memory_mb'] = peak_mem
            print(f"  {label:<15}: {peak_mem:.1f} MB")

    # 효율성 평가
    print(f"\n⚡ Efficiency Summary:")
    best_efficiency = None
    for label, stats in results.items():
        if stats['top_k'] is not None and stats['speedup'] > 1.0:
            efficiency = stats['speedup'] / (stats['flops_pct'] / 100)
            if best_efficiency is None or efficiency > best_efficiency:
                best_efficiency = efficiency
                print(f"  🌟 Best: {label} - {stats['speedup']:.2f}x speedup at {stats['flops_pct']:.0f}% FLOPs")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--analyses", type=str, default="all",
                       help="Comma-separated list of analyses (1-7) or 'all'")
    args = parser.parse_args()

    print("="*70)
    print("🔬 COMPREHENSIVE NEURON-BASED MODEL ANALYSIS")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")

    # Load checkpoint
    print("\n📂 Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_args = checkpoint.get('args', {})

    # Create model
    model = NeuronBasedLanguageModel(
        vocab_size=model_args.get('vocab_size', 30522),
        d_model=model_args.get('d_model', 768),
        d_ff=model_args.get('d_ff', 3072),
        n_layers=model_args.get('n_layers', 12),
        n_heads=model_args.get('n_heads', 12),
        max_seq_len=model_args.get('max_seq_len', 128),
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded (Epoch: {checkpoint.get('epoch', '?')}, Step: {checkpoint.get('global_step', '?')})")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Parse which analyses to run
    if args.analyses == "all":
        analyses_to_run = list(range(1, 8))
    else:
        analyses_to_run = [int(x) for x in args.analyses.split(",")]

    # Run analyses
    all_results = {}

    if 1 in analyses_to_run:
        all_results['usage_patterns'] = analyze_neuron_usage_patterns(
            model, tokenizer, args.device, num_samples=1000, top_k=768
        )

    if 2 in analyses_to_run:
        all_results['performance_sparsity'] = analyze_performance_vs_sparsity(
            model, tokenizer, args.device
        )

    if 3 in analyses_to_run:
        all_results['router_quality'] = analyze_router_quality(
            model, tokenizer, args.device, num_samples=100, top_k=768
        )

    if 4 in analyses_to_run:
        all_results['specialization'] = analyze_neuron_specialization(
            model, tokenizer, args.device, layer_idx=0
        )

    if 5 in analyses_to_run:
        all_results['layer_differences'] = analyze_layer_differences(
            model, tokenizer, args.device, num_samples=100, top_k=768
        )

    if 6 in analyses_to_run:
        all_results['dynamic_routing'] = analyze_dynamic_routing(
            model, tokenizer, args.device, top_k=768
        )

    if 7 in analyses_to_run:
        all_results['efficiency'] = analyze_computation_efficiency(
            model, tokenizer, args.device, num_samples=100
        )

    print("\n" + "="*70)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAnalyzed {len(analyses_to_run)} categories")
    print(f"Results saved in memory (extend script to save to file if needed)")


if __name__ == "__main__":
    main()
