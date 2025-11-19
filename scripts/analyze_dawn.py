"""
DAWN (Dynamic Architecture With Neurons) 종합 분석 스크립트

학습된 DAWN 모델의 상세 분석:
- Router entropy 및 routing 패턴
- 뉴런 사용 분포 및 특화
- 레이어별 특성 분석
- 성능 breakdown

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

from models.model import DAWNLanguageModel
from utils.training import CheckpointManager
from utils.data import CacheLoader


# ============================================================
# Data Loading
# ============================================================

def load_data(tokenizer_path="bert-base-uncased", max_length=128, batch_size=64):
    """데이터 로드"""
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, Dataset
    from functools import partial

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load cached texts
    train_texts = CacheLoader.load_train_texts(dataset="wikitext")
    val_texts = CacheLoader.load_validation_texts(dataset="wikitext")

    if train_texts is None or val_texts is None:
        raise ValueError("Cached data not found!")

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

    val_dataset = TextDataset(val_texts, tokenizer, max_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

    return val_loader, tokenizer


# ============================================================
# Model Loading
# ============================================================

def load_checkpoint(checkpoint_path, device='cuda'):
    """체크포인트에서 모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Config 추출
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # 기본값 사용
        config = {
            'vocab_size': 30522,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'max_seq_len': 128,
            'n_input_neurons': 2048,
            'n_process_neurons': 1024,
            'd_routing': 256,
            'dropout': 0.1
        }

    # 모델 생성
    model = DAWNLanguageModel(
        vocab_size=config.get('vocab_size', 30522),
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        max_seq_len=config.get('max_seq_len', 128),
        n_input_neurons=config.get('n_input_neurons', 2048),
        n_process_neurons=config.get('n_process_neurons', 1024),
        d_routing=config.get('d_routing', 256),
        dropout=config.get('dropout', 0.1)
    )

    # 가중치 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, config


# ============================================================
# Analysis Functions
# ============================================================

def analyze_routing_patterns(model, val_loader, device):
    """Router 패턴 분석"""
    print("\n📊 Analyzing Routing Patterns...")

    layer_entropies = defaultdict(list)
    layer_routing_scores = defaultdict(list)

    for batch in tqdm(val_loader, desc="Routing analysis"):
        input_ids = batch['input_ids'].to(device)

        with torch.no_grad():
            # Forward pass to collect routing stats
            _ = model(input_ids)

            # 각 레이어의 routing 정보 수집
            for layer_idx, layer in enumerate(model.layers):
                entropy = layer.ffn.get_routing_entropy()
                layer_entropies[layer_idx].append(entropy)

                if layer.ffn.last_routing_scores is not None:
                    scores = layer.ffn.last_routing_scores.cpu()
                    layer_routing_scores[layer_idx].append(scores)

    # 통계 계산
    n_input = model.layers[0].ffn.n_input
    max_entropy = np.log(n_input)

    results = {
        'max_entropy': max_entropy,
        'layers': {}
    }

    for layer_idx in range(len(model.layers)):
        entropies = layer_entropies[layer_idx]
        avg_entropy = np.mean(entropies)

        results['layers'][layer_idx] = {
            'avg_entropy': avg_entropy,
            'std_entropy': np.std(entropies),
            'entropy_ratio': avg_entropy / max_entropy,
            'min_entropy': np.min(entropies),
            'max_entropy': np.max(entropies)
        }

    return results


def analyze_neuron_usage(model, val_loader, device):
    """뉴런 사용 분포 분석"""
    print("\n🧠 Analyzing Neuron Usage...")

    # Reset counts
    for layer in model.layers:
        layer.ffn.reset_routing_counts()

    # Forward pass to collect usage stats
    for batch in tqdm(val_loader, desc="Usage analysis"):
        input_ids = batch['input_ids'].to(device)

        with torch.no_grad():
            model.train()  # Enable stats collection
            _ = model(input_ids)
            model.eval()

    # 각 레이어의 사용 통계 수집
    results = {'layers': {}}

    for layer_idx, layer in enumerate(model.layers):
        usage_stats = layer.ffn.get_usage_statistics()

        results['layers'][layer_idx] = {
            'total_neurons': usage_stats['total'],
            'dead_neurons': usage_stats['dead_count'],
            'dead_ratio': usage_stats['dead_ratio'],
            'gini_coefficient': usage_stats['gini'],
            'top10_concentration': usage_stats['top10_concentration']
        }

        # 사용량 분포 저장 (첫 레이어만)
        if layer_idx == 0 and usage_stats['usage_counts'] is not None:
            counts = usage_stats['usage_counts']
            sorted_counts = np.sort(counts)[::-1]
            results['layer0_top20_counts'] = sorted_counts[:20].tolist()

    return results


def analyze_neuron_specialization(model, val_loader, tokenizer, device, max_batches=50):
    """뉴런 특화 분석"""
    print("\n💎 Analyzing Neuron Specialization...")

    neuron_tokens = defaultdict(list)
    n_input = model.layers[0].ffn.n_input

    batch_count = 0
    for batch in tqdm(val_loader, desc="Specialization analysis"):
        if batch_count >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)

        with torch.no_grad():
            # Embedding
            B, S = input_ids.shape
            token_emb = model.token_embedding(input_ids)
            positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = model.position_embedding(positions)
            x = token_emb + pos_emb

            # 첫 번째 레이어의 routing 정보
            layer = model.layers[0]
            x_norm = layer.norm1(x)

            # Attention
            attn_out, _ = layer.attention(x_norm, x_norm, x_norm)
            x = x + layer.dropout(attn_out)

            # FFN routing
            x_norm = layer.norm2(x)
            input_idx, _ = layer.ffn.router(x_norm, k_input=1024)

            # 각 배치의 선택된 뉴런과 토큰 매핑
            for b in range(B):
                selected = input_idx[b].cpu().tolist()
                tokens = input_ids[b].cpu().tolist()

                for neuron_id in set(selected):
                    neuron_tokens[neuron_id].extend(tokens)

        batch_count += 1

    # 각 뉴런의 다양성 계산
    diversities = []
    specialized_neurons = []

    for neuron_id in range(n_input):
        tokens = neuron_tokens.get(neuron_id, [])
        if len(tokens) == 0:
            continue

        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        diversity = unique_tokens / total_tokens

        diversities.append(diversity)

        # Diversity < 0.3이면 특화되었다고 판단
        if diversity < 0.3 and total_tokens > 100:
            # 가장 빈번한 토큰들
            token_counts = Counter(tokens)
            top_tokens = token_counts.most_common(5)
            top_words = [(tokenizer.decode([t]), c) for t, c in top_tokens]

            specialized_neurons.append({
                'neuron_id': neuron_id,
                'diversity': diversity,
                'total_tokens': total_tokens,
                'top_tokens': top_words
            })

    results = {
        'total_analyzed': len(diversities),
        'avg_diversity': np.mean(diversities) if diversities else 0,
        'std_diversity': np.std(diversities) if diversities else 0,
        'specialized_count': len(specialized_neurons),
        'specialized_neurons': specialized_neurons[:20]  # Top 20
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
            x = token_emb + pos_emb

            # 각 레이어 통과
            for layer_idx, layer in enumerate(model.layers):
                x = layer(x)

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


def analyze_performance(model, val_loader, device):
    """성능 세부 분석"""
    print("\n🎯 Analyzing Performance...")

    all_losses = []
    all_corrects = []

    for batch in tqdm(val_loader, desc="Performance analysis"):
        input_ids = batch['input_ids'].to(device)
        labels = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            logits = outputs['logits']

            # Per-token loss
            loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            per_token_loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            # Accuracy
            preds = logits.argmax(dim=-1)
            correct = (preds == labels).float()

            all_losses.extend(per_token_loss.cpu().tolist())
            all_corrects.extend(correct.view(-1).cpu().tolist())

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
        }
    }

    return results


def analyze_aux_loss_components(model, val_loader, device):
    """Aux loss 구성 요소 분석"""
    print("\n⚖️  Analyzing Aux Loss Components...")

    # Reset counts
    for layer in model.layers:
        layer.ffn.reset_routing_counts()

    # Forward pass
    total_main_loss = 0
    total_aux_loss = 0
    n_batches = 0

    for batch in tqdm(val_loader, desc="Aux loss analysis"):
        input_ids = batch['input_ids'].to(device)
        labels = input_ids.clone()

        with torch.no_grad():
            model.train()
            outputs = model(input_ids, labels=labels)
            main_loss = outputs['loss']

            # Aux loss 계산
            aux_losses = []
            for layer in model.layers:
                aux_losses.append(layer.ffn.get_load_balance_loss())

            avg_aux = torch.stack(aux_losses).mean()

            total_main_loss += main_loss.item()
            total_aux_loss += avg_aux.item()
            n_batches += 1

            model.eval()

    results = {
        'avg_main_loss': total_main_loss / n_batches,
        'avg_aux_loss': total_aux_loss / n_batches,
        'aux_to_main_ratio': (total_aux_loss / n_batches) / (total_main_loss / n_batches)
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

    print("\n📊 ROUTING STATISTICS")
    print("-" * 40)
    for layer_idx, stats in routing_results['layers'].items():
        print(f"  Layer {layer_idx}: entropy={stats['avg_entropy']:.3f} "
              f"({stats['entropy_ratio']*100:.1f}% of max)")

    # 2. Neuron Usage
    usage_results = analyze_neuron_usage(model, val_loader, device)
    results['usage'] = usage_results

    print("\n🧠 NEURON USAGE")
    print("-" * 40)
    for layer_idx, stats in usage_results['layers'].items():
        print(f"  Layer {layer_idx}: dead={stats['dead_neurons']}/{stats['total_neurons']} "
              f"({stats['dead_ratio']*100:.1f}%), "
              f"gini={stats['gini_coefficient']:.3f}")

    # 3. Specialization
    spec_results = analyze_neuron_specialization(model, val_loader, tokenizer, device)
    results['specialization'] = spec_results

    print("\n💎 NEURON SPECIALIZATION")
    print("-" * 40)
    print(f"  Specialized neurons: {spec_results['specialized_count']}")
    print(f"  Average diversity: {spec_results['avg_diversity']:.3f}")
    if spec_results['specialized_neurons']:
        print("  Examples:")
        for neuron in spec_results['specialized_neurons'][:3]:
            tokens_str = ", ".join([f"'{w}'" for w, c in neuron['top_tokens'][:3]])
            print(f"    Neuron {neuron['neuron_id']}: {tokens_str}")

    # 4. Layer Differences
    layer_results = analyze_layer_differences(model, val_loader, device)
    results['layers'] = layer_results

    print("\n📈 LAYER-WISE ANALYSIS")
    print("-" * 40)
    for layer_idx, stats in layer_results['layers'].items():
        print(f"  Layer {layer_idx}: norm={stats['avg_norm']:.2f}, "
              f"std={stats['avg_std']:.4f}")

    # 5. Performance
    perf_results = analyze_performance(model, val_loader, device)
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
    print(f"  Aux loss: {aux_results['avg_aux_loss']:.6f}")
    print(f"  Aux/Main ratio: {aux_results['aux_to_main_ratio']:.4f}")

    print("\n" + "=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description='DAWN Model Analysis')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for analysis')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, device)
    print(f"Model loaded successfully!")

    # Load data
    print(f"\nLoading validation data...")
    val_loader, tokenizer = load_data(
        max_length=args.max_length,
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
