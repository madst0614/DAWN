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
    """Router 패턴 분석 - 각 레이어의 라우팅 통계"""
    print("\n📊 Analyzing Routing Patterns...")

    n_layers = len(model.layers)
    n_input = model.layers[0].block.n_input

    # Track neuron selection counts per layer
    layer_selection_counts = [torch.zeros(n_input, device=device) for _ in range(n_layers)]
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
                # Get router output
                k_input = layer.block.n_input // 2
                indices, weights, context = layer.block.router(x, k_input)

                # Count selections
                for b in range(batch_size):
                    layer_selection_counts[layer_idx].scatter_add_(
                        0, indices[b], torch.ones(k_input, device=device)
                    )

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
        counts = layer_selection_counts[layer_idx].cpu().numpy()
        usage_ratio = counts / (total_samples * (n_input // 2))

        results['layers'][layer_idx] = {
            'mean_usage': float(usage_ratio.mean()),
            'std_usage': float(usage_ratio.std()),
            'min_usage': float(usage_ratio.min()),
            'max_usage': float(usage_ratio.max()),
            'unused_neurons': int((counts == 0).sum()),
            'top_5_neurons': counts.argsort()[-5:][::-1].tolist()
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
                k_input = layer.block.n_input // 2
                _, weights, _, _ = layer.block.router(x, k_input)

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


def analyze_neuron_specialization(model, val_loader, tokenizer, device, max_batches=30):
    """뉴런 특화 분석 - 어떤 토큰이 어떤 뉴런을 활성화하는지"""
    print("\n💎 Analyzing Neuron Specialization...")

    n_input = model.layers[0].block.n_input

    # Track which tokens activate which neurons (first layer only for simplicity)
    neuron_token_counts = defaultdict(lambda: defaultdict(int))
    total_activations = defaultdict(int)

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Specialization analysis")):
        if batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)

        with torch.no_grad():
            B, S = input_ids.shape
            token_emb = model.token_embedding(input_ids)
            positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = model.position_embedding(positions)
            x = model.dropout(token_emb + pos_emb)

            # Analyze first layer routing
            layer = model.layers[0]
            k_input = layer.block.n_input // 2
            indices, _, _, _ = layer.block.router(x, k_input)

            # Track token-neuron associations
            for b in range(B):
                for neuron_idx in indices[b].cpu().tolist():
                    total_activations[neuron_idx] += 1
                    # Sample some tokens from this batch
                    for token_id in input_ids[b, :10].cpu().tolist():  # First 10 tokens
                        neuron_token_counts[neuron_idx][token_id] += 1

    # Compute specialization metrics
    specialized_neurons = []
    diversities = []

    for neuron_idx in range(n_input):
        if total_activations[neuron_idx] == 0:
            continue

        token_counts = neuron_token_counts[neuron_idx]
        if not token_counts:
            continue

        # Compute diversity (number of unique tokens)
        diversity = len(token_counts)
        diversities.append(diversity)

        # Find most common tokens
        top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_token_ids = [t[0] for t in top_tokens]
        top_token_words = [tokenizer.decode([t]) for t in top_token_ids]

        # Check if specialized (low diversity = specialized)
        if diversity < 50:  # Threshold for specialization
            specialized_neurons.append({
                'neuron_idx': neuron_idx,
                'diversity': diversity,
                'top_tokens': top_token_words,
                'activation_count': total_activations[neuron_idx]
            })

    results = {
        'total_analyzed': len(diversities),
        'avg_diversity': float(np.mean(diversities)) if diversities else 0,
        'std_diversity': float(np.std(diversities)) if diversities else 0,
        'specialized_count': len(specialized_neurons),
        'specialized_neurons': specialized_neurons[:10]  # Top 10
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
    print(f"  Layers: {routing_results['n_layers']}, Input neurons: {routing_results['n_input']}")
    for layer_idx, stats in routing_results['layers'].items():
        print(f"  Layer {layer_idx}: usage={stats['mean_usage']:.3f}±{stats['std_usage']:.3f}, "
              f"unused={stats['unused_neurons']}")

    # 2. Neuron Usage (Load Balance)
    usage_results = analyze_neuron_usage(model, val_loader, device)
    results['usage'] = usage_results

    print("\n🧠 NEURON USAGE (Load Balance)")
    print("-" * 40)
    for layer_idx, stats in usage_results['layers'].items():
        print(f"  Layer {layer_idx}: balance_score={stats['load_balance_score']:.3f}, "
              f"gini={stats['gini_coefficient']:.3f}")

    # 3. Specialization
    spec_results = analyze_neuron_specialization(model, val_loader, tokenizer, device)
    results['specialization'] = spec_results

    print("\n💎 NEURON SPECIALIZATION")
    print("-" * 40)
    print(f"  Analyzed neurons: {spec_results['total_analyzed']}")
    print(f"  Avg diversity: {spec_results['avg_diversity']:.1f}±{spec_results['std_diversity']:.1f}")
    print(f"  Specialized neurons: {spec_results['specialized_count']}")
    if spec_results['specialized_neurons']:
        print(f"  Top specialized:")
        for neuron in spec_results['specialized_neurons'][:3]:
            print(f"    Neuron {neuron['neuron_idx']}: {neuron['top_tokens'][:3]}")

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
    print(f"  Load balance loss: {aux_results['avg_load_balance']:.6f}")
    print(f"  Entropy loss: {aux_results['avg_entropy']:.6f}")
    print(f"  Total aux loss: {aux_results['avg_aux_loss']:.6f}")
    print(f"  Aux/Main ratio: {aux_results['aux_to_main_ratio']:.4f}")

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
