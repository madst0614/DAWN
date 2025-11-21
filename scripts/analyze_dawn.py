"""
DAWN (Dynamic Architecture With Neurons) 종합 분석 스크립트

새로운 단순화된 DAWN 모델 분석:
- InputNeurons 활성화 패턴 분석
- ProcessNeurons 활성화 패턴 분석
- 뉴런 특화 분석
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

from models.model import DAWN, DAWNLanguageModel
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

    val_dataset = TextDataset(val_texts, tokenizer, model_cfg.get('max_seq_len', 512))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

    return val_loader, tokenizer, cfg


# ============================================================
# Model Loading
# ============================================================

def load_checkpoint(checkpoint_path, device='cuda'):
    """체크포인트에서 모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_dir = Path(checkpoint_path).parent

    # Config 로드
    config_path = checkpoint_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from: {config_path}")
    else:
        print("Config file not found, using defaults")
        config = {
            'model': {
                'd_model': 512,
                'n_layers': 6,
                'max_seq_len': 128,
                'n_input': 64,
                'n_process': 128,
                'dropout': 0.1
            }
        }

    # 가중치 로드
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # vocab_size 추론
    vocab_size = 30522  # Default
    if 'token_embedding.weight' in state_dict:
        vocab_size = state_dict['token_embedding.weight'].shape[0]
        print(f"Inferred vocab_size: {vocab_size}")

    model_cfg = config.get('model', config)

    # 모델 생성 (새 DAWN)
    model = DAWN(
        vocab_size=vocab_size,
        hidden_dim=model_cfg.get('d_model', 512),
        num_layers=model_cfg.get('n_layers', 6),
        num_input_neurons=model_cfg.get('n_input', 64),
        num_process_neurons=model_cfg.get('n_process', 128),
        max_seq_len=model_cfg.get('max_seq_len', 512),
        dropout=model_cfg.get('dropout', 0.1)
    )

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # 체크포인트 정보
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"Checkpoint loss: {checkpoint['loss']:.4f}")

    return model, config


# ============================================================
# Analysis Functions
# ============================================================

def analyze_activation_patterns(model, val_loader, device, max_batches=50):
    """InputNeurons와 ProcessNeurons 활성화 패턴 분석"""
    print("\n🔥 Analyzing Activation Patterns...")

    n_layers = len(model.layers)
    num_input = model.layers[0].input_neurons.num_neurons
    num_process = model.layers[0].process_neurons.num_process_neurons

    # 레이어별 활성화 누적
    layer_input_acts = [[] for _ in range(n_layers)]
    layer_process_acts = [[] for _ in range(n_layers)]

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Activation analysis")):
        if batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)

        with torch.no_grad():
            _, all_activations = model(input_ids, return_activations=True)

            for layer_idx, acts in enumerate(all_activations):
                input_acts = acts['input_activations']  # [B, S, N_input]
                process_acts = acts['process_activations']  # [B, S, N_process]

                layer_input_acts[layer_idx].append(input_acts.cpu())
                layer_process_acts[layer_idx].append(process_acts.cpu())

    # 통계 계산
    results = {'layers': {}}

    for layer_idx in range(n_layers):
        # Concatenate all batches
        input_acts_all = torch.cat(layer_input_acts[layer_idx], dim=0)  # [total_samples, S, N_input]
        process_acts_all = torch.cat(layer_process_acts[layer_idx], dim=0)

        # 평균 활성화
        input_mean = input_acts_all.mean(dim=[0, 1])  # [N_input]
        process_mean = process_acts_all.mean(dim=[0, 1])  # [N_process]

        # Sparsity (< 0.1)
        input_sparsity = (input_acts_all < 0.1).float().mean().item()
        process_sparsity = (process_acts_all < 0.1).float().mean().item()

        # 활성 뉴런 수 (평균)
        input_active = (input_acts_all > 0.1).float().sum(dim=-1).mean().item()
        process_active = (process_acts_all > 0.1).float().sum(dim=-1).mean().item()

        results['layers'][layer_idx] = {
            'input_neurons': {
                'mean_activation': float(input_mean.mean().item()),
                'std_activation': float(input_mean.std().item()),
                'sparsity': input_sparsity,
                'avg_active_neurons': input_active,
                'top_5_neurons': input_mean.topk(5).indices.tolist()
            },
            'process_neurons': {
                'mean_activation': float(process_mean.mean().item()),
                'std_activation': float(process_mean.std().item()),
                'sparsity': process_sparsity,
                'avg_active_neurons': process_active,
                'top_5_neurons': process_mean.topk(5).indices.tolist()
            }
        }

    return results


def analyze_neuron_specialization(model, val_loader, tokenizer, device, layer_idx=0, max_batches=100):
    """
    특정 레이어의 InputNeurons 특화 분석
    어떤 뉴런이 어떤 토큰에 반응하는지
    """
    print(f"\n💎 Analyzing Neuron Specialization (Layer {layer_idx})...")

    num_input = model.layers[layer_idx].input_neurons.num_neurons
    vocab_size = tokenizer.vocab_size

    # GPU tensors for counting
    neuron_token_counts = torch.zeros(num_input, vocab_size, dtype=torch.float32, device=device)
    global_token_counts = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    neuron_activation_counts = torch.zeros(num_input, dtype=torch.float32, device=device)

    total_tokens = 0

    for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Specialization L{layer_idx}")):
        if batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)
        B, S = input_ids.shape

        with torch.no_grad():
            # Forward through layers up to target
            token_emb = model.token_embedding(input_ids)
            positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = model.position_embedding(positions)
            x = token_emb + pos_emb
            x = model.embedding_dropout(x)

            # Pass through layers
            for i in range(layer_idx + 1):
                x, activations = model.layers[i](x)

            # Get input neuron activations for target layer
            _, target_acts = model.layers[layer_idx](
                token_emb + pos_emb if layer_idx == 0 else x
            )
            input_acts = target_acts['input_activations']  # [B, S, N_input]

            # Filter valid tokens
            valid_mask = (input_ids != 0) & (input_ids != 101) & (input_ids != 102)

            # Count neuron activations (활성화 강도 기반)
            neuron_activation_counts += input_acts.sum(dim=[0, 1])

            # Count global token frequency
            valid_tokens = input_ids[valid_mask]
            global_token_counts.scatter_add_(0, valid_tokens,
                                             torch.ones_like(valid_tokens, dtype=torch.float32))
            total_tokens += valid_tokens.numel()

            # Count neuron-token co-occurrence (활성화 강도 가중)
            # [B, S, N_input] × [B, S] → [N_input, vocab_size]
            for b in range(B):
                for s in range(S):
                    if not valid_mask[b, s]:
                        continue
                    token_id = input_ids[b, s]
                    neuron_token_counts[:, token_id] += input_acts[b, s]  # Weighted by activation

    # Compute PMI
    total_activations = neuron_activation_counts.sum()
    p_neuron = neuron_activation_counts / (total_activations + 1e-10)
    p_token = global_token_counts / (total_tokens + 1e-10)
    p_joint = neuron_token_counts / (total_tokens + 1e-10)

    pmi_matrix = torch.log(
        (p_joint + 1e-10) / (p_neuron.unsqueeze(1) * p_token.unsqueeze(0) + 1e-10)
    )

    # Extract top specialized neurons
    neuron_specializations = []

    for neuron_idx in range(num_input):
        if neuron_activation_counts[neuron_idx] < 10:
            continue

        neuron_pmi = pmi_matrix[neuron_idx]
        neuron_counts = neuron_token_counts[neuron_idx]

        top_pmi_values, top_token_ids = torch.topk(neuron_pmi, k=min(10, vocab_size))

        top_pmi_values = top_pmi_values.cpu().numpy()
        top_token_ids = top_token_ids.cpu().numpy()
        top_counts = neuron_counts[top_token_ids].cpu().numpy()

        top_token_words = []
        for i, (token_id, pmi_score, count) in enumerate(zip(top_token_ids, top_pmi_values, top_counts)):
            if count < 1:
                continue
            try:
                word = tokenizer.decode([int(token_id)])
            except:
                word = f"[{token_id}]"
            top_token_words.append({
                'token': word,
                'pmi': float(pmi_score),
                'raw_count': int(count)
            })

        if top_token_words:
            avg_pmi = float(np.mean([t['pmi'] for t in top_token_words[:5]]))

            neuron_specializations.append({
                'neuron_idx': int(neuron_idx),
                'specialization_strength': avg_pmi,
                'top_tokens_pmi': top_token_words,
                'activation_count': int(neuron_activation_counts[neuron_idx].item())
            })

    neuron_specializations.sort(key=lambda x: x['specialization_strength'], reverse=True)

    return {
        'layer_idx': layer_idx,
        'total_analyzed': len(neuron_specializations),
        'avg_specialization': float(np.mean([n['specialization_strength'] for n in neuron_specializations])) if neuron_specializations else 0,
        'specialized_neurons': neuron_specializations[:20]
    }


def analyze_performance(model, val_loader, tokenizer, device):
    """성능 세부 분석"""
    print("\n🎯 Analyzing Performance...")

    all_losses = []
    all_corrects = []

    for batch in tqdm(val_loader, desc="Performance analysis"):
        input_ids = batch['input_ids'].to(device)

        # Apply MLM masking
        masked_input_ids, labels = apply_mlm_masking(input_ids.clone(), tokenizer)

        with torch.no_grad():
            logits = model(masked_input_ids)

            # Per-token loss
            loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            per_token_loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            # Accuracy
            preds = logits.argmax(dim=-1)
            valid_mask = (labels != -100)
            correct = ((preds == labels) & valid_mask).view(-1)

            valid_losses = per_token_loss[labels.view(-1) != -100].cpu().tolist()
            valid_corrects = correct[labels.view(-1) != -100].float().cpu().tolist()

            all_losses.extend(valid_losses)
            all_corrects.extend(valid_corrects)

    losses = np.array(all_losses)
    corrects = np.array(all_corrects)

    # Percentile 분석
    easy_threshold = np.percentile(losses, 25)
    hard_threshold = np.percentile(losses, 75)

    easy_mask = losses < easy_threshold
    hard_mask = losses > hard_threshold

    return {
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


def analyze_gradient_flow(model, val_loader, tokenizer, device, n_batches=10):
    """Gradient flow 분석"""
    print("\n🌊 Analyzing Gradient Flow...")

    model.train()

    layer_gradients = defaultdict(list)

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Gradient analysis")):
        if batch_idx >= n_batches:
            break

        input_ids = batch['input_ids'].to(device)
        masked_input_ids, labels = apply_mlm_masking(input_ids.clone(), tokenizer)

        model.zero_grad()

        logits = model(masked_input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        loss.backward()

        # Collect gradients per layer
        for layer_idx, layer in enumerate(model.layers):
            layer_grad_norm = 0.0
            param_count = 0
            for param in layer.parameters():
                if param.grad is not None:
                    layer_grad_norm += param.grad.norm().item() ** 2
                    param_count += 1
            if param_count > 0:
                layer_grad_norm = (layer_grad_norm / param_count) ** 0.5
                layer_gradients[layer_idx].append(layer_grad_norm)

    model.eval()

    return {
        'layer_gradient_norms': {
            idx: float(np.mean(grads)) for idx, grads in layer_gradients.items()
        }
    }


# ============================================================
# Main Analysis
# ============================================================

def comprehensive_analysis(model, val_loader, tokenizer, device):
    """DAWN 모델 종합 분석"""
    print("=" * 60)
    print("DAWN Comprehensive Analysis (New Architecture)")
    print("=" * 60)

    results = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'hidden_dim': model.hidden_dim,
            'num_layers': len(model.layers),
            'num_input_neurons': model.layers[0].input_neurons.num_neurons,
            'num_process_neurons': model.layers[0].process_neurons.num_process_neurons,
            'total_params': sum(p.numel() for p in model.parameters())
        }
    }

    # 1. Activation Patterns
    activation_results = analyze_activation_patterns(model, val_loader, device, max_batches=100)
    results['activations'] = activation_results

    print("\n🔥 ACTIVATION PATTERNS")
    print("-" * 40)
    for layer_idx, stats in activation_results['layers'].items():
        print(f"  Layer {layer_idx}:")
        print(f"    Input Neurons: sparsity={stats['input_neurons']['sparsity']:.2%}, "
              f"avg_active={stats['input_neurons']['avg_active_neurons']:.1f}")
        print(f"    Process Neurons: sparsity={stats['process_neurons']['sparsity']:.2%}, "
              f"avg_active={stats['process_neurons']['avg_active_neurons']:.1f}")

    # 2. Neuron Specialization (all layers)
    results['specialization'] = {}
    print("\n💎 NEURON SPECIALIZATION")
    print("-" * 40)

    n_layers = len(model.layers)
    for layer_idx in range(n_layers):
        spec_results = analyze_neuron_specialization(
            model, val_loader, tokenizer, device,
            layer_idx=layer_idx,
            max_batches=100
        )
        results['specialization'][layer_idx] = spec_results

        print(f"\n  Layer {layer_idx}:")
        print(f"    Analyzed: {spec_results['total_analyzed']} neurons")
        print(f"    Avg specialization (PMI): {spec_results['avg_specialization']:.3f}")
        if spec_results['specialized_neurons']:
            print(f"    Top specialized neurons:")
            for neuron in spec_results['specialized_neurons'][:3]:
                tokens = [f"{t['token']}({t['pmi']:.2f})" for t in neuron['top_tokens_pmi'][:3]]
                print(f"      Neuron {neuron['neuron_idx']}: {', '.join(tokens)}")

    # 3. Performance
    perf_results = analyze_performance(model, val_loader, tokenizer, device)
    results['performance'] = perf_results

    print("\n🎯 PERFORMANCE BREAKDOWN")
    print("-" * 40)
    print(f"  Overall accuracy: {perf_results['overall_acc']*100:.2f}%")
    print(f"  Easy samples (top 25%): {perf_results['easy_samples_acc']*100:.2f}%")
    print(f"  Hard samples (bottom 25%): {perf_results['hard_samples_acc']*100:.2f}%")

    # 4. Gradient Flow
    grad_results = analyze_gradient_flow(model, val_loader, tokenizer, device)
    results['gradient_flow'] = grad_results

    print("\n🌊 GRADIENT FLOW")
    print("-" * 40)
    for layer_idx, grad_norm in grad_results['layer_gradient_norms'].items():
        print(f"  Layer {layer_idx}: grad={grad_norm:.4f}")

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

    # Load data
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

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
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
