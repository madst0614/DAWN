"""
DAWN v7.5 Checkpoint Analysis
QK Attention Routing + Soft FFN 구조 종합 분석

분석 항목:
1. Basis 직교성 검증 (완벽해야 함!)
2. Recipe 분석 (basis 사용 패턴)
3. ⭐ Neuron Context Pattern 분석 (NEW!)
4. ⭐ 의미 vs 문맥 점수 분석 (NEW!)
5. ⭐ Attention 패턴 분석 (NEW!)
6. ⭐ V 생성 품질 분석 (NEW!)
7. Neuron 사용률 분석
8. 종합 메트릭 요약

v7.5 특징:
- Router 제거 → QK Attention weights 재활용
- 의미(X) + 문맥(Attention) 결합 뉴런 선택
- 동적 V 생성 (256→96→256)
- neuron_recipe + neuron_context_pattern 학습

Usage:
    python scripts/analyze_dawn_v75.py --checkpoint /path/to/checkpoint_folder
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
import argparse
from tqdm import tqdm
import json
from collections import defaultdict, Counter


# ============================================================
# Helper Functions
# ============================================================

def compute_gini(values):
    """Compute Gini coefficient"""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu()
        sorted_values = torch.sort(values.float())[0]
    else:
        sorted_values = torch.sort(torch.tensor(values, dtype=torch.float32))[0]

    n = len(sorted_values)
    if n == 0 or sorted_values.sum() == 0:
        return 0.0
    index = torch.arange(1, n + 1, dtype=torch.float32)
    return ((2 * index - n - 1) * sorted_values).sum() / (n * sorted_values.sum())


# ============================================================
# 1. Basis Orthogonality Analysis
# ============================================================

def analyze_basis_orthogonality(model):
    """Verify that basis is perfectly orthogonal"""
    print("\n" + "="*60)
    print("1. BASIS ORTHOGONALITY VERIFICATION")
    print("="*60)

    basis = model.shared_basis
    n_basis = basis.n_basis

    results = {}

    # Basis A orthogonality
    basis_A_flat = basis.basis_A.view(n_basis, -1)
    gram_A = torch.mm(basis_A_flat, basis_A_flat.T)
    identity_A = torch.eye(n_basis, device=gram_A.device)
    error_A = (gram_A - identity_A).abs()
    error_A_offdiag = error_A.clone()
    error_A_offdiag.fill_diagonal_(0)

    results['basis_A'] = {
        'max_off_diagonal': error_A_offdiag.max().item(),
        'mean_off_diagonal': error_A_offdiag.sum().item() / (n_basis * (n_basis - 1)),
        'diagonal_mean': torch.diag(gram_A).mean().item(),
        'diagonal_std': torch.diag(gram_A).std().item(),
    }

    # Basis embedding orthogonality
    gram_emb = torch.mm(basis.basis_emb, basis.basis_emb.T)
    error_emb = (gram_emb - identity_A).abs()
    error_emb_offdiag = error_emb.clone()
    error_emb_offdiag.fill_diagonal_(0)

    results['basis_emb'] = {
        'max_off_diagonal': error_emb_offdiag.max().item(),
        'mean_off_diagonal': error_emb_offdiag.sum().item() / (n_basis * (n_basis - 1)),
        'diagonal_mean': torch.diag(gram_emb).mean().item(),
        'diagonal_std': torch.diag(gram_emb).std().item(),
    }

    # Print results
    for basis_name, stats in results.items():
        print(f"\n{basis_name}:")
        print(f"  Max off-diagonal: {stats['max_off_diagonal']:.2e}")
        print(f"  Mean off-diagonal: {stats['mean_off_diagonal']:.2e}")
        print(f"  Diagonal mean: {stats['diagonal_mean']:.6f}")

    # Overall verdict
    max_error = max(
        results['basis_A']['max_off_diagonal'],
        results['basis_emb']['max_off_diagonal']
    )

    print(f"\n{'✅' if max_error < 1e-5 else '⚠️'} Overall: Max error = {max_error:.2e}")
    print(f"   Orthogonality {'PERFECT' if max_error < 1e-5 else 'APPROXIMATE'}!")

    return results


# ============================================================
# 2. Recipe Analysis
# ============================================================

def analyze_recipes(model):
    """Analyze how neurons combine basis elements"""
    print("\n" + "="*60)
    print("2. RECIPE ANALYSIS")
    print("="*60)

    results = {}

    for layer_idx, layer in enumerate(model.layers):
        recipe = layer.neuron_value.neuron_recipe  # [n_neurons, n_basis]
        recipe_norm = F.softmax(recipe, dim=-1)  # Normalized

        # 1. Basis usage distribution
        basis_usage = recipe_norm.mean(dim=0)  # [n_basis]

        # 2. Recipe entropy (how spread out each recipe is)
        recipe_entropy = -torch.sum(
            recipe_norm * torch.log(recipe_norm + 1e-10), dim=-1
        )  # [n_neurons]

        # 3. Recipe diversity (variance across neurons)
        recipe_std = recipe_norm.std(dim=0)  # [n_basis]

        # 4. Dominant basis per neuron
        dominant_basis = recipe_norm.argmax(dim=-1)
        dominant_counts = torch.bincount(
            dominant_basis, minlength=model.n_basis
        )

        # 5. Neuron specialization (max weight per neuron)
        max_weights = recipe_norm.max(dim=-1)[0]

        results[f'layer_{layer_idx}'] = {
            'basis_usage_mean': basis_usage.mean().item(),
            'basis_usage_std': basis_usage.std().item(),
            'basis_usage_min': basis_usage.min().item(),
            'basis_usage_max': basis_usage.max().item(),
            'recipe_entropy_mean': recipe_entropy.mean().item(),
            'recipe_entropy_std': recipe_entropy.std().item(),
            'recipe_diversity_mean': recipe_std.mean().item(),
            'neuron_specialization_mean': max_weights.mean().item(),
            'neuron_specialization_std': max_weights.std().item(),
            'dominant_basis_dist': dominant_counts.cpu().numpy().tolist(),
        }

        print(f"\nLayer {layer_idx}:")
        print(f"  Basis usage: {basis_usage.mean().item():.4f} ± {basis_usage.std().item():.4f}")
        print(f"  Recipe entropy: {recipe_entropy.mean().item():.4f} ± {recipe_entropy.std().item():.4f}")
        print(f"  Neuron specialization: {max_weights.mean().item():.4f} ± {max_weights.std().item():.4f}")

    return results


# ============================================================
# 3. ⭐ Neuron Context Pattern Analysis (NEW!)
# ============================================================

def analyze_context_patterns(model):
    """Analyze neuron_context_pattern weights"""
    print("\n" + "="*60)
    print("3. ⭐ NEURON CONTEXT PATTERN ANALYSIS")
    print("="*60)

    results = {}
    n_heads = model.layers[0].n_heads

    for layer_idx, layer in enumerate(model.layers):
        context_pattern = layer.neuron_value.neuron_context_pattern  # [n_neurons, n_heads]

        # 1. Head preference per neuron
        preferred_head = context_pattern.argmax(dim=-1)
        head_counts = torch.bincount(preferred_head, minlength=n_heads)

        # 2. Pattern diversity (how specialized each neuron is)
        pattern_entropy = -torch.sum(
            F.softmax(context_pattern, dim=-1) * F.log_softmax(context_pattern, dim=-1),
            dim=-1
        )

        # 3. Head specialization (how many neurons prefer each head)
        head_specialization = head_counts.float() / head_counts.sum()

        # 4. Context pattern magnitude
        pattern_magnitude = context_pattern.abs().mean(dim=-1)

        results[f'layer_{layer_idx}'] = {
            'pattern_entropy_mean': pattern_entropy.mean().item(),
            'pattern_entropy_std': pattern_entropy.std().item(),
            'pattern_magnitude_mean': pattern_magnitude.mean().item(),
            'pattern_magnitude_std': pattern_magnitude.std().item(),
            'head_preference_dist': head_counts.cpu().numpy().tolist(),
            'head_specialization_gini': compute_gini(head_specialization).item(),
        }

        print(f"\nLayer {layer_idx}:")
        print(f"  Pattern entropy: {pattern_entropy.mean().item():.4f} ± {pattern_entropy.std().item():.4f}")
        print(f"  Pattern magnitude: {pattern_magnitude.mean().item():.4f} ± {pattern_magnitude.std().item():.4f}")
        print(f"  Head preference: {head_counts.cpu().numpy().tolist()}")
        print(f"  Head specialization Gini: {compute_gini(head_specialization):.4f}")

    return results


# ============================================================
# 4-6. ⭐ Runtime Analysis (Semantic/Context/Attention/V)
# ============================================================

def analyze_runtime_behavior(model, dataloader, device, max_batches=10):
    """Analyze neuron selection behavior during inference"""
    print("\n" + "="*60)
    print("4-6. ⭐ RUNTIME BEHAVIOR ANALYSIS")
    print("="*60)

    model.eval()

    # ⚡ GPU-optimized accumulators (keep everything on GPU)
    n_layers = len(model.layers)
    all_neuron_usage = {f'layer_{i}': torch.zeros(model.n_neurons, device=device, dtype=torch.long)
                        for i in range(n_layers)}

    # Accumulate statistics per layer on GPU
    layer_stats = {
        'semantic_scores': torch.zeros(n_layers, device=device),
        'context_scores': torch.zeros(n_layers, device=device),
        'final_scores': torch.zeros(n_layers, device=device),
        'semantic_vs_context_ratios': torch.zeros(n_layers, device=device),
        'attn_self': torch.zeros(n_layers, device=device),
        'counts': torch.zeros(n_layers, device=device),  # for averaging
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            B, S = input_ids.shape

            # Forward pass with activations
            pos = torch.arange(S, device=device).unsqueeze(0)
            x = model.token_emb(input_ids) + model.pos_emb(pos)
            x = model.dropout(x)

            mask = model.causal_mask[:S, :S].unsqueeze(0).unsqueeze(0)

            for layer_idx, layer in enumerate(model.layers):
                # Part 1: Attention
                residual = x
                normed = layer.norm1(x)

                # Q, K
                Q = layer.q_proj(normed).view(B, S, layer.n_heads, layer.d_head).transpose(1, 2)
                K = layer.k_proj(normed).view(B, S, layer.n_heads, layer.d_head).transpose(1, 2)

                # Attention weights
                attn_scores = Q @ K.transpose(-2, -1) / (layer.d_head ** 0.5)
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                attn_weights = F.softmax(attn_scores, dim=-1)  # [B, n_heads, S, S]

                # ⭐ Analyze neuron selection
                neuron_value = layer.neuron_value

                # Semantic scores (using K - QK Attention 정보 재활용)
                K_combined = K.transpose(1, 2).reshape(B, S, layer.d_model)  # [B, S, D]
                semantic_scores = K_combined @ neuron_value.neuron_emb_semantic.T  # [B, S, n_neurons]

                # Context scores
                attn_summary = attn_weights.mean(dim=-1).transpose(1, 2)  # [B, S, n_heads]
                context_scores = attn_summary @ neuron_value.neuron_context_pattern.T  # [B, S, n_neurons]

                # Final scores
                final_scores = semantic_scores * torch.sigmoid(context_scores)

                # Top-K selection
                topk_scores, neuron_idx = torch.topk(final_scores, neuron_value.k, dim=-1)

                # ⚡ GPU-optimized statistics accumulation (no CPU sync)
                layer_stats['semantic_scores'][layer_idx] += semantic_scores.abs().mean()
                layer_stats['context_scores'][layer_idx] += context_scores.abs().mean()
                layer_stats['final_scores'][layer_idx] += final_scores.abs().mean()

                semantic_contrib = semantic_scores.abs().mean()
                context_contrib = torch.sigmoid(context_scores).abs().mean()
                layer_stats['semantic_vs_context_ratios'][layer_idx] += semantic_contrib / (context_contrib + 1e-8)

                # ⚡ Vectorized neuron usage update with bincount (much faster than loop)
                neuron_idx_flat = neuron_idx.reshape(-1)
                usage_update = torch.bincount(neuron_idx_flat, minlength=model.n_neurons)
                all_neuron_usage[f'layer_{layer_idx}'] += usage_update

                # Attention pattern statistics
                layer_stats['attn_self'][layer_idx] += torch.diagonal(attn_weights, dim1=-2, dim2=-1).mean()
                layer_stats['counts'][layer_idx] += 1

                # V generation and forward
                V, _ = neuron_value(normed, attn_weights, K)
                attn_out = (layer.attn_dropout(attn_weights) @ V).transpose(1, 2).reshape(B, S, layer.d_model)
                attn_out = layer.attn_out(attn_out)
                x = residual + layer.dropout(attn_out)

                # Part 2: FFN
                residual = x
                normed = layer.norm2(x)
                ffn_out = F.gelu(layer.w_up(normed))
                ffn_out = layer.dropout(ffn_out)
                ffn_out = layer.w_down(ffn_out)
                x = residual + layer.dropout(ffn_out)

    # ⚡ Single CPU transfer at the end (average across batches)
    for key in ['semantic_scores', 'context_scores', 'final_scores', 'semantic_vs_context_ratios', 'attn_self']:
        layer_stats[key] = (layer_stats[key] / layer_stats['counts']).cpu().numpy()

    # Compute results
    results = {}

    # 4. Semantic vs Context Analysis
    print("\n4. SEMANTIC vs CONTEXT SCORES:")
    print(f"  Semantic score (mean): {layer_stats['semantic_scores'].mean():.4f}")
    print(f"  Context score (mean): {layer_stats['context_scores'].mean():.4f}")
    print(f"  Final score (mean): {layer_stats['final_scores'].mean():.4f}")
    print(f"  Semantic/Context ratio: {layer_stats['semantic_vs_context_ratios'].mean():.4f}")

    results['semantic_context'] = {
        'semantic_mean': float(layer_stats['semantic_scores'].mean()),
        'context_mean': float(layer_stats['context_scores'].mean()),
        'final_mean': float(layer_stats['final_scores'].mean()),
        'ratio_mean': float(layer_stats['semantic_vs_context_ratios'].mean()),
    }

    # 5. Attention Pattern Analysis
    print("\n5. ATTENTION PATTERNS:")
    print(f"  Self-attention (mean): {layer_stats['attn_self'].mean():.4f}")

    results['attention'] = {
        'self_attention_mean': float(layer_stats['attn_self'].mean()),
    }

    # 6. Neuron Usage Analysis
    print("\n6. NEURON USAGE:")
    for layer_idx in range(len(model.layers)):
        usage = all_neuron_usage[f'layer_{layer_idx}']
        total = usage.sum()
        usage_rate = (usage > 0).float().mean().item()
        gini = compute_gini(usage).item()

        print(f"\nLayer {layer_idx}:")
        print(f"  Active neurons: {(usage > 0).sum().item()}/{model.n_neurons} ({usage_rate*100:.1f}%)")
        print(f"  Usage Gini: {gini:.4f}")
        print(f"  Top-10 usage: {usage.topk(10)[0].cpu().numpy()}")

        results[f'usage_layer_{layer_idx}'] = {
            'active_neurons': int((usage > 0).sum().item()),
            'usage_rate': float(usage_rate),
            'gini': float(gini),
            'total_selections': int(total.item()),
        }

    return results


# ============================================================
# 7. Summary Metrics
# ============================================================

def compute_summary_metrics(all_results):
    """Compute overall summary metrics"""
    print("\n" + "="*60)
    print("7. SUMMARY METRICS")
    print("="*60)

    summary = {}

    # Basis orthogonality
    basis_quality = all_results['basis']['basis_A']['max_off_diagonal']
    summary['basis_orthogonality'] = 'PERFECT' if basis_quality < 1e-5 else 'APPROXIMATE'

    # Recipe diversity
    recipe_results = all_results['recipes']
    recipe_entropies = [v['recipe_entropy_mean'] for k, v in recipe_results.items() if k.startswith('layer_')]
    summary['recipe_entropy_mean'] = float(np.mean(recipe_entropies))

    # Context pattern diversity
    context_results = all_results['context_patterns']
    context_entropies = [v['pattern_entropy_mean'] for k, v in context_results.items() if k.startswith('layer_')]
    summary['context_pattern_entropy_mean'] = float(np.mean(context_entropies))

    # Semantic vs Context balance
    semantic_context = all_results['runtime']['semantic_context']
    summary['semantic_context_ratio'] = semantic_context['ratio_mean']
    summary['semantic_dominance'] = 'High' if semantic_context['ratio_mean'] > 2.0 else 'Balanced'

    # Neuron usage
    usage_rates = []
    usage_ginis = []
    for layer_idx in range(4):  # Assuming 4 layers
        key = f'usage_layer_{layer_idx}'
        if key in all_results['runtime']:
            usage_rates.append(all_results['runtime'][key]['usage_rate'])
            usage_ginis.append(all_results['runtime'][key]['gini'])

    summary['neuron_usage_rate_mean'] = float(np.mean(usage_rates)) if usage_rates else 0.0
    summary['neuron_usage_gini_mean'] = float(np.mean(usage_ginis)) if usage_ginis else 0.0

    # Print summary
    print("\n📊 OVERALL SUMMARY:")
    print(f"  Basis Orthogonality: {summary['basis_orthogonality']}")
    print(f"  Recipe Entropy: {summary['recipe_entropy_mean']:.4f}")
    print(f"  Context Pattern Entropy: {summary['context_pattern_entropy_mean']:.4f}")
    print(f"  Semantic/Context Ratio: {summary['semantic_context_ratio']:.4f} ({summary['semantic_dominance']})")
    print(f"  Neuron Usage Rate: {summary['neuron_usage_rate_mean']*100:.1f}%")
    print(f"  Neuron Usage Gini: {summary['neuron_usage_gini_mean']:.4f}")

    return summary


# ============================================================
# Main Analysis Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN v7.5 Checkpoint Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint folder')
    parser.add_argument('--max-batches', type=int, default=10,
                        help='Max batches for runtime analysis')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for data loading (default: 128, increase for faster GPU processing)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint) / 'best_model.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model
    from models import create_model_by_version

    # First, detect actual model version from checkpoint
    actual_version = checkpoint.get('model_version', 'unknown')
    print(f"Checkpoint model version: {actual_version}")

    # Get model config (with backward compatibility)
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        model_version = model_config.get('model_version', actual_version)
    else:
        # Backward compatibility: infer config from checkpoint
        print("⚠️  Warning: No config found in checkpoint. Inferring from state_dict...")

        # Try to infer version from state_dict keys
        state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model'
        state_dict = checkpoint[state_dict_key]

        # Check for v7.5 specific keys
        if 'layers.0.neuron_value.neuron_recipe' in state_dict:
            model_version = '7.5'
            print(f"  Detected v7.5 architecture (neuron_value found)")
        elif 'layers.0.ffn.neuron_recipe' in state_dict:
            model_version = '7.1'
            print(f"  Detected v7.1 architecture (ffn.neuron_recipe found)")
        else:
            model_version = actual_version if actual_version != 'unknown' else '7.1'
            print(f"  Using version: {model_version}")

        # Build default config based on detected version
        model_config = {
            'vocab_size': 30522,  # BERT vocab
            'd_model': 256,
            'n_layers': 4,
            'n_heads': 4,
            'd_ff': 1024,
            'max_seq_len': 128,
            'dropout': 0.1,
        }

        # Version-specific parameters
        if model_version == '7.5':
            model_config.update({
                'n_neurons': 96,
                'neuron_k': 8,
                'n_basis': 32,
                'basis_rank': 96,
            })
        else:
            model_config.update({
                'n_neurons': 64,
                'neuron_k': 8,
                'n_basis': 32,
                'basis_rank': 64,
            })

        model_config['model_version'] = model_version

    # Check if this is actually v7.5
    if model_version != '7.5':
        print(f"\n❌ ERROR: This script is for v7.5 models only!")
        print(f"   Found: v{model_version}")
        print(f"   Please use the appropriate analysis script:")
        print(f"   - v7.0: scripts/analyze_dawn_v7.py")
        print(f"   - v7.1: scripts/analyze_dawn_v7.py")
        print(f"   - v7.2: scripts/analyze_dawn_v7.py")
        print(f"   - v7.4: scripts/analyze_dawn_v7.py")
        return

    print(f"\n✅ Confirmed v7.5 checkpoint. Proceeding with analysis...")

    model = create_model_by_version(model_version, model_config)

    # Load state dict
    state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model'
    state_dict = checkpoint[state_dict_key]

    # Handle torch.compile wrapper (remove _orig_mod. prefix)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  Removing torch.compile wrapper prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model: DAWN v{model.__version__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ⚡ Use torch.compile for faster inference (PyTorch 2.0+)
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        print("\n⚡ Compiling model with torch.compile for faster GPU execution...")
        try:
            model = torch.compile(model, mode='max-autotune')
            print("   ✅ Model compiled successfully!")
        except Exception as e:
            print(f"   ⚠️  Compilation failed: {e}")
            print("   Continuing with uncompiled model...")

    # Prepare dataloader
    from utils.data import TextDataset, collate_fn_dynamic_padding
    from torch.utils.data import DataLoader
    from functools import partial
    import pickle
    import os

    data_config = checkpoint.get('data_config', {
        'base_dir': '/content/drive/MyDrive/data',
        'val_file': 'validation/wikitext_5to1_texts.pkl'
    })

    # Load validation texts directly
    val_path = os.path.join(data_config['base_dir'], data_config['val_file'])
    print(f"\nLoading validation data from: {val_path}")

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found: {val_path}")

    with open(val_path, 'rb') as f:
        val_texts = pickle.load(f)

    print(f"Loaded {len(val_texts)} validation texts")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    max_seq_len = model_config.get('max_seq_len', 128)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=max_seq_len)

    # ⚡ GPU-optimized DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn_dynamic_padding, tokenizer=tokenizer),
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster CPU->GPU transfer
        prefetch_factor=2,  # Prefetch batches
    )

    # Run analysis
    all_results = {}

    print("\n" + "="*60)
    print("STARTING ANALYSIS")
    print("="*60)

    all_results['basis'] = analyze_basis_orthogonality(model)
    all_results['recipes'] = analyze_recipes(model)
    all_results['context_patterns'] = analyze_context_patterns(model)
    all_results['runtime'] = analyze_runtime_behavior(model, val_loader, device, args.max_batches)
    all_results['summary'] = compute_summary_metrics(all_results)

    # Save results
    output_path = Path(args.checkpoint) / 'analysis_v75.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Analysis complete! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
