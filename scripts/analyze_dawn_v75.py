"""
DAWN v7.5 Checkpoint Analysis
Dynamic Q/K/V Generation (v8 design)

분석 항목:
1. Basis 직교성 검증 (완벽해야 함!)
2. Recipe 분석 (Q/K/V recipe 3개)
3. ⭐ Runtime 분석 (라우팅 + 뉴런 사용)
4. 종합 메트릭 요약

v7.5 특징 (v8 design):
- 라우터: x만 보고 뉴런 선택
- Q/K/V 모두 동적 생성 (recipe_Q/K/V)
- 깔끔한 구조: basis_emb 없음, context score 없음

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

    basis = model.shared_basis.basis  # [n_basis, D, rank]
    n_basis = model.n_basis

    results = {}

    # Check orthogonality for each basis element
    # Each basis[i] is [D, rank], we want columns to be orthogonal
    max_errors = []
    mean_errors = []

    for i in range(n_basis):
        basis_i = basis[i]  # [D, rank]
        gram = basis_i.T @ basis_i  # [rank, rank]
        identity = torch.eye(gram.shape[0], device=gram.device)
        error = (gram - identity).abs()
        error_offdiag = error.clone()
        error_offdiag.fill_diagonal_(0)

        max_errors.append(error_offdiag.max().item())
        mean_errors.append(error_offdiag.mean().item())

    results['basis'] = {
        'max_off_diagonal': max(max_errors),
        'mean_off_diagonal': np.mean(mean_errors),
        'max_across_basis': max_errors,
    }

    print(f"\nBasis orthogonality:")
    print(f"  Max off-diagonal: {max(max_errors):.2e}")
    print(f"  Mean off-diagonal: {np.mean(mean_errors):.2e}")

    max_error = max(max_errors)
    print(f"\n{'✅' if max_error < 1e-5 else '⚠️'} Overall: Max error = {max_error:.2e}")
    print(f"   Orthogonality {'PERFECT' if max_error < 1e-5 else 'APPROXIMATE'}!")

    return results


# ============================================================
# 2. Recipe Analysis (Q/K/V)
# ============================================================

def analyze_recipes(model):
    """Analyze how neurons combine basis elements for Q/K/V/O"""
    print("\n" + "="*60)
    print("2. RECIPE ANALYSIS (Q/K/V/O)")
    print("="*60)

    results = {}

    for layer_idx, layer in enumerate(model.layers):
        qkv_module = layer.qkv_dynamic

        # Get Q/K/V/O recipes
        recipe_Q = qkv_module.neuron_recipe_Q  # [n_neurons, n_basis]
        recipe_K = qkv_module.neuron_recipe_K
        recipe_V = qkv_module.neuron_recipe_V
        recipe_O = qkv_module.neuron_recipe_O

        layer_results = {}

        for recipe_name, recipe in [('Q', recipe_Q), ('K', recipe_K), ('V', recipe_V), ('O', recipe_O)]:
            recipe_norm = F.softmax(recipe, dim=-1)  # Normalized

            # 1. Basis usage distribution
            basis_usage = recipe_norm.mean(dim=0)  # [n_basis]

            # 2. Recipe entropy (how spread out each recipe is)
            recipe_entropy = -torch.sum(
                recipe_norm * torch.log(recipe_norm + 1e-10), dim=-1
            )  # [n_neurons]

            # 3. Neuron specialization (max weight per neuron)
            max_weights = recipe_norm.max(dim=-1)[0]

            # 4. Dominant basis per neuron
            dominant_basis = recipe_norm.argmax(dim=-1)
            dominant_counts = torch.bincount(
                dominant_basis, minlength=model.n_basis
            )

            layer_results[f'recipe_{recipe_name}'] = {
                'basis_usage_mean': basis_usage.mean().item(),
                'basis_usage_std': basis_usage.std().item(),
                'recipe_entropy_mean': recipe_entropy.mean().item(),
                'recipe_entropy_std': recipe_entropy.std().item(),
                'neuron_specialization_mean': max_weights.mean().item(),
                'neuron_specialization_std': max_weights.std().item(),
                'dominant_basis_dist': dominant_counts.cpu().numpy().tolist(),
            }

        # Compare Q/K/V/O recipes
        recipe_Q_norm = F.softmax(recipe_Q, dim=-1)
        recipe_K_norm = F.softmax(recipe_K, dim=-1)
        recipe_V_norm = F.softmax(recipe_V, dim=-1)
        recipe_O_norm = F.softmax(recipe_O, dim=-1)

        # Cosine similarity between recipes
        qk_sim = F.cosine_similarity(recipe_Q_norm, recipe_K_norm, dim=-1).mean().item()
        qv_sim = F.cosine_similarity(recipe_Q_norm, recipe_V_norm, dim=-1).mean().item()
        qo_sim = F.cosine_similarity(recipe_Q_norm, recipe_O_norm, dim=-1).mean().item()
        kv_sim = F.cosine_similarity(recipe_K_norm, recipe_V_norm, dim=-1).mean().item()
        ko_sim = F.cosine_similarity(recipe_K_norm, recipe_O_norm, dim=-1).mean().item()
        vo_sim = F.cosine_similarity(recipe_V_norm, recipe_O_norm, dim=-1).mean().item()

        layer_results['recipe_similarity'] = {
            'Q_K_similarity': qk_sim,
            'Q_V_similarity': qv_sim,
            'Q_O_similarity': qo_sim,
            'K_V_similarity': kv_sim,
            'K_O_similarity': ko_sim,
            'V_O_similarity': vo_sim,
        }

        results[f'layer_{layer_idx}'] = layer_results

        print(f"\nLayer {layer_idx}:")
        for recipe_name in ['Q', 'K', 'V', 'O']:
            r = layer_results[f'recipe_{recipe_name}']
            print(f"  Recipe {recipe_name}:")
            print(f"    Basis usage: {r['basis_usage_mean']:.4f} ± {r['basis_usage_std']:.4f}")
            print(f"    Entropy: {r['recipe_entropy_mean']:.4f} ± {r['recipe_entropy_std']:.4f}")
            print(f"    Specialization: {r['neuron_specialization_mean']:.4f} ± {r['neuron_specialization_std']:.4f}")
        print(f"  Recipe similarity: Q-K={qk_sim:.4f}, Q-V={qv_sim:.4f}, Q-O={qo_sim:.4f}")
        print(f"                     K-V={kv_sim:.4f}, K-O={ko_sim:.4f}, V-O={vo_sim:.4f}")

    return results


# ============================================================
# 3. ⭐ Runtime Behavior Analysis
# ============================================================

def analyze_runtime_behavior(model, dataloader, device, max_batches=10):
    """Analyze neuron routing and usage during inference"""
    print("\n" + "="*60)
    print("3. ⭐ RUNTIME BEHAVIOR ANALYSIS")
    print("="*60)

    model.eval()

    # ⚡ GPU-optimized accumulators
    n_layers = len(model.layers)
    all_neuron_usage = {f'layer_{i}': torch.zeros(model.n_neurons, device=device, dtype=torch.long)
                        for i in range(n_layers)}

    # Accumulate statistics per layer on GPU
    layer_stats = {
        'router_scores': torch.zeros(n_layers, device=device),
        'routing_weights': torch.zeros(n_layers, device=device),
        'attn_self': torch.zeros(n_layers, device=device),
        'counts': torch.zeros(n_layers, device=device),
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)

            # Forward pass with routing info
            logits, routing_infos = model(input_ids, return_routing_info=True)

            B, S = input_ids.shape

            # Compute attention self-attention for analysis
            pos = torch.arange(S, device=device).unsqueeze(0)
            x = model.token_emb(input_ids) + model.pos_emb(pos)
            x = model.dropout(x)
            mask = model.causal_mask[:, :, :S, :S]

            for layer_idx, layer in enumerate(model.layers):
                routing_info = routing_infos[layer_idx]
                neuron_indices = routing_info['neuron_indices']  # [B, S, k]
                neuron_weights = routing_info['neuron_weights']  # [B, S, k]

                # ⚡ Statistics accumulation (no CPU sync)
                layer_stats['routing_weights'][layer_idx] += neuron_weights.mean()

                # ⚡ Vectorized neuron usage update
                neuron_idx_flat = neuron_indices.reshape(-1)
                usage_update = torch.bincount(neuron_idx_flat, minlength=model.n_neurons)
                all_neuron_usage[f'layer_{layer_idx}'] += usage_update

                # Get router scores for analysis
                residual = x
                normed = layer.norm1(x)
                router_scores = layer.qkv_dynamic.W_router(normed)  # [B, S, n_neurons]
                layer_stats['router_scores'][layer_idx] += router_scores.abs().mean()

                # Compute attention for self-attention metric
                qkv_module = layer.qkv_dynamic
                attn_out, _ = qkv_module(normed, mask)

                # For self-attention metric, we need to recompute attention weights
                # Just use a simple approximation based on the output
                layer_stats['attn_self'][layer_idx] += attn_out.abs().mean()

                layer_stats['counts'][layer_idx] += 1

                # Forward through layer
                x = residual + layer.dropout(attn_out)
                residual = x
                normed = layer.norm2(x)
                ffn_out = layer.w_down(F.gelu(layer.w_up(normed)))
                x = residual + layer.dropout(ffn_out)

    # ⚡ Single CPU transfer at the end
    for key in ['router_scores', 'routing_weights', 'attn_self']:
        layer_stats[key] = (layer_stats[key] / layer_stats['counts']).cpu().numpy()

    # Compute results
    results = {}

    # Routing Analysis
    print("\n⭐ ROUTING ANALYSIS:")
    print(f"  Router score (mean): {layer_stats['router_scores'].mean():.4f}")
    print(f"  Routing weight (mean): {layer_stats['routing_weights'].mean():.4f}")

    results['routing'] = {
        'router_score_mean': float(layer_stats['router_scores'].mean()),
        'routing_weight_mean': float(layer_stats['routing_weights'].mean()),
    }

    # Neuron Usage Analysis
    print("\n⭐ NEURON USAGE:")
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
            'top10_usage': usage.topk(10)[0].cpu().numpy().tolist(),
        }

    return results


# ============================================================
# 4. Summary Metrics
# ============================================================

def compute_summary_metrics(all_results):
    """Compute summary statistics across all analyses"""
    print("\n" + "="*60)
    print("4. SUMMARY METRICS")
    print("="*60)

    summary = {}

    # Basis orthogonality
    if 'basis' in all_results:
        summary['basis_orthogonal'] = all_results['basis']['basis']['max_off_diagonal'] < 1e-5

    # Recipe diversity (average across Q/K/V/O and layers)
    recipe_entropies = []
    recipe_specializations = []
    if 'recipes' in all_results:
        for layer_key, layer_data in all_results['recipes'].items():
            if layer_key.startswith('layer_'):
                for recipe_type in ['Q', 'K', 'V', 'O']:
                    recipe_key = f'recipe_{recipe_type}'
                    if recipe_key in layer_data:
                        recipe_entropies.append(layer_data[recipe_key]['recipe_entropy_mean'])
                        recipe_specializations.append(layer_data[recipe_key]['neuron_specialization_mean'])

    summary['recipe_entropy_avg'] = float(np.mean(recipe_entropies)) if recipe_entropies else 0.0
    summary['recipe_specialization_avg'] = float(np.mean(recipe_specializations)) if recipe_specializations else 0.0

    # Neuron usage (average across layers)
    usage_rates = []
    gini_coeffs = []
    if 'runtime' in all_results:
        for key, value in all_results['runtime'].items():
            if key.startswith('usage_layer_'):
                usage_rates.append(value['usage_rate'])
                gini_coeffs.append(value['gini'])

    summary['avg_neuron_usage_rate'] = float(np.mean(usage_rates)) if usage_rates else 0.0
    summary['avg_usage_gini'] = float(np.mean(gini_coeffs)) if gini_coeffs else 0.0

    print(f"\n📊 Summary:")
    print(f"  Basis orthogonal: {summary.get('basis_orthogonal', False)}")
    print(f"  Avg recipe entropy: {summary['recipe_entropy_avg']:.4f}")
    print(f"  Avg recipe specialization: {summary['recipe_specialization_avg']:.4f}")
    print(f"  Avg neuron usage: {summary['avg_neuron_usage_rate']*100:.1f}%")
    print(f"  Avg usage Gini: {summary['avg_usage_gini']:.4f}")

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
        model_version = model_config.get('model_version', '7.5')
    else:
        # Infer from state_dict keys
        state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model'
        state_dict = checkpoint[state_dict_key]

        if 'layers.0.qkv_dynamic.neuron_recipe_Q' in state_dict:
            model_version = '7.5'
            print("  Inferred model version: 7.5 (Dynamic Q/K/V)")
        else:
            raise ValueError("Cannot infer model version from checkpoint")

        # Infer model config from state_dict
        sample_layer = 'layers.0.qkv_dynamic'
        n_neurons = state_dict[f'{sample_layer}.neuron_recipe_Q'].shape[0]
        n_basis = state_dict[f'{sample_layer}.neuron_recipe_Q'].shape[1]

        model_config = {
            'vocab_size': state_dict['token_emb.weight'].shape[0],
            'd_model': state_dict['token_emb.weight'].shape[1],
            'n_layers': sum(1 for k in state_dict.keys() if k.startswith('layers.') and '.norm1.weight' in k),
            'n_neurons': n_neurons,
            'n_basis': n_basis,
            'model_version': model_version,
        }
        print(f"  Inferred config: {model_config}")

    # Create model
    print(f"\nCreating model v{model_version}...")
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
        print("   (Suppressing autotune logs...)")

        # Suppress autotune verbose output
        import os
        import logging
        os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
        logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
        logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

        # Disable autotune logging
        try:
            import torch._inductor.config as inductor_config
            inductor_config.trace.enabled = False
            inductor_config.trace.log_autotuning_results = False
        except:
            pass

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
    all_results['runtime'] = analyze_runtime_behavior(model, val_loader, device, args.max_batches)
    all_results['summary'] = compute_summary_metrics(all_results)

    # Save results
    output_path = Path(args.checkpoint) / 'analysis_v75.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Analysis complete! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
