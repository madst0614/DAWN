#!/usr/bin/env python3
"""
DAWN v16 Neuron Ablation Test
==============================
특정 뉴런 비활성화 시 성능 변화 측정

Usage:
    python ablation_test.py --checkpoint path/to/checkpoint.pt --val_data path/to/val.pt
"""

import argparse
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_checkpoint_path(path):
    """Resolve path to actual checkpoint file"""
    import glob
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        patterns = ["*.pt", "*.pth"]
        all_ckpts = []
        for pattern in patterns:
            all_ckpts.extend(glob.glob(os.path.join(path, pattern)))
            all_ckpts.extend(glob.glob(os.path.join(path, "**", pattern), recursive=True))
        if all_ckpts:
            for ckpt in all_ckpts:
                name = os.path.basename(ckpt).lower()
                if 'best' in name or 'final' in name:
                    return ckpt
            all_ckpts.sort(key=os.path.getmtime, reverse=True)
            return all_ckpts[0]
    raise FileNotFoundError(f"No checkpoint found: {path}")


class AblationHook:
    """Hook to ablate specific neurons during forward pass"""

    def __init__(self, neuron_type, neuron_indices):
        """
        Args:
            neuron_type: 'FR', 'FV', 'R', 'V'
            neuron_indices: list of neuron indices to ablate
        """
        self.neuron_type = neuron_type
        self.neuron_indices = neuron_indices
        self.handles = []

    def _get_pref_key(self):
        return {
            'FR': 'feature_r_pref',
            'FV': 'feature_v_pref',
            'R': 'relational_q_pref',
            'V': 'value_pref',
        }.get(self.neuron_type)

    def _get_weight_key(self):
        return {
            'FR': 'feature_r_weights',
            'FV': 'feature_v_weights',
            'R': 'relational_weights_Q',
            'V': 'value_weights',
        }.get(self.neuron_type)


def compute_perplexity(model, dataloader, device, ablate_config=None, max_batches=None, filter_fn=None):
    """
    Compute perplexity with optional neuron ablation.

    Args:
        model: DAWN model
        dataloader: validation data
        device: cuda/cpu
        ablate_config: dict of {neuron_type: [indices]} to ablate, e.g., {'R': [106]}
        max_batches: limit batches for speed
        filter_fn: function(input_ids, tokenizer) -> mask of sequences to include

    Returns:
        perplexity, n_tokens
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)

            B, L = input_ids.shape

            # Filter sequences if filter_fn provided
            if filter_fn is not None:
                mask = filter_fn(input_ids)
                if not mask.any():
                    continue
                input_ids = input_ids[mask]
                B = input_ids.shape[0]

            # Forward with routing info for ablation
            if ablate_config:
                output = model(input_ids, return_routing_info=True)
                logits = output[0] if isinstance(output, tuple) else output
                routing_infos = output[-1] if isinstance(output, tuple) and len(output) > 1 else None

                # Apply ablation by modifying the output
                # Since we can't easily hook into the forward pass,
                # we'll use a different approach: re-run with modified weights
                logits = forward_with_ablation(model, input_ids, ablate_config, device)
            else:
                output = model(input_ids)
                logits = output[0] if isinstance(output, tuple) else output

            # Compute loss (next token prediction)
            # logits: [B, L, vocab_size], targets: input_ids shifted
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum',
                ignore_index=0  # padding
            )

            # Count non-padding tokens
            n_tokens = (shift_labels != 0).sum().item()

            total_loss += loss.item()
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)

    return perplexity, total_tokens


def forward_with_ablation(model, input_ids, ablate_config, device):
    """
    Forward pass with neuron ablation.

    Modifies the router's weights temporarily to zero out specified neurons.
    """
    # Get the router
    if not hasattr(model, 'global_routers'):
        # Fallback: just run normal forward
        output = model(input_ids)
        return output[0] if isinstance(output, tuple) else output

    router = model.global_routers.neuron_router

    # Store original weights for restoration
    original_weights = {}

    # Temporarily modify neuron embeddings to produce zero weights for ablated neurons
    # This is a hack - we'll add a large negative bias to the ablated neurons
    ablation_biases = {}

    for neuron_type, indices in ablate_config.items():
        if not indices:
            continue

        # Map neuron type to embedding
        emb_name_map = {
            'FR': 'neuron_emb_feature_r',
            'FV': 'neuron_emb_feature_v',
            'R': 'neuron_emb_relational',
            'V': 'neuron_emb_value',
        }

        emb_name = emb_name_map.get(neuron_type)
        if emb_name and hasattr(router, emb_name):
            emb = getattr(router, emb_name)
            # Store original
            original_weights[emb_name] = emb.data.clone()
            # Set ablated neurons to large negative value (will become ~0 after softmax)
            emb.data[indices] = -1e9

    try:
        # Forward pass
        output = model(input_ids)
        logits = output[0] if isinstance(output, tuple) else output
    finally:
        # Restore original weights
        for emb_name, orig_data in original_weights.items():
            getattr(router, emb_name).data.copy_(orig_data)

    return logits


def create_sentence_filter(tokenizer, keywords):
    """Create a filter function for sentences containing specific keywords."""
    keyword_ids = []
    for kw in keywords:
        # Try different tokenizations
        ids = tokenizer.encode(kw, add_special_tokens=False)
        keyword_ids.extend(ids)
        # Also try with space prefix
        ids_space = tokenizer.encode(' ' + kw, add_special_tokens=False)
        keyword_ids.extend(ids_space)

    keyword_ids = set(keyword_ids)

    def filter_fn(input_ids):
        """Return mask of sequences containing any keyword."""
        # input_ids: [B, L]
        B = input_ids.shape[0]
        mask = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

        for b in range(B):
            seq_set = set(input_ids[b].cpu().tolist())
            if seq_set & keyword_ids:
                mask[b] = True

        return mask

    return filter_fn


def run_ablation_test(model, dataloader, tokenizer, device, config, args):
    """Run full ablation test suite."""

    results = {}

    print("\n" + "="*60)
    print("Neuron Ablation Test")
    print("="*60)

    # 1. Normal (baseline) perplexity
    print("\n[1/5] Computing baseline perplexity...")
    ppl_normal, n_tokens = compute_perplexity(
        model, dataloader, device,
        ablate_config=None,
        max_batches=args.max_batches
    )
    print(f"  Normal PPL: {ppl_normal:.4f} ({n_tokens:,} tokens)")
    results['normal'] = {'ppl': ppl_normal, 'n_tokens': n_tokens}

    # 2. R_106 ablated
    print("\n[2/5] Computing R_106 ablated perplexity...")
    ppl_r106, n_tokens = compute_perplexity(
        model, dataloader, device,
        ablate_config={'R': [106]},
        max_batches=args.max_batches
    )
    delta_r106 = (ppl_r106 - ppl_normal) / ppl_normal * 100
    print(f"  R_106 ablated PPL: {ppl_r106:.4f} ({delta_r106:+.2f}%)")
    results['R_106_ablated'] = {'ppl': ppl_r106, 'delta_pct': delta_r106}

    # 3. R_126 ablated (comparison)
    print("\n[3/5] Computing R_126 ablated perplexity...")
    ppl_r126, n_tokens = compute_perplexity(
        model, dataloader, device,
        ablate_config={'R': [126]},
        max_batches=args.max_batches
    )
    delta_r126 = (ppl_r126 - ppl_normal) / ppl_normal * 100
    print(f"  R_126 ablated PPL: {ppl_r126:.4f} ({delta_r126:+.2f}%)")
    results['R_126_ablated'] = {'ppl': ppl_r126, 'delta_pct': delta_r126}

    # 4. Random neuron ablated (baseline)
    print("\n[4/5] Computing random R neuron ablated perplexity...")
    n_relational = config.get('n_relational', 196)
    random_idx = np.random.randint(0, n_relational)
    while random_idx in [106, 126]:  # Avoid test neurons
        random_idx = np.random.randint(0, n_relational)

    ppl_random, n_tokens = compute_perplexity(
        model, dataloader, device,
        ablate_config={'R': [random_idx]},
        max_batches=args.max_batches
    )
    delta_random = (ppl_random - ppl_normal) / ppl_normal * 100
    print(f"  R_{random_idx} (random) ablated PPL: {ppl_random:.4f} ({delta_random:+.2f}%)")
    results['random_ablated'] = {'neuron': f'R_{random_idx}', 'ppl': ppl_random, 'delta_pct': delta_random}

    # 5. Sentence-type specific analysis
    print("\n[5/5] Sentence-type specific analysis...")

    # Filter for "of", "in", "from" sentences
    prep_filter = create_sentence_filter(tokenizer, ['of', 'in', 'from'])

    print("\n  On 'of/in/from' sentences:")
    ppl_prep_normal, n_tokens_prep = compute_perplexity(
        model, dataloader, device,
        ablate_config=None,
        max_batches=args.max_batches,
        filter_fn=prep_filter
    )
    print(f"    Normal PPL: {ppl_prep_normal:.4f} ({n_tokens_prep:,} tokens)")

    ppl_prep_r106, _ = compute_perplexity(
        model, dataloader, device,
        ablate_config={'R': [106]},
        max_batches=args.max_batches,
        filter_fn=prep_filter
    )
    delta_prep_r106 = (ppl_prep_r106 - ppl_prep_normal) / ppl_prep_normal * 100
    print(f"    R_106 ablated PPL: {ppl_prep_r106:.4f} ({delta_prep_r106:+.2f}%)")

    results['prep_sentences'] = {
        'keywords': ['of', 'in', 'from'],
        'normal_ppl': ppl_prep_normal,
        'R_106_ablated_ppl': ppl_prep_r106,
        'R_106_delta_pct': delta_prep_r106,
        'n_tokens': n_tokens_prep
    }

    # Filter for "especially" sentences (R_62 check)
    esp_filter = create_sentence_filter(tokenizer, ['especially'])

    print("\n  On 'especially' sentences:")
    ppl_esp_normal, n_tokens_esp = compute_perplexity(
        model, dataloader, device,
        ablate_config=None,
        max_batches=args.max_batches,
        filter_fn=esp_filter
    )

    if n_tokens_esp > 0:
        print(f"    Normal PPL: {ppl_esp_normal:.4f} ({n_tokens_esp:,} tokens)")

        ppl_esp_r62, _ = compute_perplexity(
            model, dataloader, device,
            ablate_config={'R': [62]},
            max_batches=args.max_batches,
            filter_fn=esp_filter
        )
        delta_esp_r62 = (ppl_esp_r62 - ppl_esp_normal) / ppl_esp_normal * 100
        print(f"    R_62 ablated PPL: {ppl_esp_r62:.4f} ({delta_esp_r62:+.2f}%)")

        results['especially_sentences'] = {
            'keywords': ['especially'],
            'normal_ppl': ppl_esp_normal,
            'R_62_ablated_ppl': ppl_esp_r62,
            'R_62_delta_pct': delta_esp_r62,
            'n_tokens': n_tokens_esp
        }
    else:
        print(f"    (No 'especially' sentences found)")
        results['especially_sentences'] = {'n_tokens': 0}

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"\nBaseline PPL: {ppl_normal:.4f}")
    print(f"\nAblation Impact:")
    print(f"  R_106: {delta_r106:+.2f}%")
    print(f"  R_126: {delta_r126:+.2f}%")
    print(f"  Random (R_{random_idx}): {delta_random:+.2f}%")

    if 'prep_sentences' in results and results['prep_sentences']['n_tokens'] > 0:
        print(f"\nOn preposition sentences (of/in/from):")
        print(f"  R_106 impact: {delta_prep_r106:+.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="DAWN v16 Neuron Ablation Test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./ablation_results")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve checkpoint path
    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")

    # Load model
    try:
        from models.model_v16 import DAWN
    except ImportError:
        from model_v16 import DAWN

    from transformers import BertTokenizer

    checkpoint = torch.load(ckpt_path, map_location=args.device)
    config = checkpoint.get('config', {})

    print(f"Model config: d_model={config.get('d_model')}, n_layers={config.get('n_layers')}")
    print(f"  n_relational={config.get('n_relational')}, top_k_relational={config.get('top_k_relational')}")

    model = DAWN(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load data
    print(f"Loading data: {args.val_data}")
    val_data = torch.load(args.val_data)
    if isinstance(val_data, dict):
        input_ids = val_data.get('input_ids', val_data.get('tokens'))
    else:
        input_ids = val_data

    if input_ids.dim() == 1:
        seq_len = config.get('max_seq_len', 512)
        n_seqs = input_ids.shape[0] // seq_len
        input_ids = input_ids[:n_seqs * seq_len].view(n_seqs, seq_len)

    dataset = torch.utils.data.TensorDataset(input_ids)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Run ablation test
    results = run_ablation_test(model, dataloader, tokenizer, args.device, config, args)

    # Save results
    import json
    output_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
