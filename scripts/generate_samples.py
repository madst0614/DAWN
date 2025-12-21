#!/usr/bin/env python3
"""
DAWN Generation Test Script

Loads model config from checkpoint, no separate config file needed.

Usage:
    python scripts/generate_samples.py \
        --checkpoint /path/to/checkpoint \
        --val_data /path/to/val_data.pt \
        --output generation_results.txt
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from models import create_model_by_version, normalize_version


def load_model(checkpoint_path):
    """Load model from checkpoint (uses model_config from checkpoint)"""
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Get config from checkpoint
    model_config = ckpt.get('model_config', ckpt.get('config', {}))
    if not model_config:
        raise ValueError("No model_config found in checkpoint")

    # Get state_dict first for auto-detection
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = ckpt

    # Auto-detect version from state_dict keys if not in config
    version = model_config.get('model_version', None)
    if version is None:
        # Check for DAWN-specific keys
        dawn_keys = ['shared_neurons.f_neurons', 'router.neuron_router.neuron_emb']
        if any(k in state_dict for k in dawn_keys):
            version = '17.1'  # Default DAWN version
        else:
            version = 'baseline'

    version = normalize_version(version)

    # Create model using checkpoint's config
    model = create_model_by_version(version, model_config)

    model.load_state_dict(state_dict)
    return model, version


def generate_text(model, tokenizer, prompt, max_new_tokens=30, temperature=0.8, top_k=50, device='cuda'):
    """Generate text with top-k sampling"""
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(generated, attention_mask=None)
            logits = output[0] if isinstance(output, tuple) else output
            next_token_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def continuation_test(model, tokenizer, val_tokens, sample_indices, context_len=128, gen_len=30, device='cuda'):
    """Test continuation on validation data"""
    results = []

    for sample_idx in sample_indices:
        start = sample_idx * 512

        context_tokens = val_tokens[start:start+context_len].unsqueeze(0).to(device)
        actual_next = val_tokens[start+context_len:start+context_len+gen_len]

        context_text = tokenizer.decode(context_tokens[0], skip_special_tokens=True)
        actual_text = tokenizer.decode(actual_next, skip_special_tokens=True)

        # Greedy generation
        with torch.no_grad():
            generated = context_tokens.clone()
            for _ in range(gen_len):
                output = model(generated, attention_mask=None)
                logits = output[0] if isinstance(output, tuple) else output
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

        generated_tokens = generated[0, context_len:context_len+gen_len]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        match = (generated_tokens.cpu() == actual_next).sum().item()

        results.append({
            'sample_idx': sample_idx,
            'context': context_text[-100:],
            'actual': actual_text,
            'generated': generated_text,
            'token_match': match,
            'match_rate': match / gen_len,
        })

    return results


def compute_perplexity(model, val_tokens, num_seqs=10, device='cuda'):
    """Compute perplexity on validation data"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i in range(num_seqs):
            start = i * 512
            seq = val_tokens[start:start+512].unsqueeze(0).to(device)

            output = model(seq, attention_mask=None)
            logits = output[0] if isinstance(output, tuple) else output

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = seq[:, 1:].contiguous()

            loss = F.cross_entropy(shift_logits.view(-1, 30522), shift_labels.view(-1))
            total_loss += loss.item() * 511
            total_tokens += 511

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(description='DAWN Generation Test')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data')
    parser.add_argument('--output', type=str, default='generation_results.txt', help='Output file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, version = load_model(args.checkpoint)
    model = model.to(args.device)
    model.eval()

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load validation data
    print(f"Loading validation data from {args.val_data}")
    val_data = torch.load(args.val_data, map_location='cpu')
    val_tokens = val_data['tokens'].long()

    # Output
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append(f"DAWN Generation Test - {version}")
    output_lines.append(f"Checkpoint: {args.checkpoint}")
    output_lines.append("=" * 70)

    # 1. Free generation
    output_lines.append("\n[FREE GENERATION]")
    prompts = [
        "The weather today is",
        "Scientists discovered that",
        "The capital of France is",
        "Once upon a time",
        "In the year 2050",
    ]

    for prompt in prompts:
        output = generate_text(model, tokenizer, prompt, max_new_tokens=30, device=args.device)
        output_lines.append(f"\nPrompt: '{prompt}'")
        output_lines.append(f"Output: {output}")

    # 2. C4 continuation
    output_lines.append("\n" + "=" * 70)
    output_lines.append("[C4 CONTINUATION TEST]")

    sample_indices = [100, 500, 1000, 2000, 5000]
    results = continuation_test(model, tokenizer, val_tokens, sample_indices, device=args.device)

    total_match = 0
    for r in results:
        output_lines.append(f"\n[Sample {r['sample_idx']}]")
        output_lines.append(f"Context: ...{r['context']}")
        output_lines.append(f"Actual:    {r['actual'][:80]}...")
        output_lines.append(f"Generated: {r['generated'][:80]}...")
        output_lines.append(f"Token match: {r['token_match']}/30 ({r['match_rate']*100:.1f}%)")
        total_match += r['token_match']

    avg_match = total_match / (len(sample_indices) * 30)
    output_lines.append(f"\nAverage token match: {avg_match*100:.1f}%")

    # 3. Perplexity
    output_lines.append("\n" + "=" * 70)
    output_lines.append("[PERPLEXITY CHECK]")

    avg_loss, ppl = compute_perplexity(model, val_tokens, num_seqs=50, device=args.device)
    output_lines.append(f"Avg Loss: {avg_loss:.4f}")
    output_lines.append(f"Perplexity: {ppl:.2f}")

    # Write output
    output_text = "\n".join(output_lines)
    print(output_text)

    with open(args.output, 'w') as f:
        f.write(output_text)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
