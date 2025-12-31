#!/usr/bin/env python3
"""
DAWN Generation Test Script

Loads model config from checkpoint, no separate config file needed.
Supports multiple checkpoints and directory paths.

Usage:
    python scripts/generate_samples.py \
        --checkpoints /path/to/ckpt1 /path/to/ckpt2 \
        --val_data /path/to/val_data.pt \
        --output generation_results.txt
"""

import argparse
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from models import create_model_by_version, normalize_version


def find_checkpoint_file(ckpt_path):
    """Find checkpoint file from path (handles directories)"""
    ckpt_path = Path(ckpt_path)

    if ckpt_path.is_file():
        return ckpt_path

    if ckpt_path.is_dir():
        # Look for common checkpoint names
        candidates = ['best_model.pt', 'checkpoint_best.pt', 'model.pt', 'latest.pt']
        for name in candidates:
            if (ckpt_path / name).exists():
                return ckpt_path / name

        # Fall back to any .pt file
        pt_files = list(ckpt_path.glob('*.pt'))
        if pt_files:
            return sorted(pt_files)[-1]  # Most recent

    raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")


def load_model(checkpoint_path):
    """Load model from checkpoint (uses model_config from checkpoint)"""
    # Find actual checkpoint file
    ckpt_file = find_checkpoint_file(checkpoint_path)
    print(f"  Loading: {ckpt_file.name}")

    # Load checkpoint
    ckpt = torch.load(ckpt_file, map_location='cpu')

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
        # Check for version-specific keys (most specific first)
        # v18.2/v18.1: has tau_proj and separate norm layers per projection
        v18_2_keys = ['router.tau_proj.weight', 'router.neuron_router.norm_fqk_Q.weight']
        # v18.0: has multi-path but no learnable tau
        v18_0_keys = ['router.neuron_router.neuron_emb']
        # DAWN generic keys
        dawn_keys = ['shared_neurons.f_neurons', 'router.neuron_router.neuron_emb']

        if all(k in state_dict for k in v18_2_keys):
            version = '18.2'  # v18.2 with learnable tau and separate norms
        elif any(k in state_dict for k in dawn_keys):
            # Check if it's v18.x by looking at the config
            if model_config.get('learnable_tau', False) or model_config.get('max_paths'):
                version = '18.2'  # Assume latest v18 if has v18 config params
            else:
                version = '17.1'  # Default DAWN version
        else:
            version = 'baseline'

    version = normalize_version(version)

    # Create model using checkpoint's config
    model = create_model_by_version(version, model_config)

    model.load_state_dict(state_dict)

    # Get model name from path
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        name = ckpt_path.name
    else:
        name = ckpt_path.parent.name

    return model, version, name


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


def compute_perplexity(model, val_tokens, num_seqs=50, seq_len=512, device='cuda'):
    """Compute perplexity on validation data"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    # Get vocab size from model
    vocab_size = model.token_emb.weight.shape[0]

    with torch.no_grad():
        for i in range(num_seqs):
            start = i * seq_len
            seq = val_tokens[start:start+seq_len].unsqueeze(0).to(device)

            output = model(seq, attention_mask=None)
            logits = output[0] if isinstance(output, tuple) else output

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = seq[:, 1:].contiguous()

            loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            total_loss += loss.item() * (seq_len - 1)
            total_tokens += (seq_len - 1)

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, ppl


def test_single_checkpoint(ckpt_path, tokenizer, val_tokens, device='cuda'):
    """Test a single checkpoint and return results"""
    results = {}

    # Load model
    model, version, name = load_model(ckpt_path)
    model = model.to(device)
    model.eval()

    results['name'] = name
    results['version'] = version
    results['path'] = str(ckpt_path)

    # 1. Free generation - organized by category
    prompts = {
        'factual': [
            "The president of the United States is",
            "The largest ocean on Earth is",
            "Einstein developed the theory of",
            "The speed of light is",
            "World War 2 ended in",
        ],
        'common_sense': [
            "If you drop a glass, it will",
            "Fire is hot, ice is",
            "Birds can fly, fish can",
            "At night, the sky is",
            "When you are hungry, you",
        ],
        'narrative': [
            "Once upon a time, there was a",
            "She walked into the room and",
            "The detective found a clue that",
            "After years of training, he finally",
        ],
        'technical': [
            "def fibonacci(n):",
            "SELECT * FROM",
            "The mitochondria is the",
            "H2O is composed of",
        ],
        'conversational': [
            "Hey, how are you",
            "I think the best way to",
            "In my opinion,",
            "The problem with this approach is",
        ],
        'ambiguous': [
            "The best thing about",
            "I never thought that",
            "It was a dark and",
            "The reason why",
        ],
    }

    results['generations'] = {}
    for category, prompt_list in prompts.items():
        results['generations'][category] = []
        for prompt in prompt_list:
            output = generate_text(model, tokenizer, prompt, max_new_tokens=30, device=device)
            results['generations'][category].append({'prompt': prompt, 'output': output})

    # 2. C4 continuation
    sample_indices = [100, 500, 1000]
    cont_results = continuation_test(model, tokenizer, val_tokens, sample_indices, device=device)
    results['continuation'] = cont_results
    results['avg_match'] = sum(r['token_match'] for r in cont_results) / (len(sample_indices) * 30)

    # 3. Perplexity
    avg_loss, ppl = compute_perplexity(model, val_tokens, num_seqs=50, device=device)
    results['loss'] = avg_loss
    results['ppl'] = ppl

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description='DAWN Generation Test')
    parser.add_argument('--checkpoints', nargs='+', required=True, help='Paths to checkpoints')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data')
    parser.add_argument('--output', type=str, default='generation_results.txt', help='Output file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    # Device check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load validation data
    print(f"Loading validation data from {args.val_data}")
    val_data = torch.load(args.val_data, map_location='cpu')
    if isinstance(val_data, dict):
        val_tokens = val_data.get('tokens', val_data.get('input_ids', None))
    else:
        val_tokens = val_data
    val_tokens = val_tokens.long().view(-1)

    # Test each checkpoint
    all_results = []
    output_lines = []

    for ckpt_path in args.checkpoints:
        print(f"\n{'='*60}")
        print(f"Testing: {ckpt_path}")
        print('='*60)

        try:
            results = test_single_checkpoint(ckpt_path, tokenizer, val_tokens, args.device)
            all_results.append(results)

            # Format output
            output_lines.append("=" * 70)
            output_lines.append(f"Model: {results['name']} ({results['version']})")
            output_lines.append(f"Path: {results['path']}")
            output_lines.append("=" * 70)

            output_lines.append("\n[FREE GENERATION]")
            for category, gens in results['generations'].items():
                output_lines.append(f"\n--- {category.upper()} ---")
                for gen in gens:
                    output_lines.append(f"Prompt: '{gen['prompt']}'")
                    output_lines.append(f"  → {gen['output']}")

            output_lines.append("\n" + "-" * 50)
            output_lines.append("[C4 CONTINUATION]")
            for r in results['continuation']:
                output_lines.append(f"\n[Sample {r['sample_idx']}]")
                output_lines.append(f"Actual:    {r['actual'][:60]}...")
                output_lines.append(f"Generated: {r['generated'][:60]}...")
                output_lines.append(f"Match: {r['token_match']}/30 ({r['match_rate']*100:.1f}%)")

            output_lines.append(f"\nAvg Match: {results['avg_match']*100:.1f}%")
            output_lines.append(f"Loss: {results['loss']:.4f} | PPL: {results['ppl']:.2f}")
            output_lines.append("")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            output_lines.append(f"ERROR for {ckpt_path}: {e}\n")

    # Summary table
    output_lines.append("\n" + "=" * 70)
    output_lines.append("SUMMARY")
    output_lines.append("=" * 70)
    output_lines.append(f"{'Model':<50} {'PPL':>8} {'Match':>8}")
    output_lines.append("-" * 70)
    for r in all_results:
        output_lines.append(f"{r['name']:<50} {r['ppl']:>8.2f} {r['avg_match']*100:>7.1f}%")

    # Write output
    output_text = "\n".join(output_lines)
    print("\n" + output_text)

    with open(args.output, 'w') as f:
        f.write(output_text)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
