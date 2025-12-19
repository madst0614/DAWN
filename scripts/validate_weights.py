#!/usr/bin/env python3
"""Validate DAWN weights on C4 validation set."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import math

from models import DAWN

# Config (v17.1)
CONFIG = {
    'vocab_size': 30522,
    'd_model': 384,
    'n_layers': 12,
    'n_heads': 6,
    'rank': 64,
    'knowledge_rank': 128,
    'n_feature_qk': 120,
    'n_feature_v': 24,
    'n_restore_qk': 120,
    'n_restore_v': 24,
    'n_feature_know': 24,
    'n_restore_know': 24,
    'top_k_feature_qk': 20,
    'top_k_feature_v': 6,
    'top_k_restore_qk': 20,
    'top_k_restore_v': 6,
    'top_k_feature_know': 4,
    'top_k_restore_know': 4,
}


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward (pass attention_mask like training)
            logits = model(input_ids, attention_mask=attention_mask)

            # Create labels with padding masked (like training)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            # Shift for causal LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Valid tokens (non-padding)
            valid_mask = (shift_labels != -100)
            valid_tokens = valid_mask.sum().item()

            # Cross entropy with ignore_index
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str, help='Path to weights file')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Max validation samples (default: 1000)')
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # Load model
    print(f"Loading model from {args.weights}")
    model = DAWN(**CONFIG)
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load C4 validation
    print("Loading C4 validation set...")
    dataset = load_dataset('allenai/c4', 'en', split='validation', streaming=True)

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.seq_len,
            padding='max_length',
            return_tensors='pt'
        )

    # Collect samples
    samples = []
    for i, example in enumerate(dataset):
        if i >= args.max_samples:
            break
        tokens = tokenizer(
            example['text'],
            truncation=True,
            max_length=args.seq_len,
            padding='max_length',
            return_tensors='pt'
        )
        samples.append({
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0)
        })

    print(f"Loaded {len(samples)} samples")

    # DataLoader
    dataloader = DataLoader(samples, batch_size=args.batch_size, shuffle=False)

    # Validate
    avg_loss, ppl = validate(model, dataloader, args.device)

    print(f"\n{'='*40}")
    print(f"Validation Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  PPL:  {ppl:.2f}")
    print(f"{'='*40}")


if __name__ == '__main__':
    main()
