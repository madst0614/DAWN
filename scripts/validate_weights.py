#!/usr/bin/env python3
"""Validate DAWN weights on C4 validation set (streaming)."""

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


def validate(model, dataloader, device, max_batches=200):
    """Evaluate model - matches training eval exactly"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", total=max_batches)):
            if batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # CLM evaluation (same as training)
            logits = model(input_ids, attention_mask=attention_mask)

            # Create labels with padding masked
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            # Shift for autoregressive loss
            B, S, V = logits.shape
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().long()

            # Count valid tokens
            valid_mask = (shift_labels != -100)
            valid_tokens = valid_mask.sum().item()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
                ignore_index=-100
            )

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def collate_fn(batch, tokenizer, max_seq_len=512):
    """Collate with FIXED padding to max_seq_len (same as training)"""
    input_ids_list = []
    attention_mask_list = []

    for sample in batch:
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        seq_len = len(input_ids)

        # Pad to fixed length
        if seq_len < max_seq_len:
            padding_len = max_seq_len - seq_len
            input_ids = torch.cat([input_ids, torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_len, dtype=torch.long)])
        elif seq_len > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            attention_mask = attention_mask[:max_seq_len]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str, help='Path to weights file')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_batches', type=int, default=200,
                       help='Max batches to evaluate (default: 200, same as training)')
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # Load model
    print(f"Loading model from {args.weights}")
    model = DAWN(**CONFIG)
    ckpt = torch.load(args.weights, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(args.device)
    model.eval()

    # Load tokenizer (same as training: bert-base-uncased)
    print("Loading tokenizer (bert-base-uncased)...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load C4 validation (streaming)
    print("Loading C4 validation set (streaming)...")
    dataset = load_dataset('allenai/c4', 'en', split='validation', streaming=True)

    # Collect samples (enough for max_batches)
    max_samples = args.max_batches * args.batch_size
    samples = []
    print(f"Tokenizing {max_samples} samples...")

    for i, example in enumerate(tqdm(dataset, total=max_samples, desc="Loading")):
        if i >= max_samples:
            break

        # Tokenize (NO padding here - will be done in collate)
        tokens = tokenizer(
            example['text'],
            truncation=True,
            max_length=args.seq_len,
            return_tensors='pt'
        )
        samples.append({
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0)
        })

    print(f"Loaded {len(samples)} samples")

    # Create dataloader with fixed padding collate
    from functools import partial
    collate = partial(collate_fn, tokenizer=tokenizer, max_seq_len=args.seq_len)
    dataloader = DataLoader(samples, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Validate
    avg_loss, ppl = validate(model, dataloader, args.device, args.max_batches)

    print(f"\n{'='*40}")
    print(f"Validation Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  PPL:  {ppl:.2f}")
    print(f"{'='*40}")


if __name__ == '__main__':
    main()
