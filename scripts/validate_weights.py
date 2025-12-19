#!/usr/bin/env python3
"""Validate DAWN weights on C4 validation set."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset
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


class TokenDataset(Dataset):
    """Dataset for pre-tokenized data (from .pt files)"""
    def __init__(self, tokens, max_length=512):
        self.tokens = tokens
        self.max_length = max_length

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        input_ids = self.tokens[idx]
        seq_len = input_ids.shape[0]
        if seq_len > self.max_length:
            input_ids = input_ids[:self.max_length]
            seq_len = self.max_length
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


def collate_fn(batch, pad_token_id=0, max_seq_len=512):
    """Collate with padding to max_seq_len"""
    input_ids_list = []
    attention_mask_list = []

    for sample in batch:
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        seq_len = input_ids.shape[0]

        if seq_len < max_seq_len:
            padding_len = max_seq_len - seq_len
            input_ids = torch.cat([input_ids, torch.full((padding_len,), pad_token_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_len, dtype=torch.long)])

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list)
    }


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask=attention_mask)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            valid_mask = (shift_labels != -100)
            valid_tokens = valid_mask.sum().item()

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
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to pre-tokenized val data (.pt file)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max validation samples (default: all)')
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

    # Load tokenizer (for pad_token_id)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load pre-tokenized val data
    print(f"Loading val data from {args.val_data}")
    data = torch.load(args.val_data)
    if isinstance(data, dict) and 'tokens' in data:
        tokens = data['tokens']
    elif isinstance(data, torch.Tensor):
        tokens = data
    else:
        raise ValueError(f"Unknown .pt format: {type(data)}")

    # Reshape if 1D
    if tokens.dim() == 1:
        num_seqs = tokens.shape[0] // args.seq_len
        tokens = tokens[:num_seqs * args.seq_len].view(num_seqs, args.seq_len)

    print(f"Val data shape: {tokens.shape}")

    # Limit samples if specified
    if args.max_samples and len(tokens) > args.max_samples:
        tokens = tokens[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")

    # Create dataset and dataloader
    dataset = TokenDataset(tokens, args.seq_len)
    from functools import partial
    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id, max_seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Validate
    avg_loss, ppl = validate(model, dataloader, args.device)

    print(f"\n{'='*40}")
    print(f"Validation Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  PPL:  {ppl:.2f}")
    print(f"{'='*40}")


if __name__ == '__main__':
    main()
