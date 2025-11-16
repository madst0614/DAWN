"""
Train Simple Neuron Pool V1 on MLM.

This is a standard Transformer with FFN replaced by neuron pool.
The model is mathematically identical - we just track neuron usage.

Usage:
    python scripts/train_v1_simple.py --num_epochs 3 --batch_size 32
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sprout.v1 import SimpleTransformerWithNeuronPool, full_analysis_report


# Reuse MLM dataset from other scripts
class MLMDataset:
    """MLM Dataset (same as before)."""

    def __init__(self, sentences, tokenizer, max_length=128, mask_prob=0.15):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()

        # MLM masking
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% [MASK], 10% random, 10% keep
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def load_data(max_samples=10000):
    """Load WikiText data."""
    print("🔄 Loading data...")

    try:
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split="train",
            streaming=True,
            trust_remote_code=False
        )
    except:
        # Fallback
        return ["This is a simple test sentence."] * max_samples

    sentences = []
    for idx, item in enumerate(dataset):
        if idx >= max_samples * 10:
            break

        text = item.get("text", "").strip()
        if not text or text.startswith("="):
            continue

        sents = [s.strip() for s in re.split(r'[.!?]+', text)]
        for sent in sents:
            if 5 <= len(sent.split()) <= 50:
                sentences.append(sent)
            if len(sentences) >= max_samples:
                break

        if len(sentences) >= max_samples:
            break

    print(f"✅ Loaded {len(sentences)} sentences")
    return sentences


def train_epoch(model, dataloader, optimizer, device, epoch, num_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']

        loss.backward()
        optimizer.step()

        # Calculate accuracy
        logits = outputs['logits']
        predictions = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct = ((predictions == labels) & mask).sum().item()
        total = mask.sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_tokens += total

        # Update progress bar
        acc = 100.0 * total_correct / total_tokens if total_tokens > 0 else 0.0
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.1f}%',
        })

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100.0 * total_correct / total_tokens

    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train V1 Simple Neuron Pool")

    # Model config
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=4)

    # Training config
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_samples", type=int, default=10000)

    # Analysis
    parser.add_argument("--analyze", action="store_true",
                        help="Run full analysis after training")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"SIMPLE NEURON POOL V1 - Training")
    print(f"{'='*70}")
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(f"✅ Loaded tokenizer (vocab: {len(tokenizer)})")

    # Model
    print(f"\n{'='*70}")
    print("MODEL")
    print(f"{'='*70}")
    print(f"  d_model: {args.d_model}")
    print(f"  d_ff: {args.d_ff}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  Total neurons: {args.n_layers * args.d_ff}")

    model = SimpleTransformerWithNeuronPool(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        n_heads=args.n_heads
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created ({total_params:,} parameters)")

    # Data
    sentences = load_data(args.max_samples)
    dataset = MLMDataset(sentences, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"✅ Created dataloader ({len(dataset)} samples, {len(dataloader)} batches)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")

    best_loss = float('inf')

    for epoch in range(args.num_epochs):
        epoch_loss, epoch_acc = train_epoch(
            model, dataloader, optimizer, device, epoch, args.num_epochs
        )

        print(f"\nEpoch {epoch+1}/{args.num_epochs}:")
        print(f"  Loss: {epoch_loss:.4f}")
        print(f"  Accuracy: {epoch_acc:.2f}%")

        if epoch_loss < best_loss:
            best_loss = epoch_loss

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss:.4f}")

    # Analysis
    if args.analyze:
        print("\nRunning analysis...")
        model.eval()
        full_analysis_report(model)

        # Visualize neuron usage
        print("\nNeuron usage after training:")
        model.visualize_neuron_usage()


if __name__ == '__main__':
    main()
