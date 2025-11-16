"""
Train SPROUT (Neuron Pool version) on Masked Language Modeling.

New architecture with unified neuron pool and hard routing.

Usage:
    python scripts/train_neuron_pool.py \\
        --num_epochs 3 \\
        --batch_size 32 \\
        --pool_size 4096 \\
        --k 128
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sprout import SPROUT_MLM


# ============================================================================
# Data Loading (reuse from original script)
# ============================================================================

def load_wikipedia_data(max_samples=50000, streaming=True):
    """Load Wikipedia dataset."""
    print("🔄 Loading Wikipedia dataset...")

    try:
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split="train",
            streaming=streaming,
            trust_remote_code=False
        )
        print("✅ Loaded Salesforce/wikitext-103")
    except Exception as e:
        print(f"⚠️  Failed to load wikitext: {e}")
        return generate_fallback_sentences(max_samples)

    # Filter and collect sentences
    sentences = []
    max_items = max_samples * 10

    for idx, item in enumerate(dataset):
        if idx >= max_items:
            break

        text = item.get("text", "").strip()
        if not text or text.startswith("="):
            continue

        sents = [s.strip() for s in re.split(r'[.!?]+', text)]

        for sent in sents:
            word_count = len(sent.split())
            if 5 <= word_count <= 50:
                sentences.append(sent)

            if len(sentences) >= max_samples:
                break

        if len(sentences) >= max_samples:
            break

    print(f"✅ Loaded {len(sentences)} sentences")
    return sentences


def generate_fallback_sentences(num_samples):
    """Generate simple fallback sentences."""
    base_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of data.",
        "Neural networks are inspired by the human brain.",
    ]
    sentences = base_sentences * (num_samples // len(base_sentences) + 1)
    return sentences[:num_samples]


class MLMDataset(Dataset):
    """Masked Language Modeling Dataset."""

    def __init__(self, sentences, tokenizer, max_length=128, mask_prob=0.15):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Create MLM labels
        labels = input_ids.clone()

        # Probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Only compute loss on masked tokens
        labels[~masked_indices] = -100

        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, replace with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 10% of the time, keep original

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SPROUT (Neuron Pool) on MLM")

    # Model config
    parser.add_argument("--d_model", type=int, default=256,
                        help="Model dimension (default: 256)")
    parser.add_argument("--pool_size", type=int, default=4096,
                        help="Number of neurons in pool (default: 4096)")
    parser.add_argument("--k", type=int, default=128,
                        help="Neurons selected per token (default: 128)")
    parser.add_argument("--n_steps", type=int, default=3,
                        help="Number of SPROUT layers (default: 3)")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--router_temperature", type=float, default=1.0,
                        help="Gumbel-Softmax temperature (default: 1.0)")
    parser.add_argument("--load_balance_weight", type=float, default=0.01,
                        help="Load balance loss weight (default: 0.01)")

    # Training config
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Warmup steps")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")

    # Data config
    parser.add_argument("--max_samples", type=int, default=50000,
                        help="Maximum training samples")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--mask_prob", type=float, default=0.15,
                        help="MLM masking probability")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=5000,
                        help="Save checkpoint every N steps")

    # Logging
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--visualize_neurons", action="store_true",
                        help="Visualize neuron usage after training")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    # Debug
    parser.add_argument("--debug_mode", action="store_true",
                        help="Use tiny dataset for debugging")

    args = parser.parse_args()
    return args


def create_dataloader(args, tokenizer):
    """Create MLM dataloader."""
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")

    if args.debug_mode:
        print("🐛 Debug mode: Using simple sentences")
        sentences = generate_fallback_sentences(500)
    else:
        sentences = load_wikipedia_data(
            max_samples=args.max_samples,
            streaming=False
        )

    dataset = MLMDataset(
        sentences=sentences,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mask_prob=args.mask_prob
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    print(f"✅ Created dataloader with {len(dataset)} samples")
    print(f"   Batches per epoch: {len(dataloader)}")

    return dataloader


def train_epoch(model, dataloader, optimizer, scheduler, scaler, args, epoch, device):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_correct = 0
    total_tokens = 0
    total_load_balance_loss = 0
    num_batches_with_lb = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", ncols=100)

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        if args.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs["loss"]
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs["loss"]

        # Backward pass
        optimizer.zero_grad()

        if args.mixed_precision and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Calculate accuracy
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct = ((predictions == labels) & mask).sum().item()
        total = mask.sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_tokens += total

        # Track load balance loss
        lb_loss = outputs.get("load_balance_loss")
        if lb_loss is not None:
            total_load_balance_loss += lb_loss.item()
            num_batches_with_lb += 1

        # Update progress bar
        if batch_idx % 10 == 0:
            acc = 100.0 * total_correct / total_tokens if total_tokens > 0 else 0.0
            postfix = {
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.1f}%',
            }
            if lb_loss is not None:
                postfix['lb_loss'] = f'{lb_loss.item():.4f}'
            progress_bar.set_postfix(postfix)

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100.0 * total_correct / total_tokens if total_tokens > 0 else 0.0
    avg_lb_loss = total_load_balance_loss / num_batches_with_lb if num_batches_with_lb > 0 else 0

    return avg_loss, avg_acc, avg_lb_loss


def main():
    """Main training function."""
    args = get_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"SPROUT (Neuron Pool) MLM Training")
    print(f"{'='*70}")
    print(f"Device: {device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(f"✅ Loaded tokenizer (vocab size: {len(tokenizer)})")

    # Model
    print(f"\n{'='*70}")
    print("CREATING MODEL")
    print(f"{'='*70}")
    print(f"  - Model dimension: {args.d_model}")
    print(f"  - Pool size: {args.pool_size}")
    print(f"  - k (neurons per token): {args.k}")
    print(f"  - Steps: {args.n_steps}")
    print(f"  - Load balance weight: {args.load_balance_weight}")

    model = SPROUT_MLM(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        pool_size=args.pool_size,
        k=args.k,
        n_steps=args.n_steps,
        n_heads=args.n_heads,
        router_temperature=args.router_temperature,
        load_balance_weight=args.load_balance_weight
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created")
    print(f"   Total params: {total_params:,}")

    # Create dataloader
    dataloader = create_dataloader(args, tokenizer)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Scheduler
    total_steps = len(dataloader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_steps
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # Training loop
    best_loss = float('inf')

    for epoch in range(args.num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*70}")

        epoch_loss, epoch_acc, epoch_lb_loss = train_epoch(
            model, dataloader, optimizer, scheduler, scaler, args, epoch, device
        )

        print(f"\nEpoch {epoch+1}/{args.num_epochs} Summary:")
        print(f"  Loss: {epoch_loss:.4f}")
        print(f"  Accuracy: {epoch_acc:.2f}%")
        if args.load_balance_weight > 0:
            print(f"  Load balance loss: {epoch_lb_loss:.4f}")

        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, "sprout_neuron_pool_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'accuracy': epoch_acc,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✅ Saved best checkpoint: {checkpoint_path}")

    # Final visualization
    if args.visualize_neurons:
        print(f"\n{'='*70}")
        print("NEURON USAGE VISUALIZATION")
        print(f"{'='*70}")

        # Get sample batch
        sample_batch = next(iter(dataloader))
        sample_ids = sample_batch['input_ids'][:4].to(device)

        model.visualize_neuron_usage(sample_ids)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoint saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
