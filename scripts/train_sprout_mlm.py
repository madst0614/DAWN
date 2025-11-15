"""
Train SPROUT on Masked Language Modeling (MLM) task.

Self-contained training script - no external dependencies on dawn utils.

Usage (Colab):
    !python scripts/train_sprout_mlm.py \
        --checkpoint_dir /content/drive/MyDrive/sprout/checkpoints/ \
        --num_epochs 3 \
        --batch_size 32

Usage (Local):
    python scripts/train_sprout_mlm.py \
        --checkpoint_dir ./checkpoints \
        --num_epochs 3
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

from sprout import SproutLanguageModel


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_wikipedia_data(max_samples=50000, streaming=True):
    """
    Load Wikipedia dataset.

    Args:
        max_samples: Maximum number of samples
        streaming: Use streaming mode

    Returns:
        List of sentences
    """
    print("🔄 Loading Wikipedia dataset...")

    try:
        # Try WikiText-103 (smaller, faster)
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
        # Fallback to simple sentences
        return generate_fallback_sentences(max_samples)

    # Filter and collect sentences
    sentences = []
    max_items = max_samples * 10  # Over-sample for filtering

    for idx, item in enumerate(dataset):
        if idx >= max_items:
            break

        text = item.get("text", "").strip()
        if not text or text.startswith("="):  # Skip titles
            continue

        # Split into sentences
        sents = [s.strip() for s in re.split(r'[.!?]+', text)]

        for sent in sents:
            word_count = len(sent.split())
            if 5 <= word_count <= 50:  # Filter by length
                sentences.append(sent)

            if len(sentences) >= max_samples:
                break

        if len(sentences) >= max_samples:
            break

    print(f"✅ Loaded {len(sentences)} sentences")
    return sentences


def generate_fallback_sentences(num_samples):
    """Generate simple fallback sentences for testing."""
    base_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of data.",
        "Neural networks are inspired by the human brain.",
        "Transformers have revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Self-supervised learning reduces the need for labeled data.",
        "Language models can generate coherent and contextual text.",
        "Gradient descent is used to optimize neural network parameters.",
    ]

    sentences = base_sentences * (num_samples // len(base_sentences) + 1)
    return sentences[:num_samples]


# ============================================================================
# MLM Dataset
# ============================================================================

class MLMDataset(Dataset):
    """
    Masked Language Modeling Dataset.

    Implements BERT-style MLM masking:
    - 80% of masked tokens → [MASK]
    - 10% of masked tokens → random token
    - 10% of masked tokens → unchanged
    """

    def __init__(self, sentences, tokenizer, max_length=128, mask_prob=0.15):
        """
        Initialize MLM dataset.

        Args:
            sentences: List of text sentences
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            mask_prob: Probability of masking (default: 15%)
        """
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
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Apply MLM masking
        input_ids, labels = self._apply_mlm_masking(input_ids, labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def _apply_mlm_masking(self, input_ids, labels):
        """
        Apply BERT-style MLM masking.

        Args:
            input_ids: Token IDs [seq_len]
            labels: Label IDs [seq_len]

        Returns:
            Masked input_ids and labels
        """
        # Create masking probability matrix
        probability_matrix = torch.full(labels.shape, self.mask_prob)

        # Don't mask special tokens
        special_tokens_mask = torch.tensor([
            self.tokenizer.get_special_tokens_mask([val], already_has_special_tokens=True)[0]
            for val in labels.tolist()
        ], dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Don't mask padding
        padding_mask = input_ids == self.tokenizer.pad_token_id
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        # Sample masked positions
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% of the time: replace with [MASK]
        indices_replaced = masked_indices & (torch.rand(labels.shape) < 0.8)
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time: replace with random token
        indices_random = masked_indices & ~indices_replaced & (torch.rand(labels.shape) < 0.5)
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 10% of the time: keep original (already done)

        return input_ids, labels


# ============================================================================
# Configuration
# ============================================================================

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SPROUT on MLM")

    # Model config
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension")
    parser.add_argument("--max_depth", type=int, default=2,
                        help="Maximum tree depth (2 = ~5 nodes)")
    parser.add_argument("--max_nodes", type=int, default=5,
                        help="Hard limit on total nodes")
    parser.add_argument("--compatibility_threshold", type=float, default=0.8,
                        help="Compatibility threshold for branching")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")

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
    parser.add_argument("--visualize_structure", action="store_true",
                        help="Visualize tree structure after training")

    # Debug
    parser.add_argument("--debug_mode", action="store_true",
                        help="Use tiny dataset for debugging")

    args = parser.parse_args()
    return args


# ============================================================================
# Training Functions
# ============================================================================

def create_dataloader(args, tokenizer):
    """Create MLM dataloader."""
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")

    # Load sentences
    if args.debug_mode:
        print("🐛 Debug mode: Using simple sentences")
        sentences = generate_fallback_sentences(500)
    else:
        sentences = load_wikipedia_data(
            max_samples=args.max_samples,
            streaming=True
        )

    print(f"   Total sentences: {len(sentences)}")

    # Create dataset
    dataset = MLMDataset(
        sentences=sentences,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mask_prob=args.mask_prob
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    print(f"   Batches per epoch: {len(dataloader)}")
    print(f"{'='*70}\n")

    return dataloader


def train_epoch(model, dataloader, optimizer, scheduler, scaler, args, epoch, device):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_correct = 0
    total_tokens = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", ncols=100)

    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
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

        # Update progress bar
        if batch_idx % 10 == 0:
            acc = 100.0 * total_correct / total_tokens if total_tokens > 0 else 0.0
            num_nodes = outputs.get("num_nodes", 0)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.1f}%',
                'nodes': num_nodes
            })

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100.0 * total_correct / total_tokens if total_tokens > 0 else 0.0

    return avg_loss, avg_acc


def main():
    """Main training function."""
    args = get_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("SPROUT MLM TRAINING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"{'='*70}\n")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")

    # Create model
    print(f"\nCreating SPROUT model...")
    print(f"  - Hidden dim: {args.hidden_dim}")
    print(f"  - Max depth: {args.max_depth}")
    print(f"  - Max nodes: {args.max_nodes}")
    print(f"  - Compatibility threshold: {args.compatibility_threshold}")

    model = SproutLanguageModel(
        vocab_size=len(tokenizer),
        hidden_dim=args.hidden_dim,
        max_depth=args.max_depth,
        compatibility_threshold=args.compatibility_threshold,
        num_heads=args.num_heads,
        max_nodes=args.max_nodes
    ).to(device)

    model_info = model.get_model_info()
    print(f"✅ Model created")
    print(f"   Total params: {model_info['total_params']:,}")
    print(f"   Initial nodes: {model_info['total_nodes']}")

    # Create dataloader
    dataloader = create_dataloader(args, tokenizer)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    total_steps = len(dataloader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_steps
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"{'='*70}\n")

    # Training loop
    best_loss = float('inf')

    for epoch in range(args.num_epochs):
        epoch_loss, epoch_acc = train_epoch(
            model, dataloader, optimizer, scheduler, scaler, args, epoch, device
        )

        print(f"\nEpoch {epoch+1}/{args.num_epochs} Summary:")
        print(f"  Loss: {epoch_loss:.4f}")
        print(f"  Accuracy: {epoch_acc:.2f}%")

        # Show structure info
        model_info = model.get_model_info()
        print(f"  Total nodes: {model_info['total_nodes']}/{args.max_nodes}")
        print(f"  Node limit reached: {model_info['node_limit_reached']}")

        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, "sprout_mlm_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'accuracy': epoch_acc,
                'model_info': model_info,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✅ Saved best checkpoint: {checkpoint_path}")

    # Final visualization
    if args.visualize_structure:
        print(f"\n{'='*70}")
        print("FINAL STRUCTURE")
        print(f"{'='*70}")
        model.visualize_structure(max_depth=args.max_depth)

        # Show statistics
        stats = model_info['sprout_stats']
        print(f"\nStatistics:")
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Nodes by depth: {stats['nodes_by_depth']}")
        print(f"  Total branches created: {stats['total_branches']}")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final nodes: {model_info['total_nodes']}/{args.max_nodes}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
