"""
Train SPROUT Brain-Like Architecture on MLM

완전히 새로운 뇌 기반 아키텍처 학습:
- 전역 뉴런 풀 (4096개 뉴런)
- Sparse 활성화 (128-256개만)
- 반복적 상호작용 (5 steps)
- 전체 시퀀스를 하나의 패턴으로 표현

Usage:
    python scripts/train_brain_like.py \
        --num_epochs 3 \
        --batch_size 32 \
        --n_neurons 4096 \
        --n_interaction_steps 5
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
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sprout_brain_like import create_brain_like_sprout


# ============================================================================
# Data Loading (Same as train_sprout_mlm.py)
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
    """Masked Language Modeling Dataset for Brain-Like SPROUT."""

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
        """Apply BERT-style MLM masking."""
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
        labels[~masked_indices] = -100

        # 80%: [MASK], 10%: random, 10%: unchanged
        indices_replaced = masked_indices & (torch.rand(labels.shape) < 0.8)
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = masked_indices & ~indices_replaced & (torch.rand(labels.shape) < 0.5)
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, labels


# ============================================================================
# Configuration
# ============================================================================

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SPROUT Brain-Like Architecture")

    # Brain-Like Model config
    parser.add_argument("--n_neurons", type=int, default=4096,
                        help="Number of neurons in global pool (default: 4096)")
    parser.add_argument("--d_state", type=int, default=256,
                        help="Neuron state dimension (default: 256)")
    parser.add_argument("--n_interaction_steps", type=int, default=5,
                        help="Number of neuron interaction steps (default: 5)")
    parser.add_argument("--initial_sparsity", type=int, default=128,
                        help="Initial number of active neurons (default: 128)")
    parser.add_argument("--final_sparsity", type=int, default=256,
                        help="Final maximum active neurons (default: 256)")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--encoder_layers", type=int, default=2,
                        help="Number of encoder layers (default: 2)")

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
    parser.add_argument("--no_mixed_precision", action="store_true",
                        help="Disable mixed precision training (default: enabled)")

    # Keep old arg for compatibility but deprecate
    parser.add_argument("--mixed_precision", action="store_true",
                        help=argparse.SUPPRESS)

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
    parser.add_argument("--analyze_activation", action="store_true",
                        help="Analyze neuron activation patterns during training")

    # Debug
    parser.add_argument("--debug_mode", action="store_true",
                        help="Use tiny dataset for debugging")

    args = parser.parse_args()

    # Set mixed_precision: default True unless --no_mixed_precision is set
    args.mixed_precision = not args.no_mixed_precision

    return args


# ============================================================================
# Training Functions
# ============================================================================

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
            streaming=True
        )

    print(f"   Total sentences: {len(sentences)}")

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
        num_workers=2,
        pin_memory=True
    )

    print(f"   Batches per epoch: {len(dataloader)}")
    print(f"{'='*70}\n")

    return dataloader


def compute_mlm_loss(logits, labels):
    """Compute MLM loss (cross entropy)."""
    # logits: [batch, vocab_size]
    # labels: [batch, seq_len]

    # 각 배치 샘플에 대해 손실 계산
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    # Brain-like 모델은 시퀀스 전체를 하나의 패턴으로 처리
    # 여러 마스크된 토큰 중 하나를 예측하도록 설계
    # 여기서는 첫 번째 마스크된 토큰을 예측

    batch_size = labels.shape[0]
    total_loss = 0
    n_valid = 0

    for i in range(batch_size):
        # 마스크된 위치 찾기
        masked_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]

        if len(masked_positions) > 0:
            # 첫 번째 마스크된 토큰 예측
            pos = masked_positions[0]
            target = labels[i, pos]

            loss = loss_fct(logits[i].unsqueeze(0), target.unsqueeze(0))
            total_loss += loss
            n_valid += 1

    if n_valid > 0:
        return total_loss / n_valid
    else:
        return torch.tensor(0.0, device=logits.device)


def train_epoch(model, dataloader, optimizer, scheduler, scaler, args, epoch, device):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_correct = 0
    total_tokens = 0

    # 활성화 패턴 추적
    activation_stats = {
        'initial_active': [],
        'final_active': [],
        'steps': args.n_interaction_steps
    }

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", ncols=100)

    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        if args.mixed_precision:
            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                loss = compute_mlm_loss(logits, labels)
        else:
            logits = model(input_ids)
            loss = compute_mlm_loss(logits, labels)

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

        # Calculate accuracy (첫 번째 마스크된 토큰에 대해)
        predictions = torch.argmax(logits, dim=-1)
        batch_size = labels.shape[0]
        batch_correct = 0
        batch_total = 0

        for i in range(batch_size):
            masked_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(masked_positions) > 0:
                pos = masked_positions[0]
                target = labels[i, pos]
                pred = predictions[i]

                if pred == target:
                    batch_correct += 1
                batch_total += 1

        total_loss += loss.item()
        total_correct += batch_correct
        total_tokens += batch_total

        # 활성화 분석 (가끔만)
        if args.analyze_activation and batch_idx % 100 == 0:
            with torch.no_grad():
                analysis = model.analyze_activation(input_ids[:1])
                activation_stats['initial_active'].append(analysis['initial_active'])
                activation_stats['final_active'].append(analysis['final_active'])

        # Update progress bar
        if batch_idx % 10 == 0:
            acc = 100.0 * total_correct / total_tokens if total_tokens > 0 else 0.0
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.1f}%'
            })

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100.0 * total_correct / total_tokens if total_tokens > 0 else 0.0

    # 활성화 통계 출력
    if args.analyze_activation and len(activation_stats['initial_active']) > 0:
        print(f"\n{'='*70}")
        print("NEURON ACTIVATION STATISTICS")
        print(f"{'='*70}")
        print(f"  Average initial active neurons: {np.mean(activation_stats['initial_active']):.1f}")
        print(f"  Average final active neurons: {np.mean(activation_stats['final_active']):.1f}")
        print(f"{'='*70}\n")

    return avg_loss, avg_acc


def main():
    """Main training function."""
    args = get_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("SPROUT BRAIN-LIKE TRAINING")
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
    print(f"\nCreating Brain-Like SPROUT model...")
    print(f"  - Global neurons: {args.n_neurons}")
    print(f"  - Neuron state dim: {args.d_state}")
    print(f"  - Interaction steps: {args.n_interaction_steps}")
    print(f"  - Initial sparsity: {args.initial_sparsity}")
    print(f"  - Final sparsity: {args.final_sparsity}")

    model = create_brain_like_sprout(
        vocab_size=len(tokenizer),
        n_neurons=args.n_neurons,
        d_state=args.d_state,
        n_heads=args.n_heads,
        encoder_layers=args.encoder_layers,
        n_interaction_steps=args.n_interaction_steps,
        initial_sparsity=args.initial_sparsity,
        final_sparsity=args.final_sparsity
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created")
    print(f"   Total params: {total_params:,}")

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
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*70}")

        epoch_loss, epoch_acc = train_epoch(
            model, dataloader, optimizer, scheduler, scaler, args, epoch, device
        )

        print(f"\nEpoch {epoch+1}/{args.num_epochs} Summary:")
        print(f"  Loss: {epoch_loss:.4f}")
        print(f"  Accuracy: {epoch_acc:.2f}%")

        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, "sprout_brain_like_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'accuracy': epoch_acc,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✅ Saved best checkpoint: {checkpoint_path}")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
