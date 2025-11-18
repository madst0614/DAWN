"""
Three-Stage Dynamic Neuron FFN Training Script

Three-Stage FFN 모델 학습

Usage:
    # 기본 학습
    python scripts/train_three_stage.py

    # 커스텀 설정
    python scripts/train_three_stage.py \
        --d_model 768 \
        --n_input_neurons 4096 \
        --n_process_neurons 2048 \
        --n_output_neurons 4096 \
        --d_process_value 512 \
        --batch_size 16 \
        --num_epochs 30 \
        --lr 3e-4

    # Mixed precision training
    python scripts/train_three_stage.py --use_amp

    # Gradient checkpointing (메모리 절약)
    python scripts/train_three_stage.py --gradient_checkpointing
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import time

from models.three_stage_ffn import ThreeStageLanguageModel
from utils.training import CheckpointManager, TrainingMonitor, count_parameters, format_time
from utils.data import CacheLoader


# ============================================================
# Dataset
# ============================================================

class TextDataset(Dataset):
    """Dataset for tokenized texts"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def load_cached_data(tokenizer_path=None, max_length=512, batch_size=32):
    """Load cached WikiText data"""
    from transformers import AutoTokenizer

    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load cached texts
    print("Loading cached WikiText data...")
    train_texts = CacheLoader.load_train_texts(dataset="wikitext")
    val_texts = CacheLoader.load_validation_texts(dataset="wikitext")

    if train_texts is None or val_texts is None:
        raise ValueError(
            "Cached data not found! "
            f"Expected at: {CacheLoader.CACHE_BASE_DIR}/{{train,validation}}/wikitext_5to1_texts.pkl"
        )

    print(f"Loaded {len(train_texts)} train texts, {len(val_texts)} val texts")

    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2
    )

    return train_loader, val_loader, tokenizer


# ============================================================
# Training Functions
# ============================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)

        # Create MLM labels (same as input for simplicity)
        labels = input_ids.clone()

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    k_input=args.k_input,
                    k_process=args.k_process,
                    k_output=args.k_output
                )
                loss = outputs['loss']
                logits = outputs['logits']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                k_input=args.k_input,
                k_process=args.k_process,
                k_output=args.k_output
            )
            loss = outputs['loss']
            logits = outputs['logits']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).sum().item()
        total_correct += correct

        batch_size, seq_len = input_ids.shape
        num_tokens = batch_size * seq_len
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct / num_tokens:.4f}"
        })

    avg_loss = total_loss / total_tokens
    avg_acc = total_correct / total_tokens
    return avg_loss, avg_acc


def evaluate(model, dataloader, device, args):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                k_input=args.k_input,
                k_process=args.k_process,
                k_output=args.k_output
            )
            loss = outputs['loss']
            logits = outputs['logits']

            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            correct = (predictions == labels).sum().item()
            total_correct += correct

            batch_size, seq_len = input_ids.shape
            num_tokens = batch_size * seq_len
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    avg_acc = total_correct / total_tokens
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description='Train Three-Stage Dynamic Neuron FFN')

    # Model architecture
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='Maximum sequence length')

    # Three-Stage FFN specific
    parser.add_argument('--n_input_neurons', type=int, default=2048,
                        help='Number of input neurons')
    parser.add_argument('--n_process_neurons', type=int, default=1024,
                        help='Number of process neurons')
    parser.add_argument('--n_output_neurons', type=int, default=2048,
                        help='Number of output neurons')
    parser.add_argument('--d_process_value', type=int, default=256,
                        help='Process value dimension')

    # Sparsity control (runtime)
    parser.add_argument('--k_input', type=int, default=None,
                        help='Number of input neurons to activate (None = n_input//8)')
    parser.add_argument('--k_process', type=int, default=None,
                        help='Number of process neurons to activate (None = n_process//8)')
    parser.add_argument('--k_output', type=int, default=None,
                        help='Number of output neurons to activate (None = n_output//8)')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Optimization
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Use gradient checkpointing to save memory')

    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Log directory')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(args.checkpoint_dir) / "three_stage" / timestamp
    log_dir = Path(args.log_dir) / "three_stage" / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Three-Stage Dynamic Neuron FFN Training")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Load data
    print(f"\n{'='*60}")
    print("Loading cached WikiText data...")
    print(f"{'='*60}")
    train_loader, val_loader, tokenizer = load_cached_data(
        max_length=args.max_seq_len,
        batch_size=args.batch_size
    )

    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print(f"\n{'='*60}")
    print("Creating Three-Stage FFN model...")
    print(f"{'='*60}")

    model = ThreeStageLanguageModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        n_input_neurons=args.n_input_neurons,
        n_process_neurons=args.n_process_neurons,
        n_output_neurons=args.n_output_neurons,
        d_process_value=args.d_process_value,
        dropout=args.dropout,
        gradient_checkpointing=args.gradient_checkpointing
    )

    model = model.to(device)

    # Model statistics
    stats = model.get_model_stats()
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"  FFN parameters: {stats['ffn_parameters']:,} ({stats['ffn_percentage']:.1f}%)")
    print(f"  Number of layers: {stats['n_layers']}")

    # Sparsity info
    if args.k_input is None:
        k_input_actual = max(args.n_input_neurons // 8, 64)
    else:
        k_input_actual = args.k_input

    if args.k_process is None:
        k_process_actual = max(args.n_process_neurons // 8, 32)
    else:
        k_process_actual = args.k_process

    if args.k_output is None:
        k_output_actual = max(args.n_output_neurons // 8, 64)
    else:
        k_output_actual = args.k_output

    print(f"\nSparsity Configuration:")
    print(f"  Input neurons: {k_input_actual}/{args.n_input_neurons} ({k_input_actual/args.n_input_neurons*100:.1f}%)")
    print(f"  Process neurons: {k_process_actual}/{args.n_process_neurons} ({k_process_actual/args.n_process_neurons*100:.1f}%)")
    print(f"  Output neurons: {k_output_actual}/{args.n_output_neurons} ({k_output_actual/args.n_output_neurons*100:.1f}%)")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * len(train_loader),
        eta_min=args.lr * 0.1
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
    if args.use_amp:
        print(f"\nUsing Automatic Mixed Precision (AMP)")

    # Checkpoint & Monitor
    ckpt_manager = CheckpointManager(str(checkpoint_dir), keep_best_n=3)
    monitor = TrainingMonitor(str(log_dir))

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}")
    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args, scaler
        )

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device, args)

        epoch_time = time.time() - epoch_start

        # Log
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        monitor.log_epoch(epoch, metrics)

        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {format_time(epoch_time)}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  New best model! (val_loss: {best_val_loss:.4f})")

        ckpt_manager.save_checkpoint(
            model, optimizer, epoch, val_loss, metrics, is_best=is_best
        )

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")


if __name__ == '__main__':
    main()
