"""
DeepSets FFN Training Script

Baseline vs DeepSets 모델 학습

Usage:
    python scripts/train_deepsets.py --model baseline
    python scripts/train_deepsets.py --model deepsets-basic
    python scripts/train_deepsets.py --model deepsets-context
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

from models.neuron_based import NeuronBasedLanguageModel as BaselineModel
from models.neuron_based_deepsets import DeepSetsLanguageModel
from utils.training import CheckpointManager, TrainingMonitor, count_parameters, format_time
from utils.data import CacheLoader
import time


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

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)

        # Create MLM labels (same as input for simplicity)
        labels = input_ids.clone()

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size, seq_len = input_ids.shape
        num_tokens = batch_size * seq_len
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total_tokens


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']

            batch_size, seq_len = input_ids.shape
            num_tokens = batch_size * seq_len
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['baseline', 'deepsets-basic', 'deepsets-context'],
                        help='Model type to train')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--sparse_k', type=int, default=512, help='Number of neurons to select')
    parser.add_argument('--d_neuron', type=int, default=128, help='Neuron info vector size (DeepSets)')
    parser.add_argument('--d_hidden', type=int, default=256, help='φ output size (DeepSets)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir) / args.model / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / args.model / datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Load data
    print("\nLoading cached WikiText data...")
    train_loader, val_loader, tokenizer = load_cached_data(
        max_length=args.max_seq_len,
        batch_size=args.batch_size
    )

    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print(f"\nCreating {args.model} model...")

    if args.model == 'baseline':
        model = BaselineModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            d_ff=args.d_ff,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            sparse_k=args.sparse_k
        )
    else:
        use_context = (args.model == 'deepsets-context')
        model = DeepSetsLanguageModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            d_ff=args.d_ff,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            sparse_k=args.sparse_k,
            d_neuron=args.d_neuron,
            d_hidden=args.d_hidden,
            use_context=use_context
        )

    model = model.to(device)

    # Parameter count
    params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")

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
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # Checkpoint & Monitor
    ckpt_manager = CheckpointManager(str(checkpoint_dir), keep_best_n=3)
    monitor = TrainingMonitor(str(log_dir))

    # Training loop
    print(f"\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, scaler
        )

        # Evaluate
        val_loss = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        # Log
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        monitor.log_epoch(epoch, metrics)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        ckpt_manager.save_checkpoint(
            model, optimizer, epoch, val_loss, metrics, is_best=is_best
        )

    print(f"\n✓ Training completed!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")


if __name__ == '__main__':
    main()
