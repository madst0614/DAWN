"""
DAWN (Dynamic Architecture With Neurons) Training Script

Usage:
    # 기본 학습 (자동으로 최신 체크포인트 이어서 학습)
    python scripts/train.py

    # 처음부터 새로 시작
    python scripts/train.py --from-scratch

    # 특정 체크포인트 폴더에서 이어서 학습
    python scripts/train.py --resume checkpoints/run_20240101_120000_1234

    # 커스텀 config 파일 사용
    python scripts/train.py --config configs/my_config.yaml

Checkpoint Options:
    (기본)           - 자동으로 최신 best_model.pt 탐색 후 이어서 학습
    --from-scratch   - 자동 탐색 비활성화, 처음부터 시작
    --resume <폴더>  - 지정한 폴더의 best_model.pt에서 이어서 학습
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress noisy torch inductor warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')
warnings.filterwarnings('ignore', message='.*online softmax.*')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import time
import numpy as np
import math

# Enable TensorFloat32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

from models.model import DAWN, DAWNLanguageModel
from utils.training import CheckpointManager, TrainingMonitor, count_parameters, format_time
from utils.data import MLM_CONFIG, apply_mlm_masking, TextDataset, collate_fn_dynamic_padding, load_data, compute_mlm_accuracy


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args, scaler=None, tokenizer=None, log_file=None):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_valid_tokens = 0
    num_batches = 0

    # Window accumulators for logging every 100 steps
    log_interval = 100
    window_loss = 0.0
    window_acc_correct = 0
    window_acc_valid = 0
    window_count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)

        # Apply MLM masking
        if tokenizer is not None:
            input_ids, labels = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)
        else:
            labels = input_ids.clone()

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(input_ids)  # [B, S, vocab_size]

                # Loss 계산
                B, S, V = logits.shape
                loss = F.cross_entropy(
                    logits.view(B * S, V),
                    labels.view(B * S),
                    ignore_index=-100
                )

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids)

            B, S, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * S, V),
                labels.view(B * S),
                ignore_index=-100
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accuracy calculation (only valid tokens)
        predictions = logits.argmax(dim=-1)
        valid_mask = (labels != -100)
        correct_predictions = (predictions == labels) & valid_mask

        correct = correct_predictions.sum().item()
        valid_tokens = valid_mask.sum().item()

        total_correct += correct
        total_valid_tokens += valid_tokens

        # Track total loss
        batch_size, seq_len = input_ids.shape
        num_tokens = batch_size * seq_len
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        num_batches += 1
        step_acc = correct / valid_tokens if valid_tokens > 0 else 0.0
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{step_acc:.4f}"
        })

        # Accumulate for window logging
        window_loss += loss.item()
        window_acc_correct += correct
        window_acc_valid += valid_tokens
        window_count += 1

        # Log aggregated metrics every 100 steps
        if log_file and (step + 1) % log_interval == 0:
            avg_window_loss = window_loss / window_count
            avg_window_acc = window_acc_correct / window_acc_valid if window_acc_valid > 0 else 0.0

            with open(log_file, 'a') as f:
                f.write(f"epoch={epoch},step={step+1},loss={avg_window_loss:.6f},"
                       f"acc={avg_window_acc:.6f}\n")

            # Reset window
            window_loss = 0.0
            window_acc_correct = 0
            window_acc_valid = 0
            window_count = 0

    # Log remaining steps at end of epoch
    if log_file and window_count > 0:
        avg_window_loss = window_loss / window_count
        avg_window_acc = window_acc_correct / window_acc_valid if window_acc_valid > 0 else 0.0

        with open(log_file, 'a') as f:
            f.write(f"epoch={epoch},step={num_batches},loss={avg_window_loss:.6f},"
                   f"acc={avg_window_acc:.6f}\n")

    avg_loss = total_loss / total_tokens
    avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0

    return avg_loss, avg_acc


def evaluate(model, dataloader, device, args, tokenizer=None):
    """Evaluate model with MLM masking"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_valid_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)

            # Apply same MLM masking as training
            if tokenizer is not None:
                masked_input_ids, labels = apply_mlm_masking(input_ids, tokenizer)
            else:
                masked_input_ids = input_ids
                labels = input_ids.clone()

            logits = model(masked_input_ids)

            B, S, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * S, V),
                labels.view(B * S),
                ignore_index=-100
            )

            # Accuracy calculation
            predictions = logits.argmax(dim=-1)
            valid_mask = (labels != -100)
            correct_predictions = (predictions == labels) & valid_mask

            correct = correct_predictions.sum().item()
            valid_tokens = valid_mask.sum().item()

            total_correct += correct
            total_valid_tokens += valid_tokens

            batch_size, seq_len = input_ids.shape
            num_tokens = batch_size * seq_len
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
    return avg_loss, avg_acc


def analyze_activations(model, input_ids, device):
    """Dynamic Neuron Transformer 활성화 패턴 분석"""
    model.eval()

    with torch.no_grad():
        _, all_selected, all_patterns = model(input_ids, return_activations=True)

    stats = {}
    for layer_idx, selected_idx in enumerate(all_selected):
        # selected_idx: [B, S, k]
        unique_neurons = torch.unique(selected_idx).numel()

        # Get total neurons from model (updated for NeuronRouter)
        if hasattr(model, '_orig_mod'):
            # Compiled model
            total_neurons = model._orig_mod.layers[layer_idx].router.n_neurons
        else:
            total_neurons = model.layers[layer_idx].router.n_neurons

        usage_ratio = unique_neurons / total_neurons

        stats[f'layer_{layer_idx}'] = {
            'unique_neurons': unique_neurons,
            'total_neurons': total_neurons,
            'usage_ratio': usage_ratio,
            'k': selected_idx.shape[-1],
        }

    return stats


def load_config(config_path):
    """Load config from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train DAWN (Dynamic Architecture With Neurons)')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint folder to resume from (e.g., checkpoints/run_20240101_120000_1234)')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Start training from scratch (disable auto-resume)')
    cli_args = parser.parse_args()

    # Load config
    config_path = Path(PROJECT_ROOT) / cli_args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)

    # Create args namespace from config
    class Args:
        pass
    args = Args()

    # Model (Dynamic Neuron Transformer)
    args.d_model = cfg['model'].get('d_model', 512)
    args.n_layers = cfg['model'].get('n_layers', 6)
    args.n_heads = cfg['model'].get('n_heads', 8)
    args.n_neurons = cfg['model'].get('n_neurons', 1024)
    args.n_patterns = cfg['model'].get('n_patterns', 512)
    args.k = cfg['model'].get('k', 8)
    args.pattern_k = cfg['model'].get('pattern_k', 16)
    args.d_ff = cfg['model'].get('d_ff', None)  # Auto-calculate if None
    args.max_seq_len = cfg['model'].get('max_seq_len', 2048)
    args.dropout = cfg['model'].get('dropout', 0.1)

    # Backward compatibility (deprecated)
    args.n_input = cfg['model'].get('n_input', None)
    args.n_process = cfg['model'].get('n_process', None)

    # Training
    args.batch_size = cfg['training']['batch_size']
    args.num_epochs = cfg['training']['num_epochs']
    args.lr = cfg['training']['lr']
    args.weight_decay = cfg['training']['weight_decay']
    args.warmup_epochs = cfg['training'].get('warmup_epochs', 1)

    # Other
    args.use_amp = cfg.get('use_amp', True)
    args.checkpoint_dir = cfg.get('checkpoint_dir', 'checkpoints')
    args.log_dir = cfg.get('log_dir', 'logs')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    base_checkpoint_dir = Path(args.checkpoint_dir)
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint loading logic
    latest_best_checkpoint = None
    checkpoint_dir = None

    if cli_args.resume:
        # Explicit resume from folder - use existing folder
        resume_folder = Path(cli_args.resume)
        if not resume_folder.is_absolute():
            resume_folder = Path(args.checkpoint_dir) / resume_folder.name

        best_ckpt = resume_folder / 'best_model.pt'
        if best_ckpt.exists():
            latest_best_checkpoint = best_ckpt
            checkpoint_dir = resume_folder  # Use existing folder
            print(f"\n✓ Resuming from: {latest_best_checkpoint}")
            print(f"✓ Continuing in same folder: {checkpoint_dir}")
        else:
            print(f"\n⚠️  Warning: Checkpoint not found at {best_ckpt}")
            print(f"    Starting from scratch instead.")

    elif not cli_args.from_scratch:
        # Auto-resume: find latest checkpoint and use its folder
        run_folders = sorted([
            d for d in base_checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith('run_')
        ], reverse=True)

        if run_folders:
            latest_folder = run_folders[0]
            best_ckpt = latest_folder / 'best_model.pt'
            if best_ckpt.exists():
                latest_best_checkpoint = best_ckpt
                checkpoint_dir = latest_folder  # Use existing folder
                print(f"\n✓ Auto-resume: Found latest checkpoint: {latest_best_checkpoint}")
                print(f"✓ Continuing in same folder: {checkpoint_dir}")

    if cli_args.from_scratch:
        print(f"\n✓ Starting from scratch (--from-scratch)")

    # Create new run folder only if not resuming
    if checkpoint_dir is None:
        import random
        from datetime import timezone, timedelta
        kst = timezone(timedelta(hours=9))
        timestamp = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
        random_suffix = random.randint(1000, 9999)
        run_name = f"run_{timestamp}_{random_suffix}"
        checkpoint_dir = base_checkpoint_dir / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Created new run folder: {checkpoint_dir}")

        # Save config for new runs
        with open(checkpoint_dir / 'config.json', 'w') as f:
            json.dump(cfg, f, indent=2)

    log_dir = checkpoint_dir
    print(f"Run folder: {checkpoint_dir}")

    print(f"\n{'='*60}")
    print(f"DAWN (Dynamic Neuron Transformer) Training")
    print(f"{'='*60}")
    print(f"\nModel: d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")
    print(f"Neurons: pool_size={args.n_neurons}, patterns={args.n_patterns}, top_k={args.k}")
    print(f"Training: batch={args.batch_size}, epochs={args.num_epochs}, lr={args.lr}")

    # Load data
    print(f"\n{'='*60}")
    print("Loading data...")
    print(f"{'='*60}")
    train_loader, val_loader, tokenizer = load_data(
        data_config=cfg['data'],
        max_length=args.max_seq_len,
        batch_size=args.batch_size
    )

    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print(f"\n{'='*60}")
    print("Creating DAWN model...")
    print(f"{'='*60}")

    model = DAWN(
        vocab_size=vocab_size,
        hidden_dim=args.d_model,
        num_layers=args.n_layers,
        n_heads=args.n_heads,
        n_neurons=args.n_neurons,
        n_patterns=args.n_patterns,
        k=args.k,
        pattern_k=args.pattern_k,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    )
    model = model.to(device)

    # Display model version
    print(f"\n📌 Model version: {DAWN.__version__}")

    # PyTorch 2.0+ compilation for speed boost
    if hasattr(torch, 'compile'):
        print(f"\nCompiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
        print(f"  Model compiled successfully!")

    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Number of layers: {args.n_layers}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )

    # Warmup + Cosine scheduler
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.num_epochs * len(train_loader)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=args.lr * 0.1
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
    if args.use_amp:
        print(f"\nUsing Automatic Mixed Precision (AMP)")

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float('inf')
    resume_checkpoint = None

    # Load checkpoint if resuming
    if latest_best_checkpoint:
        resume_checkpoint = latest_best_checkpoint

    if resume_checkpoint and resume_checkpoint.exists():
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)

        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )

        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
        if not missing_keys and not unexpected_keys:
            print("✓ All parameters loaded successfully!")

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"  Starting from epoch {start_epoch}")
        print(f"  Best val loss so far: {best_val_loss:.4f}")
    else:
        print(f"\nStarting fresh training")

    # Checkpoint & Monitor
    ckpt_manager = CheckpointManager(str(checkpoint_dir), keep_best_n=3)
    monitor = TrainingMonitor(str(log_dir))

    # Training log file (append mode if resuming)
    training_log_file = checkpoint_dir / "training_log.txt"

    # Open in append mode if resuming, write mode if new
    log_mode = 'a' if latest_best_checkpoint else 'w'
    if log_mode == 'w':
        with open(training_log_file, 'w') as f:
            f.write("# Training Log\n")
            f.write("# Step logs: epoch,step,loss,acc\n")
            f.write("# Epoch summaries: EPOCH,epoch,train_loss,train_acc,val_loss,val_acc,lr,time\n")
            f.write("\n")
    else:
        # Append separator for resumed training
        with open(training_log_file, 'a') as f:
            f.write(f"\n# === Resumed training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"  Training log: {training_log_file}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args,
            scaler, tokenizer, log_file=str(training_log_file)
        )

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device, args, tokenizer)

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

        # Write epoch summary to log
        with open(training_log_file, 'a') as f:
            f.write(f"EPOCH,{epoch},{train_loss:.6f},{train_acc:.6f},"
                   f"{val_loss:.6f},{val_acc:.6f},"
                   f"{optimizer.param_groups[0]['lr']:.6e},{epoch_time:.2f}\n")

        # Analyze activations periodically
        if epoch % 10 == 0:
            sample_batch = next(iter(val_loader))
            sample_ids = sample_batch['input_ids'][:1].to(device)
            act_stats = analyze_activations(model, sample_ids, device)
            print(f"\n  Neuron Usage Analysis (Epoch {epoch}):")
            for layer_name, stats in act_stats.items():
                print(f"    {layer_name}: {stats['unique_neurons']}/{stats['total_neurons']} neurons "
                      f"({stats['usage_ratio']:.2%} usage)")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  New best model! (val_loss: {best_val_loss:.4f})")

        ckpt_manager.save_checkpoint(
            model, optimizer, epoch, val_loss, metrics, is_best=is_best,
            scheduler=scheduler, scaler=scaler
        )

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
