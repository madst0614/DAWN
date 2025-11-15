"""
Training Script for Plastic Neural Networks

Usage:
    python train.py --model pnn --dataset wikitext-103 --epochs 15
    python train.py --config configs/pnn_wikitext103.yaml
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import BertTokenizer
from datasets import load_dataset
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import time
import json

from pnn.models.pnn import create_pnn_model
from pnn.data.dataset import MLMDataset
from pnn.utils.training import (
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    load_checkpoint
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Plastic Neural Networks')
    
    # Model
    parser.add_argument('--model', type=str, default='pnn',
                       choices=['pnn', 'pnn_hierarchical', 'bert'],
                       help='Model type: pnn (baseline), pnn_hierarchical (mountain-shaped FFN with hierarchical refiner)')
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--intermediate_size', type=str, default='2048',
                       help='FFN intermediate size (int for pnn, comma-separated list for hierarchical e.g., "1024,1536,2048,1536,1024")')
    parser.add_argument('--num_steps', type=int, default=4, help='PNN refinement steps')
    parser.add_argument('--num_blocks', type=int, default=5, help='Number of blocks for hierarchical mode')
    parser.add_argument('--use_checkpoint', action='store_true', help='Use gradient checkpointing for hierarchical mode')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--dataset', type=str, default='wikitext-103')
    parser.add_argument('--batch_size', type=int, default=384)
    parser.add_argument('--gradient_accumulation', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Warmup steps (in optimizer steps, not batches)')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_samples', type=int, default=1000000)
    
    # System
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--use_tf32', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None)
    
    # Config file
    parser.add_argument('--config', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)
    
    return args


def prepare_data(args, tokenizer):
    """Load and prepare WikiText-103 dataset"""
    print(f"\n📚 Loading {args.dataset} dataset...")
    
    # Load dataset
    if args.dataset == 'wikitext-103':
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split="train",
            streaming=True
        )
        eval_dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split="validation"
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Filter and collect training data
    train_data = []
    for item in tqdm(dataset, desc="Loading train data"):
        text = item['text'].strip()
        if len(text) > 20 and len(text.split()) >= 5:
            train_data.append(text)
            if len(train_data) >= args.max_samples:
                break
    
    print(f"✅ Loaded {len(train_data):,} training samples")
    
    # Validation data
    eval_data = []
    for item in eval_dataset:
        text = item['text'].strip()
        if len(text) > 20 and len(text.split()) >= 5:
            eval_data.append(text)
    
    print(f"✅ Loaded {len(eval_data):,} validation samples")
    
    # Create datasets
    train_dataset = MLMDataset(
        tokenizer=tokenizer,
        data=train_data,
        max_length=args.max_length,
        mask_prob=0.15
    )
    
    eval_dataset = MLMDataset(
        tokenizer=tokenizer,
        data=eval_data,
        max_length=args.max_length,
        mask_prob=0.15
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"✅ {len(train_loader)} train batches, {len(eval_loader)} eval batches")
    
    return train_loader, eval_loader


def evaluate(model, dataloader, device, use_amp=True):
    """Evaluate model on validation set"""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    hidden = model(input_ids, attention_mask)
                    loss, logits = model.get_mlm_loss(hidden, labels)
            else:
                hidden = model(input_ids, attention_mask)
                loss, logits = model.get_mlm_loss(hidden, labels)
            
            total_loss += loss.item()

            # Calculate accuracy
            preds = logits.argmax(dim=-1).view(-1)  # [B*L]
            labels_flat = labels.view(-1)  # [B*L]
            mask = (labels_flat != -100)  # [B*L]
            correct = (preds == labels_flat) & mask  # [B*L]
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, accuracy


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    args
):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    step_losses = [0.0] * args.num_steps
    step_accs = [0.0] * args.num_steps

    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", mininterval=1.0)

    for batch_idx, batch in enumerate(progress):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        if args.use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                result = model.compute_recurrent_loss(
                    input_ids, attention_mask, labels,
                    return_accuracies=True
                )
                loss = result[0] / args.gradient_accumulation
                batch_step_losses = result[1]
                batch_step_accs = result[2]

            scaler.scale(loss).backward()
        else:
            result = model.compute_recurrent_loss(
                input_ids, attention_mask, labels,
                return_accuracies=True
            )
            loss = result[0] / args.gradient_accumulation
            batch_step_losses = result[1]
            batch_step_accs = result[2]
            loss.backward()

        # Accumulate losses and accuracies
        total_loss += loss.item() * args.gradient_accumulation
        for i, sl in enumerate(batch_step_losses):
            step_losses[i] += sl
        for i, sa in enumerate(batch_step_accs):
            step_accs[i] += sa

        # Optimizer step
        if (batch_idx + 1) % args.gradient_accumulation == 0:
            if args.use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

        # Update progress bar with all step info
        losses_str = ','.join([f'{l:.4f}' for l in batch_step_losses])
        accs_str = ','.join([f'{a:.4f}' for a in batch_step_accs])
        progress.set_postfix({
            'L': f'[{losses_str}]',
            'A': f'[{accs_str}]',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    # Average losses and accuracies
    avg_loss = total_loss / len(dataloader)
    avg_step_losses = [sl / len(dataloader) for sl in step_losses]
    avg_step_accs = [sa / len(dataloader) for sa in step_accs]

    return avg_loss, avg_step_losses, avg_step_accs


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")
    
    # Enable TF32 if requested
    if args.use_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TF32 enabled")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Prepare data
    train_loader, eval_loader = prepare_data(args, tokenizer)
    
    # Create model
    print(f"\n🤖 Creating {args.model} model...")

    # Parse intermediate_size (can be int or comma-separated list)
    if ',' in args.intermediate_size:
        intermediate_size = [int(x.strip()) for x in args.intermediate_size.split(',')]
    else:
        intermediate_size = int(args.intermediate_size)

    model_config = {
        'vocab_size': tokenizer.vocab_size,
        'hidden_size': args.hidden_size,
        'num_heads': args.num_heads,
        'intermediate_size': intermediate_size,
        'max_length': args.max_length,
        'num_steps': args.num_steps,
        'dropout': args.dropout
    }

    # Add hierarchical-specific config
    if args.model == 'pnn_hierarchical':
        model_config['num_blocks'] = args.num_blocks
        model_config['use_checkpoint'] = args.use_checkpoint

    model = create_pnn_model(model_config, model_type=args.model)
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args.lr, args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    scheduler = create_scheduler(optimizer, args.warmup_steps, total_steps)
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler('cuda') if args.use_amp else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    history = {'train_loss': [], 'eval_loss': [], 'eval_acc': []}
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        history = checkpoint.get('history', history)
        print(f"📂 Resumed from epoch {start_epoch}")
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"🔬 Training for {args.epochs} epochs")
    print(f"{'='*80}\n")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, step_losses, step_accs = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, args
        )

        # Evaluate
        print("   Evaluating...")
        eval_loss, eval_acc = evaluate(model, eval_loader, device, args.use_amp)

        epoch_time = time.time() - epoch_start

        # Log results
        print(f"\n   Epoch {epoch+1}/{args.epochs}:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Step Losses: {[f'{l:.4f}' for l in step_losses]}")
        print(f"   Step Accs:   {[f'{a:.4f}' for a in step_accs]}")
        print(f"   Eval Loss:  {eval_loss:.4f}")
        print(f"   Eval Acc:   {eval_acc:.4f} ({eval_acc*100:.2f}%)")
        print(f"   Time: {epoch_time:.1f}s ({epoch_time/60:.1f}m)\n")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['eval_loss'].append(eval_loss)
        history['eval_acc'].append(eval_acc)
        
        # Save checkpoint
        is_best = eval_acc > best_acc
        if is_best:
            best_acc = eval_acc
            save_checkpoint(
                checkpoint_dir / 'best_model.pt',
                model, optimizer, scheduler, scaler,
                epoch, best_acc, history
            )
            print(f"   💾 Saved best model (acc: {best_acc:.4f})")
        
        # Save regular checkpoint
        save_checkpoint(
            checkpoint_dir / f'checkpoint_epoch{epoch}.pt',
            model, optimizer, scheduler, scaler,
            epoch, best_acc, history
        )
    
    # Final results
    print(f"\n{'='*80}")
    print(f"📊 Training Complete!")
    print(f"{'='*80}\n")
    print(f"🏆 Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"📁 Checkpoints saved in: {checkpoint_dir}")
    
    # Save final metrics
    with open(checkpoint_dir / 'metrics.json', 'w') as f:
        json.dump({
            'best_accuracy': best_acc,
            'final_train_loss': history['train_loss'][-1],
            'final_eval_loss': history['eval_loss'][-1],
            'final_eval_acc': history['eval_acc'][-1],
            'history': history
        }, f, indent=2)


if __name__ == "__main__":
    main()
