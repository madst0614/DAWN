"""
Training script for neuron-based SPROUT model
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
from datetime import datetime


from src.models.sprout_neuron_based import NeuronBasedLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train neuron-based SPROUT model")

    # Model architecture
    parser.add_argument("--vocab_size", type=int, default=30522)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=128)

    # Sparsity settings
    parser.add_argument("--initial_top_k", type=int, default=None,
                       help="Initial sparsity (None = dense)")
    parser.add_argument("--final_top_k", type=int, default=512,
                       help="Final sparsity target")
    parser.add_argument("--sparsity_warmup_steps", type=int, default=10000,
                       help="Steps to gradually increase sparsity")

    # Training settings
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--router_lr_multiplier", type=float, default=5.0,
                       help="Router learning rate = base_lr * multiplier")
    parser.add_argument("--router_loss_weight", type=float, default=0.1,
                       help="Weight for router quality loss (0 = disable)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--no_mixed_precision", action="store_true",
                       help="Disable mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing to reduce memory (trades compute for memory)")
    parser.add_argument("--enable_memory_monitor", action="store_true",
                       help="Enable detailed memory monitoring and logging")

    # Data settings
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                       help="Dataset configuration")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Limit training samples (for testing)")

    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Auto-detect checkpoint directory (Colab support)
    if args.checkpoint_dir is None:
        if os.path.exists("/content/drive/MyDrive"):
            args.checkpoint_dir = "/content/drive/MyDrive/sprout_neuron_checkpoints"
        else:
            args.checkpoint_dir = "./checkpoints/neuron_based"

    return args


def log_memory_stats(step: int = 0, prefix: str = "", suggest_optimization: bool = False):
    """
    Log CUDA memory statistics

    Args:
        step: Current training step
        prefix: Prefix for log message
        suggest_optimization: Whether to suggest optimization based on usage
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

        usage_pct = (allocated / total_memory) * 100

        print(f"[{prefix}] Step {step}: "
              f"Allocated: {allocated:.2f}GB / {total_memory:.1f}GB ({usage_pct:.1f}%) | "
              f"Reserved: {reserved:.2f}GB | "
              f"Max: {max_allocated:.2f}GB")

        # Suggestions based on memory usage
        if suggest_optimization:
            if usage_pct > 90:
                print(f"⚠️  WARNING: Very high memory usage! Consider:")
                print(f"   - Enabling --gradient_checkpointing")
                print(f"   - Reducing --batch_size")
            elif usage_pct > 75:
                print(f"⚠️  High memory usage! Monitor for OOM errors.")
            elif usage_pct < 25:
                print(f"💡 Low memory usage ({usage_pct:.1f}%). You can:")
                print(f"   - Increase --batch_size (current usage allows ~{int(allocated * 4)}GB)")
                print(f"   - Disable --gradient_checkpointing for faster training")
                print(f"   - Use larger --d_ff or --d_model")

        return allocated, reserved, max_allocated
    return 0, 0, 0


class MLMDataset:
    """Masked Language Modeling dataset"""
    def __init__(self, texts, tokenizer, max_length=128, mlm_probability=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)  # [seq_len]

        # Create MLM targets
        labels = input_ids.clone()

        # Mask tokens
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # Replace masked tokens
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% random tokens
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 10% keep original

        return {
            'input_ids': input_ids,
            'labels': labels
        }


def prepare_datasets(args):
    """Load and prepare datasets"""
    import pickle

    # Check for cached datasets first
    cache_paths = [
        "/content/drive/MyDrive/dawn_v4/cache/train/wikitext_texts.pkl",
        "/content/drive/MyDrive/dawn_v4/cache/validation/wikitext_texts.pkl"
    ]

    train_texts = None
    valid_texts = None

    # Try to load from cache
    if all(os.path.exists(p) for p in cache_paths):
        print("Found cached dataset! Loading from cache...")
        try:
            with open(cache_paths[0], 'rb') as f:
                all_train_texts = pickle.load(f)
            with open(cache_paths[1], 'rb') as f:
                original_valid_texts = pickle.load(f)

            # Combine and resplit: 100K train, rest for validation
            all_texts = all_train_texts + original_valid_texts
            train_texts = all_texts[:100000]
            valid_texts = all_texts[100000:]

            print(f"✅ Loaded from cache and resplit:")
            print(f"   Original: {len(all_train_texts)} train, {len(original_valid_texts)} valid")
            print(f"   Resplit:  {len(train_texts)} train, {len(valid_texts)} valid")
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}")
            print("Falling back to downloading dataset...")
            train_texts = None
            valid_texts = None

    # If cache loading failed or not found, download dataset
    if train_texts is None or valid_texts is None:
        print(f"Loading dataset: {args.dataset} ({args.dataset_config})")

        # Load dataset
        dataset = load_dataset(args.dataset, args.dataset_config)

        # Extract texts
        def extract_text(examples):
            return {'text': [t for t in examples['text'] if len(t.strip()) > 0]}

        dataset = dataset.map(extract_text, batched=True, remove_columns=dataset['train'].column_names)

        train_texts = dataset['train']['text']
        valid_texts = dataset['validation']['text']

    # Limit samples if specified
    if args.max_samples is not None:
        train_texts = train_texts[:args.max_samples]
        valid_texts = valid_texts[:min(args.max_samples // 10, len(valid_texts))]

    print(f"Train samples: {len(train_texts)}")
    print(f"Valid samples: {len(valid_texts)}")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create datasets
    train_dataset = MLMDataset(train_texts, tokenizer, max_length=args.max_seq_len)
    valid_dataset = MLMDataset(valid_texts, tokenizer, max_length=args.max_seq_len)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, valid_loader, tokenizer


def get_current_sparsity(step, args):
    """Calculate current top_k based on warmup schedule"""
    if args.initial_top_k is None:
        # Start dense, gradually introduce sparsity
        if step < args.sparsity_warmup_steps:
            # Gradually reduce from d_ff to final_top_k
            progress = step / args.sparsity_warmup_steps
            current_top_k = int(args.d_ff - progress * (args.d_ff - args.final_top_k))
        else:
            current_top_k = args.final_top_k
    else:
        # Gradually increase sparsity (reduce top_k)
        if step < args.sparsity_warmup_steps:
            progress = step / args.sparsity_warmup_steps
            current_top_k = int(args.initial_top_k - progress * (args.initial_top_k - args.final_top_k))
        else:
            current_top_k = args.final_top_k

    return current_top_k if current_top_k < args.d_ff else None  # None = dense


def train_epoch(model, train_loader, valid_loader, optimizer, scheduler, scaler, epoch, args, global_step):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_masked = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        ncols=120,  # 출력 폭 고정
        leave=True,  # epoch 끝나도 progress bar 유지
        position=0,  # 첫 번째 위치
        dynamic_ncols=False  # 폭 자동 조정 비활성화
    )

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(args.device)
        labels = batch['labels'].to(args.device)

        # Get current sparsity
        current_top_k = get_current_sparsity(global_step, args)

        # Forward pass
        optimizer.zero_grad()

        if args.no_mixed_precision:
            # Standard training
            outputs = model(input_ids, labels=labels, top_k=current_top_k)
            mlm_loss = outputs['loss']
            logits = outputs['logits']

            # Router quality loss (if enabled and using sparsity)
            # 간소화: 주기적으로만 계산 (10 step마다)
            router_loss = 0.0
            if args.router_loss_weight > 0 and current_top_k is not None and current_top_k < args.d_ff and global_step % 10 == 0:
                # 첫 레이어만 사용 (메모리 절약)
                with torch.no_grad():
                    token_emb = model.token_embedding(input_ids)
                    positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
                    pos_emb = model.position_embedding(positions)
                    x = token_emb + pos_emb

                # 첫 레이어만 계산
                layer = model.layers[0]
                x_norm = layer.norm1(x)
                with torch.no_grad():
                    attn_out, _ = layer.attention(x_norm, x_norm, x_norm)
                    x = x + layer.dropout(attn_out)

                x_norm = layer.norm2(x)
                router_loss = layer.ffn.compute_router_loss(x_norm, current_top_k)

            # Total loss
            loss = mlm_loss + args.router_loss_weight * router_loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

        else:
            # Mixed precision training
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, labels=labels, top_k=current_top_k)
                mlm_loss = outputs['loss']
                logits = outputs['logits']

                # Router quality loss (if enabled and using sparsity)
                # 간소화: 주기적으로만 계산 (10 step마다)
                router_loss = 0.0
                if args.router_loss_weight > 0 and current_top_k is not None and current_top_k < args.d_ff and global_step % 10 == 0:
                    # 첫 레이어만 사용 (메모리 절약)
                    with torch.no_grad():
                        token_emb = model.token_embedding(input_ids)
                        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
                        pos_emb = model.position_embedding(positions)
                        x = token_emb + pos_emb

                    # 첫 레이어만 계산
                    layer = model.layers[0]
                    x_norm = layer.norm1(x)
                    with torch.no_grad():
                        attn_out, _ = layer.attention(x_norm, x_norm, x_norm)
                        x = x + layer.dropout(attn_out)

                    x_norm = layer.norm2(x)
                    router_loss = layer.ffn.compute_router_loss(x_norm, current_top_k)

                # Total loss
                loss = mlm_loss + args.router_loss_weight * router_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Metrics
        with torch.no_grad():
            masked_positions = (labels != -100)
            if masked_positions.sum() > 0:
                predictions = logits.argmax(dim=-1)
                correct = (predictions[masked_positions] == labels[masked_positions]).sum().item()
                total_correct += correct
                total_masked += masked_positions.sum().item()

        total_loss += loss.item()
        global_step += 1

        # Logging
        if global_step % args.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100.0 * total_correct / total_masked if total_masked > 0 else 0.0
            sparsity_pct = (current_top_k / args.d_ff * 100) if current_top_k else 100.0

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%',
                'top_k': current_top_k if current_top_k else 'dense',
                'sparse%': f'{sparsity_pct:.1f}%'
            })

            # Memory monitoring
            if args.enable_memory_monitor:
                # 첫 로그에서만 최적화 제안 표시
                suggest = (global_step == args.log_interval)
                log_memory_stats(global_step, "Training", suggest_optimization=suggest)

        # Save checkpoint
        if global_step % args.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, args)

    return global_step


@torch.no_grad()
def evaluate(model, valid_loader, args, top_k):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_masked = 0

    # 학습 progress bar와 동일한 방식으로
    pbar = tqdm(
        valid_loader,
        desc="Eval",
        ncols=120,
        leave=False,  # eval 끝나면 progress bar 제거
        position=0,
        dynamic_ncols=False
    )

    for batch in pbar:
        input_ids = batch['input_ids'].to(args.device)
        labels = batch['labels'].to(args.device)

        if args.no_mixed_precision:
            outputs = model(input_ids, labels=labels, top_k=top_k)
        else:
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, labels=labels, top_k=top_k)

        loss = outputs['loss']
        logits = outputs['logits']

        total_loss += loss.item()

        # Accuracy
        masked_positions = (labels != -100)
        if masked_positions.sum() > 0:
            predictions = logits.argmax(dim=-1)
            correct = (predictions[masked_positions] == labels[masked_positions]).sum().item()
            total_correct += correct
            total_masked += masked_positions.sum().item()
    avg_loss = total_loss / len(valid_loader)
    accuracy = 100.0 * total_correct / total_masked if total_masked > 0 else 0.0

    return avg_loss, accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, args):
    """Save training checkpoint"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"checkpoint_step_{global_step}.pt"
    )

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'args': vars(args)
    }, checkpoint_path)

    print(f"\n💾 Checkpoint saved: {checkpoint_path}")

    # Save latest
    latest_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'args': vars(args)
    }, latest_path)


def main():
    args = parse_args()

    print("="*70)
    print("NEURON-BASED SPROUT TRAINING")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Mixed precision: {not args.no_mixed_precision}")
    print(f"Model: d_model={args.d_model}, d_ff={args.d_ff}, layers={args.n_layers}")
    print(f"Sparsity: {args.initial_top_k} → {args.final_top_k} over {args.sparsity_warmup_steps} steps")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("="*70)

    # Prepare data
    train_loader, valid_loader, tokenizer = prepare_datasets(args)

    # Create model
    model = NeuronBasedLanguageModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        gradient_checkpointing=args.gradient_checkpointing
    ).to(args.device)

    # Memory optimizations
    if args.gradient_checkpointing:
        print("✅ Gradient checkpointing enabled (trades ~20% speed for 30-50% less memory)")
    if args.d_ff >= 8192:
        print("✅ Large d_ff detected - automatic chunked computation will be used")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer & scheduler
    # 라우터 파라미터는 더 높은 학습률로 학습
    router_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'router' in name:
            router_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.learning_rate},
        {'params': router_params, 'lr': args.learning_rate * args.router_lr_multiplier}
    ], weight_decay=args.weight_decay)

    print(f"\nOptimizer groups:")
    print(f"  - Base parameters: lr={args.learning_rate}")
    print(f"  - Router parameters: lr={args.learning_rate * args.router_lr_multiplier} ({args.router_lr_multiplier}x)")
    print(f"  - Router loss weight: {args.router_loss_weight}")

    total_steps = len(train_loader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=args.warmup_steps / total_steps,
        anneal_strategy='cos'
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if not args.no_mixed_precision else None

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0

    if args.resume_from is not None:
        print(f"\nResuming from: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # Training loop
    print("\nStarting training...")

    # Initial memory stats with optimization suggestions
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU Total Memory: {total_memory:.2f}GB")
        print("Checking initial memory usage...")
        # 짧은 forward pass로 실제 메모리 사용량 측정
        dummy_batch = next(iter(train_loader))
        dummy_ids = dummy_batch['input_ids'][:1].to(args.device)  # 1 sample만
        with torch.no_grad():
            _ = model(dummy_ids)
        torch.cuda.synchronize()

        # 첫 실행 후 메모리 사용량 확인 및 제안
        log_memory_stats(0, "After first forward", suggest_optimization=True)
        print()  # 빈 줄 추가

    for epoch in range(start_epoch, args.num_epochs):
        global_step = train_epoch(
            model, train_loader, valid_loader, optimizer, scheduler, scaler,
            epoch, args, global_step
        )

        # Epoch evaluation
        current_top_k = get_current_sparsity(global_step, args)
        eval_loss, eval_acc = evaluate(model, valid_loader, args, current_top_k)
        print(f"\nEpoch {epoch} complete - Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.2f}%")

        # Save epoch checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, args)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
