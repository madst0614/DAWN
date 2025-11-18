"""
Hierarchical Dynamic Neuron FFN Training Script

계층적 동적 뉴런 FFN 모델 학습

Usage:
    # 기본 학습
    python scripts/train_three_stage.py

    # 커스텀 설정
    python scripts/train_three_stage.py \
        --d_model 768 \
        --n_input_neurons 4096 \
        --n_process_neurons 2048 \
        --d_routing 512 \
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import time

from models.three_stage_ffn import HierarchicalLanguageModel
from utils.training import CheckpointManager, TrainingMonitor, count_parameters, format_time
from utils.data import CacheLoader

# MLM 마스킹 설정
MLM_CONFIG = {
    "mask_prob": 0.15,
    "mask_token_ratio": 0.8,
    "random_token_ratio": 0.5
}


# ============================================================
# MLM Masking Function
# ============================================================

def apply_mlm_masking(input_ids, tokenizer, config=None):
    """
    Apply MLM-style masking (80% [MASK], 10% random, 10% keep).
    Based on dawn/utils/data_utils.py MaskingStrategy.apply_mlm_masking

    Args:
        input_ids: [B, S] Token IDs to mask
        tokenizer: Tokenizer instance
        config: Optional config dict with mask_prob, mask_token_ratio, random_token_ratio

    Returns:
        Tuple of (masked_input_ids, labels)
    """
    if config is None:
        config = MLM_CONFIG

    labels = input_ids.clone()
    mask_prob = config.get("mask_prob", 0.15)
    device = input_ids.device

    probability_matrix = torch.full(labels.shape, mask_prob, device=device)

    # ✅ Exclude special tokens (CLS, SEP, PAD, etc.) - Dawn style
    # labels is [B, S], need to preserve batch dimension
    special_tokens_mask = []
    for seq in labels.tolist():  # Iterate over batch
        seq_mask = [
            tokenizer.get_special_tokens_mask([val], already_has_special_tokens=True)[0]
            for val in seq
        ]
        special_tokens_mask.append(seq_mask)
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # ✅ Exclude padding tokens (belt and suspenders)
    padding_mask = input_ids == tokenizer.pad_token_id
    probability_matrix.masked_fill_(padding_mask, value=0.0)

    # Sample masked positions
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # Apply masking strategy
    mask_ratio = config.get("mask_token_ratio", 0.8)
    random_ratio = config.get("random_token_ratio", 0.5)

    # 80% [MASK]
    indices_replaced = masked_indices & (torch.rand(labels.shape, device=device) < mask_ratio)
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% random (of remaining)
    indices_random = (
        masked_indices
        & ~indices_replaced
        & (torch.rand(labels.shape, device=device) < random_ratio)
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]

    # 10% keep original (implicit)
    return input_ids, labels


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

def compute_load_balance_loss(model):
    """
    모든 레이어의 load balancing loss를 계산합니다.

    Args:
        model: HierarchicalFFNModel

    Returns:
        total_loss: 전체 레이어의 평균 load balance loss
    """
    total_loss = 0
    count = 0

    for layer in model.layers:
        # 각 layer의 ffn에서 load balance loss 계산
        lb_loss = layer.ffn.get_load_balance_loss()
        total_loss = total_loss + lb_loss
        count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0)


def reset_routing_stats(model):
    """모든 레이어의 routing 통계를 초기화합니다."""
    for layer in model.layers:
        layer.ffn.reset_routing_counts()


def print_diagnostic_metrics(model, epoch):
    """
    학습 진단 메트릭 출력

    - Gradient norm (exploding/vanishing 체크)
    - Neuron 사용률 (input neurons)
    - Router entropy (다양성)
    - Process neuron 사용률
    """
    print(f"\n{'='*60}")
    print(f"Diagnostic Metrics (Epoch {epoch})")
    print(f"{'='*60}")

    # 1. Gradient norm
    grad_norm = 0.0
    grad_count = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
            grad_count += 1

    if grad_count > 0:
        grad_norm = grad_norm ** 0.5
        status = "✓ OK"
        if grad_norm < 0.01:
            status = "⚠ VANISHING"
        elif grad_norm > 100:
            status = "⚠ EXPLODING"
        print(f"Gradient Norm: {grad_norm:.4f} {status}")
        print(f"  Expected: 0.1 ~ 10 | < 0.01 = vanishing | > 100 = exploding")
    else:
        print(f"Gradient Norm: N/A (no gradients)")

    # 첫 번째 레이어의 FFN 분석 (대표값)
    first_ffn = model.layers[0].ffn

    # 2. Input Neuron 사용률
    if first_ffn.input_neuron_counts is not None:
        active_input = (first_ffn.input_neuron_counts > 0).sum().item()
        total_input = first_ffn.n_input
        usage_pct = active_input / total_input * 100

        status = "✓ OK" if usage_pct > 50 else "⚠ LOW"
        print(f"\nInput Neurons (Layer 0): {active_input}/{total_input} ({usage_pct:.1f}%) {status}")
        print(f"  Expected: > 50% for good diversity")
    else:
        print(f"\nInput Neurons: N/A (no stats collected)")

    # 3. Router Entropy (다양성)
    neuron_keys = first_ffn.global_router.neuron_keys  # [n_input, d_routing]
    # 각 뉴런의 key를 확률 분포로 변환 (softmax over neurons)
    # 높은 entropy = 다양한 뉴런이 선택될 가능성
    routing_logits = neuron_keys.norm(dim=1)  # [n_input] - 각 뉴런의 key 크기
    routing_probs = F.softmax(routing_logits, dim=0)
    entropy = -(routing_probs * (routing_probs + 1e-10).log()).sum().item()
    max_entropy = torch.log(torch.tensor(float(first_ffn.n_input))).item()
    entropy_pct = entropy / max_entropy * 100

    status = "✓ OK" if entropy_pct > 50 else "⚠ LOW"
    print(f"\nRouter Entropy (Layer 0): {entropy_pct:.1f}% of max {status}")
    print(f"  Expected: > 50% for diverse routing")

    # 4. Process Neuron 사용률
    if first_ffn.process_neuron_counts is not None:
        active_process = (first_ffn.process_neuron_counts > 0).sum().item()
        total_process = first_ffn.n_process
        process_pct = active_process / total_process * 100

        status = "✓ OK" if process_pct > 50 else "⚠ LOW"
        print(f"\nProcess Neurons (Layer 0): {active_process}/{total_process} ({process_pct:.1f}%) {status}")
        print(f"  Expected: ~100% if k_process=n_process (no selection)")
    else:
        print(f"\nProcess Neurons: N/A (no stats collected)")

    print(f"{'='*60}\n")


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args, scaler=None, tokenizer=None):
    """Train for one epoch"""
    model.train()

    # Epoch 시작 시 routing 통계 초기화
    reset_routing_stats(model)

    total_loss = 0
    total_tokens = 0
    total_correct = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)

        # Apply MLM masking
        if tokenizer is not None:
            input_ids, labels = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)
        else:
            # Fallback: no masking
            labels = input_ids.clone()

        # Detailed debugging for first 10 steps of epoch 1
        debug_mode = (epoch == 1 and step < 10)

        if debug_mode:
            print(f"\n{'='*60}")
            print(f"Step {step + 1} Debugging")
            print(f"{'='*60}")
            print(f"Before Forward:")
            print(f"  Input shape: {input_ids.shape}, range: [{input_ids.min()}, {input_ids.max()}]")
            print(f"  Model embedding norm: {model.token_embedding.weight.norm():.4f}")

        optimizer.zero_grad()

        # Mixed precision training
        # Dynamic aux weight: stronger in early epochs
        if epoch <= 5:
            aux_weight = 0.5  # Strong regularization initially (was 0.05)
        else:
            aux_weight = 0.1  # Moderate regularization later (was 0.01)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    k_input=args.k_input,
                    k_process=args.k_process
                )
                loss = outputs['loss']
                logits = outputs['logits']

                # Load balancing loss 추가
                aux_loss = compute_load_balance_loss(model)
                total_loss_combined = loss + aux_weight * aux_loss

            if debug_mode:
                print(f"\nAfter Forward:")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                print(f"  Loss: {loss.item():.4f}, Aux Loss: {aux_loss.item():.4f}")

                # Routing debug
                print(f"\n[Routing Debug - Step {step+1}]")
                first_ffn = model.layers[0].ffn

                if first_ffn.last_routing_scores is not None:
                    routing_weights = first_ffn.last_routing_scores  # [B, n_input]

                    # Top-k pattern analysis
                    topk_vals, topk_idx = routing_weights[0].topk(10)
                    print(f"  Top 10 routing weights: {topk_vals.cpu().numpy()}")

                    # Distribution stats
                    print(f"  Distribution: min={routing_weights.min():.6f}, max={routing_weights.max():.6f}, mean={routing_weights.mean():.6f}")

                    # Sparsity check
                    nonzero = (routing_weights[0] > 1e-6).sum()
                    print(f"  Non-zero weights: {nonzero}/{len(routing_weights[0])}")

                    # Usage stats
                    if first_ffn.input_neuron_counts is not None:
                        counts = first_ffn.input_neuron_counts
                        active_neurons = (counts > 0).sum()
                        print(f"  Active input neurons: {active_neurons}/{len(counts)}")
                else:
                    print(f"  No routing scores available")

            scaler.scale(total_loss_combined).backward()

            if debug_mode:
                print(f"\nAfter Backward:")
                grad_issues = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm < 1e-7:
                            grad_issues.append(f"  ⚠ {name}: grad too small ({grad_norm:.2e})")
                        elif grad_norm > 100:
                            grad_issues.append(f"  ⚠ {name}: grad too large ({grad_norm:.2e})")
                    else:
                        grad_issues.append(f"  ⚠ {name}: NO GRADIENT")

                if grad_issues:
                    print("  Gradient Issues:")
                    for issue in grad_issues[:10]:  # Show first 10 issues
                        print(issue)
                else:
                    print("  ✓ All gradients OK")

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Stronger clipping
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                k_input=args.k_input,
                k_process=args.k_process
            )
            loss = outputs['loss']
            logits = outputs['logits']

            # Load balancing loss 추가
            aux_loss = compute_load_balance_loss(model)
            total_loss_combined = loss + aux_weight * aux_loss

            if debug_mode:
                print(f"\nAfter Forward:")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                print(f"  Loss: {loss.item():.4f}, Aux Loss: {aux_loss.item():.4f}")

                # Routing debug
                print(f"\n[Routing Debug - Step {step+1}]")
                first_ffn = model.layers[0].ffn

                if first_ffn.last_routing_scores is not None:
                    routing_weights = first_ffn.last_routing_scores  # [B, n_input]

                    # Top-k pattern analysis
                    topk_vals, topk_idx = routing_weights[0].topk(10)
                    print(f"  Top 10 routing weights: {topk_vals.cpu().numpy()}")

                    # Distribution stats
                    print(f"  Distribution: min={routing_weights.min():.6f}, max={routing_weights.max():.6f}, mean={routing_weights.mean():.6f}")

                    # Sparsity check
                    nonzero = (routing_weights[0] > 1e-6).sum()
                    print(f"  Non-zero weights: {nonzero}/{len(routing_weights[0])}")

                    # Usage stats
                    if first_ffn.input_neuron_counts is not None:
                        counts = first_ffn.input_neuron_counts
                        active_neurons = (counts > 0).sum()
                        print(f"  Active input neurons: {active_neurons}/{len(counts)}")
                else:
                    print(f"  No routing scores available")

            total_loss_combined.backward()

            if debug_mode:
                print(f"\nAfter Backward:")
                grad_issues = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm < 1e-7:
                            grad_issues.append(f"  ⚠ {name}: grad too small ({grad_norm:.2e})")
                        elif grad_norm > 100:
                            grad_issues.append(f"  ⚠ {name}: grad too large ({grad_norm:.2e})")
                    else:
                        grad_issues.append(f"  ⚠ {name}: NO GRADIENT")

                if grad_issues:
                    print("  Gradient Issues:")
                    for issue in grad_issues[:10]:  # Show first 10 issues
                        print(issue)
                else:
                    print("  ✓ All gradients OK")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Stronger clipping
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
            "aux": f"{aux_loss.item():.4f}",
            "w_aux": f"{(aux_weight * aux_loss).item():.5f}",
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
                k_process=args.k_process
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
    parser = argparse.ArgumentParser(description='Train Hierarchical Dynamic Neuron FFN')

    # Model architecture
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='Maximum sequence length')

    # Hierarchical FFN specific
    parser.add_argument('--n_input_neurons', type=int, default=2048,
                        help='Number of input neurons')
    parser.add_argument('--n_process_neurons', type=int, default=1024,
                        help='Number of process neurons')
    parser.add_argument('--d_routing', type=int, default=256,
                        help='Routing dimension for global router')

    # Sparsity control (runtime)
    parser.add_argument('--k_input', type=int, default=None,
                        help='Number of input neurons to activate (None = n_input//8)')
    parser.add_argument('--k_process', type=int, default=None,
                        help='Number of process neurons to activate (None = n_process//8)')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
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
    print(f"Hierarchical Dynamic Neuron FFN Training")
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
    print("Creating Hierarchical FFN model...")
    print(f"{'='*60}")

    model = HierarchicalLanguageModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        n_input_neurons=args.n_input_neurons,
        n_process_neurons=args.n_process_neurons,
        d_routing=args.d_routing,
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
    print(f"  Router parameters: {stats['router_parameters']:,} ({stats['router_percentage']:.1f}%)")
    print(f"  Number of layers: {stats['n_layers']}")

    # Sparsity info
    # Start with less aggressive sparsity to verify architecture works
    # Then gradually increase sparsity if training succeeds
    if args.k_input is None:
        k_input_actual = args.n_input_neurons // 2  # 50% (was 12.5%)
    else:
        k_input_actual = args.k_input

    if args.k_process is None:
        k_process_actual = args.n_process_neurons  # 100% - no selection (was 12.5%)
    else:
        k_process_actual = args.k_process

    print(f"\nSparsity Configuration:")
    print(f"  Input neurons: {k_input_actual}/{args.n_input_neurons} ({k_input_actual/args.n_input_neurons*100:.1f}%)")
    print(f"  Process neurons: {k_process_actual}/{args.n_process_neurons} ({k_process_actual/args.n_process_neurons*100:.1f}%)")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )

    # Warmup + Cosine Annealing scheduler
    warmup_epochs = 2
    warmup_steps = warmup_epochs * len(train_loader)
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
            model, train_loader, optimizer, scheduler, device, epoch, args, scaler, tokenizer
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

        # Print diagnostic metrics every 100 epochs (or first epoch)
        if epoch == 1 or epoch % 100 == 0:
            print_diagnostic_metrics(model, epoch)

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
