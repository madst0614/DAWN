"""
Large Batch Training Experiment (PNN-style)

주 실험과 별개로 성능 테스트용 스크립트.
PNN 검증된 큰 배치 설정을 DAWN에 적용.

모든 설정이 스크립트 안에 포함됨 (별도 config 파일 불필요).

Usage:
    # Default PNN settings (batch 384 × 3 = 1,152)
    python scripts/train_large_batch_experiment.py

    # Larger batch
    python scripts/train_large_batch_experiment.py --batch_size 512 --gradient_accumulation 6

    # Different expert
    python scripts/train_large_batch_experiment.py --expert semantic --epochs 12
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse
import time
import json
from pathlib import Path

# DAWN components
from models.expert import DeltaExpert
from models.task_heads import TaskHeads
from utils.task_registry import get_dataloader


# ============================================================================
# PNN-STYLE TRAINING CONFIGURATION (모든 설정 여기에)
# ============================================================================

# Default settings (PNN 검증된 값)
DEFAULT_CONFIG = {
    # Model
    'hidden_size': 768,
    'num_heads': 12,
    'intermediate_size': [1024, 1536, 2048, 1536, 1024],  # Mountain-shaped (PNN hierarchical)
    'num_steps': 4,
    'max_length': 128,
    'dropout': 0.1,
    'vocab_size': 30522,  # BERT vocab

    # Training - PNN verified
    'batch_size': 384,
    'gradient_accumulation': 3,  # Effective: 1,152
    'epochs': 15,
    'base_lr': 3e-4,  # PNN large batch
    'embedding_lr': 1e-4,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,

    # Data
    'task': 'mlm',
    'expert': 'lexical',

    # System
    'num_workers': 4,
    'use_amp': True,
    'use_tf32': True,
    'checkpoint_dir': './checkpoints/large_batch_experiment',
}

# ============================================================================


def create_expert_and_head(cfg):
    """Create expert and task head"""

    # Build unified config dict for DeltaExpert (DAWN style)
    expert_config = {
        # Model architecture
        'vocab_size': cfg['vocab_size'],
        'hidden_size': cfg['hidden_size'],
        'num_heads': cfg['num_heads'],
        'intermediate_size': cfg['intermediate_size'],
        'max_length': cfg['max_length'],
        'num_steps': cfg['num_steps'],
        'dropout': cfg['dropout'],
        'init_std': 0.02,

        # Delta module config (PNN hierarchical refiner style)
        'delta_module': {
            'num_blocks': 5,
            'gating_type': 'query_key',
            'temperature_scale': 'sqrt',
            'attention_dropout': 0.1,
            'attention_type': 'multihead',
            'ffn_activation': 'gelu',
            'ffn_dropout': 0.1,
            'zero_init_final_layer': True,
            'zero_init_gate': False,
        },

        # Peer integrator config (Phase 1: no peers)
        'integration': {
            'method': 'weighted_sum',
            'use_layer_norm': True,
        },
    }

    # Create expert (no shared embeddings, no peers for Phase 1)
    expert = DeltaExpert(
        config=expert_config,
        peer_names=None,  # Phase 1: independent training
        shared_embeddings=None,  # Expert has own embeddings
    )

    # Create task head
    task_head = TaskHeads(
        hidden_size=cfg['hidden_size'],
        vocab_size=cfg['vocab_size'],
    )

    return expert, task_head


def create_optimizer(expert, task_head, cfg):
    """Create optimizer with separate LRs for embeddings"""
    embedding_params = []
    other_params = []

    for name, param in list(expert.named_parameters()) + list(task_head.named_parameters()):
        if 'embedding' in name.lower():
            embedding_params.append(param)
        else:
            other_params.append(param)

    return AdamW([
        {'params': other_params, 'lr': cfg['base_lr']},
        {'params': embedding_params, 'lr': cfg['embedding_lr']}
    ], weight_decay=cfg['weight_decay'])


def create_scheduler(optimizer, cfg, num_training_steps):
    """Create linear warmup + linear decay scheduler"""
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg['warmup_steps'],
        num_training_steps=num_training_steps,
    )


def compute_recurrent_loss_and_metrics(expert, task_head, input_ids, attention_mask, labels):
    """Compute recurrent loss (PNN style)"""
    all_hiddens = expert(input_ids, attention_mask, return_all_steps=True)
    step_hiddens = all_hiddens[1:]  # Skip initial embedding
    num_steps = len(step_hiddens)

    step_losses = []
    step_accs = []
    weighted_losses = []

    # Step weights (PNN style)
    step_weights = [0.1, 0.2, 0.3, 0.4][:num_steps]
    if len(step_weights) != num_steps:
        step_weights = [1.0 / num_steps] * num_steps

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for step_idx, h in enumerate(step_hiddens):
        logits = task_head.mlm_head(task_head.mlm_norm(h))
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        step_losses.append(loss.item())
        weighted_losses.append(step_weights[step_idx] * loss)

        # Accuracy
        preds = logits.argmax(dim=-1).view(-1)
        labels_flat = labels.view(-1)
        mask = (labels_flat != -100)
        if mask.sum() > 0:
            acc = (preds == labels_flat)[mask].float().mean().item()
        else:
            acc = 0.0
        step_accs.append(acc)

    return sum(weighted_losses), step_losses, step_accs


def train_epoch(expert, task_head, dataloader, optimizer, scheduler, scaler, cfg, device, epoch):
    """Train for one epoch"""
    expert.train()
    task_head.train()

    total_loss = 0.0
    num_steps = cfg['num_steps']
    step_losses_accum = [0.0] * num_steps
    step_accs_accum = [0.0] * num_steps
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg['epochs']}", ncols=150, leave=False)

    for step, batch in enumerate(pbar, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if (labels != -100).sum() == 0:
            continue

        with autocast(device_type='cuda', dtype=torch.float16, enabled=cfg['use_amp']):
            loss, batch_step_losses, batch_step_accs = compute_recurrent_loss_and_metrics(
                expert, task_head, input_ids, attention_mask, labels
            )
            loss = loss / cfg['gradient_accumulation']

        scaler.scale(loss).backward()

        total_loss += loss.item() * cfg['gradient_accumulation']
        for i in range(num_steps):
            step_losses_accum[i] += batch_step_losses[i]
            step_accs_accum[i] += batch_step_accs[i]
        num_batches += 1

        if step % cfg['gradient_accumulation'] == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(expert.parameters()) + list(task_head.parameters()),
                cfg['gradient_clip']
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            losses_str = ','.join([f'{l:.3f}' for l in batch_step_losses])
            accs_str = ','.join([f'{a:.3f}' for a in batch_step_accs])
            pbar.set_postfix({
                'L': f'[{losses_str}]',
                'A': f'[{accs_str}]',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'gn': f'{grad_norm:.2f}',
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_step_losses = [sl / num_batches for sl in step_losses_accum]
    avg_step_accs = [sa / num_batches for sa in step_accs_accum]

    return avg_loss, avg_step_losses, avg_step_accs


@torch.no_grad()
def evaluate(expert, task_head, dataloader, cfg, device):
    """Evaluate"""
    expert.eval()
    task_head.eval()

    total_loss = 0.0
    num_steps = cfg['num_steps']
    step_losses_accum = [0.0] * num_steps
    step_accs_accum = [0.0] * num_steps
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if (labels != -100).sum() == 0:
            continue

        with autocast(device_type='cuda', dtype=torch.float16, enabled=cfg['use_amp']):
            loss, batch_step_losses, batch_step_accs = compute_recurrent_loss_and_metrics(
                expert, task_head, input_ids, attention_mask, labels
            )

        total_loss += loss.item()
        for i in range(num_steps):
            step_losses_accum[i] += batch_step_losses[i]
            step_accs_accum[i] += batch_step_accs[i]
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_step_losses = [sl / num_batches for sl in step_losses_accum]
    avg_step_accs = [sa / num_batches for sa in step_accs_accum]

    return avg_loss, avg_step_losses, avg_step_accs


def save_checkpoint(path, expert, task_head, optimizer, scheduler, scaler, epoch, metrics, cfg):
    """Save checkpoint"""
    torch.save({
        'epoch': epoch,
        'expert_state': expert.state_dict(),
        'task_head_state': task_head.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'config': cfg,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Large Batch Training Experiment")

    # Override options
    parser.add_argument('--expert', type=str, default=None, choices=['lexical', 'semantic', 'reasoning'])
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--gradient_accumulation', type=int, default=None)
    parser.add_argument('--base_lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Build config from defaults + overrides
    cfg = DEFAULT_CONFIG.copy()
    if args.expert:
        cfg['expert'] = args.expert
    if args.task:
        cfg['task'] = args.task
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.gradient_accumulation:
        cfg['gradient_accumulation'] = args.gradient_accumulation
    if args.base_lr:
        cfg['base_lr'] = args.base_lr
    if args.epochs:
        cfg['epochs'] = args.epochs
    if args.checkpoint_dir:
        cfg['checkpoint_dir'] = args.checkpoint_dir

    # Print config
    effective_batch = cfg['batch_size'] * cfg['gradient_accumulation']
    print("\n" + "=" * 70)
    print("LARGE BATCH TRAINING EXPERIMENT (PNN-style)")
    print("=" * 70)
    print(f"Expert: {cfg['expert']}")
    print(f"Task: {cfg['task']}")
    print(f"Batch: {cfg['batch_size']} × {cfg['gradient_accumulation']} = {effective_batch}")
    print(f"LR: {cfg['base_lr']} (base), {cfg['embedding_lr']} (embedding)")
    print(f"Epochs: {cfg['epochs']}")
    print(f"Checkpoints: {cfg['checkpoint_dir']}")
    print("=" * 70 + "\n")

    # Setup
    device = torch.device(args.device)
    print(f"🖥️  Device: {device}")

    if cfg['use_tf32'] and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TF32 enabled")

    # Create checkpoint dir
    checkpoint_dir = Path(cfg['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    print(f"\n🤖 Creating {cfg['expert']} expert...")
    expert, task_head = create_expert_and_head(cfg)
    expert = expert.to(device)
    task_head = task_head.to(device)

    total_params = sum(p.numel() for p in expert.parameters()) + sum(p.numel() for p in task_head.parameters())
    print(f"   Parameters: {total_params:,}")

    # Load data
    print(f"\n📚 Loading {cfg['task']} data...")
    train_dataloader = get_dataloader(
        task=cfg['task'],
        split="train",
        batch_size=cfg['batch_size'],
        max_length=cfg['max_length'],
        max_samples=1_000_000,  # PNN-style: 1M samples
        num_workers=cfg['num_workers'],
    )

    val_dataloader = get_dataloader(
        task=cfg['task'],
        split="validation",
        batch_size=cfg['batch_size'],
        max_length=cfg['max_length'],
        num_workers=cfg['num_workers'],
    )

    print(f"   Train batches: {len(train_dataloader)}")
    print(f"   Val batches: {len(val_dataloader)}")

    # Create optimizer & scheduler
    optimizer = create_optimizer(expert, task_head, cfg)
    num_training_steps = len(train_dataloader) * cfg['epochs'] // cfg['gradient_accumulation']
    scheduler = create_scheduler(optimizer, cfg, num_training_steps)
    scaler = GradScaler('cuda') if cfg['use_amp'] else None

    # Training loop
    print("\n" + "=" * 70)
    print("🚀 Starting training")
    print("=" * 70 + "\n")

    best_acc = 0.0
    history = []

    for epoch in range(1, cfg['epochs'] + 1):
        epoch_start = time.time()

        train_loss, train_step_losses, train_step_accs = train_epoch(
            expert, task_head, train_dataloader, optimizer, scheduler, scaler, cfg, device, epoch
        )

        print("   Evaluating...")
        eval_loss, eval_step_losses, eval_step_accs = evaluate(
            expert, task_head, val_dataloader, cfg, device
        )

        epoch_time = time.time() - epoch_start
        final_acc = eval_step_accs[-1]

        print(f"\n   Epoch {epoch}/{cfg['epochs']}:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Train Step L: {[f'{l:.4f}' for l in train_step_losses]}")
        print(f"   Train Step A: {[f'{a:.4f}' for a in train_step_accs]}")
        print(f"   Eval Loss:  {eval_loss:.4f}")
        print(f"   Eval Step L: {[f'{l:.4f}' for l in eval_step_losses]}")
        print(f"   Eval Step A: {[f'{a:.4f}' for a in eval_step_accs]}")
        print(f"   Final Acc: {final_acc:.4f} ({final_acc*100:.2f}%)")
        print(f"   Time: {epoch_time:.1f}s ({epoch_time/60:.1f}m)\n")

        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'eval_acc': final_acc,
            'train_step_losses': train_step_losses,
            'train_step_accs': train_step_accs,
            'eval_step_losses': eval_step_losses,
            'eval_step_accs': eval_step_accs,
            'time': epoch_time,
        }
        history.append(metrics)

        if final_acc > best_acc:
            best_acc = final_acc
            save_checkpoint(
                checkpoint_dir / 'best_model.pt',
                expert, task_head, optimizer, scheduler, scaler, epoch, metrics, cfg
            )
            print(f"   💾 Saved best model (acc: {best_acc:.4f})\n")

        if epoch == cfg['epochs']:
            save_checkpoint(
                checkpoint_dir / 'final_model.pt',
                expert, task_head, optimizer, scheduler, scaler, epoch, metrics, cfg
            )
            print(f"   💾 Saved final model\n")

    # Save history
    with open(checkpoint_dir / 'history.json', 'w') as f:
        json.dump({'best_acc': best_acc, 'config': cfg, 'history': history}, f, indent=2)

    print("=" * 70)
    print("🎉 Training complete!")
    print(f"   Best accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"   Checkpoints: {checkpoint_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
