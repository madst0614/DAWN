"""
Phase 2 Training Script for DAWN

Trains all experts collaboratively with peer prediction and integrator.
Loads Phase 1 expert checkpoints and enables peer-to-peer learning.

Usage:
    python train_phase2.py [--phase1_dir ./checkpoints/phase1] [--device cuda]
"""

import sys
import os

# Add project root to Python path (for Colab compatibility)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import argparse
from pathlib import Path
from tqdm import tqdm

from configs import DAWNConfig
from utils.train_utils import (
    DAWNTrainer,
    load_phase1_checkpoints_to_phase2,
    validate_phase1_checkpoints,
)
from utils.task_registry import get_dataloader
from models import DAWN, TaskHeads


def evaluate_task(model, task_heads, task_name, expert_name, dataloader, device):
    """Evaluate single task (Phase 1 baseline) with improved metrics"""
    model.eval()
    task_heads.eval()

    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0

    # For balanced accuracy and F1 score
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  {task_name:20s}", ncols=100, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward through specific expert (Phase 1 style)
            expert = model.experts[expert_name]
            h = expert(input_ids, attention_mask)

            # Task head
            result = task_heads(h, task_name, labels)
            loss = result["loss"]
            logits = result.get("logits")

            total_loss += loss.item()
            num_batches += 1

            # Accuracy (skip for regression)
            if logits is not None and labels is not None and task_name != "sts":
                preds = logits.argmax(dim=-1)
                mask = labels != -100
                correct += ((preds == labels) & mask).sum().item()
                total += mask.sum().item()

                # Collect predictions and labels for balanced metrics
                all_preds.extend(preds[mask].cpu().numpy().tolist())
                all_labels.extend(labels[mask].cpu().numpy().tolist())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = (correct / total * 100) if total > 0 else 0

    # Compute balanced accuracy and F1 for classification tasks
    balanced_acc = None
    f1 = None
    if len(all_preds) > 0 and task_name != "sts":
        import numpy as np
        from collections import Counter

        # Balanced Accuracy (per-class accuracy average)
        unique_labels = sorted(set(all_labels))
        per_class_acc = []
        for label in unique_labels:
            indices = [i for i, l in enumerate(all_labels) if l == label]
            if indices:
                class_correct = sum(1 for i in indices if all_preds[i] == label)
                per_class_acc.append(class_correct / len(indices) * 100)

        if per_class_acc:
            balanced_acc = sum(per_class_acc) / len(per_class_acc)

        # F1 Score (for binary/multiclass)
        if len(unique_labels) == 2:  # Binary classification
            tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
            fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1 *= 100  # Convert to percentage

    return avg_loss, accuracy, balanced_acc, f1


def run_phase1_baseline_eval(phase1_checkpoint, cfg, device, eval_samples=10000):
    """Run Phase 1 baseline evaluation before Phase 2 training"""
    print("\n" + "=" * 70)
    print("PHASE 1 BASELINE EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {phase1_checkpoint}")
    print(f"Eval samples: {eval_samples} per task")
    print("=" * 70 + "\n")

    # Load checkpoint
    print("📂 Loading Phase 1 checkpoint...")
    ckpt = torch.load(phase1_checkpoint, map_location=device)

    if ckpt.get("phase") != 1:
        print("⚠️  Warning: Not a Phase 1 checkpoint, skipping baseline eval\n")
        return

    print(f"✅ Loaded checkpoint")
    print(f"   Experts: {', '.join(ckpt.get('expert_names', []))}\n")

    # Create model (Phase 1 style)
    dawn_args = cfg.get_model_kwargs()
    dawn_args['enable_peer_prediction'] = False  # Force Phase 1 mode for baseline
    model = DAWN(**dawn_args)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)

    # Create task heads
    task_heads = TaskHeads(
        hidden_size=model_config["hidden_size"],
        vocab_size=model_config["vocab_size"],
    )
    task_heads.load_state_dict(ckpt["task_heads_state_dict"], strict=False)
    task_heads = task_heads.to(device)

    print("=" * 70)
    print("EVALUATING TASKS")
    print("=" * 70 + "\n")

    results = {}

    for task_name, task_cfg in cfg.tasks.items():
        expert_name = task_cfg.expert_name

        print(f"📊 {task_name.upper()} (expert: {expert_name})")

        # Load dataloader
        dataloader = get_dataloader(
            task=task_name,
            split="validation",
            batch_size=64,
            task_config=task_cfg.task_params,
            max_samples=eval_samples,
            verbose=False,
            num_workers=4,
        )

        # Evaluate
        loss, accuracy, balanced_acc, f1 = evaluate_task(
            model, task_heads, task_name, expert_name, dataloader, device
        )

        results[task_name] = {
            "loss": loss,
            "accuracy": accuracy,
            "balanced_acc": balanced_acc,
            "f1": f1,
            "expert": expert_name
        }

        if task_name == "sts":
            print(f"   Loss: {loss:.4f}")
        else:
            metrics = [f"Loss: {loss:.4f}", f"Acc: {accuracy:.2f}%"]
            if balanced_acc is not None:
                metrics.append(f"Bal-Acc: {balanced_acc:.2f}%")
            if f1 is not None:
                metrics.append(f"F1: {f1:.2f}%")
            print(f"   {', '.join(metrics)}")
        print()

    # Summary
    print("=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)
    print(f"{'Task':<20} {'Expert':<12} {'Loss':<8} {'Acc':<8} {'Bal-Acc':<8} {'F1':<8}")
    print("─" * 70)

    for task_name, res in results.items():
        acc_str = f"{res['accuracy']:.2f}%" if task_name != "sts" else "N/A"
        bal_str = f"{res['balanced_acc']:.2f}%" if res.get('balanced_acc') is not None else "N/A"
        f1_str = f"{res['f1']:.2f}%" if res.get('f1') is not None else "N/A"
        print(f"{task_name:<20} {res['expert']:<12} {res['loss']:<8.4f} {acc_str:<8} {bal_str:<8} {f1_str:<8}")

    print("=" * 70 + "\n")


def train_phase2_epoch(
    model,
    task_heads,
    trainer,
    task_dataloaders: dict,
    epoch: int,
    config: dict,
):
    """
    Train one epoch of Phase 2

    Iterates through tasks, computing both task loss and peer prediction loss.

    Args:
        model: DAWN model with peer prediction enabled
        task_heads: TaskHeads module
        trainer: DAWNTrainer instance
        task_dataloaders: {task_name: dataloader}
        epoch: Current epoch number
        config: Training configuration dict (converted from RuntimeConfig)

    Returns:
        Epoch metrics dict
    """
    model.train()
    task_heads.train()

    task_loss_weight = config.get("task_loss_weight", 1.0)
    peer_loss_weight = config.get("peer_prediction_loss_weight", 0.5)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    gradient_clip = config.get("gradient_clip", 1.0)
    log_every = config.get("log_every_n_steps", 500)  # Less frequent (was 100)

    epoch_metrics = {
        "total_loss": 0,
        "task_loss": 0,
        "extract_loss": 0,  # Feature extraction loss
        "steps": 0,
    }

    # Task sampling strategy
    task_sampling = config.get("task_sampling", "uniform")
    tasks = list(task_dataloaders.keys())

    print(f"\n{'='*70}")
    print(f"EPOCH {epoch + 1}")
    print(f"{'='*70}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Task sampling: {task_sampling}")
    print(f"{'='*70}\n")

    # Sequential task iteration (simpler than mixed batching)
    for task_name in tasks:
        dataloader = task_dataloaders[task_name]

        print(f"\n{'─'*70}")
        print(f"📊 TASK: {task_name.upper()}")
        print(f"{'─'*70}")

        pbar = tqdm(
            dataloader,
            desc=f"  {task_name:20s}",
            ncols=120,
            leave=True,
        )

        task_steps = 0
        task_loss_sum = 0
        extract_loss_sum = 0
        task_correct = 0
        task_total = 0

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch["input_ids"].to(trainer.device)
            attention_mask = batch["attention_mask"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)

            # Forward pass with AMP
            def forward_fn():
                # Full model forward with all experts + integrator
                output_dict = model(
                    input_ids,
                    attention_mask,
                    return_expert_outputs=True,
                )

                integrated = output_dict["integrated"]
                expert_outputs = output_dict["expert_outputs"]

                # Task loss (through task head)
                task_output = task_heads(integrated, task=task_name, labels=labels)
                task_loss = task_output["loss"]
                logits = task_output.get("logits")

                # Feature extraction loss (learning to extract from frozen peers)
                extract_loss, _ = model.compute_peer_prediction_loss(expert_outputs)

                # Combined loss
                total_loss = (task_loss_weight * task_loss) + (
                    peer_loss_weight * extract_loss
                )

                return total_loss, task_loss, extract_loss, logits

            total_loss, task_loss, extract_loss, logits = trainer.forward_with_amp(forward_fn)

            # Scale loss for gradient accumulation
            total_loss = total_loss / gradient_accumulation_steps

            # Backward
            trainer.backward_and_step(
                total_loss,
                accumulation_steps=gradient_accumulation_steps,
                gradient_clip=gradient_clip,
            )

            # Track metrics
            task_loss_sum += task_loss.item()
            extract_loss_sum += extract_loss.item()
            task_steps += 1

            # Accuracy (skip for regression tasks like STS)
            if logits is not None and labels is not None:
                if task_name != "sts":
                    preds = logits.argmax(dim=-1)
                    mask = labels != -100
                    task_correct += ((preds == labels) & mask).sum().item()
                    task_total += mask.sum().item()

            epoch_metrics["steps"] += 1
            epoch_metrics["total_loss"] += (
                total_loss.item() * gradient_accumulation_steps
            )
            epoch_metrics["task_loss"] += task_loss.item()
            epoch_metrics["extract_loss"] += extract_loss.item()

            # Update progress bar
            task_acc = (task_correct / task_total * 100) if task_total > 0 else 0.0
            pbar.set_postfix({
                "loss": f"{task_loss.item():.3f}",
                "extract": f"{extract_loss.item():.3f}",
                "acc": f"{task_acc:.1f}%" if task_name != "sts" else "N/A",
            })

            # Periodic logging
            if epoch_metrics["steps"] % log_every == 0:
                avg_task = epoch_metrics["task_loss"] / epoch_metrics["steps"]
                avg_extract = epoch_metrics["extract_loss"] / epoch_metrics["steps"]
                current_lr = trainer.get_current_lr()
                print(
                    f"\n    Step {epoch_metrics['steps']:5d}: "
                    f"task={avg_task:.4f}, extract={avg_extract:.4f}, "
                    f"lr={current_lr:.2e}"
                )

        # Task summary
        avg_task_loss = task_loss_sum / task_steps if task_steps > 0 else 0
        avg_extract_loss = extract_loss_sum / task_steps if task_steps > 0 else 0
        task_acc = (task_correct / task_total * 100) if task_total > 0 else 0.0

        print(f"\n  ✅ {task_name.upper()} Complete:")
        print(f"     Task Loss:    {avg_task_loss:.4f}")
        print(f"     Extract Loss: {avg_extract_loss:.4f}")
        if task_name != "sts":
            print(f"     Accuracy:     {task_acc:.2f}%")
        print(f"{'─'*70}")

    # Compute epoch averages
    if epoch_metrics["steps"] > 0:
        epoch_metrics["avg_total_loss"] = (
            epoch_metrics["total_loss"] / epoch_metrics["steps"]
        )
        epoch_metrics["avg_task_loss"] = (
            epoch_metrics["task_loss"] / epoch_metrics["steps"]
        )
        epoch_metrics["avg_extract_loss"] = (
            epoch_metrics["extract_loss"] / epoch_metrics["steps"]
        )
    else:
        epoch_metrics["avg_total_loss"] = 0
        epoch_metrics["avg_task_loss"] = 0
        epoch_metrics["avg_extract_loss"] = 0

    return epoch_metrics


def train_phase2(
    phase1_checkpoint_dir: str,
    cfg: DAWNConfig,
    device: str = "cuda",
    eval_samples: int = 10000,
    skip_baseline: bool = False,
):
    """
    Run Phase 2 collaborative training

    Args:
        phase1_checkpoint_dir: Directory containing Phase 1 expert checkpoints
        cfg: RuntimeConfig instance
        device: Device to train on
        eval_samples: Number of samples for baseline evaluation
        skip_baseline: Skip Phase 1 baseline evaluation
    """
    print("\n" + "=" * 70)
    print("PHASE 2: COLLABORATIVE TRAINING")
    print("=" * 70)
    print(f"Phase 1 checkpoints: {phase1_checkpoint_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {cfg.training.epochs}")
    print(f"Tasks: {', '.join(cfg.phase2_tasks)}")
    print("=" * 70 + "\n")

    # Create checkpoint directory
    checkpoint_dir = cfg.training.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Build Phase 1 checkpoint path (unified file)
    phase1_checkpoint = os.path.join(phase1_checkpoint_dir, "phase1_final.pt")

    # Validate checkpoint
    if not validate_phase1_checkpoints(phase1_checkpoint):
        raise ValueError("Phase 1 checkpoint validation failed!")

    # Run Phase 1 baseline evaluation (optional)
    if not skip_baseline:
        run_phase1_baseline_eval(
            phase1_checkpoint=phase1_checkpoint,
            cfg=cfg,
            device=device,
            eval_samples=eval_samples,
        )
    else:
        print("\n⏭️  Skipping Phase 1 baseline evaluation (--skip_baseline)\n")

    # Load Phase 1 checkpoint and create Phase 2 model
    dawn_args = cfg.get_model_kwargs()
    model, task_heads = load_phase1_checkpoints_to_phase2(
        checkpoint_path=phase1_checkpoint,
        model_config=dawn_args['config'],
        device=device,
    )

    # Configure training
    if cfg.training.freeze_experts:
        print("❄️  Freezing expert base parameters (feature extractors only)")
        for expert in model.experts.values():
            expert.freeze_base()
    else:
        print("🔥 Training all parameters (experts + feature extractors + integrator)")

    # Create trainer
    # Convert training config to dict for trainer
    phase2_config_dict = {
        "batch_size": cfg.hardware.batch_size,
        "gradient_accumulation_steps": cfg.hardware.gradient_accumulation_steps,
        "checkpoint_dir": cfg.training.checkpoint_dir,
        "log_every_n_steps": cfg.training.log_every_n_steps,
        "save_every_n_epochs": cfg.training.save_every_n_epochs,
        "mixed_precision": cfg.hardware.mixed_precision,
        "task_loss_weight": cfg.training.task_loss_weight,
        "peer_prediction_loss_weight": cfg.training.peer_prediction_loss_weight,
        "task_sampling": cfg.training.task_sampling,
        "gradient_clip": cfg.training.optimizer.gradient_clip,
        "epochs": cfg.training.epochs,
    }
    trainer = DAWNTrainer(
        model=model,
        task_heads=task_heads,
        config=phase2_config_dict,
        device=device,
        log_dir=checkpoint_dir,
    )

    # Print model info
    trainer.print_model_info()

    # Load task dataloaders
    print("\n📊 Loading task dataloaders...")
    task_dataloaders = {}

    for task_name in cfg.phase2_tasks:
        if task_name not in cfg.tasks:
            raise ValueError(f"Task '{task_name}' not found in configuration")

        task_config = cfg.tasks[task_name]
        task_specific_config = task_config.task_params
        max_samples = task_config.max_samples

        dataloader_config_dict = {
            "num_workers": cfg.dataloader.num_workers,
            "pin_memory": cfg.dataloader.pin_memory,
            "prefetch_factor": cfg.dataloader.prefetch_factor,
            "persistent_workers": cfg.dataloader.persistent_workers,
        }
        dataloader = get_dataloader(
            task=task_name,
            split="train",
            batch_size=cfg.hardware.batch_size,
            task_config=task_specific_config,
            max_samples=max_samples,
            verbose=True,
            **dataloader_config_dict,
        )

        task_dataloaders[task_name] = dataloader

    # Calculate total steps for all tasks
    total_steps = (
        sum(len(dl) for dl in task_dataloaders.values()) * cfg.training.epochs
    )
    total_steps = total_steps // cfg.hardware.gradient_accumulation_steps

    # Setup optimizer
    trainer.setup_optimizer(
        base_lr=cfg.training.optimizer.base_lr,
        embedding_lr=cfg.training.optimizer.embedding_lr,
        total_steps=total_steps,
        warmup_ratio=cfg.training.optimizer.warmup_ratio,
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    # Print trainable parameters
    trainer.print_trainable_summary()

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING PHASE 2 TRAINING")
    print("=" * 70 + "\n")

    all_epoch_metrics = []

    for epoch in range(cfg.training.epochs):
        # Train one epoch
        epoch_metrics = train_phase2_epoch(
            model=model,
            task_heads=task_heads,
            trainer=trainer,
            task_dataloaders=task_dataloaders,
            epoch=epoch,
            config=phase2_config_dict,
        )

        all_epoch_metrics.append(epoch_metrics)

        # Epoch summary
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{cfg.training.epochs} SUMMARY")
        print(f"{'='*70}")
        print(f"Avg Task Loss:    {epoch_metrics['avg_task_loss']:.4f}")
        print(f"Avg Extract Loss: {epoch_metrics['avg_extract_loss']:.4f}")
        print(f"Avg Total Loss:   {epoch_metrics['avg_total_loss']:.4f}")
        print(f"{'='*70}\n")

        # Save checkpoint
        if (epoch + 1) % cfg.training.save_every_n_epochs == 0:
            checkpoint_path = trainer.save_checkpoint(
                checkpoint_name=f"phase2_epoch{epoch+1}",
                epoch=epoch + 1,
                metrics=epoch_metrics,
            )

    # Save final checkpoint
    final_path = trainer.save_checkpoint(
        checkpoint_name="phase2_final",
        metrics=all_epoch_metrics[-1] if all_epoch_metrics else {},
    )

    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE! 🎉")
    print("=" * 70)

    print("\n📊 Training Summary:")
    print(f"{'Epoch':<10} {'Task Loss':<15} {'Extract Loss':<15} {'Total Loss':<15}")
    print("─" * 70)
    for i, metrics in enumerate(all_epoch_metrics, 1):
        print(
            f"{i:<10} {metrics['avg_task_loss']:<15.4f} "
            f"{metrics['avg_extract_loss']:<15.4f} {metrics['avg_total_loss']:<15.4f}"
        )

    print("\n" + "=" * 70)
    print(f"Final model saved: {final_path}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="DAWN Phase 2 Training")
    parser.add_argument(
        "--phase1_dir",
        type=str,
        default="./checkpoints/phase1",
        help="Directory containing Phase 1 checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override checkpoint output directory",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help="Model hidden dimension",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=5e-5,
        help="Base learning rate for Phase 2",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of Phase 2 training epochs",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate config and checkpoints without training",
    )
    parser.add_argument(
        "--freeze_experts",
        action="store_true",
        help="Freeze expert parameters, train only peer predictors and integrator",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=10000,
        help="Number of samples per task for baseline evaluation",
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip Phase 1 baseline evaluation",
    )

    args = parser.parse_args()

    # Get Phase 2 configuration with direct parameters
    print("\n🔍 Building Phase 2 configuration...")
    cfg = DAWNConfig(
        phase=2,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        learning_rate=args.base_lr,
        epochs=args.epochs,
    )

    # Override config options
    if args.checkpoint_dir:
        cfg.training.checkpoint_dir = args.checkpoint_dir
        print(f"📁 Using custom checkpoint directory: {args.checkpoint_dir}")

    if args.freeze_experts:
        cfg.training.freeze_experts = True
        print("❄️  Experts will be frozen (training peer predictors only)")

    # Print configuration summary
    cfg.print_summary()

    # Determine Phase 1 checkpoint directory
    phase1_dir = args.phase1_dir

    if not os.path.exists(phase1_dir):
        raise FileNotFoundError(f"Phase 1 checkpoint directory not found: {phase1_dir}")

    print(f"\n📁 Phase 1 checkpoints: {phase1_dir}")

    if args.validate_only:
        print("\n🔍 Validating Phase 1 checkpoint...")
        phase1_checkpoint = os.path.join(phase1_dir, "phase1_final.pt")

        if validate_phase1_checkpoints(phase1_checkpoint):
            print(
                "✅ Configuration and checkpoint are valid. Exiting (--validate_only).\n"
            )
        else:
            print("❌ Validation failed!\n")
        return

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"🖥️  Using device: {args.device}\n")

    # Run Phase 2 training
    train_phase2(
        phase1_checkpoint_dir=phase1_dir,
        cfg=cfg,
        device=args.device,
        eval_samples=args.eval_samples,
        skip_baseline=args.skip_baseline,
    )


if __name__ == "__main__":
    main()
