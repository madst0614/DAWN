"""
Evaluate Phase 1 Baseline Performance

Measures Phase 1 checkpoint performance on each task.
This provides baseline for comparing Phase 2 improvements.

Usage:
    python eval_phase1_baseline.py --phase1_checkpoint ./checkpoints/phase1/phase1_final.pt
"""

import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import argparse
from tqdm import tqdm
from configs import DAWNConfig
from utils.task_registry import get_dataloader
from models import DAWN, TaskHeads


def evaluate_task(model, task_heads, task_name, expert_name, dataloader, device):
    """Evaluate single task"""
    model.eval()
    task_heads.eval()

    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0

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

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = (correct / total * 100) if total > 0 else 0

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 Baseline")
    parser.add_argument(
        "--phase1_checkpoint",
        type=str,
        required=True,
        help="Path to phase1_final.pt",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help="Model hidden dimension (if not in checkpoint)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=12,
        help="Number of attention heads (if not in checkpoint)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=10000,
        help="Number of samples per task for evaluation",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PHASE 1 BASELINE EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {args.phase1_checkpoint}")
    print(f"Device: {args.device}")
    print(f"Eval samples: {args.eval_samples} per task")
    print("=" * 70 + "\n")

    # Load checkpoint
    print("📂 Loading Phase 1 checkpoint...")
    ckpt = torch.load(args.phase1_checkpoint, map_location=args.device)

    if ckpt.get("phase") != 1:
        raise ValueError("Not a Phase 1 checkpoint!")

    print(f"✅ Loaded checkpoint")
    print(f"   Experts: {', '.join(ckpt.get('expert_names', []))}")

    # Get config from checkpoint or create new one
    if "config" in ckpt:
        print("📋 Using config from checkpoint")
        cfg = DAWNConfig.from_dict(ckpt["config"])
    else:
        print("📋 Creating config from parameters (legacy checkpoint)")
        cfg = DAWNConfig(
            phase=1,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
        )

    # Create model
    print("\n🔧 Creating model...")
    dawn_args = cfg.get_model_kwargs()
    dawn_args['enable_peer_prediction'] = False  # Phase 1 mode
    model = DAWN(**dawn_args)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(args.device)

    # Create task heads
    task_heads = TaskHeads(
        hidden_size=cfg.model.hidden_size,
        vocab_size=cfg.model.vocab_size,
    )
    task_heads.load_state_dict(ckpt["task_heads_state_dict"], strict=False)
    task_heads = task_heads.to(args.device)

    print("✅ Model ready\n")

    # Evaluate each task
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
            split="validation",  # Use validation split
            batch_size=64,
            task_config=task_cfg.task_params,
            max_samples=args.eval_samples,
            verbose=False,
            num_workers=4,
        )

        # Evaluate
        loss, accuracy = evaluate_task(
            model, task_heads, task_name, expert_name, dataloader, args.device
        )

        results[task_name] = {"loss": loss, "accuracy": accuracy, "expert": expert_name}

        if task_name == "sts":
            print(f"   Loss: {loss:.4f}")
        else:
            print(f"   Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        print()

    # Summary
    print("=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)
    print(f"{'Task':<20} {'Expert':<15} {'Loss':<10} {'Accuracy':<10}")
    print("─" * 70)

    for task_name, res in results.items():
        acc_str = f"{res['accuracy']:.2f}%" if task_name != "sts" else "N/A"
        print(f"{task_name:<20} {res['expert']:<15} {res['loss']:<10.4f} {acc_str:<10}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
