"""
Phase 1 Training Script for DAWN

Trains each expert independently using DAWN model with shared embeddings.
Uses single DAWN instance with all experts, activating one at a time.

Usage:
    # Use default configuration (no presets needed!)
    python train_phase1.py

    # Customize parameters directly
    python train_phase1.py --batch_size 32 --base_lr 3e-5 --gradient_clip 0.3

    # Override device or checkpoint directory
    python train_phase1.py --device cuda --checkpoint_dir ./my_checkpoints
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

from configs import DAWNConfig, PHASE1_CURRICULUM
from models import DAWN, TaskHeads
from utils.task_registry import get_dataloader
from utils.model_factory import ModelFactory

import time
import sys
import json
from torch.optim import AdamW
from torch.amp import autocast
from torch.amp import GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


def train_single_task(
    model,
    task_heads,
    expert_name,
    task_name,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    scaler,
    num_epochs,
    gradient_clip=1.0,
    gradient_accumulation_steps=1,
    device="cuda",
    start_epoch=1,
    training_history=None,
    checkpoint_dir=None,
    cfg=None,  # Added config parameter
):
    """
    Train single expert on single task with tqdm progress bar.

    Progress format (PNN style with step-wise metrics):
    E1/3: 75%|████████████░░░| 750/1000 [02:30<00:50, 1.2it/s] L:[2.1,1.9,1.7,1.5] A:[78,82,85,87]%

    Args:
        start_epoch: Epoch to start from (for resuming)
        training_history: List to append epoch metrics to
        checkpoint_dir: Directory to save epoch checkpoints (None = no saving)
    """
    if training_history is None:
        training_history = []

    task_heads.freeze_all_except(task_name)

    # Track best accuracy
    best_val_acc = 0
    if training_history:
        # Resume: get best accuracy from history
        best_val_acc = max([h.get("val_accuracy", 0) for h in training_history])

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        task_heads.train()

        total_loss = 0
        correct = 0
        total = 0
        epoch_start = time.time()

        # PNN-style: Track step-wise losses and accuracies
        num_steps = model.config.get('num_steps', 4)
        step_losses_accum = [0.0] * num_steps
        step_accs_accum = [0.0] * num_steps

        # Training loop with progress bar
        pbar = tqdm(
            train_dataloader,
            desc=f"E{epoch}/{num_epochs} [Train]",
            ncols=150,  # Increased to prevent truncation
            leave=False,
        )

        for step, batch in enumerate(pbar, 1):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward with PNN-style step-wise loss computation
            with autocast(device_type='cuda', dtype=torch.float16):
                # Use recurrent loss computation (PNN style)
                expert = model.experts[expert_name]
                result = expert.compute_recurrent_loss(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_heads=task_heads,
                    task_name=task_name,
                    labels=labels,
                    return_accuracies=True
                )
                loss = result[0] / gradient_accumulation_steps
                batch_step_losses = result[1]
                batch_step_accs = result[2]

            # Backward (accumulate gradients)
            scaler.scale(loss).backward()

            # Only update optimizer every gradient_accumulation_steps
            if step % gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)

                # Clip gradients for stability
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(task_heads.parameters()),
                    gradient_clip
                )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler:
                    scheduler.step()

            # Accumulate metrics (restore original loss scale for logging)
            total_loss += loss.item() * gradient_accumulation_steps

            # Accumulate step-wise losses and accuracies (PNN style)
            for i, sl in enumerate(batch_step_losses):
                step_losses_accum[i] += sl
            for i, sa in enumerate(batch_step_accs):
                step_accs_accum[i] += sa

            # 🔍 DEBUG LOGGING: Save to file every 50 steps (no console spam)
            if cfg and getattr(cfg, 'debug_mode', False) and step % 50 == 0:
                expert = model.experts[expert_name]
                refiner = expert.delta_module.refiner

                # Check if this is hierarchical refiner with gates
                if hasattr(refiner, 'mini_gates'):
                    # Open debug log file
                    debug_log_path = os.path.join(checkpoint_dir, f"{expert_name}_{task_name}_debug.txt")
                    with open(debug_log_path, 'a') as f:
                        f.write(f"\n{'='*70}\n")
                        f.write(f"Epoch {epoch}, Step {step}, Loss: {total_loss / step:.4f}\n")
                        f.write(f"{'='*70}\n")

                        critical_warnings = []

                        # Check each mini-gate
                        for gate_idx, gate in enumerate(refiner.mini_gates):
                            temp = gate.temperature.item()
                            q_bias = gate.query_proj.bias
                            k_bias = gate.key_proj.bias
                            q_weight = gate.query_proj.weight
                            k_weight = gate.key_proj.weight

                            q_bias_max = q_bias.abs().max().item()
                            q_bias_std = q_bias.std().item()
                            k_bias_max = k_bias.abs().max().item()
                            k_bias_std = k_bias.std().item()
                            q_weight_max = q_weight.abs().max().item()
                            k_weight_max = k_weight.abs().max().item()

                            f.write(f"\nMini-Gate {gate_idx}:\n")
                            f.write(f"  Temperature: {temp:.6f}\n")
                            f.write(f"  Query bias - Max: {q_bias_max:.6f}, Std: {q_bias_std:.6f}\n")
                            f.write(f"  Key bias   - Max: {k_bias_max:.6f}, Std: {k_bias_std:.6f}\n")
                            f.write(f"  Query weight max: {q_weight_max:.6f}\n")
                            f.write(f"  Key weight max: {k_weight_max:.6f}\n")

                            # Critical warnings
                            if q_bias_max > 1.0 or k_bias_max > 1.0:
                                critical_warnings.append(f"Gate {gate_idx}: Bias explosion (Q:{q_bias_max:.2f}, K:{k_bias_max:.2f})")
                            if temp > 20.0:
                                critical_warnings.append(f"Gate {gate_idx}: Temperature very high ({temp:.1f})")

                        # Check final gate
                        if hasattr(refiner, 'final_gate'):
                            gate = refiner.final_gate
                            temp = gate.temperature.item()
                            q_bias_max = gate.query_proj.bias.abs().max().item()
                            k_bias_max = gate.key_proj.bias.abs().max().item()

                            f.write(f"\nFinal Gate:\n")
                            f.write(f"  Temperature: {temp:.6f}\n")
                            f.write(f"  Query bias max: {q_bias_max:.6f}\n")
                            f.write(f"  Key bias max: {k_bias_max:.6f}\n")

                            if q_bias_max > 1.0 or k_bias_max > 1.0:
                                critical_warnings.append(f"Final Gate: Bias explosion (Q:{q_bias_max:.2f}, K:{k_bias_max:.2f})")

                        # FFN LayerNorm weights
                        f.write(f"\nFFN LayerNorm Weights:\n")
                        for block_idx in range(len(refiner.blocks)):
                            ln_weight = refiner.blocks[block_idx]['ffn_layer_norm'].weight
                            ln_mean = ln_weight.mean().item()
                            ln_max = ln_weight.max().item()
                            f.write(f"  Block {block_idx}: Mean={ln_mean:.6f}, Max={ln_max:.6f}\n")

                        # Print critical warnings to console
                        if critical_warnings:
                            f.write(f"\n⚠️  CRITICAL WARNINGS:\n")
                            for warning in critical_warnings:
                                f.write(f"  - {warning}\n")

                            # Also print to console
                            tqdm.write(f"\n⚠️  CRITICAL at Epoch {epoch}, Step {step}:")
                            for warning in critical_warnings:
                                tqdm.write(f"  - {warning}")
                            tqdm.write(f"  (Full log: {debug_log_path})\n")

            # 💾 STEP CHECKPOINT: Save every 300 steps (debug mode only)
            if cfg and getattr(cfg, 'debug_mode', False) and checkpoint_dir and step % 300 == 0:
                checkpoint = {
                    "epoch": epoch,
                    "step": step,
                    "expert_name": expert_name,
                    "task": task_name,
                    "expert_state_dict": model.experts[expert_name].state_dict(),
                    "task_head_state_dict": task_heads.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": total_loss / step,
                }
                filename = f"{expert_name}_{task_name}_e{epoch}_s{step}.pt"
                save_path = os.path.join(checkpoint_dir, filename)
                torch.save(checkpoint, save_path)
                file_size = os.path.getsize(save_path) / (1024 * 1024)
                tqdm.write(f"💾 Step checkpoint: {filename} ({file_size:.1f} MB)")

            # Update progress bar with step-wise metrics
            avg_step_losses = [sl / step for sl in step_losses_accum]
            avg_step_accs = [(sa / step) * 100 for sa in step_accs_accum]

            losses_str = ','.join([f'{l:.4f}' for l in avg_step_losses])
            accs_str = ','.join([f'{a:.4f}' for a in avg_step_accs])

            # Get temperature from first mini-gate if using gated refiner
            expert = model.experts[expert_name]
            if hasattr(expert.delta_module.refiner, 'mini_gates'):
                first_gate = expert.delta_module.refiner.mini_gates[0]
                # QueryKeyGate has temperature parameter
                if hasattr(first_gate, 'temperature'):
                    temp = first_gate.temperature.item()
                    pbar.set_postfix_str(f"L:[{losses_str}] A:[{accs_str}] T:{temp:.2f}")
                else:
                    pbar.set_postfix_str(f"L:[{losses_str}] A:[{accs_str}]")
            else:
                # SimpleDeltaRefiner has single gate
                if hasattr(expert.delta_module.refiner, 'gate') and hasattr(expert.delta_module.refiner.gate, 'temperature'):
                    temp = expert.delta_module.refiner.gate.temperature.item()
                    pbar.set_postfix_str(f"L:[{losses_str}] A:[{accs_str}] T:{temp:.2f}")
                else:
                    pbar.set_postfix_str(f"L:[{losses_str}] A:[{accs_str}]")

        # Close progress bar
        pbar.close()

        # Handle remaining gradients if epoch didn't end on accumulation boundary
        # (e.g., if total_steps=102 and grad_accum=4, steps 101-102 need to be processed)
        if len(train_dataloader) % gradient_accumulation_steps != 0:
            print(f"\n   ℹ️  Processing remaining {len(train_dataloader) % gradient_accumulation_steps} accumulated gradients...")
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(task_heads.parameters()),
                gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        # Calculate training metrics (PNN style)
        train_loss = total_loss / len(train_dataloader)
        avg_step_losses = [sl / len(train_dataloader) for sl in step_losses_accum]
        avg_step_accs = [sa / len(train_dataloader) for sa in step_accs_accum]
        train_acc = avg_step_accs[-1] * 100 if avg_step_accs else 0  # Use final step accuracy

        # Validation
        model.eval()
        task_heads.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(
                val_dataloader,
                desc=f"E{epoch}/{num_epochs} [Val]",
                ncols=150,  # Increased to prevent truncation
                leave=False,
            )

            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward (no need for all steps during validation)
                expert = model.experts[expert_name]
                hidden = expert(input_ids, attention_mask, return_all_steps=False)
                result = task_heads(hidden, task_name, labels=labels)

                val_loss += result["loss"].item()
                logits = result.get("logits")

                if logits is not None and labels is not None:
                    # Skip accuracy for regression tasks (STS)
                    if task_name == "sts":
                        pass
                    else:
                        preds = logits.argmax(dim=-1)
                        mask = labels != -100
                        val_correct += ((preds == labels) & mask).sum().item()
                        val_total += mask.sum().item()

            val_pbar.close()

        val_loss = val_loss / len(val_dataloader)
        val_acc = (val_correct / val_total * 100) if val_total > 0 else 0

        # Print epoch summary (PNN style with step-wise metrics)
        epoch_time = time.time() - epoch_start
        print(f"\n   {'='*70}")
        print(f"   Epoch {epoch}/{num_epochs} Summary:")
        print(f"   {'='*70}")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Step Losses: {[f'{l:.4f}' for l in avg_step_losses]}")
        print(f"   Step Accs:   {[f'{a:.4f}' for a in avg_step_accs]}")
        print(f"   Train Acc:  {train_acc:.2f}% (final step)")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   Val Acc:    {val_acc:.2f}%")
        print(f"   Time:       {epoch_time:.1f}s ({epoch_time/60:.1f}m)")

        # Check if best accuracy
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"   🏆 New best accuracy: {best_val_acc:.2f}%")

        print(f"   {'='*70}\n")

        # Store epoch metrics (PNN style with step-wise info)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "step_losses": avg_step_losses,
            "step_accs": [a * 100 for a in avg_step_accs],
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "time": epoch_time,
        }
        training_history.append(epoch_metrics)

        # Save checkpoint after each epoch (efficient storage)
        if checkpoint_dir is not None:
            # Simplified checkpoint naming without presets
            # Format: {expert}_{task}
            # Example: lexical_mlm_final.pt
            stage_key = f"{expert_name}_{task_name}"

            if epoch == num_epochs:
                # Final epoch: Save expert + shared_embeddings + task_heads (~44MB)
                checkpoint = {
                    "phase": 1,
                    "expert_name": expert_name,
                    "task": task_name,
                    "epoch": epoch,
                    "is_final": True,
                    "expert_state_dict": model.experts[expert_name].state_dict(),
                    "shared_embeddings_state_dict": model.shared_embeddings.state_dict(),
                    "task_heads_state_dict": task_heads.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "metrics": epoch_metrics,
                    "training_history": training_history,
                    "config": cfg.to_dict(),  # Save full config for reproducibility
                }
                filename = f"{stage_key}_final.pt"
                save_path = os.path.join(checkpoint_dir, filename)
                torch.save(checkpoint, save_path)
                file_size = os.path.getsize(save_path) / (1024 * 1024)
                print(f"💾 Checkpoint saved: {filename} ({file_size:.1f} MB)")
            else:
                # Intermediate epoch: Save full checkpoint for resume capability
                checkpoint = {
                    "phase": 1,
                    "expert_name": expert_name,
                    "task": task_name,
                    "epoch": epoch,
                    "is_final": False,
                    "expert_state_dict": model.experts[expert_name].state_dict(),
                    "shared_embeddings_state_dict": model.shared_embeddings.state_dict(),
                    "task_heads_state_dict": task_heads.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "metrics": epoch_metrics,
                    "training_history": training_history,
                    "config": cfg.to_dict() if cfg else {},  # Save full config for reproducibility
                }
                filename = f"{stage_key}_epoch{epoch}.pt"
                save_path = os.path.join(checkpoint_dir, filename)
                torch.save(checkpoint, save_path)
                file_size = os.path.getsize(save_path) / (1024 * 1024)
                print(f"💾 Checkpoint saved: {filename} ({file_size:.1f} MB)")

    return {"loss": train_loss, "accuracy": train_acc, "history": training_history}


def train_phase1_curriculum(
    cfg: DAWNConfig,
    device: str = "cuda",
):
    """
    Run full Phase 1 curriculum using DAWN model

    Creates single DAWN instance with:
    - All experts (shared embeddings)
    - No integrator (Phase 1 mode)
    - No peer prediction

    For each stage:
    - Activates (unfreezes) target expert
    - Trains on task
    - Saves checkpoint
    - Deactivates (freezes) expert

    Args:
        cfg: RuntimeConfig instance with all configuration
        device: Device to train on
    """
    curriculum = PHASE1_CURRICULUM

    print("\n" + "=" * 70)
    print("PHASE 1: EXPERT SPECIALIZATION")
    print("=" * 70)
    print(f"Total stages: {len(curriculum)}")
    print(f"Device: {device}")
    print(f"Checkpoint directory: {cfg.training.checkpoint_dir}")
    print("=" * 70 + "\n")

    # Create checkpoint directory
    checkpoint_dir = cfg.training.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create DAWN model (Phase 1 mode: all experts, shared embeddings)
    print("🔧 Creating DAWN model (Phase 1 mode)...")
    dawn_args = cfg.get_model_kwargs()
    model = DAWN(**dawn_args)
    model = model.to(device)

    # Print model info
    ModelFactory._print_model_info(model, cfg, phase=1)

    # Check parameter initialization (debug)
    print("\n🔍 Checking parameter initialization...")
    nan_params = []
    large_params = []
    zero_params = []

    for name, param in model.named_parameters():
        # Skip empty parameters (e.g., peer_blend_weights when max_sources=1)
        if param.numel() == 0:
            continue

        if torch.isnan(param).any():
            nan_params.append(name)
        if torch.isinf(param).any():
            nan_params.append(f"{name} (inf)")
        max_val = param.abs().max().item()
        if max_val > 1.0 and 'temperature' not in name:
            large_params.append((name, max_val))
        if max_val == 0.0:
            zero_params.append(name)

    if nan_params:
        print(f"   ❌ Found {len(nan_params)} parameters with NaN/Inf:")
        for name in nan_params[:5]:
            print(f"      • {name}")
    else:
        print(f"   ✅ No NaN/Inf in parameters")

    if large_params:
        print(f"   ⚠️  Found {len(large_params)} parameters with large values:")
        for name, val in large_params[:5]:
            print(f"      • {name}: max={val:.4f}")

    if zero_params:
        print(f"   ℹ️  Found {len(zero_params)} zero-initialized parameters:")
        for name in zero_params[:5]:
            print(f"      • {name}")
        if len(zero_params) > 5:
            print(f"      ... and {len(zero_params) - 5} more")

    # Check DeltaRefiner small-init (changed from zero-init for stability)
    print("\n🔍 Checking DeltaRefiner small-init...")
    for expert_name, expert in model.experts.items():
        delta_refiner = expert.delta_module.refiner
        for i, block in enumerate(delta_refiner.blocks):
            final_linear = block['ffn'][3]
            weight_mean = final_linear.weight.abs().mean().item()
            bias_mean = final_linear.bias.abs().mean().item()
            # Small init should have mean around 0.008 (std=0.01)
            if weight_mean < 0.02 and bias_mean < 0.02:
                print(f"   ✅ Expert '{expert_name}' block {i}: small-init (w={weight_mean:.4f}, b={bias_mean:.4f})")
            else:
                print(f"   ⚠️  Expert '{expert_name}' block {i}: unexpected init (w={weight_mean:.4f}, b={bias_mean:.4f})")

    # Create TaskHeads (shared across all training)
    task_heads = TaskHeads(
        hidden_size=cfg.model.hidden_size,
        vocab_size=cfg.model.vocab_size,
    )
    task_heads = task_heads.to(device)
    print(f"✅ TaskHeads initialized\n")

    # Track completed stages and training histories
    completed_stages = set()
    all_training_history = {}  # {stage_key: [epoch_metrics]}

    # Process each stage
    for stage_idx, stage in enumerate(curriculum):
        expert_name = stage.expert_name
        task_name = stage.task_name
        # Use training preset epochs (overrides curriculum)
        num_epochs = cfg.training.get_epochs_for_task(task_name)
        stage_name = stage.name

        print(f"\n{'━'*70}")
        print(f"STAGE {stage_idx + 1}/{len(curriculum)}: {stage_name}")
        print(f"{'━'*70}")
        print(f"Expert: {expert_name}")
        print(f"Task: {task_name}")
        print(f"Epochs: {num_epochs} (customizable via config)")
        print(f"{'━'*70}\n")

        # Simplified stage key without presets
        stage_key = f"{expert_name}_{task_name}"

        # Check for existing checkpoint to resume from
        checkpoint_path = None
        start_epoch = 1
        training_history = []

        print(f"🔍 Searching for checkpoint: {stage_key}")
        print(f"   Checkpoint directory: {checkpoint_dir}")

        # List what's actually in the checkpoint directory
        if os.path.exists(checkpoint_dir):
            existing_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if existing_files:
                print(f"   📁 Found {len(existing_files)} checkpoint files in directory:")
                for f in sorted(existing_files)[:10]:  # Show first 10
                    print(f"      • {f}")
                if len(existing_files) > 10:
                    print(f"      ... and {len(existing_files) - 10} more")
            else:
                print(f"   📁 Checkpoint directory is empty")
        else:
            print(f"   ⚠️  Checkpoint directory does not exist yet")

        # Strategy 1: Check final checkpoint
        final_checkpoint_path = os.path.join(checkpoint_dir, f"{stage_key}_final.pt")
        print(f"   Strategy 1: Checking {os.path.basename(final_checkpoint_path)}... ", end="")
        if os.path.exists(final_checkpoint_path):
            checkpoint_path = final_checkpoint_path
            print("✅ FOUND")
        else:
            print("❌ not found")

        # Strategy 2: Check epoch checkpoints
        if not checkpoint_path:
            print(f"   Strategy 2: Checking epoch checkpoints...")
            for epoch_num in range(num_epochs, 0, -1):
                intermediate_path = os.path.join(checkpoint_dir, f"{stage_key}_epoch{epoch_num}.pt")
                if os.path.exists(intermediate_path):
                    checkpoint_path = intermediate_path
                    print(f"      ✅ FOUND: {os.path.basename(intermediate_path)}")
                    break
            if not checkpoint_path:
                print(f"      ❌ No epoch checkpoints found")

        # Strategy 3: Look for pretrained checkpoint (e.g., from PNN conversion)
        if not checkpoint_path and os.path.exists(checkpoint_dir):
            pretrained_path = os.path.join(checkpoint_dir, f"{expert_name}_{task_name}_pretrained.pt")
            print(f"   Strategy 3: Checking for pretrained checkpoint: {expert_name}_{task_name}_pretrained.pt... ", end="")
            if os.path.exists(pretrained_path):
                checkpoint_path = pretrained_path
                print("✅ FOUND")
                print(f"💡 Found pretrained checkpoint (e.g., from PNN conversion)")
            else:
                print("❌ not found")

        # Strategy 4: Fallback - search for legacy preset-based checkpoints
        if not checkpoint_path and os.path.exists(checkpoint_dir):
            import glob
            # Look for pattern: *_*_{expert_name}_{task_name}_*.pt (legacy format)
            pattern = os.path.join(checkpoint_dir, f"*_*_{expert_name}_{task_name}_*.pt")
            print(f"   Strategy 4: Searching for legacy preset-based checkpoints...")
            matching_files = glob.glob(pattern)

            if matching_files:
                print(f"      Found {len(matching_files)} legacy checkpoint(s):")
                for f in matching_files:
                    print(f"      - {os.path.basename(f)}")

                # Prefer final checkpoints over epoch checkpoints
                final_files = [f for f in matching_files if f.endswith("_final.pt")]
                if final_files:
                    checkpoint_path = final_files[0]  # Take first final checkpoint
                    print(f"      ✅ SELECTED: {os.path.basename(checkpoint_path)} (legacy format)")
                else:
                    # Find highest epoch number
                    checkpoint_path = max(matching_files, key=os.path.getmtime)
                    print(f"      ✅ SELECTED: {os.path.basename(checkpoint_path)} (most recent)")

                print(f"💡 Loading legacy checkpoint - will save in new format")
            else:
                print(f"      ❌ No legacy checkpoints found")

        if not os.path.exists(checkpoint_dir):
            print(f"   ⚠️  Checkpoint directory does not exist: {checkpoint_dir}")

        if not checkpoint_path:
            print(f"   ❓ No checkpoint found - starting from scratch\n")

        if checkpoint_path:
            print(f"📂 Found checkpoint: {os.path.basename(checkpoint_path)}")

            # Retry loading with exponential backoff (for network filesystem issues)
            max_retries = 3
            checkpoint = None
            for attempt in range(max_retries):
                try:
                    checkpoint = torch.load(
                        checkpoint_path,
                        map_location=device,
                        weights_only=False  # Required for loading model state dicts
                    )
                    break
                except (OSError, RuntimeError) as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        print(f"⚠️  Load attempt {attempt + 1} failed: {e}")
                        print(f"   Retrying in {wait_time}s...")
                        import time
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Failed to load checkpoint after {max_retries} attempts: {e}")
                        print(f"   Skipping checkpoint and starting from scratch")
                        checkpoint = None
                        break

            if checkpoint is None:
                # Failed to load, treat as if no checkpoint exists
                checkpoint_path = None

        if checkpoint_path and checkpoint:
            # Check if already completed
            if checkpoint.get("is_final", False) and checkpoint.get("epoch", 0) >= num_epochs:
                print(f"✅ Stage already completed (epoch {checkpoint['epoch']}/{num_epochs})")
                completed_stages.add(stage_key)

                # Load model state (new efficient format)
                if "expert_state_dict" in checkpoint:
                    # Check if this is a pretrained checkpoint (from PNN conversion)
                    is_pretrained = '_pretrained.pt' in checkpoint_path
                    if is_pretrained:
                        print(f"💡 PRETRAINED CHECKPOINT DETECTED")
                        print(f"   Loading full expert state (PNN → DAWN)")

                    # Load full expert state (PNN structure = DAWN Expert structure)
                    try:
                        expert_state = checkpoint["expert_state_dict"]

                        # Handle position embeddings size mismatch (e.g., PNN 128 vs DAWN 512)
                        pos_emb_key = 'position_embeddings.weight'
                        if pos_emb_key in expert_state:
                            ckpt_pos_emb = expert_state[pos_emb_key]
                            model_pos_emb = model.experts[expert_name].position_embeddings.weight

                            if ckpt_pos_emb.shape != model_pos_emb.shape:
                                print(f"⚠️  Position embeddings size mismatch:")
                                print(f"   Checkpoint: {ckpt_pos_emb.shape} | Model: {model_pos_emb.shape}")

                                # Tile to fill (safer than partial copy)
                                import math
                                num_repeats = math.ceil(model_pos_emb.shape[0] / ckpt_pos_emb.shape[0])
                                repeated = ckpt_pos_emb.repeat(num_repeats, 1)
                                model_pos_emb.data.copy_(repeated[:model_pos_emb.shape[0]])
                                print(f"   ✅ Tiled position embeddings: {ckpt_pos_emb.shape[0]} → {model_pos_emb.shape[0]}")

                                # Remove from state_dict to avoid error
                                expert_state = {k: v for k, v in expert_state.items() if k != pos_emb_key}

                        # Load remaining state
                        missing_keys, unexpected_keys = model.experts[expert_name].load_state_dict(expert_state, strict=False)
                        print(f"✅ Loaded expert '{expert_name}' state")
                        if missing_keys:
                            print(f"   ⚠️  Missing keys: {len(missing_keys)} (will use random init)")
                        if unexpected_keys:
                            print(f"   ⚠️  Unexpected keys: {len(unexpected_keys)} (ignored)")
                    except Exception as e:
                        print(f"❌ Failed to load expert state: {type(e).__name__}: {e}")
                        raise

                    # Load shared embeddings
                    if "shared_embeddings_state_dict" in checkpoint:
                        try:
                            shared_emb_state = checkpoint["shared_embeddings_state_dict"]

                            # Handle position embeddings size mismatch in shared embeddings
                            pos_key = 'position.weight'
                            if pos_key in shared_emb_state:
                                ckpt_pos = shared_emb_state[pos_key]
                                model_pos = model.shared_embeddings['position'].weight

                                if ckpt_pos.shape != model_pos.shape:
                                    print(f"⚠️  Shared position embeddings size mismatch:")
                                    print(f"   Checkpoint: {ckpt_pos.shape} | Model: {model_pos.shape}")

                                    # Tile to fill (safer than partial copy + random init)
                                    import math
                                    num_repeats = math.ceil(model_pos.shape[0] / ckpt_pos.shape[0])
                                    repeated = ckpt_pos.repeat(num_repeats, 1)
                                    model_pos.data.copy_(repeated[:model_pos.shape[0]])
                                    print(f"   ✅ Tiled shared position embeddings: {ckpt_pos.shape[0]} → {model_pos.shape[0]}")
                                    print(f"   ℹ️  All positions now have learned values (no random init)")

                                    # Remove from state_dict to avoid error
                                    shared_emb_state = {k: v for k, v in shared_emb_state.items() if k != pos_key}

                            # Load remaining shared embeddings
                            model.shared_embeddings.load_state_dict(shared_emb_state, strict=False)
                            print(f"✅ Loaded shared embeddings state")
                        except Exception as e:
                            print(f"❌ Failed to load shared embeddings: {type(e).__name__}: {e}")
                            raise
                    else:
                        # Pretrained checkpoint: Extract embeddings from expert_state_dict
                        print(f"💡 No shared_embeddings_state_dict found - extracting from expert")
                        try:
                            expert_state = checkpoint["expert_state_dict"]
                            embedding_keys = {
                                'token': 'token_embeddings.weight',
                                'position': 'position_embeddings.weight',
                                'layer_norm': 'embedding_layer_norm',
                            }

                            for emb_name, key_prefix in embedding_keys.items():
                                if emb_name in ['token', 'position']:
                                    key = key_prefix
                                    if key in expert_state:
                                        ckpt_emb = expert_state[key]
                                        model_emb = model.shared_embeddings[emb_name].weight

                                        # Handle size mismatch
                                        if ckpt_emb.shape != model_emb.shape:
                                            print(f"   ⚠️  {emb_name} size mismatch: {ckpt_emb.shape} vs {model_emb.shape}")
                                            if emb_name == 'position':
                                                # Copy what we can
                                                min_len = min(ckpt_emb.shape[0], model_emb.shape[0])
                                                model_emb.data[:min_len].copy_(ckpt_emb[:min_len])
                                                print(f"   ✅ Extracted {emb_name} embeddings (first {min_len} positions)")
                                            else:
                                                print(f"   ❌ Skipping {emb_name} embeddings due to size mismatch")
                                        else:
                                            model_emb.data.copy_(ckpt_emb)
                                            print(f"   ✅ Extracted {emb_name} embeddings")
                                else:  # layer_norm
                                    for param_name in ['weight', 'bias']:
                                        key = f"{key_prefix}.{param_name}"
                                        if key in expert_state:
                                            getattr(model.shared_embeddings[emb_name], param_name).data.copy_(expert_state[key])
                                    print(f"   ✅ Extracted embedding layer_norm")

                            # Dropout is not a parameter, skip
                            print(f"✅ Extracted shared embeddings from expert state")
                        except Exception as e:
                            print(f"⚠️  Failed to extract embeddings: {type(e).__name__}: {e}")
                            print(f"⚠️  Will use randomly initialized shared embeddings")

                    # Try to load task_heads, but handle missing heads gracefully
                    if "task_heads_state_dict" in checkpoint:
                        try:
                            task_heads.load_state_dict(checkpoint["task_heads_state_dict"], strict=False)
                            print(f"✅ Loaded task heads state (non-strict)")
                        except RuntimeError as e:
                            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                                print(f"⚠️  Task heads structure changed: {e}")
                                print(f"⚠️  Task heads will be re-initialized for this task")
                            else:
                                print(f"❌ Failed to load task_heads: {type(e).__name__}: {e}")
                                raise
                        except Exception as e:
                            print(f"❌ Unexpected error loading task_heads: {type(e).__name__}: {e}")
                            raise
                    else:
                        print(f"💡 No task_heads_state_dict found - will use fresh task heads")
                else:
                    # Old format: Load full model state dict
                    try:
                        model.load_state_dict(checkpoint["model_state_dict"])
                        print(f"✅ Loaded legacy checkpoint (full model)")
                    except Exception as e:
                        print(f"❌ Failed to load legacy model state: {type(e).__name__}: {e}")
                        raise

                    try:
                        task_heads.load_state_dict(checkpoint["task_heads_state_dict"], strict=False)
                        print(f"✅ Loaded task heads state (non-strict)")
                    except RuntimeError as e:
                        if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                            print(f"⚠️  Task heads structure changed: {e}")
                            print(f"⚠️  Task heads will be re-initialized for this task")
                        else:
                            print(f"❌ Failed to load task_heads: {type(e).__name__}: {e}")
                            raise
                    except Exception as e:
                        print(f"❌ Unexpected error loading task_heads: {type(e).__name__}: {e}")
                        raise

                # Store history
                if "training_history" in checkpoint:
                    all_training_history[stage_key] = checkpoint["training_history"]

                print(f"⏭️  Skipping to next stage\n")
                continue
            else:
                # Resume from checkpoint
                start_epoch = checkpoint.get("epoch", 0) + 1
                if start_epoch > num_epochs:
                    start_epoch = 1

                print(f"🔄 Resuming from epoch {start_epoch}/{num_epochs}")

                # Load states (new efficient format)
                if "expert_state_dict" in checkpoint:
                    try:
                        expert_state = checkpoint["expert_state_dict"]

                        # Handle position embeddings size mismatch (e.g., PNN 128 vs DAWN 512)
                        pos_emb_key = 'position_embeddings.weight'
                        if pos_emb_key in expert_state:
                            ckpt_pos_emb = expert_state[pos_emb_key]
                            model_pos_emb = model.experts[expert_name].position_embeddings.weight

                            if ckpt_pos_emb.shape != model_pos_emb.shape:
                                print(f"⚠️  Position embeddings size mismatch:")
                                print(f"   Checkpoint: {ckpt_pos_emb.shape} | Model: {model_pos_emb.shape}")

                                # Tile to fill (safer than partial copy + random init)
                                import math
                                num_repeats = math.ceil(model_pos_emb.shape[0] / ckpt_pos_emb.shape[0])
                                repeated = ckpt_pos_emb.repeat(num_repeats, 1)
                                model_pos_emb.data.copy_(repeated[:model_pos_emb.shape[0]])
                                print(f"   ✅ Tiled position embeddings: {ckpt_pos_emb.shape[0]} → {model_pos_emb.shape[0]}")
                                print(f"   ℹ️  All positions now have learned values (no random init)")

                                # Remove from state_dict to avoid error
                                expert_state = {k: v for k, v in expert_state.items() if k != pos_emb_key}

                        # Load remaining state
                        model.experts[expert_name].load_state_dict(expert_state, strict=False)
                        print(f"✅ Loaded expert '{expert_name}' state")
                    except Exception as e:
                        print(f"❌ Failed to load expert state: {type(e).__name__}: {e}")
                        raise

                    try:
                        shared_emb_state = checkpoint["shared_embeddings_state_dict"]

                        # Handle position embeddings size mismatch in shared embeddings
                        pos_key = 'position.weight'
                        if pos_key in shared_emb_state:
                            ckpt_pos = shared_emb_state[pos_key]
                            model_pos = model.shared_embeddings['position'].weight

                            if ckpt_pos.shape != model_pos.shape:
                                print(f"⚠️  Shared position embeddings size mismatch:")
                                print(f"   Checkpoint: {ckpt_pos.shape} | Model: {model_pos.shape}")

                                # Tile to fill (safer than partial copy + random init)
                                import math
                                num_repeats = math.ceil(model_pos.shape[0] / ckpt_pos.shape[0])
                                repeated = ckpt_pos.repeat(num_repeats, 1)
                                model_pos.data.copy_(repeated[:model_pos.shape[0]])
                                print(f"   ✅ Tiled shared position embeddings: {ckpt_pos.shape[0]} → {model_pos.shape[0]}")
                                print(f"   ℹ️  All positions now have learned values (no random init)")

                                # Remove from state_dict to avoid error
                                shared_emb_state = {k: v for k, v in shared_emb_state.items() if k != pos_key}

                        # Load remaining shared embeddings
                        model.shared_embeddings.load_state_dict(shared_emb_state, strict=False)
                        print(f"✅ Loaded shared embeddings state")
                    except Exception as e:
                        print(f"❌ Failed to load shared embeddings: {type(e).__name__}: {e}")
                        raise

                    # Try to load task_heads, but handle missing heads gracefully
                    if "task_heads_state_dict" in checkpoint:
                        try:
                            task_heads.load_state_dict(checkpoint["task_heads_state_dict"], strict=False)
                            print(f"✅ Loaded task heads state (non-strict)")
                        except RuntimeError as e:
                            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                                print(f"⚠️  Task heads structure changed: {e}")
                                print(f"⚠️  Task heads will be re-initialized for this task")
                            else:
                                print(f"❌ Failed to load task_heads: {type(e).__name__}: {e}")
                                raise
                        except Exception as e:
                            print(f"❌ Unexpected error loading task_heads: {type(e).__name__}: {e}")
                            raise
                    else:
                        print(f"💡 No task_heads_state_dict found - will use fresh task heads")
                else:
                    # Old format: Load full model state dict
                    try:
                        model.load_state_dict(checkpoint["model_state_dict"])
                        print(f"✅ Loaded legacy checkpoint (full model)")
                    except Exception as e:
                        print(f"❌ Failed to load legacy model state: {type(e).__name__}: {e}")
                        raise

                    try:
                        task_heads.load_state_dict(checkpoint["task_heads_state_dict"], strict=False)
                        print(f"✅ Loaded task heads state (non-strict)")
                    except RuntimeError as e:
                        if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                            print(f"⚠️  Task heads structure changed: {e}")
                            print(f"⚠️  Task heads will be re-initialized for this task")
                        else:
                            print(f"❌ Failed to load task_heads: {type(e).__name__}: {e}")
                            raise
                    except Exception as e:
                        print(f"❌ Unexpected error loading task_heads: {type(e).__name__}: {e}")
                        raise

                # Load training history
                if "training_history" in checkpoint:
                    training_history = checkpoint["training_history"]
                    print(f"📊 Loaded training history: {len(training_history)} epochs")

        if stage_key in completed_stages:
            continue

        # Set active expert (no freezing needed - forward handles it)
        print(f"🎯 Training expert: {expert_name}")
        expert = model.experts[expert_name]

        # Get task configuration
        if task_name not in cfg.tasks:
            raise ValueError(f"Task '{task_name}' not found in configuration")

        task_config = cfg.tasks[task_name]
        task_specific_config = task_config.task_params
        max_samples = task_config.max_samples

        # Create train and validation dataloaders
        print(f"📊 Loading {task_name} dataset...")
        dataloader_config_dict = {
            "num_workers": cfg.dataloader.num_workers,
            "pin_memory": cfg.dataloader.pin_memory,
            "prefetch_factor": cfg.dataloader.prefetch_factor,
            "persistent_workers": cfg.dataloader.persistent_workers,
        }

        train_dataloader = get_dataloader(
            task=task_name,
            split="train",
            batch_size=cfg.hardware.batch_size,
            task_config=task_specific_config,
            max_samples=max_samples,
            verbose=True,
            **dataloader_config_dict,
        )

        # Validation set (fallback to train if validation split doesn't exist)
        val_max_samples = min(10000, max_samples // 10) if max_samples else 10000
        try:
            val_dataloader = get_dataloader(
                task=task_name,
                split="validation",
                batch_size=cfg.hardware.batch_size,
                task_config=task_specific_config,
                max_samples=val_max_samples,
                verbose=False,
                **dataloader_config_dict,
            )
        except Exception as e:
            print(f"⚠️  Validation split not available, using train split for validation")
            val_dataloader = get_dataloader(
                task=task_name,
                split="train",
                batch_size=cfg.hardware.batch_size,
                task_config=task_specific_config,
                max_samples=val_max_samples,
                verbose=False,
                **dataloader_config_dict,
            )

        # Calculate total steps for scheduler
        total_steps = len(train_dataloader) * num_epochs

        # Setup optimizer with parameter groups (lower LR for embeddings)
        embedding_params = []
        model_params = []

        for name, param in model.named_parameters():
            if "shared_embeddings" in name:
                embedding_params.append(param)
            else:
                model_params.append(param)

        task_head_params = list(task_heads.parameters())

        optimizer = AdamW(
            [
                {"params": embedding_params, "lr": cfg.training.embedding_lr},
                {"params": model_params, "lr": cfg.training.base_lr},
                {"params": task_head_params, "lr": cfg.training.base_lr},
            ],
            weight_decay=cfg.training.weight_decay,
        )

        # Setup scheduler with warmup
        warmup_ratio = cfg.training.warmup_ratio

        # Extended warmup for pretrained checkpoints (more stable)
        if checkpoint_path and '_pretrained.pt' in checkpoint_path:
            warmup_ratio = min(0.2, warmup_ratio * 2)  # Double warmup, max 20%
            print(f"💡 Using extended warmup ({warmup_ratio*100:.0f}%) for pretrained checkpoint")

        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed precision scaler
        scaler = GradScaler('cuda')

        # Resume optimizer/scheduler states if continuing from checkpoint
        if start_epoch > 1 and checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    print(f"✅ Restored optimizer state")
                except ValueError as e:
                    # Parameter groups changed - this is expected when architecture changes
                    print(f"⚠️  Optimizer state incompatible (parameter groups changed): {e}")
                    print(f"⚠️  Starting with fresh optimizer state")
                except Exception as e:
                    print(f"❌ Unexpected error restoring optimizer: {type(e).__name__}: {e}")
                    print(f"⚠️  Starting with fresh optimizer state")

            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    print(f"✅ Restored scheduler state")
                except Exception as e:
                    # Scheduler state incompatible - less critical, can continue with fresh scheduler
                    print(f"⚠️  Could not restore scheduler state: {type(e).__name__}: {e}")
                    print(f"⚠️  Starting with fresh scheduler state")

        print(f"🎓 Optimizer: AdamW (base_lr={cfg.training.base_lr}, embedding_lr={cfg.training.embedding_lr})")
        print(f"📈 Scheduler: Linear warmup ({warmup_steps} steps) + decay")
        print(f"⚡ Mixed precision: Enabled\n")

        # Train on this task
        metrics = train_single_task(
            model=model,
            task_heads=task_heads,
            expert_name=expert_name,
            task_name=task_name,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            num_epochs=num_epochs,
            gradient_clip=cfg.training.gradient_clip,
            gradient_accumulation_steps=cfg.hardware.gradient_accumulation_steps,
            device=device,
            start_epoch=start_epoch,
            training_history=training_history,
            checkpoint_dir=checkpoint_dir,
            cfg=cfg,  # Pass config
        )

        # Store history
        final_history = metrics.get("history", training_history)
        all_training_history[stage_key] = final_history

        print(f"\n✅ Stage {stage_idx + 1} complete!\n")

    # Save final checkpoint with all experts
    print(f"\n💾 Saving final Phase 1 checkpoint...")
    final_checkpoint = {
        "phase": 1,
        "model_state_dict": model.state_dict(),
        "task_heads_state_dict": task_heads.state_dict(),
        "expert_names": cfg.expert_names,
        "is_final": True,
        "all_training_history": all_training_history,
        "completed_stages": list(completed_stages),
    }
    final_path = os.path.join(checkpoint_dir, "phase1_final.pt")
    torch.save(final_checkpoint, final_path)
    file_size = os.path.getsize(final_path) / (1024 * 1024)
    print(f"✅ Final checkpoint saved: {final_path} ({file_size:.1f} MB)")
    print(f"📊 Total training history: {sum(len(h) for h in all_training_history.values())} epochs across {len(all_training_history)} stages")

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE! 🎉")
    print("=" * 70)
    print(f"\nAll experts have been trained independently.")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"\nFinal checkpoint: {final_path}")
    print("\n" + "=" * 70)
    print("Next step: Run Phase 2 training (train_phase2.py)")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="DAWN Phase 1 Training")

    # Basic options
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
        help="Override checkpoint directory",
    )

    # Model parameters (directly customizable) - defaults match PNN hierarchical structure
    parser.add_argument("--hidden_size", type=int, default=768, help="Model hidden dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=str, default="1024,1536,2048,1536,1024",
                        help="FFN intermediate sizes (mountain-shaped, PNN: 1024,1536,2048,1536,1024)")
    parser.add_argument("--num_steps", type=int, default=4, help="Number of refinement steps")

    # Training parameters (directly customizable) - Match PNN hierarchical success
    parser.add_argument("--base_lr", type=float, default=2e-4, help="Base learning rate (PNN hierarchical: 2e-4)")
    parser.add_argument("--embedding_lr", type=float, default=1e-4, help="Embedding learning rate (PNN hierarchical: 1e-4)")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping (PNN hierarchical: 1.0)")
    parser.add_argument("--default_epochs", type=int, default=6, help="Default epochs per task")

    # Hardware override
    parser.add_argument("--batch_size", type=int, default=None, help="Override auto-detected batch size")

    # Debug mode
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Enable debug mode (logs gate statistics to file)",
    )

    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate config without training",
    )

    args = parser.parse_args()

    # Parse intermediate_size (comma-separated string to list)
    if isinstance(args.intermediate_size, str):
        intermediate_size = [int(x.strip()) for x in args.intermediate_size.split(',')]
    else:
        intermediate_size = args.intermediate_size

    # Get Phase 1 configuration using new simplified system
    print("\n🔍 Building Phase 1 configuration...")
    cfg = DAWNConfig(
        phase=1,
        # Model parameters
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        intermediate_size=intermediate_size,
        num_steps=args.num_steps,
        # Training parameters
        learning_rate=args.base_lr,
        embedding_lr=args.embedding_lr,
        gradient_clip=args.gradient_clip,
        epochs=args.default_epochs,
        # Checkpoint directory
        checkpoint_dir=args.checkpoint_dir,
        # Debug mode
        debug_mode=args.debug_mode,
    )

    # Override batch size if specified
    if args.batch_size is not None:
        print(f"📦 Overriding batch size: {cfg.hardware.batch_size} -> {args.batch_size}")
        cfg.hardware.batch_size = args.batch_size

    # Show debug mode status
    if cfg.debug_mode:
        print(f"🔍 Debug mode: ENABLED")
        print(f"   Gate statistics will be logged to: {{checkpoint_dir}}/{{expert}}_{{task}}_debug.txt")
    else:
        print(f"🔍 Debug mode: DISABLED")

    # Print configuration summary
    cfg.print_summary()

    if args.validate_only:
        print("✅ Configuration is valid. Exiting (--validate_only).\n")
        return

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"🖥️  Using device: {args.device}\n")

    # Run Phase 1 curriculum
    train_phase1_curriculum(
        cfg=cfg,
        device=args.device,
    )


if __name__ == "__main__":
    main()
