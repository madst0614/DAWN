"""
Training Utilities for DAWN

Phase 1/2 공통 핵심 기능:
- Optimizer setup with parameter groups
- Mixed precision helpers
- Backward & gradient clipping
- Checkpointing
- Monitoring

Phase-specific training loop는 별도 구현
"""

import torch
import torch.nn as nn
import warnings
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
from datetime import datetime

# Suppress autocast deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")


class DAWNTrainer:
    """
    DAWN Trainer - Core utilities

    Provides common training infrastructure for Phase 1 and Phase 2.
    Phase-specific training loops should be implemented separately.
    """

    def __init__(self, model, task_heads, config, device="cuda", log_dir=None):
        """
        Args:
            model: DAWN instance
            task_heads: TaskHeads module
            config: Training config dict
            device: Device to use
            log_dir: Directory for logs and checkpoints
        """
        self.model = model.to(device)
        self.task_heads = task_heads.to(device)
        self.config = config
        self.device = device
        self.log_dir = log_dir or "./logs"

        # Training state
        self.optimizer = None
        self.scheduler = None

        # Mixed precision
        self.use_amp = config.get("mixed_precision", True)
        if self.use_amp:
            try:
                from torch.amp import GradScaler as NewGradScaler

                self.scaler = NewGradScaler("cuda")
            except ImportError:
                from torch.cuda.amp import GradScaler

                self.scaler = GradScaler()
        else:
            self.scaler = None

        # Monitoring
        self.global_step = 0
        self.current_epoch = 0

        os.makedirs(self.log_dir, exist_ok=True)

    # ========================================================================
    # OPTIMIZER & SCHEDULER SETUP
    # ========================================================================

    def setup_optimizer(
        self,
        base_lr=5e-5,
        embedding_lr=1e-5,
        total_steps=None,
        warmup_ratio=0.1,
        weight_decay=0.01,
    ):
        """
        Setup optimizer with parameter groups

        Parameter groups:
        - Embeddings: Lower LR (typically 1e-5)
        - Experts: Base LR (typically 5e-5)
        - TaskHeads: Base LR
        - Integrator: Base LR
        - Peer predictors: Base LR (Phase 2)

        Args:
            base_lr: Base learning rate for most parameters
            embedding_lr: Lower LR for embeddings
            total_steps: Total training steps (for scheduler)
            warmup_ratio: Warmup ratio
            weight_decay: Weight decay
        """
        param_groups = self._get_optimizer_param_groups(
            base_lr=base_lr, embedding_lr=embedding_lr, weight_decay=weight_decay
        )

        self.optimizer = self._create_optimizer(param_groups)

        if total_steps:
            self.scheduler = self._create_scheduler(
                total_steps=total_steps, warmup_ratio=warmup_ratio
            )

        # Log parameter groups
        print(f"\n{'='*70}")
        print("OPTIMIZER SETUP")
        print(f"{'='*70}")
        for i, group in enumerate(param_groups):
            num_params = sum(p.numel() for p in group["params"])
            print(
                f"Group {i}: {group['name']:<20} LR={group['lr']:.2e}  "
                f"Params={num_params:>12,}"
            )
        print(f"{'='*70}\n")

    def _get_optimizer_param_groups(self, base_lr, embedding_lr, weight_decay):
        """
        Organize parameters into groups with different learning rates

        Returns:
            List of parameter group dicts
        """
        # Collect parameters by category
        embedding_params = []
        expert_params = []
        task_head_params = []
        integrator_params = []
        peer_predictor_params = []

        # Model parameters
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "embedding" in name.lower():
                embedding_params.append(param)
            elif "experts." in name:
                if "peer_predictors" in name:
                    peer_predictor_params.append(param)
                else:
                    expert_params.append(param)
            elif "integrator" in name:
                integrator_params.append(param)

        # TaskHeads parameters
        for param in self.task_heads.parameters():
            if param.requires_grad:
                task_head_params.append(param)

        # Build parameter groups
        param_groups = []

        if embedding_params:
            param_groups.append(
                {
                    "params": embedding_params,
                    "lr": embedding_lr,
                    "weight_decay": weight_decay,
                    "name": "embeddings",
                }
            )

        if expert_params:
            param_groups.append(
                {
                    "params": expert_params,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                    "name": "experts",
                }
            )

        if peer_predictor_params:
            param_groups.append(
                {
                    "params": peer_predictor_params,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                    "name": "peer_predictors",
                }
            )

        if integrator_params:
            param_groups.append(
                {
                    "params": integrator_params,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                    "name": "integrator",
                }
            )

        if task_head_params:
            param_groups.append(
                {
                    "params": task_head_params,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                    "name": "task_heads",
                }
            )

        return param_groups

    def _create_optimizer(self, param_groups):
        """Create AdamW optimizer"""
        return AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    def _create_scheduler(self, total_steps, warmup_ratio):
        """Create linear warmup + decay scheduler"""
        warmup_steps = int(total_steps * warmup_ratio)
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    # ========================================================================
    # FORWARD & BACKWARD HELPERS
    # ========================================================================

    def forward_with_amp(self, fn, *args, **kwargs):
        """
        Execute forward pass with optional AMP

        Args:
            fn: Forward function to call
            *args, **kwargs: Arguments to forward function

        Returns:
            Output of forward function
        """
        if self.use_amp:
            with autocast():
                return fn(*args, **kwargs)
        else:
            return fn(*args, **kwargs)

    def backward_and_step(
        self, loss, accumulation_steps=1, gradient_clip=1.0, step_scheduler=True
    ):
        """
        Backward pass with gradient accumulation and clipping

        Args:
            loss: Loss tensor (pre-scaled by accumulation_steps)
            accumulation_steps: Number of accumulation steps
            gradient_clip: Max gradient norm
            step_scheduler: Whether to step scheduler
        """
        if self.use_amp:
            # Scale loss and backward
            self.scaler.scale(loss).backward()

            # Step every accumulation_steps
            if (self.global_step + 1) % accumulation_steps == 0:
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)

                # Clip gradients
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), gradient_clip
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.task_heads.parameters(), gradient_clip
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Scheduler step
                if step_scheduler and self.scheduler:
                    self.scheduler.step()
        else:
            # Standard backward
            loss.backward()

            if (self.global_step + 1) % accumulation_steps == 0:
                # Clip gradients
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), gradient_clip
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.task_heads.parameters(), gradient_clip
                    )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Scheduler step
                if step_scheduler and self.scheduler:
                    self.scheduler.step()

        self.global_step += 1

    # ========================================================================
    # CHECKPOINTING
    # ========================================================================

    def save_checkpoint(
        self, checkpoint_name, epoch=None, metrics=None, extra_state=None
    ):
        """
        Save training checkpoint

        Args:
            checkpoint_name: Name for checkpoint file
            epoch: Current epoch (optional)
            metrics: Training metrics dict (optional)
            extra_state: Additional state to save (optional)
        """
        checkpoint = {
            "epoch": epoch or self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "task_heads_state_dict": self.task_heads.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer else None
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "metrics": metrics or {},
        }

        # Add extra state if provided
        if extra_state:
            checkpoint.update(extra_state)

        checkpoint_path = os.path.join(self.log_dir, f"{checkpoint_name}.pt")
        torch.save(checkpoint, checkpoint_path)

        print(f"💾 Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, load_optimizer=True):
        """
        Load training checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state

        Returns:
            checkpoint: Full checkpoint dict
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model and task heads
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.task_heads.load_state_dict(checkpoint["task_heads_state_dict"])

        # Load optimizer and scheduler
        if load_optimizer:
            if self.optimizer and checkpoint.get("optimizer_state_dict"):
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.scheduler and checkpoint.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)

        print(f"✅ Checkpoint loaded: {checkpoint_path}")
        print(f"   Epoch: {self.current_epoch} | Step: {self.global_step}")

        return checkpoint

    def save_expert_checkpoint(self, expert_name, checkpoint_name, metrics=None):
        """
        Save individual expert checkpoint (for Phase 1)

        Args:
            expert_name: Name of expert to save
            checkpoint_name: Name for checkpoint file
            metrics: Training metrics (optional)
        """
        expert_state = self.model.experts[expert_name].state_dict()

        checkpoint = {
            "expert_name": expert_name,
            "expert_state_dict": expert_state,
            "task_heads_state_dict": self.task_heads.state_dict(),
            "metrics": metrics or {},
        }

        checkpoint_path = os.path.join(
            self.log_dir, f"phase1_{expert_name}_{checkpoint_name}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        print(f"💾 Expert checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    # ========================================================================
    # MONITORING
    # ========================================================================

    def print_trainable_summary(self):
        """Print summary of trainable parameters"""
        counts = {
            "embeddings": 0,
            "experts": {},
            "peer_predictors": 0,
            "integrator": 0,
            "task_heads": 0,
        }

        # Count model parameters
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "embedding" in name.lower():
                counts["embeddings"] += param.numel()
            elif "experts." in name:
                if "peer_predictors" in name:
                    counts["peer_predictors"] += param.numel()
                else:
                    # Extract expert name
                    parts = name.split(".")
                    if len(parts) > 1:
                        expert_name = parts[1]
                        if expert_name not in counts["experts"]:
                            counts["experts"][expert_name] = 0
                        counts["experts"][expert_name] += param.numel()
            elif "integrator" in name:
                counts["integrator"] += param.numel()

        # Count task head parameters
        for param in self.task_heads.parameters():
            if param.requires_grad:
                counts["task_heads"] += param.numel()

        # Calculate totals
        total_trainable = (
            counts["embeddings"]
            + counts["peer_predictors"]
            + counts["integrator"]
            + counts["task_heads"]
            + sum(counts["experts"].values())
        )

        total_params = sum(p.numel() for p in self.model.parameters()) + sum(
            p.numel() for p in self.task_heads.parameters()
        )

        # Print summary
        print(f"\n{'='*70}")
        print("TRAINABLE PARAMETERS SUMMARY")
        print(f"{'='*70}")

        if counts["embeddings"] > 0:
            print(f"Embeddings:        {counts['embeddings']:>12,}")

        for expert_name, count in sorted(counts["experts"].items()):
            if count > 0:
                print(f"Expert ({expert_name:10s}): {count:>12,}")

        if counts["peer_predictors"] > 0:
            print(f"Peer Predictors:   {counts['peer_predictors']:>12,}")

        if counts["integrator"] > 0:
            print(f"Integrator:        {counts['integrator']:>12,}")

        if counts["task_heads"] > 0:
            print(f"Task Heads:        {counts['task_heads']:>12,}")

        print(f"{'-'*70}")
        print(f"TOTAL TRAINABLE:   {total_trainable:>12,}")
        print(f"TOTAL MODEL:       {total_params:>12,}")
        print(f"TRAINABLE %:       {100*total_trainable/total_params:>11.2f}%")
        print(f"{'='*70}\n")

    def get_current_lr(self):
        """Get current learning rate from optimizer"""
        if self.optimizer:
            return self.optimizer.param_groups[0]["lr"]
        return None

    def print_model_info(self):
        """Print model architecture information"""
        model_info = self.model.get_model_info()

        print(f"\n{'='*70}")
        print("MODEL INFORMATION")
        print(f"{'='*70}")
        print(f"Phase:             {model_info.get('phase', 'N/A')}")
        print(f"Num Experts:       {model_info.get('num_experts', 'N/A')}")
        print(f"Expert Names:      {', '.join(model_info.get('expert_names', []))}")
        print(f"Hidden Size:       {model_info.get('hidden_size', 'N/A')}")
        print(f"Total Params:      {model_info.get('total_params', 0):,}")
        print(f"Trainable Params:  {model_info.get('trainable_params', 0):,}")
        if "per_expert_params" in model_info:
            print(f"Per Expert Params: {model_info['per_expert_params']:,}")
        print(f"{'='*70}\n")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def format_time(seconds):
    """Format seconds to readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def compute_metrics(predictions, labels, task_type="classification"):
    """
    Compute evaluation metrics

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        task_type: Type of task ("classification" or "regression")

    Returns:
        dict: Metrics
    """
    if task_type == "classification":
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return {"accuracy": accuracy}

    elif task_type == "regression":
        mse = ((predictions - labels) ** 2).mean().item()
        mae = (predictions - labels).abs().mean().item()
        return {"mse": mse, "mae": mae}

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    def save_phase1_expert_checkpoint(
        self,
        expert_name: str,
        task: str,
        expert_state_dict: dict,
        metrics: Optional[dict] = None,
        epoch: Optional[int] = None,
        is_final: bool = False,
        shared_embeddings_state_dict: Optional[dict] = None,
    ) -> str:
        """
        Save Phase 1 checkpoint for single expert

        This saves ONLY the expert being trained, along with ALL task heads.
        This allows for continual learning across tasks for the same expert.

        Args:
            expert_name: Name of expert (lexical, syntactic, semantic)
            task: Task name being trained on
            expert_state_dict: Single expert's state dict
            metrics: Training metrics (optional)
            epoch: Epoch number (optional)
            is_final: If True, save as final checkpoint for this expert
            shared_embeddings_state_dict: Shared embeddings state dict (Phase 1)

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            # Metadata
            "phase": 1,
            "expert_name": expert_name,
            "task": task,
            "epoch": epoch,
            "is_final": is_final,
            "global_step": self.global_step,
            # Model states - ONLY trained expert + ALL task heads
            "expert_state_dict": expert_state_dict,
            "task_heads_state_dict": self.task_heads.state_dict(),
            # Training states
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer else None
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            # Metrics
            "metrics": metrics or {},
        }

        # Add shared embeddings if provided
        if shared_embeddings_state_dict is not None:
            checkpoint["shared_embeddings_state_dict"] = shared_embeddings_state_dict

        # Generate filename
        if is_final:
            filename = f"{expert_name}_final.pt"
        else:
            epoch_str = f"_epoch{epoch}" if epoch is not None else ""
            filename = f"{expert_name}_{task}{epoch_str}.pt"

        checkpoint_path = os.path.join(self.log_dir, filename)

        try:
            torch.save(checkpoint, checkpoint_path)

            # Verify and show file size
            if os.path.exists(checkpoint_path):
                file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
                print(f"💾 Saved: {filename} ({file_size:.1f}MB)")
            else:
                print(f"❌ ERROR: Checkpoint not created: {checkpoint_path}")

        except Exception as e:
            print(f"❌ ERROR saving checkpoint: {e}")
            raise

        return checkpoint_path

    def load_phase1_expert_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = False,
    ) -> dict:
        """
        Load Phase 1 expert checkpoint

        Args:
            checkpoint_path: Path to checkpoint
            load_optimizer: Whether to load optimizer/scheduler states

        Returns:
            Full checkpoint dictionary
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Validate
        required_keys = [
            "phase",
            "expert_name",
            "expert_state_dict",
            "task_heads_state_dict",
        ]
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Invalid Phase 1 checkpoint: missing '{key}'")

        if checkpoint["phase"] != 1:
            raise ValueError(f"Not a Phase 1 checkpoint (phase={checkpoint['phase']})")

        # Load task heads (always load all heads)
        self.task_heads.load_state_dict(checkpoint["task_heads_state_dict"])

        # Load optimizer/scheduler if requested
        if load_optimizer:
            if self.optimizer and checkpoint.get("optimizer_state_dict"):
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.scheduler and checkpoint.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.global_step = checkpoint.get("global_step", 0)

        expert_name = checkpoint["expert_name"]
        task = checkpoint["task"]
        epoch = checkpoint.get("epoch", "N/A")

        print(f"✅ Phase 1 checkpoint loaded: {checkpoint_path}")
        print(f"   Expert: {expert_name} | Task: {task} | Epoch: {epoch}")

        return checkpoint


# ============================================================================
# Phase 1 → Phase 2 Transition Utilities
# ============================================================================


def load_phase1_checkpoints_to_phase2(
    checkpoint_path: str,
    model_config: dict,
    device: str = "cuda",
) -> tuple:
    """
    Load Phase 1 unified checkpoint and construct Phase 2 model

    Args:
        checkpoint_path: Path to phase1_final.pt (unified checkpoint)
        model_config: Model configuration dict (from training_config.py)
        device: Device to load to

    Returns:
        (model, task_heads): Phase 2 DAWN model and TaskHeads
    """
    print("\n" + "=" * 70)
    print("LOADING PHASE 1 CHECKPOINT → PHASE 2 MODEL")
    print("=" * 70)

    # Import here to avoid circular dependency
    from models import DAWN, TaskHeads

    # 1. Load unified checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"📂 Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Validate
    if ckpt.get("phase") != 1:
        raise ValueError(f"Not a Phase 1 checkpoint: {checkpoint_path}")
    if not ckpt.get("is_final"):
        print(f"⚠️  Warning: This is not a final checkpoint (is_final={ckpt.get('is_final')})")

    print(f"✅ Loaded Phase 1 checkpoint")
    print(f"   Experts: {', '.join(ckpt.get('expert_names', []))}")

    # 2. Create Phase 2 model (with peer prediction enabled)
    # Note: model_config should be from get_dawn_args() in the calling code
    model = DAWN(
        config=model_config,
        enable_peer_prediction=True,
        active_experts=None,
    )
    model = model.to(device)

    print(f"\n🔧 Created Phase 2 model with peer prediction enabled")

    # 3. Load full model state (includes all experts + shared embeddings)
    model_state = ckpt["model_state_dict"]
    model.load_state_dict(model_state, strict=False)
    # strict=False because peer_predictors/integrator are new in Phase 2

    print(f"✅ Loaded model weights (all experts + shared embeddings)")

    # 4. Load task heads
    task_heads_state = ckpt["task_heads_state_dict"]

    # Create TaskHeads
    task_heads = TaskHeads(
        hidden_size=model_config["hidden_size"],
        vocab_size=model_config["vocab_size"],
    )
    task_heads.load_state_dict(task_heads_state, strict=False)
    task_heads = task_heads.to(device)

    print(f"✅ Loaded TaskHeads (all heads)")

    # 5. Print summary
    print("\n" + "=" * 70)
    print("PHASE 2 MODEL READY")
    print("=" * 70)
    print(f"✅ Experts loaded: {', '.join(ckpt.get('expert_names', []))}")
    print(f"✅ Feature extractors: Initialized (will be trained)")
    print(f"✅ Integrator: PassThrough (simple averaging - stable)")
    print(f"✅ TaskHeads: All heads loaded")
    print("\n💡 Phase 2 Goal: Learn feature extraction (integrator stays simple)")
    print("💡 Phase 3 Plan: Learn attention integrator (extractors frozen)")
    print("=" * 70 + "\n")

    return model, task_heads


def validate_phase1_checkpoints(checkpoint_path: str) -> bool:
    """
    Validate Phase 1 unified checkpoint before loading

    Args:
        checkpoint_path: Path to phase1_final.pt

    Returns:
        True if checkpoint is valid
    """
    print("\n🔍 Validating Phase 1 checkpoint...")
    print(f"   Path: {checkpoint_path}")

    # Check existence
    if not os.path.exists(checkpoint_path):
        print(f"❌ File not found - {checkpoint_path}")
        return False

    try:
        # Load and check
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Check phase
        if ckpt.get("phase") != 1:
            print(f"❌ Not a Phase 1 checkpoint")
            return False

        # Check required keys
        required = ["model_state_dict", "task_heads_state_dict", "expert_names"]
        missing = [k for k in required if k not in ckpt]
        if missing:
            print(f"❌ Missing keys - {missing}")
            return False

        # Get info
        expert_names = ckpt.get("expert_names", [])
        is_final = ckpt.get("is_final", False)
        history = ckpt.get("all_training_history", {})

        status = "FINAL" if is_final else "intermediate"
        print(f"✅ Valid Phase 1 checkpoint ({status})")
        print(f"   Experts: {', '.join(expert_names)}")
        print(f"   Training stages: {len(history)}")

        return True

    except Exception as e:
        print(f"❌ Error loading - {e}")
        return False


# ============================================================================
# Phase 1 Training Loop Utilities
# ============================================================================


def train_phase1_single_task(
    expert,
    task_heads,
    trainer,
    task_name: str,
    dataloader,
    num_epochs: int,
    config: dict,
    epoch_callback=None,
) -> dict:
    """
    Train single expert on single task (Phase 1)

    Args:
        expert: Single DeltaExpert instance
        task_heads: TaskHeads module
        trainer: DAWNTrainer instance
        task_name: Task to train on
        dataloader: Task dataloader
        num_epochs: Number of epochs
        config: Training configuration
        epoch_callback: Optional callback(epoch_num, epoch_metrics) called after each epoch

    Returns:
        Training metrics dict
    """
    from tqdm import tqdm

    # Freeze all task heads except current
    task_heads.freeze_all_except(task_name)

    print(f"\n{'='*70}")
    print(f"TRAINING: {task_name.upper()}")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Active task head: {task_name}")
    print(f"{'='*70}\n")

    # Training metrics
    total_loss = 0
    total_steps = 0
    epoch_losses = []
    epoch_accuracies = []

    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    gradient_clip = config.get("gradient_clip", 1.0)
    log_every = config.get("log_every_n_steps", 1500)  # epoch 중간에 1~2번만

    for epoch in range(num_epochs):
        expert.train()
        task_heads.train()

        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        num_batches = len(dataloader)
        progress_bar = tqdm(dataloader, desc=f"Ep{epoch+1}/{num_epochs}", ncols=80, bar_format='{desc}:{percentage:3.0f}%|{bar}|{postfix}')

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch["input_ids"].to(trainer.device)
            attention_mask = batch["attention_mask"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)

            # Forward pass with AMP
            batch_loss = None
            batch_logits = None
            step_representations = []

            def forward_fn():
                nonlocal batch_loss, batch_logits, step_representations
                # Expert forward with all step outputs
                hidden = expert(input_ids, attention_mask, return_all_steps=True)

                # hidden is now a list of [step0, step1, step2, step3, step4]
                if isinstance(hidden, list):
                    step_representations = hidden
                    final_hidden = hidden[-1]  # Last step
                else:
                    final_hidden = hidden

                # Task head forward
                output = task_heads(final_hidden, task=task_name, labels=labels)
                batch_loss = output["loss"]
                batch_logits = output.get("logits", None)

                return batch_loss

            loss = trainer.forward_with_amp(forward_fn)

            # Calculate accuracy for this batch
            if batch_logits is not None and task_name in ["mlm", "span_masking", "token_deletion", "text_infilling"]:
                # Token-level tasks
                predictions = torch.argmax(batch_logits, dim=-1)
                mask = (labels != -100)
                correct = ((predictions == labels) & mask).sum().item()
                total = mask.sum().item()
                epoch_correct += correct
                epoch_total += total

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Backward
            trainer.backward_and_step(
                loss,
                accumulation_steps=gradient_accumulation_steps,
                gradient_clip=gradient_clip,
            )

            # Track metrics
            epoch_loss += loss.item() * gradient_accumulation_steps
            total_loss += loss.item() * gradient_accumulation_steps
            total_steps += 1

            # Update progress bar with step losses every 10%
            progress_pct = int((batch_idx + 1) / num_batches * 100)
            prev_pct = int(batch_idx / num_batches * 100)

            if progress_pct % 10 == 0 and prev_pct % 10 != 0 and step_representations:
                # Calculate loss for each refinement step
                step_losses_list = []
                with torch.no_grad():
                    for step_idx, step_hidden in enumerate(step_representations):
                        step_output = task_heads(step_hidden, task=task_name, labels=labels)
                        step_losses_list.append(step_output["loss"].item())

                # Format step losses with step numbers
                step_str = ", ".join([f"s{i}:{l:.2f}" for i, l in enumerate(step_losses_list)])

                # Calculate accuracy
                acc = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0.0

                # Update progress bar postfix (single line update)
                progress_bar.set_postfix_str(f"{step_str} | acc={acc:.1f}%")

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)

        # Calculate epoch accuracy
        epoch_acc = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0.0
        epoch_accuracies.append(epoch_acc)

        # Print final summary (after progress bar completes)
        progress_bar.close()
        print(f"  → Final: loss={avg_epoch_loss:.4f}, acc={epoch_acc:.2f}%")

        # Call epoch callback if provided
        if epoch_callback:
            epoch_metrics = {
                "epoch": epoch + 1,
                "loss": avg_epoch_loss,
                "accuracy": epoch_acc,
                "total_correct": epoch_correct,
                "total_samples": epoch_total,
            }
            epoch_callback(epoch + 1, epoch_metrics)

    # Final metrics
    avg_loss = total_loss / total_steps if total_steps > 0 else 0

    metrics = {
        "avg_loss": avg_loss,
        "total_steps": total_steps,
        "epoch_losses": epoch_losses,
        "epoch_accuracies": epoch_accuracies,
        "final_loss": epoch_losses[-1] if epoch_losses else 0,
        "final_accuracy": epoch_accuracies[-1] if epoch_accuracies else 0,
    }

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE: {task_name.upper()}")
    print(f"{'='*70}")
    print(f"Total Steps: {total_steps}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Final Epoch Loss: {metrics['final_loss']:.4f}")
    print(f"{'='*70}\n")

    return metrics


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing DAWNTrainer utilities...")

    # Mock objects for testing
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 64)
            self.experts = nn.ModuleDict(
                {
                    "lexical": nn.Linear(64, 64),
                    "semantic": nn.Linear(64, 64),
                }
            )
            self.integrator = nn.Linear(64, 64)

        def get_model_info(self):
            return {
                "phase": 1,
                "num_experts": 2,
                "expert_names": ["lexical", "semantic"],
                "hidden_size": 64,
                "total_params": sum(p.numel() for p in self.parameters()),
                "trainable_params": sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                ),
            }

    class MockTaskHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlm_head = nn.Linear(64, 100)

    model = MockModel()
    task_heads = MockTaskHeads()
    config = {"mixed_precision": False}

    trainer = DAWNTrainer(model, task_heads, config, device="cpu")

    # Test optimizer setup
    trainer.setup_optimizer(base_lr=5e-5, embedding_lr=1e-5)

    # Test monitoring
    trainer.print_trainable_summary()
    trainer.print_model_info()

    print("✅ All tests passed!")
