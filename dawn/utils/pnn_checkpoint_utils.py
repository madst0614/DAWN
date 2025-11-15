"""
PNN Checkpoint Utilities

Enhanced checkpoint saving for standalone PNN training.
Compatible with expert_loader.py CheckpointRegistry.
"""

import torch
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime


def save_pnn_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    accuracy: float,
    loss: float,
    history: Optional[Dict] = None,
    expert_name: str = "pnn_model",
    model_config: Optional[Dict] = None
):
    """
    Save PNN training checkpoint with expert_loader compatibility

    Args:
        path: Checkpoint save path
        model: PNN model
        optimizer: Optimizer
        scheduler: LR scheduler
        scaler: GradScaler (optional)
        epoch: Current epoch
        step: Global step
        accuracy: Current accuracy
        loss: Current loss
        history: Training history dict
        expert_name: Expert name for this model
        model_config: Model configuration dict
    """
    checkpoint = {
        # Core model state
        "model_state_dict": model.state_dict(),

        # Training states
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,

        # Training metadata (for CheckpointRegistry)
        "epoch": epoch,
        "step": step,
        "accuracy": accuracy,
        "mlm_acc": accuracy,  # Alias for backward compatibility
        "loss": loss,
        "best_acc": accuracy,  # Will be updated by training loop
        "history": history or {},

        # Expert metadata (for expert_loader)
        "expert_name": expert_name,
        "model_config": model_config or _extract_model_config(model),

        # Timestamp
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(checkpoint, path)
    print(f"💾 Checkpoint saved: {Path(path).name}")
    print(f"   Epoch: {epoch} | Step: {step} | Acc: {accuracy:.2%} | Loss: {loss:.4f}")


def load_pnn_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device: str = 'cuda'
):
    """
    Load PNN training checkpoint

    Args:
        path: Checkpoint path
        model: PNN model
        optimizer: Optimizer (optional)
        scheduler: Scheduler (optional)
        scaler: GradScaler (optional)
        device: Device to load on

    Returns:
        checkpoint: Full checkpoint dict
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Load model
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer/scheduler if provided
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Print info
    expert_name = checkpoint.get("expert_name", "unknown")
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    accuracy = checkpoint.get("accuracy", 0)

    print(f"✅ Loaded checkpoint: {Path(path).name}")
    print(f"   Expert: {expert_name} | Epoch: {epoch} | Step: {step} | Acc: {accuracy:.2%}")

    return checkpoint


def _extract_model_config(model) -> Dict:
    """
    Extract model config from PNN model

    Args:
        model: PNN model instance

    Returns:
        Model configuration dict
    """
    config = {}

    # Try to get config from model attributes
    if hasattr(model, 'hidden_size'):
        config['hidden_size'] = model.hidden_size
    if hasattr(model, 'vocab_size'):
        config['vocab_size'] = model.vocab_size
    if hasattr(model, 'num_steps'):
        config['num_steps'] = model.num_steps
    if hasattr(model, 'num_heads'):
        config['num_heads'] = model.num_heads

    # Try to get from delta_refiner
    if hasattr(model, 'delta_refiner'):
        refiner = model.delta_refiner
        if hasattr(refiner, 'num_blocks'):
            config['num_blocks'] = refiner.num_blocks

    return config


# ============================================================================
# Convenience Wrappers
# ============================================================================

class PNNCheckpointManager:
    """
    Checkpoint manager for PNN training

    Usage:
        manager = PNNCheckpointManager(
            checkpoint_dir='/path/to/checkpoints',
            expert_name='lexical'
        )

        # During training
        manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            step=step,
            accuracy=acc,
            loss=loss
        )

        # Load
        checkpoint = manager.load_latest(model, optimizer, scheduler, scaler)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        expert_name: str = 'pnn_model',
        model_config: Optional[Dict] = None,
        keep_last_n: int = 3
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            expert_name: Name for this expert
            model_config: Model configuration
            keep_last_n: Keep only last N checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.expert_name = expert_name
        self.model_config = model_config
        self.keep_last_n = keep_last_n

        self.best_accuracy = 0.0
        self.history = {}

    def save(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch: int,
        step: int,
        accuracy: float,
        loss: float,
        is_best: bool = False,
        is_final: bool = False
    ):
        """
        Save checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer
            scheduler: Scheduler
            scaler: GradScaler
            epoch: Current epoch
            step: Global step
            accuracy: Current accuracy
            loss: Current loss
            is_best: If True, also save as best checkpoint
            is_final: If True, save as final checkpoint
        """
        # Update best
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy

        # Update history
        if epoch not in self.history:
            self.history[epoch] = []
        self.history[epoch].append({
            'step': step,
            'accuracy': accuracy,
            'loss': loss
        })

        # Regular checkpoint
        if not is_final:
            filename = f"{self.expert_name}_epoch{epoch}_step{step}.pt"
        else:
            filename = f"{self.expert_name}_final.pt"

        path = self.checkpoint_dir / filename

        save_pnn_checkpoint(
            path=str(path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            step=step,
            accuracy=accuracy,
            loss=loss,
            history=self.history,
            expert_name=self.expert_name,
            model_config=self.model_config or _extract_model_config(model)
        )

        # Save best if this is best
        if is_best:
            best_path = self.checkpoint_dir / f"{self.expert_name}_best.pt"
            save_pnn_checkpoint(
                path=str(best_path),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=step,
                accuracy=accuracy,
                loss=loss,
                history=self.history,
                expert_name=self.expert_name,
                model_config=self.model_config or _extract_model_config(model)
            )
            print(f"   💎 New best model saved!")

        # Cleanup old checkpoints
        if not is_final and self.keep_last_n > 0:
            self._cleanup_old_checkpoints()

    def load_latest(
        self,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        device='cuda'
    ):
        """Load latest checkpoint"""
        checkpoints = sorted(
            self.checkpoint_dir.glob(f"{self.expert_name}_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not checkpoints:
            print("No checkpoints found")
            return None

        latest = checkpoints[0]
        return load_pnn_checkpoint(
            str(latest), model, optimizer, scheduler, scaler, device
        )

    def load_best(
        self,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        device='cuda'
    ):
        """Load best checkpoint"""
        best_path = self.checkpoint_dir / f"{self.expert_name}_best.pt"

        if not best_path.exists():
            print("Best checkpoint not found")
            return None

        return load_pnn_checkpoint(
            str(best_path), model, optimizer, scheduler, scaler, device
        )

    def _cleanup_old_checkpoints(self):
        """Keep only last N checkpoints"""
        checkpoints = sorted(
            self.checkpoint_dir.glob(f"{self.expert_name}_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Remove old checkpoints (keep last N + best)
        for ckpt in checkpoints[self.keep_last_n:]:
            if 'best' not in ckpt.name and 'final' not in ckpt.name:
                ckpt.unlink()
                print(f"   🗑️  Removed old checkpoint: {ckpt.name}")


# Example usage
if __name__ == "__main__":
    """
    Example: Using PNNCheckpointManager
    """
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR
    from torch.cuda.amp import GradScaler

    # Mock model
    class MockPNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 768
            self.vocab_size = 30522
            self.num_steps = 4
            self.embedding = nn.Embedding(30522, 768)
            self.linear = nn.Linear(768, 768)

    model = MockPNN()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000)
    scaler = GradScaler()

    # Create manager
    manager = PNNCheckpointManager(
        checkpoint_dir='/tmp/pnn_checkpoints',
        expert_name='lexical',
        keep_last_n=3
    )

    # Simulate training
    for epoch in range(5):
        for step in range(100):
            # Training...
            accuracy = 0.45 + epoch * 0.01 + step * 0.0001
            loss = 2.0 - epoch * 0.1

            # Save every 50 steps
            if step % 50 == 0:
                is_best = accuracy > manager.best_accuracy
                manager.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    step=epoch * 100 + step,
                    accuracy=accuracy,
                    loss=loss,
                    is_best=is_best
                )

    # Save final
    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=4,
        step=499,
        accuracy=0.50,
        loss=1.5,
        is_final=True
    )

    print("\n✅ PNN checkpoint manager demo complete!")
    print(f"Checkpoints saved to: {manager.checkpoint_dir}")
