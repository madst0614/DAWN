"""Training utilities"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(model, lr, weight_decay):
    """Create AdamW optimizer"""
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, warmup_steps, total_steps):
    """Create learning rate scheduler with warmup"""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, (total_steps - step) / (total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path, model, optimizer, scheduler, scaler, epoch, best_acc, history
):
    """Save training checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_acc": best_acc,
        "history": history,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint
