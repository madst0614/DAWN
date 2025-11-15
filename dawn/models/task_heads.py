"""
Task Heads Module

Separate prediction heads for all tasks.
Shared across Phase 1 (expert training) and Phase 2 (collaborative training).

Each expert learns representations, task heads make predictions.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class TaskHeads(nn.Module):
    """
    Multi-Task Prediction Heads
    
    Converts hidden representations to task-specific predictions.
    Used in both Phase 1 and Phase 2 with the same weights.
    
    Args:
        hidden_size: Hidden dimension from encoder
        vocab_size: Vocabulary size for MLM
        config: Optional task-specific configs
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.config = config or {}
        
        # ============================================
        # Lexical Tasks
        # ============================================

        # Masked Language Modeling
        # Pre-normalization for stability (prevents gradient explosion)
        self.mlm_norm = nn.LayerNorm(hidden_size)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        
        # Word-in-Context (binary classification)
        self.wic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)
        )
        
        # ============================================
        # Syntactic Tasks
        # ============================================
        
        # Span Masking (same as MLM)
        self.span_masking_norm = nn.LayerNorm(hidden_size)
        self.span_masking_head = nn.Linear(hidden_size, vocab_size)

        # Token Deletion (binary: keep or delete)
        self.token_deletion_norm = nn.LayerNorm(hidden_size)
        self.token_deletion_head = nn.Linear(hidden_size, 2)

        # Text Infilling (same as MLM)
        self.text_infilling_norm = nn.LayerNorm(hidden_size)
        self.text_infilling_head = nn.Linear(hidden_size, vocab_size)

        # Sentence Permutation (predict sentence order)
        self.sentence_permutation_head = nn.Linear(hidden_size, 2)  # Binary: correct order or not
        
        # ============================================
        # Semantic Tasks
        # ============================================
        
        # Natural Language Inference (3-way classification)
        self.nli_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3)
        )
        
        # Paraphrase Detection (binary classification)
        self.paraphrase_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)
        )
        
        # Semantic Textual Similarity (regression)
        self.sts_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with standard initialization"""
        init_std = 0.02

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        hidden: torch.Tensor,
        task: str,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for specific task
        
        Args:
            hidden: [B, L, D] hidden representations
            task: Task name (mlm, wic, nli, etc.)
            labels: Optional labels for loss computation
            
        Returns:
            dict with 'logits' and optionally 'loss'
        """
        
        if task == "mlm":
            # Normalize hidden before MLM head (prevents gradient explosion)
            logits = self.mlm_head(self.mlm_norm(hidden))

        elif task == "wic":
            # Use [CLS] token
            cls_hidden = hidden[:, 0, :]  # [B, D]
            logits = self.wic_head(cls_hidden)

        elif task == "span_masking":
            # Normalize hidden before head
            logits = self.span_masking_head(self.span_masking_norm(hidden))

        elif task == "token_deletion":
            # Binary classification per token
            logits = self.token_deletion_head(self.token_deletion_norm(hidden))

        elif task == "text_infilling":
            # Normalize hidden before head
            logits = self.text_infilling_head(self.text_infilling_norm(hidden))

        elif task == "sentence_permutation":
            # Use [CLS] token for sentence-level prediction
            cls_hidden = hidden[:, 0, :]
            logits = self.sentence_permutation_head(cls_hidden)

        elif task == "nli":
            # Use [CLS] token
            cls_hidden = hidden[:, 0, :]  # [B, D]
            logits = self.nli_head(cls_hidden)
        
        elif task == "paraphrase":
            # Use [CLS] token
            cls_hidden = hidden[:, 0, :]  # [B, D]
            logits = self.paraphrase_head(cls_hidden)
        
        elif task == "sts":
            # Use [CLS] token, regression output
            cls_hidden = hidden[:, 0, :]  # [B, D]
            logits = self.sts_head(cls_hidden).squeeze(-1)  # [B]
        
        else:
            raise ValueError(f"Unknown task: {task}")
        
        result = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            loss = self.compute_loss(logits, labels, task)
            result["loss"] = loss
        
        return result
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        task: str
    ) -> torch.Tensor:
        """
        Compute task-specific loss

        Args:
            logits: Model predictions
            labels: Ground truth labels
            task: Task name

        Returns:
            loss: Scalar loss tensor
        """

        # Clamp logits to prevent overflow in loss computation
        logits = torch.clamp(logits, min=-50.0, max=50.0)

        if task in ["mlm", "span_masking", "text_infilling"]:
            # Cross-entropy for token prediction
            vocab_size = logits.size(-1)
            loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

        elif task == "token_deletion":
            # Binary classification per token
            num_classes = logits.size(-1)
            loss = nn.functional.cross_entropy(
                logits.view(-1, num_classes),
                labels.view(-1),
                ignore_index=-100
            )

        elif task in ["wic", "nli", "paraphrase", "sentence_permutation"]:
            # Cross-entropy for classification
            loss = nn.functional.cross_entropy(
                logits,
                labels
            )

        elif task == "sts":
            # MSE for regression
            loss = nn.functional.mse_loss(
                logits,
                labels.float()
            )

        else:
            raise ValueError(f"Unknown task for loss: {task}")

        return loss
    
    def get_head_for_task(self, task: str) -> nn.Module:
        """Get the specific head module for a task"""
        task_to_head = {
            "mlm": self.mlm_head,
            "wic": self.wic_head,
            "span_masking": self.span_masking_head,
            "token_deletion": self.token_deletion_head,
            "text_infilling": self.text_infilling_head,
            "sentence_permutation": self.sentence_permutation_head,
            "nli": self.nli_head,
            "paraphrase": self.paraphrase_head,
            "sts": self.sts_head,
        }
        
        if task not in task_to_head:
            raise ValueError(f"Unknown task: {task}")
        
        return task_to_head[task]
    
    def freeze_all_except(self, task: str):
        """Freeze all heads except for specified task"""
        # Freeze all
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze specific task head
        head = self.get_head_for_task(task)
        for param in head.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all task heads"""
        for param in self.parameters():
            param.requires_grad = True


# ============================================================================
# Utility Functions
# ============================================================================

def create_task_heads(hidden_size: int, vocab_size: int, config: Optional[Dict] = None) -> TaskHeads:
    """
    Factory function to create TaskHeads
    
    Args:
        hidden_size: Hidden dimension
        vocab_size: Vocabulary size
        config: Optional configuration
        
    Returns:
        TaskHeads instance
    """
    return TaskHeads(hidden_size, vocab_size, config)


def load_task_heads(checkpoint_path: str, hidden_size: int, vocab_size: int) -> TaskHeads:
    """
    Load TaskHeads from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        hidden_size: Hidden dimension
        vocab_size: Vocabulary size
        
    Returns:
        TaskHeads with loaded weights
    """
    task_heads = TaskHeads(hidden_size, vocab_size)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract task_heads state dict
    if "task_heads_state_dict" in checkpoint:
        task_heads.load_state_dict(checkpoint["task_heads_state_dict"])
    else:
        # Fallback: try to load from model_state_dict with prefix
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        task_heads_state = {
            k.replace("task_heads.", ""): v
            for k, v in state_dict.items()
            if k.startswith("task_heads.")
        }
        if task_heads_state:
            task_heads.load_state_dict(task_heads_state)
        else:
            raise ValueError("No task_heads state dict found in checkpoint")
    
    return task_heads


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing TaskHeads...")
    
    # Create instance
    hidden_size = 768
    vocab_size = 30522
    batch_size = 4
    seq_len = 128
    
    task_heads = TaskHeads(hidden_size, vocab_size)
    
    # Test each task
    hidden = torch.randn(batch_size, seq_len, hidden_size)
    
    print("\n" + "="*70)
    print("Task Head Outputs:")
    print("="*70)
    
    # MLM
    result = task_heads(hidden, task="mlm")
    print(f"MLM logits shape: {result['logits'].shape}")  # [B, L, V]
    
    # WiC
    result = task_heads(hidden, task="wic")
    print(f"WiC logits shape: {result['logits'].shape}")  # [B, 2]
    
    # NLI
    result = task_heads(hidden, task="nli")
    print(f"NLI logits shape: {result['logits'].shape}")  # [B, 3]
    
    # STS
    result = task_heads(hidden, task="sts")
    print(f"STS logits shape: {result['logits'].shape}")  # [B]
    
    # Test loss computation
    print("\n" + "="*70)
    print("Loss Computation:")
    print("="*70)
    
    # MLM loss
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    result = task_heads(hidden, task="mlm", labels=labels)
    print(f"MLM loss: {result['loss'].item():.4f}")
    
    # Classification loss
    labels = torch.randint(0, 2, (batch_size,))
    result = task_heads(hidden, task="wic", labels=labels)
    print(f"WiC loss: {result['loss'].item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in task_heads.parameters())
    print(f"\n" + "="*70)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print("="*70)
