"""
SPROUT Language Model wrapper for MLM training.

Integrates SPROUT with token embeddings and MLM head.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List

from .model import SPROUT


class SproutLanguageModel(nn.Module):
    """
    SPROUT-based language model for MLM training.

    Architecture:
        Token Embeddings → SPROUT (dynamic tree) → MLM Head
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        max_depth: int = 2,  # Limited to ~5 nodes total
        compatibility_threshold: float = 0.8,
        num_heads: int = 4,
        ffn_mult: int = 4,
        max_nodes: int = 5,  # Hard limit on total nodes
        dropout: float = 0.1,
    ):
        """
        Initialize SPROUT language model.

        Args:
            vocab_size: Vocabulary size
            hidden_dim: Hidden dimension
            max_depth: Maximum tree depth (2 = ~5 nodes max)
            compatibility_threshold: Branching threshold
            num_heads: Number of attention heads
            ffn_mult: FFN expansion multiplier
            max_nodes: Hard limit on total nodes
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        # Token embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(512, hidden_dim)  # Max seq length 512
        self.dropout = nn.Dropout(dropout)

        # SPROUT core
        self.sprout = SPROUT(
            dim=hidden_dim,
            max_depth=max_depth,
            compatibility_threshold=compatibility_threshold,
            num_heads=num_heads,
            ffn_mult=ffn_mult,
            max_nodes=max_nodes
        )

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size),
        )

        # Node count tracking
        self._node_limit_reached = False

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)

        # Initialize MLM head
        for module in self.mlm_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        context_bias: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq] token IDs
            attention_mask: [batch, seq] attention mask
            labels: [batch, seq] MLM labels (-100 for non-masked)
            context_bias: Optional context for routing

        Returns:
            Dictionary with:
                - logits: [batch, seq, vocab_size]
                - loss: MLM loss (if labels provided)
                - path_log: Routing decisions
        """
        batch_size, seq_len = input_ids.shape

        # Check node limit
        num_nodes = self.sprout.count_total_nodes()
        if num_nodes >= self.max_nodes and not self._node_limit_reached:
            self._node_limit_reached = True
            print(f"⚠️  Node limit reached: {num_nodes}/{self.max_nodes} nodes")

        # Embeddings
        token_embeds = self.embeddings(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)

        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        # SPROUT forward (only if under node limit or in eval mode)
        if self.training and num_nodes >= self.max_nodes:
            # Node limit reached: use current structure without creating new nodes
            # Temporarily increase threshold to prevent branching
            original_threshold = self.sprout.compatibility_threshold
            self.sprout.compatibility_threshold = 1.0  # Prevent any new branches

            output, path_log = self.sprout(hidden_states, context_bias)

            # Restore threshold
            self.sprout.compatibility_threshold = original_threshold
        else:
            output, path_log = self.sprout(hidden_states, context_bias)

        # MLM head
        logits = self.mlm_head(output)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten for loss calculation
            loss = loss_fct(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )

        return {
            "logits": logits,
            "loss": loss,
            "path_log": path_log,
            "num_nodes": num_nodes,
        }

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "total_nodes": self.sprout.count_total_nodes(),
            "max_nodes": self.max_nodes,
            "node_limit_reached": self._node_limit_reached,
            "sprout_stats": self.sprout.get_statistics(),
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    def visualize_structure(self, max_depth: int = 3):
        """Visualize SPROUT structure."""
        self.sprout.visualize_structure(max_depth=max_depth)

    def is_converged(self, window: int = 100, threshold: float = 0.001) -> bool:
        """Check if structure has converged."""
        return self.sprout.is_converged(window=window, threshold=threshold)
