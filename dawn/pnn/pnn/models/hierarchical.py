"""
Plastic Neural Networks (PNN) - Main Model Implementation

Paper: "Plastic Neural Networks: Learning Through Iterative Delta Refinement"
Author: Seungho Choi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


class DeltaValidator(nn.Module):
    """
    Dimension-wise validator for delta proposals.

    Determines which dimensions of a proposed delta should be
    applied to the current representation through query-key matching:

    - Query (from h_current): "What refinements do I need?"
    - Key (from delta): "What refinements do I offer?"
    - Validity: Compatibility between need and offer
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Query: "What refinements do I need?"
        self.query_proj = nn.Linear(hidden_size, hidden_size)

        # Key: "What refinements do I offer?"
        self.key_proj = nn.Linear(hidden_size, hidden_size)

        # Learnable temperature for validity scaling
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(hidden_size))

    def forward(self, h: torch.Tensor, delta_proposal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Current representation [batch, seq_len, hidden]
            delta_proposal: Proposed update [batch, seq_len, hidden]

        Returns:
            validity: Element-wise validity gates [batch, seq_len, hidden]
        """
        # Compute query and key
        query = self.query_proj(h)
        key = self.key_proj(delta_proposal)

        # Element-wise compatibility
        compatibility = query * key

        # Temperature-scaled sigmoid for validity
        validity = torch.sigmoid(compatibility / self.temperature)

        return validity


class DeltaRefinerHierarchical(nn.Module):
    """
    Hierarchical Mini-Delta Accumulation with Mountain-shaped FFN

    Each block proposes a mini-delta, accumulated to form final delta.
    All processing references h_original + accumulated_delta.

    Key features:
    - Each block generates mini-delta based on h_original + accumulated_delta
    - Mini-gates for each block's proposal
    - Final gate on total accumulated delta
    - Supports per-block intermediate sizes for mountain-shaped capacity
    - Small initialization (0.02 / sqrt(num_blocks)) for stability
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int | list[int] = 2048,
        dropout: float = 0.1,
        num_blocks: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks

        # Handle intermediate_size: int or list
        if isinstance(intermediate_size, int):
            # Uniform size for all blocks
            intermediate_sizes = [intermediate_size] * num_blocks
        else:
            # Per-block sizes (mountain-shaped)
            assert (
                len(intermediate_size) == num_blocks
            ), f"intermediate_size list length ({len(intermediate_size)}) must match num_blocks ({num_blocks})"
            intermediate_sizes = intermediate_size

        # Create transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block_intermediate_size = intermediate_sizes[i]
            block = nn.ModuleDict(
                {
                    "attention": nn.MultiheadAttention(
                        embed_dim=hidden_size,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True,
                    ),
                    "attn_layer_norm": nn.LayerNorm(hidden_size),
                    "attn_dropout": nn.Dropout(dropout),
                    "ffn": nn.Sequential(
                        nn.Linear(hidden_size, block_intermediate_size),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(block_intermediate_size, hidden_size),
                        nn.Dropout(dropout),
                    ),
                    "ffn_layer_norm": nn.LayerNorm(hidden_size),
                }
            )

            # Zero-initialize final FFN layer for stable training
            nn.init.zeros_(block["ffn"][3].weight)
            nn.init.zeros_(block["ffn"][3].bias)

            self.blocks.append(block)

        # Mini-gates for each block
        self.mini_gates = nn.ModuleList(
            [DeltaValidator(hidden_size) for _ in range(num_blocks)]
        )

        # Final gate for accumulated delta
        self.final_gate = DeltaValidator(hidden_size)

    def forward(
        self, h: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute delta through hierarchical mini-delta accumulation.

        Args:
            h: Current representation [batch, seq_len, hidden]
            attention_mask: Attention mask

        Returns:
            delta: Final gated accumulated delta
        """
        h_original = h
        accumulated_delta = torch.zeros_like(h)

        for i, block in enumerate(self.blocks):
            # Current state = original + accumulated changes
            h_with_delta = h_original + accumulated_delta

            # Attention block
            attn_out, _ = block["attention"](
                h_with_delta,
                h_with_delta,
                h_with_delta,
                key_padding_mask=attention_mask,
            )
            h_attn = block["attn_layer_norm"](
                h_with_delta + block["attn_dropout"](attn_out)
            )

            # Propose mini-delta
            mini_delta_proposal = block["ffn"](h_attn)
            mini_delta_proposal = block["ffn_layer_norm"](mini_delta_proposal)

            # Gate this block's proposal
            mini_gate = self.mini_gates[i](h_original, mini_delta_proposal)
            mini_delta = mini_gate * mini_delta_proposal

            # Accumulate
            accumulated_delta = accumulated_delta + mini_delta

        # Final gate on total accumulated delta
        final_gate = self.final_gate(h_original, accumulated_delta)
        final_delta = final_gate * accumulated_delta

        return final_delta


class PlasticNeuralNetwork(nn.Module):
    """
    Plastic Neural Network (PNN)

    Learns through iterative delta refinement rather than deep layer stacks.
    Supports both single-block (DeltaRefiner) and hierarchical (DeltaRefinerHierarchical)
    refinement strategies.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int | list[int] = 2048,
        max_length: int = 128,
        num_steps: int = 4,
        dropout: float = 0.1,
        use_hierarchical: bool = False,
        num_blocks: int = 5,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        self.use_hierarchical = use_hierarchical
        self.use_checkpoint = use_checkpoint

        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        self.embedding_layer_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        # Delta refiner (shared across steps)
        if use_hierarchical:
            # Hierarchical refiner with multiple blocks
            self.delta_refiner = DeltaRefinerHierarchical(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                num_blocks=num_blocks,
            )
        else:
            # Single-block refiner
            # intermediate_size must be int for single-block
            if isinstance(intermediate_size, list):
                raise ValueError(
                    "intermediate_size must be int for non-hierarchical mode"
                )
            self.delta_refiner = DeltaRefiner(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )

        # MLM head
        self.mlm_head = nn.Linear(hidden_size, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following BERT"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_all_steps: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with recurrent delta refinement.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_all_steps: Return outputs from all refinement steps

        Returns:
            hidden: Final representation [batch, seq_len, hidden]
            or list of representations if return_all_steps=True
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)

        # Add position embeddings
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        position_embeds = self.position_embeddings(position_ids)

        hidden = token_embeds + position_embeds
        hidden = self.embedding_layer_norm(hidden)
        hidden = self.embedding_dropout(hidden)

        # Convert attention mask format (1 = valid, 0 = masked)
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask == 0  # True = masked

        all_outputs = [hidden] if return_all_steps else None

        # Recurrent refinement
        for step in range(self.num_steps):
            # Use gradient checkpointing if enabled (only for hierarchical)
            if self.use_checkpoint and self.use_hierarchical and self.training:
                delta = checkpoint(
                    self.delta_refiner, hidden, attn_mask, use_reentrant=False
                )
            else:
                delta = self.delta_refiner(hidden, attn_mask)
            hidden = hidden + delta

            if return_all_steps:
                all_outputs.append(hidden)

        if return_all_steps:
            return all_outputs
        else:
            return hidden
