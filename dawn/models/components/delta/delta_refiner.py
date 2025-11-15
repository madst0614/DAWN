"""
Delta Refiner - Hierarchical Mini-Delta Accumulation

PNN at micro-level: accumulates mini-deltas across multiple blocks.

Architecture:
- Post-LN (PNN style): Attention → LayerNorm, FFN → LayerNorm
- FFN output normalization is KEY for stability with mini-delta accumulation
- Each block proposes mini-delta, which is gated and accumulated
- Final delta = FinalGate(h_original, accumulated_delta)

Memory Optimizations:
- Gradient checkpointing support for reduced memory usage
- Immediate tensor cleanup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from torch.utils.checkpoint import checkpoint
import numpy as np

from .gate import QueryKeyGate


class DeltaRefiner(nn.Module):
    """
    Hierarchical Mini-Delta Accumulation

    Processes input through multiple blocks, each proposing a mini-delta.
    All processing references h_original + accumulated_delta.

    Args:
        hidden_size: int
        num_heads: int
        intermediate_size: int or List[int] - supports mountain-shaped
        dropout: float
        config: Dict with refiner-specific configs
        num_blocks: int (default 5)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size,  # int or list
        dropout: float,
        config: Dict,
        num_blocks: int = 5,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks

        # Config
        attention_dropout = config.get('attention_dropout', dropout)
        ffn_dropout = config.get('ffn_dropout', dropout)
        self.debug_mode = config.get('debug_mode', False)  # NaN checking only in debug mode

        # Memory optimization flags
        self.use_gradient_checkpointing = config.get('use_gradient_checkpointing', True)  # Enable by default

        # Handle intermediate_size: int or list (mountain-shaped)
        if isinstance(intermediate_size, int):
            intermediate_sizes = [intermediate_size] * num_blocks
        else:
            assert len(intermediate_size) == num_blocks, \
                f"intermediate_size list length ({len(intermediate_size)}) must match num_blocks ({num_blocks})"
            intermediate_sizes = intermediate_size

        # Create blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block_intermediate_size = intermediate_sizes[i]

            block = nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=attention_dropout,
                    batch_first=True,
                ),
                'attn_layer_norm': nn.LayerNorm(hidden_size),
                'attn_dropout': nn.Dropout(attention_dropout),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_size, block_intermediate_size),
                    nn.GELU(),
                    nn.Dropout(ffn_dropout),
                    nn.Linear(block_intermediate_size, hidden_size),
                    nn.Dropout(ffn_dropout),
                ),
                'ffn_layer_norm': nn.LayerNorm(hidden_size),
            })

            # Zero init for final FFN layer (most conservative)
            # Combined with sigmoid gates, this provides:
            # 1. Perfect identity mapping initially (no delta)
            # 2. Stable gradient flow through bounded [0,1] gates
            # 3. No amplification issues
            nn.init.zeros_(block['ffn'][3].weight)
            nn.init.zeros_(block['ffn'][3].bias)

            self.blocks.append(block)

        # Mini-gates and final gate (simple sigmoid gates)
        self.mini_gates = nn.ModuleList([
            QueryKeyGate(hidden_size, config)
            for _ in range(num_blocks)
        ])
        self.final_gate = QueryKeyGate(hidden_size, config)

    def _process_block(
        self,
        block_idx: int,
        h_original: torch.Tensor,
        accumulated_delta: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process a single block - can be checkpointed

        Using PNN's Post-LN architecture:
        - Attention → LayerNorm(h + attn_out)
        - FFN → LayerNorm(ffn_out)

        This provides better stability for mini-delta accumulation.
        """
        block = self.blocks[block_idx]

        # Current state = original + accumulated changes
        h_with_delta = h_original + accumulated_delta

        # Post-LN: Attention THEN LayerNorm (PNN style)
        attn_out, _ = block['attention'](
            h_with_delta, h_with_delta, h_with_delta,
            key_padding_mask=attention_mask,
        )

        # Debug mode: Check attention output
        if self.debug_mode and not torch.isfinite(attn_out).all():
            raise RuntimeError(
                f"NaN/Inf in DeltaRefiner block {block_idx} attention output!\n"
                f"  Input h_with_delta range: [{h_with_delta.min():.4f}, {h_with_delta.max():.4f}]\n"
                f"  Attention output range: [{attn_out.min():.4f}, {attn_out.max():.4f}]"
            )

        # Post-LN: LayerNorm AFTER residual connection
        h_attn = block['attn_layer_norm'](
            h_with_delta + block['attn_dropout'](attn_out)
        )

        # Debug mode: Check post-attention LayerNorm
        if self.debug_mode and not torch.isfinite(h_attn).all():
            raise RuntimeError(
                f"NaN/Inf in DeltaRefiner block {block_idx} h_attn after LayerNorm!\n"
                f"  h_with_delta range: [{h_with_delta.min():.4f}, {h_with_delta.max():.4f}]\n"
                f"  attn_out range: [{attn_out.min():.4f}, {attn_out.max():.4f}]\n"
                f"  h_attn range: [{h_attn.min():.4f}, {h_attn.max():.4f}]"
            )

        # Post-LN: FFN THEN LayerNorm (PNN style)
        mini_delta_proposal = block['ffn'](h_attn)
        mini_delta_proposal = block['ffn_layer_norm'](mini_delta_proposal)

        # Debug mode: Check FFN output
        if self.debug_mode and not torch.isfinite(mini_delta_proposal).all():
            num_nan = torch.isnan(mini_delta_proposal).sum().item()
            num_inf = torch.isinf(mini_delta_proposal).sum().item()
            raise RuntimeError(
                f"NaN/Inf detected in DeltaRefiner block {block_idx} FFN output!\n"
                f"  NaN count: {num_nan}, Inf count: {num_inf}\n"
                f"  Input h_attn range: [{h_attn.min():.4f}, {h_attn.max():.4f}]\n"
                f"  FFN output range: [{mini_delta_proposal.min():.4f}, {mini_delta_proposal.max():.4f}]\n"
                f"  This indicates a critical numerical instability - training stopped."
            )

        # Gate this block's proposal
        gate_values = self.mini_gates[block_idx](h_original, mini_delta_proposal)
        mini_delta = gate_values * mini_delta_proposal

        return mini_delta

    def forward(
        self,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hierarchical mini-delta accumulation

        Args:
            h: [B, L, D]
            attention_mask: [B, L] (True = masked)

        Returns:
            delta: [B, L, D] - final gated delta
        """
        h_original = h
        accumulated_delta = torch.zeros_like(h)

        for block_idx in range(self.num_blocks):
            # Process block: computes mini_delta based on (h_original + accumulated_delta)
            if self.use_gradient_checkpointing and self.training:
                # Gradient checkpointing for memory efficiency
                mini_delta = checkpoint(
                    self._process_block,
                    block_idx,
                    h_original,
                    accumulated_delta,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                mini_delta = self._process_block(
                    block_idx, h_original, accumulated_delta, attention_mask
                )

            # Accumulate mini-deltas
            accumulated_delta = accumulated_delta + mini_delta

        # Final gate (QueryKeyGate returns gate values [0,1])
        gate_values = self.final_gate(h_original, accumulated_delta)
        final_delta = gate_values * accumulated_delta

        return final_delta
