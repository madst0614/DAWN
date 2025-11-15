"""
Simple Delta Refiner - Multi-Block Refiner WITHOUT Any Gates

Ablation study implementation:
- Multiple attention + FFN blocks (like hierarchical)
- Post-LN architecture (PNN style)
- Raw accumulation without ANY gates (key difference!)
- NO mini-gates, NO final gate

Used for ablation study: testing if gating is actually necessary.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List


class SimpleDeltaRefiner(nn.Module):
    """
    Simple Multi-Block Delta Refiner - NO Gates

    Multiple blocks with raw accumulation:
    1. Each block: Attention + FFN → mini-delta
    2. Accumulate all mini-deltas (NO gating!)
    3. Return raw accumulated delta (NO final gate!)

    Key difference from hierarchical:
    - Hierarchical: 5 mini-gates + 1 final gate (6 total)
    - Simple: NO gates at all (0 total)

    This ablation tests whether gating is necessary for multi-block refinement.

    Args:
        hidden_size: int
        num_heads: int
        intermediate_size: int or List[int] - FFN intermediate dimension(s)
        dropout: float
        config: Dict with refiner-specific configs
        num_blocks: int - number of refinement blocks
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size,  # int or List[int]
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
        self.debug_mode = config.get('debug_mode', False)

        # Handle intermediate_size: int or list
        if isinstance(intermediate_size, int):
            intermediate_sizes = [intermediate_size] * num_blocks
        else:
            intermediate_sizes = intermediate_size
            if len(intermediate_sizes) != num_blocks:
                raise ValueError(
                    f"intermediate_size list length ({len(intermediate_sizes)}) "
                    f"must match num_blocks ({num_blocks})"
                )

        # Create multiple blocks (similar to hierarchical)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=attention_dropout,
                    batch_first=True
                ),
                'attn_layer_norm': nn.LayerNorm(hidden_size),
                'attn_dropout': nn.Dropout(attention_dropout),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_size, intermediate_sizes[i]),
                    nn.GELU(),
                    nn.Dropout(ffn_dropout),
                    nn.Linear(intermediate_sizes[i], hidden_size),
                    nn.Dropout(ffn_dropout)
                ),
                'ffn_layer_norm': nn.LayerNorm(hidden_size),
            })

            # Zero-initialize final FFN layer
            nn.init.zeros_(block['ffn'][3].weight)
            nn.init.zeros_(block['ffn'][3].bias)

            self.blocks.append(block)

        # NO gates at all - pure ablation study

    def forward(
        self,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Raw multi-block refinement without ANY gates

        Args:
            h: [B, L, D] - current representation
            attention_mask: [B, L] (True = masked)

        Returns:
            delta: [B, L, D] - raw accumulated delta (NO gating!)
        """
        h_original = h
        accumulated_delta = torch.zeros_like(h)

        # Process all blocks and accumulate (NO gates at all!)
        for block_idx, block in enumerate(self.blocks):
            # Current state = original + accumulated changes
            h_with_delta = h_original + accumulated_delta

            # Post-LN: Attention THEN LayerNorm
            attn_out, _ = block['attention'](
                h_with_delta, h_with_delta, h_with_delta,
                key_padding_mask=attention_mask,
            )
            h_attn = block['attn_layer_norm'](
                h_with_delta + block['attn_dropout'](attn_out)
            )

            # Debug mode: Check attention
            if self.debug_mode and not torch.isfinite(h_attn).all():
                raise RuntimeError(
                    f"NaN/Inf in SimpleDeltaRefiner block {block_idx} attention!\n"
                    f"  h_with_delta range: [{h_with_delta.min():.4f}, {h_with_delta.max():.4f}]\n"
                    f"  h_attn range: [{h_attn.min():.4f}, {h_attn.max():.4f}]"
                )

            # Post-LN: FFN THEN LayerNorm
            mini_delta = block['ffn'](h_attn)
            mini_delta = block['ffn_layer_norm'](mini_delta)

            # Debug mode: Check FFN
            if self.debug_mode and not torch.isfinite(mini_delta).all():
                num_nan = torch.isnan(mini_delta).sum().item()
                num_inf = torch.isinf(mini_delta).sum().item()
                raise RuntimeError(
                    f"NaN/Inf in SimpleDeltaRefiner block {block_idx} FFN!\n"
                    f"  NaN: {num_nan}, Inf: {num_inf}\n"
                    f"  mini_delta range: [{mini_delta.min():.4f}, {mini_delta.max():.4f}]"
                )

            # Raw accumulation (NO gating!)
            accumulated_delta = accumulated_delta + mini_delta

        # Return raw accumulated delta (NO final gate!)
        return accumulated_delta
