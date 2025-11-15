"""
GeGLU Gate - Stabilized Gated Linear Unit for Recurrent Delta Refinement

Advanced gating mechanism optimized for recurrent mini-delta accumulation.
Balances expressiveness (value transformation) with stability (bounded gates).

Key features:
- Value normalization: prevents amplification in recurrent accumulation
- SiLU (Swish) gates: smooth, semi-bounded (used in LLaMA 3)
- Value transformation: maintains expressiveness (GeGLU advantage)
- Balanced expressiveness vs stability

Architecture:
    1. Combine h and delta_raw
    2. Project to value and gate
    3. Normalize value (KEY for stability!)
    4. SiLU gate (x * sigmoid(x))
    5. Gated output with learnable scale

Reference:
- "GLU Variants Improve Transformer" (Shazeer, 2020)
- LLaMA 3 (SwiGLU variant)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class GeGLUGate(nn.Module):
    """
    Stabilized GeGLU-based gating for recurrent delta refinement

    Balances expressiveness (value transformation) with stability (bounded gates).

    Architecture:
        1. Combine h and delta_raw → [B, L, 2D]
        2. Project to hidden representation → [B, L, D]
        3. Project to value and gate → [B, L, 2D]
        4. Split into value and gate_input → [B, L, D] each
        5. Normalize value (prevents amplification!)
        6. SiLU gate (x * sigmoid(x)) - semi-bounded
        7. Gated output with learnable scale

    Key stability features:
        - Value normalization: prevents recurrent amplification
        - SiLU gates: smooth semi-bounded activation (x * sigmoid(x))
        - Learnable output scale: balances stability with expressiveness

    Args:
        hidden_size: int - dimension of hidden states
        config: Optional[Dict] - configuration options
            - gate_bias: bool (default True) - use bias in linear layers
            - gate_dropout: float (default 0.0) - dropout rate
            - gate_output_scale_init: float (default 0.15) - initial output scale
            - enable_detailed_logging: bool (default False) - verbose logging

    Example:
        gate = GeGLUGate(hidden_size=768)
        output = gate(h, delta_raw)  # [B, L, D] → [B, L, D]
    """

    def __init__(self, hidden_size: int, config: Optional[Dict] = None):
        super().__init__()

        self.hidden_size = hidden_size

        # Configuration
        config = config or {}
        self.use_bias = config.get('gate_bias', True)
        self.dropout_rate = config.get('gate_dropout', 0.0)
        self.detailed_logging = config.get('enable_detailed_logging', False)
        output_scale_init = config.get('gate_output_scale_init', 0.15)

        # Input combination: [h, delta] → hidden representation
        self.input_proj = nn.Linear(
            hidden_size * 2,  # Concatenated h and delta
            hidden_size,
            bias=self.use_bias
        )

        # GLU projection: hidden → [value, gate] (2x for split)
        self.gate_proj = nn.Linear(
            hidden_size,
            hidden_size * 2,  # Will be split into value and gate
            bias=self.use_bias
        )

        # Value normalization (KEY for stability in recurrent accumulation!)
        # Prevents value amplification across multiple blocks/steps
        self.value_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # Learnable output scale (initialized from config)
        # Unrestricted to allow maximum expressiveness (stability from value norm + SiLU)
        self.output_scale = nn.Parameter(torch.tensor(output_scale_init))

        # Optional dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = None

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for stability

        Uses Xavier uniform initialization with gain=1.0
        This provides balanced initialization without special scaling
        """
        # Xavier uniform for both projections
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=1.0)

        # Zero bias initialization
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        if self.gate_proj.bias is not None:
            nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        h: torch.Tensor,
        delta_raw: torch.Tensor,
        debug: bool = False
    ) -> torch.Tensor:
        """
        Compute gated output with stabilized value-gate mechanism

        Args:
            h: [B, L, D] - current hidden state (query context)
            delta_raw: [B, L, D] - raw delta from refiner (key/value)
            debug: bool - return debug info (default False)

        Returns:
            output: [B, L, D] - gated values ready for integration
            debug_info: Optional[Dict] - statistics (if debug=True)

        Mathematical formulation:
            combined = concat(h, delta_raw)              # [B, L, 2D]
            hidden = input_proj(combined)                # [B, L, D]
            value, gate_input = split(gate_proj(hidden)) # [B, L, D] each
            value_normed = LayerNorm(value)              # [B, L, D] - KEY!
            gate = silu(gate_input)                      # [B, L, D] - semi-bounded
            output = value_normed * gate * scale         # Learnable scale
        """
        B, L, D = h.shape

        # Sanity check
        assert delta_raw.shape == h.shape, \
            f"Shape mismatch: h={h.shape}, delta_raw={delta_raw.shape}"

        # 1. Combine inputs: concatenate h and delta
        combined = torch.cat([h, delta_raw], dim=-1)  # [B, L, 2D]

        # 2. Project to hidden representation
        # This allows the gate to learn complex relationships between h and delta
        hidden = self.input_proj(combined)  # [B, L, D]

        # 3. Project to value and gate space
        projected = self.gate_proj(hidden)  # [B, L, 2D]

        # 4. Split into value and gate components
        value, gate_input = projected.chunk(2, dim=-1)  # [B, L, D] each

        # 5. NORMALIZE VALUE (KEY for stability!)
        # Prevents amplification in recurrent accumulation
        value_normed = self.value_norm(value)

        # 6. SiLU (Swish) gate: x * sigmoid(x)
        # Semi-bounded, smooth activation (LLaMA 3 style)
        # Better gradient flow than pure sigmoid, more bounded than GELU
        gate = F.silu(gate_input)

        # 7. Element-wise gating
        # Value (normalized) modulated by gate (semi-bounded)
        gated = value_normed * gate  # [B, L, D]

        # 8. Apply dropout if configured (only during training)
        if self.dropout is not None and self.training:
            gated = self.dropout(gated)

        # 9. Apply learnable output scale (unrestricted for expressiveness)
        output = gated * self.output_scale

        # Logging for monitoring (minimal by default)
        if self.detailed_logging:
            print(f"[GeGLUGate] value_normed=[{value_normed.min():.2f}, {value_normed.max():.2f}], "
                  f"gate=[{gate.min():.2f}, {gate.max():.2f}] (mean={gate.mean():.3f}), "
                  f"output=[{output.min():.2f}, {output.max():.2f}], "
                  f"scale={self.output_scale.item():.4f}")

        # Debug mode: return statistics
        if debug:
            with torch.no_grad():
                debug_info = {
                    'hidden_range': (hidden.min().item(), hidden.max().item()),
                    'value_range': (value.min().item(), value.max().item()),
                    'value_normed_range': (value_normed.min().item(), value_normed.max().item()),
                    'gate_range': (gate.min().item(), gate.max().item()),
                    'gated_range': (gated.min().item(), gated.max().item()),
                    'output_range': (output.min().item(), output.max().item()),
                    'gate_mean': gate.mean().item(),
                    'gate_std': gate.std().item(),
                    'output_mean': output.mean().item(),
                    'output_std': output.std().item(),
                    'output_scale': self.output_scale.item(),
                }
            return output, debug_info

        return output

    def extra_repr(self) -> str:
        """String representation for debugging"""
        return (f'hidden_size={self.hidden_size}, '
                f'bias={self.use_bias}, '
                f'dropout={self.dropout_rate}')
