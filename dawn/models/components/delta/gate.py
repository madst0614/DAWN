"""
Query-Key Gate - PNN-style Adaptive Gating

Query-key based gating that returns gate values (0~1)
for explicit multiplication with proposals.

Key principles:
- Separate query/key projections for explicit semantic matching
- Query (from h): "What refinements do I need?"
- Key (from proposal): "What refinements do I offer?"
- Element-wise compatibility with learnable temperature scaling
- Returns gate values, not gated output
- Caller performs explicit multiplication
- Bounded output for stability

Architecture:
    query = W_q(h)
    key = W_k(proposal)
    compatibility = query * key
    gate = sigmoid(compatibility / temperature)

Used by:
- DeltaRefiner: Mini-delta and final delta gating
- ExpertIntegrator: Final integration gating
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional


class QueryKeyGate(nn.Module):
    """
    PNN-style Query-Key Gate (DeltaValidator)

    Returns gate values in [0, 1] for explicit multiplication.
    Architecture: Separate query/key projections with temperature scaling

    Query (from h): "What refinements do I need?"
    Key (from proposal): "What refinements do I offer?"
    Compatibility: Element-wise matching between need and offer

    Usage:
        gate_values = gate(h, proposal)
        output = gate_values * proposal

    Args:
        hidden_size: int
        config: Optional[Dict] - gate configuration
            - gate_bias: bool (default: True)
            - temperature_init: float (default: sqrt(hidden_size))
    """

    def __init__(self, hidden_size: int, config: Optional[Dict] = None):
        super().__init__()

        self.hidden_size = hidden_size

        # Configuration
        config = config or {}
        self.use_bias = config.get('gate_bias', True)
        self.debug_mode = config.get('debug_mode', False)  # NaN checking only in debug mode
        temperature_init = config.get('temperature_init', math.sqrt(hidden_size))

        # Query: "What refinements do I need?"
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=self.use_bias)

        # Key: "What refinements do I offer?"
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=self.use_bias)

        # Learnable temperature for scaling
        self.temperature = nn.Parameter(torch.ones(1) * temperature_init)

        self._init_weights()

    def _init_weights(self):
        """Initialize for stability"""
        # Xavier uniform for both projections
        nn.init.xavier_uniform_(self.query_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.key_proj.weight, gain=1.0)

        # Zero bias (neutral start)
        if self.query_proj.bias is not None:
            nn.init.zeros_(self.query_proj.bias)
        if self.key_proj.bias is not None:
            nn.init.zeros_(self.key_proj.bias)

    def forward(self, h: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
        """
        Compute sigmoid gate values via query-key matching

        Args:
            h: [B, L, D] - current representation (query source)
            proposal: [B, L, D] - proposed update (key source)

        Returns:
            gate: [B, L, D] - gate values in [0, 1]

        Note: Caller multiplies:
            output = gate(h, proposal) * proposal
        """
        # Compute query and key
        query = self.query_proj(h)              # [B, L, D] - "What do I need?"
        key = self.key_proj(proposal)           # [B, L, D] - "What can I provide?"

        # Element-wise compatibility
        compatibility = query * key             # [B, L, D]

        # Temperature-scaled sigmoid for gate values
        gate = torch.sigmoid(compatibility / self.temperature)  # [B, L, D]

        # Debug mode: Only check for NaN (detailed stats saved to file by train script)
        if self.debug_mode and not torch.isfinite(gate).all():
            print(f"\n🚨 NaN detected in QueryKeyGate!")
            print(f"  h range: [{h.min():.4f}, {h.max():.4f}]")
            print(f"  proposal range: [{proposal.min():.4f}, {proposal.max():.4f}]")
            print(f"  query range: [{query.min():.4f}, {query.max():.4f}]")
            print(f"  key range: [{key.min():.4f}, {key.max():.4f}]")
            print(f"  compatibility range: [{compatibility.min():.4f}, {compatibility.max():.4f}]")
            print(f"  temperature: {self.temperature.item():.4f}")
            print(f"  scaled compat range: [{(compatibility/self.temperature).min():.4f}, {(compatibility/self.temperature).max():.4f}]")
            print(f"  gate range: [{gate.min():.4f}, {gate.max():.4f}]")
            print(f"  Query bias max: {self.query_proj.bias.abs().max():.4f}")
            print(f"  Key bias max: {self.key_proj.bias.abs().max():.4f}")
            raise RuntimeError("NaN in QueryKeyGate - see diagnostics above")

        return gate

    def extra_repr(self) -> str:
        return (f'hidden_size={self.hidden_size}, bias={self.use_bias}, '
                f'temperature={self.temperature.item():.2f}')
