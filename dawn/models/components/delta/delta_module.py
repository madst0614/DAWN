"""
Delta Module - Universal Delta Generator

Clean interface for delta generation.
Wraps DeltaRefiner with simple API.

Used by:
- DeltaExpert: Process self and interpreted peers
- ExpertIntegrator: Process base and interpreted experts
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .delta_refiner import DeltaRefiner
from .simple_delta_refiner import SimpleDeltaRefiner


class DeltaModule(nn.Module):
    """
    Generate refined delta from any input

    Universal processor that can handle:
    - Self state
    - Interpreted peer context
    - Interpreted expert context

    Args:
        hidden_size: int
        num_blocks: int - number of refinement blocks
        num_heads: int
        intermediate_size: int or List[int]
        dropout: float
        config: Dict - must contain 'refiner' with 'type' key
                     - 'type': 'gated' (default) or 'simple'
    """

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int,
        num_heads: int,
        intermediate_size,  # int or List[int]
        dropout: float,
        config: Dict,
    ):
        super().__init__()

        refiner_config = config.get('refiner', {})
        refiner_type = refiner_config.get('type', 'gated')

        if refiner_type == 'simple':
            # SimpleDeltaRefiner: Multi-block without mini-gates
            self.refiner = SimpleDeltaRefiner(
                hidden_size=hidden_size,
                num_blocks=num_blocks,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                config=refiner_config,
            )
        elif refiner_type == 'gated':
            self.refiner = DeltaRefiner(
                hidden_size=hidden_size,
                num_blocks=num_blocks,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                config=refiner_config,
            )
        else:
            raise ValueError(f"Unknown refiner type: {refiner_type}. Use 'gated' or 'simple'")

    def forward(
        self,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate delta from input

        Args:
            h: [B, L, D] - any source input
            attention_mask: [B, L] (True = masked)

        Returns:
            delta: [B, L, D]
        """
        delta = self.refiner(h, attention_mask)

        return delta
