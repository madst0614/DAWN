"""
Expert Context - Interpret Expert Outputs

Manages interpretation of expert outputs in base expert's context.

Used by: ExpertIntegrator (Phase 2, model-level)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List

from .context_interpreter import ContextInterpreter


class ExpertContext(nn.Module):
    """
    Manage expert context interpretation

    Contains interpreters for all non-base experts.
    Interprets their outputs in base expert's coordinate system.

    Args:
        hidden_size: int
        expert_names: List[str] - names of non-base experts
        config: Dict with num_heads, projection_rank, etc.
    """

    def __init__(
        self,
        hidden_size: int,
        expert_names: List[str],
        config: Dict,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.expert_names = sorted(expert_names)

        # Config
        num_heads = config.get('num_heads', 8)
        projection_rank = config.get('projection_rank', 256)

        # Create interpreter for each expert
        self.interpreters = nn.ModuleDict({
            name: ContextInterpreter(
                hidden_size=hidden_size,
                num_heads=num_heads,
                projection_rank=projection_rank,
                config=config,
            )
            for name in self.expert_names
        })

    def forward(
        self,
        h_base: torch.Tensor,
        expert_outputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Interpret all available expert contexts

        Args:
            h_base: [B, L, D] - base expert's state
            expert_outputs: {expert_name: [B, L, D]} - other experts
            attention_mask: [B, L] (True = masked)

        Returns:
            interpreted: List of [B, L, D] - interpreted expert contexts
        """
        interpreted = []

        for name in self.expert_names:
            if name in expert_outputs:
                h_expert = expert_outputs[name]
                h_interpreted = self.interpreters[name](
                    h_self=h_base,
                    h_other=h_expert,
                    attention_mask=attention_mask,
                )
                interpreted.append(h_interpreted)

        return interpreted
