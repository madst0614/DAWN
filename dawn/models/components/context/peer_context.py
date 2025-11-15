"""
Peer Context - Interpret Peer Outputs

Manages interpretation of peer expert outputs in current expert's context.

Used by: DeltaExpert (Phase 2)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List

from .context_interpreter import ContextInterpreter


class PeerContext(nn.Module):
    """
    Manage peer context interpretation

    Contains interpreters for all peers.
    Provides unified interface for peer context processing.

    Args:
        hidden_size: int
        peer_names: List[str] - names of peer experts
        config: Dict with num_heads, projection_rank, etc.
    """

    def __init__(
        self,
        hidden_size: int,
        peer_names: List[str],
        config: Dict,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.peer_names = sorted(peer_names)

        # Config
        num_heads = config.get('num_heads', 8)
        projection_rank = config.get('projection_rank', 256)

        # Create interpreter for each peer
        self.interpreters = nn.ModuleDict({
            name: ContextInterpreter(
                hidden_size=hidden_size,
                num_heads=num_heads,
                projection_rank=projection_rank,
                config=config,
            )
            for name in self.peer_names
        })

    def forward(
        self,
        h_self: torch.Tensor,
        peer_outputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Interpret all available peer contexts

        Args:
            h_self: [B, L, D] - my current state
            peer_outputs: {peer_name: [B, L, D]} - peer broadcasts
            attention_mask: [B, L] (True = masked)

        Returns:
            interpreted: List of [B, L, D] - interpreted peer contexts
        """
        interpreted = []

        for name in self.peer_names:
            if name in peer_outputs:
                h_peer = peer_outputs[name]
                h_interpreted = self.interpreters[name](
                    h_self=h_self,
                    h_other=h_peer,
                    attention_mask=attention_mask,
                )
                interpreted.append(h_interpreted)

        return interpreted
