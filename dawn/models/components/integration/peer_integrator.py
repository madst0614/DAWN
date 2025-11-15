"""
Peer Integrator - Gain Modulation Integration

Integrates deltas from self and peer experts through gain-based modulation.

Key principles:
- Self delta is primary (1.0 weight), normalized first for consistency
- Peer influences provide position-wise and dimension-wise gain modulation
- Each token position gets its own gain based on local context
- All operations in Self's representation space
- Post-LN architecture (PNN style) for stability

Scaling (Option 3 - Phase 1/2 consistency):
- Phase 1: self_delta / sqrt(D)
- Phase 2: (self_delta / sqrt(D)) * (1 + peer_gain)
Both phases use same base scaling for smooth transition!

Used by:
- DeltaExpert: Integrate self + peer-informed deltas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict


class PeerIntegrator(nn.Module):
    """
    Integrate self and peer deltas through gain modulation

    Self delta remains primary, peer influences provide
    position-wise and dimension-wise gain adjustments.

    Post-LN Architecture (PNN style):
    - Linear → GELU → Linear → LayerNorm → Tanh

    Option 3 Scaling (Phase 1/2 consistency):
    - Phase 1: self_delta / sqrt(D)
    - Phase 2: (self_delta / sqrt(D)) × (1 + peer_gain)
    where peer_gain is computed per-position and per-dimension

    Args:
        hidden_size: int
        max_sources: int - maximum number of delta sources (1 self + N peers)
        config: Dict with integration configs
    """

    def __init__(
        self,
        hidden_size: int,
        max_sources: int,
        config: Dict,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_sources = max_sources
        self.debug_mode = config.get('debug_mode', False)

        # Peer context → Gain field
        # Each peer proposes dimension-wise gain modulation
        # Post-LN architecture (PNN style)
        self.peer_to_gain = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, hidden_size),
                nn.LayerNorm(hidden_size),  # Post-LN: after computation
                nn.Tanh(),  # -1 ~ +1 bounded
            )
            for _ in range(max_sources - 1)  # Exclude self
        ])

        # Learnable blending weights for peer gains
        # Determines relative importance of each peer's gain
        self.peer_blend_weights = nn.Parameter(
            torch.zeros(max_sources - 1)
        )

        # Overall gain strength (how strongly peers influence)
        # Sigmoid-bounded to 0~1, then gain is -strength ~ +strength
        self.gain_strength = nn.Parameter(torch.tensor(0.2))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stability"""
        # Xavier init for gain projections
        for peer_gain in self.peer_to_gain:
            for module in peer_gain.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        h_current: torch.Tensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate self and peer deltas through gain modulation

        Args:
            h_current: [B, L, D] - Self's current state
            deltas: [num_sources, B, L, D]
                    deltas[0] = self_delta (from h_self)
                    deltas[1:] = peer_deltas (from interpreted_peer_h)

        Returns:
            modulated_delta: [B, L, D] - gain-modulated self delta
        """
        num_sources, B, L, D = deltas.shape

        # Input validation (debug mode only)
        if self.debug_mode:
            if not torch.isfinite(h_current).all():
                raise RuntimeError(
                    f"NaN/Inf in PeerIntegrator h_current! "
                    f"range: [{h_current.min():.4f}, {h_current.max():.4f}]"
                )
            if not torch.isfinite(deltas).all():
                raise RuntimeError(
                    f"NaN/Inf in PeerIntegrator deltas input! "
                    f"range: [{deltas.min():.4f}, {deltas.max():.4f}]"
                )

        # ✅ Phase 1: Single source (self only) → bypass
        if num_sources == 1:
            return deltas[0] / math.sqrt(D)

        # ✅ Phase 2+: Self + Peers
        self_delta = deltas[0]  # [B, L, D]
        peer_deltas = deltas[1:]  # [num_peers, B, L, D]
        num_peers = num_sources - 1

        # ✅ Compute peer influences (difference from self perspective)
        # This captures "what peers suggest differently"
        peer_influences = peer_deltas - self_delta.unsqueeze(0)
        # [num_peers, B, L, D]

        # ✅ Compute gain from each peer's influence (position-dependent)
        gains = []
        for i in range(num_peers):
            # Peer's influence combined with current state
            # Each position gets its own gain based on local context
            combined = h_current + peer_influences[i]  # [B, L, D]

            # Compute gain: -1 ~ +1 (position-wise, dimension-wise)
            gain = self.peer_to_gain[i](combined)  # [B, L, D]
            gains.append(gain)

        if not gains:
            # Should not happen, but safety check
            return self_delta / math.sqrt(D)

        gains = torch.stack(gains, dim=0)  # [num_peers, B, L, D]

        # ✅ Blend peer gains with learned weights
        blend_weights = F.softmax(
            self.peer_blend_weights[:num_peers], dim=0
        )  # [num_peers]

        # Weighted sum of gains
        final_gain = torch.einsum('pbld,p->bld', gains, blend_weights)  # [B, L, D]

        # ✅ Apply gain strength (sigmoid-bounded)
        strength = torch.sigmoid(self.gain_strength)  # 0 ~ 1
        bounded_gain = final_gain * strength  # [B, L, D]

        if self.debug_mode and not torch.isfinite(bounded_gain).all():
            raise RuntimeError(
                f"NaN/Inf in bounded_gain! "
                f"range: [{bounded_gain.min():.4f}, {bounded_gain.max():.4f}]"
            )

        # ✅ Option 3: Normalize self_delta first (consistent with Phase 1)
        # Phase 1: self_delta / sqrt(D)
        # Phase 2: (self_delta / sqrt(D)) * (1 + gain)
        # Both phases use same base scaling!
        self_delta_scaled = self_delta / math.sqrt(D)  # [B, L, D]

        # ✅ Apply peer gain to normalized self delta
        # Gain multiplier: 1.0 + bounded_gain
        # With strength=0.2: multiplier range is 0.8 ~ 1.2
        modulated = self_delta_scaled * (1.0 + bounded_gain)  # [B, L, D]

        if self.debug_mode and not torch.isfinite(modulated).all():
            print(f"❌ NaN in PeerIntegrator modulated delta!")
            print(f"  self_delta_scaled: [{self_delta_scaled.min():.4f}, {self_delta_scaled.max():.4f}]")
            print(f"  bounded_gain: [{bounded_gain.min():.4f}, {bounded_gain.max():.4f}]")
            print(f"  strength: {strength.item():.4f}")
            raise RuntimeError("NaN in PeerIntegrator")

        # ✅ No additional scaling (already normalized)
        return modulated
