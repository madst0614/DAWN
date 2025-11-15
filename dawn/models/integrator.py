"""
Expert Integrator - Multi-Expert Integration with Cross-Attention

Integrates multiple expert outputs through:
1. Cross-expert attention (position-wise dynamic weighting)
2. Per-expert gating (dimension-wise selective integration)
3. Weighted combination (learnable blending)
4. Final gating and PNN update

Philosophy: h_final = h_base + gated_delta
where delta combines attention-weighted expert contributions

Used by: DAWN Phase 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .components.context.expert_context import ExpertContext
from .components.delta.delta_module import DeltaModule
from .components.delta.gate import QueryKeyGate


class CrossExpertAttention(nn.Module):
    """
    Attend across expert deltas for position-wise dynamic weighting

    Given current state, determines which expert is most relevant
    at each position through attention mechanism.

    Args:
        hidden_size: int
        num_heads: int
        dropout: float
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Multi-head attention across expert deltas
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_current: torch.Tensor,
        expert_deltas: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Attend across expert deltas

        Args:
            h_current: Current state [B, L, D]
            expert_deltas: All expert deltas [num_experts, B, L, D]
            attention_mask: [B, L] (True = masked)

        Returns:
            attended_delta: Weighted combination [B, L, D]
        """
        num_experts, B, L, D = expert_deltas.shape

        # Reshape: treat experts as sequence dimension
        # [num_experts, B, L, D] → [B, L, num_experts, D] → [B*L, num_experts, D]
        deltas_reshaped = expert_deltas.permute(1, 2, 0, 3).reshape(
            B * L, num_experts, D
        )

        # Query from current state: [B, L, D] → [B*L, 1, D]
        query = h_current.reshape(B * L, 1, D)

        # Key/Value from expert deltas
        key = value = deltas_reshaped

        # Attention: which expert to focus on at each position
        attended, _ = self.attention(
            query, key, value,
            need_weights=False,
        )  # [B*L, 1, D]

        # Reshape back: [B*L, 1, D] → [B, L, D]
        attended = attended.squeeze(1).reshape(B, L, D)

        # Residual + norm
        attended = self.layer_norm(h_current + self.dropout(attended))

        return attended


class ExpertBlender(nn.Module):
    """
    Learnable blending of expert deltas

    Uses learnable weights to blend expert deltas with dimension-wise control.

    Args:
        hidden_size: int
        num_experts: int
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts

        # Learnable blend weights [num_experts]
        self.blend_weights = nn.Parameter(
            torch.zeros(num_experts)
        )

        # Optional: dimension-wise modulation
        self.dimension_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        h_current: torch.Tensor,
        expert_deltas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend expert deltas

        Args:
            h_current: Current state [B, L, D]
            expert_deltas: Gated expert deltas [num_experts, B, L, D]

        Returns:
            blended_delta: [B, L, D]
        """
        num_experts, B, L, D = expert_deltas.shape

        # Base blend weights
        base_weights = F.softmax(self.blend_weights, dim=0)  # [num_experts]

        # Context-dependent modulation
        context = h_current.mean(dim=1)  # [B, D]
        context_weights = self.dimension_modulation(context)  # [B, num_experts]

        # Combine: base (global) + context (local)
        # base_weights: [num_experts]
        # context_weights: [B, num_experts]
        combined_weights = base_weights.unsqueeze(0) * context_weights  # [B, num_experts]
        combined_weights = combined_weights / (
            combined_weights.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Blend deltas
        # expert_deltas: [num_experts, B, L, D]
        # combined_weights: [B, num_experts]
        blended = torch.einsum('nbld,bn->bld', expert_deltas, combined_weights)

        return blended


class ExpertIntegrator(nn.Module):
    """
    Multi-Expert Integration with Cross-Attention

    Combines base expert with other experts through:
    1. Expert context interpretation
    2. Delta generation per expert
    3. Per-expert gating (dimension-wise)
    4. Cross-expert attention (position-wise)
    5. Learnable blending (weighted combination)
    6. Final gating and PNN update

    Args:
        hidden_size: int
        expert_names: List[str] - all expert names
        base_expert: str - which expert to use as base
        config: Dict
    """

    def __init__(
        self,
        hidden_size: int,
        expert_names: List[str],
        base_expert: str,
        config: Dict,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.base_expert = base_expert
        self.other_experts = [e for e in expert_names if e != base_expert]
        self.num_experts = len(expert_names)

        # Expert Context Interpreter
        expert_context_config = config.get('expert_context', {})
        self.expert_context = ExpertContext(
            hidden_size=hidden_size,
            expert_names=self.other_experts,
            config=expert_context_config,
        )

        # Delta Module (shared for all sources)
        delta_config = config.get('delta_module', {})
        self.delta_module = DeltaModule(
            hidden_size=hidden_size,
            num_blocks=delta_config.get('num_blocks', 3),
            num_heads=delta_config.get('num_heads', 8),
            intermediate_size=delta_config.get('intermediate_size', hidden_size * 4),
            dropout=delta_config.get('dropout', 0.1),
            config=delta_config,
        )

        # Per-expert gates (dimension-wise selective integration)
        gate_config = config.get('gate', {})
        self.per_expert_gates = nn.ModuleList([
            QueryKeyGate(hidden_size, gate_config)
            for _ in range(self.num_experts)
        ])

        # Cross-Expert Attention (position-wise dynamic weighting)
        attention_config = config.get('cross_attention', {})
        self.use_cross_attention = attention_config.get('enabled', True)
        if self.use_cross_attention:
            self.cross_attention = CrossExpertAttention(
                hidden_size=hidden_size,
                num_heads=attention_config.get('num_heads', 8),
                dropout=attention_config.get('dropout', 0.1),
            )

        # Expert Blender (learnable weighted combination)
        self.use_blending = config.get('use_blending', True)
        if self.use_blending:
            self.expert_blender = ExpertBlender(
                hidden_size=hidden_size,
                num_experts=self.num_experts,
            )

        # Final gate
        self.final_gate = QueryKeyGate(hidden_size, gate_config)

    def forward(
        self,
        expert_outputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-expert integration

        Args:
            expert_outputs: {expert_name: [B, L, D]}
            attention_mask: [B, L] (True = masked)

        Returns:
            h_final: [B, L, D]
        """
        # Base expert
        h_base = expert_outputs[self.base_expert]

        # Other experts (filter available ones)
        other_outputs = {
            name: output
            for name, output in expert_outputs.items()
            if name in self.other_experts
        }

        # 1. Interpret other experts in base context
        if other_outputs:
            interpreted = self.expert_context(
                h_base=h_base,
                expert_outputs=other_outputs,
                attention_mask=attention_mask,
            )
        else:
            interpreted = []

        # 2. Generate deltas per expert
        delta_base = self.delta_module(h_base, attention_mask)
        delta_others = [
            self.delta_module(h, attention_mask)
            for h in interpreted
        ]
        all_deltas = [delta_base] + delta_others

        # 3. Apply per-expert gates (dimension-wise)
        gated_deltas = []
        for i, delta in enumerate(all_deltas):
            gate_values = self.per_expert_gates[i](h_base, delta)
            gated_delta = gate_values * delta
            gated_deltas.append(gated_delta)

        # Stack: [num_experts, B, L, D]
        gated_deltas_stacked = torch.stack(gated_deltas, dim=0)

        # 4. Cross-expert attention (position-wise)
        if self.use_cross_attention:
            attended_delta = self.cross_attention(
                h_current=h_base,
                expert_deltas=gated_deltas_stacked,
                attention_mask=attention_mask,
            )
        else:
            attended_delta = gated_deltas_stacked.mean(dim=0)

        # 5. Learnable blending (weighted combination)
        if self.use_blending:
            blended_delta = self.expert_blender(
                h_current=h_base,
                expert_deltas=gated_deltas_stacked,
            )
        else:
            blended_delta = gated_deltas_stacked.mean(dim=0)

        # 6. Combine attention and blending
        # Attention: position-wise dynamic weighting
        # Blending: learnable weighted combination
        combined_delta = (attended_delta + blended_delta) / 2.0

        # 7. Final gating (overall acceptance)
        final_gate_values = self.final_gate(h_base, combined_delta)
        final_delta = final_gate_values * combined_delta

        # 8. PNN update
        h_final = h_base + final_delta

        return h_final
