"""
DAWN - Decomposable Architecture With Neural Networks

Multi-expert coordination with Peer Context architecture.

Phase 1: Experts train independently (no peer context)
Phase 2: Experts collaborate via peer context + final integration

Container for:
- Multiple DeltaExperts (shared embeddings)
- ExpertIntegrator (Phase 2 only)
- TaskHeads (shared across phases)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List

from .expert import DeltaExpert
from .integrator import ExpertIntegrator


class DAWN(nn.Module):
    """
    DAWN: Multi-Expert Container

    Args:
        config: Model configuration dictionary
        enable_peer_prediction: Enable peer context (Phase 2)
        active_experts: Subset of experts to create (None = all)
    """

    def __init__(
        self,
        config: Dict,
        enable_peer_prediction: bool = False,
        active_experts: Optional[List[str]] = None,
    ):
        super().__init__()

        self.config = config
        self.expert_names = config["expert_names"]
        self.hidden_size = config["hidden_size"]
        self.enable_peer_prediction = enable_peer_prediction

        # Create shared embeddings for all experts
        self.shared_embeddings = nn.ModuleDict({
            "token": nn.Embedding(config["vocab_size"], config["hidden_size"]),
            "position": nn.Embedding(config["max_length"], config["hidden_size"]),
            "layer_norm": nn.LayerNorm(config["hidden_size"]),
            "dropout": nn.Dropout(config["dropout"]),
        })

        # Initialize shared embeddings
        init_std = config.get('init_std', 0.02)
        nn.init.normal_(self.shared_embeddings["token"].weight, mean=0.0, std=init_std)
        nn.init.normal_(self.shared_embeddings["position"].weight, mean=0.0, std=init_std)

        # Initialize LayerNorm explicitly (usually defaults to 1 and 0, but be safe)
        nn.init.ones_(self.shared_embeddings["layer_norm"].weight)
        nn.init.zeros_(self.shared_embeddings["layer_norm"].bias)

        # Determine which experts to create
        experts_to_create = active_experts if active_experts else self.expert_names

        # Create experts with shared embeddings
        self.experts = nn.ModuleDict()
        for name in experts_to_create:
            # Phase 2: provide peer names for peer context
            peer_names = (
                [n for n in self.expert_names if n != name]
                if enable_peer_prediction
                else None
            )

            self.experts[name] = DeltaExpert(
                config=config,
                peer_names=peer_names,
                shared_embeddings=self.shared_embeddings
            )

        # Expert Integrator (Phase 2 only)
        if enable_peer_prediction:
            integrator_config = config.get('expert_integrator', {})
            base_expert = integrator_config.get('base_expert', experts_to_create[0])

            self.expert_integrator = ExpertIntegrator(
                hidden_size=self.hidden_size,
                expert_names=list(experts_to_create),
                base_expert=base_expert,
                config=integrator_config,
            )
        else:
            self.expert_integrator = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        active_expert: Optional[str] = None,
        return_expert_outputs: bool = False,
        peer_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Forward pass through expert(s)

        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            active_expert: Specific expert (Phase 1), None = all (Phase 2)
            return_expert_outputs: Return dict with all expert outputs
            peer_outputs: {expert_name: output} - for Phase 2

        Returns:
            output: [B, L, D] or dict of expert outputs
        """
        # Phase 1: Single active expert
        if active_expert is not None:
            if active_expert not in self.experts:
                raise ValueError(f"Expert '{active_expert}' not found in model")

            output = self.experts[active_expert](
                input_ids=input_ids,
                attention_mask=attention_mask,
                peer_outputs=None,  # Phase 1: no peers
            )

            if return_expert_outputs:
                return {"expert_outputs": {active_expert: output}, "integrated": output}
            return output

        # Phase 2: All experts (with peer context and integration)

        # Convert attention mask for internal use
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask == 0  # True = masked

        expert_outputs = {}

        # Run all experts in parallel (in practice, sequentially here)
        for name in self.experts.keys():
            # Get peer outputs (excluding self)
            peer_outputs_for_expert = {
                peer_name: peer_out
                for peer_name, peer_out in (peer_outputs or {}).items()
                if peer_name != name
            } if peer_outputs else None

            output = self.experts[name](
                input_ids=input_ids,
                attention_mask=attention_mask,
                peer_outputs=peer_outputs_for_expert,
            )

            expert_outputs[name] = output

        # Integrate experts (Phase 2)
        if self.expert_integrator is not None:
            integrated = self.expert_integrator(
                expert_outputs=expert_outputs,
                attention_mask=attn_mask,
            )
        else:
            # Fallback: use first expert
            integrated = expert_outputs[list(expert_outputs.keys())[0]]

        if return_expert_outputs:
            return {
                "expert_outputs": expert_outputs,
                "integrated": integrated
            }

        return integrated

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "expert_names": self.expert_names,
            "active_experts": list(self.experts.keys()),
            "all_experts": self.expert_names,
            "hidden_size": self.hidden_size,
            "vocab_size": self.config["vocab_size"],
            "shared_embeddings": True,
            "has_peer_prediction": self.enable_peer_prediction,
            "has_expert_integrator": self.expert_integrator is not None,
        }
