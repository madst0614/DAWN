"""
Context Router for Firing Pattern Weights.

Computes weights for combining neuron firing patterns
based on input context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ContextRouter(nn.Module):
    """
    Context-based router that computes pattern combining weights.

    Given input and selected neurons, computes:
    1. Which neurons to select (top-k)
    2. How much weight to give each selected neuron's pattern
    """

    def __init__(
        self,
        d_model: int,
        pool_size: int,
        k: int,
        temperature: float = 1.0
    ):
        """
        Initialize context router.

        Args:
            d_model: Model dimension
            pool_size: Number of neurons in pool
            k: Number of neurons to select
            temperature: Temperature for Gumbel-Softmax (default: 1.0)
        """
        super().__init__()
        self.d_model = d_model
        self.pool_size = pool_size
        self.k = k
        self.temperature = temperature

        # Selection network: which neurons to use
        self.selection_query = nn.Linear(d_model, pool_size)

        # Weighting network: how to combine selected patterns
        self.weight_query = nn.Linear(d_model, pool_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute neuron selection and pattern weights.

        Args:
            x: Input tensor [batch, seq, d_model]

        Returns:
            selected_indices: Selected neuron indices [batch, seq, k]
            pattern_weights: Weights for combining patterns [batch, seq, k]
            selection_scores: Raw selection scores [batch, seq, pool_size]
        """
        # 1. Compute selection scores [batch, seq, pool_size]
        selection_scores = self.selection_query(x)

        if self.training:
            # Add Gumbel noise for exploration
            gumbel = -torch.log(-torch.log(torch.rand_like(selection_scores) + 1e-10) + 1e-10)
            selection_scores = (selection_scores + gumbel) / self.temperature

        # 2. Select top-k neurons [batch, seq, k]
        _, selected_indices = torch.topk(selection_scores, self.k, dim=-1)

        # 3. Compute pattern combining weights [batch, seq, pool_size]
        weight_scores = self.weight_query(x)

        # Get weights for selected neurons only
        # Gather weights for selected indices
        pattern_weights = torch.gather(weight_scores, -1, selected_indices)
        # [batch, seq, k]

        return selected_indices, pattern_weights, selection_scores

    def get_selection_entropy(self, selection_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute selection entropy (diversity measure).

        Args:
            selection_scores: Raw selection scores [batch, seq, pool_size]

        Returns:
            Entropy per token [batch, seq]
        """
        probs = F.softmax(selection_scores, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy

    def get_weight_distribution_stats(
        self,
        pattern_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get statistics about pattern weight distribution.

        Args:
            pattern_weights: Pattern weights [batch, seq, k]

        Returns:
            mean_max_weight: Average of max weights
            weight_entropy: Entropy of weight distribution
        """
        # Normalize weights
        normalized_weights = F.softmax(pattern_weights, dim=-1)

        # Max weight (how dominant is the top pattern?)
        max_weights = normalized_weights.max(dim=-1)[0]
        mean_max_weight = max_weights.mean()

        # Entropy (how spread out are the weights?)
        weight_entropy = -torch.sum(
            normalized_weights * torch.log(normalized_weights + 1e-10),
            dim=-1
        ).mean()

        return mean_max_weight, weight_entropy


class AdaptiveContextRouter(nn.Module):
    """
    Adaptive router that adjusts k based on input complexity.

    For simple inputs: use fewer neurons
    For complex inputs: use more neurons
    """

    def __init__(
        self,
        d_model: int,
        pool_size: int,
        k_min: int = 32,
        k_max: int = 256,
        temperature: float = 1.0
    ):
        """
        Initialize adaptive context router.

        Args:
            d_model: Model dimension
            pool_size: Number of neurons
            k_min: Minimum neurons to select (default: 32)
            k_max: Maximum neurons to select (default: 256)
            temperature: Gumbel-Softmax temperature (default: 1.0)
        """
        super().__init__()
        self.d_model = d_model
        self.pool_size = pool_size
        self.k_min = k_min
        self.k_max = k_max
        self.temperature = temperature

        # Selection network
        self.selection_query = nn.Linear(d_model, pool_size)

        # Weighting network
        self.weight_query = nn.Linear(d_model, pool_size)

        # Complexity estimator: decides how many neurons to use
        self.complexity_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward with adaptive k selection.

        Args:
            x: Input tensor [batch, seq, d_model]

        Returns:
            selected_indices: Variable-length selected indices (padded)
            pattern_weights: Pattern weights (padded)
            selection_scores: Raw scores
            k_values: Number of neurons used per token [batch, seq]
        """
        batch, seq, _ = x.shape

        # 1. Estimate complexity and determine k [batch, seq]
        complexity = self.complexity_net(x).squeeze(-1)  # [batch, seq]

        # k = k_min + complexity * (k_max - k_min)
        k_values = (self.k_min + complexity * (self.k_max - self.k_min)).long()

        # 2. Compute selection scores
        selection_scores = self.selection_query(x)

        if self.training:
            gumbel = -torch.log(-torch.log(torch.rand_like(selection_scores) + 1e-10) + 1e-10)
            selection_scores = (selection_scores + gumbel) / self.temperature

        # 3. Select neurons (use max k for now, mask later)
        k_max_used = self.k_max
        _, selected_indices = torch.topk(selection_scores, k_max_used, dim=-1)

        # 4. Compute pattern weights
        weight_scores = self.weight_query(x)
        pattern_weights = torch.gather(weight_scores, -1, selected_indices)

        # 5. Create mask for actual k values
        # positions: [batch, seq, k_max]
        positions = torch.arange(k_max_used, device=x.device).unsqueeze(0).unsqueeze(0)
        positions = positions.expand(batch, seq, -1)

        # mask: True where position < k_values
        mask = positions < k_values.unsqueeze(-1)

        # Zero out weights beyond k_values
        pattern_weights = pattern_weights * mask.float()

        return selected_indices, pattern_weights, selection_scores, k_values
