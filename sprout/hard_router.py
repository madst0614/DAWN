"""
Hard Router for SPROUT.

Each token selects k neurons from the pool using hard routing
with Gumbel-Softmax for differentiability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HardRouter(nn.Module):
    """
    Hard router that selects k neurons for each token.

    Uses Gumbel-Softmax trick during training for differentiability.
    """

    def __init__(self, d_model: int, pool_size: int, k: int, temperature: float = 1.0):
        """
        Initialize hard router.

        Args:
            d_model: Model dimension
            pool_size: Total number of neurons in pool
            k: Number of neurons to select per token
            temperature: Gumbel-Softmax temperature (default: 1.0)
        """
        super().__init__()
        self.k = k
        self.pool_size = pool_size
        self.temperature = temperature

        # Query network: maps token to neuron scores
        self.query = nn.Linear(d_model, pool_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select k neurons for each token.

        Args:
            x: Input tensor [batch, seq, d_model]

        Returns:
            indices: Selected neuron indices [batch, seq, k]
            scores: Routing scores [batch, seq, pool_size]
        """
        # Compute routing scores [batch, seq, pool_size]
        scores = self.query(x)

        if self.training:
            # Add Gumbel noise for exploration during training
            gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
            scores = (scores + gumbel) / self.temperature

        # Select top-k neurons [batch, seq, k]
        _, indices = torch.topk(scores, self.k, dim=-1)

        return indices, scores

    def get_routing_entropy(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate routing entropy to measure diversity.

        Args:
            scores: Routing scores [batch, seq, pool_size]

        Returns:
            Entropy per token [batch, seq]
        """
        probs = F.softmax(scores, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy

    def get_load_balance_loss(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Calculate load balance loss to encourage even neuron usage.

        Args:
            indices: Selected neuron indices [batch, seq, k]

        Returns:
            Load balance loss (scalar)
        """
        batch, seq, k = indices.shape

        # Count usage of each neuron
        usage = torch.zeros(self.pool_size, device=indices.device)
        flat_indices = indices.view(-1)
        usage.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))

        # Normalize to get distribution
        usage_dist = usage / (batch * seq * k)

        # Ideal uniform distribution
        uniform_dist = 1.0 / self.pool_size

        # KL divergence from uniform
        kl_div = torch.sum(usage_dist * torch.log((usage_dist + 1e-10) / uniform_dist))

        return kl_div
