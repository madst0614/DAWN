"""
Learned Neuron Pool with Firing Patterns.

Each neuron has a learned "firing pattern" - a unique signature
that represents what this neuron specializes in.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LearnedNeuronPool(nn.Module):
    """
    Pool of neurons, each with a learned firing pattern.

    Each neuron has:
    - firing_pattern: [d_ff] - learned signature
    - This pattern represents the neuron's specialization

    Patterns are linearly combined based on context,
    then activated with GELU once.
    """

    def __init__(self, pool_size: int, d_model: int, d_ff: int):
        """
        Initialize learned neuron pool.

        Args:
            pool_size: Number of neurons in pool
            d_model: Model dimension
            d_ff: Firing pattern dimension (hidden size)
        """
        super().__init__()
        self.pool_size = pool_size
        self.d_model = d_model
        self.d_ff = d_ff

        # Each neuron's unique firing pattern [pool_size, d_ff]
        self.firing_patterns = nn.Parameter(torch.randn(pool_size, d_ff))

        # Shared W2: maps from firing pattern space to model space
        self.W2 = nn.Linear(d_ff, d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize firing patterns with small random values
        nn.init.normal_(self.firing_patterns, std=0.02)

        # Initialize W2
        nn.init.xavier_uniform_(self.W2.weight)
        if self.W2.bias is not None:
            nn.init.zeros_(self.W2.bias)

    def forward(
        self,
        selected_indices: torch.Tensor,
        pattern_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine firing patterns and produce output.

        Args:
            selected_indices: Selected neuron indices [batch, seq, k]
            pattern_weights: Weights for combining patterns [batch, seq, k]

        Returns:
            Output tensor [batch, seq, d_model]
        """
        batch, seq, k = selected_indices.shape

        # Get firing patterns for selected neurons [batch, seq, k, d_ff]
        selected_patterns = self.firing_patterns[selected_indices]

        # Normalize weights to sum to 1
        weights = F.softmax(pattern_weights, dim=-1)  # [batch, seq, k]

        # Weighted combination of patterns [batch, seq, d_ff]
        # Each position gets a unique combination based on weights
        combined_pattern = torch.einsum('bsk,bskf->bsf', weights, selected_patterns)

        # Apply GELU activation to combined pattern (only once!)
        activated = F.gelu(combined_pattern)  # [batch, seq, d_ff]

        # Project back to model space [batch, seq, d_model]
        output = self.W2(activated)

        return output

    def get_pattern_similarity(self, neuron_i: int, neuron_j: int) -> float:
        """
        Compute cosine similarity between two neuron patterns.

        Args:
            neuron_i: First neuron index
            neuron_j: Second neuron index

        Returns:
            Cosine similarity [-1, 1]
        """
        pattern_i = self.firing_patterns[neuron_i]
        pattern_j = self.firing_patterns[neuron_j]

        similarity = F.cosine_similarity(
            pattern_i.unsqueeze(0),
            pattern_j.unsqueeze(0)
        )

        return similarity.item()

    def get_pattern_norms(self) -> torch.Tensor:
        """
        Get L2 norm of each neuron's firing pattern.

        Returns:
            Norms [pool_size]
        """
        norms = torch.norm(self.firing_patterns, dim=-1)
        return norms


class ContextualNeuronPool(nn.Module):
    """
    Neuron pool with context-dependent pattern modulation.

    Each neuron has:
    - Base firing pattern (inherent specialty)
    - Context modulation (adapts to current input)
    """

    def __init__(
        self,
        pool_size: int,
        d_model: int,
        d_ff: int,
        modulation_dim: int = 64
    ):
        """
        Initialize contextual neuron pool.

        Args:
            pool_size: Number of neurons
            d_model: Model dimension
            d_ff: Firing pattern dimension
            modulation_dim: Dimension for context modulation (default: 64)
        """
        super().__init__()
        self.pool_size = pool_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.modulation_dim = modulation_dim

        # Base firing patterns [pool_size, d_ff]
        self.base_patterns = nn.Parameter(torch.randn(pool_size, d_ff))

        # Context modulation network
        # Maps input -> low-dim adjustments for each neuron
        self.context_modulation = nn.Linear(d_model, pool_size * modulation_dim)

        # Low-dim to high-dim projection for pattern adjustment
        self.adjustment_proj = nn.Parameter(torch.randn(modulation_dim, d_ff))

        # Shared W2
        self.W2 = nn.Linear(d_ff, d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.base_patterns, std=0.02)
        nn.init.xavier_uniform_(self.context_modulation.weight)
        nn.init.normal_(self.adjustment_proj, std=0.01)
        nn.init.xavier_uniform_(self.W2.weight)
        if self.W2.bias is not None:
            nn.init.zeros_(self.W2.bias)

    def forward(
        self,
        x: torch.Tensor,
        selected_indices: torch.Tensor,
        pattern_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward with context-modulated patterns.

        Args:
            x: Input tensor [batch, seq, d_model]
            selected_indices: Selected neuron indices [batch, seq, k]
            pattern_weights: Combining weights [batch, seq, k]

        Returns:
            Output tensor [batch, seq, d_model]
        """
        batch, seq, k = selected_indices.shape

        # 1. Get base patterns for selected neurons
        base_patterns = self.base_patterns[selected_indices]  # [batch, seq, k, d_ff]

        # 2. Compute context modulation
        modulation = self.context_modulation(x)  # [batch, seq, pool_size * mod_dim]
        modulation = modulation.view(batch, seq, self.pool_size, self.modulation_dim)

        # Select modulation for chosen neurons
        # Gather along pool_size dimension
        selected_modulation = torch.gather(
            modulation,
            2,
            selected_indices.unsqueeze(-1).expand(-1, -1, -1, self.modulation_dim)
        )  # [batch, seq, k, mod_dim]

        # Project to pattern space
        adjustments = torch.matmul(
            selected_modulation,
            self.adjustment_proj
        )  # [batch, seq, k, d_ff]

        # 3. Adjust patterns
        adjusted_patterns = base_patterns + adjustments  # [batch, seq, k, d_ff]

        # 4. Combine patterns
        weights = F.softmax(pattern_weights, dim=-1)  # [batch, seq, k]
        combined_pattern = torch.einsum('bsk,bskf->bsf', weights, adjusted_patterns)

        # 5. Activate (once!)
        activated = F.gelu(combined_pattern)

        # 6. Project to output
        output = self.W2(activated)

        return output
