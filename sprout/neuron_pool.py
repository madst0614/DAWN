"""
Unified Neuron Pool for SPROUT.

Each token selects k neurons from a shared pool of neurons.
Neurons self-organize and specialize through training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class NeuronPool(nn.Module):
    """
    Unified neuron pool - all neurons exist in a single shared pool.

    Each neuron has its own weights (w_in, w_out) and biases.
    Tokens select k neurons from the pool to process their features.
    """

    def __init__(self, pool_size: int, d_model: int, d_ff: int):
        """
        Initialize neuron pool.

        Args:
            pool_size: Total number of neurons in the pool
            d_model: Model dimension
            d_ff: Feed-forward dimension (hidden size)
        """
        super().__init__()
        self.pool_size = pool_size
        self.d_model = d_model
        self.d_ff = d_ff

        # Each neuron has its own weights
        self.w_in = nn.Parameter(torch.randn(pool_size, d_model, d_ff))
        self.w_out = nn.Parameter(torch.randn(pool_size, d_ff, d_model))
        self.bias_in = nn.Parameter(torch.zeros(pool_size, d_ff))
        self.bias_out = nn.Parameter(torch.zeros(pool_size, d_model))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.w_in)
        nn.init.xavier_uniform_(self.w_out)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with selected neurons.

        Args:
            x: Input tensor [batch, seq, d_model]
            indices: Selected neuron indices [batch, seq, k]

        Returns:
            Output from selected neurons [batch, seq, k, d_model]
        """
        batch, seq, _ = x.shape
        k = indices.shape[-1]

        # Select weights for chosen neurons [batch, seq, k, d_model, d_ff]
        selected_w_in = self.w_in[indices]  # [batch, seq, k, d_model, d_ff]
        selected_w_out = self.w_out[indices]  # [batch, seq, k, d_ff, d_model]
        selected_b_in = self.bias_in[indices]  # [batch, seq, k, d_ff]
        selected_b_out = self.bias_out[indices]  # [batch, seq, k, d_model]

        # Apply neuron transformations
        # x: [batch, seq, d_model] -> [batch, seq, 1, 1, d_model]
        x_expanded = x.unsqueeze(2).unsqueeze(3)

        # Hidden: [batch, seq, k, d_ff]
        hidden = torch.matmul(x_expanded, selected_w_in).squeeze(3) + selected_b_in
        hidden = F.gelu(hidden)

        # Output: [batch, seq, k, d_model]
        output = torch.matmul(hidden.unsqueeze(3), selected_w_out).squeeze(3) + selected_b_out

        return output

    def get_neuron_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Count how many times each neuron was used.

        Args:
            indices: Neuron indices [batch, seq, k]

        Returns:
            Usage counts [pool_size]
        """
        usage = torch.zeros(self.pool_size, device=indices.device)
        flat_indices = indices.view(-1)
        usage.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
        return usage
