"""
Simple Neuron Pool - V1

Reinterprets traditional Transformer FFN as a neuron pool.
Each layer uses a fixed subset of neurons (no dynamic routing yet).

This is IDENTICAL to traditional Transformer - just a different view.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SimpleNeuronPool(nn.Module):
    """
    Unified neuron pool for all layers.

    Each layer uses a fixed range of neurons:
    - Layer 0: neurons [0, d_ff)
    - Layer 1: neurons [d_ff, 2*d_ff)
    - Layer 2: neurons [2*d_ff, 3*d_ff)
    ...

    This is mathematically identical to having separate FFN per layer.
    """

    def __init__(self, n_layers: int = 6, d_model: int = 256, d_ff: int = 1024):
        """
        Initialize simple neuron pool.

        Args:
            n_layers: Number of layers (default: 6)
            d_model: Model dimension (default: 256)
            d_ff: Neurons per layer (default: 1024)
        """
        super().__init__()

        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff

        # Total neurons across all layers
        total_neurons = n_layers * d_ff  # e.g., 6 * 1024 = 6144

        # All neuron input weights [total_neurons, d_model]
        self.W_in = nn.Parameter(torch.randn(total_neurons, d_model))
        self.b_in = nn.Parameter(torch.zeros(total_neurons))

        # All neuron output weights [total_neurons, d_model]
        self.W_out = nn.Parameter(torch.randn(total_neurons, d_model))
        self.b_out = nn.Parameter(torch.zeros(total_neurons))

        # Layer ranges (fixed assignment)
        self.layer_ranges = [
            (i * d_ff, (i + 1) * d_ff)
            for i in range(n_layers)
        ]

        # Usage tracking
        self.register_buffer('usage_count', torch.zeros(total_neurons))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights like standard FFN."""
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_out)
        nn.init.zeros_(self.b_in)
        nn.init.zeros_(self.b_out)

    def forward_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Forward pass using neurons for a specific layer.

        This is IDENTICAL to traditional FFN:
        FFN(x) = W2 @ GELU(W1 @ x + b1) + b2

        Args:
            x: Input tensor [batch, seq, d_model]
            layer_idx: Layer index (0 to n_layers-1)

        Returns:
            Output tensor [batch, seq, d_model]
        """
        # Get neuron range for this layer
        start, end = self.layer_ranges[layer_idx]

        # Extract weights for this layer's neurons
        W_in_layer = self.W_in[start:end]    # [d_ff, d_model]
        b_in_layer = self.b_in[start:end]    # [d_ff]
        W_out_layer = self.W_out[start:end]  # [d_ff, d_model]
        b_out_layer = self.b_out[start:end]  # [d_ff]

        # Forward pass (same as traditional FFN)
        # hidden = W1 @ x + b1
        hidden = torch.einsum('nd,bsd->bsn', W_in_layer, x) + b_in_layer
        # [batch, seq, d_ff]

        # activated = GELU(hidden)
        activated = F.gelu(hidden)

        # output = W2 @ activated + b2
        output = torch.einsum('nd,bsn->bsd', W_out_layer, activated) + b_out_layer
        # [batch, seq, d_model]

        # Track usage
        if self.training:
            with torch.no_grad():
                batch_size = x.shape[0]
                seq_len = x.shape[1]
                self.usage_count[start:end] += batch_size * seq_len

        return output

    def get_layer_neuron_usage(self, layer_idx: int) -> torch.Tensor:
        """
        Get usage counts for neurons in a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Usage counts [d_ff]
        """
        start, end = self.layer_ranges[layer_idx]
        return self.usage_count[start:end]

    def get_most_used_neurons(self, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the most frequently used neurons across all layers.

        Args:
            top_k: Number of top neurons to return

        Returns:
            indices: Neuron indices [top_k]
            counts: Usage counts [top_k]
        """
        values, indices = self.usage_count.topk(top_k)
        return indices, values

    def get_neuron_signature(self, neuron_id: int) -> dict:
        """
        Get input/output patterns for a specific neuron.

        Args:
            neuron_id: Global neuron ID

        Returns:
            Dictionary with neuron patterns
        """
        return {
            'input_pattern': self.W_in[neuron_id].detach().cpu(),
            'output_pattern': self.W_out[neuron_id].detach().cpu(),
            'input_bias': self.b_in[neuron_id].item(),
            'output_bias': self.b_out[neuron_id].item(),
            'usage_count': self.usage_count[neuron_id].item(),
        }

    def get_all_usage_stats(self) -> dict:
        """Get comprehensive usage statistics."""
        stats = {}

        for layer_idx in range(self.n_layers):
            layer_usage = self.get_layer_neuron_usage(layer_idx)

            stats[f'layer_{layer_idx}'] = {
                'mean_usage': layer_usage.mean().item(),
                'std_usage': layer_usage.std().item(),
                'min_usage': layer_usage.min().item(),
                'max_usage': layer_usage.max().item(),
                'total_usage': layer_usage.sum().item(),
            }

        # Global stats
        stats['global'] = {
            'total_neurons': len(self.usage_count),
            'active_neurons': (self.usage_count > 0).sum().item(),
            'mean_usage': self.usage_count.mean().item(),
            'max_usage': self.usage_count.max().item(),
        }

        return stats

    def reset_usage_stats(self):
        """Reset usage tracking."""
        self.usage_count.zero_()


def test_equivalence_to_ffn():
    """
    Test that SimpleNeuronPool is equivalent to traditional FFN.
    """
    print("Testing equivalence to traditional FFN...")

    d_model = 256
    d_ff = 1024

    # Traditional FFN
    traditional_ffn = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.GELU(),
        nn.Linear(d_ff, d_model)
    )

    # Our neuron pool (1 layer only for testing)
    neuron_pool = SimpleNeuronPool(n_layers=1, d_model=d_model, d_ff=d_ff)

    # Copy weights from traditional FFN to neuron pool
    with torch.no_grad():
        neuron_pool.W_in.data = traditional_ffn[0].weight.data  # [d_ff, d_model]
        neuron_pool.b_in.data = traditional_ffn[0].bias.data    # [d_ff]
        neuron_pool.W_out.data = traditional_ffn[2].weight.data.T  # [d_ff, d_model]
        neuron_pool.b_out.data = traditional_ffn[2].bias.data  # [d_model]

    # Test input
    x = torch.randn(2, 10, d_model)

    # Forward pass
    out_traditional = traditional_ffn(x)
    out_neuron = neuron_pool.forward_layer(x, layer_idx=0)

    # Check equivalence
    is_close = torch.allclose(out_traditional, out_neuron, atol=1e-5)
    max_diff = (out_traditional - out_neuron).abs().max().item()

    print(f"  Outputs close: {is_close}")
    print(f"  Max difference: {max_diff:.2e}")

    if is_close:
        print("✅ Test passed! SimpleNeuronPool is equivalent to traditional FFN")
    else:
        print("❌ Test failed! There's a difference")

    return is_close


if __name__ == '__main__':
    test_equivalence_to_ffn()
