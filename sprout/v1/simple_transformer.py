"""
Simple Transformer with Neuron Pool - V1

Traditional Transformer architecture but with FFN replaced by neuron pool.
Mathematically identical to standard Transformer.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .simple_neuron_pool import SimpleNeuronPool


class SimpleTransformerWithNeuronPool(nn.Module):
    """
    Transformer with neuron pool instead of separate FFNs.

    Architecture:
        Embeddings → n_layers × (Attention + NeuronPool) → Output

    This is IDENTICAL to traditional Transformer.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 256,
        d_ff: int = 1024,
        n_layers: int = 6,
        n_heads: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize simple transformer with neuron pool.

        Args:
            vocab_size: Vocabulary size (default: 30000)
            d_model: Model dimension (default: 256)
            d_ff: Feed-forward dimension (default: 1024)
            n_layers: Number of layers (default: 6)
            n_heads: Number of attention heads (default: 4)
            max_seq_len: Maximum sequence length (default: 512)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Attention layers (standard)
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(
                d_model,
                n_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layers)
        ])

        # Neuron pool (replaces separate FFNs)
        self.neuron_pool = SimpleNeuronPool(
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff
        )

        # Layer norms
        self.norms1 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.norms2 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch, seq]
            attention_mask: Attention mask [batch, seq]
            labels: MLM labels [batch, seq] (-100 for non-masked)

        Returns:
            Dictionary with logits and loss
        """
        batch, seq = input_ids.shape

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        pos_emb = self.pos_embedding(pos_ids)

        # Combine embeddings
        h = self.dropout(token_emb + pos_emb)

        # Transformer layers
        for layer_idx in range(self.n_layers):
            # Self-attention
            residual = h
            attn_out, _ = self.attentions[layer_idx](h, h, h)
            h = self.norms1[layer_idx](residual + self.dropout(attn_out))

            # Neuron pool (specific layer's neurons)
            residual = h
            ffn_out = self.neuron_pool.forward_layer(h, layer_idx)
            h = self.norms2[layer_idx](residual + self.dropout(ffn_out))

        # Output projection
        logits = self.output_proj(h)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )

        return {
            "logits": logits,
            "loss": loss,
        }

    def get_neuron_usage_stats(self) -> dict:
        """Get neuron usage statistics."""
        return self.neuron_pool.get_all_usage_stats()

    def visualize_neuron_usage(self):
        """Print neuron usage visualization."""
        stats = self.get_neuron_usage_stats()

        print("\n" + "="*70)
        print("NEURON USAGE STATISTICS (Simple Neuron Pool V1)")
        print("="*70)

        # Per-layer stats
        for layer_idx in range(self.n_layers):
            layer_stats = stats[f'layer_{layer_idx}']
            print(f"\nLayer {layer_idx}:")
            print(f"  Mean usage: {layer_stats['mean_usage']:.1f}")
            print(f"  Std usage: {layer_stats['std_usage']:.1f}")
            print(f"  Min usage: {layer_stats['min_usage']:.0f}")
            print(f"  Max usage: {layer_stats['max_usage']:.0f}")
            print(f"  Total usage: {layer_stats['total_usage']:.0f}")

        # Global stats
        global_stats = stats['global']
        print(f"\nGlobal:")
        print(f"  Total neurons: {global_stats['total_neurons']}")
        print(f"  Active neurons: {global_stats['active_neurons']}")
        print(f"  Mean usage: {global_stats['mean_usage']:.1f}")
        print(f"  Max usage: {global_stats['max_usage']:.0f}")

        print("="*70 + "\n")

    def reset_neuron_usage_stats(self):
        """Reset neuron usage tracking."""
        self.neuron_pool.reset_usage_stats()

    def get_most_used_neurons(self, top_k: int = 10) -> dict:
        """Get most used neurons with their layer information."""
        indices, counts = self.neuron_pool.get_most_used_neurons(top_k)

        results = []
        for idx, count in zip(indices, counts):
            idx = idx.item()
            count = count.item()

            layer = idx // self.d_ff
            local_id = idx % self.d_ff

            results.append({
                'global_id': idx,
                'layer': layer,
                'local_id': local_id,
                'usage_count': count,
            })

        return results


def test_simple_transformer():
    """
    Test that SimpleTransformerWithNeuronPool works correctly.
    """
    print("Testing SimpleTransformerWithNeuronPool...")

    # Create model
    model = SimpleTransformerWithNeuronPool(
        vocab_size=1000,
        d_model=128,
        d_ff=512,
        n_layers=3,
        n_heads=4
    )

    # Test input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    labels[:, :5] = -100  # Mask first half

    # Forward pass
    outputs = model(input_ids, labels=labels)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")

    # Check shapes
    assert outputs['logits'].shape == (batch_size, seq_len, 1000)
    assert outputs['loss'] is not None

    print("✅ Test passed! Model works correctly")

    # Test neuron usage
    print("\nNeuron usage after forward pass:")
    model.visualize_neuron_usage()

    return True


if __name__ == '__main__':
    test_simple_transformer()
