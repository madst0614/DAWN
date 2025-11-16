"""
SPROUT for Masked Language Modeling.

Neuron pool-based architecture with hard routing.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .sprout_layer import SPROUTLayer


class SPROUT_MLM(nn.Module):
    """
    SPROUT model for Masked Language Modeling.

    Architecture:
        Token Embeddings → n_steps × SPROUTLayer → Output Projection
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        pool_size: int = 4096,
        k: int = 128,
        n_steps: int = 3,
        n_heads: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        router_temperature: float = 1.0,
        load_balance_weight: float = 0.0
    ):
        """
        Initialize SPROUT MLM model.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension (default: 256)
            pool_size: Number of neurons in pool (default: 4096)
            k: Number of neurons selected per token (default: 128)
            n_steps: Number of SPROUT layers (default: 3)
            n_heads: Number of attention heads (default: 4)
            max_seq_len: Maximum sequence length (default: 512)
            dropout: Dropout probability (default: 0.1)
            router_temperature: Temperature for Gumbel-Softmax (default: 1.0)
            load_balance_weight: Weight for load balance loss (default: 0.0)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pool_size = pool_size
        self.k = k
        self.n_steps = n_steps
        self.load_balance_weight = load_balance_weight

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # SPROUT layers
        self.sprout_layers = nn.ModuleList([
            SPROUTLayer(
                d_model=d_model,
                pool_size=pool_size,
                k=k,
                n_heads=n_heads,
                dropout=dropout,
                router_temperature=router_temperature
            )
            for _ in range(n_steps)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

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
        return_routing: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch, seq]
            attention_mask: Attention mask [batch, seq]
            labels: MLM labels [batch, seq] (-100 for non-masked)
            return_routing: If True, return routing information

        Returns:
            Dictionary with:
                - logits: [batch, seq, vocab_size]
                - loss: MLM loss (if labels provided)
                - load_balance_loss: Load balance loss (if enabled)
                - routing_info: Routing information (if return_routing=True)
        """
        batch, seq = input_ids.shape

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        pos_emb = self.pos_embedding(pos_ids)

        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)

        # Process through SPROUT layers
        routing_info = []
        total_load_balance_loss = 0.0

        for i, layer in enumerate(self.sprout_layers):
            x, layer_routing = layer(x, return_routing=return_routing or (self.load_balance_weight > 0))

            if layer_routing is not None:
                if return_routing:
                    layer_routing['step'] = i
                    routing_info.append(layer_routing)

                if self.load_balance_weight > 0:
                    total_load_balance_loss += layer_routing['load_balance_loss']

        # Output projection
        logits = self.output_proj(x)

        # Calculate loss if labels provided
        loss = None
        mlm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )

            # Combine with load balance loss
            loss = mlm_loss
            if self.load_balance_weight > 0:
                avg_load_balance_loss = total_load_balance_loss / len(self.sprout_layers)
                loss = loss + self.load_balance_weight * avg_load_balance_loss

        result = {
            "logits": logits,
            "loss": loss,
            "mlm_loss": mlm_loss,
            "load_balance_loss": total_load_balance_loss / len(self.sprout_layers) if self.load_balance_weight > 0 else None,
        }

        if return_routing:
            result["routing_info"] = routing_info

        return result

    def get_neuron_usage_stats(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get neuron usage statistics for given input.

        Args:
            input_ids: Input token IDs [batch, seq]

        Returns:
            Dictionary with usage statistics per layer
        """
        self.eval()
        with torch.no_grad():
            batch, seq = input_ids.shape

            # Embeddings
            token_emb = self.token_embedding(input_ids)
            pos_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, -1)
            pos_emb = self.pos_embedding(pos_ids)
            x = token_emb + pos_emb

            usage_stats = {}

            for i, layer in enumerate(self.sprout_layers):
                x, routing = layer(x, return_routing=True)

                # Get neuron usage
                indices = routing['indices']  # [batch, seq, k]
                usage = layer.neuron_pool.get_neuron_usage(indices)

                usage_stats[f'layer_{i}'] = {
                    'usage_counts': usage,
                    'usage_percent': 100 * (usage > 0).sum().item() / self.pool_size,
                    'avg_usage': usage.mean().item(),
                    'max_usage': usage.max().item(),
                    'routing_entropy': routing['entropy'].mean().item()
                }

        return usage_stats

    def visualize_neuron_usage(self, input_ids: torch.Tensor):
        """Print neuron usage visualization."""
        stats = self.get_neuron_usage_stats(input_ids)

        print("\n" + "="*70)
        print("NEURON USAGE STATISTICS")
        print("="*70)

        for layer_name, layer_stats in stats.items():
            print(f"\n{layer_name}:")
            print(f"  Active neurons: {layer_stats['usage_percent']:.1f}%")
            print(f"  Average usage: {layer_stats['avg_usage']:.2f}")
            print(f"  Max usage: {layer_stats['max_usage']:.0f}")
            print(f"  Routing entropy: {layer_stats['routing_entropy']:.4f}")

        print("="*70 + "\n")
