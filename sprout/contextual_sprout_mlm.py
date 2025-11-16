"""
Contextual SPROUT for Masked Language Modeling.

Uses learned firing patterns with context-based combination.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .contextual_sprout_layer import ContextualSPROUTLayer, AdvancedContextualSPROUTLayer


class ContextualSPROUT_MLM(nn.Module):
    """
    Contextual SPROUT model for MLM.

    Architecture:
        Token Embeddings → n_steps × ContextualSPROUTLayer → Output Projection

    Each layer:
        - Selects k neurons via context routing
        - Combines their firing patterns with learned weights
        - Applies GELU once to combined pattern
        - Self-attention for information mixing
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        pool_size: int = 4096,
        k: int = 128,
        d_ff: int = 1024,
        n_steps: int = 3,
        n_heads: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        router_temperature: float = 1.0,
        use_advanced: bool = False,
        modulation_dim: int = 64
    ):
        """
        Initialize contextual SPROUT MLM.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension (default: 256)
            pool_size: Number of neurons (default: 4096)
            k: Neurons selected per token (default: 128)
            d_ff: Firing pattern dimension (default: 1024)
            n_steps: Number of SPROUT layers (default: 3)
            n_heads: Attention heads (default: 4)
            max_seq_len: Maximum sequence length (default: 512)
            dropout: Dropout probability (default: 0.1)
            router_temperature: Gumbel-Softmax temperature (default: 1.0)
            use_advanced: Use advanced layer with pattern modulation (default: False)
            modulation_dim: Modulation dimension for advanced layer (default: 64)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pool_size = pool_size
        self.k = k
        self.d_ff = d_ff
        self.n_steps = n_steps
        self.use_advanced = use_advanced

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # SPROUT layers
        self.sprout_layers = nn.ModuleList()

        LayerClass = AdvancedContextualSPROUTLayer if use_advanced else ContextualSPROUTLayer

        for _ in range(n_steps):
            if use_advanced:
                layer = AdvancedContextualSPROUTLayer(
                    d_model=d_model,
                    pool_size=pool_size,
                    k=k,
                    d_ff=d_ff,
                    modulation_dim=modulation_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    router_temperature=router_temperature
                )
            else:
                layer = ContextualSPROUTLayer(
                    d_model=d_model,
                    pool_size=pool_size,
                    k=k,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    dropout=dropout,
                    router_temperature=router_temperature
                )
            self.sprout_layers.append(layer)

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
            Dictionary with logits, loss, and optional routing info
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

        for i, layer in enumerate(self.sprout_layers):
            x, layer_routing = layer(x, return_routing=return_routing)

            if layer_routing is not None:
                layer_routing['step'] = i
                routing_info.append(layer_routing)

        # Output projection
        logits = self.output_proj(x)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )

        result = {
            "logits": logits,
            "loss": loss,
        }

        if return_routing:
            result["routing_info"] = routing_info

        return result

    def get_pattern_usage_stats(self, input_ids: torch.Tensor) -> Dict[str, any]:
        """
        Get firing pattern usage statistics.

        Args:
            input_ids: Input token IDs [batch, seq]

        Returns:
            Dictionary with pattern usage statistics per layer
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
                selected_indices = routing['selected_indices']  # [batch, seq, k]
                flat_indices = selected_indices.view(-1)

                # Count usage
                usage = torch.zeros(self.pool_size, device=input_ids.device)
                usage.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))

                # Pattern weight statistics
                pattern_weights = routing['pattern_weights']  # [batch, seq, k]

                usage_stats[f'layer_{i}'] = {
                    'usage_counts': usage,
                    'usage_percent': 100 * (usage > 0).sum().item() / self.pool_size,
                    'avg_usage': usage.mean().item(),
                    'max_usage': usage.max().item(),
                    'selection_entropy': routing['selection_entropy'].mean().item(),
                    'mean_max_weight': routing['mean_max_weight'].item(),
                    'weight_entropy': routing['weight_entropy'].item(),
                }

        return usage_stats

    def visualize_pattern_usage(self, input_ids: torch.Tensor):
        """
        Print pattern usage visualization.

        Args:
            input_ids: Input token IDs [batch, seq]
        """
        stats = self.get_pattern_usage_stats(input_ids)

        print("\n" + "="*70)
        print("FIRING PATTERN USAGE STATISTICS")
        print("="*70)

        for layer_name, layer_stats in stats.items():
            print(f"\n{layer_name}:")
            print(f"  Active neurons: {layer_stats['usage_percent']:.1f}%")
            print(f"  Average usage: {layer_stats['avg_usage']:.2f}")
            print(f"  Max usage: {layer_stats['max_usage']:.0f}")
            print(f"  Selection entropy: {layer_stats['selection_entropy']:.4f}")
            print(f"  Mean max weight: {layer_stats['mean_max_weight']:.4f}")
            print(f"  Weight entropy: {layer_stats['weight_entropy']:.4f}")

        print("="*70 + "\n")

    def get_pattern_similarity_matrix(self, layer_idx: int = 0) -> torch.Tensor:
        """
        Get similarity matrix between firing patterns in a layer.

        Args:
            layer_idx: Layer index (default: 0)

        Returns:
            Similarity matrix [pool_size, pool_size]
        """
        layer = self.sprout_layers[layer_idx]
        firing_patterns = layer.neuron_pool.firing_patterns  # [pool_size, d_ff]

        # Compute cosine similarity
        normalized = torch.nn.functional.normalize(firing_patterns, p=2, dim=1)
        similarity = torch.mm(normalized, normalized.t())

        return similarity
