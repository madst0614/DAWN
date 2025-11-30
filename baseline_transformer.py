"""
Vanilla Transformer Baseline

DAWN과 공정한 비교를 위한 Standard Transformer 구현

목적:
- DAWN의 성능을 평가할 기준선
- 같은 조건에서 비교 (파라미터, 데이터, epoch)

구조:
- Standard Multi-Head Attention
- Standard FFN (W_up → GELU → W_down)
- Pre-LayerNorm (DAWN과 동일)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StandardFFN(nn.Module):
    """Standard Feed-Forward Network

    x → W_up → GELU → W_down → output

    가장 기본적인 FFN 구조
    """

    def __init__(self, d_model=256, d_ff=1024, dropout=0.1):
        super().__init__()

        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.w_up(x)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.w_down(h)
        return h


class StandardAttention(nn.Module):
    """Standard Multi-Head Attention"""

    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, D = x.shape

        # Project
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)

        return out


class TransformerLayer(nn.Module):
    """Standard Transformer Layer (Pre-LayerNorm)"""

    def __init__(self, d_model=256, n_heads=4, d_ff=1024, dropout=0.1):
        super().__init__()

        self.attn = StandardAttention(d_model, n_heads, dropout)
        self.ffn = StandardFFN(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention with residual
        normed = self.norm1(x)
        x = x + self.dropout(self.attn(normed, mask))

        # FFN with residual
        normed = self.norm2(x)
        x = x + self.dropout(self.ffn(normed))

        return x


class VanillaTransformer(nn.Module):
    """Vanilla Transformer for Language Modeling

    DAWN과 공정한 비교를 위한 Baseline

    파라미터 비교 (d_model=256, n_layers=4, d_ff=1024):
    - Embedding: 256 * vocab_size
    - Attention per layer: 4 * 256 * 256 = 262K
    - FFN per layer: 256 * 1024 + 1024 * 256 = 524K
    - Total per layer: 786K
    - 4 layers: 3.1M (+ embeddings)
    """

    __version__ = "baseline"

    def __init__(self, vocab_size, d_model=256, d_ff=1024,
                 n_layers=4, n_heads=4,
                 max_seq_len=512, dropout=0.1,
                 **kwargs):  # kwargs for compatibility
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # Weight tying

        # Config
        self.config = {
            'model_version': self.__version__,
            'vocab_size': vocab_size,
            'd_model': d_model,
            'd_ff': d_ff,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'max_seq_len': max_seq_len,
            'dropout': dropout,
        }

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None, return_routing_info=False):
        B, S = input_ids.shape

        # Embeddings
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        # Causal mask
        mask = torch.tril(torch.ones(S, S, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Layers
        for layer in self.layers:
            x = layer(x, mask)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        # Loss calculation (compatible with DAWN train.py)
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            if return_routing_info:
                return loss, logits, []  # Empty routing info
            return loss, logits

        if return_routing_info:
            return logits, []
        return logits

    def get_config(self):
        """Return model configuration (train.py compatible)"""
        return self.config

    def count_parameters(self):
        """Count trainable parameters (train.py compatible)"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_auxiliary_losses(self):
        """Return auxiliary losses (train.py compatible) - none for baseline"""
        return {}

    def get_num_params(self):
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Breakdown
        emb_params = self.token_emb.weight.numel() + self.pos_emb.weight.numel()
        attn_params = sum(
            sum(p.numel() for p in layer.attn.parameters())
            for layer in self.layers
        )
        ffn_params = sum(
            sum(p.numel() for p in layer.ffn.parameters())
            for layer in self.layers
        )

        return {
            'total': total,
            'trainable': trainable,
            'embedding': emb_params,
            'attention': attn_params,
            'ffn': ffn_params,
        }


# ============================================
# Parameter Comparison
# ============================================
def compare_params():
    """DAWN vs Baseline 파라미터 비교"""

    vocab_size = 30522
    config = {
        'vocab_size': vocab_size,
        'd_model': 256,
        'd_ff': 1024,
        'n_layers': 4,
        'n_heads': 4,
        'max_seq_len': 128,
        'dropout': 0.1,
    }

    # Baseline
    baseline = VanillaTransformer(**config)
    baseline_params = baseline.get_num_params()

    print("=" * 60)
    print("Parameter Comparison")
    print("=" * 60)

    print(f"\n📊 Vanilla Transformer (Baseline):")
    print(f"  Total: {baseline_params['total']:,}")
    print(f"  Embedding: {baseline_params['embedding']:,}")
    print(f"  Attention: {baseline_params['attention']:,}")
    print(f"  FFN: {baseline_params['ffn']:,}")

    # DAWN estimates
    print(f"\n📊 DAWN v7.0 (estimated):")
    print(f"  Total: ~10,223,616")
    print(f"  Basis (fixed): ~10M (not trained)")
    print(f"  Recipe: ~8K")
    print(f"  Router: ~800K")
    print(f"  W_down: ~1M")

    print(f"\n📊 DAWN v7.1 (estimated):")
    print(f"  Total: ~9,174,020")
    print(f"  Basis (fixed): ~10M (not trained)")
    print(f"  Recipe: ~8K")
    print(f"  Router: ~800K")
    print(f"  W_down: 0 (removed!)")

    print(f"\n📊 비교:")
    print(f"  Baseline vs v7.0: {baseline_params['total'] - 10223616:+,}")
    print(f"  Baseline vs v7.1: {baseline_params['total'] - 9174020:+,}")

    return baseline


# ============================================
# Test
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("Vanilla Transformer Baseline")
    print("=" * 60)

    # Compare parameters
    baseline = compare_params()

    # Test forward
    print(f"\n📊 Forward Pass Test:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline = baseline.to(device)

    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 30522, (batch_size, seq_len)).to(device)
    labels = torch.randint(0, 30522, (batch_size, seq_len)).to(device)

    # Forward
    logits = baseline(input_ids)
    print(f"  Input: {input_ids.shape}")
    print(f"  Output: {logits.shape}")

    # Loss
    loss, loss_dict, _ = baseline.get_loss(input_ids, labels)
    print(f"  Loss: {loss.item():.4f}")

    # With return_activations (compatibility)
    logits, activations = baseline(input_ids, return_activations=True)
    print(f"  Activations: {activations} (empty for baseline)")

    print(f"\n✅ Baseline ready for training!")
