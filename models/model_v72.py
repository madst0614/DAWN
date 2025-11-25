"""
DAWN v7.2 - Standard FFN + Neuron Routing

실험 목적:
- 병목이 FFN(Recipe@Basis)인가, Router인가?
- Router + Neuron 선택은 유지
- FFN만 Standard로 교체

구조:
- Router: DAWN 그대로 (Neuron 선택)
- Neuron: 특징 정보 제공 (embedding)
- FFN: Standard (W_up → GELU → W_down)
- Neuron 정보를 FFN 입력에 추가

예상:
- v7.2 ≈ Baseline → FFN(Recipe@Basis)이 병목
- v7.2 < Baseline → Router도 병목
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 1. Simple Router (DAWN과 동일)
# ============================================
class SimpleRouter(nn.Module):
    """Neuron Router - DAWN과 동일"""

    def __init__(self, d_model=256, n_heads=4, k=8):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.k = k

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.score_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x, neuron_emb, mask=None):
        B, S, D = x.shape

        # Self-attention for context
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, S, D)

        # Combined representation
        combined = torch.cat([x, context], dim=-1)
        query = self.score_proj(combined)

        # Score against neurons
        scores = torch.matmul(query, neuron_emb.T)

        # Top-k selection
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        return topk_idx, weights


# ============================================
# 2. Standard FFN with Neuron Info
# ============================================
class NeuronAugmentedFFN(nn.Module):
    """Standard FFN + Neuron 정보

    구조:
    1. Router가 Neuron 선택
    2. Neuron embedding 조합 → 입력에 추가
    3. Standard FFN 적용

    vs DAWN:
    - DAWN: Neuron이 FFN weight 결정
    - v7.2: Neuron이 FFN 입력 augment
    """

    def __init__(self, d_model=256, d_ff=1024, n_neurons=64, dropout=0.1):
        super().__init__()

        self.n_neurons = n_neurons
        self.d_model = d_model

        # Neuron embedding (routing + augmentation용)
        self.neuron_emb = nn.Parameter(
            torch.randn(n_neurons, d_model) * 0.02
        )

        # Standard FFN
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Neuron info projection (optional, 더 표현력 있게)
        self.neuron_proj = nn.Linear(d_model, d_model)

    def forward(self, x, neuron_idx, neuron_weights):
        """
        Args:
            x: [B, S, D] input
            neuron_idx: [B, S, k] selected neuron indices
            neuron_weights: [B, S, k] selection weights

        Returns:
            output: [B, S, D]
        """
        B, S, D = x.shape

        # 1. Neuron embedding 조합
        selected_emb = self.neuron_emb[neuron_idx]  # [B, S, k, D]
        neuron_info = (selected_emb * neuron_weights.unsqueeze(-1)).sum(dim=2)
        # [B, S, D]

        # 2. Neuron info projection
        neuron_info = self.neuron_proj(neuron_info)

        # 3. FFN 입력에 추가
        x_aug = x + neuron_info

        # 4. Standard FFN
        h = self.w_up(x_aug)
        h = F.gelu(h)
        h = self.dropout(h)
        out = self.w_down(h)

        return out


# ============================================
# 3. DAWN Layer v7.2
# ============================================
class DAWNLayer(nn.Module):
    """DAWN Layer with Standard FFN"""

    def __init__(self, d_model=256, d_ff=1024, n_heads=4,
                 n_neurons=64, neuron_k=8, dropout=0.1):
        super().__init__()

        self.router = SimpleRouter(
            d_model=d_model,
            n_heads=n_heads,
            k=neuron_k
        )

        self.ffn = NeuronAugmentedFFN(
            d_model=d_model,
            d_ff=d_ff,
            n_neurons=n_neurons,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_indices=False):
        # 1. Get neuron embeddings
        neuron_emb = self.ffn.neuron_emb

        # 2. Router selects neurons
        normed = self.norm1(x)
        neuron_idx, neuron_weights = self.router(normed, neuron_emb, mask)

        # 3. Standard FFN with neuron info
        normed = self.norm2(x)
        ffn_out = self.ffn(normed, neuron_idx, neuron_weights)
        x = x + self.dropout(ffn_out)

        if return_indices:
            return x, neuron_idx
        return x


# ============================================
# 4. DAWN Model v7.2
# ============================================
class DAWN(nn.Module):
    """DAWN v7.2 - Standard FFN + Neuron Routing

    실험 목적:
    - FFN(Recipe@Basis)이 병목인지 확인
    - Router + Neuron 선택은 유지
    - FFN만 Standard로 교체
    """

    __version__ = "7.2"

    def __init__(self, vocab_size, d_model=256, d_ff=1024,
                 n_layers=4, n_heads=4,
                 n_neurons=64, neuron_k=8,
                 max_seq_len=512, dropout=0.1,
                 # Ignored params (for compatibility)
                 n_basis=32, basis_rank=64, **kwargs):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_neurons = n_neurons

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Layers
        self.layers = nn.ModuleList([
            DAWNLayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                n_neurons=n_neurons,
                neuron_k=neuron_k,
                dropout=dropout
            )
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
            'n_neurons': n_neurons,
            'neuron_k': neuron_k,
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

    def forward(self, input_ids, return_activations=False):
        B, S = input_ids.shape

        # Embeddings
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        # Causal mask
        mask = torch.tril(torch.ones(S, S, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Layers
        all_neuron_idx = []
        for layer in self.layers:
            if return_activations:
                x, neuron_idx = layer(x, mask, return_indices=True)
                all_neuron_idx.append(neuron_idx)
            else:
                x = layer(x, mask)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        if return_activations:
            return logits, all_neuron_idx
        return logits

    def get_loss(self, input_ids, labels,
                 diversity_weight=0.0, load_balance_weight=0.0, **kwargs):
        """Compute loss with optional regularization"""
        logits, neuron_indices = self.forward(input_ids, return_activations=True)

        # CE loss
        ce_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=-100
        )

        total_loss = ce_loss
        loss_dict = {'ce': ce_loss.item()}

        # Load balance loss (optional)
        if load_balance_weight > 0:
            lb_loss = 0
            for layer_indices in neuron_indices:
                flat_idx = layer_indices.reshape(-1)
                counts = torch.bincount(flat_idx, minlength=self.n_neurons).float()
                freq = counts / counts.sum()
                uniform = 1.0 / self.n_neurons
                lb_loss += ((freq - uniform) ** 2).sum() * self.n_neurons
            lb_loss = lb_loss / len(neuron_indices)
            total_loss = total_loss + load_balance_weight * lb_loss
            loss_dict['load_balance'] = lb_loss.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict, logits

    def get_num_params(self):
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Breakdown
        ffn_params = sum(
            sum(p.numel() for p in layer.ffn.parameters())
            for layer in self.layers
        )
        router_params = sum(
            sum(p.numel() for p in layer.router.parameters())
            for layer in self.layers
        )

        return {
            'total': total,
            'trainable': trainable,
            'ffn': ffn_params,
            'router': router_params,
        }


# ============================================
# Convenience exports
# ============================================
DAWNLanguageModel = DAWN


# ============================================
# Test
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("DAWN v7.2 - Standard FFN + Neuron Routing")
    print("=" * 60)

    config = {
        'vocab_size': 30522,
        'd_model': 256,
        'd_ff': 1024,
        'n_layers': 4,
        'n_heads': 4,
        'n_neurons': 64,
        'neuron_k': 8,
        'max_seq_len': 128,
        'dropout': 0.1,
    }

    model = DAWN(**config)
    params = model.get_num_params()

    print(f"\n📊 Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  FFN: {params['ffn']:,}")
    print(f"  Router: {params['router']:,}")

    # Compare with others
    print(f"\n📊 비교:")
    print(f"  Baseline: ~11,005,952")
    print(f"  v7.0: ~10,223,616")
    print(f"  v7.1: ~9,174,020")
    print(f"  v7.2: {params['total']:,}")

    # Test forward
    print(f"\n📊 Forward Pass:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len)).to(device)
    labels = torch.randint(0, config['vocab_size'], (batch_size, seq_len)).to(device)

    logits, neuron_indices = model(input_ids, return_activations=True)
    print(f"  Input: {input_ids.shape}")
    print(f"  Output: {logits.shape}")
    print(f"  Neuron indices: {neuron_indices[0].shape}")

    # Loss
    loss, loss_dict, _ = model.get_loss(input_ids, labels, load_balance_weight=0.01)
    print(f"  Loss: {loss.item():.4f}")

    print(f"\n✅ DAWN v7.2 Ready!")
    print(f"   Standard FFN + Neuron Routing")
    print(f"   병목 탐색 실험용")
