"""
DAWN v7.2 - Basis Residual FFN

핵심 아이디어:
- Neuron = Recipe (Basis 조합 방법)
- Neuron embedding = Routing용으로만 사용
- FFN: Recipe로 필터 생성 → Residual 적용 → Standard FFN

구조:
1. Shared Basis (전체 Layer 공유, 고정)
   - Basis_A: [32, 256, 64] - 기본 필터
   - basis_emb: [32, 256] - Routing 보조

2. Per Layer Neuron Recipe (학습됨)
   - neuron_recipe: [64, 32] - Basis 조합 비율

3. Forward Pass
   - Router: neuron_emb = recipe @ basis_emb 사용
   - FFN: recipe로 필터 생성 → delta 적용 → standard FFN

철학:
- Neuron = 필터 조합 레시피
- Recipe @ Basis_A = 동적 필터
- Bottleneck 우회 (residual + standard FFN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 1. Fixed Orthogonal Basis (Shared)
# ============================================
class FixedOrthogonalBasis(nn.Module):
    """고정된 직교 Basis - 전체 Layer 공유, 학습 안 함

    구성:
    - Basis_A: [n_basis, d_model, rank] - 기본 필터
    - basis_emb: [n_basis, d_model] - Routing 보조 벡터

    역할:
    - 모든 Layer가 같은 Basis 공유
    - Recipe로 조합하여 동적 필터 생성
    """

    def __init__(self, n_basis=32, d_model=256, basis_rank=64):
        super().__init__()

        self.n_basis = n_basis
        self.d_model = d_model
        self.basis_rank = basis_rank

        # ===== 고정 직교 Basis (학습 X) =====

        # Basis A: [n_basis, d_model, rank] - 필터 조각
        basis_A = self._create_orthogonal_basis(n_basis, d_model, basis_rank)
        self.register_buffer('basis_A', basis_A)

        # Basis embedding: [n_basis, d_model] - Routing용
        basis_emb = self._create_orthogonal_vectors(n_basis, d_model)
        self.register_buffer('basis_emb', basis_emb)

    def _create_orthogonal_basis(self, n_basis, dim1, dim2):
        """직교 basis 생성"""
        random_tensor = torch.randn(n_basis, dim1, dim2)
        flat = random_tensor.view(n_basis, -1)
        orthogonal = self._gram_schmidt(flat)
        return orthogonal.view(n_basis, dim1, dim2)

    def _create_orthogonal_vectors(self, n_basis, dim):
        """직교 벡터들 생성"""
        if n_basis <= dim:
            random_matrix = torch.randn(dim, n_basis)
            q, r = torch.linalg.qr(random_matrix)
            return q[:, :n_basis].T
        else:
            vectors = torch.randn(n_basis, dim)
            return F.normalize(vectors, dim=-1)

    def _gram_schmidt(self, vectors):
        """Gram-Schmidt 직교화"""
        n, d = vectors.shape
        orthogonal = torch.zeros_like(vectors)

        for i in range(n):
            v = vectors[i].clone()
            for j in range(i):
                # Fixed: use v instead of vectors[i]
                proj = torch.dot(orthogonal[j], v) * orthogonal[j]
                v = v - proj

            # Normalize with zero check
            norm = torch.norm(v)
            if norm > 1e-10:
                orthogonal[i] = v / norm
            else:
                # Fallback to random if near-zero
                orthogonal[i] = F.normalize(torch.randn_like(v), dim=0)

        return orthogonal


# ============================================
# 2. Simple Router (DAWN과 동일)
# ============================================
class SimpleRouter(nn.Module):
    """Neuron Router - DAWN과 동일"""

    def __init__(self, d_model=256, n_heads=4, k=8, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.k = k

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.score_proj = nn.Linear(d_model * 2, d_model)
        self.attn_dropout = nn.Dropout(dropout)

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
        attn = self.attn_dropout(attn)  # Add dropout

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
# 3. Basis Residual FFN
# ============================================
class BasisResidualFFN(nn.Module):
    """Recipe 기반 Residual FFN

    핵심:
    - Neuron = Recipe (Basis 조합 방법)
    - neuron_emb = recipe @ basis_emb (routing용)
    - FFN: recipe로 필터 생성 → residual → standard FFN

    구조:
    1. Router: neuron_emb로 neuron 선택
    2. FFN: 선택된 recipe로 필터 생성
       - W_A = token_recipe @ Basis_A
       - delta = x @ W_A @ W_A.T
       - x_filtered = x + alpha * delta
    3. Standard FFN(x_filtered)
    """

    def __init__(self, shared_basis, n_neurons=64, d_model=256,
                 d_ff=1024, dropout=0.1):
        super().__init__()

        self.basis = shared_basis
        self.n_neurons = n_neurons
        self.d_model = d_model

        # ===== Neuron Recipe (학습됨!) =====
        # [n_neurons, n_basis] - 각 neuron의 basis 조합 비율
        self.neuron_recipe = nn.Parameter(
            torch.randn(n_neurons, shared_basis.n_basis) * 0.5
        )

        # Standard FFN
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Residual alpha (with explicit dtype)
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    @property
    def neuron_emb(self):
        """Neuron embedding - Routing용으로만!

        Returns:
            [n_neurons, d_model]
        """
        # Recipe 정규화
        recipe_norm = F.softmax(self.neuron_recipe, dim=-1)
        # Basis embedding과 조합
        return torch.matmul(recipe_norm, self.basis.basis_emb)

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
        k = neuron_idx.shape[-1]

        # ============ 1. Recipe 가져오기 ============
        # [B, S, k, n_basis]
        selected_recipe = self.neuron_recipe[neuron_idx]

        # ============ 2. Weighted sum ============
        # Recipe 정규화
        selected_recipe = F.softmax(selected_recipe, dim=-1)

        # [B, S, k, n_basis] * [B, S, k, 1] -> sum -> [B, S, n_basis]
        token_recipe = (selected_recipe * neuron_weights.unsqueeze(-1)).sum(dim=2)

        # ============ 3. 필터 생성 ============
        # token_recipe: [B, S, n_basis]
        # basis.basis_A: [n_basis, d_model, rank]
        # W_A: [B, S, d_model, rank]
        W_A = torch.einsum('bsn,ndr->bsdr', token_recipe, self.basis.basis_A)

        # ============ 4. 필터 적용 ============
        # Projection: delta = x @ W_A @ W_A^T
        # h = x @ W_A: [B, S, d_model] @ [B, S, d_model, rank] -> [B, S, rank]
        h = torch.einsum('bsd,bsdr->bsr', x, W_A)

        # delta = h @ W_A^T: [B, S, rank] @ [B, S, rank, d_model] -> [B, S, d_model]
        # Fixed: transpose W_A properly
        delta = torch.einsum('bsr,bsrd->bsd', h, W_A.transpose(-2, -1))

        # ============ 5. Residual ============
        x_filtered = x + self.alpha * delta

        # ============ 6. Standard FFN ============
        h = self.w_up(x_filtered)
        h = F.gelu(h)
        h = self.dropout(h)
        output = self.w_down(h)

        return output


# ============================================
# 4. DAWN Layer v7.2
# ============================================
class DAWNLayer(nn.Module):
    """DAWN Layer with Basis Residual FFN"""

    def __init__(self, shared_basis, d_model=256, d_ff=1024, n_heads=4,
                 n_neurons=64, neuron_k=8, dropout=0.1):
        super().__init__()

        self.router = SimpleRouter(
            d_model=d_model,
            n_heads=n_heads,
            k=neuron_k,
            dropout=dropout
        )

        self.ffn = BasisResidualFFN(
            shared_basis=shared_basis,
            n_neurons=n_neurons,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_indices=False):
        # 1. Get neuron embeddings (routing용)
        neuron_emb = self.ffn.neuron_emb

        # 2. Router selects neurons
        normed = self.norm1(x)
        neuron_idx, neuron_weights = self.router(normed, neuron_emb, mask)

        # 3. Basis Residual FFN
        normed = self.norm2(x)
        ffn_out = self.ffn(normed, neuron_idx, neuron_weights)
        x = x + self.dropout(ffn_out)

        if return_indices:
            return x, neuron_idx
        return x


# ============================================
# 5. DAWN Model v7.2
# ============================================
class DAWN(nn.Module):
    """DAWN v7.2 - Basis Residual FFN

    핵심 변경:
    - Neuron = Recipe (Basis 조합 방법)
    - neuron_emb = recipe @ basis_emb (routing용)
    - FFN: recipe로 필터 생성 → residual → standard FFN

    구조:
    - Shared Basis: 전체 layer 공유, 고정
    - Per Layer Recipe: 학습됨
    - Bottleneck 우회: residual + standard FFN
    """

    __version__ = "7.2"

    def __init__(self, vocab_size, d_model=256, d_ff=1024,
                 n_layers=4, n_heads=4,
                 n_neurons=64, neuron_k=8,
                 n_basis=32, basis_rank=64,
                 max_seq_len=512, dropout=0.1, **kwargs):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.n_basis = n_basis
        self.basis_rank = basis_rank

        # ===== Shared Basis (전체 Layer 공유, 고정) =====
        self.shared_basis = FixedOrthogonalBasis(
            n_basis=n_basis,
            d_model=d_model,
            basis_rank=basis_rank
        )

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal mask (cached for efficiency)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', causal_mask)

        # Layers (모두 같은 basis 공유)
        self.layers = nn.ModuleList([
            DAWNLayer(
                shared_basis=self.shared_basis,
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
            'n_basis': n_basis,
            'basis_rank': basis_rank,
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

        # Causal mask (use cached)
        mask = self.causal_mask[:S, :S].unsqueeze(0).unsqueeze(0)

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

        # Basis (고정, 학습 X)
        basis_params = sum(p.numel() for p in self.shared_basis.parameters())

        # Recipe (학습됨)
        recipe_params = sum(
            layer.ffn.neuron_recipe.numel()
            for layer in self.layers
        )

        # FFN (standard part만)
        ffn_params = sum(
            sum(p.numel() for p in layer.ffn.parameters() if p.requires_grad)
            for layer in self.layers
        )

        # Router
        router_params = sum(
            sum(p.numel() for p in layer.router.parameters())
            for layer in self.layers
        )

        return {
            'total': total,
            'trainable': trainable,
            'basis': basis_params,
            'recipe': recipe_params,
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
    print("DAWN v7.2 - Basis Residual FFN")
    print("=" * 60)

    config = {
        'vocab_size': 30522,
        'd_model': 256,
        'd_ff': 1024,
        'n_layers': 4,
        'n_heads': 4,
        'n_neurons': 64,
        'neuron_k': 8,
        'n_basis': 32,
        'basis_rank': 64,
        'max_seq_len': 128,
        'dropout': 0.1,
    }

    model = DAWN(**config)
    params = model.get_num_params()

    print(f"\n📊 Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Basis (fixed): {params['basis']:,}")
    print(f"  Recipe: {params['recipe']:,}")
    print(f"  FFN: {params['ffn']:,}")
    print(f"  Router: {params['router']:,}")

    # Breakdown
    print(f"\n📊 구조 분석:")
    print(f"  Shared Basis:")
    print(f"    - Basis_A: [32, 256, 64] = {32*256*64:,}")
    print(f"    - basis_emb: [32, 256] = {32*256:,}")
    print(f"  Per Layer:")
    print(f"    - Recipe: [64, 32] = {64*32:,}")
    print(f"    - Standard FFN: {256*1024 + 1024*256:,}")
    print(f"    - Alpha: 1")

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
    print(f"   Recipe 기반 Residual FFN")
    print(f"   Neuron = Basis 조합 레시피")
    print(f"   Bottleneck 우회 (residual + standard FFN)")
