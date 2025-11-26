"""
DAWN v7.5 - QK Attention Routing + Soft FFN

핵심 변경사항:
1. Router 제거 → QK Attention weights 재활용
2. 뉴런 선택: 의미(X) + 문맥(Attention) 결합
3. 동적 V 생성: 256 → 64 → 256 (sparse semantic)
4. Soft FFN 유지 (성능 보장)

철학:
- Attention이 문맥 파악 → 같은 정보로 뉴런 선택
- 원본 X로 뉴런 라우팅 (정보 손실 전)
- Softmax 비선형성만으로 충분
- 파라미터 효율적 (~11M)
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
    - Basis_A: [n_basis, d_model, rank] - 필터 기본 블록
    - basis_emb: [n_basis, d_model] - Routing 보조 벡터

    역할:
    - 모든 Layer가 같은 Basis 공유
    - Recipe로 조합하여 동적 필터 생성
    """

    def __init__(self, n_basis=32, d_model=256, basis_rank=96):
        super().__init__()

        self.n_basis = n_basis
        self.d_model = d_model
        self.basis_rank = basis_rank

        # Basis A: [n_basis, d_model, rank] - 필터 조각
        basis_A = self._create_orthogonal_basis(n_basis, d_model, basis_rank)
        self.register_buffer('basis_A', basis_A)

        # Basis embedding: [n_basis, d_model] - Routing용
        basis_emb = self._create_orthogonal_vectors(n_basis, d_model)
        self.register_buffer('basis_emb', basis_emb)

    def _create_orthogonal_basis(self, n_basis, dim1, dim2):
        """직교 basis 생성 [n_basis, dim1, dim2]"""
        random_tensor = torch.randn(n_basis, dim1, dim2)
        flat = random_tensor.view(n_basis, -1)
        orthogonal = self._gram_schmidt(flat)
        return orthogonal.view(n_basis, dim1, dim2)

    def _create_orthogonal_vectors(self, n_basis, dim):
        """직교 벡터들 생성 [n_basis, dim]"""
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
                proj = torch.dot(orthogonal[j], v) * orthogonal[j]
                v = v - proj

            norm = torch.norm(v)
            if norm > 1e-10:
                orthogonal[i] = v / norm
            else:
                orthogonal[i] = F.normalize(torch.randn_like(v), dim=0)

        return orthogonal


# ============================================
# 2. Neuron-Based Value Generator
# ============================================
class NeuronBasedValue(nn.Module):
    """뉴런 기반 동적 V 생성

    핵심:
    1. 원본 X (의미) + Attention weights (문맥)로 뉴런 선택
    2. 선택된 뉴런으로 sparse representation (256→96)
    3. 복원하여 V 생성 (96→256)
    4. Multi-head 형태로 reshape

    뉴런 선택:
    - semantic_score: X @ neuron_emb (의미 매칭)
    - context_score: Attn_pattern @ context_pattern (문맥 매칭)
    - final_score: semantic × sigmoid(context) (둘 다 맞아야!)
    """

    def __init__(self, shared_basis, n_neurons=96, d_model=256,
                 n_heads=4, neuron_k=8, dropout=0.1):
        super().__init__()

        self.basis = shared_basis
        self.n_neurons = n_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.k = neuron_k

        # Neuron Recipe (의미 차원) - 학습됨!
        self.neuron_recipe = nn.Parameter(
            torch.randn(n_neurons, shared_basis.n_basis) * 0.5
        )

        # Neuron Context Pattern (문맥 차원) - 학습됨!
        self.neuron_context_pattern = nn.Parameter(
            torch.randn(n_neurons, n_heads) * 0.5
        )

        # V 복원 projection
        self.v_out = nn.Linear(shared_basis.basis_rank, d_model)
        self.dropout = nn.Dropout(dropout)

    @property
    def neuron_emb_semantic(self):
        """의미 기반 임베딩 [n_neurons, d_model]

        Recipe를 basis_emb와 조합하여 생성
        """
        recipe_norm = F.softmax(self.neuron_recipe, dim=-1)
        return torch.matmul(recipe_norm, self.basis.basis_emb)

    def forward(self, x, attn_weights):
        """
        Args:
            x: [B, S, D] - 원본 임베딩 (아직 섞이지 않음!)
            attn_weights: [B, n_heads, S, S] - attention 패턴

        Returns:
            V: [B, n_heads, S, d_head] - 동적 생성된 Value
            neuron_idx: [B, S, k] - 선택된 뉴런 인덱스
        """
        B, S, D = x.shape

        # ========== 1. 뉴런 선택 (의미 + 문맥) ==========

        # 의미 점수: "이 단어는 어떤 뉴런?"
        semantic_scores = x @ self.neuron_emb_semantic.T  # [B, S, n_neurons]

        # 문맥 점수: "이 문맥 패턴은 어떤 뉴런?"
        # Attention 패턴 요약 (head별 평균)
        attn_summary = attn_weights.mean(dim=-1).transpose(1, 2)  # [B, S, n_heads]
        context_scores = attn_summary @ self.neuron_context_pattern.T  # [B, S, n_neurons]

        # 결합: 의미도 맞고 문맥도 맞아야!
        final_scores = semantic_scores * torch.sigmoid(context_scores)

        # Top-K 선택
        topk_scores, neuron_idx = torch.topk(final_scores, self.k, dim=-1)
        neuron_weights = F.softmax(topk_scores, dim=-1)  # [B, S, k]

        # ========== 2. 선택된 뉴런으로 sparse representation ==========

        # Recipe 가져오기
        selected_recipe = self.neuron_recipe[neuron_idx]  # [B, S, k, n_basis]
        selected_recipe = F.softmax(selected_recipe, dim=-1)

        # Token별 recipe (weighted sum)
        token_recipe = (selected_recipe * neuron_weights.unsqueeze(-1)).sum(dim=2)
        # [B, S, n_basis]

        # Basis 조합으로 필터 생성
        W_A = torch.einsum('bsn,ndr->bsdr', token_recipe, self.basis.basis_A)
        # [B, S, d_model, basis_rank]

        # 256 → 96 압축 (sparse semantic space)
        v_semantic = torch.einsum('bsd,bsdr->bsr', x, W_A)
        # [B, S, basis_rank=96]

        # ========== 3. 96 → 256 복원 ==========
        V = self.v_out(v_semantic)  # [B, S, 256]
        V = self.dropout(V)

        # ========== 4. Multi-head 형태로 reshape ==========
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        # [B, n_heads, S, d_head]

        return V, neuron_idx


# ============================================
# 3. DAWN Layer v7.3
# ============================================
class DAWNLayer(nn.Module):
    """DAWN Layer with QK Attention Routing

    구조:
    1. Q, K 생성 및 Attention weights 계산
    2. 원본 X + Attention으로 뉴런 선택 & 동적 V 생성
    3. Attention 적용 (weights @ V)
    4. Soft FFN (표준, 성능 보장)
    """

    def __init__(self, shared_basis, d_model=256, d_ff=1024,
                 n_heads=4, n_neurons=96, neuron_k=8, dropout=0.1):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        # Q, K projection (표준 Attention)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.attn_out = nn.Linear(d_model, d_model)

        # Neuron-based V 생성
        self.neuron_value = NeuronBasedValue(
            shared_basis=shared_basis,
            n_neurons=n_neurons,
            d_model=d_model,
            n_heads=n_heads,
            neuron_k=neuron_k,
            dropout=dropout
        )

        # Soft FFN (표준)
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_indices=False):
        B, S, D = x.shape

        # ========== Part 1: Attention with Dynamic V ==========
        residual = x
        normed = self.norm1(x)

        # 1. Q, K 생성
        Q = self.q_proj(normed).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(normed).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        # [B, n_heads, S, d_head]

        # 2. Attention weights 계산
        attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, n_heads, S, S]
        attn_weights_dropout = self.attn_dropout(attn_weights)

        # 3. 원본 normed + attn_weights로 뉴런 선택 & V 생성
        # 핵심: 원본 normed 사용! (아직 정보 섞이지 않음)
        V, neuron_idx = self.neuron_value(normed, attn_weights)
        # V: [B, n_heads, S, d_head]

        # 4. Attention 적용
        attn_out = (attn_weights_dropout @ V).transpose(1, 2).reshape(B, S, D)
        attn_out = self.attn_out(attn_out)
        x = residual + self.dropout(attn_out)

        # ========== Part 2: Soft FFN ==========
        residual = x
        normed = self.norm2(x)

        ffn_out = F.gelu(self.w_up(normed))
        ffn_out = self.dropout(ffn_out)
        ffn_out = self.w_down(ffn_out)
        x = residual + self.dropout(ffn_out)

        if return_indices:
            return x, neuron_idx
        return x


# ============================================
# 4. DAWN Model v7.5
# ============================================
class DAWN(nn.Module):
    """DAWN v7.5 - QK Attention Routing + Soft FFN

    핵심 혁신:
    1. Router 제거 → QK Attention weights 재활용
    2. 의미(X) + 문맥(Attention) 결합 뉴런 선택
    3. 동적 V 생성 (256→96→256)
    4. Softmax 비선형성만으로 충분
    5. 파라미터 효율적 (~11M)

    구조:
    - Shared Basis: 전체 layer 공유, 고정
    - Per Layer: QK Attention + Neuron V + Soft FFN
    """

    __version__ = "7.5"

    def __init__(self, vocab_size, d_model=256, d_ff=1024,
                 n_layers=4, n_heads=4,
                 n_neurons=96, neuron_k=8,
                 n_basis=32, basis_rank=96,
                 max_seq_len=128, dropout=0.1, **kwargs):
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

        # Causal mask (cached)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', causal_mask)

        # Layers
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

        # Causal mask
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
                freq = counts / (counts.sum() + 1e-10)
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

        # Per layer breakdown
        layer_params = {}
        if len(self.layers) > 0:
            layer = self.layers[0]

            # Attention (Q, K, Out)
            attn_params = (
                layer.q_proj.weight.numel() +
                layer.k_proj.weight.numel() +
                layer.attn_out.weight.numel()
            )

            # Neuron V
            neuron_params = (
                layer.neuron_value.neuron_recipe.numel() +
                layer.neuron_value.neuron_context_pattern.numel() +
                layer.neuron_value.v_out.weight.numel() +
                layer.neuron_value.v_out.bias.numel()
            )

            # FFN
            ffn_params = (
                layer.w_up.weight.numel() + layer.w_up.bias.numel() +
                layer.w_down.weight.numel() + layer.w_down.bias.numel()
            )

            layer_params = {
                'attention': attn_params,
                'neuron_v': neuron_params,
                'ffn': ffn_params,
                'total_per_layer': attn_params + neuron_params + ffn_params
            }

        return {
            'total': total,
            'trainable': trainable,
            'basis': basis_params,
            'per_layer': layer_params,
            'embeddings': self.token_emb.weight.numel() + self.pos_emb.weight.numel(),
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
    print("DAWN v7.5 - QK Attention Routing + Soft FFN")
    print("=" * 60)

    config = {
        'vocab_size': 30522,
        'd_model': 256,
        'd_ff': 1024,
        'n_layers': 4,
        'n_heads': 4,
        'n_neurons': 96,
        'neuron_k': 8,
        'n_basis': 32,
        'basis_rank': 96,
        'max_seq_len': 128,
        'dropout': 0.1,
    }

    model = DAWN(**config)
    params = model.get_num_params()

    print(f"\n📊 Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Embeddings: {params['embeddings']:,}")
    print(f"  Basis (fixed): {params['basis']:,}")

    if 'per_layer' in params and params['per_layer']:
        print(f"\n📊 Per Layer:")
        print(f"  Attention (Q,K,Out): {params['per_layer']['attention']:,}")
        print(f"  Neuron V: {params['per_layer']['neuron_v']:,}")
        print(f"  FFN: {params['per_layer']['ffn']:,}")
        print(f"  Total: {params['per_layer']['total_per_layer']:,}")
        print(f"  4 Layers: {params['per_layer']['total_per_layer'] * 4:,}")

    # Target comparison
    target = 11_000_000
    diff = params['total'] - target
    print(f"\n🎯 Target: {target:,}")
    print(f"   Current: {params['total']:,}")
    print(f"   Diff: {diff:+,} ({diff/target*100:+.1f}%)")

    # Test forward
    print(f"\n📊 Forward Pass:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len)).to(device)
    labels = torch.randint(0, config['vocab_size'], (batch_size, seq_len)).to(device)

    # Forward
    logits, neuron_indices = model(input_ids, return_activations=True)
    print(f"  Input: {input_ids.shape}")
    print(f"  Output: {logits.shape}")
    print(f"  Neuron indices: {neuron_indices[0].shape}")

    # Loss
    loss, loss_dict, _ = model.get_loss(input_ids, labels, load_balance_weight=0.01)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss dict: {loss_dict}")

    # Neuron usage analysis
    all_used = torch.cat([idx.reshape(-1) for idx in neuron_indices])
    unique, counts = torch.unique(all_used, return_counts=True)
    print(f"\n📊 Neuron Usage:")
    print(f"  Active neurons: {len(unique)}/{config['n_neurons']}")
    print(f"  Usage rate: {len(unique)/config['n_neurons']*100:.1f}%")
