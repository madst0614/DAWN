"""
DAWN v7.0 - Dynamic Architecture With Neurons

핵심 철학:
- Basis: 고정된 직교 좌표계 (학습 X, 전체 공유)
- Neuron: Basis 조합 recipe (학습 O, Layer별)
- Router: Neuron 선택 (학습 O, Layer별)

v6.0 → v7.0 변경:
1. Basis 고정 (register_buffer) - 학습 안 함
2. Basis 전체 Layer 공유
3. Neuron embedding = recipe @ basis_emb (파생)
4. Orthogonality loss 제거 (수학적 보장)
5. n_basis 증가 (8 → 32), n_neurons 감소 (256 → 64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 1. Fixed Orthogonal Basis (핵심!)
# ============================================
class FixedOrthogonalBasis(nn.Module):
    """고정된 직교 Basis - 전체 Layer 공유, 학습 안 함

    역할:
    - 의미 공간의 "좌표축" 정의
    - 모든 Layer가 같은 좌표계 사용
    - 수학적으로 직교 보장
    """

    def __init__(self, n_basis=32, d_model=256, d_ff=1024, basis_rank=64):
        super().__init__()

        self.n_basis = n_basis
        self.d_model = d_model
        self.d_ff = d_ff
        self.basis_rank = basis_rank

        # ===== 고정 직교 Basis (학습 X) =====

        # Basis A: [n_basis, d_model, rank] - Up projection 조각
        basis_A = self._create_orthogonal_basis(n_basis, d_model, basis_rank)
        self.register_buffer('basis_A', basis_A)

        # Basis B: [n_basis, rank, d_ff] - Down projection 조각
        basis_B = self._create_orthogonal_basis(n_basis, basis_rank, d_ff)
        self.register_buffer('basis_B', basis_B)

        # Basis embedding: [n_basis, d_model] - Routing용 의미 벡터
        basis_emb = self._create_orthogonal_vectors(n_basis, d_model)
        self.register_buffer('basis_emb', basis_emb)

    def _create_orthogonal_basis(self, n_basis, dim1, dim2):
        """직교 basis 생성 (QR decomposition)"""
        # Random 초기화
        random_tensor = torch.randn(n_basis, dim1, dim2)

        # 각 basis를 flatten하고 직교화
        flat = random_tensor.view(n_basis, -1)  # [n_basis, dim1*dim2]

        # Gram-Schmidt로 직교화 (n_basis <= dim1*dim2 가정)
        orthogonal = self._gram_schmidt(flat)

        # 다시 reshape
        return orthogonal.view(n_basis, dim1, dim2)

    def _create_orthogonal_vectors(self, n_basis, dim):
        """직교 벡터들 생성"""
        if n_basis <= dim:
            # QR decomposition 사용
            random_matrix = torch.randn(dim, n_basis)
            q, r = torch.linalg.qr(random_matrix)
            return q[:, :n_basis].T  # [n_basis, dim]
        else:
            # n_basis > dim이면 완전 직교 불가, 최대한 분산
            vectors = torch.randn(n_basis, dim)
            return F.normalize(vectors, dim=-1)

    def _gram_schmidt(self, vectors):
        """Gram-Schmidt 직교화"""
        n, d = vectors.shape
        orthogonal = torch.zeros_like(vectors)

        for i in range(n):
            v = vectors[i].clone()
            for j in range(i):
                # 이전 벡터들에 대한 projection 제거
                proj = torch.dot(orthogonal[j], vectors[i]) * orthogonal[j]
                v = v - proj
            # 정규화
            orthogonal[i] = F.normalize(v, dim=0)

        return orthogonal

    def get_neuron_emb(self, recipe):
        """Recipe로부터 neuron embedding 계산

        Args:
            recipe: [n_neurons, n_basis] - 각 neuron의 basis 조합 비율

        Returns:
            neuron_emb: [n_neurons, d_model]
        """
        # Softmax로 조합 비율 정규화
        weights = F.softmax(recipe, dim=-1)
        # Basis embedding의 가중 조합
        return torch.matmul(weights, self.basis_emb)


# ============================================
# 2. Simple Router (neuron_emb 외부에서 받음)
# ============================================
class SimpleRouter(nn.Module):
    """간단한 Neuron Router

    v7.0: neuron_emb를 외부(FFN)에서 받음
    """

    def __init__(self, d_model=256, n_heads=4, k=8):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.k = k

        # Context aggregation
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Score combination
        self.score_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x, neuron_emb, mask=None):
        """
        Args:
            x: [B, S, D] input tokens
            neuron_emb: [n_neurons, D] neuron embeddings (from recipe)
            mask: optional attention mask

        Returns:
            selected_idx: [B, S, k]
            weights: [B, S, k]
        """
        B, S, D = x.shape

        # 1. Self-attention for context
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, S, D)

        # 2. Combined representation
        combined = torch.cat([x, context], dim=-1)
        query = self.score_proj(combined)  # [B, S, D]

        # 3. Score against neurons
        scores = torch.matmul(query, neuron_emb.T)  # [B, S, n_neurons]

        # 4. Top-k selection
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        return topk_idx, weights


# ============================================
# 3. Recipe-based FFN
# ============================================
class RecipeFFN(nn.Module):
    """Recipe 기반 FFN

    핵심:
    - shared_basis에서 고정 Basis 사용
    - neuron_recipe만 학습
    - neuron_emb는 recipe @ basis_emb로 파생
    """

    def __init__(self, shared_basis, n_neurons=64, d_model=256):
        super().__init__()

        self.basis = shared_basis  # 공유 (고정)
        self.n_neurons = n_neurons
        self.n_basis = shared_basis.n_basis
        self.d_model = d_model

        # ===== 학습되는 파라미터 =====
        # Neuron recipe: 각 neuron이 basis를 어떻게 조합하는지
        self.neuron_recipe = nn.Parameter(
            torch.randn(n_neurons, self.n_basis) * 0.5
        )

        # Down projection
        self.W_down = nn.Linear(shared_basis.d_ff, d_model)

    @property
    def neuron_emb(self):
        """Neuron embedding = Basis embedding의 가중 조합 (파생!)"""
        return self.basis.get_neuron_emb(self.neuron_recipe)

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

        # 1. 선택된 neuron들의 recipe 가져오기
        # neuron_recipe: [n_neurons, n_basis]
        selected_recipes = self.neuron_recipe[neuron_idx]  # [B, S, k, n_basis]

        # 2. Softmax로 recipe 정규화
        selected_recipes = F.softmax(selected_recipes, dim=-1)

        # 3. Weighted sum → token별 최종 recipe
        # [B, S, k, n_basis] * [B, S, k, 1] → [B, S, n_basis]
        token_recipe = (selected_recipes * neuron_weights.unsqueeze(-1)).sum(dim=2)

        # 4. Recipe로 FFN weight 구성
        # token_recipe @ basis_A → weighted basis
        # [B, S, n_basis] @ [n_basis, d_model, rank] → [B, S, d_model, rank]
        W_A = torch.einsum('bsn,ndr->bsdr', token_recipe, self.basis.basis_A)

        # [B, S, n_basis] @ [n_basis, rank, d_ff] → [B, S, rank, d_ff]
        W_B = torch.einsum('bsn,nrf->bsrf', token_recipe, self.basis.basis_B)

        # 5. FFN 적용
        # x @ W_A: [B, S, D] @ [B, S, D, rank] → [B, S, rank]
        h = torch.einsum('bsd,bsdr->bsr', x, W_A)

        # h @ W_B: [B, S, rank] @ [B, S, rank, d_ff] → [B, S, d_ff]
        h = torch.einsum('bsr,bsrf->bsf', h, W_B)

        # 6. Activation + Down projection
        h = F.gelu(h)
        output = self.W_down(h)

        return output


# ============================================
# 4. DAWN Layer (v7.0)
# ============================================
class DAWNLayer(nn.Module):
    """DAWN v7.0 Layer

    구성:
    - Router: neuron 선택 (neuron_emb는 FFN에서 받음)
    - FFN: recipe 기반 dynamic FFN
    """

    def __init__(self, shared_basis, d_model=256, n_heads=4,
                 n_neurons=64, neuron_k=8):
        super().__init__()

        # Router
        self.router = SimpleRouter(
            d_model=d_model,
            n_heads=n_heads,
            k=neuron_k
        )

        # Recipe-based FFN
        self.ffn = RecipeFFN(
            shared_basis=shared_basis,
            n_neurons=n_neurons,
            d_model=d_model
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_indices=False):
        """
        Args:
            x: [B, S, D]
            mask: attention mask
            return_indices: return selected neuron indices

        Returns:
            x: [B, S, D]
            (optional) neuron_idx: [B, S, k]
        """
        # 1. Get neuron embeddings from FFN
        neuron_emb = self.ffn.neuron_emb  # [n_neurons, D]

        # 2. Router selects neurons
        normed = self.norm1(x)
        neuron_idx, neuron_weights = self.router(normed, neuron_emb, mask)

        # 3. FFN with selected neurons
        normed = self.norm2(x)
        ffn_out = self.ffn(normed, neuron_idx, neuron_weights)
        x = x + ffn_out

        if return_indices:
            return x, neuron_idx
        return x


# ============================================
# 5. DAWN Model (v7.0)
# ============================================
class DAWN(nn.Module):
    """DAWN v7.0 - Fixed Orthogonal Basis

    핵심 설계:
    - 32개 고정 직교 Basis (전체 공유)
    - 64개 Neuron per layer (recipe로 정의)
    - Token-level dynamic FFN

    장점:
    - 직교성 100% 보장
    - 파라미터 효율 (공유)
    - 해석 가능 (고정 좌표계)
    - 멀티모달 확장 용이
    """

    __version__ = "7.0"

    def __init__(self, vocab_size, d_model=256, d_ff=1024,
                 n_layers=4, n_heads=4,
                 n_basis=32, basis_rank=64,
                 n_neurons=64, neuron_k=8,
                 max_seq_len=512, dropout=0.1,
                 **kwargs):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.n_basis = n_basis

        # ===== 공유 고정 Basis =====
        self.shared_basis = FixedOrthogonalBasis(
            n_basis=n_basis,
            d_model=d_model,
            d_ff=d_ff,
            basis_rank=basis_rank
        )

        # ===== Embeddings =====
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # ===== Layers =====
        self.layers = nn.ModuleList([
            DAWNLayer(
                shared_basis=self.shared_basis,
                d_model=d_model,
                n_heads=n_heads,
                n_neurons=n_neurons,
                neuron_k=neuron_k
            )
            for _ in range(n_layers)
        ])

        # ===== Output =====
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # Weight tying

        # Config 저장
        self.config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'd_ff': d_ff,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_basis': n_basis,
            'basis_rank': basis_rank,
            'n_neurons': n_neurons,
            'neuron_k': neuron_k,
            'max_seq_len': max_seq_len,
            'dropout': dropout,
        }

        # For compatibility
        self.hidden_dim = d_model

        self._init_weights()

    def _init_weights(self):
        """Initialize learnable weights"""
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
        """
        Args:
            input_ids: [B, S]
            return_activations: return neuron indices per layer

        Returns:
            logits: [B, S, vocab_size]
            (optional) all_neuron_idx: list of [B, S, k]
        """
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

    def get_loss(self, input_ids, labels):
        """Compute loss (no orthogonality loss needed!)

        Args:
            input_ids: [B, S]
            labels: [B, S]

        Returns:
            total_loss, loss_dict
        """
        logits = self.forward(input_ids)

        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=-100
        )

        return loss, {'ce': loss.item(), 'total': loss.item()}

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """Auto-regressive generation"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids)
                next_logits = logits[:, -1, :] / temperature

                if top_k:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def analyze_basis_usage(self):
        """Analyze how each layer uses the shared basis"""
        results = {}

        for i, layer in enumerate(self.layers):
            recipe = layer.ffn.neuron_recipe  # [n_neurons, n_basis]

            # Softmax로 정규화
            recipe_norm = F.softmax(recipe, dim=-1)

            # 각 basis의 평균 사용률
            basis_usage = recipe_norm.mean(dim=0).detach().cpu().numpy()

            # Recipe 다양성 (neuron 간 차이)
            recipe_std = recipe_norm.std(dim=0).mean().item()

            # Neuron embedding 분석
            neuron_emb = layer.ffn.neuron_emb

            # Neuron 간 유사도
            neuron_sim = torch.mm(
                F.normalize(neuron_emb, dim=-1),
                F.normalize(neuron_emb, dim=-1).T
            )
            # 대각선 제외한 평균 유사도
            mask = 1 - torch.eye(self.n_neurons, device=neuron_sim.device)
            avg_similarity = (neuron_sim * mask).sum() / mask.sum()

            results[f'layer_{i}'] = {
                'basis_usage': basis_usage.tolist(),
                'recipe_diversity': recipe_std,
                'neuron_similarity': avg_similarity.item(),
            }

        return results

    def get_num_params(self):
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Breakdown
        basis_params = sum(
            b.numel() for b in self.shared_basis.buffers()
        )  # Fixed, not trainable

        recipe_params = sum(
            layer.ffn.neuron_recipe.numel()
            for layer in self.layers
        )
        router_params = sum(
            sum(p.numel() for p in layer.router.parameters())
            for layer in self.layers
        )

        emb_params = self.token_emb.weight.numel() + self.pos_emb.weight.numel()

        return {
            'total': total,
            'trainable': trainable,
            'basis_fixed': basis_params,
            'recipe': recipe_params,
            'router': router_params,
            'embeddings': emb_params,
        }

    def verify_orthogonality(self):
        """Verify that the basis is truly orthogonal

        Returns:
            dict with orthogonality metrics
        """
        basis = self.shared_basis

        # Check Basis A
        basis_A_flat = basis.basis_A.view(self.n_basis, -1)
        gram_A = torch.mm(basis_A_flat, basis_A_flat.T)
        off_diag_A = gram_A - torch.diag(torch.diag(gram_A))
        max_off_diag_A = off_diag_A.abs().max().item()

        # Check Basis B
        basis_B_flat = basis.basis_B.view(self.n_basis, -1)
        gram_B = torch.mm(basis_B_flat, basis_B_flat.T)
        off_diag_B = gram_B - torch.diag(torch.diag(gram_B))
        max_off_diag_B = off_diag_B.abs().max().item()

        # Check Basis embedding
        gram_emb = torch.mm(basis.basis_emb, basis.basis_emb.T)
        off_diag_emb = gram_emb - torch.diag(torch.diag(gram_emb))
        max_off_diag_emb = off_diag_emb.abs().max().item()

        return {
            'basis_A_max_off_diag': max_off_diag_A,
            'basis_B_max_off_diag': max_off_diag_B,
            'basis_emb_max_off_diag': max_off_diag_emb,
            'is_orthogonal': max(max_off_diag_A, max_off_diag_B, max_off_diag_emb) < 1e-5
        }


# ============================================
# 6. Helper Functions
# ============================================
def create_model(config):
    """Create DAWN model from config"""
    return DAWN(**config)


def count_parameters(model):
    """Count trainable parameters"""
    return model.get_num_params()


# Backward compatibility
DAWNLanguageModel = DAWN
OrthogonalBasisFFN = RecipeFFN
NeuronRouter = SimpleRouter
Layer = DAWNLayer


# ============================================
# 7. Test
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("DAWN v7.0 - Fixed Orthogonal Basis")
    print("=" * 60)

    # Config
    config = {
        'vocab_size': 30000,
        'd_model': 256,
        'd_ff': 1024,
        'n_layers': 4,
        'n_heads': 4,
        'n_basis': 32,
        'basis_rank': 64,
        'n_neurons': 64,
        'neuron_k': 8,
        'max_seq_len': 512,
        'dropout': 0.1,
    }

    # Create model
    model = DAWN(**config)

    # Parameter count
    params = model.get_num_params()
    print(f"\nParameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Basis (fixed): {params['basis_fixed']:,}")
    print(f"  Recipe: {params['recipe']:,}")
    print(f"  Router: {params['router']:,}")
    print(f"  Embeddings: {params['embeddings']:,}")

    # Check basis orthogonality
    print(f"\nBasis Orthogonality Check:")
    orth = model.verify_orthogonality()
    print(f"  Basis A - Max off-diagonal: {orth['basis_A_max_off_diag']:.6f}")
    print(f"  Basis B - Max off-diagonal: {orth['basis_B_max_off_diag']:.6f}")
    print(f"  Basis Emb - Max off-diagonal: {orth['basis_emb_max_off_diag']:.6f}")
    print(f"  Is Orthogonal: {orth['is_orthogonal']}")

    # Test forward
    print(f"\nForward Pass Test:")
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
    print(f"  Neuron indices per layer: {neuron_indices[0].shape}")

    # Loss
    loss, loss_dict = model.get_loss(input_ids, labels)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss components: {loss_dict}")

    # Basis usage
    print(f"\nBasis Usage Analysis:")
    analysis = model.analyze_basis_usage()
    for layer_name, stats in analysis.items():
        print(f"  {layer_name}:")
        print(f"    Recipe diversity: {stats['recipe_diversity']:.4f}")
        print(f"    Neuron similarity: {stats['neuron_similarity']:.4f}")

    print(f"\n✅ DAWN v7.0 Ready!")
