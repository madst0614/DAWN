"""
DAWN v7.1 - Dynamic Architecture With Neurons

핵심 변경 (v7.0 → v7.1):
- W_down 제거!
- Up/Down이 같은 Basis 사용 (Symmetric)
- 파라미터 대폭 감소

철학:
- Basis = 의미 공간의 좌표축
- Up: 토큰 → 의미 공간으로 인코딩
- Down: 의미 공간 → 토큰으로 디코딩
- 같은 축으로 올라갔다 내려오기!

구조:
x → W_A → W_B → GELU → W_B.T → W_A.T → output
    (up)              (down = transpose)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 1. Fixed Orthogonal Basis (v7.0과 동일)
# ============================================
class FixedOrthogonalBasis(nn.Module):
    """고정된 직교 Basis - 전체 Layer 공유, 학습 안 함"""

    def __init__(self, n_basis=32, d_model=256, d_ff=1024, basis_rank=64):
        super().__init__()

        self.n_basis = n_basis
        self.d_model = d_model
        self.d_ff = d_ff
        self.basis_rank = basis_rank

        # ===== 고정 직교 Basis (학습 X) =====

        # Basis A: [n_basis, d_model, rank]
        basis_A = self._create_orthogonal_basis(n_basis, d_model, basis_rank)
        self.register_buffer('basis_A', basis_A)

        # Basis B: [n_basis, rank, d_ff]
        basis_B = self._create_orthogonal_basis(n_basis, basis_rank, d_ff)
        self.register_buffer('basis_B', basis_B)

        # Basis embedding: [n_basis, d_model]
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
                proj = torch.dot(orthogonal[j], vectors[i]) * orthogonal[j]
                v = v - proj
            norm = torch.norm(v)
            if norm > 1e-10:
                orthogonal[i] = v / norm
            else:
                orthogonal[i] = v

        return orthogonal

    def get_neuron_emb(self, recipe):
        """Recipe로부터 neuron embedding 계산"""
        weights = F.softmax(recipe, dim=-1)
        return torch.matmul(weights, self.basis_emb)


# ============================================
# 2. Simple Router (v7.0과 동일)
# ============================================
class SimpleRouter(nn.Module):
    """Neuron Router"""

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
# 3. Symmetric Basis FFN (v7.1 핵심!)
# ============================================
class SymmetricBasisFFN(nn.Module):
    """Symmetric Basis FFN - W_down 제거!

    핵심 변경:
    - Up과 Down이 같은 Basis 사용
    - Down = Up의 Transpose
    - W_down 파라미터 완전 제거

    수학적 의미:
    - Up: x @ W_A @ W_B = x를 의미 공간으로
    - Down: h @ W_B.T @ W_A.T = 의미 공간에서 토큰 공간으로
    - 같은 축으로 투영했다가 복원
    """

    def __init__(self, shared_basis, n_neurons=64, d_model=256):
        super().__init__()

        self.basis = shared_basis
        self.n_neurons = n_neurons
        self.n_basis = shared_basis.n_basis
        self.d_model = d_model

        # ===== 학습되는 파라미터 =====
        # Neuron recipe만! (W_down 없음)
        self.neuron_recipe = nn.Parameter(
            torch.randn(n_neurons, self.n_basis) * 2.0
        )

        # Scaling factor (선택적, 안정성 위해)
        self.output_scale = nn.Parameter(torch.ones(1))

    @property
    def neuron_emb(self):
        """Neuron embedding = Basis embedding의 가중 조합"""
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
        selected_recipes = self.neuron_recipe[neuron_idx]  # [B, S, k, n_basis]
        selected_recipes = F.softmax(selected_recipes, dim=-1)

        # 2. Weighted sum → token별 최종 recipe
        token_recipe = (selected_recipes * neuron_weights.unsqueeze(-1)).sum(dim=2)
        # [B, S, n_basis]

        # 3. Basis 조합으로 W_A, W_B 구성
        W_A = torch.einsum('bsn,ndr->bsdr', token_recipe, self.basis.basis_A)
        # [B, S, d_model, rank]

        W_B = torch.einsum('bsn,nrf->bsrf', token_recipe, self.basis.basis_B)
        # [B, S, rank, d_ff]

        # 4. Up projection: x → h
        h = torch.einsum('bsd,bsdr->bsr', x, W_A)      # [B, S, rank]
        h = torch.einsum('bsr,bsrf->bsf', h, W_B)      # [B, S, d_ff]

        # 5. Activation
        h = F.gelu(h)

        # 6. Down projection: Transpose! (핵심!)
        # W_B.T: [B, S, d_ff, rank]
        h = torch.einsum('bsf,bsrf->bsr', h, W_B)      # [B, S, rank]
        # W_A.T: [B, S, rank, d_model]
        output = torch.einsum('bsr,bsdr->bsd', h, W_A)  # [B, S, d_model]

        # 7. Output scaling (선택적)
        output = output * self.output_scale

        return output


# ============================================
# 4. DAWN Layer (v7.1)
# ============================================
class DAWNLayer(nn.Module):
    """DAWN v7.1 Layer with Symmetric FFN"""

    def __init__(self, shared_basis, d_model=256, n_heads=4,
                 n_neurons=64, neuron_k=8):
        super().__init__()

        self.router = SimpleRouter(
            d_model=d_model,
            n_heads=n_heads,
            k=neuron_k
        )

        # v7.1: Symmetric FFN (W_down 없음!)
        self.ffn = SymmetricBasisFFN(
            shared_basis=shared_basis,
            n_neurons=n_neurons,
            d_model=d_model
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_indices=False):
        # 1. Get neuron embeddings
        neuron_emb = self.ffn.neuron_emb

        # 2. Router selects neurons
        normed = self.norm1(x)
        neuron_idx, neuron_weights = self.router(normed, neuron_emb, mask)

        # 3. Symmetric FFN
        normed = self.norm2(x)
        ffn_out = self.ffn(normed, neuron_idx, neuron_weights)
        x = x + ffn_out

        if return_indices:
            return x, neuron_idx
        return x


# ============================================
# 5. DAWN Model (v7.1)
# ============================================
class DAWN(nn.Module):
    """DAWN v7.1 - Symmetric Basis FFN

    핵심 변경:
    - W_down 제거로 파라미터 대폭 감소
    - Up/Down이 같은 Basis 사용
    - 더 해석 가능한 구조

    파라미터 비교 (per layer):
    - v7.0: W_down = 1024 × 256 = 262K
    - v7.1: W_down = 0 (제거!)
    """

    __version__ = "7.1"

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

        # Config
        self.config = {
            'model_version': self.__version__,
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

    def compute_load_balance(self, neuron_indices):
        """Compute load balance loss for a layer"""
        flat_indices = neuron_indices.flatten()
        counts = torch.bincount(flat_indices, minlength=self.n_neurons).float()

        total = counts.sum()
        if total == 0:
            return torch.tensor(0.0, device=neuron_indices.device)

        freq = counts / total
        mean_freq = freq.mean()
        std_freq = freq.std()
        cv = std_freq / (mean_freq + 1e-8)

        return cv

    def get_loss(self, input_ids, labels, diversity_weight=0.0, load_balance_weight=0.0):
        """Compute loss with optional regularization

        Args:
            input_ids: [B, S]
            labels: [B, S]
            diversity_weight: Weight for recipe diversity loss
            load_balance_weight: Weight for load balance loss

        Returns:
            total_loss, loss_dict, logits
        """
        # Forward with activation tracking if load balance is enabled
        if load_balance_weight > 0:
            logits, neuron_indices = self.forward(input_ids, return_activations=True)
        else:
            logits = self.forward(input_ids)

        # CE loss
        ce_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=-100
        )

        total_loss = ce_loss
        loss_dict = {'ce': ce_loss.item()}

        # Recipe diversity loss
        if diversity_weight > 0:
            div_loss = 0
            for layer in self.layers:
                recipe = layer.ffn.neuron_recipe
                recipe_norm = F.softmax(recipe, dim=-1)
                recipe_normalized = F.normalize(recipe_norm, dim=-1)
                similarity = torch.mm(recipe_normalized, recipe_normalized.T)
                mask = 1 - torch.eye(self.n_neurons, device=similarity.device)
                avg_sim = (similarity * mask).sum() / mask.sum()
                div_loss += avg_sim
            div_loss = div_loss / len(self.layers)
            total_loss = total_loss + diversity_weight * div_loss
            loss_dict['diversity'] = div_loss.item()

        # Load balance loss
        if load_balance_weight > 0:
            lb_loss = 0
            for layer_indices in neuron_indices:
                lb_loss += self.compute_load_balance(layer_indices)
            lb_loss = lb_loss / len(neuron_indices)
            total_loss = total_loss + load_balance_weight * lb_loss
            loss_dict['load_balance'] = lb_loss.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict, logits

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
            recipe = layer.ffn.neuron_recipe
            recipe_norm = F.softmax(recipe, dim=-1)

            basis_usage = recipe_norm.mean(dim=0).detach().cpu().numpy()
            recipe_std = recipe_norm.std(dim=0).mean().item()

            neuron_emb = layer.ffn.neuron_emb
            neuron_sim = torch.mm(
                F.normalize(neuron_emb, dim=-1),
                F.normalize(neuron_emb, dim=-1).T
            )
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
        )

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
        """Verify that the basis is truly orthogonal"""
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
SymmetricFFN = SymmetricBasisFFN
NeuronRouter = SimpleRouter
Layer = DAWNLayer


# ============================================
# 7. Test
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("DAWN v7.1 - Symmetric Basis FFN")
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
    print(f"\n📊 Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Recipe: {params['recipe']:,}")
    print(f"  Router: {params['router']:,}")

    # v7.0 vs v7.1 comparison
    print(f"\n📊 v7.0 vs v7.1 비교:")
    v70_wdown = 1024 * 256 * 4  # W_down per layer × 4 layers
    print(f"  v7.0 W_down: {v70_wdown:,} params")
    print(f"  v7.1 W_down: 0 params (제거!)")
    print(f"  절약: {v70_wdown:,} params ({v70_wdown/params['total']*100:.1f}%)")

    # Check basis orthogonality
    print(f"\n📊 Basis 직교성:")
    orth = model.verify_orthogonality()
    print(f"  Basis A - Max off-diagonal: {orth['basis_A_max_off_diag']:.6f}")
    print(f"  Basis B - Max off-diagonal: {orth['basis_B_max_off_diag']:.6f}")
    print(f"  Basis Emb - Max off-diagonal: {orth['basis_emb_max_off_diag']:.6f}")
    print(f"  Is Orthogonal: {orth['is_orthogonal']}")

    # Test forward
    print(f"\n📊 Forward Pass Test:")
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
    loss, loss_dict, _ = model.get_loss(input_ids, labels,
                                         diversity_weight=0.05,
                                         load_balance_weight=0.01)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss components: {loss_dict}")

    # Basis usage
    print(f"\n📊 Basis Usage Analysis:")
    analysis = model.analyze_basis_usage()
    for layer_name, stats in analysis.items():
        print(f"  {layer_name}:")
        print(f"    Recipe diversity: {stats['recipe_diversity']:.4f}")
        print(f"    Neuron similarity: {stats['neuron_similarity']:.4f}")

    print(f"\n✅ DAWN v7.1 Ready!")
    print(f"   W_down 제거됨!")
    print(f"   Up/Down이 같은 Basis 사용!")
