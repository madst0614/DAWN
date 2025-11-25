"""
DAWN v8.0 - Unified Tensor Product Architecture

혁명적 변화:
- Attention + FFN 통합!
- 정보 수집과 변환을 하나의 operation으로
- Row basis = Query (어떤 정보?)
- Col basis = Transformation (어떻게 변환?)

철학:
기존: 정보 수집(Attention) → 변환(FFN) (2단계)
v8.0: 정보 수집하면서 동시에 변환 (1단계!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 1. Unified Tensor Product Basis
# ============================================
class UnifiedTensorBasis(nn.Module):
    """통합 Tensor Product Basis

    Row basis: Query-like (어떤 정보를 수집?)
    Col basis: Value-like + Transform (어떻게 변환?)

    핵심:
    output = Σᵢⱼ attention(i,j) × (Row_i @ Col_j)

    where:
    - attention(i,j) = token i가 token j를 볼 때의 weight
    - Row_i = token i의 query basis
    - Col_j = token j의 key-value-transform basis
    """

    def __init__(self, n_row=8, n_col=8, d_model=256, mid_dim=32):
        super().__init__()

        self.n_row = n_row
        self.n_col = n_col
        self.d_model = d_model
        self.mid_dim = mid_dim

        # Row basis: Query generation
        # [n_row, d_model, mid_dim]
        # "어떤 정보가 필요한가?"
        row_basis = self._create_orthogonal_basis(n_row, d_model, mid_dim)
        self.register_buffer('row_basis', row_basis)

        # Col basis: Key-Value-Transform
        # [n_col, mid_dim, d_model]
        # "어떤 정보를 제공하고 어떻게 변환?"
        col_basis = self._create_orthogonal_basis(n_col, mid_dim, d_model)
        self.register_buffer('col_basis', col_basis)

        # Basis embedding for routing
        # [n_row * n_col, d_model]
        basis_emb = self._create_basis_embeddings()
        self.register_buffer('basis_emb', basis_emb)

    def _create_orthogonal_basis(self, n_basis, dim1, dim2):
        """직교 basis 생성"""
        random_tensor = torch.randn(n_basis, dim1, dim2)
        flat = random_tensor.view(n_basis, -1)

        # Gram-Schmidt
        orthogonal = torch.zeros_like(flat)
        for i in range(n_basis):
            v = flat[i].clone()
            for j in range(i):
                proj = torch.dot(orthogonal[j], flat[i]) * orthogonal[j]
                v = v - proj

            norm = torch.norm(v)
            if norm > 1e-10:
                orthogonal[i] = v / norm
            else:
                orthogonal[i] = v

        return orthogonal.view(n_basis, dim1, dim2) * math.sqrt(dim2)

    def _create_basis_embeddings(self):
        """각 (row, col) 조합의 embedding"""
        embeddings = []

        for i in range(self.n_row):
            for j in range(self.n_col):
                # Row + Col 정보 결합
                # row_basis[i]: [d_model, mid_dim] → mean over mid_dim → [d_model]
                row_emb = self.row_basis[i].mean(dim=-1)  # [d_model]
                # col_basis[j]: [mid_dim, d_model] → mean over mid_dim → [d_model]
                col_emb = self.col_basis[j].mean(dim=0)   # [d_model]

                # 평균으로 조합
                combined = (row_emb + col_emb) / 2
                embeddings.append(combined)

        embeddings = torch.stack(embeddings)
        embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def get_neuron_emb(self, recipe):
        """Recipe로부터 neuron embedding 계산"""
        weights = F.softmax(recipe, dim=-1)
        return torch.matmul(weights, self.basis_emb)


# ============================================
# 2. Unified Tensor Product Block
# ============================================
class UnifiedTensorBlock(nn.Module):
    """통합 Tensor Product Block

    Attention + FFN을 하나로!

    동작:
    1. Recipe로 각 token의 (Row, Col) 조합 선택
    2. Row로 Query 생성 (어떤 정보 필요?)
    3. Col로 Key-Value 생성 (어떤 정보 제공?)
    4. Attention으로 정보 수집
    5. 수집된 정보를 Col로 변환
    6. 모든 (Row, Col) 조합의 weighted sum
    """

    def __init__(self, shared_basis, n_neurons=64, d_model=256):
        super().__init__()

        self.basis = shared_basis
        self.n_neurons = n_neurons
        self.n_row = shared_basis.n_row
        self.n_col = shared_basis.n_col
        self.d_model = d_model
        self.mid_dim = shared_basis.mid_dim

        # Neuron recipes: [n_neurons, n_row * n_col]
        n_combinations = self.n_row * self.n_col
        self.neuron_recipe = nn.Parameter(
            torch.randn(n_neurons, n_combinations) * 0.02  # 작게!
        )

        # Output projection (residual connection 위해)
        self.out_proj = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # transform weight

    @property
    def neuron_emb(self):
        """Neuron embedding for routing"""
        return self.basis.get_neuron_emb(self.neuron_recipe)

    def forward(self, x, neuron_idx, neuron_weights):
        """
        Args:
            x: [B, S, d_model]
            neuron_idx: [B, S, k]
            neuron_weights: [B, S, k]

        Returns:
            output: [B, S, d_model]
        """
        B, S, D = x.shape
        k = neuron_idx.shape[-1]

        # 1. 선택된 neuron recipes
        selected_recipes = self.neuron_recipe[neuron_idx]  # [B, S, k, 64]
        selected_recipes = F.softmax(selected_recipes, dim=-1)

        # 2. Token별 최종 recipe (weighted combination)
        token_recipe = (selected_recipes * neuron_weights.unsqueeze(-1)).sum(dim=2)
        # [B, S, 64] → [B, S, 8, 8] (row × col)
        token_recipe_2d = token_recipe.view(B, S, self.n_row, self.n_col)

        # Normalize
        recipe_norm = F.softmax(token_recipe_2d.view(B, S, -1), dim=-1)
        token_recipe_2d = recipe_norm.view(B, S, self.n_row, self.n_col)

        # 3. Row weights (어떤 query pattern?)
        row_weights = token_recipe_2d.sum(dim=-1)  # [B, S, n_row]

        # 4. Col weights (어떤 key-value pattern?)
        col_weights = token_recipe_2d.sum(dim=-2)  # [B, S, n_col]

        # 5. Generate Queries (Row basis)
        # [B, S, n_row] × [n_row, d_model, mid] → [B, S, d_model, mid]
        queries = torch.einsum('bsr,rij->bsij', row_weights, self.basis.row_basis)
        # Sum over d_model to get query vectors (정보 보존)
        queries = queries.sum(dim=2)  # [B, S, mid]

        # 6. Generate Keys (Col basis)
        # [B, S, n_col] × [n_col, mid, d_model] → [B, S, mid, d_model]
        keys = torch.einsum('bsc,cij->bsij', col_weights, self.basis.col_basis)
        keys = keys.transpose(-2, -1).sum(dim=2)  # [B, S, mid]

        # 7. Attention scores
        # [B, S, mid] @ [B, S, mid].T → [B, S, S]
        attention_scores = torch.bmm(queries, keys.transpose(-2, -1)) / math.sqrt(self.mid_dim)

        # Causal mask (for language modeling)
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        attention_scores.masked_fill_(causal_mask, float('-inf'))

        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 8. Gather information with attention
        # [B, S, S] @ [B, S, d_model] → [B, S, d_model]
        context = torch.bmm(attention_weights, x)

        # 9. Transform with Col basis
        # [B, S, n_col] × [n_col, mid, d_model] → [B, S, mid, d_model]
        transform = torch.einsum('bsc,cij->bsij', col_weights, self.basis.col_basis)

        # Apply transformation to context (weighted sum)
        output = context + self.alpha * transform.sum(dim=2)  # [B, S, d_model]

        # GELU activation
        output = F.gelu(output)

        # Output projection
        output = self.out_proj(output)

        return output


# ============================================
# 3. Simple Router (v7.0과 동일)
# ============================================
class SimpleRouter(nn.Module):
    """간단한 neuron router"""

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

        # Score neurons
        scores = torch.matmul(query, neuron_emb.T)

        # Top-k
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        return topk_idx, weights


# ============================================
# 4. DAWN Layer v8.0
# ============================================
class DAWNLayer(nn.Module):
    """DAWN Layer v8.0 - Unified Architecture

    기존: Attention → FFN (2 modules)
    v8.0: Unified Tensor Product (1 module!)
    """

    def __init__(self, shared_basis, d_model=256, n_heads=4,
                 n_neurons=64, neuron_k=8):
        super().__init__()

        self.router = SimpleRouter(d_model, n_heads, neuron_k)
        self.unified_block = UnifiedTensorBlock(shared_basis, n_neurons, d_model)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_indices=False):
        # Get neuron embeddings
        neuron_emb = self.unified_block.neuron_emb

        # Router selects neurons
        normed = self.norm(x)
        neuron_idx, neuron_weights = self.router(normed, neuron_emb, mask)

        # Unified block (Attention + FFN 통합!)
        output = self.unified_block(normed, neuron_idx, neuron_weights)

        # Residual
        x = x + output

        if return_indices:
            return x, neuron_idx
        return x


# ============================================
# 5. DAWN Model v8.0
# ============================================
class DAWN(nn.Module):
    """DAWN v8.0 - Unified Tensor Product Architecture

    혁명적 변화:
    - Attention + FFN 통합
    - 정보 수집과 변환을 하나의 operation으로
    - 더 효율적, 더 해석 가능
    """

    __version__ = "8.0"

    def __init__(self, vocab_size, d_model=256,
                 n_layers=4, n_heads=4,
                 n_row=8, n_col=8, mid_dim=32,
                 n_neurons=64, neuron_k=8,
                 max_seq_len=512, dropout=0.1,
                 **kwargs):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_neurons = n_neurons

        # Shared Unified Tensor Basis
        self.shared_basis = UnifiedTensorBasis(
            n_row=n_row,
            n_col=n_col,
            d_model=d_model,
            mid_dim=mid_dim
        )

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Layers (Unified!)
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

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # Weight tying

        # Config
        self.config = {
            'model_version': self.__version__,
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_row': n_row,
            'n_col': n_col,
            'mid_dim': mid_dim,
            'n_neurons': n_neurons,
            'neuron_k': neuron_k,
        }

    def forward(self, x, return_neuron_indices=False):
        B, S = x.shape

        # Embeddings
        positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.dropout(x)

        # Layers
        neuron_indices = []
        for layer in self.layers:
            if return_neuron_indices:
                x, indices = layer(x, return_indices=True)
                neuron_indices.append(indices)
            else:
                x = layer(x)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        if return_neuron_indices:
            return logits, neuron_indices
        return logits

    def compute_diversity_loss(self):
        """Recipe diversity loss"""
        total_loss = 0.0
        for layer in self.layers:
            recipe = layer.unified_block.neuron_recipe
            recipe_norm = F.softmax(recipe, dim=-1)

            # Cosine similarity
            recipe_normalized = F.normalize(recipe_norm, dim=-1)
            sim_matrix = torch.mm(recipe_normalized, recipe_normalized.T)

            # Off-diagonal mean
            mask = 1 - torch.eye(self.n_neurons, device=recipe.device)
            loss = (sim_matrix * mask).sum() / mask.sum()
            total_loss += loss

        return total_loss / len(self.layers)

    def compute_load_balance(self, neuron_indices):
        """Load balance loss"""
        all_indices = neuron_indices.reshape(-1)
        counts = torch.bincount(all_indices, minlength=self.n_neurons)
        target = counts.sum() / self.n_neurons
        loss = ((counts.float() - target) ** 2).mean()
        return loss

    def get_loss(self, x, targets, diversity_weight=0.0, load_balance_weight=0.0):
        """Forward with auxiliary losses (compatible with training script)"""
        return self.forward_with_loss(
            x, targets,
            diversity_weight=diversity_weight,
            load_balance_weight=load_balance_weight
        )

    def forward_with_loss(self, x, targets,
                         diversity_weight=0.0,
                         load_balance_weight=0.0):
        """Forward with auxiliary losses"""
        logits, neuron_indices = self.forward(x, return_neuron_indices=True)

        # Cross entropy
        ce_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )

        # Diversity loss
        diversity_loss = torch.tensor(0.0, device=x.device)
        if diversity_weight > 0:
            diversity_loss = self.compute_diversity_loss()

        # Load balance loss
        lb_loss = torch.tensor(0.0, device=x.device)
        if load_balance_weight > 0:
            for layer_indices in neuron_indices:
                lb_loss += self.compute_load_balance(layer_indices)
            lb_loss = lb_loss / len(neuron_indices)

        # Total loss
        total_loss = ce_loss + diversity_weight * diversity_loss + load_balance_weight * lb_loss

        loss_dict = {
            'ce': ce_loss.item(),
            'diversity': diversity_loss.item() if diversity_weight > 0 else 0.0,
            'load_balance': lb_loss.item() if load_balance_weight > 0 else 0.0,
            'total': total_loss.item()
        }

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


# ============================================
# Backward Compatibility
# ============================================
DAWNLanguageModel = DAWN  # Alias


def create_model(vocab_size, **kwargs):
    """Create DAWN v8.0 model"""
    return DAWN(vocab_size=vocab_size, **kwargs)


def create_model_v80(vocab_size, **kwargs):
    """Create DAWN v8.0 model (explicit version)"""
    return DAWN(vocab_size=vocab_size, **kwargs)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
