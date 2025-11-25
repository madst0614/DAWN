"""
DAWN v7.4 - TT Weighted Karcher Mean

v7.0 기반:
- Fixed Orthogonal Basis (TT 형태)
- Neuron recipes
- Router selection

v7.4 추가:
- TT Karcher Mean (weighted centroid)
- Neuron 조합 시 rank 증폭
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .tt_karcher import TTBasisWithKarcher, TTKarcherMean


class KarcherFFN(nn.Module):
    """
    v7.4 FFN with Karcher Mean

    8개 Neuron을 Karcher mean으로 조합
    → Rank 증폭 효과
    """

    def __init__(self, shared_basis, n_neurons=64, d_model=256):
        super().__init__()

        self.basis = shared_basis  # TTBasisWithKarcher
        self.n_neurons = n_neurons
        self.d_model = d_model

        # Neuron recipes
        self.neuron_recipes = nn.Parameter(
            torch.randn(n_neurons, shared_basis.n_basis) * 0.5
        )

        # Down projection
        self.w_down = nn.Linear(1024, d_model)

    @property
    def neuron_emb(self):
        """Neuron embedding for routing"""
        recipe_norm = F.softmax(self.neuron_recipes, dim=-1)
        return torch.matmul(recipe_norm, self.basis.basis_emb)

    def forward(self, x, neuron_idx, neuron_weights):
        """
        Args:
            x: [B, S, 256]
            neuron_idx: [B, S, k] selected neuron indices
            neuron_weights: [B, S, k] neuron weights
        """
        B, S, D = x.shape
        k = neuron_idx.shape[-1]

        # 1. 선택된 neuron들의 recipes
        selected_recipes = self.neuron_recipes[neuron_idx]  # [B, S, k, 32]
        selected_recipes = F.softmax(selected_recipes, dim=-1)

        # 2. 각 neuron의 TT cores 생성
        neuron_cores_A_list = []
        neuron_cores_B_list = []

        for i in range(k):
            recipe_i = selected_recipes[:, :, i, :]  # [B, S, 32]
            cores_A, cores_B = self.basis.get_neuron_tt_cores(recipe_i)
            neuron_cores_A_list.append(cores_A)
            neuron_cores_B_list.append(cores_B)

        # 3. Karcher Mean으로 centroid 찾기!
        centroid_A, centroid_B = self.basis.karcher(
            neuron_cores_A_list,
            neuron_cores_B_list,
            neuron_weights
        )

        # 4. Centroid TT로 FFN 적용
        output = self.apply_tt_ffn(x, centroid_A, centroid_B)

        return output

    def apply_tt_ffn(self, x, cores_A, cores_B):
        """
        TT cores로 FFN 적용

        TT Contraction 원리:
        - Input matrix를 fold: x[i,j]
        - TT cores: core1[i,r,k], core2[r,j,l]
        - Contract: sum_i,j,r [ x[i,j] * core1[i,r,k] * core2[r,j,l] ] = output[k,l]

        Basis_A: [256→64] = [16×16 → 8×8]
        Basis_B: [64→1024] = [8×8 → 32×32]

        Args:
            x: [B, S, 256]
            cores_A: Dict{'core1': [B, S, 16, rank, 8],
                         'core2': [B, S, rank, 16, 8]}
            cores_B: Dict{'core1': [B, S, 8, rank, 32],
                         'core2': [B, S, rank, 8, 32]}
        """
        B, S, D = x.shape

        # === Basis_A: [256] → [64] ===
        # x를 fold: [B, S, 256] → [B, S, 16, 16]
        x_fold = x.view(B, S, 16, 16)  # [B, S, i, j]

        # TT contraction for Basis_A
        # x_fold: [B, S, i, j]
        # cores_A['core1']: [B, S, i, r, k]
        # cores_A['core2']: [B, S, r, j, l]

        # Step 1: contract over i dimension
        temp = torch.einsum('bsij,bsirk->bsjrk', x_fold, cores_A['core1'])
        # temp: [B, S, j, r, k] - i가 사라짐

        # Step 2: contract over j and r dimensions
        h = torch.einsum('bsjrk,bsrjl->bskl', temp, cores_A['core2'])
        # h: [B, S, k, l] = [B, S, 8, 8]
        h = h.reshape(B, S, 64)

        # === Basis_B: [64] → [1024] ===
        # h를 fold: [B, S, 64] → [B, S, 8, 8]
        h_fold = h.view(B, S, 8, 8)  # [B, S, i, j]

        # TT contraction for Basis_B
        # h_fold: [B, S, i, j]
        # cores_B['core1']: [B, S, i, r, k]
        # cores_B['core2']: [B, S, r, j, l]

        # Step 1: contract over i dimension
        temp = torch.einsum('bsij,bsirk->bsjrk', h_fold, cores_B['core1'])
        # temp: [B, S, j, r, k] - i가 사라짐

        # Step 2: contract over j and r dimensions
        output = torch.einsum('bsjrk,bsrjl->bskl', temp, cores_B['core2'])
        # output: [B, S, k, l] = [B, S, 32, 32]
        output = output.reshape(B, S, 1024)

        # GELU
        output = F.gelu(output)

        # Down projection: [1024] → [256]
        output = self.w_down(output)

        return output


class SimpleRouter(nn.Module):
    """
    v7.0과 동일한 Router
    """

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

        # Self-attention
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


class DAWNLayerV74(nn.Module):
    """
    DAWN Layer v7.4
    """

    def __init__(self, shared_basis, d_model=256, n_heads=4,
                 n_neurons=64, neuron_k=8):
        super().__init__()

        self.router = SimpleRouter(d_model, n_heads, neuron_k)
        self.ffn = KarcherFFN(shared_basis, n_neurons, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_indices=False):
        # Router
        neuron_emb = self.ffn.neuron_emb
        normed = self.norm1(x)
        neuron_idx, neuron_weights = self.router(normed, neuron_emb, mask)

        # Karcher FFN
        normed = self.norm2(x)
        ffn_out = self.ffn(normed, neuron_idx, neuron_weights)
        x = x + ffn_out

        if return_indices:
            return x, neuron_idx
        return x


class DAWN(nn.Module):
    """
    DAWN v7.4 - TT Weighted Karcher Mean

    Changes from v7.0:
    - TT Basis storage (메모리 효율)
    - Karcher Mean neuron combination (rank 증폭)
    """

    __version__ = "7.4"

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

        # Shared TT Basis with Karcher
        self.shared_basis = TTBasisWithKarcher(
            n_basis=n_basis,
            d_model=d_model,
            d_ff=d_ff,
            basis_rank=basis_rank
        )

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Layers
        self.layers = nn.ModuleList([
            DAWNLayerV74(
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
        self.head.weight = self.token_emb.weight

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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
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
        all_neuron_indices = []
        for layer in self.layers:
            if return_activations:
                x, neuron_idx = layer(x, mask, return_indices=True)
                all_neuron_indices.append(neuron_idx)
            else:
                x = layer(x, mask)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        if return_activations:
            return logits, all_neuron_indices
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
        """Compute loss with optional diversity and load balance regularization"""
        if load_balance_weight > 0:
            logits, neuron_indices = self.forward(input_ids, return_activations=True)
        else:
            logits = self.forward(input_ids)

        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=-100
        )

        # Recipe diversity loss
        diversity_loss = 0.0
        if diversity_weight > 0:
            for layer in self.layers:
                recipe = layer.ffn.neuron_recipes
                recipe_norm = F.softmax(recipe, dim=-1)

                recipe_normalized = F.normalize(recipe_norm, dim=-1)
                similarity = torch.mm(recipe_normalized, recipe_normalized.T)

                mask = 1 - torch.eye(self.n_neurons, device=similarity.device)
                avg_similarity = (similarity * mask).sum() / mask.sum()

                diversity_loss += avg_similarity

            diversity_loss = diversity_loss / len(self.layers)

        # Load balance loss
        lb_loss = 0.0
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

    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable,
            'basis': sum(p.numel() for p in self.shared_basis.parameters()),
            'recipes': sum(
                layer.ffn.neuron_recipes.numel()
                for layer in self.layers
            ),
        }


# Backward compatibility
DAWNV74 = DAWN
DAWNLanguageModel = DAWN
DAWNLayer = DAWNLayerV74


def create_model(config):
    """Create DAWN model from config"""
    return DAWN(**config)


def create_model_v74(vocab_size, **kwargs):
    """
    DAWN v7.4 생성
    """
    default_config = {
        'd_model': 256,
        'd_ff': 1024,
        'n_layers': 4,
        'n_heads': 4,
        'n_basis': 32,
        'basis_rank': 64,
        'n_neurons': 64,
        'neuron_k': 8,
        'max_seq_len': 512,
        'dropout': 0.1
    }
    default_config.update(kwargs)

    model = DAWN(vocab_size, **default_config)

    print(f"\n{'='*70}")
    print(f"DAWN v7.4 - TT Weighted Karcher Mean")
    print(f"{'='*70}")

    params = model.get_num_params()
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Basis parameters: {params['basis']:,}")
    print(f"Recipe parameters: {params['recipes']:,}")
    print(f"{'='*70}\n")

    return model
