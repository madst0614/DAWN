"""
DAWN v6.0 - Dynamic Architecture With Neurons

핵심 철학:
- Basis: 직교 표현 공간 (8개, rank=64)
- Neurons: 자유로운 좌표 (512개, 8차원)
- Router: 동적 neuron 선택
- FFN: Token별 동적 구성

변경점 (v5.1 → v6.0):
1. Basis 수 감소 (16 → 8), Rank 증가 (32 → 64)
2. Basis 직교성 강화 (Soft regularization, λ=0.1)
3. Neuron diversity loss 제거 (자연스러운 clustering 허용)
4. 문장 레벨 FFN → Token 레벨 FFN
5. Token residual MLP 제거 (불필요)
6. 구조 단순화: Basis → Neuron → FFN (3단계 → 2단계)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 1. Neuron Router (v6.0)
# ============================================
class NeuronRouter(nn.Module):
    """Context-aware neuron selection

    v6.0: 단순화된 routing
    - Cross-attention으로 context 수집
    - Token + Context 기반 neuron 선택
    """

    def __init__(self, n_neurons=256, d_model=256, n_heads=4, k=16):
        super().__init__()
        self.n_neurons = n_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.k = k

        # Neuron embeddings (routing용)
        self.neuron_emb = nn.Parameter(
            torch.randn(n_neurons, d_model) * 0.02
        )

        # Cross-token attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Dynamic path mixing
        self.path_gate = nn.Linear(d_model * 2, 2)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, S, D] input tokens
            mask: [B, 1, S, S] causal mask

        Returns:
            selected_idx: [B, S, k] 선택된 neuron indices
            weights: [B, S, k] neuron weights
            context: [B, S, D] context vectors
        """
        B, S, D = x.shape

        # 1. Cross-token attention (context 수집)
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, S, D)

        # 2. Neuron scoring
        token_scores = torch.matmul(x, self.neuron_emb.T)  # [B, S, n_neurons]
        context_scores = torch.matmul(context, self.neuron_emb.T)

        # 3. Dynamic mixing
        gate_input = torch.cat([x, context], dim=-1)
        gate = F.softmax(self.path_gate(gate_input), dim=-1)  # [B, S, 2]

        scores = gate[:, :, 0:1] * token_scores + gate[:, :, 1:2] * context_scores

        # 4. Top-k selection
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        return topk_idx, weights, context


# ============================================
# 2. Orthogonal Basis FFN (v6.0 핵심!)
# ============================================
class OrthogonalBasisFFN(nn.Module):
    """FFN with orthogonal basis decomposition

    v6.0 핵심 변경:
    - 8개 basis (역할 명확)
    - Rank 64 (충분한 표현력)
    - Soft orthogonality (λ=0.1)
    - Neuron = basis 공간의 자유로운 점
    - Token별 FFN 구성 (문장 레벨 X)
    """

    def __init__(self, n_neurons=256, d_model=256, d_ff=1024,
                 n_basis=8, basis_rank=64):
        super().__init__()

        self.n_basis = n_basis
        self.n_neurons = n_neurons
        self.d_model = d_model
        self.d_ff = d_ff
        self.basis_rank = basis_rank

        # ===== Orthogonal Basis =====
        # 8개의 강력한 basis (각 rank=64)
        self.basis_A = nn.Parameter(
            torch.randn(n_basis, d_model, basis_rank) * 0.05
        )
        self.basis_B = nn.Parameter(
            torch.randn(n_basis, basis_rank, d_ff) * 0.05
        )

        # ===== Free Neurons =====
        # 각 neuron = 8차원 공간의 점 (자유로움!)
        # 비슷해도 OK (의미적 clustering)
        self.neuron_coords = nn.Parameter(
            torch.randn(n_neurons, n_basis) * 0.5
        )

        # Down projection
        self.W_down = nn.Linear(d_ff, d_model)

    @property
    def neuron_coef_A(self):
        """Backward compatibility: v5.x had separate coef_A and coef_B"""
        return self.neuron_coords

    @property
    def neuron_coef_B(self):
        """Backward compatibility: v5.x had separate coef_A and coef_B"""
        return self.neuron_coords

    def orthogonality_loss(self):
        """Basis 직교성 유지 (Soft regularization)

        각 basis의 A 행렬들이 서로 직교하도록
        """
        # Flatten basis_A: [n_basis, d_model * rank]
        basis_flat = self.basis_A.view(self.n_basis, -1)

        # Normalize
        basis_norm = F.normalize(basis_flat, p=2, dim=1)

        # Gram matrix
        gram = torch.mm(basis_norm, basis_norm.T)

        # Target: identity
        identity = torch.eye(self.n_basis, device=gram.device)

        # MSE loss
        loss = ((gram - identity) ** 2).sum()
        loss = loss / (self.n_basis * (self.n_basis - 1))

        return loss

    def forward(self, x, neuron_idx, neuron_weights, return_loss=False):
        """
        Args:
            x: [B, S, D] input tokens
            neuron_idx: [B, S, k] selected neuron indices
            neuron_weights: [B, S, k] neuron weights

        Returns:
            output: [B, S, D] FFN output
        """
        B, S, D = x.shape
        k = neuron_idx.shape[-1]

        # 1. 선택된 neuron들의 좌표 가져오기
        coords = self.neuron_coords[neuron_idx]  # [B, S, k, n_basis]

        # 2. Weighted combination → token별 좌표
        token_coords = (neuron_weights.unsqueeze(-1) * coords).sum(dim=2)
        # [B, S, n_basis]

        # 3. Token별 FFN 구성
        # W_up = Σ(coord_i × basis_A_i × basis_B_i)

        # 방법: einsum으로 효율적 계산
        # token_coords: [B, S, n_basis]
        # basis_A: [n_basis, d_model, rank]
        # basis_B: [n_basis, rank, d_ff]

        # Step 3a: Weighted basis_A
        # [B, S, n_basis] @ [n_basis, d_model, rank] → [B, S, d_model, rank]
        weighted_A = torch.einsum('bsn,ndr->bsdr', token_coords, self.basis_A)

        # Step 3b: Weighted basis_B
        # [B, S, n_basis] @ [n_basis, rank, d_ff] → [B, S, rank, d_ff]
        weighted_B = torch.einsum('bsn,nrf->bsrf', token_coords, self.basis_B)

        # Step 3c: Compose W_up = A @ B
        # [B, S, d_model, rank] @ [B, S, rank, d_ff] → [B, S, d_model, d_ff]
        # 하지만 이건 메모리 폭발! 대신:

        # 4. 직접 FFN 적용 (메모리 효율적)
        # h = x @ A @ B
        # Step 4a: x @ weighted_A
        # [B, S, D] → [B, S, 1, D] @ [B, S, D, rank] → [B, S, 1, rank] → [B, S, rank]
        h = torch.einsum('bsd,bsdr->bsr', x, weighted_A)

        # Step 4b: h @ weighted_B
        # [B, S, rank] @ [B, S, rank, d_ff] → [B, S, d_ff]
        h = torch.einsum('bsr,bsrf->bsf', h, weighted_B)

        # 5. Activation
        h = F.gelu(h)

        # 6. Down projection
        output = self.W_down(h)

        if return_loss:
            orth_loss = self.orthogonality_loss()
            return output, orth_loss

        return output


# ============================================
# 3. DAWN Layer (v6.0)
# ============================================
class DAWNLayer(nn.Module):
    """Single DAWN layer

    v6.0: 단순화된 구조
    - Router → FFN (직접 연결)
    - Token residual MLP 제거
    """

    def __init__(self, d_model=256, d_ff=1024, n_heads=4,
                 n_neurons=256, neuron_k=16,
                 n_basis=8, basis_rank=64):
        super().__init__()

        # Neuron Router
        self.router = NeuronRouter(
            n_neurons=n_neurons,
            d_model=d_model,
            n_heads=n_heads,
            k=neuron_k
        )

        # Orthogonal Basis FFN
        self.ffn = OrthogonalBasisFFN(
            n_neurons=n_neurons,
            d_model=d_model,
            d_ff=d_ff,
            n_basis=n_basis,
            basis_rank=basis_rank
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    @property
    def basis_ffn(self):
        """Backward compatibility: v5.x called it basis_ffn"""
        return self.ffn

    @property
    def neuron_router(self):
        """Backward compatibility: v5.x called it neuron_router"""
        return self.router

    def forward(self, x, mask=None, return_details=False, return_losses=False):
        """
        Args:
            x: [B, S, D]
            mask: causal mask
            return_details: return neuron indices
            return_losses: return orthogonality loss
        """
        # 1. Router (normalized input)
        normed = self.norm1(x)
        neuron_idx, neuron_weights, context = self.router(normed, mask)

        # 2. Neuron info aggregation (residual)
        # 선택된 neuron embeddings의 weighted sum
        neuron_embs = self.router.neuron_emb[neuron_idx]  # [B, S, k, D]
        neuron_info = (neuron_weights.unsqueeze(-1) * neuron_embs).sum(dim=2)
        x = x + neuron_info

        # 3. FFN
        normed = self.norm2(x)
        if return_losses:
            ffn_out, orth_loss = self.ffn(
                normed, neuron_idx, neuron_weights, return_loss=True
            )
        else:
            ffn_out = self.ffn(normed, neuron_idx, neuron_weights)
        x = x + ffn_out

        # Return
        if return_losses:
            if return_details:
                return x, neuron_idx, orth_loss
            return x, neuron_idx, orth_loss

        if return_details:
            return x, neuron_idx
        return x, neuron_idx


# ============================================
# 4. DAWN Model (v6.0)
# ============================================
class DAWN(nn.Module):
    """Dynamic Architecture With Neurons v6.0

    핵심 설계:
    - 8 Orthogonal Basis (명확한 역할)
    - 64 Rank (충분한 표현력)
    - 512 Free Neurons (자연스러운 clustering)
    - Token-level Dynamic FFN

    개선점:
    - 단순한 구조 (Basis → Neuron → FFN)
    - 명확한 Gradient flow
    - 해석 가능한 Basis
    - 자연스러운 Neuron 유사성
    """

    __version__ = "6.0"

    def __init__(self, vocab_size, d_model=256, d_ff=1024,
                 n_layers=4, n_heads=4,
                 n_neurons=256, neuron_k=16,
                 n_basis=8, basis_rank=64,
                 max_seq_len=512, dropout=0.1,
                 # Backward compatibility
                 hidden_dim=None, num_layers=None, k=None,
                 neuron_rank=None, mod_rank=None,
                 **kwargs):  # Ignore old params
        super().__init__()

        # Backward compatibility
        if hidden_dim is not None:
            d_model = hidden_dim
        if num_layers is not None:
            n_layers = num_layers
        if k is not None:
            neuron_k = k

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # DAWN Layers
        self.layers = nn.ModuleList([
            DAWNLayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                n_neurons=n_neurons,
                neuron_k=neuron_k,
                n_basis=n_basis,
                basis_rank=basis_rank
            )
            for _ in range(n_layers)
        ])

        # Output
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
            'n_neurons': n_neurons,
            'neuron_k': neuron_k,
            'n_basis': n_basis,
            'basis_rank': basis_rank,
            'max_seq_len': max_seq_len,
            'dropout': dropout,
        }

        # For compatibility
        self.hidden_dim = d_model
        self.n_neurons = n_neurons

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
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

    def forward(self, input_ids, return_activations=False, return_losses=False):
        """
        Args:
            input_ids: [B, S] token indices
            return_activations: return neuron indices per layer
            return_losses: return orthogonality losses

        Returns:
            logits: [B, S, vocab_size]
            (optional) neuron_indices: list of [B, S, k] per layer
            (optional) losses: dict of loss components
        """
        B, S = input_ids.shape

        # Embeddings
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        # Causal mask
        mask = torch.tril(torch.ones(S, S, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        # Process layers
        all_neuron_idx = []
        orth_losses = []

        for layer in self.layers:
            if return_losses:
                x, neuron_idx, orth_loss = layer(
                    x, mask, return_details=True, return_losses=True
                )
                orth_losses.append(orth_loss)
            else:
                x, neuron_idx = layer(x, mask, return_details=return_activations)

            if return_activations:
                all_neuron_idx.append(neuron_idx)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        # Return
        if return_losses:
            losses = {
                'orth': orth_losses,
                'orth_total': sum(orth_losses) / len(orth_losses)
            }
            if return_activations:
                return logits, all_neuron_idx, losses
            return logits, losses

        if return_activations:
            return logits, all_neuron_idx
        return logits

    def get_loss(self, input_ids, labels, orth_weight=0.1):
        """Compute total loss with orthogonality regularization

        Args:
            input_ids: [B, S]
            labels: [B, S]
            orth_weight: weight for orthogonality loss (default 0.1)

        Returns:
            total_loss, loss_dict
        """
        logits, losses = self.forward(input_ids, return_losses=True)

        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=-100
        )

        # Total loss
        total_loss = ce_loss + orth_weight * losses['orth_total']

        loss_dict = {
            'ce': ce_loss.item(),
            'orth': losses['orth_total'].item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict

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
        """Analyze basis usage across layers"""
        results = {}

        for i, layer in enumerate(self.layers):
            ffn = layer.ffn

            # Neuron coordinates statistics
            coords = ffn.neuron_coords  # [n_neurons, n_basis]

            # Mean activation per basis
            basis_importance = coords.abs().mean(dim=0)

            # Active neurons per basis (threshold=0.1)
            active_per_basis = (coords.abs() > 0.1).sum(dim=0).float()

            # Orthogonality measure
            basis_flat = ffn.basis_A.view(ffn.n_basis, -1)
            basis_norm = F.normalize(basis_flat, p=2, dim=1)
            gram = torch.mm(basis_norm, basis_norm.T)
            off_diag = gram - torch.eye(ffn.n_basis, device=gram.device)
            orthogonality = 1 - off_diag.abs().mean()

            results[f'layer_{i}'] = {
                'basis_importance': basis_importance.detach().cpu().numpy(),
                'active_per_basis': active_per_basis.detach().cpu().numpy(),
                'orthogonality': orthogonality.item(),
                'neuron_coord_mean': coords.mean().item(),
                'neuron_coord_std': coords.std().item(),
            }

        return results


# ============================================
# 5. Helper Functions
# ============================================
def create_model(config):
    """Create DAWN model from config"""
    return DAWN(**config)


def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Breakdown
    breakdown = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                breakdown[name] = params

    return total, breakdown


# Backward compatibility
DAWNLanguageModel = DAWN
Layer = DAWNLayer
