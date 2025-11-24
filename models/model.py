import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 1. Low-rank Neuron Router (v5.0)
# ============================================
class NeuronRouter(nn.Module):
    """Context-aware neuron selection with low-rank neuron pool

    v5.0: Low-rank neuron embeddings (91% parameter reduction)
    - neurons = neuron_A @ neuron_B
    - Orthogonality loss on low-rank factors
    """

    def __init__(self, n_neurons=512, d_model=256, n_heads=4, k=16,
                 neuron_rank=16):
        super().__init__()
        self.n_neurons = n_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.k = k

        # Low-rank neuron pool
        self.neuron_A = nn.Parameter(
            torch.randn(n_neurons, neuron_rank) * 0.02
        )
        self.neuron_B = nn.Parameter(
            torch.randn(neuron_rank, d_model) * 0.02
        )

        # Cross-token attention for context collection
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Dynamic mixing weights (token-based vs context-based)
        self.path_proj = nn.Linear(d_model * 2, 2)

    @property
    def neurons(self):
        """Compose full neurons from low-rank factors"""
        return torch.matmul(self.neuron_A, self.neuron_B)
        # [n_neurons, rank] @ [rank, d_model] → [n_neurons, d_model]

    def forward(self, x, mask=None, return_loss=False):
        B, S, D = x.shape

        # Get full neurons
        neurons_full = self.neurons  # [n_neurons, d_model]

        # 1. Cross-token attention - collect context
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, S, D)

        # 2. Neuron scoring - Bottom-up (token) + Top-down (context)
        token_scores = torch.matmul(x, neurons_full.T)
        context_scores = torch.matmul(context, neurons_full.T)

        # 3. Dynamic mixing
        combined = torch.cat([x, context], dim=-1)
        weights = F.softmax(self.path_proj(combined), dim=-1)

        scores = weights[:, :, 0:1] * token_scores + \
                 weights[:, :, 1:2] * context_scores

        # 4. Top-k selection
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)

        selected = neurons_full[topk_idx]  # [B, S, k, d_model]

        if return_loss:
            ortho_loss = self.compute_orthogonality_loss()
            return selected, topk_idx, topk_weights, context, ortho_loss

        return selected, topk_idx, topk_weights, context

    def compute_orthogonality_loss(self):
        """Orthogonality on low-rank factors"""
        A_norm = F.normalize(self.neuron_A, p=2, dim=1)
        gram = torch.mm(A_norm, A_norm.T)
        identity = torch.eye(self.n_neurons, device=gram.device)
        ortho_loss = ((gram - identity) ** 2).sum()
        ortho_loss = ortho_loss / (self.n_neurons * (self.n_neurons - 1))
        return ortho_loss


# ============================================
# 2. Basis FFN (v5.1)
# ============================================
class BasisFFN(nn.Module):
    """FFN with hierarchical basis decomposition + token residual

    v5.1: Coarse-to-fine architecture
    - basis [n_basis] → neurons [n_neurons] → sentence FFN [1] (coarse)
    - token residual network (d_model → 256 → d_ff) (fine)
    - Combined: h = h_coarse + 0.1 * h_fine

    Benefits:
    - Coarse: Shared structure across tokens (efficient)
    - Fine: Token-specific adjustments (expressive)
    - Scaled residual: Stable training with fine-grained control
    """

    def __init__(self, n_neurons=512, d_model=256, d_ff=1024,
                 n_basis=16, basis_rank=32, mod_rank=None):
        super().__init__()

        # Backward compatibility: mod_rank ignored in v5.1
        if mod_rank is not None:
            pass  # v5.0 compatibility, parameter not used

        self.n_basis = n_basis
        self.n_neurons = n_neurons
        self.d_model = d_model
        self.d_ff = d_ff

        # FFN Basis: [n_basis, D, rank] @ [n_basis, rank, d_ff]
        # v5.0: Increased init scale for better gradient flow
        self.basis_A = nn.Parameter(
            torch.randn(n_basis, d_model, basis_rank) * 0.05  # 0.02 → 0.05
        )
        self.basis_B = nn.Parameter(
            torch.randn(n_basis, basis_rank, d_ff) * 0.05  # 0.02 → 0.05
        )

        # Neuron-to-basis composition weights
        # v5.0: Orthogonal initialization via QR decomposition
        # Ensures neurons start with diverse, independent basis combinations
        Q_A, _ = torch.linalg.qr(torch.randn(n_neurons, n_basis))
        Q_B, _ = torch.linalg.qr(torch.randn(n_neurons, n_basis))

        self.neuron_coef_A = nn.Parameter(Q_A)
        self.neuron_coef_B = nn.Parameter(Q_B)

        # v5.1: Token residual network (coarse + fine)
        # Provides fine-grained token-level adjustments on top of coarse sentence FFN
        self.token_residual = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, d_ff)
        )
        self.residual_scale = 0.1  # Scale factor for fine adjustment

        # Down projection
        self.down = nn.Linear(d_ff, d_model)

    def compute_sparsity_loss(self):
        """Encourage each neuron to use multiple basis blocks

        Prevents neurons from collapsing to use only 1-2 basis.
        Target: at least 3 basis per neuron.
        """
        # Count active connections per neuron (threshold = 0.1)
        threshold = 0.1
        active_A = (self.neuron_coef_A.abs() > threshold).sum(dim=1).float()
        active_B = (self.neuron_coef_B.abs() > threshold).sum(dim=1).float()

        # Encourage at least 3 basis per neuron
        target = 3.0
        loss_A = F.relu(target - active_A).mean()
        loss_B = F.relu(target - active_B).mean()

        return loss_A + loss_B

    def compute_diversity_loss(self):
        """Encourage neurons to maintain diverse basis combinations during training

        Penalizes high cosine similarity between neuron coefficient vectors.
        This acts as a backup to orthogonal initialization.
        """
        # Normalize coefficient vectors
        coef_A_norm = F.normalize(self.neuron_coef_A, p=2, dim=1)
        coef_B_norm = F.normalize(self.neuron_coef_B, p=2, dim=1)

        # Compute pairwise similarity matrices
        sim_A = torch.mm(coef_A_norm, coef_A_norm.T)
        sim_B = torch.mm(coef_B_norm, coef_B_norm.T)

        # Penalize off-diagonal similarities (exclude self-similarity)
        n = sim_A.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=sim_A.device)

        # MSE loss on squared similarities (stronger penalty for high similarity)
        loss_A = sim_A[mask].pow(2).mean()
        loss_B = sim_B[mask].pow(2).mean()

        return loss_A + loss_B

    def forward(self, x, selected_neurons, neuron_idx, neuron_weights, return_loss=False):
        """
        Args:
            x: [B, S, D] - input tokens
            selected_neurons: [B, S, k, D] - from router
            neuron_idx: [B, S, k] - selected neuron indices
            neuron_weights: [B, S, k] - weights for selected neurons
        """
        B, S = x.shape[:2]

        # =====================================================
        # STEP 1: Compose sentence-level FFN from basis
        # =====================================================
        W_up = self.compose_sentence_ffn(neuron_idx, neuron_weights)
        # [B, D, d_ff]

        # =====================================================
        # STEP 2: Coarse sentence-level FFN (shared structure)
        # =====================================================
        h_coarse = torch.bmm(x, W_up)  # [B, S, D] @ [B, D, d_ff] → [B, S, d_ff]
        h_coarse = F.gelu(h_coarse)

        # =====================================================
        # STEP 3: Fine token-level adjustment (v5.1)
        # =====================================================
        # Token signature from selected neurons
        token_sig = (selected_neurons * neuron_weights.unsqueeze(-1)).sum(dim=2)
        # [B, S, D]

        # Token residual network (full MLP for expressiveness)
        h_fine = self.token_residual(token_sig)
        # [B, S, d_ff]

        # Combine coarse + fine with scaled residual
        h = h_coarse + self.residual_scale * h_fine

        # =====================================================
        # STEP 4: Down projection
        # =====================================================
        output = self.down(h)

        if return_loss:
            sparsity_loss = self.compute_sparsity_loss()
            diversity_loss = self.compute_diversity_loss()
            return output, sparsity_loss, diversity_loss

        return output

    def compose_sentence_ffn(self, neuron_idx, neuron_weights):
        """Compose sentence-level FFN from selected neurons

        Args:
            neuron_idx: [B, S, k]
            neuron_weights: [B, S, k]

        Returns:
            W_up: [B, D, d_ff] - sentence-level FFN weights
        """
        B, S, k = neuron_idx.shape

        # Flatten to sentence level
        idx_flat = neuron_idx.view(B, -1)  # [B, S*k]
        weights_flat = neuron_weights.view(B, -1)  # [B, S*k]

        # Normalize weights across sentence
        weights_flat = weights_flat / (weights_flat.sum(dim=1, keepdim=True) + 1e-8)

        # Get neuron coefficients for selected neurons
        coef_A = self.neuron_coef_A[idx_flat]  # [B, S*k, n_basis]
        coef_B = self.neuron_coef_B[idx_flat]  # [B, S*k, n_basis]

        # Weighted average → sentence-level basis coefficients
        sent_coef_A = (weights_flat.unsqueeze(-1) * coef_A).sum(dim=1)
        # [B, n_basis]
        sent_coef_B = (weights_flat.unsqueeze(-1) * coef_B).sum(dim=1)
        # [B, n_basis]

        # Compose from basis
        # [B, n_basis] @ [n_basis, D, rank] → [B, D, rank]
        A = torch.einsum('bi,idr->bdr', sent_coef_A, self.basis_A)

        # [B, n_basis] @ [n_basis, rank, d_ff] → [B, rank, d_ff]
        B_mat = torch.einsum('bi,irf->brf', sent_coef_B, self.basis_B)

        # Final composition: [B, D, rank] @ [B, rank, d_ff] → [B, D, d_ff]
        W_up = torch.bmm(A, B_mat)

        return W_up


# ============================================
# 3. DAWN Layer (v5.0)
# ============================================
class Layer(nn.Module):
    """Single DAWN layer with neuron routing and basis FFN

    v5.1: Token residual network (coarse + fine)
    - Coarse: Sentence-level FFN (shared structure)
    - Fine: Token-level residual (individual adjustments)
    """

    def __init__(self, d_model=256, d_ff=1024, n_heads=4,
                 n_neurons=512, neuron_rank=16, neuron_k=16,
                 n_basis=16, basis_rank=32, mod_rank=None):
        super().__init__()

        # Backward compatibility: mod_rank ignored in v5.1
        if mod_rank is not None:
            pass  # v5.0 compatibility, parameter not used

        # Neuron router
        self.neuron_router = NeuronRouter(
            n_neurons=n_neurons,
            d_model=d_model,
            n_heads=n_heads,
            k=neuron_k,
            neuron_rank=neuron_rank
        )

        # Basis FFN (v5.1 with token residual)
        self.basis_ffn = BasisFFN(
            n_neurons=n_neurons,
            d_model=d_model,
            d_ff=d_ff,
            n_basis=n_basis,
            basis_rank=basis_rank,
            mod_rank=mod_rank  # Pass for compatibility
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_details=False, return_losses=False):
        # 1. Neuron routing
        normed = self.norm1(x)
        if return_losses:
            selected_neurons, topk_idx, topk_weights, context, ortho_loss = \
                self.neuron_router(normed, mask, return_loss=True)
        else:
            selected_neurons, topk_idx, topk_weights, context = \
                self.neuron_router(normed, mask)

        # 2. Aggregate neuron information (residual)
        neuron_info = (topk_weights.unsqueeze(-1) * selected_neurons).sum(dim=2)
        x = x + neuron_info  # [B, S, D]

        # 3. Basis FFN
        normed = self.norm2(x)
        if return_losses:
            ffn_out, sparsity_loss, diversity_loss = self.basis_ffn(
                normed, selected_neurons, topk_idx, topk_weights, return_loss=True
            )
        else:
            ffn_out = self.basis_ffn(normed, selected_neurons, topk_idx, topk_weights)
        x = x + ffn_out

        # Return
        if return_losses:
            if return_details:
                return x, topk_idx, ortho_loss, sparsity_loss, diversity_loss
            return x, topk_idx, ortho_loss, sparsity_loss, diversity_loss

        if return_details:
            return x, topk_idx
        return x, topk_idx


# ============================================
# 4. DAWN Model (v5.1)
# ============================================
class DAWN(nn.Module):
    """Dynamic Architecture With Neurons

    v5.1: Token Residual Network (Coarse + Fine)

    Key improvements over v5.0:
    - Token residual: h = h_coarse + 0.1 * h_fine
    - Coarse: Sentence-level FFN (shared structure, efficient)
    - Fine: Token residual network (individual adjustments, expressive)
    - Better balance between efficiency and expressiveness

    v5.0 improvements (retained):
    - Low-rank neuron embeddings (91% parameter reduction)
    - Hierarchical FFN basis decomposition (compositional)
    - Orthogonal initialization + diversity loss (prevents redundancy)
    """

    __version__ = "5.1"

    def __init__(self, vocab_size, d_model=256, d_ff=1024, n_layers=4, n_heads=4,
                 n_neurons=512, neuron_rank=16, neuron_k=16,
                 n_basis=16, basis_rank=32, mod_rank=None,
                 max_seq_len=512, dropout=0.1,
                 # Backward compatibility (v5.0 and earlier)
                 hidden_dim=None, num_layers=None, k=None,
                 n_patterns=None, pattern_k=None, rank=None,
                 pattern_dropout=None, use_base=None,
                 num_input_neurons=None, num_process_neurons=None):
        super().__init__()

        # Backward compatibility: mod_rank ignored in v5.1
        if mod_rank is not None:
            pass  # v5.0 compatibility, parameter not used

        # Backward compatibility
        if hidden_dim is not None:
            d_model = hidden_dim
        if num_layers is not None:
            n_layers = num_layers
        if k is not None:
            neuron_k = k
        if num_input_neurons is not None:
            n_neurons = num_input_neurons * 16

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        # Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # DAWN layers
        self.layers = nn.ModuleList([
            Layer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                n_neurons=n_neurons,
                neuron_rank=neuron_rank,
                neuron_k=neuron_k,
                n_basis=n_basis,
                basis_rank=basis_rank,
                mod_rank=mod_rank  # v5.0 compatibility (ignored)
            )
            for _ in range(n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # Weight tying

        # Store for compatibility
        self.hidden_dim = d_model
        self.n_neurons = n_neurons

        # Store config
        self.config = {
            'd_model': d_model,
            'd_ff': d_ff,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_neurons': n_neurons,
            'neuron_rank': neuron_rank,
            'neuron_k': neuron_k,
            'n_basis': n_basis,
            'basis_rank': basis_rank,
            'mod_rank': mod_rank,  # v5.0 compatibility (not used in v5.1)
            'vocab_size': vocab_size,
            'max_seq_len': max_seq_len,
            'dropout': dropout,
        }

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
        B, S = input_ids.shape

        # Embedding
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        # Causal mask
        mask = torch.tril(torch.ones(S, S, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Process through layers
        all_neuron_idx = []
        ortho_losses = []
        sparsity_losses = []
        diversity_losses = []

        for layer in self.layers:
            if return_losses:
                if return_activations:
                    x, neuron_idx, ortho_loss, sparsity_loss, diversity_loss = layer(
                        x, mask, return_details=True, return_losses=True
                    )
                    all_neuron_idx.append(neuron_idx)
                else:
                    x, neuron_idx, ortho_loss, sparsity_loss, diversity_loss = layer(
                        x, mask, return_details=False, return_losses=True
                    )
                ortho_losses.append(ortho_loss)
                sparsity_losses.append(sparsity_loss)
                diversity_losses.append(diversity_loss)
            elif return_activations:
                x, neuron_idx = layer(x, mask, return_details=True)
                all_neuron_idx.append(neuron_idx)
            else:
                x, neuron_idx = layer(x, mask)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        # Return
        if return_losses:
            losses = {
                'neuron_ortho': ortho_losses,
                'basis_sparsity': sparsity_losses,
                'basis_diversity': diversity_losses
            }
            if return_activations:
                return logits, all_neuron_idx, losses
            return logits, losses

        if return_activations:
            return logits, all_neuron_idx
        return logits

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
# 5. Helper functions
# ============================================
def create_model(config):
    """Create DAWN model from config dict"""
    return DAWN(**config)


# Backward compatibility
DAWNLanguageModel = DAWN
