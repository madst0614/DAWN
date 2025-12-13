"""
DAWN v17: Hierarchical Neuron Circuits

2-level routing:
  Level 1: Circuit 선택 (top-k from n_circuits)
  Level 2: Circuit 내 Neuron weighting (softmax)

핵심: Circuit 내 neuron들이 협력 학습
     - Level 2 softmax로 역할 분화
     - gradient가 circuit 내 전체 neuron으로 흐름
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class SharedCircuits(nn.Module):
    """
    Circuit = [neurons_per_circuit, d_model]
    협력하는 뉴런들의 묶음

    2-level selection:
    - Circuit embedding: circuit 선택용
    - Neuron embedding: circuit 내 neuron 선택용
    """
    def __init__(
        self,
        d_model: int,
        neurons_per_circuit: int,
        n_circuits_r: int,
        n_circuits_v: int,
        n_circuits_rel: int,
        n_circuits_val: int,
        n_knowledge: int,
        knowledge_rank: int = 128,
        d_space: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.neurons_per_circuit = neurons_per_circuit
        self.d_space = d_space

        # Feature circuits: [n_circuits, neurons_per_circuit, d_model]
        # These are the actual weight matrices for computation
        self.feature_r_circuits = nn.Parameter(
            torch.zeros(n_circuits_r, neurons_per_circuit, d_model)
        )
        self.feature_v_circuits = nn.Parameter(
            torch.zeros(n_circuits_v, neurons_per_circuit, d_model)
        )

        # Expansion circuits
        self.relational_circuits = nn.Parameter(
            torch.zeros(n_circuits_rel, neurons_per_circuit, d_model)
        )
        self.value_circuits = nn.Parameter(
            torch.zeros(n_circuits_val, neurons_per_circuit, d_model)
        )

        # Neuron embeddings: [n_circuits, neurons_per_circuit, d_space]
        # For selecting neurons within each circuit
        self.feature_r_neuron_emb = nn.Parameter(
            torch.randn(n_circuits_r, neurons_per_circuit, d_space) * 0.02
        )
        self.feature_v_neuron_emb = nn.Parameter(
            torch.randn(n_circuits_v, neurons_per_circuit, d_space) * 0.02
        )
        self.relational_neuron_emb = nn.Parameter(
            torch.randn(n_circuits_rel, neurons_per_circuit, d_space) * 0.02
        )
        self.value_neuron_emb = nn.Parameter(
            torch.randn(n_circuits_val, neurons_per_circuit, d_space) * 0.02
        )

        # Knowledge neurons (same as v16)
        self.knowledge_neurons_K = nn.Parameter(torch.zeros(n_knowledge, knowledge_rank))
        self.knowledge_neurons_V = nn.Parameter(torch.zeros(n_knowledge, d_model))

        self._init_parameters()

    def _init_parameters(self):
        # Orthogonal init within each circuit
        for circuits in [self.feature_r_circuits, self.feature_v_circuits,
                         self.relational_circuits, self.value_circuits]:
            for i in range(circuits.shape[0]):
                nn.init.orthogonal_(circuits.data[i])

        nn.init.normal_(self.knowledge_neurons_K, std=0.02)
        nn.init.normal_(self.knowledge_neurons_V, std=0.02)


class CircuitRouter(nn.Module):
    """
    2-level Circuit Router

    Level 1: Circuit 선택 (top-k) - circuit_emb 사용
    Level 2: Circuit 내 Neuron weighting - neuron_emb 사용
    """
    def __init__(
        self,
        d_model: int,
        n_circuits_r: int,
        n_circuits_v: int,
        n_circuits_rel: int,
        n_circuits_val: int,
        n_knowledge: int,
        neurons_per_circuit: int = 64,
        d_space: int = 64,
        dropout: float = 0.1,
        excitability_tau: float = 1.5,
        excitability_ema_alpha: float = 0.01,
        langevin_alpha: float = 0.0003,
        langevin_beta: float = 0.0006,
    ):
        super().__init__()
        self.neurons_per_circuit = neurons_per_circuit
        self.d_space = d_space
        self.tau = excitability_tau
        self.ema_alpha = excitability_ema_alpha
        self.langevin_alpha = langevin_alpha
        self.langevin_beta = langevin_beta

        # Circuit counts
        self.n_circuits_r = n_circuits_r
        self.n_circuits_v = n_circuits_v
        self.n_circuits_rel = n_circuits_rel
        self.n_circuits_val = n_circuits_val
        self.n_knowledge = n_knowledge

        total = n_circuits_r + n_circuits_v + n_circuits_rel + n_circuits_val + n_knowledge

        # Level 1: Circuit selection projection
        self.proj = nn.Linear(d_model, d_space, bias=False)
        # Separate projections for Q/K
        self.proj_rel_Q = nn.Linear(d_model, d_space, bias=False)
        self.proj_rel_K = nn.Linear(d_model, d_space, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.circuit_emb = nn.Parameter(torch.randn(total, d_space) * 0.02)

        # Index boundaries
        self.idx_r = (0, n_circuits_r)
        self.idx_v = (n_circuits_r, n_circuits_r + n_circuits_v)
        self.idx_rel = (n_circuits_r + n_circuits_v,
                        n_circuits_r + n_circuits_v + n_circuits_rel)
        self.idx_val = (n_circuits_r + n_circuits_v + n_circuits_rel,
                        n_circuits_r + n_circuits_v + n_circuits_rel + n_circuits_val)
        self.idx_k = (n_circuits_r + n_circuits_v + n_circuits_rel + n_circuits_val, total)

        # Level 2: Neuron selection projection (shared, applied to selected circuit's neuron_emb)
        # Q/K용 별도 projection
        self.neuron_proj = nn.Linear(d_model, d_space, bias=False)
        self.neuron_proj_rel_Q = nn.Linear(d_model, d_space, bias=False)
        self.neuron_proj_rel_K = nn.Linear(d_model, d_space, bias=False)

        # Usage tracking (per circuit)
        self.register_buffer('usage_r', torch.zeros(n_circuits_r))
        self.register_buffer('usage_v', torch.zeros(n_circuits_v))
        self.register_buffer('usage_rel', torch.zeros(n_circuits_rel))
        self.register_buffer('usage_val', torch.zeros(n_circuits_val))
        self.register_buffer('usage_k', torch.zeros(n_knowledge))

        # Excitability weights
        self.register_buffer('exc_r', torch.ones(n_circuits_r) * 0.3)
        self.register_buffer('exc_v', torch.ones(n_circuits_v) * 0.3)
        self.register_buffer('exc_rel', torch.ones(n_circuits_rel) * 0.3)
        self.register_buffer('exc_val', torch.ones(n_circuits_val) * 0.3)
        self.register_buffer('exc_k', torch.ones(n_knowledge) * 0.3)

    def get_circuit_logits(self, x, circuit_type):
        """Level 1: Get circuit selection logits"""
        # x: [B, S, D] or [B, D]

        # Use different projection for Q vs K
        if circuit_type == 'relational_Q':
            h = self.dropout(self.proj_rel_Q(x))
        elif circuit_type == 'relational_K':
            h = self.dropout(self.proj_rel_K(x))
        else:
            h = self.dropout(self.proj(x))

        emb_norm = F.normalize(self.circuit_emb, dim=-1)

        if h.dim() == 2:
            logits = h @ emb_norm.T  # [B, total]
        else:
            logits = torch.einsum('bsd,nd->bsn', h, emb_norm)  # [B, S, total]

        # Slice for circuit type
        if circuit_type == 'feature_r':
            idx = self.idx_r
            usage, exc = self.usage_r, self.exc_r
        elif circuit_type == 'feature_v':
            idx = self.idx_v
            usage, exc = self.usage_v, self.exc_v
        elif circuit_type in ['relational', 'relational_Q', 'relational_K']:
            idx = self.idx_rel
            usage, exc = self.usage_rel, self.exc_rel
        elif circuit_type == 'value':
            idx = self.idx_val
            usage, exc = self.usage_val, self.exc_val
        elif circuit_type == 'knowledge':
            idx = self.idx_k
            usage, exc = self.usage_k, self.exc_k
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")

        logits = logits[..., idx[0]:idx[1]]

        # Add excitability during training
        if self.training:
            excitability = torch.clamp(1.0 - usage / self.tau, 0.0, 1.0)
            logits = logits + excitability * exc

        return logits

    def get_inner_weights(self, x_agg, selected_idx, neuron_emb, circuit_type):
        """
        Level 2: Get neuron weights using neuron embeddings

        Args:
            x_agg: [B, D] - importance-weighted aggregated input
            selected_idx: [B, k] - selected circuit indices
            neuron_emb: [n_circuits, neurons, d_space] - neuron embeddings
            circuit_type: for selecting projection

        Returns:
            [B, k, neurons] - soft weights for neurons in each selected circuit
        """
        B, k = selected_idx.shape

        # Project x to neuron selection space
        if circuit_type == 'relational_Q':
            h = self.neuron_proj_rel_Q(x_agg)  # [B, d_space]
        elif circuit_type == 'relational_K':
            h = self.neuron_proj_rel_K(x_agg)
        else:
            h = self.neuron_proj(x_agg)

        # Gather selected circuits' neuron embeddings
        # neuron_emb: [n_circuits, neurons, d_space]
        # selected_idx: [B, k] → [B, k, neurons, d_space]
        idx_expanded = selected_idx.view(B, k, 1, 1).expand(
            B, k, self.neurons_per_circuit, self.d_space
        )
        selected_neuron_emb = neuron_emb.unsqueeze(0).expand(B, -1, -1, -1).gather(1, idx_expanded)
        # selected_neuron_emb: [B, k, neurons, d_space]

        # Normalize embeddings
        selected_neuron_emb = F.normalize(selected_neuron_emb, dim=-1)

        # Compute similarity: h [B, d_space] vs neuron_emb [B, k, neurons, d_space]
        logits = torch.einsum('bd,bknd->bkn', h, selected_neuron_emb)  # [B, k, neurons]

        # Softmax within each circuit
        weights = F.softmax(logits, dim=-1)  # [B, k, neurons]

        return weights

    def update_usage(self, weights, circuit_type, mask=None):
        """Update circuit usage EMA"""
        if not self.training:
            return

        # weights: [B, n_circuits] after top-k
        usage = (weights > 0).float().mean(dim=0)

        decay = 1 - self.ema_alpha
        if circuit_type == 'feature_r':
            self.usage_r.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif circuit_type == 'feature_v':
            self.usage_v.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif circuit_type == 'relational':
            self.usage_rel.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif circuit_type == 'value':
            self.usage_val.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif circuit_type == 'knowledge':
            self.usage_k.mul_(decay).add_(usage, alpha=self.ema_alpha)

    def update_excitability(self):
        """Langevin dynamics for excitability"""
        threshold = 0.01
        for usage, exc in [(self.usage_r, self.exc_r), (self.usage_v, self.exc_v),
                           (self.usage_rel, self.exc_rel), (self.usage_val, self.exc_val),
                           (self.usage_k, self.exc_k)]:
            dead = (usage < threshold).float()
            dw = -self.langevin_alpha * exc + self.langevin_beta * dead
            exc.add_(dw).clamp_(0.1, 0.5)


class GlobalSSM(nn.Module):
    """Selective SSM for importance scoring"""
    def __init__(self, d_model: int, state_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        self.A_log = nn.Parameter(torch.randn(d_model, state_dim) * 0.1)
        self.W_delta = nn.Linear(d_model, d_model, bias=False)
        self.W_B = nn.Linear(d_model, state_dim, bias=False)
        self.W_C = nn.Linear(d_model, state_dim, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.importance_proj = nn.Linear(d_model, d_model, bias=False)
        self.temperature = 0.5

    def forward(self, x, attention_mask=None):
        B, S, D = x.shape

        delta = F.softplus(self.W_delta(x))
        B_sel = self.W_B(x)
        C_sel = self.W_C(x)
        A = -torch.exp(self.A_log)

        if MAMBA_AVAILABLE:
            # Cast to input dtype for AMP compatibility
            x_t = x.transpose(1, 2).contiguous()
            delta_t = delta.transpose(1, 2).contiguous()
            A_t = A.to(x.dtype)  # A must match input dtype for mamba kernel
            B_t = B_sel.transpose(1, 2).contiguous()
            C_t = C_sel.transpose(1, 2).contiguous()

            y = selective_scan_fn(x_t, delta_t, A_t, B_t, C_t,
                                  D=None, z=None, delta_bias=None,
                                  delta_softplus=False, return_last_state=False)
            ssm_out = y.transpose(1, 2).contiguous()
        else:
            ssm_out = self._slow_ssm(x, delta, A, B_sel, C_sel)

        ssm_out = self.norm(ssm_out)

        # Importance from final state
        h_final = ssm_out[:, -1, :]
        h_proj = self.importance_proj(h_final)
        raw = torch.einsum('bsd,bd->bs', x, h_proj)

        if attention_mask is not None:
            raw = raw.masked_fill(attention_mask == 0, float('-inf'))

        importance = F.softmax(raw / self.temperature, dim=-1)
        return importance, raw

    def _slow_ssm(self, x, delta, A, B_sel, C_sel):
        B, S, D = x.shape
        h = torch.zeros(B, D, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(S):
            dt = delta[:, t, :, None]
            decay = torch.exp(dt * A[None, :, :])
            h = h * decay + (dt * x[:, t, :, None]) * B_sel[:, t, None, :]
            y = torch.einsum('bdn,bn->bd', h, C_sel[:, t, :])
            outputs.append(y)

        return torch.stack(outputs, dim=1)


class AttentionModule(nn.Module):
    """
    v17 Attention with 2-level circuit routing

    Key: 선택된 k개 circuit 각각에 대해 별도의 inner weight 적용
    """
    def __init__(self, d_model: int, n_heads: int, neurons_per_circuit: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.neurons_per_circuit = neurons_per_circuit

        self.expand_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, circuits: SharedCircuits, selection, attention_mask=None):
        """
        selection contains:
        - circuit_r_idx: [B, k_r] selected circuit indices
        - circuit_r_weights: [B, k_r] weights for selected circuits
        - inner_r: [B, k_r, neurons_per_circuit] per-circuit neuron weights
        """
        B, S, D = x.shape

        # === Feature R compression ===
        # 1. Gather selected circuits
        # 2. Apply per-circuit inner weights
        # 3. Weighted sum → W_r

        # circuit_r_idx: [B, k] → gather circuits
        # circuits.feature_r_circuits: [n_circuits, neurons, d_model]
        k_r = selection['circuit_r_idx'].shape[1]
        idx_r = selection['circuit_r_idx'].view(B, k_r, 1, 1).expand(B, k_r, self.neurons_per_circuit, D)
        selected_r = circuits.feature_r_circuits.unsqueeze(0).expand(B, -1, -1, -1).gather(1, idx_r)
        # selected_r: [B, k_r, neurons, d_model]

        # inner_r: [B, k_r, neurons] → [B, k_r, neurons, 1]
        inner_r = selection['inner_r'].unsqueeze(-1)
        # Apply inner weights
        weighted_r = selected_r * inner_r  # [B, k_r, neurons, d_model]

        # circuit weights: [B, k_r] → [B, k_r, 1, 1]
        circuit_w_r = selection['circuit_r_weights'].view(B, k_r, 1, 1)
        # Weighted sum over selected circuits
        W_r = (weighted_r * circuit_w_r).sum(dim=1)  # [B, neurons, d_model]

        # Compress
        h_r = torch.einsum('bsd,bnd->bsn', x, W_r)  # [B, S, neurons]

        # === Feature V compression ===
        k_v = selection['circuit_v_idx'].shape[1]
        idx_v = selection['circuit_v_idx'].view(B, k_v, 1, 1).expand(B, k_v, self.neurons_per_circuit, D)
        selected_v = circuits.feature_v_circuits.unsqueeze(0).expand(B, -1, -1, -1).gather(1, idx_v)
        inner_v = selection['inner_v'].unsqueeze(-1)
        weighted_v = selected_v * inner_v
        circuit_w_v = selection['circuit_v_weights'].view(B, k_v, 1, 1)
        W_v = (weighted_v * circuit_w_v).sum(dim=1)
        h_v = torch.einsum('bsd,bnd->bsn', x, W_v)

        # === Relational Q expansion ===
        k_rel = selection['circuit_rel_Q_idx'].shape[1]
        idx_rel_Q = selection['circuit_rel_Q_idx'].view(B, k_rel, 1, 1).expand(B, k_rel, self.neurons_per_circuit, D)
        selected_rel_Q = circuits.relational_circuits.unsqueeze(0).expand(B, -1, -1, -1).gather(1, idx_rel_Q)
        inner_rel_Q = selection['inner_rel_Q'].unsqueeze(-1)
        weighted_rel_Q = selected_rel_Q * inner_rel_Q
        circuit_w_rel_Q = selection['circuit_rel_Q_weights'].view(B, k_rel, 1, 1)
        W_rel_Q = (weighted_rel_Q * circuit_w_rel_Q).sum(dim=1)
        Q = torch.einsum('bsn,bnd->bsd', h_r, W_rel_Q)

        # === Relational K expansion ===
        idx_rel_K = selection['circuit_rel_K_idx'].view(B, k_rel, 1, 1).expand(B, k_rel, self.neurons_per_circuit, D)
        selected_rel_K = circuits.relational_circuits.unsqueeze(0).expand(B, -1, -1, -1).gather(1, idx_rel_K)
        inner_rel_K = selection['inner_rel_K'].unsqueeze(-1)
        weighted_rel_K = selected_rel_K * inner_rel_K
        circuit_w_rel_K = selection['circuit_rel_K_weights'].view(B, k_rel, 1, 1)
        W_rel_K = (weighted_rel_K * circuit_w_rel_K).sum(dim=1)
        K = torch.einsum('bsn,bnd->bsd', h_r, W_rel_K)

        # === Value expansion ===
        k_val = selection['circuit_val_idx'].shape[1]
        idx_val = selection['circuit_val_idx'].view(B, k_val, 1, 1).expand(B, k_val, self.neurons_per_circuit, D)
        selected_val = circuits.value_circuits.unsqueeze(0).expand(B, -1, -1, -1).gather(1, idx_val)
        inner_val = selection['inner_val'].unsqueeze(-1)
        weighted_val = selected_val * inner_val
        circuit_w_val = selection['circuit_val_weights'].view(B, k_val, 1, 1)
        W_val = (weighted_val * circuit_w_val).sum(dim=1)
        V = torch.einsum('bsn,bnd->bsd', h_v, W_val)

        # === Multi-head attention ===
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        if attention_mask is not None:
            causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
            pad = (attention_mask == 0).view(B, 1, 1, S)
            mask = ~(causal.unsqueeze(0).unsqueeze(0) | pad)
            attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        else:
            attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.dropout(self.expand_O(attn_out))


class MemoryModule(nn.Module):
    """2-stage knowledge retrieval (same as v16)"""
    def __init__(self, d_model: int, n_knowledge: int, knowledge_rank: int = 128,
                 coarse_k: int = 16, fine_k: int = 8):
        super().__init__()
        self.n_knowledge = n_knowledge
        self.knowledge_rank = knowledge_rank
        self.coarse_k = coarse_k
        self.fine_k = fine_k

        self.query_proj = nn.Linear(d_model, knowledge_rank, bias=False)

    def forward(self, x, circuits: SharedCircuits, router: CircuitRouter, mask=None):
        B, S, D = x.shape

        # Coarse selection
        logits = router.get_circuit_logits(x, 'knowledge')  # [B, S, n_knowledge]
        _, coarse_idx = torch.topk(logits, self.coarse_k, dim=-1)  # [B, S, coarse_k]

        if self.training:
            router.update_usage((logits > logits.median()).float().mean(dim=1), 'knowledge')

        # Fine selection
        query = self.query_proj(x)  # [B, S, rank]

        K_all = circuits.knowledge_neurons_K  # [n_knowledge, rank]
        V_all = circuits.knowledge_neurons_V  # [n_knowledge, d_model]

        # Gather candidates
        idx_k = coarse_idx.unsqueeze(-1).expand(B, S, self.coarse_k, self.knowledge_rank)
        K_cand = K_all.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1).gather(2, idx_k)

        scores = torch.einsum('bsd,bscd->bsc', query, K_cand) / math.sqrt(self.knowledge_rank)
        top_scores, top_local = torch.topk(scores, self.fine_k, dim=-1)
        weights = F.softmax(top_scores, dim=-1)

        top_global = coarse_idx.gather(-1, top_local)
        idx_v = top_global.unsqueeze(-1).expand(B, S, self.fine_k, D)
        V_sel = V_all.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1).gather(2, idx_v)

        output = (V_sel * weights.unsqueeze(-1)).sum(dim=2)
        return output


class DAWNBlock(nn.Module):
    """Single DAWN v17 block"""
    def __init__(self, d_model: int, n_heads: int, neurons_per_circuit: int,
                 n_knowledge: int, knowledge_rank: int = 128,
                 coarse_k: int = 16, fine_k: int = 8, dropout: float = 0.1):
        super().__init__()

        self.attn = AttentionModule(d_model, n_heads, neurons_per_circuit, dropout)
        self.memory = MemoryModule(d_model, n_knowledge, knowledge_rank, coarse_k, fine_k)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, circuits, router, selection, attention_mask=None):
        # Attention
        h = self.norm1(x)
        x = x + self.attn(h, circuits, selection, attention_mask)

        # Memory
        h = self.norm2(x)
        x = x + self.dropout(self.memory(h, circuits, router, attention_mask))

        return x


class DAWN(nn.Module):
    """
    DAWN v17: Hierarchical Neuron Circuits

    2-level routing:
    - Level 1: Circuit selection (top-k)
    - Level 2: Neuron weighting within circuit (softmax, applied before circuit sum)
    """
    __version__ = "17.0"

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 384,
        n_layers: int = 12,
        n_heads: int = 6,
        max_seq_len: int = 512,
        # Circuit config
        neurons_per_circuit: int = 64,
        n_circuits_r: int = 96,
        n_circuits_v: int = 24,
        n_circuits_rel: int = 96,
        n_circuits_val: int = 16,
        top_k_circuits_r: int = 12,
        top_k_circuits_v: int = 6,
        top_k_circuits_rel: int = 12,
        top_k_circuits_val: int = 4,
        # Knowledge
        n_knowledge: int = 300,
        knowledge_rank: int = 128,
        coarse_k: int = 16,
        fine_k: int = 8,
        # Other
        state_dim: int = 64,
        d_space: int = 64,
        dropout: float = 0.1,
        # Excitability
        excitability_tau: float = 1.5,
        excitability_ema_alpha: float = 0.01,
        langevin_alpha: float = 0.0003,
        langevin_beta: float = 0.0006,
        **kwargs
    ):
        super().__init__()

        # Save config
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.neurons_per_circuit = neurons_per_circuit
        self.n_circuits_r = n_circuits_r
        self.n_circuits_v = n_circuits_v
        self.n_circuits_rel = n_circuits_rel
        self.n_circuits_val = n_circuits_val
        self.top_k_r = top_k_circuits_r
        self.top_k_v = top_k_circuits_v
        self.top_k_rel = top_k_circuits_rel
        self.top_k_val = top_k_circuits_val
        self.n_knowledge = n_knowledge
        self.knowledge_rank = knowledge_rank

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Shared circuits (with d_space for neuron embeddings)
        self.circuits = SharedCircuits(
            d_model, neurons_per_circuit,
            n_circuits_r, n_circuits_v, n_circuits_rel, n_circuits_val,
            n_knowledge, knowledge_rank, d_space
        )

        # Router
        self.router = CircuitRouter(
            d_model, n_circuits_r, n_circuits_v, n_circuits_rel, n_circuits_val, n_knowledge,
            neurons_per_circuit, d_space, dropout,
            excitability_tau, excitability_ema_alpha, langevin_alpha, langevin_beta
        )

        # SSM for importance
        self.ssm = GlobalSSM(d_model, state_dim)

        # Layers
        self.layers = nn.ModuleList([
            DAWNBlock(d_model, n_heads, neurons_per_circuit,
                      n_knowledge, knowledge_rank, coarse_k, fine_k, dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _get_selection(self, x, importance):
        """Get circuit indices, weights, and per-circuit inner weights"""
        B = x.shape[0]

        def topk_select(logits, k):
            # logits: [B, S, n_circuits]
            # importance-weighted aggregation to batch level
            agg = torch.einsum('bs,bsn->bn', importance, F.softmax(logits, dim=-1))
            vals, idx = torch.topk(agg, k, dim=-1)  # [B, k]
            # Normalize weights
            weights = vals / (vals.sum(dim=-1, keepdim=True) + 1e-8)
            return idx, weights

        # Level 1: Circuit selection (returns idx and weights)
        circuit_r_idx, circuit_r_w = topk_select(self.router.get_circuit_logits(x, 'feature_r'), self.top_k_r)
        circuit_v_idx, circuit_v_w = topk_select(self.router.get_circuit_logits(x, 'feature_v'), self.top_k_v)
        circuit_rel_Q_idx, circuit_rel_Q_w = topk_select(self.router.get_circuit_logits(x, 'relational_Q'), self.top_k_rel)
        circuit_rel_K_idx, circuit_rel_K_w = topk_select(self.router.get_circuit_logits(x, 'relational_K'), self.top_k_rel)
        circuit_val_idx, circuit_val_w = topk_select(self.router.get_circuit_logits(x, 'value'), self.top_k_val)

        # Aggregate x for neuron selection
        x_agg = torch.einsum('bs,bsd->bd', importance, x)  # [B, D]

        # Level 2: Inner neuron weights using neuron embeddings [B, k, neurons]
        inner_r = self.router.get_inner_weights(
            x_agg, circuit_r_idx, self.circuits.feature_r_neuron_emb, 'feature_r')
        inner_v = self.router.get_inner_weights(
            x_agg, circuit_v_idx, self.circuits.feature_v_neuron_emb, 'feature_v')
        inner_rel_Q = self.router.get_inner_weights(
            x_agg, circuit_rel_Q_idx, self.circuits.relational_neuron_emb, 'relational_Q')
        inner_rel_K = self.router.get_inner_weights(
            x_agg, circuit_rel_K_idx, self.circuits.relational_neuron_emb, 'relational_K')
        inner_val = self.router.get_inner_weights(
            x_agg, circuit_val_idx, self.circuits.value_neuron_emb, 'value')

        # Update usage (create full sparse tensor for tracking)
        if self.training:
            def to_sparse(idx, n):
                sparse = torch.zeros(B, n, device=x.device)
                sparse.scatter_(1, idx, 1.0)
                return sparse
            self.router.update_usage(to_sparse(circuit_r_idx, self.n_circuits_r), 'feature_r')
            self.router.update_usage(to_sparse(circuit_v_idx, self.n_circuits_v), 'feature_v')
            rel_sparse = torch.maximum(
                to_sparse(circuit_rel_Q_idx, self.n_circuits_rel),
                to_sparse(circuit_rel_K_idx, self.n_circuits_rel)
            )
            self.router.update_usage(rel_sparse, 'relational')
            self.router.update_usage(to_sparse(circuit_val_idx, self.n_circuits_val), 'value')

        return {
            'circuit_r_idx': circuit_r_idx, 'circuit_r_weights': circuit_r_w,
            'circuit_v_idx': circuit_v_idx, 'circuit_v_weights': circuit_v_w,
            'circuit_rel_Q_idx': circuit_rel_Q_idx, 'circuit_rel_Q_weights': circuit_rel_Q_w,
            'circuit_rel_K_idx': circuit_rel_K_idx, 'circuit_rel_K_weights': circuit_rel_K_w,
            'circuit_val_idx': circuit_val_idx, 'circuit_val_weights': circuit_val_w,
            'inner_r': inner_r, 'inner_v': inner_v,
            'inner_rel_Q': inner_rel_Q, 'inner_rel_K': inner_rel_K,
            'inner_val': inner_val,
        }

    def forward(self, input_ids, labels=None, attention_mask=None, return_routing_info=False):
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos = torch.arange(S, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        routing_infos = []

        for layer in self.layers:
            # Get importance
            importance, _ = self.ssm(x, attention_mask)

            # Get selection
            selection = self._get_selection(x, importance)

            # Forward
            x = layer(x, self.circuits, self.router, selection, attention_mask)

            if return_routing_info:
                routing_infos.append({
                    'importance': importance.detach(),
                    'circuit_r_idx': selection['circuit_r_idx'].detach(),
                    'circuit_r_weights': selection['circuit_r_weights'].detach(),
                    'inner_r': selection['inner_r'].detach(),
                })

        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size),
                                   shift_labels.view(-1), ignore_index=-100)
            if return_routing_info:
                return loss, logits, routing_infos
            return loss, logits

        if return_routing_info:
            return logits, routing_infos
        return logits

    def get_auxiliary_losses(self):
        """Diversity and orthogonality losses"""
        # Circuit diversity
        div_loss = 0.0
        for circuits in [self.circuits.feature_r_circuits, self.circuits.feature_v_circuits,
                         self.circuits.relational_circuits, self.circuits.value_circuits]:
            flat = circuits.view(circuits.shape[0], -1)
            norm = F.normalize(flat, dim=-1)
            sim = norm @ norm.T
            mask = ~torch.eye(circuits.shape[0], dtype=torch.bool, device=circuits.device)
            div_loss += sim[mask].abs().mean()

        # Neuron orthogonality within circuits
        orth_loss = 0.0
        count = 0
        for circuits in [self.circuits.feature_r_circuits, self.circuits.feature_v_circuits,
                         self.circuits.relational_circuits, self.circuits.value_circuits]:
            for i in range(circuits.shape[0]):
                neurons = circuits[i]  # [neurons_per_circuit, d_model]
                gram = neurons @ neurons.T
                I = torch.eye(neurons.shape[0], device=neurons.device)
                orth_loss += ((gram - I) ** 2).mean()
                count += 1

        return {
            'circuit_div': div_loss / 4,
            'neuron_orth': orth_loss / count,
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self):
        return {
            'version': self.__version__,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'neurons_per_circuit': self.neurons_per_circuit,
            'n_circuits_r': self.n_circuits_r,
            'n_circuits_v': self.n_circuits_v,
            'n_circuits_rel': self.n_circuits_rel,
            'n_circuits_val': self.n_circuits_val,
            'n_knowledge': self.n_knowledge,
        }
