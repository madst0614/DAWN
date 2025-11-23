import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 1. 문맥 기반 뉴런 라우터 (v4.5)
# ============================================
class NeuronRouter(nn.Module):
    """뉴런 선택만 담당 - aggregate 제거!"""

    def __init__(self, n_neurons=512, d_model=256, n_heads=4, k=16):
        super().__init__()
        self.n_neurons = n_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.k = k

        # Neuron pool
        self.neurons = nn.Parameter(torch.randn(n_neurons, d_model) * 0.02)

        # Cross-token attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Path mixing (token vs context)
        self.path_proj = nn.Linear(d_model * 2, 2)

    def forward(self, x, mask=None, return_loss=False):
        B, S, D = x.shape

        # 1. Cross-token attention (문맥 수집)
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, S, D)

        # 2. Bottom-up: 토큰 기반 뉴런 점수
        token_scores = torch.matmul(x, self.neurons.T)  # [B, S, n_neurons]

        # 3. Top-down: 문맥 기반 뉴런 점수
        context_scores = torch.matmul(context, self.neurons.T)  # [B, S, n_neurons]

        # 4. Dynamic mixing
        combined = torch.cat([x, context], dim=-1)  # [B, S, 2*D]
        weights = F.softmax(self.path_proj(combined), dim=-1)  # [B, S, 2]

        scores = weights[:, :, 0:1] * token_scores + \
                 weights[:, :, 1:2] * context_scores  # [B, S, n_neurons]

        # 5. Top-k 선택
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)

        # 6. 선택된 뉴런 (v4.5: aggregate 제거!)
        selected = self.neurons[topk_idx]  # [B, S, k, d_model]

        if return_loss:
            ortho_loss = self.compute_orthogonality_loss()
            return selected, topk_idx, topk_weights, context, ortho_loss

        return selected, topk_idx, topk_weights, context

    def compute_orthogonality_loss(self):
        """뉴런 벡터 직교성 강화"""
        neurons_norm = F.normalize(self.neurons, p=2, dim=1)
        gram = torch.mm(neurons_norm, neurons_norm.T)
        identity = torch.eye(self.n_neurons, device=gram.device)
        ortho_loss = ((gram - identity) ** 2).sum()
        ortho_loss = ortho_loss / (self.n_neurons * (self.n_neurons - 1))
        return ortho_loss


# ============================================
# 2. 패턴별 변환 FFN (v4.5)
# ============================================
class InteractionFFN(nn.Module):
    """v4.5: Pattern-specific Up Projection with Cross-neuron Gating

    핵심 아이디어:
    1. Cross-neuron gating: 선택된 뉴런들이 서로 보고 feature 조절
    2. Pattern selection: Context + neuron 조합으로 패턴 선택
    3. Pattern-specific up: 각 패턴이 256→1024 변환을 다르게 수행!
       - Pattern 5: "시각적 feature 조합" 공간
       - Pattern 12: "의미적 feature 조합" 공간
    4. Down projection: 1024→256으로 필요한 정보만 추출
    """

    def __init__(self, n_neurons=512, d_model=256, d_ff=1024,
                 n_patterns=32, k_patterns=4, n_heads=4, rank=64,
                 pattern_dropout=0.0, use_base=True):
        super().__init__()
        self.n_patterns = n_patterns
        self.k_patterns = k_patterns
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank = rank
        self.pattern_dropout = pattern_dropout
        self.use_base = use_base

        # =====================================================
        # Part 1: Cross-Neuron Gating
        # =====================================================
        self.neuron_q = nn.Linear(d_model, d_model)
        self.neuron_k = nn.Linear(d_model, d_model)
        self.neuron_v = nn.Linear(d_model, d_model)

        # =====================================================
        # Part 2: Pattern Queries
        # =====================================================
        self.pattern_queries = nn.Parameter(
            torch.randn(n_patterns, d_model) * 0.02
        )

        # =====================================================
        # Part 3: Pattern-Specific Up Projection (Low-rank)
        # =====================================================
        # 각 패턴: [d_model, rank] @ [rank, d_ff] = [d_model, d_ff]
        self.pattern_up_A = nn.Parameter(
            torch.randn(n_patterns, d_model, rank) * 0.02
        )
        self.pattern_up_B = nn.Parameter(
            torch.randn(n_patterns, rank, d_ff) * 0.01
        )

        # Optional: Base projection for stability
        if use_base:
            self.up_base = nn.Linear(d_model, d_ff)
            nn.init.normal_(self.up_base.weight, std=0.01)
            if self.up_base.bias is not None:
                nn.init.zeros_(self.up_base.bias)

        # =====================================================
        # Part 4: Down Projection
        # =====================================================
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x, selected_neurons, topk_neuron_weights, context,
                return_pattern_weights=False, return_loss=False):
        """
        Args:
            x: [B, S, d_model] - input
            selected_neurons: [B, S, K, d_model] - router가 선택한 뉴런들
            topk_neuron_weights: [B, S, K] - 각 뉴런 중요도
            context: [B, S, d_model] - router의 context
        """
        B, S, K, D = selected_neurons.shape

        # =====================================================
        # STEP 1: Cross-Neuron Gating
        # =====================================================

        # Reshape for multi-head attention
        neurons_flat = selected_neurons.view(B * S, K, D)

        # Q, K, V projection
        q = self.neuron_q(neurons_flat).view(B * S, K, self.n_heads, self.d_head)
        k = self.neuron_k(neurons_flat).view(B * S, K, self.n_heads, self.d_head)
        v = self.neuron_v(neurons_flat).view(B * S, K, self.n_heads, self.d_head)

        # V를 gating vector로 변환
        v = torch.sigmoid(v)  # [0, 1] range

        # Transpose: [B*S, n_heads, K, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B*S, n_heads, K, K]

        # Apply attention to gates
        gates = torch.matmul(attn_weights, v)  # [B*S, n_heads, K, d_head]
        gates = gates.transpose(1, 2).contiguous().view(B, S, K, D)

        # Apply gating to neurons
        modulated_neurons = selected_neurons * gates

        # =====================================================
        # STEP 2: Aggregate Modulated Neurons
        # =====================================================

        # Weighted average by neuron importance
        aggregated = (topk_neuron_weights.unsqueeze(-1) * modulated_neurons).sum(dim=2)
        # [B, S, D]

        # =====================================================
        # STEP 3: Pattern Selection
        # =====================================================

        # 3-1. Neuron-based scores
        neuron_pattern_scores = torch.matmul(
            aggregated,
            self.pattern_queries.T
        ) / (D ** 0.5)  # [B, S, n_patterns]

        # 3-2. Context-based scores
        context_pattern_scores = torch.matmul(
            context,
            self.pattern_queries.T
        )  # [B, S, n_patterns]

        # 3-3. Combine
        pattern_scores = 0.5 * neuron_pattern_scores + 0.5 * context_pattern_scores

        # 3-4. Pattern Dropout (training only)
        if self.training and self.pattern_dropout > 0:
            drop_mask = torch.rand(
                1, 1, self.n_patterns,
                device=pattern_scores.device
            ) > self.pattern_dropout
            pattern_scores = pattern_scores.masked_fill(
                ~drop_mask,
                float('-inf')
            )

        # 3-5. Top-k pattern selection
        topk_scores, topk_pattern_idx = torch.topk(
            pattern_scores, self.k_patterns, dim=-1
        )
        topk_pattern_weights = F.softmax(topk_scores, dim=-1)

        # =====================================================
        # STEP 4: Pattern-Specific Up Projection
        # =====================================================

        combined = x + aggregated  # Residual

        # 4-1. Gather selected patterns' matrices
        # topk_pattern_idx: [B, S, k_patterns]
        flat_idx = topk_pattern_idx.view(-1)  # [B*S*k_patterns]

        # Gather: [B*S*k_patterns, D, rank] and [B*S*k_patterns, rank, d_ff]
        A_flat = self.pattern_up_A[flat_idx]  # [B*S*k_patterns, D, rank]
        B_flat = self.pattern_up_B[flat_idx]  # [B*S*k_patterns, rank, d_ff]

        # Reshape
        A = A_flat.view(B, S, self.k_patterns, D, self.rank)
        B = B_flat.view(B, S, self.k_patterns, self.rank, self.d_ff)

        # 4-2. Pattern-specific projections
        # combined: [B, S, D] → [B, S, 1, D] for broadcasting
        combined_exp = combined.unsqueeze(2)  # [B, S, 1, D]

        # Matmul: combined @ A
        # [B, S, 1, D] @ [B, S, k_patterns, D, rank]
        # Use bmm efficiently
        h_mid = torch.matmul(
            combined_exp.unsqueeze(3),  # [B, S, 1, 1, D]
            A  # [B, S, k_patterns, D, rank]
        ).squeeze(3)  # [B, S, k_patterns, rank]

        # Matmul: h_mid @ B
        # [B, S, k_patterns, rank] @ [B, S, k_patterns, rank, d_ff]
        h_patterns = torch.matmul(
            h_mid.unsqueeze(3),  # [B, S, k_patterns, 1, rank]
            B  # [B, S, k_patterns, rank, d_ff]
        ).squeeze(3)  # [B, S, k_patterns, d_ff]

        # 4-3. Weighted combination
        h_pattern = (topk_pattern_weights.unsqueeze(-1) * h_patterns).sum(dim=2)
        # [B, S, d_ff]

        # 4-4. Optional: Base projection
        if self.use_base:
            h_base = self.up_base(combined)  # [B, S, d_ff]
            h = 0.1 * h_base + 0.9 * h_pattern
        else:
            h = h_pattern

        # =====================================================
        # STEP 5: Non-linearity
        # =====================================================

        h = F.gelu(h)

        # =====================================================
        # STEP 6: Down Projection
        # =====================================================

        output = self.down(h)

        # =====================================================
        # STEP 7: Return
        # =====================================================

        if return_loss:
            load_loss = self.compute_load_balancing_loss(pattern_scores)
            if return_pattern_weights:
                full_weights = torch.zeros(B, S, self.n_patterns, device=x.device)
                full_weights.scatter_(-1, topk_pattern_idx, topk_pattern_weights)
                return output, full_weights, load_loss
            return output, load_loss

        if return_pattern_weights:
            full_weights = torch.zeros(B, S, self.n_patterns, device=x.device)
            full_weights.scatter_(-1, topk_pattern_idx, topk_pattern_weights)
            return output, full_weights

        return output

    def compute_load_balancing_loss(self, pattern_scores):
        """패턴 사용 균등화"""
        pattern_probs = F.softmax(pattern_scores, dim=-1)
        pattern_usage = pattern_probs.mean(dim=(0, 1))
        target = 1.0 / self.n_patterns
        load_loss = ((pattern_usage - target) ** 2).sum()
        return load_loss


# ============================================
# 3. 단일 레이어 (v4.5)
# ============================================
class Layer(nn.Module):
    """v4.5: Cross-neuron gating + Pattern-specific up projection"""

    def __init__(self, d_model=256, d_ff=1024, n_heads=4,
                 n_neurons=512, n_patterns=32, neuron_k=16, pattern_k=4,
                 rank=64, pattern_dropout=0.0, use_base=True):
        super().__init__()

        # 뉴런 선택
        self.neuron_router = NeuronRouter(n_neurons, d_model, n_heads, neuron_k)

        # 뉴런 상호작용 + Pattern-specific FFN
        self.neuron_interaction = InteractionFFN(
            n_neurons, d_model, d_ff, n_patterns, pattern_k, n_heads,
            rank, pattern_dropout, use_base
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_details=False, return_losses=False):
        # 1. 뉴런 라우팅 (선택만!)
        normed = self.norm1(x)
        if return_losses:
            selected_neurons, topk_idx, topk_weights, context, ortho_loss = \
                self.neuron_router(normed, mask, return_loss=True)
        else:
            selected_neurons, topk_idx, topk_weights, context = \
                self.neuron_router(normed, mask)

        # 2. 뉴런 상호작용 + Pattern-specific FFN
        normed = self.norm2(x)
        if return_losses:
            if return_details:
                interaction_out, pattern_weights, load_loss = \
                    self.neuron_interaction(
                        normed, selected_neurons, topk_weights, context,
                        return_pattern_weights=True, return_loss=True
                    )
            else:
                interaction_out, load_loss = \
                    self.neuron_interaction(
                        normed, selected_neurons, topk_weights, context,
                        return_loss=True
                    )
                pattern_weights = None
        else:
            if return_details:
                interaction_out, pattern_weights = \
                    self.neuron_interaction(
                        normed, selected_neurons, topk_weights, context,
                        return_pattern_weights=True
                    )
            else:
                interaction_out = \
                    self.neuron_interaction(
                        normed, selected_neurons, topk_weights, context
                    )
                pattern_weights = None

        x = x + interaction_out

        # Return
        if return_losses:
            if return_details:
                return x, topk_idx, pattern_weights, load_loss, ortho_loss
            return x, topk_idx, load_loss, ortho_loss

        if return_details:
            return x, topk_idx, pattern_weights
        return x, topk_idx


# ============================================
# 4. DAWN 모델 (v4.5)
# ============================================
class DAWN(nn.Module):
    """Dynamic Architecture With Neurons"""

    __version__ = "4.5"
    # v4.5: Pattern-specific up projection + Cross-neuron gating

    def __init__(self, vocab_size, d_model=256, d_ff=1024, n_layers=4, n_heads=4,
                 n_neurons=512, n_patterns=32, neuron_k=16, pattern_k=4, rank=64,
                 max_seq_len=512, dropout=0.1, pattern_dropout=0.0, use_base=True,
                 # Backward compatibility
                 hidden_dim=None, num_layers=None, k=None,
                 num_input_neurons=None, num_process_neurons=None):
        super().__init__()

        # Backward compatibility
        if hidden_dim is not None:
            d_model = hidden_dim
        if num_layers is not None:
            n_layers = num_layers
        if k is not None:
            neuron_k = k
        if num_input_neurons is not None:
            n_neurons = num_input_neurons * 16
        if num_process_neurons is not None:
            n_patterns = num_process_neurons * 4

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_neurons = n_neurons

        # Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Layers
        self.layers = nn.ModuleList([
            Layer(d_model, d_ff, n_heads, n_neurons, n_patterns,
                  neuron_k, pattern_k, rank, pattern_dropout, use_base)
            for _ in range(n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

        # Store for compatibility
        self.hidden_dim = d_model

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

    def forward(self, input_ids, return_activations=False, return_losses=False):
        B, S = input_ids.shape

        # Embedding
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        # Causal mask
        mask = torch.tril(torch.ones(S, S, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Layers
        all_selected = []
        all_patterns = []
        pattern_load_losses = []
        neuron_ortho_losses = []

        for layer in self.layers:
            if return_losses:
                if return_activations:
                    x, selected_idx, pattern_weights, load_loss, ortho_loss = layer(
                        x, mask, return_details=True, return_losses=True
                    )
                    all_selected.append(selected_idx)
                    all_patterns.append(pattern_weights)
                else:
                    x, selected_idx, load_loss, ortho_loss = layer(
                        x, mask, return_details=False, return_losses=True
                    )
                pattern_load_losses.append(load_loss)
                neuron_ortho_losses.append(ortho_loss)
            elif return_activations:
                x, selected_idx, pattern_weights = layer(x, mask, return_details=True)
                all_selected.append(selected_idx)
                all_patterns.append(pattern_weights)
            else:
                x, selected_idx = layer(x, mask, return_details=False)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        # Return
        if return_losses:
            losses = {
                'pattern_load': pattern_load_losses,
                'neuron_ortho': neuron_ortho_losses
            }
            if return_activations:
                return logits, all_selected, all_patterns, losses
            return logits, losses

        if return_activations:
            return logits, all_selected, all_patterns
        return logits

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
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
    """Config로부터 모델 생성"""
    return DAWN(**config)


# Backward compatibility
DAWNLanguageModel = DAWN
