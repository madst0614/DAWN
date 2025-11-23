import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 1. 문맥 기반 뉴런 라우터
# ============================================
class NeuronRouter(nn.Module):
    """Full-rank neuron routing (v4.4: returns context for downstream)"""

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

        # 6. 선택된 뉴런 조합
        selected = self.neurons[topk_idx]  # [B, S, k, d_model]
        output = torch.sum(topk_weights.unsqueeze(-1) * selected, dim=2)

        # 7. Orthogonality loss (v4.2)
        if return_loss:
            ortho_loss = self.compute_orthogonality_loss()
            # v4.4: context도 반환
            return output, topk_idx, topk_weights, selected, context, ortho_loss

        # v4.4: context 반환
        return output, topk_idx, topk_weights, selected, context

    def compute_orthogonality_loss(self):
        """뉴런 벡터 직교성 강화"""
        neurons_norm = F.normalize(self.neurons, p=2, dim=1)
        gram = torch.mm(neurons_norm, neurons_norm.T)
        identity = torch.eye(self.n_neurons, device=gram.device)
        ortho_loss = ((gram - identity) ** 2).sum()
        ortho_loss = ortho_loss / (self.n_neurons * (self.n_neurons - 1))
        return ortho_loss


# ============================================
# 2. 뉴런 상호작용 FFN (v4.4 NEW!)
# ============================================
class InteractionFFN(nn.Module):
    """Neuron interaction modeling with pattern-based FFN

    v4.4의 핵심 아이디어:
    1. Cross-neuron attention: 선택된 뉴런들이 서로 상호작용
    2. Pattern-based interpretation: 패턴이 "어떤 상호작용 패턴"을 볼지 선택
    3. Context-aware: 같은 뉴런 조합이라도 context에 따라 다른 패턴

    직관:
    - 뉴런 A (edge) + 뉴런 B (corner) → Self-attention → "사각형"
    - 패턴: "형태 인식", "질감 인식" 등 다양한 관점
    - Context: 문맥에 따라 어떤 패턴이 적합한지 선택
    """

    def __init__(self, n_neurons=512, d_model=256, d_ff=1024,
                 n_patterns=32, k_patterns=4, pattern_dropout=0.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_patterns = n_patterns
        self.k_patterns = k_patterns
        self.d_model = d_model
        self.pattern_dropout = pattern_dropout

        # =====================================================
        # Part 1: Cross-Neuron Interaction
        # =====================================================
        # 뉴런들이 서로를 보고 상호작용하기 위한 attention
        self.neuron_q = nn.Linear(d_model, d_model)
        self.neuron_k = nn.Linear(d_model, d_model)
        self.neuron_v = nn.Linear(d_model, d_model)

        # =====================================================
        # Part 2: Pattern Queries
        # =====================================================
        # 패턴 = "어떤 상호작용 패턴을 볼 것인가"
        self.pattern_queries = nn.Parameter(
            torch.randn(n_patterns, d_model) * 0.02
        )

        # =====================================================
        # Part 3: FFN
        # =====================================================
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x, router_out, topk_neuron_idx, topk_neuron_weights,
                selected_neurons, context, return_pattern_weights=False,
                return_loss=False):
        """
        Args:
            x: [B, S, d_model] - input
            router_out: [B, S, d_model] - neuron combination output (not used here)
            topk_neuron_idx: [B, S, K] - selected neuron indices
            topk_neuron_weights: [B, S, K] - neuron importance weights
            selected_neurons: [B, S, K, d_model] - selected neuron vectors
            context: [B, S, d_model] - context from NeuronRouter
            return_pattern_weights: bool
            return_loss: bool
        """
        B, S, K, D = selected_neurons.shape

        # =====================================================
        # STEP 1: Cross-Neuron Interaction (뉴런 간 상호작용)
        # =====================================================

        # 뉴런들을 [B*S, K, D]로 reshape
        neurons_flat = selected_neurons.view(B * S, K, D)

        # Self-attention: 각 뉴런이 다른 뉴런들을 봄
        q = self.neuron_q(neurons_flat)  # [B*S, K, D]
        k_proj = self.neuron_k(neurons_flat)
        v = self.neuron_v(neurons_flat)

        # Attention scores
        attn_scores = torch.matmul(q, k_proj.transpose(-2, -1)) / (D ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B*S, K, K]

        # 상호작용 결과
        # neuron_i' = neuron_i + Σ(attention_weights[i,j] * neuron_j)
        interacted_neurons = torch.matmul(attn_weights, v)  # [B*S, K, D]
        interacted_neurons = interacted_neurons.view(B, S, K, D)

        # =====================================================
        # STEP 2: Pattern Selection (패턴 선택)
        # =====================================================

        # 2-1. 상호작용한 뉴런들과 패턴 매칭
        # [n_patterns, D] × [B, S, K, D] → [n_patterns, B, S, K]
        neuron_pattern_similarity = torch.einsum(
            'pd,bskd->pbsk',
            self.pattern_queries,
            interacted_neurons
        ) / (D ** 0.5)

        # 2-2. 뉴런 중요도로 가중
        neuron_pattern_similarity = neuron_pattern_similarity.permute(1, 2, 0, 3)
        # [B, S, n_patterns, K]

        weighted_similarity = (
            neuron_pattern_similarity * topk_neuron_weights.unsqueeze(2)
        )

        # 2-3. 각 패턴의 점수 = 뉴런들의 기여도 합
        neuron_based_scores = weighted_similarity.sum(dim=-1)  # [B, S, n_patterns]

        # 2-4. Context 기반 패턴 점수
        context_based_scores = torch.matmul(
            context,  # [B, S, D]
            self.pattern_queries.T  # [D, n_patterns]
        )  # [B, S, n_patterns]

        # 2-5. 결합 (뉴런 조합 + 문맥)
        # 같은 뉴런이라도 context에 따라 다른 패턴!
        pattern_scores = 0.5 * neuron_based_scores + 0.5 * context_based_scores

        # 2-6. Pattern Dropout (training only)
        if self.training and self.pattern_dropout > 0:
            drop_mask = torch.rand(
                1, 1, self.n_patterns,
                device=pattern_scores.device
            ) > self.pattern_dropout
            pattern_scores = pattern_scores.masked_fill(
                ~drop_mask,
                float('-inf')
            )

        # 2-7. Top-k 패턴 선택
        topk_scores, topk_pattern_idx = torch.topk(
            pattern_scores, self.k_patterns, dim=-1
        )
        topk_pattern_weights = F.softmax(topk_scores, dim=-1)

        # =====================================================
        # STEP 3: Pattern-Guided Aggregation (패턴별 집계)
        # =====================================================

        # 선택된 패턴들이 상호작용 결과를 어떻게 집계할지 결정
        # [B, S, k_patterns, K]
        selected_pattern_attn = neuron_pattern_similarity.gather(
            2,
            topk_pattern_idx.unsqueeze(-1).expand(-1, -1, -1, K)
        )

        # 패턴 가중치 적용
        final_attn = (
            topk_pattern_weights.unsqueeze(-1) *
            F.softmax(selected_pattern_attn, dim=-1)
        ).sum(dim=2)  # [B, S, K]

        # 최종 뉴런 조합 (상호작용 + 패턴 해석)
        aggregated = (
            final_attn.unsqueeze(-1) * interacted_neurons
        ).sum(dim=2)  # [B, S, D]

        # =====================================================
        # STEP 4: FFN Transformation
        # =====================================================

        # aggregated: 상호작용 + 패턴 기반 집계 결과
        combined = x + aggregated  # residual connection

        h = self.up(combined)
        h = F.gelu(h)
        output = self.down(h)

        # =====================================================
        # STEP 5: Regularization Loss
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
# 3. 단일 레이어 (v4.4)
# ============================================
class Layer(nn.Module):
    """v4.4: Explicit neuron interaction modeling"""

    def __init__(self, d_model=256, d_ff=1024, n_heads=4,
                 n_neurons=512, n_patterns=32, neuron_k=16, pattern_k=16,
                 pattern_dropout=0.0):
        super().__init__()

        # 뉴런 선택
        self.neuron_router = NeuronRouter(n_neurons, d_model, n_heads, neuron_k)

        # 뉴런 상호작용 + FFN
        self.neuron_interaction = InteractionFFN(
            n_neurons, d_model, d_ff, n_patterns, pattern_k, pattern_dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_details=False, return_losses=False):
        # 1. 뉴런 라우팅
        normed = self.norm1(x)
        if return_losses:
            router_out, topk_idx, topk_weights, selected_neurons, context, ortho_loss = \
                self.neuron_router(normed, mask, return_loss=True)
        else:
            router_out, topk_idx, topk_weights, selected_neurons, context = \
                self.neuron_router(normed, mask)
        x = x + router_out

        # 2. 뉴런 상호작용 + FFN
        normed = self.norm2(x)
        if return_losses:
            if return_details:
                interaction_out, pattern_weights, load_loss = \
                    self.neuron_interaction(
                        normed, router_out, topk_idx, topk_weights,
                        selected_neurons, context,
                        return_pattern_weights=True, return_loss=True
                    )
            else:
                interaction_out, load_loss = \
                    self.neuron_interaction(
                        normed, router_out, topk_idx, topk_weights,
                        selected_neurons, context,
                        return_loss=True
                    )
                pattern_weights = None
        else:
            if return_details:
                interaction_out, pattern_weights = \
                    self.neuron_interaction(
                        normed, router_out, topk_idx, topk_weights,
                        selected_neurons, context,
                        return_pattern_weights=True
                    )
            else:
                interaction_out = \
                    self.neuron_interaction(
                        normed, router_out, topk_idx, topk_weights,
                        selected_neurons, context
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
# 4. DAWN 모델 (v4.4)
# ============================================
class DAWN(nn.Module):
    """Dynamic Architecture With Neurons"""

    __version__ = "4.4"  # v4.4: Explicit Neuron Interaction
    # v4.3: Strong Regularization (10x weights: 0.1 load, 0.01 ortho)
    # v4.4: Explicit neuron interaction modeling via cross-neuron attention

    def __init__(self, vocab_size, d_model=256, d_ff=1024, n_layers=4, n_heads=4,
                 n_neurons=512, n_patterns=32, neuron_k=16, pattern_k=16,
                 max_seq_len=512, dropout=0.1, pattern_dropout=0.0,
                 # Backward compatibility
                 hidden_dim=None, num_layers=None, k=None,
                 num_input_neurons=None, num_process_neurons=None,
                 adapt_rank=None, process_rank=None):
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
                  neuron_k, pattern_k, pattern_dropout)
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
