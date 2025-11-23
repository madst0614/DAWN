import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 1. 문맥 기반 뉴런 라우터
# ============================================
class NeuronRouter(nn.Module):
    """Full-rank neuron routing (v4.1: Connection removed for simplicity)"""

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
            return output, topk_idx, topk_weights, selected, ortho_loss

        return output, topk_idx, topk_weights, selected

    def compute_orthogonality_loss(self):
        """뉴런 벡터 직교성 강화 (v4.2)

        목표: 각 뉴런이 서로 다른 방향을 가리키도록
        방법: Gram matrix가 identity에 가까워지도록
        """
        # Normalize neurons to unit vectors
        neurons_norm = F.normalize(self.neurons, p=2, dim=1)  # [n_neurons, d_model]

        # Compute Gram matrix (cosine similarity)
        gram = torch.mm(neurons_norm, neurons_norm.T)  # [n_neurons, n_neurons]

        # Target: identity matrix (orthogonal)
        identity = torch.eye(self.n_neurons, device=gram.device)

        # L2 loss
        ortho_loss = ((gram - identity) ** 2).sum()

        # Normalize by number of off-diagonal elements
        ortho_loss = ortho_loss / (self.n_neurons * (self.n_neurons - 1))

        return ortho_loss


# ============================================
# 2. 패턴 기반 동적 FFN
# ============================================
class PatternFFN(nn.Module):
    """v4.1: Weighted-After Pattern Selection

    핵심 아이디어:
    - 패턴 쿼리와 뉴런 벡터의 유사도 계산 (순수 매칭)
    - 뉴런 가중치를 곱해서 기여도 반영
    - 합산하여 패턴 점수 도출

    직관적 해석:
    1. "이 뉴런은 이 패턴과 얼마나 맞나?" (유사도)
    2. "이 뉴런이 얼마나 중요한가?" (가중치)
    3. "종합 점수" (유사도 × 가중치의 합)
    """

    def __init__(self, n_neurons=512, d_model=256, d_ff=1024,
                 n_patterns=32, k_patterns=4):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_patterns = n_patterns
        self.k_patterns = k_patterns
        self.d_model = d_model

        # 패턴 쿼리 (각 패턴의 "의미" 벡터)
        self.pattern_queries = nn.Parameter(
            torch.randn(n_patterns, d_model) * 0.02
        )

        # Pattern gates
        self.gates = nn.Parameter(torch.randn(n_patterns, d_ff) * 0.02)

        # FFN
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x, router_out, topk_neuron_idx, topk_neuron_weights,
                selected_neurons, return_pattern_weights=False, return_loss=False):
        """
        Args:
            x: [B, S, d_model] - input
            router_out: [B, S, d_model] - neuron combination output
            topk_neuron_idx: [B, S, K] - selected neuron indices
            topk_neuron_weights: [B, S, K] - neuron importance weights
            selected_neurons: [B, S, K, d_model] - selected neuron vectors
            return_pattern_weights: bool - return full pattern weight matrix
            return_loss: bool - return load balancing loss (v4.2)
        """
        B, S, K, D = selected_neurons.shape

        # 1️⃣ 뉴런-패턴 유사도 계산 (순수 매칭)
        # pattern_queries: [n_patterns, D]
        # selected_neurons: [B, S, K, D]
        neuron_pattern_similarity = torch.einsum(
            'pd,bskd->pbsk',
            self.pattern_queries,
            selected_neurons
        ) / (D ** 0.5)
        # [n_patterns, B, S, K]
        # 예: [패턴7, batch, seq, 뉴런0] = 패턴7과 선택된 첫 뉴런의 유사도

        # 2️⃣ 뉴런 중요도 가중치 적용
        # topk_neuron_weights: [B, S, K]
        weighted_similarity = neuron_pattern_similarity * topk_neuron_weights
        # [n_patterns, B, S, K]
        # 예: [패턴7, batch, seq, 뉴런0] = (유사도) × (중요도)

        # 3️⃣ 뉴런들의 기여도 합산
        pattern_scores = weighted_similarity.sum(dim=-1)  # sum over K
        # [n_patterns, B, S]

        # Transpose to [B, S, n_patterns]
        pattern_scores = pattern_scores.permute(1, 2, 0)

        # 4️⃣ Top-k 패턴 선택
        topk_scores, topk_pattern_idx = torch.topk(
            pattern_scores, self.k_patterns, dim=-1
        )
        topk_pattern_weights = F.softmax(topk_scores, dim=-1)

        # 5️⃣ 선택된 패턴의 gate 조합
        selected_gates = self.gates[topk_pattern_idx]  # [B, S, k_patterns, d_ff]
        ffn_gate = torch.sum(
            topk_pattern_weights.unsqueeze(-1) * selected_gates,
            dim=2
        )  # [B, S, d_ff]

        # 6️⃣ Gated FFN
        h = self.up(x)
        h = h * torch.sigmoid(ffn_gate)
        h = F.gelu(h)
        output = self.down(h)

        # 7️⃣ Load balancing loss (v4.2)
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
        """패턴 사용 균등화 (v4.2)

        목표: 모든 패턴이 골고루 선택되도록
        방법: 각 패턴의 평균 선택 확률이 uniform에 가까워지도록

        Args:
            pattern_scores: [B, S, n_patterns] - raw pattern scores before top-k
        """
        # Convert scores to probabilities
        pattern_probs = F.softmax(pattern_scores, dim=-1)  # [B, S, n_patterns]

        # Average probability across batch and sequence
        pattern_usage = pattern_probs.mean(dim=(0, 1))  # [n_patterns]

        # Uniform target
        target = 1.0 / self.n_patterns

        # L2 loss: encourage uniform distribution
        load_loss = ((pattern_usage - target) ** 2).sum()

        return load_loss


# ============================================
# 3. 단일 레이어
# ============================================
class Layer(nn.Module):
    """단일 레이어 (v4.2: Load Balancing + Orthogonality)"""

    def __init__(self, d_model=256, d_ff=1024, n_heads=4,
                 n_neurons=512, n_patterns=32, neuron_k=16, pattern_k=16):
        super().__init__()

        self.router = NeuronRouter(n_neurons, d_model, n_heads, neuron_k)
        self.ffn = PatternFFN(n_neurons, d_model, d_ff, n_patterns, pattern_k)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_details=False, return_losses=False):
        # 1. 뉴런 라우팅
        normed = self.norm1(x)
        if return_losses:
            router_out, topk_idx, topk_weights, selected_neurons, ortho_loss = self.router(
                normed, mask, return_loss=True
            )
        else:
            router_out, topk_idx, topk_weights, selected_neurons = self.router(normed, mask)
        x = x + router_out

        # 2. 패턴 FFN
        normed = self.norm2(x)
        if return_losses:
            if return_details:
                ffn_out, pattern_weights, load_loss = self.ffn(
                    normed, router_out, topk_idx, topk_weights, selected_neurons,
                    return_pattern_weights=True, return_loss=True
                )
            else:
                ffn_out, load_loss = self.ffn(
                    normed, router_out, topk_idx, topk_weights, selected_neurons,
                    return_loss=True
                )
                pattern_weights = None
        else:
            if return_details:
                ffn_out, pattern_weights = self.ffn(
                    normed, router_out, topk_idx, topk_weights, selected_neurons,
                    return_pattern_weights=True
                )
            else:
                ffn_out = self.ffn(normed, router_out, topk_idx, topk_weights, selected_neurons)
                pattern_weights = None
        x = x + ffn_out

        # Return based on flags
        if return_losses:
            if return_details:
                return x, topk_idx, pattern_weights, load_loss, ortho_loss
            return x, topk_idx, load_loss, ortho_loss

        if return_details:
            return x, topk_idx, pattern_weights
        return x, topk_idx


# ============================================
# 4. DAWN 모델
# ============================================
class DAWN(nn.Module):
    """Dynamic Architecture With Neurons"""

    __version__ = "4.2"  # 버전 관리
    # v1.0: NeuronPool + NeuronAttention (separate) - deprecated
    # v2.0: Unified NeuronRouter (no connections)
    # v2.1: NeuronRouter with inter-layer connections
    # v3.0: NeuronRouter with bottom-up/top-down gating
    # v3.1: Dynamic mixing with learned path weights
    # v3.2: Low-rank neurons/patterns for forced diversity
    # v3.4: Full-rank with increased capacity (512 neurons, 256 patterns)
    # v3.5: 뉴런 조합 기반 단순 패턴 선택 (32 patterns, 87% 파라미터 감소)
    # v3.6: Attention-based pattern selection (Q-K attention for pattern matching)
    # v3.7: Orthogonal init + Learnable temperature (collapse 방지)
    # v4.0: Neuron-Pattern Affinity Matching (뉴런 분화 → 패턴 분화 유도)
    # v4.1: Weighted-After Pattern Selection (유사도 → 가중치 → 합산, 직관적)
    # v4.2: Load Balancing + Orthogonality (패턴 균등 + 뉴런 다양성)

    def __init__(self, vocab_size, d_model=256, d_ff=1024, n_layers=4, n_heads=4,
                 n_neurons=512, n_patterns=32, neuron_k=16, pattern_k=16,
                 max_seq_len=512, dropout=0.1,
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
            Layer(d_model, d_ff, n_heads, n_neurons, n_patterns, neuron_k, pattern_k)
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

        # Return based on flags
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
# 5. 학습 유틸리티
# ============================================
class DAWNLanguageModel(DAWN):
    """Language Model wrapper (backward compatibility)"""
    pass


class DAWNTrainer:
    """DAWN 학습 헬퍼"""
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_step(self, input_ids, targets):
        self.model.train()

        logits = self.model(input_ids)
        B, S, V = logits.shape

        loss = F.cross_entropy(
            logits.view(B * S, V),
            targets.view(B * S),
            ignore_index=-100
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def analyze_activations(self, input_ids):
        """뉴런 선택 패턴 분석"""
        self.model.eval()

        with torch.no_grad():
            _, all_selected, all_patterns = self.model(input_ids, return_activations=True)

        analysis = {}
        for layer_idx, selected_idx in enumerate(all_selected):
            # selected_idx: [B, S, k]
            unique_neurons = torch.unique(selected_idx).numel()

            # Get router from layer
            if hasattr(self.model, '_orig_mod'):
                total_neurons = self.model._orig_mod.layers[layer_idx].router.n_neurons
            else:
                total_neurons = self.model.layers[layer_idx].router.n_neurons

            analysis[f'layer_{layer_idx}'] = {
                'unique_neurons': unique_neurons,
                'total_neurons': total_neurons,
                'usage_ratio': unique_neurons / total_neurons
            }

        return analysis

    def analyze_connections(self):
        """레이어 간 connection 분석"""
        if hasattr(self.model, '_orig_mod'):
            return self.model._orig_mod.get_connection_stats()
        return self.model.get_connection_stats()


# ============================================
# 6. 모델 생성 헬퍼
# ============================================
def create_model(config):
    """Config로부터 모델 생성"""
    return DAWN(**config)


# Backward compatibility exports
DynamicNeuronTransformer = DAWN
InputNeurons = NeuronRouter  # Old name
ProcessNeurons = PatternFFN  # Old name
