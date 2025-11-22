import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 1. 문맥 기반 뉴런 라우터
# ============================================
class NeuronRouter(nn.Module):
    """문맥 기반 뉴런 라우팅"""

    def __init__(self, n_neurons=256, d_model=256, n_heads=4, k=8,
                 prev_n_neurons=None):
        super().__init__()
        self.n_neurons = n_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.k = k

        # 뉴런 풀
        self.neurons = nn.Parameter(torch.randn(n_neurons, d_model) * 0.02)

        # cross-token attention용
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 뉴런 선택용 (dynamic mixing)
        self.path_proj = nn.Linear(d_model * 2, 2)  # 2 paths: token vs context

        # 이전 레이어와의 connection (있으면)
        self.has_connection = prev_n_neurons is not None
        if self.has_connection:
            self.connection = nn.Linear(prev_n_neurons, n_neurons, bias=False)
            nn.init.zeros_(self.connection.weight)  # 처음엔 영향 없게

    def forward(self, x, mask=None, prev_selection=None):
        B, S, D = x.shape

        # 1. cross-token attention (문맥 수집)
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

        # 4. Dynamic mixing: 상황에 따라 bottom-up vs top-down 비율 조절
        combined = torch.cat([x, context], dim=-1)  # [B, S, 2*D]
        weights = F.softmax(self.path_proj(combined), dim=-1)  # [B, S, 2]

        scores = weights[:, :, 0:1] * token_scores + \
                 weights[:, :, 1:2] * context_scores  # [B, S, n_neurons]

        # 5. Lateral: 이전 레이어 selection이 현재 점수 조절
        if self.has_connection and prev_selection is not None:
            influence = self.connection(prev_selection)  # [B, S, n_neurons]
            scores = scores + influence

        # 6. Top-k 선택
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)

        # 7. 선택된 뉴런 조합
        selected = self.neurons[topk_idx]
        output = torch.sum(topk_weights.unsqueeze(-1) * selected, dim=2)

        # 8. 다음 레이어로 전달할 selection (soft version)
        selection_out = torch.zeros(B, S, self.n_neurons, device=x.device)
        selection_out.scatter_(-1, topk_idx, topk_weights)

        return output, topk_idx, topk_weights, selection_out


# ============================================
# 2. 패턴 기반 동적 FFN
# ============================================
class PatternFFN(nn.Module):
    """패턴 기반 동적 FFN (with gating)"""

    def __init__(self, d_model=256, d_ff=1024, n_patterns=128, k=16):
        super().__init__()
        self.n_patterns = n_patterns
        self.k = k

        # 패턴 저장소
        self.patterns = nn.Parameter(torch.randn(n_patterns, d_model) * 0.02)
        self.gates = nn.Parameter(torch.randn(n_patterns, d_ff) * 0.02)

        # 패턴 선택용 (dynamic mixing - NeuronRouter와 일관성)
        self.path_proj = nn.Linear(d_model * 2, 2)  # 2 paths: input vs router

        # FFN
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x, router_out, return_pattern_weights=False):
        B, S, D = x.shape

        # 1. Bottom-up: 입력 기반 패턴 점수
        pattern_scores = torch.matmul(x, self.patterns.T)  # [B, S, n_patterns]

        # 2. Top-down: 라우터 출력 기반 패턴 점수
        router_scores = torch.matmul(router_out, self.patterns.T)  # [B, S, n_patterns]

        # 3. Dynamic mixing: 상황에 따라 input vs router 비율 조절
        combined = torch.cat([x, router_out], dim=-1)  # [B, S, 2*D]
        weights = F.softmax(self.path_proj(combined), dim=-1)  # [B, S, 2]

        scores = weights[:, :, 0:1] * pattern_scores + \
                 weights[:, :, 1:2] * router_scores  # [B, S, n_patterns]

        # 4. Top-k 선택
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)

        # 5. FFN gate 조합
        selected_gates = self.gates[topk_idx]
        ffn_gate = torch.sum(topk_weights.unsqueeze(-1) * selected_gates, dim=2)

        # 6. Gated FFN
        h = self.up(x)
        h = h * torch.sigmoid(ffn_gate)
        h = F.gelu(h)
        output = self.down(h)

        if return_pattern_weights:
            # 분석용 전체 weights 복원
            full_weights = torch.zeros(B, S, self.n_patterns, device=x.device)
            full_weights.scatter_(-1, topk_idx, topk_weights)
            return output, full_weights

        return output


# ============================================
# 3. 단일 레이어
# ============================================
class Layer(nn.Module):
    """단일 레이어"""

    def __init__(self, d_model=256, d_ff=1024, n_heads=4,
                 n_neurons=256, n_patterns=128, neuron_k=8, pattern_k=16,
                 prev_n_neurons=None):
        super().__init__()

        self.router = NeuronRouter(n_neurons, d_model, n_heads, neuron_k,
                                   prev_n_neurons=prev_n_neurons)
        self.ffn = PatternFFN(d_model, d_ff, n_patterns, pattern_k)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, prev_selection=None, return_details=False):
        # 1. 뉴런 라우팅 (문맥 기반 + 이전 레이어 영향)
        normed = self.norm1(x)
        router_out, topk_idx, topk_weights, selection_out = self.router(
            normed, mask, prev_selection
        )
        x = x + router_out

        # 2. 패턴 FFN
        normed = self.norm2(x)
        if return_details:
            ffn_out, pattern_weights = self.ffn(normed, router_out, return_pattern_weights=True)
        else:
            ffn_out = self.ffn(normed, router_out)
            pattern_weights = None
        x = x + ffn_out

        if return_details:
            return x, topk_idx, pattern_weights, selection_out
        return x, topk_idx, selection_out


# ============================================
# 4. DAWN 모델
# ============================================
class DAWN(nn.Module):
    """Dynamic Architecture With Neurons"""

    __version__ = "3.1"  # 버전 관리
    # v1.0: NeuronPool + NeuronAttention (separate) - deprecated
    # v2.0: Unified NeuronRouter (no connections)
    # v2.1: NeuronRouter with inter-layer connections
    # v3.0: NeuronRouter with bottom-up/top-down gating
    # v3.1: Dynamic mixing with learned path weights

    def __init__(self, vocab_size, d_model=256, d_ff=1024, n_layers=4, n_heads=4,
                 n_neurons=256, n_patterns=128, neuron_k=8, pattern_k=16,
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

        # Layers (with connection)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            prev_n = n_neurons if i > 0 else None  # 첫 레이어는 connection 없음
            self.layers.append(
                Layer(d_model, d_ff, n_heads, n_neurons, n_patterns,
                      neuron_k, pattern_k, prev_n_neurons=prev_n)
            )

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

    def forward(self, input_ids, return_activations=False):
        B, S = input_ids.shape

        # Embedding
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        # Causal mask
        mask = torch.tril(torch.ones(S, S, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Layers (with connection propagation)
        all_selected = []
        all_patterns = []
        prev_selection = None

        for layer in self.layers:
            if return_activations:
                x, selected_idx, pattern_weights, selection_out = layer(
                    x, mask, prev_selection, return_details=True
                )
                all_selected.append(selected_idx)
                all_patterns.append(pattern_weights)
            else:
                x, selected_idx, selection_out = layer(
                    x, mask, prev_selection, return_details=False
                )

            prev_selection = selection_out  # 다음 레이어로 전달

        # Output
        x = self.norm(x)
        logits = self.head(x)

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

    def get_connection_stats(self):
        """레이어 간 connection 분석"""
        stats = {}
        for i, layer in enumerate(self.layers):
            if layer.router.has_connection:
                weight = layer.router.connection.weight.data
                stats[f'layer_{i}'] = {
                    'mean': weight.mean().item(),
                    'std': weight.std().item(),
                    'max': weight.max().item(),
                    'min': weight.min().item(),
                    'sparsity': (weight.abs() < 0.01).float().mean().item()
                }
        return stats


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
