import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 1. 뉴런 풀 (QK Retrieval 기반)
# ============================================
class NeuronPool(nn.Module):
    def __init__(self, n_neurons=1024, d_model=768, k=8):
        super().__init__()
        self.n_neurons = n_neurons
        self.k = k

        # 뉴런들 (학습 가능)
        self.neurons = nn.Parameter(torch.randn(n_neurons, d_model) * 0.02)

        # Query projection
        self.q_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape

        # Query
        q = self.q_proj(x)  # [B, S, D]

        # 뉴런 선택 (QK retrieval)
        scores = torch.matmul(q, self.neurons.T)  # [B, S, n_neurons]

        # Top-k 선택
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)  # [B, S, k]
        topk_weights = F.softmax(topk_scores, dim=-1)  # [B, S, k]

        # 선택된 뉴런 가져오기
        selected_neurons = self.neurons[topk_idx]  # [B, S, k, D]

        # 가중합
        output = torch.sum(topk_weights.unsqueeze(-1) * selected_neurons, dim=2)  # [B, S, D]

        return output, topk_idx, topk_weights


# ============================================
# 2. 패턴 기반 동적 FFN
# ============================================
class PatternFFN(nn.Module):
    def __init__(self, d_model=768, d_ff=3072, n_patterns=512):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # 패턴 저장소
        self.pattern_keys = nn.Parameter(torch.randn(n_patterns, d_model) * 0.02)
        self.pattern_gates = nn.Parameter(torch.randn(n_patterns, d_ff) * 0.02)

        # 확장/축소
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x, set_repr):
        # x: [B, S, D]
        # set_repr: [B, S, D] - 선택된 뉴런 Set의 표현

        B, S, D = x.shape

        # 1. Set 기반 패턴 retrieval
        scores = torch.matmul(set_repr, self.pattern_keys.T)  # [B, S, n_patterns]
        weights = F.softmax(scores, dim=-1)
        gate = torch.matmul(weights, self.pattern_gates)  # [B, S, d_ff]

        # 2. 확장
        h = self.up_proj(x)  # [B, S, d_ff]

        # 3. 패턴 적용 (활성/억제)
        h = h * torch.sigmoid(gate)
        h = F.gelu(h)

        # 4. 축소
        output = self.down_proj(h)  # [B, S, D]

        return output


# ============================================
# 3. 동적 Attention (뉴런 기반)
# ============================================
class NeuronAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=8, n_neurons=1024, k=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.k = k

        # 뉴런 풀
        self.neuron_pool = NeuronPool(n_neurons, d_model, k)

        # QKV projections (선택된 뉴런 기반)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, S, D = x.shape

        # 1. 뉴런 선택
        neuron_out, selected_idx, selected_weights = self.neuron_pool(x)

        # 2. QKV (뉴런 출력 기반)
        q = self.q_proj(neuron_out).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(neuron_out).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(neuron_out).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # 3. Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 4. Causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)

        # 5. Attention output
        attn_out = torch.matmul(weights, v)  # [B, n_heads, S, d_head]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        output = self.out_proj(attn_out)

        return output, neuron_out, selected_idx


# ============================================
# 4. 동적 Transformer Layer
# ============================================
class DynamicNeuronLayer(nn.Module):
    def __init__(self, d_model=768, d_ff=3072, n_heads=8,
                 n_neurons=1024, n_patterns=512, k=8):
        super().__init__()

        # Attention (뉴런 기반)
        self.attention = NeuronAttention(d_model, n_heads, n_neurons, k)

        # FFN (패턴 기반)
        self.ffn = PatternFFN(d_model, d_ff, n_patterns)

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 1. Attention with residual
        normed = self.norm1(x)
        attn_out, neuron_repr, selected_idx = self.attention(normed, mask)
        x = x + attn_out

        # 2. FFN with residual (패턴은 뉴런 표현 기반)
        normed = self.norm2(x)
        ffn_out = self.ffn(normed, neuron_repr)
        x = x + ffn_out

        return x, selected_idx


# ============================================
# 5. 전체 모델 (DAWN 호환 인터페이스)
# ============================================
class DynamicNeuronTransformer(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, d_ff=3072,
                 n_layers=6, n_heads=8, n_neurons=1024,
                 n_patterns=512, k=8, max_seq_len=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # Layers
        self.layers = nn.ModuleList([
            DynamicNeuronLayer(d_model, d_ff, n_heads, n_neurons, n_patterns, k)
            for _ in range(n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, return_activations=False):
        """
        순전파

        Args:
            input_ids: [B, S]
            return_activations: bool - 뉴런 선택 정보 반환 여부

        Returns:
            logits: [B, S, vocab_size]
            all_selected: (선택적) 레이어별 선택된 뉴런 인덱스
        """
        B, S = input_ids.shape

        # 1. Embeddings
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.embedding_dropout(x)

        # 2. Causal mask
        mask = torch.tril(torch.ones(S, S, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        # 3. Layers
        all_selected = []
        for layer in self.layers:
            x, selected_idx = layer(x, mask)
            if return_activations:
                all_selected.append(selected_idx)

        # 4. Output
        x = self.norm(x)
        logits = self.lm_head(x)

        if return_activations:
            return logits, all_selected
        else:
            return logits

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """자동회귀 생성"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ============================================
# DAWN 호환성 레이어
# ============================================
class DAWN(DynamicNeuronTransformer):
    """
    DAWN (Dynamic Architecture With Neurons)
    Dynamic Neuron Transformer 기반 재구현

    기존 DAWN 인터페이스 호환성 유지
    """
    def __init__(
        self,
        vocab_size,
        hidden_dim=512,
        num_layers=6,
        num_input_neurons=None,  # Deprecated, mapped to n_neurons
        num_process_neurons=None,  # Deprecated, mapped to n_patterns
        adapt_rank=None,  # Deprecated
        process_rank=None,  # Deprecated
        max_seq_len=2048,
        dropout=0.1,
        # New Dynamic Neuron Transformer params
        n_heads=8,
        n_neurons=1024,
        n_patterns=512,
        k=8,
        d_ff=None
    ):
        # Map old params to new architecture
        d_model = hidden_dim

        # Auto-calculate d_ff if not provided (standard is 4x d_model)
        if d_ff is None:
            d_ff = d_model * 4

        # Override with old params if provided (backward compatibility)
        if num_input_neurons is not None:
            n_neurons = num_input_neurons * 16  # Scale up for neuron pool
        if num_process_neurons is not None:
            n_patterns = num_process_neurons * 4  # Scale up for pattern pool

        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            n_layers=num_layers,
            n_heads=n_heads,
            n_neurons=n_neurons,
            n_patterns=n_patterns,
            k=k,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        # Store for compatibility
        self.hidden_dim = hidden_dim


# ========== 학습 유틸리티 ==========

class DAWNTrainer:
    """DAWN 학습 헬퍼 (Dynamic Neuron Transformer 호환)"""
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
            _, all_selected = self.model(input_ids, return_activations=True)

        analysis = {}
        for layer_idx, selected_idx in enumerate(all_selected):
            # selected_idx: [B, S, k]
            unique_neurons = torch.unique(selected_idx).numel()
            total_neurons = self.model.layers[0].attention.neuron_pool.n_neurons

            analysis[f'layer_{layer_idx}'] = {
                'unique_neurons_used': unique_neurons,
                'total_neurons': total_neurons,
                'usage_ratio': unique_neurons / total_neurons,
                'avg_neurons_per_token': selected_idx.shape[-1],  # k
            }

        return analysis


# ========== 호환성 ==========

# Backward compatibility aliases
DAWNLanguageModel = DAWN
InputNeurons = NeuronPool
ProcessNeurons = PatternFFN

def _from_config(cls, config, vocab_size):
    """Config dict로부터 모델 생성"""
    model_cfg = config.get('model', {})
    return cls(
        vocab_size=vocab_size,
        hidden_dim=model_cfg.get('d_model', 512),
        num_layers=model_cfg.get('n_layers', 6),
        max_seq_len=model_cfg.get('max_seq_len', 2048),
        dropout=model_cfg.get('dropout', 0.1),
        n_heads=model_cfg.get('n_heads', 8),
        n_neurons=model_cfg.get('n_neurons', 1024),
        n_patterns=model_cfg.get('n_patterns', 512),
        k=model_cfg.get('k', 8),
        d_ff=model_cfg.get('d_ff', None)
    )

DAWN.from_config = classmethod(_from_config)


def create_model(vocab_size=50000, **kwargs):
    """DAWN 모델 생성 (Dynamic Neuron Transformer 기반)"""
    default_config = {
        'hidden_dim': 512,
        'num_layers': 6,
        'n_heads': 8,
        'n_neurons': 1024,
        'n_patterns': 512,
        'k': 8,
        'max_seq_len': 2048,
        'dropout': 0.1
    }
    default_config.update(kwargs)

    model = DAWN(vocab_size=vocab_size, **default_config)

    # 파라미터 분석
    total_params = sum(p.numel() for p in model.parameters())

    # 레이어별 파라미터
    layer_params = sum(p.numel() for p in model.layers[0].parameters())
    embedding_params = sum(p.numel() for p in model.token_emb.parameters()) + \
                      sum(p.numel() for p in model.pos_emb.parameters())

    print(f"Dynamic Neuron Transformer - DAWN")
    print(f"=" * 70)
    print(f"Total parameters: {total_params:,}")
    print(f"  - Embeddings: {embedding_params:,}")
    print(f"  - Per layer: {layer_params:,}")
    print(f"  - Number of layers: {model.n_layers}")
    print(f"  - Neurons per layer: {model.layers[0].attention.neuron_pool.n_neurons}")
    print(f"  - Patterns per layer: {model.layers[0].ffn.pattern_keys.shape[0]}")
    print(f"  - Memory footprint: ~{total_params * 4 / 1e9:.2f} GB (FP32)")
    print(f"=" * 70)

    return model


def example_usage():
    """사용 예시"""
    print("=" * 70)
    print("DAWN: Dynamic Architecture With Neurons")
    print("Dynamic Neuron Transformer Implementation")
    print("=" * 70)
    print("\nArchitecture:")
    print("  - NeuronPool: QK-based retrieval for dynamic neuron selection")
    print("  - PatternFFN: Pattern-based dynamic feed-forward networks")
    print("  - NeuronAttention: Neuron-guided multi-head attention")
    print("=" * 70)

    model = create_model(vocab_size=10000)

    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    trainer = DAWNTrainer(model, optimizer, device=device)

    # 테스트
    batch_size = 8
    seq_len = 128
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    targets = torch.randint(0, 10000, (batch_size, seq_len))

    if device == 'cuda':
        input_ids = input_ids.cuda()
        targets = targets.cuda()

    loss = trainer.train_step(input_ids, targets)
    print(f"\nTest Loss: {loss:.4f}")

    analysis = trainer.analyze_activations(input_ids[:1])
    print("\nNeuron Usage Analysis:")
    for layer_name, stats in list(analysis.items())[:2]:
        print(f"  {layer_name}:")
        print(f"    - Unique neurons used: {stats['unique_neurons_used']}/{stats['total_neurons']}")
        print(f"    - Usage ratio: {stats['usage_ratio']:.2%}")

    print("\n" + "=" * 70)
    print("Ready for training!")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()
