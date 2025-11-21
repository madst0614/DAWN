import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BalancedInputNeurons(nn.Module):
    """
    균형잡힌 기본 특징 감지 (V1)
    공유 변환 + 뉴런별 작은 특화

    파라미터: ~1.3M (기존 66M의 1/50)
    표현력: 충분히 유지
    """
    def __init__(self, hidden_dim, num_input_neurons, adapt_rank=16):
        super().__init__()
        self.num_neurons = num_input_neurons
        self.hidden_dim = hidden_dim
        self.adapt_rank = adapt_rank

        # 패턴 템플릿
        self.patterns = nn.Parameter(
            torch.randn(num_input_neurons, hidden_dim)
        )

        # 공유 변환 (모든 뉴런이 같이 사용)
        self.shared_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # 각 뉴런별 작은 특화 (Low-rank adaptation)
        # 각 뉴런이 자기만의 약간의 조정 추가
        self.neuron_adapt_down = nn.Parameter(
            torch.randn(num_input_neurons, hidden_dim, adapt_rank) * 0.02
        )
        self.neuron_adapt_up = nn.Parameter(
            torch.randn(num_input_neurons, adapt_rank, hidden_dim) * 0.02
        )

    def forward(self, x):
        B, S, H = x.shape
        N = self.num_neurons

        # 1. 패턴 매칭 → 활성화
        x_norm = F.normalize(x, dim=-1)
        patterns_norm = F.normalize(self.patterns, dim=-1)
        pattern_matches = torch.matmul(x_norm, patterns_norm.t())
        activations = torch.sigmoid(pattern_matches)  # [B, S, N]

        # 2. 공유 변환 (기본 feature 추출)
        shared = self.shared_transform(x)  # [B, S, H]

        # 3. 뉴런별 작은 특화
        # [B, S, H] @ [N, H, r] → [B, S, N, r]
        down = torch.einsum('bsh,nhr->bsnr', shared, self.neuron_adapt_down)
        # [B, S, N, r] @ [N, r, H] → [B, S, N, H]
        specialized = torch.einsum('bsnr,nrh->bsnh', down, self.neuron_adapt_up)

        # 공유 + 특화
        # [B, S, H] + [B, S, N, H]
        features = shared.unsqueeze(2) + specialized

        # 4. 활성화 적용
        activated_features = features * activations.unsqueeze(-1)
        intermediate = activated_features.sum(dim=2)

        return intermediate, activations


class LateralConnections(nn.Module):
    """
    특징 간 관계 계산 (V2/V4, 통사 처리)
    활성화 패턴들이 서로 소통
    """
    def __init__(self, num_input_neurons):
        super().__init__()
        self.num_input_neurons = num_input_neurons

        # 활성화 패턴 간 attention (경량: 1 head)
        self.pattern_attention = nn.MultiheadAttention(
            embed_dim=num_input_neurons,
            num_heads=1,
            batch_first=True
        )

    def forward(self, activations):
        """
        activations: [B, S, N_in] - InputNeurons의 활성화 패턴

        Returns:
            relational_acts: [B, S, N_in] - 관계 정보가 전파된 활성화
            attn_weights: [B, S, S] - attention weights
        """
        B, S, N_in = activations.shape

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(S, S, device=activations.device) * float('-inf'),
            diagonal=1
        )

        # 활성화 패턴 간 소통
        relational_acts, attn_weights = self.pattern_attention(
            activations,
            activations,
            activations,
            attn_mask=causal_mask
        )

        return relational_acts, attn_weights


class LowRankProcessNeurons(nn.Module):
    """
    전역 패턴 통합 (IT, 의미 통합) - Low-Rank 최적화

    각 ProcessNeuron이 독립적 변환을 유지하면서
    Low-Rank 분해로 메모리와 계산량 절약
    """
    def __init__(self, hidden_dim, num_input_neurons, num_process_neurons, rank=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_process_neurons = num_process_neurons
        self.rank = rank

        # 2D 패턴 감지기
        self.pattern_detector = nn.Conv2d(
            in_channels=1,
            out_channels=num_process_neurons,
            kernel_size=(5, num_input_neurons),
            padding=(2, 0)
        )

        # Low-Rank 분해
        self.down_proj = nn.Parameter(
            torch.randn(num_process_neurons, hidden_dim, rank) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(num_process_neurons, rank, hidden_dim) * 0.02
        )

    def forward(self, intermediate, enriched_activations):
        """
        intermediate: [B, S, H]
        enriched_activations: [B, S, N_in]
        """
        B, S, N_in = enriched_activations.shape
        N_proc = self.num_process_neurons
        H = self.hidden_dim

        # 1. 2D 패턴 감지
        act_map = enriched_activations.unsqueeze(1)
        pattern_responses = self.pattern_detector(act_map)
        process_activations = torch.sigmoid(
            pattern_responses.squeeze(-1).transpose(1, 2)
        )

        # 2. Low-Rank 변환
        down = torch.einsum('bsh,nhr->bsnr', intermediate, self.down_proj)
        transformed = torch.einsum('bsnr,nrh->bsnh', down, self.up_proj)

        # 3. 활성화 가중치 적용
        weighted = transformed * process_activations.unsqueeze(-1)
        output = weighted.sum(dim=2)

        return output, process_activations


class DAWNLayer(nn.Module):
    """
    단일 DAWN 레이어 (균형잡힌 버전)

    1. BalancedInputNeurons: 기본 특징 + 약간의 특화
    2. LateralConnections: 특징 간 관계
    3. LowRankProcessNeurons: 전역 패턴 통합
    """
    def __init__(self, hidden_dim, num_input_neurons=64, num_process_neurons=128,
                 adapt_rank=16, process_rank=128):
        super().__init__()

        # Stage 1: 균형잡힌 기본 특징 감지
        self.input_neurons = BalancedInputNeurons(
            hidden_dim, num_input_neurons, adapt_rank=adapt_rank
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Stage 2: Lateral connections
        self.lateral_connections = LateralConnections(num_input_neurons)

        # Stage 3: 전역 패턴 통합
        self.process_neurons = LowRankProcessNeurons(
            hidden_dim, num_input_neurons, num_process_neurons, rank=process_rank
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Stage 1: 기본 특징 감지
        intermediate, input_acts = self.input_neurons(x)
        x = self.norm1(x + intermediate)

        # Stage 2: Lateral connections
        relational_acts, attn_weights = self.lateral_connections(input_acts)
        enriched_acts = input_acts + relational_acts

        # Stage 3: 전역 패턴 통합
        output, process_acts = self.process_neurons(x, enriched_acts)
        x = self.norm2(x + output)

        return x, {
            'input_activations': input_acts,
            'relational_activations': relational_acts,
            'enriched_activations': enriched_acts,
            'process_activations': process_acts,
            'attention_weights': attn_weights
        }


class DAWN(nn.Module):
    """
    DAWN (Dynamic Architecture With Neurons)
    균형잡힌 최적화 버전

    파라미터 분배:
    - InputNeurons: ~1.3M per layer (경량)
    - ProcessNeurons: ~17M per layer (중량)
    - 총: ~142M (기존 542M의 1/4)

    철학:
    - InputNeurons: 기본 특징 + 약간의 특화
    - ProcessNeurons: 복잡한 관계와 패턴
    - 생물학적으로 타당한 균형
    """
    def __init__(
        self,
        vocab_size,
        hidden_dim=512,
        num_layers=6,
        num_input_neurons=64,
        num_process_neurons=128,
        adapt_rank=16,
        process_rank=128,
        max_seq_len=2048,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 임베딩
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # DAWN 레이어들
        self.layers = nn.ModuleList([
            DAWNLayer(
                hidden_dim,
                num_input_neurons,
                num_process_neurons,
                adapt_rank=adapt_rank,
                process_rank=process_rank
            )
            for _ in range(num_layers)
        ])

        # 출력
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

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
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, return_activations=False):
        """
        순전파

        Args:
            input_ids: [B, S]
            return_activations: bool

        Returns:
            logits: [B, S, vocab_size]
            activations: (선택적)
        """
        B, S = input_ids.shape

        # 임베딩
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb
        x = self.embedding_dropout(x)

        # 레이어별 처리
        all_activations = []
        for layer in self.layers:
            x, activations = layer(x)
            if return_activations:
                all_activations.append(activations)

        # 출력
        x = self.output_norm(x)
        logits = self.output_projection(x)

        if return_activations:
            return logits, all_activations
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


# ========== 학습 유틸리티 ==========

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
        self.model.eval()

        with torch.no_grad():
            _, all_activations = self.model(input_ids, return_activations=True)

        analysis = {}
        for layer_idx, acts in enumerate(all_activations):
            input_acts = acts['input_activations']
            relational_acts = acts['relational_activations']
            enriched_acts = acts['enriched_activations']
            process_acts = acts['process_activations']

            analysis[f'layer_{layer_idx}'] = {
                'input_mean': input_acts.mean().item(),
                'input_std': input_acts.std().item(),
                'input_sparsity': (input_acts < 0.1).float().mean().item(),
                'relational_mean': relational_acts.mean().item(),
                'relational_std': relational_acts.std().item(),
                'enriched_mean': enriched_acts.mean().item(),
                'enriched_std': enriched_acts.std().item(),
                'process_mean': process_acts.mean().item(),
                'process_std': process_acts.std().item(),
                'process_sparsity': (process_acts < 0.1).float().mean().item(),
            }

        return analysis


# ========== 호환성 ==========

# Backward compatibility aliases
InputNeurons = BalancedInputNeurons
ProcessNeurons = LowRankProcessNeurons

DAWNLanguageModel = DAWN

def _from_config(cls, config, vocab_size):
    """Config dict로부터 모델 생성"""
    model_cfg = config.get('model', {})
    return cls(
        vocab_size=vocab_size,
        hidden_dim=model_cfg.get('d_model', 512),
        num_layers=model_cfg.get('n_layers', 6),
        num_input_neurons=model_cfg.get('n_input', 64),
        num_process_neurons=model_cfg.get('n_process', 128),
        adapt_rank=model_cfg.get('adapt_rank', 16),
        process_rank=model_cfg.get('process_rank', 128),
        max_seq_len=model_cfg.get('max_seq_len', 2048),
        dropout=model_cfg.get('dropout', 0.1)
    )

DAWN.from_config = classmethod(_from_config)


def create_model(vocab_size=50000):
    """DAWN 모델 생성"""
    model = DAWN(
        vocab_size=vocab_size,
        hidden_dim=512,
        num_layers=6,
        num_input_neurons=64,
        num_process_neurons=128,
        adapt_rank=16,
        process_rank=128,
        max_seq_len=2048,
        dropout=0.1
    )

    # 파라미터 분석
    total_params = sum(p.numel() for p in model.parameters())

    # 레이어별 파라미터
    input_params = sum(p.numel() for p in model.layers[0].input_neurons.parameters())
    process_params = sum(p.numel() for p in model.layers[0].process_neurons.parameters())

    print(f"Total parameters: {total_params:,}")
    print(f"  - Embeddings + Output: {(total_params - 6*(input_params + process_params)):,}")
    print(f"  - InputNeurons per layer: {input_params:,}")
    print(f"  - ProcessNeurons per layer: {process_params:,}")
    print(f"  - Memory footprint: ~{total_params * 4 / 1e9:.2f} GB")

    return model


def example_usage():
    """사용 예시"""
    print("="*70)
    print("DAWN: Dynamic Architecture With Neurons (BALANCED)")
    print("="*70)
    print("\nArchitecture Philosophy:")
    print("  - InputNeurons: Lightweight with specialization")
    print("  - ProcessNeurons: Heavyweight for complex patterns")
    print("  - Biologically inspired balance")
    print("\nOptimizations:")
    print("  - Shared + Low-rank adaptation (InputNeurons)")
    print("  - Low-rank factorization (ProcessNeurons)")
    print("  - 1-head attention (LateralConnections)")
    print("="*70)

    model = create_model(vocab_size=10000)
    model = model.cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    trainer = DAWNTrainer(model, optimizer)

    # 테스트
    batch_size = 8
    seq_len = 128
    input_ids = torch.randint(0, 10000, (batch_size, seq_len)).cuda()
    targets = torch.randint(0, 10000, (batch_size, seq_len)).cuda()

    loss = trainer.train_step(input_ids, targets)
    print(f"\nTest Loss: {loss:.4f}")

    analysis = trainer.analyze_activations(input_ids[:1])
    print("\nActivation Analysis (Layer 0):")
    for key, value in list(analysis['layer_0'].items())[:5]:
        print(f"  {key}: {value:.4f}")

    print("\n" + "="*70)
    print("Ready for training!")
    print("="*70)


if __name__ == "__main__":
    example_usage()
