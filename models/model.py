import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InputNeurons(nn.Module):
    """
    각 토큰이 자기 패턴에 맞는 input neuron을 선택적으로 활성화
    기본 feature detection 수행
    """
    def __init__(self, hidden_dim, num_input_neurons):
        super().__init__()

        self.num_neurons = num_input_neurons
        self.hidden_dim = hidden_dim

        # 모든 뉴런의 패턴을 하나의 행렬로
        self.patterns = nn.Parameter(
            torch.randn(num_input_neurons, hidden_dim)
        )

        # 모든 뉴런의 feature extractor를 하나로
        self.feature_extractors = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim * num_input_neurons)
        )

    def forward(self, x):
        # x: [B, S, H]
        B, S, H = x.shape
        N = self.num_neurons

        # 1. 모든 패턴 매칭 병렬 계산
        x_norm = F.normalize(x, dim=-1)  # [B, S, H]
        patterns_norm = F.normalize(self.patterns, dim=-1)  # [N, H]

        # 배치 행렬곱
        pattern_matches = torch.matmul(x_norm, patterns_norm.t())  # [B, S, N]
        activations = torch.sigmoid(pattern_matches)

        # 2. 모든 feature 추출 병렬
        all_features = self.feature_extractors(x)  # [B, S, H*N]
        all_features = all_features.view(B, S, N, H)  # [B, S, N, H]

        # 3. 활성화 적용
        activated_features = all_features * activations.unsqueeze(-1)  # [B, S, N, H]

        # 4. 중간 표현 (모든 input neuron의 기여 합산)
        intermediate = activated_features.sum(dim=2)  # [B, S, H]

        return intermediate, activations


class ProcessNeurons(nn.Module):
    """
    InputNeuron의 활성 패턴을 보고 선택적으로 활성화
    전체 시퀀스를 엮어서 토큰 풍부화
    """
    def __init__(self, hidden_dim, num_input_neurons, num_process_neurons):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_process_neurons = num_process_neurons

        # 각 ProcessNeuron이 반응하는 InputNeuron 활성 패턴
        self.pattern_templates = nn.Parameter(
            torch.randn(num_process_neurons, num_input_neurons)
        )

        # 각 ProcessNeuron의 변환 (Linear로 최적화!)
        self.neuron_transforms = nn.Linear(
            hidden_dim,
            hidden_dim * num_process_neurons,
            bias=False
        )

    def forward(self, intermediate, input_activations):
        # intermediate: [B, S, H] - InputNeurons의 출력
        # input_activations: [B, S, N_in] - InputNeurons의 활성 패턴

        B, S, H = intermediate.shape
        N_proc = self.num_process_neurons

        # 1. 패턴 매칭: 어떤 ProcessNeuron을 활성화할지
        act_norm = F.normalize(input_activations, dim=-1)
        template_norm = F.normalize(self.pattern_templates, dim=-1)
        pattern_matches = torch.matmul(act_norm, template_norm.t())  # [B, S, N_proc]
        process_activations = torch.sigmoid(pattern_matches)

        # 2. 각 ProcessNeuron이 보는 context 계산
        # 활성화된 토큰들의 정보만 모음
        weighted = intermediate.unsqueeze(2) * process_activations.unsqueeze(-1)  # [B, S, N_proc, H]

        # 시퀀스 통합 (각 ProcessNeuron별로)
        contexts = weighted.sum(dim=1)  # [B, N_proc, H]
        normalization = process_activations.sum(dim=1).unsqueeze(-1) + 1e-8  # [B, N_proc, 1]
        contexts = contexts / normalization  # [B, N_proc, H]

        # 3. 모든 ProcessNeuron 변환 병렬 실행 (Linear 사용으로 최적화!)
        all_transformed = self.neuron_transforms(contexts)  # [B, H*N_proc]
        transformed = all_transformed.view(B, N_proc, H)  # [B, N_proc, H]

        # 4. 각 토큰에 분배 (expand 없이 직접 broadcasting)
        distributed = transformed.unsqueeze(1) * process_activations.unsqueeze(-1)  # [B, 1, N_proc, H] * [B, S, N_proc, 1]

        # 5. 모든 ProcessNeuron 출력 통합
        combined = distributed.sum(dim=2)  # [B, S, H]

        return combined, process_activations


class DAWNLayer(nn.Module):
    """
    단일 DAWN 레이어
    InputNeurons → ProcessNeurons → 토큰 풍부화
    """
    def __init__(self, hidden_dim, num_input_neurons=64, num_process_neurons=128):
        super().__init__()

        # 1단계: 선택적 feature 추출
        self.input_neurons = InputNeurons(hidden_dim, num_input_neurons)

        # 2단계: 전체 맥락 보고 토큰 풍부화
        self.process_neurons = ProcessNeurons(
            hidden_dim,
            num_input_neurons,
            num_process_neurons
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, S, H]

        # InputNeurons: 기본 feature 추출 + 활성 패턴
        intermediate, input_acts = self.input_neurons(x)
        x = self.norm1(x + intermediate)

        # ProcessNeurons: 활성 패턴 보고 토큰들 엮어서 풍부화
        enriched, process_acts = self.process_neurons(x, input_acts)
        x = self.norm2(x + enriched)

        return x, {
            'input_activations': input_acts,
            'process_activations': process_acts
        }


class DAWN(nn.Module):
    """
    전체 DAWN 모델
    """
    def __init__(
        self,
        vocab_size,
        hidden_dim=512,
        num_layers=6,
        num_input_neurons=64,
        num_process_neurons=128,
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
            DAWNLayer(hidden_dim, num_input_neurons, num_process_neurons)
            for _ in range(num_layers)
        ])

        # 출력
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # 가중치 초기화
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
        Args:
            input_ids: [B, S] - 토큰 ID
            return_activations: bool - 활성 패턴 반환 여부

        Returns:
            logits: [B, S, vocab_size]
            activations: (선택적) 각 레이어의 활성 패턴
        """
        B, S = input_ids.shape

        # 임베딩
        token_emb = self.token_embedding(input_ids)  # [B, S, H]

        # 위치 임베딩
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)  # [1, S]
        pos_emb = self.position_embedding(positions)  # [1, S, H]

        x = token_emb + pos_emb
        x = self.embedding_dropout(x)

        # 레이어별 처리 (점진적 풍부화)
        all_activations = []
        for layer in self.layers:
            x, activations = layer(x)
            if return_activations:
                all_activations.append(activations)

        # 최종 출력
        x = self.output_norm(x)
        logits = self.output_projection(x)  # [B, S, vocab_size]

        if return_activations:
            return logits, all_activations
        else:
            return logits

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        자동회귀 생성

        Args:
            input_ids: [B, S] - 초기 시퀀스
            max_new_tokens: int - 생성할 토큰 수
            temperature: float - 샘플링 온도
            top_k: int - top-k 샘플링

        Returns:
            generated: [B, S + max_new_tokens]
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 현재 시퀀스로 예측
                logits = self.forward(input_ids)  # [B, S, vocab]

                # 마지막 토큰의 logits
                next_token_logits = logits[:, -1, :] / temperature  # [B, vocab]

                # Top-k 필터링
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

                # 샘플링
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

                # 시퀀스에 추가
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ========== 학습 유틸리티 ==========

class DAWNTrainer:
    """
    DAWN 학습을 위한 헬퍼 클래스
    """
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_step(self, input_ids, targets):
        """
        단일 학습 스텝

        Args:
            input_ids: [B, S]
            targets: [B, S]

        Returns:
            loss: float
        """
        self.model.train()

        # Forward
        logits = self.model(input_ids)  # [B, S, vocab]

        # Loss 계산
        B, S, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B * S, V),
            targets.view(B * S),
            ignore_index=-100
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()

    def analyze_activations(self, input_ids):
        """
        활성 패턴 분석

        Args:
            input_ids: [B, S]

        Returns:
            analysis: dict - 각 레이어의 활성 통계
        """
        self.model.eval()

        with torch.no_grad():
            _, all_activations = self.model(input_ids, return_activations=True)

        analysis = {}
        for layer_idx, acts in enumerate(all_activations):
            input_acts = acts['input_activations']  # [B, S, N_in]
            process_acts = acts['process_activations']  # [B, S, N_proc]

            analysis[f'layer_{layer_idx}'] = {
                'input_mean': input_acts.mean().item(),
                'input_std': input_acts.std().item(),
                'input_sparsity': (input_acts < 0.1).float().mean().item(),
                'process_mean': process_acts.mean().item(),
                'process_std': process_acts.std().item(),
                'process_sparsity': (process_acts < 0.1).float().mean().item(),
            }

        return analysis


# ========== 사용 예시 ==========

def create_model(vocab_size=50000):
    """
    DAWN 모델 생성
    """
    model = DAWN(
        vocab_size=vocab_size,
        hidden_dim=512,
        num_layers=6,
        num_input_neurons=64,
        num_process_neurons=128,
        max_seq_len=2048,
        dropout=0.1
    )

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def example_usage():
    """
    사용 예시
    """
    # 모델 생성
    model = create_model(vocab_size=10000)
    model = model.cuda()

    # 옵티마이저
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # 트레이너
    trainer = DAWNTrainer(model, optimizer)

    # 더미 데이터
    batch_size = 8
    seq_len = 128

    input_ids = torch.randint(0, 10000, (batch_size, seq_len)).cuda()
    targets = torch.randint(0, 10000, (batch_size, seq_len)).cuda()

    # 학습 스텝
    loss = trainer.train_step(input_ids, targets)
    print(f"Loss: {loss:.4f}")

    # 활성 분석
    analysis = trainer.analyze_activations(input_ids[:1])
    print("\nActivation Analysis:")
    for layer, stats in analysis.items():
        print(f"\n{layer}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")

    # 생성
    prompt = torch.randint(0, 10000, (1, 10)).cuda()
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"\nGenerated shape: {generated.shape}")


# ========== 하위 호환성을 위한 별칭 ==========

# 기존 코드와의 호환성을 위해 별칭 제공
DAWNLanguageModel = DAWN

# from_config 메서드 추가
def _from_config(cls, config, vocab_size):
    """Create model from config dict."""
    model_cfg = config.get('model', {})
    return cls(
        vocab_size=vocab_size,
        hidden_dim=model_cfg.get('d_model', 512),
        num_layers=model_cfg.get('n_layers', 6),
        num_input_neurons=model_cfg.get('n_input', 64),
        num_process_neurons=model_cfg.get('n_process', 128),
        max_seq_len=model_cfg.get('max_seq_len', 2048),
        dropout=model_cfg.get('dropout', 0.1)
    )

DAWN.from_config = classmethod(_from_config)


if __name__ == "__main__":
    example_usage()
