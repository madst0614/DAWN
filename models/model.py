import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelationalInputNeurons(nn.Module):
    """
    각 토큰 → 활성화 패턴 변환 (관계 정보 포함)
    [B, S, H] → [B, S, N] 관계 반영된 활성화 지형
    """
    def __init__(self, hidden_dim, num_input_neurons):
        super().__init__()
        self.num_neurons = num_input_neurons
        self.hidden_dim = hidden_dim

        # 패턴 템플릿
        self.patterns = nn.Parameter(
            torch.randn(num_input_neurons, hidden_dim)
        )

        # Feature extractor
        self.feature_extractors = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim * num_input_neurons)
        )

        # 관계 계산용 경량 Attention
        self.relation_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            batch_first=True
        )

    def forward(self, x):
        B, S, H = x.shape
        N = self.num_neurons

        # 1. 기본 활성화 (의미 패턴)
        x_norm = F.normalize(x, dim=-1)
        patterns_norm = F.normalize(self.patterns, dim=-1)
        pattern_matches = torch.matmul(x_norm, patterns_norm.t())
        base_activations = torch.sigmoid(pattern_matches)  # [B, S, N]

        # 2. 관계 계산 (Causal Attention)
        # 미래 토큰을 보지 않도록 causal mask
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device) * float('-inf'),
            diagonal=1
        )
        _, attn_weights = self.relation_attn(x, x, x, attn_mask=causal_mask)
        # attn_weights: [B, S, S] - 토큰 간 관계 행렬

        # 3. 관계를 통한 활성화 전파
        # 각 토큰이 관련있는 토큰들의 활성화를 받아옴
        relational_activations = torch.matmul(attn_weights, base_activations)
        # [B, S, S] @ [B, S, N] = [B, S, N]

        # 4. 기본 활성화 + 관계 활성화
        combined_activations = base_activations + 0.5 * relational_activations

        # 5. Feature 추출 (관계 반영된 활성화로)
        all_features = self.feature_extractors(x)
        all_features = all_features.view(B, S, N, H)
        activated_features = all_features * combined_activations.unsqueeze(-1)
        intermediate = activated_features.sum(dim=2)

        return intermediate, combined_activations


class ProcessNeurons(nn.Module):
    """
    활성화 지형 → 패턴 인식 → 토큰 풍부화
    2D Conv로 지형의 패턴을 병렬 감지
    """
    def __init__(self, hidden_dim, num_input_neurons, num_process_neurons):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_process_neurons = num_process_neurons

        # 2D 패턴 감지기
        # 활성화 지형 [S, N_in]에서 패턴 찾기
        self.pattern_detector = nn.Conv2d(
            in_channels=1,
            out_channels=num_process_neurons,
            kernel_size=(5, num_input_neurons),  # 5토큰 윈도우, 모든 뉴런
            padding=(2, 0),  # 시퀀스 길이 유지
            bias=True
        )

        # 각 ProcessNeuron의 변환
        self.neuron_transforms = nn.Parameter(
            torch.randn(num_process_neurons, hidden_dim, hidden_dim) * 0.02
        )

    def forward(self, intermediate, input_activations):
        B, S, N_in = input_activations.shape
        N_proc = self.num_process_neurons
        H = self.hidden_dim

        # 1. 활성화 지형을 2D 이미지로
        # [B, S, N_in] → [B, 1, S, N_in]
        act_map = input_activations.unsqueeze(1)

        # 2. 2D Conv로 패턴 감지 (완전 병렬!)
        # [B, 1, S, N_in] → [B, N_proc, S, 1]
        pattern_responses = self.pattern_detector(act_map)

        # [B, N_proc, S, 1] → [B, S, N_proc]
        process_activations = torch.sigmoid(pattern_responses.squeeze(-1).transpose(1, 2))

        # 3. 각 ProcessNeuron의 변환 적용
        # intermediate: [B, S, H]
        # 각 위치에서 활성화된 ProcessNeuron들의 변환을 적용

        # [B, S, H] → [B, S, 1, H] → [B, S, N_proc, H]
        x_expanded = intermediate.unsqueeze(2).expand(-1, -1, N_proc, -1)

        # [B, S, N_proc, H] @ [N_proc, H, H] → [B, S, N_proc, H]
        transformed = torch.einsum('bsnh,nhd->bsnd', x_expanded, self.neuron_transforms)

        # 4. 활성화 가중치 적용
        weighted = transformed * process_activations.unsqueeze(-1)

        # 5. 합산
        output = weighted.sum(dim=2)  # [B, S, H]

        return output, process_activations


class DAWNLayer(nn.Module):
    """
    RelationalInputNeurons → ProcessNeurons
    관계 반영 활성화 지형 생성 → 지형 패턴 인식
    """
    def __init__(self, hidden_dim, num_input_neurons=64, num_process_neurons=128):
        super().__init__()

        # 관계 정보를 포함하는 InputNeurons
        self.input_neurons = RelationalInputNeurons(hidden_dim, num_input_neurons)

        self.process_neurons = ProcessNeurons(
            hidden_dim,
            num_input_neurons,
            num_process_neurons
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # 1. 관계 반영된 활성화 지형 생성
        intermediate, input_acts = self.input_neurons(x)
        x = self.norm1(x + intermediate)

        # 2. 지형 패턴 인식 & 토큰 풍부화
        enriched, process_acts = self.process_neurons(x, input_acts)
        x = self.norm2(x + enriched)

        return x, {
            'input_activations': input_acts,
            'process_activations': process_acts
        }


class DAWN(nn.Module):
    """
    전체 DAWN 모델
    위치 + 의미 + 관계 정보를 활성화 지형으로 통합
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

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DAWNLayer(hidden_dim, num_input_neurons, num_process_neurons)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
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
        B, S = input_ids.shape

        # 임베딩 (위치 + 의미)
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb
        x = self.embedding_dropout(x)

        # 레이어별 처리 (관계 정보 점진적 추가)
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
            process_acts = acts['process_activations']

            analysis[f'layer_{layer_idx}'] = {
                'input_mean': input_acts.mean().item(),
                'input_std': input_acts.std().item(),
                'input_sparsity': (input_acts < 0.1).float().mean().item(),
                'process_mean': process_acts.mean().item(),
                'process_std': process_acts.std().item(),
                'process_sparsity': (process_acts < 0.1).float().mean().item(),
            }

        return analysis


# ========== 호환성 ==========

DAWNLanguageModel = DAWN

def _from_config(cls, config, vocab_size):
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


def create_model(vocab_size=50000):
    model = DAWN(
        vocab_size=vocab_size,
        hidden_dim=512,
        num_layers=6,
        num_input_neurons=64,
        num_process_neurons=128,
        max_seq_len=2048,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    return model


if __name__ == "__main__":
    model = create_model(vocab_size=10000)
    print("\nDAWN model created successfully!")
    print("Architecture: Relational Activation Landscape → 2D Pattern Recognition")
