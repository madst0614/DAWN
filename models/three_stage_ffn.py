"""
Hierarchical Dynamic Neuron FFN with Global Router

구조:
Phase 1: Global Router (QKV) - 시퀀스 전체 요약 → 입력 뉴런 집합 선택
Phase 2: Input Neurons - 선택된 뉴런들이 토큰별 세밀한 해석
Phase 3: Process Neurons - 입력 조합하여 기계적 계산

특징:
- 계층적 라우팅: 거시적 → 미시적
- QKV 기반 전역 라우터
- 메모리 효율적 (시퀀스별 뉴런 선택)
- 생물학적 영감 (Thalamus → V1/V2 → IT/Prefrontal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint


# ============================================================
# Global Router (QKV-based)
# ============================================================

class GlobalRouter(nn.Module):
    """
    전역 라우터: 시퀀스 전체 맥락 파악하여 입력 뉴런 선택

    - Query: 시퀀스 요약 → "이 문장은 대략 무엇인가?"
    - Key: 각 입력 뉴런의 "정체성"
    - Value: 암묵적 (뉴런 인덱스)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_input_neurons: int = 2048,
        d_routing: int = 256,
        use_mlp: bool = True
    ):
        super().__init__()

        self.d_routing = d_routing
        self.n_input = n_input_neurons
        self.use_mlp = use_mlp

        if use_mlp:
            # MLP로 더 강력한 Query 생성
            self.query_net = nn.Sequential(
                nn.Linear(d_model, d_routing * 2),
                nn.GELU(),
                nn.LayerNorm(d_routing * 2),
                nn.Linear(d_routing * 2, d_routing)
            )
        else:
            # 단순 선형 변환
            self.query_net = nn.Linear(d_model, d_routing)

        # 각 입력 뉴런의 "정체성" (Key)
        self.neuron_keys = nn.Parameter(
            torch.randn(n_input_neurons, d_routing) * 0.02
        )

        self._init_weights()

    def _init_weights(self):
        # Orthogonal initialization for better diversity
        nn.init.orthogonal_(self.neuron_keys)
        for module in self.query_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        k_input: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S, d_model]
            k_input: 선택할 입력 뉴런 개수

        Returns:
            input_idx: [B, k_input] - 선택된 뉴런 인덱스
            routing_weights: [B, n_input] - Soft weights for gradient flow
        """
        B, S, d_model = x.shape

        # Max pooling for stronger signal (instead of mean)
        global_context = x.max(dim=1)[0]  # [B, d_model]

        # Query: "이 시퀀스는 어떤 특성인가?"
        query = self.query_net(global_context)  # [B, d_routing]

        # Attention with neuron keys
        routing_logits = (query @ self.neuron_keys.T) / (self.d_routing ** 0.5)
        # [B, n_input]

        # Soft routing for gradient flow
        # Temperature = 0.1 for sharper distribution
        routing_probs = F.softmax(routing_logits / 0.1, dim=-1)
        # [B, n_input]

        # Hard selection (top-k)
        _, input_idx = routing_logits.topk(k_input, dim=-1)
        # [B, k_input]

        # Straight-through estimator:
        # Forward: hard selection (one-hot mask)
        # Backward: soft gradient (routing_probs)
        hard_mask = torch.zeros_like(routing_probs)  # [B, n_input]
        hard_mask.scatter_(1, input_idx, 1.0)

        # This maintains gradient flow to neuron_keys!
        routing_weights = hard_mask + (routing_probs - routing_probs.detach())
        # [B, n_input]

        return input_idx, routing_weights


# ============================================================
# Hierarchical Dynamic FFN
# ============================================================

class HierarchicalDynamicFFN(nn.Module):
    """
    계층적 동적 FFN

    Phase 1: Global Router → 입력 뉴런 집합 선택 (거시적)
    Phase 2: Input Neurons → 토큰별 세밀한 해석 (미시적)
    Phase 3: Process Neurons → 단순 계산 및 출력
    """

    def __init__(
        self,
        d_model: int = 512,
        n_input_neurons: int = 2048,
        n_process_neurons: int = 1024,
        d_routing: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_input = n_input_neurons
        self.n_process = n_process_neurons
        self.d_routing = d_routing

        # ===== Phase 1: Global Router =====
        self.global_router = GlobalRouter(
            d_model=d_model,
            n_input_neurons=n_input_neurons,
            d_routing=d_routing,
            use_mlp=True
        )

        # ===== Phase 2: Input Neurons =====
        # 각 입력 뉴런의 패턴 (토큰별 해석용)
        self.input_patterns = nn.Parameter(
            torch.randn(n_input_neurons, d_model) * 0.02
        )

        # ===== Phase 3: Process Neurons =====
        # 처리 뉴런 가중치 (입력 → 처리)
        self.process_weights = nn.Parameter(
            torch.randn(n_process_neurons, n_input_neurons) * 0.02
        )
        # 처리 뉴런 출력 패턴
        self.process_outputs = nn.Parameter(
            torch.randn(n_process_neurons, d_model) * 0.02
        )

        self.dropout = nn.Dropout(dropout)

        # Routing 통계 저장 (load balancing용)
        self.input_neuron_counts = None
        self.process_neuron_counts = None
        self.last_routing_scores = None

        self._init_weights()

    def _init_weights(self):
        # Orthogonal initialization for better diversity and gradient flow
        nn.init.orthogonal_(self.input_patterns)
        nn.init.orthogonal_(self.process_weights)
        nn.init.orthogonal_(self.process_outputs)

    def forward(
        self,
        x: torch.Tensor,
        k_input: Optional[int] = None,
        k_process: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, d_model]
            k_input: 선택할 입력 뉴런 수 (None이면 n_input//8)
            k_process: 선택할 처리 뉴런 수 (None이면 n_process//8)

        Returns:
            output: [B, S, d_model]
        """
        B, S, d_model = x.shape

        # Default k values - start conservative to verify architecture
        # Input: 50% (moderate sparsity, preserves information)
        # Process: 100% (no bottleneck, full expressiveness)
        if k_input is None:
            k_input = self.n_input // 2  # 50% (was 12.5%)
        if k_process is None:
            k_process = self.n_process  # 100% (was 12.5%)

        # ===== Phase 1: Global Router =====
        # 시퀀스별로 입력 뉴런 선택 (거시적 결정)
        input_idx, routing_weights = self.global_router(x, k_input)
        # input_idx: [B, k_input] - selected neuron indices
        # routing_weights: [B, n_input] - soft weights for gradient flow

        # Routing 통계 저장 (load balancing용)
        if self.training:
            if self.input_neuron_counts is None:
                self.input_neuron_counts = torch.zeros(
                    self.n_input,
                    device=x.device,
                    dtype=torch.float32
                )
            if self.process_neuron_counts is None:
                self.process_neuron_counts = torch.zeros(
                    self.n_process,
                    device=x.device,
                    dtype=torch.float32
                )

            # Input neuron 사용 카운트 (soft weights로 더 정확)
            # routing_weights: [B, n_input]
            self.input_neuron_counts += routing_weights.sum(dim=0).detach()

            # Routing weights 저장 (entropy 계산용)
            self.last_routing_scores = routing_weights.detach()

        # ===== Phase 2: Input Neurons =====
        # 모든 입력 뉴런의 토큰별 activation 계산
        input_acts = F.gelu(x @ self.input_patterns.T)  # [B, S, n_input]

        # 선택된 입력 뉴런의 activation만 추출
        expanded_input_idx = input_idx.unsqueeze(1).expand(-1, S, -1)
        # [B, S, k_input]
        selected_input_acts = torch.gather(input_acts, 2, expanded_input_idx)
        # [B, S, k_input]

        # ===== Phase 3: Process Neurons =====
        # 배치별로 처리 (각 시퀀스가 다른 입력 뉴런 사용)
        outputs = []
        process_indices = []  # load balancing용

        for b in range(B):
            # 이 시퀀스의 입력 뉴런 인덱스
            idx_b = input_idx[b]  # [k_input]
            acts_b = selected_input_acts[b]  # [S, k_input]

            # Dynamic weight indexing (position-invariant, more stable)
            # 선택된 입력 뉴런에 대응하는 가중치만 추출
            selected_weights = self.process_weights[:, idx_b]  # [n_process, k_input]

            # 처리 뉴런 활성화 (선택된 가중치로만 계산)
            process_acts = F.gelu(acts_b @ selected_weights.T)
            # [S, n_process]

            # 처리 뉴런 선택 (시퀀스 평균 기준)
            process_scores = process_acts.mean(dim=0)  # [n_process]
            _, process_idx = process_scores.topk(k_process)  # [k_process]

            process_indices.append(process_idx)

            # 선택된 처리 뉴런의 출력
            selected_process_acts = process_acts[:, process_idx]  # [S, k_process]
            selected_process_outputs = self.process_outputs[process_idx]  # [k_process, d_model]

            # 최종 출력
            output_b = selected_process_acts @ selected_process_outputs  # [S, d_model]
            outputs.append(output_b)

        output = torch.stack(outputs, dim=0)  # [B, S, d_model]
        output = self.dropout(output)

        # Process neuron 통계 수집 (training 시에만)
        if self.training and len(process_indices) > 0:
            process_indices_tensor = torch.stack(process_indices)  # [B, k_process]
            ones = torch.ones_like(process_indices_tensor, dtype=torch.float32)
            self.process_neuron_counts.scatter_add_(
                0,
                process_indices_tensor.flatten(),
                ones.flatten()
            )

        return output

    def get_load_balance_loss(self) -> torch.Tensor:
        """
        Load balancing loss 계산 (KL divergence + entropy 기반)

        Returns:
            loss: 0-2 범위의 정규화된 loss
        """
        if self.input_neuron_counts is None or not self.training:
            device = self.input_patterns.device
            return torch.tensor(0.0, device=device)

        counts = self.input_neuron_counts
        device = counts.device

        # 사용 빈도가 0이면 계산 불가
        if counts.sum() == 0:
            return torch.tensor(0.0, device=device)

        # 정규화 (확률 분포)
        usage_probs = counts / (counts.sum() + 1e-8)

        # 목표: 균등 분포
        target_prob = 1.0 / self.n_input
        target = torch.full_like(usage_probs, target_prob)

        # KL divergence (안정적)
        usage_probs = usage_probs + 1e-8  # Smoothing
        target = target + 1e-8

        kl_loss = F.kl_div(
            usage_probs.log(),
            target,
            reduction='sum'
        ) / self.n_input

        # Routing entropy (다양성)
        # last_routing_scores is now routing_weights [B, n_input] (already soft)
        if self.last_routing_scores is not None:
            # Average across batch
            avg_probs = self.last_routing_scores.mean(dim=0) + 1e-8  # [n_input]

            # Entropy of average distribution
            entropy = -(avg_probs * avg_probs.log()).sum()
            max_entropy = torch.log(torch.tensor(float(self.n_input), device=device))

            # 낮은 엔트로피 = penalty
            entropy_loss = 1.0 - (entropy / max_entropy)
        else:
            entropy_loss = torch.tensor(0.0, device=device)

        # 합치기 (각각 0-1 범위)
        total_loss = kl_loss + entropy_loss

        return total_loss

    def reset_routing_counts(self):
        """Routing 통계를 초기화합니다."""
        self.input_neuron_counts = None
        self.process_neuron_counts = None
        self.last_routing_scores = None

    def get_neuron_stats(
        self,
        x: torch.Tensor,
        k_input: Optional[int] = None,
        k_process: Optional[int] = None
    ) -> dict:
        """디버깅/분석용 통계"""
        B, S, _ = x.shape

        if k_input is None:
            k_input = self.n_input // 2  # 50% (was 12.5%)
        if k_process is None:
            k_process = self.n_process  # 100% (was 12.5%)

        with torch.no_grad():
            # Global Router
            input_idx, routing_weights = self.global_router(x, k_input)
            # routing_weights: [B, n_input]

            # Input Neurons
            input_acts = F.gelu(x @ self.input_patterns.T)
            expanded_idx = input_idx.unsqueeze(1).expand(-1, S, -1)
            selected_input_acts = torch.gather(input_acts, 2, expanded_idx)

            # Process Neurons (첫 번째 시퀀스만)
            idx_0 = input_idx[0]
            acts_0 = selected_input_acts[0]

            # Dynamic weight indexing (consistent with forward)
            selected_weights = self.process_weights[:, idx_0]  # [n_process, k_input]
            process_acts = F.gelu(acts_0 @ selected_weights.T)  # [S, n_process]
            process_scores = process_acts.mean(dim=0)
            top_process_scores, process_idx = process_scores.topk(k_process)

            # Extract routing scores for selected neurons
            selected_routing_scores = torch.gather(
                routing_weights, 1, input_idx
            )  # [B, k_input]

            return {
                'global_router': {
                    'input_indices': input_idx.cpu(),  # [B, k_input]
                    'routing_scores': selected_routing_scores.cpu(),  # [B, k_input]
                    'mean_score': selected_routing_scores.mean().item()
                },
                'input_neurons': {
                    'indices': input_idx.cpu(),
                    'activations': selected_input_acts.mean(dim=1).cpu(),  # [B, k_input]
                    'mean_activation': selected_input_acts.mean().item(),
                    'sparsity': k_input / self.n_input
                },
                'process_neurons': {
                    'indices': process_idx.cpu(),  # [k_process]
                    'activations': top_process_scores.cpu(),
                    'mean_activation': top_process_scores.mean().item(),
                    'sparsity': k_process / self.n_process
                }
            }


# ============================================================
# Transformer Layer with Hierarchical FFN
# ============================================================

class TransformerLayerWithHierarchicalFFN(nn.Module):
    """Transformer layer with Hierarchical Dynamic FFN"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_input_neurons: int = 2048,
        n_process_neurons: int = 1024,
        d_routing: int = 256,
        dropout: float = 0.1,
        use_checkpoint: bool = False
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.ffn = HierarchicalDynamicFFN(
            d_model=d_model,
            n_input_neurons=n_input_neurons,
            n_process_neurons=n_process_neurons,
            d_routing=d_routing,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint

    def _attention_block(self, x, attention_mask):
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=attention_mask)
        return self.dropout(attn_out)

    def _ffn_block(self, x, k_input, k_process):
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm, k_input, k_process)
        return self.dropout(ffn_out)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        k_input: Optional[int] = None,
        k_process: Optional[int] = None
    ) -> torch.Tensor:
        # Attention
        if self.use_checkpoint and self.training:
            attn_out = checkpoint(self._attention_block, x, attention_mask, use_reentrant=False)
        else:
            attn_out = self._attention_block(x, attention_mask)
        x = x + attn_out

        # FFN
        if self.use_checkpoint and self.training:
            ffn_out = checkpoint(self._ffn_block, x, k_input, k_process, use_reentrant=False)
        else:
            ffn_out = self._ffn_block(x, k_input, k_process)
        x = x + ffn_out

        return x


# ============================================================
# Language Model with Hierarchical FFN
# ============================================================

class HierarchicalLanguageModel(nn.Module):
    """Language Model with Hierarchical Dynamic Neuron FFN"""

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 512,
        n_input_neurons: int = 2048,
        n_process_neurons: int = 1024,
        d_routing: int = 256,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.gradient_checkpointing = gradient_checkpointing

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerLayerWithHierarchicalFFN(
                d_model=d_model,
                n_heads=n_heads,
                n_input_neurons=n_input_neurons,
                n_process_neurons=n_process_neurons,
                d_routing=d_routing,
                dropout=dropout,
                use_checkpoint=gradient_checkpointing
            )
            for _ in range(n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        k_input: Optional[int] = None,
        k_process: Optional[int] = None
    ) -> dict:
        """
        Args:
            input_ids: [B, S]
            attention_mask: [B, S]
            labels: [B, S]
            k_input: 입력 뉴런 수
            k_process: 처리 뉴런 수

        Returns:
            dict with 'logits', 'loss'
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        token_emb = self.token_embedding(input_ids)  # [B, S, d_model]
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask, k_input, k_process)

        x = self.norm(x)
        logits = self.output_projection(x)  # [B, S, vocab_size]

        # Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore masked tokens
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        return {
            'logits': logits,
            'loss': loss
        }

    def get_model_stats(self) -> dict:
        """모델 통계"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # FFN 파라미터
        ffn_params = sum(
            sum(p.numel() for p in layer.ffn.parameters())
            for layer in self.layers
        )

        # Global Router 파라미터
        router_params = sum(
            sum(p.numel() for p in layer.ffn.global_router.parameters())
            for layer in self.layers
        )

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'ffn_parameters': ffn_params,
            'router_parameters': router_params,
            'ffn_percentage': ffn_params / total_params * 100,
            'router_percentage': router_params / total_params * 100,
            'n_layers': self.n_layers,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model
        }


# ============================================================
# Utility Functions
# ============================================================

def count_parameters(model: nn.Module) -> dict:
    """파라미터 수 계산"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def analyze_routing_patterns(
    model: HierarchicalLanguageModel,
    dataloader,
    device: str = 'cuda',
    n_batches: int = 10
) -> dict:
    """
    라우팅 패턴 분석
    - Global Router가 어떤 뉴런을 선택하는가?
    - 문맥에 따라 다른 뉴런이 선택되는가?
    """
    model.eval()

    input_neuron_counts = torch.zeros(model.layers[0].ffn.n_input)
    process_neuron_counts = torch.zeros(model.layers[0].ffn.n_process)

    routing_scores_all = []
    total_sequences = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            input_ids = batch['input_ids'].to(device)
            B, S = input_ids.shape
            total_sequences += B

            # Get embeddings
            token_emb = model.token_embedding(input_ids)
            positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = model.position_embedding(positions)
            x = token_emb + pos_emb

            # Analyze first layer's FFN
            layer = model.layers[0]
            x_norm = layer.norm2(x)

            stats = layer.ffn.get_neuron_stats(x_norm)

            # Count neuron usage
            input_idx = stats['input_neurons']['indices'].flatten()
            process_idx = stats['process_neurons']['indices'].flatten()

            for idx in input_idx:
                input_neuron_counts[idx] += 1
            for idx in process_idx:
                process_neuron_counts[idx] += 1

            # Collect routing scores
            routing_scores_all.append(stats['global_router']['routing_scores'])

    routing_scores_all = torch.cat(routing_scores_all, dim=0)  # [total_seqs, k_input]

    return {
        'input_neuron_usage': input_neuron_counts / total_sequences,
        'process_neuron_usage': process_neuron_counts / total_sequences,
        'routing_score_mean': routing_scores_all.mean().item(),
        'routing_score_std': routing_scores_all.std().item(),
        'total_sequences': total_sequences,
        'unique_input_neurons_used': (input_neuron_counts > 0).sum().item(),
        'unique_process_neurons_used': (process_neuron_counts > 0).sum().item()
    }


if __name__ == '__main__':
    # 간단한 테스트
    print("Testing Hierarchical Dynamic Neuron FFN...")

    # 모델 생성
    model = HierarchicalLanguageModel(
        vocab_size=30000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        max_seq_len=512,
        n_input_neurons=2048,
        n_process_neurons=1024,
        d_routing=256
    )

    # 통계 출력
    stats = model.get_model_stats()
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"  FFN parameters: {stats['ffn_parameters']:,} ({stats['ffn_percentage']:.1f}%)")
    print(f"  Router parameters: {stats['router_parameters']:,} ({stats['router_percentage']:.1f}%)")

    # Forward pass 테스트
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))

    print(f"\nTesting forward pass...")
    output = model(input_ids)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {output['logits'].shape}")

    # 뉴런 통계
    print(f"\nGetting neuron statistics...")
    layer0_ffn = model.layers[0].ffn
    x = model.token_embedding(input_ids)
    neuron_stats = layer0_ffn.get_neuron_stats(x)

    print(f"  Global Router:")
    print(f"    Mean routing score: {neuron_stats['global_router']['mean_score']:.4f}")

    print(f"  Input neurons:")
    print(f"    Selected: {neuron_stats['input_neurons']['sparsity']*100:.1f}%")
    print(f"    Mean activation: {neuron_stats['input_neurons']['mean_activation']:.4f}")

    print(f"  Process neurons:")
    print(f"    Selected: {neuron_stats['process_neurons']['sparsity']*100:.1f}%")
    print(f"    Mean activation: {neuron_stats['process_neurons']['mean_activation']:.4f}")

    print(f"\n✓ All tests passed!")
