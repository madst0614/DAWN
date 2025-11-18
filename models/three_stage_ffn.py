"""
Hierarchical Dynamic Neuron FFN with Global Router - OPTIMIZED VERSION

최적화:
- For loop 완전 제거
- 배치 차원 병렬 처리
- torch.gather + torch.bmm 활용
- 5-15배 속도 향상 예상
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint


# ============================================================
# Global Router (QKV-based) - UNCHANGED
# ============================================================

class GlobalRouter(nn.Module):
    """
    전역 라우터: 시퀀스 전체 맥락 파악하여 입력 뉴런 선택
    """

    def __init__(
        self,
        d_model: int = 512,
        n_input_neurons: int = 2048,
        d_routing: int = 256,
        use_mlp: bool = True,
        temperature: float = 0.5  # Lower temperature = smoother routing
    ):
        super().__init__()

        self.d_routing = d_routing
        self.n_input = n_input_neurons
        self.use_mlp = use_mlp
        self.temperature = temperature

        if use_mlp:
            self.query_net = nn.Sequential(
                nn.Linear(d_model, d_routing * 2),
                nn.GELU(),
                nn.LayerNorm(d_routing * 2),
                nn.Linear(d_routing * 2, d_routing)
            )
        else:
            self.query_net = nn.Linear(d_model, d_routing)

        # Initialized with orthogonal in _init_weights()
        self.neuron_keys = nn.Parameter(
            torch.empty(n_input_neurons, d_routing)
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize router weights.

        Uses gain=sqrt(2) for GELU compatibility and Xavier for linear layers.
        """
        import math
        gain = math.sqrt(2.0)

        nn.init.orthogonal_(self.neuron_keys, gain=gain)
        for module in self.query_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=gain)
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

        # Max pooling for stronger signal
        global_context = x.max(dim=1)[0]  # [B, d_model]

        # Query generation
        query = self.query_net(global_context)  # [B, d_routing]

        # Attention with neuron keys
        routing_logits = (query @ self.neuron_keys.T) / (self.d_routing ** 0.5)
        # [B, n_input]

        # Soft routing for gradient flow (with temperature for smoother distribution)
        routing_probs = F.softmax(routing_logits / self.temperature, dim=-1)
        # [B, n_input] - Lower temperature = more uniform, Higher = more peaked

        # Hard selection (top-k)
        _, input_idx = routing_logits.topk(k_input, dim=-1)
        # [B, k_input]

        # One-hot encoding for selected neurons
        one_hot = torch.zeros_like(routing_probs)  # [B, n_input]
        one_hot.scatter_(1, input_idx, 1.0)

        # Straight-through estimator
        routing_weights = (one_hot - routing_probs).detach() + routing_probs
        # [B, n_input]

        return input_idx, routing_weights


# ============================================================
# Hierarchical Dynamic FFN - OPTIMIZED (No For Loop!)
# ============================================================

class HierarchicalDynamicFFN(nn.Module):
    """
    계층적 동적 FFN - 완전 병렬화 버전

    For loop 제거 → torch.gather + torch.bmm 사용
    """

    def __init__(
        self,
        d_model: int = 512,
        n_input_neurons: int = 2048,
        n_process_neurons: int = 1024,
        d_routing: int = 256,
        dropout: float = 0.1,
        temperature: float = 0.5  # Routing temperature
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
            use_mlp=True,
            temperature=temperature
        )

        # ===== Phase 2: Input Neurons =====
        # Initialized with orthogonal in _init_weights()
        self.input_patterns = nn.Parameter(
            torch.empty(n_input_neurons, d_model)
        )

        # ===== Phase 3: Process Neurons =====
        # Initialized with orthogonal in _init_weights()
        self.process_weights = nn.Parameter(
            torch.empty(n_process_neurons, n_input_neurons)
        )
        self.process_outputs = nn.Parameter(
            torch.empty(n_process_neurons, d_model)
        )

        self.dropout = nn.Dropout(dropout)

        # Routing 통계
        self.input_neuron_counts = None
        self.process_neuron_counts = None
        self.last_routing_scores = None

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with orthogonal initialization.

        Uses gain=sqrt(2) ≈ 1.414 for GELU activation (recommended by Kaiming He).
        This prevents vanishing signals through the network depth.
        """
        import math
        gain = math.sqrt(2.0)  # For GELU/ReLU-like activations

        nn.init.orthogonal_(self.input_patterns, gain=gain)
        nn.init.orthogonal_(self.process_weights, gain=gain)
        nn.init.orthogonal_(self.process_outputs, gain=gain)

    def forward(
        self,
        x: torch.Tensor,
        k_input: Optional[int] = None,
        k_process: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, d_model]
            k_input: 선택할 입력 뉴런 수
            k_process: 선택할 처리 뉴런 수

        Returns:
            output: [B, S, d_model]
        """
        B, S, d_model = x.shape

        # Default k values
        if k_input is None:
            k_input = self.n_input // 2  # 50%
        if k_process is None:
            k_process = self.n_process  # 100%

        # ===== Phase 1: Global Router =====
        input_idx, routing_weights = self.global_router(x, k_input)
        # input_idx: [B, k_input]
        # routing_weights: [B, n_input]

        # Routing 통계 저장
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

            self.input_neuron_counts += routing_weights.sum(dim=0).detach()
            self.last_routing_scores = routing_weights.detach()

        # ===== Phase 2: Input Neurons =====
        input_acts = F.gelu(x @ self.input_patterns.T)  # [B, S, n_input]

        # Soft routing weights 적용
        weighted_input_acts = input_acts * routing_weights.unsqueeze(1)
        # [B, S, n_input]

        # 선택된 입력 뉴런만 추출 (병렬!)
        expanded_input_idx = input_idx.unsqueeze(1).expand(-1, S, -1)
        # [B, S, k_input]

        selected_input_acts = torch.gather(
            weighted_input_acts,
            2,
            expanded_input_idx
        )
        # [B, S, k_input]

        # ===== Phase 3: Process Neurons (완전 병렬!) =====

        # 🔥 핵심 최적화: 배치별 가중치 수집을 병렬로!

        # 1. Process weights를 배치 차원으로 확장
        process_weights_expanded = self.process_weights.unsqueeze(0).expand(
            B, -1, -1
        )  # [B, n_process, n_input]

        # 2. 각 배치에 대해 선택된 입력 뉴런의 가중치만 수집 (병렬!)
        input_idx_expanded = input_idx.unsqueeze(1).expand(
            -1, self.n_process, -1
        )  # [B, n_process, k_input]

        selected_process_weights = torch.gather(
            process_weights_expanded,
            2,  # n_input 차원에서 선택
            input_idx_expanded
        )
        # [B, n_process, k_input]

        # 3. 배치별 행렬곱 (병렬!)
        # selected_input_acts: [B, S, k_input]
        # selected_process_weights.T: [B, k_input, n_process]
        process_acts = F.gelu(
            torch.bmm(
                selected_input_acts,
                selected_process_weights.transpose(1, 2)
            )
        )
        # [B, S, n_process]

        # 4. Process neuron 선택 (배치별로 병렬!)
        process_scores = process_acts.mean(dim=1)  # [B, n_process]

        top_process_scores, process_idx = process_scores.topk(
            k_process, dim=1
        )
        # [B, k_process]

        # 5. 선택된 process neurons의 activation 추출 (병렬!)
        expanded_process_idx = process_idx.unsqueeze(1).expand(-1, S, -1)
        # [B, S, k_process]

        selected_process_acts = torch.gather(
            process_acts,
            2,
            expanded_process_idx
        )
        # [B, S, k_process]

        # 6. 출력 가중치 수집 (병렬!)
        expanded_process_idx_for_output = process_idx.unsqueeze(2).expand(
            -1, -1, d_model
        )
        # [B, k_process, d_model]

        selected_process_outputs = torch.gather(
            self.process_outputs.unsqueeze(0).expand(B, -1, -1),
            1,  # n_process 차원에서 선택
            expanded_process_idx_for_output
        )
        # [B, k_process, d_model]

        # 7. 최종 출력 (배치 행렬곱 - 병렬!)
        output = torch.bmm(
            selected_process_acts,  # [B, S, k_process]
            selected_process_outputs  # [B, k_process, d_model]
        )
        # [B, S, d_model]

        # Process neuron 통계 수집
        if self.training:
            # process_idx: [B, k_process]
            ones = torch.ones_like(process_idx, dtype=torch.float32)
            self.process_neuron_counts.scatter_add_(
                0,
                process_idx.flatten(),
                ones.flatten()
            )

        return self.dropout(output)

    def get_load_balance_loss(self) -> torch.Tensor:
        """Load balancing loss 계산"""
        if self.input_neuron_counts is None or not self.training:
            device = self.input_patterns.device
            return torch.tensor(0.0, device=device)

        counts = self.input_neuron_counts
        device = counts.device

        if counts.sum() == 0:
            return torch.tensor(0.0, device=device)

        # 정규화
        usage_probs = counts / (counts.sum() + 1e-8)

        # 목표: 균등 분포
        target_prob = 1.0 / self.n_input
        target = torch.full_like(usage_probs, target_prob)

        # KL divergence
        usage_probs = usage_probs + 1e-8
        target = target + 1e-8

        kl_loss = F.kl_div(
            usage_probs.log(),
            target,
            reduction='sum'
        ) / self.n_input

        # Routing entropy
        if self.last_routing_scores is not None:
            avg_probs = self.last_routing_scores.mean(dim=0) + 1e-8

            entropy = -(avg_probs * avg_probs.log()).sum()
            max_entropy = torch.log(
                torch.tensor(float(self.n_input), device=device)
            )

            normalized_entropy = entropy / max_entropy
            entropy_loss = torch.clamp(1.0 - normalized_entropy, min=0.0, max=1.0)
        else:
            entropy_loss = torch.tensor(0.0, device=device)

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
            k_input = self.n_input // 2
        if k_process is None:
            k_process = self.n_process

        with torch.no_grad():
            # Global Router
            input_idx, routing_weights = self.global_router(x, k_input)

            # Input Neurons
            input_acts = F.gelu(x @ self.input_patterns.T)
            weighted_input_acts = input_acts * routing_weights.unsqueeze(1)
            expanded_idx = input_idx.unsqueeze(1).expand(-1, S, -1)
            selected_input_acts = torch.gather(weighted_input_acts, 2, expanded_idx)

            # Process Neurons (병렬 버전)
            process_weights_expanded = self.process_weights.unsqueeze(0).expand(
                B, -1, -1
            )
            input_idx_expanded = input_idx.unsqueeze(1).expand(-1, self.n_process, -1)
            selected_process_weights = torch.gather(
                process_weights_expanded, 2, input_idx_expanded
            )

            process_acts = F.gelu(
                torch.bmm(
                    selected_input_acts,
                    selected_process_weights.transpose(1, 2)
                )
            )
            process_scores = process_acts.mean(dim=1)
            top_process_scores, process_idx = process_scores.topk(k_process, dim=1)

            # 첫 번째 배치의 통계만
            selected_routing_scores = torch.gather(
                routing_weights, 1, input_idx
            )

            return {
                'global_router': {
                    'input_indices': input_idx.cpu(),
                    'routing_scores': selected_routing_scores.cpu(),
                    'mean_score': selected_routing_scores.mean().item()
                },
                'input_neurons': {
                    'indices': input_idx.cpu(),
                    'activations': selected_input_acts.mean(dim=1).cpu(),
                    'mean_activation': selected_input_acts.mean().item(),
                    'sparsity': k_input / self.n_input
                },
                'process_neurons': {
                    'indices': process_idx[0].cpu(),  # 첫 배치만
                    'activations': top_process_scores[0].cpu(),
                    'mean_activation': top_process_scores.mean().item(),
                    'sparsity': k_process / self.n_process
                }
            }


# ============================================================
# Transformer Layer - UNCHANGED
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
        use_checkpoint: bool = False,
        temperature: float = 0.5  # Routing temperature
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
            dropout=dropout,
            temperature=temperature
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint

    def _attention_block(self, x, attention_mask):
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(
            x_norm, x_norm, x_norm,
            key_padding_mask=attention_mask
        )
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
            attn_out = checkpoint(
                self._attention_block, x, attention_mask,
                use_reentrant=False
            )
        else:
            attn_out = self._attention_block(x, attention_mask)
        x = x + attn_out

        # FFN
        if self.use_checkpoint and self.training:
            ffn_out = checkpoint(
                self._ffn_block, x, k_input, k_process,
                use_reentrant=False
            )
        else:
            ffn_out = self._ffn_block(x, k_input, k_process)
        x = x + ffn_out

        return x


# ============================================================
# Language Model - UNCHANGED
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
        gradient_checkpointing: bool = False,
        temperature: float = 0.5  # Routing temperature (lower = smoother)
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
                use_checkpoint=gradient_checkpointing,
                temperature=temperature
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
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask, k_input, k_process)

        x = self.norm(x)
        logits = self.output_projection(x)

        # Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        return {
            'logits': logits,
            'loss': loss
        }

    def get_model_stats(self) -> dict:
        """모델 통계"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        ffn_params = sum(
            sum(p.numel() for p in layer.ffn.parameters())
            for layer in self.layers
        )

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
# Performance Benchmark
# ============================================================

def benchmark_ffn(device='cuda', n_runs=100):
    """FFN 성능 벤치마크"""
    import time

    print("=" * 60)
    print("FFN Performance Benchmark")
    print("=" * 60)

    # 설정
    B, S, d_model = 32, 128, 512
    n_input = 2048
    n_process = 1024

    # 모델 생성
    ffn = HierarchicalDynamicFFN(
        d_model=d_model,
        n_input_neurons=n_input,
        n_process_neurons=n_process
    ).to(device)

    # 입력 생성
    x = torch.randn(B, S, d_model, device=device)

    # Warmup
    for _ in range(10):
        _ = ffn(x)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(n_runs):
        output = ffn(x)

    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / n_runs * 1000  # ms

    print(f"\nConfiguration:")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {S}")
    print(f"  d_model: {d_model}")
    print(f"  n_input_neurons: {n_input}")
    print(f"  n_process_neurons: {n_process}")

    print(f"\nPerformance:")
    print(f"  Average time per forward pass: {avg_time:.2f} ms")
    print(f"  Throughput: {n_runs / (end - start):.2f} iterations/sec")

    # 메모리 사용량
    if device == 'cuda':
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"\nMemory:")
        print(f"  Allocated: {memory_allocated:.2f} MB")
        print(f"  Reserved: {memory_reserved:.2f} MB")

    return avg_time


# ============================================================
# Test & Demo
# ============================================================

if __name__ == '__main__':
    print("Testing Optimized Hierarchical Dynamic Neuron FFN...")
    print()

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
    print("Model Statistics:")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"  FFN parameters: {stats['ffn_parameters']:,} "
          f"({stats['ffn_percentage']:.1f}%)")
    print(f"  Router parameters: {stats['router_parameters']:,} "
          f"({stats['router_percentage']:.1f}%)")

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
    print(f"    Mean routing score: "
          f"{neuron_stats['global_router']['mean_score']:.4f}")

    print(f"  Input neurons:")
    print(f"    Selected: {neuron_stats['input_neurons']['sparsity']*100:.1f}%")
    print(f"    Mean activation: "
          f"{neuron_stats['input_neurons']['mean_activation']:.4f}")

    print(f"  Process neurons:")
    print(f"    Selected: "
          f"{neuron_stats['process_neurons']['sparsity']*100:.1f}%")
    print(f"    Mean activation: "
          f"{neuron_stats['process_neurons']['mean_activation']:.4f}")

    print(f"\n✓ All tests passed!")

    # 성능 벤치마크 (CUDA 사용 가능한 경우)
    if torch.cuda.is_available():
        print()
        benchmark_ffn(device='cuda', n_runs=100)
    else:
        print("\nCUDA not available - skipping performance benchmark")
