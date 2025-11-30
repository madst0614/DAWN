"""
DAWN v10.1: Triton Optimized Kernels

ScatterMoE/MegaBlocks 방식 참고:
- Grouped GEMM으로 뉴런별 matmul 배치 처리
- 블록 타일링 + coalesced memory access
- Forward/Backward 모두 Triton

핵심 아이디어:
1. 토큰을 뉴런별로 그룹핑 (permute)
2. 각 뉴런 그룹에 대해 배치 matmul
3. 결과를 원래 순서로 복원 (unpermute)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available")


# ============================================================
# Utility: Permute/Unpermute for Grouped Processing
# ============================================================

def compute_permutation(topk_idx: torch.Tensor, num_neurons: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    토큰-뉴런 쌍을 뉴런별로 그룹핑하기 위한 permutation 계산

    Args:
        topk_idx: [BS, k] - 각 토큰이 선택한 뉴런 인덱스
        num_neurons: N

    Returns:
        sorted_idx: [BS*k] - 정렬된 (토큰, 슬롯) 인덱스
        neuron_counts: [N] - 뉴런별 토큰 수
        neuron_offsets: [N+1] - 뉴런별 시작 오프셋
    """
    BS, k = topk_idx.shape

    # Flatten: [BS, k] -> [BS*k]
    flat_idx = topk_idx.view(-1)

    # 뉴런 인덱스로 정렬
    sorted_neuron_idx, sorted_idx = torch.sort(flat_idx, stable=True)

    # 뉴런별 카운트
    neuron_counts = torch.bincount(flat_idx, minlength=num_neurons)

    # 뉴런별 오프셋 (cumsum)
    neuron_offsets = torch.zeros(num_neurons + 1, dtype=torch.int32, device=topk_idx.device)
    neuron_offsets[1:] = torch.cumsum(neuron_counts, dim=0)

    return sorted_idx, neuron_counts, neuron_offsets


# ============================================================
# Triton Kernels
# ============================================================

if TRITON_AVAILABLE:

    # ----------------------------------------------------------
    # Grouped GEMM Forward Kernel
    # ----------------------------------------------------------
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        ],
        key=['total_tokens', 'D', 'R'],
    )
    @triton.jit
    def grouped_compress_fwd_kernel(
        # Input/Output pointers
        x_ptr,              # [BS, D] - 입력 (permuted)
        neurons_ptr,        # [N, D, R] - 뉴런 가중치
        output_ptr,         # [BS*k, R] - 출력 (permuted)
        # Permutation info
        sorted_idx_ptr,     # [BS*k] - permutation 인덱스
        neuron_offsets_ptr, # [N+1] - 뉴런별 오프셋
        # Sizes
        total_tokens,       # BS * k
        N,                  # num neurons
        D: tl.constexpr,    # input dim
        R: tl.constexpr,    # output dim (rank)
        k,                  # top-k
        # Strides
        stride_x_bs, stride_x_d,
        stride_n_n, stride_n_d, stride_n_r,
        stride_o_t, stride_o_r,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Grouped GEMM for compress operation.
        각 프로그램이 하나의 (뉴런, 출력 블록) 담당.
        """
        # Program IDs
        pid_n = tl.program_id(0)  # 뉴런 인덱스
        pid_block = tl.program_id(1)  # 출력 블록 인덱스

        if pid_n >= N:
            return

        # 이 뉴런의 토큰 범위
        start_offset = tl.load(neuron_offsets_ptr + pid_n)
        end_offset = tl.load(neuron_offsets_ptr + pid_n + 1)
        num_tokens = end_offset - start_offset

        if num_tokens == 0:
            return

        # 출력 R 차원에서의 블록 범위
        r_start = pid_block * BLOCK_N
        r_offs = r_start + tl.arange(0, BLOCK_N)
        r_mask = r_offs < R

        # 토큰 블록별로 처리
        for m_start in range(0, num_tokens, BLOCK_M):
            m_offs = m_start + tl.arange(0, BLOCK_M)
            m_mask = (m_offs < num_tokens)

            # 실제 토큰 인덱스 가져오기
            perm_idx = start_offset + m_offs
            perm_mask = m_mask

            # sorted_idx에서 원래 토큰-슬롯 인덱스 가져오기
            token_slot_idx = tl.load(sorted_idx_ptr + perm_idx, mask=perm_mask, other=0)
            token_idx = token_slot_idx // k  # 원래 토큰 인덱스

            # 누적기 초기화
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # D 차원을 블록으로 순회
            for d_start in range(0, D, BLOCK_K):
                d_offs = d_start + tl.arange(0, BLOCK_K)
                d_mask = d_offs < D

                # x[token_idx, d_offs] 로드: [BLOCK_M, BLOCK_K]
                x_ptrs = x_ptr + token_idx[:, None] * stride_x_bs + d_offs[None, :] * stride_x_d
                x_block = tl.load(x_ptrs, mask=perm_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

                # neurons[pid_n, d_offs, r_offs] 로드: [BLOCK_K, BLOCK_N]
                n_ptrs = neurons_ptr + pid_n * stride_n_n + d_offs[:, None] * stride_n_d + r_offs[None, :] * stride_n_r
                n_block = tl.load(n_ptrs, mask=d_mask[:, None] & r_mask[None, :], other=0.0).to(tl.float32)

                # GEMM: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
                acc += tl.dot(x_block, n_block)

            # 결과 저장
            out_ptrs = output_ptr + perm_idx[:, None] * stride_o_t + r_offs[None, :] * stride_o_r
            tl.store(out_ptrs, acc, mask=perm_mask[:, None] & r_mask[None, :])


    # ----------------------------------------------------------
    # Unpermute + Weighted Sum Kernel
    # ----------------------------------------------------------
    @triton.jit
    def unpermute_weighted_sum_kernel(
        # Input/Output
        permuted_ptr,       # [BS*k, R] - permuted 출력
        weights_ptr,        # [BS, k] - 가중치
        output_ptr,         # [BS, R] - 최종 출력
        sorted_idx_ptr,     # [BS*k] - permutation 인덱스 (역방향용)
        # Sizes
        BS, k: tl.constexpr, R: tl.constexpr,
        # Strides
        stride_p_t, stride_p_r,
        stride_w_bs, stride_w_k,
        stride_o_bs, stride_o_r,
        # Block
        BLOCK_R: tl.constexpr,
    ):
        """
        Permuted 출력을 원래 순서로 복원하면서 weighted sum 수행.
        각 프로그램이 하나의 토큰 담당.
        """
        pid_bs = tl.program_id(0)
        pid_r = tl.program_id(1)

        if pid_bs >= BS:
            return

        r_start = pid_r * BLOCK_R
        r_offs = r_start + tl.arange(0, BLOCK_R)
        r_mask = r_offs < R

        # 누적기
        acc = tl.zeros((BLOCK_R,), dtype=tl.float32)

        # k개 슬롯 순회
        for s in range(k):
            # 가중치 로드
            w = tl.load(weights_ptr + pid_bs * stride_w_bs + s * stride_w_k)

            # permuted 출력에서 해당 값 찾기
            # sorted_idx의 역함수 필요 -> 직접 계산
            token_slot = pid_bs * k + s

            # permuted_ptr[token_slot, r_offs] 로드
            p_ptrs = permuted_ptr + token_slot * stride_p_t + r_offs * stride_p_r
            p_vals = tl.load(p_ptrs, mask=r_mask, other=0.0)

            acc += w * p_vals

        # 저장
        out_ptrs = output_ptr + pid_bs * stride_o_bs + r_offs * stride_o_r
        tl.store(out_ptrs, acc, mask=r_mask)


    # ----------------------------------------------------------
    # Backward: grad_x kernel
    # ----------------------------------------------------------
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        ],
        key=['BS', 'D', 'R'],
    )
    @triton.jit
    def grouped_compress_bwd_x_kernel(
        # Inputs
        grad_out_ptr,       # [BS*k, R] - permuted grad output
        neurons_ptr,        # [N, D, R]
        weights_ptr,        # [BS, k]
        # Output
        grad_x_ptr,         # [BS, D]
        # Permutation
        sorted_idx_ptr,     # [BS*k]
        topk_idx_ptr,       # [BS, k] - 원래 뉴런 인덱스
        # Sizes
        BS, k: tl.constexpr, N, D: tl.constexpr, R: tl.constexpr,
        # Strides
        stride_go_t, stride_go_r,
        stride_n_n, stride_n_d, stride_n_r,
        stride_w_bs, stride_w_k,
        stride_gx_bs, stride_gx_d,
        stride_idx_bs, stride_idx_k,
        # Blocks
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        grad_x[t] = sum_s w[s] * (grad_out[t,s] @ neurons[idx[t,s]].T)
        각 프로그램이 하나의 토큰 담당.
        """
        pid_bs = tl.program_id(0)
        pid_d = tl.program_id(1)

        if pid_bs >= BS:
            return

        d_start = pid_d * BLOCK_N
        d_offs = d_start + tl.arange(0, BLOCK_N)
        d_mask = d_offs < D

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for s in range(k):
            # 가중치
            w = tl.load(weights_ptr + pid_bs * stride_w_bs + s * stride_w_k)

            # 뉴런 인덱스
            n_idx = tl.load(topk_idx_ptr + pid_bs * stride_idx_bs + s * stride_idx_k)

            # grad_out[pid_bs*k + s, :] @ neurons[n_idx, d_offs, :].T
            token_slot = pid_bs * k + s

            slot_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

            for r_start in range(0, R, BLOCK_K):
                r_offs = r_start + tl.arange(0, BLOCK_K)
                r_mask_inner = r_offs < R

                # grad_out[token_slot, r_offs]
                go_ptrs = grad_out_ptr + token_slot * stride_go_t + r_offs * stride_go_r
                go_vals = tl.load(go_ptrs, mask=r_mask_inner, other=0.0)  # [BLOCK_K]

                # neurons[n_idx, d_offs, r_offs] -> [BLOCK_N, BLOCK_K]
                n_ptrs = neurons_ptr + n_idx * stride_n_n + d_offs[:, None] * stride_n_d + r_offs[None, :] * stride_n_r
                n_block = tl.load(n_ptrs, mask=d_mask[:, None] & r_mask_inner[None, :], other=0.0)

                # [BLOCK_N, BLOCK_K] @ [BLOCK_K] -> [BLOCK_N]
                slot_acc += tl.sum(n_block * go_vals[None, :], axis=1)

            acc += w * slot_acc

        # 저장
        gx_ptrs = grad_x_ptr + pid_bs * stride_gx_bs + d_offs * stride_gx_d
        tl.store(gx_ptrs, acc, mask=d_mask)


    # ----------------------------------------------------------
    # Backward: grad_neurons kernel (atomic add)
    # ----------------------------------------------------------
    @triton.jit
    def grouped_compress_bwd_neurons_kernel(
        # Inputs
        x_ptr,              # [BS, D]
        grad_out_ptr,       # [BS*k, R]
        weights_ptr,        # [BS, k]
        # Output
        grad_neurons_ptr,   # [N, D, R]
        # Permutation
        sorted_idx_ptr,     # [BS*k]
        neuron_offsets_ptr, # [N+1]
        # Sizes
        BS, k: tl.constexpr, N, D: tl.constexpr, R: tl.constexpr,
        # Strides
        stride_x_bs, stride_x_d,
        stride_go_t, stride_go_r,
        stride_w_bs, stride_w_k,
        stride_gn_n, stride_gn_d, stride_gn_r,
        # Blocks
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        """
        grad_neurons[n] += sum_{t,s: idx[t,s]==n} w[t,s] * outer(x[t], grad_out[t,s])
        각 프로그램이 하나의 뉴런 담당.
        """
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_r = tl.program_id(2)

        if pid_n >= N:
            return

        d_start = pid_d * BLOCK_D
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        r_start = pid_r * BLOCK_R
        r_offs = r_start + tl.arange(0, BLOCK_R)
        r_mask = r_offs < R

        # 이 뉴런의 토큰 범위
        start_offset = tl.load(neuron_offsets_ptr + pid_n)
        end_offset = tl.load(neuron_offsets_ptr + pid_n + 1)
        num_tokens = end_offset - start_offset

        # 누적기
        acc = tl.zeros((BLOCK_D, BLOCK_R), dtype=tl.float32)

        # 이 뉴런을 선택한 모든 토큰 순회
        for t in range(num_tokens):
            perm_idx = start_offset + t

            # 원래 토큰-슬롯 인덱스
            token_slot = tl.load(sorted_idx_ptr + perm_idx)
            token_idx = token_slot // k
            slot_idx = token_slot % k

            # 가중치
            w = tl.load(weights_ptr + token_idx * stride_w_bs + slot_idx * stride_w_k)

            # x[token_idx, d_offs]
            x_ptrs = x_ptr + token_idx * stride_x_bs + d_offs * stride_x_d
            x_vals = tl.load(x_ptrs, mask=d_mask, other=0.0)  # [BLOCK_D]

            # grad_out[token_slot, r_offs]
            go_ptrs = grad_out_ptr + token_slot * stride_go_t + r_offs * stride_go_r
            go_vals = tl.load(go_ptrs, mask=r_mask, other=0.0)  # [BLOCK_R]

            # outer product: [BLOCK_D] x [BLOCK_R] -> [BLOCK_D, BLOCK_R]
            acc += w * (x_vals[:, None] * go_vals[None, :])

        # 저장
        gn_ptrs = grad_neurons_ptr + pid_n * stride_gn_n + d_offs[:, None] * stride_gn_d + r_offs[None, :] * stride_gn_r
        tl.store(gn_ptrs, acc, mask=d_mask[:, None] & r_mask[None, :])


    # ----------------------------------------------------------
    # Backward: grad_weights kernel
    # ----------------------------------------------------------
    @triton.jit
    def grouped_compress_bwd_weights_kernel(
        # Inputs
        x_ptr,              # [BS, D]
        neurons_ptr,        # [N, D, R]
        grad_out_ptr,       # [BS*k, R]
        topk_idx_ptr,       # [BS, k]
        # Output
        grad_weights_ptr,   # [BS, k]
        # Sizes
        BS, k: tl.constexpr, N, D: tl.constexpr, R: tl.constexpr,
        # Strides
        stride_x_bs, stride_x_d,
        stride_n_n, stride_n_d, stride_n_r,
        stride_go_t, stride_go_r,
        stride_idx_bs, stride_idx_k,
        stride_gw_bs, stride_gw_k,
        # Block
        BLOCK_K: tl.constexpr,
    ):
        """
        grad_weights[t,s] = sum_r grad_out[t,s,r] * (x[t] @ neurons[idx[t,s]])[r]
        각 프로그램이 하나의 (토큰, 슬롯) 담당.
        """
        pid_bs = tl.program_id(0)
        pid_s = tl.program_id(1)

        if pid_bs >= BS or pid_s >= k:
            return

        # 뉴런 인덱스
        n_idx = tl.load(topk_idx_ptr + pid_bs * stride_idx_bs + pid_s * stride_idx_k)
        token_slot = pid_bs * k + pid_s

        # x[pid_bs] @ neurons[n_idx] -> [R]
        # 그리고 grad_out[token_slot]와 dot product

        acc = 0.0

        for r_start in range(0, R, BLOCK_K):
            r_offs = r_start + tl.arange(0, BLOCK_K)
            r_mask = r_offs < R

            # grad_out[token_slot, r_offs]
            go_ptrs = grad_out_ptr + token_slot * stride_go_t + r_offs * stride_go_r
            go_vals = tl.load(go_ptrs, mask=r_mask, other=0.0)

            # x[pid_bs] @ neurons[n_idx, :, r_offs] -> [BLOCK_K]
            proj = tl.zeros((BLOCK_K,), dtype=tl.float32)

            for d in range(D):
                x_val = tl.load(x_ptr + pid_bs * stride_x_bs + d * stride_x_d)
                n_vals = tl.load(neurons_ptr + n_idx * stride_n_n + d * stride_n_d + r_offs * stride_n_r,
                                mask=r_mask, other=0.0)
                proj += x_val * n_vals

            acc += tl.sum(go_vals * proj)

        # 저장
        tl.store(grad_weights_ptr + pid_bs * stride_gw_bs + pid_s * stride_gw_k, acc)


# ============================================================
# Expand Kernels (비슷한 구조, R과 D 역할 교환)
# ============================================================

if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        ],
        key=['total_tokens', 'R', 'D'],
    )
    @triton.jit
    def grouped_expand_fwd_kernel(
        # Input/Output pointers
        x_ptr,              # [BS, R] - 입력
        neurons_ptr,        # [N, R, D] - 뉴런 가중치
        output_ptr,         # [BS*k, D] - 출력 (permuted)
        # Permutation info
        sorted_idx_ptr,     # [BS*k]
        neuron_offsets_ptr, # [N+1]
        # Sizes
        total_tokens,       # BS * k
        N,                  # num neurons
        R: tl.constexpr,    # input dim (rank)
        D: tl.constexpr,    # output dim
        k,                  # top-k
        # Strides
        stride_x_bs, stride_x_r,
        stride_n_n, stride_n_r, stride_n_d,
        stride_o_t, stride_o_d,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Grouped GEMM for expand operation."""
        pid_n = tl.program_id(0)
        pid_block = tl.program_id(1)

        if pid_n >= N:
            return

        start_offset = tl.load(neuron_offsets_ptr + pid_n)
        end_offset = tl.load(neuron_offsets_ptr + pid_n + 1)
        num_tokens = end_offset - start_offset

        if num_tokens == 0:
            return

        d_start = pid_block * BLOCK_N
        d_offs = d_start + tl.arange(0, BLOCK_N)
        d_mask = d_offs < D

        for m_start in range(0, num_tokens, BLOCK_M):
            m_offs = m_start + tl.arange(0, BLOCK_M)
            m_mask = (m_offs < num_tokens)

            perm_idx = start_offset + m_offs
            perm_mask = m_mask

            token_slot_idx = tl.load(sorted_idx_ptr + perm_idx, mask=perm_mask, other=0)
            token_idx = token_slot_idx // k

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for r_start in range(0, R, BLOCK_K):
                r_offs = r_start + tl.arange(0, BLOCK_K)
                r_mask_inner = r_offs < R

                x_ptrs = x_ptr + token_idx[:, None] * stride_x_bs + r_offs[None, :] * stride_x_r
                x_block = tl.load(x_ptrs, mask=perm_mask[:, None] & r_mask_inner[None, :], other=0.0).to(tl.float32)

                n_ptrs = neurons_ptr + pid_n * stride_n_n + r_offs[:, None] * stride_n_r + d_offs[None, :] * stride_n_d
                n_block = tl.load(n_ptrs, mask=r_mask_inner[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

                acc += tl.dot(x_block, n_block)

            out_ptrs = output_ptr + perm_idx[:, None] * stride_o_t + d_offs[None, :] * stride_o_d
            tl.store(out_ptrs, acc, mask=perm_mask[:, None] & d_mask[None, :])


# ============================================================
# Autograd Functions
# ============================================================

class TritonCompressFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, neurons, topk_idx, weights):
        """
        Args:
            x: [B, S, D]
            neurons: [N, D, R]
            topk_idx: [B, S, k]
            weights: [B, S, k]
        Returns:
            output: [B, S, R]
        """
        B, S, D = x.shape
        N, _, R = neurons.shape
        k = topk_idx.shape[-1]
        BS = B * S

        # Flatten
        x_flat = x.view(BS, D).contiguous()
        topk_flat = topk_idx.view(BS, k).contiguous()
        weights_flat = weights.view(BS, k).contiguous()

        # Compute permutation
        sorted_idx, neuron_counts, neuron_offsets = compute_permutation(topk_flat, N)

        # Allocate output (permuted order)
        permuted_output = torch.zeros(BS * k, R, device=x.device, dtype=x.dtype)

        # Launch grouped GEMM kernel
        num_r_blocks = triton.cdiv(R, 64)
        grid = (N, num_r_blocks)

        grouped_compress_fwd_kernel[grid](
            x_flat, neurons, permuted_output,
            sorted_idx, neuron_offsets,
            BS * k, N, D, R, k,
            x_flat.stride(0), x_flat.stride(1),
            neurons.stride(0), neurons.stride(1), neurons.stride(2),
            permuted_output.stride(0), permuted_output.stride(1),
        )

        # Unpermute and weighted sum
        output = torch.zeros(BS, R, device=x.device, dtype=x.dtype)

        num_r_blocks = triton.cdiv(R, 32)
        grid = (BS, num_r_blocks)

        unpermute_weighted_sum_kernel[grid](
            permuted_output, weights_flat, output, sorted_idx,
            BS, k, R,
            permuted_output.stride(0), permuted_output.stride(1),
            weights_flat.stride(0), weights_flat.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_R=32,
        )

        # Save for backward
        ctx.save_for_backward(x_flat, neurons, topk_flat, weights_flat, sorted_idx, neuron_offsets)
        ctx.shape = (B, S, D, R, N, k, BS)

        return output.view(B, S, R)

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, neurons, topk_flat, weights_flat, sorted_idx, neuron_offsets = ctx.saved_tensors
        B, S, D, R, N, k, BS = ctx.shape
        original_dtype = grad_output.dtype

        # Convert to float32 for computation
        grad_out_flat = grad_output.reshape(BS, R).float().contiguous()
        x_flat = x_flat.float()
        neurons = neurons.float()
        weights_flat = weights_flat.float()

        # grad_out을 permuted order로 확장 (각 슬롯별로 복제)
        grad_out_permuted = torch.zeros(BS * k, R, device=grad_out_flat.device, dtype=torch.float32)
        for s in range(k):
            grad_out_permuted[s::k] = grad_out_flat

        # grad_x
        grad_x = torch.zeros(BS, D, device=x_flat.device, dtype=torch.float32)
        num_d_blocks = triton.cdiv(D, 64)
        grid = (BS, num_d_blocks)

        grouped_compress_bwd_x_kernel[grid](
            grad_out_permuted, neurons, weights_flat,
            grad_x,
            sorted_idx, topk_flat,
            BS, k, N, D, R,
            grad_out_permuted.stride(0), grad_out_permuted.stride(1),
            neurons.stride(0), neurons.stride(1), neurons.stride(2),
            weights_flat.stride(0), weights_flat.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            topk_flat.stride(0), topk_flat.stride(1),
        )

        # grad_neurons
        grad_neurons = torch.zeros_like(neurons, dtype=torch.float32)
        num_d_blocks = triton.cdiv(D, 32)
        num_r_blocks = triton.cdiv(R, 32)
        grid = (N, num_d_blocks, num_r_blocks)

        grouped_compress_bwd_neurons_kernel[grid](
            x_flat, grad_out_permuted, weights_flat,
            grad_neurons,
            sorted_idx, neuron_offsets,
            BS, k, N, D, R,
            x_flat.stride(0), x_flat.stride(1),
            grad_out_permuted.stride(0), grad_out_permuted.stride(1),
            weights_flat.stride(0), weights_flat.stride(1),
            grad_neurons.stride(0), grad_neurons.stride(1), grad_neurons.stride(2),
            BLOCK_D=32, BLOCK_R=32,
        )

        # grad_weights
        grad_weights = torch.zeros(BS, k, device=x_flat.device, dtype=torch.float32)
        grid = (BS, k)

        grouped_compress_bwd_weights_kernel[grid](
            x_flat, neurons, grad_out_permuted, topk_flat,
            grad_weights,
            BS, k, N, D, R,
            x_flat.stride(0), x_flat.stride(1),
            neurons.stride(0), neurons.stride(1), neurons.stride(2),
            grad_out_permuted.stride(0), grad_out_permuted.stride(1),
            topk_flat.stride(0), topk_flat.stride(1),
            grad_weights.stride(0), grad_weights.stride(1),
            BLOCK_K=32,
        )

        # Cast back to original dtype
        return grad_x.reshape(B, S, D).to(original_dtype), grad_neurons.to(original_dtype), None, grad_weights.reshape(B, S, k).to(original_dtype)


class TritonExpandFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, neurons, topk_idx, weights):
        B, S, R = x.shape
        N, _, D = neurons.shape
        k = topk_idx.shape[-1]
        BS = B * S

        x_flat = x.view(BS, R).contiguous()
        topk_flat = topk_idx.view(BS, k).contiguous()
        weights_flat = weights.view(BS, k).contiguous()

        sorted_idx, neuron_counts, neuron_offsets = compute_permutation(topk_flat, N)

        permuted_output = torch.zeros(BS * k, D, device=x.device, dtype=x.dtype)

        num_d_blocks = triton.cdiv(D, 64)
        grid = (N, num_d_blocks)

        grouped_expand_fwd_kernel[grid](
            x_flat, neurons, permuted_output,
            sorted_idx, neuron_offsets,
            BS * k, N, R, D, k,
            x_flat.stride(0), x_flat.stride(1),
            neurons.stride(0), neurons.stride(1), neurons.stride(2),
            permuted_output.stride(0), permuted_output.stride(1),
        )

        # Unpermute and weighted sum
        output = torch.zeros(BS, D, device=x.device, dtype=x.dtype)

        num_d_blocks = triton.cdiv(D, 32)
        grid = (BS, num_d_blocks)

        # Reuse unpermute kernel (R->D)
        unpermute_weighted_sum_kernel[grid](
            permuted_output, weights_flat, output, sorted_idx,
            BS, k, D,
            permuted_output.stride(0), permuted_output.stride(1),
            weights_flat.stride(0), weights_flat.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_R=32,
        )

        ctx.save_for_backward(x_flat, neurons, topk_flat, weights_flat, sorted_idx, neuron_offsets)
        ctx.shape = (B, S, R, D, N, k, BS)

        return output.view(B, S, D)

    @staticmethod
    def backward(ctx, grad_output):
        # Similar structure to compress backward, with R and D swapped
        x_flat, neurons, topk_flat, weights_flat, sorted_idx, neuron_offsets = ctx.saved_tensors
        B, S, R, D, N, k, BS = ctx.shape
        original_dtype = grad_output.dtype

        # Convert to float32 for computation
        grad_out_flat = grad_output.reshape(BS, D).float().contiguous()
        x_flat = x_flat.float()
        neurons = neurons.float()
        weights_flat = weights_flat.float()

        grad_x = torch.zeros(BS, R, device=x_flat.device, dtype=torch.float32)
        grad_neurons = torch.zeros_like(neurons, dtype=torch.float32)
        grad_weights = torch.zeros(BS, k, device=x_flat.device, dtype=torch.float32)

        # PyTorch fallback
        for b in range(BS):
            for s in range(k):
                n_idx = topk_flat[b, s]
                w = weights_flat[b, s]

                grad_x[b] += w * (grad_out_flat[b] @ neurons[n_idx].T)
                grad_neurons[n_idx] += w * torch.outer(x_flat[b], grad_out_flat[b])

                proj = x_flat[b] @ neurons[n_idx]
                grad_weights[b, s] = (grad_out_flat[b] * proj).sum()

        # Cast back to original dtype
        return grad_x.reshape(B, S, R).to(original_dtype), grad_neurons.to(original_dtype), None, grad_weights.reshape(B, S, k).to(original_dtype)


# ============================================================
# Wrapper Functions
# ============================================================

def triton_compress(x, neurons, topk_idx, weights):
    if TRITON_AVAILABLE and x.is_cuda:
        return TritonCompressFunction.apply(x, neurons, topk_idx, weights)
    else:
        return pytorch_compress_fallback(x, neurons, topk_idx, weights)


def triton_expand(x, neurons, topk_idx, weights):
    if TRITON_AVAILABLE and x.is_cuda:
        return TritonExpandFunction.apply(x, neurons, topk_idx, weights)
    else:
        return pytorch_expand_fallback(x, neurons, topk_idx, weights)


def pytorch_compress_fallback(x, neurons, topk_idx, weights):
    """PyTorch fallback"""
    B, S, D = x.shape
    N, _, R = neurons.shape
    k = topk_idx.shape[-1]

    # Dense computation
    neurons_flat = neurons.permute(1, 0, 2).reshape(D, N * R)
    all_proj = (x @ neurons_flat).view(B, S, N, R)

    # Gather selected
    topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, R)
    selected = all_proj.gather(2, topk_idx_exp)

    # Weighted sum
    output = (selected * weights.unsqueeze(-1)).sum(dim=2)
    return output


def pytorch_expand_fallback(x, neurons, topk_idx, weights):
    """PyTorch fallback"""
    B, S, R = x.shape
    N, _, D = neurons.shape
    k = topk_idx.shape[-1]

    neurons_flat = neurons.permute(1, 0, 2).reshape(R, N * D)
    all_proj = (x @ neurons_flat).view(B, S, N, D)

    topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    selected = all_proj.gather(2, topk_idx_exp)

    output = (selected * weights.unsqueeze(-1)).sum(dim=2)
    return output


# ============================================================
# Test & Benchmark
# ============================================================

def test_correctness():
    """정확성 테스트"""
    if not TRITON_AVAILABLE:
        print("Triton not available, skipping test")
        return

    B, S, D, R, N, k = 2, 4, 64, 16, 32, 4

    x = torch.randn(B, S, D, device='cuda', dtype=torch.float32, requires_grad=True)
    neurons = torch.randn(N, D, R, device='cuda', dtype=torch.float32, requires_grad=True)
    topk_idx = torch.randint(0, N, (B, S, k), device='cuda')
    weights = torch.softmax(torch.randn(B, S, k, device='cuda'), dim=-1)
    weights.requires_grad = True

    # Triton forward
    out_triton = triton_compress(x, neurons, topk_idx, weights)

    # PyTorch reference
    x_ref = x.detach().clone().requires_grad_(True)
    neurons_ref = neurons.detach().clone().requires_grad_(True)
    weights_ref = weights.detach().clone().requires_grad_(True)
    out_ref = pytorch_compress_fallback(x_ref, neurons_ref, topk_idx, weights_ref)

    # Compare forward
    fwd_diff = (out_triton - out_ref).abs().max()
    print(f"Forward diff: {fwd_diff:.6f}")

    # Backward
    grad_out = torch.randn_like(out_triton)

    out_triton.backward(grad_out)
    out_ref.backward(grad_out)

    grad_x_diff = (x.grad - x_ref.grad).abs().max()
    grad_n_diff = (neurons.grad - neurons_ref.grad).abs().max()
    grad_w_diff = (weights.grad - weights_ref.grad).abs().max()

    print(f"grad_x diff: {grad_x_diff:.6f}")
    print(f"grad_neurons diff: {grad_n_diff:.6f}")
    print(f"grad_weights diff: {grad_w_diff:.6f}")

    if fwd_diff < 1e-3 and grad_x_diff < 1e-3:
        print("✅ Test passed!")
    else:
        print("❌ Test failed!")


def benchmark():
    """상세 속도 벤치마크 - 병목 식별"""
    import time

    print("\n" + "="*60)
    print("TRITON KERNEL BENCHMARK")
    print("="*60)

    if not TRITON_AVAILABLE:
        print("Triton not available")
        return

    # 실제 DAWN v10.1 크기
    configs = [
        # (B, S, D, R, N, k, name)
        (8, 128, 320, 64, 224, 8, "Training batch"),
        (1, 128, 320, 64, 224, 8, "Single sample"),
        (32, 128, 320, 64, 224, 8, "Large batch"),
    ]

    for B, S, D, R, N, k, name in configs:
        print(f"\n--- {name}: B={B}, S={S}, D={D}, R={R}, N={N}, k={k} ---")

        x_compress = torch.randn(B, S, D, device='cuda', dtype=torch.float32, requires_grad=True)
        x_expand = torch.randn(B, S, R, device='cuda', dtype=torch.float32, requires_grad=True)
        neurons = torch.randn(N, D, R, device='cuda', dtype=torch.float32, requires_grad=True)
        topk_idx = torch.randint(0, N, (B, S, k), device='cuda')
        weights = torch.softmax(torch.randn(B, S, k, device='cuda'), dim=-1).requires_grad_(True)

        n_iters = 20
        torch.cuda.synchronize()

        # ============ COMPRESS FORWARD ============
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iters):
            out = triton_compress(x_compress, neurons, topk_idx, weights)
        torch.cuda.synchronize()
        compress_fwd_triton = (time.time() - start) / n_iters * 1000

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iters):
            out = pytorch_compress_fallback(x_compress.detach(), neurons.detach(), topk_idx, weights.detach())
        torch.cuda.synchronize()
        compress_fwd_pytorch = (time.time() - start) / n_iters * 1000

        # ============ COMPRESS BACKWARD ============
        x_c = x_compress.detach().clone().requires_grad_(True)
        n_c = neurons.detach().clone().requires_grad_(True)
        w_c = weights.detach().clone().requires_grad_(True)
        out_c = triton_compress(x_c, n_c, topk_idx, w_c)
        grad_out = torch.randn_like(out_c)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iters):
            if x_c.grad is not None:
                x_c.grad.zero_()
                n_c.grad.zero_()
                w_c.grad.zero_()
            out_c = triton_compress(x_c, n_c, topk_idx, w_c)
            out_c.backward(grad_out, retain_graph=True)
        torch.cuda.synchronize()
        compress_bwd_triton = (time.time() - start) / n_iters * 1000

        x_p = x_compress.detach().clone().requires_grad_(True)
        n_p = neurons.detach().clone().requires_grad_(True)
        w_p = weights.detach().clone().requires_grad_(True)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iters):
            if x_p.grad is not None:
                x_p.grad.zero_()
                n_p.grad.zero_()
                w_p.grad.zero_()
            out_p = pytorch_compress_fallback(x_p, n_p, topk_idx, w_p)
            out_p.backward(grad_out, retain_graph=True)
        torch.cuda.synchronize()
        compress_bwd_pytorch = (time.time() - start) / n_iters * 1000

        # ============ EXPAND FORWARD ============
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iters):
            out = triton_expand(x_expand, neurons, topk_idx, weights)
        torch.cuda.synchronize()
        expand_fwd_triton = (time.time() - start) / n_iters * 1000

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iters):
            out = pytorch_expand_fallback(x_expand.detach(), neurons.detach(), topk_idx, weights.detach())
        torch.cuda.synchronize()
        expand_fwd_pytorch = (time.time() - start) / n_iters * 1000

        # ============ EXPAND BACKWARD ============
        x_e = x_expand.detach().clone().requires_grad_(True)
        n_e = neurons.detach().clone().requires_grad_(True)
        w_e = weights.detach().clone().requires_grad_(True)
        out_e = triton_expand(x_e, n_e, topk_idx, w_e)
        grad_out_e = torch.randn_like(out_e)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iters):
            if x_e.grad is not None:
                x_e.grad.zero_()
                n_e.grad.zero_()
                w_e.grad.zero_()
            out_e = triton_expand(x_e, n_e, topk_idx, w_e)
            out_e.backward(grad_out_e, retain_graph=True)
        torch.cuda.synchronize()
        expand_bwd_triton = (time.time() - start) / n_iters * 1000

        x_ep = x_expand.detach().clone().requires_grad_(True)
        n_ep = neurons.detach().clone().requires_grad_(True)
        w_ep = weights.detach().clone().requires_grad_(True)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iters):
            if x_ep.grad is not None:
                x_ep.grad.zero_()
                n_ep.grad.zero_()
                w_ep.grad.zero_()
            out_ep = pytorch_expand_fallback(x_ep, n_ep, topk_idx, w_ep)
            out_ep.backward(grad_out_e, retain_graph=True)
        torch.cuda.synchronize()
        expand_bwd_pytorch = (time.time() - start) / n_iters * 1000

        # ============ RESULTS ============
        print(f"\n{'Operation':<25} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
        print("-" * 65)
        print(f"{'Compress Forward':<25} {compress_fwd_triton:<15.2f} {compress_fwd_pytorch:<15.2f} {compress_fwd_pytorch/compress_fwd_triton:<10.2f}x")
        print(f"{'Compress Backward':<25} {compress_bwd_triton:<15.2f} {compress_bwd_pytorch:<15.2f} {compress_bwd_pytorch/compress_bwd_triton:<10.2f}x")
        print(f"{'Expand Forward':<25} {expand_fwd_triton:<15.2f} {expand_fwd_pytorch:<15.2f} {expand_fwd_pytorch/expand_fwd_triton:<10.2f}x")
        print(f"{'Expand Backward':<25} {expand_bwd_triton:<15.2f} {expand_bwd_pytorch:<15.2f} {expand_bwd_pytorch/expand_bwd_triton:<10.2f}x")

        total_triton = compress_fwd_triton + compress_bwd_triton + expand_fwd_triton + expand_bwd_triton
        total_pytorch = compress_fwd_pytorch + compress_bwd_pytorch + expand_fwd_pytorch + expand_bwd_pytorch
        print(f"\n{'TOTAL (one layer)':<25} {total_triton:<15.2f} {total_pytorch:<15.2f} {total_pytorch/total_triton:<10.2f}x")
        print(f"{'8 layers estimate':<25} {total_triton*8:<15.2f} {total_pytorch*8:<15.2f}")

        # 병목 식별
        print(f"\n🔍 Bottleneck Analysis:")
        ops = [
            ("Compress Fwd", compress_fwd_triton),
            ("Compress Bwd", compress_bwd_triton),
            ("Expand Fwd", expand_fwd_triton),
            ("Expand Bwd", expand_bwd_triton),
        ]
        ops.sort(key=lambda x: x[1], reverse=True)
        for name, time_ms in ops:
            pct = time_ms / total_triton * 100
            bar = "█" * int(pct / 5)
            print(f"  {name:<15} {time_ms:>8.2f} ms ({pct:>5.1f}%) {bar}")

    print("\n" + "="*60)
    print("If Triton is slower than PyTorch, disable with USE_TRITON=False")
    print("="*60)


if __name__ == "__main__":
    print("Testing Triton kernels...")
    test_correctness()
    print("\nRunning benchmark...")
    benchmark()
