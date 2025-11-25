"""
TT Karcher Mean Module for DAWN v7.4

Weighted centroid on TT manifold
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TTRepresentation:
    """
    TT (Tensor Train) 표현
    """
    def __init__(self, core1, core2):
        """
        Args:
            core1: [16, rank, 32] or [B, S, 16, rank, 32]
            core2: [rank, 16, 32] or [B, S, rank, 16, 32]
        """
        self.core1 = core1
        self.core2 = core2
        self.rank = core1.shape[-2]

    def to_full_matrix(self):
        """
        TT cores를 full matrix로 복원
        Returns: [256, 1024] or [B, S, 256, 1024]
        """
        # Contract cores
        # core1: [..., 16, rank, 32]
        # core2: [..., rank, 16, 32]
        result = torch.einsum('...irk,...rjl->...ijkl', self.core1, self.core2)

        # Reshape to matrix
        shape = result.shape[:-4]
        return result.reshape(*shape, 256, 1024)


class TTKarcherMean(nn.Module):
    """
    Weighted Karcher Mean on TT Manifold (Memory-efficient version)

    8개 Neuron의 TT 표현을 weight로 조합
    → 단순 weighted average (메모리 효율)
    """

    def __init__(self, max_iter=5, tolerance=1e-6, step_size=0.5):
        super().__init__()
        # Note: iterations disabled for memory efficiency
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.step_size = step_size

    def forward(self, neuron_cores_A, neuron_cores_B, weights):
        """
        Args:
            neuron_cores_A: List of k TT cores for Basis_A
            neuron_cores_B: List of k TT cores for Basis_B
            weights: [B, S, k] neuron weights (softmax)

        Returns:
            centroid_A: Weighted centroid TT for Basis_A
            centroid_B: Weighted centroid TT for Basis_B
        """
        # Simple weighted average (memory efficient)
        centroid_A = self.weighted_average(neuron_cores_A, weights)
        centroid_B = self.weighted_average(neuron_cores_B, weights)

        return centroid_A, centroid_B

    def weighted_average(self, tt_cores_list, weights):
        """
        Memory-efficient weighted average of TT cores
        """
        B, S, k = weights.shape

        # Stack and compute weighted sum in one operation
        # Stack cores: [k, B, S, ...]
        core1_stack = torch.stack([tt['core1'] for tt in tt_cores_list], dim=0)
        core2_stack = torch.stack([tt['core2'] for tt in tt_cores_list], dim=0)

        # weights: [B, S, k] -> [k, B, S, 1, 1, 1]
        w = weights.permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Weighted sum
        core1_avg = (w * core1_stack).sum(dim=0)
        core2_avg = (w * core2_stack).sum(dim=0)

        return {'core1': core1_avg, 'core2': core2_avg}


class TTBasisWithKarcher(nn.Module):
    """
    TT Basis + Karcher Mean FFN

    v7.0의 FixedOrthogonalBasis를 확장
    """

    def __init__(self, n_basis=32, d_model=256, d_ff=1024, basis_rank=64):
        super().__init__()

        self.n_basis = n_basis
        self.d_model = d_model
        self.d_ff = d_ff
        self.basis_rank = basis_rank

        # Basis를 TT 형태로 저장
        # Basis_A: [256, 64] → TT [16, rank, 8] + [rank, 16, 8]
        self.basis_A_core1 = nn.Parameter(
            torch.randn(n_basis, 16, basis_rank, 8) * 0.02,
            requires_grad=False  # v7.4: 일단 고정 (학습 가능 버전은 v7.5)
        )
        self.basis_A_core2 = nn.Parameter(
            torch.randn(n_basis, basis_rank, 16, 8) * 0.02,
            requires_grad=False
        )

        # Basis_B: [64, 1024] → TT [8, rank, 32] + [rank, 8, 32]
        self.basis_B_core1 = nn.Parameter(
            torch.randn(n_basis, 8, basis_rank, 32) * 0.02,
            requires_grad=False
        )
        self.basis_B_core2 = nn.Parameter(
            torch.randn(n_basis, basis_rank, 8, 32) * 0.02,
            requires_grad=False
        )

        # Basis embeddings (routing용)
        self.basis_emb = nn.Parameter(
            torch.randn(n_basis, d_model) * 0.02,
            requires_grad=False
        )

        # 직교 초기화
        self._init_orthogonal()

        # Karcher mean module
        self.karcher = TTKarcherMean(max_iter=5, tolerance=1e-6)

    def _init_orthogonal(self):
        """초기 직교화 - row-wise normalization for TT cores"""
        with torch.no_grad():
            # Basis_A cores: [n_basis, 16, rank, 8] and [n_basis, rank, 16, 8]
            # Normalize along appropriate dimensions
            self.basis_A_core1.data = F.normalize(self.basis_A_core1.data, dim=2)
            self.basis_A_core2.data = F.normalize(self.basis_A_core2.data, dim=1)

            # Basis_B cores: [n_basis, 8, rank, 32] and [n_basis, rank, 8, 32]
            self.basis_B_core1.data = F.normalize(self.basis_B_core1.data, dim=2)
            self.basis_B_core2.data = F.normalize(self.basis_B_core2.data, dim=1)

    def get_neuron_tt_cores(self, neuron_recipe):
        """
        Neuron recipe로 TT cores 조합

        Args:
            neuron_recipe: [B, S, n_basis]
        Returns:
            cores_A: Dict with core1, core2
            cores_B: Dict with core1, core2
        """
        # Recipe로 basis cores 가중 조합
        cores_A = {
            'core1': torch.einsum('bsn,nirk->bsirk', neuron_recipe, self.basis_A_core1),
            'core2': torch.einsum('bsn,nrjl->bsrjl', neuron_recipe, self.basis_A_core2)
        }

        cores_B = {
            'core1': torch.einsum('bsn,nirk->bsirk', neuron_recipe, self.basis_B_core1),
            'core2': torch.einsum('bsn,nrjl->bsrjl', neuron_recipe, self.basis_B_core2)
        }

        return cores_A, cores_B
