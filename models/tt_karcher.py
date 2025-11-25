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
    Weighted Karcher Mean on TT Manifold

    8개 Neuron의 TT 표현을 weight로 조합
    → 균형점(centroid) 찾기
    """

    def __init__(self, max_iter=5, tolerance=1e-6, step_size=0.5):
        super().__init__()

        self.max_iter = max_iter
        self.tolerance = tolerance
        self.step_size = step_size

    def forward(self, neuron_cores_A, neuron_cores_B, weights):
        """
        Args:
            neuron_cores_A: List of 8 TT cores for Basis_A
                Each: {'core1': [B, S, 16, rank, 8],
                       'core2': [B, S, rank, 16, 8]}
            neuron_cores_B: List of 8 TT cores for Basis_B
                Each: {'core1': [B, S, 8, rank, 32],
                       'core2': [B, S, rank, 8, 32]}
            weights: [B, S, 8] neuron weights (softmax)

        Returns:
            centroid_A: Weighted centroid TT for Basis_A
            centroid_B: Weighted centroid TT for Basis_B
        """
        # Find centroid for Basis_A
        centroid_A = self.find_centroid(neuron_cores_A, weights)

        # Find centroid for Basis_B
        centroid_B = self.find_centroid(neuron_cores_B, weights)

        return centroid_A, centroid_B

    def find_centroid(self, tt_cores_list, weights):
        """
        TT manifold에서 weighted centroid 찾기

        Args:
            tt_cores_list: List[Dict] - 8개 TT cores
            weights: [B, S, 8]
        """
        # 초기 추정: weighted average
        center = self.weighted_average_init(tt_cores_list, weights)

        # Iterative refinement
        for iteration in range(self.max_iter):
            # 각 neuron에서 center로의 tangent
            tangents = []
            for i, tt_cores in enumerate(tt_cores_list):
                w = weights[:, :, i]  # [B, S]
                tangent = self.compute_tangent(center, tt_cores)

                # Weight 적용
                weighted_tangent = {
                    'core1': w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * tangent['core1'],
                    'core2': w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * tangent['core2']
                }
                tangents.append(weighted_tangent)

            # Total tangent (합력)
            total_tangent = self.sum_tangents(tangents)

            # Convergence check
            tangent_norm = self.compute_tangent_norm(total_tangent)
            if tangent_norm < self.tolerance:
                break

            # Move center along tangent
            step = self.step_size / (iteration + 1)  # Decreasing step
            center = self.move_along_tangent(center, total_tangent, step)

        return center

    def weighted_average_init(self, tt_cores_list, weights):
        """
        초기 추정: weighted average of cores
        """
        B, S = weights.shape[:2]

        # Weighted sum of cores
        core1_sum = sum(
            weights[:, :, i].view(B, S, 1, 1, 1) * tt['core1']
            for i, tt in enumerate(tt_cores_list)
        )
        core2_sum = sum(
            weights[:, :, i].view(B, S, 1, 1, 1) * tt['core2']
            for i, tt in enumerate(tt_cores_list)
        )

        # Orthogonalize
        core1_orth = self.orthogonalize_core(core1_sum)
        core2_orth = self.orthogonalize_core(core2_sum)

        return {'core1': core1_orth, 'core2': core2_orth}

    def compute_tangent(self, center, target):
        """
        Center에서 target으로의 tangent vector
        """
        # 간단 버전: target - center
        tangent = {
            'core1': target['core1'] - center['core1'],
            'core2': target['core2'] - center['core2']
        }
        return tangent

    def sum_tangents(self, tangents):
        """
        여러 tangent의 합
        """
        total = {
            'core1': sum(t['core1'] for t in tangents),
            'core2': sum(t['core2'] for t in tangents)
        }
        return total

    def compute_tangent_norm(self, tangent):
        """
        Tangent의 norm (수렴 판단용)
        """
        norm1 = torch.norm(tangent['core1'])
        norm2 = torch.norm(tangent['core2'])
        return (norm1 + norm2).item()

    def move_along_tangent(self, center, tangent, step_size):
        """
        Center를 tangent 방향으로 이동
        """
        new_center = {
            'core1': center['core1'] + step_size * tangent['core1'],
            'core2': center['core2'] + step_size * tangent['core2']
        }

        # Orthogonalize to stay on manifold
        new_center['core1'] = self.orthogonalize_core(new_center['core1'])
        new_center['core2'] = self.orthogonalize_core(new_center['core2'])

        return new_center

    def orthogonalize_core(self, core):
        """
        Core를 직교화 (TT manifold 유지)
        """
        shape = core.shape
        # Reshape for QR
        # [..., d1, rank, d2] → [..., d1*rank, d2]
        core_2d = core.reshape(*shape[:-3], shape[-3] * shape[-2], shape[-1])

        # QR decomposition
        Q, R = torch.linalg.qr(core_2d)

        # Reshape back
        Q = Q.reshape(*shape[:-3], shape[-3], shape[-2], shape[-1])

        return Q


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
