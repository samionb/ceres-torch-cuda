from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .linear import get_optional_backend
from .problem import EvaluateOptions, ParameterBlock, Problem
from .types import CovarianceAlgorithmType, SparseLinearAlgebraLibraryType


@dataclass
class CovarianceOptions:
    sparse_linear_algebra_library_type: SparseLinearAlgebraLibraryType = SparseLinearAlgebraLibraryType.NO_SPARSE
    algorithm_type: CovarianceAlgorithmType = CovarianceAlgorithmType.DENSE_SVD
    column_pivot_threshold: float = -1.0
    min_reciprocal_condition_number: float = 1e-14
    null_space_rank: int = 0
    num_threads: int = 1
    apply_loss_function: bool = True


class Covariance:
    def __init__(self, options: Optional[CovarianceOptions] = None) -> None:
        self.options = options or CovarianceOptions()
        self._problem: Optional[Problem] = None
        self._tangent_covariance: Optional[torch.Tensor] = None
        self._blocks: set[tuple[ParameterBlock, ParameterBlock]] = set()
        self._slices: dict[ParameterBlock, slice] = {}

    def compute(
        self,
        covariance_blocks: Sequence[tuple[ParameterBlock | torch.Tensor, ParameterBlock | torch.Tensor]]
        | Sequence[ParameterBlock | torch.Tensor],
        problem: Problem,
    ) -> bool:
        self._problem = problem
        if not covariance_blocks:
            self._blocks = set()
            return True
        first = covariance_blocks[0]  # type: ignore[index]
        if isinstance(first, tuple):
            pairs = [(problem._require_parameter_block(a), problem._require_parameter_block(b)) for a, b in covariance_blocks]  # type: ignore[misc]
        else:
            params = [problem._require_parameter_block(p) for p in covariance_blocks]  # type: ignore[assignment]
            pairs = [(a, b) for a in params for b in params]
        self._blocks = set(pairs) | {(b, a) for a, b in pairs}
        evaluation = problem.evaluate(
            EvaluateOptions(apply_loss_function=self.options.apply_loss_function),
            compute_jacobian=True,
        )
        if evaluation.jacobian is None:
            return False
        J = evaluation.jacobian
        self._slices = problem.parameter_tangent_slices(active_only=True)
        if J.shape[1] == 0:
            self._tangent_covariance = J.new_zeros((0, 0))
            return True
        if self.options.algorithm_type is CovarianceAlgorithmType.SPARSE_QR:
            backend = get_optional_backend("sparse_qr_covariance")
            if backend is not None:
                self._tangent_covariance = backend(J, options=self.options, slices=self._slices)  # type: ignore[assignment]
                return True
            return self._compute_qr_covariance(J)
        return self._compute_svd_covariance(J)

    def _compute_svd_covariance(self, J: torch.Tensor) -> bool:
        _, S, Vh = torch.linalg.svd(J, full_matrices=False)
        if S.numel() == 0:
            return False
        threshold = self._svd_threshold(S, J.shape)
        rank = int(torch.sum(S > threshold).detach().cpu())
        inv_s2 = torch.where(S > threshold, 1.0 / (S * S), torch.zeros_like(S))
        if self.options.null_space_rank > 0 and self.options.null_space_rank < inv_s2.numel():
            inv_s2[-self.options.null_space_rank :] = 0.0
        self._tangent_covariance = (Vh.T * inv_s2) @ Vh
        if rank < J.shape[1] and self.options.null_space_rank == 0:
            return False
        if torch.any((S / S.max()) < self.options.min_reciprocal_condition_number) and self.options.null_space_rank == 0:
            return False
        return True

    def _compute_qr_covariance(self, J: torch.Tensor) -> bool:
        if J.shape[0] < J.shape[1]:
            if self.options.null_space_rank != 0:
                return self._compute_svd_covariance(J)
            return False
        _, R = torch.linalg.qr(J, mode="reduced")
        diag = torch.abs(torch.diagonal(R))
        if diag.numel() == 0:
            return False
        threshold = self._qr_threshold(diag, J.shape)
        rank = int(torch.sum(diag > threshold).detach().cpu())
        if rank < J.shape[1]:
            if self.options.null_space_rank != 0:
                return self._compute_svd_covariance(J)
            return False
        if torch.any((diag / diag.max()) < self.options.min_reciprocal_condition_number) and self.options.null_space_rank == 0:
            return False
        eye = torch.eye(R.shape[1], dtype=R.dtype, device=R.device)
        R_inv = torch.linalg.solve_triangular(R, eye, upper=True)
        self._tangent_covariance = R_inv @ R_inv.T
        return True

    def get_covariance_block(self, a: ParameterBlock | torch.Tensor, b: ParameterBlock | torch.Tensor) -> torch.Tensor:
        block = self._get_tangent_block(a, b)
        assert self._problem is not None
        pa = self._problem._require_parameter_block(a)
        pb = self._problem._require_parameter_block(b)
        Ja = self._ambient_to_tangent_basis(pa, dtype=block.dtype, device=block.device)
        Jb = self._ambient_to_tangent_basis(pb, dtype=block.dtype, device=block.device)
        return Ja @ block @ Jb.T

    def get_covariance_block_in_tangent_space(
        self, a: ParameterBlock | torch.Tensor, b: ParameterBlock | torch.Tensor
    ) -> torch.Tensor:
        return self._get_tangent_block(a, b)

    def get_covariance_matrix(self, parameter_blocks: Sequence[ParameterBlock | torch.Tensor]) -> torch.Tensor:
        rows = [torch.cat([self.get_covariance_block(a, b) for b in parameter_blocks], dim=1) for a in parameter_blocks]
        return torch.cat(rows, dim=0)

    def get_covariance_matrix_in_tangent_space(self, parameter_blocks: Sequence[ParameterBlock | torch.Tensor]) -> torch.Tensor:
        rows = [
            torch.cat([self.get_covariance_block_in_tangent_space(a, b) for b in parameter_blocks], dim=1)
            for a in parameter_blocks
        ]
        return torch.cat(rows, dim=0)

    def _get_tangent_block(self, a: ParameterBlock | torch.Tensor, b: ParameterBlock | torch.Tensor) -> torch.Tensor:
        if self._problem is None or self._tangent_covariance is None:
            raise RuntimeError("Covariance.compute must be called first")
        pa = self._problem._require_parameter_block(a)
        pb = self._problem._require_parameter_block(b)
        if (pa, pb) not in self._blocks:
            raise KeyError("Requested covariance block was not computed")
        sa = self._slices.get(pa, slice(0, 0))
        sb = self._slices.get(pb, slice(0, 0))
        return self._tangent_covariance[sa, sb]

    def _ambient_to_tangent_basis(
        self,
        block: ParameterBlock,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if block.tangent_size == 0:
            return torch.zeros((block.size, 0), dtype=dtype, device=device)
        return block.manifold.plus_jacobian(block.tensor.detach().reshape(-1)).to(dtype=dtype, device=device)

    def _svd_threshold(self, singular_values: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        if self.options.column_pivot_threshold >= 0:
            return singular_values.new_tensor(self.options.column_pivot_threshold)
        return 20.0 * (shape[0] + shape[1]) * torch.finfo(singular_values.dtype).eps * singular_values.max()

    def _qr_threshold(self, diagonal: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        if self.options.column_pivot_threshold >= 0:
            return diagonal.new_tensor(self.options.column_pivot_threshold)
        return 20.0 * (shape[0] + shape[1]) * torch.finfo(diagonal.dtype).eps * diagonal.max()
