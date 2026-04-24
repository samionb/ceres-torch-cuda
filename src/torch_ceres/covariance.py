from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .problem import ParameterBlock, Problem
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
            pairs = [(problem._coerce_parameter_block(a), problem._coerce_parameter_block(b)) for a, b in covariance_blocks]  # type: ignore[misc]
        else:
            params = [problem._coerce_parameter_block(p) for p in covariance_blocks]  # type: ignore[assignment]
            pairs = [(a, b) for a in params for b in params]
        self._blocks = set(pairs) | {(b, a) for a, b in pairs}
        evaluation = problem.evaluate(compute_jacobian=True)
        if evaluation.jacobian is None:
            return False
        J = evaluation.jacobian
        self._slices = problem.parameter_tangent_slices(active_only=True)
        if J.shape[1] == 0:
            self._tangent_covariance = J.new_zeros((0, 0))
            return True
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        if S.numel() == 0:
            return False
        threshold = self._svd_threshold(S, J.shape)
        inv_s2 = torch.where(S > threshold, 1.0 / (S * S), torch.zeros_like(S))
        if self.options.null_space_rank > 0 and self.options.null_space_rank < inv_s2.numel():
            inv_s2[-self.options.null_space_rank :] = 0.0
        self._tangent_covariance = (Vh.T * inv_s2) @ Vh
        if torch.any((S / S.max()) < self.options.min_reciprocal_condition_number) and self.options.null_space_rank == 0:
            return False
        return True

    def get_covariance_block(self, a: ParameterBlock | torch.Tensor, b: ParameterBlock | torch.Tensor) -> torch.Tensor:
        block = self._get_tangent_block(a, b)
        assert self._problem is not None
        pa = self._problem._coerce_parameter_block(a)
        pb = self._problem._coerce_parameter_block(b)
        Ja = pa.manifold.plus_jacobian(pa.tensor.detach().reshape(-1)).to(dtype=block.dtype, device=block.device)
        Jb = pb.manifold.plus_jacobian(pb.tensor.detach().reshape(-1)).to(dtype=block.dtype, device=block.device)
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
        pa = self._problem._coerce_parameter_block(a)
        pb = self._problem._coerce_parameter_block(b)
        if (pa, pb) not in self._blocks:
            raise KeyError("Requested covariance block was not computed")
        sa = self._slices.get(pa, slice(0, 0))
        sb = self._slices.get(pb, slice(0, 0))
        return self._tangent_covariance[sa, sb]

    def _svd_threshold(self, singular_values: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        if self.options.column_pivot_threshold >= 0:
            return singular_values.new_tensor(self.options.column_pivot_threshold)
        return 20.0 * (shape[0] + shape[1]) * torch.finfo(singular_values.dtype).eps * singular_values.max()

