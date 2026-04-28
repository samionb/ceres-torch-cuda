from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .linear import OptionalBackendUnavailable, get_optional_backend
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

    def validate(self) -> None:
        checks = [
            (self.column_pivot_threshold >= -1.0, "column_pivot_threshold must be >= -1"),
            (self.min_reciprocal_condition_number >= 0.0, "min_reciprocal_condition_number must be >= 0"),
            (self.null_space_rank >= -1, "null_space_rank must be >= -1"),
            (self.num_threads > 0, "num_threads must be > 0"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)


@dataclass
class CovarianceSummary:
    algorithm_type: CovarianceAlgorithmType
    success: bool = False
    message: str = "Covariance.compute was not called."
    rank: int = 0
    nullity: int = 0
    num_rows: int = 0
    num_columns: int = 0
    num_requested_blocks: int = 0
    num_computed_blocks: int = 0
    ambient_dimension: int = 0
    tangent_dimension: int = 0
    max_singular_value: float = 0.0
    min_retained_singular_value: float = 0.0
    reciprocal_condition_number: float = 0.0
    requested_null_space_rank: int = 0

    def BriefReport(self) -> str:
        return (
            "ceres-torch Covariance Report: "
            f"Algorithm: {self.algorithm_type.value}, "
            f"Rank: {self.rank}, "
            f"Nullity: {self.nullity}, "
            f"Success: {self.success}"
        )

    def FullReport(self) -> str:
        return "\n".join(
            [
                "Covariance Summary (ceres-torch)",
                "",
                f"Algorithm: {self.algorithm_type.value}",
                f"Success: {self.success}",
                f"Message: {self.message}",
                f"Rows: {self.num_rows}",
                f"Columns: {self.num_columns}",
                f"Ambient dimension: {self.ambient_dimension}",
                f"Tangent dimension: {self.tangent_dimension}",
                f"Requested blocks: {self.num_requested_blocks}",
                f"Computed blocks: {self.num_computed_blocks}",
                f"Rank: {self.rank}",
                f"Nullity: {self.nullity}",
                f"Requested null space rank: {self.requested_null_space_rank}",
                f"Max singular value: {self.max_singular_value:.12e}",
                f"Min retained singular value: {self.min_retained_singular_value:.12e}",
                f"Reciprocal condition number: {self.reciprocal_condition_number:.12e}",
            ]
        )


class Covariance:
    def __init__(self, options: Optional[CovarianceOptions] = None) -> None:
        self.options = options or CovarianceOptions()
        self._problem: Optional[Problem] = None
        self._tangent_covariance: Optional[torch.Tensor] = None
        self._blocks: set[tuple[ParameterBlock, ParameterBlock]] = set()
        self._slices: dict[ParameterBlock, slice] = {}
        self.summary = CovarianceSummary(
            algorithm_type=self.options.algorithm_type,
            requested_null_space_rank=self.options.null_space_rank,
        )

    def compute(
        self,
        covariance_blocks: Sequence[tuple[ParameterBlock | torch.Tensor, ParameterBlock | torch.Tensor]]
        | Sequence[ParameterBlock | torch.Tensor],
        problem: Problem,
    ) -> bool:
        self.options.validate()
        self.summary = CovarianceSummary(
            algorithm_type=self.options.algorithm_type,
            requested_null_space_rank=self.options.null_space_rank,
        )
        self._problem = problem
        if not covariance_blocks:
            self._blocks = set()
            self.summary.success = True
            self.summary.message = "No covariance blocks requested."
            self.summary.ambient_dimension = problem.num_parameters()
            self.summary.tangent_dimension = problem.num_effective_parameters()
            return True
        pairs = self._normalize_covariance_blocks(covariance_blocks, problem)
        self._blocks = set(pairs) | {(b, a) for a, b in pairs}
        evaluation = problem.evaluate(
            EvaluateOptions(apply_loss_function=self.options.apply_loss_function),
            compute_jacobian=True,
        )
        if evaluation.jacobian is None:
            self.summary.message = "Problem evaluation did not produce a Jacobian."
            return False
        J = evaluation.jacobian
        self._slices = problem.parameter_tangent_slices(active_only=True)
        self.summary.num_requested_blocks = len(pairs)
        self.summary.num_computed_blocks = len(self._blocks)
        self.summary.ambient_dimension = problem.num_parameters()
        self.summary.tangent_dimension = J.shape[1]
        if J.shape[1] == 0:
            self._tangent_covariance = J.new_zeros((0, 0))
            self._set_summary_from_singular_values(
                J.new_zeros(0),
                J.shape,
                success=True,
                message="No active tangent parameters.",
            )
            return True
        if self.options.algorithm_type is CovarianceAlgorithmType.SPARSE_QR:
            backend = get_optional_backend("sparse_qr_covariance")
            if backend is not None:
                try:
                    self._tangent_covariance = backend(J, options=self.options, slices=self._slices)  # type: ignore[assignment]
                    if self._tangent_covariance.shape != (J.shape[1], J.shape[1]):
                        self.summary.message = (
                            "Optional sparse QR backend returned covariance with shape "
                            f"{tuple(self._tangent_covariance.shape)}, expected {(J.shape[1], J.shape[1])}."
                        )
                        self._tangent_covariance = None
                        return False
                    self._set_summary_from_singular_values(
                        torch.linalg.svdvals(J),
                        J.shape,
                        success=True,
                        message="Computed covariance with optional sparse QR backend.",
                    )
                    return True
                except OptionalBackendUnavailable:
                    pass
            return self._compute_qr_covariance(J)
        return self._compute_svd_covariance(J)

    Compute = compute

    def _compute_svd_covariance(self, J: torch.Tensor) -> bool:
        _, S, Vh = torch.linalg.svd(J, full_matrices=False)
        if S.numel() == 0:
            self.summary.message = "SVD returned no singular values."
            return False
        threshold = self._svd_threshold(S, J.shape)
        keep = S > threshold
        eigen_ratios = _eigenvalue_ratios_from_singular_values(S)
        if self.options.null_space_rank == -1:
            keep = keep & (eigen_ratios >= self.options.min_reciprocal_condition_number)
        elif self.options.null_space_rank > 0:
            if self.options.null_space_rank > S.numel():
                self._set_summary_from_singular_values(
                    S,
                    J.shape,
                    success=False,
                    message="Requested null_space_rank exceeds the number of singular values.",
                    keep=keep,
                )
                self._tangent_covariance = (Vh.T * torch.zeros_like(S)) @ Vh
                return False
            keep = keep.clone()
            keep[-self.options.null_space_rank :] = False
        inv_s2 = torch.where(keep, 1.0 / (S * S), torch.zeros_like(S))
        self._tangent_covariance = (Vh.T * inv_s2) @ Vh
        rank = int(torch.sum(keep).detach().cpu())
        if self.options.null_space_rank == 0 and rank < J.shape[1]:
            self._set_summary_from_singular_values(
                S,
                J.shape,
                success=False,
                message="Jacobian is rank deficient.",
                keep=keep,
            )
            return False
        if torch.any(eigen_ratios[keep] < self.options.min_reciprocal_condition_number):
            self._set_summary_from_singular_values(
                S,
                J.shape,
                success=False,
                message="Retained covariance spectrum is below min_reciprocal_condition_number.",
                keep=keep,
            )
            return False
        self._set_summary_from_singular_values(
            S,
            J.shape,
            success=True,
            message="Computed covariance with dense SVD.",
            keep=keep,
        )
        return True

    def _compute_qr_covariance(self, J: torch.Tensor) -> bool:
        if J.shape[0] < J.shape[1]:
            self.summary = CovarianceSummary(
                algorithm_type=self.options.algorithm_type,
                success=False,
                message="Sparse QR covariance requires rows >= columns and cannot use null_space_rank.",
                rank=J.shape[0],
                nullity=J.shape[1] - J.shape[0],
                num_rows=J.shape[0],
                num_columns=J.shape[1],
                num_requested_blocks=self.summary.num_requested_blocks,
                num_computed_blocks=self.summary.num_computed_blocks,
                ambient_dimension=self.summary.ambient_dimension,
                tangent_dimension=self.summary.tangent_dimension,
                requested_null_space_rank=self.options.null_space_rank,
            )
            return False
        _, R = torch.linalg.qr(J, mode="reduced")
        diag = torch.abs(torch.diagonal(R))
        if diag.numel() == 0:
            self.summary.message = "QR returned an empty diagonal."
            return False
        threshold = self._qr_threshold(diag, J.shape)
        rank = int(torch.sum(diag > threshold).detach().cpu())
        if rank < J.shape[1]:
            self._set_summary_from_singular_values(
                torch.linalg.svdvals(J),
                J.shape,
                success=False,
                message="Sparse QR detected a rank deficient Jacobian and cannot use null_space_rank.",
            )
            return False
        singular_values = torch.linalg.svdvals(J)
        if torch.any(_eigenvalue_ratios_from_singular_values(singular_values) < self.options.min_reciprocal_condition_number) and self.options.null_space_rank == 0:
            self._set_summary_from_singular_values(
                singular_values,
                J.shape,
                success=False,
                message="Sparse QR spectrum is below min_reciprocal_condition_number.",
            )
            return False
        eye = torch.eye(R.shape[1], dtype=R.dtype, device=R.device)
        R_inv = torch.linalg.solve_triangular(R, eye, upper=True)
        self._tangent_covariance = R_inv @ R_inv.T
        self._set_summary_from_singular_values(
            singular_values,
            J.shape,
            success=True,
            message="Computed covariance with QR.",
        )
        return True

    def get_covariance_block(
        self,
        a: ParameterBlock | torch.Tensor,
        b: ParameterBlock | torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        block = self._get_tangent_block(a, b)
        assert self._problem is not None
        pa = self._problem._require_parameter_block(a)
        pb = self._problem._require_parameter_block(b)
        Ja = self._ambient_to_tangent_basis(pa, dtype=block.dtype, device=block.device)
        Jb = self._ambient_to_tangent_basis(pb, dtype=block.dtype, device=block.device)
        return _copy_to_output(Ja @ block @ Jb.T, out)

    def get_covariance_block_in_tangent_space(
        self,
        a: ParameterBlock | torch.Tensor,
        b: ParameterBlock | torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _copy_to_output(self._get_tangent_block(a, b), out)

    def get_covariance_matrix(
        self,
        parameter_blocks: Sequence[ParameterBlock | torch.Tensor],
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows = [torch.cat([self.get_covariance_block(a, b) for b in parameter_blocks], dim=1) for a in parameter_blocks]
        return _copy_to_output(torch.cat(rows, dim=0), out)

    def get_covariance_matrix_in_tangent_space(
        self,
        parameter_blocks: Sequence[ParameterBlock | torch.Tensor],
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows = [
            torch.cat([self.get_covariance_block_in_tangent_space(a, b) for b in parameter_blocks], dim=1)
            for a in parameter_blocks
        ]
        return _copy_to_output(torch.cat(rows, dim=0), out)

    GetCovarianceBlock = get_covariance_block
    GetCovarianceBlockInTangentSpace = get_covariance_block_in_tangent_space
    GetCovarianceMatrix = get_covariance_matrix
    GetCovarianceMatrixInTangentSpace = get_covariance_matrix_in_tangent_space

    def has_covariance_block(self, a: ParameterBlock | torch.Tensor, b: ParameterBlock | torch.Tensor) -> bool:
        if self._problem is None:
            return False
        try:
            pa = self._problem._require_parameter_block(a)
            pb = self._problem._require_parameter_block(b)
        except KeyError:
            return False
        return (pa, pb) in self._blocks

    HasCovarianceBlock = has_covariance_block

    def rank(self) -> int:
        return self.summary.rank

    def nullity(self) -> int:
        return self.summary.nullity

    def reciprocal_condition_number(self) -> float:
        return self.summary.reciprocal_condition_number

    Rank = rank
    Nullity = nullity
    ReciprocalConditionNumber = reciprocal_condition_number

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

    def _set_summary_from_singular_values(
        self,
        singular_values: torch.Tensor,
        shape: tuple[int, int],
        *,
        success: bool,
        message: str,
        keep: torch.Tensor | None = None,
    ) -> None:
        if singular_values.numel() == 0:
            rank = 0
            max_sigma = 0.0
            min_retained = 0.0
            reciprocal_condition = 0.0
        else:
            if keep is None:
                threshold = self._svd_threshold(singular_values, shape)
                keep = singular_values > threshold
            rank = int(torch.sum(keep).detach().cpu())
            max_sigma_t = singular_values.max()
            max_sigma = float(max_sigma_t.detach().cpu())
            retained = singular_values[keep]
            min_retained = float(retained.min().detach().cpu()) if retained.numel() else 0.0
            if retained.numel() and max_sigma > 0.0:
                reciprocal_condition = float(((retained.min() / max_sigma_t) ** 2).detach().cpu())
            else:
                reciprocal_condition = 0.0
        self.summary = CovarianceSummary(
            algorithm_type=self.options.algorithm_type,
            success=success,
            message=message,
            rank=rank,
            nullity=max(0, shape[1] - rank),
            num_rows=shape[0],
            num_columns=shape[1],
            num_requested_blocks=self.summary.num_requested_blocks,
            num_computed_blocks=self.summary.num_computed_blocks,
            ambient_dimension=self.summary.ambient_dimension,
            tangent_dimension=self.summary.tangent_dimension,
            max_singular_value=max_sigma,
            min_retained_singular_value=min_retained,
            reciprocal_condition_number=reciprocal_condition,
            requested_null_space_rank=self.options.null_space_rank,
        )

    def _normalize_covariance_blocks(
        self,
        covariance_blocks: Sequence[tuple[ParameterBlock | torch.Tensor, ParameterBlock | torch.Tensor]]
        | Sequence[ParameterBlock | torch.Tensor],
        problem: Problem,
    ) -> list[tuple[ParameterBlock, ParameterBlock]]:
        first = covariance_blocks[0]  # type: ignore[index]
        if isinstance(first, tuple):
            pairs = [(problem._require_parameter_block(a), problem._require_parameter_block(b)) for a, b in covariance_blocks]  # type: ignore[misc]
            seen: set[frozenset[ParameterBlock]] = set()
            for a, b in pairs:
                key = frozenset((a, b))
                if key in seen:
                    raise ValueError("covariance_blocks cannot contain duplicate block pairs")
                seen.add(key)
            return pairs
        params = [problem._require_parameter_block(p) for p in covariance_blocks]  # type: ignore[assignment]
        if len(set(params)) != len(params):
            raise ValueError("parameter_blocks cannot contain duplicates")
        return [(a, b) for a in params for b in params]


def _eigenvalue_ratios_from_singular_values(singular_values: torch.Tensor) -> torch.Tensor:
    if singular_values.numel() == 0:
        return singular_values
    max_s = singular_values.max().clamp_min(torch.finfo(singular_values.dtype).tiny)
    return (singular_values / max_s) ** 2


def _copy_to_output(value: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        return value
    if out.numel() != value.numel():
        raise ValueError("Output tensor has the wrong number of elements")
    out.reshape(-1).copy_(value.reshape(-1).to(dtype=out.dtype, device=out.device))
    return out
