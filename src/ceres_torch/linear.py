from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Callable, Iterable, Optional, Sequence

import torch

from .types import DoglegType, LinearSolverType, PreconditionerType, VisibilityClusteringType


@dataclass
class LinearSolverSummary:
    residual_norm: float
    num_iterations: int
    success: bool
    message: str = ""


@dataclass
class LinearSolverResult:
    x: torch.Tensor
    summary: LinearSolverSummary


@dataclass(frozen=True)
class VisibilityClusterStructure:
    visibility_clustering_type: VisibilityClusteringType
    preconditioner_type: PreconditionerType
    membership: tuple[int, ...]
    cluster_pairs: tuple[tuple[int, int], ...]
    block_pairs: tuple[tuple[int, int], ...]
    num_clusters: int
    camera_graph_edges: tuple[tuple[int, int, float], ...] = ()
    cluster_graph_edges: tuple[tuple[int, int, float], ...] = ()


@dataclass
class NormalEquationPreconditioner:
    preconditioner_type: PreconditionerType
    message: str
    block_sizes: tuple[int, ...] = ()
    diagonal_inverse: Optional[torch.Tensor] = None
    block_factors: Optional[list[tuple[slice, torch.Tensor, bool]]] = None
    power_series_matrix: Optional[torch.Tensor] = None
    base_preconditioner: Optional["NormalEquationPreconditioner"] = None
    max_power_series_iterations: int = 0
    power_series_tolerance: float = 0.0
    matrix_factor: Optional[torch.Tensor] = None
    matrix_is_cholesky: bool = False
    visibility_structure: Optional[VisibilityClusterStructure] = None

    def apply(self, residual: torch.Tensor) -> torch.Tensor:
        if self.preconditioner_type is PreconditionerType.IDENTITY:
            return residual
        if self.matrix_factor is not None:
            if self.matrix_is_cholesky:
                return torch.cholesky_solve(residual.reshape(-1, 1), self.matrix_factor).reshape(-1)
            return _solve_square_system(self.matrix_factor, residual)
        if self.power_series_matrix is not None and self.base_preconditioner is not None:
            result = self.base_preconditioner.apply(residual)
            term = result
            for _ in range(max(0, self.max_power_series_iterations - 1)):
                term = self.base_preconditioner.apply(self.power_series_matrix @ term)
                if torch.linalg.norm(term) <= self.power_series_tolerance * torch.linalg.norm(result).clamp_min(1.0):
                    break
                result = result + term
            return result
        if self.block_factors is not None:
            z = torch.zeros_like(residual)
            for block_slice, factor, is_cholesky in self.block_factors:
                block_residual = residual[block_slice]
                if is_cholesky:
                    z[block_slice] = torch.cholesky_solve(block_residual.reshape(-1, 1), factor).reshape(-1)
                else:
                    z[block_slice] = _solve_square_system(factor, block_residual)
            return z
        assert self.diagonal_inverse is not None
        return self.diagonal_inverse * residual


OptionalBackend = Callable[..., torch.Tensor | LinearSolverResult]
_optional_backends: dict[str, OptionalBackend] = {}


class OptionalBackendUnavailable(RuntimeError):
    pass


def register_optional_backend(name: str, backend: OptionalBackend) -> None:
    _optional_backends[name] = backend


def unregister_optional_backend(name: str) -> None:
    _optional_backends.pop(name, None)


def clear_optional_backends() -> None:
    _optional_backends.clear()


def get_optional_backend(name: str) -> OptionalBackend | None:
    return _optional_backends.get(name)


def solve_linear_system(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    solver_type: LinearSolverType = LinearSolverType.DENSE_QR,
    damping: Optional[torch.Tensor] = None,
    num_eliminate: int = 0,
    min_iterations: int = 0,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
    preconditioner_type: PreconditionerType = PreconditionerType.JACOBI,
    block_sizes: Optional[Sequence[int]] = None,
    use_mixed_precision: bool = False,
    max_refinement_iterations: int = 0,
    max_num_spse_iterations: int = 5,
    use_spse_initialization: bool = False,
    spse_tolerance: float = 0.1,
    visibility: Optional[Sequence[Iterable[int]]] = None,
    visibility_clustering_type: VisibilityClusteringType = VisibilityClusteringType.CANONICAL_VIEWS,
) -> LinearSolverResult:
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix")
    b = b.reshape(-1)
    if b.numel() != A.shape[0]:
        raise ValueError("b must have one entry per row of A")
    if A.shape[1] == 0:
        x = b.new_zeros(0)
        return LinearSolverResult(
            x,
            LinearSolverSummary(float(torch.linalg.norm(b).detach().cpu()), 0, True, "zero-column system"),
        )
    if A.shape[0] == 0:
        x = b.new_zeros(A.shape[1])
        return LinearSolverResult(x, LinearSolverSummary(0.0, 0, True, "empty-row system"))

    if solver_type is LinearSolverType.DENSE_SCHUR:
        backend_result = _try_optional_linear_backend(
            "dense_schur",
            A,
            b,
            damping=damping,
            num_eliminate=num_eliminate,
            solver_type=solver_type,
            block_sizes=block_sizes,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            tolerance=tolerance,
            preconditioner_type=preconditioner_type,
            max_num_spse_iterations=max_num_spse_iterations,
            use_spse_initialization=use_spse_initialization,
            spse_tolerance=spse_tolerance,
            visibility=visibility,
            visibility_clustering_type=visibility_clustering_type,
        )
        if backend_result is None:
            backend_result = _try_optional_linear_backend(
                "block_schur",
                A,
                b,
                damping=damping,
                num_eliminate=num_eliminate,
                solver_type=solver_type,
                block_sizes=block_sizes,
                min_iterations=min_iterations,
                max_iterations=max_iterations,
                tolerance=tolerance,
                preconditioner_type=preconditioner_type,
                max_num_spse_iterations=max_num_spse_iterations,
                use_spse_initialization=use_spse_initialization,
                spse_tolerance=spse_tolerance,
                visibility=visibility,
                visibility_clustering_type=visibility_clustering_type,
            )
        if backend_result is not None:
            return backend_result
        x = schur_solve_dense(A, b, num_eliminate, damping=damping)
        x = _refine_solution(
            A,
            b,
            x,
            damping=damping,
            max_refinement_iterations=max_refinement_iterations if use_mixed_precision else 0,
        )
        return _summarize_solution(A, b, x, 1, "dense schur")

    if solver_type is LinearSolverType.ITERATIVE_SCHUR and num_eliminate > 0:
        backend_result = _try_optional_linear_backend(
            "iterative_schur",
            A,
            b,
            damping=damping,
            num_eliminate=num_eliminate,
            solver_type=solver_type,
            block_sizes=block_sizes,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            tolerance=tolerance,
            preconditioner_type=preconditioner_type,
            max_num_spse_iterations=max_num_spse_iterations,
            use_spse_initialization=use_spse_initialization,
            spse_tolerance=spse_tolerance,
            visibility=visibility,
            visibility_clustering_type=visibility_clustering_type,
        )
        if backend_result is None:
            backend_result = _try_optional_linear_backend(
                "block_schur",
                A,
                b,
                damping=damping,
                num_eliminate=num_eliminate,
                solver_type=solver_type,
                block_sizes=block_sizes,
                min_iterations=min_iterations,
                max_iterations=max_iterations,
                tolerance=tolerance,
                preconditioner_type=preconditioner_type,
                max_num_spse_iterations=max_num_spse_iterations,
                use_spse_initialization=use_spse_initialization,
                spse_tolerance=spse_tolerance,
                visibility=visibility,
                visibility_clustering_type=visibility_clustering_type,
            )
        if backend_result is not None:
            return backend_result
        return iterative_schur_solve(
            A,
            b,
            num_eliminate,
            damping=damping,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            tolerance=tolerance,
            preconditioner_type=preconditioner_type,
            block_sizes=block_sizes,
            max_num_spse_iterations=max_num_spse_iterations,
            use_spse_initialization=use_spse_initialization,
            spse_tolerance=spse_tolerance,
            visibility=visibility,
            visibility_clustering_type=visibility_clustering_type,
        )

    if solver_type in {LinearSolverType.CGNR, LinearSolverType.ITERATIVE_SCHUR}:
        return conjugate_gradient_normal_equations(
            A,
            b,
            damping=damping,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            tolerance=tolerance,
            preconditioner_type=preconditioner_type,
            block_sizes=block_sizes,
        )

    if solver_type in {LinearSolverType.SPARSE_NORMAL_CHOLESKY, LinearSolverType.SPARSE_SCHUR}:
        backend_names = (
            ["sparse_schur", "block_schur", "sparse_cholesky"]
            if solver_type is LinearSolverType.SPARSE_SCHUR
            else ["sparse_normal_cholesky", "sparse_cholesky"]
        )
        for backend_name in backend_names:
            backend_result = _try_optional_linear_backend(
                backend_name,
                A,
                b,
                damping=damping,
                num_eliminate=num_eliminate,
                solver_type=solver_type,
                block_sizes=block_sizes,
                min_iterations=min_iterations,
                max_iterations=max_iterations,
                tolerance=tolerance,
                preconditioner_type=preconditioner_type,
                max_num_spse_iterations=max_num_spse_iterations,
                use_spse_initialization=use_spse_initialization,
                spse_tolerance=spse_tolerance,
                visibility=visibility,
                visibility_clustering_type=visibility_clustering_type,
            )
            if backend_result is not None:
                return backend_result
        solver_type = LinearSolverType.DENSE_SCHUR if solver_type is LinearSolverType.SPARSE_SCHUR else LinearSolverType.DENSE_NORMAL_CHOLESKY

    if solver_type is LinearSolverType.DENSE_SCHUR:
        backend_result = _try_optional_linear_backend(
            "dense_schur",
            A,
            b,
            damping=damping,
            num_eliminate=num_eliminate,
            solver_type=solver_type,
            block_sizes=block_sizes,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            tolerance=tolerance,
            preconditioner_type=preconditioner_type,
            max_num_spse_iterations=max_num_spse_iterations,
            use_spse_initialization=use_spse_initialization,
            spse_tolerance=spse_tolerance,
            visibility=visibility,
            visibility_clustering_type=visibility_clustering_type,
        )
        if backend_result is None:
            backend_result = _try_optional_linear_backend(
                "block_schur",
                A,
                b,
                damping=damping,
                num_eliminate=num_eliminate,
                solver_type=solver_type,
                block_sizes=block_sizes,
                min_iterations=min_iterations,
                max_iterations=max_iterations,
                tolerance=tolerance,
                preconditioner_type=preconditioner_type,
                max_num_spse_iterations=max_num_spse_iterations,
                use_spse_initialization=use_spse_initialization,
                spse_tolerance=spse_tolerance,
                visibility=visibility,
                visibility_clustering_type=visibility_clustering_type,
            )
        if backend_result is not None:
            return backend_result
        x = schur_solve_dense(A, b, num_eliminate, damping=damping)
        return _summarize_solution(A, b, x, 1, "dense schur")

    if solver_type is LinearSolverType.DENSE_NORMAL_CHOLESKY:
        lhs = A.T @ A
        rhs = A.T @ b
        if damping is not None:
            lhs = lhs + torch.diag(damping.reshape(-1))
        try:
            chol = torch.linalg.cholesky(lhs)
            x = torch.cholesky_solve(rhs.reshape(-1, 1), chol).reshape(-1)
            x = _refine_solution(
                A,
                b,
                x,
                damping=damping,
                max_refinement_iterations=max_refinement_iterations if use_mixed_precision else 0,
            )
            return _summarize_solution(A, b, x, 1, "dense normal cholesky")
        except RuntimeError:
            x = torch.linalg.lstsq(lhs, rhs).solution.reshape(-1)
            x = _refine_solution(
                A,
                b,
                x,
                damping=damping,
                max_refinement_iterations=max_refinement_iterations if use_mixed_precision else 0,
            )
            return _summarize_solution(A, b, x, 1, "normal equations lstsq fallback")

    if damping is not None:
        D = torch.diag(torch.sqrt(torch.clamp(damping.reshape(-1), min=0.0)))
        A_aug = torch.cat([A, D], dim=0)
        b_aug = torch.cat([b.reshape(-1), b.new_zeros(A.shape[1])], dim=0)
    else:
        A_aug = A
        b_aug = b.reshape(-1)
    if use_mixed_precision and A_aug.dtype is torch.float64:
        x = torch.linalg.lstsq(A_aug.float(), b_aug.float()).solution.to(dtype=A_aug.dtype).reshape(-1)
    else:
        x = torch.linalg.lstsq(A_aug, b_aug).solution.reshape(-1)
    x = _refine_augmented_solution(A_aug, b_aug, x, max_refinement_iterations if use_mixed_precision else 0)
    return _summarize_solution(A_aug, b_aug, x, 1, "dense qr/lstsq")


def conjugate_gradient_normal_equations(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    damping: Optional[torch.Tensor],
    max_iterations: int,
    tolerance: float,
    preconditioner_type: PreconditionerType,
    block_sizes: Optional[Sequence[int]] = None,
    min_iterations: int = 0,
) -> LinearSolverResult:
    rhs = A.T @ b.reshape(-1)
    diag = damping.reshape(-1) if damping is not None else torch.zeros(A.shape[1], dtype=A.dtype, device=A.device)

    def matvec(x: torch.Tensor) -> torch.Tensor:
        return A.T @ (A @ x) + diag * x

    x = torch.zeros_like(rhs)
    r = rhs - matvec(x)
    preconditioner = build_normal_equation_preconditioner(
        A,
        damping=damping,
        preconditioner_type=preconditioner_type,
        block_sizes=block_sizes,
    )
    z = preconditioner.apply(r)
    p = z.clone()
    rz_old = torch.dot(r, z)
    b_norm = torch.linalg.norm(rhs).clamp_min(torch.finfo(A.dtype).eps)
    if min_iterations <= 0 and torch.linalg.norm(r) <= tolerance * b_norm:
        return _summarize_solution(A, b, x, 0, f"cgnr {preconditioner.message}", success=True)
    iterations = 0
    success = False
    for k in range(max_iterations):
        Ap = matvec(p)
        denom = torch.dot(p, Ap).clamp_min(torch.finfo(A.dtype).eps)
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        iterations = k + 1
        if iterations >= min_iterations and torch.linalg.norm(r) <= tolerance * b_norm:
            success = True
            break
        z = preconditioner.apply(r)
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old.clamp_min(torch.finfo(A.dtype).eps)
        p = z + beta * p
        rz_old = rz_new
    return _summarize_solution(A, b, x, iterations, f"cgnr {preconditioner.message}", success=success)


def build_normal_equation_preconditioner(
    A: torch.Tensor,
    *,
    damping: Optional[torch.Tensor] = None,
    preconditioner_type: PreconditionerType = PreconditionerType.JACOBI,
    block_sizes: Optional[Sequence[int]] = None,
) -> NormalEquationPreconditioner:
    if preconditioner_type is PreconditionerType.IDENTITY:
        return NormalEquationPreconditioner(preconditioner_type, "identity")

    diag = damping.reshape(-1) if damping is not None else torch.zeros(A.shape[1], dtype=A.dtype, device=A.device)
    normalized_block_sizes = _normalize_block_sizes(block_sizes, A.shape[1])
    H = A.T @ A + torch.diag(diag)
    if preconditioner_type is PreconditionerType.CLUSTER_TRIDIAGONAL and len(normalized_block_sizes) > 1:
        return _build_cluster_tridiagonal_preconditioner(
            H,
            preconditioner_type=preconditioner_type,
            block_sizes=normalized_block_sizes,
            message_prefix="cluster_tridiagonal",
        )
    if _uses_block_preconditioner(preconditioner_type) and any(size > 1 for size in normalized_block_sizes):
        return _build_matrix_preconditioner(
            H,
            preconditioner_type=preconditioner_type,
            block_sizes=normalized_block_sizes,
            message_prefix="block_jacobi",
        )

    diagonal = torch.diagonal(H)
    diagonal_inverse = 1.0 / torch.clamp(diagonal, min=torch.finfo(H.dtype).eps)
    return NormalEquationPreconditioner(
        preconditioner_type,
        f"diagonal/{preconditioner_type.value}",
        diagonal_inverse=diagonal_inverse,
    )


def build_schur_complement_preconditioner(
    A: torch.Tensor,
    *,
    damping: Optional[torch.Tensor] = None,
    num_eliminate: int,
    preconditioner_type: PreconditionerType = PreconditionerType.JACOBI,
    block_sizes: Optional[Sequence[int]] = None,
    max_num_spse_iterations: int = 5,
    spse_tolerance: float = 0.1,
    visibility: Optional[Sequence[Iterable[int]]] = None,
    visibility_clustering_type: VisibilityClusteringType = VisibilityClusteringType.CANONICAL_VIEWS,
) -> NormalEquationPreconditioner:
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix")
    if num_eliminate <= 0 or num_eliminate >= A.shape[1]:
        raise ValueError("num_eliminate must split eliminated and retained Schur columns")
    system = _build_schur_complement_system(A, damping=damping, num_eliminate=num_eliminate, b=None)
    retained_block_sizes = _remaining_block_sizes_after_elimination(block_sizes, num_eliminate, A.shape[1])
    if preconditioner_type is PreconditionerType.IDENTITY:
        return NormalEquationPreconditioner(preconditioner_type, "schur_identity")
    if preconditioner_type is PreconditionerType.SCHUR_POWER_SERIES_EXPANSION:
        base = _build_matrix_preconditioner(
            system.Hbb,
            preconditioner_type=PreconditionerType.SCHUR_JACOBI,
            block_sizes=retained_block_sizes,
            message_prefix="schur_power_series_base",
        )
        return NormalEquationPreconditioner(
            preconditioner_type,
            f"schur_power_series/{max_num_spse_iterations}",
            block_sizes=retained_block_sizes,
            power_series_matrix=system.schur_eliminated_term,
            base_preconditioner=base,
            max_power_series_iterations=max_num_spse_iterations,
            power_series_tolerance=spse_tolerance,
        )
    if preconditioner_type is PreconditionerType.CLUSTER_TRIDIAGONAL and len(retained_block_sizes) > 1:
        if visibility is not None:
            return _build_visibility_cluster_preconditioner(
                system.S,
                preconditioner_type=preconditioner_type,
                block_sizes=retained_block_sizes,
                visibility=visibility,
                visibility_clustering_type=visibility_clustering_type,
                message_prefix="schur_cluster_tridiagonal",
            )
        return _build_cluster_tridiagonal_preconditioner(
            system.S,
            preconditioner_type=preconditioner_type,
            block_sizes=retained_block_sizes,
            message_prefix="schur_cluster_tridiagonal",
        )
    if preconditioner_type is PreconditionerType.CLUSTER_JACOBI and visibility is not None:
        return _build_visibility_cluster_preconditioner(
            system.S,
            preconditioner_type=preconditioner_type,
            block_sizes=retained_block_sizes,
            visibility=visibility,
            visibility_clustering_type=visibility_clustering_type,
            message_prefix="schur_cluster_jacobi",
        )
    if _uses_block_preconditioner(preconditioner_type) and any(size > 1 for size in retained_block_sizes):
        return _build_matrix_preconditioner(
            system.S,
            preconditioner_type=preconditioner_type,
            block_sizes=retained_block_sizes,
            message_prefix="schur_block_jacobi",
        )
    diagonal_inverse = 1.0 / torch.clamp(torch.diagonal(system.S), min=torch.finfo(A.dtype).eps)
    return NormalEquationPreconditioner(
        preconditioner_type,
        f"schur_diagonal/{preconditioner_type.value}",
        block_sizes=retained_block_sizes,
        diagonal_inverse=diagonal_inverse,
    )


def jacobi_damping_from_jacobian(
    J: torch.Tensor,
    *,
    min_diagonal: float,
    max_diagonal: float,
    radius: float,
    jacobi_scaling: bool = True,
) -> torch.Tensor:
    if J.shape[1] == 0:
        return J.new_zeros(0)
    diag = torch.sum(J * J, dim=0) if jacobi_scaling else torch.ones(J.shape[1], dtype=J.dtype, device=J.device)
    diag = torch.clamp(diag, min=min_diagonal, max=max_diagonal)
    return diag / max(radius, torch.finfo(J.dtype).eps)


def dogleg_step(
    J: torch.Tensor,
    r: torch.Tensor,
    radius: float,
    *,
    dogleg_type: DoglegType = DoglegType.TRADITIONAL_DOGLEG,
) -> torch.Tensor:
    if radius <= 0.0 or J.shape[1] == 0:
        return J.new_zeros(J.shape[1])
    if dogleg_type is DoglegType.SUBSPACE_DOGLEG:
        return _subspace_dogleg_step(J, r, radius)
    return _traditional_dogleg_step(J, r, radius)


def _traditional_dogleg_step(J: torch.Tensor, r: torch.Tensor, radius: float) -> torch.Tensor:
    g = J.T @ r
    H = J.T @ J
    gn = _solve_square_system(H, -g)
    if torch.linalg.norm(gn) <= radius:
        return gn
    sd = _steepest_descent_step(J, g)
    sd_norm = torch.linalg.norm(sd)
    if sd_norm >= radius:
        return (radius / sd_norm.clamp_min(torch.finfo(J.dtype).eps)) * sd
    diff = gn - sd
    a = torch.dot(diff, diff)
    b = 2.0 * torch.dot(sd, diff)
    c = torch.dot(sd, sd) - radius * radius
    tau = (-b + torch.sqrt(torch.clamp(b * b - 4.0 * a * c, min=0.0))) / (2.0 * a).clamp_min(torch.finfo(J.dtype).eps)
    return sd + tau * diff


def _subspace_dogleg_step(J: torch.Tensor, r: torch.Tensor, radius: float) -> torch.Tensor:
    g = J.T @ r
    H = J.T @ J
    gn = _solve_square_system(H, -g)
    if torch.linalg.norm(gn) <= radius:
        return gn
    sd = _steepest_descent_step(J, g)
    sd_norm = torch.linalg.norm(sd)
    if sd_norm >= radius:
        return (radius / sd_norm.clamp_min(torch.finfo(J.dtype).eps)) * sd

    basis = _orthonormal_basis((-g, gn))
    if basis.shape[1] == 0:
        return J.new_zeros(J.shape[1])
    if basis.shape[1] < 2:
        return _traditional_dogleg_step(J, r, radius)

    reduced_jacobian = J @ basis
    reduced_hessian = reduced_jacobian.T @ reduced_jacobian
    reduced_gradient = reduced_jacobian.T @ r.reshape(-1)
    reduced_step = _trust_region_boundary_step(reduced_hessian, reduced_gradient, radius)
    return basis @ reduced_step


def _steepest_descent_step(J: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    Jg = J @ g
    eps = torch.finfo(J.dtype).eps
    denom = torch.dot(Jg, Jg).clamp_min(eps)
    alpha = torch.dot(g, g) / denom
    return -alpha * g


def _orthonormal_basis(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    basis: list[torch.Tensor] = []
    for vector in vectors:
        candidate = vector.reshape(-1).clone()
        candidate_norm = torch.linalg.norm(candidate)
        for q in basis:
            candidate = candidate - torch.dot(q, candidate) * q
        norm = torch.linalg.norm(candidate)
        threshold = 100.0 * torch.finfo(candidate.dtype).eps * max(float(candidate_norm.detach().cpu()), 1.0)
        if float(norm.detach().cpu()) > threshold:
            basis.append(candidate / norm)
    if not basis:
        first = next(iter(vectors))
        return first.new_zeros((first.numel(), 0))
    return torch.stack(basis, dim=1)


def _trust_region_boundary_step(H: torch.Tensor, g: torch.Tensor, radius: float) -> torch.Tensor:
    unconstrained = _solve_square_system(H, -g)
    if torch.linalg.norm(unconstrained) <= radius:
        return unconstrained
    eye = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    radius_value = max(float(radius), float(torch.finfo(H.dtype).tiny))

    def solve_shifted(shift: float) -> torch.Tensor:
        return _solve_square_system(H + shift * eye, -g)

    low = 0.0
    high = 1.0
    y = solve_shifted(high)
    for _ in range(80):
        if float(torch.linalg.norm(y).detach().cpu()) <= radius_value:
            break
        high *= 2.0
        y = solve_shifted(high)
    for _ in range(80):
        mid = 0.5 * (low + high)
        y = solve_shifted(mid)
        if float(torch.linalg.norm(y).detach().cpu()) > radius_value:
            low = mid
        else:
            high = mid
    y = solve_shifted(high)
    y_norm = torch.linalg.norm(y)
    if float(y_norm.detach().cpu()) > radius_value:
        return (radius / y_norm.clamp_min(torch.finfo(H.dtype).eps)) * y
    return y


def schur_solve_dense(
    A: torch.Tensor,
    b: torch.Tensor,
    num_eliminate: int,
    *,
    damping: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if num_eliminate <= 0 or num_eliminate >= A.shape[1]:
        if damping is not None:
            D = torch.diag(torch.sqrt(torch.clamp(damping.reshape(-1), min=0.0)))
            A = torch.cat([A, D], dim=0)
            b = torch.cat([b.reshape(-1), b.new_zeros(D.shape[0])], dim=0)
        return torch.linalg.lstsq(A, b.reshape(-1)).solution.reshape(-1)

    H = A.T @ A
    rhs = A.T @ b.reshape(-1)
    if damping is not None:
        H = H + torch.diag(damping.reshape(-1))

    Haa = H[:num_eliminate, :num_eliminate]
    Hab = H[:num_eliminate, num_eliminate:]
    Hba = H[num_eliminate:, :num_eliminate]
    Hbb = H[num_eliminate:, num_eliminate:]
    ga = rhs[:num_eliminate]
    gb = rhs[num_eliminate:]

    Haa_inv_Hab = _solve_square_system(Haa, Hab)
    Haa_inv_ga = _solve_square_system(Haa, ga)
    S = Hbb - Hba @ Haa_inv_Hab
    rhs_b = gb - Hba @ Haa_inv_ga
    xb = _solve_square_system(S, rhs_b)
    xa = Haa_inv_ga - Haa_inv_Hab @ xb
    return torch.cat([xa, xb])


def iterative_schur_solve(
    A: torch.Tensor,
    b: torch.Tensor,
    num_eliminate: int,
    *,
    damping: Optional[torch.Tensor] = None,
    min_iterations: int = 0,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
    preconditioner_type: PreconditionerType = PreconditionerType.JACOBI,
    block_sizes: Optional[Sequence[int]] = None,
    max_num_spse_iterations: int = 5,
    use_spse_initialization: bool = False,
    spse_tolerance: float = 0.1,
    visibility: Optional[Sequence[Iterable[int]]] = None,
    visibility_clustering_type: VisibilityClusteringType = VisibilityClusteringType.CANONICAL_VIEWS,
) -> LinearSolverResult:
    if num_eliminate <= 0 or num_eliminate >= A.shape[1]:
        return conjugate_gradient_normal_equations(
            A,
            b,
            damping=damping,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            tolerance=tolerance,
            preconditioner_type=preconditioner_type,
            block_sizes=block_sizes,
        )
    system = _build_schur_complement_system(A, damping=damping, num_eliminate=num_eliminate, b=b)
    assert system.schur_rhs is not None and system.ga is not None
    preconditioner = build_schur_complement_preconditioner(
        A,
        damping=damping,
        num_eliminate=num_eliminate,
        preconditioner_type=preconditioner_type,
        block_sizes=block_sizes,
        max_num_spse_iterations=max_num_spse_iterations,
        spse_tolerance=spse_tolerance,
        visibility=visibility,
        visibility_clustering_type=visibility_clustering_type,
    )
    initial_x: torch.Tensor | None = None
    message_suffix = ""
    if use_spse_initialization:
        spse = build_schur_complement_preconditioner(
            A,
            damping=damping,
            num_eliminate=num_eliminate,
            preconditioner_type=PreconditionerType.SCHUR_POWER_SERIES_EXPANSION,
            block_sizes=block_sizes,
            max_num_spse_iterations=max_num_spse_iterations,
            spse_tolerance=spse_tolerance,
        )
        initial_x = spse.apply(system.schur_rhs)
        message_suffix = f" spse_init/{max_num_spse_iterations}"
    xb, iterations, success = _preconditioned_conjugate_gradient(
        system.S,
        system.schur_rhs,
        preconditioner,
        initial_x=initial_x,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    xa = system.Haa_inv_ga - system.Haa_inv_Hab @ xb
    x = torch.cat([xa, xb])
    result = _summarize_solution(
        A,
        b,
        x,
        iterations,
        f"iterative schur {preconditioner.message}{message_suffix}",
        success=success,
    )
    return result


@dataclass
class _SchurComplementSystem:
    Haa: torch.Tensor
    Hab: torch.Tensor
    Hba: torch.Tensor
    Hbb: torch.Tensor
    S: torch.Tensor
    Haa_inv_Hab: torch.Tensor
    Haa_inv_ga: torch.Tensor
    schur_eliminated_term: torch.Tensor
    ga: Optional[torch.Tensor] = None
    gb: Optional[torch.Tensor] = None
    schur_rhs: Optional[torch.Tensor] = None


def _build_schur_complement_system(
    A: torch.Tensor,
    *,
    damping: Optional[torch.Tensor],
    num_eliminate: int,
    b: Optional[torch.Tensor],
) -> _SchurComplementSystem:
    H = A.T @ A
    if damping is not None:
        H = H + torch.diag(damping.reshape(-1))
    Haa = H[:num_eliminate, :num_eliminate]
    Hab = H[:num_eliminate, num_eliminate:]
    Hba = H[num_eliminate:, :num_eliminate]
    Hbb = H[num_eliminate:, num_eliminate:]
    Haa_inv_Hab = _solve_square_system(Haa, Hab)
    schur_eliminated_term = Hba @ Haa_inv_Hab
    S = Hbb - schur_eliminated_term
    if b is None:
        return _SchurComplementSystem(
            Haa=Haa,
            Hab=Hab,
            Hba=Hba,
            Hbb=Hbb,
            S=S,
            Haa_inv_Hab=Haa_inv_Hab,
            Haa_inv_ga=Haa.new_zeros(num_eliminate),
            schur_eliminated_term=schur_eliminated_term,
        )
    rhs = A.T @ b.reshape(-1)
    ga = rhs[:num_eliminate]
    gb = rhs[num_eliminate:]
    Haa_inv_ga = _solve_square_system(Haa, ga)
    schur_rhs = gb - Hba @ Haa_inv_ga
    return _SchurComplementSystem(
        Haa=Haa,
        Hab=Hab,
        Hba=Hba,
        Hbb=Hbb,
        S=S,
        Haa_inv_Hab=Haa_inv_Hab,
        Haa_inv_ga=Haa_inv_ga,
        schur_eliminated_term=schur_eliminated_term,
        ga=ga,
        gb=gb,
        schur_rhs=schur_rhs,
    )


def create_schur_complement_visibility_graph(
    visibility: Sequence[Iterable[int]],
) -> dict[tuple[int, int], float]:
    normalized = _normalize_visibility(visibility)
    inverse_visibility: dict[int, list[int]] = {}
    for camera_id, visible_points in enumerate(normalized):
        for point_id in visible_points:
            inverse_visibility.setdefault(point_id, []).append(camera_id)

    pair_counts: dict[tuple[int, int], int] = {}
    for cameras in inverse_visibility.values():
        ordered = sorted(cameras)
        for i, camera1 in enumerate(ordered):
            for camera2 in ordered[i + 1 :]:
                pair_counts[(camera1, camera2)] = pair_counts.get((camera1, camera2), 0) + 1

    edges = {(camera_id, camera_id): 1.0 for camera_id in range(len(normalized))}
    for (camera1, camera2), count in pair_counts.items():
        denom = sqrt(float(len(normalized[camera1]) * len(normalized[camera2])))
        if denom > 0.0:
            edges[(camera1, camera2)] = float(count) / denom
    return dict(sorted(edges.items()))


def single_linkage_visibility_clustering(
    visibility: Sequence[Iterable[int]],
    *,
    min_similarity: float = 0.9,
) -> tuple[int, ...]:
    graph = create_schur_complement_visibility_graph(visibility)
    num_vertices = len(_normalize_visibility(visibility))
    parent = {vertex: vertex for vertex in range(num_vertices)}

    def find(vertex: int) -> int:
        root = vertex
        while parent[root] != root:
            root = parent[root]
        while parent[vertex] != vertex:
            next_vertex = parent[vertex]
            parent[vertex] = root
            vertex = next_vertex
        return root

    for (vertex1, vertex2), weight in graph.items():
        if vertex1 >= vertex2 or weight < min_similarity:
            continue
        root1 = find(vertex1)
        root2 = find(vertex2)
        if root1 == root2:
            continue
        if root1 < root2:
            parent[root2] = root1
        else:
            parent[root1] = root2
    return _flatten_cluster_membership({vertex: find(vertex) for vertex in range(num_vertices)}, num_vertices)


def canonical_views_visibility_clustering(
    visibility: Sequence[Iterable[int]],
    *,
    min_views: int = 3,
    size_penalty_weight: float = 3.0,
    similarity_penalty_weight: float = 0.0,
    view_score_weight: float = 0.0,
) -> tuple[int, ...]:
    graph = create_schur_complement_visibility_graph(visibility)
    num_vertices = len(_normalize_visibility(visibility))
    valid_views = set(range(num_vertices))
    centers: list[int] = []
    view_to_canonical_view: dict[int, int] = {}
    view_to_canonical_view_similarity: dict[int, float] = {}

    while valid_views:
        best_difference = -float("inf")
        best_view = min(valid_views)
        for candidate in sorted(valid_views):
            difference = view_score_weight
            for neighbor in _graph_neighbors(graph, candidate):
                old_similarity = view_to_canonical_view_similarity.get(neighbor, 0.0)
                new_similarity = _graph_edge_weight(graph, neighbor, candidate)
                if new_similarity > old_similarity:
                    difference += new_similarity - old_similarity
            difference -= size_penalty_weight
            for center in centers:
                difference -= similarity_penalty_weight * _graph_edge_weight(graph, center, candidate)
            if difference > best_difference:
                best_difference = difference
                best_view = candidate

        if best_difference <= 0.0 and len(centers) >= min_views:
            break
        centers.append(best_view)
        valid_views.remove(best_view)
        for neighbor in _graph_neighbors(graph, best_view):
            old_similarity = view_to_canonical_view_similarity.get(neighbor, 0.0)
            new_similarity = _graph_edge_weight(graph, neighbor, best_view)
            if new_similarity > old_similarity:
                view_to_canonical_view[neighbor] = best_view
                view_to_canonical_view_similarity[neighbor] = new_similarity

    center_to_cluster = {center: index for index, center in enumerate(centers)}
    membership: dict[int, int] = {}
    for view in range(num_vertices):
        center = view_to_canonical_view.get(view)
        if center is None:
            membership[view] = -1
        else:
            membership[view] = center_to_cluster[center]
    return _flatten_cluster_membership(membership, num_vertices)


def build_visibility_cluster_structure(
    visibility: Sequence[Iterable[int]],
    *,
    preconditioner_type: PreconditionerType = PreconditionerType.CLUSTER_JACOBI,
    visibility_clustering_type: VisibilityClusteringType = VisibilityClusteringType.CANONICAL_VIEWS,
) -> VisibilityClusterStructure:
    if preconditioner_type not in {PreconditionerType.CLUSTER_JACOBI, PreconditionerType.CLUSTER_TRIDIAGONAL}:
        raise ValueError("visibility clustering is only defined for CLUSTER_JACOBI and CLUSTER_TRIDIAGONAL")
    normalized = _normalize_visibility(visibility)
    if visibility_clustering_type is VisibilityClusteringType.SINGLE_LINKAGE:
        membership = single_linkage_visibility_clustering(normalized)
    elif visibility_clustering_type is VisibilityClusteringType.CANONICAL_VIEWS:
        membership = canonical_views_visibility_clustering(normalized)
    else:
        raise ValueError(f"Unknown visibility clustering type: {visibility_clustering_type}")

    num_clusters = max(membership) + 1 if membership else 0
    cluster_pairs: set[tuple[int, int]] = {(cluster, cluster) for cluster in range(num_clusters)}
    cluster_graph = _create_cluster_visibility_graph(normalized, membership, num_clusters)
    if preconditioner_type is PreconditionerType.CLUSTER_TRIDIAGONAL and num_clusters > 1:
        cluster_pairs.update(degree2_maximum_spanning_forest_edges(num_clusters, cluster_graph))
    block_pairs = []
    for block1 in range(len(membership)):
        for block2 in range(block1, len(membership)):
            cluster1, cluster2 = sorted((membership[block1], membership[block2]))
            if (cluster1, cluster2) in cluster_pairs:
                block_pairs.append((block1, block2))
    return VisibilityClusterStructure(
        visibility_clustering_type=visibility_clustering_type,
        preconditioner_type=preconditioner_type,
        membership=membership,
        cluster_pairs=tuple(sorted(cluster_pairs)),
        block_pairs=tuple(block_pairs),
        num_clusters=num_clusters,
        camera_graph_edges=_edge_dict_to_tuple(create_schur_complement_visibility_graph(normalized)),
        cluster_graph_edges=_edge_dict_to_tuple(cluster_graph),
    )


def degree2_maximum_spanning_forest_edges(
    num_vertices: int,
    weighted_edges: dict[tuple[int, int], float],
) -> tuple[tuple[int, int], ...]:
    parent = {vertex: vertex for vertex in range(num_vertices)}
    degree = {vertex: 0 for vertex in range(num_vertices)}

    def find(vertex: int) -> int:
        root = vertex
        while parent[root] != root:
            root = parent[root]
        while parent[vertex] != vertex:
            next_vertex = parent[vertex]
            parent[vertex] = root
            vertex = next_vertex
        return root

    edges: list[tuple[float, tuple[int, int]]] = []
    for (vertex1, vertex2), weight in weighted_edges.items():
        if vertex1 == vertex2:
            continue
        edge = (vertex1, vertex2) if vertex1 < vertex2 else (vertex2, vertex1)
        edges.append((float(weight), edge))
    forest_edges: list[tuple[int, int]] = []
    for _weight, (vertex1, vertex2) in sorted(edges, reverse=True):
        if degree[vertex1] == 2 or degree[vertex2] == 2:
            continue
        root1 = find(vertex1)
        root2 = find(vertex2)
        if root1 == root2:
            continue
        forest_edges.append((vertex1, vertex2))
        degree[vertex1] += 1
        degree[vertex2] += 1
        if root2 < root1:
            root1, root2 = root2, root1
        parent[root2] = root1
    return tuple(sorted(forest_edges))


def _build_visibility_cluster_preconditioner(
    matrix: torch.Tensor,
    *,
    preconditioner_type: PreconditionerType,
    block_sizes: Sequence[int],
    visibility: Sequence[Iterable[int]],
    visibility_clustering_type: VisibilityClusteringType,
    message_prefix: str,
) -> NormalEquationPreconditioner:
    normalized_block_sizes = _normalize_block_sizes(block_sizes, matrix.shape[0])
    if len(visibility) != len(normalized_block_sizes):
        raise ValueError("visibility must contain one entry per retained Schur block")
    structure = build_visibility_cluster_structure(
        visibility,
        preconditioner_type=preconditioner_type,
        visibility_clustering_type=visibility_clustering_type,
    )
    slices = _block_slices(normalized_block_sizes)
    approximation = torch.zeros_like(matrix)
    for block1, block2 in structure.block_pairs:
        slice1 = slices[block1]
        slice2 = slices[block2]
        approximation[slice1, slice2] = matrix[slice1, slice2]
        if block1 != block2:
            approximation[slice2, slice1] = matrix[slice2, slice1]

    factor, is_cholesky = _factor_preconditioner_matrix(approximation)
    if not is_cholesky and preconditioner_type is PreconditionerType.CLUSTER_TRIDIAGONAL:
        scaled = approximation.clone()
        for block1, block2 in structure.block_pairs:
            if structure.membership[block1] == structure.membership[block2]:
                continue
            slice1 = slices[block1]
            slice2 = slices[block2]
            scaled[slice1, slice2] *= 0.5
            if block1 != block2:
                scaled[slice2, slice1] *= 0.5
        scaled_factor, scaled_is_cholesky = _factor_preconditioner_matrix(scaled)
        if scaled_is_cholesky:
            factor, is_cholesky = scaled_factor, True
        else:
            factor = scaled
    return NormalEquationPreconditioner(
        preconditioner_type,
        (
            f"{message_prefix}_visibility/{visibility_clustering_type.value}/"
            f"clusters:{structure.num_clusters}/pairs:{len(structure.block_pairs)}"
        ),
        normalized_block_sizes,
        matrix_factor=factor,
        matrix_is_cholesky=is_cholesky,
        visibility_structure=structure,
    )


def _normalize_visibility(visibility: Sequence[Iterable[int]]) -> tuple[frozenset[int], ...]:
    normalized: list[frozenset[int]] = []
    for visible_points in visibility:
        points: set[int] = set()
        for point in visible_points:
            point_id = int(point)
            if point_id < 0:
                raise ValueError("visibility point ids must be non-negative")
            points.add(point_id)
        normalized.append(frozenset(points))
    return tuple(normalized)


def _graph_neighbors(graph: dict[tuple[int, int], float], vertex: int) -> tuple[int, ...]:
    neighbors: set[int] = set()
    for vertex1, vertex2 in graph:
        if vertex1 == vertex:
            neighbors.add(vertex2)
        if vertex2 == vertex:
            neighbors.add(vertex1)
    return tuple(sorted(neighbors))


def _graph_edge_weight(graph: dict[tuple[int, int], float], vertex1: int, vertex2: int) -> float:
    edge = (vertex1, vertex2) if vertex1 <= vertex2 else (vertex2, vertex1)
    return graph.get(edge, 0.0)


def _flatten_cluster_membership(membership: dict[int, int], num_vertices: int) -> tuple[int, ...]:
    cluster_to_index: dict[int, int] = {}
    flattened: list[int] = []
    for vertex in range(num_vertices):
        cluster = membership.get(vertex, -1)
        if cluster < 0:
            cluster = vertex
        if cluster not in cluster_to_index:
            cluster_to_index[cluster] = len(cluster_to_index)
        flattened.append(cluster_to_index[cluster])
    return tuple(flattened)


def _create_cluster_visibility_graph(
    visibility: tuple[frozenset[int], ...],
    membership: tuple[int, ...],
    num_clusters: int,
) -> dict[tuple[int, int], float]:
    cluster_visibility: list[set[int]] = [set() for _ in range(num_clusters)]
    for block_id, visible_points in enumerate(visibility):
        cluster_visibility[membership[block_id]].update(visible_points)
    cluster_graph: dict[tuple[int, int], float] = {}
    for cluster1 in range(num_clusters):
        for cluster2 in range(cluster1 + 1, num_clusters):
            intersection_size = len(cluster_visibility[cluster1].intersection(cluster_visibility[cluster2]))
            if intersection_size:
                cluster_graph[(cluster1, cluster2)] = float(intersection_size)
    return dict(sorted(cluster_graph.items()))


def _edge_dict_to_tuple(edges: dict[tuple[int, int], float]) -> tuple[tuple[int, int, float], ...]:
    return tuple((vertex1, vertex2, weight) for (vertex1, vertex2), weight in sorted(edges.items()))


def _factor_preconditioner_matrix(matrix: torch.Tensor) -> tuple[torch.Tensor, bool]:
    try:
        return torch.linalg.cholesky(matrix), True
    except RuntimeError:
        return matrix, False


def _build_matrix_preconditioner(
    matrix: torch.Tensor,
    *,
    preconditioner_type: PreconditionerType,
    block_sizes: Sequence[int],
    message_prefix: str,
) -> NormalEquationPreconditioner:
    normalized_block_sizes = _normalize_block_sizes(block_sizes, matrix.shape[0])
    block_factors: list[tuple[slice, torch.Tensor, bool]] = []
    offset = 0
    for size in normalized_block_sizes:
        block_slice = slice(offset, offset + size)
        block = matrix[block_slice, block_slice]
        try:
            factor = torch.linalg.cholesky(block)
            is_cholesky = True
        except RuntimeError:
            factor = block
            is_cholesky = False
        block_factors.append((block_slice, factor, is_cholesky))
        offset += size
    return NormalEquationPreconditioner(
        preconditioner_type,
        f"{message_prefix}/{preconditioner_type.value}",
        normalized_block_sizes,
        block_factors=block_factors,
    )


def _build_cluster_tridiagonal_preconditioner(
    matrix: torch.Tensor,
    *,
    preconditioner_type: PreconditionerType,
    block_sizes: Sequence[int],
    message_prefix: str,
) -> NormalEquationPreconditioner:
    normalized_block_sizes = _normalize_block_sizes(block_sizes, matrix.shape[0])
    slices = _block_slices(normalized_block_sizes)
    approximation = torch.zeros_like(matrix)
    for i, block_slice in enumerate(slices):
        approximation[block_slice, block_slice] = matrix[block_slice, block_slice]
        if i + 1 < len(slices):
            next_slice = slices[i + 1]
            approximation[block_slice, next_slice] = matrix[block_slice, next_slice]
            approximation[next_slice, block_slice] = matrix[next_slice, block_slice]
    try:
        factor = torch.linalg.cholesky(approximation)
        is_cholesky = True
    except RuntimeError:
        factor = approximation
        is_cholesky = False
    return NormalEquationPreconditioner(
        preconditioner_type,
        f"{message_prefix}/{preconditioner_type.value}",
        normalized_block_sizes,
        matrix_factor=factor,
        matrix_is_cholesky=is_cholesky,
    )


def _preconditioned_conjugate_gradient(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioner: NormalEquationPreconditioner,
    *,
    initial_x: Optional[torch.Tensor] = None,
    min_iterations: int = 0,
    max_iterations: int,
    tolerance: float,
) -> tuple[torch.Tensor, int, bool]:
    x = torch.zeros_like(rhs.reshape(-1)) if initial_x is None else initial_x.reshape(-1).clone()
    r = rhs.reshape(-1) - matrix @ x
    z = preconditioner.apply(r)
    p = z.clone()
    rz_old = torch.dot(r, z)
    rhs_norm = torch.linalg.norm(rhs).clamp_min(torch.finfo(rhs.dtype).eps)
    if min_iterations <= 0 and torch.linalg.norm(r) <= tolerance * rhs_norm:
        return x, 0, True
    iterations = 0
    success = False
    for k in range(max_iterations):
        Ap = matrix @ p
        denom = torch.dot(p, Ap)
        if torch.abs(denom) <= torch.finfo(matrix.dtype).eps:
            success = bool(torch.linalg.norm(r) <= tolerance * rhs_norm)
            break
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        iterations = k + 1
        if iterations >= min_iterations and torch.linalg.norm(r) <= tolerance * rhs_norm:
            success = True
            break
        z = preconditioner.apply(r)
        rz_new = torch.dot(r, z)
        if torch.abs(rz_old) <= torch.finfo(matrix.dtype).eps:
            break
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return x, iterations, success


def _block_slices(block_sizes: Sequence[int]) -> tuple[slice, ...]:
    slices: list[slice] = []
    offset = 0
    for size in block_sizes:
        next_offset = offset + size
        slices.append(slice(offset, next_offset))
        offset = next_offset
    return tuple(slices)


def _remaining_block_sizes_after_elimination(
    block_sizes: Optional[Sequence[int]],
    num_eliminate: int,
    num_columns: int,
) -> tuple[int, ...]:
    normalized = _normalize_block_sizes(block_sizes, num_columns)
    remaining: list[int] = []
    offset = 0
    for size in normalized:
        next_offset = offset + size
        if next_offset <= num_eliminate:
            offset = next_offset
            continue
        if offset < num_eliminate < next_offset:
            remaining.append(next_offset - num_eliminate)
        else:
            remaining.append(size)
        offset = next_offset
    if not remaining:
        raise ValueError("num_eliminate leaves no retained Schur columns")
    return tuple(remaining)


def _uses_block_preconditioner(preconditioner_type: PreconditionerType) -> bool:
    return preconditioner_type in {
        PreconditionerType.SCHUR_JACOBI,
        PreconditionerType.SCHUR_POWER_SERIES_EXPANSION,
        PreconditionerType.CLUSTER_JACOBI,
        PreconditionerType.CLUSTER_TRIDIAGONAL,
        PreconditionerType.SUBSET,
    }


def _normalize_block_sizes(block_sizes: Optional[Sequence[int]], num_columns: int) -> tuple[int, ...]:
    if block_sizes is None:
        return tuple(1 for _ in range(num_columns))
    normalized = tuple(int(size) for size in block_sizes if int(size) > 0)
    if sum(normalized) != num_columns:
        raise ValueError("block_sizes must sum to the number of columns in A")
    return normalized


def _try_optional_linear_backend(
    name: str,
    A: torch.Tensor,
    b: torch.Tensor,
    **kwargs: Any,
) -> LinearSolverResult | None:
    backend = get_optional_backend(name)
    if backend is None:
        return None
    try:
        raw_result = backend(A, b, **kwargs)
    except TypeError:
        raw_result = backend(A, b)
    except OptionalBackendUnavailable:
        return None
    if isinstance(raw_result, LinearSolverResult):
        return raw_result
    return _summarize_solution(A, b, raw_result, 1, f"optional {name} backend")


def _refine_solution(
    A: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    *,
    damping: Optional[torch.Tensor],
    max_refinement_iterations: int,
) -> torch.Tensor:
    if max_refinement_iterations <= 0:
        return x
    if damping is not None:
        D = torch.diag(torch.sqrt(torch.clamp(damping.reshape(-1), min=0.0)))
        A_aug = torch.cat([A, D], dim=0)
        b_aug = torch.cat([b.reshape(-1), b.new_zeros(A.shape[1])], dim=0)
    else:
        A_aug = A
        b_aug = b.reshape(-1)
    return _refine_augmented_solution(A_aug, b_aug, x, max_refinement_iterations)


def _refine_augmented_solution(
    A: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    max_refinement_iterations: int,
) -> torch.Tensor:
    for _ in range(max_refinement_iterations):
        residual = b.reshape(-1) - A @ x.reshape(-1)
        if torch.linalg.norm(residual) <= 10.0 * torch.finfo(A.dtype).eps * torch.linalg.norm(b.reshape(-1)).clamp_min(1.0):
            break
        correction = torch.linalg.lstsq(A, residual).solution.reshape(-1)
        x = x + correction
    return x


def _solve_square_system(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve(A, b)
    except RuntimeError:
        return torch.linalg.lstsq(A, b).solution


def _summarize_solution(
    A: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    iterations: int,
    message: str,
    *,
    success: bool = True,
) -> LinearSolverResult:
    residual_norm = float(torch.linalg.norm(A @ x.reshape(-1) - b.reshape(-1)).detach().cpu())
    return LinearSolverResult(x.reshape(-1), LinearSolverSummary(residual_norm, iterations, success, message))
