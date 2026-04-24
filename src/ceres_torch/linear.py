from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .types import LinearSolverType, PreconditionerType


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


OptionalBackend = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
_optional_backends: dict[str, OptionalBackend] = {}


def register_optional_backend(name: str, backend: OptionalBackend) -> None:
    _optional_backends[name] = backend


def solve_linear_system(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    solver_type: LinearSolverType = LinearSolverType.DENSE_QR,
    damping: Optional[torch.Tensor] = None,
    num_eliminate: int = 0,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
    preconditioner_type: PreconditionerType = PreconditionerType.JACOBI,
    use_mixed_precision: bool = False,
    max_refinement_iterations: int = 0,
) -> LinearSolverResult:
    if A.numel() == 0:
        x = b.new_zeros(b.shape)
        return LinearSolverResult(x, LinearSolverSummary(0.0, 0, True, "empty system"))

    if solver_type is LinearSolverType.DENSE_SCHUR or (
        solver_type is LinearSolverType.ITERATIVE_SCHUR and num_eliminate > 0
    ):
        x = schur_solve_dense(A, b, num_eliminate, damping=damping)
        x = _refine_solution(
            A,
            b,
            x,
            damping=damping,
            max_refinement_iterations=max_refinement_iterations if use_mixed_precision else 0,
        )
        return _summarize_solution(A, b, x, 1, "dense schur")

    if solver_type in {LinearSolverType.CGNR, LinearSolverType.ITERATIVE_SCHUR}:
        return conjugate_gradient_normal_equations(
            A,
            b,
            damping=damping,
            max_iterations=max_iterations,
            tolerance=tolerance,
            preconditioner_type=preconditioner_type,
        )

    if solver_type in {LinearSolverType.SPARSE_NORMAL_CHOLESKY, LinearSolverType.SPARSE_SCHUR}:
        backend_name = "sparse_cholesky"
        if backend_name in _optional_backends:
            x = _optional_backends[backend_name](A, b)
            return _summarize_solution(A, b, x, 1, "optional sparse backend")
        solver_type = LinearSolverType.DENSE_SCHUR if solver_type is LinearSolverType.SPARSE_SCHUR else LinearSolverType.DENSE_NORMAL_CHOLESKY

    if solver_type is LinearSolverType.DENSE_SCHUR:
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
) -> LinearSolverResult:
    rhs = A.T @ b.reshape(-1)
    diag = damping.reshape(-1) if damping is not None else torch.zeros(A.shape[1], dtype=A.dtype, device=A.device)

    def matvec(x: torch.Tensor) -> torch.Tensor:
        return A.T @ (A @ x) + diag * x

    x = torch.zeros_like(rhs)
    r = rhs - matvec(x)
    if _uses_diagonal_preconditioner(preconditioner_type):
        M_inv = 1.0 / torch.clamp(torch.sum(A * A, dim=0) + diag, min=torch.finfo(A.dtype).eps)
        z = M_inv * r
    else:
        z = r.clone()
    p = z.clone()
    rz_old = torch.dot(r, z)
    b_norm = torch.linalg.norm(rhs).clamp_min(torch.finfo(A.dtype).eps)
    iterations = 0
    success = False
    for k in range(max_iterations):
        Ap = matvec(p)
        denom = torch.dot(p, Ap).clamp_min(torch.finfo(A.dtype).eps)
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        iterations = k + 1
        if torch.linalg.norm(r) <= tolerance * b_norm:
            success = True
            break
        z = M_inv * r if _uses_diagonal_preconditioner(preconditioner_type) else r
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old.clamp_min(torch.finfo(A.dtype).eps)
        p = z + beta * p
        rz_old = rz_new
    return _summarize_solution(A, b, x, iterations, "cgnr", success=success)


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


def dogleg_step(J: torch.Tensor, r: torch.Tensor, radius: float) -> torch.Tensor:
    g = J.T @ r
    H = J.T @ J
    try:
        gn = torch.linalg.solve(H, -g)
    except RuntimeError:
        gn = torch.linalg.lstsq(H, -g).solution
    if torch.linalg.norm(gn) <= radius:
        return gn
    Jg = J @ g
    denom = torch.dot(Jg, Jg).clamp_min(torch.finfo(J.dtype).eps)
    alpha = torch.dot(g, g) / denom
    sd = -alpha * g
    sd_norm = torch.linalg.norm(sd)
    if sd_norm >= radius:
        return (radius / sd_norm.clamp_min(torch.finfo(J.dtype).eps)) * sd
    diff = gn - sd
    a = torch.dot(diff, diff)
    b = 2.0 * torch.dot(sd, diff)
    c = torch.dot(sd, sd) - radius * radius
    tau = (-b + torch.sqrt(torch.clamp(b * b - 4.0 * a * c, min=0.0))) / (2.0 * a).clamp_min(torch.finfo(J.dtype).eps)
    return sd + tau * diff


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


def _uses_diagonal_preconditioner(preconditioner_type: PreconditionerType) -> bool:
    return preconditioner_type in {
        PreconditionerType.JACOBI,
        PreconditionerType.SCHUR_JACOBI,
        PreconditionerType.SCHUR_POWER_SERIES_EXPANSION,
        PreconditionerType.CLUSTER_JACOBI,
        PreconditionerType.CLUSTER_TRIDIAGONAL,
        PreconditionerType.SUBSET,
    }


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
