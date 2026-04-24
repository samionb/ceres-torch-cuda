from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import torch

from .linear import (
    LinearSolverResult,
    LinearSolverSummary,
    OptionalBackendUnavailable,
    get_optional_backend,
    register_optional_backend,
    unregister_optional_backend,
)


SCIPY_SPARSE_BACKEND_NAMES = (
    "sparse_normal_cholesky",
    "sparse_cholesky",
    "sparse_schur",
    "sparse_qr_covariance",
)


@dataclass(frozen=True)
class NativeSparseBackendInfo:
    available: bool
    backend: str
    registered: tuple[str, ...]
    message: str


def scipy_sparse_available() -> bool:
    return importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("scipy") is not None


def native_sparse_backends_available() -> bool:
    return scipy_sparse_available()


def register_native_sparse_backends(*, overwrite: bool = True) -> NativeSparseBackendInfo:
    return register_scipy_sparse_backends(overwrite=overwrite)


def register_scipy_sparse_backends(*, overwrite: bool = True) -> NativeSparseBackendInfo:
    if not scipy_sparse_available():
        return NativeSparseBackendInfo(
            available=False,
            backend="scipy-superlu",
            registered=(),
            message="SciPy is not installed; native sparse backends were not registered.",
        )
    backends = {
        "sparse_normal_cholesky": scipy_sparse_normal_cholesky,
        "sparse_cholesky": scipy_sparse_normal_cholesky,
        "sparse_schur": scipy_sparse_schur,
        "sparse_qr_covariance": scipy_sparse_qr_covariance,
    }
    registered: list[str] = []
    for name, backend in backends.items():
        if overwrite or get_optional_backend(name) is None:
            register_optional_backend(name, backend)
            registered.append(name)
    return NativeSparseBackendInfo(
        available=True,
        backend="scipy-superlu",
        registered=tuple(registered),
        message="Registered SciPy/SuperLU sparse direct backends.",
    )


def unregister_scipy_sparse_backends() -> None:
    for name in SCIPY_SPARSE_BACKEND_NAMES:
        unregister_optional_backend(name)


def unregister_native_sparse_backends() -> None:
    unregister_scipy_sparse_backends()


def scipy_sparse_normal_cholesky(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    damping: torch.Tensor | None = None,
    **_: Any,
) -> LinearSolverResult:
    sp, _spla, _np = _scipy_modules()
    A_sp = sp.csc_matrix(_numpy_2d(A))
    b_np = _numpy_vector(b)
    H = A_sp.T @ A_sp
    if damping is not None:
        H = H + sp.diags(_numpy_vector(damping), format="csc")
    rhs = A_sp.T @ b_np
    x_np = _solve_sparse_direct(H, rhs)
    x = _torch_like(x_np, A, shape=(A.shape[1],))
    return LinearSolverResult(
        x,
        _summary(A, b, x, damping=damping, iterations=1, message="scipy sparse normal equations"),
    )


def scipy_sparse_schur(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    damping: torch.Tensor | None = None,
    num_eliminate: int = 0,
    **_: Any,
) -> LinearSolverResult:
    sp, _spla, np = _scipy_modules()
    n = A.shape[1]
    if num_eliminate <= 0 or num_eliminate >= n:
        result = scipy_sparse_normal_cholesky(A, b, damping=damping)
        result.summary.message = "scipy sparse schur normal fallback"
        return result

    A_sp = sp.csc_matrix(_numpy_2d(A))
    b_np = _numpy_vector(b)
    H = A_sp.T @ A_sp
    if damping is not None:
        H = H + sp.diags(_numpy_vector(damping), format="csc")
    rhs = A_sp.T @ b_np

    e = int(num_eliminate)
    Haa = H[:e, :e].tocsc()
    Hab = H[:e, e:].tocsc()
    Hba = H[e:, :e].tocsc()
    Hbb = H[e:, e:].tocsc()
    ga = np.asarray(rhs[:e]).reshape(-1)
    gb = np.asarray(rhs[e:]).reshape(-1)

    Haa_inv_Hab = _solve_sparse_direct(Haa, Hab.toarray()) if Hab.shape[1] else np.zeros((e, 0), dtype=b_np.dtype)
    Haa_inv_ga = _solve_sparse_direct(Haa, ga)
    S = Hbb - Hba @ Haa_inv_Hab
    rhs_b = gb - Hba @ Haa_inv_ga
    xb = _solve_sparse_direct(sp.csc_matrix(S), rhs_b)
    xa = Haa_inv_ga - Haa_inv_Hab @ xb
    x = _torch_like(np.concatenate([xa, xb]), A, shape=(n,))
    return LinearSolverResult(
        x,
        _summary(A, b, x, damping=damping, iterations=1, message="scipy sparse schur"),
    )


def scipy_sparse_qr_covariance(
    J: torch.Tensor,
    *,
    options: Any | None = None,
    **_: Any,
) -> torch.Tensor:
    sp, _spla, np = _scipy_modules()
    if J.shape[1] == 0:
        return J.new_zeros((0, 0))
    if J.shape[0] < J.shape[1] and getattr(options, "null_space_rank", 0) == 0:
        raise OptionalBackendUnavailable("Sparse direct covariance requires rows >= columns for strict rank policy.")
    J_sp = sp.csc_matrix(_numpy_2d(J))
    H = (J_sp.T @ J_sp).tocsc()
    identity = np.eye(H.shape[0], dtype=_numpy_2d(J).dtype)
    covariance = _solve_sparse_direct(H, identity)
    covariance = 0.5 * (covariance + covariance.T)
    return _torch_like(covariance, J, shape=(J.shape[1], J.shape[1]))


def _scipy_modules() -> tuple[Any, Any, Any]:
    if not scipy_sparse_available():
        raise OptionalBackendUnavailable("SciPy sparse backend is unavailable.")
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    return sp, spla, np


def _solve_sparse_direct(matrix: Any, rhs: Any) -> Any:
    _sp, spla, np = _scipy_modules()
    try:
        lu = spla.splu(matrix.tocsc())
        rhs_array = np.asarray(rhs)
        if rhs_array.ndim == 2:
            columns = [np.asarray(lu.solve(rhs_array[:, i])).reshape(-1) for i in range(rhs_array.shape[1])]
            return np.stack(columns, axis=1) if columns else np.zeros((matrix.shape[0], 0), dtype=rhs_array.dtype)
        return np.asarray(lu.solve(rhs_array)).reshape(rhs_array.shape)
    except Exception as exc:  # SciPy raises RuntimeError for singular SuperLU factorizations.
        raise OptionalBackendUnavailable(str(exc)) from exc


def _numpy_2d(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().contiguous().numpy()


def _numpy_vector(tensor: torch.Tensor) -> Any:
    return tensor.detach().reshape(-1).cpu().contiguous().numpy()


def _torch_like(values: Any, like: torch.Tensor, *, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.as_tensor(values, dtype=like.dtype, device=like.device).reshape(shape)


def _summary(
    A: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    *,
    damping: torch.Tensor | None,
    iterations: int,
    message: str,
) -> LinearSolverSummary:
    residual = A @ x.reshape(-1) - b.reshape(-1).to(dtype=A.dtype, device=A.device)
    residual_norm_sq = torch.sum(residual * residual)
    if damping is not None:
        d = damping.reshape(-1).to(dtype=A.dtype, device=A.device)
        residual_norm_sq = residual_norm_sq + torch.sum(torch.clamp(d, min=0.0) * x.reshape(-1) ** 2)
    return LinearSolverSummary(
        residual_norm=float(torch.sqrt(torch.clamp(residual_norm_sq, min=0.0)).detach().cpu()),
        num_iterations=iterations,
        success=True,
        message=message,
    )
