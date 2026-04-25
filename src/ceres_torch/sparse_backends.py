from __future__ import annotations

import importlib
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
SUITESPARSE_QR_MODULE_NAMES = ("sparseqr", "suitesparseqr")


@dataclass(frozen=True)
class NativeSparseBackendInfo:
    available: bool
    backend: str
    registered: tuple[str, ...]
    message: str


def scipy_sparse_available() -> bool:
    return importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("scipy") is not None


def suitesparseqr_available() -> bool:
    return scipy_sparse_available() and _find_suitesparseqr_module_name() is not None


def native_sparse_backends_available() -> bool:
    return scipy_sparse_available() or suitesparseqr_available()


def register_native_sparse_backends(*, overwrite: bool = True) -> NativeSparseBackendInfo:
    scipy_info = register_scipy_sparse_backends(overwrite=overwrite)
    suitesparseqr_info = register_suitesparseqr_sparse_qr_backend(overwrite=overwrite)
    registered = scipy_info.registered + tuple(name for name in suitesparseqr_info.registered if name not in scipy_info.registered)
    available = scipy_info.available or suitesparseqr_info.available
    backend_names = [info.backend for info in (scipy_info, suitesparseqr_info) if info.available]
    return NativeSparseBackendInfo(
        available=available,
        backend="+".join(backend_names) if backend_names else "none",
        registered=registered,
        message="; ".join(info.message for info in (scipy_info, suitesparseqr_info) if info.message),
    )


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


def register_suitesparseqr_sparse_qr_backend(*, overwrite: bool = True) -> NativeSparseBackendInfo:
    module_name = _find_suitesparseqr_module_name()
    if module_name is None:
        return NativeSparseBackendInfo(
            available=False,
            backend="suitesparseqr",
            registered=(),
            message="SuiteSparseQR Python bindings are not installed; sparse QR covariance backend was not registered.",
        )
    if not scipy_sparse_available():
        return NativeSparseBackendInfo(
            available=False,
            backend=module_name,
            registered=(),
            message="SciPy is required to pass CSC matrices to SuiteSparseQR bindings.",
        )
    if overwrite or get_optional_backend("sparse_qr_covariance") is None:
        register_optional_backend("sparse_qr_covariance", suitesparseqr_sparse_qr_covariance)
        registered = ("sparse_qr_covariance",)
    else:
        registered = ()
    return NativeSparseBackendInfo(
        available=True,
        backend=module_name,
        registered=registered,
        message=f"Registered {module_name} sparse QR covariance backend.",
    )


def unregister_suitesparseqr_sparse_qr_backend() -> None:
    unregister_optional_backend("sparse_qr_covariance")


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


def suitesparseqr_sparse_qr_covariance(
    J: torch.Tensor,
    *,
    options: Any | None = None,
    **_: Any,
) -> torch.Tensor:
    sp, _spla, np = _scipy_modules()
    if J.shape[1] == 0:
        return J.new_zeros((0, 0))
    if J.shape[0] < J.shape[1] and getattr(options, "null_space_rank", 0) == 0:
        raise OptionalBackendUnavailable("SuiteSparseQR covariance requires rows >= columns for strict rank policy.")
    module = _suitesparseqr_module()
    qr = getattr(module, "qr", None)
    if qr is None:
        raise OptionalBackendUnavailable("SuiteSparseQR binding does not expose a qr(A) factorization API.")
    try:
        factorization = qr(sp.csc_matrix(_numpy_2d(J)))
    except TypeError as exc:
        raise OptionalBackendUnavailable(f"SuiteSparseQR qr(A) call failed: {exc}") from exc
    R, permutation = _extract_qr_factor(factorization, J.shape[1], np)
    covariance = _covariance_from_upper_qr(R, permutation, J.shape, options, np)
    return _torch_like(covariance, J, shape=(J.shape[1], J.shape[1]))


def _scipy_modules() -> tuple[Any, Any, Any]:
    if not scipy_sparse_available():
        raise OptionalBackendUnavailable("SciPy sparse backend is unavailable.")
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    return sp, spla, np


def _find_suitesparseqr_module_name() -> str | None:
    for name in SUITESPARSE_QR_MODULE_NAMES:
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ValueError):
            continue
    return None


def _suitesparseqr_module() -> Any:
    module_name = _find_suitesparseqr_module_name()
    if module_name is None:
        raise OptionalBackendUnavailable("SuiteSparseQR Python bindings are unavailable.")
    return importlib.import_module(module_name)


def _extract_qr_factor(factorization: Any, num_columns: int, np: Any) -> tuple[Any, Any | None]:
    if hasattr(factorization, "R"):
        R = getattr(factorization, "R")
        permutation = getattr(factorization, "E", None)
        if permutation is None:
            permutation = getattr(factorization, "permutation", None)
        return _numpy_matrix(R, np), _numpy_permutation(permutation, num_columns, np)

    pieces = list(factorization) if isinstance(factorization, (tuple, list)) else [factorization]
    matrix_candidates = []
    permutation = None
    for piece in pieces:
        matrix = _try_numpy_matrix(piece, np)
        if matrix is not None and matrix.ndim == 2 and matrix.shape[1] >= num_columns and matrix.shape[0] >= num_columns:
            score = _upper_triangular_score(matrix[:num_columns, :num_columns], np)
            matrix_candidates.append((score, matrix))
        if permutation is None:
            permutation = _numpy_permutation(piece, num_columns, np)

    if not matrix_candidates:
        raise OptionalBackendUnavailable("SuiteSparseQR factorization did not include an R matrix.")
    matrix_candidates.sort(key=lambda item: item[0])
    return matrix_candidates[0][1], permutation


def _try_numpy_matrix(value: Any, np: Any) -> Any | None:
    try:
        return _numpy_matrix(value, np)
    except Exception:
        return None


def _numpy_matrix(value: Any, np: Any) -> Any:
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray())
    return np.asarray(value)


def _numpy_permutation(value: Any, num_columns: int, np: Any) -> Any | None:
    if value is None:
        return None
    array = _try_numpy_matrix(value, np)
    if array is None:
        return None
    if array.ndim == 1 and array.shape[0] == num_columns and np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64, copy=False)
    if array.ndim == 2 and array.shape == (num_columns, num_columns):
        row_counts = np.sum(np.abs(array) > 0.5, axis=1)
        col_counts = np.sum(np.abs(array) > 0.5, axis=0)
        if np.all(row_counts == 1) and np.all(col_counts == 1):
            return array
    return None


def _upper_triangular_score(matrix: Any, np: Any) -> float:
    lower = np.tril(matrix, k=-1)
    total = np.linalg.norm(matrix) + np.finfo(matrix.dtype).eps
    return float(np.linalg.norm(lower) / total)


def _covariance_from_upper_qr(R: Any, permutation: Any | None, shape: tuple[int, int], options: Any | None, np: Any) -> Any:
    num_columns = shape[1]
    R_square = np.triu(np.asarray(R[:num_columns, :num_columns]))
    diagonal = np.abs(np.diag(R_square))
    if diagonal.size == 0:
        raise OptionalBackendUnavailable("SuiteSparseQR returned an empty R factor.")
    pivot_threshold = getattr(options, "column_pivot_threshold", -1.0)
    if pivot_threshold >= 0.0:
        threshold = pivot_threshold
    else:
        threshold = 20.0 * (shape[0] + shape[1]) * np.finfo(R_square.dtype).eps * float(np.max(diagonal))
    if np.any(diagonal <= threshold):
        raise OptionalBackendUnavailable("SuiteSparseQR detected a rank deficient Jacobian.")
    R_t = torch.as_tensor(R_square)
    eye_t = torch.eye(num_columns, dtype=R_t.dtype)
    R_inv_t = torch.linalg.solve_triangular(R_t, eye_t, upper=True)
    covariance = R_inv_t.matmul(R_inv_t.T).cpu().numpy()
    if permutation is None:
        return 0.5 * (covariance + covariance.T)
    if getattr(permutation, "ndim", 0) == 1:
        restored = np.zeros_like(covariance)
        restored[np.ix_(permutation, permutation)] = covariance
        covariance = restored
    else:
        covariance = permutation @ covariance @ permutation.T
    return 0.5 * (covariance + covariance.T)


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
