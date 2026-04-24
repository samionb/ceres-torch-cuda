from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
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


CUDA_BACKEND_NAMES = (
    "sparse_normal_cholesky",
    "sparse_cholesky",
    "sparse_schur",
    "block_schur",
)

_EXTENSION_NAME = "ceres_torch_cuda_ext"
_LOADED_EXTENSION: Any | None = None


@dataclass(frozen=True)
class CudaExtensionInfo:
    available: bool
    backend: str
    registered: tuple[str, ...]
    source_paths: tuple[str, ...]
    message: str


def torch_cuda_backend_available() -> bool:
    return torch.version.cuda is not None and torch.cuda.is_available()


def get_torch_cuda_backend_info() -> CudaExtensionInfo:
    source_paths = cuda_extension_source_paths()
    if torch.version.cuda is None:
        return CudaExtensionInfo(
            available=False,
            backend="torch-cuda",
            registered=(),
            source_paths=source_paths,
            message="PyTorch is not CUDA-enabled.",
        )
    if not torch.cuda.is_available():
        return CudaExtensionInfo(
            available=False,
            backend="torch-cuda",
            registered=(),
            source_paths=source_paths,
            message="No CUDA device is available.",
        )
    return CudaExtensionInfo(
        available=True,
        backend="torch-cuda",
        registered=(),
        source_paths=source_paths,
        message="PyTorch CUDA backend is available.",
    )


def cuda_extension_source_paths() -> tuple[str, ...]:
    root = Path(__file__).resolve().parents[2]
    native_dir = root / "native" / "cuda"
    return (
        str(native_dir / "ceres_torch_cuda.cpp"),
        str(native_dir / "ceres_torch_cuda_kernel.cu"),
    )


def get_cuda_extension_info() -> CudaExtensionInfo:
    source_paths = cuda_extension_source_paths()
    missing_sources = [path for path in source_paths if not Path(path).exists()]
    if missing_sources:
        return CudaExtensionInfo(
            available=False,
            backend="cuda-extension",
            registered=(),
            source_paths=source_paths,
            message=f"CUDA extension sources are missing: {missing_sources}",
        )
    if torch.version.cuda is None:
        return CudaExtensionInfo(
            available=False,
            backend="cuda-extension",
            registered=(),
            source_paths=source_paths,
            message="PyTorch is not CUDA-enabled.",
        )
    if not torch.cuda.is_available():
        return CudaExtensionInfo(
            available=False,
            backend="cuda-extension",
            registered=(),
            source_paths=source_paths,
            message="No CUDA device is available for extension execution.",
        )
    if shutil.which("nvcc") is None:
        return CudaExtensionInfo(
            available=False,
            backend="cuda-extension",
            registered=(),
            source_paths=source_paths,
            message="nvcc was not found on PATH.",
        )
    if shutil.which("ninja") is None:
        return CudaExtensionInfo(
            available=False,
            backend="cuda-extension",
            registered=(),
            source_paths=source_paths,
            message="ninja was not found on PATH.",
        )
    return CudaExtensionInfo(
        available=True,
        backend="cuda-extension",
        registered=(),
        source_paths=source_paths,
        message="CUDA extension build inputs are available.",
    )


def cuda_extension_build_available() -> bool:
    return get_cuda_extension_info().available


def load_cuda_extension(*, verbose: bool | None = None, force: bool = False) -> Any:
    global _LOADED_EXTENSION
    if _LOADED_EXTENSION is not None and not force:
        return _LOADED_EXTENSION
    info = get_cuda_extension_info()
    if not info.available:
        raise OptionalBackendUnavailable(info.message)
    from torch.utils.cpp_extension import load

    if verbose is None:
        verbose = os.environ.get("CERES_TORCH_CUDA_EXTENSION_VERBOSE") == "1"
    _LOADED_EXTENSION = load(
        name=_EXTENSION_NAME,
        sources=list(info.source_paths),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )
    return _LOADED_EXTENSION


def register_cuda_sparse_backends(*, overwrite: bool = True, backend: str = "torch") -> CudaExtensionInfo:
    if backend == "extension":
        return register_cuda_extension_backends(overwrite=overwrite)
    if backend != "torch":
        raise ValueError("backend must be 'torch' or 'extension'")
    info = get_torch_cuda_backend_info()
    if not info.available:
        return info
    backends = {
        "sparse_normal_cholesky": cuda_sparse_normal_cholesky,
        "sparse_cholesky": cuda_sparse_normal_cholesky,
        "sparse_schur": cuda_block_schur,
        "block_schur": cuda_block_schur,
    }
    registered: list[str] = []
    for name, backend_fn in backends.items():
        if overwrite or get_optional_backend(name) is None:
            register_optional_backend(name, backend_fn)
            registered.append(name)
    return CudaExtensionInfo(
        available=True,
        backend=info.backend,
        registered=tuple(registered),
        source_paths=info.source_paths,
        message="Registered PyTorch CUDA sparse/block-Schur backends.",
    )


def register_cuda_extension_backends(*, overwrite: bool = True) -> CudaExtensionInfo:
    info = get_cuda_extension_info()
    if not info.available:
        return info
    backends = {
        "sparse_normal_cholesky": cuda_extension_sparse_normal_cholesky,
        "sparse_cholesky": cuda_extension_sparse_normal_cholesky,
        "sparse_schur": cuda_extension_block_schur,
        "block_schur": cuda_extension_block_schur,
    }
    registered: list[str] = []
    for name, backend in backends.items():
        if overwrite or get_optional_backend(name) is None:
            register_optional_backend(name, backend)
            registered.append(name)
    return CudaExtensionInfo(
        available=True,
        backend=info.backend,
        registered=tuple(registered),
        source_paths=info.source_paths,
        message="Registered CUDA extension sparse/block-Schur backends.",
    )


def unregister_cuda_sparse_backends() -> None:
    for name in CUDA_BACKEND_NAMES:
        unregister_optional_backend(name)


def cuda_sparse_normal_cholesky(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    damping: torch.Tensor | None = None,
    **_: Any,
) -> LinearSolverResult:
    _require_cuda_tensor(A, "A")
    _require_cuda_tensor(b, "b")
    H, rhs = _torch_normal_equations(A, b, damping)
    x = torch.linalg.solve(H, rhs.reshape(-1, 1)).reshape(-1)
    return LinearSolverResult(
        x.reshape(-1),
        _summary(A, b, x, damping=damping, iterations=1, message="torch cuda sparse normal equations"),
    )


def cuda_block_schur(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    damping: torch.Tensor | None = None,
    num_eliminate: int = 0,
    **_: Any,
) -> LinearSolverResult:
    _require_cuda_tensor(A, "A")
    _require_cuda_tensor(b, "b")
    H, rhs = _torch_normal_equations(A, b, damping)
    n = H.shape[0]
    if num_eliminate <= 0 or num_eliminate >= n:
        x = torch.linalg.solve(H, rhs.reshape(-1, 1)).reshape(-1)
    else:
        e = int(num_eliminate)
        Haa = H[:e, :e]
        Hab = H[:e, e:]
        Hba = H[e:, :e]
        Hbb = H[e:, e:]
        ga = rhs[:e]
        gb = rhs[e:]
        Haa_inv_Hab = torch.linalg.solve(Haa, Hab)
        Haa_inv_ga = torch.linalg.solve(Haa, ga.reshape(-1, 1)).reshape(-1)
        S = Hbb - Hba @ Haa_inv_Hab
        rhs_b = gb - Hba @ Haa_inv_ga
        xb = torch.linalg.solve(S, rhs_b.reshape(-1, 1)).reshape(-1)
        xa = Haa_inv_ga - Haa_inv_Hab @ xb
        x = torch.cat([xa, xb])
    return LinearSolverResult(
        x.reshape(-1),
        _summary(A, b, x, damping=damping, iterations=1, message="torch cuda block schur"),
    )


def cuda_extension_sparse_normal_cholesky(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    damping: torch.Tensor | None = None,
    **_: Any,
) -> LinearSolverResult:
    _require_cuda_tensor(A, "A")
    _require_cuda_tensor(b, "b")
    extension = load_cuda_extension()
    damping_t = _damping_or_empty(A, damping)
    x = extension.normal_equations_solve(A.contiguous(), b.reshape(-1).contiguous(), damping_t.contiguous())
    return LinearSolverResult(
        x.reshape(-1),
        _summary(A, b, x, damping=damping, iterations=1, message="cuda extension sparse normal equations"),
    )


def cuda_extension_block_schur(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    damping: torch.Tensor | None = None,
    num_eliminate: int = 0,
    **_: Any,
) -> LinearSolverResult:
    _require_cuda_tensor(A, "A")
    _require_cuda_tensor(b, "b")
    extension = load_cuda_extension()
    damping_t = _damping_or_empty(A, damping)
    x = extension.block_schur_solve(
        A.contiguous(),
        b.reshape(-1).contiguous(),
        int(num_eliminate),
        damping_t.contiguous(),
    )
    return LinearSolverResult(
        x.reshape(-1),
        _summary(A, b, x, damping=damping, iterations=1, message="cuda extension block schur"),
    )


def _torch_normal_equations(
    A: torch.Tensor,
    b: torch.Tensor,
    damping: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    H = A.T @ A
    if damping is not None:
        d = _damping_or_empty(A, damping)
        H = H + torch.diag(d)
    rhs = A.T @ b.reshape(-1)
    return H, rhs


def _require_cuda_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cuda":
        raise OptionalBackendUnavailable(f"{name} must be a CUDA tensor for the CUDA extension backend.")
    if tensor.dtype not in {torch.float32, torch.float64}:
        raise OptionalBackendUnavailable(f"{name} must be float32 or float64 for the CUDA extension backend.")


def _damping_or_empty(A: torch.Tensor, damping: torch.Tensor | None) -> torch.Tensor:
    if damping is None:
        return A.new_empty(0)
    _require_cuda_tensor(damping, "damping")
    return damping.reshape(-1).to(dtype=A.dtype, device=A.device)


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
