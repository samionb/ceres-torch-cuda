from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .costs import AnalyticCostFunction, NormalPrior
from .covariance import Covariance
from .cuda_backends import register_cuda_sparse_backends, torch_cuda_backend_available, unregister_cuda_sparse_backends
from .linear import solve_linear_system
from .problem import Problem
from .solver import solve
from .sparse_backends import native_sparse_backends_available, register_native_sparse_backends, unregister_native_sparse_backends
from .types import LinearSolverType, PreconditionerType, SolverOptions


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    device: str
    dtype: str
    median_seconds: float
    min_seconds: float
    max_seconds: float
    iterations: int
    metric: float


def time_callable(
    name: str,
    fn: Callable[[], float | torch.Tensor],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 3,
) -> BenchmarkResult:
    device = torch.device(device)
    for _ in range(warmup):
        fn()
        _synchronize_if_needed(device)
    timings: list[float] = []
    metric = 0.0
    gcold = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            raw_metric = fn()
            _synchronize_if_needed(device)
            timings.append(time.perf_counter() - start)
            metric = _to_float(raw_metric)
    finally:
        if gcold:
            gc.enable()
    sorted_timings = sorted(timings)
    return BenchmarkResult(
        name=name,
        device=device.type,
        dtype=str(dtype).replace("torch.", ""),
        median_seconds=sorted_timings[len(sorted_timings) // 2],
        min_seconds=min(timings),
        max_seconds=max(timings),
        iterations=repeats,
        metric=metric,
    )


def dense_linear_benchmark(
    *,
    rows: int = 204800,
    cols: int = 256,
    solver_type: LinearSolverType = LinearSolverType.DENSE_QR,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 3,
) -> BenchmarkResult:
    generator = torch.Generator(device="cpu").manual_seed(7)
    A = _randn((rows, cols), generator=generator, dtype=dtype, device=device)
    x_true = _randn((cols,), generator=generator, dtype=dtype, device=device)
    b = A @ x_true

    def run() -> torch.Tensor:
        result = solve_linear_system(A, b, solver_type=solver_type, tolerance=1e-12, max_iterations=cols * 4)
        return torch.linalg.norm(A @ result.x - b)

    return time_callable(
        f"linear/{solver_type.value}/{rows}x{cols}",
        run,
        device=device,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )


def schur_benchmark(
    *,
    rows: int = 409600,
    eliminate: int = 512,
    remain: int = 128,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 3,
) -> BenchmarkResult:
    generator = torch.Generator(device="cpu").manual_seed(11)
    cols = eliminate + remain
    A = _randn((rows, cols), generator=generator, dtype=dtype, device=device)
    x_true = _randn((cols,), generator=generator, dtype=dtype, device=device)
    b = A @ x_true

    def run() -> torch.Tensor:
        result = solve_linear_system(A, b, solver_type=LinearSolverType.DENSE_SCHUR, num_eliminate=eliminate)
        return torch.linalg.norm(A @ result.x - b)

    return time_callable(
        f"schur/dense/{rows}x{cols}/elim{eliminate}",
        run,
        device=device,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )


def iterative_schur_benchmark(
    *,
    rows: int = 409600,
    eliminate: int = 512,
    remain: int = 128,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 3,
    preconditioner_type: PreconditionerType = PreconditionerType.SCHUR_JACOBI,
    use_spse_initialization: bool = False,
    max_num_spse_iterations: int = 5,
) -> BenchmarkResult:
    generator = torch.Generator(device="cpu").manual_seed(13)
    cols = eliminate + remain
    A = _randn((rows, cols), generator=generator, dtype=dtype, device=device)
    x_true = _randn((cols,), generator=generator, dtype=dtype, device=device)
    b = A @ x_true

    def run() -> torch.Tensor:
        result = solve_linear_system(
            A,
            b,
            solver_type=LinearSolverType.ITERATIVE_SCHUR,
            num_eliminate=eliminate,
            preconditioner_type=preconditioner_type,
            block_sizes=[eliminate, remain],
            tolerance=1e-12,
            max_iterations=max(20, remain * 4),
            use_spse_initialization=use_spse_initialization,
            max_num_spse_iterations=max_num_spse_iterations,
        )
        return torch.linalg.norm(A @ result.x - b)

    suffix = "/spse_init" if use_spse_initialization else ""
    return time_callable(
        f"schur/iterative/{preconditioner_type.value}{suffix}/{rows}x{cols}/elim{eliminate}",
        run,
        device=device,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )


def iterative_schur_spse_benchmark(
    *,
    rows: int = 409600,
    eliminate: int = 512,
    remain: int = 128,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 3,
    max_num_spse_iterations: int = 5,
) -> BenchmarkResult:
    return iterative_schur_benchmark(
        rows=rows,
        eliminate=eliminate,
        remain=remain,
        device=device,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
        preconditioner_type=PreconditionerType.SCHUR_POWER_SERIES_EXPANSION,
        use_spse_initialization=True,
        max_num_spse_iterations=max_num_spse_iterations,
    )


def cluster_tridiagonal_benchmark(
    *,
    rows: int = 204800,
    cols: int = 256,
    num_blocks: int = 16,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 3,
) -> BenchmarkResult:
    generator = torch.Generator(device="cpu").manual_seed(29)
    A = _randn((rows, cols), generator=generator, dtype=dtype, device=device)
    x_true = _randn((cols,), generator=generator, dtype=dtype, device=device)
    b = A @ x_true
    block_sizes = _balanced_block_sizes(cols, num_blocks)

    def run() -> torch.Tensor:
        result = solve_linear_system(
            A,
            b,
            solver_type=LinearSolverType.CGNR,
            preconditioner_type=PreconditionerType.CLUSTER_TRIDIAGONAL,
            block_sizes=block_sizes,
            tolerance=1e-12,
            max_iterations=max(50, cols * 4),
        )
        return torch.linalg.norm(A @ result.x - b)

    return time_callable(
        f"linear/CGNR/CLUSTER_TRIDIAGONAL/{rows}x{cols}/blocks{len(block_sizes)}",
        run,
        device=device,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )


def sparse_direct_benchmark(
    *,
    rows: int = 200000,
    cols: int = 320,
    density: float = 0.08,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 3,
) -> BenchmarkResult:
    if not native_sparse_backends_available():
        raise RuntimeError("Native sparse backends are not available; install the 'sparse' extra.")
    generator = torch.Generator(device="cpu").manual_seed(23)
    A = _randn((rows, cols), generator=generator, dtype=dtype, device=device)
    mask = (torch.rand((rows, cols), generator=generator, dtype=dtype) < density).to(device=device)
    A = A * mask
    A[:cols, :] = A[:cols, :] + torch.eye(cols, dtype=dtype, device=device)
    x_true = _randn((cols,), generator=generator, dtype=dtype, device=device)
    b = A @ x_true

    def run() -> torch.Tensor:
        register_native_sparse_backends()
        try:
            result = solve_linear_system(A, b, solver_type=LinearSolverType.SPARSE_NORMAL_CHOLESKY)
        finally:
            unregister_native_sparse_backends()
        return torch.linalg.norm(A @ result.x - b)

    return time_callable(
        f"linear/SPARSE_NORMAL_CHOLESKY/scipy/{rows}x{cols}/density{density:g}",
        run,
        device=device,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )


def cuda_block_schur_benchmark(
    *,
    rows: int = 409600,
    eliminate: int = 512,
    remain: int = 128,
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 3,
) -> BenchmarkResult:
    if not torch_cuda_backend_available():
        raise RuntimeError("PyTorch CUDA backend is not available.")
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(31)
    cols = eliminate + remain
    A = _randn((rows, cols), generator=generator, dtype=dtype, device=device)
    x_true = _randn((cols,), generator=generator, dtype=dtype, device=device)
    b = A @ x_true

    def run() -> torch.Tensor:
        register_cuda_sparse_backends()
        try:
            result = solve_linear_system(
                A,
                b,
                solver_type=LinearSolverType.SPARSE_SCHUR,
                num_eliminate=eliminate,
            )
        finally:
            unregister_cuda_sparse_backends()
        return torch.linalg.norm(A @ result.x - b)

    return time_callable(
        f"cuda/block_schur/torch/{rows}x{cols}/elim{eliminate}",
        run,
        device=device,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )


def solver_curve_fit_benchmark(
    *,
    num_observations: int = 40000,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 1,
) -> BenchmarkResult:
    t = torch.linspace(0.0, 2.0, num_observations, dtype=dtype, device=device)
    y = 2.5 * torch.exp(-1.3 * t) + 0.7

    def run() -> torch.Tensor:
        params = torch.tensor([1.0, -0.5, 0.0], dtype=dtype, device=device)
        problem = Problem()
        problem.AddResidualBlock(
            AnalyticCostFunction(
                lambda p: p[0] * torch.exp(p[1] * t) + p[2] - y,
                lambda p: [
                    torch.stack(
                        [
                            torch.exp(p[1] * t),
                            p[0] * t * torch.exp(p[1] * t),
                            torch.ones_like(t),
                        ],
                        dim=1,
                    )
                ],
                [3],
                num_observations,
            ),
            None,
            [params],
        )
        summary = solve(SolverOptions(max_num_iterations=30, gradient_tolerance=1e-12), problem)
        return torch.tensor(summary.final_cost, dtype=dtype, device=device)

    return time_callable(
        f"solver/curve_fit/{num_observations}",
        run,
        device=device,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )


def covariance_benchmark(
    *,
    rows: int = 100000,
    cols: int = 160,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 1,
) -> BenchmarkResult:
    generator = torch.Generator(device="cpu").manual_seed(19)
    A = _randn((rows, cols), generator=generator, dtype=dtype, device=device)
    x = torch.zeros(cols, dtype=dtype, device=device)

    def run() -> torch.Tensor:
        problem = Problem()
        block = problem.AddParameterBlock(x.clone())
        problem.AddResidualBlock(NormalPrior(A, torch.zeros(cols, dtype=dtype, device=device)), None, [block.tensor])
        covariance = Covariance()
        if not covariance.compute([(block, block)], problem):
            raise RuntimeError("Covariance benchmark problem is rank deficient")
        return torch.linalg.norm(covariance.get_covariance_block(block, block))

    return time_callable(
        f"covariance/dense_svd/{rows}x{cols}",
        run,
        device=device,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )


def run_default_benchmarks(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 3,
) -> list[BenchmarkResult]:
    return [
        dense_linear_benchmark(device=device, dtype=dtype, warmup=warmup, repeats=repeats),
        dense_linear_benchmark(
            solver_type=LinearSolverType.CGNR,
            device=device,
            dtype=dtype,
            warmup=warmup,
            repeats=repeats,
        ),
        schur_benchmark(device=device, dtype=dtype, warmup=warmup, repeats=repeats),
        iterative_schur_benchmark(device=device, dtype=dtype, warmup=warmup, repeats=repeats),
        iterative_schur_spse_benchmark(device=device, dtype=dtype, warmup=warmup, repeats=repeats),
        cluster_tridiagonal_benchmark(device=device, dtype=dtype, warmup=warmup, repeats=repeats),
        solver_curve_fit_benchmark(device=device, dtype=dtype, warmup=warmup, repeats=max(1, repeats // 2)),
        covariance_benchmark(device=device, dtype=dtype, warmup=warmup, repeats=max(1, repeats // 2)),
    ]


def format_benchmark_results(results: Sequence[BenchmarkResult]) -> str:
    lines = ["name,device,dtype,median_seconds,min_seconds,max_seconds,iterations,metric"]
    for result in results:
        lines.append(
            f"{result.name},{result.device},{result.dtype},"
            f"{result.median_seconds:.9f},{result.min_seconds:.9f},{result.max_seconds:.9f},"
            f"{result.iterations},{result.metric:.9e}"
        )
    return "\n".join(lines)


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _to_float(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=dtype).to(device=device)


def _balanced_block_sizes(total: int, num_blocks: int) -> list[int]:
    num_blocks = max(1, min(total, num_blocks))
    base = total // num_blocks
    remainder = total % num_blocks
    return [base + (1 if i < remainder else 0) for i in range(num_blocks)]
