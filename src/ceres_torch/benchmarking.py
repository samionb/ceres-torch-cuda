from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .costs import AutoDiffCostFunction, NormalPrior
from .covariance import Covariance
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
    repeats: int = 5,
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
    rows: int = 160,
    cols: int = 40,
    solver_type: LinearSolverType = LinearSolverType.DENSE_QR,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 5,
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
    rows: int = 240,
    eliminate: int = 36,
    remain: int = 18,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 5,
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
    rows: int = 240,
    eliminate: int = 36,
    remain: int = 18,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 5,
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
            preconditioner_type=PreconditionerType.SCHUR_JACOBI,
            block_sizes=[eliminate, remain],
            tolerance=1e-12,
            max_iterations=max(20, remain * 4),
        )
        return torch.linalg.norm(A @ result.x - b)

    return time_callable(
        f"schur/iterative/{rows}x{cols}/elim{eliminate}",
        run,
        device=device,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
    )


def sparse_direct_benchmark(
    *,
    rows: int = 200,
    cols: int = 48,
    density: float = 0.15,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 5,
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


def solver_curve_fit_benchmark(
    *,
    num_observations: int = 80,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 3,
) -> BenchmarkResult:
    t = torch.linspace(0.0, 2.0, num_observations, dtype=dtype, device=device)
    y = 2.5 * torch.exp(-1.3 * t) + 0.7

    def run() -> torch.Tensor:
        params = torch.tensor([1.0, -0.5, 0.0], dtype=dtype, device=device)
        problem = Problem()
        for ti, yi in zip(t, y):
            problem.AddResidualBlock(
                AutoDiffCostFunction(lambda p, ti=ti, yi=yi: p[0] * torch.exp(p[1] * ti) + p[2] - yi, [3], 1),
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
    rows: int = 120,
    cols: int = 24,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warmup: int = 1,
    repeats: int = 3,
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
    repeats: int = 5,
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
