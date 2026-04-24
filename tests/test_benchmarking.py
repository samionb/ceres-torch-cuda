import math
import os

import pytest
import torch

import ceres_torch as tc


RUN_BENCHMARKS = os.environ.get("CERES_TORCH_RUN_BENCHMARKS") == "1"
BENCHMARK_MAX_SECONDS = float(os.environ.get("CERES_TORCH_BENCHMARK_MAX_SECONDS", "10.0"))


def test_benchmark_harness_smoke_and_csv_format() -> None:
    result = tc.dense_linear_benchmark(rows=24, cols=6, warmup=0, repeats=1)
    text = tc.format_benchmark_results([result])

    assert result.name.startswith("linear/")
    assert result.median_seconds >= 0.0
    assert result.metric < 1e-8
    assert text.splitlines()[0] == "name,device,dtype,median_seconds,min_seconds,max_seconds,iterations,metric"
    assert result.name in text


@pytest.mark.performance
@pytest.mark.skipif(not RUN_BENCHMARKS, reason="Set CERES_TORCH_RUN_BENCHMARKS=1 to run performance gates")
def test_cpu_default_benchmark_suite_performance_gate() -> None:
    results = tc.run_default_benchmarks(device="cpu", dtype=torch.float64, warmup=0, repeats=1)

    assert len(results) == 5
    for result in results:
        assert math.isfinite(result.metric)
        assert result.median_seconds <= BENCHMARK_MAX_SECONDS, result
    residual_results = [r for r in results if r.name.startswith(("linear/", "schur/"))]
    assert all(r.metric < 1e-6 for r in residual_results)
    solver_results = [r for r in results if r.name.startswith("solver/")]
    assert all(r.metric < 1e-8 for r in solver_results)


@pytest.mark.performance
@pytest.mark.skipif(not RUN_BENCHMARKS, reason="Set CERES_TORCH_RUN_BENCHMARKS=1 to run performance gates")
@pytest.mark.skipif(not tc.cuda_available(), reason="CUDA is not available in this PyTorch environment")
def test_cuda_dense_linear_benchmark_performance_gate() -> None:
    result = tc.dense_linear_benchmark(rows=64, cols=16, device="cuda", dtype=torch.float64, warmup=1, repeats=2)

    assert result.device == "cuda"
    assert result.metric < 1e-6
    assert result.median_seconds <= BENCHMARK_MAX_SECONDS
