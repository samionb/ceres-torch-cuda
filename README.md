# ceres-torch

`ceres-torch` is a pure Python/PyTorch implementation of the public modeling and
solver concepts exposed by Ceres Solver. The project targets Ceres 2.3.0 behavior
as the compatibility reference while using idiomatic Python objects, dataclasses,
and PyTorch tensors.

This first implementation slice includes:

- Ceres-like `Problem`, `SolverOptions`, `solve`, `CostFunction`, `LossFunction`,
  `Manifold`, `Covariance`, and `GradientProblem` APIs. Common Ceres-style
  CamelCase aliases are also available on `Problem`.
- PyTorch autograd, analytic, and numeric-differentiated residual blocks.
- Robust losses, common manifolds, rotation utilities, interpolation helpers,
  Ceres-style robust loss correction, dense/iterative linear solvers,
  trust-region LM/dogleg, line-search fallback, and covariance blocks.
- CPU execution with device/dtype propagation. CUDA works when the installed
  PyTorch build supports it.
- Optional SciPy/SuperLU sparse direct backends for sparse normal equations,
  sparse Schur, and covariance. CUDA sparse/block-Schur requests can use a
  PyTorch CUDA backend; an opt-in native CUDA extension tier remains available
  for future custom kernels.

Full Ceres parity is intentionally tracked as a staged engineering effort in
[`docs/TRACEABILITY.md`](docs/TRACEABILITY.md).

## Quick Start

```python
import torch
import ceres_torch as tc

x = torch.tensor([0.5], dtype=torch.float64)

problem = tc.Problem()
problem.add_residual_block(
    tc.AutoDiffCostFunction(lambda x: 10.0 - x, [1]),
    None,
    [x],
)

summary = tc.solve(tc.SolverOptions(max_num_iterations=25), problem)
print(x, summary.BriefReport())
```

## Development

```powershell
cd C:\Git\ceres-torch-cuda
python -m pytest
```

Opt-in performance gates and benchmark CSV output:

```powershell
$env:CERES_TORCH_RUN_BENCHMARKS=1
python -m pytest -m performance
python benchmarks\run_benchmarks.py --device cpu --repeats 5
```

Set `CERES_TORCH_BENCHMARK_MAX_SECONDS` to adjust the per-case pytest
performance gate for slower or faster machines.

Optional native sparse/direct backends:

```powershell
python -m pip install -e ".[sparse]"
```

```python
import ceres_torch as tc

tc.register_native_sparse_backends()
```

Once registered, `SPARSE_NORMAL_CHOLESKY`, `SPARSE_SCHUR`, and
`SPARSE_QR` covariance requests use SciPy/SuperLU when the system is suitable,
then fall back to the pure PyTorch paths if the optional backend is unavailable.
The `tc.sparse_direct_benchmark(...)` helper provides a small smoke benchmark
for this native sparse path.

Optional CUDA backend:

```powershell
python -m pytest tests\test_cuda_smoke.py tests\test_cuda_extensions.py
```

```python
import ceres_torch as tc

tc.register_cuda_sparse_backends()
```

The default CUDA backend is pure PyTorch and requires only a CUDA-enabled
PyTorch build and a CUDA device. The native extension sources under
`native/cuda` can still be built explicitly for experimental custom kernels:

```powershell
python -m pip install -e ".[cuda]"
$env:CERES_TORCH_BUILD_CUDA_EXTENSIONS=1
python -m pytest -m native_extension
```
