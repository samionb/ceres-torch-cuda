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
- Optional-extension placeholders for future cuDSS/cuSPARSE and block-Schur
  acceleration.

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
