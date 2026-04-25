# Ceres 2.3.0 Traceability Matrix

This matrix maps the local Ceres checkout to `ceres-torch` implementation
areas. Status values:

- `implemented`: present in the current Python package.
- `partial`: usable first slice exists, but behavior/performance parity is not
  complete.
- `planned`: explicit full-parity work remains.

| Ceres area | Reference | ceres-torch area | Status | Acceptance target |
| --- | --- | --- | --- | --- |
| `Problem`, residual/parameter blocks | `include/ceres/problem.h` | `ceres_torch.problem` | implemented | Add/remove/evaluate residuals, bounds, constants, manifolds, public introspection |
| `Solver::Options`, enums, summaries | `include/ceres/solver.h`, `types.h` | `ceres_torch.types`, `solver` | partial | Least-squares/gradient option validation and reports match Ceres semantics |
| Cost functions | `cost_function.h`, autodiff/numeric headers | `ceres_torch.costs` | implemented | Analytic/autograd/numeric Jacobians agree on test functions |
| Robust losses | `loss_function.h`, `corrector.cc` | `ceres_torch.losses` | implemented | Values/derivatives match formulas and solver uses Ceres robust correction |
| Manifolds | `manifold.h`, sphere/line/product/autodiff | `ceres_torch.manifolds` | partial | Identity/Jacobian properties, analytic sphere/line Jacobians, Ceres-style aliases, right-multiply helper, quaternion layout parity |
| Rotation helpers | `rotation.h` | `ceres_torch.rotation` | partial | Angle-axis, quaternion, scaled quaternion matrices, legacy Euler helpers, aliases, cross/dot products, and point rotation |
| Interpolation | `cubic_interpolation.h` | `ceres_torch.interpolation` | partial | Catmull-Rom/Ceres Hermite kernel, cubic/bicubic scalar and vector sample values and derivatives |
| Trust region minimizer | `trust_region_minimizer.cc` | `ceres_torch.solver` | partial | LM/dogleg convergence, radius updates, projected constrained line search, nonmonotonic windows with best-state restoration, inner iterations, progress counters |
| Line search minimizer | `line_search_minimizer.cc` | `ceres_torch.gradient_solver`, `solver` | partial | Armijo/Wolfe, shared interpolation modes, steepest, NCG, BFGS/LBFGS coverage, counters in first/least-squares solvers |
| Dense linear solvers | dense QR/Cholesky files | `ceres_torch.linear` | partial | QR/Cholesky residual norms match Ceres tolerances |
| Sparse/Schur solvers | Schur, CGNR, sparse Cholesky files | `ceres_torch.linear`, `schur`, `sparse_backends`, `cuda_backends`, `native/cuda` | partial | Dense Schur with ordering, pure PyTorch iterative paths, SciPy/SuperLU sparse normal and Schur backend, PyTorch CUDA sparse/block-Schur backend, opt-in native CUDA extension |
| Preconditioners | Jacobi, Schur, cluster, subset files | `ceres_torch.linear` | partial | Identity/Jacobi, exact block-Jacobi Schur/cluster/subset aliases from parameter block structure, specialized cluster graph forms planned |
| Covariance | `covariance.h`, `covariance_impl.cc` | `ceres_torch.covariance`, `sparse_backends` | partial | Dense SVD/QR covariance blocks, loss toggle, constants, Ceres eigenvalue-ratio rank policy, rank summary, SciPy/SuperLU sparse direct covariance backend |
| GradientProblemSolver | `gradient_problem_solver.h` | `ceres_torch.gradient_solver` | partial | General unconstrained minimization with validation, reports, counters, line search |
| Callbacks/logging | `iteration_callback.h`, callbacks files | `ceres_torch.callbacks` | implemented | Iteration/evaluation callback behavior and summary visibility |
| Tiny solver | `tiny_solver.h` | `ceres_torch.tiny_solver` | partial | Small fixed-size LM parity with summary/report API |
| C API | `c_api.h` | Not cloned | planned exception | Python callable/module support replaces C ABI |
| Examples/data | `examples`, `data` | `examples`, tests | partial | Port all tutorial examples and BAL/NIST/SLAM validations |
| CUDA | CUDA internal files | PyTorch device + optional backends + `native/cuda` | partial | CUDA tensor smoke tests, PyTorch CUDA sparse/block-Schur backend tests, opt-in native extension build/load test |
| Performance benchmarks | internal benchmark/test matrix | `ceres_torch.benchmarking`, `benchmarks` | partial | Opt-in dense, Schur, iterative, covariance, solver, and CUDA benchmark gates |

## Full-Parity Backlog

1. Expand solver parity: exact Ceres LM radius update, inexact LM forcing
   sequences, deeper nonmonotonic step-evaluator parity, richer constrained
   line-search edge cases, richer inner-iteration ordering, and detailed
   timing counters.
2. Extend optional sparse backends beyond the SciPy/SuperLU CPU direct path and
   PyTorch CUDA sparse/block-Schur path with SuiteSparse-like sparse QR behavior
   and production cuDSS/cuSPARSE Cholesky kernels where PyTorch cannot expose the
   needed primitive directly.
3. Port generated bundle-adjustment solver matrix tests and compare against the
   local Ceres binaries once built using `GoldenSolverResult` assertions.
4. Fill exact Ceres covariance sparse QR behavior and broader gauge-invariance
   examples beyond the dense rank/nullity policy now covered.
5. Port all examples under `C:\Git\ceres-solver\examples` and all public helper
   tests under `internal/ceres/*test.cc`.

## Ported Examples

- `hello_world.py`
- `hello_world_analytic_diff.py`
- `hello_world_numeric_diff.py`
- `curve_fitting.py`
- `robust_curve_fitting.py`
- `powell.py`
- `rosenbrock.py`
- `rosenbrock_analytic_diff.py`
- `rosenbrock_numeric_diff.py`
- `tiny_bundle_adjustment.py`
- `sampled_function.py`
- `pose_graph_2d.py`
- `circle_fit.py`
- `iteration_callback_example.py`
- `evaluation_callback_example.py`
