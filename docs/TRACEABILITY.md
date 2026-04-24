# Ceres 2.3.0 Traceability Matrix

This matrix maps the local Ceres checkout to `torch-ceres` implementation
areas. Status values:

- `implemented`: present in the current Python package.
- `partial`: usable first slice exists, but behavior/performance parity is not
  complete.
- `planned`: explicit full-parity work remains.

| Ceres area | Reference | torch-ceres area | Status | Acceptance target |
| --- | --- | --- | --- | --- |
| `Problem`, residual/parameter blocks | `include/ceres/problem.h` | `torch_ceres.problem` | implemented | Add/remove/evaluate residuals, bounds, constants, manifolds |
| `Solver::Options`, enums, summaries | `include/ceres/solver.h`, `types.h` | `torch_ceres.types`, `solver` | partial | Option validation and reports match Ceres semantics |
| Cost functions | `cost_function.h`, autodiff/numeric headers | `torch_ceres.costs` | implemented | Analytic/autograd/numeric Jacobians agree on test functions |
| Robust losses | `loss_function.h` | `torch_ceres.losses` | implemented | Values/derivatives match formulas and solver uses row scaling |
| Manifolds | `manifold.h`, sphere/line/product/autodiff | `torch_ceres.manifolds` | partial | Identity/Jacobian properties and quaternion layout parity |
| Rotation helpers | `rotation.h` | `torch_ceres.rotation` | partial | Angle-axis, quaternion, matrix conversions and point rotation |
| Interpolation | `cubic_interpolation.h` | `torch_ceres.interpolation` | partial | Cubic/bicubic sample values and derivatives |
| Trust region minimizer | `trust_region_minimizer.cc` | `torch_ceres.solver` | partial | LM and dogleg convergence on Ceres examples |
| Line search minimizer | `line_search_minimizer.cc` | `torch_ceres.gradient_solver`, `solver` | partial | Armijo/Wolfe, steepest, NCG, BFGS/LBFGS parity |
| Dense linear solvers | dense QR/Cholesky files | `torch_ceres.linear` | partial | QR/Cholesky residual norms match Ceres tolerances |
| Sparse/Schur solvers | Schur, CGNR, sparse Cholesky files | `torch_ceres.linear`, `schur` | partial | Pure PyTorch iterative paths; direct sparse via optional backends |
| Preconditioners | Jacobi, Schur, cluster, subset files | `torch_ceres.linear` | partial | Identity/Jacobi implemented; Schur/cluster/subset parity planned |
| Covariance | `covariance.h`, `covariance_impl.cc` | `torch_ceres.covariance` | partial | Dense SVD covariance blocks; sparse QR backend planned |
| GradientProblemSolver | `gradient_problem_solver.h` | `torch_ceres.gradient_solver` | partial | General unconstrained minimization with line search |
| Callbacks/logging | `iteration_callback.h`, callbacks files | `torch_ceres.callbacks` | implemented | Callback termination and summary visibility |
| Tiny solver | `tiny_solver.h` | `torch_ceres.tiny_solver` | planned | Small fixed-size LM parity |
| C API | `c_api.h` | Not cloned | planned exception | Python callable/module support replaces C ABI |
| Examples/data | `examples`, `data` | `examples`, tests | partial | Port all tutorial examples and BAL/NIST/SLAM validations |
| CUDA | CUDA internal files | PyTorch device + optional backends | partial | CUDA tensors work; extension backends needed for Ceres-scale sparse |

## Full-Parity Backlog

1. Expand solver parity: exact Ceres LM radius update, inexact LM forcing
   sequences, nonmonotonic trust region windows, full projected constrained
   line search, inner iterations, and detailed timing counters.
2. Add native optional sparse backends for SuiteSparse-like sparse QR behavior,
   cuDSS/cuSPARSE Cholesky, and block-Schur CUDA kernels.
3. Port generated bundle-adjustment solver matrix tests and compare against the
   local Ceres binaries once built.
4. Fill exact Ceres covariance sparse QR behavior, rank-deficiency policy, and
   gauge-invariance examples.
5. Port all examples under `C:\Git\ceres-solver\examples` and all public helper
   tests under `internal/ceres/*test.cc`.

