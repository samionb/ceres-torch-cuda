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
| `Solver::Options`, enums, summaries | `include/ceres/solver.h`, `types.h` | `ceres_torch.types`, `solver` | partial | Option validation and reports match Ceres semantics |
| Cost functions | `cost_function.h`, autodiff/numeric headers | `ceres_torch.costs` | implemented | Analytic/autograd/numeric Jacobians agree on test functions |
| Robust losses | `loss_function.h`, `corrector.cc` | `ceres_torch.losses` | implemented | Values/derivatives match formulas and solver uses Ceres robust correction |
| Manifolds | `manifold.h`, sphere/line/product/autodiff | `ceres_torch.manifolds` | partial | Identity/Jacobian properties and quaternion layout parity |
| Rotation helpers | `rotation.h` | `ceres_torch.rotation` | partial | Angle-axis, quaternion, matrix conversions, cross products, and point rotation |
| Interpolation | `cubic_interpolation.h` | `ceres_torch.interpolation` | partial | Cubic/bicubic sample values and derivatives |
| Trust region minimizer | `trust_region_minimizer.cc` | `ceres_torch.solver` | partial | LM/dogleg convergence, radius updates, nonmonotonic windows, progress counters |
| Line search minimizer | `line_search_minimizer.cc` | `ceres_torch.gradient_solver`, `solver` | partial | Armijo/Wolfe, steepest, NCG, BFGS/LBFGS coverage in first/least-squares solvers |
| Dense linear solvers | dense QR/Cholesky files | `ceres_torch.linear` | partial | QR/Cholesky residual norms match Ceres tolerances |
| Sparse/Schur solvers | Schur, CGNR, sparse Cholesky files | `ceres_torch.linear`, `schur` | partial | Dense Schur with ordering, pure PyTorch iterative paths; direct sparse via optional backends |
| Preconditioners | Jacobi, Schur, cluster, subset files | `ceres_torch.linear` | partial | Identity/Jacobi plus pure-core diagonal Schur/cluster/subset aliases; exact block forms planned |
| Covariance | `covariance.h`, `covariance_impl.cc` | `ceres_torch.covariance` | partial | Dense SVD/QR covariance blocks, loss toggle, constants, rank policy; sparse backend planned |
| GradientProblemSolver | `gradient_problem_solver.h` | `ceres_torch.gradient_solver` | partial | General unconstrained minimization with line search |
| Callbacks/logging | `iteration_callback.h`, callbacks files | `ceres_torch.callbacks` | implemented | Iteration/evaluation callback behavior and summary visibility |
| Tiny solver | `tiny_solver.h` | `ceres_torch.tiny_solver` | partial | Small fixed-size LM parity with summary/report API |
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

## Ported Examples

- `hello_world.py`
- `curve_fitting.py`
- `robust_curve_fitting.py`
- `powell.py`
- `rosenbrock.py`
- `tiny_bundle_adjustment.py`
- `sampled_function.py`
- `pose_graph_2d.py`
