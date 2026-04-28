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
| `Solver::Options`, enums, summaries | `include/ceres/solver.h`, `types.h` | `ceres_torch.types`, `solver` | implemented | Least-squares/gradient option validation, gradient checking, fixed-cost reports, cross-option guards, reports, iteration counters, and timing fields match Ceres-style semantics |
| Cost functions | `cost_function.h`, autodiff/numeric headers | `ceres_torch.costs` | implemented | Analytic/autograd/numeric Jacobians agree on test functions |
| Robust losses | `loss_function.h`, `corrector.cc` | `ceres_torch.losses` | implemented | Values/derivatives match formulas and solver uses Ceres robust correction |
| Manifolds | `manifold.h`, sphere/line/product/autodiff | `ceres_torch.manifolds` | implemented | Identity/Jacobian properties, analytic sphere/line Jacobians, Ceres-style aliases, right-multiply helper, quaternion and Eigen-quaternion layout parity, product and AutoDiff manifold coverage |
| Rotation helpers | `rotation.h` | `ceres_torch.rotation` | implemented | Angle-axis, quaternion, scaled quaternion matrices, quaternion order conversion, legacy Euler helpers, aliases, cross/dot products, point rotation, robust pi-rotation matrix conversion, and small-angle autodiff safety |
| Interpolation | `cubic_interpolation.h` | `ceres_torch.interpolation` | implemented | Catmull-Rom/Ceres Hermite kernel, cubic/bicubic scalar and vector sample values and derivatives, Ceres finite-grid clamping and flat storage layouts, autodiff-compatible chain-rule behavior |
| Trust region minimizer | `trust_region_minimizer.cc` | `ceres_torch.solver` | partial | LM convergence, traditional/subspace dogleg steps, Ceres-style LM radius updates, projected constrained line search, nonmonotonic step evaluator with best-state restoration, inner iterations, progress and detailed timing counters |
| Line search minimizer | `line_search_minimizer.cc` | `ceres_torch.gradient_solver`, `solver` | implemented | Armijo/Wolfe, shared interpolation modes, steepest, NCG, BFGS/LBFGS coverage, direction restarts, active-bound projected-gradient convergence, callback state visibility, counters, and timing reports in first/least-squares solvers |
| Dense linear solvers | dense QR/Cholesky files | `ceres_torch.linear` | implemented | Dense QR and normal Cholesky cover damping, rank-deficient fallback, mixed-precision refinement, shape validation, and residual norms within Ceres-style tolerances |
| Sparse/Schur solvers | Schur, CGNR, sparse Cholesky files | `ceres_torch.linear`, `schur`, `sparse_backends`, `cuda_backends`, `native/cuda` | implemented | Dense Schur with ordering, pure PyTorch CGNR and iterative Schur CG paths, minimum iterative solve counts, SPSE initialization, full option and visibility pass-through to optional Schur backends, SciPy/SuperLU sparse normal and Schur backend, SuiteSparseQR-style covariance hook, PyTorch CUDA sparse/block-Schur backend, opt-in native CUDA extension |
| Preconditioners | Jacobi, Schur, cluster, subset files | `ceres_torch.linear` | implemented | Identity/Jacobi, exact block-Jacobi Schur/cluster/subset aliases, Schur power-series expansion preconditioner, ordered block cluster-tridiagonal approximation, and Ceres-style visibility graph, canonical/single-linkage clustering, cluster-Jacobi, and degree-2 cluster-tridiagonal graph forms |
| Covariance | `covariance.h`, `covariance_impl.cc` | `ceres_torch.covariance`, `sparse_backends` | implemented | Dense SVD/QR covariance blocks, loss toggle, constants, Ceres eigenvalue-ratio rank policy, Ceres-style accessors/output copies, reports, duplicate request guards, optional backend type/shape validation, manifold tangent/ambient covariance, rank summary, SciPy/SuperLU sparse direct covariance backend, optional SuiteSparseQR-style sparse QR covariance backend |
| GradientProblemSolver | `gradient_problem_solver.h` | `ceres_torch.gradient_solver` | implemented | General unconstrained minimization with validation, reports, counters, line search, callback state visibility, manifold validation, and Ceres-style timing fields |
| Callbacks/logging | `iteration_callback.h`, callbacks files | `ceres_torch.callbacks` | implemented | Iteration/evaluation callback behavior and summary visibility |
| Tiny solver | `tiny_solver.h` | `ceres_torch.tiny_solver` | implemented | Small dense LM behavior with Ceres-style statuses, option validation, cost/gradient/step/function termination, and summary/report API |
| C API | `c_api.h` | Not cloned | planned exception | Python callable/module support replaces C ABI |
| Examples/data | `examples`, `data` | `examples`, tests | partial | Port all tutorial examples, More-Garbow-Hillstrom subset, and BAL/NIST/SLAM validations |
| CUDA | CUDA internal files | PyTorch device + optional backends + `native/cuda` | partial | CUDA tensor smoke tests, PyTorch CUDA sparse/block-Schur backend tests, opt-in native extension build/load test |
| Performance benchmarks | internal benchmark/test matrix | `ceres_torch.benchmarking`, `benchmarks` | partial | Opt-in dense, dense/iterative Schur, SPSE initialization, cluster-tridiagonal, sparse-direct, covariance, solver, and CUDA benchmark gates |

## Full-Parity Backlog

1. Expand solver parity: inexact LM forcing sequences, richer constrained
   line-search edge cases, richer inner-iteration ordering beyond the current
   Ceres-style acceptance ordering, and deeper timing breakdowns for less common
   minimizer paths.
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
- `bicubic_interpolation.py`
- `pose_graph_2d.py`
- `circle_fit.py`
- `iteration_callback_example.py`
- `evaluation_callback_example.py`
- `robot_pose_mle.py`
- `simple_bundle_adjuster.py`
- `more_garbow_hillstrom.py`
