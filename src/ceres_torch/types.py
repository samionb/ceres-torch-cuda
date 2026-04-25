from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class AutoName(Enum):
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[Any]) -> str:
        return name

    def __str__(self) -> str:
        return self.value


class LinearSolverType(AutoName):
    DENSE_NORMAL_CHOLESKY = auto()
    DENSE_QR = auto()
    SPARSE_NORMAL_CHOLESKY = auto()
    DENSE_SCHUR = auto()
    SPARSE_SCHUR = auto()
    ITERATIVE_SCHUR = auto()
    CGNR = auto()


class PreconditionerType(AutoName):
    IDENTITY = auto()
    JACOBI = auto()
    SCHUR_JACOBI = auto()
    SCHUR_POWER_SERIES_EXPANSION = auto()
    CLUSTER_JACOBI = auto()
    CLUSTER_TRIDIAGONAL = auto()
    SUBSET = auto()


class VisibilityClusteringType(AutoName):
    CANONICAL_VIEWS = auto()
    SINGLE_LINKAGE = auto()


class SparseLinearAlgebraLibraryType(AutoName):
    SUITE_SPARSE = auto()
    EIGEN_SPARSE = auto()
    ACCELERATE_SPARSE = auto()
    CUDA_SPARSE = auto()
    NO_SPARSE = auto()


class LinearSolverOrderingType(AutoName):
    AMD = auto()
    NESDIS = auto()


class DenseLinearAlgebraLibraryType(AutoName):
    EIGEN = auto()
    LAPACK = auto()
    CUDA = auto()


class LoggingType(AutoName):
    SILENT = auto()
    PER_MINIMIZER_ITERATION = auto()


class MinimizerType(AutoName):
    LINE_SEARCH = auto()
    TRUST_REGION = auto()


class LineSearchDirectionType(AutoName):
    STEEPEST_DESCENT = auto()
    NONLINEAR_CONJUGATE_GRADIENT = auto()
    LBFGS = auto()
    BFGS = auto()


class NonlinearConjugateGradientType(AutoName):
    FLETCHER_REEVES = auto()
    POLAK_RIBIERE = auto()
    HESTENES_STIEFEL = auto()


class LineSearchType(AutoName):
    ARMIJO = auto()
    WOLFE = auto()


class TrustRegionStrategyType(AutoName):
    LEVENBERG_MARQUARDT = auto()
    DOGLEG = auto()


class DoglegType(AutoName):
    TRADITIONAL_DOGLEG = auto()
    SUBSPACE_DOGLEG = auto()


class TerminationType(AutoName):
    CONVERGENCE = auto()
    NO_CONVERGENCE = auto()
    FAILURE = auto()
    USER_SUCCESS = auto()
    USER_FAILURE = auto()


class CallbackReturnType(AutoName):
    SOLVER_CONTINUE = auto()
    SOLVER_ABORT = auto()
    SOLVER_TERMINATE_SUCCESSFULLY = auto()


class DumpFormatType(AutoName):
    CONSOLE = auto()
    TEXTFILE = auto()


class NumericDiffMethodType(AutoName):
    CENTRAL = auto()
    FORWARD = auto()
    RIDDERS = auto()


class LineSearchInterpolationType(AutoName):
    BISECTION = auto()
    QUADRATIC = auto()
    CUBIC = auto()


class CovarianceAlgorithmType(AutoName):
    DENSE_SVD = auto()
    SPARSE_QR = auto()


@dataclass
class IterationSummary:
    iteration: int = 0
    step_is_valid: bool = False
    step_is_nonmonotonic: bool = False
    step_is_successful: bool = False
    cost: float = 0.0
    cost_change: float = 0.0
    gradient_max_norm: float = 0.0
    gradient_norm: float = 0.0
    step_norm: float = 0.0
    relative_decrease: float = 0.0
    trust_region_radius: float = 0.0
    eta: float = 0.0
    step_size: float = 0.0
    line_search_function_evaluations: int = 0
    line_search_gradient_evaluations: int = 0
    line_search_iterations: int = 0
    linear_solver_iterations: int = 0
    residual_evaluation_time_in_seconds: float = 0.0
    jacobian_evaluation_time_in_seconds: float = 0.0
    linear_solver_time_in_seconds: float = 0.0
    line_search_time_in_seconds: float = 0.0
    inner_iteration_time_in_seconds: float = 0.0
    iteration_time_in_seconds: float = 0.0
    step_solver_time_in_seconds: float = 0.0
    cumulative_time_in_seconds: float = 0.0


@dataclass
class SolverOptions:
    minimizer_type: MinimizerType = MinimizerType.TRUST_REGION
    line_search_direction_type: LineSearchDirectionType = LineSearchDirectionType.LBFGS
    line_search_type: LineSearchType = LineSearchType.WOLFE
    nonlinear_conjugate_gradient_type: NonlinearConjugateGradientType = (
        NonlinearConjugateGradientType.FLETCHER_REEVES
    )
    max_lbfgs_rank: int = 20
    use_approximate_eigenvalue_bfgs_scaling: bool = False
    line_search_interpolation_type: LineSearchInterpolationType = LineSearchInterpolationType.CUBIC
    min_line_search_step_size: float = 1e-9
    line_search_sufficient_function_decrease: float = 1e-4
    max_line_search_step_contraction: float = 1e-3
    min_line_search_step_contraction: float = 0.6
    max_num_line_search_step_size_iterations: int = 20
    max_num_line_search_direction_restarts: int = 5
    line_search_sufficient_curvature_decrease: float = 0.9
    max_line_search_step_expansion: float = 10.0
    trust_region_strategy_type: TrustRegionStrategyType = TrustRegionStrategyType.LEVENBERG_MARQUARDT
    dogleg_type: DoglegType = DoglegType.TRADITIONAL_DOGLEG
    use_nonmonotonic_steps: bool = False
    max_consecutive_nonmonotonic_steps: int = 5
    max_num_iterations: int = 50
    max_solver_time_in_seconds: float = 1e9
    num_threads: int = 1
    initial_trust_region_radius: float = 1e4
    max_trust_region_radius: float = 1e16
    min_trust_region_radius: float = 1e-32
    min_relative_decrease: float = 1e-3
    min_lm_diagonal: float = 1e-6
    max_lm_diagonal: float = 1e32
    max_num_consecutive_invalid_steps: int = 5
    function_tolerance: float = 1e-6
    gradient_tolerance: float = 1e-10
    parameter_tolerance: float = 1e-8
    linear_solver_type: LinearSolverType = LinearSolverType.DENSE_QR
    preconditioner_type: PreconditionerType = PreconditionerType.JACOBI
    visibility_clustering_type: VisibilityClusteringType = VisibilityClusteringType.CANONICAL_VIEWS
    dense_linear_algebra_library_type: DenseLinearAlgebraLibraryType = (
        DenseLinearAlgebraLibraryType.EIGEN
    )
    sparse_linear_algebra_library_type: SparseLinearAlgebraLibraryType = (
        SparseLinearAlgebraLibraryType.NO_SPARSE
    )
    linear_solver_ordering_type: LinearSolverOrderingType = LinearSolverOrderingType.AMD
    use_explicit_schur_complement: bool = False
    dynamic_sparsity: bool = False
    use_mixed_precision_solves: bool = False
    max_num_refinement_iterations: int = 0
    min_linear_solver_iterations: int = 0
    max_linear_solver_iterations: int = 500
    max_num_spse_iterations: int = 5
    use_spse_initialization: bool = False
    spse_tolerance: float = 0.1
    eta: float = 1e-1
    jacobi_scaling: bool = True
    use_inner_iterations: bool = False
    inner_iteration_tolerance: float = 1e-3
    logging_type: LoggingType = LoggingType.PER_MINIMIZER_ITERATION
    minimizer_progress_to_stdout: bool = False
    check_gradients: bool = False
    gradient_check_relative_precision: float = 1e-8
    gradient_check_numeric_derivative_relative_step_size: float = 1e-6
    update_state_every_iteration: bool = False
    callbacks: list[Callable[[IterationSummary], CallbackReturnType]] = field(default_factory=list)

    def validate(self) -> None:
        checks = [
            (self.max_num_iterations >= 0, "max_num_iterations must be >= 0"),
            (self.max_solver_time_in_seconds >= 0, "max_solver_time_in_seconds must be >= 0"),
            (self.function_tolerance >= 0, "function_tolerance must be >= 0"),
            (self.gradient_tolerance >= 0, "gradient_tolerance must be >= 0"),
            (self.parameter_tolerance >= 0, "parameter_tolerance must be >= 0"),
            (self.num_threads > 0, "num_threads must be > 0"),
            (self.initial_trust_region_radius > 0, "initial_trust_region_radius must be > 0"),
            (self.min_trust_region_radius > 0, "min_trust_region_radius must be > 0"),
            (self.max_trust_region_radius > 0, "max_trust_region_radius must be > 0"),
            (
                self.min_trust_region_radius <= self.initial_trust_region_radius <= self.max_trust_region_radius,
                "trust region radii must satisfy min <= initial <= max",
            ),
            (self.max_lbfgs_rank > 0, "max_lbfgs_rank must be > 0"),
            (self.min_line_search_step_size > 0, "min_line_search_step_size must be > 0"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)


@dataclass
class SolverSummary:
    minimizer_type: MinimizerType = MinimizerType.TRUST_REGION
    termination_type: TerminationType = TerminationType.FAILURE
    message: str = "torch_ceres.solve was not called."
    initial_cost: float = -1.0
    final_cost: float = -1.0
    fixed_cost: float = 0.0
    iterations: list[IterationSummary] = field(default_factory=list)
    num_successful_steps: int = 0
    num_unsuccessful_steps: int = 0
    num_residual_evaluations: int = 0
    num_jacobian_evaluations: int = 0
    num_linear_solves: int = 0
    num_line_search_steps: int = 0
    num_line_search_function_evaluations: int = 0
    num_line_search_gradient_evaluations: int = 0
    line_search_total_time_in_seconds: float = 0.0
    preprocessor_time_in_seconds: float = 0.0
    minimizer_time_in_seconds: float = 0.0
    postprocessor_time_in_seconds: float = 0.0
    residual_evaluation_time_in_seconds: float = 0.0
    jacobian_evaluation_time_in_seconds: float = 0.0
    linear_solver_time_in_seconds: float = 0.0
    inner_iteration_time_in_seconds: float = 0.0
    total_time_in_seconds: float = 0.0
    num_parameter_blocks: int = 0
    num_parameters: int = 0
    num_effective_parameters: int = 0
    num_residual_blocks: int = 0
    num_residuals: int = 0
    linear_solver_type_given: LinearSolverType = LinearSolverType.DENSE_QR
    linear_solver_type_used: LinearSolverType = LinearSolverType.DENSE_QR
    trust_region_strategy_type: TrustRegionStrategyType = TrustRegionStrategyType.LEVENBERG_MARQUARDT
    line_search_direction_type: LineSearchDirectionType = LineSearchDirectionType.LBFGS
    line_search_type: LineSearchType = LineSearchType.WOLFE

    def IsSolutionUsable(self) -> bool:
        return self.termination_type in {
            TerminationType.CONVERGENCE,
            TerminationType.NO_CONVERGENCE,
            TerminationType.USER_SUCCESS,
        }

    def BriefReport(self) -> str:
        return (
            "ceres-torch Solver Report: "
            f"Iterations: {len(self.iterations)}, "
            f"Initial cost: {self.initial_cost:.6e}, "
            f"Final cost: {self.final_cost:.6e}, "
            f"Termination: {self.termination_type.value}"
        )

    def FullReport(self) -> str:
        return "\n".join(
            [
                "Solver Summary (ceres-torch)",
                "",
                f"Minimizer: {self.minimizer_type.value}",
                f"Linear solver: {self.linear_solver_type_used.value}",
                f"Parameter blocks: {self.num_parameter_blocks}",
                f"Parameters: {self.num_parameters}",
                f"Effective parameters: {self.num_effective_parameters}",
                f"Residual blocks: {self.num_residual_blocks}",
                f"Residuals: {self.num_residuals}",
                f"Initial cost: {self.initial_cost:.12e}",
                f"Final cost: {self.final_cost:.12e}",
                f"Successful steps: {self.num_successful_steps}",
                f"Unsuccessful steps: {self.num_unsuccessful_steps}",
                f"Linear solves: {self.num_linear_solves}",
                f"Line search steps: {self.num_line_search_steps}",
                f"Line search function evaluations: {self.num_line_search_function_evaluations}",
                f"Line search gradient evaluations: {self.num_line_search_gradient_evaluations}",
                f"Residual evaluation time (s): {self.residual_evaluation_time_in_seconds:.6f}",
                f"Jacobian evaluation time (s): {self.jacobian_evaluation_time_in_seconds:.6f}",
                f"Linear solver time (s): {self.linear_solver_time_in_seconds:.6f}",
                f"Line search time (s): {self.line_search_total_time_in_seconds:.6f}",
                f"Inner iteration time (s): {self.inner_iteration_time_in_seconds:.6f}",
                f"Preprocessor time (s): {self.preprocessor_time_in_seconds:.6f}",
                f"Minimizer time (s): {self.minimizer_time_in_seconds:.6f}",
                f"Postprocessor time (s): {self.postprocessor_time_in_seconds:.6f}",
                f"Total time (s): {self.total_time_in_seconds:.6f}",
                f"Termination: {self.termination_type.value} ({self.message})",
            ]
        )


DYNAMIC = -1
