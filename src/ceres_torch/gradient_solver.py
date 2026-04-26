from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .costs import AutoDiffFirstOrderFunction, FirstOrderFunction
from .line_search import next_line_search_step_size
from .manifolds import EuclideanManifold, Manifold
from .types import (
    CallbackReturnType,
    IterationSummary,
    LineSearchDirectionType,
    LineSearchInterpolationType,
    LineSearchType,
    LoggingType,
    NonlinearConjugateGradientType,
    TerminationType,
)


@dataclass
class GradientProblem:
    function: FirstOrderFunction
    manifold: Optional[Manifold] = None

    @classmethod
    def from_callable(cls, functor, size: int, manifold: Optional[Manifold] = None) -> "GradientProblem":
        return cls(AutoDiffFirstOrderFunction(functor), manifold or EuclideanManifold(size))


@dataclass
class GradientProblemSolverOptions:
    line_search_direction_type: LineSearchDirectionType = LineSearchDirectionType.LBFGS
    line_search_type: LineSearchType = LineSearchType.WOLFE
    nonlinear_conjugate_gradient_type: NonlinearConjugateGradientType = NonlinearConjugateGradientType.FLETCHER_REEVES
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
    max_num_iterations: int = 50
    max_solver_time_in_seconds: float = 1e9
    function_tolerance: float = 1e-6
    gradient_tolerance: float = 1e-10
    parameter_tolerance: float = 1e-8
    logging_type: LoggingType = LoggingType.PER_MINIMIZER_ITERATION
    minimizer_progress_to_stdout: bool = False
    update_state_every_iteration: bool = False
    callbacks: list = field(default_factory=list)

    def validate(self) -> None:
        checks = [
            (self.max_num_iterations >= 0, "max_num_iterations must be >= 0"),
            (self.max_solver_time_in_seconds >= 0, "max_solver_time_in_seconds must be >= 0"),
            (self.function_tolerance >= 0, "function_tolerance must be >= 0"),
            (self.gradient_tolerance >= 0, "gradient_tolerance must be >= 0"),
            (self.parameter_tolerance >= 0, "parameter_tolerance must be >= 0"),
            (self.max_lbfgs_rank > 0, "max_lbfgs_rank must be > 0"),
            (self.min_line_search_step_size > 0, "min_line_search_step_size must be > 0"),
            (self.max_num_line_search_step_size_iterations > 0, "max_num_line_search_step_size_iterations must be > 0"),
            (self.max_num_line_search_direction_restarts >= 0, "max_num_line_search_direction_restarts must be >= 0"),
            (
                0.0 < self.line_search_sufficient_function_decrease < 1.0,
                "line_search_sufficient_function_decrease must be in (0, 1)",
            ),
            (
                0.0 < self.line_search_sufficient_curvature_decrease < 1.0,
                "line_search_sufficient_curvature_decrease must be in (0, 1)",
            ),
            (self.max_line_search_step_expansion > 1.0, "max_line_search_step_expansion must be > 1"),
            (
                0.0 < self.min_line_search_step_contraction < 1.0,
                "min_line_search_step_contraction must be in (0, 1)",
            ),
            (
                0.0 < self.max_line_search_step_contraction < 1.0,
                "max_line_search_step_contraction must be in (0, 1)",
            ),
            (
                self.max_line_search_step_contraction <= self.min_line_search_step_contraction,
                "max_line_search_step_contraction must be <= min_line_search_step_contraction",
            ),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)


@dataclass
class GradientProblemSolverSummary:
    termination_type: TerminationType = TerminationType.FAILURE
    message: str = "torch_ceres.gradient_solve was not called."
    initial_cost: float = -1.0
    final_cost: float = -1.0
    iterations: list[IterationSummary] = field(default_factory=list)
    num_cost_evaluations: int = 0
    num_gradient_evaluations: int = 0
    num_successful_steps: int = 0
    num_unsuccessful_steps: int = 0
    num_line_search_steps: int = 0
    num_line_search_function_evaluations: int = 0
    num_line_search_gradient_evaluations: int = 0
    num_line_search_direction_restarts: int = 0
    total_time_in_seconds: float = 0.0
    cost_evaluation_time_in_seconds: float = 0.0
    gradient_evaluation_time_in_seconds: float = 0.0
    line_search_polynomial_minimization_time_in_seconds: float = 0.0
    line_search_total_time_in_seconds: float = 0.0
    num_parameters: int = -1
    num_tangent_parameters: int = -1
    line_search_direction_type: LineSearchDirectionType = LineSearchDirectionType.LBFGS
    line_search_type: LineSearchType = LineSearchType.WOLFE
    line_search_interpolation_type: LineSearchInterpolationType = LineSearchInterpolationType.CUBIC
    nonlinear_conjugate_gradient_type: NonlinearConjugateGradientType = NonlinearConjugateGradientType.FLETCHER_REEVES
    max_lbfgs_rank: int = -1

    def IsSolutionUsable(self) -> bool:
        return self.termination_type in {TerminationType.CONVERGENCE, TerminationType.NO_CONVERGENCE, TerminationType.USER_SUCCESS}

    def BriefReport(self) -> str:
        return (
            "ceres-torch Gradient Report: "
            f"Iterations: {len(self.iterations)}, "
            f"Initial cost: {self.initial_cost:.6e}, "
            f"Final cost: {self.final_cost:.6e}, "
            f"Termination: {self.termination_type.value}"
        )

    def FullReport(self) -> str:
        return "\n".join(
            [
                "Gradient Solver Summary (ceres-torch)",
                "",
                f"Line search direction: {self.line_search_direction_type.value}",
                f"Line search type: {self.line_search_type.value}",
                f"Parameters: {self.num_parameters}",
                f"Tangent parameters: {self.num_tangent_parameters}",
                f"Initial cost: {self.initial_cost:.12e}",
                f"Final cost: {self.final_cost:.12e}",
                f"Iterations: {len(self.iterations)}",
                f"Cost evaluations: {self.num_cost_evaluations}",
                f"Gradient evaluations: {self.num_gradient_evaluations}",
                f"Successful steps: {self.num_successful_steps}",
                f"Unsuccessful steps: {self.num_unsuccessful_steps}",
                f"Line search steps: {self.num_line_search_steps}",
                f"Line search function evaluations: {self.num_line_search_function_evaluations}",
                f"Line search gradient evaluations: {self.num_line_search_gradient_evaluations}",
                f"Line search direction restarts: {self.num_line_search_direction_restarts}",
                f"Cost evaluation time (s): {self.cost_evaluation_time_in_seconds:.6f}",
                f"Gradient evaluation time (s): {self.gradient_evaluation_time_in_seconds:.6f}",
                f"Line search polynomial time (s): {self.line_search_polynomial_minimization_time_in_seconds:.6f}",
                f"Line search time (s): {self.line_search_total_time_in_seconds:.6f}",
                f"Total time (s): {self.total_time_in_seconds:.6f}",
                f"Termination: {self.termination_type.value} ({self.message})",
            ]
        )


def gradient_solve(
    options: GradientProblemSolverOptions,
    problem: GradientProblem,
    parameters: torch.Tensor,
) -> GradientProblemSolverSummary:
    options.validate()
    start = time.perf_counter()
    manifold = problem.manifold or EuclideanManifold(parameters.numel())
    if manifold.ambient_size != parameters.numel():
        raise ValueError("GradientProblem manifold ambient size must match parameters")
    work = parameters.detach().clone()
    summary = GradientProblemSolverSummary(
        num_parameters=parameters.numel(),
        num_tangent_parameters=manifold.tangent_size,
        line_search_direction_type=options.line_search_direction_type,
        line_search_type=options.line_search_type,
        line_search_interpolation_type=options.line_search_interpolation_type,
        nonlinear_conjugate_gradient_type=options.nonlinear_conjugate_gradient_type,
        max_lbfgs_rank=options.max_lbfgs_rank,
    )
    value, _ambient_grad, evaluation_time = _timed_value_and_gradient(problem.function, work)
    summary.num_cost_evaluations += 1
    summary.num_gradient_evaluations += 1
    summary.gradient_evaluation_time_in_seconds += evaluation_time
    summary.initial_cost = float(value.detach().cpu())
    summary.final_cost = summary.initial_cost
    previous_grad: Optional[torch.Tensor] = None
    previous_direction: Optional[torch.Tensor] = None
    s_history: list[torch.Tensor] = []
    y_history: list[torch.Tensor] = []
    inverse_hessian: Optional[torch.Tensor] = None

    for iteration in range(options.max_num_iterations + 1):
        iter_start = time.perf_counter()
        value, ambient_grad, evaluation_time = _timed_value_and_gradient(problem.function, work)
        summary.num_cost_evaluations += 1
        summary.num_gradient_evaluations += 1
        summary.gradient_evaluation_time_in_seconds += evaluation_time
        plus_jac = manifold.plus_jacobian(work.reshape(-1)).to(dtype=work.dtype, device=work.device)
        tangent_grad = plus_jac.T @ ambient_grad.reshape(-1)
        grad_norm = torch.linalg.norm(tangent_grad)
        grad_max = torch.max(torch.abs(tangent_grad)) if tangent_grad.numel() else tangent_grad.new_tensor(0.0)
        cost = float(value.detach().cpu())
        iter_summary = IterationSummary(
            iteration=iteration,
            cost=cost,
            gradient_norm=float(grad_norm.detach().cpu()),
            gradient_max_norm=float(grad_max.detach().cpu()),
            jacobian_evaluation_time_in_seconds=evaluation_time,
        )
        summary.iterations.append(iter_summary)
        _maybe_log_progress(options, iter_summary)
        summary.final_cost = cost
        if float(grad_max.detach().cpu()) <= options.gradient_tolerance:
            summary.termination_type = TerminationType.CONVERGENCE
            summary.message = "Gradient tolerance reached."
            iter_summary.iteration_time_in_seconds = time.perf_counter() - iter_start
            iter_summary.cumulative_time_in_seconds = time.perf_counter() - start
            break
        if iteration == options.max_num_iterations:
            summary.termination_type = TerminationType.NO_CONVERGENCE
            summary.message = "Maximum iterations reached."
            iter_summary.iteration_time_in_seconds = time.perf_counter() - iter_start
            iter_summary.cumulative_time_in_seconds = time.perf_counter() - start
            break

        if inverse_hessian is None or inverse_hessian.shape[0] != tangent_grad.numel():
            inverse_hessian = torch.eye(tangent_grad.numel(), dtype=tangent_grad.dtype, device=tangent_grad.device)
        direction = _search_direction(
            options,
            tangent_grad,
            previous_grad,
            previous_direction,
            s_history,
            y_history,
            inverse_hessian,
        )
        directional_derivative = torch.dot(tangent_grad, direction)
        if directional_derivative >= 0:
            summary.num_line_search_direction_restarts += 1
            if summary.num_line_search_direction_restarts > options.max_num_line_search_direction_restarts:
                summary.termination_type = TerminationType.FAILURE
                summary.message = "Line search direction restart limit reached."
                iter_summary.step_is_successful = False
                iter_summary.iteration_time_in_seconds = time.perf_counter() - iter_start
                iter_summary.cumulative_time_in_seconds = time.perf_counter() - start
                break
            direction = -tangent_grad
            directional_derivative = -torch.dot(tangent_grad, tangent_grad)
            s_history.clear()
            y_history.clear()
            inverse_hessian = torch.eye(tangent_grad.numel(), dtype=tangent_grad.dtype, device=tangent_grad.device)
        callback_snapshot = parameters.detach().clone()
        work_snapshot = work.detach().clone()
        accepted = False
        accepted_value: torch.Tensor | None = None
        accepted_tangent_grad: torch.Tensor | None = None
        directions = [direction]
        trial_evaluations = 0
        if not torch.allclose(direction, -tangent_grad):
            directions.append(-tangent_grad)
        line_search_start = time.perf_counter()
        for trial_index, trial_direction in enumerate(directions):
            if trial_index > 0:
                summary.num_line_search_direction_restarts += 1
                if summary.num_line_search_direction_restarts > options.max_num_line_search_direction_restarts:
                    break
            step_size = 1.0
            trial_derivative = torch.dot(tangent_grad, trial_direction)
            previous_step_size: float | None = None
            previous_candidate_cost: float | None = None
            for ls_iter in range(options.max_num_line_search_step_size_iterations):
                candidate = manifold.plus(work_snapshot.reshape(-1), step_size * trial_direction).reshape_as(work)
                cand_value, cand_ambient_grad, candidate_eval_time = _timed_value_and_gradient(problem.function, candidate.detach())
                summary.num_cost_evaluations += 1
                summary.num_gradient_evaluations += 1
                summary.gradient_evaluation_time_in_seconds += candidate_eval_time
                iter_summary.jacobian_evaluation_time_in_seconds += candidate_eval_time
                trial_evaluations += 1
                candidate_cost = float(cand_value.detach().cpu())
                candidate_plus_jac = manifold.plus_jacobian(candidate.detach().reshape(-1)).to(
                    dtype=work.dtype, device=work.device
                )
                cand_tangent_grad = candidate_plus_jac.T @ cand_ambient_grad.reshape(-1)
                armijo_ok = cand_value <= value + options.line_search_sufficient_function_decrease * step_size * trial_derivative
                if options.line_search_type is LineSearchType.WOLFE:
                    curvature_ok = torch.abs(torch.dot(cand_tangent_grad, trial_direction)) <= (
                        options.line_search_sufficient_curvature_decrease * torch.abs(trial_derivative)
                    )
                else:
                    curvature_ok = True
                if armijo_ok and curvature_ok:
                    work = candidate.detach().clone()
                    if options.update_state_every_iteration:
                        with torch.no_grad():
                            parameters.reshape(-1).copy_(work.reshape(-1))
                    accepted = True
                    direction = trial_direction
                    accepted_value = cand_value.detach()
                    accepted_tangent_grad = cand_tangent_grad.detach()
                    iter_summary.step_size = step_size
                    iter_summary.line_search_iterations = ls_iter + 1
                    iter_summary.line_search_function_evaluations = trial_evaluations
                    iter_summary.line_search_gradient_evaluations = trial_evaluations
                    iter_summary.line_search_time_in_seconds = time.perf_counter() - line_search_start
                    iter_summary.step_norm = float(torch.linalg.norm(step_size * trial_direction).detach().cpu())
                    iter_summary.cost_change = cost - candidate_cost
                    iter_summary.step_is_successful = True
                    summary.num_line_search_steps += iter_summary.line_search_iterations
                    summary.num_line_search_function_evaluations += trial_evaluations
                    summary.num_line_search_gradient_evaluations += trial_evaluations
                    summary.line_search_total_time_in_seconds += iter_summary.line_search_time_in_seconds
                    summary.num_successful_steps += 1
                    break
                interpolation_start = time.perf_counter()
                next_step_size = next_line_search_step_size(
                    options,
                    step_size=step_size,
                    cost=cost,
                    candidate_cost=candidate_cost,
                    directional_derivative=float(trial_derivative.detach().cpu()),
                    previous_step_size=previous_step_size,
                    previous_candidate_cost=previous_candidate_cost,
                )
                summary.line_search_polynomial_minimization_time_in_seconds += time.perf_counter() - interpolation_start
                previous_step_size = step_size
                previous_candidate_cost = candidate_cost
                step_size = next_step_size
                if step_size < options.min_line_search_step_size:
                    break
            if accepted:
                break
        if not accepted:
            work = work_snapshot
            if options.update_state_every_iteration:
                with torch.no_grad():
                    parameters.reshape(-1).copy_(callback_snapshot.reshape(-1))
            summary.num_unsuccessful_steps += 1
            summary.num_line_search_steps += trial_evaluations
            summary.num_line_search_function_evaluations += trial_evaluations
            summary.num_line_search_gradient_evaluations += trial_evaluations
            failed_line_search_time = time.perf_counter() - line_search_start
            summary.line_search_total_time_in_seconds += failed_line_search_time
            iter_summary.line_search_time_in_seconds = failed_line_search_time
            summary.termination_type = TerminationType.NO_CONVERGENCE
            summary.message = "Line search failed."
            iter_summary.iteration_time_in_seconds = time.perf_counter() - iter_start
            iter_summary.cumulative_time_in_seconds = time.perf_counter() - start
            break
        assert accepted_value is not None and accepted_tangent_grad is not None
        new_value = accepted_value
        new_tangent_grad = accepted_tangent_grad
        summary.final_cost = float(new_value.detach().cpu())
        iter_summary.iteration_time_in_seconds = time.perf_counter() - iter_start
        iter_summary.cumulative_time_in_seconds = time.perf_counter() - start
        s = (iter_summary.step_size * direction).detach()
        y = (new_tangent_grad - tangent_grad).detach()
        s_history.append(s)
        y_history.append(y)
        if len(s_history) > options.max_lbfgs_rank:
            s_history.pop(0)
            y_history.pop(0)
        if options.line_search_direction_type is LineSearchDirectionType.BFGS:
            inverse_hessian = _bfgs_update(inverse_hessian, s, y)
        previous_grad = tangent_grad.detach()
        previous_direction = direction.detach()
        new_grad_max = torch.max(torch.abs(new_tangent_grad)) if new_tangent_grad.numel() else new_tangent_grad.new_tensor(0.0)
        if (
            float(new_grad_max.detach().cpu()) <= 10.0 * options.gradient_tolerance
            and abs(cost - float(new_value.detach().cpu())) <= options.function_tolerance * max(cost, 1.0)
        ):
            summary.termination_type = TerminationType.CONVERGENCE
            summary.message = "Function tolerance reached."
            summary.final_cost = float(new_value.detach().cpu())
            break
        for callback in options.callbacks:
            result = callback(iter_summary)
            if result is CallbackReturnType.SOLVER_ABORT:
                summary.termination_type = TerminationType.USER_FAILURE
                summary.message = "User callback aborted."
                summary.total_time_in_seconds = time.perf_counter() - start
                if not options.update_state_every_iteration:
                    with torch.no_grad():
                        parameters.reshape(-1).copy_(callback_snapshot.reshape(-1))
                return summary
            if result is CallbackReturnType.SOLVER_TERMINATE_SUCCESSFULLY:
                summary.termination_type = TerminationType.USER_SUCCESS
                summary.message = "User callback terminated successfully."
                summary.total_time_in_seconds = time.perf_counter() - start
                with torch.no_grad():
                    parameters.reshape(-1).copy_(work.reshape(-1))
                return summary
        if time.perf_counter() - start >= options.max_solver_time_in_seconds:
            summary.termination_type = TerminationType.NO_CONVERGENCE
            summary.message = "Maximum solver time reached."
            break
    if summary.termination_type is not TerminationType.USER_FAILURE:
        with torch.no_grad():
            parameters.reshape(-1).copy_(work.reshape(-1))
    summary.total_time_in_seconds = time.perf_counter() - start
    return summary


def _timed_value_and_gradient(
    function: FirstOrderFunction,
    parameters: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    start = time.perf_counter()
    value, gradient = function.value_and_gradient(parameters)
    return value, gradient, time.perf_counter() - start


def _maybe_log_progress(options: GradientProblemSolverOptions, iteration: IterationSummary) -> None:
    if not options.minimizer_progress_to_stdout or options.logging_type is LoggingType.SILENT:
        return
    print(
        f"{iteration.iteration:4d}: "
        f"f:{iteration.cost: .6e} "
        f"g:{iteration.gradient_max_norm: .3e} "
        f"h:{iteration.step_norm: .3e} "
        f"a:{iteration.step_size: .3e}"
    )


def _search_direction(
    options: GradientProblemSolverOptions,
    grad: torch.Tensor,
    previous_grad: Optional[torch.Tensor],
    previous_direction: Optional[torch.Tensor],
    s_history: list[torch.Tensor],
    y_history: list[torch.Tensor],
    inverse_hessian: Optional[torch.Tensor],
) -> torch.Tensor:
    if options.line_search_direction_type is LineSearchDirectionType.STEEPEST_DESCENT:
        return -grad
    if options.line_search_direction_type is LineSearchDirectionType.NONLINEAR_CONJUGATE_GRADIENT and previous_grad is not None and previous_direction is not None:
        if options.nonlinear_conjugate_gradient_type is NonlinearConjugateGradientType.POLAK_RIBIERE:
            beta = torch.dot(grad, grad - previous_grad) / torch.dot(previous_grad, previous_grad).clamp_min(torch.finfo(grad.dtype).eps)
        elif options.nonlinear_conjugate_gradient_type is NonlinearConjugateGradientType.HESTENES_STIEFEL:
            y = grad - previous_grad
            beta = torch.dot(grad, y) / torch.dot(previous_direction, y).clamp_min(torch.finfo(grad.dtype).eps)
        else:
            beta = torch.dot(grad, grad) / torch.dot(previous_grad, previous_grad).clamp_min(torch.finfo(grad.dtype).eps)
        return -grad + torch.clamp(beta, min=0.0) * previous_direction
    if options.line_search_direction_type is LineSearchDirectionType.LBFGS and s_history:
        return _lbfgs_two_loop(
            grad,
            s_history,
            y_history,
            use_approximate_eigenvalue_scaling=options.use_approximate_eigenvalue_bfgs_scaling,
        )
    if options.line_search_direction_type is LineSearchDirectionType.BFGS and inverse_hessian is not None:
        return -(inverse_hessian @ grad)
    return -grad


def _lbfgs_two_loop(
    grad: torch.Tensor,
    s_history: list[torch.Tensor],
    y_history: list[torch.Tensor],
    *,
    use_approximate_eigenvalue_scaling: bool,
) -> torch.Tensor:
    q = grad.clone()
    alphas: list[torch.Tensor] = []
    rhos: list[torch.Tensor] = []
    for s, y in reversed(list(zip(s_history, y_history))):
        rho = 1.0 / torch.dot(y, s).clamp_min(torch.finfo(grad.dtype).eps)
        alpha = rho * torch.dot(s, q)
        q = q - alpha * y
        alphas.append(alpha)
        rhos.append(rho)
    if s_history and use_approximate_eigenvalue_scaling:
        s, y = s_history[-1], y_history[-1]
        gamma = torch.dot(s, y) / torch.dot(y, y).clamp_min(torch.finfo(grad.dtype).eps)
        r = gamma * q
    else:
        r = q
    for s, y, alpha, rho in zip(s_history, y_history, reversed(alphas), reversed(rhos)):
        beta = rho * torch.dot(y, r)
        r = r + s * (alpha - beta)
    return -r


def _bfgs_update(inverse_hessian: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ys = torch.dot(y, s)
    if bool((ys <= 10.0 * torch.finfo(s.dtype).eps).detach().cpu()):
        return torch.eye(inverse_hessian.shape[0], dtype=inverse_hessian.dtype, device=inverse_hessian.device)
    rho = 1.0 / ys
    eye = torch.eye(inverse_hessian.shape[0], dtype=inverse_hessian.dtype, device=inverse_hessian.device)
    sy = torch.outer(s, y)
    ys_outer = torch.outer(y, s)
    return (eye - rho * sy) @ inverse_hessian @ (eye - rho * ys_outer) + rho * torch.outer(s, s)
