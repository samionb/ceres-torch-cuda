from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .costs import AutoDiffFirstOrderFunction, FirstOrderFunction
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


@dataclass
class GradientProblemSolverSummary:
    termination_type: TerminationType = TerminationType.FAILURE
    message: str = "torch_ceres.gradient_solve was not called."
    initial_cost: float = -1.0
    final_cost: float = -1.0
    iterations: list[IterationSummary] = field(default_factory=list)
    num_cost_evaluations: int = 0
    num_gradient_evaluations: int = 0
    total_time_in_seconds: float = 0.0
    num_parameters: int = -1
    num_tangent_parameters: int = -1
    line_search_direction_type: LineSearchDirectionType = LineSearchDirectionType.LBFGS
    line_search_type: LineSearchType = LineSearchType.WOLFE

    def IsSolutionUsable(self) -> bool:
        return self.termination_type in {TerminationType.CONVERGENCE, TerminationType.NO_CONVERGENCE, TerminationType.USER_SUCCESS}

    def BriefReport(self) -> str:
        return (
            "Torch Ceres Gradient Report: "
            f"Iterations: {len(self.iterations)}, "
            f"Initial cost: {self.initial_cost:.6e}, "
            f"Final cost: {self.final_cost:.6e}, "
            f"Termination: {self.termination_type.value}"
        )


def gradient_solve(
    options: GradientProblemSolverOptions,
    problem: GradientProblem,
    parameters: torch.Tensor,
) -> GradientProblemSolverSummary:
    start = time.perf_counter()
    manifold = problem.manifold or EuclideanManifold(parameters.numel())
    summary = GradientProblemSolverSummary(
        num_parameters=parameters.numel(),
        num_tangent_parameters=manifold.tangent_size,
        line_search_direction_type=options.line_search_direction_type,
        line_search_type=options.line_search_type,
    )
    value, ambient_grad = problem.function.value_and_gradient(parameters.detach())
    summary.initial_cost = float(value.detach().cpu())
    summary.final_cost = summary.initial_cost
    previous_grad: Optional[torch.Tensor] = None
    previous_direction: Optional[torch.Tensor] = None
    s_history: list[torch.Tensor] = []
    y_history: list[torch.Tensor] = []

    for iteration in range(options.max_num_iterations + 1):
        value, ambient_grad = problem.function.value_and_gradient(parameters.detach())
        summary.num_gradient_evaluations += 1
        plus_jac = manifold.plus_jacobian(parameters.detach().reshape(-1)).to(dtype=parameters.dtype, device=parameters.device)
        tangent_grad = plus_jac.T @ ambient_grad.reshape(-1)
        grad_norm = torch.linalg.norm(tangent_grad)
        grad_max = torch.max(torch.abs(tangent_grad)) if tangent_grad.numel() else tangent_grad.new_tensor(0.0)
        cost = float(value.detach().cpu())
        iter_summary = IterationSummary(
            iteration=iteration,
            cost=cost,
            gradient_norm=float(grad_norm.detach().cpu()),
            gradient_max_norm=float(grad_max.detach().cpu()),
        )
        summary.iterations.append(iter_summary)
        summary.final_cost = cost
        if float(grad_max.detach().cpu()) <= options.gradient_tolerance:
            summary.termination_type = TerminationType.CONVERGENCE
            summary.message = "Gradient tolerance reached."
            break
        if iteration == options.max_num_iterations:
            summary.termination_type = TerminationType.NO_CONVERGENCE
            summary.message = "Maximum iterations reached."
            break

        direction = _search_direction(options, tangent_grad, previous_grad, previous_direction, s_history, y_history)
        directional_derivative = torch.dot(tangent_grad, direction)
        if directional_derivative >= 0:
            direction = -tangent_grad
            directional_derivative = -torch.dot(tangent_grad, tangent_grad)
        snapshot = parameters.detach().clone()
        accepted = False
        directions = [direction]
        if not torch.allclose(direction, -tangent_grad):
            directions.append(-tangent_grad)
        for trial_direction in directions:
            step_size = 1.0
            trial_derivative = torch.dot(tangent_grad, trial_direction)
            for ls_iter in range(options.max_num_line_search_step_size_iterations):
                candidate = manifold.plus(snapshot.reshape(-1), step_size * trial_direction).reshape_as(parameters)
                cand_value, _ = problem.function.value_and_gradient(candidate.detach())
                summary.num_cost_evaluations += 1
                if cand_value <= value + options.line_search_sufficient_function_decrease * step_size * trial_derivative:
                    with torch.no_grad():
                        parameters.reshape(-1).copy_(candidate.reshape(-1))
                    accepted = True
                    direction = trial_direction
                    iter_summary.step_size = step_size
                    iter_summary.line_search_iterations = ls_iter + 1
                    iter_summary.step_is_successful = True
                    break
                step_size *= options.min_line_search_step_contraction
                if step_size < options.min_line_search_step_size:
                    break
            if accepted:
                break
        if not accepted:
            summary.termination_type = TerminationType.NO_CONVERGENCE
            summary.message = "Line search failed."
            break
        new_value, new_ambient_grad = problem.function.value_and_gradient(parameters.detach())
        new_tangent_grad = plus_jac.T @ new_ambient_grad.reshape(-1)
        s_history.append(step_size * direction.detach())
        y_history.append((new_tangent_grad - tangent_grad).detach())
        if len(s_history) > options.max_lbfgs_rank:
            s_history.pop(0)
            y_history.pop(0)
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
                return summary
            if result is CallbackReturnType.SOLVER_TERMINATE_SUCCESSFULLY:
                summary.termination_type = TerminationType.USER_SUCCESS
                summary.message = "User callback terminated successfully."
                summary.total_time_in_seconds = time.perf_counter() - start
                return summary
    summary.total_time_in_seconds = time.perf_counter() - start
    return summary


def _search_direction(
    options: GradientProblemSolverOptions,
    grad: torch.Tensor,
    previous_grad: Optional[torch.Tensor],
    previous_direction: Optional[torch.Tensor],
    s_history: list[torch.Tensor],
    y_history: list[torch.Tensor],
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
        return _lbfgs_two_loop(grad, s_history, y_history)
    return -grad


def _lbfgs_two_loop(grad: torch.Tensor, s_history: list[torch.Tensor], y_history: list[torch.Tensor]) -> torch.Tensor:
    q = grad.clone()
    alphas: list[torch.Tensor] = []
    rhos: list[torch.Tensor] = []
    for s, y in reversed(list(zip(s_history, y_history))):
        rho = 1.0 / torch.dot(y, s).clamp_min(torch.finfo(grad.dtype).eps)
        alpha = rho * torch.dot(s, q)
        q = q - alpha * y
        alphas.append(alpha)
        rhos.append(rho)
    if s_history:
        s, y = s_history[-1], y_history[-1]
        gamma = torch.dot(s, y) / torch.dot(y, y).clamp_min(torch.finfo(grad.dtype).eps)
        r = gamma * q
    else:
        r = q
    for s, y, alpha, rho in zip(s_history, y_history, reversed(alphas), reversed(rhos)):
        beta = rho * torch.dot(y, r)
        r = r + s * (alpha - beta)
    return -r
