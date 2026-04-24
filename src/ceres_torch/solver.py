from __future__ import annotations

import time

import torch

from .linear import dogleg_step, jacobi_damping_from_jacobian, solve_linear_system
from .problem import Problem
from .types import (
    CallbackReturnType,
    IterationSummary,
    LinearSolverType,
    MinimizerType,
    SolverOptions,
    SolverSummary,
    TerminationType,
    TrustRegionStrategyType,
)


def solve(options: SolverOptions, problem: Problem) -> SolverSummary:
    options.validate()
    if options.minimizer_type is MinimizerType.LINE_SEARCH:
        return _line_search_solve(options, problem)
    return _trust_region_solve(options, problem)


def _trust_region_solve(options: SolverOptions, problem: Problem) -> SolverSummary:
    start = time.perf_counter()
    summary = _new_summary(options, problem)
    radius = options.initial_trust_region_radius
    consecutive_invalid = 0
    previous_cost: float | None = None

    initial = problem.evaluate(compute_jacobian=True)
    summary.initial_cost = float(initial.cost.detach().cpu())
    summary.final_cost = summary.initial_cost
    if initial.gradient is None:
        summary.termination_type = TerminationType.CONVERGENCE
        summary.message = "No active parameters."
        return summary

    for iteration in range(options.max_num_iterations + 1):
        iter_start = time.perf_counter()
        evaluation = problem.evaluate(compute_jacobian=True)
        summary.num_jacobian_evaluations += 1
        cost = float(evaluation.cost.detach().cpu())
        J = evaluation.jacobian
        r = evaluation.residuals
        g = evaluation.gradient
        assert J is not None and g is not None
        grad_norm = torch.linalg.norm(g)
        grad_max = torch.max(torch.abs(g)) if g.numel() else g.new_tensor(0.0)

        iter_summary = IterationSummary(
            iteration=iteration,
            step_is_valid=True,
            step_is_successful=True,
            cost=cost,
            gradient_norm=float(grad_norm.detach().cpu()),
            gradient_max_norm=float(grad_max.detach().cpu()),
            trust_region_radius=radius,
            eta=options.eta,
        )

        if iteration == 0:
            summary.iterations.append(iter_summary)
            if _gradient_converged(grad_max, options):
                summary.termination_type = TerminationType.CONVERGENCE
                summary.message = "Gradient tolerance reached."
                summary.final_cost = cost
                break
            if options.max_num_iterations == 0:
                summary.termination_type = TerminationType.NO_CONVERGENCE
                summary.message = "Maximum iterations reached."
                summary.final_cost = cost
                break
            continue

        if J.shape[1] == 0:
            summary.termination_type = TerminationType.CONVERGENCE
            summary.message = "No active parameters."
            summary.final_cost = cost
            break

        if options.trust_region_strategy_type is TrustRegionStrategyType.DOGLEG:
            step = dogleg_step(J, r, radius)
            linear_iterations = 1
        else:
            damping = jacobi_damping_from_jacobian(
                J,
                min_diagonal=options.min_lm_diagonal,
                max_diagonal=options.max_lm_diagonal,
                radius=radius,
                jacobi_scaling=options.jacobi_scaling,
            )
            linear_result = solve_linear_system(
                J,
                -r,
                solver_type=_effective_linear_solver(options.linear_solver_type),
                damping=damping,
                max_iterations=options.max_linear_solver_iterations,
                tolerance=options.eta,
                preconditioner_type=options.preconditioner_type,
            )
            step = linear_result.x
            linear_iterations = linear_result.summary.num_iterations
            summary.num_linear_solves += 1

        if not torch.all(torch.isfinite(step)):
            consecutive_invalid += 1
            radius = max(radius * 0.25, options.min_trust_region_radius)
            iter_summary.step_is_valid = False
            iter_summary.step_is_successful = False
            summary.num_unsuccessful_steps += 1
            if consecutive_invalid > options.max_num_consecutive_invalid_steps:
                summary.termination_type = TerminationType.FAILURE
                summary.message = "Too many invalid trust-region steps."
                break
            summary.iterations.append(iter_summary)
            continue

        snapshot = problem.snapshot()
        problem.apply_delta(step)
        candidate = problem.evaluate(compute_jacobian=False)
        summary.num_residual_evaluations += 1
        candidate_cost = float(candidate.cost.detach().cpu())
        actual_decrease = cost - candidate_cost
        model_decrease = _model_decrease(J, r, step)
        rho = actual_decrease / max(model_decrease, torch.finfo(J.dtype).eps)
        accepted = actual_decrease > 0 and rho >= options.min_relative_decrease

        iter_summary.step_norm = float(torch.linalg.norm(step).detach().cpu())
        iter_summary.cost_change = actual_decrease
        iter_summary.relative_decrease = float(rho)
        iter_summary.linear_solver_iterations = linear_iterations
        iter_summary.iteration_time_in_seconds = time.perf_counter() - iter_start
        iter_summary.cumulative_time_in_seconds = time.perf_counter() - start

        if accepted:
            consecutive_invalid = 0
            summary.num_successful_steps += 1
            summary.final_cost = candidate_cost
            radius = min(options.max_trust_region_radius, radius * max(1.5, 1.0 + 2.0 * max(rho, 0.0)))
            iter_summary.step_is_successful = True
            if options.update_state_every_iteration:
                pass
        else:
            problem.restore(snapshot)
            summary.num_unsuccessful_steps += 1
            radius = max(options.min_trust_region_radius, radius * 0.25)
            iter_summary.step_is_successful = False

        summary.iterations.append(iter_summary)
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

        if accepted:
            if _parameter_converged(step, problem, options):
                summary.termination_type = TerminationType.CONVERGENCE
                summary.message = "Parameter tolerance reached."
                break
            if previous_cost is not None and abs(previous_cost - candidate_cost) <= options.function_tolerance * max(previous_cost, 1.0):
                summary.termination_type = TerminationType.CONVERGENCE
                summary.message = "Function tolerance reached."
                break
            previous_cost = candidate_cost
            next_eval = problem.evaluate(compute_jacobian=True)
            if next_eval.gradient is not None and _gradient_converged(torch.max(torch.abs(next_eval.gradient)), options):
                summary.termination_type = TerminationType.CONVERGENCE
                summary.message = "Gradient tolerance reached."
                summary.final_cost = float(next_eval.cost.detach().cpu())
                break

        if radius <= options.min_trust_region_radius:
            summary.termination_type = TerminationType.NO_CONVERGENCE
            summary.message = "Minimum trust-region radius reached."
            break
        if time.perf_counter() - start >= options.max_solver_time_in_seconds:
            summary.termination_type = TerminationType.NO_CONVERGENCE
            summary.message = "Maximum solver time reached."
            break
    else:
        summary.termination_type = TerminationType.NO_CONVERGENCE
        summary.message = "Maximum iterations reached."

    if summary.termination_type is TerminationType.FAILURE and summary.message == "torch_ceres.solve was not called.":
        summary.message = "Solver failed."
    if summary.termination_type is TerminationType.FAILURE:
        pass
    elif not summary.message or summary.message == "torch_ceres.solve was not called.":
        summary.termination_type = TerminationType.NO_CONVERGENCE
        summary.message = "Maximum iterations reached."
    summary.total_time_in_seconds = time.perf_counter() - start
    return summary


def _line_search_solve(options: SolverOptions, problem: Problem) -> SolverSummary:
    start = time.perf_counter()
    summary = _new_summary(options, problem)
    initial = problem.evaluate(compute_jacobian=True)
    summary.initial_cost = float(initial.cost.detach().cpu())
    summary.final_cost = summary.initial_cost

    for iteration in range(options.max_num_iterations + 1):
        evaluation = problem.evaluate(compute_jacobian=True)
        summary.num_jacobian_evaluations += 1
        cost = float(evaluation.cost.detach().cpu())
        g = evaluation.gradient
        assert g is not None
        grad_max = torch.max(torch.abs(g)) if g.numel() else g.new_tensor(0.0)
        iter_summary = IterationSummary(
            iteration=iteration,
            cost=cost,
            gradient_norm=float(torch.linalg.norm(g).detach().cpu()),
            gradient_max_norm=float(grad_max.detach().cpu()),
        )
        summary.iterations.append(iter_summary)
        if _gradient_converged(grad_max, options):
            summary.termination_type = TerminationType.CONVERGENCE
            summary.message = "Gradient tolerance reached."
            summary.final_cost = cost
            break
        if iteration == options.max_num_iterations:
            summary.termination_type = TerminationType.NO_CONVERGENCE
            summary.message = "Maximum iterations reached."
            break
        direction = -g
        directional_derivative = torch.dot(g, direction)
        step_size = 1.0
        accepted = False
        snapshot = problem.snapshot()
        for ls_iter in range(options.max_num_line_search_step_size_iterations):
            problem.restore(snapshot)
            problem.apply_delta(step_size * direction)
            candidate = problem.evaluate(compute_jacobian=False)
            candidate_cost = float(candidate.cost.detach().cpu())
            summary.num_residual_evaluations += 1
            if candidate_cost <= cost + options.line_search_sufficient_function_decrease * step_size * float(directional_derivative.detach().cpu()):
                accepted = True
                summary.final_cost = candidate_cost
                iter_summary.step_size = step_size
                iter_summary.line_search_iterations = ls_iter + 1
                iter_summary.step_is_successful = True
                summary.num_successful_steps += 1
                break
            step_size *= options.min_line_search_step_contraction
            if step_size < options.min_line_search_step_size:
                break
        if not accepted:
            problem.restore(snapshot)
            summary.num_unsuccessful_steps += 1
            summary.termination_type = TerminationType.NO_CONVERGENCE
            summary.message = "Line search failed to find a decreasing step."
            break
    summary.total_time_in_seconds = time.perf_counter() - start
    return summary


def _new_summary(options: SolverOptions, problem: Problem) -> SolverSummary:
    return SolverSummary(
        minimizer_type=options.minimizer_type,
        num_parameter_blocks=problem.num_parameter_blocks(),
        num_parameters=problem.num_parameters(),
        num_effective_parameters=problem.num_effective_parameters(),
        num_residual_blocks=problem.num_residual_blocks(),
        num_residuals=problem.num_residuals(),
        linear_solver_type_given=options.linear_solver_type,
        linear_solver_type_used=_effective_linear_solver(options.linear_solver_type),
        trust_region_strategy_type=options.trust_region_strategy_type,
        line_search_direction_type=options.line_search_direction_type,
        line_search_type=options.line_search_type,
    )


def _effective_linear_solver(requested: LinearSolverType) -> LinearSolverType:
    if requested in {LinearSolverType.SPARSE_NORMAL_CHOLESKY, LinearSolverType.SPARSE_SCHUR}:
        return LinearSolverType.DENSE_NORMAL_CHOLESKY
    if requested in {LinearSolverType.DENSE_SCHUR, LinearSolverType.ITERATIVE_SCHUR}:
        return requested
    return requested


def _model_decrease(J: torch.Tensor, r: torch.Tensor, step: torch.Tensor) -> float:
    before = 0.5 * torch.dot(r, r)
    after_r = r + J @ step
    after = 0.5 * torch.dot(after_r, after_r)
    return float(torch.clamp(before - after, min=0.0).detach().cpu())


def _gradient_converged(grad_max: torch.Tensor, options: SolverOptions) -> bool:
    return bool(float(grad_max.detach().cpu()) <= options.gradient_tolerance)


def _parameter_converged(step: torch.Tensor, problem: Problem, options: SolverOptions) -> bool:
    state_norm_sq = 0.0
    for block in problem.parameter_blocks:
        state_norm_sq += float(torch.sum(block.tensor.detach().reshape(-1) ** 2).cpu())
    state_norm = state_norm_sq**0.5
    step_norm = float(torch.linalg.norm(step).detach().cpu())
    return step_norm <= options.parameter_tolerance * (state_norm + options.parameter_tolerance)

