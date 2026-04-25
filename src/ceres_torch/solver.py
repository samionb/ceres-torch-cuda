from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .linear import dogleg_step, jacobi_damping_from_jacobian, solve_linear_system
from .line_search import next_line_search_step_size
from .problem import EvaluateOptions, ParameterBlock, Problem
from .types import (
    CallbackReturnType,
    IterationSummary,
    LineSearchDirectionType,
    LineSearchType,
    LinearSolverType,
    MinimizerType,
    NonlinearConjugateGradientType,
    SolverOptions,
    SolverSummary,
    TerminationType,
    TrustRegionStrategyType,
    LoggingType,
)


def solve(options: SolverOptions, problem: Problem) -> SolverSummary:
    options.validate()
    if options.minimizer_type is MinimizerType.LINE_SEARCH:
        return _line_search_solve(options, problem)
    return _trust_region_solve(options, problem)


def _trust_region_solve(options: SolverOptions, problem: Problem) -> SolverSummary:
    start = time.perf_counter()
    summary = _new_summary(options, problem)
    active_blocks = _linear_solver_parameter_order(problem)
    num_eliminate = _num_eliminate_for_schur(active_blocks, _effective_linear_solver(options.linear_solver_type))
    lm_radius = _LevenbergMarquardtRadiusState(options.initial_trust_region_radius, options.max_trust_region_radius)
    radius = lm_radius.radius
    consecutive_invalid = 0
    previous_cost: float | None = None

    initial = problem.evaluate(EvaluateOptions(parameter_blocks=active_blocks), compute_jacobian=True)
    summary.initial_cost = float(initial.cost.detach().cpu())
    summary.final_cost = summary.initial_cost
    best_cost = summary.initial_cost
    best_snapshot = problem.snapshot()
    step_evaluator = _TrustRegionStepEvaluator(
        summary.initial_cost,
        options.max_consecutive_nonmonotonic_steps if options.use_nonmonotonic_steps else 0,
    )
    if initial.gradient is None:
        summary.termination_type = TerminationType.CONVERGENCE
        summary.message = "No active parameters."
        return summary

    for iteration in range(options.max_num_iterations + 1):
        iter_start = time.perf_counter()
        evaluation = problem.evaluate(EvaluateOptions(parameter_blocks=active_blocks), compute_jacobian=True)
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
            _maybe_log_progress(options, iter_summary)
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
            linear_start = time.perf_counter()
            step = dogleg_step(J, r, radius)
            linear_iterations = 1
            linear_time = time.perf_counter() - linear_start
        else:
            damping = jacobi_damping_from_jacobian(
                J,
                min_diagonal=options.min_lm_diagonal,
                max_diagonal=options.max_lm_diagonal,
                radius=radius,
                jacobi_scaling=options.jacobi_scaling,
            )
            linear_start = time.perf_counter()
            linear_result = solve_linear_system(
                J,
                -r,
                solver_type=_effective_linear_solver(options.linear_solver_type),
                damping=damping,
                num_eliminate=num_eliminate,
                max_iterations=options.max_linear_solver_iterations,
                tolerance=options.eta,
                preconditioner_type=options.preconditioner_type,
                block_sizes=[block.tangent_size for block in active_blocks],
                use_mixed_precision=options.use_mixed_precision_solves,
                max_refinement_iterations=options.max_num_refinement_iterations,
            )
            linear_time = time.perf_counter() - linear_start
            step = linear_result.x
            linear_iterations = linear_result.summary.num_iterations
            summary.num_linear_solves += 1

        if not torch.all(torch.isfinite(step)):
            consecutive_invalid += 1
            radius = _rejected_trust_region_radius(
                options.trust_region_strategy_type,
                radius,
                lm_radius,
            )
            iter_summary.step_is_valid = False
            iter_summary.step_is_successful = False
            summary.num_unsuccessful_steps += 1
            if consecutive_invalid > options.max_num_consecutive_invalid_steps:
                summary.termination_type = TerminationType.FAILURE
                summary.message = "Too many invalid trust-region steps."
                break
            summary.iterations.append(iter_summary)
            _maybe_log_progress(options, iter_summary)
            continue

        snapshot = problem.snapshot()
        if _has_bounds(active_blocks) and options.max_num_line_search_step_size_iterations > 0:
            line_search_start = time.perf_counter()
            step, line_search_iterations, line_search_evaluations = _projected_line_search_step(
                problem,
                active_blocks,
                snapshot,
                step,
                cost,
                g,
                options,
            )
            line_search_time = time.perf_counter() - line_search_start
            summary.num_residual_evaluations += line_search_evaluations
            summary.num_line_search_steps += line_search_iterations
            summary.num_line_search_function_evaluations += line_search_evaluations
            summary.line_search_total_time_in_seconds += line_search_time
            iter_summary.line_search_iterations = line_search_iterations
            iter_summary.line_search_function_evaluations = line_search_evaluations
            problem.restore(snapshot)
        problem.apply_delta(step, active_blocks=active_blocks)
        candidate = problem.evaluate(compute_jacobian=False)
        summary.num_residual_evaluations += 1
        candidate_cost = float(candidate.cost.detach().cpu())
        model_decrease = _model_decrease(J, r, step)
        step_is_valid = model_decrease > 0.0
        inner_iterations_were_useful = False
        if step_is_valid and options.use_inner_iterations:
            trust_region_candidate_cost = candidate_cost
            candidate_cost, inner_steps = _run_inner_iterations(
                problem,
                active_blocks,
                options,
                current_cost=candidate_cost,
            )
            if inner_steps:
                summary.num_residual_evaluations += inner_steps
            inner_model_decrease = trust_region_candidate_cost - candidate_cost
            model_decrease += inner_model_decrease
            inner_iterations_were_useful = candidate_cost < min(cost, trust_region_candidate_cost)
        rho = step_evaluator.step_quality(candidate_cost, model_decrease) if step_is_valid else 0.0
        accepted = step_is_valid and (inner_iterations_were_useful or rho > options.min_relative_decrease)

        iter_summary.step_norm = float(torch.linalg.norm(step).detach().cpu())
        iter_summary.cost_change = cost - candidate_cost
        iter_summary.relative_decrease = float(rho)
        iter_summary.step_is_valid = step_is_valid
        iter_summary.linear_solver_iterations = linear_iterations
        iter_summary.step_solver_time_in_seconds = linear_time
        iter_summary.iteration_time_in_seconds = time.perf_counter() - iter_start
        iter_summary.cumulative_time_in_seconds = time.perf_counter() - start

        if accepted:
            consecutive_invalid = 0
            summary.num_successful_steps += 1
            step_evaluator.step_accepted(candidate_cost, model_decrease)
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_snapshot = problem.snapshot()
                iter_summary.step_is_nonmonotonic = False
            else:
                iter_summary.step_is_nonmonotonic = candidate_cost > best_cost
            summary.final_cost = best_cost
            radius = _accepted_trust_region_radius(
                options.trust_region_strategy_type,
                radius,
                rho,
                iter_summary.step_norm,
                lm_radius,
            )
            iter_summary.trust_region_radius = radius
            iter_summary.step_is_successful = True
            if options.update_state_every_iteration:
                pass
        else:
            problem.restore(snapshot)
            summary.num_unsuccessful_steps += 1
            radius = _rejected_trust_region_radius(
                options.trust_region_strategy_type,
                radius,
                lm_radius,
            )
            iter_summary.trust_region_radius = radius
            iter_summary.step_is_successful = False

        summary.iterations.append(iter_summary)
        _maybe_log_progress(options, iter_summary)
        callback_snapshot: list[torch.Tensor] | None = None
        if options.update_state_every_iteration:
            callback_snapshot = problem.snapshot()
            problem.restore(best_snapshot)
        for callback in options.callbacks:
            result = callback(iter_summary)
            if result is CallbackReturnType.SOLVER_ABORT:
                summary.termination_type = TerminationType.USER_FAILURE
                summary.message = "User callback aborted."
                problem.restore(best_snapshot)
                summary.final_cost = best_cost
                summary.total_time_in_seconds = time.perf_counter() - start
                return summary
            if result is CallbackReturnType.SOLVER_TERMINATE_SUCCESSFULLY:
                summary.termination_type = TerminationType.USER_SUCCESS
                summary.message = "User callback terminated successfully."
                problem.restore(best_snapshot)
                summary.final_cost = best_cost
                summary.total_time_in_seconds = time.perf_counter() - start
                return summary
        if callback_snapshot is not None:
            problem.restore(callback_snapshot)

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
            next_eval = problem.evaluate(EvaluateOptions(parameter_blocks=active_blocks), compute_jacobian=True)
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
    problem.restore(best_snapshot)
    summary.final_cost = best_cost
    summary.total_time_in_seconds = time.perf_counter() - start
    return summary


def _line_search_solve(options: SolverOptions, problem: Problem) -> SolverSummary:
    start = time.perf_counter()
    summary = _new_summary(options, problem)
    active_blocks = _linear_solver_parameter_order(problem)
    initial = problem.evaluate(EvaluateOptions(parameter_blocks=active_blocks), compute_jacobian=True)
    summary.initial_cost = float(initial.cost.detach().cpu())
    summary.final_cost = summary.initial_cost
    previous_grad: torch.Tensor | None = None
    previous_direction: torch.Tensor | None = None
    s_history: list[torch.Tensor] = []
    y_history: list[torch.Tensor] = []
    inverse_hessian: torch.Tensor | None = None

    for iteration in range(options.max_num_iterations + 1):
        evaluation = problem.evaluate(EvaluateOptions(parameter_blocks=active_blocks), compute_jacobian=True)
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
        _maybe_log_progress(options, iter_summary)
        if _gradient_converged(grad_max, options):
            summary.termination_type = TerminationType.CONVERGENCE
            summary.message = "Gradient tolerance reached."
            summary.final_cost = cost
            break
        if iteration == options.max_num_iterations:
            summary.termination_type = TerminationType.NO_CONVERGENCE
            summary.message = "Maximum iterations reached."
            break
        if inverse_hessian is None or inverse_hessian.shape[0] != g.numel():
            inverse_hessian = torch.eye(g.numel(), dtype=g.dtype, device=g.device)
        direction = _line_search_direction(
            options,
            g,
            previous_grad,
            previous_direction,
            s_history,
            y_history,
            inverse_hessian,
        )
        directional_derivative = torch.dot(g, direction)
        if directional_derivative >= 0:
            direction = -g
            directional_derivative = -torch.dot(g, g)
            s_history.clear()
            y_history.clear()
            inverse_hessian = torch.eye(g.numel(), dtype=g.dtype, device=g.device)
        accepted = False
        snapshot = problem.snapshot()
        accepted_direction = direction
        accepted_gradient: torch.Tensor | None = None
        directions = [direction]
        trial_evaluations = 0
        if not torch.allclose(direction, -g):
            directions.append(-g)
        for trial_direction in directions:
            step_size = 1.0
            trial_derivative = torch.dot(g, trial_direction)
            previous_step_size: float | None = None
            previous_candidate_cost: float | None = None
            for ls_iter in range(options.max_num_line_search_step_size_iterations):
                problem.restore(snapshot)
                problem.apply_delta(step_size * trial_direction, active_blocks=active_blocks)
                candidate = problem.evaluate(EvaluateOptions(parameter_blocks=active_blocks), compute_jacobian=True)
                summary.num_jacobian_evaluations += 1
                trial_evaluations += 1
                candidate_cost = float(candidate.cost.detach().cpu())
                candidate_grad = candidate.gradient if candidate.gradient is not None else g.new_zeros(g.shape)
                armijo_ok = candidate_cost <= cost + options.line_search_sufficient_function_decrease * step_size * float(trial_derivative.detach().cpu())
                if options.line_search_type is LineSearchType.WOLFE:
                    curvature_ok = torch.abs(torch.dot(candidate_grad, trial_direction)) <= (
                        options.line_search_sufficient_curvature_decrease * torch.abs(trial_derivative)
                    )
                else:
                    curvature_ok = True
                if armijo_ok and curvature_ok:
                    accepted = True
                    accepted_direction = trial_direction
                    accepted_gradient = candidate_grad.detach()
                    summary.final_cost = candidate_cost
                    iter_summary.step_size = step_size
                    iter_summary.line_search_iterations = ls_iter + 1
                    iter_summary.line_search_function_evaluations = trial_evaluations
                    iter_summary.line_search_gradient_evaluations = trial_evaluations
                    iter_summary.step_norm = float(torch.linalg.norm(step_size * trial_direction).detach().cpu())
                    iter_summary.cost_change = cost - candidate_cost
                    iter_summary.step_is_successful = True
                    summary.num_line_search_steps += iter_summary.line_search_iterations
                    summary.num_line_search_function_evaluations += trial_evaluations
                    summary.num_line_search_gradient_evaluations += trial_evaluations
                    summary.num_successful_steps += 1
                    break
                next_step_size = next_line_search_step_size(
                    options,
                    step_size=step_size,
                    cost=cost,
                    candidate_cost=candidate_cost,
                    directional_derivative=float(trial_derivative.detach().cpu()),
                    previous_step_size=previous_step_size,
                    previous_candidate_cost=previous_candidate_cost,
                )
                previous_step_size = step_size
                previous_candidate_cost = candidate_cost
                step_size = next_step_size
                if step_size < options.min_line_search_step_size:
                    break
            if accepted:
                break
        if not accepted:
            problem.restore(snapshot)
            summary.num_unsuccessful_steps += 1
            summary.num_line_search_steps += trial_evaluations
            summary.num_line_search_function_evaluations += trial_evaluations
            summary.num_line_search_gradient_evaluations += trial_evaluations
            summary.termination_type = TerminationType.NO_CONVERGENCE
            summary.message = "Line search failed to find a decreasing step."
            break
        if accepted_gradient is None:
            accepted_evaluation = problem.evaluate(EvaluateOptions(parameter_blocks=active_blocks), compute_jacobian=True)
            summary.num_jacobian_evaluations += 1
            accepted_gradient = accepted_evaluation.gradient if accepted_evaluation.gradient is not None else g.new_zeros(g.shape)
            summary.final_cost = float(accepted_evaluation.cost.detach().cpu())
        s = (iter_summary.step_size * accepted_direction).detach()
        y = (accepted_gradient - g).detach()
        if _has_positive_curvature(s, y):
            s_history.append(s)
            y_history.append(y)
            if len(s_history) > options.max_lbfgs_rank:
                s_history.pop(0)
                y_history.pop(0)
            if options.line_search_direction_type is LineSearchDirectionType.BFGS and inverse_hessian is not None:
                inverse_hessian = _bfgs_update(inverse_hessian, s, y)
        else:
            s_history.clear()
            y_history.clear()
            inverse_hessian = torch.eye(g.numel(), dtype=g.dtype, device=g.device)
        previous_grad = g.detach()
        previous_direction = accepted_direction.detach()
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
        accepted_grad_max = torch.max(torch.abs(accepted_gradient)) if accepted_gradient.numel() else accepted_gradient.new_tensor(0.0)
        if _gradient_converged(accepted_grad_max, options):
            summary.termination_type = TerminationType.CONVERGENCE
            summary.message = "Gradient tolerance reached."
            break
        if _parameter_converged(s, problem, options):
            summary.termination_type = TerminationType.CONVERGENCE
            summary.message = "Parameter tolerance reached."
            break
        if abs(cost - summary.final_cost) <= options.function_tolerance * max(cost, 1.0):
            summary.termination_type = TerminationType.CONVERGENCE
            summary.message = "Function tolerance reached."
            break
        if time.perf_counter() - start >= options.max_solver_time_in_seconds:
            summary.termination_type = TerminationType.NO_CONVERGENCE
            summary.message = "Maximum solver time reached."
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


def _linear_solver_parameter_order(problem: Problem) -> list[ParameterBlock]:
    blocks = [b for b in problem.parameter_blocks if b.tangent_size > 0]
    if not any(b.ordering_group is not None for b in blocks):
        return blocks
    max_group = max((b.ordering_group for b in blocks if b.ordering_group is not None), default=0)
    original_index = {block: i for i, block in enumerate(blocks)}
    return sorted(
        blocks,
        key=lambda block: (
            block.ordering_group if block.ordering_group is not None else max_group + 1,
            original_index[block],
        ),
    )


def _num_eliminate_for_schur(blocks: list[ParameterBlock], solver_type: LinearSolverType) -> int:
    if solver_type not in {
        LinearSolverType.DENSE_SCHUR,
        LinearSolverType.ITERATIVE_SCHUR,
        LinearSolverType.SPARSE_SCHUR,
    }:
        return 0
    if not blocks or blocks[0].ordering_group is None:
        return 0
    first_group = blocks[0].ordering_group
    if not any(block.ordering_group != first_group for block in blocks):
        return 0
    return sum(block.tangent_size for block in blocks if block.ordering_group == first_group)


def _effective_linear_solver(requested: LinearSolverType) -> LinearSolverType:
    if requested is LinearSolverType.SPARSE_NORMAL_CHOLESKY:
        return LinearSolverType.DENSE_NORMAL_CHOLESKY
    if requested is LinearSolverType.SPARSE_SCHUR:
        return LinearSolverType.DENSE_SCHUR
    if requested in {LinearSolverType.DENSE_SCHUR, LinearSolverType.ITERATIVE_SCHUR}:
        return requested
    return requested


def _model_decrease(J: torch.Tensor, r: torch.Tensor, step: torch.Tensor) -> float:
    before = 0.5 * torch.dot(r, r)
    after_r = r + J @ step
    after = 0.5 * torch.dot(after_r, after_r)
    return float(torch.clamp(before - after, min=0.0).detach().cpu())


@dataclass
class _LevenbergMarquardtRadiusState:
    radius: float
    max_radius: float
    decrease_factor: float = 2.0

    def step_accepted(self, step_quality: float) -> float:
        denominator = max(1.0 / 3.0, 1.0 - (2.0 * step_quality - 1.0) ** 3)
        self.radius = min(self.max_radius, self.radius / denominator)
        self.decrease_factor = 2.0
        return self.radius

    def step_rejected(self) -> float:
        self.radius = self.radius / self.decrease_factor
        self.decrease_factor *= 2.0
        return self.radius


class _TrustRegionStepEvaluator:
    def __init__(self, initial_cost: float, max_consecutive_nonmonotonic_steps: int) -> None:
        self.max_consecutive_nonmonotonic_steps = max(0, int(max_consecutive_nonmonotonic_steps))
        self.minimum_cost = initial_cost
        self.current_cost = initial_cost
        self.reference_cost = initial_cost
        self.candidate_cost = initial_cost
        self.accumulated_reference_model_cost_change = 0.0
        self.accumulated_candidate_model_cost_change = 0.0
        self.num_consecutive_nonmonotonic_steps = 0

    def step_quality(self, cost: float, model_cost_change: float) -> float:
        if model_cost_change <= 0.0:
            return float("-inf")
        relative_decrease = (self.current_cost - cost) / model_cost_change
        historical_relative_decrease = (self.reference_cost - cost) / (
            self.accumulated_reference_model_cost_change + model_cost_change
        )
        return max(relative_decrease, historical_relative_decrease)

    def step_accepted(self, cost: float, model_cost_change: float) -> None:
        self.current_cost = cost
        self.accumulated_candidate_model_cost_change += model_cost_change
        self.accumulated_reference_model_cost_change += model_cost_change

        if self.current_cost < self.minimum_cost:
            self.minimum_cost = self.current_cost
            self.num_consecutive_nonmonotonic_steps = 0
            self.candidate_cost = self.current_cost
            self.accumulated_candidate_model_cost_change = 0.0
        else:
            self.num_consecutive_nonmonotonic_steps += 1
            if self.current_cost > self.candidate_cost:
                self.candidate_cost = self.current_cost
                self.accumulated_candidate_model_cost_change = 0.0

        if self.num_consecutive_nonmonotonic_steps == self.max_consecutive_nonmonotonic_steps:
            self.reference_cost = self.candidate_cost
            self.accumulated_reference_model_cost_change = self.accumulated_candidate_model_cost_change


def _accepted_trust_region_radius(
    strategy_type: TrustRegionStrategyType,
    radius: float,
    rho: float,
    step_norm: float,
    lm_radius: _LevenbergMarquardtRadiusState,
) -> float:
    if strategy_type is TrustRegionStrategyType.LEVENBERG_MARQUARDT:
        return lm_radius.step_accepted(rho)
    return _updated_trust_region_radius(radius, rho, step_norm, lm_radius.max_radius)


def _rejected_trust_region_radius(
    strategy_type: TrustRegionStrategyType,
    radius: float,
    lm_radius: _LevenbergMarquardtRadiusState,
) -> float:
    if strategy_type is TrustRegionStrategyType.LEVENBERG_MARQUARDT:
        return lm_radius.step_rejected()
    return radius * 0.25


def _has_bounds(blocks: list[ParameterBlock]) -> bool:
    return any(block.lower_bound is not None or block.upper_bound is not None for block in blocks)


def _projected_line_search_step(
    problem: Problem,
    active_blocks: list[ParameterBlock],
    snapshot: list[torch.Tensor],
    step: torch.Tensor,
    cost: float,
    gradient: torch.Tensor,
    options: SolverOptions,
) -> tuple[torch.Tensor, int, int]:
    directional_derivative = float(torch.dot(gradient, step).detach().cpu())
    if directional_derivative >= 0.0:
        return step, 0, 0

    step_size = 1.0
    previous_step_size: float | None = None
    previous_candidate_cost: float | None = None
    evaluations = 0
    iterations = 0
    for _ in range(options.max_num_line_search_step_size_iterations):
        iterations += 1
        problem.restore(snapshot)
        problem.apply_delta(step_size * step, active_blocks=active_blocks)
        candidate = problem.evaluate(EvaluateOptions(parameter_blocks=active_blocks), compute_jacobian=False)
        evaluations += 1
        candidate_cost = float(candidate.cost.detach().cpu())
        sufficient_decrease = cost + options.line_search_sufficient_function_decrease * step_size * directional_derivative
        if candidate_cost <= sufficient_decrease:
            problem.restore(snapshot)
            return step_size * step, iterations, evaluations

        next_step_size = next_line_search_step_size(
            options,
            step_size=step_size,
            cost=cost,
            candidate_cost=candidate_cost,
            directional_derivative=directional_derivative,
            previous_step_size=previous_step_size,
            previous_candidate_cost=previous_candidate_cost,
        )
        previous_step_size = step_size
        previous_candidate_cost = candidate_cost
        step_size = next_step_size
        if step_size < options.min_line_search_step_size:
            break

    problem.restore(snapshot)
    return step, iterations, evaluations


def _run_inner_iterations(
    problem: Problem,
    active_blocks: list[ParameterBlock],
    options: SolverOptions,
    *,
    current_cost: float,
) -> tuple[float, int]:
    if not options.use_inner_iterations:
        return current_cost, 0
    evaluations = 0
    for block in active_blocks:
        if block.tangent_size == 0:
            continue
        evaluation = problem.evaluate(
            EvaluateOptions(parameter_blocks=[block], new_evaluation_point=False),
            compute_jacobian=True,
        )
        J = evaluation.jacobian
        if J is None or J.shape[1] == 0:
            continue
        step = solve_linear_system(
            J,
            -evaluation.residuals,
            solver_type=LinearSolverType.DENSE_QR,
            tolerance=options.eta,
            max_iterations=options.max_linear_solver_iterations,
            block_sizes=[block.tangent_size],
        ).x
        if not torch.all(torch.isfinite(step)):
            continue
        snapshot = problem.snapshot()
        problem.apply_delta(step, active_blocks=[block])
        candidate = problem.evaluate(EvaluateOptions(new_evaluation_point=False), compute_jacobian=False)
        evaluations += 1
        candidate_cost = float(candidate.cost.detach().cpu())
        improvement = current_cost - candidate_cost
        required = options.inner_iteration_tolerance * max(abs(current_cost), 1e-12)
        if improvement > 0.0 and improvement >= required:
            current_cost = candidate_cost
        else:
            problem.restore(snapshot)
    return current_cost, evaluations


def _updated_trust_region_radius(
    radius: float,
    rho: float,
    step_norm: float,
    max_radius: float,
) -> float:
    if rho > 0.75:
        return min(max_radius, max(2.0 * radius, 3.0 * max(step_norm, torch.finfo(torch.float64).eps)))
    if rho < 0.25:
        return max(radius * 0.5, torch.finfo(torch.float64).tiny)
    return radius


def _maybe_log_progress(options: SolverOptions, iteration: IterationSummary) -> None:
    if not options.minimizer_progress_to_stdout or options.logging_type is LoggingType.SILENT:
        return
    print(
        f"{iteration.iteration:4d}: "
        f"f:{iteration.cost: .6e} "
        f"d:{iteration.cost_change: .3e} "
        f"g:{iteration.gradient_max_norm: .3e} "
        f"h:{iteration.step_norm: .3e} "
        f"rho:{iteration.relative_decrease: .3e} "
        f"mu:{iteration.trust_region_radius: .3e}"
    )


def _line_search_direction(
    options: SolverOptions,
    grad: torch.Tensor,
    previous_grad: torch.Tensor | None,
    previous_direction: torch.Tensor | None,
    s_history: list[torch.Tensor],
    y_history: list[torch.Tensor],
    inverse_hessian: torch.Tensor | None,
) -> torch.Tensor:
    if options.line_search_direction_type is LineSearchDirectionType.STEEPEST_DESCENT:
        return -grad
    if (
        options.line_search_direction_type is LineSearchDirectionType.NONLINEAR_CONJUGATE_GRADIENT
        and previous_grad is not None
        and previous_direction is not None
    ):
        beta = _nonlinear_conjugate_gradient_beta(options, grad, previous_grad, previous_direction)
        return -grad + torch.clamp(beta, min=0.0) * previous_direction
    if options.line_search_direction_type is LineSearchDirectionType.LBFGS and s_history:
        return _lbfgs_two_loop(grad, s_history, y_history)
    if options.line_search_direction_type is LineSearchDirectionType.BFGS and inverse_hessian is not None:
        return -(inverse_hessian @ grad)
    return -grad


def _nonlinear_conjugate_gradient_beta(
    options: SolverOptions,
    grad: torch.Tensor,
    previous_grad: torch.Tensor,
    previous_direction: torch.Tensor,
) -> torch.Tensor:
    eps = torch.finfo(grad.dtype).eps
    if options.nonlinear_conjugate_gradient_type is NonlinearConjugateGradientType.POLAK_RIBIERE:
        return torch.dot(grad, grad - previous_grad) / torch.dot(previous_grad, previous_grad).clamp_min(eps)
    if options.nonlinear_conjugate_gradient_type is NonlinearConjugateGradientType.HESTENES_STIEFEL:
        y = grad - previous_grad
        return torch.dot(grad, y) / torch.dot(previous_direction, y).clamp_min(eps)
    return torch.dot(grad, grad) / torch.dot(previous_grad, previous_grad).clamp_min(eps)


def _lbfgs_two_loop(grad: torch.Tensor, s_history: list[torch.Tensor], y_history: list[torch.Tensor]) -> torch.Tensor:
    q = grad.clone()
    alphas: list[torch.Tensor] = []
    rhos: list[torch.Tensor] = []
    eps = torch.finfo(grad.dtype).eps
    for s, y in reversed(list(zip(s_history, y_history))):
        rho = 1.0 / torch.dot(y, s).clamp_min(eps)
        alpha = rho * torch.dot(s, q)
        q = q - alpha * y
        alphas.append(alpha)
        rhos.append(rho)
    s, y = s_history[-1], y_history[-1]
    gamma = torch.dot(s, y) / torch.dot(y, y).clamp_min(eps)
    r = gamma * q
    for s, y, alpha, rho in zip(s_history, y_history, reversed(alphas), reversed(rhos)):
        beta = rho * torch.dot(y, r)
        r = r + s * (alpha - beta)
    return -r


def _bfgs_update(inverse_hessian: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not _has_positive_curvature(s, y):
        return torch.eye(inverse_hessian.shape[0], dtype=inverse_hessian.dtype, device=inverse_hessian.device)
    ys = torch.dot(y, s)
    rho = 1.0 / ys
    eye = torch.eye(inverse_hessian.shape[0], dtype=inverse_hessian.dtype, device=inverse_hessian.device)
    sy = torch.outer(s, y)
    ys_outer = torch.outer(y, s)
    return (eye - rho * sy) @ inverse_hessian @ (eye - rho * ys_outer) + rho * torch.outer(s, s)


def _has_positive_curvature(s: torch.Tensor, y: torch.Tensor) -> bool:
    threshold = 10.0 * torch.finfo(s.dtype).eps * torch.linalg.norm(s) * torch.linalg.norm(y)
    return bool((torch.dot(s, y) > threshold).detach().cpu())


def _gradient_converged(grad_max: torch.Tensor, options: SolverOptions) -> bool:
    return bool(float(grad_max.detach().cpu()) <= options.gradient_tolerance)


def _parameter_converged(step: torch.Tensor, problem: Problem, options: SolverOptions) -> bool:
    state_norm_sq = 0.0
    for block in problem.parameter_blocks:
        state_norm_sq += float(torch.sum(block.tensor.detach().reshape(-1) ** 2).cpu())
    state_norm = state_norm_sq**0.5
    step_norm = float(torch.linalg.norm(step).detach().cpu())
    return step_norm <= options.parameter_tolerance * (state_norm + options.parameter_tolerance)
