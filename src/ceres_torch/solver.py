from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .costs import GradientChecker
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
    preprocessor_start = start
    summary = _new_summary(options, problem)
    active_blocks = _linear_solver_parameter_order(problem)
    summary.fixed_cost = _fixed_cost(problem)
    gradient_check_failure = _gradient_check_failure(options, problem)
    if gradient_check_failure is not None:
        summary.termination_type = TerminationType.FAILURE
        summary.message = gradient_check_failure
        summary.preprocessor_time_in_seconds = time.perf_counter() - preprocessor_start
        summary.total_time_in_seconds = time.perf_counter() - start
        return summary
    num_eliminate = _num_eliminate_for_schur(active_blocks, _effective_linear_solver(options.linear_solver_type))
    schur_visibility = _schur_visibility_from_residual_blocks(problem, active_blocks, num_eliminate)
    lm_radius = _LevenbergMarquardtRadiusState(options.initial_trust_region_radius, options.max_trust_region_radius)
    radius = lm_radius.radius
    consecutive_invalid = 0
    previous_cost: float | None = None
    inner_iterations_are_enabled = options.use_inner_iterations
    summary.preprocessor_time_in_seconds = time.perf_counter() - preprocessor_start
    minimizer_start = time.perf_counter()

    initial, initial_eval_time = _timed_evaluate(
        problem,
        EvaluateOptions(parameter_blocks=active_blocks),
        compute_jacobian=True,
    )
    summary.jacobian_evaluation_time_in_seconds += initial_eval_time
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
        return _finalize_trust_region_summary(summary, start, minimizer_start, problem, best_snapshot, best_cost)

    for iteration in range(options.max_num_iterations + 1):
        iter_start = time.perf_counter()
        evaluation, jacobian_eval_time = _timed_evaluate(
            problem,
            EvaluateOptions(parameter_blocks=active_blocks),
            compute_jacobian=True,
        )
        summary.jacobian_evaluation_time_in_seconds += jacobian_eval_time
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
            jacobian_evaluation_time_in_seconds=jacobian_eval_time,
        )

        if iteration == 0:
            iter_summary.iteration_time_in_seconds = time.perf_counter() - iter_start
            iter_summary.cumulative_time_in_seconds = time.perf_counter() - start
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
            step = dogleg_step(J, r, radius, dogleg_type=options.dogleg_type)
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
                min_iterations=options.min_linear_solver_iterations,
                max_iterations=options.max_linear_solver_iterations,
                tolerance=options.eta,
                preconditioner_type=options.preconditioner_type,
                block_sizes=[block.tangent_size for block in active_blocks],
                use_mixed_precision=options.use_mixed_precision_solves,
                max_refinement_iterations=options.max_num_refinement_iterations,
                max_num_spse_iterations=options.max_num_spse_iterations,
                use_spse_initialization=options.use_spse_initialization,
                spse_tolerance=options.spse_tolerance,
                visibility=schur_visibility,
                visibility_clustering_type=options.visibility_clustering_type,
            )
            linear_time = time.perf_counter() - linear_start
            step = linear_result.x
            linear_iterations = linear_result.summary.num_iterations
            summary.num_linear_solves += 1
        summary.linear_solver_time_in_seconds += linear_time

        if not torch.all(torch.isfinite(step)):
            consecutive_invalid += 1
            radius = _rejected_trust_region_radius(
                options.trust_region_strategy_type,
                radius,
                lm_radius,
            )
            iter_summary.step_is_valid = False
            iter_summary.step_is_successful = False
            iter_summary.linear_solver_iterations = linear_iterations
            iter_summary.linear_solver_time_in_seconds = linear_time
            iter_summary.step_solver_time_in_seconds = linear_time
            iter_summary.trust_region_radius = radius
            iter_summary.iteration_time_in_seconds = time.perf_counter() - iter_start
            iter_summary.cumulative_time_in_seconds = time.perf_counter() - start
            summary.num_unsuccessful_steps += 1
            if consecutive_invalid >= options.max_num_consecutive_invalid_steps:
                summary.termination_type = TerminationType.FAILURE
                summary.message = "Too many invalid trust-region steps."
                break
            summary.iterations.append(iter_summary)
            _maybe_log_progress(options, iter_summary)
            continue

        snapshot = problem.snapshot()
        if _has_bounds(active_blocks) and options.max_num_line_search_step_size_iterations > 0:
            line_search_start = time.perf_counter()
            step, line_search_iterations, line_search_evaluations, line_search_residual_time = _projected_line_search_step(
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
            summary.residual_evaluation_time_in_seconds += line_search_residual_time
            iter_summary.line_search_iterations = line_search_iterations
            iter_summary.line_search_function_evaluations = line_search_evaluations
            iter_summary.line_search_time_in_seconds = line_search_time
            iter_summary.residual_evaluation_time_in_seconds += line_search_residual_time
            problem.restore(snapshot)
        problem.apply_delta(step, active_blocks=active_blocks)
        candidate, residual_eval_time = _timed_evaluate(problem, compute_jacobian=False)
        summary.residual_evaluation_time_in_seconds += residual_eval_time
        summary.num_residual_evaluations += 1
        candidate_cost = float(candidate.cost.detach().cpu())
        model_decrease = _model_decrease(J, r, step)
        step_is_valid = model_decrease > 0.0
        inner_iterations_were_useful = False
        if step_is_valid and inner_iterations_are_enabled:
            inner_start = time.perf_counter()
            trust_region_candidate_cost = candidate_cost
            inner_result = _run_inner_iterations(
                problem,
                active_blocks,
                options,
                current_cost=candidate_cost,
            )
            candidate_cost = inner_result.cost
            inner_time = time.perf_counter() - inner_start
            summary.num_inner_iteration_steps += 1
            if inner_result.residual_evaluations:
                summary.num_residual_evaluations += inner_result.residual_evaluations
            if inner_result.jacobian_evaluations:
                summary.num_jacobian_evaluations += inner_result.jacobian_evaluations
            summary.residual_evaluation_time_in_seconds += inner_result.residual_time_in_seconds
            summary.jacobian_evaluation_time_in_seconds += inner_result.jacobian_time_in_seconds
            summary.inner_iteration_time_in_seconds += inner_time
            iter_summary.residual_evaluation_time_in_seconds += inner_result.residual_time_in_seconds
            iter_summary.jacobian_evaluation_time_in_seconds += inner_result.jacobian_time_in_seconds
            iter_summary.inner_iteration_time_in_seconds = inner_time
            inner_model_decrease = trust_region_candidate_cost - candidate_cost
            model_decrease += inner_model_decrease
            inner_iterations_were_useful = candidate_cost < min(cost, trust_region_candidate_cost)
            inner_iterations_are_enabled = inner_result.relative_progress > options.inner_iteration_tolerance
        rho = step_evaluator.step_quality(candidate_cost, model_decrease) if step_is_valid else 0.0
        accepted = step_is_valid and (inner_iterations_were_useful or rho > options.min_relative_decrease)

        iter_summary.step_norm = float(torch.linalg.norm(step).detach().cpu())
        iter_summary.cost_change = cost - candidate_cost
        iter_summary.relative_decrease = float(rho)
        iter_summary.step_is_valid = step_is_valid
        iter_summary.linear_solver_iterations = linear_iterations
        iter_summary.linear_solver_time_in_seconds = linear_time
        iter_summary.step_solver_time_in_seconds = linear_time
        iter_summary.residual_evaluation_time_in_seconds += residual_eval_time
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
                return _finalize_trust_region_summary(summary, start, minimizer_start, problem, best_snapshot, best_cost)
            if result is CallbackReturnType.SOLVER_TERMINATE_SUCCESSFULLY:
                summary.termination_type = TerminationType.USER_SUCCESS
                summary.message = "User callback terminated successfully."
                return _finalize_trust_region_summary(summary, start, minimizer_start, problem, best_snapshot, best_cost)
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
            next_eval, next_jacobian_time = _timed_evaluate(
                problem,
                EvaluateOptions(parameter_blocks=active_blocks),
                compute_jacobian=True,
            )
            summary.jacobian_evaluation_time_in_seconds += next_jacobian_time
            summary.num_jacobian_evaluations += 1
            if next_eval.gradient is not None and _gradient_converged(torch.max(torch.abs(next_eval.gradient)), options):
                summary.termination_type = TerminationType.CONVERGENCE
                summary.message = "Gradient tolerance reached."
                summary.final_cost = float(next_eval.cost.detach().cpu())
                break

        if radius <= options.min_trust_region_radius:
            summary.termination_type = TerminationType.CONVERGENCE
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
    return _finalize_trust_region_summary(summary, start, minimizer_start, problem, best_snapshot, best_cost)


def _line_search_solve(options: SolverOptions, problem: Problem) -> SolverSummary:
    start = time.perf_counter()
    preprocessor_start = start
    summary = _new_summary(options, problem)
    active_blocks = _linear_solver_parameter_order(problem)
    summary.fixed_cost = _fixed_cost(problem)
    gradient_check_failure = _gradient_check_failure(options, problem)
    if gradient_check_failure is not None:
        summary.termination_type = TerminationType.FAILURE
        summary.message = gradient_check_failure
        summary.preprocessor_time_in_seconds = time.perf_counter() - preprocessor_start
        summary.total_time_in_seconds = time.perf_counter() - start
        return summary
    summary.preprocessor_time_in_seconds = time.perf_counter() - preprocessor_start
    minimizer_start = time.perf_counter()
    initial, initial_eval_time = _timed_evaluate(
        problem,
        EvaluateOptions(parameter_blocks=active_blocks),
        compute_jacobian=True,
    )
    summary.jacobian_evaluation_time_in_seconds += initial_eval_time
    summary.initial_cost = float(initial.cost.detach().cpu())
    summary.final_cost = summary.initial_cost
    previous_grad: torch.Tensor | None = None
    previous_direction: torch.Tensor | None = None
    s_history: list[torch.Tensor] = []
    y_history: list[torch.Tensor] = []
    inverse_hessian: torch.Tensor | None = None

    for iteration in range(options.max_num_iterations + 1):
        iter_start = time.perf_counter()
        evaluation, jacobian_eval_time = _timed_evaluate(
            problem,
            EvaluateOptions(parameter_blocks=active_blocks),
            compute_jacobian=True,
        )
        summary.jacobian_evaluation_time_in_seconds += jacobian_eval_time
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
            jacobian_evaluation_time_in_seconds=jacobian_eval_time,
        )
        summary.iterations.append(iter_summary)
        _maybe_log_progress(options, iter_summary)
        if _gradient_converged(grad_max, options):
            summary.termination_type = TerminationType.CONVERGENCE
            summary.message = "Gradient tolerance reached."
            summary.final_cost = cost
            iter_summary.iteration_time_in_seconds = time.perf_counter() - iter_start
            iter_summary.cumulative_time_in_seconds = time.perf_counter() - start
            break
        if iteration == options.max_num_iterations:
            summary.termination_type = TerminationType.NO_CONVERGENCE
            summary.message = "Maximum iterations reached."
            iter_summary.iteration_time_in_seconds = time.perf_counter() - iter_start
            iter_summary.cumulative_time_in_seconds = time.perf_counter() - start
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
        direction_restarts = 0
        if _has_bounds(active_blocks):
            direction = _project_line_search_direction_for_bounds(direction, active_blocks)
        directional_derivative = torch.dot(g, direction)
        if torch.linalg.norm(direction) <= torch.finfo(direction.dtype).eps or directional_derivative >= 0:
            direction_restarts += 1
            direction = -g
            if _has_bounds(active_blocks):
                direction = _project_line_search_direction_for_bounds(direction, active_blocks)
            directional_derivative = torch.dot(g, direction)
            s_history.clear()
            y_history.clear()
            inverse_hessian = torch.eye(g.numel(), dtype=g.dtype, device=g.device)
        if torch.linalg.norm(direction) <= torch.finfo(direction.dtype).eps or directional_derivative >= 0:
            summary.termination_type = TerminationType.CONVERGENCE
            summary.message = "Projected gradient tolerance reached."
            summary.final_cost = cost
            iter_summary.line_search_direction_restarts = direction_restarts
            iter_summary.iteration_time_in_seconds = time.perf_counter() - iter_start
            iter_summary.cumulative_time_in_seconds = time.perf_counter() - start
            break
        accepted = False
        snapshot = problem.snapshot()
        accepted_direction = direction
        accepted_gradient: torch.Tensor | None = None
        directions = [direction]
        trial_evaluations = 0
        fallback_direction = -g
        if _has_bounds(active_blocks):
            fallback_direction = _project_line_search_direction_for_bounds(fallback_direction, active_blocks)
        if torch.linalg.norm(fallback_direction) > torch.finfo(fallback_direction.dtype).eps and not torch.allclose(direction, fallback_direction):
            directions.append(fallback_direction)
        line_search_start = time.perf_counter()
        for trial_index, trial_direction in enumerate(directions):
            if trial_index > 0:
                direction_restarts += 1
                if direction_restarts > options.max_num_line_search_direction_restarts:
                    break
            step_size = 1.0
            trial_derivative = torch.dot(g, trial_direction)
            previous_step_size: float | None = None
            previous_candidate_cost: float | None = None
            for ls_iter in range(options.max_num_line_search_step_size_iterations):
                problem.restore(snapshot)
                problem.apply_delta(step_size * trial_direction, active_blocks=active_blocks)
                candidate, candidate_eval_time = _timed_evaluate(
                    problem,
                    EvaluateOptions(parameter_blocks=active_blocks),
                    compute_jacobian=True,
                )
                summary.jacobian_evaluation_time_in_seconds += candidate_eval_time
                iter_summary.jacobian_evaluation_time_in_seconds += candidate_eval_time
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
                    iter_summary.line_search_direction_restarts = direction_restarts
                    iter_summary.line_search_time_in_seconds = time.perf_counter() - line_search_start
                    iter_summary.step_norm = float(torch.linalg.norm(step_size * trial_direction).detach().cpu())
                    iter_summary.cost_change = cost - candidate_cost
                    iter_summary.step_is_successful = True
                    summary.num_line_search_steps += iter_summary.line_search_iterations
                    summary.num_line_search_function_evaluations += trial_evaluations
                    summary.num_line_search_gradient_evaluations += trial_evaluations
                    summary.num_line_search_direction_restarts += direction_restarts
                    summary.line_search_total_time_in_seconds += iter_summary.line_search_time_in_seconds
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
            summary.num_line_search_direction_restarts += direction_restarts
            failed_line_search_time = time.perf_counter() - line_search_start
            summary.line_search_total_time_in_seconds += failed_line_search_time
            iter_summary.line_search_direction_restarts = direction_restarts
            iter_summary.line_search_time_in_seconds = failed_line_search_time
            summary.termination_type = TerminationType.NO_CONVERGENCE
            summary.message = "Line search failed to find a decreasing step."
            iter_summary.iteration_time_in_seconds = time.perf_counter() - iter_start
            iter_summary.cumulative_time_in_seconds = time.perf_counter() - start
            break
        if accepted_gradient is None:
            accepted_evaluation, accepted_eval_time = _timed_evaluate(
                problem,
                EvaluateOptions(parameter_blocks=active_blocks),
                compute_jacobian=True,
            )
            summary.jacobian_evaluation_time_in_seconds += accepted_eval_time
            iter_summary.jacobian_evaluation_time_in_seconds += accepted_eval_time
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
        iter_summary.iteration_time_in_seconds = time.perf_counter() - iter_start
        iter_summary.cumulative_time_in_seconds = time.perf_counter() - start
        for callback in options.callbacks:
            result = callback(iter_summary)
            if result is CallbackReturnType.SOLVER_ABORT:
                summary.termination_type = TerminationType.USER_FAILURE
                summary.message = "User callback aborted."
                return _finalize_line_search_summary(summary, start, minimizer_start)
            if result is CallbackReturnType.SOLVER_TERMINATE_SUCCESSFULLY:
                summary.termination_type = TerminationType.USER_SUCCESS
                summary.message = "User callback terminated successfully."
                return _finalize_line_search_summary(summary, start, minimizer_start)
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
    return _finalize_line_search_summary(summary, start, minimizer_start)


def _timed_evaluate(
    problem: Problem,
    options: EvaluateOptions | None = None,
    *,
    compute_jacobian: bool,
):
    eval_start = time.perf_counter()
    result = problem.evaluate(options, compute_jacobian=compute_jacobian)
    return result, time.perf_counter() - eval_start


def _fixed_cost(problem: Problem) -> float:
    fixed_residual_blocks = [
        residual_block
        for residual_block in problem.residual_blocks
        if all(block.tangent_size == 0 for block in residual_block.parameter_blocks)
    ]
    if not fixed_residual_blocks:
        return 0.0
    evaluation = problem.evaluate(
        EvaluateOptions(
            parameter_blocks=[],
            residual_blocks=fixed_residual_blocks,
        ),
        compute_jacobian=False,
    )
    return float(evaluation.cost.detach().cpu())


def _gradient_check_failure(options: SolverOptions, problem: Problem) -> str | None:
    if not options.check_gradients:
        return None
    checker = GradientChecker(
        relative_precision=options.gradient_check_relative_precision,
        relative_step_size=options.gradient_check_numeric_derivative_relative_step_size,
    )
    worst_error = 0.0
    for index, residual_block in enumerate(problem.residual_blocks):
        parameters = [block.tensor.detach().clone() for block in residual_block.parameter_blocks]
        result = checker.probe(residual_block.cost_function, parameters)
        worst_error = max(worst_error, float(result["max_relative_error"]))
        if not bool(result["ok"]):
            name = residual_block.name or f"#{index}"
            return (
                f"Gradient check failed for residual block {name}: "
                f"max relative error {worst_error:.3e} exceeds "
                f"{options.gradient_check_relative_precision:.3e}."
            )
    return None


def _finalize_trust_region_summary(
    summary: SolverSummary,
    start: float,
    minimizer_start: float,
    problem: Problem,
    best_snapshot: list[torch.Tensor],
    best_cost: float,
) -> SolverSummary:
    if summary.minimizer_time_in_seconds == 0.0:
        summary.minimizer_time_in_seconds = time.perf_counter() - minimizer_start
    postprocessor_start = time.perf_counter()
    problem.restore(best_snapshot)
    summary.final_cost = best_cost
    summary.postprocessor_time_in_seconds += time.perf_counter() - postprocessor_start
    summary.total_time_in_seconds = time.perf_counter() - start
    return summary


def _finalize_line_search_summary(
    summary: SolverSummary,
    start: float,
    minimizer_start: float,
) -> SolverSummary:
    if summary.minimizer_time_in_seconds == 0.0:
        summary.minimizer_time_in_seconds = time.perf_counter() - minimizer_start
    summary.postprocessor_time_in_seconds += max(0.0, time.perf_counter() - start - summary.preprocessor_time_in_seconds - summary.minimizer_time_in_seconds)
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
        dogleg_type=options.dogleg_type,
        line_search_direction_type=options.line_search_direction_type,
        line_search_type=options.line_search_type,
        line_search_interpolation_type=options.line_search_interpolation_type,
        nonlinear_conjugate_gradient_type=options.nonlinear_conjugate_gradient_type,
        preconditioner_type=options.preconditioner_type,
        max_lbfgs_rank=options.max_lbfgs_rank,
        dense_linear_algebra_library_type=options.dense_linear_algebra_library_type,
        sparse_linear_algebra_library_type=options.sparse_linear_algebra_library_type,
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


def _schur_visibility_from_residual_blocks(
    problem: Problem,
    blocks: list[ParameterBlock],
    num_eliminate: int,
) -> tuple[tuple[int, ...], ...] | None:
    if num_eliminate <= 0:
        return None
    eliminated: list[ParameterBlock] = []
    retained: list[ParameterBlock] = []
    offset = 0
    for block in blocks:
        next_offset = offset + block.tangent_size
        if block.tangent_size == 0:
            continue
        if next_offset <= num_eliminate:
            eliminated.append(block)
        elif offset >= num_eliminate:
            retained.append(block)
        else:
            return None
        offset = next_offset
    if not eliminated or not retained:
        return None

    eliminated_index = {block: index for index, block in enumerate(eliminated)}
    retained_index = {block: index for index, block in enumerate(retained)}
    visibility: list[set[int]] = [set() for _ in retained]
    for residual_block in problem.residual_blocks:
        visible_eliminated = {
            eliminated_index[block]
            for block in residual_block.parameter_blocks
            if block in eliminated_index
        }
        if not visible_eliminated:
            continue
        for block in residual_block.parameter_blocks:
            retained_id = retained_index.get(block)
            if retained_id is not None:
                visibility[retained_id].update(visible_eliminated)
    if not any(visibility):
        return None
    return tuple(tuple(sorted(points)) for points in visibility)


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


def _project_line_search_direction_for_bounds(
    direction: torch.Tensor,
    active_blocks: list[ParameterBlock],
) -> torch.Tensor:
    projected = direction.clone()
    offset = 0
    for block in active_blocks:
        n = block.tangent_size
        segment = projected[offset : offset + n]
        offset += n
        if n == 0 or n != block.size:
            continue
        value = block.tensor.detach().reshape(-1).to(dtype=segment.dtype, device=segment.device)
        tolerance = 10.0 * torch.finfo(segment.dtype).eps * torch.maximum(torch.abs(value), torch.ones_like(value))
        if block.lower_bound is not None:
            lower = block.lower_bound.reshape(-1).to(dtype=segment.dtype, device=segment.device)
            at_lower = value <= lower + tolerance
            segment.copy_(torch.where(at_lower & (segment < 0), torch.zeros_like(segment), segment))
        if block.upper_bound is not None:
            upper = block.upper_bound.reshape(-1).to(dtype=segment.dtype, device=segment.device)
            at_upper = value >= upper - tolerance
            segment.copy_(torch.where(at_upper & (segment > 0), torch.zeros_like(segment), segment))
    return projected


def _projected_line_search_step(
    problem: Problem,
    active_blocks: list[ParameterBlock],
    snapshot: list[torch.Tensor],
    step: torch.Tensor,
    cost: float,
    gradient: torch.Tensor,
    options: SolverOptions,
) -> tuple[torch.Tensor, int, int, float]:
    directional_derivative = float(torch.dot(gradient, step).detach().cpu())
    if directional_derivative >= 0.0:
        return step, 0, 0, 0.0

    step_size = 1.0
    previous_step_size: float | None = None
    previous_candidate_cost: float | None = None
    evaluations = 0
    iterations = 0
    residual_time = 0.0
    for _ in range(options.max_num_line_search_step_size_iterations):
        iterations += 1
        problem.restore(snapshot)
        problem.apply_delta(step_size * step, active_blocks=active_blocks)
        candidate, candidate_time = _timed_evaluate(
            problem,
            EvaluateOptions(parameter_blocks=active_blocks),
            compute_jacobian=False,
        )
        residual_time += candidate_time
        evaluations += 1
        candidate_cost = float(candidate.cost.detach().cpu())
        sufficient_decrease = cost + options.line_search_sufficient_function_decrease * step_size * directional_derivative
        if candidate_cost <= sufficient_decrease:
            problem.restore(snapshot)
            return step_size * step, iterations, evaluations, residual_time

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
    return step, iterations, evaluations, residual_time


@dataclass
class _InnerIterationResult:
    cost: float
    residual_evaluations: int = 0
    jacobian_evaluations: int = 0
    residual_time_in_seconds: float = 0.0
    jacobian_time_in_seconds: float = 0.0
    accepted_steps: int = 0
    relative_progress: float = 0.0


def _run_inner_iterations(
    problem: Problem,
    active_blocks: list[ParameterBlock],
    options: SolverOptions,
    *,
    current_cost: float,
) -> _InnerIterationResult:
    if not options.use_inner_iterations:
        return _InnerIterationResult(cost=current_cost)
    initial_cost = current_cost
    residual_evaluations = 0
    jacobian_evaluations = 0
    residual_time = 0.0
    jacobian_time = 0.0
    accepted_steps = 0
    for block in active_blocks:
        if block.tangent_size == 0:
            continue
        evaluation, evaluation_time = _timed_evaluate(
            problem,
            EvaluateOptions(parameter_blocks=[block], new_evaluation_point=False),
            compute_jacobian=True,
        )
        jacobian_time += evaluation_time
        jacobian_evaluations += 1
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
        candidate, candidate_time = _timed_evaluate(
            problem,
            EvaluateOptions(new_evaluation_point=False),
            compute_jacobian=False,
        )
        residual_time += candidate_time
        residual_evaluations += 1
        candidate_cost = float(candidate.cost.detach().cpu())
        improvement = current_cost - candidate_cost
        required = options.inner_iteration_tolerance * max(abs(current_cost), 1e-12)
        if improvement > 0.0 and improvement >= required:
            current_cost = candidate_cost
            accepted_steps += 1
        else:
            problem.restore(snapshot)
    if initial_cost > 0.0:
        relative_progress = 1.0 - current_cost / initial_cost
    else:
        relative_progress = 0.0
    return _InnerIterationResult(
        cost=current_cost,
        residual_evaluations=residual_evaluations,
        jacobian_evaluations=jacobian_evaluations,
        residual_time_in_seconds=residual_time,
        jacobian_time_in_seconds=jacobian_time,
        accepted_steps=accepted_steps,
        relative_progress=relative_progress,
    )


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
        return _lbfgs_two_loop(
            grad,
            s_history,
            y_history,
            use_approximate_eigenvalue_scaling=options.use_approximate_eigenvalue_bfgs_scaling,
        )
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
    eps = torch.finfo(grad.dtype).eps
    for s, y in reversed(list(zip(s_history, y_history))):
        rho = 1.0 / torch.dot(y, s).clamp_min(eps)
        alpha = rho * torch.dot(s, q)
        q = q - alpha * y
        alphas.append(alpha)
        rhos.append(rho)
    s, y = s_history[-1], y_history[-1]
    gamma = torch.dot(s, y) / torch.dot(y, y).clamp_min(eps)
    r = gamma * q if use_approximate_eigenvalue_scaling else q
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
