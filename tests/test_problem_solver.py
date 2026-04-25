import torch
import pytest

import ceres_torch as tc
from ceres_torch.solver import _LevenbergMarquardtRadiusState, _TrustRegionStepEvaluator


def test_problem_evaluate_autodiff_jacobian() -> None:
    x = torch.tensor([2.0], dtype=torch.float64)
    problem = tc.Problem()
    problem.add_residual_block(tc.AutoDiffCostFunction(lambda x: x * x - 4.0, [1]), None, [x])
    result = problem.evaluate(compute_jacobian=True)
    torch.testing.assert_close(result.residuals, torch.tensor([0.0], dtype=torch.float64))
    assert result.jacobian is not None
    torch.testing.assert_close(result.jacobian, torch.tensor([[4.0]], dtype=torch.float64))


def test_solve_hello_world() -> None:
    x = torch.tensor([0.5], dtype=torch.float64)
    problem = tc.Problem()
    problem.add_residual_block(tc.AutoDiffCostFunction(lambda x: 10.0 - x, [1]), None, [x])
    summary = tc.solve(tc.SolverOptions(max_num_iterations=25, gradient_tolerance=1e-12), problem)
    assert summary.IsSolutionUsable()
    torch.testing.assert_close(x, torch.tensor([10.0], dtype=torch.float64), atol=1e-6, rtol=1e-6)


def test_bounds_are_enforced() -> None:
    x = torch.tensor([0.0], dtype=torch.float64)
    problem = tc.Problem()
    problem.add_parameter_block(x)
    problem.set_bounds(x, lower=torch.tensor([0.0], dtype=torch.float64), upper=torch.tensor([1.0], dtype=torch.float64))
    problem.add_residual_block(tc.AutoDiffCostFunction(lambda x: 10.0 - x, [1]), None, [x])
    summary = tc.solve(tc.SolverOptions(max_num_iterations=10), problem)
    assert summary.IsSolutionUsable()
    assert 0.0 <= x.item() <= 1.0
    torch.testing.assert_close(x, torch.tensor([1.0], dtype=torch.float64), atol=1e-6, rtol=1e-6)


def test_trust_region_projected_line_search_rescales_bound_constrained_step() -> None:
    x = torch.tensor([0.32], dtype=torch.float64)
    problem = tc.Problem()
    problem.AddParameterBlock(x)
    problem.SetParameterUpperBound(x, 0, 1.57)
    problem.AddResidualBlock(
        tc.AutoDiffCostFunction(lambda x: torch.sin(5.0 * x[0]).reshape(1), [1], 1),
        None,
        [x],
    )

    summary = tc.solve(
        tc.SolverOptions(
            max_num_iterations=1,
            initial_trust_region_radius=1e4,
            gradient_tolerance=0.0,
            function_tolerance=0.0,
            parameter_tolerance=0.0,
            min_relative_decrease=1e-3,
        ),
        problem,
    )

    step_iteration = summary.iterations[1]
    assert summary.num_successful_steps == 1
    assert summary.num_unsuccessful_steps == 0
    assert summary.num_line_search_steps == step_iteration.line_search_iterations
    assert summary.num_line_search_function_evaluations == step_iteration.line_search_function_evaluations
    assert step_iteration.line_search_iterations > 1
    assert x.item() < 1.57
    assert summary.final_cost < summary.initial_cost
    assert "Line search steps" in summary.FullReport()


def test_gradient_problem_rosenbrock() -> None:
    x = torch.tensor([-1.2, 1.0], dtype=torch.float64)

    def rosenbrock(x: torch.Tensor) -> torch.Tensor:
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    problem = tc.GradientProblem.from_callable(rosenbrock, size=2)
    options = tc.GradientProblemSolverOptions(
        max_num_iterations=200,
        line_search_direction_type=tc.LineSearchDirectionType.LBFGS,
        gradient_tolerance=1e-8,
    )
    summary = tc.gradient_solve(options, problem, x)
    assert summary.IsSolutionUsable()
    torch.testing.assert_close(x, torch.tensor([1.0, 1.0], dtype=torch.float64), atol=1e-4, rtol=1e-4)


def test_gradient_problem_bfgs_direction() -> None:
    x = torch.tensor([-1.2, 1.0], dtype=torch.float64)

    def rosenbrock(x: torch.Tensor) -> torch.Tensor:
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    problem = tc.GradientProblem.from_callable(rosenbrock, size=2)
    options = tc.GradientProblemSolverOptions(
        max_num_iterations=200,
        line_search_direction_type=tc.LineSearchDirectionType.BFGS,
        gradient_tolerance=1e-8,
    )
    summary = tc.gradient_solve(options, problem, x)
    assert summary.IsSolutionUsable()
    torch.testing.assert_close(x, torch.tensor([1.0, 1.0], dtype=torch.float64), atol=1e-4, rtol=1e-4)


def test_gradient_problem_options_validate_and_reports_include_counters(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match="max_num_iterations"):
        tc.GradientProblemSolverOptions(max_num_iterations=-1).validate()
    with pytest.raises(ValueError, match="max_line_search_step_contraction"):
        tc.GradientProblemSolverOptions(
            max_line_search_step_contraction=0.8,
            min_line_search_step_contraction=0.6,
        ).validate()

    x = torch.tensor([-1.2, 1.0], dtype=torch.float64)

    def rosenbrock(x: torch.Tensor) -> torch.Tensor:
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    summary = tc.gradient_solve(
        tc.GradientProblemSolverOptions(
            max_num_iterations=1,
            minimizer_progress_to_stdout=True,
            logging_type=tc.LoggingType.PER_MINIMIZER_ITERATION,
        ),
        tc.GradientProblem.from_callable(rosenbrock, size=2),
        x,
    )

    assert summary.num_cost_evaluations >= summary.num_gradient_evaluations
    assert "Gradient Solver Summary" in summary.FullReport()
    assert "Cost evaluations" in summary.FullReport()
    assert "0:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "interpolation_type",
    [
        tc.LineSearchInterpolationType.BISECTION,
        tc.LineSearchInterpolationType.QUADRATIC,
        tc.LineSearchInterpolationType.CUBIC,
    ],
)
def test_gradient_problem_line_search_interpolation_modes_converge(
    interpolation_type: tc.LineSearchInterpolationType,
) -> None:
    x = torch.tensor([-1.2, 1.0], dtype=torch.float64)

    def rosenbrock(x: torch.Tensor) -> torch.Tensor:
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    summary = tc.gradient_solve(
        tc.GradientProblemSolverOptions(
            max_num_iterations=250,
            line_search_direction_type=tc.LineSearchDirectionType.LBFGS,
            line_search_interpolation_type=interpolation_type,
            function_tolerance=1e-12,
            gradient_tolerance=1e-8,
            parameter_tolerance=1e-12,
        ),
        tc.GradientProblem.from_callable(rosenbrock, size=2),
        x,
    )

    successful = [iteration for iteration in summary.iterations if iteration.step_is_successful]
    assert summary.IsSolutionUsable()
    assert successful
    assert any(iteration.line_search_function_evaluations >= iteration.line_search_iterations for iteration in successful)
    assert all(iteration.cumulative_time_in_seconds >= 0.0 for iteration in successful)
    torch.testing.assert_close(x, torch.tensor([1.0, 1.0], dtype=torch.float64), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "direction_type,ncg_type",
    [
        (tc.LineSearchDirectionType.LBFGS, None),
        (tc.LineSearchDirectionType.BFGS, None),
        (
            tc.LineSearchDirectionType.NONLINEAR_CONJUGATE_GRADIENT,
            tc.NonlinearConjugateGradientType.POLAK_RIBIERE,
        ),
    ],
)
def test_solve_line_search_direction_modes_converge_on_rosenbrock(
    direction_type: tc.LineSearchDirectionType,
    ncg_type: tc.NonlinearConjugateGradientType | None,
) -> None:
    x = torch.tensor([-1.2, 1.0], dtype=torch.float64)
    problem = tc.Problem()
    problem.AddResidualBlock(
        tc.AutoDiffCostFunction(lambda x: torch.stack([1.0 - x[0], 10.0 * (x[1] - x[0] ** 2)]), [2], 2),
        None,
        [x],
    )
    options = tc.SolverOptions(
        minimizer_type=tc.MinimizerType.LINE_SEARCH,
        line_search_direction_type=direction_type,
        line_search_type=tc.LineSearchType.WOLFE,
        max_num_iterations=250,
        function_tolerance=1e-12,
        gradient_tolerance=1e-8,
        parameter_tolerance=1e-12,
    )
    if ncg_type is not None:
        options.nonlinear_conjugate_gradient_type = ncg_type

    summary = tc.solve(options, problem)

    assert summary.IsSolutionUsable()
    assert summary.line_search_direction_type is direction_type
    torch.testing.assert_close(x, torch.tensor([1.0, 1.0], dtype=torch.float64), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "interpolation_type",
    [
        tc.LineSearchInterpolationType.BISECTION,
        tc.LineSearchInterpolationType.QUADRATIC,
        tc.LineSearchInterpolationType.CUBIC,
    ],
)
def test_solve_line_search_interpolation_modes_converge(interpolation_type: tc.LineSearchInterpolationType) -> None:
    x = torch.tensor([-1.2, 1.0], dtype=torch.float64)
    problem = tc.Problem()
    problem.AddResidualBlock(
        tc.AutoDiffCostFunction(lambda x: torch.stack([1.0 - x[0], 10.0 * (x[1] - x[0] ** 2)]), [2], 2),
        None,
        [x],
    )

    summary = tc.solve(
        tc.SolverOptions(
            minimizer_type=tc.MinimizerType.LINE_SEARCH,
            line_search_direction_type=tc.LineSearchDirectionType.LBFGS,
            line_search_interpolation_type=interpolation_type,
            max_num_iterations=250,
            function_tolerance=1e-12,
            gradient_tolerance=1e-8,
            parameter_tolerance=1e-12,
        ),
        problem,
    )

    assert summary.IsSolutionUsable()
    torch.testing.assert_close(x, torch.tensor([1.0, 1.0], dtype=torch.float64), atol=1e-4, rtol=1e-4)


def test_dense_schur_respects_parameter_ordering_groups() -> None:
    camera = torch.tensor([0.0], dtype=torch.float64)
    point = torch.tensor([0.0], dtype=torch.float64)
    problem = tc.Problem()
    problem.AddParameterBlock(camera)
    problem.AddParameterBlock(point)
    problem.SetParameterBlockOrderingGroup(point, 0)
    problem.SetParameterBlockOrderingGroup(camera, 1)
    problem.AddResidualBlock(
        tc.AutoDiffCostFunction(lambda point, camera: torch.stack([point[0] + camera[0] - 3.0, 2.0 * point[0] - camera[0]]), [1, 1], 2),
        None,
        [point, camera],
    )

    summary = tc.solve(
        tc.SolverOptions(
            linear_solver_type=tc.LinearSolverType.DENSE_SCHUR,
            max_num_iterations=25,
            gradient_tolerance=1e-12,
        ),
        problem,
    )

    assert summary.IsSolutionUsable()
    assert summary.linear_solver_type_used is tc.LinearSolverType.DENSE_SCHUR
    assert problem.GetParameterBlockOrderingGroup(point) == 0
    torch.testing.assert_close(point, torch.tensor([1.0], dtype=torch.float64), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(camera, torch.tensor([2.0], dtype=torch.float64), atol=1e-6, rtol=1e-6)


def test_trust_region_radius_expands_after_high_quality_steps() -> None:
    x = torch.tensor([0.5], dtype=torch.float64)
    problem = tc.Problem()
    problem.AddResidualBlock(tc.AutoDiffCostFunction(lambda x: 10.0 - x, [1]), None, [x])

    summary = tc.solve(
        tc.SolverOptions(
            max_num_iterations=3,
            initial_trust_region_radius=1.0,
            gradient_tolerance=0.0,
            function_tolerance=0.0,
            parameter_tolerance=0.0,
        ),
        problem,
    )

    assert len(summary.iterations) >= 3
    assert summary.iterations[2].trust_region_radius > summary.iterations[1].trust_region_radius
    assert summary.iterations[1].step_solver_time_in_seconds >= 0.0


def test_levenberg_marquardt_radius_update_matches_ceres_strategy() -> None:
    state = _LevenbergMarquardtRadiusState(radius=2.0, max_radius=20.0)

    assert state.radius == 2.0
    assert state.step_rejected() == 1.0
    assert state.step_rejected() == 0.25
    assert state.step_accepted(1.0) == pytest.approx(0.75)
    assert state.step_accepted(1.0) == pytest.approx(2.25)
    assert state.step_accepted(0.25) == pytest.approx(2.0)
    assert state.step_accepted(1.0) == pytest.approx(6.0)
    assert state.step_accepted(1.0) == pytest.approx(18.0)
    assert state.step_accepted(1.0) == pytest.approx(20.0)


def test_nonmonotonic_step_evaluator_uses_accumulated_model_change() -> None:
    evaluator = _TrustRegionStepEvaluator(initial_cost=10.0, max_consecutive_nonmonotonic_steps=2)

    assert evaluator.step_quality(9.0, 2.0) == pytest.approx(0.5)
    evaluator.step_accepted(9.0, 2.0)

    assert evaluator.step_quality(9.5, 1.0) == pytest.approx(1.0 / 6.0)
    evaluator.step_accepted(9.5, 1.0)

    assert evaluator.step_quality(9.4, 1.0) == pytest.approx(0.15)
    evaluator.step_accepted(9.4, 1.0)

    assert evaluator.step_quality(9.6, 1.0) == pytest.approx(-0.05)


def test_inner_iterations_improve_blockwise_nonlinear_problem() -> None:
    def run(use_inner_iterations: bool) -> tuple[tc.SolverSummary, torch.Tensor, torch.Tensor]:
        x = torch.tensor([-1.2], dtype=torch.float64)
        y = torch.tensor([1.0], dtype=torch.float64)
        problem = tc.Problem()
        problem.AddResidualBlock(
            tc.AutoDiffCostFunction(lambda x, y: torch.stack([10.0 * (y[0] - x[0] ** 2), 1.0 - x[0]]), [1, 1], 2),
            None,
            [x, y],
        )
        summary = tc.solve(
            tc.SolverOptions(
                max_num_iterations=10,
                use_inner_iterations=use_inner_iterations,
                inner_iteration_tolerance=0.0,
                gradient_tolerance=0.0,
                function_tolerance=0.0,
                parameter_tolerance=0.0,
            ),
            problem,
        )
        return summary, x, y

    plain_summary, _, _ = run(False)
    inner_summary, _, _ = run(True)

    assert inner_summary.final_cost < plain_summary.final_cost
    assert inner_summary.num_residual_evaluations > plain_summary.num_residual_evaluations


def test_nonmonotonic_trust_region_restores_best_state_for_callbacks_and_final_state() -> None:
    x = torch.tensor([-20.0], dtype=torch.float64)
    problem = tc.Problem()
    problem.AddParameterBlock(x)
    problem.AddResidualBlock(
        tc.AutoDiffCostFunction(lambda x: (torch.sin(3.0 * x[0]) + 0.1 * x[0]).reshape(1), [1], 1),
        None,
        [x],
    )
    callback_costs: list[float] = []

    def callback(iteration: tc.IterationSummary) -> tc.CallbackReturnType:
        if iteration.step_is_nonmonotonic:
            callback_costs.append(float(problem.evaluate(compute_jacobian=False).cost.detach().cpu()))
        return tc.CallbackReturnType.SOLVER_CONTINUE

    summary = tc.solve(
        tc.SolverOptions(
            max_num_iterations=8,
            use_nonmonotonic_steps=True,
            max_consecutive_nonmonotonic_steps=5,
            update_state_every_iteration=True,
            callbacks=[callback],
            initial_trust_region_radius=0.01,
            gradient_tolerance=0.0,
            function_tolerance=0.0,
            parameter_tolerance=0.0,
            min_relative_decrease=1e-6,
        ),
        problem,
    )

    nonmonotonic_steps = [iteration for iteration in summary.iterations if iteration.step_is_nonmonotonic]
    assert nonmonotonic_steps
    uphill_step = nonmonotonic_steps[-1]
    uphill_candidate_cost = uphill_step.cost - uphill_step.cost_change
    final_evaluated_cost = float(problem.evaluate(compute_jacobian=False).cost.detach().cpu())

    assert callback_costs
    assert summary.final_cost < uphill_candidate_cost
    assert callback_costs[-1] < uphill_candidate_cost
    torch.testing.assert_close(torch.tensor(final_evaluated_cost), torch.tensor(summary.final_cost))


def test_minimizer_progress_to_stdout_logs_iteration(capsys: pytest.CaptureFixture[str]) -> None:
    x = torch.tensor([0.5], dtype=torch.float64)
    problem = tc.Problem()
    problem.AddResidualBlock(tc.AutoDiffCostFunction(lambda x: 10.0 - x, [1]), None, [x])

    tc.solve(
        tc.SolverOptions(
            max_num_iterations=0,
            minimizer_progress_to_stdout=True,
            logging_type=tc.LoggingType.PER_MINIMIZER_ITERATION,
        ),
        problem,
    )

    output = capsys.readouterr().out
    assert "0:" in output
    assert "f:" in output


def test_powell_function_converges() -> None:
    x1 = torch.tensor([3.0], dtype=torch.float64)
    x2 = torch.tensor([-1.0], dtype=torch.float64)
    x3 = torch.tensor([0.0], dtype=torch.float64)
    x4 = torch.tensor([1.0], dtype=torch.float64)
    problem = tc.Problem()
    problem.AddResidualBlock(tc.AutoDiffCostFunction(lambda x1, x2: x1 + 10.0 * x2, [1, 1]), None, [x1, x2])
    problem.AddResidualBlock(tc.AutoDiffCostFunction(lambda x3, x4: torch.sqrt(x3.new_tensor(5.0)) * (x3 - x4), [1, 1]), None, [x3, x4])
    problem.AddResidualBlock(tc.AutoDiffCostFunction(lambda x2, x3: (x2 - 2.0 * x3) ** 2, [1, 1]), None, [x2, x3])
    problem.AddResidualBlock(tc.AutoDiffCostFunction(lambda x1, x4: torch.sqrt(x1.new_tensor(10.0)) * (x1 - x4) ** 2, [1, 1]), None, [x1, x4])
    summary = tc.solve(
        tc.SolverOptions(max_num_iterations=500, function_tolerance=1e-12, parameter_tolerance=1e-12),
        problem,
    )
    assert summary.IsSolutionUsable()
    final = torch.cat([x1, x2, x3, x4])
    assert torch.linalg.norm(final) < 1e-3
