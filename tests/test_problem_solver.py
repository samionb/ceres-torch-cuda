import torch
import pytest

import ceres_torch as tc


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
