import torch

import ceres_torch as tc


def test_solver_golden_result_comparison_helper() -> None:
    x = torch.tensor([0.5], dtype=torch.float64)
    problem = tc.Problem()
    problem.AddResidualBlock(tc.AutoDiffCostFunction(lambda x: 10.0 - x, [1], 1), None, [x])

    summary = tc.solve(tc.SolverOptions(max_num_iterations=25, gradient_tolerance=1e-12), problem)
    golden = tc.GoldenSolverResult(
        initial_cost=45.125,
        final_cost=summary.final_cost,
        termination_type=summary.termination_type,
        num_iterations=len(summary.iterations),
        parameter_values=[[10.0]],
    )

    tc.assert_solver_matches_golden(summary, golden, parameters=[x], cost_atol=1e-9, parameter_atol=1e-6)


def test_solver_golden_result_from_dict() -> None:
    golden = tc.GoldenSolverResult.from_dict(
        {
            "initial_cost": 1.0,
            "final_cost": 0.0,
            "termination_type": "CONVERGENCE",
            "num_iterations": 3,
            "parameter_values": [[1.0, 2.0]],
        }
    )

    assert golden.termination_type is tc.TerminationType.CONVERGENCE
    assert golden.parameter_values == [[1.0, 2.0]]
