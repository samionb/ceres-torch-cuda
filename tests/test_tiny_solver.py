import pytest
import torch

import ceres_torch as tc


def test_tiny_solver_summary_and_convergence() -> None:
    x = torch.tensor([3.0], dtype=torch.float64)
    solver = tc.TinySolver(
        tc.AutoDiffCostFunction(lambda x: x * x - 4.0, [1], 1),
        tc.TinySolverOptions(max_num_iterations=50, gradient_tolerance=1e-12),
    )

    summary = solver.solve(x)

    assert summary.IsSolutionUsable()
    assert "Tiny Solver Report" in summary.BriefReport()
    assert summary.status in {tc.TinySolverStatus.GRADIENT_TOO_SMALL, tc.TinySolverStatus.COST_CHANGE_TOO_SMALL}
    assert summary.gradient_max_norm >= 0.0
    torch.testing.assert_close(x, torch.tensor([2.0], dtype=torch.float64), atol=1e-6, rtol=1e-6)


def test_tiny_solver_cost_threshold_and_option_validation() -> None:
    with pytest.raises(ValueError, match="max_num_iterations"):
        tc.TinySolverOptions(max_num_iterations=-1).validate()
    with pytest.raises(ValueError, match="initial_trust_region_radius"):
        tc.TinySolverOptions(initial_trust_region_radius=0.0).validate()

    x = torch.tensor([0.0], dtype=torch.float64)
    solver = tc.TinySolver(
        tc.AutoDiffCostFunction(lambda x: x, [1], 1),
        tc.TinySolverOptions(cost_threshold=1e-12),
    )

    summary = solver.solve(x)

    assert summary.termination_type is tc.TerminationType.CONVERGENCE
    assert summary.status is tc.TinySolverStatus.GRADIENT_TOO_SMALL
    assert summary.iterations == 0


def test_tiny_solver_reports_max_iterations_when_not_converged() -> None:
    x = torch.tensor([10.0], dtype=torch.float64)
    solver = tc.TinySolver(
        tc.AutoDiffCostFunction(lambda x: x * x + 1.0, [1], 1),
        tc.TinySolverOptions(max_num_iterations=0, gradient_tolerance=0.0, cost_threshold=0.0),
    )

    summary = solver.solve(x)

    assert summary.termination_type is tc.TerminationType.NO_CONVERGENCE
    assert summary.status is tc.TinySolverStatus.HIT_MAX_ITERATIONS
    assert summary.iterations == 0
