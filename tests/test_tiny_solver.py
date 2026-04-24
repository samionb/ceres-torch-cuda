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
    torch.testing.assert_close(x, torch.tensor([2.0], dtype=torch.float64), atol=1e-6, rtol=1e-6)
