import torch

import ceres_torch as tc


def test_trivial_loss() -> None:
    s = torch.tensor(2.0, dtype=torch.float64)
    rho = tc.TrivialLoss().evaluate(s)
    torch.testing.assert_close(rho[0], s)
    torch.testing.assert_close(rho[1], torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(rho[2], torch.tensor(0.0, dtype=torch.float64))


def test_huber_loss_outlier_region() -> None:
    s = torch.tensor(4.0, dtype=torch.float64)
    rho0, rho1, rho2 = tc.HuberLoss(1.0).evaluate(s)
    torch.testing.assert_close(rho0, torch.tensor(3.0, dtype=torch.float64))
    torch.testing.assert_close(rho1, torch.tensor(0.5, dtype=torch.float64))
    assert rho2 < 0


def test_composed_scaled_loss() -> None:
    s = torch.tensor(0.25, dtype=torch.float64)
    loss = tc.ScaledLoss(tc.ComposedLoss(tc.CauchyLoss(2.0), tc.SoftLOneLoss(1.0)), 3.0)
    rho0, rho1, rho2 = loss.evaluate(s)
    assert torch.isfinite(rho0)
    assert rho1 > 0
    assert torch.isfinite(rho2)


def test_convex_loss_uses_triggs_jacobian_correction() -> None:
    x = torch.tensor([2.0], dtype=torch.float64)
    problem = tc.Problem()
    problem.add_residual_block(
        tc.AutoDiffCostFunction(lambda x: torch.stack([x[0], x[0].new_tensor(1.0)]), [1]),
        tc.TolerantLoss(1.0, 1.0),
        [x],
    )
    result = problem.evaluate(compute_jacobian=True)
    assert result.jacobian is not None
    raw_residual = torch.tensor([2.0, 1.0], dtype=torch.float64)
    raw_jacobian = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    _, rho1, _ = tc.TolerantLoss(1.0, 1.0).evaluate(torch.dot(raw_residual, raw_residual))
    first_order_only = torch.sqrt(rho1) * raw_jacobian
    assert not torch.allclose(result.jacobian, first_order_only)
    torch.testing.assert_close(result.jacobian.T @ result.residuals, rho1 * (raw_jacobian.T @ raw_residual))
