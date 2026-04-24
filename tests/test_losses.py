import torch

import torch_ceres as tc


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

