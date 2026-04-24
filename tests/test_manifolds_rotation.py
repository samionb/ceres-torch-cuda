import torch

import ceres_torch as tc


def test_quaternion_plus_minus_roundtrip() -> None:
    manifold = tc.QuaternionManifold()
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    delta = torch.tensor([0.01, -0.02, 0.03], dtype=torch.float64)
    y = manifold.plus(q, delta)
    recovered = manifold.minus(y, q)
    torch.testing.assert_close(recovered, delta, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(torch.linalg.norm(y), torch.tensor(1.0, dtype=torch.float64))


def test_subset_manifold_jacobian() -> None:
    manifold = tc.SubsetManifold(4, [1, 3])
    x = torch.arange(4, dtype=torch.float64)
    delta = torch.tensor([10.0, 20.0], dtype=torch.float64)
    y = manifold.plus(x, delta)
    torch.testing.assert_close(y, torch.tensor([10.0, 1.0, 22.0, 3.0], dtype=torch.float64))
    assert manifold.plus_jacobian(x).shape == (4, 2)


def test_angle_axis_rotates_point() -> None:
    aa = torch.tensor([0.0, 0.0, torch.pi / 2], dtype=torch.float64)
    point = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    rotated = tc.angle_axis_rotate_point(aa, point)
    torch.testing.assert_close(rotated, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-7, rtol=1e-7)

