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


def test_manifold_ceres_style_aliases_and_right_multiply() -> None:
    manifold = tc.SubsetManifold(4, [1, 3])
    x = torch.arange(4, dtype=torch.float64)
    delta = torch.tensor([0.25, -0.5], dtype=torch.float64)
    ambient_matrix = torch.arange(12, dtype=torch.float64).reshape(3, 4)

    torch.testing.assert_close(manifold.Plus(x, delta), manifold.plus(x, delta))
    torch.testing.assert_close(manifold.Minus(manifold.Plus(x, delta), x), delta)
    torch.testing.assert_close(manifold.PlusJacobian(x), manifold.plus_jacobian(x))
    torch.testing.assert_close(manifold.MinusJacobian(x), manifold.minus_jacobian(x))
    torch.testing.assert_close(
        manifold.RightMultiplyByPlusJacobian(x, ambient_matrix),
        ambient_matrix @ manifold.plus_jacobian(x),
    )


def test_product_and_sphere_manifold_identities() -> None:
    product = tc.ProductManifold(tc.EuclideanManifold(2), tc.SphereManifold(3))
    x = torch.tensor([1.0, -2.0, 0.0, 0.0, 2.0], dtype=torch.float64)
    delta = torch.tensor([0.5, -0.25, 0.1, -0.2], dtype=torch.float64)

    y = product.plus(x, delta)
    recovered = product.minus(y, x)

    torch.testing.assert_close(recovered, delta, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(torch.linalg.norm(y[2:]), torch.linalg.norm(x[2:]), atol=1e-10, rtol=1e-10)


def test_line_manifold_analytic_jacobian_matches_plus_linearization() -> None:
    manifold = tc.LineManifold(3)
    x = torch.tensor([1.0, -2.0, 0.5, 0.0, 0.0, 1.0], dtype=torch.float64)
    delta = torch.tensor([0.2, -0.1, 0.03, -0.04], dtype=torch.float64)
    eps = torch.tensor(1e-6, dtype=torch.float64)

    J = manifold.plus_jacobian(x)
    y = manifold.plus(x, eps * delta)
    linearized = x + eps * (J @ delta)

    assert J.shape == (6, 4)
    assert manifold.minus_jacobian(x).shape == (4, 6)
    torch.testing.assert_close(y, linearized, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(manifold.minus(y, x), eps * delta, atol=1e-10, rtol=1e-10)


def test_angle_axis_rotates_point() -> None:
    aa = torch.tensor([0.0, 0.0, torch.pi / 2], dtype=torch.float64)
    point = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    rotated = tc.angle_axis_rotate_point(aa, point)
    torch.testing.assert_close(rotated, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-7, rtol=1e-7)


def test_rotation_matrix_roundtrip_and_rotate_point() -> None:
    angle_axis = torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64)
    point = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)

    matrix = tc.angle_axis_to_rotation_matrix(angle_axis)
    recovered = tc.rotation_matrix_to_angle_axis(matrix)

    torch.testing.assert_close(recovered, angle_axis, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(tc.rotation_matrix_rotate_point(matrix, point), tc.angle_axis_rotate_point(angle_axis, point))


def test_quaternion_inverse_and_cross_product() -> None:
    q = tc.angle_axis_to_quaternion(torch.tensor([0.1, 0.2, -0.3], dtype=torch.float64))
    identity = tc.quaternion_product(q, tc.quaternion_inverse(q))
    torch.testing.assert_close(identity, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64), atol=1e-8, rtol=1e-8)

    a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    b = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(tc.cross_product(a, b), torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(tc.dot_product(a, b), torch.tensor(0.0, dtype=torch.float64))


def test_quaternion_scaled_rotation_matches_ceres_semantics() -> None:
    q = torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(tc.quaternion_to_scaled_rotation_matrix(q), 4.0 * torch.eye(3, dtype=torch.float64))
    torch.testing.assert_close(tc.quaternion_to_rotation_matrix(q), torch.eye(3, dtype=torch.float64))
    torch.testing.assert_close(tc.QuaternionToScaledRotation(q), tc.quaternion_to_scaled_rotation_matrix(q))
    torch.testing.assert_close(tc.QuaternionToRotation(q), tc.quaternion_to_rotation_matrix(q))


def test_ceres_style_rotation_aliases_and_euler_roundtrip() -> None:
    euler = torch.tensor([10.0, -20.0, 30.0], dtype=torch.float64)
    matrix = tc.EulerAnglesToRotationMatrix(euler)
    recovered = tc.RotationMatrixToEulerAngles(matrix)

    torch.testing.assert_close(recovered, euler, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(tc.RotationMatrixToQuaternion(matrix), tc.rotation_matrix_to_quaternion(matrix))
    torch.testing.assert_close(
        tc.AngleAxisToRotationMatrix(tc.RotationMatrixToAngleAxis(matrix)),
        matrix,
        atol=1e-8,
        rtol=1e-8,
    )


def test_generic_euler_rotation_axes_support_batches() -> None:
    angles = torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.4, -0.1]], dtype=torch.float64)
    rotations = tc.EulerAnglesToRotation(angles, axes="ZYX", intrinsic=True)
    identity = torch.eye(3, dtype=torch.float64).expand(2, 3, 3)
    torch.testing.assert_close(rotations @ rotations.transpose(-1, -2), identity, atol=1e-10, rtol=1e-10)
