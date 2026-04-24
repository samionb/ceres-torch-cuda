import torch

import ceres_torch as tc


def test_covariance_dense_svd_block() -> None:
    x = torch.tensor([0.0], dtype=torch.float64)
    problem = tc.Problem()
    block = problem.add_parameter_block(x)
    problem.add_residual_block(tc.AutoDiffCostFunction(lambda x: 2.0 * x - 1.0, [1]), None, [x])
    covariance = tc.Covariance(tc.CovarianceOptions(algorithm_type=tc.CovarianceAlgorithmType.DENSE_SVD))
    assert covariance.compute([(block, block)], problem)
    cov = covariance.get_covariance_block(block, block)
    torch.testing.assert_close(cov, torch.tensor([[0.25]], dtype=torch.float64), atol=1e-9, rtol=1e-9)


def test_covariance_rank_deficiency_policy() -> None:
    x = torch.tensor([0.0], dtype=torch.float64)
    y = torch.tensor([0.0], dtype=torch.float64)
    problem = tc.Problem()
    bx = problem.add_parameter_block(x)
    by = problem.add_parameter_block(y)
    problem.add_residual_block(tc.AutoDiffCostFunction(lambda x, y: x + y, [1, 1]), None, [x, y])

    strict = tc.Covariance(tc.CovarianceOptions(algorithm_type=tc.CovarianceAlgorithmType.DENSE_SVD))
    assert not strict.compute([(bx, bx), (by, by)], problem)

    pseudo_inverse = tc.Covariance(
        tc.CovarianceOptions(algorithm_type=tc.CovarianceAlgorithmType.DENSE_SVD, null_space_rank=-1)
    )
    assert pseudo_inverse.compute([(bx, bx), (by, by), (bx, by)], problem)
    matrix = pseudo_inverse.get_covariance_matrix_in_tangent_space([bx, by])
    torch.testing.assert_close(matrix, torch.full((2, 2), 0.25, dtype=torch.float64), atol=1e-9, rtol=1e-9)


def test_covariance_respects_apply_loss_function_option() -> None:
    x = torch.tensor([0.0], dtype=torch.float64)
    problem = tc.Problem()
    block = problem.add_parameter_block(x)
    problem.add_residual_block(tc.AutoDiffCostFunction(lambda x: x - 10.0, [1]), tc.HuberLoss(1.0), [x])

    robust = tc.Covariance(tc.CovarianceOptions(apply_loss_function=True))
    plain = tc.Covariance(tc.CovarianceOptions(apply_loss_function=False))

    assert robust.compute([(block, block)], problem)
    assert plain.compute([(block, block)], problem)
    torch.testing.assert_close(robust.get_covariance_block(block, block), torch.tensor([[10.0]], dtype=torch.float64))
    torch.testing.assert_close(plain.get_covariance_block(block, block), torch.tensor([[1.0]], dtype=torch.float64))


def test_covariance_sparse_qr_matches_dense_svd_for_full_rank_problem() -> None:
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    problem = tc.Problem()
    block = problem.add_parameter_block(x)
    A = torch.tensor([[2.0, 0.0], [1.0, 3.0], [0.5, -1.0]], dtype=torch.float64)
    problem.add_residual_block(tc.NormalPrior(A, torch.zeros(2, dtype=torch.float64)), None, [x])

    dense = tc.Covariance(tc.CovarianceOptions(algorithm_type=tc.CovarianceAlgorithmType.DENSE_SVD))
    qr = tc.Covariance(tc.CovarianceOptions(algorithm_type=tc.CovarianceAlgorithmType.SPARSE_QR))

    assert dense.compute([(block, block)], problem)
    assert qr.compute([(block, block)], problem)
    torch.testing.assert_close(qr.get_covariance_block(block, block), dense.get_covariance_block(block, block), atol=1e-10, rtol=1e-10)


def test_covariance_constant_parameter_blocks_return_zero_ambient_blocks() -> None:
    fixed = torch.tensor([3.0], dtype=torch.float64)
    variable = torch.tensor([0.0], dtype=torch.float64)
    problem = tc.Problem()
    fixed_block = problem.add_parameter_block(fixed)
    variable_block = problem.add_parameter_block(variable)
    problem.SetParameterBlockConstant(fixed)
    problem.add_residual_block(tc.AutoDiffCostFunction(lambda variable: 2.0 * variable - 1.0, [1]), None, [variable])

    covariance = tc.Covariance()

    assert covariance.compute([(fixed_block, fixed_block), (fixed_block, variable_block), (variable_block, variable_block)], problem)
    torch.testing.assert_close(covariance.get_covariance_block(fixed_block, fixed_block), torch.zeros((1, 1), dtype=torch.float64))
    torch.testing.assert_close(covariance.get_covariance_block(fixed_block, variable_block), torch.zeros((1, 1), dtype=torch.float64))
    torch.testing.assert_close(covariance.get_covariance_block(variable_block, variable_block), torch.tensor([[0.25]], dtype=torch.float64))


def test_cubic_interpolator_linear_data() -> None:
    grid = tc.Grid1D(torch.arange(5, dtype=torch.float64))
    interpolator = tc.CubicInterpolator(grid)
    torch.testing.assert_close(interpolator.evaluate(torch.tensor(2.5, dtype=torch.float64)), torch.tensor(2.5, dtype=torch.float64))


def test_cubic_interpolator_derivative_for_linear_data() -> None:
    grid = tc.Grid1D(torch.arange(6, dtype=torch.float64), x0=-1.0, spacing=0.5)
    interpolator = tc.CubicInterpolator(grid)
    value, derivative = interpolator.evaluate_with_derivative(torch.tensor(0.25, dtype=torch.float64))
    torch.testing.assert_close(value, torch.tensor(2.5, dtype=torch.float64))
    torch.testing.assert_close(derivative, torch.tensor(2.0, dtype=torch.float64))


def test_catmull_rom_kernel_and_vector_valued_grid1d() -> None:
    p0 = torch.tensor([0.0, 1.0], dtype=torch.float64)
    p1 = torch.tensor([1.0, 3.0], dtype=torch.float64)
    p2 = torch.tensor([2.0, 5.0], dtype=torch.float64)
    p3 = torch.tensor([3.0, 7.0], dtype=torch.float64)
    x = torch.tensor(0.25, dtype=torch.float64)

    torch.testing.assert_close(tc.CubicHermiteSpline(p0, p1, p2, p3, x), torch.tensor([1.25, 3.5], dtype=torch.float64))
    torch.testing.assert_close(tc.CubicHermiteSplineDerivative(p0, p1, p2, p3, x), torch.tensor([1.0, 2.0], dtype=torch.float64))

    samples = torch.stack([torch.tensor([float(i), 2.0 * i + 1.0], dtype=torch.float64) for i in range(6)])
    interpolator = tc.CubicInterpolator(tc.Grid1D(samples))
    value, derivative = interpolator.Evaluate(torch.tensor(2.25, dtype=torch.float64))

    assert interpolator.grid.data_dimension == 2
    torch.testing.assert_close(value, torch.tensor([2.25, 5.5], dtype=torch.float64))
    torch.testing.assert_close(derivative, torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_bicubic_interpolator_derivatives_for_planar_data() -> None:
    rows = torch.arange(5, dtype=torch.float64).reshape(-1, 1)
    cols = torch.arange(6, dtype=torch.float64).reshape(1, -1)
    data = 2.0 * rows + 3.0 * cols
    grid = tc.Grid2D(data, row0=-1.0, col0=2.0, row_spacing=0.5, col_spacing=2.0)
    interpolator = tc.BiCubicInterpolator(grid)

    value, drow, dcol = interpolator.evaluate_with_derivatives(
        torch.tensor(0.25, dtype=torch.float64),
        torch.tensor(7.0, dtype=torch.float64),
    )

    torch.testing.assert_close(value, torch.tensor(12.5, dtype=torch.float64))
    torch.testing.assert_close(drow, torch.tensor(4.0, dtype=torch.float64))
    torch.testing.assert_close(dcol, torch.tensor(1.5, dtype=torch.float64))


def test_bicubic_interpolator_vector_valued_samples() -> None:
    rows = torch.arange(5, dtype=torch.float64).reshape(-1, 1)
    cols = torch.arange(6, dtype=torch.float64).reshape(1, -1)
    first = 2.0 * rows + 3.0 * cols
    second = rows - 4.0 * cols
    data = torch.stack([first, second], dim=-1)
    interpolator = tc.BiCubicInterpolator(tc.Grid2D(data, row_spacing=0.5, col_spacing=2.0))

    value, drow, dcol = interpolator.Evaluate(torch.tensor(1.25, dtype=torch.float64), torch.tensor(3.0, dtype=torch.float64))

    assert interpolator.grid.data_dimension == 2
    torch.testing.assert_close(value, torch.tensor([9.5, -3.5], dtype=torch.float64))
    torch.testing.assert_close(drow, torch.tensor([4.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(dcol, torch.tensor([1.5, -2.0], dtype=torch.float64))
