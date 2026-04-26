import torch
import pytest

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
    assert covariance.summary.success
    assert covariance.Rank() == 1
    assert covariance.Nullity() == 0
    assert covariance.ReciprocalConditionNumber() == 1.0


def test_covariance_rank_deficiency_policy() -> None:
    x = torch.tensor([0.0], dtype=torch.float64)
    y = torch.tensor([0.0], dtype=torch.float64)
    problem = tc.Problem()
    bx = problem.add_parameter_block(x)
    by = problem.add_parameter_block(y)
    problem.add_residual_block(tc.AutoDiffCostFunction(lambda x, y: x + y, [1, 1]), None, [x, y])

    strict = tc.Covariance(tc.CovarianceOptions(algorithm_type=tc.CovarianceAlgorithmType.DENSE_SVD))
    assert not strict.compute([(bx, bx), (by, by)], problem)
    assert strict.summary.rank == 1
    assert strict.summary.nullity == 1
    assert "rank deficient" in strict.summary.message

    pseudo_inverse = tc.Covariance(
        tc.CovarianceOptions(algorithm_type=tc.CovarianceAlgorithmType.DENSE_SVD, null_space_rank=-1)
    )
    assert pseudo_inverse.compute([(bx, bx), (by, by), (bx, by)], problem)
    matrix = pseudo_inverse.get_covariance_matrix_in_tangent_space([bx, by])
    torch.testing.assert_close(matrix, torch.full((2, 2), 0.25, dtype=torch.float64), atol=1e-9, rtol=1e-9)
    assert pseudo_inverse.summary.rank == 1
    assert pseudo_inverse.summary.nullity == 1


def test_covariance_options_validate() -> None:
    with pytest.raises(ValueError, match="null_space_rank"):
        tc.CovarianceOptions(null_space_rank=-2).validate()
    with pytest.raises(ValueError, match="min_reciprocal_condition_number"):
        tc.CovarianceOptions(min_reciprocal_condition_number=-1.0).validate()


def test_covariance_near_singular_uses_eigenvalue_ratio_policy() -> None:
    x = torch.zeros(2, dtype=torch.float64)
    problem = tc.Problem()
    block = problem.add_parameter_block(x)
    A = torch.tensor([[1.0, 0.0], [0.0, 1e-4]], dtype=torch.float64)
    problem.add_residual_block(tc.NormalPrior(A, torch.zeros(2, dtype=torch.float64)), None, [x])

    strict = tc.Covariance(
        tc.CovarianceOptions(
            algorithm_type=tc.CovarianceAlgorithmType.DENSE_SVD,
            min_reciprocal_condition_number=1e-6,
        )
    )
    assert not strict.compute([(block, block)], problem)
    assert strict.summary.rank == 2
    assert strict.summary.reciprocal_condition_number < 1e-6

    truncated = tc.Covariance(
        tc.CovarianceOptions(
            algorithm_type=tc.CovarianceAlgorithmType.DENSE_SVD,
            min_reciprocal_condition_number=1e-6,
            null_space_rank=-1,
        )
    )
    assert truncated.compute([(block, block)], problem)
    cov = truncated.get_covariance_block(block, block)
    torch.testing.assert_close(cov, torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64)))
    assert truncated.summary.rank == 1
    assert truncated.summary.nullity == 1


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


def test_grid1d_ceres_flat_layouts_and_clamped_window() -> None:
    interleaved = torch.tensor([1, 5, 2, 6, 3, 7], dtype=torch.float64)
    stacked = torch.tensor([1, 2, 3, 5, 6, 7], dtype=torch.float64)

    interleaved_grid = tc.Grid1D(interleaved, begin=10, end=13, data_dimension=2, interleaved=True)
    stacked_grid = tc.Grid1D(stacked, begin=10, end=13, data_dimension=2, interleaved=False)

    torch.testing.assert_close(interleaved_grid.get(11), torch.tensor([2.0, 6.0], dtype=torch.float64))
    torch.testing.assert_close(stacked_grid.get(11), torch.tensor([2.0, 6.0], dtype=torch.float64))
    torch.testing.assert_close(stacked_grid.get(9), torch.tensor([1.0, 5.0], dtype=torch.float64))
    torch.testing.assert_close(stacked_grid.get(99), torch.tensor([3.0, 7.0], dtype=torch.float64))
    assert interleaved_grid.data_dimension == 2
    assert interleaved_grid.DATA_DIMENSION == 2
    torch.testing.assert_close(interleaved_grid.GetValue(11), torch.tensor([2.0, 6.0], dtype=torch.float64))


@pytest.mark.parametrize(
    "coefficients",
    [
        (0.0, 0.0, 0.0, 0.5),
        (0.0, 0.0, 1.0, 0.5),
        (0.0, 0.4, 1.0, 0.5),
    ],
)
@pytest.mark.parametrize("data_dimension", [1, 2, 3])
def test_cubic_interpolator_matches_ceres_polynomial_cases(
    coefficients: tuple[float, float, float, float],
    data_dimension: int,
) -> None:
    a, b, c, d = coefficients
    samples = []
    for x in range(10):
        base = a * x**3 + b * x**2 + c * x + d
        samples.extend((dim * dim + 1.0) * base for dim in range(data_dimension))
    interpolator = tc.CubicInterpolator(
        tc.Grid1D(torch.tensor(samples, dtype=torch.float64), end=10, data_dimension=data_dimension)
    )

    for x in torch.linspace(1.0, 8.0, 25, dtype=torch.float64):
        value, derivative = interpolator.Evaluate(x)
        expected_value = torch.tensor(
            [(dim * dim + 1.0) * (a * x.item() ** 3 + b * x.item() ** 2 + c * x.item() + d) for dim in range(data_dimension)],
            dtype=torch.float64,
        )
        expected_derivative = torch.tensor(
            [(dim * dim + 1.0) * (3.0 * a * x.item() ** 2 + 2.0 * b * x.item() + c) for dim in range(data_dimension)],
            dtype=torch.float64,
        )
        if data_dimension == 1:
            expected_value = expected_value.reshape(())
            expected_derivative = expected_derivative.reshape(())
        torch.testing.assert_close(value, expected_value, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(derivative, expected_derivative, atol=1e-12, rtol=1e-12)


def test_cubic_interpolator_autograd_matches_reported_derivative() -> None:
    values = torch.tensor([1.0, 2.0, 2.0, 5.0, 3.0, 9.0, 2.0, 7.0], dtype=torch.float64)
    interpolator = tc.CubicInterpolator(tc.Grid1D(values, end=4, data_dimension=2))
    x = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)

    value, derivative = interpolator.Evaluate(x)
    grad0 = torch.autograd.grad(value[0], x, retain_graph=True)[0]
    grad1 = torch.autograd.grad(value[1], x)[0]

    torch.testing.assert_close(torch.stack([grad0, grad1]), derivative, atol=1e-12, rtol=1e-12)


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


def test_grid2d_ceres_flat_storage_layouts() -> None:
    row_major_interleaved = torch.tensor(
        [1, 4, 2, 8, 3, 12, 2, 8, 3, 12, 4, 16],
        dtype=torch.float64,
    )
    row_major_stacked = torch.tensor(
        [1, 2, 3, 2, 3, 4, 4, 8, 12, 8, 12, 16],
        dtype=torch.float64,
    )
    col_major_interleaved = torch.tensor(
        [1, 4, 2, 8, 2, 8, 3, 12, 3, 12, 4, 16],
        dtype=torch.float64,
    )
    col_major_stacked = torch.tensor(
        [1, 2, 2, 3, 3, 4, 4, 8, 8, 12, 12, 16],
        dtype=torch.float64,
    )

    grids = [
        tc.Grid2D(row_major_interleaved, row_end=2, col_end=3, data_dimension=2, row_major=True, interleaved=True),
        tc.Grid2D(row_major_stacked, row_end=2, col_end=3, data_dimension=2, row_major=True, interleaved=False),
        tc.Grid2D(col_major_interleaved, row_end=2, col_end=3, data_dimension=2, row_major=False, interleaved=True),
        tc.Grid2D(col_major_stacked, row_end=2, col_end=3, data_dimension=2, row_major=False, interleaved=False),
    ]

    for grid in grids:
        for row in range(2):
            for col in range(3):
                expected = torch.tensor([row + col + 1.0, 4.0 * (row + col + 1.0)], dtype=torch.float64)
                torch.testing.assert_close(grid.get(row, col), expected)
        torch.testing.assert_close(grid.get(-10, -10), torch.tensor([1.0, 4.0], dtype=torch.float64))
        torch.testing.assert_close(grid.get(10, 10), torch.tensor([4.0, 16.0], dtype=torch.float64))
        assert grid.DATA_DIMENSION == 2
        torch.testing.assert_close(grid.GetValue(0, 1), torch.tensor([2.0, 8.0], dtype=torch.float64))


def test_grid2d_ceres_one_dimensional_out_of_bounds_clamping() -> None:
    data = torch.tensor([1, 2, 3, 2, 3, 4], dtype=torch.float64)
    grid = tc.Grid2D(data, row_end=2, col_end=3, data_dimension=1)

    expected = {
        (-1, -1): 1.0,
        (-1, 0): 1.0,
        (-1, 1): 2.0,
        (-1, 2): 3.0,
        (-1, 3): 3.0,
        (0, 3): 3.0,
        (1, 3): 4.0,
        (2, 3): 4.0,
        (2, 2): 4.0,
        (2, 1): 3.0,
        (2, 0): 2.0,
        (2, -1): 2.0,
        (1, -1): 2.0,
        (0, -1): 1.0,
    }

    for (row, col), value in expected.items():
        torch.testing.assert_close(grid.GetValue(row, col), torch.tensor(value, dtype=torch.float64))


def test_bicubic_interpolator_matches_shaped_grid_with_flat_stacked_data() -> None:
    rows = torch.arange(5, dtype=torch.float64).reshape(-1, 1)
    cols = torch.arange(6, dtype=torch.float64).reshape(1, -1)
    first = 2.0 * rows + 3.0 * cols
    second = rows - 4.0 * cols
    shaped = torch.stack([first, second], dim=-1)
    flat_stacked = torch.cat([first.reshape(-1), second.reshape(-1)])

    shaped_interpolator = tc.BiCubicInterpolator(tc.Grid2D(shaped, row_spacing=0.5, col_spacing=2.0))
    flat_interpolator = tc.BiCubicInterpolator(
        tc.Grid2D(
            flat_stacked,
            row_spacing=0.5,
            col_spacing=2.0,
            row_end=5,
            col_end=6,
            data_dimension=2,
            row_major=True,
            interleaved=False,
        )
    )

    shaped_value = shaped_interpolator.Evaluate(torch.tensor(1.25, dtype=torch.float64), torch.tensor(3.0, dtype=torch.float64))
    flat_value = flat_interpolator.Evaluate(torch.tensor(1.25, dtype=torch.float64), torch.tensor(3.0, dtype=torch.float64))

    for flat_tensor, shaped_tensor in zip(flat_value, shaped_value):
        torch.testing.assert_close(flat_tensor, shaped_tensor)


@pytest.mark.parametrize(
    "coefficients",
    [
        torch.zeros((3, 3), dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.0], [0.1, 0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.1, 0.2], [0.1, 0.0, 0.0], [0.2, 0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.1, 0.2], [0.1, 0.3, 0.0], [0.2, 0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.3, 0.1, 0.2], [0.1, 0.0, 0.0], [0.2, 0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.3, -0.4, 0.2], [-0.4, 0.0, 0.0], [0.2, 0.0, 1.0]], dtype=torch.float64),
    ],
)
@pytest.mark.parametrize("data_dimension", [1, 2, 3])
def test_bicubic_interpolator_matches_ceres_quadratic_cases(
    coefficients: torch.Tensor,
    data_dimension: int,
) -> None:
    def evaluate(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        x = torch.stack([row, col, row.new_tensor(1.0)])
        return x @ coefficients @ x

    rows = []
    for row in range(10):
        row_values = []
        for col in range(10):
            base = evaluate(torch.tensor(float(row), dtype=torch.float64), torch.tensor(float(col), dtype=torch.float64))
            row_values.append(torch.stack([(dim * dim + 1.0) * base for dim in range(data_dimension)]))
        rows.append(torch.stack(row_values))
    data = torch.stack(rows)
    if data_dimension == 1:
        data = data[..., 0]
    interpolator = tc.BiCubicInterpolator(tc.Grid2D(data))

    for row in torch.linspace(1.0, 8.0, 9, dtype=torch.float64):
        for col in torch.linspace(1.0, 8.0, 9, dtype=torch.float64):
            value, drow, dcol = interpolator.Evaluate(row, col)
            x = torch.stack([row, col, row.new_tensor(1.0)])
            base = x @ coefficients @ x
            base_drow = (coefficients[0, :] + coefficients[:, 0]) @ x
            base_dcol = (coefficients[1, :] + coefficients[:, 1]) @ x
            expected_value = torch.stack([(dim * dim + 1.0) * base for dim in range(data_dimension)])
            expected_drow = torch.stack([(dim * dim + 1.0) * base_drow for dim in range(data_dimension)])
            expected_dcol = torch.stack([(dim * dim + 1.0) * base_dcol for dim in range(data_dimension)])
            if data_dimension == 1:
                expected_value = expected_value.reshape(())
                expected_drow = expected_drow.reshape(())
                expected_dcol = expected_dcol.reshape(())
            torch.testing.assert_close(value, expected_value, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(drow, expected_drow, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(dcol, expected_dcol, atol=1e-12, rtol=1e-12)


def test_bicubic_interpolator_autograd_matches_reported_derivatives() -> None:
    data = torch.tensor(
        [
            [[1.0, 5.0], [2.0, 10.0], [2.0, 6.0], [3.0, 5.0]],
            [[1.0, 2.0], [2.0, 2.0], [2.0, 2.0], [3.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    interpolator = tc.BiCubicInterpolator(tc.Grid2D(data))
    row = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    col = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)

    value, drow, dcol = interpolator.Evaluate(row, col)
    grad_row0, grad_col0 = torch.autograd.grad(value[0], (row, col), retain_graph=True)
    grad_row1, grad_col1 = torch.autograd.grad(value[1], (row, col))

    torch.testing.assert_close(torch.stack([grad_row0, grad_row1]), drow, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.stack([grad_col0, grad_col1]), dcol, atol=1e-12, rtol=1e-12)
