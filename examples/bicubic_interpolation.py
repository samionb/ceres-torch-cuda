"""Ceres bicubic_interpolation.cc port."""

from __future__ import annotations

import torch

import ceres_torch as ct


GRID_ROWS_HALF = 9
GRID_COLS_HALF = 11


def surface_value(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * x - y * x + y * y


def build_interpolator(dtype: torch.dtype = torch.float64) -> ct.BiCubicInterpolator:
    values = []
    for row in range(-GRID_ROWS_HALF, GRID_ROWS_HALF + 1):
        y = torch.tensor(float(row), dtype=dtype)
        for col in range(-GRID_COLS_HALF, GRID_COLS_HALF + 1):
            x = torch.tensor(float(col), dtype=dtype)
            values.append(surface_value(x, y))

    grid = ct.Grid2D(
        torch.stack(values),
        row_begin=-GRID_ROWS_HALF,
        row_end=GRID_ROWS_HALF + 1,
        col_begin=-GRID_COLS_HALF,
        col_end=GRID_COLS_HALF + 1,
        data_dimension=1,
        row_major=True,
        interleaved=True,
    )
    return ct.BiCubicInterpolator(grid)


def run(dtype: torch.dtype = torch.float64):
    interpolator = build_interpolator(dtype=dtype)
    true_shift = torch.tensor([1.234, 2.345], dtype=dtype)
    estimated_shift = torch.tensor([3.1415, 1.337], dtype=dtype)
    base_points = torch.tensor(
        [
            [-2.0, -3.0],
            [-2.0, 3.0],
            [2.0, 3.0],
            [2.0, -3.0],
        ],
        dtype=dtype,
    )

    problem = ct.Problem()
    problem.AddParameterBlock(estimated_shift)

    for point in base_points:
        target_point = point + true_shift
        target_value = surface_value(target_point[0], target_point[1]).detach()

        def residual(shift, p=point, value=target_value):
            sample = p + shift
            interpolated = interpolator.evaluate(sample[1], sample[0])
            return (interpolated - value).reshape(1)

        problem.AddResidualBlock(
            ct.AutoDiffCostFunction(residual, [2], 1),
            None,
            [estimated_shift],
        )

    options = ct.SolverOptions(
        max_num_iterations=50,
        function_tolerance=1e-14,
        gradient_tolerance=1e-14,
        parameter_tolerance=1e-14,
    )
    summary = ct.solve(options, problem)
    return summary, estimated_shift, true_shift


def main() -> None:
    summary, estimated_shift, true_shift = run()
    print(summary.BriefReport())
    print(f"estimated_shift={estimated_shift.tolist()}")
    print(f"true_shift={true_shift.tolist()}")


if __name__ == "__main__":
    main()
