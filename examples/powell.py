import math

import torch

import ceres_torch as tc


def main() -> None:
    dtype = torch.float64
    x1 = torch.tensor([3.0], dtype=dtype)
    x2 = torch.tensor([-1.0], dtype=dtype)
    x3 = torch.tensor([0.0], dtype=dtype)
    x4 = torch.tensor([1.0], dtype=dtype)

    problem = tc.Problem()
    problem.add_residual_block(tc.AutoDiffCostFunction(lambda x1, x2: x1 + 10.0 * x2, [1, 1]), None, [x1, x2])
    problem.add_residual_block(
        tc.AutoDiffCostFunction(lambda x3, x4: math.sqrt(5.0) * (x3 - x4), [1, 1]),
        None,
        [x3, x4],
    )
    problem.add_residual_block(
        tc.AutoDiffCostFunction(lambda x2, x3: (x2 - 2.0 * x3) ** 2, [1, 1]),
        None,
        [x2, x3],
    )
    problem.add_residual_block(
        tc.AutoDiffCostFunction(lambda x1, x4: math.sqrt(10.0) * (x1 - x4) ** 2, [1, 1]),
        None,
        [x1, x4],
    )

    summary = tc.solve(
        tc.SolverOptions(max_num_iterations=500, function_tolerance=1e-12, parameter_tolerance=1e-12),
        problem,
    )
    print(summary.BriefReport())
    print(f"x1={x1.item():.8f} x2={x2.item():.8f} x3={x3.item():.8f} x4={x4.item():.8f}")


if __name__ == "__main__":
    main()
