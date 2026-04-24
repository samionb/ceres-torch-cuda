import torch

import ceres_torch as tc


def main() -> None:
    x = torch.tensor([0.5], dtype=torch.float64)
    problem = tc.Problem()
    problem.add_residual_block(tc.AutoDiffCostFunction(lambda x: 10.0 - x, [1]), None, [x])
    summary = tc.solve(tc.SolverOptions(max_num_iterations=25), problem)
    print(summary.BriefReport())
    print(f"x = {x.item():.12f}")


if __name__ == "__main__":
    main()

