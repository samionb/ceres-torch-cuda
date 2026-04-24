import torch

import ceres_torch as tc


def run() -> tuple[tc.SolverSummary, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    xs = torch.linspace(0.0, 1.0, 40, dtype=dtype)
    ys = torch.exp(0.3 + 1.7 * xs)
    m = torch.tensor([0.0], dtype=dtype)
    c = torch.tensor([0.0], dtype=dtype)
    problem = tc.Problem()
    for x, y in zip(xs, ys):
        problem.add_residual_block(
            tc.AutoDiffCostFunction(lambda m, c, x=x, y=y: y - torch.exp(m[0] * x + c[0]), [1, 1]),
            tc.HuberLoss(1.0),
            [m, c],
        )
    summary = tc.solve(tc.SolverOptions(max_num_iterations=50), problem)
    return summary, m, c


def main() -> None:
    summary, m, c = run()
    print(summary.BriefReport())
    print(f"m = {m.item():.6f}, c = {c.item():.6f}")


if __name__ == "__main__":
    main()
