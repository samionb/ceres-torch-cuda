import torch

import ceres_torch as tc


def run() -> tuple[tc.SolverSummary, torch.Tensor]:
    x = torch.tensor([0.5], dtype=torch.float64)
    problem = tc.Problem()
    problem.AddResidualBlock(
        tc.AnalyticCostFunction(
            lambda x: 10.0 - x,
            lambda x: [torch.tensor([[-1.0]], dtype=x.dtype, device=x.device)],
            [1],
            1,
        ),
        None,
        [x],
    )
    summary = tc.solve(tc.SolverOptions(max_num_iterations=25, gradient_tolerance=1e-12), problem)
    return summary, x


def main() -> None:
    summary, x = run()
    print(summary.BriefReport())
    print(f"x = {x.item():.12f}")


if __name__ == "__main__":
    main()
