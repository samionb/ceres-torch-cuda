import torch

import ceres_torch as tc


def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def run() -> tuple[tc.GradientProblemSolverSummary, torch.Tensor]:
    x = torch.tensor([-1.2, 1.0], dtype=torch.float64)
    problem = tc.GradientProblem(tc.NumericDiffFirstOrderFunction(rosenbrock), tc.EuclideanManifold(2))
    options = tc.GradientProblemSolverOptions(
        line_search_direction_type=tc.LineSearchDirectionType.BFGS,
        max_num_iterations=200,
        gradient_tolerance=1e-8,
    )
    summary = tc.gradient_solve(options, problem, x)
    return summary, x


def main() -> None:
    summary, x = run()
    print(summary.BriefReport())
    print(f"x={x[0].item():.8f} y={x[1].item():.8f}")


if __name__ == "__main__":
    main()
