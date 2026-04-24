import torch

import ceres_torch as tc


def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def main() -> None:
    x = torch.tensor([-1.2, 1.0], dtype=torch.float64)
    problem = tc.GradientProblem.from_callable(rosenbrock, size=2)
    options = tc.GradientProblemSolverOptions(line_search_direction_type=tc.LineSearchDirectionType.BFGS)
    summary = tc.gradient_solve(options, problem, x)
    print(summary.BriefReport())
    print(f"x={x[0].item():.8f} y={x[1].item():.8f}")


if __name__ == "__main__":
    main()

