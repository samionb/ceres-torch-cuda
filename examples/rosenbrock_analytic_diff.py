import torch

import ceres_torch as tc


class RosenbrockAnalytic(tc.FirstOrderFunction):
    def value_and_gradient(self, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = parameters
        value = (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2
        gradient = torch.stack(
            [
                -2.0 * (1.0 - x) - 400.0 * x * (y - x * x),
                200.0 * (y - x * x),
            ]
        )
        return value.detach(), gradient.detach()


def run() -> tuple[tc.GradientProblemSolverSummary, torch.Tensor]:
    x = torch.tensor([-1.2, 1.0], dtype=torch.float64)
    problem = tc.GradientProblem(RosenbrockAnalytic(), tc.EuclideanManifold(2))
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
