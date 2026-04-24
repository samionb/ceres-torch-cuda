import torch

import ceres_torch as tc


def run() -> tuple[tc.SolverSummary, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    true_center = torch.tensor([1.5, -2.0], dtype=dtype)
    true_radius = torch.tensor([3.0], dtype=dtype)
    angles = torch.linspace(0.0, 2.0 * torch.pi, 16, dtype=dtype)[:-1]
    points = torch.stack(
        [
            true_center[0] + true_radius[0] * torch.cos(angles),
            true_center[1] + true_radius[0] * torch.sin(angles),
        ],
        dim=1,
    )

    center = torch.tensor([0.0, 0.0], dtype=dtype)
    radius = torch.tensor([1.0], dtype=dtype)
    problem = tc.Problem()
    problem.AddParameterBlock(center)
    problem.AddParameterBlock(radius)
    problem.SetParameterLowerBound(radius, 0, 0.0)
    for point in points:
        problem.AddResidualBlock(
            tc.AutoDiffCostFunction(lambda c, r, p=point: torch.linalg.norm(p - c) - r[0], [2, 1], 1),
            None,
            [center, radius],
        )

    summary = tc.solve(tc.SolverOptions(max_num_iterations=100, gradient_tolerance=1e-12), problem)
    return summary, center, radius


def main() -> None:
    summary, center, radius = run()
    print(summary.BriefReport())
    print(f"center={center.tolist()} radius={radius.item():.8f}")


if __name__ == "__main__":
    main()
