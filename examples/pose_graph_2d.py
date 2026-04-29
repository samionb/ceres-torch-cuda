import torch

import ceres_torch as tc


def normalize_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def relative_pose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ca = torch.cos(a[2])
    sa = torch.sin(a[2])
    delta = b[:2] - a[:2]
    local = torch.stack([ca * delta[0] + sa * delta[1], -sa * delta[0] + ca * delta[1]])
    return torch.cat([local, normalize_angle(b[2] - a[2]).reshape(1)])


def run(
    *,
    dtype: torch.dtype = torch.float64,
    max_num_iterations: int = 50,
) -> tuple[tc.SolverSummary, torch.Tensor, torch.Tensor, torch.Tensor]:
    true_poses = [
        torch.tensor([0.0, 0.0, 0.0], dtype=dtype),
        torch.tensor([1.0, 0.0, 0.1], dtype=dtype),
        torch.tensor([2.0, 0.2, 0.05], dtype=dtype),
        torch.tensor([3.0, 0.1, 0.0], dtype=dtype),
    ]
    estimates = [pose + torch.tensor([0.05, -0.03, 0.02], dtype=dtype) for pose in true_poses]
    initial_estimates = torch.stack([pose.clone() for pose in estimates])
    measurements = [(i, i + 1, relative_pose(true_poses[i], true_poses[i + 1]).detach()) for i in range(len(true_poses) - 1)]

    problem = tc.Problem()
    problem.SetParameterBlockConstant(estimates[0])
    for i, j, measurement in measurements:
        problem.AddResidualBlock(
            tc.AutoDiffCostFunction(lambda a, b, m=measurement: relative_pose(a, b) - m, [3, 3], 3),
            None,
            [estimates[i], estimates[j]],
        )

    summary = tc.solve(
        tc.SolverOptions(max_num_iterations=max_num_iterations, linear_solver_type=tc.LinearSolverType.DENSE_QR),
        problem,
    )
    return summary, initial_estimates, torch.stack(estimates), torch.stack(true_poses)


def main() -> None:
    summary, _initial, estimates, _true = run()
    print(summary.BriefReport())
    for i, pose in enumerate(estimates):
        print(f"pose {i}: {pose.tolist()}")


if __name__ == "__main__":
    main()
