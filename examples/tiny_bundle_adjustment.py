import torch

import ceres_torch as tc


def project(camera: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    angle_axis = camera[:3]
    translation = camera[3:6]
    focal = camera[6]
    p = tc.angle_axis_rotate_point(angle_axis, point) + translation
    return focal * p[:2] / p[2].clamp_min(1e-9)


def run(
    *,
    dtype: torch.dtype = torch.float64,
    max_num_iterations: int = 30,
) -> tuple[tc.SolverSummary, torch.Tensor, torch.Tensor, torch.Tensor]:
    camera = torch.tensor([0.01, -0.02, 0.03, 0.1, -0.2, 3.0, 800.0], dtype=dtype)
    points = [
        torch.tensor([0.2, 0.1, 4.0], dtype=dtype),
        torch.tensor([-0.3, 0.2, 4.5], dtype=dtype),
        torch.tensor([0.1, -0.2, 5.0], dtype=dtype),
    ]
    observations = [project(camera, p).detach() for p in points]

    camera_est = camera + torch.tensor([0.05, -0.03, 0.02, 0.2, 0.1, -0.2, 20.0], dtype=dtype)
    offsets = [
        torch.tensor([0.05, -0.02, 0.03], dtype=dtype),
        torch.tensor([-0.03, 0.04, -0.02], dtype=dtype),
        torch.tensor([0.02, -0.04, 0.05], dtype=dtype),
    ]
    point_est = [p + offset for p, offset in zip(points, offsets)]
    initial_reprojection_errors = torch.stack([project(camera_est, p) - obs for p, obs in zip(point_est, observations)])

    problem = tc.Problem()
    for point, obs in zip(point_est, observations):
        problem.add_residual_block(
            tc.AutoDiffCostFunction(lambda cam, pt, obs=obs: project(cam, pt) - obs, [7, 3]),
            None,
            [camera_est, point],
        )
    options = tc.SolverOptions(max_num_iterations=max_num_iterations, linear_solver_type=tc.LinearSolverType.DENSE_QR)
    summary = tc.solve(options, problem)
    final_reprojection_errors = torch.stack([project(camera_est, p) - obs for p, obs in zip(point_est, observations)])
    return summary, camera_est, initial_reprojection_errors, final_reprojection_errors


def main() -> None:
    summary, camera_est, _initial_errors, _final_errors = run()
    print(summary.BriefReport())
    print("camera", camera_est)


if __name__ == "__main__":
    main()
