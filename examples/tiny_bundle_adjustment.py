import torch

import torch_ceres as tc


def project(camera: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    angle_axis = camera[:3]
    translation = camera[3:6]
    focal = camera[6]
    p = tc.angle_axis_rotate_point(angle_axis, point) + translation
    return focal * p[:2] / p[2].clamp_min(1e-9)


def main() -> None:
    dtype = torch.float64
    camera = torch.tensor([0.01, -0.02, 0.03, 0.1, -0.2, 3.0, 800.0], dtype=dtype)
    points = [
        torch.tensor([0.2, 0.1, 4.0], dtype=dtype),
        torch.tensor([-0.3, 0.2, 4.5], dtype=dtype),
        torch.tensor([0.1, -0.2, 5.0], dtype=dtype),
    ]
    observations = [project(camera, p).detach() for p in points]

    camera_est = camera + torch.tensor([0.05, -0.03, 0.02, 0.2, 0.1, -0.2, 20.0], dtype=dtype)
    point_est = [p + 0.05 * torch.randn_like(p) for p in points]

    problem = tc.Problem()
    for point, obs in zip(point_est, observations):
        problem.add_residual_block(
            tc.AutoDiffCostFunction(lambda cam, pt, obs=obs: project(cam, pt) - obs, [7, 3]),
            None,
            [camera_est, point],
        )
    options = tc.SolverOptions(max_num_iterations=30, linear_solver_type=tc.LinearSolverType.DENSE_QR)
    summary = tc.solve(options, problem)
    print(summary.BriefReport())
    print("camera", camera_est)


if __name__ == "__main__":
    main()

