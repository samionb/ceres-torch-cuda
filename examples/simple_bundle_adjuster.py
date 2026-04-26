from dataclasses import dataclass
from pathlib import Path

import torch

import ceres_torch as tc


@dataclass
class BALProblem:
    camera_indices: list[int]
    point_indices: list[int]
    observations: torch.Tensor
    cameras: torch.Tensor
    points: torch.Tensor

    @property
    def num_cameras(self) -> int:
        return int(self.cameras.shape[0])

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def num_observations(self) -> int:
        return int(self.observations.shape[0])

    @classmethod
    def from_file(cls, path: str | Path, *, dtype: torch.dtype = torch.float64) -> "BALProblem":
        return cls.from_text(Path(path).read_text(), dtype=dtype)

    @classmethod
    def from_text(cls, text: str, *, dtype: torch.dtype = torch.float64) -> "BALProblem":
        tokens = text.split()
        cursor = 0

        def next_int() -> int:
            nonlocal cursor
            value = int(tokens[cursor])
            cursor += 1
            return value

        def next_float() -> float:
            nonlocal cursor
            value = float(tokens[cursor])
            cursor += 1
            return value

        num_cameras = next_int()
        num_points = next_int()
        num_observations = next_int()
        camera_indices: list[int] = []
        point_indices: list[int] = []
        observations: list[tuple[float, float]] = []
        for _ in range(num_observations):
            camera_indices.append(next_int())
            point_indices.append(next_int())
            observations.append((next_float(), next_float()))
        cameras = torch.tensor([next_float() for _ in range(9 * num_cameras)], dtype=dtype).reshape(num_cameras, 9)
        points = torch.tensor([next_float() for _ in range(3 * num_points)], dtype=dtype).reshape(num_points, 3)
        if cursor != len(tokens):
            raise ValueError("BAL input contains trailing values")
        return cls(
            camera_indices=camera_indices,
            point_indices=point_indices,
            observations=torch.tensor(observations, dtype=dtype),
            cameras=cameras,
            points=points,
        )


def snavely_reprojection(camera: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    p = tc.angle_axis_rotate_point(camera[:3], point) + camera[3:6]
    xp = -p[0] / p[2]
    yp = -p[1] / p[2]
    r2 = xp * xp + yp * yp
    distortion = 1.0 + r2 * (camera[7] + camera[8] * r2)
    return camera[6] * distortion * torch.stack([xp, yp])


def build_problem(
    bal_problem: BALProblem,
    *,
    fix_cameras: bool = False,
) -> tuple[tc.Problem, list[torch.Tensor], list[torch.Tensor]]:
    problem = tc.Problem()
    camera_blocks = [bal_problem.cameras[i] for i in range(bal_problem.num_cameras)]
    point_blocks = [bal_problem.points[i] for i in range(bal_problem.num_points)]
    for camera in camera_blocks:
        problem.AddParameterBlock(camera)
        if fix_cameras:
            problem.SetParameterBlockConstant(camera)
    for point in point_blocks:
        problem.AddParameterBlock(point)
        problem.SetParameterBlockOrderingGroup(point, 0)

    for camera_index, point_index, observation in zip(
        bal_problem.camera_indices,
        bal_problem.point_indices,
        bal_problem.observations,
    ):
        problem.AddResidualBlock(
            tc.AutoDiffCostFunction(
                lambda camera, point, observation=observation: snavely_reprojection(camera, point) - observation,
                [9, 3],
                2,
            ),
            None,
            [camera_blocks[camera_index], point_blocks[point_index]],
        )
    return problem, camera_blocks, point_blocks


def run(
    path: str | Path | None = None,
    *,
    bal_problem: BALProblem | None = None,
    fix_cameras: bool = False,
    max_num_iterations: int = 50,
) -> tuple[tc.SolverSummary, torch.Tensor, torch.Tensor, BALProblem]:
    if bal_problem is None:
        if path is None:
            text, _true_cameras, _true_points = make_tiny_bal_problem()
            bal_problem = BALProblem.from_text(text)
        else:
            bal_problem = BALProblem.from_file(path)
    problem, _camera_blocks, _point_blocks = build_problem(bal_problem, fix_cameras=fix_cameras)
    summary = tc.solve(
        tc.SolverOptions(
            max_num_iterations=max_num_iterations,
            linear_solver_type=tc.LinearSolverType.DENSE_SCHUR,
            gradient_tolerance=1e-12,
            function_tolerance=1e-12,
            parameter_tolerance=1e-12,
        ),
        problem,
    )
    return summary, bal_problem.cameras, bal_problem.points, bal_problem


def make_tiny_bal_problem() -> tuple[str, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    true_cameras = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0],
            [0.0, 0.02, 0.0, -0.4, 0.0, 0.05, 800.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    true_points = torch.tensor(
        [
            [0.2, -0.1, 4.0],
            [-0.3, 0.15, 4.5],
            [0.1, 0.2, 5.0],
        ],
        dtype=dtype,
    )
    initial_points = true_points + torch.tensor(
        [
            [0.04, -0.03, 0.15],
            [-0.05, 0.02, -0.12],
            [0.03, 0.04, 0.10],
        ],
        dtype=dtype,
    )
    observation_rows = []
    for camera_index, camera in enumerate(true_cameras):
        for point_index, point in enumerate(true_points):
            observation = snavely_reprojection(camera, point)
            observation_rows.append((camera_index, point_index, float(observation[0]), float(observation[1])))
    lines = [f"{true_cameras.shape[0]} {true_points.shape[0]} {len(observation_rows)}"]
    lines.extend(f"{ci} {pi} {x:.17g} {y:.17g}" for ci, pi, x, y in observation_rows)
    lines.extend(f"{value:.17g}" for value in true_cameras.reshape(-1).tolist())
    lines.extend(f"{value:.17g}" for value in initial_points.reshape(-1).tolist())
    return "\n".join(lines) + "\n", true_cameras, true_points


def main() -> None:
    summary, cameras, points, _bal = run(fix_cameras=True)
    print(summary.BriefReport())
    print(f"cameras={cameras.tolist()}")
    print(f"points={points.tolist()}")


if __name__ == "__main__":
    main()
