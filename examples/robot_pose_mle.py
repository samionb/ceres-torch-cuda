import math

import torch

import ceres_torch as tc


def simulate_robot(
    *,
    corridor_length: float,
    pose_separation: float,
    odometry_stddev: float,
    range_stddev: float,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    num_steps = int(math.ceil(corridor_length / pose_separation))
    true_odometry = []
    observed_odometry = []
    observed_ranges = []
    robot_location = 0.0
    for _ in range(num_steps):
        actual_odometry = min(pose_separation, corridor_length - robot_location)
        robot_location += actual_odometry
        actual_range = corridor_length - robot_location
        odometry_noise = odometry_stddev * torch.randn((), generator=generator, dtype=dtype).item()
        range_noise = range_stddev * torch.randn((), generator=generator, dtype=dtype).item()
        true_odometry.append(actual_odometry)
        observed_odometry.append(actual_odometry + odometry_noise)
        observed_ranges.append(actual_range + range_noise)
    return (
        torch.tensor(true_odometry, dtype=dtype),
        torch.tensor(observed_odometry, dtype=dtype),
        torch.tensor(observed_ranges, dtype=dtype),
    )


def range_errors(odometry: torch.Tensor, ranges: torch.Tensor, corridor_length: float) -> torch.Tensor:
    return torch.cumsum(odometry.reshape(-1), dim=0) + ranges.reshape(-1) - corridor_length


def run(
    *,
    corridor_length: float = 30.0,
    pose_separation: float = 0.5,
    odometry_stddev: float = 0.1,
    range_stddev: float = 0.01,
    seed: int = 0,
    max_num_iterations: int = 50,
) -> tuple[tc.SolverSummary, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    _true_odometry, observed_odometry, observed_ranges = simulate_robot(
        corridor_length=corridor_length,
        pose_separation=pose_separation,
        odometry_stddev=odometry_stddev,
        range_stddev=range_stddev,
        dtype=dtype,
        seed=seed,
    )
    initial_odometry = observed_odometry.clone()
    odometry = [torch.tensor([float(value)], dtype=dtype) for value in observed_odometry]

    problem = tc.Problem()
    for pose_index, range_reading in enumerate(observed_ranges):
        involved_blocks = odometry[: pose_index + 1]

        def range_residual(*relative_poses: torch.Tensor, reading=range_reading) -> torch.Tensor:
            global_pose = sum(relative_pose[0] for relative_pose in relative_poses)
            return ((global_pose + reading - corridor_length) / range_stddev).reshape(1)

        problem.AddResidualBlock(
            tc.DynamicAutoDiffCostFunction(range_residual, [1] * len(involved_blocks), 1),
            None,
            involved_blocks,
        )

        observed = observed_odometry[pose_index]

        def odometry_residual(relative_pose: torch.Tensor, mean=observed) -> torch.Tensor:
            return ((relative_pose[0] - mean) / odometry_stddev).reshape(1)

        problem.AddResidualBlock(
            tc.AutoDiffCostFunction(odometry_residual, [1], 1),
            None,
            [odometry[pose_index]],
        )

    summary = tc.solve(
        tc.SolverOptions(
            max_num_iterations=max_num_iterations,
            gradient_tolerance=1e-12,
            function_tolerance=1e-12,
            parameter_tolerance=1e-12,
        ),
        problem,
    )
    final_odometry = torch.cat([value.detach().reshape(1) for value in odometry])
    initial_range_errors = range_errors(initial_odometry, observed_ranges, corridor_length)
    final_range_errors = range_errors(final_odometry, observed_ranges, corridor_length)
    return summary, initial_odometry, final_odometry, observed_ranges, initial_range_errors, final_range_errors


def main() -> None:
    summary, initial_odometry, final_odometry, _ranges, initial_errors, final_errors = run()
    print(summary.BriefReport())
    print(f"initial_range_rmse={torch.sqrt(torch.mean(initial_errors ** 2)).item():.8f}")
    print(f"final_range_rmse={torch.sqrt(torch.mean(final_errors ** 2)).item():.8f}")
    print(f"initial_total_odometry={initial_odometry.sum().item():.8f}")
    print(f"final_total_odometry={final_odometry.sum().item():.8f}")


if __name__ == "__main__":
    main()
