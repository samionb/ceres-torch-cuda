from dataclasses import dataclass
from typing import Callable

import torch

import ceres_torch as ct


@dataclass(frozen=True)
class MGHProblem:
    name: str
    initial: tuple[float, ...]
    residual: Callable[[torch.Tensor], torch.Tensor]
    expected: tuple[float, ...]
    lower: tuple[float, ...] | None = None
    upper: tuple[float, ...] | None = None


def rosenbrock_residual(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]])


def beale_residual(x: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            1.5 - x[0] * (1.0 - x[1]),
            2.25 - x[0] * (1.0 - x[1] * x[1]),
            2.625 - x[0] * (1.0 - x[1] * x[1] * x[1]),
        ]
    )


PROBLEMS = {
    "rosenbrock": MGHProblem(
        name="rosenbrock",
        initial=(-1.2, 1.0),
        residual=rosenbrock_residual,
        expected=(1.0, 1.0),
    ),
    "beale": MGHProblem(
        name="beale",
        initial=(1.0, 1.0),
        residual=beale_residual,
        expected=(3.0, 0.5),
        lower=(0.6, 0.5),
        upper=(10.0, 100.0),
    ),
}


def solve_problem(
    problem_name: str,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    max_num_iterations: int = 200,
) -> tuple[ct.SolverSummary, torch.Tensor, MGHProblem]:
    spec = PROBLEMS[problem_name]
    x = torch.tensor(spec.initial, dtype=dtype, device=device)
    problem = ct.Problem()
    problem.AddParameterBlock(x)
    if spec.lower is not None or spec.upper is not None:
        lower = None if spec.lower is None else torch.tensor(spec.lower, dtype=dtype, device=device)
        upper = None if spec.upper is None else torch.tensor(spec.upper, dtype=dtype, device=device)
        problem.SetBounds(x, lower=lower, upper=upper)
    problem.AddResidualBlock(ct.AutoDiffCostFunction(spec.residual, [len(spec.initial)]), None, [x])
    summary = ct.solve(
        ct.SolverOptions(
            max_num_iterations=max_num_iterations,
            function_tolerance=1e-12,
            gradient_tolerance=1e-12,
            parameter_tolerance=1e-12,
        ),
        problem,
    )
    return summary, x, spec


def run(
    problem_name: str = "all",
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[str, tuple[ct.SolverSummary, torch.Tensor, MGHProblem]]:
    names = list(PROBLEMS) if problem_name == "all" else [problem_name]
    return {name: solve_problem(name, dtype=dtype, device=device) for name in names}


if __name__ == "__main__":
    for name, (summary, x, spec) in run().items():
        print(name, summary.BriefReport(), x.detach().cpu().tolist(), "target", list(spec.expected))
