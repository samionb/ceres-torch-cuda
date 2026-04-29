from dataclasses import dataclass
from typing import Callable

import torch

import ceres_torch as tc


@dataclass(frozen=True)
class NISTRegressionProblem:
    name: str
    initial_values: tuple[tuple[float, ...], ...]
    certified: tuple[float, ...]
    certified_cost: float
    x: tuple[float, ...]
    y: tuple[float, ...]
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def exponential_adsorption_model(parameters: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return parameters[0] * (1.0 - torch.exp(-parameters[1] * x))


PROBLEMS = {
    "Misra1a": NISTRegressionProblem(
        name="Misra1a",
        initial_values=((500.0, 0.0001), (250.0, 0.0005)),
        certified=(2.3894212918e2, 5.5015643181e-4),
        certified_cost=1.2455138894e-1 / 2.0,
        x=(77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0),
        y=(10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01, 66.40, 75.47, 81.78),
        model=exponential_adsorption_model,
    ),
    "BoxBOD": NISTRegressionProblem(
        name="BoxBOD",
        initial_values=((100.0, 0.75),),
        certified=(2.1380940889e2, 5.4723748542e-1),
        certified_cost=1.1680088766e3 / 2.0,
        x=(1.0, 2.0, 3.0, 5.0, 7.0, 10.0),
        y=(109.0, 149.0, 149.0, 191.0, 213.0, 224.0),
        model=exponential_adsorption_model,
    ),
}


def solve_problem(
    problem_name: str = "Misra1a",
    *,
    start_index: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[tc.SolverSummary, torch.Tensor, NISTRegressionProblem]:
    spec = PROBLEMS[problem_name]
    x = torch.tensor(spec.x, dtype=dtype, device=device)
    y = torch.tensor(spec.y, dtype=dtype, device=device)
    parameters = torch.tensor(spec.initial_values[start_index], dtype=dtype, device=device)

    problem = tc.Problem()
    problem.AddResidualBlock(
        tc.AutoDiffCostFunction(lambda b: y - spec.model(b, x), [len(spec.certified)], y.numel()),
        None,
        [parameters],
    )
    summary = tc.solve(
        tc.SolverOptions(
            max_num_iterations=500,
            linear_solver_type=tc.LinearSolverType.DENSE_QR,
            function_tolerance=1e-14,
            gradient_tolerance=1e-14,
            parameter_tolerance=1e-14,
        ),
        problem,
    )
    return summary, parameters, spec


def run(
    problem_name: str = "Misra1a",
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[int, tuple[tc.SolverSummary, torch.Tensor, NISTRegressionProblem]]:
    spec = PROBLEMS[problem_name]
    return {
        start_index: solve_problem(problem_name, start_index=start_index, dtype=dtype, device=device)
        for start_index in range(len(spec.initial_values))
    }


def log_relative_error(estimated: torch.Tensor, certified: torch.Tensor) -> float:
    error = torch.max(torch.abs(estimated.detach().cpu() - certified) / torch.clamp(torch.abs(certified), min=1.0))
    error_value = float(error)
    if error_value == 0.0:
        return float("inf")
    return -torch.log10(torch.tensor(error_value)).item()


if __name__ == "__main__":
    for name in PROBLEMS:
        for start, (summary, parameters, spec) in run(name).items():
            certified = torch.tensor(spec.certified, dtype=torch.float64)
            print(
                name,
                "start",
                start + 1,
                summary.BriefReport(),
                parameters.detach().cpu().tolist(),
                "lre",
                f"{log_relative_error(parameters, certified):.2f}",
            )
