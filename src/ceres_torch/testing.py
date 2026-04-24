from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import SolverSummary, TerminationType


def assert_close(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float = 1e-7, atol: float = 1e-9) -> None:
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def finite_difference_jacobian(fun, x: torch.Tensor, step: float = 1e-6) -> torch.Tensor:
    x = x.detach().clone()
    base = fun(x).reshape(-1)
    columns = []
    for i in range(x.numel()):
        xp = x.clone().reshape(-1)
        xm = x.clone().reshape(-1)
        h = step * max(float(abs(x.reshape(-1)[i]).cpu()), 1.0)
        xp[i] += h
        xm[i] -= h
        columns.append(((fun(xp.reshape_as(x)) - fun(xm.reshape_as(x))).reshape(-1) / (2.0 * h)))
    return torch.stack(columns, dim=1) if columns else base.new_zeros((base.numel(), 0))


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def test_devices(include_cuda: bool = True) -> list[torch.device]:
    devices = [torch.device("cpu")]
    if include_cuda and cuda_available():
        devices.append(torch.device("cuda"))
    return devices


@dataclass(frozen=True)
class GoldenSolverResult:
    initial_cost: float
    final_cost: float
    termination_type: TerminationType
    num_iterations: int | None = None
    parameter_values: list[list[float]] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "GoldenSolverResult":
        termination = data["termination_type"]
        if not isinstance(termination, TerminationType):
            termination = TerminationType[termination]
        return cls(
            initial_cost=float(data["initial_cost"]),
            final_cost=float(data["final_cost"]),
            termination_type=termination,
            num_iterations=data.get("num_iterations"),
            parameter_values=data.get("parameter_values"),
        )


def assert_solver_matches_golden(
    summary: SolverSummary,
    golden: GoldenSolverResult,
    *,
    parameters: list[torch.Tensor] | None = None,
    cost_rtol: float = 1e-6,
    cost_atol: float = 1e-9,
    parameter_rtol: float = 1e-6,
    parameter_atol: float = 1e-9,
    iteration_tolerance: int | None = None,
) -> None:
    assert summary.termination_type is golden.termination_type
    torch.testing.assert_close(
        torch.tensor(summary.initial_cost, dtype=torch.float64),
        torch.tensor(golden.initial_cost, dtype=torch.float64),
        rtol=cost_rtol,
        atol=cost_atol,
    )
    torch.testing.assert_close(
        torch.tensor(summary.final_cost, dtype=torch.float64),
        torch.tensor(golden.final_cost, dtype=torch.float64),
        rtol=cost_rtol,
        atol=cost_atol,
    )
    if golden.num_iterations is not None:
        if iteration_tolerance is None:
            assert len(summary.iterations) == golden.num_iterations
        else:
            assert abs(len(summary.iterations) - golden.num_iterations) <= iteration_tolerance
    if golden.parameter_values is not None:
        if parameters is None:
            raise AssertionError("Golden parameter values require parameters to compare")
        assert len(parameters) == len(golden.parameter_values)
        for actual, expected in zip(parameters, golden.parameter_values):
            torch.testing.assert_close(
                actual.detach().cpu().reshape(-1),
                torch.tensor(expected, dtype=actual.dtype).reshape(-1),
                rtol=parameter_rtol,
                atol=parameter_atol,
            )
