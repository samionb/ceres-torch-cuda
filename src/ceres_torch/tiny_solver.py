from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import torch

from .costs import CostFunction
from .linear import solve_linear_system
from .types import LinearSolverType, TerminationType


class TinySolverStatus(Enum):
    GRADIENT_TOO_SMALL = auto()
    RELATIVE_STEP_SIZE_TOO_SMALL = auto()
    COST_TOO_SMALL = auto()
    HIT_MAX_ITERATIONS = auto()
    COST_CHANGE_TOO_SMALL = auto()


@dataclass
class TinySolverOptions:
    max_num_iterations: int = 50
    gradient_tolerance: float = 1e-10
    parameter_tolerance: float = 1e-8
    function_tolerance: float = 1e-6
    cost_threshold: float = torch.finfo(torch.float64).eps
    initial_trust_region_radius: float = 1e4

    def validate(self) -> None:
        checks = [
            (self.max_num_iterations >= 0, "max_num_iterations must be >= 0"),
            (self.gradient_tolerance >= 0.0, "gradient_tolerance must be >= 0"),
            (self.parameter_tolerance >= 0.0, "parameter_tolerance must be >= 0"),
            (self.function_tolerance >= 0.0, "function_tolerance must be >= 0"),
            (self.cost_threshold >= 0.0, "cost_threshold must be >= 0"),
            (self.initial_trust_region_radius > 0.0, "initial_trust_region_radius must be > 0"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)


@dataclass
class TinySolverSummary:
    termination_type: TerminationType
    initial_cost: float
    final_cost: float
    iterations: int
    gradient_max_norm: float = -1.0
    status: TinySolverStatus = TinySolverStatus.HIT_MAX_ITERATIONS
    message: str = ""

    def IsSolutionUsable(self) -> bool:
        return self.termination_type in {TerminationType.CONVERGENCE, TerminationType.NO_CONVERGENCE}

    def BriefReport(self) -> str:
        return (
            "Tiny Solver Report: "
            f"Iterations: {self.iterations}, "
            f"Initial cost: {self.initial_cost:.6e}, "
            f"Final cost: {self.final_cost:.6e}, "
            f"Gradient max norm: {self.gradient_max_norm:.6e}, "
            f"Status: {self.status.name}, "
            f"Termination: {self.termination_type.value}"
        )


class TinySolver:
    def __init__(self, cost_function: CostFunction, options: TinySolverOptions | None = None) -> None:
        self.cost_function = cost_function
        self.options = options or TinySolverOptions()

    def solve(self, parameters: torch.Tensor) -> TinySolverSummary:
        self.options.validate()
        residual, jacobians = self.cost_function.compute_jacobians([parameters.detach()])
        J = jacobians[0]
        cost = 0.5 * torch.dot(residual, residual)
        gradient = J.T @ residual
        gradient_max_norm = float(torch.max(torch.abs(gradient)).detach().cpu()) if gradient.numel() else 0.0
        initial = float(cost.detach().cpu())
        final = initial
        termination = TerminationType.NO_CONVERGENCE
        message = "Maximum iterations reached."
        status = TinySolverStatus.HIT_MAX_ITERATIONS
        iteration = 0
        if gradient_max_norm < self.options.gradient_tolerance:
            return TinySolverSummary(
                TerminationType.CONVERGENCE,
                initial,
                final,
                iteration,
                gradient_max_norm,
                TinySolverStatus.GRADIENT_TOO_SMALL,
                "Gradient tolerance reached.",
            )
        if initial < self.options.cost_threshold:
            return TinySolverSummary(
                TerminationType.CONVERGENCE,
                initial,
                final,
                iteration,
                gradient_max_norm,
                TinySolverStatus.COST_TOO_SMALL,
                "Cost threshold reached.",
            )

        inverse_radius = 1.0 / self.options.initial_trust_region_radius
        decrease_factor = 2.0
        for iteration in range(1, self.options.max_num_iterations + 1):
            residual, jacobians = self.cost_function.compute_jacobians([parameters.detach()])
            J = jacobians[0]
            g = J.T @ residual
            gradient_max_norm = float(torch.max(torch.abs(g)).detach().cpu()) if g.numel() else 0.0
            if gradient_max_norm < self.options.gradient_tolerance:
                termination = TerminationType.CONVERGENCE
                message = "Gradient tolerance reached."
                status = TinySolverStatus.GRADIENT_TOO_SMALL
                final = float(cost.detach().cpu())
                break
            if float(cost.detach().cpu()) < self.options.cost_threshold:
                termination = TerminationType.CONVERGENCE
                message = "Cost threshold reached."
                status = TinySolverStatus.COST_TOO_SMALL
                final = float(cost.detach().cpu())
                break
            H = J.T @ J
            diagonal = torch.clamp(torch.diagonal(H), min=1e-6, max=1e32)
            damping = inverse_radius * diagonal
            step = solve_linear_system(J, -residual, solver_type=LinearSolverType.DENSE_QR, damping=damping).x
            if torch.linalg.norm(step) <= self.options.parameter_tolerance * (
                torch.linalg.norm(parameters.detach()) + self.options.parameter_tolerance
            ):
                termination = TerminationType.CONVERGENCE
                message = "Parameter tolerance reached."
                status = TinySolverStatus.RELATIVE_STEP_SIZE_TOO_SMALL
                final = float(cost.detach().cpu())
                break
            snapshot = parameters.detach().clone()
            with torch.no_grad():
                parameters.reshape(-1).add_(step.reshape(-1))
            new_residual = self.cost_function.residuals(parameters.detach())
            new_cost = 0.5 * torch.dot(new_residual, new_residual)
            cost_change = 2.0 * (cost - new_cost)
            model_cost_change = torch.dot(step, -2.0 * g - H @ step)
            rho = cost_change / model_cost_change.clamp_min(torch.finfo(J.dtype).eps)
            if bool((rho > 0.0).detach().cpu()):
                final = float(new_cost.detach().cpu())
                cost = new_cost
                if abs(float(cost_change.detach().cpu())) < self.options.function_tolerance:
                    termination = TerminationType.CONVERGENCE
                    message = "Function tolerance reached."
                    status = TinySolverStatus.COST_CHANGE_TOO_SMALL
                    break
                tmp = 2.0 * float(rho.detach().cpu()) - 1.0
                inverse_radius = inverse_radius * max(1.0 / 3.0, 1.0 - tmp * tmp * tmp)
                decrease_factor = 2.0
            else:
                with torch.no_grad():
                    parameters.reshape(-1).copy_(snapshot.reshape(-1))
                if abs(float(cost_change.detach().cpu())) < self.options.function_tolerance:
                    termination = TerminationType.CONVERGENCE
                    message = "Function tolerance reached."
                    status = TinySolverStatus.COST_CHANGE_TOO_SMALL
                    final = float(cost.detach().cpu())
                    break
                inverse_radius *= decrease_factor
                decrease_factor *= 2.0
        else:
            iteration = self.options.max_num_iterations
        return TinySolverSummary(termination, initial, final, iteration, gradient_max_norm, status, message)
