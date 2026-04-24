from __future__ import annotations

from dataclasses import dataclass

import torch

from .costs import CostFunction
from .linear import jacobi_damping_from_jacobian, solve_linear_system
from .types import LinearSolverType, TerminationType


@dataclass
class TinySolverOptions:
    max_num_iterations: int = 50
    gradient_tolerance: float = 1e-10
    parameter_tolerance: float = 1e-8
    function_tolerance: float = 1e-6
    initial_trust_region_radius: float = 1e4


@dataclass
class TinySolverSummary:
    termination_type: TerminationType
    initial_cost: float
    final_cost: float
    iterations: int


class TinySolver:
    def __init__(self, cost_function: CostFunction, options: TinySolverOptions | None = None) -> None:
        self.cost_function = cost_function
        self.options = options or TinySolverOptions()

    def solve(self, parameters: torch.Tensor) -> TinySolverSummary:
        radius = self.options.initial_trust_region_radius
        residual, jacobians = self.cost_function.compute_jacobians([parameters.detach()])
        cost = 0.5 * torch.dot(residual, residual)
        initial = float(cost.detach().cpu())
        final = initial
        termination = TerminationType.NO_CONVERGENCE
        for iteration in range(1, self.options.max_num_iterations + 1):
            residual, jacobians = self.cost_function.compute_jacobians([parameters.detach()])
            J = jacobians[0]
            g = J.T @ residual
            if torch.max(torch.abs(g)) <= self.options.gradient_tolerance:
                termination = TerminationType.CONVERGENCE
                break
            damping = jacobi_damping_from_jacobian(J, min_diagonal=1e-6, max_diagonal=1e32, radius=radius)
            step = solve_linear_system(J, -residual, solver_type=LinearSolverType.DENSE_QR, damping=damping).x
            snapshot = parameters.detach().clone()
            with torch.no_grad():
                parameters.reshape(-1).add_(step.reshape(-1))
            new_residual = self.cost_function.residuals(parameters.detach())
            new_cost = 0.5 * torch.dot(new_residual, new_residual)
            if new_cost < cost:
                radius *= 2.0
                final = float(new_cost.detach().cpu())
                if torch.linalg.norm(step) <= self.options.parameter_tolerance * (
                    torch.linalg.norm(parameters.detach()) + self.options.parameter_tolerance
                ):
                    termination = TerminationType.CONVERGENCE
                    break
                if abs(float(cost.detach().cpu()) - final) <= self.options.function_tolerance * max(float(cost.detach().cpu()), 1.0):
                    termination = TerminationType.CONVERGENCE
                    break
                cost = new_cost
            else:
                with torch.no_grad():
                    parameters.reshape(-1).copy_(snapshot.reshape(-1))
                radius *= 0.25
        else:
            iteration = self.options.max_num_iterations
        return TinySolverSummary(termination, initial, final, iteration)

