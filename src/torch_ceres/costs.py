from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import torch

from .types import NumericDiffMethodType


def as_residual_tensor(value: torch.Tensor | Sequence[float] | float, like: Optional[torch.Tensor] = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value
    elif like is not None:
        result = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    else:
        result = torch.as_tensor(value, dtype=torch.float64)
    return result.reshape(-1)


class CostFunction:
    parameter_block_sizes: Optional[list[int]] = None
    num_residuals: Optional[int] = None

    def residuals(self, *parameters: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute(self, parameters: Sequence[torch.Tensor]) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        return self.residuals(*parameters), None

    def compute_jacobians(self, parameters: Sequence[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        active = [p.detach().clone().requires_grad_(True) for p in parameters]

        def wrapped(*args: torch.Tensor) -> torch.Tensor:
            return as_residual_tensor(self.residuals(*args), args[0] if args else None)

        residual = wrapped(*active)
        jac = torch.autograd.functional.jacobian(wrapped, tuple(active), vectorize=False)
        if isinstance(jac, torch.Tensor):
            jac = (jac,)
        elif len(active) == 1 and len(jac) == 1 and isinstance(jac[0], tuple):
            jac = jac[0]
        jacobians = [j.reshape(residual.numel(), p.numel()).detach() for j, p in zip(jac, active)]
        return residual.detach(), jacobians

    def __call__(self, *parameters: torch.Tensor) -> torch.Tensor:
        return self.residuals(*parameters)


class CallableCostFunction(CostFunction):
    def __init__(
        self,
        functor: Callable[..., torch.Tensor],
        parameter_block_sizes: Optional[Sequence[int]] = None,
        num_residuals: Optional[int] = None,
    ) -> None:
        self.functor = functor
        self.parameter_block_sizes = list(parameter_block_sizes) if parameter_block_sizes is not None else None
        self.num_residuals = num_residuals

    def residuals(self, *parameters: torch.Tensor) -> torch.Tensor:
        return as_residual_tensor(self.functor(*parameters), parameters[0] if parameters else None)


class AutoDiffCostFunction(CallableCostFunction):
    pass


class DynamicAutoDiffCostFunction(AutoDiffCostFunction):
    pass


class AnalyticCostFunction(CostFunction):
    def __init__(
        self,
        residual_fun: Callable[..., torch.Tensor],
        jacobian_fun: Callable[..., Sequence[torch.Tensor]],
        parameter_block_sizes: Optional[Sequence[int]] = None,
        num_residuals: Optional[int] = None,
    ) -> None:
        self.residual_fun = residual_fun
        self.jacobian_fun = jacobian_fun
        self.parameter_block_sizes = list(parameter_block_sizes) if parameter_block_sizes is not None else None
        self.num_residuals = num_residuals

    def residuals(self, *parameters: torch.Tensor) -> torch.Tensor:
        return as_residual_tensor(self.residual_fun(*parameters), parameters[0] if parameters else None)

    def compute_jacobians(self, parameters: Sequence[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        residual = self.residuals(*parameters)
        jacobians = [
            torch.as_tensor(j, dtype=residual.dtype, device=residual.device).reshape(residual.numel(), p.numel())
            for j, p in zip(self.jacobian_fun(*parameters), parameters)
        ]
        return residual, jacobians


class NumericDiffCostFunction(CallableCostFunction):
    def __init__(
        self,
        functor: Callable[..., torch.Tensor],
        parameter_block_sizes: Optional[Sequence[int]] = None,
        num_residuals: Optional[int] = None,
        method: NumericDiffMethodType = NumericDiffMethodType.CENTRAL,
        relative_step_size: float = 1e-6,
    ) -> None:
        super().__init__(functor, parameter_block_sizes, num_residuals)
        self.method = method
        self.relative_step_size = relative_step_size

    def compute_jacobians(self, parameters: Sequence[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        base = [p.detach().clone() for p in parameters]
        residual = self.residuals(*base)
        jacobians: list[torch.Tensor] = []
        for block_index, param in enumerate(base):
            columns = []
            flat = param.reshape(-1)
            for col in range(flat.numel()):
                h = self.relative_step_size * max(float(abs(flat[col]).detach().cpu()), 1.0)
                step = torch.zeros_like(flat)
                step[col] = h

                def eval_at(delta: torch.Tensor) -> torch.Tensor:
                    shifted = [p.clone() for p in base]
                    shifted_flat = shifted[block_index].reshape(-1)
                    shifted_flat.add_(delta)
                    return self.residuals(*shifted)

                if self.method is NumericDiffMethodType.FORWARD:
                    deriv = (eval_at(step) - residual) / h
                elif self.method is NumericDiffMethodType.RIDDERS:
                    d1 = (eval_at(step) - eval_at(-step)) / (2.0 * h)
                    half_step = step * 0.5
                    d2 = (eval_at(half_step) - eval_at(-half_step)) / h
                    deriv = (4.0 * d2 - d1) / 3.0
                else:
                    deriv = (eval_at(step) - eval_at(-step)) / (2.0 * h)
                columns.append(deriv.reshape(-1))
            jacobians.append(torch.stack(columns, dim=1) if columns else residual.new_zeros((residual.numel(), 0)))
        return residual, jacobians


class DynamicNumericDiffCostFunction(NumericDiffCostFunction):
    pass


class ConditionedCostFunction(CostFunction):
    def __init__(self, wrapped: CostFunction, conditioners: Iterable[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self.wrapped = wrapped
        self.conditioners = list(conditioners)
        self.parameter_block_sizes = wrapped.parameter_block_sizes
        self.num_residuals = wrapped.num_residuals

    def residuals(self, *parameters: torch.Tensor) -> torch.Tensor:
        residual = self.wrapped.residuals(*parameters)
        if len(self.conditioners) != residual.numel():
            raise ValueError("ConditionedCostFunction needs one conditioner per residual")
        return torch.stack([as_residual_tensor(fn(r), residual)[0] for fn, r in zip(self.conditioners, residual)])


class NormalPrior(CostFunction):
    def __init__(self, A: torch.Tensor, b: torch.Tensor) -> None:
        self.A = A
        self.b = b
        self.parameter_block_sizes = [b.numel()]
        self.num_residuals = A.shape[0]

    def residuals(self, x: torch.Tensor) -> torch.Tensor:
        A = self.A.to(dtype=x.dtype, device=x.device)
        b = self.b.to(dtype=x.dtype, device=x.device)
        return (A @ (x.reshape(-1) - b.reshape(-1))).reshape(-1)

    def compute_jacobians(self, parameters: Sequence[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = parameters[0]
        return self.residuals(x), [self.A.to(dtype=x.dtype, device=x.device)]


class CostFunctionToFunctor:
    def __init__(self, cost_function: CostFunction) -> None:
        self.cost_function = cost_function

    def __call__(self, *parameters: torch.Tensor) -> torch.Tensor:
        return self.cost_function.residuals(*parameters)


class DynamicCostFunctionToFunctor(CostFunctionToFunctor):
    pass


class FirstOrderFunction:
    def value_and_gradient(self, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class AutoDiffFirstOrderFunction(FirstOrderFunction):
    def __init__(self, functor: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.functor = functor

    def value_and_gradient(self, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = parameters.detach().clone().requires_grad_(True)
        value = self.functor(x).reshape(())
        (grad,) = torch.autograd.grad(value, x)
        return value.detach(), grad.detach()


class NumericDiffFirstOrderFunction(FirstOrderFunction):
    def __init__(self, functor: Callable[[torch.Tensor], torch.Tensor], relative_step_size: float = 1e-6) -> None:
        self.functor = functor
        self.relative_step_size = relative_step_size

    def value_and_gradient(self, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = parameters.detach().clone()
        value = self.functor(x).reshape(())
        grad = torch.zeros_like(x.reshape(-1))
        flat = x.reshape(-1)
        for i in range(flat.numel()):
            h = self.relative_step_size * max(float(abs(flat[i]).cpu()), 1.0)
            xp = x.clone().reshape(-1)
            xm = x.clone().reshape(-1)
            xp[i] += h
            xm[i] -= h
            grad[i] = (self.functor(xp.reshape_as(x)) - self.functor(xm.reshape_as(x))) / (2.0 * h)
        return value.detach(), grad.reshape_as(parameters).detach()


@dataclass
class GradientChecker:
    relative_precision: float = 1e-8
    relative_step_size: float = 1e-6

    def probe(self, cost_function: CostFunction, parameters: Sequence[torch.Tensor]) -> dict[str, float | bool]:
        residual, analytic = cost_function.compute_jacobians(parameters)
        numeric_cost = NumericDiffCostFunction(
            lambda *xs: cost_function.residuals(*xs),
            method=NumericDiffMethodType.CENTRAL,
            relative_step_size=self.relative_step_size,
        )
        _, numeric = numeric_cost.compute_jacobians(parameters)
        max_relative_error = 0.0
        for a, n in zip(analytic, numeric):
            denom = torch.clamp(torch.maximum(torch.abs(a), torch.abs(n)), min=torch.finfo(a.dtype).eps)
            rel = torch.max(torch.abs(a - n) / denom).item() if a.numel() else 0.0
            max_relative_error = max(max_relative_error, rel)
        return {
            "ok": max_relative_error <= self.relative_precision,
            "max_relative_error": max_relative_error,
            "num_residuals": residual.numel(),
        }
