from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def _as_tensor_like(value: float, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


class LossFunction:
    def evaluate(self, sq_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __call__(self, sq_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.evaluate(sq_norm)


class TrivialLoss(LossFunction):
    def evaluate(self, sq_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return sq_norm, torch.ones_like(sq_norm), torch.zeros_like(sq_norm)


@dataclass(frozen=True)
class HuberLoss(LossFunction):
    a: float

    def evaluate(self, sq_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = _as_tensor_like(self.a * self.a, sq_norm)
        a = _as_tensor_like(self.a, sq_norm)
        mask = sq_norm <= b
        sqrt_s = torch.sqrt(torch.clamp(sq_norm, min=torch.finfo(sq_norm.dtype).tiny))
        rho0 = torch.where(mask, sq_norm, 2.0 * a * sqrt_s - b)
        rho1 = torch.where(mask, torch.ones_like(sq_norm), a / sqrt_s)
        rho2 = torch.where(mask, torch.zeros_like(sq_norm), -0.5 * a / (sqrt_s * sq_norm))
        return rho0, rho1, rho2


@dataclass(frozen=True)
class SoftLOneLoss(LossFunction):
    a: float

    def evaluate(self, sq_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = _as_tensor_like(self.a * self.a, sq_norm)
        z = 1.0 + sq_norm / b
        sqrt_z = torch.sqrt(z)
        rho0 = 2.0 * b * (sqrt_z - 1.0)
        rho1 = 1.0 / sqrt_z
        rho2 = -0.5 / (b * z * sqrt_z)
        return rho0, rho1, rho2


@dataclass(frozen=True)
class CauchyLoss(LossFunction):
    a: float

    def evaluate(self, sq_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = _as_tensor_like(self.a * self.a, sq_norm)
        z = 1.0 + sq_norm / b
        rho0 = b * torch.log(z)
        rho1 = 1.0 / z
        rho2 = -1.0 / (b * z * z)
        return rho0, rho1, rho2


@dataclass(frozen=True)
class ArctanLoss(LossFunction):
    a: float

    def evaluate(self, sq_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = _as_tensor_like(self.a, sq_norm)
        z = sq_norm / a
        denom = 1.0 + z * z
        rho0 = a * torch.atan(z)
        rho1 = 1.0 / denom
        rho2 = -2.0 * z / (a * denom * denom)
        return rho0, rho1, rho2


@dataclass(frozen=True)
class TolerantLoss(LossFunction):
    a: float
    b: float

    def evaluate(self, sq_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = _as_tensor_like(self.a, sq_norm)
        b = _as_tensor_like(self.b, sq_norm)
        z = (sq_norm - a) / b
        c = b * torch.nn.functional.softplus(-a / b)
        rho0 = b * torch.nn.functional.softplus(z) - c
        rho1 = torch.sigmoid(z)
        rho2 = rho1 * (1.0 - rho1) / b
        return rho0, rho1, rho2


@dataclass(frozen=True)
class TukeyLoss(LossFunction):
    a: float

    def evaluate(self, sq_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a2 = _as_tensor_like(self.a * self.a, sq_norm)
        z = sq_norm / a2
        inner = 1.0 - z
        mask = sq_norm <= a2
        rho0 = torch.where(mask, (a2 / 3.0) * (1.0 - inner**3), a2 / 3.0)
        rho1 = torch.where(mask, inner**2, torch.zeros_like(sq_norm))
        rho2 = torch.where(mask, -2.0 * inner / a2, torch.zeros_like(sq_norm))
        return rho0, rho1, rho2


@dataclass
class ScaledLoss(LossFunction):
    rho: Optional[LossFunction]
    scale: float

    def evaluate(self, sq_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = TrivialLoss() if self.rho is None else self.rho
        rho0, rho1, rho2 = base.evaluate(sq_norm)
        scale = _as_tensor_like(self.scale, sq_norm)
        return scale * rho0, scale * rho1, scale * rho2


@dataclass
class ComposedLoss(LossFunction):
    outer: LossFunction
    inner: LossFunction

    def evaluate(self, sq_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g0, g1, g2 = self.inner.evaluate(sq_norm)
        f0, f1, f2 = self.outer.evaluate(g0)
        return f0, f1 * g1, f2 * g1 * g1 + f1 * g2


class LossFunctionWrapper(LossFunction):
    def __init__(self, rho: Optional[LossFunction]) -> None:
        self.rho = rho

    def reset(self, rho: Optional[LossFunction]) -> None:
        self.rho = rho

    def evaluate(self, sq_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.rho or TrivialLoss()).evaluate(sq_norm)


def loss_or_trivial(loss: Optional[LossFunction]) -> LossFunction:
    return loss if loss is not None else TrivialLoss()


def robust_weight(loss: Optional[LossFunction], residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sq_norm = torch.sum(residual * residual)
    rho0, rho1, _ = loss_or_trivial(loss).evaluate(sq_norm)
    weight = torch.sqrt(torch.clamp(rho1, min=0.0))
    return 0.5 * rho0, weight

