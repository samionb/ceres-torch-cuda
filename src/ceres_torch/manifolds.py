from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .rotation import (
    convert_ceres_to_eigen_quaternion,
    convert_eigen_to_ceres_quaternion,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_product,
)


class Manifold:
    ambient_size: int
    tangent_size: int

    def AmbientSize(self) -> int:
        return self.ambient_size

    def TangentSize(self) -> int:
        return self.tangent_size

    def plus(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def Plus(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self.plus(x, delta)

    def minus(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def Minus(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.minus(y, x)

    def plus_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros(self.tangent_size, dtype=x.dtype, device=x.device, requires_grad=True)

        def wrapped(d: torch.Tensor) -> torch.Tensor:
            return self.plus(x, d)

        return torch.autograd.functional.jacobian(wrapped, delta).reshape(self.ambient_size, self.tangent_size).detach()

    def PlusJacobian(self, x: torch.Tensor) -> torch.Tensor:
        return self.plus_jacobian(x)

    def minus_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        y = x.detach().clone().requires_grad_(True)

        def wrapped(z: torch.Tensor) -> torch.Tensor:
            return self.minus(z, x)

        return torch.autograd.functional.jacobian(wrapped, y).reshape(self.tangent_size, self.ambient_size).detach()

    def MinusJacobian(self, x: torch.Tensor) -> torch.Tensor:
        return self.minus_jacobian(x)

    def right_multiply_by_plus_jacobian(self, x: torch.Tensor, ambient_matrix: torch.Tensor) -> torch.Tensor:
        plus_jacobian = self.plus_jacobian(x).to(dtype=ambient_matrix.dtype, device=ambient_matrix.device)
        return ambient_matrix @ plus_jacobian

    def RightMultiplyByPlusJacobian(self, x: torch.Tensor, ambient_matrix: torch.Tensor) -> torch.Tensor:
        return self.right_multiply_by_plus_jacobian(x, ambient_matrix)


@dataclass
class EuclideanManifold(Manifold):
    size: int

    @property
    def ambient_size(self) -> int:
        return self.size

    @property
    def tangent_size(self) -> int:
        return self.size

    def plus(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1) + delta.reshape(-1)

    def minus(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return y.reshape(-1) - x.reshape(-1)

    def plus_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return torch.eye(self.size, dtype=x.dtype, device=x.device)

    def minus_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return torch.eye(self.size, dtype=x.dtype, device=x.device)


class SubsetManifold(Manifold):
    def __init__(self, size: int, constant_parameters: Sequence[int]) -> None:
        self.size = size
        self.constant_parameters = set(int(i) for i in constant_parameters)
        self.variable_indices = [i for i in range(size) if i not in self.constant_parameters]

    @property
    def ambient_size(self) -> int:
        return self.size

    @property
    def tangent_size(self) -> int:
        return len(self.variable_indices)

    def plus(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        y = x.reshape(-1).clone()
        if self.variable_indices:
            y[self.variable_indices] = y[self.variable_indices] + delta.reshape(-1)
        return y

    def minus(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return (y.reshape(-1) - x.reshape(-1))[self.variable_indices]

    def plus_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        J = x.new_zeros((self.size, self.tangent_size))
        for col, idx in enumerate(self.variable_indices):
            J[idx, col] = 1.0
        return J

    def minus_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return self.plus_jacobian(x).T


class QuaternionManifold(Manifold):
    ambient_size = 4
    tangent_size = 3

    def plus(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x = normalize_quaternion(x.reshape(4))
        delta = delta.reshape(3)
        norm_delta = torch.linalg.norm(delta)
        tiny = torch.as_tensor(torch.finfo(delta.dtype).eps, dtype=delta.dtype, device=delta.device)
        scale = torch.where(norm_delta > tiny, torch.sin(norm_delta) / norm_delta, 1.0 - norm_delta * norm_delta / 6.0)
        q_delta = torch.cat([torch.cos(norm_delta).reshape(1), scale.reshape(1) * delta])
        return normalize_quaternion(quaternion_product(q_delta, x))

    def minus(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        q = quaternion_product(normalize_quaternion(y.reshape(4)), quaternion_conjugate(normalize_quaternion(x.reshape(4))))
        u = q[1:]
        u_norm = torch.linalg.norm(u)
        tiny = torch.as_tensor(torch.finfo(q.dtype).eps, dtype=q.dtype, device=q.device)
        scale = torch.where(u_norm > tiny, torch.atan2(u_norm, q[0]) / u_norm, torch.ones_like(u_norm))
        return scale * u

    def plus_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        x = normalize_quaternion(x.reshape(4))
        w, a, b, c = x
        return torch.stack(
            [
                torch.stack([-a, -b, -c]),
                torch.stack([w, c, -b]),
                torch.stack([-c, w, a]),
                torch.stack([b, -a, w]),
            ]
        ).to(dtype=x.dtype, device=x.device)


class EigenQuaternionManifold(QuaternionManifold):
    def plus(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        ceres_q = convert_eigen_to_ceres_quaternion(x.reshape(4))
        return convert_ceres_to_eigen_quaternion(super().plus(ceres_q, delta))

    def minus(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return super().minus(convert_eigen_to_ceres_quaternion(y.reshape(4)), convert_eigen_to_ceres_quaternion(x.reshape(4)))

    def plus_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        J = super().plus_jacobian(convert_eigen_to_ceres_quaternion(x.reshape(4)))
        return torch.cat([J[1:], J[:1]], dim=0)


def _sphere_basis(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1)
    x = x / torch.linalg.norm(x).clamp_min(torch.finfo(x.dtype).tiny)
    P = torch.eye(x.numel(), dtype=x.dtype, device=x.device) - torch.outer(x, x)
    U, _, _ = torch.linalg.svd(P)
    return U[:, : x.numel() - 1]


class SphereManifold(Manifold):
    def __init__(self, size: int) -> None:
        if size <= 1:
            raise ValueError("SphereManifold size must be greater than 1")
        self.size = size

    @property
    def ambient_size(self) -> int:
        return self.size

    @property
    def tangent_size(self) -> int:
        return self.size - 1

    def plus(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1)
        norm_x = torch.linalg.norm(x).clamp_min(torch.finfo(x.dtype).tiny)
        unit_x = x / norm_x
        delta = delta.reshape(-1)
        theta = torch.linalg.norm(delta)
        if float(theta.detach().cpu()) == 0.0:
            return x.clone()
        tangent = _sphere_basis(unit_x) @ (delta / theta)
        return norm_x * (torch.cos(theta) * unit_x + torch.sin(theta) * tangent)

    def minus(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1)
        y = y.reshape(-1)
        norm_x = torch.linalg.norm(x).clamp_min(torch.finfo(x.dtype).tiny)
        unit_x = x / norm_x
        unit_y = y / torch.linalg.norm(y).clamp_min(torch.finfo(y.dtype).tiny)
        dot = torch.clamp(torch.dot(unit_x, unit_y), -1.0, 1.0)
        perp = unit_y - dot * unit_x
        perp_norm = torch.linalg.norm(perp)
        if float(perp_norm.detach().cpu()) == 0.0:
            return x.new_zeros(self.tangent_size)
        angle = torch.atan2(perp_norm, dot)
        return _sphere_basis(unit_x).T @ (angle * perp / perp_norm)

    def plus_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return _sphere_basis(x.reshape(-1))

    def minus_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return _sphere_basis(x.reshape(-1)).T


class LineManifold(Manifold):
    def __init__(self, ambient_dimension: int) -> None:
        if ambient_dimension <= 1:
            raise ValueError("LineManifold ambient_dimension must be greater than 1")
        self.size = ambient_dimension
        self._sphere = SphereManifold(ambient_dimension)

    @property
    def ambient_size(self) -> int:
        return 2 * self.size

    @property
    def tangent_size(self) -> int:
        return 2 * (self.size - 1)

    def plus(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1)
        origin, direction = x[: self.size], x[self.size :]
        delta_o, delta_d = delta[: self.size - 1], delta[self.size - 1 :]
        basis = _sphere_basis(direction)
        new_origin = origin + basis @ delta_o
        new_direction = self._sphere.plus(direction, delta_d)
        return torch.cat([new_origin, new_direction])

    def minus(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y = y.reshape(-1)
        x = x.reshape(-1)
        x_origin, x_direction = x[: self.size], x[self.size :]
        y_origin, y_direction = y[: self.size], y[self.size :]
        basis = _sphere_basis(x_direction)
        delta_o = basis.T @ (y_origin - x_origin)
        delta_d = self._sphere.minus(y_direction, x_direction)
        return torch.cat([delta_o, delta_d])

    def plus_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1)
        direction = x[self.size :]
        basis = _sphere_basis(direction)
        zeros = x.new_zeros((self.size, self.size - 1))
        return torch.cat(
            [
                torch.cat([basis, zeros], dim=1),
                torch.cat([zeros, basis], dim=1),
            ],
            dim=0,
        )

    def minus_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1)
        direction = x[self.size :]
        basis_t = _sphere_basis(direction).T
        zeros = x.new_zeros((self.size - 1, self.size))
        return torch.cat(
            [
                torch.cat([basis_t, zeros], dim=1),
                torch.cat([zeros, basis_t], dim=1),
            ],
            dim=0,
        )


class ProductManifold(Manifold):
    def __init__(self, *manifolds: Manifold) -> None:
        self.manifolds = list(manifolds)

    @property
    def ambient_size(self) -> int:
        return sum(m.ambient_size for m in self.manifolds)

    @property
    def tangent_size(self) -> int:
        return sum(m.tangent_size for m in self.manifolds)

    def plus(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        xs = torch.split(x.reshape(-1), [m.ambient_size for m in self.manifolds])
        ds = torch.split(delta.reshape(-1), [m.tangent_size for m in self.manifolds])
        return torch.cat([m.plus(xi, di) for m, xi, di in zip(self.manifolds, xs, ds)])

    def minus(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ys = torch.split(y.reshape(-1), [m.ambient_size for m in self.manifolds])
        xs = torch.split(x.reshape(-1), [m.ambient_size for m in self.manifolds])
        return torch.cat([m.minus(yi, xi) for m, yi, xi in zip(self.manifolds, ys, xs)])

    def plus_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        xs = torch.split(x.reshape(-1), [m.ambient_size for m in self.manifolds])
        blocks = [m.plus_jacobian(xi) for m, xi in zip(self.manifolds, xs)]
        return torch.block_diag(*blocks)

    def minus_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        xs = torch.split(x.reshape(-1), [m.ambient_size for m in self.manifolds])
        blocks = [m.minus_jacobian(xi) for m, xi in zip(self.manifolds, xs)]
        return torch.block_diag(*blocks)


class AutoDiffManifold(Manifold):
    def __init__(
        self,
        ambient_size: int,
        tangent_size: int,
        plus_fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        minus_fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self._ambient_size = ambient_size
        self._tangent_size = tangent_size
        self.plus_fun = plus_fun
        self.minus_fun = minus_fun

    @property
    def ambient_size(self) -> int:
        return self._ambient_size

    @property
    def tangent_size(self) -> int:
        return self._tangent_size

    def plus(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self.plus_fun(x.reshape(-1), delta.reshape(-1)).reshape(-1)

    def minus(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.minus_fun(y.reshape(-1), x.reshape(-1)).reshape(-1)
