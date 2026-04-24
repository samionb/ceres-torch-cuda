from __future__ import annotations

from dataclasses import dataclass

import torch


def cubic_hermite_spline(p0: torch.Tensor, p1: torch.Tensor, m0: torch.Tensor, m1: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x2 = x * x
    x3 = x2 * x
    h00 = 2 * x3 - 3 * x2 + 1
    h10 = x3 - 2 * x2 + x
    h01 = -2 * x3 + 3 * x2
    h11 = x3 - x2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def cubic_hermite_spline_derivative(
    p0: torch.Tensor,
    p1: torch.Tensor,
    m0: torch.Tensor,
    m1: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    x2 = x * x
    dh00 = 6 * x2 - 6 * x
    dh10 = 3 * x2 - 4 * x + 1
    dh01 = -6 * x2 + 6 * x
    dh11 = 3 * x2 - 2 * x
    return dh00 * p0 + dh10 * m0 + dh01 * p1 + dh11 * m1


@dataclass
class Grid1D:
    data: torch.Tensor
    x0: float = 0.0
    spacing: float = 1.0

    def get(self, idx: int) -> torch.Tensor:
        idx = max(0, min(idx, self.data.shape[0] - 1))
        return self.data[idx]


class CubicInterpolator:
    def __init__(self, grid: Grid1D) -> None:
        self.grid = grid

    def evaluate(self, x: torch.Tensor | float) -> torch.Tensor:
        x_t = torch.as_tensor(x, dtype=self.grid.data.dtype, device=self.grid.data.device)
        value, _ = self._evaluate_unit_interval(x_t)
        return value

    def evaluate_with_derivative(self, x: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
        x_t = torch.as_tensor(x, dtype=self.grid.data.dtype, device=self.grid.data.device)
        value, unit_derivative = self._evaluate_unit_interval(x_t)
        return value, unit_derivative / self.grid.spacing

    def _evaluate_unit_interval(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = (x_t - self.grid.x0) / self.grid.spacing
        i = torch.floor(u).to(torch.long)
        t = u - i
        i0 = int(i.detach().cpu())
        p0 = self.grid.get(i0)
        p1 = self.grid.get(i0 + 1)
        m0 = 0.5 * (self.grid.get(i0 + 1) - self.grid.get(i0 - 1))
        m1 = 0.5 * (self.grid.get(i0 + 2) - self.grid.get(i0))
        return cubic_hermite_spline(p0, p1, m0, m1, t), cubic_hermite_spline_derivative(p0, p1, m0, m1, t)


@dataclass
class Grid2D:
    data: torch.Tensor
    row0: float = 0.0
    col0: float = 0.0
    row_spacing: float = 1.0
    col_spacing: float = 1.0

    def get(self, row: int, col: int) -> torch.Tensor:
        row = max(0, min(row, self.data.shape[0] - 1))
        col = max(0, min(col, self.data.shape[1] - 1))
        return self.data[row, col]


class BiCubicInterpolator:
    def __init__(self, grid: Grid2D) -> None:
        self.grid = grid

    def evaluate(self, row: torch.Tensor | float, col: torch.Tensor | float) -> torch.Tensor:
        value, _, _ = self.evaluate_with_derivatives(row, col)
        return value

    def evaluate_with_derivatives(
        self,
        row: torch.Tensor | float,
        col: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_t = torch.as_tensor(row, dtype=self.grid.data.dtype, device=self.grid.data.device)
        col_t = torch.as_tensor(col, dtype=self.grid.data.dtype, device=self.grid.data.device)
        u = (row_t - self.grid.row0) / self.grid.row_spacing
        v = (col_t - self.grid.col0) / self.grid.col_spacing
        i = int(torch.floor(u).detach().cpu())
        j = int(torch.floor(v).detach().cpu())
        tu = u - torch.floor(u)
        tv = v - torch.floor(v)
        rows = []
        col_derivatives = []
        for di in range(-1, 3):
            samples = torch.stack([self.grid.get(i + di, j + dj) for dj in range(-1, 3)])
            interp, dcol = CubicInterpolator(Grid1D(samples)).evaluate_with_derivative(1.0 + tv)
            rows.append(interp)
            col_derivatives.append(dcol)
        row_interp = CubicInterpolator(Grid1D(torch.stack(rows)))
        value, drow_unit = row_interp.evaluate_with_derivative(1.0 + tu)
        dcol_unit = CubicInterpolator(Grid1D(torch.stack(col_derivatives))).evaluate(1.0 + tu)
        return value, drow_unit / self.grid.row_spacing, dcol_unit / self.grid.col_spacing
