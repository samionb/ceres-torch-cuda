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
        u = (x_t - self.grid.x0) / self.grid.spacing
        i = torch.floor(u).to(torch.long)
        t = u - i
        i0 = int(i.detach().cpu())
        p0 = self.grid.get(i0)
        p1 = self.grid.get(i0 + 1)
        m0 = 0.5 * (self.grid.get(i0 + 1) - self.grid.get(i0 - 1))
        m1 = 0.5 * (self.grid.get(i0 + 2) - self.grid.get(i0))
        return cubic_hermite_spline(p0, p1, m0, m1, t)


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
        row_t = torch.as_tensor(row, dtype=self.grid.data.dtype, device=self.grid.data.device)
        col_t = torch.as_tensor(col, dtype=self.grid.data.dtype, device=self.grid.data.device)
        u = (row_t - self.grid.row0) / self.grid.row_spacing
        v = (col_t - self.grid.col0) / self.grid.col_spacing
        i = int(torch.floor(u).detach().cpu())
        j = int(torch.floor(v).detach().cpu())
        tu = u - torch.floor(u)
        tv = v - torch.floor(v)
        rows = []
        for di in range(-1, 3):
            samples = torch.stack([self.grid.get(i + di, j + dj) for dj in range(-1, 3)])
            interp = CubicInterpolator(Grid1D(samples)).evaluate(1.0 + tv)
            rows.append(interp)
        return CubicInterpolator(Grid1D(torch.stack(rows))).evaluate(1.0 + tu)

