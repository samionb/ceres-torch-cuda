from __future__ import annotations

import torch


def catmull_rom_spline(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3)
    b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)
    c = 0.5 * (-p0 + p2)
    d = p1
    return d + x * (c + x * (b + x * a))


def catmull_rom_spline_derivative(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3)
    b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)
    c = 0.5 * (-p0 + p2)
    return c + x * (2.0 * b + 3.0 * a * x)


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


class Grid1D:
    def __init__(
        self,
        data: torch.Tensor,
        x0: float = 0.0,
        spacing: float = 1.0,
        *,
        begin: int = 0,
        end: int | None = None,
        data_dimension: int | None = None,
        interleaved: bool = True,
    ) -> None:
        self.data = torch.as_tensor(data)
        self.x0 = x0
        self.spacing = spacing
        self.begin = int(begin)
        self.interleaved = interleaved

        if self.data.ndim == 1:
            self._data_dimension = int(data_dimension or 1)
            if self._data_dimension < 1:
                raise ValueError("Grid1D data_dimension must be >= 1")
            if end is None:
                if self.data.numel() % self._data_dimension != 0:
                    raise ValueError("Flat Grid1D data length must be divisible by data_dimension")
                self.end = self.begin + self.data.numel() // self._data_dimension
            else:
                self.end = int(end)
            self._num_values = self.end - self.begin
            if self._num_values <= 0:
                raise ValueError("Grid1D requires begin < end")
            if self.data.numel() != self._num_values * self._data_dimension:
                raise ValueError("Flat Grid1D data length does not match [begin, end) and data_dimension")
            self._flat_layout = True
        else:
            inferred_dimension = int(torch.tensor(self.data.shape[1:]).prod().item())
            self._data_dimension = int(data_dimension or inferred_dimension)
            if self._data_dimension != inferred_dimension:
                raise ValueError("Shaped Grid1D data_dimension must match trailing tensor dimensions")
            self.end = int(end) if end is not None else self.begin + self.data.shape[0]
            self._num_values = self.end - self.begin
            if self._num_values <= 0:
                raise ValueError("Grid1D requires begin < end")
            if self.data.shape[0] != self._num_values:
                raise ValueError("Shaped Grid1D leading dimension must match [begin, end)")
            self._flat_layout = False

    def get(self, idx: int) -> torch.Tensor:
        idx = max(self.begin, min(idx, self.end - 1)) - self.begin
        if not self._flat_layout:
            return self.data[idx]
        if self.interleaved:
            offset = self._data_dimension * idx
            value = self.data[offset : offset + self._data_dimension]
        else:
            value = torch.stack([self.data[channel * self._num_values + idx] for channel in range(self._data_dimension)])
        return value[0] if self._data_dimension == 1 else value

    @property
    def data_dimension(self) -> int:
        return self._data_dimension

    @property
    def DATA_DIMENSION(self) -> int:
        return self._data_dimension

    GetValue = get


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

    Evaluate = evaluate_with_derivative

    def _evaluate_unit_interval(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = (x_t - self.grid.x0) / self.grid.spacing
        i = torch.floor(u).to(torch.long)
        t = u - i
        i0 = int(i.detach().cpu())
        p0 = self.grid.get(i0 - 1)
        p1 = self.grid.get(i0)
        p2 = self.grid.get(i0 + 1)
        p3 = self.grid.get(i0 + 2)
        return catmull_rom_spline(p0, p1, p2, p3, t), catmull_rom_spline_derivative(p0, p1, p2, p3, t)


class Grid2D:
    def __init__(
        self,
        data: torch.Tensor,
        row0: float = 0.0,
        col0: float = 0.0,
        row_spacing: float = 1.0,
        col_spacing: float = 1.0,
        *,
        row_begin: int = 0,
        row_end: int | None = None,
        col_begin: int = 0,
        col_end: int | None = None,
        data_dimension: int | None = None,
        row_major: bool = True,
        interleaved: bool = True,
    ) -> None:
        self.data = torch.as_tensor(data)
        self.row0 = row0
        self.col0 = col0
        self.row_spacing = row_spacing
        self.col_spacing = col_spacing
        self.row_begin = int(row_begin)
        self.col_begin = int(col_begin)
        self.row_major = row_major
        self.interleaved = interleaved

        if self.data.ndim == 1:
            if row_end is None or col_end is None:
                raise ValueError("Flat Grid2D data requires row_end and col_end")
            self.row_end = int(row_end)
            self.col_end = int(col_end)
            self._data_dimension = int(data_dimension or 1)
            if self._data_dimension < 1:
                raise ValueError("Grid2D data_dimension must be >= 1")
            self._num_rows = self.row_end - self.row_begin
            self._num_cols = self.col_end - self.col_begin
            if self._num_rows <= 0 or self._num_cols <= 0:
                raise ValueError("Grid2D requires row_begin < row_end and col_begin < col_end")
            self._num_values = self._num_rows * self._num_cols
            if self.data.numel() != self._num_values * self._data_dimension:
                raise ValueError("Flat Grid2D data length does not match extents and data_dimension")
            self._flat_layout = True
        else:
            self.row_end = int(row_end) if row_end is not None else self.row_begin + self.data.shape[0]
            self.col_end = int(col_end) if col_end is not None else self.col_begin + self.data.shape[1]
            inferred_dimension = 1 if self.data.ndim == 2 else int(torch.tensor(self.data.shape[2:]).prod().item())
            self._data_dimension = int(data_dimension or inferred_dimension)
            if self._data_dimension != inferred_dimension:
                raise ValueError("Shaped Grid2D data_dimension must match trailing tensor dimensions")
            self._num_rows = self.row_end - self.row_begin
            self._num_cols = self.col_end - self.col_begin
            if self._num_rows <= 0 or self._num_cols <= 0:
                raise ValueError("Grid2D requires row_begin < row_end and col_begin < col_end")
            if self.data.shape[:2] != (self._num_rows, self._num_cols):
                raise ValueError("Shaped Grid2D leading dimensions must match extents")
            self._num_values = self._num_rows * self._num_cols
            self._flat_layout = False

    def get(self, row: int, col: int) -> torch.Tensor:
        row_idx = max(self.row_begin, min(row, self.row_end - 1)) - self.row_begin
        col_idx = max(self.col_begin, min(col, self.col_end - 1)) - self.col_begin
        if not self._flat_layout:
            return self.data[row_idx, col_idx]
        linear = self._num_cols * row_idx + col_idx if self.row_major else self._num_rows * col_idx + row_idx
        if self.interleaved:
            offset = self._data_dimension * linear
            value = self.data[offset : offset + self._data_dimension]
        else:
            value = torch.stack([self.data[channel * self._num_values + linear] for channel in range(self._data_dimension)])
        return value[0] if self._data_dimension == 1 else value

    @property
    def data_dimension(self) -> int:
        return self._data_dimension

    @property
    def DATA_DIMENSION(self) -> int:
        return self._data_dimension

    GetValue = get


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

    Evaluate = evaluate_with_derivatives


CubicHermiteSpline = catmull_rom_spline
CubicHermiteSplineDerivative = catmull_rom_spline_derivative
