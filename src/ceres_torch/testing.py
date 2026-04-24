from __future__ import annotations

import torch


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

