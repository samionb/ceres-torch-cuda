from __future__ import annotations

import torch

from .linear import schur_solve_dense


def solve_dense_schur(jacobian: torch.Tensor, residuals: torch.Tensor, num_eliminate: int) -> torch.Tensor:
    return schur_solve_dense(jacobian, residuals, num_eliminate)


def solve_iterative_schur(jacobian: torch.Tensor, residuals: torch.Tensor, num_eliminate: int) -> torch.Tensor:
    # Pure PyTorch first slice: use explicit dense Schur. Optional CUDA/sparse
    # backends can replace this function through the linear backend registry.
    return schur_solve_dense(jacobian, residuals, num_eliminate)

