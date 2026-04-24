from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import torch

from .costs import CallableCostFunction, CostFunction
from .losses import LossFunction, robustify_residual_and_jacobian
from .manifolds import EuclideanManifold, Manifold


@dataclass
class CRSMatrix:
    num_rows: int
    num_cols: int
    rows: list[int]
    cols: list[int]
    values: list[float]

    @classmethod
    def from_dense(cls, matrix: torch.Tensor) -> "CRSMatrix":
        rows: list[int] = [0]
        cols: list[int] = []
        values: list[float] = []
        cpu = matrix.detach().cpu()
        for i in range(cpu.shape[0]):
            nz = torch.nonzero(cpu[i] != 0, as_tuple=False).reshape(-1)
            cols.extend(int(j) for j in nz)
            values.extend(float(cpu[i, j]) for j in nz)
            rows.append(len(cols))
        return cls(cpu.shape[0], cpu.shape[1], rows, cols, values)


@dataclass(eq=False)
class ParameterBlock:
    tensor: torch.Tensor
    manifold: Optional[Manifold] = None
    constant: bool = False
    name: Optional[str] = None
    lower_bound: Optional[torch.Tensor] = None
    upper_bound: Optional[torch.Tensor] = None
    ordering_group: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("ParameterBlock tensor must be a torch.Tensor")
        if self.manifold is None:
            self.manifold = EuclideanManifold(self.tensor.numel())
        if self.manifold.ambient_size != self.tensor.numel():
            raise ValueError("Manifold ambient size must equal parameter tensor numel")

    @property
    def size(self) -> int:
        return self.tensor.numel()

    @property
    def tangent_size(self) -> int:
        return 0 if self.constant else self.manifold.tangent_size

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def clone_value(self) -> torch.Tensor:
        return self.tensor.detach().clone()

    def project_bounds(self, value: torch.Tensor) -> torch.Tensor:
        y = value.reshape(-1)
        if self.lower_bound is not None:
            y = torch.maximum(y, self.lower_bound.to(dtype=y.dtype, device=y.device).reshape(-1))
        if self.upper_bound is not None:
            y = torch.minimum(y, self.upper_bound.to(dtype=y.dtype, device=y.device).reshape(-1))
        return y.reshape_as(self.tensor)


@dataclass(eq=False)
class ResidualBlock:
    cost_function: CostFunction
    loss_function: Optional[LossFunction]
    parameter_blocks: list[ParameterBlock]
    name: Optional[str] = None


@dataclass
class EvaluateOptions:
    parameter_blocks: Optional[Sequence[ParameterBlock | torch.Tensor]] = None
    residual_blocks: Optional[Sequence[ResidualBlock]] = None
    apply_loss_function: bool = True
    num_threads: int = 1


@dataclass
class EvaluationResult:
    cost: torch.Tensor
    residuals: torch.Tensor
    gradient: Optional[torch.Tensor] = None
    jacobian: Optional[torch.Tensor] = None
    crs_jacobian: Optional[CRSMatrix] = None


class Problem:
    def __init__(self) -> None:
        self.parameter_blocks: list[ParameterBlock] = []
        self.residual_blocks: list[ResidualBlock] = []
        self._tensor_to_block: dict[int, ParameterBlock] = {}

    def add_parameter_block(
        self,
        values: torch.Tensor,
        size: Optional[int] = None,
        manifold: Optional[Manifold] = None,
        *,
        name: Optional[str] = None,
    ) -> ParameterBlock:
        if size is not None and values.numel() != size:
            raise ValueError(f"Expected parameter block size {size}, got {values.numel()}")
        key = id(values)
        if key in self._tensor_to_block:
            block = self._tensor_to_block[key]
            if manifold is not None:
                block.manifold = manifold
            if name is not None:
                block.name = name
            return block
        block = ParameterBlock(values, manifold=manifold, name=name)
        self.parameter_blocks.append(block)
        self._tensor_to_block[key] = block
        return block

    AddParameterBlock = add_parameter_block

    def add_residual_block(
        self,
        cost_function: CostFunction | callable,
        loss_function: Optional[LossFunction],
        parameter_blocks: Sequence[ParameterBlock | torch.Tensor],
        *,
        name: Optional[str] = None,
    ) -> ResidualBlock:
        if not isinstance(cost_function, CostFunction):
            cost_function = CallableCostFunction(cost_function)
        blocks = [self._coerce_parameter_block(p) for p in parameter_blocks]
        if cost_function.parameter_block_sizes is not None:
            expected = list(cost_function.parameter_block_sizes)
            got = [b.size for b in blocks]
            if expected != got:
                raise ValueError(f"Cost function parameter sizes {expected} do not match blocks {got}")
        residual = ResidualBlock(cost_function, loss_function, blocks, name=name)
        self.residual_blocks.append(residual)
        return residual

    AddResidualBlock = add_residual_block

    def remove_residual_block(self, residual_block: ResidualBlock) -> None:
        self.residual_blocks.remove(residual_block)

    RemoveResidualBlock = remove_residual_block

    def remove_parameter_block(self, parameter_block: ParameterBlock | torch.Tensor) -> None:
        block = self._coerce_parameter_block(parameter_block)
        self.residual_blocks = [rb for rb in self.residual_blocks if block not in rb.parameter_blocks]
        self.parameter_blocks.remove(block)
        self._tensor_to_block.pop(id(block.tensor), None)

    RemoveParameterBlock = remove_parameter_block

    def set_parameter_block_constant(self, parameter_block: ParameterBlock | torch.Tensor) -> None:
        self._coerce_parameter_block(parameter_block).constant = True

    SetParameterBlockConstant = set_parameter_block_constant

    def set_parameter_block_variable(self, parameter_block: ParameterBlock | torch.Tensor) -> None:
        self._coerce_parameter_block(parameter_block).constant = False

    SetParameterBlockVariable = set_parameter_block_variable

    def is_parameter_block_constant(self, parameter_block: ParameterBlock | torch.Tensor) -> bool:
        return self._coerce_parameter_block(parameter_block).constant

    IsParameterBlockConstant = is_parameter_block_constant

    def set_manifold(self, parameter_block: ParameterBlock | torch.Tensor, manifold: Optional[Manifold]) -> None:
        block = self._coerce_parameter_block(parameter_block)
        block.manifold = manifold or EuclideanManifold(block.size)
        if block.manifold.ambient_size != block.size:
            raise ValueError("Manifold ambient size must equal parameter block size")

    SetManifold = set_manifold

    def set_bounds(
        self,
        parameter_block: ParameterBlock | torch.Tensor,
        lower: Optional[torch.Tensor | float] = None,
        upper: Optional[torch.Tensor | float] = None,
    ) -> None:
        block = self._coerce_parameter_block(parameter_block)
        if lower is not None:
            block.lower_bound = torch.as_tensor(lower, dtype=block.dtype, device=block.device).broadcast_to(block.tensor.shape).clone()
        if upper is not None:
            block.upper_bound = torch.as_tensor(upper, dtype=block.dtype, device=block.device).broadcast_to(block.tensor.shape).clone()

    SetBounds = set_bounds

    def set_parameter_lower_bound(self, parameter_block: ParameterBlock | torch.Tensor, index: int, lower: float) -> None:
        block = self._coerce_parameter_block(parameter_block)
        lb = (
            block.lower_bound.clone()
            if block.lower_bound is not None
            else torch.full_like(block.tensor.reshape(-1), -torch.inf)
        )
        lb.reshape(-1)[index] = lower
        block.lower_bound = lb.reshape_as(block.tensor)

    SetParameterLowerBound = set_parameter_lower_bound

    def set_parameter_upper_bound(self, parameter_block: ParameterBlock | torch.Tensor, index: int, upper: float) -> None:
        block = self._coerce_parameter_block(parameter_block)
        ub = (
            block.upper_bound.clone()
            if block.upper_bound is not None
            else torch.full_like(block.tensor.reshape(-1), torch.inf)
        )
        ub.reshape(-1)[index] = upper
        block.upper_bound = ub.reshape_as(block.tensor)

    SetParameterUpperBound = set_parameter_upper_bound

    def get_parameter_blocks(self) -> list[ParameterBlock]:
        return list(self.parameter_blocks)

    GetParameterBlocks = get_parameter_blocks

    def get_residual_blocks(self) -> list[ResidualBlock]:
        return list(self.residual_blocks)

    GetResidualBlocks = get_residual_blocks

    def num_parameter_blocks(self) -> int:
        return len(self.parameter_blocks)

    NumParameterBlocks = num_parameter_blocks

    def num_parameters(self) -> int:
        return sum(b.size for b in self.parameter_blocks)

    NumParameters = num_parameters

    def num_effective_parameters(self) -> int:
        return sum(b.tangent_size for b in self.parameter_blocks)

    def num_residual_blocks(self) -> int:
        return len(self.residual_blocks)

    NumResidualBlocks = num_residual_blocks

    def num_residuals(self) -> int:
        return int(sum(self.evaluate_residual_block(rb, compute_jacobians=False).residuals.numel() for rb in self.residual_blocks))

    NumResiduals = num_residuals

    def evaluate(self, options: Optional[EvaluateOptions] = None, *, compute_jacobian: bool = True) -> EvaluationResult:
        options = options or EvaluateOptions()
        residual_blocks = list(options.residual_blocks or self.residual_blocks)
        active_blocks = self._selected_parameter_blocks(options.parameter_blocks)
        active_to_col, total_cols = self._active_column_map(active_blocks)

        cost_terms: list[torch.Tensor] = []
        residual_terms: list[torch.Tensor] = []
        jacobian_rows: list[torch.Tensor] = []

        dtype, device = self._default_dtype_device()
        for rb in residual_blocks:
            result = self._evaluate_residual_block_internal(
                rb,
                compute_jacobians=compute_jacobian,
                apply_loss=options.apply_loss_function,
                active_to_col=active_to_col,
                total_cols=total_cols,
            )
            cost_terms.append(result.cost)
            residual_terms.append(result.residuals)
            if compute_jacobian and result.jacobian is not None:
                jacobian_rows.append(result.jacobian)
            dtype, device = result.cost.dtype, result.cost.device

        cost = torch.stack(cost_terms).sum() if cost_terms else torch.zeros((), dtype=dtype, device=device)
        residuals = torch.cat(residual_terms) if residual_terms else torch.zeros(0, dtype=dtype, device=device)
        jacobian = torch.cat(jacobian_rows, dim=0) if compute_jacobian and jacobian_rows else None
        gradient = jacobian.T @ residuals if jacobian is not None else None
        return EvaluationResult(
            cost=cost,
            residuals=residuals,
            gradient=gradient,
            jacobian=jacobian,
            crs_jacobian=CRSMatrix.from_dense(jacobian) if jacobian is not None else None,
        )

    Evaluate = evaluate

    def evaluate_residual_block(
        self,
        residual_block: ResidualBlock,
        *,
        apply_loss_function: bool = True,
        compute_jacobians: bool = True,
    ) -> EvaluationResult:
        active_to_col, total_cols = self._active_column_map(self.parameter_blocks)
        return self._evaluate_residual_block_internal(
            residual_block,
            compute_jacobians=compute_jacobians,
            apply_loss=apply_loss_function,
            active_to_col=active_to_col,
            total_cols=total_cols,
        )

    EvaluateResidualBlock = evaluate_residual_block

    def snapshot(self) -> list[torch.Tensor]:
        return [b.clone_value() for b in self.parameter_blocks]

    def restore(self, snapshot: Sequence[torch.Tensor]) -> None:
        with torch.no_grad():
            for block, value in zip(self.parameter_blocks, snapshot):
                block.tensor.reshape(-1).copy_(value.to(dtype=block.dtype, device=block.device).reshape(-1))

    def apply_delta(self, delta: torch.Tensor, active_blocks: Optional[Sequence[ParameterBlock]] = None) -> None:
        active = list(active_blocks or [b for b in self.parameter_blocks if b.tangent_size > 0])
        offset = 0
        with torch.no_grad():
            for block in active:
                n = block.tangent_size
                d = delta[offset : offset + n].to(dtype=block.dtype, device=block.device)
                offset += n
                if n == 0:
                    continue
                updated = block.manifold.plus(block.tensor.detach().reshape(-1), d).reshape_as(block.tensor)
                updated = block.project_bounds(updated)
                block.tensor.reshape(-1).copy_(updated.reshape(-1))

    def parameter_tangent_slices(self, active_only: bool = True) -> dict[ParameterBlock, slice]:
        blocks = [b for b in self.parameter_blocks if (not active_only or b.tangent_size > 0)]
        slices: dict[ParameterBlock, slice] = {}
        offset = 0
        for block in blocks:
            n = block.tangent_size
            slices[block] = slice(offset, offset + n)
            offset += n
        return slices

    def _evaluate_residual_block_internal(
        self,
        rb: ResidualBlock,
        *,
        compute_jacobians: bool,
        apply_loss: bool,
        active_to_col: dict[ParameterBlock, slice],
        total_cols: int,
    ) -> EvaluationResult:
        params = [b.tensor.detach().clone() for b in rb.parameter_blocks]
        if compute_jacobians:
            residual, ambient_jacobians = rb.cost_function.compute_jacobians(params)
        else:
            residual = rb.cost_function.residuals(*params).detach().reshape(-1)
            ambient_jacobians = None

        residual = residual.reshape(-1)
        jacobian = None
        if compute_jacobians and ambient_jacobians is not None:
            jacobian = residual.new_zeros((residual.numel(), total_cols))
            for block, J_ambient in zip(rb.parameter_blocks, ambient_jacobians):
                if block not in active_to_col:
                    continue
                col_slice = active_to_col[block]
                J_ambient = J_ambient.to(dtype=residual.dtype, device=residual.device)
                plus_jacobian = block.manifold.plus_jacobian(block.tensor.detach().reshape(-1)).to(
                    dtype=residual.dtype, device=residual.device
                )
                jacobian[:, col_slice] = J_ambient @ plus_jacobian
        if apply_loss:
            cost, corrected_residual, jacobian = robustify_residual_and_jacobian(rb.loss_function, residual, jacobian)
        else:
            cost = 0.5 * torch.sum(residual * residual)
            corrected_residual = residual
        return EvaluationResult(cost=cost, residuals=corrected_residual, jacobian=jacobian)

    def _active_column_map(self, blocks: Sequence[ParameterBlock]) -> tuple[dict[ParameterBlock, slice], int]:
        active_to_col: dict[ParameterBlock, slice] = {}
        offset = 0
        for block in blocks:
            if block.tangent_size == 0:
                continue
            active_to_col[block] = slice(offset, offset + block.tangent_size)
            offset += block.tangent_size
        return active_to_col, offset

    def _selected_parameter_blocks(
        self, blocks: Optional[Sequence[ParameterBlock | torch.Tensor]]
    ) -> list[ParameterBlock]:
        return [self._coerce_parameter_block(b) for b in blocks] if blocks is not None else list(self.parameter_blocks)

    def _coerce_parameter_block(self, value: ParameterBlock | torch.Tensor) -> ParameterBlock:
        if isinstance(value, ParameterBlock):
            return value
        key = id(value)
        if key not in self._tensor_to_block:
            return self.add_parameter_block(value)
        return self._tensor_to_block[key]

    def _default_dtype_device(self) -> tuple[torch.dtype, torch.device]:
        if self.parameter_blocks:
            b = self.parameter_blocks[0]
            return b.dtype, b.device
        return torch.float64, torch.device("cpu")


def add_residual_blocks(problem: Problem, blocks: Iterable[ResidualBlock]) -> None:
    problem.residual_blocks.extend(blocks)
