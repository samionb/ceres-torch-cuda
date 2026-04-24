from __future__ import annotations

from typing import Protocol

from .types import CallbackReturnType, IterationSummary


class IterationCallback(Protocol):
    def __call__(self, summary: IterationSummary) -> CallbackReturnType:
        ...


class EvaluationCallback(Protocol):
    def prepare_for_evaluation(self, evaluate_jacobians: bool, new_evaluation_point: bool) -> None:
        ...


class LoggingCallback:
    def __init__(self, to_stdout: bool = True) -> None:
        self.to_stdout = to_stdout

    def __call__(self, summary: IterationSummary) -> CallbackReturnType:
        if self.to_stdout:
            print(
                f"{summary.iteration:4d}: "
                f"f:{summary.cost: .6e} "
                f"d:{summary.cost_change: .3e} "
                f"g:{summary.gradient_max_norm: .3e} "
                f"h:{summary.step_norm: .3e} "
                f"rho:{summary.relative_decrease: .3e} "
                f"mu:{summary.trust_region_radius: .3e}"
            )
        return CallbackReturnType.SOLVER_CONTINUE
