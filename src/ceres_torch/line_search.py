from __future__ import annotations

import math
from typing import Any

from .types import LineSearchInterpolationType


def next_line_search_step_size(
    options: Any,
    *,
    step_size: float,
    cost: float,
    candidate_cost: float,
    directional_derivative: float,
    previous_step_size: float | None,
    previous_candidate_cost: float | None,
) -> float:
    if options.line_search_interpolation_type is LineSearchInterpolationType.BISECTION:
        next_step_size = 0.5 * step_size
    elif options.line_search_interpolation_type is LineSearchInterpolationType.QUADRATIC:
        next_step_size = quadratic_interpolation_step(step_size, cost, candidate_cost, directional_derivative)
    else:
        next_step_size = cubic_interpolation_step(
            step_size,
            cost,
            candidate_cost,
            directional_derivative,
            previous_step_size,
            previous_candidate_cost,
        )
    lower_factor = max(0.5, min(options.max_line_search_step_contraction, options.min_line_search_step_contraction))
    lower = step_size * lower_factor
    upper = step_size * max(options.max_line_search_step_contraction, options.min_line_search_step_contraction)
    return min(max(next_step_size, lower), upper)


def quadratic_interpolation_step(
    step_size: float,
    cost: float,
    candidate_cost: float,
    directional_derivative: float,
) -> float:
    denom = 2.0 * (candidate_cost - cost - directional_derivative * step_size)
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.5 * step_size
    step = -(directional_derivative * step_size * step_size) / denom
    return step if math.isfinite(step) and step > 0.0 else 0.5 * step_size


def cubic_interpolation_step(
    step_size: float,
    cost: float,
    candidate_cost: float,
    directional_derivative: float,
    previous_step_size: float | None,
    previous_candidate_cost: float | None,
) -> float:
    if previous_step_size is None or previous_candidate_cost is None or previous_step_size == step_size:
        return quadratic_interpolation_step(step_size, cost, candidate_cost, directional_derivative)
    t1, t2 = previous_step_size, step_size
    y1 = previous_candidate_cost - cost - directional_derivative * t1
    y2 = candidate_cost - cost - directional_derivative * t2
    det = t1 * t1 * t2 * t2 * (t2 - t1)
    if det == 0.0 or not math.isfinite(det):
        return quadratic_interpolation_step(step_size, cost, candidate_cost, directional_derivative)
    c2 = (y1 * t2**3 - y2 * t1**3) / det
    c3 = (y2 * t1 * t1 - y1 * t2 * t2) / det
    if not math.isfinite(c2) or not math.isfinite(c3):
        return quadratic_interpolation_step(step_size, cost, candidate_cost, directional_derivative)
    if abs(c3) < 1e-30:
        return quadratic_interpolation_step(step_size, cost, candidate_cost, directional_derivative)
    discriminant = c2 * c2 - 3.0 * c3 * directional_derivative
    if discriminant < 0.0:
        return quadratic_interpolation_step(step_size, cost, candidate_cost, directional_derivative)
    roots = [
        (-c2 + math.sqrt(discriminant)) / (3.0 * c3),
        (-c2 - math.sqrt(discriminant)) / (3.0 * c3),
    ]
    candidates = [root for root in roots if math.isfinite(root) and root > 0.0 and root < step_size]
    return min(candidates, default=quadratic_interpolation_step(step_size, cost, candidate_cost, directional_derivative))
