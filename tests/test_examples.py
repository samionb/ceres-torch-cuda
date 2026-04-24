import importlib.util
from pathlib import Path

import torch

import ceres_torch as tc


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load_example(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_circle_fit_example_converges() -> None:
    module = load_example("circle_fit")
    summary, center, radius = module.run()

    assert summary.IsSolutionUsable()
    torch.testing.assert_close(center, torch.tensor([1.5, -2.0], dtype=torch.float64), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(radius, torch.tensor([3.0], dtype=torch.float64), atol=1e-6, rtol=1e-6)


def test_iteration_callback_example_terminates_through_callback() -> None:
    module = load_example("iteration_callback_example")
    summary, x, callback = module.run()

    assert summary.termination_type is tc.TerminationType.USER_SUCCESS
    assert callback.calls > 0
    torch.testing.assert_close(x, torch.tensor([10.0], dtype=torch.float64), atol=1e-3, rtol=1e-6)


def test_evaluation_callback_example_records_point_freshness() -> None:
    module = load_example("evaluation_callback_example")
    result, callback = module.run()

    torch.testing.assert_close(result.cost, torch.tensor(0.0, dtype=torch.float64))
    assert callback.calls == [(False, False), (True, True)]
