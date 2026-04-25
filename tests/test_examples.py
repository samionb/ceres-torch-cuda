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


def test_hello_world_example_variants_converge() -> None:
    for name in ["hello_world", "hello_world_analytic_diff", "hello_world_numeric_diff"]:
        module = load_example(name)
        summary, x = module.run()

        assert summary.IsSolutionUsable()
        torch.testing.assert_close(x, torch.tensor([10.0], dtype=torch.float64), atol=1e-6, rtol=1e-6)


def test_curve_fitting_example_converges() -> None:
    module = load_example("curve_fitting")
    summary, m, c = module.run()

    assert summary.IsSolutionUsable()
    torch.testing.assert_close(m, torch.tensor([1.7], dtype=torch.float64), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(c, torch.tensor([0.3], dtype=torch.float64), atol=1e-3, rtol=1e-3)


def test_robust_curve_fitting_reduces_outlier_bias() -> None:
    module = load_example("robust_curve_fitting")
    plain_summary, plain_m, plain_c = module.run(use_robust_loss=False)
    robust_summary, robust_m, robust_c = module.run(use_robust_loss=True)

    target = torch.tensor([0.3, 0.1], dtype=torch.float64)
    plain_error = torch.linalg.norm(torch.cat([plain_m, plain_c]) - target)
    robust_error = torch.linalg.norm(torch.cat([robust_m, robust_c]) - target)

    assert plain_summary.IsSolutionUsable()
    assert robust_summary.IsSolutionUsable()
    assert robust_error < plain_error


def test_rosenbrock_example_variants_converge() -> None:
    for name in ["rosenbrock", "rosenbrock_analytic_diff", "rosenbrock_numeric_diff"]:
        module = load_example(name)
        summary, x = module.run()

        assert summary.IsSolutionUsable()
        torch.testing.assert_close(x, torch.tensor([1.0, 1.0], dtype=torch.float64), atol=1e-4, rtol=1e-4)


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


def test_bicubic_interpolation_example_recovers_shift() -> None:
    module = load_example("bicubic_interpolation")
    summary, estimated_shift, true_shift = module.run()

    assert summary.IsSolutionUsable()
    torch.testing.assert_close(estimated_shift, true_shift, atol=1e-8, rtol=1e-8)
