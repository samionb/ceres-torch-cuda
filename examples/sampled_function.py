import torch

import ceres_torch as tc


def run(
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.linspace(0.0, 6.0, 13, dtype=dtype)
    values = torch.sin(xs)
    interpolator = tc.CubicInterpolator(tc.Grid1D(values, x0=0.0, spacing=0.5))

    x = torch.tensor(2.25, dtype=dtype)
    value, derivative = interpolator.evaluate_with_derivative(x)
    return x, value, derivative, torch.cos(x)


def main() -> None:
    x, value, derivative, _expected_derivative = run()
    print(f"f({x.item():.2f})={value.item():.8f} df/dx={derivative.item():.8f}")


if __name__ == "__main__":
    main()
