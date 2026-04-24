import torch

import ceres_torch as tc


def main() -> None:
    xs = torch.linspace(0.0, 6.0, 13, dtype=torch.float64)
    values = torch.sin(xs)
    interpolator = tc.CubicInterpolator(tc.Grid1D(values, x0=0.0, spacing=0.5))

    x = torch.tensor(2.25, dtype=torch.float64)
    value, derivative = interpolator.evaluate_with_derivative(x)
    print(f"f({x.item():.2f})={value.item():.8f} df/dx={derivative.item():.8f}")


if __name__ == "__main__":
    main()
