import torch

import torch_ceres as tc


def test_covariance_dense_svd_block() -> None:
    x = torch.tensor([0.0], dtype=torch.float64)
    problem = tc.Problem()
    block = problem.add_parameter_block(x)
    problem.add_residual_block(tc.AutoDiffCostFunction(lambda x: 2.0 * x - 1.0, [1]), None, [x])
    covariance = tc.Covariance(tc.CovarianceOptions(algorithm_type=tc.CovarianceAlgorithmType.DENSE_SVD))
    assert covariance.compute([(block, block)], problem)
    cov = covariance.get_covariance_block(block, block)
    torch.testing.assert_close(cov, torch.tensor([[0.25]], dtype=torch.float64), atol=1e-9, rtol=1e-9)


def test_cubic_interpolator_linear_data() -> None:
    grid = tc.Grid1D(torch.arange(5, dtype=torch.float64))
    interpolator = tc.CubicInterpolator(grid)
    torch.testing.assert_close(interpolator.evaluate(torch.tensor(2.5, dtype=torch.float64)), torch.tensor(2.5, dtype=torch.float64))

