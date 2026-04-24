import pytest
import torch

import ceres_torch as tc


cuda = pytest.mark.skipif(not tc.cuda_available(), reason="CUDA is not available in this PyTorch environment")


@cuda
def test_cuda_problem_solve_preserves_device_and_dtype() -> None:
    x = torch.tensor([0.5], dtype=torch.float64, device="cuda")
    problem = tc.Problem()
    problem.AddResidualBlock(tc.AutoDiffCostFunction(lambda x: 10.0 - x, [1], 1), None, [x])

    summary = tc.solve(tc.SolverOptions(max_num_iterations=25, gradient_tolerance=1e-12), problem)

    assert summary.IsSolutionUsable()
    assert x.device.type == "cuda"
    assert x.dtype is torch.float64
    torch.testing.assert_close(x.cpu(), torch.tensor([10.0], dtype=torch.float64), atol=1e-6, rtol=1e-6)


@cuda
def test_cuda_covariance_dense_svd() -> None:
    x = torch.tensor([0.0], dtype=torch.float64, device="cuda")
    problem = tc.Problem()
    block = problem.AddParameterBlock(x)
    problem.AddResidualBlock(tc.AutoDiffCostFunction(lambda x: 2.0 * x - 1.0, [1], 1), None, [x])

    covariance = tc.Covariance()

    assert covariance.compute([(block, block)], problem)
    torch.testing.assert_close(
        covariance.get_covariance_block(block, block).cpu(),
        torch.tensor([[0.25]], dtype=torch.float64),
    )
