import torch

import ceres_torch as tc
import torch_ceres as old_tc


class RecordingEvaluationCallback:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []

    def prepare_for_evaluation(self, evaluate_jacobians: bool, new_evaluation_point: bool) -> None:
        self.calls.append((evaluate_jacobians, new_evaluation_point))


def test_compatibility_import_path_points_at_renamed_package() -> None:
    assert old_tc.Problem is tc.Problem


def test_problem_public_modeling_introspection() -> None:
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    y = torch.tensor([4.0], dtype=torch.float64)
    loss = tc.HuberLoss(1.0)
    cost = tc.AutoDiffCostFunction(lambda x, y: torch.stack([x[0] + y[0], x[2] - y[0]]), [3, 1], 2)

    problem = tc.Problem()
    x_block = problem.AddParameterBlock(x)
    residual = problem.AddResidualBlock(cost, loss, [x, y])

    assert problem.HasParameterBlock(x)
    assert not problem.HasParameterBlock(torch.tensor([1.0], dtype=torch.float64))
    assert problem.ParameterBlockSize(x) == 3
    assert problem.ParameterBlockTangentSize(x) == 3
    residual_parameters = problem.GetParameterBlocksForResidualBlock(residual)
    assert residual_parameters[0] is x_block
    assert residual_parameters[1].tensor is y
    assert problem.GetResidualBlocksForParameterBlock(x) == [residual]
    assert problem.GetCostFunctionForResidualBlock(residual) is cost
    assert problem.GetLossFunctionForResidualBlock(residual) is loss
    assert problem.NumResiduals() == 2

    problem.SetParameterLowerBound(x, 1, -2.5)
    problem.SetParameterUpperBound(x, 1, 5.5)
    assert problem.GetParameterLowerBound(x, 1) == -2.5
    assert problem.GetParameterUpperBound(x, 1) == 5.5
    assert problem.GetParameterLowerBound(x, 0) == -torch.finfo(torch.float64).max
    assert problem.GetParameterUpperBound(x, 0) == torch.finfo(torch.float64).max


def test_manifold_introspection_separates_explicit_manifold_from_constant_state() -> None:
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    manifold = tc.SubsetManifold(3, [1])
    problem = tc.Problem()
    problem.AddParameterBlock(x)

    assert not problem.HasManifold(x)
    assert problem.GetManifold(x) is None

    problem.SetManifold(x, manifold)
    assert problem.HasManifold(x)
    assert problem.GetManifold(x) is manifold
    assert problem.ParameterBlockTangentSize(x) == 2

    problem.SetParameterBlockConstant(x)
    assert problem.IsParameterBlockConstant(x)
    assert problem.ParameterBlockTangentSize(x) == 2
    assert problem.NumEffectiveParameters() == 0

    problem.SetManifold(x, None)
    assert not problem.HasManifold(x)
    assert problem.GetManifold(x) is None


def test_evaluation_callback_tracks_jacobian_and_point_freshness_without_num_residuals_side_effects() -> None:
    callback = RecordingEvaluationCallback()
    x = torch.tensor([2.0], dtype=torch.float64)
    cost = tc.AutoDiffCostFunction(lambda x: x * x - 4.0, [1], 1)
    problem = tc.Problem(evaluation_callback=callback)
    residual = problem.AddResidualBlock(cost, None, [x])

    assert problem.NumResiduals() == 1
    assert callback.calls == []

    problem.Evaluate(compute_jacobian=True)
    problem.EvaluateResidualBlockAssumingParametersUnchanged(residual, compute_jacobians=False)

    assert callback.calls == [(True, True), (False, False)]


def test_crs_matrix_roundtrip_to_dense() -> None:
    dense = torch.tensor([[1.0, 0.0, 2.0], [0.0, -3.0, 0.0]], dtype=torch.float64)
    crs = tc.CRSMatrix.from_dense(dense)
    torch.testing.assert_close(crs.to_dense(dtype=torch.float64), dense)
