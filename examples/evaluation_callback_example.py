import torch

import ceres_torch as tc


class RecordingEvaluationCallback:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []

    def prepare_for_evaluation(self, evaluate_jacobians: bool, new_evaluation_point: bool) -> None:
        self.calls.append((evaluate_jacobians, new_evaluation_point))


def run() -> tuple[tc.EvaluationResult, RecordingEvaluationCallback]:
    callback = RecordingEvaluationCallback()
    x = torch.tensor([2.0], dtype=torch.float64)
    problem = tc.Problem(evaluation_callback=callback)
    residual = problem.AddResidualBlock(tc.AutoDiffCostFunction(lambda x: x * x - 4.0, [1], 1), None, [x])
    problem.EvaluateResidualBlockAssumingParametersUnchanged(residual, compute_jacobians=False)
    result = problem.Evaluate(compute_jacobian=True)
    return result, callback


def main() -> None:
    result, callback = run()
    print(f"cost={result.cost.item():.8f} callback_calls={callback.calls}")


if __name__ == "__main__":
    main()
