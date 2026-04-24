import torch

import ceres_torch as tc


class TerminateWhenSmall:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.calls = 0

    def __call__(self, summary: tc.IterationSummary) -> tc.CallbackReturnType:
        self.calls += 1
        if summary.cost <= self.threshold:
            return tc.CallbackReturnType.SOLVER_TERMINATE_SUCCESSFULLY
        return tc.CallbackReturnType.SOLVER_CONTINUE


def run() -> tuple[tc.SolverSummary, torch.Tensor, TerminateWhenSmall]:
    x = torch.tensor([0.5], dtype=torch.float64)
    problem = tc.Problem()
    problem.AddResidualBlock(tc.AutoDiffCostFunction(lambda x: 10.0 - x, [1], 1), None, [x])
    callback = TerminateWhenSmall(1e-6)
    summary = tc.solve(tc.SolverOptions(max_num_iterations=25, callbacks=[callback]), problem)
    return summary, x, callback


def main() -> None:
    summary, x, callback = run()
    print(summary.BriefReport())
    print(f"x={x.item():.8f} callback_calls={callback.calls}")


if __name__ == "__main__":
    main()
