import torch

import ceres_torch as tc


def test_dense_schur_matches_damped_dense_qr_solution() -> None:
    A = torch.tensor(
        [
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 3.0, 0.0, 1.0],
            [1.0, -1.0, 2.0, 1.0],
            [0.0, 1.0, -1.0, 2.0],
            [1.0, 2.0, 0.0, -1.0],
            [3.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    b = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 2.0], dtype=torch.float64)
    damping = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)

    dense = tc.solve_linear_system(A, b, solver_type=tc.LinearSolverType.DENSE_QR, damping=damping)
    schur = tc.solve_linear_system(
        A,
        b,
        solver_type=tc.LinearSolverType.DENSE_SCHUR,
        damping=damping,
        num_eliminate=2,
    )

    assert schur.summary.success
    torch.testing.assert_close(schur.x, dense.x, atol=1e-10, rtol=1e-10)
