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


def test_cgnr_accepts_ceres_preconditioner_families_as_diagonal_core_paths() -> None:
    A = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [1.0, 0.0, 2.0],
            [2.0, -1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    true_x = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    b = A @ true_x
    expected = tc.solve_linear_system(A, b, solver_type=tc.LinearSolverType.DENSE_QR).x

    for preconditioner in [
        tc.PreconditionerType.SCHUR_JACOBI,
        tc.PreconditionerType.SCHUR_POWER_SERIES_EXPANSION,
        tc.PreconditionerType.CLUSTER_JACOBI,
        tc.PreconditionerType.CLUSTER_TRIDIAGONAL,
        tc.PreconditionerType.SUBSET,
    ]:
        result = tc.solve_linear_system(
            A,
            b,
            solver_type=tc.LinearSolverType.CGNR,
            preconditioner_type=preconditioner,
            tolerance=1e-12,
            max_iterations=100,
        )
        assert result.summary.success
        torch.testing.assert_close(result.x, expected, atol=1e-8, rtol=1e-8)


def test_mixed_precision_iterative_refinement_recovers_double_solution() -> None:
    A = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0 + 1e-7],
            [1.0, 1.0 - 1e-7],
        ],
        dtype=torch.float64,
    )
    true_x = torch.tensor([2.0, -3.0], dtype=torch.float64)
    b = A @ true_x

    low_precision = tc.solve_linear_system(
        A,
        b,
        solver_type=tc.LinearSolverType.DENSE_QR,
        use_mixed_precision=True,
        max_refinement_iterations=0,
    )
    refined = tc.solve_linear_system(
        A,
        b,
        solver_type=tc.LinearSolverType.DENSE_QR,
        use_mixed_precision=True,
        max_refinement_iterations=5,
    )

    assert refined.summary.residual_norm < low_precision.summary.residual_norm
    torch.testing.assert_close(refined.x, true_x, atol=1e-6, rtol=1e-6)
