import torch
import pytest

import ceres_torch as tc


@pytest.mark.parametrize(
    "solver_type",
    [tc.LinearSolverType.DENSE_QR, tc.LinearSolverType.DENSE_NORMAL_CHOLESKY],
)
def test_dense_linear_solvers_match_full_rank_least_squares(solver_type: tc.LinearSolverType) -> None:
    A = torch.tensor(
        [
            [3.0, 0.0, 1.0],
            [0.0, 4.0, -1.0],
            [2.0, 0.0, 0.0],
            [0.0, -1.0, 5.0],
        ],
        dtype=torch.float64,
    )
    expected = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    b = A @ expected

    result = tc.solve_linear_system(A, b, solver_type=solver_type)

    assert result.summary.success
    assert result.summary.residual_norm < 1e-10
    torch.testing.assert_close(result.x, expected, atol=1e-10, rtol=1e-10)


def test_dense_qr_and_normal_cholesky_match_with_damping() -> None:
    A = torch.tensor(
        [
            [2.0, 0.0],
            [1.0, 3.0],
            [0.5, -1.0],
        ],
        dtype=torch.float64,
    )
    b = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    damping = torch.tensor([0.2, 0.4], dtype=torch.float64)

    qr = tc.solve_linear_system(A, b, solver_type=tc.LinearSolverType.DENSE_QR, damping=damping)
    cholesky = tc.solve_linear_system(A, b, solver_type=tc.LinearSolverType.DENSE_NORMAL_CHOLESKY, damping=damping)

    assert qr.summary.success
    assert cholesky.summary.success
    torch.testing.assert_close(cholesky.x, qr.x, atol=1e-10, rtol=1e-10)


def test_dense_normal_cholesky_rank_deficient_fallback_matches_qr() -> None:
    A = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ],
        dtype=torch.float64,
    )
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    qr = tc.solve_linear_system(A, b, solver_type=tc.LinearSolverType.DENSE_QR)
    cholesky = tc.solve_linear_system(A, b, solver_type=tc.LinearSolverType.DENSE_NORMAL_CHOLESKY)

    assert cholesky.summary.success
    assert "fallback" in cholesky.summary.message
    torch.testing.assert_close(cholesky.x, qr.x, atol=1e-10, rtol=1e-10)


def test_dense_linear_solver_validates_shapes_and_zero_column_systems() -> None:
    A = torch.zeros((3, 0), dtype=torch.float64)
    b = torch.tensor([1.0, -2.0, 2.0], dtype=torch.float64)
    result = tc.solve_linear_system(A, b)

    assert result.x.shape == torch.Size([0])
    assert result.summary.message == "zero-column system"
    assert result.summary.residual_norm == pytest.approx(3.0)

    with pytest.raises(ValueError, match="2D"):
        tc.solve_linear_system(torch.zeros(3, dtype=torch.float64), b)
    with pytest.raises(ValueError, match="one entry per row"):
        tc.solve_linear_system(torch.eye(2, dtype=torch.float64), b)


def test_subspace_dogleg_solves_reduced_model_no_worse_than_traditional() -> None:
    J = torch.tensor(
        [
            [1.0, 0.2],
            [0.3, 1.7],
            [2.0, -0.5],
            [-0.4, 1.2],
        ],
        dtype=torch.float64,
    )
    r = torch.tensor([1.0, -2.0, 0.5, 1.5], dtype=torch.float64)
    radius = 0.35

    traditional = tc.dogleg_step(J, r, radius, dogleg_type=tc.DoglegType.TRADITIONAL_DOGLEG)
    subspace = tc.dogleg_step(J, r, radius, dogleg_type=tc.DoglegType.SUBSPACE_DOGLEG)

    def model_cost(step: torch.Tensor) -> torch.Tensor:
        linearized = r + J @ step
        return 0.5 * torch.dot(linearized, linearized)

    assert torch.linalg.norm(subspace) <= radius + 1e-12
    assert model_cost(subspace) <= model_cost(traditional) + 1e-12


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


def test_iterative_schur_matches_dense_schur_solution() -> None:
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

    dense = tc.solve_linear_system(
        A,
        b,
        solver_type=tc.LinearSolverType.DENSE_SCHUR,
        damping=damping,
        num_eliminate=2,
    )
    iterative = tc.solve_linear_system(
        A,
        b,
        solver_type=tc.LinearSolverType.ITERATIVE_SCHUR,
        damping=damping,
        num_eliminate=2,
        preconditioner_type=tc.PreconditionerType.SCHUR_JACOBI,
        block_sizes=[1, 1, 2],
        max_iterations=20,
        tolerance=1e-14,
    )

    assert iterative.summary.success
    assert iterative.summary.num_iterations > 0
    assert iterative.summary.message.startswith("iterative schur")
    torch.testing.assert_close(iterative.x, dense.x, atol=1e-10, rtol=1e-10)


def test_iterative_solvers_honor_minimum_iteration_count() -> None:
    A = torch.eye(3, dtype=torch.float64)
    b = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)

    result = tc.solve_linear_system(
        A,
        b,
        solver_type=tc.LinearSolverType.CGNR,
        min_iterations=3,
        max_iterations=5,
        tolerance=1e-14,
    )

    assert result.summary.success
    assert result.summary.num_iterations == 3
    torch.testing.assert_close(result.x, b, atol=1e-12, rtol=1e-12)


def test_iterative_schur_uses_spse_initialization_when_requested() -> None:
    A = torch.tensor(
        [
            [3.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.0],
            [1.0, 1.0, 2.0, -1.0],
            [0.5, -1.0, 1.0, 2.0],
            [2.0, 0.0, -1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    b = torch.tensor([0.4, -1.0, 0.7, 1.5, -0.3], dtype=torch.float64)
    damping = torch.tensor([0.2, 0.3, 0.4, 0.5], dtype=torch.float64)

    cold = tc.solve_linear_system(
        A,
        b,
        solver_type=tc.LinearSolverType.ITERATIVE_SCHUR,
        damping=damping,
        num_eliminate=2,
        block_sizes=[1, 1, 2],
        max_iterations=0,
        use_spse_initialization=False,
    )
    warm = tc.solve_linear_system(
        A,
        b,
        solver_type=tc.LinearSolverType.ITERATIVE_SCHUR,
        damping=damping,
        num_eliminate=2,
        block_sizes=[1, 1, 2],
        max_iterations=0,
        max_num_spse_iterations=4,
        use_spse_initialization=True,
        spse_tolerance=0.0,
    )

    assert "spse_init/4" in warm.summary.message
    assert warm.summary.residual_norm < cold.summary.residual_norm


def test_schur_power_series_preconditioner_applies_truncated_series() -> None:
    A = torch.tensor(
        [
            [3.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.0],
            [1.0, 1.0, 2.0, -1.0],
            [0.5, -1.0, 1.0, 2.0],
            [2.0, 0.0, -1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    residual = torch.tensor([0.7, -1.2], dtype=torch.float64)
    damping = torch.tensor([0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
    preconditioner = tc.build_schur_complement_preconditioner(
        A,
        damping=damping,
        num_eliminate=2,
        preconditioner_type=tc.PreconditionerType.SCHUR_POWER_SERIES_EXPANSION,
        block_sizes=[1, 1, 2],
        max_num_spse_iterations=3,
        spse_tolerance=0.0,
    )

    H = A.T @ A + torch.diag(damping)
    Haa = H[:2, :2]
    Hab = H[:2, 2:]
    Hba = H[2:, :2]
    Hbb = H[2:, 2:]
    eliminated = Hba @ torch.linalg.solve(Haa, Hab)
    Hbb_inv = torch.linalg.inv(Hbb)
    expected = Hbb_inv @ residual
    term = expected
    for _ in range(2):
        term = Hbb_inv @ (eliminated @ term)
        expected = expected + term

    assert preconditioner.message == "schur_power_series/3"
    assert preconditioner.block_sizes == (2,)
    torch.testing.assert_close(preconditioner.apply(residual), expected, atol=1e-10, rtol=1e-10)


def test_cluster_tridiagonal_preconditioner_solves_ordered_block_band() -> None:
    A = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.5, -1.0, 2.0],
        ],
        dtype=torch.float64,
    )
    damping = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)
    residual = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    preconditioner = tc.build_normal_equation_preconditioner(
        A,
        damping=damping,
        preconditioner_type=tc.PreconditionerType.CLUSTER_TRIDIAGONAL,
        block_sizes=[1, 1, 1],
    )
    H = A.T @ A + torch.diag(damping)
    expected_matrix = H.clone()
    expected_matrix[0, 2] = 0.0
    expected_matrix[2, 0] = 0.0
    expected = torch.linalg.solve(expected_matrix, residual)

    assert preconditioner.message.startswith("cluster_tridiagonal/")
    assert preconditioner.block_sizes == (1, 1, 1)
    torch.testing.assert_close(preconditioner.apply(residual), expected, atol=1e-10, rtol=1e-10)


def test_block_jacobi_preconditioner_applies_exact_normal_blocks() -> None:
    A = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.0, 1.0, 4.0],
            [2.0, -1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    damping = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    residual = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    preconditioner = tc.build_normal_equation_preconditioner(
        A,
        damping=damping,
        preconditioner_type=tc.PreconditionerType.SCHUR_JACOBI,
        block_sizes=[2, 1],
    )
    H = A.T @ A + torch.diag(damping)
    expected = torch.cat(
        [
            torch.linalg.solve(H[:2, :2], residual[:2].reshape(-1, 1)).reshape(-1),
            torch.linalg.solve(H[2:, 2:], residual[2:].reshape(-1, 1)).reshape(-1),
        ]
    )

    assert preconditioner.message.startswith("block_jacobi/")
    assert preconditioner.block_sizes == (2, 1)
    torch.testing.assert_close(preconditioner.apply(residual), expected)


def test_block_sizes_must_match_column_count() -> None:
    A = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="block_sizes"):
        tc.build_normal_equation_preconditioner(
            A,
            preconditioner_type=tc.PreconditionerType.SCHUR_JACOBI,
            block_sizes=[2, 2],
        )


def test_cgnr_accepts_ceres_preconditioner_families_with_block_structure() -> None:
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
            block_sizes=[2, 1],
            tolerance=1e-12,
            max_iterations=100,
        )
        assert result.summary.success
        if preconditioner is tc.PreconditionerType.CLUSTER_TRIDIAGONAL:
            assert "cluster_tridiagonal/" in result.summary.message
        else:
            assert "block_jacobi/" in result.summary.message
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
