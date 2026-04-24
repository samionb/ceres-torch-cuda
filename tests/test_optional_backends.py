import pytest
import torch

import ceres_torch as tc


def test_sparse_linear_solver_delegates_to_registered_backend() -> None:
    A = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]], dtype=torch.float64)
    expected = torch.tensor([1.0, -2.0], dtype=torch.float64)
    b = A @ expected
    calls: list[dict[str, object]] = []

    def backend(A: torch.Tensor, b: torch.Tensor, **kwargs: object) -> torch.Tensor:
        calls.append(kwargs)
        return torch.linalg.lstsq(A, b).solution.reshape(-1)

    tc.register_optional_backend("sparse_normal_cholesky", backend)
    try:
        result = tc.solve_linear_system(A, b, solver_type=tc.LinearSolverType.SPARSE_NORMAL_CHOLESKY)
    finally:
        tc.unregister_optional_backend("sparse_normal_cholesky")

    assert calls
    assert calls[0]["solver_type"] is tc.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    assert "optional sparse_normal_cholesky backend" in result.summary.message
    torch.testing.assert_close(result.x, expected)


def test_sparse_schur_delegates_to_registered_block_backend() -> None:
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0], [0.0, 1.0]], dtype=torch.float64)
    b = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    dense = tc.solve_linear_system(A, b, solver_type=tc.LinearSolverType.DENSE_SCHUR, num_eliminate=1)
    calls: list[int] = []

    def backend(A: torch.Tensor, b: torch.Tensor, **kwargs: object) -> torch.Tensor:
        num_eliminate = int(kwargs["num_eliminate"])
        calls.append(num_eliminate)
        return tc.schur_solve_dense(A, b, num_eliminate)

    tc.register_optional_backend("sparse_schur", backend)
    try:
        result = tc.solve_linear_system(
            A,
            b,
            solver_type=tc.LinearSolverType.SPARSE_SCHUR,
            num_eliminate=1,
        )
    finally:
        tc.unregister_optional_backend("sparse_schur")

    assert calls == [1]
    assert "optional sparse_schur backend" in result.summary.message
    torch.testing.assert_close(result.x, dense.x)


def test_covariance_sparse_qr_delegates_to_registered_backend() -> None:
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    problem = tc.Problem()
    block = problem.add_parameter_block(x)
    problem.add_residual_block(tc.NormalPrior(torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)), None, [x])
    calls: list[tuple[int, int]] = []

    def backend(J: torch.Tensor, **kwargs: object) -> torch.Tensor:
        calls.append((J.shape[0], J.shape[1]))
        return 3.0 * torch.eye(J.shape[1], dtype=J.dtype, device=J.device)

    covariance = tc.Covariance(tc.CovarianceOptions(algorithm_type=tc.CovarianceAlgorithmType.SPARSE_QR))
    tc.register_optional_backend("sparse_qr_covariance", backend)
    try:
        assert covariance.compute([(block, block)], problem)
    finally:
        tc.unregister_optional_backend("sparse_qr_covariance")

    assert calls == [(2, 2)]
    torch.testing.assert_close(covariance.get_covariance_block(block, block), 3.0 * torch.eye(2, dtype=torch.float64))


def test_optional_backend_registry_helpers() -> None:
    def backend(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return b

    tc.register_optional_backend("temporary", backend)
    assert tc.get_optional_backend("temporary") is backend
    tc.clear_optional_backends()
    assert tc.get_optional_backend("temporary") is None


def test_scipy_native_sparse_normal_backend_matches_dense_solver() -> None:
    pytest.importorskip("scipy")
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
    damping = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

    info = tc.register_native_sparse_backends()
    try:
        sparse = tc.solve_linear_system(
            A,
            b,
            solver_type=tc.LinearSolverType.SPARSE_NORMAL_CHOLESKY,
            damping=damping,
        )
        dense = tc.solve_linear_system(
            A,
            b,
            solver_type=tc.LinearSolverType.DENSE_NORMAL_CHOLESKY,
            damping=damping,
        )
    finally:
        tc.unregister_native_sparse_backends()

    assert info.available
    assert "sparse_normal_cholesky" in info.registered
    assert sparse.summary.message == "scipy sparse normal equations"
    torch.testing.assert_close(sparse.x, dense.x, atol=1e-10, rtol=1e-10)


def test_scipy_native_sparse_schur_backend_matches_dense_schur() -> None:
    pytest.importorskip("scipy")
    A = torch.tensor(
        [
            [2.0, 0.0, 1.0],
            [0.0, 3.0, -1.0],
            [1.0, -1.0, 2.0],
            [0.5, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    b = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)

    tc.register_scipy_sparse_backends()
    try:
        sparse = tc.solve_linear_system(
            A,
            b,
            solver_type=tc.LinearSolverType.SPARSE_SCHUR,
            num_eliminate=1,
        )
        dense = tc.solve_linear_system(
            A,
            b,
            solver_type=tc.LinearSolverType.DENSE_SCHUR,
            num_eliminate=1,
        )
    finally:
        tc.unregister_scipy_sparse_backends()

    assert sparse.summary.message == "scipy sparse schur"
    torch.testing.assert_close(sparse.x, dense.x, atol=1e-10, rtol=1e-10)


def test_scipy_native_sparse_covariance_backend_matches_dense_svd() -> None:
    pytest.importorskip("scipy")
    x = torch.zeros(3, dtype=torch.float64)
    A = torch.tensor(
        [
            [2.0, 0.0, 1.0],
            [0.0, 3.0, -1.0],
            [1.0, 1.0, 0.0],
            [0.5, -1.0, 2.0],
        ],
        dtype=torch.float64,
    )
    problem = tc.Problem()
    block = problem.add_parameter_block(x)
    problem.add_residual_block(tc.NormalPrior(A, torch.zeros(3, dtype=torch.float64)), None, [x])

    tc.register_native_sparse_backends()
    try:
        sparse = tc.Covariance(tc.CovarianceOptions(algorithm_type=tc.CovarianceAlgorithmType.SPARSE_QR))
        dense = tc.Covariance(tc.CovarianceOptions(algorithm_type=tc.CovarianceAlgorithmType.DENSE_SVD))
        assert sparse.compute([(block, block)], problem)
        assert dense.compute([(block, block)], problem)
    finally:
        tc.unregister_native_sparse_backends()

    torch.testing.assert_close(
        sparse.get_covariance_block(block, block),
        dense.get_covariance_block(block, block),
        atol=1e-10,
        rtol=1e-10,
    )
