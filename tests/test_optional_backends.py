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
    calls: list[dict[str, object]] = []

    def backend(A: torch.Tensor, b: torch.Tensor, **kwargs: object) -> torch.Tensor:
        num_eliminate = int(kwargs["num_eliminate"])
        calls.append(kwargs)
        return tc.schur_solve_dense(A, b, num_eliminate)

    tc.register_optional_backend("sparse_schur", backend)
    try:
        result = tc.solve_linear_system(
            A,
            b,
            solver_type=tc.LinearSolverType.SPARSE_SCHUR,
            num_eliminate=1,
            min_iterations=2,
            max_iterations=9,
            tolerance=1e-4,
            preconditioner_type=tc.PreconditionerType.CLUSTER_TRIDIAGONAL,
            block_sizes=[1, 1],
            max_num_spse_iterations=4,
            use_spse_initialization=True,
            spse_tolerance=0.2,
            visibility=[{0}],
            visibility_clustering_type=tc.VisibilityClusteringType.SINGLE_LINKAGE,
        )
    finally:
        tc.unregister_optional_backend("sparse_schur")

    assert calls
    assert calls[0]["solver_type"] is tc.LinearSolverType.SPARSE_SCHUR
    assert calls[0]["num_eliminate"] == 1
    assert calls[0]["min_iterations"] == 2
    assert calls[0]["max_iterations"] == 9
    assert calls[0]["tolerance"] == 1e-4
    assert calls[0]["preconditioner_type"] is tc.PreconditionerType.CLUSTER_TRIDIAGONAL
    assert calls[0]["max_num_spse_iterations"] == 4
    assert calls[0]["use_spse_initialization"] is True
    assert calls[0]["spse_tolerance"] == 0.2
    assert calls[0]["visibility_clustering_type"] is tc.VisibilityClusteringType.SINGLE_LINKAGE
    assert "optional sparse_schur backend" in result.summary.message
    torch.testing.assert_close(result.x, dense.x)


def test_iterative_schur_backend_receives_full_solver_options() -> None:
    A = torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0], [1.0, 1.0, 2.0]], dtype=torch.float64)
    b = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    captured: list[dict[str, object]] = []

    def backend(A: torch.Tensor, b: torch.Tensor, **kwargs: object) -> torch.Tensor:
        captured.append(kwargs)
        return torch.linalg.lstsq(A, b).solution.reshape(-1)

    tc.register_optional_backend("iterative_schur", backend)
    try:
        result = tc.solve_linear_system(
            A,
            b,
            solver_type=tc.LinearSolverType.ITERATIVE_SCHUR,
            num_eliminate=1,
            min_iterations=2,
            max_iterations=7,
            tolerance=1e-5,
            preconditioner_type=tc.PreconditionerType.SCHUR_POWER_SERIES_EXPANSION,
            block_sizes=[1, 2],
            max_num_spse_iterations=4,
            use_spse_initialization=True,
            spse_tolerance=0.25,
        )
    finally:
        tc.unregister_optional_backend("iterative_schur")

    assert captured
    assert captured[0]["solver_type"] is tc.LinearSolverType.ITERATIVE_SCHUR
    assert captured[0]["num_eliminate"] == 1
    assert captured[0]["min_iterations"] == 2
    assert captured[0]["max_iterations"] == 7
    assert captured[0]["tolerance"] == 1e-5
    assert captured[0]["preconditioner_type"] is tc.PreconditionerType.SCHUR_POWER_SERIES_EXPANSION
    assert captured[0]["block_sizes"] == [1, 2]
    assert captured[0]["max_num_spse_iterations"] == 4
    assert captured[0]["use_spse_initialization"] is True
    assert captured[0]["spse_tolerance"] == 0.25
    assert "optional iterative_schur backend" in result.summary.message


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


def test_covariance_sparse_qr_rejects_backend_shape_mismatch() -> None:
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    problem = tc.Problem()
    block = problem.add_parameter_block(x)
    problem.add_residual_block(tc.NormalPrior(torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)), None, [x])

    def backend(J: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return torch.eye(J.shape[1] + 1, dtype=J.dtype, device=J.device)

    covariance = tc.Covariance(tc.CovarianceOptions(algorithm_type=tc.CovarianceAlgorithmType.SPARSE_QR))
    tc.register_optional_backend("sparse_qr_covariance", backend)
    try:
        assert not covariance.compute([(block, block)], problem)
    finally:
        tc.unregister_optional_backend("sparse_qr_covariance")

    assert "expected (2, 2)" in covariance.summary.message


def test_covariance_sparse_qr_rejects_non_tensor_backend_result() -> None:
    x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    problem = tc.Problem()
    block = problem.add_parameter_block(x)
    problem.add_residual_block(tc.NormalPrior(torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)), None, [x])

    def backend(_J: torch.Tensor, **_kwargs: object) -> list[float]:
        return [1.0, 0.0, 0.0, 1.0]

    covariance = tc.Covariance(tc.CovarianceOptions(algorithm_type=tc.CovarianceAlgorithmType.SPARSE_QR))
    tc.register_optional_backend("sparse_qr_covariance", backend)  # type: ignore[arg-type]
    try:
        assert not covariance.compute([(block, block)], problem)
    finally:
        tc.unregister_optional_backend("sparse_qr_covariance")

    assert "did not return a torch.Tensor" in covariance.summary.message


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


def test_suitesparseqr_covariance_backend_uses_qr_factorization(monkeypatch: pytest.MonkeyPatch) -> None:
    sp = pytest.importorskip("scipy.sparse")
    import numpy as np
    import ceres_torch.sparse_backends as sparse_backends

    class FakeSuiteSparseQR:
        def qr(self, matrix):
            dense = torch.as_tensor(matrix.toarray(), dtype=torch.float64)
            _Q, R = torch.linalg.qr(dense, mode="reduced")
            return None, sp.csc_matrix(R.numpy()), np.arange(matrix.shape[1])

    monkeypatch.setattr(sparse_backends, "_suitesparseqr_module", lambda: FakeSuiteSparseQR())
    J = torch.tensor(
        [
            [2.0, 0.0],
            [1.0, 3.0],
            [0.5, -1.0],
        ],
        dtype=torch.float64,
    )

    covariance = tc.suitesparseqr_sparse_qr_covariance(J)
    expected = torch.linalg.inv(J.T @ J)
    torch.testing.assert_close(covariance, expected, atol=1e-10, rtol=1e-10)


def test_suitesparseqr_registration_overrides_sparse_qr_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("scipy")
    import ceres_torch.sparse_backends as sparse_backends

    monkeypatch.setattr(sparse_backends, "_find_suitesparseqr_module_name", lambda: "sparseqr")
    info = tc.register_suitesparseqr_sparse_qr_backend()
    try:
        assert info.available
        assert info.backend == "sparseqr"
        assert info.registered == ("sparse_qr_covariance",)
        assert tc.get_optional_backend("sparse_qr_covariance") is tc.suitesparseqr_sparse_qr_covariance
    finally:
        tc.unregister_suitesparseqr_sparse_qr_backend()


def test_unregister_native_sparse_backends_clears_suitesparseqr_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    import ceres_torch.sparse_backends as sparse_backends

    monkeypatch.setattr(sparse_backends, "scipy_sparse_available", lambda: True)
    monkeypatch.setattr(sparse_backends, "_find_suitesparseqr_module_name", lambda: "sparseqr")

    tc.register_native_sparse_backends()
    assert tc.get_optional_backend("sparse_normal_cholesky") is not None
    assert tc.get_optional_backend("sparse_qr_covariance") is not None

    tc.unregister_native_sparse_backends()

    assert tc.get_optional_backend("sparse_normal_cholesky") is None
    assert tc.get_optional_backend("sparse_qr_covariance") is None
