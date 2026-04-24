import os
from pathlib import Path

import pytest
import torch

import ceres_torch as tc


RUN_CUDA_EXTENSION_BUILD = os.environ.get("CERES_TORCH_BUILD_CUDA_EXTENSIONS") == "1"


def test_cuda_extension_info_reports_source_files() -> None:
    info = tc.get_cuda_extension_info()

    assert info.backend == "cuda-extension"
    assert info.source_paths
    assert all(Path(path).exists() for path in info.source_paths)
    if not info.available:
        assert info.message


def test_cuda_extension_backends_reject_cpu_tensors() -> None:
    A = torch.eye(2, dtype=torch.float64)
    b = torch.ones(2, dtype=torch.float64)

    with pytest.raises(tc.OptionalBackendUnavailable, match="CUDA tensor"):
        tc.cuda_sparse_normal_cholesky(A, b)
    with pytest.raises(tc.OptionalBackendUnavailable, match="CUDA tensor"):
        tc.cuda_block_schur(A, b, num_eliminate=1)


def test_register_cuda_sparse_backends_is_safe_when_unavailable() -> None:
    info = tc.register_cuda_sparse_backends()
    try:
        if info.available:
            assert {"sparse_normal_cholesky", "sparse_schur", "block_schur"}.issubset(info.registered)
        else:
            assert info.registered == ()
            assert tc.get_optional_backend("block_schur") is None
    finally:
        tc.unregister_cuda_sparse_backends()


@pytest.mark.native_extension
@pytest.mark.skipif(not RUN_CUDA_EXTENSION_BUILD, reason="Set CERES_TORCH_BUILD_CUDA_EXTENSIONS=1 to build CUDA extension")
@pytest.mark.skipif(not tc.cuda_extension_build_available(), reason=tc.get_cuda_extension_info().message)
def test_cuda_extension_block_schur_matches_dense_solver() -> None:
    A = torch.tensor(
        [
            [2.0, 0.0, 1.0],
            [0.0, 3.0, -1.0],
            [1.0, -1.0, 2.0],
            [0.5, 0.0, 1.0],
        ],
        dtype=torch.float64,
        device="cuda",
    )
    b = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64, device="cuda")

    result = tc.cuda_block_schur(A, b, num_eliminate=1)
    dense = tc.solve_linear_system(A, b, solver_type=tc.LinearSolverType.DENSE_SCHUR, num_eliminate=1)

    assert result.summary.message == "cuda extension block schur"
    torch.testing.assert_close(result.x, dense.x, atol=1e-8, rtol=1e-8)
