#include <torch/extension.h>

#include <ATen/ops/linalg_solve.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__global__ void normal_equations_kernel(const scalar_t* __restrict__ A,
                                        const scalar_t* __restrict__ b,
                                        const scalar_t* __restrict__ damping,
                                        scalar_t* __restrict__ H,
                                        scalar_t* __restrict__ rhs,
                                        int64_t rows,
                                        int64_t cols,
                                        bool has_damping) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t h_size = cols * cols;
  const int64_t total = h_size + cols;
  if (index >= total) {
    return;
  }

  if (index < h_size) {
    const int64_t col_i = index / cols;
    const int64_t col_j = index - col_i * cols;
    scalar_t sum = scalar_t(0);
    for (int64_t row = 0; row < rows; ++row) {
      sum += A[row * cols + col_i] * A[row * cols + col_j];
    }
    if (has_damping && col_i == col_j) {
      sum += damping[col_i];
    }
    H[index] = sum;
    return;
  }

  const int64_t col = index - h_size;
  scalar_t sum = scalar_t(0);
  for (int64_t row = 0; row < rows; ++row) {
    sum += A[row * cols + col] * b[row];
  }
  rhs[col] = sum;
}

std::tuple<torch::Tensor, torch::Tensor> build_normal_equations(torch::Tensor A,
                                                                torch::Tensor b,
                                                                torch::Tensor damping) {
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(A.dim() == 2, "A must be a matrix");
  TORCH_CHECK(b.dim() == 1, "b must be a vector");
  TORCH_CHECK(A.size(0) == b.size(0), "A rows must match b size");
  TORCH_CHECK(A.scalar_type() == b.scalar_type(), "A and b must have the same dtype");
  const bool has_damping = damping.numel() != 0;
  if (has_damping) {
    TORCH_CHECK(damping.is_cuda(), "damping must be a CUDA tensor");
    TORCH_CHECK(damping.dim() == 1, "damping must be a vector");
    TORCH_CHECK(damping.size(0) == A.size(1), "damping size must match A columns");
    TORCH_CHECK(damping.scalar_type() == A.scalar_type(), "damping dtype must match A dtype");
  }

  A = A.contiguous();
  b = b.contiguous();
  damping = damping.contiguous();

  const int64_t rows = A.size(0);
  const int64_t cols = A.size(1);
  auto H = torch::empty({cols, cols}, A.options());
  auto rhs = torch::empty({cols}, A.options());
  const int threads = 256;
  const int64_t total = cols * cols + cols;
  const int blocks = static_cast<int>((total + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "normal_equations_kernel", [&] {
    const scalar_t* damping_ptr = has_damping ? damping.data_ptr<scalar_t>() : nullptr;
    normal_equations_kernel<scalar_t><<<blocks, threads>>>(
        A.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        damping_ptr,
        H.data_ptr<scalar_t>(),
        rhs.data_ptr<scalar_t>(),
        rows,
        cols,
        has_damping);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {H, rhs};
}

torch::Tensor solve_square(torch::Tensor A, torch::Tensor b) {
  if (b.dim() == 1) {
    return at::linalg_solve(A, b.unsqueeze(1)).squeeze(1);
  }
  return at::linalg_solve(A, b);
}

}  // namespace

torch::Tensor normal_equations_solve_cuda(torch::Tensor A, torch::Tensor b, torch::Tensor damping) {
  auto [H, rhs] = build_normal_equations(A, b, damping);
  return solve_square(H, rhs);
}

torch::Tensor block_schur_solve_cuda(torch::Tensor A,
                                     torch::Tensor b,
                                     int64_t num_eliminate,
                                     torch::Tensor damping) {
  auto [H, rhs] = build_normal_equations(A, b, damping);
  const int64_t cols = H.size(0);
  if (num_eliminate <= 0 || num_eliminate >= cols) {
    return solve_square(H, rhs);
  }

  const int64_t e = num_eliminate;
  auto Haa = H.slice(0, 0, e).slice(1, 0, e);
  auto Hab = H.slice(0, 0, e).slice(1, e, cols);
  auto Hba = H.slice(0, e, cols).slice(1, 0, e);
  auto Hbb = H.slice(0, e, cols).slice(1, e, cols);
  auto ga = rhs.slice(0, 0, e);
  auto gb = rhs.slice(0, e, cols);

  auto Haa_inv_Hab = solve_square(Haa, Hab);
  auto Haa_inv_ga = solve_square(Haa, ga);
  auto S = Hbb - torch::matmul(Hba, Haa_inv_Hab);
  auto rhs_b = gb - torch::matmul(Hba, Haa_inv_ga);
  auto xb = solve_square(S, rhs_b);
  auto xa = Haa_inv_ga - torch::matmul(Haa_inv_Hab, xb);
  return torch::cat({xa, xb}, 0);
}
