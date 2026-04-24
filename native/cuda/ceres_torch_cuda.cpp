#include <torch/extension.h>

torch::Tensor normal_equations_solve_cuda(torch::Tensor A, torch::Tensor b, torch::Tensor damping);
torch::Tensor block_schur_solve_cuda(torch::Tensor A,
                                     torch::Tensor b,
                                     int64_t num_eliminate,
                                     torch::Tensor damping);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("normal_equations_solve",
        &normal_equations_solve_cuda,
        "Solve damped normal equations using a CUDA normal-equation kernel");
  m.def("block_schur_solve",
        &block_schur_solve_cuda,
        "Solve a block-Schur normal system using CUDA normal-equation assembly");
}
