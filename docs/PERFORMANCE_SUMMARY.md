# Performance Summary

Run timestamp: 2026-04-29T21:20:58+03:00

Environment:

- OS shell: Windows PowerShell
- Conda environment: `env2`
- Python: 3.11.0
- PyTorch: 2.8.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- CPU: AMD Ryzen 7 9700X 8-Core Processor
- GPU: NVIDIA GeForce RTX 5070 Ti, driver 591.86, 16303 MiB

Validation commands:

```powershell
$env:CERES_TORCH_RUN_BENCHMARKS='1'
$env:CERES_TORCH_BENCHMARK_MAX_SECONDS='30'
conda run -n env2 python -m pytest tests\test_benchmarking.py -m performance -q
```

Result: 4 passed, 6 deselected.

The benchmark defaults were enlarged from smoke-sized matrices to medium workloads:

- Dense/CGNR linear: `2048x256`
- Dense/iterative Schur: `4096x640`, eliminating 512 variables
- Cluster-tridiagonal CGNR: `2048x256`, 16 blocks
- Sparse direct: `2000x320`, density `0.08`
- Curve fitting: 400 residual observations
- Dense covariance: `1000x160`

Default CPU benchmark command:

```powershell
conda run -n env2 python benchmarks\run_benchmarks.py --device cpu --dtype float64 --warmup 1 --repeats 3
```

| Benchmark | Device | Median (s) | Min (s) | Max (s) | Repeats | Metric |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `linear/DENSE_QR/2048x256` | cpu | 0.011366100 | 0.009673000 | 0.011418800 | 3 | 1.298809143e-12 |
| `linear/CGNR/2048x256` | cpu | 0.054742500 | 0.049490400 | 0.054885000 | 3 | 9.985342917e-10 |
| `schur/dense/4096x640/elim512` | cpu | 0.013351400 | 0.010278700 | 0.013394800 | 3 | 5.176453269e-12 |
| `schur/iterative/SCHUR_JACOBI/4096x640/elim512` | cpu | 0.025949200 | 0.020344700 | 0.028356900 | 3 | 5.116177559e-12 |
| `schur/iterative/SCHUR_POWER_SERIES_EXPANSION/spse_init/4096x640/elim512` | cpu | 0.037386900 | 0.036241000 | 0.039135800 | 3 | 3.533976227e-09 |
| `linear/CGNR/CLUSTER_TRIDIAGONAL/2048x256/blocks16` | cpu | 0.042535900 | 0.041992800 | 0.048768600 | 3 | 1.029618312e-09 |
| `solver/curve_fit/400` | cpu | 2.202827300 | 2.202827300 | 2.202827300 | 1 | 7.330748407e-14 |
| `covariance/dense_svd/1000x160` | cpu | 0.657943200 | 0.657943200 | 0.657943200 | 1 | 1.641717374e-02 |

Default CUDA benchmark command:

```powershell
conda run -n env2 python benchmarks\run_benchmarks.py --device cuda --dtype float64 --warmup 1 --repeats 3
```

| Benchmark | Device | Median (s) | Min (s) | Max (s) | Repeats | Metric |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `linear/DENSE_QR/2048x256` | cuda | 0.003874700 | 0.003828000 | 0.004616800 | 3 | 4.016452835e-13 |
| `linear/CGNR/2048x256` | cuda | 0.098517900 | 0.097872800 | 0.098554400 | 3 | 9.985338353e-10 |
| `schur/dense/4096x640/elim512` | cuda | 0.010122500 | 0.009882100 | 0.010192600 | 3 | 3.630059702e-12 |
| `schur/iterative/SCHUR_JACOBI/4096x640/elim512` | cuda | 0.018289100 | 0.018223800 | 0.018417400 | 3 | 3.613871476e-12 |
| `schur/iterative/SCHUR_POWER_SERIES_EXPANSION/spse_init/4096x640/elim512` | cuda | 0.030906500 | 0.030897400 | 0.031642200 | 3 | 3.533999831e-09 |
| `linear/CGNR/CLUSTER_TRIDIAGONAL/2048x256/blocks16` | cuda | 0.082315000 | 0.081834400 | 0.083603000 | 3 | 1.029739181e-09 |
| `solver/curve_fit/400` | cuda | 9.785884700 | 9.785884700 | 9.785884700 | 1 | 7.330748429e-14 |
| `covariance/dense_svd/1000x160` | cuda | 0.630727200 | 0.630727200 | 0.630727200 | 1 | 1.641717374e-02 |

Gate-specific backend benchmarks:

| Benchmark | Device | Median (s) | Min (s) | Max (s) | Repeats | Metric |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `linear/SPARSE_NORMAL_CHOLESKY/scipy/2000x320/density0.08` | cpu | 0.009300500 | 0.009217800 | 0.009578300 | 3 | 2.223122937e-13 |
| `cuda/block_schur/torch/4096x640/elim512` | cuda | 0.011166500 | 0.010420900 | 0.011232500 | 3 | 3.903393465e-12 |

Notes:

- The pytest performance gates passed with `CERES_TORCH_BENCHMARK_MAX_SECONDS=30`.
- CUDA tests used the PyTorch CUDA backend. Native CUDA extension build benchmarks remain opt-in through `CERES_TORCH_BUILD_CUDA_EXTENSIONS=1`.
