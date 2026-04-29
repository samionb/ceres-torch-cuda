# Performance Summary

Run timestamp: 2026-04-29T21:07:18+03:00

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

Default CPU benchmark command:

```powershell
conda run -n env2 python benchmarks\run_benchmarks.py --device cpu --dtype float64 --warmup 1 --repeats 5
```

| Benchmark | Device | Median (s) | Min (s) | Max (s) | Repeats | Metric |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `linear/DENSE_QR/160x40` | cpu | 0.000116200 | 0.000112400 | 0.000221500 | 5 | 5.397476863e-14 |
| `linear/CGNR/160x40` | cpu | 0.006576700 | 0.006493600 | 0.007717100 | 5 | 2.110957117e-09 |
| `schur/dense/240x54/elim36` | cpu | 0.000183800 | 0.000172300 | 0.000226300 | 5 | 6.956253840e-14 |
| `schur/iterative/SCHUR_JACOBI/240x54/elim36` | cpu | 0.000356400 | 0.000353600 | 0.000440800 | 5 | 1.412476743e-13 |
| `schur/iterative/SCHUR_POWER_SERIES_EXPANSION/spse_init/240x54/elim36` | cpu | 0.001259900 | 0.001251000 | 0.001273400 | 5 | 5.104261089e-09 |
| `linear/CGNR/CLUSTER_TRIDIAGONAL/180x36/blocks6` | cpu | 0.007961600 | 0.007832000 | 0.008774300 | 5 | 1.095381894e-09 |
| `solver/curve_fit/80` | cpu | 0.479364000 | 0.445086100 | 0.479364000 | 2 | 2.304934798e-10 |
| `covariance/dense_svd/120x24` | cpu | 0.018430200 | 0.018348600 | 0.018430200 | 2 | 5.638091295e-02 |

Default CUDA benchmark command:

```powershell
conda run -n env2 python benchmarks\run_benchmarks.py --device cuda --dtype float64 --warmup 1 --repeats 5
```

| Benchmark | Device | Median (s) | Min (s) | Max (s) | Repeats | Metric |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `linear/DENSE_QR/160x40` | cuda | 0.001260500 | 0.001153700 | 0.001800600 | 5 | 2.914779495e-14 |
| `linear/CGNR/160x40` | cuda | 0.037074600 | 0.036675400 | 0.038554300 | 5 | 2.110958125e-09 |
| `schur/dense/240x54/elim36` | cuda | 0.000804600 | 0.000766800 | 0.000861500 | 5 | 6.519344555e-14 |
| `schur/iterative/SCHUR_JACOBI/240x54/elim36` | cuda | 0.001569700 | 0.001443500 | 0.001590200 | 5 | 1.013046115e-13 |
| `schur/iterative/SCHUR_POWER_SERIES_EXPANSION/spse_init/240x54/elim36` | cuda | 0.007181600 | 0.006876900 | 0.008052600 | 5 | 5.104283002e-09 |
| `linear/CGNR/CLUSTER_TRIDIAGONAL/180x36/blocks6` | cuda | 0.042060400 | 0.041532100 | 0.043308800 | 5 | 1.095367964e-09 |
| `solver/curve_fit/80` | cuda | 1.847808500 | 1.793656600 | 1.847808500 | 2 | 2.304934798e-10 |
| `covariance/dense_svd/120x24` | cuda | 0.016428700 | 0.015852800 | 0.016428700 | 2 | 5.638091295e-02 |

Gate-specific backend benchmarks:

| Benchmark | Device | Median (s) | Min (s) | Max (s) | Repeats | Metric |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `linear/SPARSE_NORMAL_CHOLESKY/scipy/64x16/density0.2` | cpu | 0.000646900 | 0.000626400 | 0.000784000 | 5 | 1.017484158e-14 |
| `linear/DENSE_QR/64x16` | cuda | 0.000595600 | 0.000566100 | 0.000799600 | 5 | 5.759854580e-15 |
| `cuda/block_schur/torch/64x20/elim12` | cuda | 0.001035600 | 0.001009200 | 0.001497000 | 5 | 2.007355686e-14 |

Notes:

- The pytest performance gates passed with `CERES_TORCH_BENCHMARK_MAX_SECONDS=30`.
- CUDA tests used the PyTorch CUDA backend. Native CUDA extension build benchmarks remain opt-in through `CERES_TORCH_BUILD_CUDA_EXTENSIONS=1`.
