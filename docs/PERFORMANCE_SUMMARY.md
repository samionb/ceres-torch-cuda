# Performance Summary

Run timestamp: 2026-04-29T21:46:06+03:00

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
$env:CERES_TORCH_BENCHMARK_MAX_SECONDS='120'
conda run -n env2 python -m pytest tests\test_benchmarking.py -m performance -q
```

Result: 4 passed, 6 deselected in 117.31 seconds.

The benchmark defaults were enlarged by roughly 100x more row/residual input data over the previous medium workload:

- Dense/CGNR linear: `204800x256`
- Dense/iterative Schur: `409600x640`, eliminating 512 variables
- Cluster-tridiagonal CGNR: `204800x256`, 16 blocks
- Sparse direct: `200000x320`, density `0.08`
- Curve fitting: 40000 residual observations with an analytic Jacobian
- Dense covariance: `100000x160`

Default CPU benchmark command:

```powershell
conda run -n env2 python benchmarks\run_benchmarks.py --device cpu --dtype float64 --warmup 1 --repeats 3
```

| Benchmark | Device | Median (s) | Min (s) | Max (s) | Repeats | Metric |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `linear/DENSE_QR/204800x256` | cpu | 1.671766700 | 1.633468900 | 1.697797900 | 3 | 9.159227711e-11 |
| `linear/CGNR/204800x256` | cpu | 0.376619300 | 0.372843200 | 0.378550600 | 3 | 6.169606910e-10 |
| `schur/dense/409600x640/elim512` | cpu | 1.168345000 | 1.157258800 | 1.178864700 | 3 | 4.256626386e-10 |
| `schur/iterative/SCHUR_JACOBI/409600x640/elim512` | cpu | 2.176115200 | 2.161323100 | 2.207530800 | 3 | 4.105658347e-10 |
| `schur/iterative/SCHUR_POWER_SERIES_EXPANSION/spse_init/409600x640/elim512` | cpu | 3.228644500 | 3.214516600 | 3.235038500 | 3 | 2.332721250e-09 |
| `linear/CGNR/CLUSTER_TRIDIAGONAL/204800x256/blocks16` | cpu | 0.354656500 | 0.354181800 | 0.357206700 | 3 | 6.889286204e-09 |
| `solver/curve_fit/40000` | cpu | 23.127901300 | 23.127901300 | 23.127901300 | 1 | 7.581149983e-12 |
| `covariance/dense_svd/100000x160` | cpu | 59.863862400 | 59.863862400 | 59.863862400 | 1 | 1.268016329e-04 |

Default CUDA benchmark command:

```powershell
conda run -n env2 python benchmarks\run_benchmarks.py --device cuda --dtype float64 --warmup 1 --repeats 3
```

| Benchmark | Device | Median (s) | Min (s) | Max (s) | Repeats | Metric |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `linear/DENSE_QR/204800x256` | cuda | 0.747905700 | 0.747705900 | 0.748769200 | 3 | 5.948415870e-12 |
| `linear/CGNR/204800x256` | cuda | 0.055129000 | 0.055065600 | 0.055531100 | 3 | 6.041733900e-10 |
| `schur/dense/409600x640/elim512` | cuda | 0.511351700 | 0.511266500 | 0.511624800 | 3 | 3.301662906e-10 |
| `schur/iterative/SCHUR_JACOBI/409600x640/elim512` | cuda | 1.013853600 | 1.013772400 | 1.014035100 | 3 | 2.924908557e-10 |
| `schur/iterative/SCHUR_POWER_SERIES_EXPANSION/spse_init/409600x640/elim512` | cuda | 1.517637800 | 1.516892200 | 1.517967300 | 3 | 2.326623096e-09 |
| `linear/CGNR/CLUSTER_TRIDIAGONAL/204800x256/blocks16` | cuda | 0.055655700 | 0.055649000 | 0.055656600 | 3 | 6.896990072e-09 |
| `solver/curve_fit/40000` | cuda | 21.552487600 | 21.552487600 | 21.552487600 | 1 | 7.581149983e-12 |
| `covariance/dense_svd/100000x160` | cuda | 60.578102400 | 60.578102400 | 60.578102400 | 1 | 1.268016329e-04 |

Gate-specific backend benchmarks:

| Benchmark | Device | Median (s) | Min (s) | Max (s) | Repeats | Metric |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `linear/SPARSE_NORMAL_CHOLESKY/scipy/200000x320/density0.08` | cpu | 0.960530200 | 0.952481200 | 0.969964800 | 3 | 1.181590163e-11 |
| `cuda/block_schur/torch/409600x640/elim512` | cuda | 0.511368800 | 0.511308900 | 0.511713100 | 3 | 3.015507426e-10 |

Notes:

- The pytest performance gates passed with `CERES_TORCH_BENCHMARK_MAX_SECONDS=120`.
- CUDA tests used the PyTorch CUDA backend. Native CUDA extension build benchmarks remain opt-in through `CERES_TORCH_BUILD_CUDA_EXTENSIONS=1`.
