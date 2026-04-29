from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ceres_torch as tc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ceres-torch performance benchmarks.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.device == "cuda" and not tc.cuda_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    results = tc.run_default_benchmarks(
        device=args.device,
        dtype=dtype,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    text = tc.format_benchmark_results(results)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
