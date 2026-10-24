#!/usr/bin/env python3
"""Compare CPU and CUDA matrix multiplication under explicit conditions."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def measure(device: str, size: int, dtype: torch.dtype, warmup: int, repeats: int) -> dict:
    generator = torch.Generator(device=device).manual_seed(42)
    left = torch.randn((size, size), device=device, dtype=dtype, generator=generator)
    right = torch.randn((size, size), device=device, dtype=dtype, generator=generator)

    for _ in range(warmup):
        _ = left @ right
    if device == "cuda":
        torch.cuda.synchronize()

    samples_ms = []
    checksum = 0.0
    for _ in range(repeats):
        started = time.perf_counter()
        result = left @ right
        if device == "cuda":
            torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - started) * 1000)
        checksum = float(result[0, 0])

    median_ms = statistics.median(samples_ms)
    operations = 2 * size**3
    return {
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
        "size": size,
        "warmup": warmup,
        "repeats": repeats,
        "samples_ms": [round(value, 3) for value in samples_ms],
        "median_ms": round(median_ms, 3),
        "estimated_tflops": round(operations / (median_ms / 1000) / 1e12, 3),
        "checksum_first_element": checksum,
        "peak_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        if device == "cuda"
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(42)
    results = [measure("cpu", args.size, torch.float32, 1, args.repeats)]
    if torch.cuda.is_available():
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            torch.cuda.reset_peak_memory_stats()
            results.append(measure("cuda", args.size, dtype, args.warmup, args.repeats))

    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "notes": [
            "Timing excludes tensor allocation and CPU-GPU transfer.",
            "Median is reported after warm-up; TFLOPS is 2*N^3/time.",
            "Different dtypes have different numeric precision.",
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
