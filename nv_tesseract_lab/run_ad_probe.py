"""Run NVIDIA's official AD Diffusion quick example and preserve raw evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sample_gpu(stop: threading.Event, samples: list[dict[str, float]]) -> None:
    command = [
        "/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    while not stop.is_set():
        try:
            raw = subprocess.check_output(command, text=True, timeout=2).strip()
            memory, utilization, power = [float(value.strip()) for value in raw.split(",")]
            samples.append(
                {
                    "elapsed_seconds": time.perf_counter(),
                    "memory_used_mib": memory,
                    "utilization_gpu_percent": utilization,
                    "power_draw_w": power,
                }
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        stop.wait(0.2)


def main() -> None:
    ad_root = Path.cwd()
    if not (ad_root / "examples" / "quick_example.py").exists():
        raise SystemExit("Run from NV-Tesseract/ad_diffusion")
    results_root = ad_root.parent.parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    label = os.environ.get("AD_PROBE_LABEL", "run")

    samples: list[dict[str, float]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop, samples), daemon=True)
    sampler.start()
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "examples/quick_example.py"],
        cwd=ad_root,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    stop.set()
    sampler.join(timeout=3)

    console_log = completed.stdout + completed.stderr
    console_path = results_root / f"ad_quick_example_console_{label}.log"
    console_path.write_text(console_log, encoding="utf-8")

    datasets = ad_root / "examples" / "datasets"
    results = pd.read_csv(datasets / "anomaly_results.csv")
    labels = pd.read_csv(datasets / "sample_timeseries_labels.csv")
    results_copy = results_root / f"ad_results_{label}.csv"
    labels_copy = results_root / f"ad_labels_{label}.csv"
    results.to_csv(results_copy, index=False)
    labels.to_csv(labels_copy, index=False)
    actual = labels["is_anomaly"].astype(bool)
    predicted = results["Anomaly"].astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    first_sample = samples[0]["elapsed_seconds"] if samples else 0.0
    for sample in samples:
        sample["elapsed_seconds"] -= first_sample
    raw = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": f"{sys.executable} examples/quick_example.py",
        "exit_code": completed.returncode,
        "total_elapsed_seconds": elapsed,
        "rows": len(results),
        "numeric_features": 3,
        "nsample": 15,
        "dpm_solver_steps": 20,
        "detected_anomalies": int(predicted.sum()),
        "ground_truth_anomalies": int(actual.sum()),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": {
            "mean": float(results["MAE"].mean()),
            "std": float(results["MAE"].std()),
            "max": float(results["MAE"].max()),
        },
        "gpu_samples": samples,
        "gpu_peak_memory_used_mib": max((s["memory_used_mib"] for s in samples), default=None),
        "gpu_peak_utilization_percent": max(
            (s["utilization_gpu_percent"] for s in samples), default=None
        ),
        "gpu_peak_power_draw_w": max((s["power_draw_w"] for s in samples), default=None),
        "console_log": str(console_path),
        "results_csv": str(results_copy),
        "labels_csv": str(labels_copy),
    }
    output = results_root / f"ad_probe_{label}.json"
    output.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in raw.items() if key != "gpu_samples"}, indent=2))
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
