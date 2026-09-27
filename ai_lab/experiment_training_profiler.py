#!/usr/bin/env python3
"""Locate FashionMNIST CNN training waits using normal timing and torch.profiler."""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiment_fashion_cnn import SmallCNN


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "results" / "training_profiler"
SEED = 42
BATCH_SIZE = 256
WARM_STEPS = 40
TIMED_STEPS = 100
PROFILE_STEPS = 20
ROUNDS = 3


def make_loader(workers: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.RandomResizedCrop(28, scale=(0.75, 1.0)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    dataset = datasets.FashionMNIST(DATA, train=True, download=True, transform=transform)
    arguments = dict(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    if workers:
        arguments["prefetch_factor"] = 2
    return DataLoader(**arguments)


def make_model(initial_state: dict) -> tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    model = SmallCNN().cuda()
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer, nn.CrossEntropyLoss()


def step(batch: tuple[torch.Tensor, torch.Tensor], model: nn.Module,
         optimizer: torch.optim.Optimizer, loss_fn: nn.Module, annotate: bool = False) -> None:
    if annotate:
        with record_function("h2d_transfer"):
            images, labels = (tensor.to("cuda", non_blocking=True) for tensor in batch)
        with record_function("forward_backward_adam"):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
    else:
        images, labels = (tensor.to("cuda", non_blocking=True) for tensor in batch)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()


def timed_run(workers: int, initial_state: dict, round_number: int) -> tuple[dict, list[dict]]:
    torch.manual_seed(SEED)
    loader = make_loader(workers)
    model, optimizer, loss_fn = make_model(initial_state)
    iterator = iter(loader)
    for _ in range(WARM_STEPS):
        step(next(iterator), model, optimizer, loss_fn)
    torch.cuda.synchronize()

    rows = []
    started = time.perf_counter()
    for step_index in range(TIMED_STEPS):
        before_next = time.perf_counter_ns()
        batch = next(iterator)
        after_next = time.perf_counter_ns()
        step(batch, model, optimizer, loss_fn)
        after_enqueue = time.perf_counter_ns()
        rows.append({
            "workers": workers,
            "round": round_number,
            "step": step_index + 1,
            "batch_wait_ms": (after_next - before_next) / 1e6,
            "enqueue_ms": (after_enqueue - after_next) / 1e6,
        })
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "workers": workers,
        "round": round_number,
        "timed_steps": TIMED_STEPS,
        "images": TIMED_STEPS * BATCH_SIZE,
        "elapsed_seconds": elapsed,
        "images_per_second": TIMED_STEPS * BATCH_SIZE / elapsed,
        "median_batch_wait_ms": statistics.median(row["batch_wait_ms"] for row in rows),
        "p95_batch_wait_ms": float(np.percentile([row["batch_wait_ms"] for row in rows], 95)),
    }
    del iterator, loader, model, optimizer
    torch.cuda.empty_cache()
    return result, rows


def profile_run(workers: int, initial_state: dict) -> dict:
    torch.manual_seed(SEED)
    loader = make_loader(workers)
    model, optimizer, loss_fn = make_model(initial_state)
    iterator = iter(loader)
    for _ in range(WARM_STEPS):
        step(next(iterator), model, optimizer, loss_fn)
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=False, profile_memory=False, with_stack=False) as prof:
        for _ in range(PROFILE_STEPS):
            with record_function("batch_wait"):
                batch = next(iterator)
            step(batch, model, optimizer, loss_fn, annotate=True)
            prof.step()
        torch.cuda.synchronize()
    trace_path = OUT / f"workers_{workers}_trace.json"
    prof.export_chrome_trace(str(trace_path))
    table_path = OUT / f"workers_{workers}_operator_table.txt"
    table_path.write_text(prof.key_averages().table(sort_by="self_device_time_total", row_limit=25), encoding="utf-8")

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    events = trace.get("traceEvents", [])
    stages = defaultdict(list)
    kernel_count = 0
    kernel_total_us = 0.0
    for event in events:
        if event.get("ph") != "X":
            continue
        duration = event.get("dur", 0)
        if event.get("cat") == "user_annotation" and event.get("name") in {
            "batch_wait", "h2d_transfer", "forward_backward_adam"
        }:
            stages[event["name"]].append(duration / 1000.0)
        if "kernel" in str(event.get("cat", "")).lower():
            kernel_count += 1
            kernel_total_us += duration
    result = {
        "workers": workers,
        "trace": str(trace_path),
        "trace_bytes": trace_path.stat().st_size,
        "operator_table": str(table_path),
        "profile_steps": PROFILE_STEPS,
        "stages": {name: {"count": len(values), "median_ms": statistics.median(values),
                          "p95_ms": float(np.percentile(values, 95))} for name, values in stages.items()},
        "kernel_events": kernel_count,
        "kernel_duration_sum_ms": kernel_total_us / 1000.0,
    }
    del iterator, loader, model, optimizer
    torch.cuda.empty_cache()
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the CPU/CUDA profiler comparison")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    initial_state = {key: value.clone() for key, value in SmallCNN().state_dict().items()}
    runs = []
    all_rows = []
    for round_number in range(1, ROUNDS + 1):
        order = (0, 4) if round_number % 2 else (4, 0)
        for workers in order:
            run, rows = timed_run(workers, initial_state, round_number)
            runs.append(run)
            all_rows.extend(rows)
            print(f"round={round_number} workers={workers} images/s={run['images_per_second']:.1f} "
                  f"batch_wait_median={run['median_batch_wait_ms']:.3f}ms", flush=True)
    with (OUT / "timed_steps_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["workers", "round", "step", "batch_wait_ms", "enqueue_ms"])
        writer.writeheader()
        writer.writerows(all_rows)

    traces = {str(workers): profile_run(workers, initial_state) for workers in (0, 4)}
    summary = {}
    for workers in (0, 4):
        relevant = [run for run in runs if run["workers"] == workers]
        summary[str(workers)] = {
            "rounds": ROUNDS,
            "median_images_per_second": statistics.median(run["images_per_second"] for run in relevant),
            "median_elapsed_seconds": statistics.median(run["elapsed_seconds"] for run in relevant),
            "median_of_round_batch_wait_medians_ms": statistics.median(run["median_batch_wait_ms"] for run in relevant),
        }
    payload = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0)},
        "protocol": {"seed": SEED, "dataset": "FashionMNIST train", "transform":
                     "RandomResizedCrop(28, scale=(0.75,1.0)), RandomRotation(15), ToTensor",
                     "batch_size": BATCH_SIZE, "workers": [0, 4], "pin_memory": True,
                     "persistent_workers_if_workers_positive": True, "prefetch_factor_if_workers_positive": 2,
                     "precision": "FP32, TF32 disabled", "warm_steps": WARM_STEPS,
                     "timed_steps": TIMED_STEPS, "timing_rounds": ROUNDS,
                     "profile_steps_separate": PROFILE_STEPS,
                     "batch_wait_definition": "wall time inside next(DataLoader iterator); may overlap GPU work",
                     "timed_elapsed_definition": "100 steps, CUDA synchronize at end; profiler disabled"},
        "runs": runs,
        "summary": summary,
        "profiler": traces,
    }
    (OUT / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "profiler": traces}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
