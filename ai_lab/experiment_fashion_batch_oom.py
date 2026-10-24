#!/usr/bin/env python3
"""Measure FashionMNIST batch-size throughput and find an actual CUDA OOM boundary."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiment_fashion_cnn import SmallCNN


REAL_BATCHES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
OOM_BATCHES = [4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072]


def make_loader(data_dir: Path, batch_size: int) -> DataLoader:
    dataset = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        num_workers=0,
        pin_memory=True,
    )


def train_epoch(data_dir: Path, initial_state: dict, batch_size: int) -> dict:
    loader = make_loader(data_dir, batch_size)
    model = SmallCNN().cuda(); model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Compile/library warm-up at the same shape, on a disposable model.
    warm_model = copy.deepcopy(model)
    warm_opt = torch.optim.Adam(warm_model.parameters(), lr=1e-3)
    warm_images = torch.zeros(batch_size, 1, 28, 28, device="cuda")
    warm_labels = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    warm_opt.zero_grad(set_to_none=True)
    warm_loss = loss_fn(warm_model(warm_images), warm_labels)
    warm_loss.backward(); warm_opt.step(); torch.cuda.synchronize()
    del warm_model, warm_opt, warm_images, warm_labels, warm_loss
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    model.train(); seen = 0; steps = 0; loss_sum = 0.0
    torch.cuda.synchronize(); started = time.perf_counter()
    for images, labels in loader:
        images = images.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward(); optimizer.step()
        seen += labels.numel(); steps += 1; loss_sum += float(loss.detach()) * labels.size(0)
    torch.cuda.synchronize(); seconds = time.perf_counter() - started
    result = {
        "batch_size": batch_size,
        "samples": seen,
        "steps": steps,
        "seconds": seconds,
        "images_per_second": seen / seconds,
        "mean_step_ms": seconds / steps * 1000,
        "train_loss": loss_sum / seen,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }
    del model, optimizer, loader, loss, images, labels
    gc.collect(); torch.cuda.empty_cache()
    return result


def try_synthetic_step(batch_size: int, initial_state: dict) -> dict:
    model = optimizer = images = labels = loss = None
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    before_free, before_total = torch.cuda.mem_get_info()
    try:
        model = SmallCNN().cuda(); model.load_state_dict(initial_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        images = torch.zeros(batch_size, 1, 28, 28, device="cuda")
        labels = torch.zeros(batch_size, dtype=torch.long, device="cuda")
        torch.cuda.synchronize(); started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), labels)
        loss.backward(); optimizer.step(); torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        return {
            "batch_size": batch_size,
            "status": "ok",
            "step_seconds": seconds,
            "images_per_second": batch_size / seconds,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "free_before_mib": before_free / 1024**2,
            "total_mib": before_total / 1024**2,
        }
    except torch.OutOfMemoryError as exc:
        message = str(exc)
        requested = re.search(r"Tried to allocate ([0-9.]+ [GM]iB)", message)
        return {
            "batch_size": batch_size,
            "status": "oom",
            "exception": type(exc).__name__,
            "requested": requested.group(1) if requested else None,
            "message": message,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "free_before_mib": before_free / 1024**2,
            "total_mib": before_total / 1024**2,
        }
    finally:
        del model, optimizer, images, labels, loss
        gc.collect(); torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-oom-probes", action="store_true", help="Also run large synthetic OOM probes; can cause heavy GPU/host memory pressure.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.manual_seed(42)
    initial_state = copy.deepcopy(SmallCNN().state_dict())
    throughput_runs = {str(batch): [] for batch in REAL_BATCHES}
    for repeat in range(args.repeats):
        order = REAL_BATCHES if repeat % 2 == 0 else list(reversed(REAL_BATCHES))
        for batch in order:
            throughput_runs[str(batch)].append(train_epoch(args.data_dir, initial_state, batch))

    summary = {}
    for batch_text, runs in throughput_runs.items():
        ips = np.array([run["images_per_second"] for run in runs])
        seconds = np.array([run["seconds"] for run in runs])
        summary[batch_text] = {
            "median_images_per_second": float(np.median(ips)),
            "min_images_per_second": float(ips.min()),
            "max_images_per_second": float(ips.max()),
            "median_epoch_seconds": float(np.median(seconds)),
            "median_step_ms": float(np.median([run["mean_step_ms"] for run in runs])),
            "peak_allocated_mib": max(run["peak_allocated_mib"] for run in runs),
            "peak_reserved_mib": max(run["peak_reserved_mib"] for run in runs),
        }

    oom_runs = []
    for batch in (OOM_BATCHES if args.include_oom_probes else []):
        outcome = try_synthetic_step(batch, initial_state)
        oom_runs.append(outcome)
        if outcome["status"] == "oom":
            break

    # WSL/WDDM may allow allocations beyond the board's dedicated VRAM and
    # become extremely slow before a natural OOM. Add a reproducible allocator
    # limit so readers can inspect and recover from a real OutOfMemoryError
    # without exhausting host memory. This is not labeled as the physical limit.
    controlled_fraction = 0.25
    torch.cuda.set_per_process_memory_fraction(controlled_fraction, 0)
    controlled_oom_runs = []
    try:
        for batch in ([8192, 16384, 24576, 32768, 49152] if args.include_oom_probes else []):
            outcome = try_synthetic_step(batch, initial_state)
            controlled_oom_runs.append(outcome)
            if outcome["status"] == "oom":
                break
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
        torch.cuda.empty_cache()

    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "protocol": {
            "real_dataset": "FashionMNIST train 60,000",
            "precision": "FP32 with TF32 disabled",
            "repeats": args.repeats,
            "repeat_order": "ascending, descending, ascending",
            "warmup": "one synthetic train step at identical batch shape, excluded",
            "timing": "DataLoader + H2D + forward + backward + Adam",
            "oom_probe": "one synthetic FP32 train step, increasing batch until first OutOfMemoryError",
            "controlled_oom_fraction": controlled_fraction,
            "controlled_oom_note": "allocator safety limit, not the physical VRAM boundary",
        },
        "throughput_summary": summary,
        "throughput_runs": throughput_runs,
        "oom_runs": oom_runs,
        "controlled_oom_runs": controlled_oom_runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    font_path = Path("/mnt/c/Windows/Fonts/meiryo.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), dpi=150); fig.patch.set_facecolor("#07111f")
    for ax in axes:
        ax.set_facecolor("#0b172a"); ax.tick_params(colors="#dbeafe"); ax.grid(alpha=.18, color="white")
        for spine in ax.spines.values(): spine.set_visible(False)
    batches = np.array([int(value) for value in summary])
    median_ips = np.array([summary[str(value)]["median_images_per_second"] for value in batches])
    min_ips = np.array([summary[str(value)]["min_images_per_second"] for value in batches])
    max_ips = np.array([summary[str(value)]["max_images_per_second"] for value in batches])
    axes[0].plot(batches, median_ips, "o-", color="#38bdf8", linewidth=2)
    axes[0].fill_between(batches, min_ips, max_ips, color="#38bdf8", alpha=.18)
    axes[0].set_xscale("log", base=2); axes[0].set_xlabel("batch size", color="#dbeafe")
    axes[0].set_ylabel("images/s（1 epoch）", color="#dbeafe"); axes[0].set_title("実データthroughput：中央値と範囲", color="white")

    ok = [run for run in oom_runs if run["status"] == "ok"]
    oom = [run for run in controlled_oom_runs if run["status"] == "oom"]
    axes[1].plot([run["batch_size"] for run in ok], [run["peak_allocated_mib"] for run in ok], "o-", color="#34d399", label="peak allocated")
    axes[1].plot([run["batch_size"] for run in ok], [run["peak_reserved_mib"] for run in ok], "s--", color="#fbbf24", label="peak reserved")
    if oom:
        axes[1].axvline(oom[0]["batch_size"], color="#fb7185", linestyle=":", linewidth=2, label=f"25%制限でOOM: {oom[0]['batch_size']:,}")
    axes[1].set_xscale("log", base=2); axes[1].set_xlabel("synthetic batch size", color="#dbeafe")
    axes[1].set_ylabel("PyTorch peak memory [MiB]", color="#dbeafe"); axes[1].set_title("巨大batchで実際にOOMまで試す", color="white")
    axes[1].legend(facecolor="#0b172a", labelcolor="white")
    fig.suptitle("RTX 5070 Ti：batch size・throughput・VRAM・OOM", color="white", fontsize=19)
    fig.text(.5, .02, "FashionMNIST CNN 105,866 params / FP32 / WSL2 / DataLoader・転送・backward・Adam込み", ha="center", color="#cbd5e1", fontsize=10)
    fig.tight_layout(rect=(.02,.06,.98,.92)); args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, facecolor=fig.get_facecolor()); plt.close(fig)
    print(json.dumps({"throughput_summary": summary, "natural_runs": oom_runs, "controlled_oom_runs": controlled_oom_runs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
