#!/usr/bin/env python3
"""Benchmark DataLoader workers, persistent workers and pinned memory on WSL2."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
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


WORKERS = [0, 1, 2, 4, 8, 12, 16]


def make_transform(workload: str):
    if workload == "light":
        return transforms.ToTensor()
    if workload == "augmented":
        return transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.75, 1.0)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
    raise ValueError(workload)


def make_loader(data_dir: Path, workload: str, workers: int, pin_memory: bool) -> DataLoader:
    dataset = datasets.FashionMNIST(
        data_dir,
        train=True,
        download=True,
        transform=make_transform(workload),
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": 256,
        "shuffle": True,
        "generator": torch.Generator().manual_seed(42),
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def run_config(data_dir: Path, initial_state: dict, workload: str, workers: int, pin_memory: bool) -> dict:
    loader = make_loader(data_dir, workload, workers, pin_memory)
    model = SmallCNN().cuda(); model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # GPU/library warm-up only. DataLoader startup remains part of epoch 1.
    images = torch.zeros(256, 1, 28, 28, device="cuda")
    labels = torch.zeros(256, dtype=torch.long, device="cuda")
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(images), labels); loss.backward(); optimizer.step()
    torch.cuda.synchronize(); del images, labels, loss

    epochs = []
    for epoch in [1, 2]:
        model.train(); seen = 0; steps = 0; loss_sum = 0.0
        torch.cuda.synchronize(); started = time.perf_counter()
        for images, labels in loader:
            images = images.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward(); optimizer.step()
            seen += labels.numel(); steps += 1
            loss_sum += float(loss.detach()) * labels.size(0)
        torch.cuda.synchronize(); seconds = time.perf_counter() - started
        epochs.append({
            "epoch": epoch,
            "phase": "first_with_worker_startup" if epoch == 1 else "warm_persistent",
            "seconds": seconds,
            "images_per_second": seen / seconds,
            "mean_step_ms": seconds / steps * 1000,
            "train_loss": loss_sum / seen,
        })
    result = {
        "workload": workload,
        "workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
        "prefetch_factor": 2 if workers > 0 else None,
        "epochs": epochs,
    }
    del loader, model, optimizer, images, labels, loss
    gc.collect(); torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.manual_seed(42)
    initial_state = copy.deepcopy(SmallCNN().state_dict())

    runs = []
    for workload in ["light", "augmented"]:
        for workers in WORKERS:
            runs.append(run_config(args.data_dir, initial_state, workload, workers, True))
    # Isolate pinned-memory impact at a practical parallel worker count.
    runs.append(run_config(args.data_dir, initial_state, "augmented", 4, False))
    pin_trials = {"false": [], "true": []}
    for repeat in range(5):
        order = [False, True] if repeat % 2 == 0 else [True, False]
        for enabled in order:
            trial = run_config(args.data_dir, initial_state, "augmented", 4, enabled)
            pin_trials[str(enabled).lower()].append({
                "repeat": repeat + 1,
                "first_seconds": trial["epochs"][0]["seconds"],
                "warm_seconds": trial["epochs"][1]["seconds"],
            })
    high_worker_trials = {"8": [], "12": [], "16": []}
    high_workers = [8, 12, 16]
    for repeat in range(5):
        order = high_workers[repeat % 3:] + high_workers[:repeat % 3]
        for workers in order:
            trial = run_config(args.data_dir, initial_state, "augmented", workers, True)
            high_worker_trials[str(workers)].append({
                "repeat": repeat + 1,
                "warm_seconds": trial["epochs"][1]["seconds"],
                "warm_images_per_second": trial["epochs"][1]["images_per_second"],
            })

    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "logical_cpus": os.cpu_count(),
        "protocol": {
            "dataset": "FashionMNIST train 60,000",
            "batch_size": 256,
            "precision": "FP32, TF32 disabled",
            "timing": "DataLoader + H2D + forward + backward + Adam",
            "epoch_1": "includes worker process startup and initial prefetch",
            "epoch_2": "same loader with persistent workers, warm",
            "prefetch_factor": 2,
        },
        "runs": runs,
        "pin_memory_trials": pin_trials,
        "high_worker_trials": high_worker_trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def find(workload, workers, pin=True):
        return next(run for run in runs if run["workload"] == workload and run["workers"] == workers and run["pin_memory"] == pin)

    font_path = Path("/mnt/c/Windows/Fonts/meiryo.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), dpi=150); fig.patch.set_facecolor("#07111f")
    for ax in axes:
        ax.set_facecolor("#0b172a"); ax.tick_params(colors="#dbeafe"); ax.grid(axis="y", alpha=.18, color="white")
        for spine in ax.spines.values(): spine.set_visible(False)
    for workload, color in [("light", "#38bdf8"), ("augmented", "#a78bfa")]:
        warm_ips = [find(workload, worker)["epochs"][1]["images_per_second"] for worker in WORKERS]
        if workload == "augmented":
            for worker in high_workers:
                index = WORKERS.index(worker)
                warm_ips[index] = float(np.median([trial["warm_images_per_second"] for trial in high_worker_trials[str(worker)]]))
        axes[0].plot(WORKERS, warm_ips, "o-", linewidth=2, color=color, label=workload)
    axes[0].set_xlabel("num_workers", color="#dbeafe"); axes[0].set_ylabel("warm epoch images/s", color="#dbeafe")
    axes[0].set_title("transform負荷で最適worker数が変わる", color="white"); axes[0].legend(facecolor="#0b172a", labelcolor="white")

    first = [find("augmented", worker)["epochs"][0]["seconds"] for worker in WORKERS]
    warm = [find("augmented", worker)["epochs"][1]["seconds"] for worker in WORKERS]
    x = np.arange(len(WORKERS)); width = .36
    axes[1].bar(x-width/2, first, width, color="#fb7185", label="epoch 1（起動込み）")
    axes[1].bar(x+width/2, warm, width, color="#34d399", label="epoch 2（persistent）")
    axes[1].set_xticks(x, WORKERS); axes[1].set_xlabel("num_workers", color="#dbeafe"); axes[1].set_ylabel("augmented epoch [s]", color="#dbeafe")
    axes[1].set_title("worker起動費と温間性能を分離", color="white"); axes[1].legend(facecolor="#0b172a", labelcolor="white")
    pin_on = float(np.median([trial["warm_seconds"] for trial in pin_trials["true"]]))
    pin_off = float(np.median([trial["warm_seconds"] for trial in pin_trials["false"]]))
    fig.suptitle("WSL2 PyTorch DataLoader bottleneckを実測", color="white", fontsize=19)
    fig.text(.5, .02, f"FashionMNIST CNN / batch 256 / 60,000 images / pin_memory workers=4 各5回中央値: OFF {pin_off:.3f}s → ON {pin_on:.3f}s", ha="center", color="#cbd5e1", fontsize=10)
    fig.subplots_adjust(left=.09, right=.97, bottom=.15, top=.82, wspace=.22)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, facecolor=fig.get_facecolor()); plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
