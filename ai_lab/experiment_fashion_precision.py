#!/usr/bin/env python3
"""Compare FP32, TF32, FP16 AMP and BF16 AMP on the same FashionMNIST CNN."""

from __future__ import annotations

import argparse
import copy
import json
import time
from contextlib import nullcontext
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


MODES = {
    "fp32": {"autocast": None, "tf32": False, "grad_scaler": False},
    "tf32": {"autocast": None, "tf32": True, "grad_scaler": False},
    "fp16_amp": {"autocast": torch.float16, "tf32": True, "grad_scaler": True},
    "bf16_amp": {"autocast": torch.bfloat16, "tf32": True, "grad_scaler": False},
}


def make_loaders(data_dir: Path, seed: int):
    transform = transforms.ToTensor()
    train_set = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, generator=generator, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


def autocast_context(dtype):
    return torch.autocast(device_type="cuda", dtype=dtype) if dtype is not None else nullcontext()


def evaluate(model, loader, dtype) -> float:
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
            with autocast_context(dtype):
                predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum()); total += labels.numel()
    return correct / total


def set_tf32(enabled: bool) -> None:
    precision = "tf32" if enabled else "ieee"
    # PyTorch 2.9+ API. The older allow_tf32 booleans are being deprecated.
    torch.backends.cuda.matmul.fp32_precision = precision
    torch.backends.cudnn.fp32_precision = precision


def train_mode(data_dir: Path, initial_state: dict, mode_name: str) -> dict:
    config = MODES[mode_name]
    set_tf32(config["tf32"])
    train_loader, test_loader = make_loaders(data_dir, seed=42)
    model = SmallCNN().cuda(); model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=config["grad_scaler"])

    warmup_model = copy.deepcopy(model)
    warmup_optimizer = torch.optim.Adam(warmup_model.parameters(), lr=1e-3)
    dummy_images = torch.zeros(256, 1, 28, 28, device="cuda")
    dummy_labels = torch.zeros(256, dtype=torch.long, device="cuda")
    for _ in range(3):
        warmup_optimizer.zero_grad(set_to_none=True)
        with autocast_context(config["autocast"]):
            warmup_loss = loss_function(warmup_model(dummy_images), dummy_labels)
        if config["grad_scaler"]:
            scaler.scale(warmup_loss).backward(); scaler.step(warmup_optimizer); scaler.update()
        else:
            warmup_loss.backward(); warmup_optimizer.step()
    del warmup_model, warmup_optimizer, dummy_images, dummy_labels
    torch.cuda.empty_cache()
    # The synthetic warm-up must not advance FP16's dynamic loss-scale state.
    scaler = torch.amp.GradScaler("cuda", enabled=config["grad_scaler"])
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()

    epochs = []
    for epoch in range(1, 3):
        model.train(); loss_sum = 0.0; correct = 0; seen = 0
        torch.cuda.synchronize(); started = time.perf_counter()
        for images, labels in train_loader:
            images = images.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config["autocast"]):
                logits = model(images)
                loss = loss_function(logits, labels)
            if config["grad_scaler"]:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum()); seen += labels.size(0)
        torch.cuda.synchronize(); seconds = time.perf_counter() - started
        test_accuracy = evaluate(model, test_loader, config["autocast"])
        epochs.append({
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "test_accuracy": test_accuracy,
            "train_seconds": seconds,
            "images_per_second": seen / seconds,
        })
    return {
        "mode": mode_name,
        "config": {"autocast": str(config["autocast"]), "tf32": config["tf32"], "grad_scaler": config["grad_scaler"]},
        "epochs": epochs,
        "total_train_seconds": sum(item["train_seconds"] for item in epochs),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        "final_grad_scale": float(scaler.get_scale()) if config["grad_scaler"] else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(42)
    initial_state = copy.deepcopy(SmallCNN().state_dict())
    mode_names = list(MODES)
    runs = {mode: [] for mode in mode_names}
    # Rotate the execution order so a single mode is not always advantaged by
    # cold-start or thermal state. Every run still gets identical weights/data.
    for repeat in range(args.repeats):
        order = mode_names[repeat % len(mode_names):] + mode_names[:repeat % len(mode_names)]
        for mode in order:
            run = train_mode(args.data_dir, initial_state, mode)
            run["repeat"] = repeat + 1
            runs[mode].append(run)

    results = {}
    for mode, mode_runs in runs.items():
        times = np.array([run["total_train_seconds"] for run in mode_runs])
        accuracies = np.array([run["epochs"][-1]["test_accuracy"] for run in mode_runs])
        results[mode] = {
            "median_train_seconds": float(np.median(times)),
            "min_train_seconds": float(times.min()),
            "max_train_seconds": float(times.max()),
            "median_test_accuracy": float(np.median(accuracies)),
            "min_test_accuracy": float(accuracies.min()),
            "max_test_accuracy": float(accuracies.max()),
            "peak_allocated_mib": max(run["peak_allocated_mib"] for run in mode_runs),
            "peak_reserved_mib": max(run["peak_reserved_mib"] for run in mode_runs),
            "final_grad_scale": mode_runs[-1]["final_grad_scale"],
        }
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "tf32_supported": torch.cuda.is_tf32_supported(),
        "protocol": {
            "same_initial_state": True, "same_shuffle_seed": 42,
            "full_train_samples": 60000, "batch_size": 256, "epochs": 2,
            "repeats_per_mode": args.repeats,
            "order": "rotated each repeat to reduce cold-start/thermal order bias",
            "warmup": "3 synthetic train steps per mode, excluded from timing",
            "timing": "DataLoader + H2D + forward + backward + Adam; test excluded",
        },
        "summary": results,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    font_path = Path("/mnt/c/Windows/Fonts/meiryo.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path); plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), dpi=150); fig.patch.set_facecolor("#07111f")
    labels = list(results); colors = ["#38bdf8", "#60a5fa", "#a78bfa", "#34d399"]
    for ax in axes:
        ax.set_facecolor("#0b172a"); ax.tick_params(colors="#dbeafe"); ax.grid(axis="y", alpha=.18, color="white")
        for spine in ax.spines.values(): spine.set_visible(False)
    times = [results[name]["median_train_seconds"] for name in labels]
    lower = [results[name]["median_train_seconds"]-results[name]["min_train_seconds"] for name in labels]
    upper = [results[name]["max_train_seconds"]-results[name]["median_train_seconds"] for name in labels]
    bars = axes[0].bar(labels, times, color=colors, yerr=[lower, upper], capsize=5)
    axes[0].set_ylabel("2 epoch end-to-end [s]", color="#dbeafe"); axes[0].set_title(f"学習時間（各{args.repeats}回の中央値・範囲）", color="white", fontsize=15)
    axes[0].tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, times, strict=True): axes[0].text(bar.get_x()+bar.get_width()/2, value*1.01, f"{value:.3f}s", ha="center", color="white")
    accuracy = [results[name]["median_test_accuracy"]*100 for name in labels]
    allocated = [results[name]["peak_allocated_mib"] for name in labels]
    width=.36; x=np.arange(len(labels))
    axes[1].bar(x-width/2, accuracy, width, label="test accuracy [%]", color=colors)
    second=axes[1].twinx(); second.bar(x+width/2, allocated, width, label="allocated MiB", color="#fbbf24", alpha=.75)
    axes[1].set_xticks(x, labels, rotation=15); axes[1].set_ylim(80,90); axes[1].set_ylabel("test accuracy [%]", color="#dbeafe"); second.set_ylabel("peak allocated [MiB]", color="#fbbf24"); second.tick_params(colors="#fbbf24")
    axes[1].set_title("精度とPyTorch allocated", color="white", fontsize=15)
    fig.suptitle("RTX 5070 TiでFP32・TF32・FP16・BF16を実モデル比較", color="white", fontsize=19)
    fig.text(.5,.02,f"FashionMNIST 60,000 / CNN 105,866 params / batch 256 / 各{args.repeats}回 / 同じ初期重み・shuffle",ha="center",color="#cbd5e1",fontsize=10)
    fig.tight_layout(rect=(.02,.06,.98,.92)); args.figure.parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.figure,facecolor=fig.get_facecolor()); plt.close(fig)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
