#!/usr/bin/env python3
"""End-to-end CPU/CUDA CNN training comparison on full FashionMNIST."""

from __future__ import annotations

import argparse
import copy
import json
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


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_loaders(data_dir: Path, seed: int, pin_memory: bool) -> tuple:
    transform = transforms.ToTensor()
    train_set = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=256,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0, pin_memory=pin_memory)
    return train_loader, test_loader, train_set, test_set


def evaluate(model, loader, device: str) -> tuple[float, list]:
    model.eval()
    correct = 0
    total = 0
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device == "cuda")
            labels = labels.to(device, non_blocking=device == "cuda")
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum())
            total += labels.numel()
            for truth, predicted in zip(labels.cpu(), predictions.cpu(), strict=True):
                confusion[truth, predicted] += 1
    return correct / total, confusion.tolist()


def train_device(data_dir: Path, initial_state: dict, device: str, seed: int) -> dict:
    torch.manual_seed(seed)
    train_loader, test_loader, train_set, test_set = make_loaders(data_dir, seed, pin_memory=device == "cuda")
    model = SmallCNN().to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    epochs = []
    for epoch in range(1, 4):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=device == "cuda")
            labels = labels.to(device, non_blocking=device == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum())
            seen += labels.size(0)
        if device == "cuda":
            torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - started
        test_accuracy, confusion = evaluate(model, test_loader, device)
        epochs.append({
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "test_accuracy": test_accuracy,
            "train_seconds": epoch_seconds,
            "images_per_second": seen / epoch_seconds,
        })
    return {
        "device": device,
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": epochs,
        "total_train_seconds": sum(item["train_seconds"] for item in epochs),
        "final_confusion_matrix": confusion,
        "peak_cuda_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    base_model = SmallCNN()
    initial_state = copy.deepcopy(base_model.state_dict())
    sample_output = base_model(torch.zeros(2, 1, 28, 28))
    if sample_output.shape != (2, 10):
        raise RuntimeError(f"unexpected output shape: {sample_output.shape}")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    results = {device: train_device(args.data_dir, initial_state, device, seed=42) for device in devices}
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "protocol": {
            "dataset": "FashionMNIST full train/test",
            "transform": "ToTensor only; pixels in [0,1]",
            "architecture": "Conv1x16 -> pool -> Conv16x32 -> pool -> Linear1568x64 -> 10",
            "batch_size": 256,
            "test_batch_size": 512,
            "epochs": 3,
            "optimizer": "Adam lr=1e-3",
            "num_workers": 0,
            "same_initial_state": True,
            "same_shuffle_seed": 42,
            "timing_includes_dataloader_transfer_forward_backward_step": True,
            "timing_excludes_test_evaluation": True,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    font_path = Path("/mnt/c/Windows/Fonts/meiryo.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), dpi=150)
    fig.patch.set_facecolor("#07111f")
    colors = {"cpu": "#38bdf8", "cuda": "#34d399"}
    for ax in axes:
        ax.set_facecolor("#0b172a"); ax.tick_params(colors="#dbeafe"); ax.grid(alpha=0.18, color="white")
        for spine in ax.spines.values(): spine.set_visible(False)
    epochs = [1, 2, 3]
    for device in devices:
        axes[0].plot(epochs, [item["test_accuracy"]*100 for item in results[device]["epochs"]], marker="o", linewidth=2.5, label=device.upper(), color=colors[device])
    axes[0].set_xticks(epochs); axes[0].set_xlabel("epoch", color="#dbeafe"); axes[0].set_ylabel("test accuracy [%]", color="#dbeafe")
    axes[0].set_title("同じ初期重みのテスト正解率", color="white", fontsize=15); axes[0].legend()
    width = 0.35; positions = np.arange(3)
    for index, device in enumerate(devices):
        values = [item["train_seconds"] for item in results[device]["epochs"]]
        axes[1].bar(positions + (index-(len(devices)-1)/2)*width, values, width, label=device.upper(), color=colors[device])
    axes[1].set_xticks(positions, ["epoch 1", "epoch 2", "epoch 3"]); axes[1].set_ylabel("end-to-end train [s]", color="#dbeafe")
    axes[1].set_title("DataLoader・転送・逆伝播込み", color="white", fontsize=15); axes[1].legend()
    fig.suptitle("FashionMNIST CNN：CPUとRTX 5070 Tiを実モデルで比較", color="white", fontsize=19)
    gpu_note = f"CUDA peak={results['cuda']['peak_cuda_memory_mib']} MiB" if "cuda" in results else "CUDA unavailable"
    fig.text(0.5, 0.02, f"full train 60,000 / test 10,000 / batch 256 / 3 epoch / num_workers=0 / {gpu_note}", ha="center", color="#cbd5e1", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.92)); args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, facecolor=fig.get_facecolor()); plt.close(fig)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
