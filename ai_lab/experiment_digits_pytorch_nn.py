#!/usr/bin/env python3
"""Compare an equivalent PyTorch MLP with the from-scratch NumPy network."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

from experiment_digits_numpy_nn import TwoLayerNet


def build_model(seed: int, dtype=torch.float32) -> nn.Sequential:
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    numpy_initial = TwoLayerNet(64, 64, 10, seed)
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(numpy_initial.params["w1"].T).to(dtype))
        model[0].bias.copy_(torch.from_numpy(numpy_initial.params["b1"]).to(dtype))
        model[2].weight.copy_(torch.from_numpy(numpy_initial.params["w2"].T).to(dtype))
        model[2].bias.copy_(torch.from_numpy(numpy_initial.params["b2"]).to(dtype))
    return model.to(dtype=dtype)


def numpy_torch_equivalence(x: np.ndarray, y: np.ndarray) -> dict:
    numpy_model = TwoLayerNet(64, 8, 10, seed=42)
    numpy_probabilities, _ = numpy_model.forward(x)
    numpy_loss, numpy_gradients = numpy_model.loss_and_gradients(x, y)

    torch_model = nn.Sequential(nn.Linear(64, 8), nn.ReLU(), nn.Linear(8, 10)).double()
    with torch.no_grad():
        torch_model[0].weight.copy_(torch.from_numpy(numpy_model.params["w1"].T))
        torch_model[0].bias.copy_(torch.from_numpy(numpy_model.params["b1"]))
        torch_model[2].weight.copy_(torch.from_numpy(numpy_model.params["w2"].T))
        torch_model[2].bias.copy_(torch.from_numpy(numpy_model.params["b2"]))
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    logits = torch_model(x_tensor)
    loss = nn.CrossEntropyLoss()(logits, y_tensor)
    loss.backward()
    torch_probabilities = torch.softmax(logits, dim=1).detach().numpy()
    torch_gradients = {
        "w1": torch_model[0].weight.grad.detach().numpy().T,
        "b1": torch_model[0].bias.grad.detach().numpy(),
        "w2": torch_model[2].weight.grad.detach().numpy().T,
        "b2": torch_model[2].bias.grad.detach().numpy(),
    }
    gradient_max_abs = {
        key: float(np.max(np.abs(numpy_gradients[key] - torch_gradients[key])))
        for key in numpy_gradients
    }
    return {
        "numpy_loss": numpy_loss,
        "torch_loss": float(loss.detach()),
        "loss_abs_difference": abs(numpy_loss - float(loss.detach())),
        "probability_max_abs_difference": float(np.max(np.abs(numpy_probabilities - torch_probabilities))),
        "gradient_max_abs_difference": gradient_max_abs,
        "passed_under_1e-10": max(gradient_max_abs.values()) < 1e-10,
    }


def tensor_accuracy(model, x, y) -> float:
    with torch.no_grad():
        return float((model(x).argmax(dim=1) == y).float().mean().cpu())


def train_one(arrays: tuple, device: str, seed: int) -> dict:
    x_train_np, y_train_np, x_val_np, y_val_np, x_test_np, y_test_np = arrays
    model = build_model(seed).to(device)
    x_train = torch.from_numpy(x_train_np.astype(np.float32)).to(device)
    y_train = torch.from_numpy(y_train_np).to(device)
    x_val = torch.from_numpy(x_val_np.astype(np.float32)).to(device)
    y_val = torch.from_numpy(y_val_np).to(device)
    x_test = torch.from_numpy(x_test_np.astype(np.float32)).to(device)
    y_test = torch.from_numpy(y_test_np).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(len(x_train_np)) for _ in range(240)]
    best_val = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch, order in enumerate(orders, start=1):
        model.train()
        for start in range(0, len(order), 64):
            indices = torch.from_numpy(order[start:start + 64]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x_train[indices]), y_train[indices])
            loss.backward()
            optimizer.step()
        train_loss = float(loss_function(model(x_train), y_train).detach().cpu())
        train_accuracy = tensor_accuracy(model, x_train, y_train)
        val_accuracy = tensor_accuracy(model, x_val, y_val)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "val_accuracy": val_accuracy})
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if stale >= 40:
            break
    if device == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.to(device).eval()
    if device == "cuda":
        torch.cuda.synchronize()
    predict_started = time.perf_counter()
    with torch.no_grad():
        predictions = model(x_test).argmax(dim=1)
    if device == "cuda":
        torch.cuda.synchronize()
    predict_ms = (time.perf_counter() - predict_started) * 1000
    return {
        "device": device,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_accuracy": best_val,
        "train_accuracy_at_best": tensor_accuracy(model, x_train, y_train),
        "test_accuracy": float((predictions == y_test).float().mean().cpu()),
        "train_seconds": train_seconds,
        "predict_360_ms": predict_ms,
        "mistakes": int((predictions != y_test).sum().cpu()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }


def summarize(runs: list[dict]) -> dict:
    accuracies = [run["test_accuracy"] for run in runs]
    times = [run["train_seconds"] for run in runs]
    return {
        "mean_test_accuracy": statistics.mean(accuracies),
        "sample_std_test_accuracy": statistics.stdev(accuracies),
        "min_test_accuracy": min(accuracies),
        "max_test_accuracy": max(accuracies),
        "mean_train_seconds": statistics.mean(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    digits = load_digits()
    x = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(np.int64)
    x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full)
    arrays = (x_train, y_train, x_val, y_val, x_test, y_test)
    equivalence = numpy_torch_equivalence(x_train[:5], y_train[:5])
    if not equivalence["passed_under_1e-10"]:
        raise RuntimeError(f"NumPy/PyTorch gradient mismatch: {equivalence}")
    seeds = [0, 1, 2, 3, 4, 42]
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    runs = {device: [train_one(arrays, device, seed) for seed in seeds] for device in devices}
    summaries = {device: summarize(device_runs) for device, device_runs in runs.items()}
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "protocol": {
            "architecture": "64 -> 64 ReLU -> 10 logits",
            "optimizer": "torch.optim.SGD(lr=0.05, momentum=0.9)",
            "loss": "CrossEntropyLoss (raw logits; no explicit Softmax)",
            "same_split_as_numpy": True,
            "exact_same_initial_weights_as_numpy": True,
            "same_batch_order_between_devices": True,
            "data_preloaded_on_each_device": True,
            "timing_includes_training_and validation_each_epoch": True,
        },
        "numpy_torch_equivalence": equivalence,
        "summaries": summaries,
        "runs": runs,
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
    positions = np.arange(len(seeds))
    width = 0.36
    for index, device in enumerate(devices):
        values = [run["test_accuracy"] * 100 for run in runs[device]]
        axes[0].bar(positions + (index - (len(devices)-1)/2)*width, values, width, label=device.upper(), color=colors[device])
    axes[0].set_xticks(positions, [str(seed) for seed in seeds]); axes[0].set_ylim(90, 100)
    axes[0].set_xlabel("重み・バッチ順seed", color="#dbeafe"); axes[0].set_ylabel("test accuracy [%]", color="#dbeafe")
    axes[0].set_title("同一条件のテスト正解率", color="white", fontsize=15); axes[0].legend()
    time_labels = [device.upper() for device in devices]
    time_values = [summaries[device]["mean_train_seconds"] for device in devices]
    bars = axes[1].bar(time_labels, time_values, color=[colors[device] for device in devices], width=0.55)
    axes[1].set_ylabel("平均学習時間 [s]", color="#dbeafe"); axes[1].set_title("この小さなネットのCPU/GPU時間", color="white", fontsize=15)
    for bar, value in zip(bars, time_values, strict=True):
        axes[1].text(bar.get_x()+bar.get_width()/2, value*1.02, f"{value:.3f}s", ha="center", color="white", fontsize=13)
    fig.suptitle("同じ2層NNをPyTorchへ移植：自動微分とCPU/CUDA実測", color="white", fontsize=19)
    max_grad = max(equivalence["gradient_max_abs_difference"].values())
    fig.text(0.5, 0.02, f"NumPy↔PyTorch 最大勾配差={max_grad:.2e} / probability差={equivalence['probability_max_abs_difference']:.2e} / データは各deviceへ事前配置", ha="center", color="#cbd5e1", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.92))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, facecolor=fig.get_facecolor()); plt.close(fig)
    compact = {"equivalence": equivalence, "summaries": summaries, "seed42": {device: {key: value for key, value in next(run for run in runs[device] if run['seed']==42).items() if key != 'history'} for device in devices}}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
