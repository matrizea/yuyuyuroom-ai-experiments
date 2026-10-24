#!/usr/bin/env python3
"""Train and verify a two-layer neural network using only NumPy."""

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
from matplotlib import font_manager
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class TwoLayerNet:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int):
        rng = np.random.default_rng(seed)
        self.params = {
            "w1": rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size)),
            "b1": np.zeros(hidden_size),
            "w2": rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_size)),
            "b2": np.zeros(output_size),
        }

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        hidden_pre = x @ self.params["w1"] + self.params["b1"]
        hidden = np.maximum(hidden_pre, 0)
        logits = hidden @ self.params["w2"] + self.params["b2"]
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
        return probabilities, (x, hidden_pre, hidden, probabilities)

    def loss_and_gradients(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
        probabilities, cache = self.forward(x)
        sample_count = x.shape[0]
        loss = -np.log(probabilities[np.arange(sample_count), y] + 1e-12).mean()
        x_cached, hidden_pre, hidden, _ = cache
        d_logits = probabilities.copy()
        d_logits[np.arange(sample_count), y] -= 1
        d_logits /= sample_count
        gradients = {
            "w2": hidden.T @ d_logits,
            "b2": d_logits.sum(axis=0),
        }
        d_hidden = d_logits @ self.params["w2"].T
        d_hidden[hidden_pre <= 0] = 0
        gradients["w1"] = x_cached.T @ d_hidden
        gradients["b1"] = d_hidden.sum(axis=0)
        return float(loss), gradients

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0].argmax(axis=1)


def accuracy(model: TwoLayerNet, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(x) == y))


def gradient_check(model: TwoLayerNet, x: np.ndarray, y: np.ndarray, checks: int = 24) -> dict:
    _, analytical = model.loss_and_gradients(x, y)
    rng = np.random.default_rng(2026)
    epsilon = 1e-5
    records = []
    keys = list(model.params)
    for _ in range(checks):
        key = keys[int(rng.integers(len(keys)))]
        parameter = model.params[key]
        index = tuple(int(rng.integers(size)) for size in parameter.shape)
        original = parameter[index]
        parameter[index] = original + epsilon
        plus, _ = model.loss_and_gradients(x, y)
        parameter[index] = original - epsilon
        minus, _ = model.loss_and_gradients(x, y)
        parameter[index] = original
        numerical = (plus - minus) / (2 * epsilon)
        analytic = float(analytical[key][index])
        relative_error = abs(numerical - analytic) / max(1e-12, abs(numerical) + abs(analytic))
        records.append({
            "parameter": key,
            "index": index,
            "numerical": numerical,
            "analytical": analytic,
            "relative_error": relative_error,
        })
    return {
        "checks": checks,
        "max_relative_error": max(item["relative_error"] for item in records),
        "mean_relative_error": statistics.mean(item["relative_error"] for item in records),
        "passed_under_1e-5": all(item["relative_error"] < 1e-5 for item in records),
        "records": records,
    }


def train_one(x_train, y_train, x_val, y_val, x_test, y_test, seed: int) -> dict:
    model = TwoLayerNet(64, 64, 10, seed)
    rng = np.random.default_rng(seed)
    learning_rate = 0.05
    momentum = 0.9
    batch_size = 64
    max_epochs = 240
    patience = 40
    velocity = {key: np.zeros_like(value) for key, value in model.params.items()}
    best_val = -1.0
    best_params = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            _, gradients = model.loss_and_gradients(x_train[indices], y_train[indices])
            for key in model.params:
                velocity[key] = momentum * velocity[key] - learning_rate * gradients[key]
                model.params[key] += velocity[key]
        train_loss, _ = model.loss_and_gradients(x_train, y_train)
        train_accuracy = accuracy(model, x_train, y_train)
        val_accuracy = accuracy(model, x_val, y_val)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        })
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_epoch = epoch
            best_params = copy.deepcopy(model.params)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    elapsed = time.perf_counter() - started
    model.params = best_params
    predict_started = time.perf_counter()
    test_predictions = model.predict(x_test)
    predict_ms = (time.perf_counter() - predict_started) * 1000
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_accuracy": best_val,
        "train_accuracy_at_best": accuracy(model, x_train, y_train),
        "test_accuracy": float(np.mean(test_predictions == y_test)),
        "train_seconds": elapsed,
        "predict_360_ms": predict_ms,
        "parameter_count": sum(value.size for value in model.params.values()),
        "mistakes": int(np.sum(test_predictions != y_test)),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    digits = load_digits()
    features = digits.data.astype(np.float64) / 16.0
    labels = digits.target.astype(np.int64)
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )
    check_model = TwoLayerNet(64, 8, 10, seed=42)
    check = gradient_check(check_model, x_train[:5], y_train[:5])
    if not check["passed_under_1e-5"]:
        raise RuntimeError(f"gradient check failed: {check['max_relative_error']}")

    seeds = [0, 1, 2, 3, 4, 42]
    runs = [train_one(x_train, y_train, x_val, y_val, x_test, y_test, seed) for seed in seeds]
    test_scores = [run["test_accuracy"] for run in runs]
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "test_split_seed": 42,
            "test_size": len(x_test),
            "validation_size": len(x_val),
            "training_size": len(x_train),
            "input_scaling": "pixel / 16.0",
            "architecture": "64 -> 64 ReLU -> 10 softmax",
            "optimizer": "mini-batch SGD",
            "learning_rate": 0.05,
            "momentum": 0.9,
            "batch_size": 64,
            "max_epochs": 240,
            "early_stopping_patience": 40,
            "selection": "best validation accuracy; test not used for selection",
        },
        "gradient_check": check,
        "summary": {
            "mean_test_accuracy": statistics.mean(test_scores),
            "sample_std_test_accuracy": statistics.stdev(test_scores),
            "min_test_accuracy": min(test_scores),
            "max_test_accuracy": max(test_scores),
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    seed42 = next(run for run in runs if run["seed"] == 42)
    history = seed42["history"]
    font_path = Path("/mnt/c/Windows/Fonts/meiryo.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), dpi=150)
    fig.patch.set_facecolor("#07111f")
    for ax in axes:
        ax.set_facecolor("#0b172a")
        ax.tick_params(colors="#dbeafe")
        ax.grid(alpha=0.18, color="white")
        for spine in ax.spines.values():
            spine.set_visible(False)
    epochs = [item["epoch"] for item in history]
    axes[0].plot(epochs, [item["train_loss"] for item in history], color="#38bdf8", linewidth=2.5)
    axes[0].axvline(seed42["best_epoch"], color="#fbbf24", linestyle="--", label=f"best epoch={seed42['best_epoch']}")
    axes[0].set_title("交差エントロピー損失", color="white", fontsize=15)
    axes[0].set_xlabel("epoch", color="#dbeafe"); axes[0].set_ylabel("loss", color="#dbeafe"); axes[0].legend()
    axes[1].plot(epochs, np.array([item["train_accuracy"] for item in history]) * 100, label="train", color="#a78bfa", linewidth=2.5)
    axes[1].plot(epochs, np.array([item["val_accuracy"] for item in history]) * 100, label="validation", color="#34d399", linewidth=2.5)
    axes[1].axvline(seed42["best_epoch"], color="#fbbf24", linestyle="--")
    axes[1].set_title("学習と検証の正解率", color="white", fontsize=15)
    axes[1].set_xlabel("epoch", color="#dbeafe"); axes[1].set_ylabel("accuracy [%]", color="#dbeafe"); axes[1].legend()
    fig.suptitle("NumPyだけで作る2層ニューラルネット：Digits実測", color="white", fontsize=20)
    fig.text(0.5, 0.02, f"seed=42 / test={seed42['test_accuracy']*100:.2f}% / 6初期値平均={payload['summary']['mean_test_accuracy']*100:.2f}% / 勾配チェック最大相対誤差={check['max_relative_error']:.2e}", ha="center", color="#cbd5e1", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.92))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps({"summary": payload["summary"], "gradient_check": {key: value for key, value in check.items() if key != "records"}, "seed42": {key: value for key, value in seed42.items() if key != "history"}}, indent=2))


if __name__ == "__main__":
    main()
