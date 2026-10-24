#!/usr/bin/env python3
"""Compare classic classifiers on sklearn Digits with repeated splits."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_models() -> dict:
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=42),
        ),
        "rbf_svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale"),
        ),
    }


def evaluate_once(features, labels, seed: int) -> dict:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    result = {}
    for name, model in build_models().items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        predicted = model.predict(x_test)
        predict_ms = (time.perf_counter() - started) * 1000
        result[name] = {
            "train_accuracy": accuracy_score(y_train, model.predict(x_train)),
            "test_accuracy": accuracy_score(y_test, predicted),
            "fit_ms": fit_ms,
            "predict_360_ms": predict_ms,
            "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
            "mistake_indices": np.flatnonzero(predicted != y_test).tolist(),
            "predicted": predicted.tolist(),
            "y_test": y_test.tolist(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--mistakes-figure", type=Path, required=True)
    args = parser.parse_args()

    digits = load_digits()
    seeds = [0, 1, 2, 3, 4, 42]
    runs = {str(seed): evaluate_once(digits.data, digits.target, seed) for seed in seeds}
    summary = {}
    for name in build_models():
        scores = [runs[str(seed)][name]["test_accuracy"] for seed in seeds]
        summary[name] = {
            "mean_test_accuracy": statistics.mean(scores),
            "sample_std_test_accuracy": statistics.stdev(scores),
            "min_test_accuracy": min(scores),
            "max_test_accuracy": max(scores),
            "seed42": runs["42"][name],
        }

    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "sklearn.datasets.load_digits",
            "samples": int(digits.data.shape[0]),
            "features": int(digits.data.shape[1]),
            "image_shape": list(digits.images.shape[1:]),
            "classes": digits.target_names.tolist(),
            "pixel_range": [float(digits.data.min()), float(digits.data.max())],
        },
        "protocol": {
            "test_size": 0.2,
            "stratified": True,
            "seeds": seeds,
            "scaling_fitted_on_training_data_only": True,
        },
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    font_path = Path("/mnt/c/Windows/Fonts/meiryo.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    names = list(summary)
    labels = ["ダミー\n最多クラス", "ロジスティック\n回帰", "RBF-SVM"]
    means = [summary[name]["mean_test_accuracy"] * 100 for name in names]
    stds = [summary[name]["sample_std_test_accuracy"] * 100 for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), dpi=150, gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor("#07111f")
    for ax in axes:
        ax.set_facecolor("#0b172a")
        ax.tick_params(colors="#dbeafe")
        for spine in ax.spines.values():
            spine.set_visible(False)

    bars = axes[0].bar(labels, means, yerr=stds, capsize=6, color=["#94a3b8", "#38bdf8", "#34d399"])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("テスト正解率 [%]", color="#dbeafe")
    axes[0].set_title("6分割seedの平均 ± 標本SD", color="white", fontsize=14)
    axes[0].grid(axis="y", alpha=0.18, color="white")
    for bar, value in zip(bars, means, strict=True):
        axes[0].text(bar.get_x() + bar.get_width()/2, value + 3, f"{value:.2f}%", ha="center", color="white", fontweight="bold")

    matrix = np.array(summary["rbf_svm"]["seed42"]["confusion_matrix"])
    image = axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_title("RBF-SVM 混同行列（seed=42）", color="white", fontsize=14)
    axes[1].set_xlabel("予測ラベル", color="#dbeafe")
    axes[1].set_ylabel("正解ラベル", color="#dbeafe")
    axes[1].set_xticks(range(10)); axes[1].set_yticks(range(10))
    for row in range(10):
        for col in range(10):
            axes[1].text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > matrix.max()/2 else "#07111f", fontsize=9)
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("8×8手書き数字：古典機械学習を同条件で比較", color="white", fontsize=20, y=0.98)
    fig.text(0.5, 0.015, "学習80% / テスト20% / stratifyあり。StandardScalerは学習データだけでfit。", ha="center", color="#cbd5e1", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.93))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, facecolor=fig.get_facecolor())
    plt.close(fig)

    x_train, x_test, y_train, y_test, image_train, image_test = train_test_split(
        digits.data,
        digits.target,
        digits.images,
        test_size=0.2,
        random_state=42,
        stratify=digits.target,
    )
    svm = build_models()["rbf_svm"]
    svm.fit(x_train, y_train)
    predicted = svm.predict(x_test)
    mistake_positions = np.flatnonzero(predicted != y_test)
    mistake_fig, mistake_axes = plt.subplots(2, 4, figsize=(12, 6.75), dpi=150)
    mistake_fig.patch.set_facecolor("#07111f")
    flat_axes = mistake_axes.ravel()
    for ax, position in zip(flat_axes, mistake_positions, strict=False):
        ax.imshow(image_test[position], cmap="gray_r", interpolation="nearest")
        ax.set_title(
            f"正解 {y_test[position]} → 予測 {predicted[position]}",
            color="#fda4af",
            fontsize=13,
        )
        ax.axis("off")
    for ax in flat_axes[len(mistake_positions):]:
        ax.axis("off")
    mistake_fig.suptitle(
        f"RBF-SVMが間違えた{len(mistake_positions)}枚（seed=42 / test 360枚）",
        color="white",
        fontsize=20,
    )
    mistake_fig.text(
        0.5,
        0.025,
        "正解率だけでなく、どの数字同士を混同したかを見る。8×8なので人間にも曖昧な画像がある。",
        ha="center",
        color="#cbd5e1",
        fontsize=11,
    )
    mistake_fig.tight_layout(rect=(0.03, 0.06, 0.97, 0.92), h_pad=4.0, w_pad=1.5)
    args.mistakes_figure.parent.mkdir(parents=True, exist_ok=True)
    mistake_fig.savefig(args.mistakes_figure, facecolor=mistake_fig.get_facecolor())
    plt.close(mistake_fig)
    print(json.dumps({name: {key: round(value, 6) for key, value in values.items() if isinstance(value, float)} for name, values in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
