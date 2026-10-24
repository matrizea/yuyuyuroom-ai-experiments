#!/usr/bin/env python3
"""Render the Ollama 4B/8B/12B benchmark from saved JSON."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.json.read_text(encoding="utf-8"))

    models = ["gemma3:4b", "qwen3:8b", "gemma3:12b"]
    labels = ["Gemma 3\n4.3B", "Qwen 3\n8.2B*", "Gemma 3\n12.2B"]
    colors = ["#38bdf8", "#a78bfa", "#fb7185"]
    summary = payload["summary"]
    runs = payload["runs"]

    font_path = Path("/mnt/c/Windows/Fonts/meiryo.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()

    fig, axes = plt.subplots(1, 3, figsize=(14, 7.1), dpi=160)
    fig.patch.set_facecolor("#07111f")
    for ax in axes:
        ax.set_facecolor("#0b172a")
        ax.tick_params(colors="#dbeafe")
        ax.grid(axis="y", alpha=0.18, color="white")
        for spine in ax.spines.values():
            spine.set_visible(False)

    x = np.arange(3)
    speed = [summary[m]["warm_tokens_per_s_median"] for m in models]
    axes[0].bar(x, speed, color=colors, width=0.66)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("output tokens/s（温間中央値）", color="#dbeafe")
    axes[0].set_title("生成速度", color="white", fontsize=14)
    for i, value in enumerate(speed):
        axes[0].text(i, value + 4, f"{value:.1f}", ha="center", color="white", weight="bold")
    axes[0].set_ylim(0, max(speed) * 1.2)

    cold_values = []
    cold_err_low = []
    cold_err_high = []
    for model in models:
        values = [r["load_duration_s"] for r in runs if r["model"] == model and r["phase"] == "cold"]
        median = float(np.median(values))
        cold_values.append(median)
        cold_err_low.append(median - min(values))
        cold_err_high.append(max(values) - median)
    axes[1].bar(x, cold_values, color=colors, width=0.66)
    axes[1].errorbar(x, cold_values, yerr=[cold_err_low, cold_err_high], fmt="none", ecolor="white", capsize=5)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("API load_duration [s]", color="#dbeafe")
    axes[1].set_title("アンロード後の再ロード", color="white", fontsize=14)
    for i, value in enumerate(cold_values):
        axes[1].text(i, value + 0.25, f"{value:.2f}s", ha="center", color="white", weight="bold")
    axes[1].set_ylim(0, max(cold_values) * 1.28)

    disk_gib = []
    for model in models:
        tag = next(item for item in payload["models_from_tags"] if item["name"] == model)
        disk_gib.append(tag["size"] / 1024**3)
    gpu_gib = [summary[m]["peak_gpu_used_mib"] / 1024 for m in models]
    width = 0.34
    axes[2].bar(x - width / 2, disk_gib, width, color="#fbbf24", label="model file")
    axes[2].bar(x + width / 2, gpu_gib, width, color="#34d399", label="peak GPU used")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("GiB", color="#dbeafe")
    axes[2].set_title("保存容量とGPU使用量", color="white", fontsize=14)
    axes[2].legend(facecolor="#0b172a", labelcolor="white", loc="upper left")
    for i, (disk, gpu) in enumerate(zip(disk_gib, gpu_gib)):
        axes[2].text(i - width / 2, disk + 0.3, f"{disk:.1f}", ha="center", color="white", fontsize=9)
        axes[2].text(i + width / 2, gpu + 0.3, f"{gpu:.1f}", ha="center", color="white", fontsize=9)
    axes[2].axhline(16303 / 1024, color="#f87171", linestyle="--", linewidth=1.5, label="16 GB GPU")
    axes[2].set_ylim(0, 17.5)

    fig.suptitle("RTX 5070 Ti 16GBでローカルLLM 4B / 8B / 12Bを実測", color="white", fontsize=19, y=0.965)
    fig.text(
        0.5,
        0.035,
        "Ollama 0.32.15 / Q4_K_M / num_ctx 4096 / 256 output tokens / cold→warm × 3　*8Bのみ別model family",
        ha="center",
        color="#cbd5e1",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.17, top=0.84, wspace=0.28)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
