#!/usr/bin/env python3
"""Render Gemma 3 4B Q4_K_M vs Q8_0 results from JSON."""

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
    data = json.loads(args.json.read_text(encoding="utf-8"))
    models = data["models"]
    labels = ["Q4_K_M", "Q8_0"]
    colors = ["#38bdf8", "#fb7185"]
    summary = [data["summary"][model] for model in models]

    font_path = Path("/mnt/c/Windows/Fonts/meiryo.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()

    fig, axes = plt.subplots(1, 3, figsize=(14, 7), dpi=160)
    fig.patch.set_facecolor("#07111f")
    for ax in axes:
        ax.set_facecolor("#0b172a")
        ax.tick_params(colors="#dbeafe")
        ax.grid(axis="y", alpha=0.18, color="white")
        for spine in ax.spines.values():
            spine.set_visible(False)
    x = np.arange(2)

    disk = [row["file_size_gib"] for row in summary]
    gpu = [row["peak_gpu_used_mib"] / 1024 for row in summary]
    width = 0.34
    axes[0].bar(x - width / 2, disk, width, color="#fbbf24", label="model file")
    axes[0].bar(x + width / 2, gpu, width, color="#34d399", label="peak GPU used")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("GiB", color="#dbeafe")
    axes[0].set_title("容量とGPU使用量", color="white", fontsize=14)
    axes[0].legend(facecolor="#0b172a", labelcolor="white")
    for i, (a, b) in enumerate(zip(disk, gpu)):
        axes[0].text(i - width / 2, a + 0.15, f"{a:.1f}", ha="center", color="white")
        axes[0].text(i + width / 2, b + 0.15, f"{b:.1f}", ha="center", color="white")
    axes[0].set_ylim(0, 7)

    speed = [row["warm_tokens_per_s_median"] for row in summary]
    axes[1].bar(x, speed, color=colors, width=0.58)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("output tokens/s", color="#dbeafe")
    axes[1].set_title("温間生成速度（3回中央値）", color="white", fontsize=14)
    for i, value in enumerate(speed):
        axes[1].text(i, value + 4, f"{value:.1f}", ha="center", color="white", weight="bold")
    axes[1].set_ylim(0, 210)

    accuracy = [100 * row["mcq_correct"] / row["mcq_total"] for row in summary]
    consistency = [100 * row["order_consistent_questions"] / 24 for row in summary]
    axes[2].bar(x - width / 2, accuracy, width, color="#a78bfa", label="96回答の正答率")
    axes[2].bar(x + width / 2, consistency, width, color="#f59e0b", label="順序一貫 24問")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("percent", color="#dbeafe")
    axes[2].set_title("固定四択probe", color="white", fontsize=14)
    axes[2].legend(facecolor="#0b172a", labelcolor="white", loc="lower right")
    for i, (acc, con, row) in enumerate(zip(accuracy, consistency, summary)):
        axes[2].text(i - width / 2, acc + 1.5, f"{row['mcq_correct']}/96", ha="center", color="white")
        axes[2].text(i + width / 2, con + 1.5, f"{row['order_consistent_questions']}/24", ha="center", color="white")
    axes[2].set_ylim(0, 105)

    fig.suptitle("Gemma 3 4B：Q4_K_MとQ8_0は何が変わる？", color="white", fontsize=19, y=0.965)
    fig.text(
        0.5,
        0.035,
        f"RTX 5070 Ti 16GB / num_ctx 4096 / 256 tokens　自由記述の文字一致度 {data['free_response_character_similarity']*100:.1f}%　※四択は標準benchmarkではない",
        ha="center",
        color="#cbd5e1",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.84, wspace=0.28)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
