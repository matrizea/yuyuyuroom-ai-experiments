#!/usr/bin/env python3
"""Render Ollama 4K/8K/16K context experiment."""

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
    contexts = data["context_settings"]
    labels = ["4K", "8K", "16K"]
    colors = ["#38bdf8", "#a78bfa", "#fb7185"]
    summaries = [data["summary"][str(ctx)] for ctx in contexts]

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
    x = np.arange(3)

    tokens = [s["prompt_eval_count_median"] for s in summaries]
    axes[0].bar(x, tokens, color=colors, width=0.62)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("実prompt tokens", color="#dbeafe")
    axes[0].set_title("contextの約70%を入力", color="white", fontsize=14)
    for i, (value, summary) in enumerate(zip(tokens, summaries)):
        axes[0].text(i, value + 350, f"{value:,}\n取得 {summary['retrieved']}/3", ha="center", color="white")
    axes[0].set_ylim(0, 13500)

    ttfc = [s["ttfc_after_load_median_s"] for s in summaries]
    axes[1].bar(x, ttfc, color=colors, width=0.62)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("load除外TTFC [s]", color="#dbeafe")
    axes[1].set_title("長文を読んで最初に答えるまで", color="white", fontsize=14)
    for i, value in enumerate(ttfc):
        axes[1].text(i, value + 0.035, f"{value:.3f}s", ha="center", color="white")
    axes[1].set_ylim(0, 1.2)

    total_gpu = [s["peak_gpu_used_mib"] / 1024 for s in summaries]
    ollama_vram = [s["ollama_size_vram_bytes"] / 1024**3 for s in summaries]
    width = 0.34
    axes[2].bar(x - width / 2, total_gpu, width, color="#34d399", label="nvidia-smi peak")
    axes[2].bar(x + width / 2, ollama_vram, width, color="#fbbf24", label="Ollama size_vram")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("GiB", color="#dbeafe")
    axes[2].set_title("context設定とGPU memory", color="white", fontsize=14)
    axes[2].legend(facecolor="#0b172a", labelcolor="white", loc="lower right")
    for i, (a, b) in enumerate(zip(total_gpu, ollama_vram)):
        axes[2].text(i - width / 2, a + 0.12, f"{a:.2f}", ha="center", color="white", fontsize=9)
        axes[2].text(i + width / 2, b + 0.12, f"{b:.2f}", ha="center", color="white", fontsize=9)
    axes[2].set_ylim(0, 5.8)

    speed_text = " / ".join(f"{s['prompt_eval_tokens_per_s_median']/1000:.1f}k" for s in summaries)
    fig.suptitle("Ollama context 4K / 8K / 16K：長文とVRAMを実測", color="white", fontsize=19, y=0.965)
    fig.text(
        0.5,
        0.035,
        f"Gemma 3 4B Q4_K_M / beginning・middle・end　prompt処理速度: {speed_text} tok/s（4K / 8K / 16K）",
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
