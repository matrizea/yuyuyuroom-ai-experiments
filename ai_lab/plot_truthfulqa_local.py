#!/usr/bin/env python3
"""Render local TruthfulQA two-order benchmark results."""

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
    labels = ["Gemma 3\n4B", "Qwen 3\n8B*", "Gemma 3\n12B"]
    summaries = [data["summaries"][model] for model in models]
    colors = ["#38bdf8", "#a78bfa", "#fb7185"]

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
    width = 0.34

    response_acc = [100 * s["correct_responses"] / s["total_responses"] for s in summaries]
    strict_acc = [100 * s["strict_both_orders_correct"] / s["questions"] for s in summaries]
    axes[0].bar(x - width / 2, response_acc, width, color="#34d399", label="148 response")
    axes[0].bar(x + width / 2, strict_acc, width, color="#fbbf24", label="両順序正解 74問")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("accuracy [%]", color="#dbeafe")
    axes[0].set_title("TruthfulQA 2択", color="white", fontsize=14)
    axes[0].legend(facecolor="#0b172a", labelcolor="white", loc="lower right")
    for i, (a, b) in enumerate(zip(response_acc, strict_acc)):
        axes[0].text(i - width / 2, a + 1.3, f"{a:.1f}", ha="center", color="white", fontsize=9)
        axes[0].text(i + width / 2, b + 1.3, f"{b:.1f}", ha="center", color="white", fontsize=9)
    axes[0].set_ylim(0, 100)

    both_wrong = [100 * s["both_orders_wrong"] / s["questions"] for s in summaries]
    consistent = [100 * s["content_consistent"] / s["questions"] for s in summaries]
    axes[1].bar(x - width / 2, both_wrong, width, color="#f87171", label="両順序で誤答")
    axes[1].bar(x + width / 2, consistent, width, color="#60a5fa", label="answer内容が一貫")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("questions [%]", color="#dbeafe")
    axes[1].set_title("誤答と順序頑健性", color="white", fontsize=14)
    axes[1].legend(facecolor="#0b172a", labelcolor="white", loc="center right")
    for i, (a, b, s) in enumerate(zip(both_wrong, consistent, summaries)):
        axes[1].text(i - width / 2, a + 1.4, f"{s['both_orders_wrong']}/74", ha="center", color="white", fontsize=9)
        axes[1].text(i + width / 2, b + 1.4, f"{s['content_consistent']}/74", ha="center", color="white", fontsize=9)
    axes[1].set_ylim(0, 105)

    correct_a = []
    correct_b = []
    for model in models:
        items = data["evaluations"][model]
        correct_a.append(100 * sum(item["orders"][0]["is_correct"] for item in items) / len(items))
        correct_b.append(100 * sum(item["orders"][1]["is_correct"] for item in items) / len(items))
    axes[2].bar(x - width / 2, correct_a, width, color="#c084fc", label="正解をAに配置")
    axes[2].bar(x + width / 2, correct_b, width, color="#fb923c", label="正解をBに配置")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("accuracy [%]", color="#dbeafe")
    axes[2].set_title("choice position bias", color="white", fontsize=14)
    axes[2].legend(facecolor="#0b172a", labelcolor="white", loc="lower right")
    for i, (a, b) in enumerate(zip(correct_a, correct_b)):
        axes[2].text(i - width / 2, a + 1.3, f"{a:.1f}", ha="center", color="white", fontsize=9)
        axes[2].text(i + width / 2, b + 1.3, f"{b:.1f}", ha="center", color="white", fontsize=9)
    axes[2].set_ylim(0, 100)

    fig.suptitle("ローカルLLM 4B / 8B / 12BをTruthfulQAで検証", color="white", fontsize=19, y=0.965)
    fig.text(
        0.5,
        0.035,
        "TruthfulQA 2025 two-choice / 37 categories × 2 questions / correctをA・B両方へ配置　*8Bのみ別model family",
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
