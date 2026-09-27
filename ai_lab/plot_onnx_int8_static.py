#!/usr/bin/env python3
"""Make an evidence chart from the recorded ONNX INT8 experiment JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
DATA = BASE / "results" / "onnx_int8_static" / "preprocessed" / "result.json"
OUTPUT = BASE.parent / "figures" / "onnx-int8-static-cpu-comparison.png"


def main() -> None:
    result = json.loads(DATA.read_text(encoding="utf-8"))
    batches = [1, 32, 256]
    fp32_ms = [result["timings"][str(batch)]["fp32"]["median_ms"] for batch in batches]
    int8_ms = [result["timings"][str(batch)]["int8"]["median_ms"] for batch in batches]
    fp32_kib = result["artifacts"]["fp32"]["bytes"] / 1024
    int8_kib = result["artifacts"]["int8"]["bytes"] / 1024

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    blue, amber = "#2979a8", "#da8142"

    x = np.arange(len(batches))
    width = 0.34
    axes[0].bar(x - width / 2, fp32_ms, width, label="FP32", color=blue)
    axes[0].bar(x + width / 2, int8_ms, width, label="INT8", color=amber)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, [str(batch) for batch in batches])
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("Median CPU inference time (ms, log scale)")
    axes[0].set_title("Latency: smaller is better")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    for offset, values in ((-width / 2, fp32_ms), (width / 2, int8_ms)):
        for center, value in zip(x + offset, values):
            axes[0].annotate(f"{value:.3f}", (center, value), xytext=(0, 4),
                             textcoords="offset points", ha="center", fontsize=8)

    axes[1].bar(["FP32", "INT8"], [fp32_kib, int8_kib], color=[blue, amber], width=0.55)
    axes[1].set_ylim(0, max(fp32_kib, int8_kib) * 1.2)
    axes[1].set_ylabel("ONNX file size (KiB)")
    axes[1].set_title("Stored model: smaller is better")
    axes[1].grid(axis="y", alpha=0.2)
    for index, value in enumerate((fp32_kib, int8_kib)):
        axes[1].annotate(f"{value:.1f} KiB", (index, value), xytext=(0, 4),
                         textcoords="offset points", ha="center")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=170, facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
