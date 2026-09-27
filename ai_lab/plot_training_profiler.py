#!/usr/bin/env python3
"""Visualize measured profiler spans and independent unprofiled throughput."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
RESULT = BASE / "results" / "training_profiler"
FIGURE = BASE.parent / "figures" / "pytorch-profiler-dataloader-wait.png"
STAGES = ("batch_wait", "h2d_transfer", "forward_backward_adam")
COLORS = {"batch_wait": "#4285b4", "h2d_transfer": "#a8b0b8", "forward_backward_adam": "#d88947"}


def first_events(workers: int) -> dict[str, list[dict]]:
    payload = json.loads((RESULT / f"workers_{workers}_trace.json").read_text(encoding="utf-8"))
    spans = {}
    for stage in STAGES:
        spans[stage] = sorted(
            [event for event in payload["traceEvents"]
             if event.get("ph") == "X" and event.get("cat") == "user_annotation" and event.get("name") == stage],
            key=lambda event: event["ts"],
        )[:5]
        if len(spans[stage]) != 5:
            raise ValueError(f"Expected five {stage} events, got {len(spans[stage])}")
    return spans


def main() -> None:
    result = json.loads((RESULT / "result.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.8), layout="constrained",
                              gridspec_kw={"height_ratios": [1.55, 1.55, 0.95]})
    for index, workers in enumerate((0, 4)):
        ax = axes[index]
        spans = first_events(workers)
        origin = spans["batch_wait"][0]["ts"]
        for step_index in range(5):
            for stage in STAGES:
                event = spans[stage][step_index]
                left_ms = (event["ts"] - origin) / 1000.0
                duration_ms = event["dur"] / 1000.0
                ax.broken_barh([(left_ms, duration_ms)], (step_index + 0.65, 0.7),
                               facecolors=COLORS[stage], label=stage if step_index == 0 else None)
        ax.set_ylim(0.4, 5.6)
        ax.invert_yaxis()
        ax.set_yticks(range(1, 6), [f"Step {step}" for step in range(1, 6)])
        ax.set_xlabel("Time from first recorded batch wait (ms)")
        ax.set_title(f"Profiler trace: num_workers={workers} (first 5 steps; each panel has its own x-axis scale)")
        ax.grid(axis="x", alpha=0.2)
        if index == 0:
            ax.legend(loc="upper right", frameon=False, ncol=3)

    ax = axes[2]
    throughput = [result["summary"][str(workers)]["median_images_per_second"] for workers in (0, 4)]
    bars = ax.barh(["0 workers", "4 workers"], throughput, color=["#4285b4", "#d88947"], height=0.55)
    ax.invert_yaxis()
    ax.set_xlim(0, max(throughput) * 1.2)
    ax.set_xlabel("Images/s, median of three unprofiled 100-step runs")
    ax.set_title("Independent timing (profiler off)")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, throughput):
        ax.annotate(f"{value:,.0f}", (value, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", va="center")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=160, facecolor="white")
    plt.close(fig)
    print(FIGURE)


if __name__ == "__main__":
    main()
