"""Create non-cherry-picked figures from the saved Sony/FX backtest artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "last_value": "Last value",
    "nv_univariate": "NV-Tesseract: Sony only",
    "nv_sony_plus_usdjpy": "NV-Tesseract: Sony + USD/JPY",
}
COLORS = {
    "last_value": "#6b7280",
    "nv_univariate": "#e45756",
    "nv_sony_plus_usdjpy": "#2a9d8f",
}


def main() -> None:
    root = Path(__file__).resolve().parent
    results = root / "results"
    figures = root / "figures"
    figures.mkdir(exist_ok=True)

    aggregate = pd.read_csv(results / "sony_fx_backtest_aggregate.csv")
    records = json.loads((results / "sony_fx_backtest_per_origin.json").read_text(encoding="utf-8"))
    per_origin = pd.json_normalize(records)

    models = list(LABELS)
    horizons = [1, 5, 20]
    x = np.arange(len(horizons))
    width = 0.24

    fig, (bar_ax, dot_ax) = plt.subplots(2, 1, figsize=(11.5, 9), constrained_layout=True)
    for index, model in enumerate(models):
        subset = aggregate[aggregate["model"] == model].set_index("horizon").loc[horizons]
        bar_ax.bar(
            x + (index - 1) * width,
            subset["mae"],
            width,
            label=LABELS[model],
            color=COLORS[model],
        )
    bar_ax.set_title("Sony adjusted-close forecast error across 20 rolling origins")
    bar_ax.set_ylabel("Mean absolute error (JPY; lower is better)")
    bar_ax.set_xticks(x, ["1 day", "5 days", "20 days"])
    bar_ax.grid(axis="y", alpha=0.25)
    bar_ax.legend(fontsize=9)

    baseline = per_origin[per_origin["model"] == "last_value"].set_index("origin_number")
    positions = np.arange(1, len(baseline) + 1)
    for model in models[1:]:
        model_rows = per_origin[per_origin["model"] == model].set_index("origin_number")
        delta = model_rows["metrics.20.mae"] - baseline["metrics.20.mae"]
        dot_ax.scatter(
            positions,
            delta,
            label=LABELS[model],
            color=COLORS[model],
            s=38,
            alpha=0.85,
        )
    dot_ax.axhline(0, color="#111827", linewidth=1.2)
    dot_ax.set_title("Each origin: 20-day MAE difference from last-value baseline")
    dot_ax.set_xlabel("Backtest origin (chronological order)")
    dot_ax.set_ylabel("MAE difference (JPY; below zero wins)")
    dot_ax.set_xticks([1, 5, 10, 15, 20])
    dot_ax.grid(alpha=0.2)
    dot_ax.legend(fontsize=9)
    comparison_path = figures / "sony-fx-backtest-all-origins.png"
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)

    latest_origin = max(item["origin_number"] for item in records)
    latest = {item["model"]: item for item in records if item["origin_number"] == latest_origin}
    aligned = pd.read_csv(results / "sony_usdjpy_aligned.csv", parse_dates=["timestamp"])
    origin_date = pd.Timestamp(latest["last_value"]["origin_date"])
    future = aligned[aligned["timestamp"] > origin_date].head(20)
    if len(future) != 20:
        raise RuntimeError(f"Expected 20 future rows for latest origin, got {len(future)}")

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    history = aligned[aligned["timestamp"] <= origin_date].tail(60)
    ax.plot(history["timestamp"], history["sony_adjusted_close_jpy"], color="#2563eb", label="Observed history")
    ax.plot(future["timestamp"], latest["last_value"]["actual"], color="#111827", linewidth=2.1, label="Actual")
    for model in models:
        ax.plot(
            future["timestamp"],
            latest[model]["predicted"],
            color=COLORS[model],
            linestyle="--",
            linewidth=1.8,
            label=LABELS[model],
        )
    ax.axvline(origin_date, color="#6b7280", linestyle=":", linewidth=1.2)
    ax.set_title("Latest held-out 20-trading-day window")
    ax.set_xlabel("Market date")
    ax.set_ylabel("Sony adjusted close (JPY)")
    ax.grid(alpha=0.22)
    ax.legend(ncol=2, fontsize=9)
    latest_path = figures / "sony-fx-backtest-latest-window.png"
    fig.savefig(latest_path, dpi=180)
    plt.close(fig)

    print(comparison_path)
    print(latest_path)


if __name__ == "__main__":
    main()
