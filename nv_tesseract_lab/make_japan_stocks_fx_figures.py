"""Create figures from the fixed five-stock NV-Tesseract panel artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "last_value": "Last value",
    "nv_univariate": "NV-Tesseract: stock only",
    "nv_stock_plus_usdjpy": "NV-Tesseract: stock + USD/JPY",
}
COLORS = {
    "last_value": "#6b7280",
    "nv_univariate": "#e45756",
    "nv_stock_plus_usdjpy": "#2a9d8f",
}
COMPANIES = {
    "6758.T": "Sony",
    "7203.T": "Toyota",
    "7974.T": "Nintendo",
    "9984.T": "SoftBank Group",
    "8306.T": "MUFG",
}


def main() -> None:
    root = Path(__file__).resolve().parent
    results = root / "results" / "japan_stocks_fx"
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    panel = pd.read_csv(results / "panel_aggregate.csv")
    stocks = pd.read_csv(results / "per_stock_aggregate.csv")
    models = list(LABELS)
    horizons = [1, 5, 20]

    fig, (macro_ax, stock_ax) = plt.subplots(2, 1, figsize=(11.5, 9.4), constrained_layout=True)
    x = np.arange(len(horizons))
    width = 0.24
    for index, model in enumerate(models):
        subset = panel[panel["model"] == model].set_index("horizon").loc[horizons]
        macro_ax.bar(
            x + (index - 1) * width,
            subset["macro_smape_percent"],
            width,
            label=LABELS[model],
            color=COLORS[model],
        )
    macro_ax.set_title("Five-stock panel: average percentage forecast error")
    macro_ax.set_ylabel("Macro sMAPE (%) — lower is better")
    macro_ax.set_xticks(x, ["1 day", "5 days", "20 days"])
    macro_ax.grid(axis="y", alpha=0.25)
    macro_ax.legend(fontsize=9)

    symbols = list(COMPANIES)
    x2 = np.arange(len(symbols))
    for index, model in enumerate(models):
        subset = (
            stocks[(stocks["model"] == model) & (stocks["horizon"] == 20)]
            .set_index("symbol")
            .loc[symbols]
        )
        stock_ax.bar(
            x2 + (index - 1) * width,
            subset["smape_percent"],
            width,
            label=LABELS[model],
            color=COLORS[model],
        )
    stock_ax.set_title("20-day error by stock (20 held-out origins each)")
    stock_ax.set_ylabel("sMAPE (%) — lower is better")
    stock_ax.set_xticks(x2, [COMPANIES[symbol] for symbol in symbols])
    stock_ax.grid(axis="y", alpha=0.25)
    stock_ax.legend(fontsize=9)
    output = figures / "nv-tesseract-five-stocks-usdjpy-backtest.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)

    delta = stocks[stocks["model"] != "last_value"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True, constrained_layout=True)
    for ax, horizon in zip(axes, horizons, strict=True):
        subset = delta[delta["horizon"] == horizon]
        for model, marker in (("nv_univariate", "o"), ("nv_stock_plus_usdjpy", "s")):
            part = subset[subset["model"] == model].set_index("symbol").loc[symbols]
            ax.scatter(
                part["mean_mae_delta_vs_last_value"],
                np.arange(len(symbols)),
                color=COLORS[model],
                marker=marker,
                s=55,
                label=LABELS[model],
            )
        ax.axvline(0, color="#111827", linewidth=1.1)
        ax.set_title(f"{horizon}-day horizon")
        ax.set_xlabel("Mean MAE difference (JPY; left wins)")
        ax.grid(alpha=0.22)
    axes[0].set_yticks(np.arange(len(symbols)), [COMPANIES[symbol] for symbol in symbols])
    axes[-1].legend(fontsize=8, loc="lower right")
    delta_output = figures / "nv-tesseract-five-stocks-mae-delta.png"
    fig.savefig(delta_output, dpi=180)
    plt.close(fig)
    print(output)
    print(delta_output)


if __name__ == "__main__":
    main()
