"""Validate and visualize the saved Toyota learning/regime experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAB = Path(__file__).resolve().parent
RESULTS = LAB / "results" / "toyota_learning_regime"
FIGURES = LAB / "figures"
HORIZONS = (1, 5, 20)
SELECTED = [
    "last_value",
    "nv_univariate",
    "nv_stock_plus_usdjpy",
    "finetuned_short_800_ensemble",
    "finetuned_all_pre_eval_ensemble",
]
LABELS = {
    "last_value": "Last value",
    "nv_univariate": "NV zero-shot",
    "nv_stock_plus_usdjpy": "NV + USD/JPY",
    "finetuned_short_800_ensemble": "Fine-tuned 800 rows",
    "finetuned_all_pre_eval_ensemble": "Fine-tuned 1,067 rows",
}
REGIME_LABELS = {
    "before_2025_04_03_auto_tariff": "Before Apr. 3\n(n=3)",
    "2025_04_03_to_2025_09_04_transition": "Apr. 3–Sep. 4\n(n=6)",
    "after_2025_09_04_japan_agreement_order": "After Sep. 4\n(n=11)",
}


def metric(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = predicted - actual
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.mean(np.where(denominator == 0, 0.0, 200.0 * np.abs(error) / denominator))
    return float(np.mean(np.abs(error))), float(smape)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    records = json.loads((RESULTS / "per_origin.json").read_text(encoding="utf-8"))
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    overall = pd.read_csv(RESULTS / "aggregate_overall.csv")
    regimes = pd.read_csv(RESULTS / "aggregate_by_regime.csv")

    assert summary["validation_ends"] < summary["first_scored_forecast_starts"]
    assert summary["training_history_conditions"] == {"short_800": 800, "all_pre_eval": 1067}
    assert len(records) == 220  # 3 existing + 8 fine-tuned model rows, each at 20 origins
    for epoch_path in (RESULTS / "training").glob("*_seed*/epoch_metrics.json"):
        epochs = json.loads(epoch_path.read_text())
        assert len(epochs) == 5
        assert min(epochs, key=lambda row: row["val_mse"])["epoch"] == 1
    grouped = {}
    for row in records:
        key = (row["origin_number"], row["model"])
        assert key not in grouped
        grouped[key] = row
        actual = np.asarray(row["actual"], dtype=float)
        predicted = np.asarray(row["predicted"], dtype=float)
        assert actual.shape == predicted.shape == (20,)
        assert np.isfinite(actual).all() and np.isfinite(predicted).all()
        for horizon in HORIZONS:
            mae, smape = metric(actual[:horizon], predicted[:horizon])
            saved = row["metrics"][str(horizon)]
            assert np.isclose(mae, saved["mae"], rtol=0, atol=1e-9)
            assert np.isclose(smape, saved["smape_percent"], rtol=0, atol=1e-9)

    for origin in range(1, 21):
        truth = grouped[(origin, "last_value")]["actual"]
        for model in SELECTED[1:]:
            assert np.allclose(grouped[(origin, model)]["actual"], truth, rtol=0, atol=0)
        for history in ("short_800", "all_pre_eval"):
            seeds = np.stack(
                [grouped[(origin, f"finetuned_{history}_seed{seed}")]["predicted"] for seed in (42, 43, 44)]
            )
            ensemble = np.asarray(grouped[(origin, f"finetuned_{history}_ensemble")]["predicted"])
            assert np.allclose(ensemble, seeds.mean(axis=0), rtol=0, atol=1e-10)

    seed_rows = []
    for history in ("short_800", "all_pre_eval"):
        for seed in (42, 43, 44):
            model = f"finetuned_{history}_seed{seed}"
            model_rows = [row for row in records if row["model"] == model]
            for horizon in HORIZONS:
                seed_rows.append(
                    {
                        "history": history,
                        "seed": seed,
                        "horizon": horizon,
                        "mean_smape_percent": float(
                            np.mean([row["metrics"][str(horizon)]["smape_percent"] for row in model_rows])
                        ),
                    }
                )
    pd.DataFrame(seed_rows).to_csv(RESULTS / "seed_stability.csv", index=False)
    selected_regimes = regimes[regimes["model"].isin(SELECTED)].copy()
    selected_regimes.to_csv(RESULTS / "regime_summary_selected.csv", index=False)

    colors = ["#595959", "#d95f02", "#1b9e77", "#7570b3", "#1f78b4"]
    fig, ax = plt.subplots(figsize=(11.5, 6.4), constrained_layout=True)
    x = np.arange(len(HORIZONS))
    width = 0.16
    for index, (model, color) in enumerate(zip(SELECTED, colors, strict=True)):
        values = [
            float(overall[(overall.model == model) & (overall.horizon == horizon)].smape_percent.iloc[0])
            for horizon in HORIZONS
        ]
        bars = ax.bar(x + (index - 2) * width, values, width, label=LABELS[model], color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [f"{h}-day" for h in HORIZONS])
    ax.set_ylabel("Mean sMAPE (%) — lower is better")
    ax.set_title("Toyota: more fine-tuning history helps, but last-value still wins")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=3, frameon=False, loc="upper left")
    fig.savefig(FIGURES / "nv-tesseract-toyota-finetuning-history.png", dpi=180)
    plt.close(fig)

    regime_order = list(REGIME_LABELS)
    regime_models = [
        "last_value",
        "nv_stock_plus_usdjpy",
        "finetuned_short_800_ensemble",
        "finetuned_all_pre_eval_ensemble",
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.4), constrained_layout=True)
    x = np.arange(len(regime_order))
    width = 0.2
    for index, model in enumerate(regime_models):
        values = [
            float(
                regimes[
                    (regimes.regime == label) & (regimes.model == model) & (regimes.horizon == 20)
                ].smape_percent.iloc[0]
            )
            for label in regime_order
        ]
        bars = ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=LABELS[model],
            color=colors[[0, 2, 3, 4][index]],
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [REGIME_LABELS[item] for item in regime_order])
    ax.set_ylabel("20-day mean sMAPE (%) — lower is better")
    ax.set_title("Toyota forecasts by policy time slice (descriptive, not causal)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2, frameon=False, loc="upper right")
    fig.savefig(FIGURES / "nv-tesseract-toyota-policy-regimes.png", dpi=180)
    plt.close(fig)

    print("PASS: leakage boundary, 220 records, metrics, truth, and ensembles verified")
    print(overall[overall.model.isin(SELECTED)].to_string(index=False))
    print("\n20-day regime results")
    print(selected_regimes[selected_regimes.horizon == 20].to_string(index=False))


if __name__ == "__main__":
    main()
