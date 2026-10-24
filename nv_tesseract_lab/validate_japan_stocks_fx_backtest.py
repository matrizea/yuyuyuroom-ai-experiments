"""Validate the five-stock backtest artifacts without rerunning inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    results = Path(__file__).resolve().parent / "results" / "japan_stocks_fx"
    records = json.loads((results / "per_origin.json").read_text(encoding="utf-8"))
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    per_stock = pd.read_csv(results / "per_stock_aggregate.csv")
    panel = pd.read_csv(results / "panel_aggregate.csv")
    timings = pd.read_csv(results / "timings.csv")
    frame = pd.json_normalize(records)

    assert len(records) == 5 * 20 * 3
    assert len(timings) == 5 * 20 * 2
    assert len(per_stock) == 5 * 3 * 3
    assert len(panel) == 3 * 3
    assert len(summary["stocks"]) == 5
    assert set(frame["model"]) == {
        "last_value",
        "nv_univariate",
        "nv_stock_plus_usdjpy",
    }
    for (symbol, origin), group in frame.groupby(["symbol", "origin_number"]):
        assert len(group) == 3, (symbol, origin)
        actuals = [np.asarray(value, dtype=float) for value in group["actual"]]
        assert all(np.array_equal(actuals[0], value) for value in actuals[1:])
        for _, row in group.iterrows():
            predicted = np.asarray(row["predicted"], dtype=float)
            assert len(predicted) == len(actuals[0]) == 20
            assert np.isfinite(predicted).all() and np.isfinite(actuals[0]).all()
            for horizon in (1, 5, 20):
                mae = np.mean(np.abs(predicted[:horizon] - actuals[0][:horizon]))
                assert np.isclose(mae, row[f"metrics.{horizon}.mae"], atol=1e-10)
    for _, row in per_stock.iterrows():
        selected = frame[(frame["symbol"] == row["symbol"]) & (frame["model"] == row["model"])]
        expected = selected[f"metrics.{int(row['horizon'])}.mae"].mean()
        assert np.isclose(expected, row["mae"], atol=1e-10)
    for _, row in panel.iterrows():
        selected = per_stock[
            (per_stock["model"] == row["model"]) & (per_stock["horizon"] == row["horizon"])
        ]
        assert np.isclose(selected["mae"].mean(), row["macro_mae"], atol=1e-10)
        assert np.isclose(
            selected["smape_percent"].mean(), row["macro_smape_percent"], atol=1e-10
        )
    print("PASS: 300 forecasts, 200 timings, per-stock metrics, and panel aggregates are consistent")


if __name__ == "__main__":
    main()
