"""Validate saved Sony/FX backtest artifacts without rerunning the models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    results = Path(__file__).resolve().parent / "results"
    aggregate = pd.read_csv(results / "sony_fx_backtest_aggregate.csv")
    timings = pd.read_csv(results / "sony_fx_backtest_timings.csv")
    aligned = pd.read_csv(results / "sony_usdjpy_aligned.csv")
    records = json.loads((results / "sony_fx_backtest_per_origin.json").read_text(encoding="utf-8"))
    summary = json.loads((results / "sony_fx_backtest.json").read_text(encoding="utf-8"))
    sony_raw = json.loads((results / "sony_6758t_raw.json").read_text(encoding="utf-8"))
    fx_raw = json.loads((results / "usdjpy_raw.json").read_text(encoding="utf-8"))

    assert len(aggregate) == 9
    assert len(timings) == 40
    assert len(records) == 60
    assert len(aligned) == summary["rows"] == 1627
    assert sony_raw["chart"]["result"] and fx_raw["chart"]["result"]
    assert not aligned[["sony_adjusted_close_jpy", "usd_jpy"]].isna().any().any()
    assert np.isfinite(aligned[["sony_adjusted_close_jpy", "usd_jpy"]].to_numpy()).all()

    record_frame = pd.json_normalize(records)
    assert set(record_frame["model"]) == {
        "last_value",
        "nv_univariate",
        "nv_sony_plus_usdjpy",
    }
    assert record_frame["origin_number"].nunique() == 20
    for origin_number, group in record_frame.groupby("origin_number"):
        assert len(group) == 3, origin_number
        actuals = [np.asarray(value, dtype=float) for value in group["actual"]]
        assert all(np.array_equal(actuals[0], value) for value in actuals[1:])
        assert all(len(value) == 20 and np.isfinite(value).all() for value in actuals)
        for _, row in group.iterrows():
            predicted = np.asarray(row["predicted"], dtype=float)
            assert len(predicted) == 20 and np.isfinite(predicted).all()
            for horizon in (1, 5, 20):
                expected = float(np.mean(np.abs(predicted[:horizon] - actuals[0][:horizon])))
                assert np.isclose(expected, row[f"metrics.{horizon}.mae"], atol=1e-10)

    for _, row in aggregate.iterrows():
        selected = record_frame[record_frame["model"] == row["model"]]
        expected = selected[f"metrics.{int(row['horizon'])}.mae"].mean()
        assert np.isclose(expected, row["mae"], atol=1e-10)

    print("PASS: raw JSON, aligned data, 60 forecasts, metrics, and aggregates are consistent")


if __name__ == "__main__":
    main()
