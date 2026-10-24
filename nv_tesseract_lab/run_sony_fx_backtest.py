"""Rolling backtest of NV-Tesseract on Sony Group stock with USD/JPY.

This is a model-evaluation experiment, not an investment recommendation.
Run from the NV-Tesseract ``forecasting`` directory with its existing venv.
The script downloads public daily Yahoo Finance chart data, preserves the raw
responses, and compares NV-Tesseract against a last-value baseline on hidden,
already-known periods.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch


SEED = 42
SEQ_LEN = 512
FORECAST_HORIZON = 20
MODEL_HORIZON = 72
N_ORIGINS = 20
ORIGIN_STEP = 20  # non-overlapping 20-trading-day evaluation windows
HORIZONS = (1, 5, 20)
SONY_SYMBOL = "6758.T"
FX_SYMBOL = "JPY=X"  # Yahoo Finance symbol for USD/JPY
TARGET_COLUMN = "sony_adjusted_close_jpy"
FX_COLUMN = "usd_jpy"
TIMESTAMP_COLUMN = "timestamp"
PERIOD1 = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
PERIOD2 = int(datetime(2027, 1, 1, tzinfo=timezone.utc).timestamp())


@dataclass(frozen=True)
class MetricRow:
    model: str
    horizon: int
    mae: float
    rmse: float
    smape_percent: float
    endpoint_direction_accuracy: float | None
    origin_win_rate_vs_last_value: float | None
    mean_mae_delta_vs_last_value: float | None
    bootstrap95_mae_delta_low: float | None
    bootstrap95_mae_delta_high: float | None


def yahoo_chart_url(symbol: str) -> str:
    query = urllib.parse.urlencode(
        {
            "period1": PERIOD1,
            "period2": PERIOD2,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?{query}"


def download_chart(symbol: str, raw_path: Path) -> pd.DataFrame:
    request = urllib.request.Request(
        yahoo_chart_url(symbol),
        headers={"User-Agent": "Mozilla/5.0 NV-Tesseract reproducible backtest"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    raw_path.write_bytes(payload)
    document = json.loads(payload)
    result = document["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adjclose is None:
        adjclose = quote["close"]
    frame = pd.DataFrame(
        {
            "timestamp_raw": result["timestamp"],
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "adjusted_close": adjclose,
            "volume": quote.get("volume", [None] * len(result["timestamp"])),
        }
    )
    # Exchange timestamps are normalized to the UTC calendar date for a stable
    # daily join. Rows without a completed adjusted close are excluded.
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        frame.pop("timestamp_raw"), unit="s", utc=True
    ).dt.tz_localize(None).dt.normalize()
    frame = frame.dropna(subset=["adjusted_close"]).drop_duplicates(TIMESTAMP_COLUMN)
    frame = frame.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    return frame


def metric_values(actual: np.ndarray, predicted: np.ndarray, last_input: float) -> dict[str, float]:
    error = predicted - actual
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.mean(np.where(denominator == 0, 0.0, 200.0 * np.abs(error) / denominator))
    actual_direction = np.sign(actual[-1] - last_input)
    predicted_direction = np.sign(predicted[-1] - last_input)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "smape_percent": float(smape),
        "endpoint_direction_correct": float(actual_direction == predicted_direction),
    }


def aggregate_metrics(per_origin: list[dict[str, Any]]) -> list[MetricRow]:
    rows: list[MetricRow] = []
    naive_mae: dict[int, dict[str, float]] = {}
    for horizon in HORIZONS:
        naive_mae[horizon] = {
            row["origin_date"]: row["metrics"][str(horizon)]["mae"]
            for row in per_origin
            if row["model"] == "last_value"
        }

    for model in ("last_value", "nv_univariate", "nv_sony_plus_usdjpy"):
        model_rows = [row for row in per_origin if row["model"] == model]
        for horizon in HORIZONS:
            values = [row["metrics"][str(horizon)] for row in model_rows]
            wins: float | None = None
            mean_delta: float | None = None
            bootstrap_low: float | None = None
            bootstrap_high: float | None = None
            if model != "last_value":
                differences = np.array(
                    [
                        value["mae"] - naive_mae[horizon][row["origin_date"]]
                        for row, value in zip(model_rows, values, strict=True)
                    ],
                    dtype=float,
                )
                wins = float(
                    np.mean(differences < 0)
                )
                mean_delta = float(np.mean(differences))
                rng = np.random.default_rng(SEED + horizon + (0 if model == "nv_univariate" else 1000))
                bootstrap_means = np.mean(
                    rng.choice(differences, size=(10_000, len(differences)), replace=True),
                    axis=1,
                )
                bootstrap_low, bootstrap_high = [
                    float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975])
                ]
            rows.append(
                MetricRow(
                    model=model,
                    horizon=horizon,
                    mae=float(np.mean([value["mae"] for value in values])),
                    rmse=float(np.sqrt(np.mean([value["rmse"] ** 2 for value in values]))),
                    smape_percent=float(np.mean([value["smape_percent"] for value in values])),
                    endpoint_direction_accuracy=(
                        None
                        if model == "last_value"
                        else float(np.mean([value["endpoint_direction_correct"] for value in values]))
                    ),
                    origin_win_rate_vs_last_value=wins,
                    mean_mae_delta_vs_last_value=mean_delta,
                    bootstrap95_mae_delta_low=bootstrap_low,
                    bootstrap95_mae_delta_high=bootstrap_high,
                )
            )
    return rows


def main() -> None:
    forecasting_root = Path.cwd()
    if not (forecasting_root / "sdk").is_dir():
        raise SystemExit("Run from NV-Tesseract/forecasting")
    sys.path.insert(0, str(forecasting_root))
    from sdk.forecasting import ForecastingConfig, perform_forecasting

    results_dir = Path(
        os.environ.get(
            "NV_TESSERACT_RESULTS_DIR",
            forecasting_root.parent.parent / "results",
        )
    ).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    sony = download_chart(SONY_SYMBOL, results_dir / "sony_6758t_raw.json")
    fx = download_chart(FX_SYMBOL, results_dir / "usdjpy_raw.json")
    sony_daily = sony[[TIMESTAMP_COLUMN, "adjusted_close"]].rename(
        columns={"adjusted_close": TARGET_COLUMN}
    )
    fx_daily = fx[[TIMESTAMP_COLUMN, "adjusted_close"]].rename(
        columns={"adjusted_close": FX_COLUMN}
    )
    combined = sony_daily.merge(fx_daily, on=TIMESTAMP_COLUMN, how="left").sort_values(
        TIMESTAMP_COLUMN
    )
    # Only past FX values may fill a Tokyo trading date with no exact FX row.
    combined[FX_COLUMN] = combined[FX_COLUMN].ffill()
    combined = combined.dropna().reset_index(drop=True)
    # Avoid treating an in-progress same-day quote as a completed daily close.
    today_jst = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Tokyo")).date()
    combined = combined[combined[TIMESTAMP_COLUMN].dt.date < today_jst].reset_index(drop=True)
    combined.to_csv(results_dir / "sony_usdjpy_aligned.csv", index=False)

    minimum_rows = SEQ_LEN + FORECAST_HORIZON + (N_ORIGINS - 1) * ORIGIN_STEP
    if len(combined) < minimum_rows:
        raise RuntimeError(f"Need at least {minimum_rows} aligned rows, got {len(combined)}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    config_common = dict(
        seq_len=SEQ_LEN,
        forecast_horizon=FORECAST_HORIZON,
        model_horizon=MODEL_HORIZON,
        timestamp_column=TIMESTAMP_COLUMN,
        target_column=TARGET_COLUMN,
        seed=SEED,
        batch_size=1,
        num_workers=0,
        save_preds=None,
    )
    origins = [
        len(combined) - FORECAST_HORIZON - ORIGIN_STEP * offset
        for offset in range(N_ORIGINS - 1, -1, -1)
    ]
    per_origin: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []

    for origin_number, origin in enumerate(origins, start=1):
        train = combined.iloc[origin - SEQ_LEN : origin].copy()
        truth = combined.iloc[origin : origin + FORECAST_HORIZON].copy()
        actual = truth[TARGET_COLUMN].to_numpy(dtype=float)
        last_input = float(train[TARGET_COLUMN].iloc[-1])

        predictions: dict[str, np.ndarray] = {
            "last_value": np.full(FORECAST_HORIZON, last_input, dtype=float)
        }
        for model_name, cross_channel, columns in (
            ("nv_univariate", False, [TIMESTAMP_COLUMN, TARGET_COLUMN]),
            (
                "nv_sony_plus_usdjpy",
                True,
                [TIMESTAMP_COLUMN, TARGET_COLUMN, FX_COLUMN],
            ),
        ):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            forecast = perform_forecasting(
                df=train[columns],
                config=ForecastingConfig(
                    **config_common,
                    use_cross_channel=cross_channel,
                ),
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            predicted = forecast[f"{TARGET_COLUMN}_forecast"].to_numpy(dtype=float)
            if len(predicted) != FORECAST_HORIZON or not np.isfinite(predicted).all():
                raise RuntimeError(f"Invalid prediction for {model_name} at origin {origin_number}")
            predictions[model_name] = predicted
            timings.append(
                {
                    "origin_number": origin_number,
                    "origin_date": str(train[TIMESTAMP_COLUMN].iloc[-1].date()),
                    "model": model_name,
                    "elapsed_seconds": elapsed,
                }
            )

        for model_name, predicted in predictions.items():
            metrics = {
                str(horizon): metric_values(
                    actual[:horizon], predicted[:horizon], last_input
                )
                for horizon in HORIZONS
            }
            per_origin.append(
                {
                    "origin_number": origin_number,
                    "origin_date": str(train[TIMESTAMP_COLUMN].iloc[-1].date()),
                    "forecast_start": str(truth[TIMESTAMP_COLUMN].iloc[0].date()),
                    "forecast_end": str(truth[TIMESTAMP_COLUMN].iloc[-1].date()),
                    "model": model_name,
                    "last_input": last_input,
                    "actual": actual.tolist(),
                    "predicted": predicted.tolist(),
                    "metrics": metrics,
                }
            )
        print(
            f"origin {origin_number:02d}/{N_ORIGINS}: "
            f"{train[TIMESTAMP_COLUMN].iloc[-1].date()} -> {truth[TIMESTAMP_COLUMN].iloc[-1].date()}",
            flush=True,
        )

    aggregate = aggregate_metrics(per_origin)
    aggregate_frame = pd.DataFrame([asdict(row) for row in aggregate])
    aggregate_frame.to_csv(results_dir / "sony_fx_backtest_aggregate.csv", index=False)
    pd.DataFrame(timings).to_csv(results_dir / "sony_fx_backtest_timings.csv", index=False)
    (results_dir / "sony_fx_backtest_per_origin.json").write_text(
        json.dumps(per_origin, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sony_returns = combined[TARGET_COLUMN].pct_change()
    fx_returns = combined[FX_COLUMN].pct_change()
    metadata = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "model evaluation only; not investment advice",
        "symbols": {"sony": SONY_SYMBOL, "usd_jpy": FX_SYMBOL},
        "data_source": "Yahoo Finance chart API",
        "sony_code_verified_against": "Sony Group investor relations: TSE code 6758",
        "price_field": "adjusted_close",
        "alignment": "Sony trading dates; exact-date USD/JPY joined and past-only forward fill",
        "rows": len(combined),
        "data_start": str(combined[TIMESTAMP_COLUMN].iloc[0].date()),
        "data_end": str(combined[TIMESTAMP_COLUMN].iloc[-1].date()),
        "return_correlation_sony_usdjpy": float(sony_returns.corr(fx_returns)),
        "seq_len": SEQ_LEN,
        "forecast_horizon": FORECAST_HORIZON,
        "model_horizon": MODEL_HORIZON,
        "origins": N_ORIGINS,
        "origin_step": ORIGIN_STEP,
        "horizons": list(HORIZONS),
        "seed": SEED,
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=forecasting_root, text=True
        ).strip(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "aggregate": [asdict(row) for row in aggregate],
        "timing_summary_seconds": {
            model: {
                "first": float(next(t["elapsed_seconds"] for t in timings if t["model"] == model)),
                "median": float(
                    np.median([t["elapsed_seconds"] for t in timings if t["model"] == model])
                ),
                "max": float(max(t["elapsed_seconds"] for t in timings if t["model"] == model)),
            }
            for model in ("nv_univariate", "nv_sony_plus_usdjpy")
        },
    }
    (results_dir / "sony_fx_backtest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(aggregate_frame.to_string(index=False), flush=True)
    print(json.dumps(metadata["timing_summary_seconds"], indent=2), flush=True)
    print(f"raw_result={results_dir / 'sony_fx_backtest.json'}", flush=True)


if __name__ == "__main__":
    main()
