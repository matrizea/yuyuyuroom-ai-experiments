"""Backtest NV-Tesseract on five Japanese stocks with and without USD/JPY.

This is a model-evaluation experiment, not an investment recommendation.
The five-stock panel is fixed before inference to avoid selecting only favorable
symbols after seeing results. Run from NV-Tesseract/forecasting.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from run_sony_fx_backtest import (
    FORECAST_HORIZON,
    FX_COLUMN,
    FX_SYMBOL,
    HORIZONS,
    MODEL_HORIZON,
    N_ORIGINS,
    ORIGIN_STEP,
    SEED,
    SEQ_LEN,
    TIMESTAMP_COLUMN,
    MetricRow,
    download_chart,
    metric_values,
)


STOCKS = {
    "6758.T": {"company": "Sony Group", "sector_role": "electronics/media"},
    "7203.T": {"company": "Toyota Motor", "sector_role": "automobile/exporter"},
    "7974.T": {"company": "Nintendo", "sector_role": "games/global entertainment"},
    "9984.T": {"company": "SoftBank Group", "sector_role": "technology investment"},
    "8306.T": {"company": "Mitsubishi UFJ Financial Group", "sector_role": "banking"},
}
TARGET_COLUMN = "adjusted_close_jpy"


def aggregate_one_stock(records: list[dict[str, Any]]) -> list[MetricRow]:
    rows: list[MetricRow] = []
    baseline = {
        horizon: {
            row["origin_number"]: row["metrics"][str(horizon)]["mae"]
            for row in records
            if row["model"] == "last_value"
        }
        for horizon in HORIZONS
    }
    for model in ("last_value", "nv_univariate", "nv_stock_plus_usdjpy"):
        model_rows = [row for row in records if row["model"] == model]
        for horizon in HORIZONS:
            values = [row["metrics"][str(horizon)] for row in model_rows]
            differences = None
            if model != "last_value":
                differences = np.asarray(
                    [
                        value["mae"] - baseline[horizon][row["origin_number"]]
                        for row, value in zip(model_rows, values, strict=True)
                    ]
                )
                rng = np.random.default_rng(
                    SEED + horizon + (0 if model == "nv_univariate" else 1000)
                )
                bootstrap = rng.choice(
                    differences, size=(10_000, len(differences)), replace=True
                ).mean(axis=1)
                low, high = np.quantile(bootstrap, [0.025, 0.975])
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
                    origin_win_rate_vs_last_value=(
                        None if differences is None else float(np.mean(differences < 0))
                    ),
                    mean_mae_delta_vs_last_value=(
                        None if differences is None else float(np.mean(differences))
                    ),
                    bootstrap95_mae_delta_low=(None if differences is None else float(low)),
                    bootstrap95_mae_delta_high=(None if differences is None else float(high)),
                )
            )
    return rows


def aggregate_panel(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Macro-average stocks; bootstrap time origins as correlated market blocks."""
    frame = pd.json_normalize(records)
    output: list[dict[str, Any]] = []
    for model in ("last_value", "nv_univariate", "nv_stock_plus_usdjpy"):
        model_frame = frame[frame["model"] == model]
        for horizon in HORIZONS:
            mae_column = f"metrics.{horizon}.mae"
            rmse_column = f"metrics.{horizon}.rmse"
            smape_column = f"metrics.{horizon}.smape_percent"
            direction_column = f"metrics.{horizon}.endpoint_direction_correct"
            stock_means = model_frame.groupby("symbol")[[mae_column, smape_column]].mean()
            row: dict[str, Any] = {
                "model": model,
                "horizon": horizon,
                "macro_mae": float(stock_means[mae_column].mean()),
                "pooled_rmse": float(np.sqrt(np.mean(np.square(model_frame[rmse_column])))),
                "macro_smape_percent": float(stock_means[smape_column].mean()),
                "endpoint_direction_accuracy": (
                    None if model == "last_value" else float(model_frame[direction_column].mean())
                ),
                "origin_stock_win_rate_vs_last_value": None,
                "mean_mae_delta_vs_last_value": None,
                "time_block_bootstrap95_delta_low": None,
                "time_block_bootstrap95_delta_high": None,
                "mean_smape_delta_vs_last_value": None,
                "time_block_bootstrap95_smape_delta_low": None,
                "time_block_bootstrap95_smape_delta_high": None,
            }
            if model != "last_value":
                baseline = frame[frame["model"] == "last_value"][
                    ["symbol", "origin_number", mae_column]
                ].rename(columns={mae_column: "baseline_mae"})
                paired = model_frame.merge(baseline, on=["symbol", "origin_number"])
                paired["delta"] = paired[mae_column] - paired["baseline_mae"]
                origin_blocks = paired.groupby("origin_number")["delta"].mean().to_numpy()
                rng = np.random.default_rng(
                    SEED + horizon + (0 if model == "nv_univariate" else 1000)
                )
                bootstrap = rng.choice(
                    origin_blocks, size=(10_000, len(origin_blocks)), replace=True
                ).mean(axis=1)
                low, high = np.quantile(bootstrap, [0.025, 0.975])
                baseline_smape = frame[frame["model"] == "last_value"][
                    ["symbol", "origin_number", smape_column]
                ].rename(columns={smape_column: "baseline_smape"})
                paired_smape = model_frame.merge(
                    baseline_smape, on=["symbol", "origin_number"]
                )
                paired_smape["smape_delta"] = (
                    paired_smape[smape_column] - paired_smape["baseline_smape"]
                )
                smape_origin_blocks = (
                    paired_smape.groupby("origin_number")["smape_delta"].mean().to_numpy()
                )
                smape_bootstrap = rng.choice(
                    smape_origin_blocks,
                    size=(10_000, len(smape_origin_blocks)),
                    replace=True,
                ).mean(axis=1)
                smape_low, smape_high = np.quantile(smape_bootstrap, [0.025, 0.975])
                row.update(
                    {
                        "origin_stock_win_rate_vs_last_value": float(np.mean(paired["delta"] < 0)),
                        "mean_mae_delta_vs_last_value": float(paired["delta"].mean()),
                        "time_block_bootstrap95_delta_low": float(low),
                        "time_block_bootstrap95_delta_high": float(high),
                        "mean_smape_delta_vs_last_value": float(
                            paired_smape["smape_delta"].mean()
                        ),
                        "time_block_bootstrap95_smape_delta_low": float(smape_low),
                        "time_block_bootstrap95_smape_delta_high": float(smape_high),
                    }
                )
            output.append(row)
    return output


def main() -> None:
    forecasting_root = Path.cwd()
    if not (forecasting_root / "sdk").is_dir():
        raise SystemExit("Run from NV-Tesseract/forecasting")
    sys.path.insert(0, str(forecasting_root))
    from sdk.forecasting import ForecastingConfig, perform_forecasting

    results_dir = Path(
        os.environ.get(
            "NV_TESSERACT_PANEL_RESULTS_DIR",
            forecasting_root.parent.parent / "results" / "japan_stocks_fx",
        )
    ).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    fx = download_chart(FX_SYMBOL, results_dir / "usdjpy_raw.json")
    fx_daily = fx[[TIMESTAMP_COLUMN, "adjusted_close"]].rename(
        columns={"adjusted_close": FX_COLUMN}
    )
    today_jst = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Tokyo")).date()
    minimum_rows = SEQ_LEN + FORECAST_HORIZON + (N_ORIGINS - 1) * ORIGIN_STEP
    per_origin: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    stock_metadata: list[dict[str, Any]] = []

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    for symbol, identity in STOCKS.items():
        safe_symbol = symbol.lower().replace(".", "_")
        stock = download_chart(symbol, results_dir / f"{safe_symbol}_raw.json")
        stock_daily = stock[[TIMESTAMP_COLUMN, "adjusted_close"]].rename(
            columns={"adjusted_close": TARGET_COLUMN}
        )
        combined = stock_daily.merge(fx_daily, on=TIMESTAMP_COLUMN, how="left").sort_values(
            TIMESTAMP_COLUMN
        )
        combined[FX_COLUMN] = combined[FX_COLUMN].ffill()
        combined = combined.dropna().reset_index(drop=True)
        combined = combined[combined[TIMESTAMP_COLUMN].dt.date < today_jst].reset_index(drop=True)
        combined.to_csv(results_dir / f"{safe_symbol}_aligned.csv", index=False)
        if len(combined) < minimum_rows:
            raise RuntimeError(f"{symbol}: need {minimum_rows} rows, got {len(combined)}")

        origins = [
            len(combined) - FORECAST_HORIZON - ORIGIN_STEP * offset
            for offset in range(N_ORIGINS - 1, -1, -1)
        ]
        symbol_records: list[dict[str, Any]] = []
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
        for origin_number, origin in enumerate(origins, start=1):
            train = combined.iloc[origin - SEQ_LEN : origin].copy()
            truth = combined.iloc[origin : origin + FORECAST_HORIZON].copy()
            actual = truth[TARGET_COLUMN].to_numpy(dtype=float)
            last_input = float(train[TARGET_COLUMN].iloc[-1])
            predictions = {
                "last_value": np.full(FORECAST_HORIZON, last_input, dtype=float)
            }
            for model_name, cross_channel, columns in (
                ("nv_univariate", False, [TIMESTAMP_COLUMN, TARGET_COLUMN]),
                (
                    "nv_stock_plus_usdjpy",
                    True,
                    [TIMESTAMP_COLUMN, TARGET_COLUMN, FX_COLUMN],
                ),
            ):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                forecast = perform_forecasting(
                    df=train[columns],
                    config=ForecastingConfig(**config_common, use_cross_channel=cross_channel),
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                predicted = forecast[f"{TARGET_COLUMN}_forecast"].to_numpy(dtype=float)
                if len(predicted) != FORECAST_HORIZON or not np.isfinite(predicted).all():
                    raise RuntimeError(f"Invalid {model_name} prediction: {symbol} origin {origin_number}")
                predictions[model_name] = predicted
                timings.append(
                    {
                        "symbol": symbol,
                        "origin_number": origin_number,
                        "origin_date": str(train[TIMESTAMP_COLUMN].iloc[-1].date()),
                        "model": model_name,
                        "elapsed_seconds": elapsed,
                    }
                )

            for model_name, predicted in predictions.items():
                record = {
                    "symbol": symbol,
                    "company": identity["company"],
                    "origin_number": origin_number,
                    "origin_date": str(train[TIMESTAMP_COLUMN].iloc[-1].date()),
                    "forecast_start": str(truth[TIMESTAMP_COLUMN].iloc[0].date()),
                    "forecast_end": str(truth[TIMESTAMP_COLUMN].iloc[-1].date()),
                    "model": model_name,
                    "last_input": last_input,
                    "actual": actual.tolist(),
                    "predicted": predicted.tolist(),
                    "metrics": {
                        str(horizon): metric_values(
                            actual[:horizon], predicted[:horizon], last_input
                        )
                        for horizon in HORIZONS
                    },
                }
                symbol_records.append(record)
                per_origin.append(record)
            print(
                f"{symbol} origin {origin_number:02d}/{N_ORIGINS}: "
                f"{train[TIMESTAMP_COLUMN].iloc[-1].date()} -> "
                f"{truth[TIMESTAMP_COLUMN].iloc[-1].date()}",
                flush=True,
            )

        returns = combined[TARGET_COLUMN].pct_change()
        fx_returns = combined[FX_COLUMN].pct_change()
        stock_metadata.append(
            {
                "symbol": symbol,
                **identity,
                "rows": len(combined),
                "data_start": str(combined[TIMESTAMP_COLUMN].iloc[0].date()),
                "data_end": str(combined[TIMESTAMP_COLUMN].iloc[-1].date()),
                "return_correlation_with_usdjpy": float(returns.corr(fx_returns)),
            }
        )

    aggregate_rows: list[dict[str, Any]] = []
    for symbol in STOCKS:
        stock_records = [row for row in per_origin if row["symbol"] == symbol]
        for row in aggregate_one_stock(stock_records):
            aggregate_rows.append({"symbol": symbol, **asdict(row)})
    panel_rows = aggregate_panel(per_origin)
    pd.DataFrame(aggregate_rows).to_csv(results_dir / "per_stock_aggregate.csv", index=False)
    pd.DataFrame(panel_rows).to_csv(results_dir / "panel_aggregate.csv", index=False)
    pd.DataFrame(timings).to_csv(results_dir / "timings.csv", index=False)
    (results_dir / "per_origin.json").write_text(
        json.dumps(per_origin, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "model evaluation only; not investment advice",
        "panel_selection": "fixed before inference: diverse large Japanese stocks",
        "stocks": stock_metadata,
        "fx_symbol": FX_SYMBOL,
        "data_source": "Yahoo Finance chart API",
        "price_field": "adjusted_close",
        "alignment": "stock trading dates; exact-date USD/JPY joined and past-only forward fill",
        "seq_len": SEQ_LEN,
        "forecast_horizon": FORECAST_HORIZON,
        "model_horizon": MODEL_HORIZON,
        "origins_per_stock": N_ORIGINS,
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
        "panel_aggregate": panel_rows,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(panel_rows).to_string(index=False), flush=True)
    print(f"raw_result={results_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
