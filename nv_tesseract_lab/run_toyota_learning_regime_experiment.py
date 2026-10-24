"""Fine-tune NV-Tesseract on Toyota with two history sizes, then score held-out 2025-2026.

The published model's 512-step input length stays fixed.  "History size" here
means the number of pre-evaluation rows from which sliding training examples
are created.  All training and validation targets end before the first scored
forecast, so later policy regimes cannot leak into model fitting.

Run from NV-Tesseract/forecasting with its existing virtual environment.
This is model evaluation, not investment advice.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SEQ_LEN = 512
TRAIN_HORIZON = 72
SCORE_HORIZON = 20
HORIZONS = (1, 5, 20)
TRAIN_STRIDE = 5
VALIDATION_TARGET_ROWS = 160
SHORT_HISTORY_ROWS = 800
EPOCHS = 5
SEEDS = (42, 43, 44)
SYMBOL = "7203.T"
TARGET = "adjusted_close_jpy"
TIMESTAMP = "timestamp"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def regime(origin_date: str) -> str:
    """Label by information available on the forecast origin date."""
    date = pd.Timestamp(origin_date)
    if date < pd.Timestamp("2025-04-03"):
        return "before_2025_04_03_auto_tariff"
    if date < pd.Timestamp("2025-09-05"):
        return "2025_04_03_to_2025_09_04_transition"
    return "after_2025_09_04_japan_agreement_order"


def metrics(actual: np.ndarray, predicted: np.ndarray, last_input: float) -> dict[str, float]:
    error = predicted - actual
    denominator = np.abs(actual) + np.abs(predicted)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "smape_percent": float(
            np.mean(np.where(denominator == 0, 0.0, 200.0 * np.abs(error) / denominator))
        ),
        "endpoint_direction_correct": float(
            np.sign(predicted[-1] - last_input) == np.sign(actual[-1] - last_input)
        ),
    }


def aggregate(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.json_normalize(records)
    models = list(dict.fromkeys(frame["model"].tolist()))
    overall: list[dict[str, Any]] = []
    by_regime: list[dict[str, Any]] = []
    for grouping, destination in ((None, overall), ("regime", by_regime)):
        groups = [("all", frame)] if grouping is None else list(frame.groupby(grouping))
        for group_name, group in groups:
            for model in models:
                model_frame = group[group["model"] == model]
                if model_frame.empty:
                    continue
                for horizon in HORIZONS:
                    smape_col = f"metrics.{horizon}.smape_percent"
                    mae_col = f"metrics.{horizon}.mae"
                    direction_col = f"metrics.{horizon}.endpoint_direction_correct"
                    baseline = group[group["model"] == "last_value"][["origin_number", smape_col]].rename(
                        columns={smape_col: "baseline_smape"}
                    )
                    paired = model_frame.merge(baseline, on="origin_number")
                    delta = paired[smape_col] - paired["baseline_smape"]
                    row = {
                        "regime": group_name,
                        "model": model,
                        "horizon": horizon,
                        "n_origins": int(len(model_frame)),
                        "mae": float(model_frame[mae_col].mean()),
                        "smape_percent": float(model_frame[smape_col].mean()),
                        "direction_accuracy": (
                            None if model == "last_value" else float(model_frame[direction_col].mean())
                        ),
                        "mean_smape_delta_vs_last": (
                            0.0 if model == "last_value" else float(delta.mean())
                        ),
                        "win_rate_vs_last": (
                            None if model == "last_value" else float(np.mean(delta < 0))
                        ),
                    }
                    if model == "last_value":
                        row["bootstrap95_smape_delta_low"] = 0.0
                        row["bootstrap95_smape_delta_high"] = 0.0
                    else:
                        rng = np.random.default_rng(10_000 + horizon + len(destination) * 17)
                        sampled = rng.choice(delta.to_numpy(), size=(10_000, len(delta)), replace=True).mean(axis=1)
                        low, high = np.quantile(sampled, (0.025, 0.975))
                        row["bootstrap95_smape_delta_low"] = float(low)
                        row["bootstrap95_smape_delta_high"] = float(high)
                    destination.append(row)
    return pd.DataFrame(overall), pd.DataFrame(by_regime)


def main() -> None:
    forecasting_root = Path.cwd()
    if not (forecasting_root / "sdk").is_dir():
        raise SystemExit("Run from NV-Tesseract/forecasting")
    lab_root = Path(__file__).resolve().parent
    panel_dir = lab_root / "results" / "japan_stocks_fx"
    results_dir = lab_root / "results" / "toyota_learning_regime"
    training_dir = results_dir / "training"
    results_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    aligned = pd.read_csv(panel_dir / "7203_t_aligned.csv", parse_dates=[TIMESTAMP])
    existing = json.loads((panel_dir / "per_origin.json").read_text(encoding="utf-8"))
    existing = [row for row in existing if row["symbol"] == SYMBOL]
    baseline_rows = [row for row in existing if row["model"] in {"last_value", "nv_univariate", "nv_stock_plus_usdjpy"}]
    origins = sorted(
        {row["origin_number"]: row for row in baseline_rows if row["model"] == "last_value"}.values(),
        key=lambda row: row["origin_number"],
    )
    first_origin_date = pd.Timestamp(origins[0]["origin_date"])
    first_origin_index = int(aligned.index[aligned[TIMESTAMP] == first_origin_date][0]) + 1
    train_end = first_origin_index - VALIDATION_TARGET_ROWS
    if train_end < SHORT_HISTORY_ROWS:
        raise RuntimeError("Not enough pre-evaluation rows for the fixed short-history condition")

    all_train = aligned.iloc[:train_end][[TIMESTAMP, TARGET]].copy()
    short_train = all_train.iloc[-SHORT_HISTORY_ROWS:].copy()
    validation = aligned.iloc[train_end - SEQ_LEN : first_origin_index][[TIMESTAMP, TARGET]].copy()
    datasets = {"short_800": short_train, "all_pre_eval": all_train}
    for name, frame in datasets.items():
        frame.to_csv(training_dir / f"{name}_train.csv", index=False)
    validation.to_csv(training_dir / "validation.csv", index=False)

    training_runs: list[dict[str, Any]] = []
    finetune_script = forecasting_root / "examples" / "finetune_example.py"
    for history_name in datasets:
        for seed in SEEDS:
            out_dir = training_dir / f"{history_name}_seed{seed}"
            command = [
                sys.executable,
                str(finetune_script),
                "--train-csv",
                str(training_dir / f"{history_name}_train.csv"),
                "--val-csv",
                str(training_dir / "validation.csv"),
                "--timestamp-col",
                TIMESTAMP,
                "--target-cols",
                TARGET,
                "--seq-len",
                str(SEQ_LEN),
                "--forecast-horizon",
                str(TRAIN_HORIZON),
                "--stride",
                str(TRAIN_STRIDE),
                "--epochs",
                str(EPOCHS),
                "--batch-size",
                "8",
                "--lr",
                "1e-4",
                "--seed",
                str(seed),
                "--output-dir",
                str(out_dir),
                "--num-gpus",
                "1",
                "--local-files-only",
            ]
            log_path = training_dir / f"{history_name}_seed{seed}.log"
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=forecasting_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            elapsed = time.perf_counter() - started
            log_path.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"Fine-tune failed: {history_name} seed {seed}; see {log_path}")
            metadata = json.loads((out_dir / "finetune_metadata.json").read_text())
            epoch_metrics = json.loads((out_dir / "epoch_metrics.json").read_text())
            training_runs.append(
                {
                    "history": history_name,
                    "seed": seed,
                    "train_rows": int(len(datasets[history_name])),
                    "validation_rows_including_context": int(len(validation)),
                    "elapsed_seconds": elapsed,
                    "best_epoch": metadata["best_epoch"],
                    "metrics_at_saved_best": metadata["metrics"],
                    "all_epoch_metrics": epoch_metrics,
                    "checkpoint_sha256": sha256(out_dir / "best_model.pt"),
                }
            )
            print(f"trained {history_name} seed={seed} in {elapsed:.1f}s", flush=True)

    sys.path.insert(0, str(forecasting_root))
    from sdk.forecasting import ForecastingConfig, perform_forecasting

    records: list[dict[str, Any]] = []
    for row in baseline_rows:
        kept = dict(row)
        kept["regime"] = regime(row["origin_date"])
        records.append(kept)

    for history_name in datasets:
        seed_predictions: dict[int, list[np.ndarray]] = {row["origin_number"]: [] for row in origins}
        for seed in SEEDS:
            out_dir = training_dir / f"{history_name}_seed{seed}"
            config = ForecastingConfig(
                standardizer_pkl=str(out_dir / "standardizer.pkl"),
                ckpt=str(out_dir / "best_model.pt"),
                seq_len=SEQ_LEN,
                forecast_horizon=SCORE_HORIZON,
                model_horizon=TRAIN_HORIZON,
                timestamp_column=TIMESTAMP,
                target_column=TARGET,
                seed=seed,
                batch_size=1,
                num_workers=0,
                use_cross_channel=False,
            )
            for origin in origins:
                origin_idx = int(aligned.index[aligned[TIMESTAMP] == pd.Timestamp(origin["origin_date"])][0]) + 1
                history = aligned.iloc[origin_idx - SEQ_LEN : origin_idx][[TIMESTAMP, TARGET]]
                forecast = perform_forecasting(df=history, config=config)
                prediction = forecast[f"{TARGET}_forecast"].to_numpy(dtype=float)
                seed_predictions[origin["origin_number"]].append(prediction)
        for origin in origins:
            origin_idx = int(aligned.index[aligned[TIMESTAMP] == pd.Timestamp(origin["origin_date"])][0]) + 1
            truth = aligned.iloc[origin_idx : origin_idx + SCORE_HORIZON]
            actual = truth[TARGET].to_numpy(dtype=float)
            last_input = float(aligned.iloc[origin_idx - 1][TARGET])
            predictions = seed_predictions[origin["origin_number"]]
            for seed, predicted in zip(SEEDS, predictions, strict=True):
                model_name = f"finetuned_{history_name}_seed{seed}"
                records.append(
                    {
                        "symbol": SYMBOL,
                        "company": "Toyota Motor",
                        "origin_number": origin["origin_number"],
                        "origin_date": origin["origin_date"],
                        "forecast_start": str(truth[TIMESTAMP].iloc[0].date()),
                        "forecast_end": str(truth[TIMESTAMP].iloc[-1].date()),
                        "regime": regime(origin["origin_date"]),
                        "model": model_name,
                        "last_input": last_input,
                        "actual": actual.tolist(),
                        "predicted": predicted.tolist(),
                        "metrics": {
                            str(h): metrics(actual[:h], predicted[:h], last_input) for h in HORIZONS
                        },
                    }
                )
            ensemble = np.mean(np.stack(predictions), axis=0)
            records.append(
                {
                    "symbol": SYMBOL,
                    "company": "Toyota Motor",
                    "origin_number": origin["origin_number"],
                    "origin_date": origin["origin_date"],
                    "forecast_start": str(truth[TIMESTAMP].iloc[0].date()),
                    "forecast_end": str(truth[TIMESTAMP].iloc[-1].date()),
                    "regime": regime(origin["origin_date"]),
                    "model": f"finetuned_{history_name}_ensemble",
                    "last_input": last_input,
                    "actual": actual.tolist(),
                    "predicted": ensemble.tolist(),
                    "metrics": {str(h): metrics(actual[:h], ensemble[:h], last_input) for h in HORIZONS},
                }
            )

    overall, regimes = aggregate(records)
    (results_dir / "per_origin.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    overall.to_csv(results_dir / "aggregate_overall.csv", index=False)
    regimes.to_csv(results_dir / "aggregate_by_regime.csv", index=False)
    pd.DataFrame(training_runs).to_json(
        results_dir / "training_runs.json", orient="records", indent=2
    )
    summary = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "model evaluation only; not investment advice",
        "symbol": SYMBOL,
        "fixed_input_length": SEQ_LEN,
        "training_forecast_horizon": TRAIN_HORIZON,
        "scored_horizon": SCORE_HORIZON,
        "training_history_conditions": {name: len(frame) for name, frame in datasets.items()},
        "validation_target_rows": VALIDATION_TARGET_ROWS,
        "validation_ends": str(validation[TIMESTAMP].iloc[-1].date()),
        "first_scored_forecast_starts": origins[0]["forecast_start"],
        "epochs": EPOCHS,
        "stride": TRAIN_STRIDE,
        "seeds": list(SEEDS),
        "regime_rule": "origin-date labels; descriptive time slices, not a causal estimator",
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=forecasting_root, text=True
        ).strip(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_runs": training_runs,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overall[overall["model"].isin([
        "last_value",
        "nv_univariate",
        "nv_stock_plus_usdjpy",
        "finetuned_short_800_ensemble",
        "finetuned_all_pre_eval_ensemble",
    ])].to_string(index=False), flush=True)
    print(f"results={results_dir}", flush=True)


if __name__ == "__main__":
    main()
