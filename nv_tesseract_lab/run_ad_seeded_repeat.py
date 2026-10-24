"""Check whether explicit PyTorch seeding stabilizes AD Diffusion output."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    ad_root = Path.cwd()
    sys.path.insert(0, str(ad_root))
    sys.path.insert(0, str(ad_root / "examples"))
    from quick_example import create_synthetic_dataset  # pylint: disable=import-outside-toplevel
    from sdk.anomaly_analysis import (  # pylint: disable=import-outside-toplevel
        perform_anomaly_analysis_with_diffusion,
    )

    results_root = ad_root.parent.parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    full_df = create_synthetic_dataset(n_samples=500, n_features=3, anomaly_rate=0.08)
    analysis_df = full_df[["sensor_1", "sensor_2", "sensor_3"]].copy()

    records = []
    outputs: list[pd.DataFrame] = []
    for run_number in (1, 2):
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = perform_anomaly_analysis_with_diffusion(
            df=analysis_df,
            threshold_strategy="scs",
            model_path="final_model.pth",
            model_config_path="curriculum_medium.yaml",
            nsample=15,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        result.to_csv(results_root / f"ad_seeded_result_{run_number}.csv", index=False)
        outputs.append(result)
        records.append(
            {
                "run": run_number,
                "elapsed_seconds": elapsed,
                "detected_anomalies": int(result["Anomaly"].sum()),
                "mae_mean": float(result["MAE"].mean()),
                "mae_max": float(result["MAE"].max()),
            }
        )

    comparison = {
        "seed": 42,
        "runs": records,
        "anomaly_columns_identical": bool(outputs[0]["Anomaly"].equals(outputs[1]["Anomaly"])),
        "mae_max_absolute_difference": float(
            np.max(np.abs(outputs[0]["MAE"].to_numpy() - outputs[1]["MAE"].to_numpy()))
        ),
    }
    (results_root / "ad_seeded_repeat.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
