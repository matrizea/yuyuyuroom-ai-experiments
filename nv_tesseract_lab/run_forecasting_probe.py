"""Reproducible NV-Tesseract forecasting probe for the site article.

Run from the repository's ``forecasting`` directory with its uv environment.
The script uses NVIDIA's bundled ETTh single-feature sample, performs the same
512-point to 100-point forecast twice, and preserves raw outputs as JSON/CSV.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SEED = 42
SEQ_LEN = 512
FORECAST_HORIZON = 100
TARGET_COLUMN = "LULL"
TIMESTAMP_COLUMN = "timestamp"


def gpu_snapshot() -> dict[str, str] | None:
    """Return one raw nvidia-smi snapshot without failing the experiment."""
    command = [
        "/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    fields = [part.strip() for part in output.split(",")]
    if len(fields) != 6:
        return {"raw": output}
    return dict(
        zip(
            [
                "name",
                "driver_version",
                "memory_total_mib",
                "memory_used_mib",
                "utilization_gpu_percent",
                "power_draw_w",
            ],
            fields,
            strict=True,
        )
    )


def main() -> None:
    forecasting_root = Path.cwd()
    sdk_root = forecasting_root / "sdk"
    if not sdk_root.exists():
        raise SystemExit("Run this script from NV-Tesseract/forecasting")
    sys.path.insert(0, str(forecasting_root))

    from sdk.forecasting import (  # pylint: disable=import-outside-toplevel
        DEVICE,
        ForecastingConfig,
        perform_forecasting,
    )

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    results_dir = Path(
        os.environ.get(
            "NV_TESSERACT_RESULTS_DIR",
            forecasting_root.parent.parent / "results",
        )
    ).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = sdk_root / "tests" / "datasets" / "ETTh_single_feature.csv"
    source_df = pd.read_csv(csv_path)
    input_df = source_df.tail(SEQ_LEN).copy()

    config = ForecastingConfig(
        seq_len=SEQ_LEN,
        forecast_horizon=FORECAST_HORIZON,
        # The published base checkpoint has a native 72-step head.  Keep that
        # native size even for a 100-step request so the SDK can extend the
        # forecast autoregressively, as the official quick example does.
        model_horizon=72,
        timestamp_column=TIMESTAMP_COLUMN,
        target_column=TARGET_COLUMN,
        seed=SEED,
        batch_size=1,
        num_workers=0,
        use_cross_channel=False,
        save_preds=None,
    )

    metadata: dict[str, object] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=forecasting_root, text=True
        ).strip(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_selected_by_sdk": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_compute_capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
        "input_csv": str(csv_path),
        "input_rows_available": len(source_df),
        "input_rows_used": len(input_df),
        "input_start": str(input_df[TIMESTAMP_COLUMN].iloc[0]),
        "input_end": str(input_df[TIMESTAMP_COLUMN].iloc[-1]),
        "target_column": TARGET_COLUMN,
        "seq_len": SEQ_LEN,
        "forecast_horizon": FORECAST_HORIZON,
        "model_horizon": 72,
        "seed": SEED,
        "batch_size": 1,
        "use_cross_channel": False,
        "gpu_before": gpu_snapshot(),
        "runs": [],
    }

    for run_index in range(1, 3):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        forecast_df = perform_forecasting(df=input_df, config=config)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        output_path = results_dir / f"forecast_run_{run_index}.csv"
        forecast_df.to_csv(output_path, index=False)
        forecast_column = f"{TARGET_COLUMN}_forecast"
        run_record = {
            "run": run_index,
            "elapsed_seconds": elapsed,
            "output_rows": len(forecast_df),
            "output_columns": list(forecast_df.columns),
            "forecast_first_10": forecast_df[forecast_column].head(10).tolist(),
            "forecast_min": float(forecast_df[forecast_column].min()),
            "forecast_max": float(forecast_df[forecast_column].max()),
            "forecast_mean": float(forecast_df[forecast_column].mean()),
            "torch_peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
            ),
            "torch_peak_reserved_mib": (
                torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_available() else None
            ),
            "gpu_after_run": gpu_snapshot(),
            "output_csv": str(output_path),
        }
        metadata["runs"].append(run_record)
        print(json.dumps(run_record, ensure_ascii=False, indent=2), flush=True)

    metadata["gpu_after"] = gpu_snapshot()
    json_path = results_dir / "forecast_probe.json"
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"raw_result={json_path}", flush=True)


if __name__ == "__main__":
    main()
