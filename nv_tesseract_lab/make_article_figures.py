"""Create article figures strictly from saved NV-Tesseract experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    lab_root = Path(__file__).resolve().parent
    repo_root = lab_root / "NV-Tesseract"
    results_root = lab_root / "results"
    figures_root = lab_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(
        repo_root / "forecasting" / "sdk" / "tests" / "datasets" / "ETTh_single_feature.csv"
    ).tail(512)
    forecast = pd.read_csv(results_root / "forecast_run_1.csv")
    source["timestamp"] = pd.to_datetime(source["timestamp"])
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.plot(source["timestamp"].tail(168), source["LULL"].tail(168), label="Observed LULL", linewidth=1.8)
    ax.plot(
        forecast["timestamp"],
        forecast["LULL_forecast"],
        label="NV-Tesseract forecast (100 steps)",
        linewidth=2.0,
        color="#e45756",
    )
    ax.axvline(source["timestamp"].iloc[-1], color="#555", linestyle="--", linewidth=1)
    ax.set_title("NV-Tesseract forecasting output on the bundled ETTh sample")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("LULL")
    ax.grid(alpha=0.25)
    ax.legend()
    forecast_path = figures_root / "nv-tesseract-etth-forecast.png"
    fig.savefig(forecast_path, dpi=180)
    plt.close(fig)

    ad_result = pd.read_csv(results_root / "ad_seeded_result_1.csv")
    labels = pd.read_csv(results_root / "ad_labels_run_1.csv")
    x = range(len(ad_result))
    detected = ad_result["Anomaly"].astype(bool)
    actual = labels["is_anomaly"].astype(bool)

    fig, (signal_ax, score_ax) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True
    )
    signal_ax.plot(x, ad_result["sensor_1"], color="#4c78a8", linewidth=1, label="sensor_1")
    signal_ax.scatter(
        ad_result.index[actual],
        ad_result.loc[actual, "sensor_1"],
        s=12,
        facecolors="none",
        edgecolors="#f2cf5b",
        label="Ground truth anomaly",
    )
    signal_ax.scatter(
        ad_result.index[detected],
        ad_result.loc[detected, "sensor_1"],
        s=20,
        color="#e45756",
        marker="x",
        label="Detected anomaly",
    )
    signal_ax.set_title("AD Diffusion on NVIDIA's synthetic sample (PyTorch seed = 42)")
    signal_ax.set_ylabel("sensor_1")
    signal_ax.grid(alpha=0.2)
    signal_ax.legend(ncol=3, fontsize=9)

    score_ax.plot(x, ad_result["MAE"], color="#72b7b2", linewidth=1.2, label="Reconstruction MAE")
    score_ax.scatter(
        ad_result.index[detected], ad_result.loc[detected, "MAE"], color="#e45756", s=18, label="Detected"
    )
    score_ax.set_xlabel("Sample index")
    score_ax.set_ylabel("MAE")
    score_ax.grid(alpha=0.2)
    score_ax.legend(fontsize=9)
    ad_path = figures_root / "nv-tesseract-ad-diffusion-result.png"
    fig.savefig(ad_path, dpi=180)
    plt.close(fig)

    forecasting_meta = json.loads((results_root / "forecast_probe.json").read_text(encoding="utf-8"))
    seeded_meta = json.loads((results_root / "ad_seeded_repeat.json").read_text(encoding="utf-8"))
    summary = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>NV-Tesseract 実行結果</title>
<style>
body{{font-family:system-ui,sans-serif;background:#111827;color:#e5e7eb;margin:0;padding:32px}}
main{{max-width:1040px;margin:auto}} h1{{font-size:30px}} .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}
.card{{background:#1f2937;border:1px solid #374151;border-radius:14px;padding:20px}} .value{{font-size:31px;font-weight:700;color:#60a5fa}}
code{{color:#86efac}} table{{border-collapse:collapse;width:100%;margin-top:22px}} th,td{{padding:10px;border-bottom:1px solid #374151;text-align:left}}
</style></head><body><main><h1>NV-Tesseract — WSL2 + RTX 5070 Ti 実行結果</h1>
<p><code>CUDA available: true / {forecasting_meta['torch']} / {forecasting_meta['gpu_name']}</code></p>
<div class="grid"><div class="card"><div>予測出力</div><div class="value">100 points</div></div>
<div class="card"><div>初回（読込込み）</div><div class="value">{forecasting_meta['runs'][0]['elapsed_seconds']:.3f} s</div></div>
<div class="card"><div>同一プロセス再実行</div><div class="value">{forecasting_meta['runs'][1]['elapsed_seconds']:.3f} s</div></div></div>
<table><tr><th>確認項目</th><th>結果</th></tr>
<tr><td>予測GPUメモリ（PyTorch peak allocated）</td><td>{forecasting_meta['runs'][0]['torch_peak_allocated_mib']:.1f} MiB</td></tr>
<tr><td>AD Diffusion seed固定 2回</td><td>{seeded_meta['runs'][0]['detected_anomalies']}件 / {seeded_meta['runs'][1]['detected_anomalies']}件</td></tr>
<tr><td>AD判定列の一致</td><td>{str(seeded_meta['anomaly_columns_identical']).lower()}</td></tr>
<tr><td>MAE最大絶対差</td><td>{seeded_meta['mae_max_absolute_difference']}</td></tr></table>
</main></body></html>"""
    (lab_root / "result_summary.html").write_text(summary, encoding="utf-8")
    print(forecast_path)
    print(ad_path)


if __name__ == "__main__":
    main()
