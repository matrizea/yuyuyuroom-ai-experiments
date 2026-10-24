#!/usr/bin/env python3
"""Generate a publication figure and human-readable HTML dashboard from raw JSON."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--dashboard", type=Path, required=True)
    args = parser.parse_args()

    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    results = benchmark["results"]

    font_path = Path("/mnt/c/Windows/Fonts/meiryo.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()

    labels = [
        "CPU\nFP32",
        "GPU\nFP32",
        "GPU\nFP16",
        "GPU\nBF16",
    ]
    latencies = [item["median_ms"] for item in results]
    colors = ["#94a3b8", "#38bdf8", "#a78bfa", "#34d399"]
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
    fig.patch.set_facecolor("#07111f")
    ax.set_facecolor("#0b172a")
    bars = ax.bar(labels, latencies, color=colors, width=0.68)
    ax.set_yscale("log")
    ax.set_ylabel("行列積の中央値 [ms]（対数軸）", color="#dbeafe", fontsize=13)
    ax.set_title("PyTorch + WSL 2 + RTX 5070 Ti：4096² 行列積", color="white", fontsize=20, pad=18)
    ax.text(
        0.5,
        1.01,
        "配列生成・CPU↔GPU転送を除外 / warm-up後7回の中央値",
        transform=ax.transAxes,
        ha="center",
        color="#93c5fd",
        fontsize=11,
    )
    ax.tick_params(colors="#dbeafe", labelsize=12)
    ax.grid(axis="y", which="both", alpha=0.18, color="white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, value in zip(bars, latencies, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.13,
            f"{value:.3f} ms",
            ha="center",
            va="bottom",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
    fp32_speedup = results[0]["median_ms"] / results[1]["median_ms"]
    ax.text(
        0.02,
        0.94,
        f"同じFP32で GPUはCPUの約{fp32_speedup:.1f}倍",
        transform=ax.transAxes,
        color="#7dd3fc",
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.5,
        0.02,
        "注意：データ型が違う棒は精度も違う。小規模処理や転送込みでは同じ倍率にならない。",
        ha="center",
        color="#cbd5e1",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.95))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, facecolor=fig.get_facecolor())
    plt.close(fig)

    cards = "".join(
        f"""
        <div class="card">
          <div class="label">{html.escape(label.replace(chr(10), ' '))}</div>
          <div class="value">{item['median_ms']:.3f}<span> ms</span></div>
          <div class="meta">{item['estimated_tflops']:.3f} TFLOPS（推定）</div>
        </div>"""
        for label, item in zip(labels, results, strict=True)
    )
    dashboard = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PyTorch CUDA 動作確認</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#07111f;color:#e5eefc;font-family:'Yu Gothic UI','Meiryo',sans-serif}}
.wrap{{width:min(1180px,94vw);margin:32px auto}} .eyebrow{{color:#38bdf8;font-weight:700;letter-spacing:.12em}}
h1{{font-size:clamp(28px,4vw,48px);margin:10px 0}} .lead{{font-size:19px;color:#bfdbfe}}
.status{{display:inline-flex;gap:10px;align-items:center;background:#0f2c25;border:1px solid #34d399;padding:10px 16px;border-radius:999px;color:#6ee7b7;font-weight:700}}
.dot{{width:10px;height:10px;border-radius:50%;background:#34d399;box-shadow:0 0 14px #34d399}}
.spec{{margin:24px 0;padding:22px;background:#0b172a;border:1px solid #1e3a5f;border-radius:18px;display:grid;grid-template-columns:repeat(4,1fr);gap:16px}}
.spec b{{display:block;color:#93c5fd;font-size:13px;margin-bottom:6px}} .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.card{{background:linear-gradient(145deg,#0d1d35,#102746);border:1px solid #21466f;border-radius:18px;padding:22px}}
.label{{color:#93c5fd;font-weight:700}} .value{{font-size:32px;font-weight:800;margin:10px 0}} .value span{{font-size:16px;color:#9fb3cc}}
.meta{{color:#cbd5e1}} .explain{{margin-top:24px;background:#10233d;border-left:5px solid #38bdf8;padding:18px 22px;border-radius:10px;line-height:1.8}}
code{{color:#a7f3d0}} @media(max-width:800px){{.spec,.grid{{grid-template-columns:1fr 1fr}}}} 
</style></head><body><main class="wrap">
<div class="eyebrow">AI LAB / ACTUAL MACHINE RESULT</div>
<h1>PyTorchからRTX 5070 Tiを使えた</h1>
<p class="lead">WSL 2上のPythonからCUDA行列積を実行し、環境情報と7回計測をJSONへ保存。</p>
<div class="status"><span class="dot"></span>torch.cuda.is_available() = True</div>
<section class="spec">
  <div><b>GPU</b>{html.escape(environment['cuda']['gpu']['name'])}</div>
  <div><b>PyTorch</b>{html.escape(environment['packages']['torch'])}</div>
  <div><b>CUDA runtime</b>{html.escape(environment['cuda']['runtime'])}</div>
  <div><b>GPU memory</b>{environment['cuda']['gpu']['total_memory_gib']:.2f} GiB</div>
</section>
<section class="grid">{cards}</section>
<section class="explain"><b>この数値の意味</b><br>
同じFP32の4096²行列積ではGPUがCPUの約{fp32_speedup:.1f}倍。ただし、配列生成とCPU↔GPU転送は計測外です。
<code>CUDAが使える</code>ことと、あらゆるPython処理が同じ倍率で速くなることは別です。</section>
</main></body></html>"""
    args.dashboard.parent.mkdir(parents=True, exist_ok=True)
    args.dashboard.write_text(dashboard, encoding="utf-8")


if __name__ == "__main__":
    main()
