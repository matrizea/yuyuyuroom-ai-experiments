"""Plot the paired FP16/FP8 latency, file size and GT differences from evidence."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def delta_panel(ax, rows, title):
    ax.axvline(0, color="#64748b", lw=1, linestyle="--")
    for index, (label, mean, interval) in enumerate(rows):
        low, high = interval
        ax.plot([low, high], [index, index], color="#b45309", lw=3)
        ax.scatter([mean], [index], color="#b45309", s=42, zorder=3)
        ax.annotate(f"{mean:+.6f}", (mean, index), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [row[0] for row in rows])
    ax.set_ylim(-.6, len(rows)-.4)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("FP8 minus FP16; 95% scene/sequence bootstrap CI")
    ax.grid(axis="x", alpha=.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=ROOT/"evidence")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"article_figures")
    args = parser.parse_args()
    benchmark = read(args.evidence_dir/"fp8_benchmark.json")
    nyu = read(args.evidence_dir/"results_dense_fp8_nyu_gpu/summary.json")
    sintel = read(args.evidence_dir/"results_dense_fp8_sintel_gpu/summary.json")
    if benchmark["status"] != "complete":
        raise ValueError("The paired benchmark is incomplete")
    for result in (nyu, sintel):
        if result["reference"] != "fp16" or result["completed_images"] != result["expected_images"]:
            raise ValueError("Use complete GT comparisons with FP16 reference")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "figure.facecolor":"white", "axes.axisbelow":True})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.3))
    names, labels = ["fp16", "fp8"], ["FP16 (opset 21)", "FP8 W8A8"]
    positions = np.arange(2)
    modes = [("host_input_output", "Host input/output", -.17, "#2563a8"),
             ("device_resident_input_output", "GPU resident input/output", .17, "#7aa3b5")]
    maximum = 0.
    for mode, label, offset, color in modes:
        values = [benchmark["models"][name][mode]["median_ms"] for name in names]
        maximum = max(maximum, *values)
        axes[0, 0].bar(positions+offset, values, width=.31, color=color, label=label)
        for position, value in zip(positions+offset, values):
            axes[0, 0].text(position, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0, 0].set_xticks(positions, labels)
    axes[0, 0].set_ylim(0, maximum*1.22)
    axes[0, 0].set_ylabel("Median warm call (ms)")
    axes[0, 0].set_title("Latency: same runtime and input")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(axis="y", alpha=.2)
    sizes = [benchmark["models"][name]["source_graph"]["file_size_mib"] for name in names]
    axes[0, 1].bar(positions, sizes, color=["#2563a8", "#b45309"], width=.5)
    for position, value in zip(positions, sizes):
        axes[0, 1].text(position, value, f"{value:.2f}", ha="center", va="bottom")
    axes[0, 1].set_xticks(positions, labels)
    axes[0, 1].set_ylim(0, max(sizes)*1.2)
    axes[0, 1].set_ylabel("ONNX file (MiB)")
    axes[0, 1].set_title("File size; excludes runtime workspace")
    axes[0, 1].grid(axis="y", alpha=.2)
    rows = []
    for kind, label in (("measured", "Measured GT"), ("inpainted", "Inpainted GT")):
        comparison = nyu["ground_truth_variants"][kind]["models"]["fp8"]["versus_reference"]
        rows.append((label, comparison["mean_delta_abs_rel"], comparison["scene_cluster_ci95_delta_abs_rel"]))
    delta_panel(axes[1, 0], rows, "NYUv2 128 images / 95 scenes: AbsRel delta")
    comparison = sintel["ground_truth_variants"]["rendered_dense"]["models"]["fp8"]["versus_reference_inverse_nrmse"]
    delta_panel(axes[1, 1], [("Rendered GT", comparison["mean_delta_inverse_nrmse"],
                            comparison["scene_cluster_ci95_delta_inverse_nrmse"])],
                "Sintel 64 images / 23 sequences: inverse NRMSE delta")
    fig.suptitle("Depth Anything V2 Small / RTX 5070 Ti / ONNX Runtime CUDA 1.31.0", fontsize=13)
    fig.text(.5, .028, "TF32 off; 518 x 518, batch 1; 20 warmups + 100 calls per block; 6 blocks per model and mode.",
             ha="center", fontsize=8)
    fig.text(.5, .009, "FP8: 48 E4M3 fixed-weight GEMMs; other floating operations FP16. GT metrics use per-image oracle alignment.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .052, 1, .965), h_pad=2.4, w_pad=2.)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/"fp8-vs-fp16-results.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
