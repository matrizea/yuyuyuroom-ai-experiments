"""Recompute the eight-model GPU dense tables from public per-image CSV evidence.

Inputs are read only. This script does not import an inference runtime, download
data, or access original RGB/GT/prediction arrays. Use a new --output-dir.

Example from the distributed experiment directory:
  python scripts/summarize_all_models_dense.py --evidence-dir evidence --output-dir results/all-models

For original local measurements, override the four input directories:
  --nyu-seven results_dense_nyu_gpu --nyu-fp8 results_fp8/nyu
  --sintel-seven results_dense_sintel_full_gpu --sintel-fp8 results_fp8/sintel
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import runtime_init
import numpy as np
from dense_utils import bootstrap_mean_delta, bootstrap_scene_mean_delta

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["fp32", "fp16", "int8_dynamic", "int4_rtn", "int4_fast",
          "int8_w8a16", "int4_w4a16", "fp8"]
METRICS = ["abs_rel", "rmse", "delta1", "inverse_nrmse",
           "clipped_inverse_depth_rate", "nonpositive_aligned_inverse_rate"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_source(directory, label):
    directory = directory.resolve()
    summary_path, csv_path = directory / "summary.json", directory / "per_image.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(row["id"], row["model"], row["gt_kind"]) for row in rows}
    require(len(keys) == len(rows), label + ": duplicate per-image rows")
    require(summary["config"]["provider"] == "CUDA", label + ": expected CUDA source, not CPU-only evidence")
    require(summary["completed_images"] == summary["expected_images"], label + ": incomplete image subset")
    require(summary["config"]["threads"] == 8 and summary["config"]["inter_op_threads"] == 1,
            label + ": mismatching thread settings")
    require(summary["config"]["cuda_use_tf32"] is False, label + ": TF32 must be disabled")
    require(summary["config"].get("limit") is None, label + ": must evaluate the full selected manifest")
    for row in rows:
        require(int(row["valid_pixels"]) > 0, label + ": empty GT mask")
        for metric in METRICS:
            require(row[metric] != "" and math.isfinite(float(row[metric])),
                    label + ": undefined/nonfinite " + metric)
    return {"summary": summary, "rows": rows, "directory": directory,
            "input_hashes": {"summary.json": sha256(summary_path), "per_image.csv": sha256(csv_path)}}


def selected(source, kind, name):
    return [row for row in source["rows"] if row["gt_kind"] == kind and row["model"] == name]


def public_model_spec(summary, name):
    spec = summary["config"]["models"][name]
    # Do not copy local absolute paths or complete private config objects.
    model_file = re.split(r"[\\/]", spec["path"])[-1]
    return {"file": model_file, "sha256": spec["sha256"], "file_size_mib": spec["file_size_mib"]}


def compare_control(original, fp8, kinds, count):
    comparisons = 0
    maximum_delta = 0.0
    for kind in kinds:
        old = selected(original, kind, "fp16")
        new = selected(fp8, kind, "fp16")
        require(len(old) == len(new) == count, "FP16 control image count mismatch")
        require([row["id"] for row in old] == [row["id"] for row in new], "FP16 control image order mismatch")
        for a, b in zip(old, new):
            require(a["scene"] == b["scene"] and a["valid_pixels"] == b["valid_pixels"],
                    "FP16 control GT metadata mismatch")
            for metric in METRICS:
                delta = abs(float(a[metric]) - float(b[metric]))
                maximum_delta = max(maximum_delta, delta)
                require(delta == 0, "FP16 opset17 vs21 control metric difference: " + metric)
                comparisons += 1
    return {"per_image_metric_comparisons": comparisons, "maximum_absolute_metric_difference": maximum_delta,
            "all_selected_metrics_exactly_equal": True,
            "note": "CSV-level check of opset17 and opset21 FP16 controls. Native-map bitwise equality is separately documented in the FP16 control evidence; this CSV-only script does not read arrays."}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "evidence")
    parser.add_argument("--nyu-seven", type=Path)
    parser.add_argument("--nyu-fp8", type=Path)
    parser.add_argument("--sintel-seven", type=Path)
    parser.add_argument("--sintel-fp8", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory; existing output is never overwritten")
    evidence = args.evidence_dir.resolve()
    paths = {
        "nyu_seven": args.nyu_seven or evidence / "results_dense_nyu_gpu",
        "nyu_fp8": args.nyu_fp8 or evidence / "results_dense_fp8_nyu_gpu",
        "sintel_seven": args.sintel_seven or evidence / "results_dense_sintel_full_gpu",
        "sintel_fp8": args.sintel_fp8 or evidence / "results_dense_fp8_sintel_gpu",
    }
    sources = {label: read_source(directory, label) for label, directory in paths.items()}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "model_order": MODELS,
              "reference": "fp32", "aggregation": "Arithmetic mean of per-image metrics, not pooled-pixel accuracy",
              "interpretation": "Oracle relative-depth evaluation: each image's GT fits nonnegative inverse-depth scale and shift. This is not uncalibrated absolute metric-distance accuracy.",
              "uncertainty": "Paired image and scene/sequence cluster bootstrap, 10000 draws, seed42; scene resampling retains image-weighted mean difference",
              "provider_note": "All rows originate from CUDA-provider sessions. Dynamic INT8 is CUDA+CPU hybrid: original profiles place DynamicQuantizeLinear and MatMulInteger on CPU. These rows are not CPU-only scores.",
              "source_inputs": {}, "datasets": {}}
    for label, source in sources.items():
        try:
            displayed = source["directory"].relative_to(evidence).as_posix()
        except ValueError:
            displayed = source["directory"].name
        result["source_inputs"][label] = {"directory": displayed, "file_sha256": source["input_hashes"],
            "manifest_sha256": source["summary"]["config"]["manifest_sha256"],
            "provider": "CUDA", "threads": 8, "inter_op_threads": 1, "cuda_use_tf32": False}
    output_rows = []
    for dataset, count, expected_kinds in (("nyu", 128, {"inpainted", "measured"}),
                                            ("sintel", 64, {"rendered_dense"})):
        original, fp8 = sources[dataset + "_seven"], sources[dataset + "_fp8"]
        a, b = original["summary"], fp8["summary"]
        require(a["completed_images"] == b["completed_images"] == count, dataset + ": expected selected image count")
        require(set(a["config"]["models"]) == set(MODELS) - {"fp8"}, dataset + ": expected seven original models")
        require(set(b["config"]["models"]) == {"fp16", "fp8"}, dataset + ": expected FP16 control and FP8")
        require(set(a["ground_truth_variants"]) == set(b["ground_truth_variants"]) == expected_kinds,
                dataset + ": expected GT variants")
        for key in ("manifest_sha256", "min_depth", "max_depth", "threads", "inter_op_threads", "cuda_use_tf32"):
            require(a["config"][key] == b["config"][key], dataset + ": mismatching " + key)
        require(a["method"] == b["method"], dataset + ": mismatching evaluation method")
        require(a["config"]["max_depth"] == (10.0 if dataset == "nyu" else 10000.0), dataset + ": unexpected GT range")
        values = {"images": count, "manifest_sha256": a["config"]["manifest_sha256"],
                  "minimum_depth_exclusive": a["config"]["min_depth"],
                  "maximum_depth_exclusive": a["config"]["max_depth"],
                  "fp16_opset_control_metric_check": compare_control(original, fp8, sorted(expected_kinds), count),
                  "ground_truth_variants": {}}
        for kind in sorted(expected_kinds):
            baseline_rows = selected(original, kind, "fp32")
            require(len(baseline_rows) == count, dataset + ": reference row count")
            baseline_ids = [row["id"] for row in baseline_rows]
            model_results = {}
            for name in MODELS:
                source = fp8 if name == "fp8" else original
                chosen = selected(source, kind, name)
                require(len(chosen) == count and [row["id"] for row in chosen] == baseline_ids,
                        dataset + "/" + name + ": mismatching image set/order")
                require(all(row["valid_pixels"] == reference["valid_pixels"] and row["scene"] == reference["scene"]
                            for row, reference in zip(chosen, baseline_rows)),
                        dataset + "/" + name + ": GT mask count/scene mismatch")
                means = {metric: float(np.mean([float(row[metric]) for row in chosen])) for metric in METRICS}
                original_model = source["summary"]["ground_truth_variants"][kind]["models"][name]
                for metric, value in means.items():
                    require(math.isclose(value, original_model[metric]["mean"], rel_tol=1e-12, abs_tol=1e-14),
                            dataset + "/" + name + ": source summary mean mismatch for " + metric)
                model = {"model": name, **public_model_spec(source["summary"], name), "images": len(chosen),
                    "scenes": len({row["scene"] for row in chosen}),
                    "valid_pixels": sum(int(row["valid_pixels"]) for row in chosen),
                    "image_means": means}
                for metric, field in (("abs_rel", "versus_fp32"), ("inverse_nrmse", "versus_fp32_inverse_nrmse")):
                    candidate = [float(row[metric]) for row in chosen]
                    reference = [float(row[metric]) for row in baseline_rows]
                    scenes = [row["scene"] for row in chosen]
                    model[field] = {**bootstrap_mean_delta(candidate, reference, metric=metric),
                                    **bootstrap_scene_mean_delta(candidate, reference, scenes, metric=metric)}
                model_results[name] = model
                output_rows.append({"dataset": dataset, "gt_kind": kind, "model": name,
                    "images": len(chosen), "scenes": model["scenes"], "valid_pixels": model["valid_pixels"],
                    **means, "model_sha256": model["sha256"], "file_size_mib": model["file_size_mib"],
                    "mean_delta_abs_rel_vs_fp32": model["versus_fp32"]["mean_delta_abs_rel"],
                    "scene_ci95_delta_abs_rel_lower": model["versus_fp32"]["scene_cluster_ci95_delta_abs_rel"][0],
                    "scene_ci95_delta_abs_rel_upper": model["versus_fp32"]["scene_cluster_ci95_delta_abs_rel"][1],
                    "mean_delta_inverse_nrmse_vs_fp32": model["versus_fp32_inverse_nrmse"]["mean_delta_inverse_nrmse"],
                    "scene_ci95_delta_inverse_nrmse_lower": model["versus_fp32_inverse_nrmse"]["scene_cluster_ci95_delta_inverse_nrmse"][0],
                    "scene_ci95_delta_inverse_nrmse_upper": model["versus_fp32_inverse_nrmse"]["scene_cluster_ci95_delta_inverse_nrmse"][1]})
            values["ground_truth_variants"][kind] = {"models": model_results}
        result["datasets"][dataset] = values
    # Inputs must remain unchanged while being summarized.
    for source in sources.values():
        for filename, expected in source["input_hashes"].items():
            require(sha256(source["directory"] / filename) == expected, "Input changed during summary generation")
    result["validation"] = {"status": "PASS", "source_inputs_unchanged": True,
        "all_24_mean_metric_rows_match_source_summaries": True, "row_count": len(output_rows),
        "no_images_or_GT_or_predictions_read": True, "no_inference_or_benchmark_performed": True}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "all_models_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "all_models_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"PASS: {len(output_rows)} dataset/GT/model rows; eight models; source inputs unchanged; saved {args.output_dir}")


if __name__ == "__main__":
    main()
