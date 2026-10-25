"""Plot eight cached CUDA-experiment outputs on two explicitly selected NYUv2 images.

No inference is run. RGB images are never opened. Only model predictions are
rendered; measured GT and its validity mask are read privately to verify labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dense_utils import restore_prediction


MODELS = ["fp32", "fp16", "int8_dynamic", "int4_rtn", "int4_fast", "int8_w8a16", "int4_w4a16", "fp8"]
LABELS = {
    "fp32": "FP32", "fp16": "FP16", "int8_dynamic": "dynamic INT8",
    "int4_rtn": "INT4 (FP32)", "int4_fast": "INT4 (INT8 allowed)",
    "int8_w8a16": "W8A16", "int4_w4a16": "W4A16", "fp8": "FP8 W8A8",
}
DISPLAY_CAPTION_JA = (
    "色はGTへ整合する前の逆深度出力で、同じ例の8モデル全体の最小・最大値を8モデルに共通適用した。"
    "明るい色ほど近く、暗い色ほど遠い。全出力値を色域に含め、表示のクリッピングもモデル別の正規化も行わない。"
    "AbsRelは実測GTの共通有効画素で画像・モデルごとに倍率と切片を合わせ、距離へ戻した後の誤差であり、"
    "この表示色の正規化とは別の処理である。元RGB画像・GT深度画像は掲載していない。"
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def workspace() -> Path:
    for candidate in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (candidate / "results_dense_nyu_gpu/per_image.csv").is_file():
            return candidate
    raise FileNotFoundError("Run inside the experiment workspace with cached dense predictions")


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def independent_absrel(prediction: np.ndarray, depth: np.ndarray, mask: np.ndarray, maximum: float) -> dict:
    """Independent least-squares solver cross-check of the cached metric CSV.

    Solve a two-column system, rather than calling evaluate_prediction's centered
    covariance implementation. The constraint and inverse-depth floor are equal.
    """
    p = prediction[mask].astype(np.float64)
    z = depth[mask].astype(np.float64)
    design = np.column_stack((p, np.ones_like(p)))
    unconstrained, shift = np.linalg.lstsq(design, 1 / z, rcond=None)[0]
    scale = max(float(unconstrained), 0.0)
    if scale == 0:
        shift = float(np.mean(1 / z))
    aligned_inverse = scale * p + float(shift)
    predicted_depth = 1 / np.maximum(aligned_inverse, 1 / maximum)
    return {"abs_rel": float(np.mean(np.abs(predicted_depth - z) / z)),
            "alignment_scale": scale, "alignment_shift": float(shift),
            "valid_pixels": int(mask.sum())}


def hide_axes(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#a6adb8")
        spine.set_linewidth(0.8)


def render_summary(path: Path, identifier: str, kind: str, maps: dict, metrics: dict,
                   lower: float, upper: float) -> None:
    fig = plt.figure(figsize=(16, 26.8), dpi=100, facecolor="white")
    grid = fig.add_gridspec(4, 2, left=.045, right=.955, bottom=.115, top=.895, hspace=.25, wspace=.075)
    for index, model in enumerate(MODELS):
        axis = fig.add_subplot(grid[index // 2, index % 2])
        colored = axis.imshow(maps[model], cmap="inferno", vmin=lower, vmax=upper, interpolation="nearest")
        axis.set_title(f"{LABELS[model]}\nMeasured-GT AbsRel = {metrics[model]:.6f}", fontsize=20, pad=13)
        hide_axes(axis)
    title = "Fixed first test image" if kind == "fixed" else "Largest measured-GT AbsRel deterioration"
    fig.suptitle(f"NYUv2: all eight model outputs\n{title} | {identifier}", fontsize=26, fontweight="bold", y=.987)
    fig.text(.045, .944, "Raw inverse depth; shared min/max across all eight models (no clipping)", fontsize=19, color="#374151")
    cbar = fig.colorbar(colored, cax=fig.add_axes([.10, .080, .80, .012]), orientation="horizontal")
    cbar.set_label("Raw inverse depth (arbitrary units)   |   dark = farther, bright = nearer", fontsize=18, labelpad=13)
    cbar.ax.tick_params(labelsize=16)
    fig.text(.045, .036, "AbsRel: per-image oracle alignment to measured GT; full frame, common valid pixels, 0 < depth < 10 m.", fontsize=15, color="#374151")
    fig.text(.045, .020, "One shared display range covers all output values. No per-model normalization. No RGB or GT images are shown.", fontsize=15, color="#374151")
    fig.savefig(path, dpi=100, metadata={"Title": "NYUv2 eight cached model outputs", "Description": "Model output maps only; shared min/max across all eight models, no clipping; no input RGB or GT images"})
    plt.close(fig)


def render_individual(path: Path, identifier: str, model: str, prediction: np.ndarray,
                      metric: float, lower: float, upper: float) -> None:
    fig = plt.figure(figsize=(12.8, 11.2), dpi=100, facecolor="white")
    axis = fig.add_axes([.055, .245, .89, .64])
    colored = axis.imshow(prediction, cmap="inferno", vmin=lower, vmax=upper, interpolation="nearest")
    hide_axes(axis)
    fig.suptitle(f"{LABELS[model]} | {identifier}\nMeasured-GT AbsRel = {metric:.6f}", fontsize=25, fontweight="bold", y=.968)
    cbar = fig.colorbar(colored, cax=fig.add_axes([.10, .167, .80, .027]), orientation="horizontal")
    cbar.set_label("Raw inverse depth (arbitrary units) | dark = farther, bright = nearer", fontsize=16, labelpad=12)
    cbar.ax.tick_params(labelsize=14)
    fig.text(.055, .083, "Shared min/max across all 8 models for this image; no clipping.", fontsize=16, color="#374151")
    fig.text(.055, .055, "AbsRel uses measured-GT oracle alignment; display uses raw output.", fontsize=16, color="#374151")
    fig.text(.055, .027, "Model output only. No input RGB or ground-truth depth image is shown.", fontsize=16, color="#374151")
    fig.savefig(path, dpi=100, metadata={"Title": "NYUv2 cached prediction: " + LABELS[model], "Description": "Model output only; shared min/max across all eight models, no clipping; no input RGB or GT image"})
    plt.close(fig)


def main() -> None:
    root = workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "article_figures/all_models")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "data_dense/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = [item["id"] for item in manifest["images"]]
    if len(ids) != 128 or len(set(ids)) != 128:
        raise ValueError("Require the complete fixed NYUv2 128-image manifest")
    items = {item["id"]: item for item in manifest["images"]}
    csv_paths = {"base": root / "results_dense_nyu_gpu/per_image.csv", "fp8": root / "results_fp8/nyu/per_image.csv"}
    source_results = {"base": root / "results_dense_nyu_gpu", "fp8": root / "results_fp8/nyu"}
    source_rows = {name: csv_rows(path) for name, path in csv_paths.items()}
    summaries = {name: json.loads((folder / "summary.json").read_text(encoding="utf-8")) for name, folder in source_results.items()}
    manifest_hash = sha256(manifest_path)
    for name, summary in summaries.items():
        if summary["config"]["manifest_sha256"] != manifest_hash or summary["config"]["provider"] != "CUDA":
            raise ValueError(f"Mismatched source manifest or provider: {name}")
        if summary["config"]["min_depth"] != 0 or summary["config"]["max_depth"] != 10:
            raise ValueError("Require the unchanged NYUv2 0 < depth < 10 m evaluation range")
    registry = {}
    metrics = {}
    for model in MODELS:
        source = "fp8" if model == "fp8" else "base"
        registry[model] = {"result_directory": relative(source_results[source], root),
                           "per_image_csv": relative(csv_paths[source], root),
                           "per_image_csv_sha256": sha256(csv_paths[source]),
                           "summary_sha256": sha256(source_results[source] / "summary.json"),
                           "source_model_sha256": summaries[source]["config"]["models"][model]["sha256"],
                           "label": LABELS[model]}
        for gt_kind in ("measured", "inpainted"):
            selected = [row for row in source_rows[source] if row["model"] == model and row["gt_kind"] == gt_kind]
            if len(selected) != 128 or {row["id"] for row in selected} != set(ids):
                raise ValueError(f"Incomplete or duplicated metric rows: {model}/{gt_kind}")
            metrics[model, gt_kind] = {row["id"]: row for row in selected}
        for identifier in ids:
            if int(metrics[model, "measured"][identifier]["valid_pixels"]) != int(metrics["fp32", "measured"][identifier]["valid_pixels"]):
                raise ValueError("All variants must use the same measured-GT valid pixels")
            if metrics[model, "measured"][identifier]["scene"] != items[identifier]["scene"]:
                raise ValueError("Metric scene and manifest scene differ")
    candidates = [{"id": identifier, "model": model,
                   "measured_absrel_minus_fp32": float(metrics[model, "measured"][identifier]["abs_rel"]) - float(metrics["fp32", "measured"][identifier]["abs_rel"])}
                  for identifier in ids for model in MODELS if model != "fp32"]
    worst = min(candidates, key=lambda row: (-row["measured_absrel_minus_fp32"], row["id"], row["model"]))
    selections = {"fixed": ids[0], "worst": worst["id"]}
    metadata = {
        "dataset": manifest["dataset"], "manifest": relative(manifest_path, root), "manifest_sha256": manifest_hash,
        "subset_images": 128, "subset_seed": 42, "provider": "CUDA experiment outputs (some models have CPU fallback)",
        "models_order": MODELS, "models": registry,
        "selection": {"fixed": "First image in the existing ordered manifest; independent of error and appearance.",
                      "worst": "Maximize per-image measured-GT AbsRel(model) - AbsRel(FP32) over all 7 alternatives x 128 images. Ascending image ID then model key breaks ties.",
                      "candidate_image_model_pairs": len(candidates), "worst_pair": worst},
        "display": {"quantity": "Raw model inverse-depth output before GT alignment", "unit": "arbitrary units, not inverse meters",
                    "restore": "Bilinear align_corners=True, using dense_utils.restore_prediction, from native 518x518 to original 480x640",
                    "common_range": "Each sample uses the min and max over all eight restored model outputs, shared by its summary and all eight individual images",
                    "range_strategy": "per_sample_all_eight_models_global_min_max",
                    "per_model_normalization": False, "gt_alignment_used_for_display": False,
                    "colormap": "inferno", "direction": "Darker = farther, brighter = nearer",
                    "out_of_range": "None: every displayed value is inside the common range; below/above fractions are independently asserted zero"},
        "metric": {"name": "AbsRel", "gt_kind": "measured", "definition": "mean(abs(predicted_depth - measured_depth) / measured_depth) on common GT-only valid mask",
                   "gt_unit": "meters", "gt_distance_type": manifest["depth_type"], "mask_rule": manifest["mask_rule"],
                   "crop": "none, full 480x640", "alignment": "Per-image, per-model least squares fit s * raw_inverse_prediction + t to 1 / measured_depth, with s >= 0; inverse depth floored at 0.1 before inversion",
                   "interpretation": "Oracle-aligned relative depth shape error; not absolute-distance accuracy without GT",
                   "label_source": "Existing per_image.csv; independently recomputed for the 16 displayed image/model pairs with a two-column np.linalg.lstsq solver"},
        "privacy_and_redistribution": {"rgb_arrays_loaded": False, "gt_arrays_loaded_privately_for_metric_check": True,
                                       "rgb_or_gt_rendered": False, "rgb_or_gt_copied_to_output": False,
                                       "prediction_arrays_copied_to_output": False},
        "caption_display_ja": DISPLAY_CAPTION_JA,
        "samples": {}, "model_image_links": {model: {} for model in MODELS},
    }
    selected_rows = []
    for kind, identifier in selections.items():
        item = items[identifier]
        shape = (item["height"], item["width"])
        maps, paths = {}, {}
        depth_path, mask_path = root / item["raw_depth_path"], root / item["raw_mask_path"]
        for key, path in (("raw_depth_path", depth_path), ("raw_mask_path", mask_path)):
            if sha256(path) != item["sha256"][key]:
                raise ValueError(f"Private GT file no longer matches manifest hash: {key}")
        depth = np.load(depth_path, allow_pickle=False).astype(np.float64) * float(item["depth_unit_scale"])
        valid = np.load(mask_path, allow_pickle=False).astype(bool) & np.isfinite(depth) & (depth > 0) & (depth < 10)
        if depth.shape != shape:
            raise ValueError("Unexpected GT dimensions")
        sample_metrics = {model: float(metrics[model, "measured"][identifier]["abs_rel"]) for model in MODELS}
        for model in MODELS:
            source = "fp8" if model == "fp8" else "base"
            paths[model] = source_results[source] / "predictions" / model / (identifier + ".npy")
            native = np.load(paths[model], allow_pickle=False).squeeze()
            if native.shape != (518, 518) or native.dtype != np.float32 or not np.isfinite(native).all():
                raise ValueError(f"Require finite native float32 inverse-depth maps: {model}/{identifier}")
            maps[model] = restore_prediction(native, shape)
        lower = min(float(prediction.min()) for prediction in maps.values())
        upper = max(float(prediction.max()) for prediction in maps.values())
        if not upper > lower:
            raise ValueError("Common display range must be nonconstant")
        sample = {"id": identifier, "scene": item["scene"], "category": item["category"],
                  "selection": metadata["selection"][kind], "native_prediction_shape": [518, 518],
                  "display_shape": list(shape), "common_vmin": lower, "common_vmax": upper,
                  "fp32_map_min": float(maps["fp32"].min()), "fp32_map_max": float(maps["fp32"].max()),
                  "measured_valid_pixels": int(valid.sum()), "total_pixels": int(valid.size),
                  "measured_valid_fraction": float(valid.mean()),
                  "private_gt_hashes_checked_against_manifest": True, "models": {}}
        for model in MODELS:
            check = independent_absrel(maps[model], depth, valid, maximum=10.0)
            csv_metric = sample_metrics[model]
            error = abs(check["abs_rel"] - csv_metric)
            if not math.isclose(check["abs_rel"], csv_metric, rel_tol=5e-10, abs_tol=5e-12):
                raise ValueError(f"Recomputed AbsRel differs from source CSV: {kind}/{model}: {error}")
            if check["valid_pixels"] != int(metrics[model, "measured"][identifier]["valid_pixels"]):
                raise ValueError("Recomputed valid mask differs from source CSV")
            filename = f"nyu_all_models_{kind}_{identifier}_{model}.png"
            render_individual(output / filename, identifier, model, maps[model], csv_metric, lower, upper)
            info = {"label": LABELS[model], "measured_gt_abs_rel": csv_metric,
                    "measured_gt_absrel_independent_check": check, "absrel_absolute_verification_error": error,
                    "measured_gt_absrel_minus_fp32": csv_metric - sample_metrics["fp32"],
                    "native_prediction_file": relative(paths[model], root), "native_prediction_sha256": sha256(paths[model]),
                    "restored_prediction_min": float(maps[model].min()), "restored_prediction_max": float(maps[model].max()),
                    "display_clipped_below_fraction": float(np.mean(maps[model] < lower)),
                    "display_clipped_above_fraction": float(np.mean(maps[model] > upper)),
                    "png": filename, "png_sha256": sha256(output / filename), "png_size": [1280, 1120],
                    "alt_ja": f"{identifier}に対する{LABELS[model]}の逆深度出力。実測GTによる整合後のAbsRelは{csv_metric:.6f}。8モデル全体の最小・最大値による共通色域で明るいほど近い。クリッピングなし、元RGB・GT画像は含まない。"}
            if info["display_clipped_below_fraction"] != 0 or info["display_clipped_above_fraction"] != 0:
                raise AssertionError("All displayed values must fit within the all-model common range")
            sample["models"][model] = info
            metadata["model_image_links"][model][kind] = {"sample_id": identifier, "measured_gt_abs_rel": csv_metric,
                                                       "image_relative_path": filename, "image_sha256": info["png_sha256"]}
            selected_rows.append({"sample_kind": kind, "id": identifier, "model": model, "model_label": LABELS[model],
                                  "measured_gt_abs_rel": csv_metric, "measured_absrel_minus_fp32": info["measured_gt_absrel_minus_fp32"],
                                  "valid_pixels": check["valid_pixels"], "png": filename,
                                  "independently_recomputed_absrel": check["abs_rel"], "absrel_check_absolute_error": error,
                                  "common_vmin": lower, "common_vmax": upper,
                                  "display_clipped_below_fraction": info["display_clipped_below_fraction"],
                                  "display_clipped_above_fraction": info["display_clipped_above_fraction"]})
        summary_name = f"nyu_all_models_{kind}_{identifier}.png"
        render_summary(output / summary_name, identifier, kind, maps, sample_metrics, lower, upper)
        sample["summary_png"] = summary_name
        sample["summary_png_sha256"] = sha256(output / summary_name)
        sample["summary_png_size"] = [1600, 2680]
        sample["alt_ja"] = f"NYUv2の{identifier}に対する8モデルの逆深度出力を2列4行で比較。FP32、FP16、dynamic INT8、INT4（FP32）、INT4（INT8許可）、W8A16、W4A16、FP8 W8A8を、8モデル全体の最小・最大値による共通色域で表示。各出力に実測GTによる整合後のAbsRelを付記。クリッピングなし、元RGB・GT画像は含まない。"
        reason = (f"既存のNYUv2テスト128画像manifestの先頭{identifier}を固定例とした。誤差や見た目による選定ではない。"
                  if kind == "fixed" else
                  f"NYUv2テスト128画像×FP32以外の7モデルの896組から、実測GTに対するAbsRelのFP32比増加が最大の例{identifier}を選んだ。最大は{LABELS[worst['model']]}の+{worst['measured_absrel_minus_fp32']:.6f}。典型例や平均的な画質を表すものではない。")
        sample["caption_ja"] = reason + DISPLAY_CAPTION_JA
        metadata["samples"][kind] = sample
    with (output / "selected_image_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    with (output / "selection_candidates.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    aggregate = [{"model": model, "label": LABELS[model], "images_per_gt_kind": len(ids),
                  "measured_mean_abs_rel": float(np.mean([float(metrics[model, "measured"][identifier]["abs_rel"]) for identifier in ids])),
                  "inpainted_mean_abs_rel": float(np.mean([float(metrics[model, "inpainted"][identifier]["abs_rel"]) for identifier in ids]))}
                 for model in MODELS]
    with (output / "nyu128_aggregate_source.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    for path in output.glob("nyu_all_models_*.png"):
        with Image.open(path) as opened:
            expected = (1280, 1120) if any(path.name.endswith("_" + model + ".png") for model in MODELS) else (1600, 2680)
            if opened.size != expected:
                raise AssertionError(f"Unexpected PNG dimensions: {path.name}: {opened.size}")
    metadata["checks"] = {"all_source_rows_complete": True, "all_source_provider_cuda": True,
                          "all_source_manifest_hashes_match": True, "displayed_metric_checks": 16,
                          "max_absrel_verification_error": max(row["absrel_check_absolute_error"] for row in selected_rows),
                          "gt_mask_valid_counts_match_all_displayed_csv_rows": True,
                          "all_display_clipping_fractions_zero": True,
                          "original_rgb_files_opened": 0, "gt_or_rgb_arrays_written_to_output": 0}
    write_json(output / "metadata.json", metadata)
    script_copy = output / "plot_all_models_dense.py"
    if script_copy.resolve() != Path(__file__).resolve():
        shutil.copyfile(Path(__file__), script_copy)
    generation = {"generator": "scripts/plot_all_models_dense.py",
                  "command": ".venv/Scripts/python.exe scripts/plot_all_models_dense.py --output-dir " + relative(output, root),
                  "default_public_output_directory": "article_figures/all_models",
                  "python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                  "script_sha256": sha256(script_copy),
                  "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(output.iterdir()) if path.is_file() and path.name != "generation.json"}}
    write_json(output / "generation.json", generation)
    print(json.dumps({"output_directory": relative(output, root), "selected": selections,
                      "largest_deterioration": worst, "checks": metadata["checks"], "png_count": len(list(output.glob("*.png")))}, indent=2))


if __name__ == "__main__":
    main()
