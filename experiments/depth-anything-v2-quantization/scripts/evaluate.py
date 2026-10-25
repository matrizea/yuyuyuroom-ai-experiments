"""Compare Depth Anything V2 Small ONNX precision variants on the DA-2K split."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import runtime_init  # Load the locally installed MSVC runtime before ONNX Runtime.
import cv2
import numpy as np

from eval_utils import (ROOT, cluster_bootstrap_difference, compare_depth, preprocess,
                        sample_original_points, save_preview, sha256)

DEFAULT_MODELS = ["fp32", "int8_dynamic", "int4_rtn", "int4_fast"]


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def make_session(path: Path, threads: int):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    # Idle sessions must not spin while another variant runs on the same CPU.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def infer(session, tensor: np.ndarray) -> np.ndarray:
    output = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: tensor})[0]
    depth = np.squeeze(output)
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError(f"Invalid depth output: shape={depth.shape}, finite={np.isfinite(depth).all()}")
    return depth


def benchmark(session, warmup: int, iterations: int) -> dict:
    tensor = np.random.default_rng(42).normal(size=(1,3,518,518)).astype(np.float32)
    for _ in range(warmup):
        infer(session, tensor)
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: tensor})
        durations.append((time.perf_counter()-start)*1000)
    return {"median_ms": float(np.median(durations)), "p90_ms": float(np.percentile(durations,90)),
            "min_ms": min(durations), "iterations": iterations, "warmup": warmup,
            "input_shape": [1,3,518,518], "timing_scope": "session.run model only",
            "samples_ms": durations}


def summarize(checkpoint: dict, model_paths: dict, args) -> dict:
    entries = list(checkpoint["images"].values())
    per_image, per_pair = [], []
    for entry in entries:
        per_image.extend(entry["image_rows"])
        per_pair.extend(entry["pair_rows"])
    by_model = {}
    for name, path in model_paths.items():
        image_rows = [row for row in per_image if row["model"] == name]
        pair_rows = [row for row in per_pair if row["model"] == name]
        counts = np.array([row["correct_pairs"] for row in image_rows])
        totals = np.array([row["pair_count"] for row in image_rows])
        fp_counts = np.array([row["fp32_correct_pairs"] for row in image_rows])
        metric_stats = {}
        for metric in ["aligned_nrmse", "spearman_64x64", "raw_rmse"]:
            values = [row[metric] for row in image_rows if row[metric] is not None]
            metric_stats[metric] = {"mean": float(np.mean(values)) if values else None,
                                    "median": float(np.median(values)) if values else None,
                                    "p90": float(np.percentile(values,90)) if values else None,
                                    "max": float(np.max(values)) if values else None}
        categories = {}
        for category in sorted(set(row["category"] for row in pair_rows)):
            subset = [row for row in pair_rows if row["category"] == category]
            correct = sum(row["correct"] for row in subset)
            categories[category] = {"pairs": len(subset), "correct": correct,
                                    "accuracy": correct/len(subset)}
        correct_count = int(counts.sum())
        by_model[name] = {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                          "sha256": checkpoint["config"]["models"][name]["sha256"],
                          "file_size_mib": path.stat().st_size/(1024**2),
                          "images": len(image_rows), "pairs": len(pair_rows), "correct_pairs": correct_count,
                          "accuracy": correct_count/len(pair_rows) if pair_rows else None,
                          "fp32_correct_to_wrong": sum(row["fp32_correct"] and not row["correct"] for row in pair_rows),
                          "fp32_wrong_to_correct": sum(not row["fp32_correct"] and row["correct"] for row in pair_rows),
                          "decision_disagreements_with_fp32": sum(row["fp32_correct"] != row["correct"] for row in pair_rows),
                          "versus_fp32": cluster_bootstrap_difference(counts, fp_counts, totals),
                          "map_metrics": metric_stats, "by_category": categories,
                          "benchmark": checkpoint.get("benchmarks",{}).get(name)}
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": checkpoint["config"],
               "environment": checkpoint["environment"], "completed_images": len(entries),
               "expected_images": checkpoint["expected_images"], "models": by_model,
               "method": {"task": "DA-2K ordinal near/far accuracy; point1 is closer, predicted inverse depth point1 > point2",
                          "preprocessing": ("OpenCV RGB/255(float64), square 518x518 CUBIC resize without crop (aspect ratio may change), official ImageNet normalization, float32 NCHW"
                                            if args.resize_mode == "square" else "Official OpenCV RGB/255(float64), Resize lower_bound=518/aspect preserve/multiple14/CUBIC, ImageNet normalization, float32 NCHW"),
                          "point_sampling": "Direct bilinear sample equivalent to official align_corners=True upsampling to original image",
                          "map_metrics": "Native model output resolution; least-squares scale+shift candidate to FP32, RMSE/std(FP32); Spearman on bilinear 64x64 grid",
                          "confidence_interval": "Paired percentile bootstrap of whole images, 10000 resamples, seed42; no generalization beyond this fixed dataset is implied",
                          "preview": "Shared FP32 min/max for depth; scale+shift aligned difference clipped to +/-10% FP32 range"}}
    write_csv(args.output_dir/"per_image.csv", per_image)
    write_csv(args.output_dir/"per_pair.csv", per_pair)
    write_json(args.output_dir/"summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/eval_manifest.json")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Variant names (models/NAME.onnx), or NAME=PATH; include fp32")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resize-mode", choices=["square", "aspect"], default="square",
                        help="Square matches fixed518 ONNX; aspect requires dynamic-shape models")
    parser.add_argument("--limit", type=int, help="Smoke test with only the first N manifest images; use a separate output directory")
    parser.add_argument("--skip-benchmark", action="store_true", help="Evaluate accuracy now; benchmark later when CPU is idle")
    parser.add_argument("--benchmark-only", action="store_true", help="Run benchmarks and refresh existing checkpoint/summary")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.warmup < 0 or args.iterations < 1 or (args.limit is not None and args.limit < 1):
        parser.error("threads, iterations and limit must be positive; warmup must be nonnegative")
    cv2.setNumThreads(1)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = manifest["images"][:args.limit]
    model_paths = {}
    for value in args.models:
        name, separator, location = value.partition("=")
        if name in model_paths:
            parser.error(f"Duplicate model: {name}")
        model_paths[name] = Path(location).resolve() if separator else ROOT/"models"/(name+".onnx")
    if "fp32" not in model_paths:
        parser.error("Include fp32 to define the reference")
    model_paths = {"fp32": model_paths.pop("fp32"), **model_paths}
    for name, path in model_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} model does not exist: {path}")
    config = {"manifest_sha256": sha256(args.manifest), "limit": args.limit, "threads": args.threads,
              "inter_op_threads": 1, "execution_mode": "sequential", "allow_spinning": False,
              "models": {name: {"path": str(path), "sha256": sha256(path)} for name,path in model_paths.items()},
              "input_size": 518, "resize_mode": args.resize_mode, "previews_enabled": not args.no_previews}
    checkpoint_path = args.output_dir/"checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["config"] != config:
            raise ValueError("Checkpoint configuration differs; choose a separate --output-dir")
        print(f"Resuming {len(checkpoint['images'])}/{len(images)} images", flush=True)
    else:
        if args.benchmark_only:
            raise FileNotFoundError("--benchmark-only requires an existing checkpoint")
        import onnxruntime as ort
        checkpoint = {"config": config, "images": {}, "benchmarks": {}, "expected_images": len(images),
                      "environment": {"python": sys.version, "platform": platform.platform(),
                                      "processor": platform.processor(), "onnxruntime": ort.__version__,
                                      "numpy": np.__version__, "opencv": cv2.__version__,
                                      "providers": ["CPUExecutionProvider"]}}
        write_json(checkpoint_path, checkpoint)
    print("Loading ONNX sessions...", flush=True)
    sessions = {name: make_session(path, args.threads) for name,path in model_paths.items()}
    if not args.benchmark_only:
        start = time.perf_counter()
        for index, item in enumerate(images):
            key = item["path"]
            if key in checkpoint["images"]:
                continue
            image_path = Path(key)
            if not image_path.is_absolute():
                image_path = ROOT/image_path
            # Path/fromfile handles the original dataset's Chinese filename on Windows.
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise IOError(f"Could not read {image_path}")
            tensor = preprocess(image, resize_mode=args.resize_mode)
            depths = {name: infer(session, tensor) for name,session in sessions.items()}
            fp = depths["fp32"]
            metrics = {name: compare_depth(fp, depth) for name,depth in depths.items()}
            annotations = item["annotations"]
            points = [annotation[k] for annotation in annotations for k in ["point1","point2"]]
            values = {name: sample_original_points(depth, points, image.shape).reshape(-1,2)
                      for name,depth in depths.items()}
            correctness = {name: pairs[:,0] > pairs[:,1] for name,pairs in values.items()}
            image_rows, pair_rows = [], []
            for name in model_paths:
                image_rows.append({"image": key, "category": item["category"], "model": name,
                                   "original_h": image.shape[0], "original_w": image.shape[1],
                                   "input_h": tensor.shape[-2], "input_w": tensor.shape[-1],
                                   "pair_count": len(annotations), "correct_pairs": int(correctness[name].sum()),
                                   "fp32_correct_pairs": int(correctness["fp32"].sum()), **metrics[name]})
                for pair_index, annotation in enumerate(annotations):
                    if annotation.get("closer_point", "point1") != "point1":
                        raise ValueError("Expected official point1 closer annotation")
                    a, b = values[name][pair_index]
                    pair_rows.append({"image": key, "category": item["category"], "pair_index": pair_index,
                                      "point1_y": annotation["point1"][0], "point1_x": annotation["point1"][1],
                                      "point2_y": annotation["point2"][0], "point2_x": annotation["point2"][1],
                                      "model": name, "pred_point1": float(a), "pred_point2": float(b),
                                      "margin": float(a-b), "correct": bool(correctness[name][pair_index]),
                                      "fp32_correct": bool(correctness["fp32"][pair_index])})
            preview_path = None
            if not args.no_previews:
                preview_path = args.output_dir/"previews"/f"{index:03d}_{item['category']}_{image_path.stem}.jpg"
                save_preview(preview_path, image, depths, metrics, annotations)
            checkpoint["images"][key] = {"image_rows": image_rows, "pair_rows": pair_rows,
                                         "preview": str(preview_path.relative_to(args.output_dir)) if preview_path else None}
            write_json(checkpoint_path, checkpoint)
            if (index+1) % 16 == 0 or index+1 == len(images):
                elapsed = time.perf_counter()-start
                print(f"Evaluated {index+1}/{len(images)} images; elapsed {elapsed:.1f}s; input {tensor.shape[-2:]}", flush=True)
                summarize(checkpoint, model_paths, args)
    if not args.skip_benchmark:
        for name, session in sessions.items():
            print(f"Benchmarking {name}: {args.warmup} warmups + {args.iterations} runs", flush=True)
            checkpoint["benchmarks"][name] = benchmark(session, args.warmup, args.iterations)
            write_json(checkpoint_path, checkpoint)
    summary = summarize(checkpoint, model_paths, args)
    for name, result in summary["models"].items():
        accuracy = f"{result['accuracy']:.4%}" if result["accuracy"] is not None else "n/a"
        print(f"{name}: accuracy={accuracy}, flips wrong={result['fp32_correct_to_wrong']}, "
              f"fixed={result['fp32_wrong_to_correct']}, size={result['file_size_mib']:.2f}MiB", flush=True)
    print(f"Saved {args.output_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
