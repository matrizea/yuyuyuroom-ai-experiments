"""Compare an image with FP32/INT8/INT4 models; no ground-truth accuracy is measured.

Example:
    .venv/Scripts/python.exe scripts/compare_image.py --image C:/photos/example.jpg
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re

import runtime_init  # Preload the local Windows MSVC runtime before ONNX Runtime.
import cv2
import numpy as np

from evaluate import DEFAULT_MODELS, infer, make_session, write_json
from eval_utils import ROOT, compare_depth, preprocess, save_preview, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="Original image path; Unicode names are supported")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/custom")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Variant names (models/NAME.onnx) or NAME=PATH; include fp32")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    image_path = args.image.resolve()
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")
    model_paths = {}
    for argument in args.models:
        name, separator, location = argument.partition("=")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            parser.error(f"Invalid model name: {name!r}")
        if name in model_paths:
            parser.error(f"Duplicate model: {name}")
        path = Path(location).resolve() if separator else ROOT/"models"/(name+".onnx")
        if not path.is_file():
            parser.error(f"Model not found: {path}")
        model_paths[name] = path
    if "fp32" not in model_paths:
        parser.error("Include fp32 in --models to define the reference")
    model_paths = {"fp32": model_paths.pop("fp32"), **model_paths}
    cv2.setNumThreads(1)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Could not decode image: {image_path}")
    tensor = preprocess(image, input_size=518, resize_mode="square")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    depths = {}
    results = {}
    for name, path in model_paths.items():
        print(f"Running {name}...", flush=True)
        session = make_session(path, args.threads)
        depth = np.asarray(infer(session, tensor), dtype=np.float32)
        del session
        depth_path = output_dir/f"{name}_depth.npy"
        np.save(depth_path, depth, allow_pickle=False)
        depths[name] = depth
        results[name] = {"model_path": str(path), "model_sha256": sha256(path),
                         "file_size_mib": path.stat().st_size/(1024**2),
                         "depth_file": depth_path.name, "depth_dtype": "float32",
                         "depth_shape": list(depth.shape),
                         "map_metrics_vs_fp32": compare_depth(depths["fp32"], depth)}
    metrics = {name: result["map_metrics_vs_fp32"] for name,result in results.items()}
    preview_path = output_dir/"comparison.jpg"
    save_preview(preview_path, image, depths, metrics, [])
    preview = cv2.imdecode(np.fromfile(preview_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    preview[:28, :240] = 245
    cv2.putText(preview, "Original (no ground truth)", (7,19), cv2.FONT_HERSHEY_SIMPLEX,
                0.44, (25,25,25), 1, cv2.LINE_AA)
    success, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY,88])
    if not success:
        raise IOError(f"Could not encode preview: {preview_path}")
    preview_path.write_bytes(encoded.tobytes())
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image": str(image_path), "image_sha256": sha256(image_path),
        "original_shape_hwc": list(image.shape), "model_input_shape": list(tensor.shape),
        "ground_truth_available": False, "ordinal_accuracy": None,
        "interpretation_ja": "正解深度・近遠注釈がないため正答率は測定していません。各指標はFP32出力との差であり、実際の深度精度を保証しません。",
        "preprocessing": "Square 518x518 OpenCV CUBIC resize without crop (aspect ratio may change), RGB/255 float64, official ImageNet normalization, float32 NCHW",
        "depth_values": "Native 518x518 relative inverse depth: larger values indicate nearer surfaces; values are not distance in metres",
        "map_metrics": "Least-squares scale+shift candidate to FP32; aligned RMSE/std(FP32); Spearman on bilinear 64x64 grid; raw RMSE before alignment",
        "preview": preview_path.name,
        "preview_scales": "All depths share FP32 min/max; aligned differences share +/-10% of the FP32 range",
        "cpu_config": {"intra_op_threads": args.threads, "inter_op_threads": 1,
                       "execution_mode": "sequential", "provider": "CPUExecutionProvider"},
        "models": results,
    }
    write_json(output_dir/"comparison.json", summary)
    for name, model_metrics in metrics.items():
        error = model_metrics["aligned_nrmse"]
        correlation = model_metrics["spearman_64x64"]
        print(f"{name}: aligned_NRMSE={error}, Spearman={correlation}", flush=True)
    print("No ground-truth accuracy or latency benchmark was measured.", flush=True)
    print(f"Saved {preview_path} and {output_dir/'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
