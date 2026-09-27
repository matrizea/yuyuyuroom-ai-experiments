#!/usr/bin/env python3
"""Calibrate a saved FashionMNIST ONNX CNN and measure actual ORT CPU inference.

The test set is never used for calibration. Raw per-call timings are retained.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import platform
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torchvision import datasets


BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
RESULT_DIR = BASE / "results" / "onnx_int8_static"
FP32_MODEL = MODEL_DIR / "fashion_cnn_dynamic.onnx"
INT8_MODEL = MODEL_DIR / "fashion_cnn_static_int8_qdq.onnx"
PREPROCESSED_MODEL = MODEL_DIR / "fashion_cnn_preprocessed.onnx"
PREPROCESSED_INT8_MODEL = MODEL_DIR / "fashion_cnn_preprocessed_static_int8_qdq.onnx"
SEED = 42
CALIBRATION_IMAGES = 512
CALIBRATION_BATCH = 64
TEST_BATCH = 256
BENCHMARK_BATCHES = (1, 32, 256)
WARMUP_CALLS = 20
BENCHMARK_ROUNDS = 5
CALLS_PER_ROUND = 100


class ArrayCalibrationReader(CalibrationDataReader):
    def __init__(self, images: np.ndarray, batch_size: int) -> None:
        self.images = images
        self.batch_size = batch_size
        self.cursor = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.cursor >= len(self.images):
            return None
        batch = self.images[self.cursor : self.cursor + self.batch_size]
        self.cursor += len(batch)
        return {"image": batch}

    def rewind(self) -> None:
        self.cursor = 0


def images_to_numpy(raw: torch.Tensor) -> np.ndarray:
    # FashionMNIST ToTensor also scales uint8 pixels from 0..255 to 0..1.
    return np.ascontiguousarray(raw.numpy()[:, None, :, :], dtype=np.float32) / 255.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_summary(path: Path) -> dict[str, object]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return {
        "node_counts": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "initializer_types": dict(
            sorted(
                Counter(onnx.TensorProto.DataType.Name(init.data_type) for init in model.graph.initializer).items()
            )
        ),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
    }


def create_session(model_path: Path, optimized_path: Path) -> tuple[ort.InferenceSession, float]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(optimized_path)
    started = time.perf_counter()
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    return session, (time.perf_counter() - started) * 1000.0


def predict_all(session: ort.InferenceSession, images: np.ndarray) -> np.ndarray:
    outputs = []
    for begin in range(0, len(images), TEST_BATCH):
        outputs.append(session.run(["logits"], {"image": images[begin : begin + TEST_BATCH]})[0])
    return np.concatenate(outputs)


def benchmark(sessions: dict[str, ort.InferenceSession], images: np.ndarray) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    aggregate: dict[str, dict] = {}
    for size in BENCHMARK_BATCHES:
        input_batch = np.ascontiguousarray(images[:size])
        for session in sessions.values():
            for _ in range(WARMUP_CALLS):
                session.run(["logits"], {"image": input_batch})
        for round_index in range(BENCHMARK_ROUNDS):
            # Alternating order reduces drift from one variant always running first.
            names = ("fp32", "int8") if round_index % 2 == 0 else ("int8", "fp32")
            for name in names:
                session = sessions[name]
                for call_index in range(CALLS_PER_ROUND):
                    start = time.perf_counter_ns()
                    session.run(["logits"], {"image": input_batch})
                    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
                    rows.append(
                        {
                            "batch_size": size,
                            "variant": name,
                            "round": round_index + 1,
                            "call": call_index + 1,
                            "elapsed_ms": elapsed_ms,
                        }
                    )
        aggregate[str(size)] = {}
        for name in ("fp32", "int8"):
            values = [r["elapsed_ms"] for r in rows if r["batch_size"] == size and r["variant"] == name]
            round_medians = [
                statistics.median(
                    r["elapsed_ms"]
                    for r in rows
                    if r["batch_size"] == size and r["variant"] == name and r["round"] == round_index
                )
                for round_index in range(1, BENCHMARK_ROUNDS + 1)
            ]
            aggregate[str(size)][name] = {
                "median_ms": statistics.median(values),
                "p95_ms": float(np.percentile(values, 95)),
                "round_medians_ms": round_medians,
                "calls": len(values),
            }
        aggregate[str(size)]["int8_over_fp32_speed_ratio"] = (
            aggregate[str(size)]["fp32"]["median_ms"] / aggregate[str(size)]["int8"]["median_ms"]
        )
    return rows, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocess", action="store_true", help="Apply ORT's recommended preprocessing first")
    args = parser.parse_args()
    if not FP32_MODEL.is_file():
        raise FileNotFoundError(f"First run experiment_pytorch_onnx.py: {FP32_MODEL}")
    output_dir = RESULT_DIR / "preprocessed" if args.preprocess else RESULT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train = datasets.FashionMNIST(DATA_DIR, train=True, download=True)
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True)
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(SEED))[:CALIBRATION_IMAGES]
    calibration_images = images_to_numpy(train.data[indices])
    test_images = images_to_numpy(test.data)
    labels = test.targets.numpy()
    if len(test_images) != 10_000 or len(calibration_images) != CALIBRATION_IMAGES:
        raise RuntimeError("Unexpected FashionMNIST input sizes")

    quant_input = FP32_MODEL
    quant_output = INT8_MODEL
    preprocessing_seconds = 0.0
    if args.preprocess:
        preprocessing_started = time.perf_counter()
        quant_pre_process(input_model=FP32_MODEL, output_model_path=PREPROCESSED_MODEL)
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        quant_input = PREPROCESSED_MODEL
        quant_output = PREPROCESSED_INT8_MODEL

    calibration_started = time.perf_counter()
    quantize_static(
        model_input=str(quant_input),
        model_output=str(quant_output),
        calibration_data_reader=ArrayCalibrationReader(calibration_images, CALIBRATION_BATCH),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
    )
    calibration_seconds = time.perf_counter() - calibration_started

    sessions = {}
    session_init_ms = {}
    for name, path in (("fp32", FP32_MODEL), ("int8", quant_output)):
        sessions[name], session_init_ms[name] = create_session(path, output_dir / f"{name}_optimized.onnx")
        if sessions[name].get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError(f"Unexpected providers for {name}: {sessions[name].get_providers()}")

    fp32_logits = predict_all(sessions["fp32"], test_images)
    int8_logits = predict_all(sessions["int8"], test_images)
    fp32_pred = fp32_logits.argmax(axis=1)
    int8_pred = int8_logits.argmax(axis=1)
    difference = np.abs(fp32_logits - int8_logits)
    timing_rows, timings = benchmark(sessions, test_images)
    with (output_dir / "timings_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["batch_size", "variant", "round", "call", "elapsed_ms"])
        writer.writeheader()
        writer.writerows(timing_rows)

    model_artifacts = {}
    paths = [
        ("fp32", FP32_MODEL),
        ("int8", quant_output),
        ("fp32_optimized", output_dir / "fp32_optimized.onnx"),
        ("int8_optimized", output_dir / "int8_optimized.onnx"),
    ]
    if args.preprocess:
        paths.append(("preprocessed_fp32", PREPROCESSED_MODEL))
    for name, path in paths:
        model_artifacts[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "graph": graph_summary(path),
        }

    result = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "available_providers": ort.get_available_providers(),
            "used_providers": {name: session.get_providers() for name, session in sessions.items()},
            "cpu": platform.processor(),
        },
        "protocol": {
            "seed": SEED,
            "calibration_source": "FashionMNIST train only",
            "calibration_images": CALIBRATION_IMAGES,
            "calibration_indices": indices.tolist(),
            "calibration_batch": CALIBRATION_BATCH,
            "test_source": "FashionMNIST test only",
            "test_images": len(test_images),
            "test_batch": TEST_BATCH,
            "quantization": "static QDQ, MinMax, per-channel, QInt8 activations and weights",
            "preprocessing": args.preprocess,
            "quantization_input": str(quant_input),
            "ops_to_quantize": ["Conv", "Gemm", "MatMul"],
            "execution_provider": "CPUExecutionProvider",
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "graph_optimization": "ORT_ENABLE_ALL",
            "benchmark_batches": BENCHMARK_BATCHES,
            "warmup_calls": WARMUP_CALLS,
            "benchmark_rounds": BENCHMARK_ROUNDS,
            "calls_per_round": CALLS_PER_ROUND,
        },
        "preprocessing_seconds": preprocessing_seconds,
        "calibration_seconds": calibration_seconds,
        "session_init_ms": session_init_ms,
        "accuracy": {
            "fp32": float(np.mean(fp32_pred == labels)),
            "int8": float(np.mean(int8_pred == labels)),
            "changed_predictions": int(np.count_nonzero(fp32_pred != int8_pred)),
            "changed_correct_to_wrong": int(np.count_nonzero((fp32_pred == labels) & (int8_pred != labels))),
            "changed_wrong_to_correct": int(np.count_nonzero((fp32_pred != labels) & (int8_pred == labels))),
            "mean_abs_logit_difference": float(difference.mean()),
            "max_abs_logit_difference": float(difference.max()),
        },
        "timings": timings,
        "artifacts": model_artifacts,
    }
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy": result["accuracy"], "timings": timings, "models": {
        name: {"bytes": info["bytes"], "optimized_nodes": info["graph"]["node_counts"]}
        for name, info in model_artifacts.items()
    }}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
