"""Calibrated E4M3 W8A8 for the 48 constant MatMuls, using ORT CUDA GemmFloat8.

Other operations remain the existing FP16 model, with FP32 external IO. No
evaluation images or GT are used for calibration. GemmFloat8 consumes FP8 A/B
directly (cuBLASLt FP32 compute type, FP8 fast accumulation and FP16 output),
not dequantized FP16 B.
"""
import runtime_init

import argparse
from collections import Counter, defaultdict
import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
F8 = TensorProto.FLOAT8E4M3FN


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def session(model, profile_prefix=None):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    s = ort.InferenceSession(model if isinstance(model, bytes) else str(model),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"])
    if "CUDAExecutionProvider" not in s.get_providers():
        raise RuntimeError("CUDA failed to initialize")
    return s


def fp8_array(x, scale):
    return np.clip(x.astype(np.float32) / np.float32(scale), -448, 448).astype(ml_dtypes.float8_e4m3fn)


def profile_counts(s):
    path = Path(s.end_profiling())
    counts = defaultdict(Counter)
    for e in json.loads(path.read_text()):
        a = e.get("args", {})
        if a.get("provider") and e.get("name", "").endswith("_kernel_time"):
            counts[a["provider"]][a.get("op_name", "unknown")] += 1
    return path, {k: dict(v) for k, v in counts.items()}


def smoke(dest):
    """Exercise both aligned GEMM and the model's unaligned 1370-token extent."""
    results = []
    rng = np.random.default_rng(42)
    for m in (128, 1370):
        a = rng.normal(size=(m, 384)).astype(np.float16)
        b = rng.normal(scale=.05, size=(1152, 384)).astype(np.float16)
        sa16 = np.float16(np.max(np.abs(a)) / 448)
        sb = np.float32(np.max(np.abs(b)) / 448)
        nodes = [helper.make_node("QuantizeLinear", ["a", "sa16", "zero"], ["aq"], name="FP8_A_quantize"),
                 helper.make_node("GemmFloat8", ["aq", "bq", "", "sa32", "sb32"], ["y"],
                                  domain="com.microsoft", name="native_fp8_gemm", transA=0, transB=1,
                                  dtype=TensorProto.FLOAT16, alpha=1., beta=0.),
                 helper.make_node("Cast", ["aq"], ["aq_float"], to=TensorProto.FLOAT)]
        init = [numpy_helper.from_array(np.asarray(sa16), "sa16"),
                numpy_helper.from_array(np.asarray(sa16, dtype=np.float32), "sa32"),
                numpy_helper.from_array(np.asarray(sb), "sb32"),
                numpy_helper.from_array(np.asarray(0, dtype=ml_dtypes.float8_e4m3fn), "zero"),
                numpy_helper.from_array(fp8_array(b, sb), "bq")]
        graph = helper.make_graph(nodes, "fp8_smoke", [helper.make_tensor_value_info("a", TensorProto.FLOAT16, a.shape)],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [m, 1152]),
                                   helper.make_tensor_value_info("aq_float", TensorProto.FLOAT, a.shape)], init)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)], ir_version=10)
        onnx.checker.check_model(model)
        path = dest / f"smoke_m{m}.onnx"
        onnx.save(model, path)
        s = session(path, dest / f"smoke_m{m}_profile")
        start = time.perf_counter()
        y, actual_aq = s.run(None, {"a": a})
        elapsed = (time.perf_counter() - start) * 1000
        # CUDA Q divides FP16 input by its FP16 scale; rounding at FP8 bin
        # boundaries can differ from a Python FP32 division. Validate GEMM
        # against the actual quantized A, rather than conflating these effects.
        reference = (actual_aq * np.float32(sa16)) @ (fp8_array(b, sb).astype(np.float32) * sb).T
        p, counts = profile_counts(s)
        result = {"M": m, "K": 384, "N": 1152, "finite": bool(np.isfinite(y).all()),
                  "rmse_vs_dequantized_float32_reference": float(np.sqrt(np.mean((y.astype(np.float32)-reference)**2))),
                  "max_abs_vs_dequantized_float32_reference": float(np.max(np.abs(y-reference))),
                  "activation_bins_different_from_float32_division": int(np.count_nonzero(actual_aq != fp8_array(a, sa16).astype(np.float32))),
                  "first_run_ms": elapsed, "profile": str(p.relative_to(ROOT)), "placement": counts}
        result["relative_rmse_vs_quantized_reference"] = result["rmse_vs_dequantized_float32_reference"] / max(float(np.sqrt(np.mean(reference**2))), 1e-12)
        result["numerical_assertion"] = "allclose(actual, quantized-FP32-matmul reference, rtol=0.002, atol=0.002); relative RMSE <0.001"
        if (not result["finite"] or counts.get("CUDAExecutionProvider", {}).get("GemmFloat8", 0) != 1
                or result["relative_rmse_vs_quantized_reference"] >= 0.001
                or not np.allclose(y, reference, rtol=0.002, atol=0.002)):
            raise RuntimeError(f"Native FP8 smoke failed: {result}")
        results.append(result)
        print("SMOKE", result, flush=True)
        del s
        gc.collect()
    return results


def calibrate(model, targets, data_path, manifest_path, dest):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_sha = sha256(data_path)
    if manifest["array"]["sha256"] != input_sha:
        raise ValueError("Calibration array SHA256 does not match its manifest")
    if manifest["count"] != 64 or manifest["seed"] != 43:
        raise ValueError("Expected 64 training calibration images selected with seed43")
    if manifest["preprocessing"]["eval_utils_sha256"] != sha256(ROOT/"scripts/eval_utils.py"):
        raise ValueError("Shared preprocessing changed after calibration input generation")
    for key in ("official_nyu_test_index_overlap", "nyu_eval128_index_overlap", "official_nyu_test_scene_overlap", "exact_decoded_rgb_overlap_with_evaluation_images"):
        if manifest["separation_checks"][key] != 0:
            raise ValueError(f"Calibration/evaluation leakage check failed: {key}")
    augmented = copy.deepcopy(model)
    del augmented.graph.output[:]
    names = []
    for index, node in enumerate(targets):
        absolute, maximum = f"fp8_calib_abs_{index}", f"fp8_calib_amax_{index}"
        augmented.graph.node.extend([
            helper.make_node("Abs", [node.input[0]], [absolute], name=absolute),
            helper.make_node("ReduceMax", [absolute], [maximum], keepdims=0, name=maximum)])
        augmented.graph.output.append(helper.make_tensor_value_info(maximum, TensorProto.FLOAT16, []))
        names.append(maximum)
    inputs = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if inputs.dtype != np.float32 or inputs.shape != (64, 3, 518, 518):
        raise ValueError(f"Expected held-out calibration 64x3x518x518 float32, found {inputs.shape} {inputs.dtype}")
    s = session(augmented.SerializeToString())
    amax = np.zeros(len(targets), np.float32)
    for index in range(len(inputs)):
        values = np.asarray(s.run(names, {"image": np.asarray(inputs[index:index+1])}), dtype=np.float32).reshape(-1)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Nonfinite calibration activation {index}")
        amax = np.maximum(amax, values)
        print(f"CALIB {index+1}/{len(inputs)}", flush=True)
    del s
    gc.collect()
    payload = {"method": "Maximum absolute FP16 activation over 64 NYUv2 official train images; symmetric per-tensor E4M3",
               "input_path": str(data_path.relative_to(ROOT)), "input_sha256": input_sha,
               "manifest_path": str(manifest_path.relative_to(ROOT)), "manifest_sha256": sha256(manifest_path),
               "images": len(inputs), "layers": {n.name: {"input": n.input[0], "amax": float(v)} for n, v in zip(targets, amax)}}
    (dest / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return amax


def build(source, data_path, manifest_path, dest):
    original_source_hash = sha256(source)
    source_metadata = json.loads((ROOT/"models_gpu/metadata.json").read_text(encoding="utf-8"))
    if source_metadata["models"]["fp16"]["sha256"] != original_source_hash:
        raise ValueError("Source FP16 hash does not match the existing measured FP16 model metadata")
    model = onnx.load(source)
    original_opset = next(o.version for o in model.opset_import if o.domain == "")
    if original_opset < 21:
        model = onnx.version_converter.convert_version(model, 21)
    # Convert operator semantics with ONNX's version converter, never just
    # relabel an opset. Keep an explicit control and prove unchanged output.
    converted_control = dest / "fp16_opset21.onnx"
    onnx.save(model, converted_control)
    x = np.asarray(np.load(data_path, mmap_mode="r", allow_pickle=False)[:1])
    control_original = session(source)
    original_y = control_original.run(None, {"image": x})[0]
    del control_original
    gc.collect()
    control_converted = session(converted_control)
    converted_y = control_converted.run(None, {"image": x})[0]
    del control_converted
    gc.collect()
    conversion_max_abs = float(np.max(np.abs(original_y.astype(np.float32)-converted_y.astype(np.float32))))
    if not np.array_equal(original_y, converted_y):
        raise RuntimeError(f"ONNX opset conversion changed FP16 output: max abs {conversion_max_abs}")
    init = {i.name: i for i in model.graph.initializer}
    targets = [n for n in model.graph.node if n.op_type == "MatMul" and n.input[1] in init]
    if len(targets) != 48:
        raise ValueError(f"Expected 48 constant MatMul weights; got {len(targets)}")
    amax = calibrate(model, targets, data_path, manifest_path, dest)
    target_index = {n.name: index for index, n in enumerate(targets)}
    nodes, added_init, layer_info = [], [], []
    for node in model.graph.node:
        if node.name not in target_index:
            nodes.append(node)
            continue
        index = target_index[node.name]
        prefix = f"fp8_{index}"
        weight = numpy_helper.to_array(init[node.input[1]]).astype(np.float32)
        k, n = weight.shape
        scale_a16 = np.float16(max(float(amax[index]) / 448, np.finfo(np.float16).tiny))
        if float(scale_a16) * 448 < float(amax[index]):
            scale_a16 = np.nextafter(scale_a16, np.float16(np.inf), dtype=np.float16)
        scale_a32 = np.float32(scale_a16)
        scale_b = np.float32(max(float(np.max(np.abs(weight))) / 448, np.finfo(np.float32).tiny))
        added_init.extend([
            numpy_helper.from_array(np.asarray(scale_a16), prefix+"_sa16"),
            numpy_helper.from_array(np.asarray(scale_a32), prefix+"_sa32"),
            numpy_helper.from_array(np.asarray(scale_b), prefix+"_sb32"),
            numpy_helper.from_array(np.asarray(0, dtype=ml_dtypes.float8_e4m3fn), prefix+"_zero"),
            numpy_helper.from_array(fp8_array(weight.T.copy(), scale_b), prefix+"_weight"),
            numpy_helper.from_array(np.asarray([-1, k], np.int64), prefix+"_flat_shape"),
            numpy_helper.from_array(np.asarray([1, 1370, n], np.int64), prefix+"_restore_shape")])
        nodes.extend([
            helper.make_node("Reshape", [node.input[0], prefix+"_flat_shape"], [prefix+"_flat"], name=prefix+"_flatten"),
            helper.make_node("QuantizeLinear", [prefix+"_flat", prefix+"_sa16", prefix+"_zero"], [prefix+"_aq"], name=prefix+"_quantize"),
            helper.make_node("GemmFloat8", [prefix+"_aq", prefix+"_weight", "", prefix+"_sa32", prefix+"_sb32"], [prefix+"_y"],
                             domain="com.microsoft", name=prefix+"_native_gemm", dtype=TensorProto.FLOAT16, transA=0, transB=1, alpha=1., beta=0.),
            helper.make_node("Reshape", [prefix+"_y", prefix+"_restore_shape"], [node.output[0]], name=prefix+"_restore")])
        layer_info.append({"name": node.name, "weight_shape": list(weight.shape), "activation_amax": float(amax[index]),
                           "activation_scale": float(scale_a32), "weight_scale": float(scale_b)})
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # No target weights are shared with non-target nodes in this exported model.
    used = {i for n in nodes for i in n.input}
    original_init = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(original_init + added_init)
    if not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    for opset in model.opset_import:
        if opset.domain == "" and opset.version < 21:
            raise ValueError("Base model must be opset >=21; do not silently relabel opset")
    final_initializers = {i.name: i for i in model.graph.initializer}
    quantizers = {n.output[0]: n for n in nodes if n.op_type == "QuantizeLinear"}
    gemms = [n for n in nodes if n.op_type == "GemmFloat8"]
    if len(gemms) != 48 or not all(
            n.input[0] in quantizers
            and final_initializers[n.input[1]].data_type == F8
            and final_initializers[quantizers[n.input[0]].input[2]].data_type == F8 for n in gemms):
        raise RuntimeError("Native FP8 operand graph validation failed")
    onnx.checker.check_model(model)
    out = dest / "fp8_w8a8.onnx"
    onnx.save(model, out)
    s = session(out, dest / "model_smoke_profile")
    x = np.asarray(np.load(data_path, mmap_mode="r", allow_pickle=False)[:1])
    y = s.run(None, {"image": x})[0]
    profile_path, counts = profile_counts(s)
    if not np.isfinite(y).all() or counts.get("CUDAExecutionProvider", {}).get("GemmFloat8") != 48:
        raise RuntimeError(f"Full model FP8 smoke failed: {counts}")
    metadata = {"source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
                "source_opset": original_opset, "target_opset": 21, "opset_conversion": "onnx.version_converter.convert_version",
                "fp16_opset_conversion_control": str(converted_control.relative_to(ROOT)), "opset_conversion_control_max_abs": conversion_max_abs,
                "fp16_opset_conversion_control_sha256": sha256(converted_control),
                "model": str(out.relative_to(ROOT)), "sha256": sha256(out), "size_mib": out.stat().st_size/2**20,
                "format": "48 constant MatMuls: E4M3FN W8A8; other operations FP16; external IO FP32",
                "activation_quantization": "Static max-calibrated per-tensor, FP16-representable scale rounded upward to cover calibration amax",
                "weight_quantization": "Per-tensor max-abs /448, nearest-even E4M3FN; transpose stored weights for native GemmFloat8",
                "compute": "ORT CUDA GemmFloat8 -> cuBLASLt with FP8 A/B, FP32 compute type, FP8 fast accumulation enabled (FAST_ACCUM=1), FP16 output; not DQ+FP16 MatMul",
                "cublaslt_fast_accumulation": True,
                "source_graph_node_counts": dict(Counter(n.op_type for n in nodes)),
                "static_native_operand_check": "PASS: all 48 GemmFloat8 A inputs are QuantizeLinear outputs with E4M3FN zero points; B inputs are stored E4M3FN initializers",
                "native_claim_limit": "FP8 operand cuBLASLt path established by kernel source and CUDA profile; no instruction-level profiler used",
                "fixed_input": [1, 3, 518, 518], "transformer_sequence_length": 1370,
                "onnxruntime": ort.__version__, "provider_options": {"use_tf32": 0},
                "profile": str(profile_path.relative_to(ROOT)), "placement": counts, "finite_output": True,
                "layers": layer_info,
                "sources": ["https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/contrib_ops/cuda/math/gemm_float8.cu",
                            "https://github.com/microsoft/onnxruntime/blob/v1.31.0/docs/ContribOperators.md#com.microsoft.GemmFloat8"]}
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("BUILT", out, metadata["size_mib"], "MiB", counts, flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT/"models_gpu/fp16.onnx")
    parser.add_argument("--calibration-inputs", type=Path, default=ROOT/"data_fp8_calibration/inputs.npy")
    parser.add_argument("--calibration-manifest", type=Path, default=ROOT/"data_fp8_calibration/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"models_fp8")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ort.preload_dlls(directory="")
    smoke_results = smoke(args.output_dir)
    (args.output_dir / "gemm_smoke.json").write_text(json.dumps(smoke_results, indent=2), encoding="utf-8")
    if not args.smoke_only:
        build(args.source.resolve(), args.calibration_inputs.resolve(), args.calibration_manifest.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
