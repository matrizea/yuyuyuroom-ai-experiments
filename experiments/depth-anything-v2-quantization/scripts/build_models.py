"""Export the official ViT-S checkpoint and quantize transformer MatMul weights."""
from __future__ import annotations

import runtime_init  # noqa: F401 -- before the native ML libraries
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import urllib.request

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    target = ROOT / "models"
    target.mkdir(exist_ok=True)
    weights = target / "depth_anything_v2_vits.pth"
    url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
    if not weights.exists():
        print("Downloading official checkpoint", flush=True)
        urllib.request.urlretrieve(url, weights)
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    x = torch.randn(1, 3, 518, 518)
    fp = target / "fp32.onnx"
    if not fp.exists() or args.force:
        print("Exporting unmodified official forward, fixed 518x518", flush=True)
        with torch.inference_mode():
            torch.onnx.export(model, x, str(fp), input_names=["image"], output_names=["depth"],
                              opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(str(fp))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(fp), sess_options=options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        reference = model(x).numpy()
    actual = sess.run(None, {"image": x.numpy()})[0]
    export_error = {"max_abs": float(np.max(np.abs(actual-reference))),
                    "rmse": float(np.sqrt(np.mean((actual-reference)**2))),
                    "relative_rmse_std": float(np.sqrt(np.mean((actual-reference)**2))/np.std(reference))}
    if export_error["relative_rmse_std"] > 0.001:
        raise RuntimeError(f"Export verification failed: {export_error}")
    del sess
    p8 = target / "int8_dynamic.onnx"
    if not p8.exists() or args.force:
        print("Quantizing INT8 dynamic, per-channel signed weights, MatMul only", flush=True)
        quantize_dynamic(str(fp), str(p8), op_types_to_quantize=["MatMul"], per_channel=True,
                         weight_type=QuantType.QInt8, extra_options={"MatMulConstBOnly": True})
    for name, level in [("int4_rtn", 1), ("int4_fast", 4)]:
        dest = target / f"{name}.onnx"
        if not dest.exists() or args.force:
            print(f"Quantizing {name}: symmetric 4bit, block32, accuracy_level={level}", flush=True)
            quant = MatMulNBitsQuantizer(onnx.load(str(fp)), bits=4, block_size=32,
                                         is_symmetric=True, accuracy_level=level,
                                         op_types_to_quantize=("MatMul",))
            quant.process()
            quant.model.save_model_to_file(str(dest), use_external_data_format=False)
    original = onnx.load(str(fp))
    init = {i.name: i for i in original.graph.initializer}
    eligible = [n for n in original.graph.node if n.op_type == "MatMul" and n.input[1] in init]
    total_params = sum(int(np.prod(i.dims)) for i in original.graph.initializer if i.data_type == onnx.TensorProto.FLOAT)
    qparams = sum(int(np.prod(init[n.input[1]].dims)) for n in eligible)
    details = {}
    for name in ["fp32", "int8_dynamic", "int4_rtn", "int4_fast"]:
        p = target / f"{name}.onnx"
        m = onnx.load(str(p))
        onnx.checker.check_model(m)
        details[name] = {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                         "sha256": sha256(p), "operators": dict(Counter(n.op_type for n in m.graph.node))}
        s = ort.InferenceSession(str(p), sess_options=options, providers=["CPUExecutionProvider"])
        y = s.run(None, {"image": x.numpy()})[0]
        assert y.shape == (1, 518, 518) and np.isfinite(y).all()
        print(name, f"{p.stat().st_size/2**20:.2f} MiB", "range", float(y.min()), float(y.max()), flush=True)
        del s
    meta = {"checkpoint_url": url, "checkpoint_sha256": sha256(weights),
            "upstream_commit": subprocess.check_output(["git", "-C", str(ROOT / "vendor/Depth-Anything-V2"), "rev-parse", "HEAD"], text=True).strip(),
            "input_shape": [1, 3, 518, 518], "export_validation_random_seed42": export_error,
            "total_float_initializer_elements": total_params,
            "quantized_weight_elements": qparams, "quantized_weight_fraction": qparams/total_params,
            "quantized_matmul_count": len(eligible),
            "quantization": {"int8_dynamic": "MatMul constant weights: signed INT8 per output channel, dynamic U8 activation. Other ops FP32.",
                             "int4_rtn": "MatMul constant weights: symmetric INT4 block32, RTN, accuracy_level=1. Activations FP32; other ops FP32.",
                             "int4_fast": "Same INT4 weights, accuracy_level=4 permits internal INT8 activation approximation."},
            "models": details}
    (target / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"export_validation": export_error, "quantized_fraction": qparams/total_params, "quantized_matmuls": len(eligible)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
