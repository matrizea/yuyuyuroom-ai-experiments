"""Create GPU comparison controls without changing the existing CPU models."""
import runtime_init
import hashlib
import json
from pathlib import Path
import logging
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.quantization.onnx_model import ONNXModel

ROOT = Path(__file__).resolve().parents[1]
dest = ROOT / "models_gpu"
dest.mkdir(exist_ok=True)
logging.getLogger().setLevel(logging.WARNING)
fp = ROOT / "models/fp32.onnx"
eight = dest / "int8_weight_only.onnx"
if not eight.exists():
    q = MatMulNBitsQuantizer(onnx.load(fp), bits=8, block_size=32,
                             is_symmetric=True, accuracy_level=1,
                             op_types_to_quantize=("MatMul",))
    q.process()
    q.model.save_model_to_file(str(eight), use_external_data_format=False)
for name, source in [("fp16", fp), ("int8_w8a16", eight),
                     ("int4_w4a16", ROOT / "models/int4_rtn.onnx")]:
    out = dest / (name + ".onnx")
    if not out.exists():
        model = convert_float_to_float16(onnx.load(source), keep_io_types=True,
                                         disable_shape_infer=False)
        ONNXModel(model).topological_sort()
        onnx.save(model, out)
    else:
        model = onnx.load(out)
        ONNXModel(model).topological_sort()
        onnx.save(model, out)
    onnx.checker.check_model(str(out))
    print(name, out.stat().st_size/2**20, "MiB", flush=True)
metadata = {"source_fp32_sha256": hashlib.sha256(fp.read_bytes()).hexdigest(),
            "io_type": "FP32; internal noninteger tensors converted to FP16 in *_a16 and fp16 models",
            "matmul_quantization": "48 constant MatMul weights; symmetric RTN, group32; all other weights FP16 in W8A16/W4A16",
            "activation_note": "A16 means stored floating activation precision. CUDA may dequantize weights for GEMM; this does not certify native integer Tensor Core execution.",
            "models": {p.stem:{"path": str(p.relative_to(ROOT)),"size_mib":p.stat().st_size/2**20,
                                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in sorted(dest.glob("*.onnx"))}}
(dest / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
