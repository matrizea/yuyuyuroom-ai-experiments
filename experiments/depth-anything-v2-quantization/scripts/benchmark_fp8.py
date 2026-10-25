"""Fresh paired FP16/FP8 CUDA measurements, preserving the earlier GPU results.

Default: three balanced ABBA/BAAB rounds, 20 warmups and 100 timed calls per
block and I/O mode. A separate session records execution placement and the
optimized graph; profiling never runs inside the latency measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import runtime_init
import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source,"sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def stats(samples: list[float]) -> dict:
    return {"median_ms":float(np.median(samples)),"p90_ms":float(np.percentile(samples,90)),
            "mean_ms":float(np.mean(samples)),"min_ms":float(np.min(samples)),
            "max_ms":float(np.max(samples)),"sample_count":len(samples),"samples_ms":samples}


def schedule(rounds: int) -> list[dict]:
    result = []
    for number in range(1,rounds+1):
        order = ["fp16","fp8","fp8","fp16"] if number % 2 else ["fp8","fp16","fp16","fp8"]
        for position,name in enumerate(order,1):
            result.append({"round":number,"position":position,"model":name,
                           "mode_order":["host_input_output","device_resident_input_output"] if (number+position) % 2 == 0
                           else ["device_resident_input_output","host_input_output"]})
    return result


def graph_audit(path: Path) -> dict:
    """Read graph dtypes on CPU; no runtime session or GPU is created here."""
    model = onnx.load(path,load_external_data=True)
    type_name = lambda code: onnx.TensorProto.DataType.Name(code) if code else "UNDEFINED"
    types = {tensor.name:type_name(tensor.data_type) for tensor in model.graph.initializer}
    inventory, elements = Counter(),Counter()
    for tensor in model.graph.initializer:
        inventory[type_name(tensor.data_type)] += 1
        elements[type_name(tensor.data_type)] += int(np.prod(tensor.dims,dtype=np.int64))
    values = [*model.graph.input,*model.graph.value_info,*model.graph.output]
    for value in values:
        if value.type.HasField("tensor_type"):
            types[value.name] = type_name(value.type.tensor_type.elem_type)
    # Preserve useful FP8 edges even when the serialized graph lacks value_info.
    for node in model.graph.node:
        attributes = {attribute.name:attribute for attribute in node.attribute}
        dtype = None
        if node.op_type == "Cast" and "to" in attributes:
            dtype = type_name(attributes["to"].i)
        elif node.op_type == "Constant" and "value" in attributes:
            dtype = type_name(attributes["value"].t.data_type)
        elif node.op_type == "QuantizeLinear" and len(node.input) > 2:
            dtype = types.get(node.input[2])
        elif node.op_type == "DequantizeLinear" and len(node.input) > 1:
            dtype = types.get(node.input[1])
        if dtype and node.output:
            types[node.output[0]] = dtype
    fp8_nodes = []
    for node in model.graph.node:
        input_types = [types.get(name,"UNKNOWN") for name in node.input]
        output_types = [types.get(name,"UNKNOWN") for name in node.output]
        if "float8" in node.op_type.lower() or any("FLOAT8" in name for name in input_types+output_types):
            fp8_nodes.append({"name":node.name,"domain":node.domain,"op_type":node.op_type,
                              "input_dtypes":input_types,"output_dtypes":output_types})
    def io(value):
        tensor = value.type.tensor_type
        return {"name":value.name,"dtype":type_name(tensor.elem_type),
                "shape":[dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param
                         for dimension in tensor.shape.dim]}
    return {"path":relative(path),"sha256":sha256(path),"file_size_mib":path.stat().st_size/2**20,
            "opsets":{entry.domain or "ai.onnx":entry.version for entry in model.opset_import},
            "initializer_dtype_counts":dict(inventory),"initializer_elements_by_dtype":dict(elements),
            "known_tensor_dtype_counts":dict(Counter(types.values())),
            "operators":dict(Counter((node.domain or "ai.onnx")+"::"+node.op_type for node in model.graph.node)),
            "cast_destination_counts":dict(Counter(type_name(attribute.i) for node in model.graph.node
                                                    if node.op_type == "Cast" for attribute in node.attribute if attribute.name == "to")),
            "fp8_related_nodes":fp8_nodes,"inputs":[io(value) for value in model.graph.input],
            "outputs":[io(value) for value in model.graph.output]}


def profile_audit(path: Path, output_dir: Path) -> dict:
    events = json.loads(path.read_text(encoding="utf-8"))
    counts,durations = defaultdict(Counter),defaultdict(float)
    fp8_events = []
    for event in events:
        details = event.get("args",{})
        provider = details.get("provider")
        if not provider or not event.get("name","").endswith("_kernel_time"):
            continue
        operation = details.get("op_name","unknown")
        counts[provider][operation] += 1
        durations[provider] += event.get("dur",0)
        shapes = {key:details[key] for key in ["input_type_shape","output_type_shape"] if key in details}
        if "float8" in operation.lower() or "float8" in json.dumps(shapes).lower():
            fp8_events.append({"node_event":event.get("name"),"op_type":operation,"provider":provider,
                               "node_index":details.get("node_index"),**shapes})
    compute_events = [row for row in fp8_events
                      if "gemmfloat8" in row["op_type"].lower() or "matmulfloat8" in row["op_type"].lower()
                      or (row["op_type"] in ["Gemm","MatMul","FusedMatMul","QGemm"]
                          and "float8" in json.dumps(row.get("input_type_shape",[])).lower())]
    return {"file":path.relative_to(output_dir).as_posix(),
            "node_executions_by_provider":{name:dict(value) for name,value in counts.items()},
            "summed_profile_kernel_us_by_provider":dict(durations),"fp8_kernel_events":fp8_events,
            "fp8_compute_events":compute_events,
            "cpu_fallback_node_count":sum(counts.get("CPUExecutionProvider",{}).values()),
            "cuda_fp8_operator_execution_observed":any(row["provider"] == "CUDAExecutionProvider" for row in compute_events),
            "hardware_tensor_core_precision_verified_by_this_script":False,
            "note":"Single separate cold run; provider and op/dtype evidence, not a latency or hardware-instruction trace"}


def gpu_state(device_id: int) -> dict:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available":False}
    query = "name,driver_version,temperature.gpu,clocks.gr,clocks.mem,power.draw,power.limit,pstate,utilization.gpu,memory.used,memory.total"
    try:
        result = subprocess.run([executable,f"--id={device_id}",f"--query-gpu={query}","--format=csv,noheader,nounits"],
                                check=True,capture_output=True,text=True,timeout=10)
        return {"utc":utc(),"fields":query.split(","),"values":[value.strip() for value in result.stdout.strip().split(",")]}
    except (OSError,subprocess.SubprocessError) as error:
        return {"available":False,"error_type":type(error).__name__}


def make_session(path: Path,args,name: str,profile: bool = False):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    options.add_session_config_entry("session.intra_op.allow_spinning","0")
    options.add_session_config_entry("session.inter_op.allow_spinning","0")
    if profile:
        options.enable_profiling = True
        options.profile_file_prefix = str(args.output_dir/"profiles"/name)
        options.optimized_model_filepath = str(args.output_dir/"optimized"/(name+".onnx"))
    session = ort.InferenceSession(str(path),sess_options=options,
                                  providers=[("CUDAExecutionProvider",{"device_id":args.device_id,"use_tf32":0}),"CPUExecutionProvider"])
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDA failed to initialize; refusing a CPU-only GPU benchmark")
    return session


def model_input(session,tensor: np.ndarray) -> np.ndarray:
    dtype = session.get_inputs()[0].type
    if dtype == "tensor(float)":
        return tensor
    if dtype == "tensor(float16)":
        return tensor.astype(np.float16)
    raise ValueError(f"Unsupported external input dtype: {dtype}; comparison expects float/float16 I/O")


def measure_mode(session,tensor,args,mode: str) -> dict:
    import onnxruntime as ort
    input_name,output_name = session.get_inputs()[0].name,session.get_outputs()[0].name
    if mode == "host_input_output":
        def invoke():
            session.run([output_name],{input_name:tensor})
    else:
        device_input = ort.OrtValue.ortvalue_from_numpy(tensor,"cuda",args.device_id)
        binding = session.io_binding()
        binding.bind_ortvalue_input(input_name,device_input)
        binding.bind_output(output_name,"cuda",args.device_id)
        def invoke():
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
    for _ in range(args.warmup):
        invoke()
    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        invoke()
        samples.append((time.perf_counter()-start)*1000)
    return stats(samples)


def aggregate(payload: dict) -> None:
    for name,result in payload["models"].items():
        blocks = [block for block in payload["blocks"] if block["model"] == name]
        for mode in ["host_input_output","device_resident_input_output"]:
            samples = [sample for block in blocks for sample in block[mode]["samples_ms"]]
            if samples:
                result[mode] = {**stats(samples),"block_median_ms":[block[mode]["median_ms"] for block in blocks]}
    paired = []
    for number in range(1,payload["config"]["rounds"]+1):
        blocks = [block for block in payload["blocks"] if block["round"] == number]
        if len(blocks) != 4:
            continue
        record = {"round":number}
        for mode in ["host_input_output","device_resident_input_output"]:
            medians = {name:float(np.median([sample for block in blocks if block["model"] == name
                                            for sample in block[mode]["samples_ms"]])) for name in ["fp16","fp8"]}
            record[mode] = {"fp16_median_ms":medians["fp16"],"fp8_median_ms":medians["fp8"],
                            "fp8_over_fp16_latency_ratio":medians["fp8"]/medians["fp16"],
                            "fp8_speedup_over_fp16":medians["fp16"]/medians["fp8"]}
        paired.append(record)
    payload["paired_rounds"] = paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16-model",type=Path,default=ROOT/"models_fp8/fp16_opset21.onnx")
    parser.add_argument("--fp8-model",type=Path,default=ROOT/"models_fp8/fp8_w8a8.onnx")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results_fp8/benchmark")
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--warmup",type=int,default=20)
    parser.add_argument("--iterations",type=int,default=100)
    parser.add_argument("--device-id",type=int,default=0)
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("rounds and iterations must be positive; warmup must be nonnegative")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    destination = args.output_dir/"benchmark.json"
    if destination.exists():
        raise FileExistsError(f"Preserve prior measurements; choose another --output-dir: {relative(destination)}")
    for name in ["profiles","optimized"]:
        (args.output_dir/name).mkdir(exist_ok=True)
    paths = {"fp16":args.fp16_model.resolve(),"fp8":args.fp8_model.resolve()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(relative(path))
    payload = {"status":"initializing","created_utc":utc(),
               "config":{"rounds":args.rounds,"warmup_per_block_per_mode":args.warmup,
                         "iterations_per_block_per_mode":args.iterations,"intra_op_threads":8,"inter_op_threads":1,
                         "execution_mode":"sequential","allow_spinning":False,"input_shape":[1,3,518,518],
                         "random_seed":42,"device_id":args.device_id,"use_tf32":False,
                         "schedule":schedule(args.rounds),"prior_results_are_separate":"results_gpu/benchmark.json"},
               "environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"onnx":onnx.__version__},
               "method":{"host_timing":"session.run with CPU input and CPU output, H2D/D2H included",
                         "device_timing":"Preloaded CUDA input, CUDA-bound output, run_with_iobinding plus synchronize_outputs",
                         "excluded":"Model loading, first call, input generation/copy to CUDA, profiling, disk I/O, GPU telemetry query",
                         "aggregation":"Pool equal-sized blocks per model; report per-block and per-round medians as well",
                         "FP8_claim_limit":"Graph FP8 types and a CUDA GemmFloat8 event demonstrate execution of that runtime operator; this script does not trace native Tensor Core instructions"},
               "models":{},"blocks":[]}
    write_json(destination,payload)
    stage,name = "graph_audit",None
    try:
        for name,path in paths.items():
            payload["models"][name] = {"source_graph":graph_audit(path)}
        if not payload["models"]["fp8"]["source_graph"]["fp8_related_nodes"]:
            raise ValueError("The candidate graph has no FP8-related node/type evidence")
        import onnxruntime as ort
        ort.preload_dlls(directory="")
        payload["environment"].update({"onnxruntime":ort.__version__,"available_providers":ort.get_available_providers()})
        payload["initial_gpu_state"] = gpu_state(args.device_id)
        tensor = np.random.default_rng(42).normal(size=(1,3,518,518)).astype(np.float32)
        payload["config"]["input_sha256"] = hashlib.sha256(tensor.tobytes()).hexdigest()
        stage = "profile_and_optimized_graph"
        for name,path in paths.items():
            print(f"Auditing {name} placement and optimized graph",flush=True)
            session = make_session(path,args,name,profile=True)
            actual = model_input(session,tensor)
            output = session.run([session.get_outputs()[0].name],{session.get_inputs()[0].name:actual})[0]
            if not np.isfinite(output).all():
                raise ValueError(f"Nonfinite {name} smoke output")
            profile_path = Path(session.end_profiling())
            payload["models"][name].update({"profile":profile_audit(profile_path,args.output_dir),
                                            "optimized_graph":graph_audit(args.output_dir/"optimized"/(name+".onnx")),
                                            "providers":session.get_providers(),"provider_options":session.get_provider_options(),
                                            "smoke_output_shape":list(output.shape)})
            del session,output
            gc.collect()
            write_json(destination,payload)
        fp8_profile = payload["models"]["fp8"]["profile"]
        if not fp8_profile["cuda_fp8_operator_execution_observed"]:
            raise RuntimeError("No CUDA FP8 operator/type event was observed; refusing to label a fallback simulation as a native FP8 benchmark")
        stage = "paired_benchmark"
        sessions = {name:make_session(path,args,name) for name,path in paths.items()}
        inputs = {name:model_input(session,tensor) for name,session in sessions.items()}
        payload["status"] = "benchmark_running"
        for planned in schedule(args.rounds):
            name = planned["model"]
            block = {**planned,"started_utc":utc(),"gpu_before":gpu_state(args.device_id)}
            print(f"Round {planned['round']}/{args.rounds} block {planned['position']}/4: {name}",flush=True)
            for mode in planned["mode_order"]:
                block[mode] = measure_mode(sessions[name],inputs[name],args,mode)
            block["ended_utc"],block["gpu_after"] = utc(),gpu_state(args.device_id)
            payload["blocks"].append(block)
            aggregate(payload)
            write_json(destination,payload)
        del sessions,inputs
        gc.collect()
        payload.update({"status":"complete","completed_utc":utc(),"final_gpu_state":gpu_state(args.device_id)})
        write_json(destination,payload)
        for name,result in payload["models"].items():
            print(f"{name}: host {result['host_input_output']['median_ms']:.4f} ms; "
                  f"device {result['device_resident_input_output']['median_ms']:.4f} ms; "
                  f"CPU fallback nodes {result['profile']['cpu_fallback_node_count']}",flush=True)
        print(f"Saved {relative(destination)}",flush=True)
    except Exception as error:
        payload.update({"status":"unsupported_or_failed","failure":{"stage":stage,"model":name,
                        "type":type(error).__name__,"message":str(error).replace(str(ROOT),".")}})
        write_json(destination,payload)
        raise


if __name__ == "__main__":
    main()
