"""CUDA benchmarks plus operator placement evidence (CPU fallback is not hidden)."""
import runtime_init
import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]


def session(path, profile=False):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.enable_profiling = profile
    options.profile_file_prefix = str(ROOT / "results_gpu/profiles" / path.stem)
    return ort.InferenceSession(str(path), sess_options=options,
            providers=[("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"])


def stats(values):
    return {"median_ms":float(np.median(values)),"p90_ms":float(np.percentile(values,90)),
            "samples_ms": values}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args=parser.parse_args()
    (ROOT / "results_gpu/profiles").mkdir(parents=True, exist_ok=True)
    ort.preload_dlls(directory="")
    specs = {"fp32": ROOT/"models/fp32.onnx", "fp16": ROOT/"models_gpu/fp16.onnx",
             "int8_dynamic": ROOT/"models/int8_dynamic.onnx",
             "int4_rtn": ROOT/"models/int4_rtn.onnx",
             "int4_fast": ROOT/"models/int4_fast.onnx",
             "int8_w8a16": ROOT/"models_gpu/int8_w8a16.onnx",
             "int4_w4a16": ROOT/"models_gpu/int4_w4a16.onnx"}
    x=np.random.default_rng(42).normal(size=(1,3,518,518)).astype(np.float32)
    results={}
    for name,path in specs.items():
        print("Benchmark", name, flush=True)
        start=time.perf_counter()
        s=session(path)
        loading=time.perf_counter()-start
        if "CUDAExecutionProvider" not in s.get_providers():
            raise RuntimeError("CUDA provider failed to initialize; refusing a CPU-only GPU benchmark")
        start=time.perf_counter()
        y=s.run(None,{"image":x})[0]
        first=(time.perf_counter()-start)*1000
        assert np.isfinite(y).all()
        for _ in range(args.warmup):
            s.run(None,{"image":x})
        host=[]
        for _ in range(args.iterations):
            start=time.perf_counter();s.run(None,{"image":x});host.append((time.perf_counter()-start)*1000)
        device_x=ort.OrtValue.ortvalue_from_numpy(x,"cuda",0)
        io=s.io_binding();io.bind_ortvalue_input("image",device_x);io.bind_output("depth","cuda",0)
        for _ in range(args.warmup):
            s.run_with_iobinding(io);io.synchronize_outputs()
        device=[]
        for _ in range(args.iterations):
            start=time.perf_counter();s.run_with_iobinding(io);io.synchronize_outputs();device.append((time.perf_counter()-start)*1000)
        results[name]={"path":str(path.relative_to(ROOT)),"size_mib":path.stat().st_size/2**20,
                       "session_load_seconds":loading,"first_host_run_ms":first,
                       "host_input_output":stats(host),"device_resident_input_output":stats(device)}
        del io,device_x,s;gc.collect()
        # A separate profiled run avoids contaminating measured timings.
        s=session(path,profile=True);s.run(None,{"image":x});pf=Path(s.end_profiling())
        events=json.loads(pf.read_text())
        counts=defaultdict(Counter);duration=defaultdict(float)
        for event in events:
            info=event.get("args",{});provider=info.get("provider")
            if provider and event.get("name","").endswith("_kernel_time"):
                counts[provider][info.get("op_name","unknown")]+=1
                duration[provider]+=event.get("dur",0)
        results[name]["profile"]={"file":str(pf.relative_to(ROOT)),
                                   "node_executions_by_provider":{k:dict(v) for k,v in counts.items()},
                                   "summed_profile_kernel_us_by_provider":dict(duration),
                                   "note":"Single cold profile; summed event durations are placement evidence, not latency measurement."}
        del s;gc.collect()
        print(name, "host",results[name]["host_input_output"]["median_ms"],
              "resident", results[name]["device_resident_input_output"]["median_ms"],
              "placement",dict(counts),flush=True)
        payload={"onnxruntime":ort.__version__,"numpy":np.__version__,"provider_options":{"CUDAExecutionProvider":{"use_tf32":0}},
                 "input_shape":[1,3,518,518],"warmup":args.warmup,"iterations":args.iterations,
                 "host_timing":"session.run with H2D input / D2H output; excludes preprocessing/model loading",
                 "device_timing":"preloaded CUDA input and CUDA-bound output; run_with_iobinding + synchronize_outputs",
                 "models":results}
        (ROOT/"results_gpu/benchmark.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
