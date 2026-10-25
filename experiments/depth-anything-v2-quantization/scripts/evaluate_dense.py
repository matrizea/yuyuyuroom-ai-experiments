"""Evaluate relative inverse-depth models against per-pixel NYUv2 or Sintel GT.

Every image/model is aligned to GT using oracle inverse-depth scale and shift.
The resulting distance-space errors do not measure uncalibrated metric distance.

Examples:
  python scripts/evaluate_dense.py --provider CPU --output-dir results_dense/cpu
  python scripts/evaluate_dense.py --provider CUDA --output-dir results_dense/cuda
  python scripts/evaluate_dense.py --models fp32 --predictions torch_fp16=DIR
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

import runtime_init
import cv2
import numpy as np

from dense_utils import (SOURCES,aggregate_rows,bootstrap_mean_delta,bootstrap_scene_mean_delta,evaluate_prediction,
                         load_ground_truth,resolve_path,restore_prediction,save_dense_preview)
from evaluate import DEFAULT_MODELS,write_csv,write_json
from eval_utils import ROOT,preprocess,sha256


def make_dense_session(path: Path,args,name: str):
    import onnxruntime as ort
    if args.provider == "CUDA":
        ort.preload_dlls(directory="")
    selected = "CUDAExecutionProvider" if args.provider == "CUDA" else "CPUExecutionProvider"
    if selected not in ort.get_available_providers():
        raise RuntimeError(f"Requested {selected} is unavailable; available={ort.get_available_providers()}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    options.add_session_config_entry("session.intra_op.allow_spinning","0")
    options.add_session_config_entry("session.inter_op.allow_spinning","0")
    if args.provider == "CUDA":
        profile_dir = args.output_dir/"profiles"
        profile_dir.mkdir(parents=True,exist_ok=True)
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_dir/name)
        providers = [(selected,{"device_id":args.device_id,"use_tf32":0}),"CPUExecutionProvider"]
    else:
        providers = [selected]
    session = ort.InferenceSession(str(path),sess_options=options,providers=providers)
    if selected not in session.get_providers():
        raise RuntimeError(f"Requested {selected} failed to initialize; actual session providers={session.get_providers()}")
    return session


def runtime_prediction(session,tensor: np.ndarray) -> np.ndarray:
    input_info = session.get_inputs()[0]
    if input_info.type == "tensor(float16)":
        tensor = tensor.astype(np.float16)
    elif input_info.type != "tensor(float)":
        raise ValueError(f"Unsupported model input dtype: {input_info.type}")
    output = session.run([session.get_outputs()[0].name],{input_info.name:tensor})[0]
    output = np.squeeze(output).astype(np.float32)
    if output.ndim != 2 or not np.isfinite(output).all():
        raise ValueError(f"Model output is not a finite 2D depth map: {output.shape}")
    return output


def finish_profile(session,args) -> dict:
    path = Path(session.end_profiling())
    events = json.loads(path.read_text(encoding="utf-8"))
    counts,operators = {},{}
    for event in events:
        details = event.get("args",{})
        provider = details.get("provider")
        if event.get("cat") == "Node" and provider:
            counts[provider] = counts.get(provider,0)+1
            operation = details.get("op_name","unknown")
            operators.setdefault(provider,{})[operation] = operators.setdefault(provider,{}).get(operation,0)+1
    return {"profile":str(path.relative_to(args.output_dir)),"executed_node_events":counts,
            "executed_operators":operators,"note":"First evaluated image of this run only; CUDA sessions may place unsupported operations on CPU"}


def model_specs(args,images: list) -> dict:
    specs = {}
    for argument in args.models:
        name,separator,location = argument.partition("=")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",name) or name in specs:
            raise ValueError(f"Invalid or duplicate model name: {name!r}")
        path = Path(location).resolve() if separator else ROOT/"models"/(name+".onnx")
        if not path.is_file():
            raise FileNotFoundError(path)
        specs[name] = {"kind":"onnx","path":str(path),"sha256":sha256(path),
                       "file_size_mib":path.stat().st_size/1024**2}
    for argument in args.predictions:
        name,separator,location = argument.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",name) or name in specs:
            raise ValueError(f"Use a unique NAME=DIRECTORY for --predictions: {argument!r}")
        directory = Path(location).resolve()
        digest = hashlib.sha256()
        for item in images:
            prediction_path = directory/(str(item["id"])+".npy")
            digest.update((str(item["id"])+":"+sha256(prediction_path)+"\n").encode("utf-8"))
        specs[name] = {"kind":"predictions","path":str(directory),"prediction_set_sha256":digest.hexdigest()}
    if args.reference not in specs:
        raise ValueError(f"Reference {args.reference!r} must be included in --models or --predictions")
    return {args.reference:specs.pop(args.reference),**specs}


def summarize(checkpoint: dict,args) -> dict:
    entries = list(checkpoint["images"].values())
    rows = [row for item in entries for row in item["rows"]]
    kinds = {}
    for gt_kind in sorted(set(row["gt_kind"] for row in rows)):
        selected = [row for row in rows if row["gt_kind"] == gt_kind]
        reference = {row["id"]:row for row in selected if row["model"] == args.reference}
        models = {}
        for name,spec in checkpoint["config"]["models"].items():
            subset = [row for row in selected if row["model"] == name]
            paired = [row for row in subset if row["id"] in reference]
            inverse_paired = [row for row in paired if row["inverse_nrmse"] is not None
                              and reference[row["id"]]["inverse_nrmse"] is not None]
            category_stats,scene_stats = {},{}
            for category in sorted(set(row["category"] for row in subset)):
                category_stats[category] = aggregate_rows([row for row in subset if row["category"] == category])
            for scene in sorted(set(row["scene"] for row in subset)):
                scene_stats[scene] = aggregate_rows([row for row in subset if row["scene"] == scene])
            models[name] = {**spec,**aggregate_rows(subset),
                            "versus_reference":{
                                **bootstrap_mean_delta([row["abs_rel"] for row in paired],
                                                       [reference[row["id"]]["abs_rel"] for row in paired]),
                                **bootstrap_scene_mean_delta([row["abs_rel"] for row in paired],
                                                             [reference[row["id"]]["abs_rel"] for row in paired],
                                                             [row["scene"] for row in paired])},
                            "versus_reference_inverse_nrmse":{
                                **bootstrap_mean_delta([row["inverse_nrmse"] for row in inverse_paired],
                                                       [reference[row["id"]]["inverse_nrmse"] for row in inverse_paired],
                                                       metric="inverse_nrmse"),
                                **bootstrap_scene_mean_delta([row["inverse_nrmse"] for row in inverse_paired],
                                                             [reference[row["id"]]["inverse_nrmse"] for row in inverse_paired],
                                                             [row["scene"] for row in inverse_paired],metric="inverse_nrmse"),
                                "excluded_undefined_pairs":len(paired)-len(inverse_paired)},
                            "by_category":category_stats,"by_scene":scene_stats,
                            "scale_boundary_images":sum(row["scale_boundary_or_degenerate"] for row in subset)}
        gt_notes = {"inpainted":"Dense inpainted NYUv2 depth; filled pixels are not independent sensor measurements",
                    "measured":"Projected measured NYUv2 raw depth; only valid sensor pixels are scored",
                    "rendered_dense":"Dense synthetic rendered ground truth; original valid GT pixels are scored without hole filling"}
        kinds[gt_kind] = {"models":models,"ground_truth_note":gt_notes.get(gt_kind,"See dataset manifest for the ground-truth acquisition method")}
    previews = []
    if entries:
        primary = "inpainted" if "inpainted" in kinds else next(iter(kinds))
        primary_rows = [row for row in rows if row["gt_kind"] == primary]
        by_id = {row["id"]:row for row in primary_rows if row["model"] == args.reference}
        if by_id:
            middle = float(np.median([row["abs_rel"] for row in by_id.values()]))
            representative = min(by_id,key=lambda key:abs(by_id[key]["abs_rel"]-middle))
            entry = checkpoint["images"][representative]
            previews.append({"selection":"reference_median_absrel","id":representative,
                             "preview":entry.get("preview"),"reference_abs_rel":by_id[representative]["abs_rel"]})
            for name in checkpoint["config"]["models"]:
                if name == args.reference:
                    continue
                model_rows = [row for row in primary_rows if row["model"] == name]
                worst = max(model_rows,key=lambda row:row["abs_rel"]-by_id[row["id"]]["abs_rel"])
                entry = checkpoint["images"][worst["id"]]
                previews.append({"selection":"worst_absrel_increase","model":name,"id":worst["id"],
                                 "preview":entry.get("preview"),"delta_abs_rel":worst["abs_rel"]-by_id[worst["id"]]["abs_rel"]})
    summary = {"created_utc":datetime.now(timezone.utc).isoformat(),"config":checkpoint["config"],
               "dataset":checkpoint["dataset"],"environment":checkpoint["environment"],
               "expected_images":checkpoint["expected_images"],"completed_images":len(entries),
               "reference":args.reference,"ground_truth_variants":kinds,
               "provider_profiles":checkpoint.get("provider_profiles",{}),"selected_previews":previews,
               "method":{
                   "interpretation":"Oracle relative-depth evaluation: each prediction uses the evaluated image's GT to fit inverse-depth scale and shift; this is not absolute metric-distance accuracy",
                   "alignment":"Float64 least squares fit s*prediction+t to 1/GT_depth on all GT-valid pixels, s>=0; s=0 boundary is flagged when unconstrained fit is nonpositive or prediction is constant",
                   "prediction_restoration":"Original GT resolution, bilinear align_corners=True before alignment; no crop",
                   "preprocessing":"Fixed square518x518 OpenCV CUBIC without crop (aspect ratio can change), official RGB/ImageNet normalization",
                   "common_mask":"Only supplied GT mask AND finite GT AND min_depth<GT<max_depth; identical across models; no prediction-dependent pixel exclusion",
                   "inverse_depth_clamp":"Only for inversion into positive depth: max(aligned_inverse,1/max_depth), then reciprocal; pixels remain scored and clipped fraction is recorded; no near-distance prediction cap",
                   "inverse_error":"inverse_nrmse uses the unclipped aligned inverse prediction, RMSE divided by standard deviation of GT inverse depth on the GT-valid mask",
                   "aggregation":"Main metrics are arithmetic mean across per-image metrics; separately labeled pixel-weighted diagnostics are also provided",
                   "uncertainty":"Paired percentile bootstrap of AbsRel and inverse-NRMSE differences, 10000 resamples and seed42: both individual images and whole scene/sequence clusters. Scene resamples sum selected scene delta sums and divide by selected scene image counts, retaining the per-image mean estimand; undefined inverse-NRMSE pairs are counted and omitted only from that metric's CI",
                   "calibration":"Evaluation GT/images are not used to calibrate the quantized weights; GT is used only for the explicitly labeled per-image oracle alignment",
                   "timing":"No latency benchmark is performed by this script",
                   "sources":SOURCES}}
    write_csv(args.output_dir/"per_image.csv",rows)
    write_json(args.output_dir/"summary.json",summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest",type=Path,default=ROOT/"data_dense/manifest.json")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results_dense")
    parser.add_argument("--provider",choices=["CPU","CUDA"],default="CPU")
    parser.add_argument("--device-id",type=int,default=0)
    parser.add_argument("--models",nargs="*",default=DEFAULT_MODELS,help="Names or NAME=ONNX_PATH; use empty --models for predictions only")
    parser.add_argument("--predictions",nargs="*",default=[],help="NAME=DIRECTORY with one raw inverse-depth ID.npy per manifest image")
    parser.add_argument("--reference",default="fp32")
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--min-depth",type=float)
    parser.add_argument("--max-depth",type=float)
    parser.add_argument("--no-previews",action="store_true")
    parser.add_argument("--no-save-predictions",action="store_true",help="Do not cache native float32 model inverse-depth predictions")
    args = parser.parse_args()
    if args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("threads and limit must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = manifest["images"][:args.limit]
    if not images:
        parser.error("The manifest contains no selected images")
    ids = [str(item["id"]) for item in images]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",key) for key in ids):
        parser.error("Each image needs a unique filename-safe id")
    minimum = float(args.min_depth if args.min_depth is not None else manifest.get("min_depth",0.0))
    maximum = float(args.max_depth if args.max_depth is not None else manifest["max_depth"])
    if minimum < 0 or maximum <= minimum or not np.isfinite(maximum):
        parser.error("Require finite max_depth > min_depth >= 0")
    specs = model_specs(args,images)
    config = {"manifest_sha256":sha256(args.manifest),"limit":args.limit,"min_depth":minimum,
              "max_depth":maximum,"models":specs,"reference":args.reference,"provider":args.provider,
              "device_id":args.device_id,"threads":args.threads,"inter_op_threads":1,
              "cuda_use_tf32":False if args.provider == "CUDA" else None,
              "previews":not args.no_previews,"save_predictions":not args.no_save_predictions}
    checkpoint_path = args.output_dir/"checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["config"] != config:
            raise ValueError("Checkpoint configuration differs; choose a separate output directory")
        print(f"Resuming {len(checkpoint['images'])}/{len(images)} images",flush=True)
    else:
        import onnxruntime as ort
        checkpoint = {"config":config,"expected_images":len(images),"images":{},"provider_profiles":{},
                      "dataset":{key:value for key,value in manifest.items() if key != "images"},
                      "environment":{"python":sys.version,"platform":platform.platform(),
                                     "onnxruntime":ort.__version__,"numpy":np.__version__,"opencv":cv2.__version__,
                                     "available_providers":ort.get_available_providers()}}
        write_json(checkpoint_path,checkpoint)
    cv2.setNumThreads(1)
    sessions = {name:make_dense_session(Path(spec["path"]),args,name)
                for name,spec in specs.items() if spec["kind"] == "onnx"}
    profile_closed = set()
    for index,item in enumerate(images):
        key = str(item["id"])
        if key in checkpoint["images"]:
            continue
        image_path = resolve_path(item["image_path"])
        image = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Cannot decode {image_path}")
        tensor = preprocess(image,resize_mode="square")
        truths = load_ground_truth(item,manifest,minimum,maximum)
        if any(gt["depth"].shape != image.shape[:2] for gt in truths.values()):
            raise ValueError(f"Image and GT must describe the same uncropped image: {key}")
        predictions = {}
        for name,spec in specs.items():
            if spec["kind"] == "onnx":
                native = runtime_prediction(sessions[name],tensor)
                if args.provider == "CUDA" and name not in profile_closed:
                    checkpoint["provider_profiles"][name] = finish_profile(sessions[name],args)
                    profile_closed.add(name)
            else:
                native = np.load(Path(spec["path"])/(key+".npy"),allow_pickle=False)
            if not args.no_save_predictions:
                directory = args.output_dir/"predictions"/name
                directory.mkdir(parents=True,exist_ok=True)
                np.save(directory/(key+".npy"),native.astype(np.float32),allow_pickle=False)
            predictions[name] = restore_prediction(native,image.shape[:2])
        rows,all_metrics,all_maps = [],{},{}
        for gt_kind,truth in truths.items():
            gt_metrics,gt_maps = {},{}
            for name,prediction in predictions.items():
                metrics,maps = evaluate_prediction(prediction,truth["depth"],truth["mask"],maximum)
                gt_metrics[name],gt_maps[name] = metrics,maps
                rows.append({"id":key,"image":item["image_path"],"scene":str(item.get("scene","unknown")),
                             "category":str(item.get("category","unknown")),"gt_kind":gt_kind,"model":name,
                             **metrics})
            all_metrics[gt_kind],all_maps[gt_kind] = gt_metrics,gt_maps
        preview_path = None
        if not args.no_previews:
            primary = "inpainted" if "inpainted" in truths else next(iter(truths))
            preview_path = args.output_dir/"previews"/(key+".jpg")
            save_dense_preview(preview_path,image,predictions,all_metrics[primary],truths[primary]["depth"],
                               truths[primary]["mask"],all_maps[primary],args.reference,primary)
        checkpoint["images"][key] = {"rows":rows,"preview":str(preview_path.relative_to(args.output_dir)) if preview_path else None}
        write_json(checkpoint_path,checkpoint)
        if (index+1) % 8 == 0 or index+1 == len(images):
            print(f"Evaluated {index+1}/{len(images)} images; GT variants: {list(truths)}",flush=True)
            summarize(checkpoint,args)
    # Close any unused profiling sessions, such as a completed-run resume.
    for name,session in sessions.items():
        if args.provider == "CUDA" and name not in profile_closed:
            session.end_profiling()
    summary = summarize(checkpoint,args)
    for gt_kind,kind in summary["ground_truth_variants"].items():
        for name,result in kind["models"].items():
            print(f"{gt_kind}/{name}: AbsRel={result['abs_rel']['mean']:.6f}, RMSE={result['rmse']['mean']:.6f}, delta1={result['delta1']['mean']:.6f}",flush=True)
    print(f"Saved {args.output_dir/'summary.json'} (oracle GT alignment; not absolute-distance measurement)",flush=True)


if __name__ == "__main__":
    main()
