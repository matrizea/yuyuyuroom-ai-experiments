"""Dense GT metrics for affine-invariant inverse-depth predictions.

Metric formulas follow Depth Anything V2 metric_depth/util/metric.py.
Per-image least-squares alignment is an oracle relative-depth evaluation, not a
measurement of absolute distance accuracy without ground truth.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from eval_utils import ROOT, compare_depth, sample_bilinear

SOURCES = {
    "depth_anything_v2_relative_output": "https://arxiv.org/html/2406.09414v2#S5.SS2",
    "depth_metrics": "https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/metric_depth/util/metric.py",
    "resolution_and_image_averaging": "https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/metric_depth/train.py",
    "scale_shift_least_squares": "https://github.com/isl-org/ZoeDepth/blob/main/zoedepth/trainers/loss.py",
    "relative_evaluation_alignment": "https://openaccess.thecvf.com/content/CVPR2024/supplemental/Yang_Depth_Anything_Unleashing_CVPR_2024_supplemental.pdf",
}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT/path


def restore_prediction(prediction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Restore to original GT resolution using bilinear align_corners=True."""
    prediction = np.squeeze(np.asarray(prediction))
    if prediction.ndim != 2 or not np.isfinite(prediction).all():
        raise ValueError(f"Prediction must be a finite 2D array, got {prediction.shape}")
    if prediction.shape == tuple(shape):
        return prediction.astype(np.float64)
    ys = np.linspace(0, prediction.shape[0]-1, shape[0])[:,None]
    xs = np.linspace(0, prediction.shape[1]-1, shape[1])[None,:]
    return sample_bilinear(prediction, ys, xs)


def load_ground_truth(item: dict, manifest: dict, minimum: float, maximum: float) -> dict:
    primary_kind = item.get("gt_kind",manifest.get("gt_kind","inpainted"))
    variants = {primary_kind: (item["depth_path"], item.get("mask_path"))}
    if item.get("raw_depth_path"):
        variants["measured"] = (item["raw_depth_path"], item.get("raw_mask_path"))
    scale = float(item.get("depth_unit_scale", manifest.get("depth_unit_scale",1.0)))
    results = {}
    for name, (depth_path, mask_path) in variants.items():
        depth = np.load(resolve_path(depth_path), allow_pickle=False).astype(np.float64)*scale
        if depth.ndim != 2:
            raise ValueError(f"Expected 2D GT: {depth_path}, got {depth.shape}")
        mask = np.isfinite(depth) & (depth > minimum) & (depth < maximum)
        if mask_path:
            supplied = np.load(resolve_path(mask_path), allow_pickle=False)
            if supplied.shape != depth.shape:
                raise ValueError(f"GT mask shape mismatch: {mask_path}")
            mask &= supplied.astype(bool)
        if int(mask.sum()) < 10:
            raise ValueError(f"Fewer than 10 valid GT pixels: {depth_path}")
        results[name] = {"depth": depth, "mask": mask, "depth_path": depth_path,
                         "mask_path": mask_path}
    return results


def evaluate_prediction(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray,
                        maximum: float) -> tuple[dict, dict]:
    """Fit s*p+t to 1/z on the fixed GT mask, constraining s to be nonnegative.

    A nonpositive unconstrained solution is reported as a degenerate boundary
    solution s=0, preserving the physical near/far direction. No model-dependent
    pixels are dropped. Only the inversion step clips inverse depth at 1/max_depth.
    """
    p = prediction[mask].astype(np.float64)
    z = target[mask].astype(np.float64)
    inverse_gt = 1/z
    centered = p-p.mean()
    denominator = float(np.mean(centered*centered))
    unconstrained = float(np.mean(centered*(inverse_gt-inverse_gt.mean()))/denominator) if denominator > 0 else 0.0
    scale = max(unconstrained,0.0)
    shift = float(inverse_gt.mean()-scale*p.mean())
    aligned_inverse = scale*prediction+shift
    valid_aligned = aligned_inverse[mask]
    inverse_error = valid_aligned-inverse_gt
    inverse_rmse = float(np.sqrt(np.mean(inverse_error**2)))
    inverse_std = float(np.std(inverse_gt))
    floor = 1/maximum
    clipped_inverse = np.maximum(aligned_inverse,floor)
    aligned_depth = 1/clipped_inverse
    predicted = aligned_depth[mask]
    difference = predicted-z
    relative_error = np.abs(difference)/z
    metrics = {
        "valid_pixels": int(mask.sum()), "total_pixels": int(mask.size),
        "valid_pixel_ratio": float(mask.mean()),
        "abs_rel": float(relative_error.mean()),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "delta1": float(np.mean(np.maximum(z/predicted,predicted/z) < 1.25)),
        "inverse_nrmse": inverse_rmse/inverse_std if inverse_std > 0 else None,
        "inverse_rmse": inverse_rmse, "inverse_mae": float(np.abs(inverse_error).mean()),
        "alignment_scale": scale, "alignment_shift": shift,
        "unconstrained_scale": unconstrained,
        "scale_boundary_or_degenerate": bool(unconstrained <= 0 or denominator == 0),
        "clipped_inverse_depth_rate": float(np.mean(valid_aligned < floor)),
        "nonpositive_aligned_inverse_rate": float(np.mean(valid_aligned <= 0)),
        "inverse_depth_floor": floor,
    }
    maps = {"aligned_inverse": aligned_inverse, "aligned_depth": aligned_depth,
            "abs_rel": np.where(mask,np.abs(aligned_depth-target)/np.where(mask,target,1),np.nan)}
    return metrics,maps


def bootstrap_mean_delta(candidate: np.ndarray, reference: np.ndarray,
                         draws: int = 10000, seed: int = 42, metric: str = "abs_rel") -> dict:
    delta = np.asarray(candidate,dtype=np.float64)-np.asarray(reference,dtype=np.float64)
    if not len(delta):
        return {f"mean_delta_{metric}": None,f"ci95_delta_{metric}": [None,None],"images": 0}
    rng = np.random.default_rng(seed)
    sampled = []
    for first in range(0,draws,500):
        indices = rng.integers(0,len(delta),size=(min(500,draws-first),len(delta)))
        sampled.append(delta[indices].mean(axis=1))
    return {f"mean_delta_{metric}": float(delta.mean()),
            f"ci95_delta_{metric}": np.percentile(np.concatenate(sampled),[2.5,97.5]).tolist(),
            "images": len(delta),"draws": draws,"seed": seed,"unit": "image",
            "positive_is_worse": True}


def bootstrap_scene_mean_delta(candidate: np.ndarray, reference: np.ndarray, scenes: list[str],
                               draws: int = 10000, seed: int = 42, metric: str = "abs_rel") -> dict:
    """Resample entire scenes, retaining all their images and image weighting.

    For each bootstrap draw, sum the selected scenes' delta sums and divide by
    their selected image counts. This targets the same per-image mean difference
    as the image bootstrap while preserving within-scene frame dependence.
    """
    delta = np.asarray(candidate,dtype=np.float64)-np.asarray(reference,dtype=np.float64)
    if len(delta) != len(scenes):
        raise ValueError("Scene labels and paired metric arrays must have equal lengths")
    available = [str(scene) for scene in scenes if scene is not None and str(scene).strip() not in ["","unknown"]]
    result = {f"scene_cluster_ci95_delta_{metric}":[None,None],
              "scene_count":len(set(available)),"scene_cluster_draws":draws,
              "scene_cluster_seed":seed,"scene_cluster_unit":"scene/sequence",
              "scene_cluster_aggregation":"Resample whole scenes; sum selected scene delta sums / sum selected scene image counts"}
    if not len(delta) or len(available) != len(scenes):
        result["scene_cluster_note"] = "Unavailable because one or more images lack scene/sequence metadata"
        return result
    names,inverse = np.unique(np.asarray(scenes,dtype=str),return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse,weights=delta)
    rng = np.random.default_rng(seed)
    sampled = []
    for first in range(0,draws,500):
        selected = rng.integers(0,len(names),size=(min(500,draws-first),len(names)))
        sampled.append(sums[selected].sum(axis=1)/counts[selected].sum(axis=1))
    result[f"scene_cluster_ci95_delta_{metric}"] = np.percentile(np.concatenate(sampled),[2.5,97.5]).tolist()
    if len(names) < 2:
        result["scene_cluster_note"] = "Only one scene: interval is degenerate and cannot estimate between-scene uncertainty"
    return result


def aggregate_rows(rows: list[dict]) -> dict:
    metrics = ["abs_rel","rmse","delta1","inverse_nrmse","inverse_rmse",
               "inverse_mae","clipped_inverse_depth_rate","nonpositive_aligned_inverse_rate"]
    result = {"images": len(rows),"valid_pixels": sum(row["valid_pixels"] for row in rows)}
    for key in metrics:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = {"mean": float(np.mean(values)) if values else None,
                       "median": float(np.median(values)) if values else None,
                       "p90": float(np.percentile(values,90)) if values else None,
                       "max": float(np.max(values)) if values else None}
    if rows:
        counts = np.array([row["valid_pixels"] for row in rows],dtype=np.float64)
        result["pixel_weighted_diagnostics"] = {
            "abs_rel": float(np.average([row["abs_rel"] for row in rows],weights=counts)),
            "rmse": float(np.sqrt(np.average([row["rmse"]**2 for row in rows],weights=counts))),
            "delta1": float(np.average([row["delta1"] for row in rows],weights=counts)),
        }
    return result


def save_dense_preview(path: Path, image: np.ndarray, predictions: dict, metrics: dict,
                       target: np.ndarray, mask: np.ndarray, maps: dict, reference: str,
                       gt_kind: str, tile_w: int = 240, tile_h: int = 180) -> None:
    """Three rows: model outputs, GT errors, then differences from FP32."""
    names = list(predictions)
    label_h = 28
    canvas = np.full((3*(tile_h+label_h),(len(names)+1)*tile_w,3),245,np.uint8)

    def tile(array: np.ndarray,row: int,column: int,label: str) -> None:
        y,x = row*(tile_h+label_h),column*tile_w
        cv2.putText(canvas,label,(x+6,y+19),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,20),1,cv2.LINE_AA)
        ratio = min(tile_w/array.shape[1],tile_h/array.shape[0])
        resized = cv2.resize(array,(max(1,round(array.shape[1]*ratio)),max(1,round(array.shape[0]*ratio))))
        yy,xx = y+label_h+(tile_h-resized.shape[0])//2,x+(tile_w-resized.shape[1])//2
        canvas[yy:yy+resized.shape[0],xx:xx+resized.shape[1]] = resized

    tile(image,0,0,"Original (oracle GT alignment)")
    inverse_gt = np.zeros_like(target)
    inverse_gt[mask] = 1/target[mask]
    lo,hi = np.percentile(inverse_gt[mask],[1,99])
    normalized_gt = np.uint8(np.clip((inverse_gt-lo)/max(hi-lo,1e-12),0,1)*255)
    colored_gt = cv2.applyColorMap(normalized_gt,cv2.COLORMAP_INFERNO)
    colored_gt[~mask] = (125,125,125)
    tile(colored_gt,1,0,f"GT inverse depth: {gt_kind}")
    guide = np.full((tile_h,tile_w,3),245,np.uint8)
    for index,line in enumerate(["Row1: shared FP32 min/max", "Row2: GT relative error 0..50%", "Row3: aligned Q - FP32", "Blue/white/red: -10/0/+10%", "GT invalid pixels: gray"]):
        cv2.putText(guide,line,(6,20+index*29),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1,cv2.LINE_AA)
    tile(guide,2,0,"Common scales / GT-only mask")
    baseline = predictions[reference]
    lower,upper = float(baseline.min()),float(baseline.max())
    span = max(upper-lower,1e-12)
    for column,name in enumerate(names,1):
        prediction = predictions[name]
        image_depth = cv2.applyColorMap(np.uint8(np.clip((prediction-lower)/span,0,1)*255),cv2.COLORMAP_INFERNO)
        tile(image_depth,0,column,name+" raw inverse depth")
        gt_error = np.nan_to_num(maps[name]["abs_rel"],nan=0)
        error_color = cv2.applyColorMap(np.uint8(np.clip(gt_error/.5,0,1)*255),cv2.COLORMAP_INFERNO)
        error_color[~mask] = (125,125,125)
        tile(error_color,1,column,f"GT AbsRel={metrics[name]['abs_rel']:.4f}")
        aligned = compare_depth(baseline,prediction)
        difference = np.clip((prediction*aligned["alignment_scale"]+aligned["alignment_shift"]-baseline)/(span*.1),-1,1)
        color = np.empty((*difference.shape,3),np.uint8)
        color[...,0] = np.uint8((1-np.maximum(difference,0))*255)
        color[...,1] = np.uint8((1-np.abs(difference))*255)
        color[...,2] = np.uint8((1+np.minimum(difference,0))*255)
        tile(color,2,column,"Difference vs "+reference)
    success,encoded = cv2.imencode(".jpg",canvas,[cv2.IMWRITE_JPEG_QUALITY,90])
    if not success:
        raise IOError(f"Could not encode {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(encoded.tobytes())
