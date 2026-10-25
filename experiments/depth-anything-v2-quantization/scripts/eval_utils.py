"""Numerical and visualization helpers for the DA-2K quantization comparison."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess(image_bgr: np.ndarray, input_size: int = 518, resize_mode: str = "square") -> np.ndarray:
    """Use official normalization, with square input for the fixed-size ONNX export."""
    if not hasattr(preprocess, "transforms"):
        path = ROOT / "vendor/Depth-Anything-V2/depth_anything_v2/util/transform.py"
        specification = importlib.util.spec_from_file_location("official_da2_transform", path)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        preprocess.transforms = module
    module = preprocess.transforms
    sample = {"image": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) / 255.0}
    if resize_mode == "square":
        sample["image"] = cv2.resize(sample["image"], (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    elif resize_mode == "aspect":
        sample = module.Resize(
            input_size, input_size, resize_target=False, keep_aspect_ratio=True,
            ensure_multiple_of=14, resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        )(sample)
    else:
        raise ValueError(f"Unknown resize mode: {resize_mode}")
    sample = module.NormalizeImage(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(sample)
    return module.PrepareForNet()(sample)["image"][None]


def sample_bilinear(depth: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Sample a 2D array at source coordinates with border clamping."""
    ys = np.clip(np.asarray(ys, dtype=np.float64), 0, depth.shape[0] - 1)
    xs = np.clip(np.asarray(xs, dtype=np.float64), 0, depth.shape[1] - 1)
    y0, x0 = np.floor(ys).astype(np.intp), np.floor(xs).astype(np.intp)
    y1, x1 = np.minimum(y0 + 1, depth.shape[0] - 1), np.minimum(x0 + 1, depth.shape[1] - 1)
    wy, wx = ys - y0, xs - x0
    return ((1-wy) * ((1-wx)*depth[y0, x0] + wx*depth[y0, x1])
            + wy * ((1-wx)*depth[y1, x0] + wx*depth[y1, x1]))


def sample_original_points(depth: np.ndarray, points: list, original_shape: tuple) -> np.ndarray:
    """Equivalent to official bilinear align_corners=True upsampling then indexing."""
    points = np.asarray(points, dtype=np.float64)
    original_h, original_w = original_shape[:2]
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected row/column point pairs, got shape {points.shape}")
    if np.any(points < 0) or np.any(points[:, 0] >= original_h) or np.any(points[:, 1] >= original_w):
        raise ValueError("Annotation is outside the source image")
    ys = points[:, 0] * (depth.shape[0]-1) / max(original_h-1, 1)
    xs = points[:, 1] * (depth.shape[1]-1) / max(original_w-1, 1)
    return sample_bilinear(depth, ys, xs)


def grid_samples(depth: np.ndarray, side: int = 64) -> np.ndarray:
    ys, xs = np.meshgrid(np.linspace(0, depth.shape[0]-1, side),
                         np.linspace(0, depth.shape[1]-1, side), indexing="ij")
    return sample_bilinear(depth, ys, xs).ravel()


def compare_depth(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Compare native output maps; fit candidate scale/shift to the FP32 map."""
    if reference.shape != candidate.shape:
        raise ValueError(f"Output shapes differ: {reference.shape} vs {candidate.shape}")
    fp, q = reference.astype(np.float64), candidate.astype(np.float64)
    fp_mean, q_mean = float(fp.mean()), float(q.mean())
    fp_center, q_center = fp-fp_mean, q-q_mean
    q_variance = float(np.mean(q_center*q_center))
    scale = float(np.mean(q_center*fp_center) / q_variance) if q_variance else 0.0
    shift = fp_mean - scale*q_mean
    aligned_rmse = float(np.sqrt(np.mean((scale*q + shift-fp)**2)))
    fp_std = float(fp.std())
    fp_rank, q_rank = rankdata(grid_samples(reference)), rankdata(grid_samples(candidate))
    correlation = float(np.corrcoef(fp_rank, q_rank)[0, 1]) if fp_rank.std() and q_rank.std() else None
    return {
        "aligned_nrmse": aligned_rmse/fp_std if fp_std else None,
        "spearman_64x64": correlation,
        "raw_rmse": float(np.sqrt(np.mean((q-fp)**2))),
        "alignment_scale": scale, "alignment_shift": shift,
    }


def cluster_bootstrap_difference(candidate_counts: np.ndarray, reference_counts: np.ndarray,
                                 pair_counts: np.ndarray, draws: int = 10000, seed: int = 42) -> dict:
    """Resample whole images, keeping every annotation of a sampled image together."""
    candidate_counts = np.asarray(candidate_counts, dtype=np.float64)
    reference_counts = np.asarray(reference_counts, dtype=np.float64)
    pair_counts = np.asarray(pair_counts, dtype=np.float64)
    if not len(pair_counts):
        return {"difference_percentage_points": None, "ci95_percentage_points": [None, None]}
    rng = np.random.default_rng(seed)
    draws_result = []
    for start in range(0, draws, 500):
        selections = rng.integers(0, len(pair_counts), size=(min(500, draws-start), len(pair_counts)))
        differences = (candidate_counts[selections]-reference_counts[selections]).sum(axis=1)
        totals = pair_counts[selections].sum(axis=1)
        draws_result.append(100*differences/totals)
    distribution = np.concatenate(draws_result)
    interval = np.percentile(distribution, [2.5, 97.5]).tolist()
    return {"difference_percentage_points": float(100*(candidate_counts-reference_counts).sum()/pair_counts.sum()),
            "ci95_percentage_points": interval, "bootstrap_draws": draws, "bootstrap_seed": seed,
            "bootstrap_unit": "image"}


def save_preview(path: Path, image: np.ndarray, depths: dict[str, np.ndarray], metrics: dict,
                 annotations: list, tile_w: int = 240, tile_h: int = 200) -> None:
    """Save a compact montage with one common FP32 scale and aligned differences."""
    label_h = 28
    names = list(depths)
    columns = 1+len(names)
    canvas = np.full((2*(tile_h+label_h), columns*tile_w, 3), 245, np.uint8)

    def tile(value: np.ndarray, row: int, col: int, label: str) -> None:
        y, x = row*(tile_h+label_h), col*tile_w
        cv2.putText(canvas, label, (x+7, y+19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (25,25,25), 1, cv2.LINE_AA)
        ratio = min(tile_w/value.shape[1], tile_h/value.shape[0])
        resized = cv2.resize(value, (max(1, round(value.shape[1]*ratio)), max(1, round(value.shape[0]*ratio))))
        yy, xx = y+label_h+(tile_h-resized.shape[0])//2, x+(tile_w-resized.shape[1])//2
        canvas[yy:yy+resized.shape[0], xx:xx+resized.shape[1]] = resized

    annotated = image.copy()
    radius = max(4, round(max(image.shape[:2])/150))
    for index, annotation in enumerate(annotations):
        for key, color in [("point1", (0,230,0)), ("point2", (0,0,255))]:
            y, x = annotation[key]
            cv2.circle(annotated, (int(x),int(y)), radius, color, max(2,radius//2), cv2.LINE_AA)
    tile(annotated, 0, 0, "Original: green=near, red=far")
    fp = depths["fp32"]
    lower, upper = float(fp.min()), float(fp.max())
    span = max(upper-lower, 1e-12)
    legend = np.full((tile_h, tile_w, 3), 245, np.uint8)
    for y, line in enumerate(["Top: shared FP32 min/max", "Bottom: aligned Q - FP32", "Blue < 0; white = 0; red > 0", "Common limits: +/-10%", "of FP32 depth range"]):
        cv2.putText(legend, line, (7, 25+y*27), cv2.FONT_HERSHEY_SIMPLEX, .42, (25,25,25), 1, cv2.LINE_AA)
    tile(legend, 1, 0, "Comparison guide")
    for col, name in enumerate(names, 1):
        depth = depths[name]
        normalized = np.uint8(np.clip((depth-lower)/span, 0, 1)*255)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        tile(colored, 0, col, name + " depth (inverse)")
        aligned = depth*metrics[name]["alignment_scale"] + metrics[name]["alignment_shift"]
        difference = np.clip((aligned-fp)/(span*0.1), -1, 1)
        heatmap = np.empty((*difference.shape, 3), np.uint8)
        heatmap[..., 0] = np.uint8((1-np.maximum(difference, 0))*255)
        heatmap[..., 1] = np.uint8((1-np.abs(difference))*255)
        heatmap[..., 2] = np.uint8((1+np.minimum(difference, 0))*255)
        error = metrics[name]["aligned_nrmse"]
        label = f"Aligned diff; NRMSE={error:.4f}" if error is not None else "Aligned diff; NRMSE unavailable"
        tile(heatmap, 1, col, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        raise IOError(f"Could not encode preview: {path}")
    # OpenCV's Windows fopen path handling cannot write every Unicode filename.
    path.write_bytes(encoded.tobytes())
