"""Prepare a deterministic NYUv2 test subset with filled and measured depth masks.

Downloads the original NYU labeled MATLAB/HDF5 file (about 2.97 GB), official
split, and calibration toolbox. Does not overwrite pre-existing artifacts.
Requires numpy, scipy, h5py, and Pillow.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import stat
import time
import urllib.request
import zipfile

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_dense"
URLS = {
    "nyu_depth_v2_labeled.mat": "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
    "splits.mat": "https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat",
    "toolbox_nyu_depth_v2.zip": "https://cs.nyu.edu/~fergus/datasets/toolbox_nyu_depth_v2.zip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(path: Path, url: str) -> dict:
    if path.exists():
        print(f"Using {path.name}: {path.stat().st_size:,} bytes", flush=True)
    else:
        part = path.with_suffix(path.suffix + ".part")
        offset = part.stat().st_size if part.exists() else 0
        if path.name == "nyu_depth_v2_labeled.mat":
            parallel_download(path, url, part, offset)
            return {"path": path.relative_to(ROOT).as_posix(), "url": url,
                    "bytes": path.stat().st_size, "sha256": sha256(path)}
        headers = {"User-Agent": "DA2-quantization-dense-evaluation/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            if offset and response.status != 206:
                raise IOError(f"Server does not support resuming {part}; refusing to overwrite partial file")
            expected = int(response.headers.get("Content-Length", "0")) + offset
            downloaded, last = offset, time.monotonic()
            with part.open("ab" if offset else "xb") as destination:
                while block := response.read(8*1024*1024):
                    destination.write(block)
                    downloaded += len(block)
                    if time.monotonic() - last > 10:
                        print(f"{path.name}: {downloaded:,} / {expected:,} bytes", flush=True)
                        last = time.monotonic()
            if expected and downloaded != expected:
                raise IOError(f"Incomplete download: {downloaded} != {expected}")
        if path.exists():
            raise FileExistsError(path)
        part.rename(path)
        print(f"Downloaded {path.name}: {downloaded:,} bytes", flush=True)
    return {"path": path.relative_to(ROOT).as_posix(), "url": url,
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def parallel_download(path: Path, url: str, part: Path, offset: int) -> None:
    """Resume a large original file using bounded, verified HTTP Range requests."""
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
        size = int(response.headers["Content-Length"])
    chunks = DATA / "nyu_download_chunks"
    chunks.mkdir(exist_ok=True)
    if not chunks.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Chunk directory must remain in data_dense")
    step = 8*1024*1024
    ranges = [(start, min(start+step, size)) for start in range(offset, size, step)]

    def fetch(bounds: tuple[int, int]) -> tuple[Path, int]:
        start, end = bounds
        target = chunks/f"{start:012d}_{end:012d}.bin"
        if target.exists():
            if target.stat().st_size != end-start:
                raise FileExistsError(f"Incomplete existing chunk: {target}")
            return target, end-start
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end-1}"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    content = response.read()
                    if response.status != 206 or len(content) != end-start:
                        raise IOError(f"Invalid range response for {start}:{end}")
                with target.open("xb") as output:
                    output.write(content)
                return target, len(content)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt+1)
        raise AssertionError("Unreachable")

    received, last = offset, time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(fetch, bounds) for bounds in ranges]):
            _, length = future.result()
            received += length
            if time.monotonic()-last > 10:
                print(f"{path.name}: {received:,} / {size:,} bytes", flush=True)
                last = time.monotonic()
    with part.open("ab" if offset else "xb") as output:
        for start,end in ranges:
            target = chunks/f"{start:012d}_{end:012d}.bin"
            output.write(target.read_bytes())
            # Delete only this verified, exact, script-owned chunk path.
            if not target.resolve().is_relative_to(DATA.resolve()):
                raise ValueError("Chunk path escaped workspace")
            target.unlink()
    if part.stat().st_size != size:
        raise IOError("Assembled NYU original size mismatch")
    if path.exists():
        raise FileExistsError(path)
    part.rename(path)
    print(f"Downloaded {path.name}: {size:,} bytes", flush=True)


def extract_toolbox(path: Path) -> None:
    destination = DATA / "toolbox"
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            name = PurePosixPath(entry.filename)
            target = destination.joinpath(*name.parts)
            if name.is_absolute() or ".." in name.parts or "\\" in entry.filename or ":" in entry.filename:
                raise ValueError(f"Unsafe ZIP path: {entry.filename}")
            if stat.S_ISLNK(entry.external_attr >> 16) or not target.resolve().is_relative_to(DATA.resolve()):
                raise ValueError(f"Unsafe ZIP member: {entry.filename}")
        for entry in archive.infolist():
            target = destination.joinpath(*PurePosixPath(entry.filename).parts)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                content = archive.read(entry)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.read_bytes() != content:
                        raise FileExistsError(f"Existing toolbox member differs: {target}")
                else:
                    with target.open("xb") as output:
                        output.write(content)


def write_json(path: Path, value: object) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf8") != content:
            raise FileExistsError(f"Existing JSON differs: {path}")
    else:
        with path.open("x", encoding="utf8", newline="\n") as output:
            output.write(content)


def save_npy(path: Path, value: np.ndarray) -> None:
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), value, equal_nan=True):
            raise FileExistsError(f"Existing array differs: {path}")
    else:
        with path.open("xb") as output:
            np.save(output, value, allow_pickle=False)


def matlab_text(file: h5py.File, key: str, index: int) -> str:
    references = file[key]
    reference = references[index, 0] if references.shape[0] > 1 else references[0, index]
    return "".join(chr(int(x)) for x in file[reference][:].ravel())


def prepare(sources: dict, count: int, seed: int) -> dict:
    split = loadmat(DATA/"splits.mat")
    test_indices = sorted(int(x)-1 for x in split["testNdxs"].ravel())
    train_indices = sorted(int(x)-1 for x in split["trainNdxs"].ravel())
    assert len(test_indices) == 654 and len(train_indices) == 795
    assert not set(test_indices) & set(train_indices)
    assert sorted(test_indices+train_indices) == list(range(1449))
    selected = sorted(random.Random(seed).sample(test_indices, count))
    output = DATA/f"nyu{count}"
    for name in ("rgb", "depth_inpainted_m", "depth_raw_m", "mask_inpainted", "mask_raw"):
        (output/name).mkdir(parents=True, exist_ok=True)
    rows = []
    with h5py.File(DATA/"nyu_depth_v2_labeled.mat", "r") as file:
        assert file["images"].shape == (1449, 3, 640, 480), file["images"].shape
        assert file["depths"].shape == file["rawDepths"].shape == (1449, 640, 480)
        for number,index in enumerate(selected, 1):
            identifier = f"nyu_{index+1:04d}"
            rgb = np.asarray(file["images"][index], dtype=np.uint8).transpose(2, 1, 0).copy()
            filled = np.asarray(file["depths"][index], dtype=np.float32).T.copy()
            raw = np.asarray(file["rawDepths"][index], dtype=np.float32).T.copy()
            filled_mask = np.isfinite(filled) & (filled > 0) & (filled < 10.0)
            raw_mask = np.isfinite(raw) & (raw > 0) & (raw < 10.0)
            if min(int(filled_mask.sum()), int(raw_mask.sum())) < 10:
                raise ValueError(f"Too few valid NYU ground-truth pixels: {identifier}")
            image_path = output/"rgb"/(identifier+".png")
            if image_path.exists():
                if not np.array_equal(np.asarray(Image.open(image_path).convert("RGB")), rgb):
                    raise FileExistsError(f"Existing RGB differs: {image_path}")
            else:
                with image_path.open("xb") as destination:
                    Image.fromarray(rgb).save(destination, format="PNG", compress_level=6)
            arrays = {"depth_path": ("depth_inpainted_m", filled),
                      "raw_depth_path": ("depth_raw_m", raw),
                      "mask_path": ("mask_inpainted", filled_mask),
                      "raw_mask_path": ("mask_raw", raw_mask)}
            paths = {}
            hashes = {"image_path": sha256(image_path)}
            for key,(directory,array) in arrays.items():
                path = output/directory/(identifier+".npy")
                save_npy(path, array)
                paths[key] = path.relative_to(ROOT).as_posix()
                hashes[key] = sha256(path)
            rows.append({"id": identifier, "original_matlab_index_1based": index+1,
                         "scene": matlab_text(file,"scenes",index),
                         "category": matlab_text(file,"sceneTypes",index),
                         "image_path": image_path.relative_to(ROOT).as_posix(), **paths,
                         "depth_unit_scale": 1.0, "height": 480, "width": 640,
                         "inpainted_valid_pixels": int(filled_mask.sum()),
                         "raw_valid_pixels": int(raw_mask.sum()), "sha256": hashes})
            if number%16 == 0 or number == count:
                print(f"Prepared NYUv2 {number}/{count}", flush=True)
    total_pixels = len(rows)*480*640
    manifest = {
        "dataset": "NYUv2 labeled official test subset", "image_count": count, "gt_kind": "inpainted",
        "original_split": "Author's official testNdxs (654 test, 795 train, 1-based in splits.mat)",
        "sampling": "Sorted 0-based test indices -> random.Random(seed).sample(count) -> sort selected indices",
        "seed": seed, "original_test_images": 654, "original_train_images": 795,
        "selected_indices_1based": [x+1 for x in selected],
        "path_base": "workspace", "depth_unit": "meters", "depth_unit_scale": 1.0,
        "depth_type": "depth_camera_z_projected_to_rgb",
        "depth_type_evidence": "Author toolbox depth_plane2depth_world.m sets Z=imgDepthAbs; project_depth_map.m projects these samples to RGB pixels and stores the original depth2 values, not Euclidean ray length.",
        "min_depth": 0.0, "max_depth": 10.0, "mask_rule": "isfinite(depth) & (depth > 0) & (depth < 10.0)",
        "max_depth_exclusive": True, "crop": "none (full 480x640 image)",
        "ground_truth_kinds": {
            "inpainted": "depths: dense depth after missing samples were filled using colorization; filled values are estimates, not independent sensor measurements.",
            "measured": "rawDepths: nonlinearity-corrected Kinect measurements projected to RGB coordinates before hole filling; invalid or saturated samples are excluded by mask.",
        },
        "total_image_pixels": total_pixels,
        "inpainted_valid_pixels": sum(x["inpainted_valid_pixels"] for x in rows),
        "raw_valid_pixels": sum(x["raw_valid_pixels"] for x in rows),
        "inpainted_valid_fraction": sum(x["inpainted_valid_pixels"] for x in rows)/total_pixels,
        "raw_valid_fraction": sum(x["raw_valid_pixels"] for x in rows)/total_pixels,
        "sources": sources,
        "documentation": ["https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html",
                          "https://cs.nyu.edu/~fergus/datasets/indoor_seg_support.pdf"],
        "license": {"status": "No explicit dataset license located in the reviewed author dataset page, paper, or toolbox README; do not label the original data MIT based on third-party mirrors.",
                    "citation": "Silberman, Hoiem, Kohli, Fergus. Indoor Segmentation and Support Inference from RGBD Images. ECCV 2012.",
                    "redistribution_permission_verified": False},
        "evaluation_notes": [
            "The relative inverse-depth model needs per-image scale and shift alignment against inverse metric GT; this is an oracle-aligned shape evaluation, not an absolute metric-depth prediction result.",
            "Use a common GT-only valid mask for all model variants and separately report inpainted versus measured depth.",
            "This is a fixed 128-image subset by default and no crop, so its scores are not directly comparable to full-654 cropped benchmark results.",
        ],
        "images": rows,
    }
    write_json(DATA/"manifest.json", manifest)
    summary = {key:value for key,value in manifest.items() if key != "images"}
    summary["manifest_sha256"] = sha256(DATA/"manifest.json")
    summary["prepared_bytes"] = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    write_json(DATA/"nyu_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.count <= 654:
        parser.error("count must be in [1,654]")
    DATA.mkdir(parents=True, exist_ok=True)
    if not DATA.resolve().is_relative_to(ROOT):
        raise ValueError("Data directory must remain within the workspace")
    sources = {name: download(DATA/name, url) for name,url in URLS.items()}
    extract_toolbox(DATA/"toolbox_nyu_depth_v2.zip")
    write_json(DATA/"sources.json", sources)
    if not args.download_only:
        result = prepare(sources, args.count, args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
