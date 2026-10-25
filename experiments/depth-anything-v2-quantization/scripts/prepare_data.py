"""Download DA-2K safely and prepare deterministic, disjoint evaluation splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import stat
import time
import urllib.request
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://huggingface.co/datasets/depth-anything/DA-2K/resolve/main/DA-2K.zip"
EXPECTED_SHA256 = "ff0e48e7cc53273efd1312610e51f1ec87bea0b8a22daf37125fd81246592b81"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(path: Path) -> None:
    if path.exists():
        print(f"Using existing archive: {path}", flush=True)
        return
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        # A partial download is owned by this script. Do not overwrite it silently.
        raise FileExistsError(f"Remove or resume incomplete download first: {temporary}")
    request = urllib.request.Request(URL, headers={"User-Agent": "DA2K-quantization-evaluation/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("xb") as destination:
        total = int(response.headers.get("Content-Length", "0"))
        received, last_progress = 0, time.monotonic()
        while block := response.read(4 * 1024 * 1024):
            destination.write(block)
            received += len(block)
            if time.monotonic() - last_progress > 10:
                print(f"Downloaded {received:,} / {total:,} bytes", flush=True)
                last_progress = time.monotonic()
    # The final archive is created only when the stream is complete.
    if total and received != total:
        raise IOError(f"Incomplete download: {received} != {total}")
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)
    print(f"Downloaded {received:,} bytes", flush=True)


def checked_destination(name: str) -> Path:
    if "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive path: {name!r}")
    destination = DATA.joinpath(*relative.parts)
    if not destination.resolve().is_relative_to(DATA.resolve()):
        raise ValueError(f"Archive path escapes data directory: {name!r}")
    return destination


def crc32(path: Path) -> int:
    checksum = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xffffffff


def member_name(info: zipfile.ZipInfo) -> str:
    # DA-2K contains one Chinese filename stored as UTF-8 without the ZIP UTF-8
    # flag. Recover those original bytes; ordinary ASCII names are unchanged.
    if not info.flag_bits & 0x800:
        try:
            return info.filename.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return info.filename


def extract(archive: Path) -> list[dict]:
    members = []
    with zipfile.ZipFile(archive) as source:
        # Validate every path before creating any files.
        for info in source.infolist():
            checked_destination(info.filename)
            checked_destination(member_name(info))
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Symlink in archive: {info.filename}")
        for info in source.infolist():
            destination = checked_destination(member_name(info))
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            members.append({"path": info.filename, "bytes": info.file_size, "crc32": f"{info.CRC:08x}"})
            original_destination = checked_destination(info.filename)
            if original_destination != destination and original_destination.exists() and not destination.exists():
                # Repair an earlier extraction only after verifying both paths
                # stay in data/ and that the existing file matches this member.
                if original_destination.stat().st_size != info.file_size or crc32(original_destination) != info.CRC:
                    raise FileExistsError(f"Cannot repair differing file: {original_destination}")
                original_destination.rename(destination)
            if destination.exists():
                if destination.stat().st_size != info.file_size or crc32(destination) != info.CRC:
                    raise FileExistsError(f"Existing file differs from archive; refusing overwrite: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as src, destination.open("xb") as dst:
                while block := src.read(4 * 1024 * 1024):
                    dst.write(block)
    return members


def write_json(path: Path, value: object) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Existing generated file differs; refusing overwrite: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(content)


def make_splits(archive_hash: str, eval_per_category: int, calibration_per_category: int) -> dict:
    if min(eval_per_category, calibration_per_category) < 1:
        raise ValueError("Split sizes must be positive")
    annotation_path = DATA / "DA-2K" / "annotations.json"
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    by_category: dict[str, list[str]] = {}
    for relative, pairs in annotations.items():
        path = PurePosixPath(relative)
        if len(path.parts) != 3 or path.parts[0] != "images":
            raise ValueError(f"Unexpected annotation path: {relative}")
        image_path = checked_destination("DA-2K/" + relative)
        if not image_path.is_file() or not pairs:
            raise ValueError(f"Missing image or annotations: {relative}")
        for pair in pairs:
            if pair.get("closer_point") not in ("point1", "point2"):
                raise ValueError(f"Unexpected closer point: {relative}")
            for point_name in ("point1", "point2"):
                point = pair.get(point_name)
                if not isinstance(point, list) or len(point) != 2 or any(type(x) is not int or x < 0 for x in point):
                    raise ValueError(f"Invalid point coordinates: {relative}, {point}")
        by_category.setdefault(path.parts[1], []).append(relative)
    if len(by_category) != 8:
        raise ValueError(f"Expected eight categories, found {sorted(by_category)}")

    splits: dict[str, list[dict]] = {"evaluation": [], "calibration": []}
    population = {}
    for category, paths in sorted(by_category.items()):
        paths = sorted(paths)
        if len(paths) < eval_per_category + calibration_per_category:
            raise ValueError(f"Category {category} has too few images: {len(paths)}")
        population[category] = {"images": len(paths), "pairs": sum(len(annotations[path]) for path in paths)}
        # Resetting the RNG per category makes each category independent of category order.
        random.Random(42).shuffle(paths)
        selected = {"evaluation": paths[:eval_per_category],
                    "calibration": paths[eval_per_category:eval_per_category + calibration_per_category]}
        for split, selected_paths in selected.items():
            for relative in selected_paths:
                splits[split].append({"path": "data/DA-2K/" + relative,
                                      "category": category, "annotations": annotations[relative]})

    evaluation_paths = {image["path"] for image in splits["evaluation"]}
    calibration_paths = {image["path"] for image in splits["calibration"]}
    if evaluation_paths & calibration_paths:
        raise AssertionError("Evaluation and calibration overlap")
    common = {
        "dataset": "DA-2K", "source_url": URL, "archive_sha256": archive_hash,
        "annotations_path": "data/DA-2K/annotations.json", "annotations_sha256": sha256(annotation_path),
        "path_base": "workspace", "seed": 42,
        "sampling": "For each sorted category, shuffle sorted image paths using a fresh random.Random(42); evaluation first, calibration next.",
        "annotation_coordinates": "point1 and point2 are [row, column] in the original image; closer_point identifies the nearer point.",
        "zip_filename_decoding": "Unflagged UTF-8 member filenames are decoded by recovering CP437 bytes and decoding UTF-8 when valid. archive_manifest.json retains the ZIP library's original member names for audit.",
        "population": {"image_count": len(annotations), "pair_count": sum(map(len, annotations.values())), "categories": population},
    }
    summaries = {}
    for split, images in splits.items():
        category_counts = {category: {"images": sum(x["category"] == category for x in images),
                                     "pairs": sum(len(x["annotations"]) for x in images if x["category"] == category)}
                           for category in sorted(by_category)}
        summary = {"image_count": len(images), "pair_count": sum(len(x["annotations"]) for x in images),
                   "categories": category_counts}
        manifest = {**common, "split": split, **summary, "images": images}
        filename = "eval_manifest.json" if split == "evaluation" else "calibration_manifest.json"
        write_json(DATA / filename, manifest)
        summaries[split] = {"manifest": "data/" + filename, "manifest_sha256": sha256(DATA / filename), **summary}
    summary = {
        **common, "splits": summaries, "overlap_images": 0,
        "sampling_limitations": [
            "The evaluation is a fixed stratified subset, not the complete benchmark.",
            "Equal images per category differs from the full benchmark category proportions.",
            "Multiple point pairs from an image are correlated; use paired image-level bootstrap for confidence intervals.",
            "A few hundred images can reveal large degradation but cannot establish equivalence for small percentage-point changes.",
            "Calibration is disjoint from evaluation but drawn from the same public benchmark; it is not an independent deployment-domain validation.",
        ],
    }
    write_json(DATA / "dataset_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-only", action="store_true", help="Download/extract and record archive contents only")
    parser.add_argument("--eval-per-category", type=int, default=32)
    parser.add_argument("--calibration-per-category", type=int, default=8)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if not DATA.resolve().is_relative_to(ROOT):
        raise ValueError("Data directory must stay inside the workspace")
    archive = DATA / "DA-2K.zip"
    download(archive)
    archive_hash = sha256(archive)
    if archive_hash != EXPECTED_SHA256:
        raise ValueError(f"Dataset archive SHA256 mismatch: {archive_hash}")
    members = extract(archive)
    metadata = {"source_url": URL, "archive": str(archive.relative_to(ROOT)).replace("\\", "/"),
                "archive_bytes": archive.stat().st_size, "archive_sha256": archive_hash, "files": members}
    write_json(DATA / "archive_manifest.json", metadata)
    print(f"Archive SHA256: {archive_hash}; files: {len(members)}", flush=True)
    if not args.inspect_only:
        summary = make_splits(archive_hash, args.eval_per_category, args.calibration_per_category)
        print(json.dumps({"population": summary["population"], "splits": summary["splits"],
                          "overlap_images": summary["overlap_images"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
