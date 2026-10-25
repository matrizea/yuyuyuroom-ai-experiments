"""Prepare rendered dense Sintel GT by fetching selected ZIP members with HTTP Range.

The official 1.60GB depth and 1.78GB RGB archives are never downloaded in full.
Records original ZIP central-directory hashes and each extracted member hash/CRC.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import struct
import time
import urllib.request
import zipfile
import zlib

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data_dense"/"sintel"
URLS = {
    "depth": "https://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip",
    "rgb": "https://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def request_range(url: str, start: int, end: int) -> bytes:
    for attempt in range(4):
        try:
            request = urllib.request.Request(url,headers={"Range":f"bytes={start}-{end-1}"})
            with urllib.request.urlopen(request,timeout=60) as response:
                value = response.read()
                if response.status != 206 or len(value) != end-start:
                    raise IOError(f"Invalid HTTP Range response {response.status}, {len(value)}, expected {end-start}")
            return value
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt+1)
    raise AssertionError("Unreachable")


class RemoteZipFile(io.RawIOBase):
    def __init__(self, url: str):
        super().__init__()
        self.url, self.position = url, 0
        with urllib.request.urlopen(urllib.request.Request(url,headers={"Range":"bytes=-65536"}),timeout=60) as response:
            tail = response.read()
            if response.status != 206:
                raise IOError("HTTP Range required")
            self.size = int(response.headers["Content-Range"].split("/")[-1])
        self.cache = {(self.size-len(tail),self.size):tail}
        self.transferred = len(tail)

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        return self.position

    def read(self, length=-1):
        end = self.size if length < 0 else min(self.size,self.position+length)
        if end <= self.position:
            return b""
        for (start,stop),value in self.cache.items():
            if start <= self.position and end <= stop:
                result = value[self.position-start:end-start]
                self.position = end
                return result
        start = self.position
        result = request_range(self.url,start,end)
        self.transferred += len(result)
        self.cache[(start,end)] = result
        self.position = end
        return result


def destination(name: str) -> Path:
    relative = PurePosixPath(name)
    target = DATA.joinpath(*relative.parts)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe ZIP path: {name}")
    if not target.resolve().is_relative_to(DATA.resolve()):
        raise ValueError(f"ZIP path escaped workspace: {name}")
    return target


def fetch_member(url: str, size: int, info: zipfile.ZipInfo) -> dict:
    target = destination(info.filename)
    target.parent.mkdir(parents=True,exist_ok=True)
    transferred = 0
    if target.exists():
        content = target.read_bytes()
    else:
        # A member's local header can be read together with compressed payload.
        # ZIP64 is unnecessary for selected small files; central offsets are 64-bit safe.
        end = min(size,info.header_offset+info.compress_size+4096)
        value = request_range(url,info.header_offset,end)
        transferred += len(value)
        fields = struct.unpack("<4s5H3I2H",value[:30])
        if fields[0] != b"PK\x03\x04" or fields[2] & 1:
            raise ValueError(f"Unsupported local ZIP header: {info.filename}")
        data_offset = 30+fields[-2]+fields[-1]
        if data_offset+info.compress_size > len(value):
            value = request_range(url,info.header_offset,info.header_offset+data_offset+info.compress_size)
            transferred += len(value)
        compressed = value[data_offset:data_offset+info.compress_size]
        if info.compress_type == zipfile.ZIP_STORED:
            content = compressed
        elif info.compress_type == zipfile.ZIP_DEFLATED:
            content = zlib.decompress(compressed,-15)
        else:
            raise ValueError(f"Unsupported ZIP compression: {info.compress_type}")
        if len(content) != info.file_size or zlib.crc32(content)&0xffffffff != info.CRC:
            raise IOError(f"ZIP CRC or size mismatch: {info.filename}")
        with target.open("xb") as output:
            output.write(content)
    if len(content) != info.file_size or zlib.crc32(content)&0xffffffff != info.CRC:
        raise FileExistsError(f"Existing member differs: {target}")
    return {"member":info.filename,"bytes":info.file_size,"crc32":f"{info.CRC:08x}",
            "sha256":hashlib.sha256(content).hexdigest(),"range_bytes_transferred":transferred,
            "path":target.relative_to(ROOT).as_posix()}


def write_json(path: Path, value: object) -> None:
    content = json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if path.exists():
        if path.read_text(encoding="utf8") != content:
            raise FileExistsError(f"Generated JSON differs: {path}")
    else:
        with path.open("x",encoding="utf8",newline="\n") as output:
            output.write(content)


def preserve_previous_protocol(path: Path, value: dict) -> None:
    """Keep previous cap metadata before an authorized protocol change."""
    if path.exists():
        previous=json.loads(path.read_text(encoding="utf8"))
        if previous.get("max_depth") != value.get("max_depth"):
            backup=path.with_stem(path.stem+f"_{previous['max_depth']:g}m")
            write_json(backup,previous)
            if not path.resolve().is_relative_to((ROOT/"data_dense").resolve()):
                raise ValueError("Metadata path escaped data_dense")
            path.unlink()
    write_json(path,value)


def save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not np.array_equal(np.load(path,allow_pickle=False),value,equal_nan=True):
            raise FileExistsError(f"Existing array differs: {path}")
    else:
        with path.open("xb") as output:
            np.save(output,value,allow_pickle=False)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count",type=int,default=64)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--max-depth",type=float,default=10000.0)
    parser.add_argument("--download-only",action="store_true")
    args=parser.parse_args()
    if not math.isfinite(args.max_depth) or args.max_depth<=0:
        parser.error("max-depth must be finite and positive")
    DATA.mkdir(parents=True,exist_ok=True)
    if not DATA.resolve().is_relative_to(ROOT):
        raise ValueError("Dataset directory escaped workspace")
    remote={key:RemoteZipFile(url) for key,url in URLS.items()}
    archives={key:zipfile.ZipFile(value) for key,value in remote.items()}
    infos={key:{item.filename:item for item in value.infolist()} for key,value in archives.items()}
    by_sequence={}
    for name in sorted(infos["depth"]):
        path=PurePosixPath(name)
        if len(path.parts)==4 and path.parts[:2]==("training","depth") and path.suffix==".dpt":
            by_sequence.setdefault(path.parts[2],[]).append(path.stem)
    sequences=sorted(by_sequence)
    if not len(sequences)<=args.count<=sum(map(len,by_sequence.values())):
        parser.error("count must include at least one frame per sequence and cannot exceed the population")
    base,extra=divmod(args.count,len(sequences))
    extra_sequences=set(random.Random(args.seed).sample(sequences,extra))
    selected=[]
    for sequence in sequences:
        frames=random.Random(args.seed).sample(sorted(by_sequence[sequence]),base+(sequence in extra_sequences))
        selected.extend((sequence,frame) for frame in sorted(frames))
    wanted={"depth":["README_depth.txt","sdk/python/sintel_io.py"],"rgb":["README.txt"]}
    # The exact Python SDK path varies between releases; include all short SDK sources.
    wanted["depth"]=[name for name in infos["depth"] if name=="README_depth.txt" or (name.startswith("sdk/") and not name.endswith("/"))]
    for sequence,frame in selected:
        wanted["depth"].extend([f"training/depth/{sequence}/{frame}.dpt",f"training/camdata_left/{sequence}/{frame}.cam"])
        wanted["rgb"].append(f"training/clean/{sequence}/{frame}.png")
    records={key:[] for key in URLS}
    work=[(key,infos[key][name]) for key in URLS for name in wanted[key]]
    transferred=0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(fetch_member,URLS[key],remote[key].size,info):key for key,info in work}
        for index,future in enumerate(as_completed(futures),1):
            result=future.result();key=futures[future]
            transferred+=result.pop("range_bytes_transferred")
            records[key].append(result)
            if index%16==0 or index==len(work):
                print(f"Sintel members {index}/{len(work)}; newly transferred {transferred:,} bytes",flush=True)
    provenance={key:{"url":URLS[key],"archive_bytes":remote[key].size,
                     "archive_sha256":None,"archive_hash_note":"Full ZIP was not downloaded; central-directory and every selected decompressed member have hashes.",
                     "central_directory_cache_sha256":hashlib.sha256(b"".join(value for _,value in sorted(remote[key].cache.items()))).hexdigest(),
                     "members":sorted(records[key],key=lambda x:x["member"])} for key in URLS}
    write_json(DATA/"source_members.json",provenance)
    if args.download_only:
        print(f"Downloaded selected {len(selected)} frames from {len(sequences)} sequences",flush=True)
        return
    rows=[]
    for sequence,frame in selected:
        source=DATA/f"training/depth/{sequence}/{frame}.dpt"
        content=source.read_bytes()
        tag,width,height=struct.unpack("<fii",content[:12])
        if tag!=202021.25 or len(content)!=12+width*height*4:
            raise ValueError(f"Invalid Sintel .dpt: {source}")
        depth=np.frombuffer(content,dtype="<f4",offset=12).reshape(height,width).copy()
        image_path=DATA/f"training/clean/{sequence}/{frame}.png"
        with Image.open(image_path) as image:
            if image.size!=(width,height):
                raise ValueError(f"RGB/depth dimensions differ: {image_path}")
        # Explicit experimental range cap. The source contains finite 1e11
        # background-like values, which must not be treated as physical surfaces.
        mask=np.isfinite(depth)&(depth>0)&(depth<args.max_depth)
        if int(mask.sum())<10:
            raise ValueError(f"Too few valid GT pixels at max-depth={args.max_depth}: {sequence}/{frame}")
        identifier=f"sintel_{sequence}_{frame}"
        depth_path=DATA/"depth_m"/(identifier+".npy")
        mask_path=DATA/f"mask_{args.max_depth:g}m"/(identifier+".npy")
        save_npy(depth_path,depth)
        save_npy(mask_path,mask)
        camera_path=DATA/f"training/camdata_left/{sequence}/{frame}.cam"
        camera=camera_path.read_bytes()
        if struct.unpack("<f",camera[:4])[0]!=202021.25 or len(camera)!=172:
            raise ValueError(f"Invalid Sintel .cam: {camera_path}")
        intrinsics=np.frombuffer(camera,dtype="<f8",offset=4,count=9).reshape(3,3)
        rows.append({"id":identifier,"scene":sequence,"category":sequence,"frame":frame,
                     "image_path":image_path.relative_to(ROOT).as_posix(),
                     "depth_path":depth_path.relative_to(ROOT).as_posix(),
                     "mask_path":mask_path.relative_to(ROOT).as_posix(),
                     "camera_path":camera_path.relative_to(ROOT).as_posix(),
                     "camera_intrinsics":intrinsics.tolist(),
                     "height":height,"width":width,"depth_unit_scale":1.0,
                     "valid_pixels":int(mask.sum()),"total_pixels":int(depth.size),
                     "excluded_nonfinite_pixels":int((~np.isfinite(depth)).sum()),
                     "excluded_nonpositive_pixels":int((np.isfinite(depth)&(depth<=0)).sum()),
                     "excluded_ge80m_pixels":int((np.isfinite(depth)&(depth>=80)).sum()),
                     "excluded_by_max_depth_pixels":int((np.isfinite(depth)&(depth>=args.max_depth)).sum()),
                     "source_ge1e10_pixels":int((np.isfinite(depth)&(depth>=1e10)).sum()),
                     "source_depth_min_positive_m":float(depth[np.isfinite(depth)&(depth>0)].min()),
                     "source_depth_max_finite_m":float(depth[np.isfinite(depth)].max()),
                     "source_depth_max_below1e10_m":float(depth[np.isfinite(depth)&(depth>0)&(depth<1e10)].max()),
                     "sha256":{"image_path":sha256(image_path),"depth_path":sha256(depth_path),
                               "mask_path":sha256(mask_path),"camera_path":sha256(camera_path)}})
    total=sum(row["total_pixels"] for row in rows)
    valid=sum(row["valid_pixels"] for row in rows)
    far=sum(row["excluded_ge80m_pixels"] for row in rows)
    huge=sum(row["source_ge1e10_pixels"] for row in rows)
    clipped=sum(row["excluded_by_max_depth_pixels"] for row in rows)
    manifest={
        "dataset":"MPI Sintel clean rendered dense training subset","gt_kind":"rendered_dense",
        "image_count":len(rows),"original_split":"training (not the hidden testing split)",
        "population_images":sum(map(len,by_sequence.values())),"population_sequences":len(sequences),
        "sampling":"Sort sequences; allocate floor(count/n) frames to each; choose remainder sequences using Random(seed).sample; within each sequence independently Random(seed).sample sorted frames.",
        "seed":args.seed,"sequence_frame_counts":{s:sum(row["scene"]==s for row in rows) for s in sequences},
        "path_base":"workspace","depth_unit":"meters","depth_unit_scale":1.0,
        "depth_type":"blender_z_buffer_as_supplied","camera_z_vs_ray_verified":False,
        "depth_convention_note":"The official source states additional Blender Z-buffer pass in meters; neither source page nor SDK explicitly settles camera-axis Z versus Euclidean camera-ray distance. Values are preserved without conversion.",
        "min_depth":0.0,"max_depth":args.max_depth,"max_depth_exclusive":True,
        "mask_rule":f"isfinite(depth) & (depth > 0) & (depth < {args.max_depth:g})",
        "mask_cap_status":f"{args.max_depth:g} m is this experiment's explicit range criterion, not an official Sintel requirement. Selected depths below 1e10 are at most {max(row['source_depth_max_below1e10_m'] for row in rows):g} m; default 10000 m retains these and removes only observed 1e11 background-like values. Original arrays are retained.",
        "crop":"none (full 436x1024 image)",
        "total_image_pixels":total,"valid_pixels":valid,"valid_fraction":valid/total,
        "excluded_ge80m_pixels":far,"excluded_ge80m_fraction":far/total,
        "excluded_by_max_depth_pixels":clipped,"excluded_by_max_depth_fraction":clipped/total,
        "source_ge1e10_pixels":huge,"source_ge1e10_fraction":huge/total,
        "excluded_nonfinite_pixels":sum(row["excluded_nonfinite_pixels"] for row in rows),
        "excluded_nonpositive_pixels":sum(row["excluded_nonpositive_pixels"] for row in rows),
        "source_depth_max_below1e10_m":max(row["source_depth_max_below1e10_m"] for row in rows),
        "minimum_valid_pixels_per_image":min(row["valid_pixels"] for row in rows),
        "source_members_manifest":"data_dense/sintel/source_members.json",
        "source_members_manifest_sha256":sha256(DATA/"source_members.json"),
        "source_archive_urls":URLS,
        "selected_member_uncompressed_bytes":sum(r["bytes"] for values in records.values() for r in values),
        "documentation":["https://sintel.is.tue.mpg.de/depth","https://sintel.is.tue.mpg.de/downloads"],
        "license":{"dataset_status":"No explicit dataset redistribution license located in the reviewed MPI website and archive README/SDK; original film licensing does not by itself establish the terms of MPI's modified benchmark.",
                   "film_license":"Original Durian/Sintel material: Creative Commons Attribution 3.0, per Blender Foundation's sharing page.",
                   "film_license_source":"https://durian.blender.org/sharing",
                   "film_attribution":"(c) copyright Blender Foundation | www.sintel.org",
                   "dataset_citation":"Butler, Wulff, Stanley, Black. A naturalistic open source movie for optical flow evaluation. ECCV 2012.",
                   "dataset_redistribution_permission_verified":False},
        "evaluation_notes":[
            "Rendered depth is dense and not hole-filled sensor depth, but only the supplied GT-valid range mask is scored; this does not mean every pixel is valid or independently physically measured.",
            "Multiple selected frames share each sequence; confidence intervals should resample sequences if making claims beyond this fixed subset.",
            "Per-image affine inverse-depth oracle alignment uses GT scale/shift and does not demonstrate absolute metric depth prediction.",
        ],
        "images":rows,
    }
    manifest_path=DATA.parent/"sintel_manifest.json"
    preserve_previous_protocol(manifest_path,manifest)
    summary={key:value for key,value in manifest.items() if key!="images"}
    summary["manifest_sha256"]=sha256(manifest_path)
    summary["prepared_bytes"]=sum(path.stat().st_size for path in DATA.rglob("*") if path.is_file())
    preserve_previous_protocol(DATA.parent/"sintel_summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
