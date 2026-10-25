"""Prepare GT-free FP8 calibration tensors from the official NYUv2 TRAIN split.

The input image pixels and normalized tensors remain local. Only this script,
indices, seeds, hashes and manifest metadata are intended for public release.

Usage:
  .venv/Scripts/python.exe scripts/prepare_fp8_calibration.py
  inputs = numpy.load('data_fp8_calibration/inputs.npy', mmap_mode='r')
  batch_one = inputs[i:i+1]  # float32, [1,3,518,518]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import cv2
import h5py
import numpy as np
from scipy.io import loadmat

from eval_utils import ROOT, preprocess, sha256

COUNT, SEED = 64, 43
EVALUATION_MANIFESTS = {
    "nyuv2": ROOT/"data_dense/manifest.json",
    "sintel": ROOT/"data_dense/sintel_manifest.json",
    "da2k": ROOT/"data/eval_manifest.json",
}


def rgb_digest(rgb: np.ndarray) -> str:
    """Hash shape plus decoded uint8 RGB bytes, independent of image encoding."""
    rgb = np.ascontiguousarray(rgb,dtype=np.uint8)
    digest=hashlib.sha256()
    digest.update(np.asarray(rgb.shape,dtype="<i8").tobytes())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def text(file: h5py.File, key: str, index: int) -> str:
    refs=file[key]
    ref=refs[0,index] if refs.shape[0]==1 else refs[index,0]
    return "".join(chr(int(char)) for char in file[ref][:].ravel())


def evaluation_image_digests() -> tuple[dict,dict[str,str]]:
    """Read evaluation RGBs only, for exact decoded-image overlap checks."""
    hashes,metadata={},{}
    for name,path in EVALUATION_MANIFESTS.items():
        manifest=json.loads(path.read_text(encoding="utf8"))
        for item in manifest["images"]:
            image_path=ROOT/item.get("image_path",item.get("path"))
            bgr=cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),cv2.IMREAD_COLOR)
            if bgr is None:
                raise IOError(f"Cannot decode evaluation RGB: {image_path}")
            digest=rgb_digest(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            hashes[digest]=image_path.relative_to(ROOT).as_posix()
        metadata[name]={"manifest":path.relative_to(ROOT).as_posix(),
                        "sha256":sha256(path),"evaluation_images":len(manifest["images"])}
    return metadata,hashes


def write_json(path: Path, value: object) -> None:
    content=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if path.exists():
        if path.read_text(encoding="utf8")!=content:
            raise FileExistsError(f"Existing generated metadata differs: {path}")
    else:
        with path.open("x",encoding="utf8",newline="\n") as output:
            output.write(content)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"data_fp8_calibration")
    args=parser.parse_args()
    output_dir=args.output_dir.resolve()
    if not output_dir.is_relative_to(ROOT):
        raise ValueError("Calibration output must stay inside this experiment workspace")
    output_dir.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    original=ROOT/"data_dense/nyu_depth_v2_labeled.mat"
    split_path=ROOT/"data_dense/splits.mat"
    split=loadmat(split_path)
    train=sorted(int(x)-1 for x in split["trainNdxs"].ravel())
    test=sorted(int(x)-1 for x in split["testNdxs"].ravel())
    assert len(train)==795 and len(test)==654
    assert not set(train)&set(test)
    assert sorted(train+test)==list(range(1449))
    selected=sorted(random.Random(SEED).sample(train,COUNT))
    assert not set(selected)&set(test), "Calibration indices intersect official NYU test"
    nyu_eval=json.loads(EVALUATION_MANIFESTS["nyuv2"].read_text(encoding="utf8"))
    assert not {index+1 for index in selected}&{row["original_matlab_index_1based"] for row in nyu_eval["images"]}
    print("Checking decoded RGB overlap against all three evaluation manifests...",flush=True)
    eval_metadata,eval_rgb_hashes=evaluation_image_digests()

    final_path=output_dir/"inputs.npy"
    shape=(COUNT,3,518,518)
    existing=final_path.exists()
    if existing:
        array=np.load(final_path,mmap_mode="r",allow_pickle=False)
        if array.shape!=shape or array.dtype!=np.float32:
            raise FileExistsError("Existing calibration array has the wrong shape or dtype")
        temporary=None
    else:
        temporary=output_dir/"inputs.inprogress.npy"
        if temporary.exists():
            raise FileExistsError(f"An unfinished calibration array already exists: {temporary}")
        array=np.lib.format.open_memmap(temporary,mode="w+",dtype=np.float32,shape=shape)
    rows=[]
    with h5py.File(original,"r") as file:
        assert file["images"].shape==(1449,3,640,480)
        calibration_scenes={text(file,"scenes",index) for index in selected}
        test_scenes={text(file,"scenes",index) for index in test}
        assert not calibration_scenes&test_scenes, "Calibration and test share a NYU scene"
        for row_index,index in enumerate(selected):
            # Only RGB and scene metadata are accessed. No depth/GT variable is opened.
            rgb=np.asarray(file["images"][index],dtype=np.uint8).transpose(2,1,0).copy()
            image_digest=rgb_digest(rgb)
            if image_digest in eval_rgb_hashes:
                raise AssertionError(f"Calibration image equals an evaluation RGB: {eval_rgb_hashes[image_digest]}")
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            # Reuse the exact same implementation as scripts/evaluate.py.
            tensor=preprocess(bgr,input_size=518,resize_mode="square")
            assert tensor.dtype==np.float32 and tensor.shape==(1,3,518,518)
            assert np.isfinite(tensor).all()
            if existing:
                if not np.array_equal(array[row_index],tensor[0]):
                    raise FileExistsError(f"Existing tensor differs at row {row_index}")
            else:
                array[row_index]=tensor[0]
            rows.append({"row":row_index,"id":f"nyu_{index+1:04d}",
                         "original_matlab_index_1based":index+1,
                         "scene":text(file,"scenes",index),"category":text(file,"sceneTypes",index),
                         "decoded_rgb_sha256":image_digest,
                         "tensor_float32_nchw_sha256":hashlib.sha256(tensor[0].tobytes()).hexdigest()})
    if not existing:
        array.flush()
    del array
    if temporary is not None:
        if final_path.exists():
            raise FileExistsError(final_path)
        temporary.rename(final_path)

    manifest={
        "purpose":"Static FP8 activation calibration using RGB inputs only; evaluation GT is not used",
        "dataset":"NYUv2 original labeled dataset, official TRAIN split only",
        "count":COUNT,"seed":SEED,"selection":"Sort official 0-based train indices; random.Random(43).sample(...,64); sort selected indices",
        "official_train_images":795,"official_test_images":654,
        "selected_indices_1based":[index+1 for index in selected],
        "array":{"path":final_path.relative_to(ROOT).as_posix(),"format":"NumPy NPY, no pickle",
                 "shape":list(shape),"dtype":"float32","layout":"NCHW","input_batch_size":1,
                 "bytes":final_path.stat().st_size,"sha256":sha256(final_path),
                 "loading":"np.load(path, mmap_mode='r', allow_pickle=False); inputs[i:i+1] -> [1,3,518,518]"},
        "preprocessing":{"implementation":"scripts/eval_utils.py::preprocess(image_bgr,input_size=518,resize_mode='square')",
                         "steps":"RGB uint8 -> BGR to match evaluation reader; exact shared function converts to RGB/255 float64, square 518x518 OpenCV CUBIC with no crop, official ImageNet mean/std, float32 NCHW",
                         "eval_utils_sha256":sha256(ROOT/"scripts/eval_utils.py"),
                         "official_transform_sha256":sha256(ROOT/"vendor/Depth-Anything-V2/depth_anything_v2/util/transform.py"),
                         "opencv":cv2.__version__,"numpy":np.__version__,"h5py":h5py.__version__},
        "source":{"dataset_url":"https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
                  "dataset_sha256":sha256(original),"dataset_bytes":original.stat().st_size,
                  "split_url":"https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat",
                  "split_sha256":sha256(split_path)},
        "separation_checks":{"official_nyu_test_index_overlap":0,"nyu_eval128_index_overlap":0,
                             "official_nyu_test_scene_overlap":0,
                             "exact_decoded_rgb_overlap_with_evaluation_images":0,
                             "evaluation_images_checked":sum(item["evaluation_images"] for item in eval_metadata.values()),
                             "rgb_hash_definition":"SHA256(little-endian int64 RGB shape + contiguous uint8 RGB bytes); excludes encoding differences, but is not a perceptual near-duplicate test",
                             "evaluation_manifests":eval_metadata},
        "gt_access":{"depth_arrays_opened":False,"gt_values_used_for_calibration":False,
                     "hdf5_variables_read":["images","scenes","sceneTypes"],
                     "note":"Evaluation RGB images are decoded only to assert nonoverlap; they are never included in calibration tensors"},
        "publication":{"publish_input_array":False,"publish_rgb_or_gt":False,
                       "publish_script_indices_seed_hash_metadata":True,
                       "reason":"Normalized tensors encode original image content; no original dataset redistribution license is asserted. See data_dense/license_review.json."},
        "images":rows,
    }
    write_json(output_dir/"manifest.json",manifest)
    print(json.dumps({"ready":str(final_path),"shape":list(shape),"dtype":"float32",
                      "array_sha256":manifest["array"]["sha256"],"bytes":manifest["array"]["bytes"],
                      "calibration_scenes":len(calibration_scenes),"test_index_overlap":0,
                      "test_scene_overlap":0,"exact_evaluation_rgb_overlap":0,
                      "manifest_sha256":sha256(output_dir/"manifest.json")},indent=2),flush=True)


if __name__=="__main__":
    main()
