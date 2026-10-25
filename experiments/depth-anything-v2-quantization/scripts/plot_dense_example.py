"""Plot cached model outputs only, selecting the largest measured-GT W4A16 loss.

Run from the experiment workspace:
  .venv/Scripts/python.exe scripts/plot_dense_example.py

Neither RGB images nor ground-truth arrays are loaded or drawn.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image

MODELS = ["fp32", "fp16", "int8_w8a16", "int4_w4a16"]
LABELS = {"fp32": "FP32", "fp16": "FP16", "int8_w8a16": "INT8 W8A16", "int4_w4a16": "INT4 W4A16"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def workspace() -> Path:
    for candidate in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (candidate/"results_dense_nyu_gpu/per_image.csv").is_file():
            return candidate
    return Path.cwd()


def align_positive(candidate: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, dict]:
    q,fp = candidate.astype(np.float64),reference.astype(np.float64)
    qc,fc = q-q.mean(),fp-fp.mean()
    variance = float(np.mean(qc*qc))
    unconstrained = float(np.mean(qc*fc)/variance) if variance else 0.0
    scale = max(unconstrained,0.0)
    shift = float(fp.mean()-scale*q.mean())
    aligned = scale*q+shift
    rmse = float(np.sqrt(np.mean((aligned-fp)**2)))
    return aligned, {"scale": scale, "shift": shift, "unconstrained_scale": unconstrained,
                     "nonnegative_scale_boundary": bool(unconstrained <= 0),
                     "aligned_nrmse": rmse/float(fp.std())}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")


def main() -> None:
    root = workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",type=Path,default=root/"results_dense_nyu_gpu")
    parser.add_argument("--output-dir",type=Path,default=root/"work/depth_quant_article/dense_figure")
    args = parser.parse_args()
    result_dir,output_dir = args.results_dir.resolve(),args.output_dir.resolve()
    csv_path = result_dir/"per_image.csv"
    with csv_path.open(encoding="utf8",newline="") as stream:
        rows = list(csv.DictReader(stream))
    measured = {name:{row["id"]:row for row in rows if row["gt_kind"]=="measured" and row["model"]==name}
                for name in MODELS}
    if len(measured["fp32"]) != 128 or any(set(measured[name]) != set(measured["fp32"]) for name in MODELS):
        raise ValueError("Require all four variants on the same complete 128-image measured-GT split")
    differences = {key:float(measured["int4_w4a16"][key]["abs_rel"])-float(measured["fp32"][key]["abs_rel"])
                   for key in measured["fp32"]}
    identifier = min(differences,key=lambda key:(-differences[key],key))
    input_paths = {name:result_dir/"predictions"/name/(identifier+".npy") for name in MODELS}
    predictions = {name:np.load(path,allow_pickle=False).squeeze() for name,path in input_paths.items()}
    reference = predictions["fp32"]
    if reference.ndim != 2 or not float(reference.std()) > 0:
        raise ValueError("Reference must be a nonconstant 2D map")
    if any(value.shape != reference.shape or not np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("All cached predictions must be finite maps of the same native size")
    lower,upper = float(reference.min()),float(reference.max())
    span = upper-lower
    aligned,alignment = {},{}
    for name in MODELS:
        aligned[name],alignment[name] = align_positive(predictions[name],reference)
    absrel = {name:float(measured[name][identifier]["abs_rel"]) for name in MODELS}

    output_dir.mkdir(parents=True,exist_ok=True)
    figure_path = output_dir/"nyu_w4a16_worst_measured.png"
    fig = plt.figure(figsize=(13.8,8),dpi=100,facecolor="white")
    grid = fig.add_gridspec(2,5,width_ratios=[1,1,1,1,.055],left=.048,right=.915,
                           bottom=.145,top=.817,wspace=.11,hspace=.32)
    difference_cmap = LinearSegmentedColormap.from_list("blue_white_red",["#2166ac","#ffffff","#b2182b"],N=257)
    top_image,bottom_image = None,None
    for col,name in enumerate(MODELS):
        top = fig.add_subplot(grid[0,col])
        top_image = top.imshow(predictions[name],cmap="inferno",vmin=lower,vmax=upper,interpolation="nearest")
        top.set_title(f"{LABELS[name]}\nMeasured-GT AbsRel = {absrel[name]:.4f}",fontsize=11,pad=9)
        top.set_xticks([]);top.set_yticks([])
        bottom = fig.add_subplot(grid[1,col])
        difference_percent = 100*(aligned[name]-reference.astype(np.float64))/span
        bottom_image = bottom.imshow(difference_percent,cmap=difference_cmap,vmin=-10,vmax=10,interpolation="nearest")
        bottom.set_title(f"Aligned NRMSE = {alignment[name]['aligned_nrmse']:.4f}",fontsize=11,pad=8)
        bottom.set_xticks([]);bottom.set_yticks([])
        for axis in (top,bottom):
            for spine in axis.spines.values():
                spine.set_color("#c7cdd4");spine.set_linewidth(.7)
    top_cbar = fig.colorbar(top_image,cax=fig.add_subplot(grid[0,4]))
    top_cbar.set_label("Raw inverse depth\n(arbitrary units)",fontsize=10,labelpad=9)
    top_cbar.ax.tick_params(labelsize=9)
    bottom_cbar = fig.colorbar(bottom_image,cax=fig.add_subplot(grid[1,4]),ticks=[-10,-5,0,5,10])
    bottom_cbar.set_label("Aligned difference\n(% of FP32 map range)",fontsize=10,labelpad=9)
    bottom_cbar.ax.tick_params(labelsize=9)
    fig.suptitle("NYUv2 GPU: example with the largest W4A16 AbsRel increase",fontsize=17,fontweight="bold",x=.48,y=.978)
    fig.text(.048,.925,f"ID {identifier}  |  selected by measured-GT AbsRel(W4A16) - AbsRel(FP32), among 128 test images",
             fontsize=11,color="#374151")
    fig.text(.048,.877,"TOP: raw inverse-depth predictions, shared FP32 min/max",fontsize=11,fontweight="bold",color="#273345")
    fig.text(.048,.479,"BOTTOM: positive scale + shift aligned to FP32, then prediction - FP32",fontsize=11,fontweight="bold",color="#273345")
    fig.text(.048,.093,"Blue: below FP32  |  White: equal to FP32  |  Red: above FP32. Difference colors clip at +/-10% of the FP32 range.",fontsize=10,color="#374151")
    fig.text(.048,.060,"AbsRel uses measured valid GT pixels after per-image oracle inverse-depth alignment. The difference row measures agreement with FP32.",fontsize=9.5,color="#374151")
    fig.text(.048,.028,"Only cached model outputs are displayed. RGB inputs and ground-truth depth maps are not included.",fontsize=9.5,color="#374151")
    fig.savefig(figure_path,dpi=100,metadata={"Title":"NYUv2 cached prediction comparison", "Description":"Model outputs and differences only; no RGB or GT displayed"})
    plt.close(fig)
    with Image.open(figure_path) as figure:
        if figure.size != (1380,800):
            raise AssertionError(f"Unexpected figure dimensions: {figure.size}")

    caption = {
        "id":identifier,"dataset":"NYUv2 official test subset, 128 images, seed42","provider":"CUDA",
        "selection_reason":"Largest per-image measured-GT AbsRel(W4A16) minus AbsRel(FP32); ascending ID resolves any tie",
        "selection_images":len(differences),"measured_gt_abs_rel_increase_w4a16":differences[identifier],
        "measured_gt_valid_pixels":int(measured["fp32"][identifier]["valid_pixels"]),
        "native_prediction_shape":list(reference.shape),
        "models":{name:{"label":LABELS[name],"measured_gt_abs_rel":absrel[name],
                         "fp32_map_alignment":alignment[name],"prediction_sha256":sha256(input_paths[name])}
                  for name in MODELS},
        "raw_inverse_display":{"common_reference":"fp32","min":lower,"max":upper,"colormap":"inferno","out_of_range_colors_clipped":True},
        "difference_display":{"formula":"100 * (s * prediction + t - FP32) / (max(FP32) - min(FP32))", "alignment":"Least squares on all native map pixels with s >= 0, independently for each variant", "fixed_percent_limits":[-10,10],"colormap":"blue_white_red","colors_clipped":True},
        "nrmse_definition":"RMSE of positive-affine-aligned candidate versus FP32 divided by std(FP32), on all native map pixels",
        "gt_metric_interpretation":"Measured-GT AbsRel is copied from per_image.csv after per-image oracle alignment against inverse metric GT on measured valid pixels; it is not uncalibrated absolute-distance accuracy",
        "rgb_or_gt_arrays_loaded":False,"rgb_or_gt_rendered":False,
        "alt_ja":"NYUv2の同一入力に対するGPUモデル出力の比較。上段にFP32、FP16、INT8 W8A16、INT4 W4A16の逆深度予測を共通色域で表示し、下段にFP32へscaleとshiftを整合した差を青・白・赤で表示。INT4 W4A16で実測GTのAbsRel増加が最大だった画像を選択。元RGB画像とGT深度画像は含まない。",
        "caption_ja":f"NYUv2テスト128画像のうち、実測GTに対するAbsRelの増加（W4A16−FP32）が最大だった {identifier}。AbsRelはFP32 {absrel['fp32']:.4f}、FP16 {absrel['fp16']:.4f}、W8A16 {absrel['int8_w8a16']:.4f}、W4A16 {absrel['int4_w4a16']:.4f}。上段は共通のFP32最小・最大値で表示したモデルの逆深度出力、下段は各出力をFP32へ正のscaleとshiftで整合した差（色域はFP32レンジの±10%）。下段はGT誤差を表すものではない。AbsRelは画像ごとのGTを用いたoracle整合後の値で、絶対距離を直接推定した精度ではない。元RGB画像・GT深度画像は掲載していない。",
        "source_csv":str(csv_path),"source_csv_sha256":sha256(csv_path),
    }
    caption_path = output_dir/"caption.json"
    write_json(caption_path,caption)
    copied_script = output_dir/"plot_dense_example.py"
    if copied_script.resolve() != Path(__file__).resolve():
        shutil.copyfile(Path(__file__),copied_script)
    write_json(output_dir/"generation.json",{
        "generator":"scripts/plot_dense_example.py","python":sys.version,"matplotlib":matplotlib.__version__,"numpy":np.__version__,
        "command":".venv/Scripts/python.exe scripts/plot_dense_example.py",
        "figure":figure_path.name,"pixels":[1380,800],"figure_sha256":sha256(figure_path),
        "caption":caption_path.name,"caption_sha256":sha256(caption_path),
        "script_copy":copied_script.name,"script_sha256":sha256(copied_script),
        "input_prediction_files":{name:str(path) for name,path in input_paths.items()},
        "arrays_loaded":"Only fp32, fp16, int8_w8a16, and int4_w4a16 native prediction NPY files for the selected ID",
    })
    print(json.dumps({"figure":str(figure_path),"id":identifier,"measured_abs_rel":absrel,
                      "w4a16_minus_fp32_abs_rel":differences[identifier],
                      "aligned_nrmse":{name:alignment[name]["aligned_nrmse"] for name in MODELS}},indent=2),flush=True)


if __name__ == "__main__":
    main()
