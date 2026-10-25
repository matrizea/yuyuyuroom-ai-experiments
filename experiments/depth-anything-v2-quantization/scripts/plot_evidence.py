"""Plot the recorded CPU and CUDA measurements without redistributing source images."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "article_figures"
LABEL = {"fp32":"FP32", "fp16":"FP16", "int8_dynamic":"INT8 dynamic", "int4_rtn":"INT4 / A32",
         "int4_fast":"INT4 / internal A8 allowed", "int8_w8a16":"W8A16", "int4_w4a16":"W4A16"}
COLORS = {"fp32":"#475569", "fp16":"#0284c7", "int8_dynamic":"#047857", "int4_rtn":"#b45309",
          "int4_fast":"#7c3aed", "int8_w8a16":"#0e7490", "int4_w4a16":"#be185d"}


def main():
    OUT.mkdir(exist_ok=True)
    cpu = json.loads((ROOT/"evidence/cpu_ordinal/summary.json").read_text(encoding="utf-8"))
    gpu = json.loads((ROOT/"evidence/gpu_benchmark.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "figure.facecolor":"white", "axes.axisbelow":True})
    names = list(cpu["models"])
    y = np.arange(len(names))
    fig, axs = plt.subplots(1,3,figsize=(13,3.8), gridspec_kw={"width_ratios":[1.1,1,1.2]})
    colors = [COLORS[n] for n in names]
    for ax, key, title in [(axs[0],"latency","CPU latency (ms, lower is better)"),
                           (axs[1],"size","ONNX file (MiB, lower is better)")]:
        values = [cpu["models"][n]["benchmark"]["median_ms"] if key=="latency" else cpu["models"][n]["file_size_mib"] for n in names]
        ax.barh(y,values,color=colors,height=.58)
        for pos,value in zip(y,values): ax.text(value+max(values)*.02,pos,f"{value:.1f}",va="center",fontsize=9)
        ax.set_xlim(0,max(values)*1.23);ax.set_yticks(y,[LABEL[n] for n in names] if key=="latency" else [])
        ax.invert_yaxis();ax.set_title(title,fontsize=10);ax.grid(axis="x",alpha=.2)
    intervals=[cpu["models"][n]["versus_fp32"]["ci95_percentage_points"] for n in names]
    means=[cpu["models"][n]["versus_fp32"]["difference_percentage_points"] for n in names]
    axs[2].axvline(0,color="#64748b",lw=1)
    for pos,name,mean,(low,high) in zip(y,names,means,intervals):
        axs[2].plot([low,high],[pos,pos],color=COLORS[name],lw=2)
        axs[2].scatter(mean,pos,color=COLORS[name],s=35,zorder=3)
    axs[2].set_yticks(y,[]);axs[2].invert_yaxis();axs[2].set_title("Near/far accuracy vs FP32 (pp)",fontsize=10)
    axs[2].set_xlabel("95% CI, paired image bootstrap");axs[2].grid(axis="x",alpha=.2)
    fig.suptitle("Core i7-14700F / 8 CPU threads / fixed 518 x 518",fontsize=12)
    fig.tight_layout();fig.savefig(OUT/"cpu-quantization-results.png",dpi=180,bbox_inches="tight");plt.close(fig)
    names=list(gpu["models"]);y=np.arange(len(names));fig,axs=plt.subplots(1,2,figsize=(11,5.1),gridspec_kw={"width_ratios":[1.5,1]})
    host=[gpu["models"][n]["host_input_output"]["median_ms"] for n in names]
    resident=[gpu["models"][n]["device_resident_input_output"]["median_ms"] for n in names]
    axs[0].barh(y-.15,host,height=.28,color="#2563a8",label="Host input/output")
    axs[0].barh(y+.15,resident,height=.28,color="#6b9eab",label="GPU resident input/output")
    for pos,a,b in zip(y,host,resident):
        axs[0].text(a+.6,pos-.15,f"{a:.2f}",va="center",fontsize=8)
        axs[0].text(b+.6,pos+.15,f"{b:.2f}",va="center",fontsize=8)
    axs[0].set_yticks(y,[LABEL[n] for n in names]);axs[0].set_xlim(0,max(host+resident)*1.16);axs[0].invert_yaxis()
    axs[0].set_title("Warm latency (ms, lower is better)");axs[0].legend(loc="lower right",fontsize=8);axs[0].grid(axis="x",alpha=.2)
    sizes=[gpu["models"][n]["size_mib"] for n in names]
    axs[1].barh(y,sizes,color=[COLORS[n] for n in names],height=.58)
    for pos,value in zip(y,sizes):axs[1].text(value+1,pos,f"{value:.2f}",va="center",fontsize=9)
    axs[1].set_yticks(y,[]);axs[1].set_xlim(0,max(sizes)*1.2);axs[1].invert_yaxis();axs[1].set_title("ONNX file (MiB)");axs[1].grid(axis="x",alpha=.2)
    fig.suptitle("RTX 5070 Ti / TF32 disabled / 20 warmups + 100 runs",fontsize=12)
    fig.text(.25,.005,"INT8 dynamic includes CPU fallback and device transfers.",ha="center",fontsize=9,color="#9f1239")
    fig.tight_layout(rect=(0,.025,1,1));fig.savefig(OUT/"gpu-quantization-results.png",dpi=180,bbox_inches="tight");plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__": main()
