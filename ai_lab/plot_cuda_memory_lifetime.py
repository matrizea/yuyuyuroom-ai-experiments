"""Plot raw byte measurements from experiment_cuda_memory_lifetime.py."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
raw = json.loads(args.input.read_text(encoding="utf-8"))
ref = raw["repeats"][0]["references"]
fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
for field, marker in [("allocated_bytes", "o"), ("reserved_bytes", "s")]:
    axes[0].plot(range(6), [r[field]/1024**2 for r in ref], marker=marker,
                 label=field.removesuffix("_bytes"), linestyle="--" if field.startswith("reserved") else "-")
axes[0].set_xticks(range(6), ["Start", "Create\ntensor", "empty_cache\n(tensor live)", "del tensor\n(list live)", "Clear\nlist", "empty_cache\n(no refs)"])
axes[0].set_ylabel("PyTorch memory (MiB)")
axes[0].set_title("256 MiB tensor: references first, cache second")
axes[0].legend()
axes[0].grid(alpha=.2)
hist = raw["repeats"][0]["histories"]
values = [h["growth_bytes"]/1024**2 for h in hist]
axes[1].barh(range(len(hist)), values, color=["#be6847", "#437e91", "#be6847", "#437e91", "#437e91"])
axes[1].set_yticks(range(len(hist)), [h["mode"] for h in hist])
axes[1].invert_yaxis()
for i, val in enumerate(values):
    axes[1].text(val+.5, i, f"{val:.3f} MiB", va="center")
axes[1].set_xlim(0, 52)
axes[1].set_xlabel("Allocated growth after 10 forward passes (MiB)")
axes[1].set_title("What is kept in the list? (not a training benchmark)")
axes[1].grid(axis="x", alpha=.2)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=150)
print(args.output)
