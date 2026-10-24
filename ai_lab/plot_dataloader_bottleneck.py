#!/usr/bin/env python3
"""Re-render the DataLoader benchmark figure from its saved JSON."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.json.read_text(encoding="utf-8"))
    runs = payload["runs"]
    pin_trials = payload["pin_memory_trials"]
    high_trials = payload["high_worker_trials"]
    workers_list = [0, 1, 2, 4, 8, 12, 16]

    def find(workload, workers, pin=True):
        return next(run for run in runs if run["workload"] == workload and run["workers"] == workers and run["pin_memory"] == pin)

    font_path = Path("/mnt/c/Windows/Fonts/meiryo.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), dpi=150); fig.patch.set_facecolor("#07111f")
    for ax in axes:
        ax.set_facecolor("#0b172a"); ax.tick_params(colors="#dbeafe"); ax.grid(axis="y", alpha=.18, color="white")
        for spine in ax.spines.values(): spine.set_visible(False)
    for workload, color in [("light", "#38bdf8"), ("augmented", "#a78bfa")]:
        warm_ips = [find(workload, worker)["epochs"][1]["images_per_second"] for worker in workers_list]
        if workload == "augmented":
            for worker in [8, 12, 16]:
                warm_ips[workers_list.index(worker)] = float(np.median([trial["warm_images_per_second"] for trial in high_trials[str(worker)]]))
        axes[0].plot(workers_list, warm_ips, "o-", linewidth=2, color=color, label=workload)
    axes[0].set_xlabel("num_workers", color="#dbeafe"); axes[0].set_ylabel("warm epoch images/s", color="#dbeafe")
    axes[0].set_title("transform負荷でplateauが変わる", color="white"); axes[0].legend(facecolor="#0b172a", labelcolor="white")

    first = [find("augmented", worker)["epochs"][0]["seconds"] for worker in workers_list]
    warm = [find("augmented", worker)["epochs"][1]["seconds"] for worker in workers_list]
    x = np.arange(len(workers_list)); width = .36
    axes[1].bar(x-width/2, first, width, color="#fb7185", label="epoch 1（起動込み）")
    axes[1].bar(x+width/2, warm, width, color="#34d399", label="epoch 2（persistent）")
    axes[1].set_xticks(x, workers_list); axes[1].set_xlabel("num_workers", color="#dbeafe"); axes[1].set_ylabel("augmented epoch [s]", color="#dbeafe")
    axes[1].set_title("worker起動費と温間性能を分離", color="white"); axes[1].legend(facecolor="#0b172a", labelcolor="white")
    pin_on = float(np.median([trial["warm_seconds"] for trial in pin_trials["true"]]))
    pin_off = float(np.median([trial["warm_seconds"] for trial in pin_trials["false"]]))
    fig.suptitle("WSL2 PyTorch DataLoader bottleneckを実測", color="white", fontsize=18, y=.96)
    fig.text(.5, .035, f"FashionMNIST CNN / batch 256 / pin_memory workers=4 各5回中央値: OFF {pin_off:.3f}s → ON {pin_on:.3f}s", ha="center", color="#cbd5e1", fontsize=10)
    fig.subplots_adjust(left=.09, right=.97, bottom=.15, top=.82, wspace=.22)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, facecolor=fig.get_facecolor()); plt.close(fig)


if __name__ == "__main__":
    main()
