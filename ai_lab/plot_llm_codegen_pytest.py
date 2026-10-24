#!/usr/bin/env python3
"""Plot the reproducible local-LLM code-generation experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ai_lab" / "results" / "llm_codegen_pytest.json"
OUT = ROOT / "figures" / "llm-codegen-pytest.png"


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    labels = ["Gemma 3\n4B Q4", "Qwen 3\n8B Q4", "Gemma 3\n12B Q4"]
    keys = data["models"]
    first = [data["summary"][k]["first_pass"] for k in keys]
    final = [data["summary"][k]["final_pass"] for k in keys]

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.34
    axes[0].bar(x - width / 2, first, width, label="First generation", color="#6b8afd")
    axes[0].bar(x + width / 2, final, width, label="After one repair", color="#25b99a")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 10.8)
    axes[0].set_ylabel("Passed tasks / 10")
    axes[0].set_title("pytest pass count")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", alpha=.25)
    for xpos, value in zip(x - width / 2, first):
        axes[0].text(xpos, value + .15, str(value), ha="center", weight="bold")
    for xpos, value in zip(x + width / 2, final):
        axes[0].text(xpos, value + .15, str(value), ha="center", weight="bold")

    tasks = [task["name"] for task in data["tasks"]]
    matrix = []
    for model in keys:
        row = []
        for task in tasks:
            r = next(x for x in data["results"] if x["model"] == model and x["task"] == task)
            first_ok = bool((r.get("first_test") or {}).get("passed"))
            row.append(2 if first_ok else (1 if r["final_passed"] else 0))
        matrix.append(row)
    short = ["pal", "2sum", "RLE", "merge", "bracket", "rotate", "prefix", "top-k", "avg", "grid"]
    from matplotlib.colors import ListedColormap
    axes[1].imshow(matrix, aspect="auto", cmap=ListedColormap(["#e35d6a", "#f3b641", "#25b99a"]), vmin=0, vmax=2)
    axes[1].set_xticks(range(10), short, rotation=45, ha="right")
    axes[1].set_yticks(range(3), [x.replace("\n", " ") for x in labels])
    axes[1].set_title("Task outcome")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axes[1].text(j, i, ["fail", "fixed", "pass"][value], ha="center", va="center", fontsize=8,
                         color="white" if value != 1 else "#3a2b00", weight="bold")

    fig.suptitle("Local LLM code generation: tests, not appearance, decide", fontsize=16, weight="bold")
    fig.text(.5, .01, "10 fixed Python tasks · temperature=0 · one repair attempt · isolated pytest subprocess", ha="center", color="#555")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
