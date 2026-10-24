#!/usr/bin/env python3
"""Plot local RAG retrieval and answer results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ai_lab" / "results" / "local_rag.json"
OUT = ROOT / "figures" / "local-rag-tfidf-qwen3.png"


def main() -> None:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    s = d["summary"]
    labels = ["No retrieval\ncorrect", "RAG\ncorrect", "Source hit\n@1", "Source hit\n@3", "Expected\ncitation"]
    values = [s["no_rag_correct"], s["rag_correct"], s["retrieval_hit_at_1"],
              s["retrieval_hit_at_3"], s["rag_expected_citation"]]
    colors = ["#e35d6a", "#25b99a", "#6b8afd", "#4469dc", "#f3b641"]
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, width=.68)
    ax.set_ylim(0, 21.5)
    ax.set_ylabel("Questions / 20")
    ax.set_title("Local RAG on measurements the model could not have memorized", fontsize=16, weight="bold")
    ax.grid(axis="y", alpha=.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + .25, f"{value}/20", ha="center", weight="bold", fontsize=12)
    ax.text(.02, .94, "13 local articles → 198 heading chunks → char 2–4 gram TF-IDF → top 3 → Qwen 3 8B",
            transform=ax.transAxes, color="#444")
    ax.text(.02, .86, "The one RAG miss retrieved the right document but not the chunk containing 11,244 tokens.",
            transform=ax.transAxes, color="#8a3d47")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
