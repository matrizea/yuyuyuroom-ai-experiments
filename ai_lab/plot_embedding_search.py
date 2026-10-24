#!/usr/bin/env python3
"""Plot TF-IDF vs EmbeddingGemma local retrieval results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ai_lab" / "results" / "embedding_search.json"
OUT = ROOT / "figures" / "tfidf-vs-embeddinggemma.png"


def main() -> None:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    tf, em = d["tfidf_summary"], d["embedding_summary"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), constrained_layout=True)
    colors = ["#6b8afd", "#25b99a"]
    x = np.arange(3); w = .34
    axes[0].bar(x-w/2, [tf["hit_at_1"], tf["hit_at_3"], tf["mrr"]*20], w, label="TF-IDF", color=colors[0])
    axes[0].bar(x+w/2, [em["hit_at_1"], em["hit_at_3"], em["mrr"]*20], w, label="EmbeddingGemma", color=colors[1])
    axes[0].set_xticks(x, ["hit@1", "hit@3", "MRR × 20"]); axes[0].set_ylim(0,21)
    axes[0].set_ylabel("Questions / 20 (MRR scaled)"); axes[0].set_title("Expected-source retrieval")
    axes[0].legend(); axes[0].grid(axis="y",alpha=.25)

    axes[1].bar(["TF-IDF", "Embedding"], [tf["mean_query_wall_ms"], em["mean_query_wall_ms"]], color=colors)
    axes[1].set_yscale("log"); axes[1].set_ylabel("Mean query latency (ms, log)"); axes[1].set_title("Query latency")
    for i,v in enumerate([tf["mean_query_wall_ms"], em["mean_query_wall_ms"]]): axes[1].text(i,v*1.18,f"{v:.1f} ms",ha="center",weight="bold")

    rag = d["embedding_rag"]
    axes[2].bar(["TF-IDF RAG", "Embedding RAG"], [19, rag["correct"]], color=colors)
    axes[2].set_ylim(0,21); axes[2].set_ylabel("Correct answers / 20"); axes[2].set_title("Same Qwen 3 8B generator")
    for i,v in enumerate([19,rag["correct"]]): axes[2].text(i,v+.25,f"{v}/20",ha="center",weight="bold")
    axes[2].grid(axis="y",alpha=.25)
    fig.suptitle("Embeddings did not beat a lexical baseline on exact local measurements", fontsize=16, weight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT,dpi=180,facecolor="white")
    print(OUT)

if __name__ == "__main__": main()
