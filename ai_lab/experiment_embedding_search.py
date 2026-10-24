#!/usr/bin/env python3
"""Compare TF-IDF and EmbeddingGemma retrieval on the fixed local RAG set."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from experiment_local_rag import ARTICLE_NAMES, QUESTIONS, ROOT, CORPUS, ask, chunks_from_markdown, correct


OUT = ROOT / "ai_lab" / "results" / "embedding_search.json"
EMBED_MODEL = "embeddinggemma"


def embed(inputs: list[str]) -> tuple[np.ndarray, dict]:
    payload = json.dumps({"model": EMBED_MODEL, "input": inputs, "truncate": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as response:
        data = json.load(response)
    arr = np.asarray(data["embeddings"], dtype=np.float32)
    arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
    return arr, {"wall_s": time.perf_counter() - t0,
                 "ollama_total_s": data.get("total_duration", 0) / 1e9,
                 "prompt_eval_count": data.get("prompt_eval_count")}


def ranks(scores: np.ndarray, chunks: list[dict], source: str) -> tuple[list[int], int | None]:
    order = np.argsort(scores)[::-1]
    first = next((rank for rank, idx in enumerate(order, 1) if chunks[idx]["source"] == source), None)
    return order[:3].tolist(), first


def main() -> None:
    chunks = [c for name in ARTICLE_NAMES for c in chunks_from_markdown(CORPUS / name)]
    texts = [c["text"] for c in chunks]

    tf0 = time.perf_counter()
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
    tf_matrix = tf.fit_transform(texts)
    tf_index_s = time.perf_counter() - tf0

    batches, embs, embed_meta = [], [], []
    emb0 = time.perf_counter()
    for start in range(0, len(texts), 16):
        arr, meta = embed(texts[start:start + 16])
        embs.append(arr); embed_meta.append(meta); batches.append([start, start + len(arr)])
        print(f"embedded {start + len(arr)}/{len(texts)}", flush=True)
    emb_matrix = np.vstack(embs)
    emb_index_s = time.perf_counter() - emb0

    results = []
    for i, item in enumerate(QUESTIONS, 1):
        t0 = time.perf_counter(); tq = tf.transform([item.q]); tf_scores = (tf_matrix @ tq.T).toarray().ravel(); tf_q_s = time.perf_counter() - t0
        tf_top, tf_rank = ranks(tf_scores, chunks, item.source)
        t0 = time.perf_counter(); qemb, qmeta = embed([item.q]); emb_q_s = time.perf_counter() - t0
        emb_scores = emb_matrix @ qemb[0]
        emb_top, emb_rank = ranks(emb_scores, chunks, item.source)

        retrieved = [{**chunks[j], "score": float(emb_scores[j])} for j in emb_top]
        context = "\n\n".join(f"[{x['id']}]\n{x['text']}" for x in retrieved)
        prompt = ("提供資料だけを根拠に質問へ簡潔に答えてください。最後に根拠のchunk IDを"
                  "角括弧で1つ以上引用してください。資料に無ければ『資料内に記載なし』と答えてください。\n\n"
                  f"資料:\n{context}\n\n質問: {item.q}")
        answer, gen_s = ask(prompt)
        row = {
            "index": i, "question": item.q, "expected_source": item.source,
            "tfidf": {"rank": tf_rank, "top3": [chunks[j]["id"] for j in tf_top], "query_wall_s": tf_q_s},
            "embedding": {"rank": emb_rank, "top3": retrieved, "query_wall_s": emb_q_s,
                          "ollama_total_s": qmeta["ollama_total_s"]},
            "answer": answer, "answer_correct": correct(answer, item.patterns),
            "cites_expected_source": item.source in answer, "generation_wall_s": gen_s,
        }
        results.append(row)
        print(f"{i:02d}/20 tf_rank={tf_rank} emb_rank={emb_rank} answer={row['answer_correct']}", flush=True)

    def metrics(key: str) -> dict:
        ranks_ = [x[key]["rank"] for x in results]
        return {"hit_at_1": sum(r == 1 for r in ranks_), "hit_at_3": sum(r is not None and r <= 3 for r in ranks_),
                "mrr": float(np.mean([1/r if r else 0 for r in ranks_])),
                "mean_query_wall_ms": float(np.mean([x[key]["query_wall_s"] for x in results]) * 1000)}

    output = {
        "embedding_model": EMBED_MODEL, "dimensions": int(emb_matrix.shape[1]),
        "documents": len(ARTICLE_NAMES), "chunks": len(chunks), "questions": len(QUESTIONS),
        "indexing": {"tfidf_wall_s": tf_index_s, "embedding_wall_s": emb_index_s,
                     "embedding_batches": batches, "ollama_batch_meta": embed_meta},
        "tfidf_summary": metrics("tfidf"), "embedding_summary": metrics("embedding"),
        "embedding_rag": {"correct": sum(x["answer_correct"] for x in results),
                          "expected_citation": sum(x["cites_expected_source"] for x in results),
                          "mean_generation_wall_s": float(np.mean([x["generation_wall_s"] for x in results]))},
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("tfidf_summary", "embedding_summary", "embedding_rag")}, ensure_ascii=False, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
