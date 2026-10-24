#!/usr/bin/env python3
"""Evaluate local TF-IDF retrieval + Qwen 3 8B on facts from prior experiments."""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
OUT = ROOT / "ai_lab" / "results" / "local_rag.json"
MODEL = "qwen3:8b"
ARTICLE_NAMES = [
    "article_ollama_4b_8b_12b.md", "article_gemma_q4_q8.md",
    "article_ollama_context_4k_8k_16k.md", "article_truthfulqa_local_4b_8b_12b.md",
    "article_llm_codegen_pytest.md", "article_digits_classic_ml.md",
    "article_digits_numpy_nn.md", "article_digits_pytorch_nn.md",
    "article_fashion_cnn_cpu_gpu.md", "article_fashion_precision.md",
    "article_fashion_batch_oom.md", "article_dataloader_bottleneck.md",
    "article_hog_svm_vs_cnn.md",
]


@dataclass(frozen=True)
class Question:
    q: str
    patterns: tuple[str, ...]
    source: str


QUESTIONS = [
    Question("4B・8B・12Bのうち生成速度が最も速かったモデルと速度は？", (r"4B", r"179\.5"), "article_ollama_4b_8b_12b.md"),
    Question("12BモデルのGPU peak使用量は何GiBだった？", (r"9\.7" ,), "article_ollama_4b_8b_12b.md"),
    Question("Gemma 3 4BのQ4とQ8のファイルサイズをそれぞれ答えて。", (r"3\.(?:1095|11)", r"4\.(?:6379|64)"), "article_gemma_q4_q8.md"),
    Question("Q4とQ8の96回答中の正解数はそれぞれいくつ？", (r"79", r"85"), "article_gemma_q4_q8.md"),
    Question("context 16384条件で実際のprompt token数はいくつ？", (r"11.?244",), "article_ollama_context_4k_8k_16k.md"),
    Question("context長実験のneedle回収は全条件合計で何問中何問成功？", (r"9.{0,8}9",), "article_ollama_context_4k_8k_16k.md"),
    Question("TruthfulQAで12Bの両順序正解率は？", (r"73\.0",), "article_truthfulqa_local_4b_8b_12b.md"),
    Question("TruthfulQA実験で生成した総response数はいくつ？", (r"444",), "article_truthfulqa_local_4b_8b_12b.md"),
    Question("code生成でQwen 3 8Bは初回と修正後に何問合格した？", (r"9.{0,18}10",), "article_llm_codegen_pytest.md"),
    Question("code生成で4Bの修正後合格数はいくつ？", (r"7.{0,8}10",), "article_llm_codegen_pytest.md"),
    Question("Digitsの小型PyTorch NNでCPUとCUDAの平均正解率は？", (r"96\.44",), "article_digits_pytorch_nn.md"),
    Question("Digitsの小型NNではCPUとCUDAのどちらが何倍速かった？", (r"CPU", r"4\.38"), "article_digits_pytorch_nn.md"),
    Question("FashionMNIST CNNのend-to-endでGPUはCPUの何倍速かった？", (r"2\.02",), "article_fashion_cnn_cpu_gpu.md"),
    Question("FashionMNIST CNNのCUDA peak allocatedはいくつ？", (r"116\.5",), "article_fashion_cnn_cpu_gpu.md"),
    Question("precision比較でFP16 AMPはFP32より何%速い、または遅い？", (r"8(?:\.0)?%?.{0,12}遅",), "article_fashion_precision.md"),
    Question("batch 256は最大throughputの何%に達した？", (r"93(?:\.1)?",), "article_fashion_batch_oom.md"),
    Question("batch 8192のpeak allocatedは約何MiB？", (r"2.?704",), "article_fashion_batch_oom.md"),
    Question("augmentationありDataLoaderで0 workerのthroughputはいくつ？", (r"8.?830",), "article_dataloader_bottleneck.md"),
    Question("HOG+SVMとCNNのtest accuracyをそれぞれ答えて。", (r"88\.85", r"87\.00"), "article_hog_svm_vs_cnn.md"),
    Question("HOG+SVMに対しCNN推論はend-to-endで約何倍速かった？", (r"15\.8",), "article_hog_svm_vs_cnn.md"),
]


def chunks_from_markdown(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"(?=^##\s+)", text, flags=re.M)
    chunks = []
    for i, part in enumerate(parts):
        clean = re.sub(r"!\[[^]]*\]\([^)]*\)", "", part).strip()
        if clean:
            chunks.append({"id": f"{path.name}#{i}", "source": path.name, "text": clean[:3500]})
    return chunks


def ask(prompt: str) -> tuple[str, float]:
    payload = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 120},
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.load(response)
    return data["response"].strip(), time.perf_counter() - t0


def correct(answer: str, patterns: tuple[str, ...]) -> bool:
    compact = answer.replace(",", "").replace("，", "")
    return all(re.search(p, compact, flags=re.I | re.S) is not None for p in patterns)


def main() -> None:
    chunks = [c for name in ARTICLE_NAMES for c in chunks_from_markdown(CORPUS / name)]
    # Character n-grams work without a Japanese tokenizer and preserve numbers/model names.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    results = []
    for i, item in enumerate(QUESTIONS, 1):
        qvec = vectorizer.transform([item.q])
        scores = (matrix @ qvec.T).toarray().ravel()
        top_idx = np.argsort(scores)[::-1][:3]
        retrieved = [{**chunks[j], "score": float(scores[j])} for j in top_idx]
        context = "\n\n".join(f"[{x['id']}]\n{x['text']}" for x in retrieved)

        no_prompt = ("次の質問へ簡潔に答えてください。分からなければ『不明』と答えてください。"
                     "推測で数値を作らないでください。\n\n質問: " + item.q)
        rag_prompt = ("提供資料だけを根拠に質問へ簡潔に答えてください。最後に根拠のchunk IDを"
                      "角括弧で1つ以上引用してください。資料に無ければ『資料内に記載なし』と答えてください。\n\n"
                      f"資料:\n{context}\n\n質問: {item.q}")
        no_answer, no_s = ask(no_prompt)
        rag_answer, rag_s = ask(rag_prompt)
        top_sources = [x["source"] for x in retrieved]
        results.append({
            "index": i, "question": item.q, "expected_patterns": list(item.patterns),
            "expected_source": item.source, "retrieved": retrieved,
            "retrieval_hit_at_1": top_sources[0] == item.source,
            "retrieval_hit_at_3": item.source in top_sources,
            "no_rag": {"answer": no_answer, "correct": correct(no_answer, item.patterns), "wall_s": no_s},
            "rag": {"answer": rag_answer, "correct": correct(rag_answer, item.patterns), "wall_s": rag_s,
                    "cites_expected_source": item.source in rag_answer},
        })
        print(f"{i:02d}/20 retrieve@1={results[-1]['retrieval_hit_at_1']} "
              f"no={results[-1]['no_rag']['correct']} rag={results[-1]['rag']['correct']}", flush=True)

    summary = {
        "questions": len(results),
        "retrieval_hit_at_1": sum(x["retrieval_hit_at_1"] for x in results),
        "retrieval_hit_at_3": sum(x["retrieval_hit_at_3"] for x in results),
        "no_rag_correct": sum(x["no_rag"]["correct"] for x in results),
        "rag_correct": sum(x["rag"]["correct"] for x in results),
        "rag_expected_citation": sum(x["rag"]["cites_expected_source"] for x in results),
        "no_rag_mean_wall_s": float(np.mean([x["no_rag"]["wall_s"] for x in results])),
        "rag_mean_wall_s": float(np.mean([x["rag"]["wall_s"] for x in results])),
    }
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(), "model": MODEL,
        "retriever": {"type": "TF-IDF", "analyzer": "char_wb", "ngram_range": [2, 4],
                      "top_k": 3, "documents": len(ARTICLE_NAMES), "chunks": len(chunks)},
        "questions": [asdict(q) for q in QUESTIONS], "results": results, "summary": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
