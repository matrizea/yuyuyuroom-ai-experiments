"""Compare Gemma 3 4B Q4_K_M and Q8_0 with Ollama.

Measures cold/warm latency, generation speed, GPU usage and a small fixed
computer-science multiple-choice probe.  This imports the already-tested
streaming and GPU monitor helpers from experiment_ollama_model_sizes.py.
"""

from __future__ import annotations

import json
import re
import statistics
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import experiment_ollama_model_sizes as base


MODELS = ["gemma3:4b-it-q4_K_M", "gemma3:4b-it-q8_0"]
DISPLAY = {MODELS[0]: "Q4_K_M", MODELS[1]: "Q8_0"}
CYCLES = 3
QUESTIONS = [
    ("勾配降下法で学習率が大きすぎる場合に起きやすい現象は？", ["常に一回で最小値へ着く", "損失が振動・発散する", "勾配が必ず0になる", "訓練dataが増える"], "B"),
    ("多class分類のlogitから確率分布を作る関数は？", ["ReLU", "softmax", "dropout", "max pooling"], "B"),
    ("test dataを使ってhyperparameterを選ぶ問題は？", ["data leakage", "underflow", "deadlock", "cache miss"], "A"),
    ("training lossは低いがvalidation lossが上昇する状態は？", ["underfitting", "overfitting", "normalization", "quantization"], "B"),
    ("見逃しを減らすことを特に重視する指標は？", ["recall", "precision", "specificity", "MSE"], "A"),
    ("PCAの主成分が選ぶ方向は？", ["分散が最大", "平均が最大", "sample数が最大", "class数が最大"], "A"),
    ("Batch Normalizationが推論時に通常使う統計は？", ["現在の1 sampleだけ", "保存した移動平均", "test label", "乱数"], "B"),
    ("ReLU(x)の定義は？", ["1/(1+exp(-x))", "max(0,x)", "x^2", "log(x)"], "B"),
    ("kernel 3、stride 1、padding 0を幅28へ適用した出力幅は？", ["24", "26", "28", "30"], "B"),
    ("L2 regularizationが直接penaltyへ加えるものは？", ["weightの絶対値和", "weightの二乗和", "gradientの符号", "labelのentropy"], "B"),
    ("公平なclass分類accuracy評価でtest setに行うべきことは？", ["学習にも使う", "最終評価まで分離する", "labelを削除する", "train setへ複製する"], "B"),
    ("binary searchの時間計算量は？", ["O(1)", "O(log n)", "O(n)", "O(n^2)"], "B"),
    ("LIFOのdata構造は？", ["queue", "stack", "heap", "graph"], "B"),
    ("重みなしgraphの最短辺数経路を求める基本algorithmは？", ["DFS", "BFS", "quick sort", "gradient descent"], "B"),
    ("hash tableで異なるkeyが同じindexになる現象は？", ["collision", "pagination", "inheritance", "deadlock"], "A"),
    ("TCPがUDPと比べて提供する性質は？", ["到着順と再送による信頼性", "常に低遅延", "接続不要", "packet lossの強制"], "A"),
    ("deadlockの必要条件に含まれないものは？", ["相互排他", "hold and wait", "循環待ち", "無限のmemory"], "D"),
    ("virtual memoryの主な目的は？", ["物理memoryを抽象化したaddress空間を提供", "CPU clockを上げる", "networkを暗号化", "GPU coreを増やす"], "A"),
    ("Gitで作業treeの変更状態を確認するcommandは？", ["git status", "git clone", "git init", "git tag"], "A"),
    ("IEEE 754 floatで0.1+0.2が厳密に0.3にならない主因は？", ["2進で有限表現できない丸め", "CPUが10進演算できない禁止", "整数overflow", "network latency"], "A"),
    ("confusion matrixのtrue positiveは？", ["実際positiveをpositiveと予測", "実際negativeをpositiveと予測", "実際positiveをnegativeと予測", "実際negativeをnegativeと予測"], "A"),
    ("ROC-AUC=0.5が示す典型的状態は？", ["完全分類", "random ranking相当", "必ずoverfit", "lossが0"], "B"),
    ("SQLのPRIMARY KEYに求められる性質は？", ["rowを一意に識別", "必ず文字列", "重複を推奨", "NULLだけを格納"], "A"),
    ("Bayesの定理でP(A|B)に比例する積は？", ["P(B|A)P(A)", "P(A|B)P(B)", "P(A)+P(B)", "P(A)-P(B)"], "A"),
]


def ask_mcq(model: str, question: str, choices: list[str]) -> dict:
    letters = "ABCD"
    choice_text = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, choices))
    prompt = (
        "次の四択問題に答えてください。説明は書かず、正解の英大文字A/B/C/Dを1文字だけ出力してください。\n"
        f"問題: {question}\n{choice_text}\n回答:"
    )
    response = base.api_json(
        "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 8},
        },
    )
    text = response.get("response", "").strip()
    match = re.search(r"\b([ABCD])\b", text.upper())
    return {
        "question": question,
        "choices": choices,
        "response": text,
        "parsed": match.group(1) if match else None,
        "eval_count": response.get("eval_count"),
    }


def rotate_choices(choices: list[str], correct: str, rotation: int) -> tuple[list[str], str]:
    """Rotate choices so every question is tested with its answer at A/B/C/D."""
    rotated = choices[rotation:] + choices[:rotation]
    correct_index = (ord(correct) - ord("A") - rotation) % 4
    return rotated, "ABCD"[correct_index]


def main() -> None:
    base.PROMPT = (
        "情報科学部のB4学生へ、機械学習におけるdata leakageを説明してください。"
        "訓練・validation・testの役割、典型例、検出方法、予防策を見出し付きで700字以上にまとめてください。"
    )
    base.OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 256}
    all_runs = []
    details = {}
    quality = {}
    tags = base.api_json("/api/tags").get("models", [])

    for model in MODELS:
        details[model] = base.api_json("/api/show", {"model": model})
        for repeat in range(1, CYCLES + 1):
            base.unload(model)
            all_runs.append(base.run_streamed(model, "cold", repeat))
            all_runs.append(base.run_streamed(model, "warm", repeat))

        answers = []
        for question_index, (question, choices, correct) in enumerate(QUESTIONS):
            for rotation in range(4):
                rotated_choices, rotated_correct = rotate_choices(choices, correct, rotation)
                row = ask_mcq(model, question, rotated_choices)
                row["question_index"] = question_index
                row["rotation"] = rotation
                row["correct"] = rotated_correct
                row["is_correct"] = row["parsed"] == rotated_correct
                row["selected_text"] = (
                    rotated_choices[ord(row["parsed"]) - ord("A")] if row["parsed"] else None
                )
                answers.append(row)
        consistent_questions = 0
        for question_index in range(len(QUESTIONS)):
            selected = {
                row["selected_text"]
                for row in answers
                if row["question_index"] == question_index
            }
            if len(selected) == 1:
                consistent_questions += 1
        quality[model] = {
            "correct": sum(row["is_correct"] for row in answers),
            "total": len(answers),
            "order_consistent_questions": consistent_questions,
            "questions": len(QUESTIONS),
            "answers": answers,
        }
        base.unload(model)

    summary = {}
    for model in MODELS:
        warm = [r for r in all_runs if r["model"] == model and r["phase"] == "warm"]
        cold = [r for r in all_runs if r["model"] == model and r["phase"] == "cold"]
        tag = next(row for row in tags if row["name"] == model)
        summary[model] = {
            "display": DISPLAY[model],
            "file_size_bytes": tag["size"],
            "file_size_gib": tag["size"] / 1024**3,
            "quantization_level": details[model]["details"]["quantization_level"],
            "cold_load_median_s": statistics.median(r["load_duration_s"] for r in cold),
            "warm_ttfc_median_s": statistics.median(r["time_to_first_content_s"] for r in warm),
            "warm_tokens_per_s_median": statistics.median(r["tokens_per_s"] for r in warm),
            "warm_tokens_per_s_range": [min(r["tokens_per_s"] for r in warm), max(r["tokens_per_s"] for r in warm)],
            "peak_gpu_used_mib": max(r["gpu_peak_used_mib"] for r in all_runs if r["model"] == model),
            "mcq_correct": quality[model]["correct"],
            "mcq_total": quality[model]["total"],
            "order_consistent_questions": quality[model]["order_consistent_questions"],
        }

    q4_text = next(r["response"] for r in all_runs if r["model"] == MODELS[0] and r["phase"] == "warm")
    q8_text = next(r["response"] for r in all_runs if r["model"] == MODELS[1] and r["phase"] == "warm")
    disagreements = []
    for index in range(len(QUESTIONS) * 4):
        a = quality[MODELS[0]]["answers"][index]
        b = quality[MODELS[1]]["answers"][index]
        if a["parsed"] != b["parsed"]:
            disagreements.append({"index": index, "question_index": a["question_index"], "rotation": a["rotation"], "question": a["question"], "q4": a["parsed"], "q8": b["parsed"], "correct": a["correct"]})

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": MODELS,
        "prompt": base.PROMPT,
        "options": base.OPTIONS,
        "cycles": CYCLES,
        "summary": summary,
        "runs": all_runs,
        "quality_probe": quality,
        "free_response_character_similarity": SequenceMatcher(None, q4_text, q8_text).ratio(),
        "free_responses_exact_match": q4_text == q8_text,
        "mcq_disagreements": disagreements,
        "model_details": details,
    }
    path = Path(__file__).parent / "results" / "gemma_q4_q8.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}")
    print(json.dumps({"summary": summary, "similarity": output["free_response_character_similarity"], "disagreements": disagreements}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
