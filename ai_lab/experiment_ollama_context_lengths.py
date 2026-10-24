"""Measure Ollama context length costs and synthetic needle retrieval."""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import experiment_ollama_model_sizes as base


MODEL = "gemma3:4b-it-q4_K_M"
CONTEXTS = [4096, 8192, 16384]
POSITIONS = ["beginning", "middle", "end"]
TARGET_FILL = 0.70


def records(count: int) -> list[str]:
    return [
        f"LOG {i:05d} | sensor=S{i % 17:02d} | status=normal | value={(i * 37) % 1009:04d} | note=no anomaly."
        for i in range(count)
    ]


def make_prompt(count: int, needle: str, position: str) -> str:
    rows = records(count)
    index = {"beginning": 0, "middle": len(rows) // 2, "end": len(rows)}[position]
    rows.insert(index, f"IMPORTANT RECORD | SECRET_CODE={needle} | preserve this exact code.")
    return (
        "次の研究ログを読み、最後の質問に答えてください。通常recordの数値は計算不要です。\n"
        + "\n".join(rows)
        + "\nQUESTION: IMPORTANT RECORDのSECRET_CODEは何ですか。説明せずcodeだけを正確に出力してください。\nANSWER:"
    )


def calibration_tokens() -> tuple[int, float]:
    prompt = make_prompt(100, "CALIBRATION-0000", "middle")
    response = base.api_json(
        "/api/generate",
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 1},
        },
    )
    count = response["prompt_eval_count"]
    return count, count / 100


def main() -> None:
    calibration_count, tokens_per_record = calibration_tokens()
    base.unload(MODEL)
    runs = []
    for context in CONTEXTS:
        record_count = int(context * TARGET_FILL / tokens_per_record)
        for position in POSITIONS:
            needle = f"CTX{context}-{position.upper()}-7319"
            base.unload(MODEL)
            base.PROMPT = make_prompt(record_count, needle, position)
            base.OPTIONS = {
                "temperature": 0,
                "seed": 42,
                "num_ctx": context,
                "num_predict": 32,
            }
            result = base.run_streamed(MODEL, f"ctx{context}-{position}", 1)
            result["context_setting"] = context
            result["position"] = position
            result["record_count"] = record_count
            result["needle"] = needle
            result["retrieved"] = needle in result["response"]
            result["prompt_tokens_per_s"] = (
                result["prompt_eval_count"] / result["prompt_eval_duration_s"]
                if result["prompt_eval_duration_s"]
                else None
            )
            result["ttfc_after_load_s"] = max(
                0.0, result["time_to_first_content_s"] - result["load_duration_s"]
            )
            result["ollama_ps_after"] = base.api_json("/api/ps")
            runs.append(result)
    base.unload(MODEL)

    summary = {}
    for context in CONTEXTS:
        selected = [run for run in runs if run["context_setting"] == context]
        ps_models = selected[-1]["ollama_ps_after"].get("models", [])
        resident = ps_models[0] if ps_models else {}
        summary[str(context)] = {
            "prompt_eval_count_median": statistics.median(run["prompt_eval_count"] for run in selected),
            "prompt_eval_tokens_per_s_median": statistics.median(run["prompt_tokens_per_s"] for run in selected),
            "ttfc_after_load_median_s": statistics.median(run["ttfc_after_load_s"] for run in selected),
            "peak_gpu_used_mib": max(run["gpu_peak_used_mib"] for run in selected),
            "ollama_size_vram_bytes": resident.get("size_vram"),
            "ollama_context_length": resident.get("context_length"),
            "retrieved": sum(run["retrieved"] for run in selected),
            "total": len(selected),
        }

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "context_settings": CONTEXTS,
        "positions": POSITIONS,
        "target_fill": TARGET_FILL,
        "calibration_prompt_tokens": calibration_count,
        "estimated_tokens_per_record": tokens_per_record,
        "runs": runs,
        "summary": summary,
    }
    path = Path(__file__).parent / "results" / "ollama_context_lengths.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
