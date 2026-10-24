"""Benchmark 4B/8B/12B local LLMs with Ollama on the same 16 GB GPU.

The script intentionally uses only the Python standard library.  It streams
responses to measure time-to-first-content, reads Ollama's final token timing,
and samples total GPU memory via nvidia-smi while each request is running.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


API_ROOT = "http://127.0.0.1:11434"
MODELS = ["gemma3:4b", "qwen3:8b", "gemma3:12b"]
PROMPT = (
    "情報科学部の研究室に入ったばかりのB4学生へ、勾配降下法で学習率が大きすぎると"
    "学習が不安定になる理由を説明してください。損失曲面、更新式、発散の兆候、"
    "切り分け手順、改善策の5項目を見出し付きで扱い、具体例を含めて700字以上で書いてください。"
)
OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "num_ctx": 4096,
    "num_predict": 256,
}
CYCLES = 3
NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"


def api_json(path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_ROOT + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def gpu_sample() -> dict[str, int]:
    text = subprocess.check_output(
        [
            NVIDIA_SMI,
            "--query-gpu=memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used, free, util = (int(value.strip()) for value in text.split(","))
    return {"used_mib": used, "free_mib": free, "util_percent": util}


def unload(model: str) -> None:
    api_json(
        "/api/generate",
        {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
    )
    time.sleep(1.0)


def run_streamed(model: str, phase: str, repeat: int) -> dict:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "stream": True,
        "think": False,
        "keep_alive": "10m",
        "options": OPTIONS,
    }
    request = urllib.request.Request(
        API_ROOT + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    samples: list[dict] = []
    sampling = True

    def monitor() -> None:
        while sampling:
            try:
                samples.append({"elapsed_s": time.perf_counter() - start, **gpu_sample()})
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            time.sleep(0.05)

    start = time.perf_counter()
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    first_content_s = None
    chunks: list[str] = []
    final: dict = {}
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            for raw_line in response:
                event = json.loads(raw_line)
                content = event.get("response", "")
                if content and first_content_s is None:
                    first_content_s = time.perf_counter() - start
                chunks.append(content)
                if event.get("done"):
                    final = event
    finally:
        sampling = False
        monitor_thread.join(timeout=2)
    wall_s = time.perf_counter() - start

    result = {
        "model": model,
        "phase": phase,
        "repeat": repeat,
        "wall_s": wall_s,
        "time_to_first_content_s": first_content_s,
        "total_duration_s": final.get("total_duration", 0) / 1e9,
        "load_duration_s": final.get("load_duration", 0) / 1e9,
        "prompt_eval_count": final.get("prompt_eval_count"),
        "prompt_eval_duration_s": final.get("prompt_eval_duration", 0) / 1e9,
        "eval_count": final.get("eval_count"),
        "eval_duration_s": final.get("eval_duration", 0) / 1e9,
        "tokens_per_s": (
            final.get("eval_count", 0) / (final.get("eval_duration", 0) / 1e9)
            if final.get("eval_duration", 0)
            else None
        ),
        "done_reason": final.get("done_reason"),
        "response": "".join(chunks),
        "gpu_baseline_used_mib": samples[0]["used_mib"] if samples else None,
        "gpu_peak_used_mib": max((x["used_mib"] for x in samples), default=None),
        "gpu_peak_util_percent": max((x["util_percent"] for x in samples), default=None),
        "gpu_samples": samples,
    }
    print(
        f"{model:11s} {phase:4s} #{repeat}: "
        f"TTFC={first_content_s:.3f}s, load={result['load_duration_s']:.3f}s, "
        f"{result['tokens_per_s']:.1f} tok/s, peak={result['gpu_peak_used_mib']} MiB, "
        f"tokens={result['eval_count']}"
    )
    return result


def summarize(runs: list[dict]) -> dict:
    summary = {}
    for model in MODELS:
        model_runs = [row for row in runs if row["model"] == model]
        cold = [row for row in model_runs if row["phase"] == "cold"]
        warm = [row for row in model_runs if row["phase"] == "warm"]
        summary[model] = {
            "cold_load_median_s": statistics.median(
                row["load_duration_s"] for row in cold
            ),
            "cold_load_min_s": min(row["load_duration_s"] for row in cold),
            "cold_load_max_s": max(row["load_duration_s"] for row in cold),
            "cold_time_to_first_content_median_s": statistics.median(
                row["time_to_first_content_s"] for row in cold
            ),
            "warm_time_to_first_content_median_s": statistics.median(
                row["time_to_first_content_s"] for row in warm
            ),
            "warm_tokens_per_s_median": statistics.median(
                row["tokens_per_s"] for row in warm
            ),
            "warm_tokens_per_s_min": min(row["tokens_per_s"] for row in warm),
            "warm_tokens_per_s_max": max(row["tokens_per_s"] for row in warm),
            "peak_gpu_used_mib": max(row["gpu_peak_used_mib"] for row in model_runs),
            "eval_count": [row["eval_count"] for row in model_runs],
        }
    return summary


def main() -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    available = api_json("/api/tags")
    model_details = {}
    runs = []
    for model in MODELS:
        model_details[model] = api_json("/api/show", {"model": model})
        for repeat in range(1, CYCLES + 1):
            unload(model)
            runs.append(run_streamed(model, "cold", repeat))
            runs.append(run_streamed(model, "warm", repeat))
        runs[-1]["ollama_ps_after"] = api_json("/api/ps")
        unload(model)

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": PROMPT,
        "options": OPTIONS,
        "cycles": CYCLES,
        "models_from_tags": available.get("models", []),
        "model_details": model_details,
        "runs": runs,
        "summary": summarize(runs),
    }
    path = results_dir / "ollama_model_sizes.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
