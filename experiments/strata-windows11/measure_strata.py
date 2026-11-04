"""Measure a local Strata Chat Completions endpoint (Python standard library only)."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request


CASES = [
    (
        "japanese",
        "VRAMとメインメモリの違いを、GPUを初めて使う人に日本語で説明してください。"
        "それぞれの役割と、VRAMが足りないときに起こることを含め、300〜400字にしてください。",
        512,
    ),
    (
        "python",
        "Pythonの関数 xor_checksum(data: bytes) -> int を実装してください。"
        "全バイトの排他的論理和を返し、空なら0を返します。"
        "外部ライブラリは使わず、関数だけをコードブロックに出力してください。",
        256,
    ),
    (
        "moe",
        "MoE型の大規模言語モデルが、全パラメータを毎回計算しない仕組みを日本語で説明してください。"
        "ルーター、エキスパート、総パラメータ数と有効パラメータ数の違いを含め、"
        "400〜500字にしてください。見出しと箇条書きは使わないでください。",
        640,
    ),
]


def get_json(url, headers):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def run_case(base_url, model, case, headers, repeat):
    name, prompt, maximum = case
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": maximum,
        "temperature": 0,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_content = None
    chunks = []
    usage, timings, finish = None, None, None
    with urllib.request.urlopen(req, timeout=900) as response:
        for line in response:
            line = line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if value == "[DONE]":
                break
            event = json.loads(value)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                usage = event["usage"]
            if event.get("timings"):
                timings = event["timings"]
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    if first_content is None:
                        first_content = time.perf_counter() - started
                    chunks.append(content)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    return {
        "case": name,
        "repeat": repeat,
        "request": body,
        "first_content_seconds": first_content,
        "elapsed_seconds": time.perf_counter() - started,
        "finish_reason": finish,
        "usage": usage,
        "timings": timings,
        "answer": "".join(chunks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path, default=Path("strata-results.json"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    headers = {}
    if os.environ.get("STRATA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["STRATA_API_KEY"]
    models = get_json(base_url + "/v1/models", headers)
    model = models["data"][0]["id"]
    result = {
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": model,
        "method": "serial SSE; TTFT = first nonempty content delta; engine timings retained",
        "runs": [],
    }
    result["warmup"] = run_case(base_url, model, CASES[0], headers, 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"warmup": True, "elapsed_s": result["warmup"]["elapsed_seconds"],
                      "first_content_s": result["warmup"]["first_content_seconds"]}), flush=True)
    for repeat in range(1, args.repeats + 1):
        for case in CASES:
            run = run_case(base_url, model, case, headers, repeat)
            result["runs"].append(run)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            speed = (run["timings"] or {}).get("predicted_per_second")
            print(json.dumps({
                "case": case[0], "repeat": repeat,
                "first_content_s": run["first_content_seconds"],
                "elapsed_s": run["elapsed_seconds"],
                "decode_tokens_per_s": speed,
                "finish_reason": run["finish_reason"],
            }), flush=True)
    for name, _, _ in CASES:
        runs = [r for r in result["runs"] if r["case"] == name]
        speeds = [(r["timings"] or {}).get("predicted_per_second") for r in runs]
        speeds = [s for s in speeds if s is not None]
        print(json.dumps({"case": name, "median_decode_tokens_per_s": statistics.median(speeds) if speeds else None}), flush=True)


if __name__ == "__main__":
    main()
