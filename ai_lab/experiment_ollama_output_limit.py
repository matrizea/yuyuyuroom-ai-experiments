from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gemma3:4b-it-q4_K_M"
DEFAULT_LIMITS = [16, 64, 256]
DEFAULT_REPEATS = 3
MAX_ALLOWED_LIMIT = 512
MAX_ALLOWED_REPEATS = 10
TIMEOUT_SECONDS = 180
NUM_CTX = 4096
SEED = 42
TEMPERATURE = 0
KEEP_ALIVE = "5m"
PROMPT = (
    "1から80までの整数を、小さい順に半角スペースで区切って書いてください。"
    "数字とスペース以外は書かないでください。"
)
EXPECTED_INTEGERS = list(range(1, 81))


class SafeExperimentError(Exception):
    """Exception holding only safe error descriptions without URLs or local paths."""


def validate_base_url(raw_url: str | None, parser: argparse.ArgumentParser) -> str:
    if raw_url is None or not raw_url.strip():
        parser.error("Provide --base-url or set the OLLAMA_HOST environment variable.")

    candidate = raw_url.strip()
    if "://" not in candidate:
        parser.error("Base URL must explicitly specify http:// or https://.")

    parsed = urllib.parse.urlsplit(candidate)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        parser.error("Base URL must use http or https and include a valid host.")
    if parsed.username or parsed.password:
        parser.error("Base URL must not contain credentials.")

    normalized_path = parsed.path.rstrip("/")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))


def parse_args() -> tuple[argparse.Namespace, str]:
    parser = argparse.ArgumentParser(
        description="Measure Ollama output behavior across num_predict token limits."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_HOST"),
        help="Ollama server base URL (http:// or https://). Defaults to OLLAMA_HOST.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Existing Ollama model name. Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--limits",
        type=int,
        nargs="+",
        default= list(DEFAULT_LIMITS),
        help="Positive num_predict limits to test. Default: 16 64 256.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of full cycles over --limits. Default: {DEFAULT_REPEATS}.",
    )
    parser.add_argument(
        "--output",
        default="experiment_ollama_output_limit.json",
        help="Relative file path for UTF-8 JSON output.",
    )
    args = parser.parse_args()

    base_url = validate_base_url(args.base_url, parser)

    if not args.model or not args.model.strip():
        parser.error("--model must be a non-empty model tag.")

    if any(limit <= 0 or limit > MAX_ALLOWED_LIMIT for limit in args.limits):
        parser.error(
            f"--limits must contain only positive integers up to {MAX_ALLOWED_LIMIT}."
        )

    if args.repeats <= 0 or args.repeats > MAX_ALLOWED_REPEATS:
        parser.error(f"--repeats must be between 1 and {MAX_ALLOWED_REPEATS}.")

    out_path = Path(args.output)
    if out_path.is_absolute() or ".." in out_path.parts:
        parser.error("--output must be a relative path without '..' segments.")

    return args, base_url


def request_json(
    base_url: str, endpoint: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    url = f"{base_url}{endpoint}"
    headers = {"Accept": "application/json"}
    data: bytes | None = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"

    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            status = getattr(resp, "status", 200)
            if status < 200 or status >= 300:
                raise SafeExperimentError(f"HTTP error on {endpoint}: status={status}")
            raw_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        raise SafeExperimentError(
            f"HTTP error on {endpoint}: status={exc.code}"
        ) from None
    except urllib.error.URLError as exc:
        reason_type = type(exc.reason).__name__ if exc.reason is not None else "URLError"
        raise SafeExperimentError(
            f"Network error on {endpoint}: {reason_type}"
        ) from None
    except (TimeoutError, socket.timeout):
        raise SafeExperimentError(
            f"Request timed out on {endpoint} after {TIMEOUT_SECONDS}s"
        ) from None
    except OSError as exc:
        raise SafeExperimentError(
            f"Transport error on {endpoint}: {type(exc).__name__}"
        ) from None

    try:
        decoded = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafeExperimentError(
            f"Invalid JSON response from {endpoint}: {type(exc).__name__}"
        ) from None

    if not isinstance(decoded, dict):
        raise SafeExperimentError(f"Unexpected JSON root type from {endpoint}.")

    if "error" in decoded and decoded["error"]:
        raise SafeExperimentError(f"API error returned from {endpoint}.")

    return decoded


def fetch_server_and_model_metadata(base_url: str, model: str) -> dict[str, Any]:
    version_resp = request_json(base_url, "/api/version")
    version = version_resp.get("version")
    if not isinstance(version, str) or not version:
        raise SafeExperimentError("Missing valid 'version' in /api/version response.")

    show_resp = request_json(base_url, "/api/show", payload={"model": model})
    raw_model_info = show_resp.get("model_info")
    if not isinstance(raw_model_info, dict):
        raw_model_info = {}

    raw_details = show_resp.get("details")
    if not isinstance(raw_details, dict):
        raw_details = {}

    architecture = raw_model_info.get(
        "general.architecture", raw_model_info.get("architecture")
    )
    parameter_count = raw_model_info.get(
        "general.parameter_count", raw_model_info.get("parameter_count")
    )

    return {
        "ollama_version": version,
        "model_info": {
            "architecture": architecture,
            "parameter_count": parameter_count,
        },
        "details": {
            "family": raw_details.get("family"),
            "parameter_size": raw_details.get("parameter_size"),
            "quantization_level": raw_details.get("quantization_level"),
        },
    }


def run_generate_once(
    base_url: str, model: str, num_predict: int, repeat: int | None = None
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_ctx": NUM_CTX,
            "num_predict": num_predict,
        },
    }

    start_wall = time.perf_counter()
    resp = request_json(base_url, "/api/generate", payload=payload)
    wall_seconds = round(time.perf_counter() - start_wall, 6)

    done = resp.get("done")
    response_text = resp.get("response")
    if not isinstance(done, bool) or not done or not isinstance(response_text, str):
        raise SafeExperimentError(
            "Incomplete or malformed response from /api/generate."
        )

    extracted_numbers = [int(tok) for tok in re.findall(r"-?\d+", response_text)]
    matches_1_to_80 = extracted_numbers == EXPECTED_INTEGERS

    record: dict[str, Any] = {
        "num_predict": num_predict,
        "response": response_text,
        "response_char_count": len(response_text),
        "done": done,
        "done_reason": resp.get("done_reason"),
        "eval_count": resp.get("eval_count"),
        "prompt_eval_count": resp.get("prompt_eval_count"),
        "total_duration": resp.get("total_duration"),
        "load_duration": resp.get("load_duration"),
        "eval_duration": resp.get("eval_duration"),
        "wall_seconds": wall_seconds,
        "extracted_int_count": len(extracted_numbers),
        "matches_1_to_80": matches_1_to_80,
    }
    if repeat is not None:
        record = {"repeat": repeat, **record}
    return record


def print_compact_table(runs: list[dict[str, Any]]) -> None:
    print("num_predict | repeat | eval_count | done_reason | matches_1_to_80")
    print("-" * 65)
    for row in runs:
        print(
            f"{row['num_predict']:>11} | "
            f"{row['repeat']:>6} | "
            f"{str(row['eval_count']):>10} | "
            f"{str(row['done_reason']):<11} | "
            f"{str(row['matches_1_to_80'])}"
        )


def main() -> int:
    args, base_url = parse_args()

    try:
        metadata = fetch_server_and_model_metadata(base_url, args.model)
        warmup_record = run_generate_once(base_url, args.model, num_predict=1, repeat=None)

        runs: list[dict[str, Any]] = []
        for repeat_idx in range(1, args.repeats + 1):
            for limit in args.limits:
                run_record = run_generate_once(
                    base_url, args.model, num_predict=limit, repeat=repeat_idx
                )
                runs.append(run_record)

        output_data: dict[str, Any] = {
            "environment": {
                "python_version": platform.python_version(),
                "ollama_version": metadata["ollama_version"],
            },
            "model_metadata": {
                "model": args.model,
                "model_info": metadata["model_info"],
                "details": metadata["details"],
            },
            "config": {
                "model": args.model,
                "num_ctx": NUM_CTX,
                "seed": SEED,
                "temperature": TEMPERATURE,
                "keep_alive": KEEP_ALIVE,
                "limits": list(args.limits),
                "repeats": args.repeats,
                "prompt": PROMPT,
            },
            "warmup": warmup_record,
            "runs": runs,
        }

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(output_data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

        print_compact_table(runs)
        return 0
    except SafeExperimentError as exc:
        print(f"ExperimentError ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"UnexpectedError ({type(exc).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

