"""Generate Python functions with local LLMs and verify them using pytest."""

from __future__ import annotations

import ast
import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import experiment_ollama_model_sizes as base


MODELS = ["gemma3:4b-it-q4_K_M", "qwen3:8b", "gemma3:12b"]
TASKS = [
    {
        "name": "is_palindrome",
        "spec": "Implement is_palindrome(text). Ignore non-alphanumeric characters and case. Return bool. An empty normalized string is a palindrome.",
        "tests": '''from candidate import is_palindrome\n\ndef test_examples():\n    assert is_palindrome("A man, a plan, a canal: Panama!") is True\n    assert is_palindrome("race a car") is False\n    assert is_palindrome("") is True\n    assert is_palindrome("., !") is True\n    assert is_palindrome("No 'x' in Nixon") is True\n''',
    },
    {
        "name": "two_sum",
        "spec": "Implement two_sum(nums, target). Exactly one pair exists. Return the two distinct zero-based indices as an increasing list. Do not reuse an element.",
        "tests": '''from candidate import two_sum\n\ndef test_cases():\n    assert two_sum([2, 7, 11, 15], 9) == [0, 1]\n    assert two_sum([3, 2, 4], 6) == [1, 2]\n    assert two_sum([3, 3], 6) == [0, 1]\n    assert two_sum([-3, 4, 3, 90], 0) == [0, 2]\n''',
    },
    {
        "name": "rle_encode",
        "spec": "Implement rle_encode(text). Return consecutive runs as a list of (character, count) tuples. Preserve case. Empty input returns [].",
        "tests": '''from candidate import rle_encode\n\ndef test_cases():\n    assert rle_encode("aaabbc") == [("a", 3), ("b", 2), ("c", 1)]\n    assert rle_encode("") == []\n    assert rle_encode("Aaa") == [("A", 1), ("a", 2)]\n    assert rle_encode("x") == [("x", 1)]\n''',
    },
    {
        "name": "merge_intervals",
        "spec": "Implement merge_intervals(intervals). Each interval is [start,end] with start<=end. Sort and merge overlapping or touching intervals. Return new lists and do not mutate the input.",
        "tests": '''from candidate import merge_intervals\n\ndef test_cases():\n    x = [[1,3],[2,6],[8,10],[10,12],[15,18]]\n    before = [r[:] for r in x]\n    assert merge_intervals(x) == [[1,6],[8,12],[15,18]]\n    assert x == before\n    assert merge_intervals([]) == []\n    assert merge_intervals([[5,5]]) == [[5,5]]\n    assert merge_intervals([[-3,-1],[-2,2]]) == [[-3,2]]\n''',
    },
    {
        "name": "balanced_brackets",
        "spec": "Implement balanced_brackets(text) for (), [], and {}. Other characters are ignored. Return bool.",
        "tests": '''from candidate import balanced_brackets\n\ndef test_cases():\n    assert balanced_brackets("([]{})") is True\n    assert balanced_brackets("([)]") is False\n    assert balanced_brackets("abc{d[e](f)}") is True\n    assert balanced_brackets(")(") is False\n    assert balanced_brackets("") is True\n    assert balanced_brackets("((") is False\n''',
    },
    {
        "name": "rotate_matrix",
        "spec": "Implement rotate_matrix(matrix). Rotate a non-ragged rectangular list-of-lists 90 degrees clockwise. Return a new matrix. [] returns []. Do not mutate input.",
        "tests": '''from candidate import rotate_matrix\n\ndef test_cases():\n    x = [[1,2,3],[4,5,6]]\n    assert rotate_matrix(x) == [[4,1],[5,2],[6,3]]\n    assert x == [[1,2,3],[4,5,6]]\n    assert rotate_matrix([[1],[2],[3]]) == [[3,2,1]]\n    assert rotate_matrix([[1,2]]) == [[1],[2]]\n    assert rotate_matrix([]) == []\n''',
    },
    {
        "name": "longest_common_prefix",
        "spec": "Implement longest_common_prefix(strings). Return the longest prefix shared by every string. [] returns ''. Matching is case-sensitive.",
        "tests": '''from candidate import longest_common_prefix\n\ndef test_cases():\n    assert longest_common_prefix(["flower","flow","flight"]) == "fl"\n    assert longest_common_prefix(["dog","racecar","car"]) == ""\n    assert longest_common_prefix([]) == ""\n    assert longest_common_prefix(["solo"]) == "solo"\n    assert longest_common_prefix(["Case","case"]) == ""\n    assert longest_common_prefix(["", "abc"]) == ""\n''',
    },
    {
        "name": "top_k_frequent",
        "spec": "Implement top_k_frequent(nums, k). Return k distinct integers ordered by descending frequency; break ties by smaller integer first. Assume 0<=k<=number of distinct values.",
        "tests": '''from candidate import top_k_frequent\n\ndef test_cases():\n    assert top_k_frequent([1,1,1,2,2,3], 2) == [1,2]\n    assert top_k_frequent([4,4,1,1,2,2], 3) == [1,2,4]\n    assert top_k_frequent([-1,-1,2,2,3], 2) == [-1,2]\n    assert top_k_frequent([5], 0) == []\n''',
    },
    {
        "name": "moving_average",
        "spec": "Implement moving_average(values, window). Return arithmetic means for every contiguous window as floats. Raise ValueError if window<=0 or window>len(values).",
        "tests": '''import pytest\nfrom candidate import moving_average\n\ndef test_values():\n    assert moving_average([1,2,3,4], 2) == [1.5,2.5,3.5]\n    assert moving_average([2,4], 2) == [3.0]\n    assert moving_average([-1,1,3], 1) == [-1.0,1.0,3.0]\n\n@pytest.mark.parametrize("window", [0,-1,4])\ndef test_invalid(window):\n    with pytest.raises(ValueError):\n        moving_average([1,2,3], window)\n''',
    },
    {
        "name": "shortest_path_grid",
        "spec": "Implement shortest_path_grid(grid, start, goal). grid is rectangular with 0=open and 1=blocked; start/goal are (row,col). Move four directions. Return minimum edge count, or -1 if unreachable. Return -1 if start or goal is blocked.",
        "tests": '''from candidate import shortest_path_grid\n\ndef test_cases():\n    grid = [[0,0,1],[1,0,0],[0,0,0]]\n    assert shortest_path_grid(grid,(0,0),(2,2)) == 4\n    assert shortest_path_grid([[0]],(0,0),(0,0)) == 0\n    assert shortest_path_grid([[1]],(0,0),(0,0)) == -1\n    assert shortest_path_grid([[0,1],[1,0]],(0,0),(1,1)) == -1\n    assert shortest_path_grid([[0,0,0]],(0,2),(0,0)) == 2\n''',
    },
]

DANGEROUS_NAMES = {"eval", "exec", "open", "compile", "__import__", "input", "globals", "locals", "vars", "getattr", "setattr", "delattr", "breakpoint", "help"}


def extract_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    return (fenced.group(1) if fenced else text).strip() + "\n"


def validate_code(code: str, expected_name: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return False, f"SyntaxError: {error}"
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) != 1 or functions[0].name != expected_name:
        return False, f"Expected exactly one function named {expected_name}"
    if any(not isinstance(node, (ast.FunctionDef, ast.Expr)) or (isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant)) for node in tree.body):
        return False, "Only a function definition and optional module docstring are allowed"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef)):
            return False, f"Blocked AST node: {type(node).__name__}"
        if isinstance(node, ast.Name) and node.id in DANGEROUS_NAMES:
            return False, f"Blocked name: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"Blocked dunder attribute: {node.attr}"
    return True, "ok"


def generate(model: str, prompt: str) -> dict:
    started = time.perf_counter()
    response = base.api_json(
        "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 512},
        },
    )
    return {"text": response.get("response", ""), "wall_s": time.perf_counter() - started, "eval_count": response.get("eval_count"), "eval_duration_s": response.get("eval_duration", 0) / 1e9}


def run_pytest(directory: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [os.sys.executable, "-m", "pytest", "-q", "--disable-warnings", "--maxfail=1"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        return {"returncode": proc.returncode, "passed": proc.returncode == 0, "stdout": proc.stdout[-5000:], "stderr": proc.stderr[-3000:], "wall_s": time.perf_counter() - started, "timed_out": False}
    except subprocess.TimeoutExpired as error:
        return {"returncode": None, "passed": False, "stdout": (error.stdout or "")[-5000:] if isinstance(error.stdout, str) else "", "stderr": "pytest timeout after 10s", "wall_s": time.perf_counter() - started, "timed_out": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and execute LLM code only in a disposable environment. AST checks are not a security sandbox.")
    parser.add_argument("--execute-in-disposable-environment", action="store_true")
    args = parser.parse_args()
    if not args.execute_in_disposable_environment:
        raise SystemExit("Refusing to execute generated code. Use a disposable VM/container with no secrets or host mounts, then pass --execute-in-disposable-environment. AST filtering is not a sandbox.")
    root = Path(__file__).parent / ("generated_code_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    root.mkdir(parents=True)
    results = []
    for model in MODELS:
        base.unload(model)
        model_dir = root / model.replace(":", "_")
        for task in TASKS:
            task_dir = model_dir / task["name"]
            task_dir.mkdir(parents=True)
            prompt = (
                "Output only Python source code, with no Markdown fences or explanation. "
                "Define exactly one function. Do not import anything and do not access files, network, processes, environment variables, or user input.\n"
                f"Task: {task['spec']}"
            )
            first = generate(model, prompt)
            code = extract_code(first["text"])
            safe, safety_message = validate_code(code, task["name"])
            (task_dir / "candidate.py").write_text(code, encoding="utf-8")
            (task_dir / "test_candidate.py").write_text(task["tests"], encoding="utf-8")
            first_test = run_pytest(task_dir) if safe else {"returncode": None, "passed": False, "stdout": "", "stderr": safety_message, "wall_s": 0.0, "timed_out": False}
            repaired = None
            repair_test = None
            if not first_test["passed"]:
                feedback = (first_test["stdout"] + "\n" + first_test["stderr"])[-5000:]
                repair_prompt = (
                    "Output only corrected Python source code, with no Markdown fences or explanation. "
                    "Define exactly one function; no imports or external access.\n"
                    f"Original task: {task['spec']}\nCurrent code:\n{code}\npytest/safety feedback:\n{feedback}\nFix the implementation."
                )
                repaired = generate(model, repair_prompt)
                repair_code = extract_code(repaired["text"])
                repair_safe, repair_safety = validate_code(repair_code, task["name"])
                (task_dir / "candidate_repaired.py").write_text(repair_code, encoding="utf-8")
                (task_dir / "candidate.py").write_text(repair_code, encoding="utf-8")
                repair_test = run_pytest(task_dir) if repair_safe else {"returncode": None, "passed": False, "stdout": "", "stderr": repair_safety, "wall_s": 0.0, "timed_out": False}
            results.append({"model": model, "task": task["name"], "spec": task["spec"], "first_generation": first, "first_code": code, "first_safe": safe, "first_safety_message": safety_message, "first_test": first_test, "repair_generation": repaired, "repair_test": repair_test, "final_passed": first_test["passed"] or bool(repair_test and repair_test["passed"])})
            print(f"{model:25s} {task['name']:22s} first={first_test['passed']} final={results[-1]['final_passed']}")
        base.unload(model)

    summary = {}
    for model in MODELS:
        rows = [row for row in results if row["model"] == model]
        summary[model] = {
            "first_pass": sum(row["first_test"]["passed"] for row in rows),
            "final_pass": sum(row["final_passed"] for row in rows),
            "tasks": len(rows),
            "repairs_attempted": sum(row["repair_test"] is not None for row in rows),
            "repairs_succeeded": sum(bool(row["repair_test"] and row["repair_test"]["passed"]) for row in rows),
            "safety_rejections": sum(not row["first_safe"] for row in rows),
            "timeouts": sum(row["first_test"]["timed_out"] or bool(row["repair_test"] and row["repair_test"]["timed_out"]) for row in rows),
        }
    output = {"created_at": datetime.now(timezone.utc).isoformat(), "models": MODELS, "tasks": [{"name": t["name"], "spec": t["spec"], "tests": t["tests"]} for t in TASKS], "safety_policy": {"dangerous_names": sorted(DANGEROUS_NAMES), "timeout_s": 10, "imports_allowed": False}, "results": results, "summary": summary}
    path = Path(__file__).parent / "results" / "llm_codegen_pytest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
