from __future__ import annotations

import argparse
import contextlib
import copy
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


SEED = 42
INPUT_SHAPE = (32, 8)
REUSE_INPUT_SHAPE = (4, 8)


class SmallEvalGradModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observe orthogonal effects of model.train()/eval() and autograd mode contexts."
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Execution device (cpu or cuda). Default: cpu.",
    )
    parser.add_argument(
        "--output",
        default="experiment_eval_grad_modes.json",
        help="Relative file path for UTF-8 JSON output.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.is_absolute() or ".." in out_path.parts:
        parser.error("--output must be a relative path without '..' segments.")

    return args


def set_deterministic_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def sanitize_error_message(message: str) -> str:
    single_line = " ".join(message.strip().splitlines())
    single_line = re.sub(r"[A-Za-z]:\\[^\s'\"]+", "<path>", single_line)
    single_line = re.sub(r"/(?:home|Users|tmp|var|opt|etc|mnt|root|proc)[^\s'\"]*", "<path>", single_line)
    return single_line


def get_grad_context(grad_mode: str) -> contextlib.AbstractContextManager[Any]:
    if grad_mode == "normal":
        return contextlib.nullcontext()
    if grad_mode == "no_grad":
        return torch.no_grad()
    if grad_mode == "inference_mode":
        return torch.inference_mode()
    raise ValueError(f"Unsupported grad_mode: {grad_mode}")


def run_six_mode_combinations(device: torch.device) -> list[dict[str, Any]]:
    set_deterministic_seed(SEED, device)
    base_model = SmallEvalGradModel().to(device)
    base_state = copy.deepcopy(base_model.state_dict())
    fixed_input = torch.randn(*INPUT_SHAPE, device=device, requires_grad=False)

    if fixed_input.requires_grad:
        raise AssertionError("fixed_input.requires_grad must be False.")

    results: list[dict[str, Any]] = []
    for train_mode in ("train", "eval"):
        for grad_mode in ("normal", "no_grad", "inference_mode"):
            model = SmallEvalGradModel().to(device)
            model.load_state_dict(copy.deepcopy(base_state))

            for param in model.parameters():
                param.requires_grad_(True)
            if not all(param.requires_grad for param in model.parameters()):
                raise AssertionError("All model parameters must have requires_grad=True.")

            if train_mode == "train":
                model.train()
            else:
                model.eval()

            bn_before = int(model.bn.num_batches_tracked.item())
            set_deterministic_seed(SEED, device)

            with get_grad_context(grad_mode):
                model_training = bool(model.training)
                grad_enabled = bool(torch.is_grad_enabled())
                inference_mode_enabled = bool(torch.is_inference_mode_enabled())

                out1 = model(fixed_input)
                out2 = model(fixed_input)

                first_output_requires_grad = bool(out1.requires_grad)
                first_output_has_grad_fn = out1.grad_fn is not None
                first_output_grad_fn = (
                    type(out1.grad_fn).__name__ if out1.grad_fn is not None else None
                )
                outputs_equal = bool(torch.equal(out1, out2))
                first_output_row0 = [
                    round(float(val), 6) for val in out1[0].detach().cpu().tolist()
                ]

            bn_after = int(model.bn.num_batches_tracked.item())
            bn_delta = bn_after - bn_before

            expected_bn_delta = 2 if train_mode == "train" else 0
            expected_requires_grad = grad_mode == "normal"
            expected_training = train_mode == "train"
            expected_grad_enabled = grad_mode == "normal"
            expected_inference_enabled = grad_mode == "inference_mode"

            self_check_passed = (
                bn_delta == expected_bn_delta
                and first_output_requires_grad == expected_requires_grad
                and first_output_has_grad_fn == expected_requires_grad
                and model_training == expected_training
                and grad_enabled == expected_grad_enabled
                and inference_mode_enabled == expected_inference_enabled
            )
            if not self_check_passed:
                raise AssertionError(
                    f"Self-check failed for case ({train_mode}, {grad_mode}): "
                    f"bn_delta={bn_delta}, requires_grad={first_output_requires_grad}, "
                    f"has_grad_fn={first_output_has_grad_fn}"
                )

            results.append(
                {
                    "model_mode": train_mode,
                    "grad_mode": grad_mode,
                    "model_training": model_training,
                    "is_grad_enabled": grad_enabled,
                    "is_inference_mode_enabled": inference_mode_enabled,
                    "first_output_requires_grad": first_output_requires_grad,
                    "first_output_has_grad_fn": first_output_has_grad_fn,
                    "first_output_grad_fn": first_output_grad_fn,
                    "outputs_equal": outputs_equal,
                    "bn_num_batches_tracked_before": bn_before,
                    "bn_num_batches_tracked_after": bn_after,
                    "bn_num_batches_tracked_delta": bn_delta,
                    "first_output_row0": first_output_row0,
                    "self_check_passed": self_check_passed,
                }
            )

    return results


def run_tensor_reuse_subexperiment(device: torch.device) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # 1. Tensor created in torch.no_grad(), consumed outside by a trainable Linear layer
    set_deterministic_seed(SEED, device)
    with torch.no_grad():
        no_grad_features = torch.randn(*REUSE_INPUT_SHAPE, device=device) * 2
    head_no_grad = nn.Linear(8, 1).to(device)
    loss_no_grad = head_no_grad(no_grad_features)
    loss_no_grad.sum().backward()
    no_grad_grad_ready = (
        head_no_grad.weight.grad is not None and head_no_grad.bias.grad is not None
    )
    if not no_grad_grad_ready:
        raise AssertionError("Expected backward() to succeed for tensor created in no_grad.")
    cases.append(
        {
            "case": "no_grad_direct",
            "is_inference_tensor": bool(no_grad_features.is_inference()),
            "backward_succeeded": True,
            "weight_grad_populated": True,
            "error_type": None,
            "error_message": None,
        }
    )

    # 2. Tensor created in torch.inference_mode(), consumed directly outside (intended failure)
    set_deterministic_seed(SEED, device)
    with torch.inference_mode():
        inf_features = torch.randn(*REUSE_INPUT_SHAPE, device=device) * 2
    head_inf = nn.Linear(8, 1).to(device)
    inf_error_type: str | None = None
    inf_error_message: str | None = None
    try:
        loss_inf = head_inf(inf_features)
        loss_inf.sum().backward()
    except RuntimeError as exc:
        inf_error_type = type(exc).__name__
        inf_error_message = sanitize_error_message(str(exc))

    if inf_error_type is None:
        raise AssertionError(
            "Expected RuntimeError when backward() uses an inference tensor directly without clone()."
        )
    cases.append(
        {
            "case": "inference_mode_direct",
            "is_inference_tensor": bool(inf_features.is_inference()),
            "backward_succeeded": False,
            "weight_grad_populated": head_inf.weight.grad is not None,
            "error_type": inf_error_type,
            "error_message": inf_error_message,
        }
    )

    # 3. Tensor created in torch.inference_mode(), cloned outside before autograd use
    set_deterministic_seed(SEED, device)
    with torch.inference_mode():
        inf_features_for_clone = torch.randn(*REUSE_INPUT_SHAPE, device=device) * 2
    cloned_features = inf_features_for_clone.clone()
    head_cloned = nn.Linear(8, 1).to(device)
    loss_cloned = head_cloned(cloned_features)
    loss_cloned.sum().backward()
    cloned_grad_ready = (
        head_cloned.weight.grad is not None and head_cloned.bias.grad is not None
    )
    if not cloned_grad_ready or cloned_features.is_inference():
        raise AssertionError("Expected cloned inference tensor to work in autograd.")
    cases.append(
        {
            "case": "inference_mode_cloned_outside",
            "original_is_inference_tensor": bool(inf_features_for_clone.is_inference()),
            "is_inference_tensor": bool(cloned_features.is_inference()),
            "backward_succeeded": True,
            "weight_grad_populated": True,
            "error_type": None,
            "error_message": None,
        }
    )

    return cases


def print_compact_tables(
    mode_cases: list[dict[str, Any]], reuse_cases: list[dict[str, Any]]
) -> None:
    print(
        "model_mode | grad_mode      | training | grad_en | inf_en | out_req_grad | grad_fn | equal | bn_delta | check"
    )
    print("-" * 102)
    for row in mode_cases:
        print(
            f"{row['model_mode']:<10} | "
            f"{row['grad_mode']:<14} | "
            f"{str(row['model_training']):<8} | "
            f"{str(row['is_grad_enabled']):<7} | "
            f"{str(row['is_inference_mode_enabled']):<6} | "
            f"{str(row['first_output_requires_grad']):<12} | "
            f"{str(row['first_output_has_grad_fn']):<7} | "
            f"{str(row['outputs_equal']):<5} | "
            f"{row['bn_num_batches_tracked_delta']:>8} | "
            f"{str(row['self_check_passed']):<5}"
        )

    print()
    print("reuse_case                   | is_inference | backward_ok | error_type")
    print("-" * 74)
    for row in reuse_cases:
        print(
            f"{row['case']:<28} | "
            f"{str(row['is_inference_tensor']):<12} | "
            f"{str(row['backward_succeeded']):<11} | "
            f"{str(row['error_type'])}"
        )


def main() -> int:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Error: --device cuda requested, but CUDA is not available.", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    mode_cases = run_six_mode_combinations(device)
    reuse_cases = run_tensor_reuse_subexperiment(device)

    payload: dict[str, Any] = {
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": str(device),
            "gpu_name": gpu_name,
        },
        "config": {
            "seed": SEED,
            "input_shape": list(INPUT_SHAPE),
            "reuse_input_shape": list(REUSE_INPUT_SHAPE),
            "architecture": [
                "Linear(8, 8)",
                "BatchNorm1d(8)",
                "ReLU()",
                "Dropout(p=0.5)",
                "Linear(8, 2)",
            ],
            "input_requires_grad": False,
            "parameters_requires_grad": True,
        },
        "mode_combinations": mode_cases,
        "tensor_reuse_cases": reuse_cases,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print_compact_tables(mode_cases, reuse_cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())

