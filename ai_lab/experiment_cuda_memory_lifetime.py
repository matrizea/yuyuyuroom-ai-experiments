"""Small, bounded CUDA reference-lifetime experiment; no downloads or training."""
import argparse
import gc
import json
import platform
from pathlib import Path

import torch


def measure(label):
    torch.cuda.synchronize()
    return {"label": label, "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved()}


def clean():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def references():
    clean()
    rows = [measure("start")]
    tensor = torch.ones((8192, 8192), device="cuda", dtype=torch.float32)
    rows.append(measure("tensor_256MiB"))
    torch.cuda.empty_cache()
    rows.append(measure("empty_cache_while_live"))
    saved = [tensor]
    del tensor
    gc.collect()
    torch.cuda.empty_cache()
    rows.append(measure("del_tensor_list_still_live"))
    saved.clear()
    gc.collect()
    rows.append(measure("clear_last_reference"))
    torch.cuda.empty_cache()
    rows.append(measure("empty_cache_after_clear"))
    initial = rows[0]["allocated_bytes"]
    assert rows[1]["allocated_bytes"] - initial == 256 * 1024**2
    assert rows[2]["allocated_bytes"] == rows[1]["allocated_bytes"]
    assert rows[3]["allocated_bytes"] == rows[1]["allocated_bytes"]
    assert rows[4]["allocated_bytes"] == initial
    assert rows[5]["reserved_bytes"] <= rows[4]["reserved_bytes"]
    return rows


def history(mode):
    clean()
    x = torch.ones((1024, 1024), device="cuda", requires_grad=True)
    rows = [measure("iteration_0")]
    saved = []
    for i in range(10):
        if mode == "no_grad_cpu":
            with torch.no_grad():
                output = torch.sigmoid(x * 2)
        else:
            output = torch.sigmoid(x * 2)
        loss = output.mean()
        if mode == "loss_tensor":
            saved.append(loss)
        elif mode == "loss_item":
            saved.append(loss.item())
        elif mode == "output_detach_gpu":
            saved.append(output.detach())
        elif mode in ("output_detach_cpu", "no_grad_cpu"):
            saved.append(output.detach().cpu())
        else:
            raise ValueError(mode)
        del output, loss
        rows.append(measure(f"iteration_{i+1}"))
    sample = saved[0]
    detail = {"stored_type": type(sample).__name__,
              "device": str(sample.device) if isinstance(sample, torch.Tensor) else None,
              "requires_grad": sample.requires_grad if isinstance(sample, torch.Tensor) else False,
              "grad_fn": type(sample.grad_fn).__name__ if isinstance(sample, torch.Tensor) and sample.grad_fn else None}
    del sample
    saved.clear()
    del x
    clean()
    end = measure("cleaned")
    growth = rows[-1]["allocated_bytes"] - rows[0]["allocated_bytes"]
    if mode in ("loss_item", "output_detach_cpu", "no_grad_cpu"):
        assert growth == 0, (mode, growth)
    else:
        assert growth >= 40 * 1024**2, (mode, growth)
    assert end["allocated_bytes"] == 0
    return {"mode": mode, "rows": rows, "stored": detail, "growth_bytes": growth, "end": end}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: check the active Python and CUDA-enabled PyTorch first.")
    clean()
    free, total = torch.cuda.mem_get_info()
    if free < 1024**3:
        raise SystemExit("Less than 1 GiB free; close your own GPU workloads before running.")
    torch.manual_seed(20261024)
    result = {"environment": {"python": platform.python_version(), "platform": platform.platform(),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "device": torch.cuda.get_device_name(), "allocator": torch.cuda.memory.get_allocator_backend(),
              "free_bytes_before": free, "total_bytes": total}, "seed": 20261024, "repeats": []}
    if result["environment"]["allocator"] != "native":
        raise SystemExit("This example checks native allocator behavior; unset custom allocator configuration.")
    for repeat in range(3):
        ref = references()
        modes = [history(mode) for mode in ("loss_tensor", "loss_item", "output_detach_gpu", "output_detach_cpu", "no_grad_cpu")]
        result["repeats"].append({"repeat": repeat+1, "references": ref, "histories": modes})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["environment"], ensure_ascii=False))
    for row in result["repeats"][0]["references"]:
        print(f'{row["label"]:32s} allocated={row["allocated_bytes"]/1024**2:8.2f} MiB reserved={row["reserved_bytes"]/1024**2:8.2f} MiB')
    for row in result["repeats"][0]["histories"]:
        print(f'{row["mode"]:32s} growth={row["growth_bytes"]/1024**2:8.3f} MiB')
    print("PASS: all checks passed in 3 repetitions")


if __name__ == "__main__":
    main()
