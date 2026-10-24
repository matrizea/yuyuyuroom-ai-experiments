#!/usr/bin/env python3
"""Collect reproducible host, Python and accelerator metadata as JSON."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    gpu = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "total_memory_gib": round(properties.total_memory / 1024**3, 2),
        }

    record = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "wsl_distro": os.environ.get("WSL_DISTRO_NAME"),
        "packages": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "cuda": {
            "available": cuda_available,
            "runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": gpu,
            "nvidia_smi": command_output([
                "/usr/lib/wsl/lib/nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
