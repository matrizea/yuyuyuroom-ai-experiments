"""Refresh dense metric summaries from complete checkpoints without inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from evaluate_dense import summarize
from eval_utils import ROOT

DEFAULT_DIRECTORIES = ["results_dense_nyu_cpu","results_dense_nyu_gpu",
                       "results_dense_sintel_full_cpu","results_dense_sintel_full_gpu"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dirs",type=Path,nargs="*",help="Directories containing completed checkpoint.json files; defaults to all four final runs")
    args = parser.parse_args()
    directories = args.output_dirs or [ROOT/name for name in DEFAULT_DIRECTORIES]
    loaded = []
    # Validate every run before rewriting any artifact.
    for directory in directories:
        directory = directory.resolve()
        checkpoint_path = directory/"checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        count = len(checkpoint["images"])
        if count != checkpoint["expected_images"]:
            raise RuntimeError(f"Run is incomplete; wait for inference to finish: {directory} ({count}/{checkpoint['expected_images']})")
        loaded.append((directory,checkpoint))
    for directory,checkpoint in loaded:
        options = SimpleNamespace(output_dir=directory,reference=checkpoint["config"]["reference"])
        summary = summarize(checkpoint,options)
        kinds = ", ".join(summary["ground_truth_variants"])
        print(f"Refreshed {directory}: {summary['completed_images']} images; {kinds}",flush=True)
    print("Updated CSV/JSON summaries only; no model inference was performed.",flush=True)


if __name__ == "__main__":
    main()
