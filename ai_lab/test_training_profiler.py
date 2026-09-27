#!/usr/bin/env python3
"""Validate the profiler article's generated raw and summary artifacts."""

from __future__ import annotations

import csv
import json
import statistics
import unittest
from pathlib import Path


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results" / "training_profiler"


class ArtifactChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads((RESULTS / "result.json").read_text(encoding="utf-8"))
        with (RESULTS / "timed_steps_raw.csv").open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_six_runs_and_raw_steps(self) -> None:
        self.assertEqual(len(self.result["runs"]), 6)
        self.assertEqual(len(self.rows), 600)
        for workers in (0, 4):
            for round_number in (1, 2, 3):
                subset = [row for row in self.rows if int(row["workers"]) == workers
                          and int(row["round"]) == round_number]
                self.assertEqual(len(subset), 100)
                run = next(run for run in self.result["runs"] if run["workers"] == workers
                           and run["round"] == round_number)
                self.assertAlmostEqual(statistics.median(float(row["batch_wait_ms"]) for row in subset),
                                       run["median_batch_wait_ms"])

    def test_summary_from_raw_runs(self) -> None:
        for workers in (0, 4):
            runs = [run for run in self.result["runs"] if run["workers"] == workers]
            self.assertAlmostEqual(statistics.median(run["images_per_second"] for run in runs),
                                   self.result["summary"][str(workers)]["median_images_per_second"])

    def test_trace_annotations_and_gpu_kernels(self) -> None:
        for workers in (0, 4):
            recorded = self.result["profiler"][str(workers)]
            self.assertGreater(recorded["kernel_events"], 0)
            trace = json.loads((RESULTS / f"workers_{workers}_trace.json").read_text(encoding="utf-8"))
            for stage in ("batch_wait", "h2d_transfer", "forward_backward_adam"):
                events = [event for event in trace["traceEvents"] if event.get("ph") == "X"
                          and event.get("cat") == "user_annotation" and event.get("name") == stage]
                self.assertEqual(len(events), 20)
                self.assertEqual(recorded["stages"][stage]["count"], 20)


if __name__ == "__main__":
    unittest.main()
