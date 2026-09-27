#!/usr/bin/env python3
"""Check experiment artifact consistency and preprocessing equivalence."""

from __future__ import annotations

import csv
import json
import statistics
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision import transforms

from experiment_onnx_int8_static import BASE, images_to_numpy


RESULT_DIR = BASE / "results" / "onnx_int8_static" / "preprocessed"
RESULT = json.loads((RESULT_DIR / "result.json").read_text(encoding="utf-8"))


class ExperimentChecks(unittest.TestCase):
    def test_calibration_uses_training_only(self) -> None:
        protocol = RESULT["protocol"]
        self.assertEqual(protocol["calibration_source"], "FashionMNIST train only")
        self.assertEqual(protocol["test_source"], "FashionMNIST test only")
        self.assertEqual(len(protocol["calibration_indices"]), 512)
        self.assertEqual(len(set(protocol["calibration_indices"])), 512)
        self.assertTrue(all(0 <= index < 60_000 for index in protocol["calibration_indices"]))

    def test_pixel_scaling_matches_totensor(self) -> None:
        pixels = torch.tensor([[[0, 127], [128, 255]]], dtype=torch.uint8)
        actual = images_to_numpy(pixels)[0]
        expected = transforms.ToTensor()(pixels[0].numpy())
        np.testing.assert_array_equal(actual, expected.numpy())

    def test_optimized_graph_has_quantized_kernels(self) -> None:
        nodes = RESULT["artifacts"]["int8_optimized"]["graph"]["node_counts"]
        self.assertEqual(nodes.get("QLinearConv"), 2)
        self.assertEqual(nodes.get("QGemm"), 2)
        types = RESULT["artifacts"]["int8"]["graph"]["initializer_types"]
        self.assertGreater(types.get("INT8", 0), 0)

    def test_raw_timings_match_aggregates(self) -> None:
        with (RESULT_DIR / "timings_raw.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3 * 2 * 5 * 100)
        for batch in (1, 32, 256):
            for variant in ("fp32", "int8"):
                values = [float(row["elapsed_ms"]) for row in rows
                          if int(row["batch_size"]) == batch and row["variant"] == variant]
                self.assertEqual(len(values), 500)
                self.assertAlmostEqual(statistics.median(values), RESULT["timings"][str(batch)][variant]["median_ms"])

    def test_preprocessed_fp32_equivalence(self) -> None:
        source = BASE / "models" / "fashion_cnn_dynamic.onnx"
        processed = BASE / "models" / "fashion_cnn_preprocessed.onnx"
        rng = np.random.default_rng(123)
        images = rng.random((13, 1, 28, 28), dtype=np.float32)
        sessions = [ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                    for path in (source, processed)]
        outputs = [session.run(["logits"], {"image": images})[0] for session in sessions]
        np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
