"""Recompute paired FP8 timings and GT means/intervals from public evidence."""
from pathlib import Path
import csv
import json
import math
import statistics

from dense_utils import bootstrap_mean_delta, bootstrap_scene_mean_delta

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b, context):
    if not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12):
        raise AssertionError(f"{context}: {a} != {b}")


def main():
    benchmark = read(EVIDENCE/"fp8_benchmark.json")
    assert benchmark["status"] == "complete"
    config = benchmark["config"]
    assert config["rounds"] == 3 and config["warmup_per_block_per_mode"] == 20
    assert config["iterations_per_block_per_mode"] == 100 and len(benchmark["blocks"]) == 12
    assert config["input_shape"] == [1, 3, 518, 518] and config["use_tf32"] is False
    modes = ["host_input_output", "device_resident_input_output"]
    for name in ("fp16", "fp8"):
        result = benchmark["models"][name]
        assert result["source_graph"]["opsets"]["ai.onnx"] == 21
        assert result["profile"]["cpu_fallback_node_count"] == 0
        selected = [block for block in benchmark["blocks"] if block["model"] == name]
        assert len(selected) == 6
        for mode in modes:
            values = []
            for block in selected:
                assert len(block[mode]["samples_ms"]) == 100
                close(statistics.median(block[mode]["samples_ms"]), block[mode]["median_ms"], name+" block")
                values.extend(block[mode]["samples_ms"])
            assert len(values) == 600 and result[mode]["sample_count"] == 600
            close(statistics.median(values), result[mode]["median_ms"], name+" pooled "+mode)
            close(statistics.mean(values), result[mode]["mean_ms"], name+" pooled mean "+mode)
    for record in benchmark["paired_rounds"]:
        blocks = [block for block in benchmark["blocks"] if block["round"] == record["round"]]
        for mode in modes:
            medians = {}
            for name in ("fp16", "fp8"):
                values = [sample for block in blocks if block["model"] == name for sample in block[mode]["samples_ms"]]
                medians[name] = statistics.median(values)
                close(medians[name], record[mode][name+"_median_ms"], "round "+name)
            close(medians["fp8"]/medians["fp16"], record[mode]["fp8_over_fp16_latency_ratio"], "round ratio")
    fp8 = benchmark["models"]["fp8"]
    assert fp8["profile"]["cuda_fp8_operator_execution_observed"] is True
    assert fp8["profile"]["node_executions_by_provider"]["CUDAExecutionProvider"]["GemmFloat8"] == 48
    gemms = [node for node in fp8["source_graph"]["fp8_related_nodes"] if node["op_type"] == "GemmFloat8"]
    assert len(gemms) == 48
    assert all(node["input_dtypes"][:2] == ["FLOAT8E4M3FN", "FLOAT8E4M3FN"] for node in gemms)
    model = read(EVIDENCE/"fp8_models.json")
    assert model["sha256"] == fp8["source_graph"]["sha256"]
    close(model["size_mib"], fp8["source_graph"]["file_size_mib"], "FP8 size")
    calibration = read(EVIDENCE/"fp8_calibration_manifest.json")
    assert calibration["count"] == 64 and calibration["seed"] == 43
    assert calibration["gt_access"]["gt_values_used_for_calibration"] is False
    for key in ("official_nyu_test_index_overlap", "nyu_eval128_index_overlap",
                "official_nyu_test_scene_overlap", "exact_decoded_rgb_overlap_with_evaluation_images"):
        assert calibration["separation_checks"][key] == 0
    assert read(EVIDENCE/"fp8_calibration.json")["input_sha256"] == calibration["array"]["sha256"]
    checks = 0
    for label, expected, old_name in (("nyu", 128, "results_dense_nyu_gpu"),
                                      ("sintel", 64, "results_dense_sintel_full_gpu")):
        directory = EVIDENCE/f"results_dense_fp8_{label}_gpu"
        summary = read(directory/"summary.json")
        original = read(EVIDENCE/old_name/"summary.json")
        with (directory/"per_image.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert summary["expected_images"] == summary["completed_images"] == expected
        assert summary["reference"] == "fp16"
        for key in ("manifest_sha256", "min_depth", "max_depth", "threads", "cuda_use_tf32"):
            assert summary["config"][key] == original["config"][key], key
        assert set(summary["ground_truth_variants"]) == set(original["ground_truth_variants"])
        for kind, variants in summary["ground_truth_variants"].items():
            reference = {row["id"]:row for row in rows if row["model"] == "fp16" and row["gt_kind"] == kind}
            for name, result in variants["models"].items():
                selected = [row for row in rows if row["model"] == name and row["gt_kind"] == kind]
                assert len(selected) == expected and {row["id"] for row in selected} == set(reference)
                assert all(int(row["valid_pixels"]) == int(reference[row["id"]]["valid_pixels"]) for row in selected)
                assert result["valid_pixels"] == original["ground_truth_variants"][kind]["models"]["fp16"]["valid_pixels"]
                for metric in ("abs_rel", "rmse", "delta1", "inverse_nrmse", "clipped_inverse_depth_rate"):
                    values = [float(row[metric]) for row in selected]
                    close(statistics.mean(values), result[metric]["mean"], label+" "+kind+" "+name+" "+metric)
                for metric, field in (("abs_rel", "versus_reference"),
                                      ("inverse_nrmse", "versus_reference_inverse_nrmse")):
                    candidate = [float(row[metric]) for row in selected]
                    baseline = [float(reference[row["id"]][metric]) for row in selected]
                    scenes = [row["scene"] for row in selected]
                    recomputed = {**bootstrap_mean_delta(candidate, baseline, metric=metric),
                                  **bootstrap_scene_mean_delta(candidate, baseline, scenes, metric=metric)}
                    saved = result[field]
                    close(recomputed[f"mean_delta_{metric}"], saved[f"mean_delta_{metric}"], "GT delta")
                    for key in (f"ci95_delta_{metric}", f"scene_cluster_ci95_delta_{metric}"):
                        for calculated, recorded in zip(recomputed[key], saved[key]):
                            close(calculated, recorded, label+" "+kind+" "+name+" "+key)
                    checks += 1
    print(f"PASS: 12 paired timing blocks, CUDA FP8 operand execution, calibration separation, {checks} GT mean/CI comparisons")


if __name__ == "__main__":
    main()
