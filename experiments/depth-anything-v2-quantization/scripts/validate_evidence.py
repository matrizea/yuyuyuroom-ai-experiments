"""Recompute public result counts, means, and timings from the saved CSV/JSON."""
from pathlib import Path
import csv
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"


def read_json(path): return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8",newline="") as stream: return list(csv.DictReader(stream))


def close(a,b,context):
    if not math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12):
        raise AssertionError(f"{context}: {a} != {b}")


def main():
    summary=read_json(EVIDENCE/"cpu_ordinal/summary.json")
    pairs=read_csv(EVIDENCE/"cpu_ordinal/per_pair.csv")
    images=read_csv(EVIDENCE/"cpu_ordinal/per_image.csv")
    for name,result in summary["models"].items():
        subset=[row for row in pairs if row["model"]==name]
        im=[row for row in images if row["model"]==name]
        assert len(subset)==result["pairs"] and len(im)==result["images"]
        correct=sum(row["correct"]=="True" for row in subset)
        assert correct==result["correct_pairs"]
        close(correct/len(subset),result["accuracy"],f"{name} accuracy")
        worse=sum(row["fp32_correct"]=="True" and row["correct"]=="False" for row in subset)
        better=sum(row["fp32_correct"]=="False" and row["correct"]=="True" for row in subset)
        assert (worse,better)==(result["fp32_correct_to_wrong"],result["fp32_wrong_to_correct"])
        for metric in ["aligned_nrmse","spearman_64x64"]:
            close(statistics.mean(float(row[metric]) for row in im),result["map_metrics"][metric]["mean"],f"{name} {metric}")
        close(statistics.median(result["benchmark"]["samples_ms"]),result["benchmark"]["median_ms"],f"{name} CPU latency")
    gpu=read_json(EVIDENCE/"gpu_benchmark.json")
    for name,result in gpu["models"].items():
        for key in ["host_input_output","device_resident_input_output"]:
            assert len(result[key]["samples_ms"])==gpu["iterations"]
            close(statistics.median(result[key]["samples_ms"]),result[key]["median_ms"],f"{name}/{key}")
    dense_files=sorted(EVIDENCE.glob("results_dense_*/summary.json"))
    for path in dense_files:
        dense=read_json(path);rows=read_csv(path.with_name("per_image.csv"))
        assert dense["completed_images"]==dense["expected_images"]
        for kind,variants in dense["ground_truth_variants"].items():
            for name,result in variants["models"].items():
                selected=[row for row in rows if row["model"]==name and row["gt_kind"]==kind]
                assert len(selected)==dense["completed_images"]
                for metric in ["abs_rel","rmse","delta1","inverse_nrmse","clipped_inverse_depth_rate"]:
                    close(statistics.mean(float(row[metric]) for row in selected),result[metric]["mean"],f"{path.parent.name}/{kind}/{name}/{metric}")
                assert len(result["versus_reference"]["scene_cluster_ci95_delta_abs_rel"])==2
                assert len(result["versus_reference_inverse_nrmse"]["scene_cluster_ci95_delta_inverse_nrmse"])==2
    print(f"PASS: ordinal CSV counts/metrics, CPU/GPU timing medians, {len(dense_files)} complete dense result sets")


if __name__=="__main__": main()
