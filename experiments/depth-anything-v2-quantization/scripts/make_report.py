"""Build an offline Japanese HTML report from the quantization experiment outputs.

This script reads existing measurements and previews only. It does not run
inference, download assets, or alter models, datasets, and measurement files.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
from statistics import median
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
ORDER = ("fp32", "int8_dynamic", "int4_rtn", "int4_fast")
LABELS = {
    "fp32": "FP32",
    "int8_dynamic": "INT8 dynamic",
    "int4_rtn": "INT4 RTN · A32",
    "int4_fast": "INT4 · 内部A8許可",
}
COLORS = {"fp32": "#334155", "int8_dynamic": "#047857", "int4_rtn": "#b45309", "int4_fast": "#7c3aed"}
CATEGORIES = {
    "adverse_style": "悪条件・画風の変化",
    "aerial": "空撮",
    "indoor": "屋内",
    "non_real": "非実写",
    "object": "物体",
    "outdoor": "屋外",
    "transparent_reflective": "透明・反射",
    "underwater": "水中",
}


def e(value: object) -> str:
    return escape(str(value), quote=True)


def number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def fmt(value: object, digits: int = 2, suffix: str = "") -> str:
    result = number(value)
    return "未測定" if result is None else f"{result:,.{digits}f}{suffix}"


def pct(value: object, digits: int = 2) -> str:
    result = number(value)
    return "未測定" if result is None else f"{100 * result:.{digits}f}%"


def pp(value: object) -> str:
    result = number(value)
    return "未測定" if result is None else f"{result:+.2f} pp"


def boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0"}:
        return False
    raise ValueError(f"Unrecognized boolean in CSV: {value!r}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def label(name: str) -> str:
    return LABELS.get(name, name)


def category_label(name: str) -> str:
    return CATEGORIES.get(name, name)


def local_href(path: Path, parent: Path) -> str:
    import os
    return quote(Path(os.path.relpath(path.resolve(), parent.resolve())).as_posix(), safe="/:.-_")


def link_file(path: Path, title: str, report_dir: Path) -> str:
    return f'<a href="{e(local_href(path, report_dir))}">{e(title)}</a>'


def stat(model: dict, name: str, aggregate: str = "mean") -> object:
    return model.get("map_metrics", {}).get(name, {}).get(aggregate)


def interval_text(model: dict) -> str:
    interval = model.get("versus_fp32", {}).get("ci95_percentage_points", [None, None])
    if len(interval) != 2 or any(number(x) is None for x in interval):
        return "未測定"
    return f"[{float(interval[0]):+.2f}, {float(interval[1]):+.2f}] pp"


def delta(model: dict) -> object:
    return model.get("versus_fp32", {}).get("difference_percentage_points")


def conclusion(models: dict, complete: bool) -> list[str]:
    """Describe measurements without treating a nonsignificant result as equality."""
    sentences = []
    if not complete:
        sentences.append("評価は途中です。以下は現在までの暫定値で、最終的な採用判断には使わないでください。")
    quantized = [name for name in ORDER if name != "fp32" and name in models]
    if not quantized:
        return sentences + ["FP32以外の測定がないため、量子化による変化はまだ判断できません。"]
    for name in quantized:
        result = models[name]
        change = number(delta(result))
        if change is None:
            continue
        interval = result.get("versus_fp32", {}).get("ci95_percentage_points", [None, None])
        low, high = (number(x) for x in interval)
        observed = (f"{label(name)} の近遠正答率は {pct(result.get('accuracy'))}、"
                    f"FP32との差は {pp(change)}（画像単位の対応あり95% CI {interval_text(result)}）。")
        if low is not None and high is not None and high < 0:
            observed += "この標本では正答率の低下が確認されました。"
        elif low is not None and high is not None and low > 0:
            observed += "この標本では正答率が上がりましたが、別の画像でも改善する保証はありません。"
        else:
            observed += "CIは0を含み、この標本では正答率差を明確に区別できません。同等精度の証明ではありません。"
        sentences.append(observed)
    eligible = [name for name in quantized if number(stat(models[name], "aligned_nrmse")) is not None]
    if eligible:
        faithful = min(eligible, key=lambda name: float(stat(models[name], "aligned_nrmse")))
        sentences.append(f"FP32出力への一致度が最も高い量子化版は {label(faithful)} "
                         f"（平均 aligned NRMSE {fmt(stat(models[faithful], 'aligned_nrmse'), 5)}）。"
                         "これはFP32との出力差であり、GTに対する精度そのものではありません。")
    base_latency = number((models["fp32"].get("benchmark") or {}).get("median_ms"))
    base_size = number(models["fp32"].get("file_size_mib"))
    timed = [name for name in quantized
             if (number((models[name].get("benchmark") or {}).get("median_ms")) or 0) > 0]
    if timed and base_latency and base_size:
        fastest = min(timed, key=lambda name: float(models[name]["benchmark"]["median_ms"]))
        fastest_result = models[fastest]
        latency = float(fastest_result["benchmark"]["median_ms"])
        size = number(fastest_result.get("file_size_mib"))
        size_text = f"、ONNXファイルサイズはFP32から {100*(1-size/base_size):.1f}% 削減" if size is not None else ""
        sentences.append(f"このCPU測定で最も速い量子化版は {label(fastest)} "
                         f"（中央値 {latency:.1f} ms、FP32比 {base_latency/latency:.2f}倍の速度{size_text}）。")
    int8 = models.get("int8_dynamic")
    if int8:
        change = number(delta(int8))
        interval = int8.get("versus_fp32", {}).get("ci95_percentage_points", [None, None])
        low = number(interval[0]) if len(interval) == 2 else None
        high = number(interval[1]) if len(interval) == 2 else None
        if high is not None and high < 0:
            sentences.append("FP32を残す理由はあります。少なくとも今回のINT8設定では近遠判定を失っており、"
                             "必要な画質と省メモリ・速度の利益を照らして選ぶ必要があります。")
        elif change is not None:
            sentences.append("この結果だけからSmallにFP32が必須とは言えません。"
                             "ただし、採用前には使用する画像で近遠判定と細部を確認し、"
                             "ここで測れない小さな精度差を許容できるか判断する必要があります。")
            if complete and timed and low is not None and high is not None and low <= 0 <= high:
                sentences.append("このCPU条件では、出力差・サイズ・速度のバランスからINT8を第一候補とします。"
                                 "小さな精度差の許容可否は利用する画像で確認してください。")
    return sentences


def verify_consistency(summary: dict, images: list[dict], pairs: list[dict]) -> None:
    if "fp32" not in summary.get("models", {}):
        raise ValueError("summary.json must contain models.fp32")
    for name, result in summary["models"].items():
        image_rows = [row for row in images if row.get("model") == name]
        pair_rows = [row for row in pairs if row.get("model") == name]
        if len(image_rows) != int(result["images"]) or len(pair_rows) != int(result["pairs"]):
            raise ValueError(f"CSV and summary counts differ for {name}; regenerate after evaluation finishes")
        correct = sum(boolean(row["correct"]) for row in pair_rows)
        if correct != int(result["correct_pairs"]):
            raise ValueError(f"CSV and summary correctness differ for {name}")


def overview_table(models: dict, names: list[str]) -> str:
    base = models["fp32"]
    base_size = number(base.get("file_size_mib"))
    base_latency = number((base.get("benchmark") or {}).get("median_ms"))
    rows = []
    for name in names:
        result = models[name]
        benchmark = result.get("benchmark") or {}
        size = number(result.get("file_size_mib"))
        latency = number(benchmark.get("median_ms"))
        size_note = f"<small>{size / base_size:.1%} of FP32</small>" if size is not None and base_size else ""
        latency_note = f"<small>FP32比 {base_latency / latency:.2f}×</small>" if latency and base_latency else ""
        rows.append(f'''<tr>
          <th scope="row"><span class="dot" style="background:{COLORS.get(name, '#334155')}"></span>{e(label(name))}</th>
          <td class="em">{pct(result.get('accuracy'))}<small>{int(result['correct_pairs'])}/{int(result['pairs'])} ペア</small></td>
          <td>{pp(delta(result))}<small>{interval_text(result)}</small></td>
          <td><span class="loss">{int(result.get('fp32_correct_to_wrong', 0))}</span> / <span class="gain">{int(result.get('fp32_wrong_to_correct', 0))}</span></td>
          <td>{fmt(size, 2)}{size_note}</td>
          <td>{fmt(latency, 1)}<small>P90 {fmt(benchmark.get('p90_ms'), 1)} ms</small>{latency_note}</td>
          <td>{fmt(stat(result, 'aligned_nrmse'), 5)}<small>P90 {fmt(stat(result, 'aligned_nrmse', 'p90'), 5)}</small></td>
          <td>{fmt(stat(result, 'spearman_64x64'), 5)}</td>
        </tr>''')
    return '''<div class="table-wrap"><table class="overview"><thead><tr>
      <th>モデル</th><th>GT近遠正答率 ↑</th><th>FP32との差 / 95% CI</th><th>悪化 / 改善<small>ペア数</small></th>
      <th>サイズ<small>MiB ↓</small></th><th>推論中央値<small>ms ↓</small></th><th>aligned NRMSE ↓<small>画像平均</small></th><th>Spearman ↑<small>画像平均</small></th>
      </tr></thead><tbody>''' + "".join(rows) + "</tbody></table></div>"


def accuracy_chart(models: dict, names: list[str]) -> str:
    width, height = 880, 65 + 60 * len(names)
    left, chart_width = 175, 530
    chunks = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="GT近遠正答率の比較。横軸0から100パーセント。">']
    for tick in range(0, 101, 20):
        x = left + chart_width * tick / 100
        chunks.append(f'<line x1="{x}" x2="{x}" y1="20" y2="{height-35}" stroke="#dce4ec"/><text x="{x}" y="{height-12}" text-anchor="middle" class="axis">{tick}%</text>')
    for index, name in enumerate(names):
        y = 26 + index * 60
        accuracy = number(models[name].get("accuracy")) or 0
        chunks.append(f'<text x="{left-14}" y="{y+21}" text-anchor="end" class="chart-label">{e(label(name))}</text>')
        chunks.append(f'<rect x="{left}" y="{y}" width="{chart_width}" height="32" rx="5" fill="#eef2f6"/>')
        chunks.append(f'<rect x="{left}" y="{y}" width="{max(0,min(1,accuracy))*chart_width:.2f}" height="32" rx="5" fill="{COLORS.get(name, "#334155")}"/>')
        chunks.append(f'<text x="{left+chart_width+14}" y="{y+21}" class="chart-value">{pct(accuracy)}</text>')
    return "".join(chunks) + "</svg>"


def category_table(models: dict, names: list[str], dataset: dict) -> str:
    categories = sorted(models["fp32"].get("by_category", {}))
    split_categories = dataset.get("splits", {}).get("evaluation", {}).get("categories", {})
    rows = []
    for category in categories:
        base = models["fp32"]["by_category"][category]
        count = split_categories.get(category, {}).get("images")
        cells = []
        for name in names:
            result = models[name].get("by_category", {}).get(category)
            if not result:
                cells.append("<td>未測定</td>")
                continue
            accuracy = float(result["accuracy"])
            change = 100 * (accuracy - float(base["accuracy"]))
            cells.append(f'<td><b>{pct(accuracy)}</b><small>{result["correct"]}/{result["pairs"]} · {pp(change)}</small>'
                         f'<div class="mini-track"><i style="width:{100*accuracy:.2f}%;background:{COLORS.get(name, "#334155")}"></i></div></td>')
        rows.append(f'<tr><th scope="row">{e(category_label(category))}<small>{e(category)} · {e(count) if count is not None else "—"}画像 / {base["pairs"]}ペア</small></th>{"".join(cells)}</tr>')
    return ('<div class="table-wrap"><table><thead><tr><th>カテゴリ</th>'
            + "".join(f"<th>{e(label(name))}</th>" for name in names)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def image_index(rows: list[dict]) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for row in rows:
        result.setdefault(row["image"], {})[row["model"]] = row
    return result


def choose_examples(index: dict) -> tuple[list[tuple], list[tuple], list[tuple], str | None]:
    preferred = "int4_rtn" if any("int4_rtn" in item for item in index.values()) else "int4_fast"
    candidates = [(path, item[preferred]) for path, item in index.items()
                  if preferred in item and number(item[preferred].get("aligned_nrmse")) is not None]
    if not candidates:
        return [], [], [], None
    representatives = []
    for category in sorted({row["category"] for _, row in candidates}):
        subset = [(path, row) for path, row in candidates if row["category"] == category]
        central = median(float(row["aligned_nrmse"]) for _, row in subset)
        path, _ = min(subset, key=lambda item: (abs(float(item[1]["aligned_nrmse"])-central), item[0]))
        representatives.append((path, f"{label(preferred)} のaligned NRMSEが、このカテゴリの中央値 {central:.5f} に最も近い画像。見栄えや正答率では選んでいません。"))
    ranked = sorted(candidates, key=lambda item: (float(item[1]["aligned_nrmse"]), item[0]))
    best = [(path, f"{label(preferred)} のaligned NRMSEが全評価画像中で小さい順の第{rank}位。GT正答率の順位ではありません。")
            for rank, (path, _) in enumerate(ranked[:2], 1)]
    worst = [(path, f"{label(preferred)} のaligned NRMSEが全評価画像中で大きい順の第{rank}位。GT近遠判定の悪化と一致するとは限りません。")
             for rank, (path, _) in enumerate(reversed(ranked[-4:]), 1)]
    return representatives, best, worst, preferred


def preview_for(path: str, result_dir: Path, checkpoint: dict) -> Path | None:
    relative = checkpoint.get("images", {}).get(path, {}).get("preview")
    if relative:
        candidate = result_dir / relative
        if candidate.is_file():
            return candidate
    # Older outputs have no per-image preview column; recover by the saved stem.
    stem = Path(path).stem
    candidates = sorted((result_dir / "previews").glob(f"*_{stem}.*"))
    return candidates[0] if len(candidates) == 1 else None


def example_card(path: str, reason: str, index: dict, pairs: list[dict], names: list[str],
                 result_dir: Path, report_dir: Path, checkpoint: dict) -> str:
    image_rows = index[path]
    first = next(iter(image_rows.values()))
    preview = preview_for(path, result_dir, checkpoint)
    if preview:
        href = e(local_href(preview, report_dir))
        visual = f'<a class="preview-link" href="{href}"><img loading="lazy" src="{href}" alt="元画像、各精度の逆深度、FP32からの整合済み差分を上下2段で比較"></a>'
    else:
        visual = '<div class="missing">保存済み比較プレビューが見つかりません。画像なしで数値のみ表示しています。</div>'
    rows = []
    for name in names:
        row = image_rows.get(name)
        if not row:
            continue
        subset = [p for p in pairs if p["image"] == path and p["model"] == name]
        worse = sum(boolean(p["fp32_correct"]) and not boolean(p["correct"]) for p in subset)
        better = sum(not boolean(p["fp32_correct"]) and boolean(p["correct"]) for p in subset)
        rows.append(f'<tr><th>{e(label(name))}</th><td>{e(row["correct_pairs"])}/{e(row["pair_count"])}</td>'
                    f'<td><span class="loss">{worse}</span> / <span class="gain">{better}</span></td>'
                    f'<td>{fmt(row.get("aligned_nrmse"),5)}</td><td>{fmt(row.get("spearman_64x64"),5)}</td></tr>')
    return f'''<article class="example">
      <div class="example-heading"><span class="tag">{e(category_label(first['category']))}</span><b>{e(Path(path).name)}</b>
      <span class="muted">元画像 {e(first['original_w'])}×{e(first['original_h'])}</span></div>
      <p class="selection">選択理由：{e(reason)}</p>{visual}
      <div class="table-wrap"><table class="image-metrics"><thead><tr><th>モデル</th><th>GT正答 / 全ペア</th><th>FP32から悪化 / 改善</th><th>aligned NRMSE</th><th>Spearman</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
      <details><summary>元画像パス</summary><code>{e(path)}</code></details>
      </article>'''


CSS = """
:root{color-scheme:light;--ink:#152638;--muted:#536579;--line:#d8e2eb;--paper:#fff;--bg:#f3f6f9;--green:#047857;--red:#a32731}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.8 system-ui,-apple-system,"Segoe UI","Yu Gothic UI","Meiryo",sans-serif}
main{max-width:1240px;margin:auto;padding:36px 28px 80px}header{margin-bottom:24px}.eyebrow{font-size:12px;letter-spacing:.13em;font-weight:750;color:#38617a}h1{font-size:clamp(28px,4vw,46px);line-height:1.25;letter-spacing:-.025em;margin:12px 0 15px}h2{font-size:25px;line-height:1.4;margin:0 0 18px}h3{font-size:19px;margin:25px 0 10px}p{margin:9px 0 14px}.lead{max-width:900px;font-size:18px}.muted,.note{color:var(--muted)}.note{font-size:13px}.section{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:27px;margin:22px 0}.summary{border-top:5px solid #186b67}.summary li{margin:9px 0}.warning{background:#fff4dc;border:1px solid #e7c56c;border-radius:9px;padding:12px 16px}.callout{border-left:4px solid #3e7577;background:#eff7f6;padding:13px 17px;margin:16px 0}.facts{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.fact,.tag{display:inline-block;background:#eaf0f5;border:1px solid #d8e2eb;border-radius:5px;padding:3px 9px;font-size:12px}.facts .fact{font-size:13px;padding:6px 11px}nav{display:flex;flex-wrap:wrap;gap:8px 19px;font-size:13px}a{color:#086d82;text-decoration-thickness:1px;text-underline-offset:3px}a:hover{color:#003d4d}ul{padding-left:23px}li{margin:5px 0}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:9px;margin:14px 0}table{width:100%;border-collapse:collapse;font-size:13px;text-align:left;font-variant-numeric:tabular-nums}th,td{padding:12px 13px;border-bottom:1px solid var(--line);vertical-align:top}thead th{background:#f0f4f8;font-size:12px;color:#3e5367;white-space:nowrap}tbody th{font-weight:650}tbody tr:last-child>*{border-bottom:0}table small{display:block;font-weight:400;color:var(--muted);font-size:11px;line-height:1.6;white-space:nowrap}.overview{min-width:1040px}.overview td{white-space:nowrap}.overview th:first-child{min-width:170px}.em{font-weight:750;font-size:17px}.loss{color:var(--red);font-weight:650}.gain{color:var(--green);font-weight:650}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}.mini-track{height:5px;background:#e9eef3;border-radius:3px;margin-top:7px;width:100%;max-width:145px}.mini-track i{height:100%;display:block;border-radius:3px}.chart{max-width:900px;margin:20px auto}.chart svg{display:block;width:100%;height:auto}.axis{fill:#536579;font-size:12px}.chart-label{fill:#31465b;font-size:14px}.chart-value{fill:#152638;font-weight:650;font-size:15px}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:24px}.definition dt{font-weight:700;margin-top:13px}.definition dd{margin:4px 0 0;color:#43576a;font-size:13px}.example{margin:24px 0 32px;padding-top:18px;border-top:1px solid var(--line)}.example-heading{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;overflow-wrap:anywhere}.example-heading b{font-size:14px}.example-heading .muted{font-size:12px}.selection{font-size:13px;color:#485f72}.preview-link{display:block;background:#f1f4f7;border:1px solid #ccd8e3;border-radius:7px;overflow:hidden}.preview-link img{display:block;width:100%;height:auto}.image-metrics{min-width:650px}.image-metrics th,.image-metrics td{padding:7px 13px}.missing{background:#fff3e4;padding:24px;border:1px dashed #be8d43}.method-table th{width:23%;min-width:155px}.method-table td{overflow-wrap:anywhere}code{font:12px/1.6 ui-monospace,Consolas,monospace;overflow-wrap:anywhere}details{font-size:12px;color:var(--muted)}details summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f1f4f7;padding:13px;border-radius:7px}.source-list li{font-size:13px;margin:9px 0}.footer{font-size:12px;color:var(--muted);margin-top:30px}section{scroll-margin-top:18px}
@media(max-width:750px){main{padding:22px 12px 45px}.section{padding:18px 15px;border-radius:10px}.two-col{grid-template-columns:1fr}.lead{font-size:16px}.method-table th{min-width:105px}h2{font-size:22px}}
@media print{body{background:#fff}main{max-width:none;padding:0}.section{break-inside:avoid;border-radius:0;padding:16px 0;border-left:0;border-right:0}nav{display:none}.example{break-inside:avoid}a{color:inherit}details{display:none}.table-wrap{overflow:visible}table{font-size:10px}.overview{min-width:0}.overview th:first-child{min-width:0}.overview td{white-space:normal}th,td{padding:7px 5px}table small{white-space:normal}.em{font-size:13px}}
"""


def build_report(summary: dict, image_rows: list[dict], pair_rows: list[dict], metadata: dict,
                 dataset: dict, result_dir: Path, report_path: Path, checkpoint: dict) -> str:
    models = summary["models"]
    names = [name for name in ORDER if name in models] + [name for name in models if name not in ORDER]
    reference = models["fp32"]
    completed = int(summary.get("completed_images", reference["images"]))
    expected = int(summary.get("expected_images", completed))
    complete = completed == expected
    split = dataset.get("splits", {}).get("evaluation", {})
    config = summary.get("config", {})
    environment = summary.get("environment", {})
    threads = config.get("threads", 8)
    index = image_index(image_rows)
    representatives, best, worst, selected_variant = choose_examples(index)
    int8_candidates = [(path, rows["int8_dynamic"]) for path, rows in index.items()
                       if "int8_dynamic" in rows and number(rows["int8_dynamic"].get("aligned_nrmse")) is not None]
    int8_ranked = sorted(int8_candidates, key=lambda item: (-float(item[1]["aligned_nrmse"]), item[0]))
    int8_examples = [(path, f"INT8 dynamic のaligned NRMSEが全評価画像中で大きい順の第{rank}位。GT近遠判定の正誤とは独立に選択しています。")
                     for rank, (path, _) in enumerate(int8_ranked[:2], 1)]
    report_dir = report_path.parent
    examples = lambda items: "".join(example_card(path, why, index, pair_rows, names, result_dir,
                                                 report_dir, checkpoint) for path, why in items)
    conclusion_html = "".join(f"<li>{e(sentence)}</li>" for sentence in conclusion(models, complete))
    validation = metadata.get("export_validation_random_seed42", {})
    coverage = number(metadata.get("quantized_weight_fraction"))
    quant_count = metadata.get("quantized_matmul_count", "—")
    quantized_count = metadata.get("quantized_weight_elements", "—")
    total_count = metadata.get("total_float_initializer_elements", "—")
    benchmarks = [result.get("benchmark") for result in models.values() if result.get("benchmark")]
    if benchmarks:
        benchmark_description = (f"各モデルを独立にウォームアップ {benchmarks[0].get('warmup', '—')}回、"
                                 f"本測定 {benchmarks[0].get('iterations', '—')}回。"
                                 "同一のseed42乱数入力、batch=1。session.runのみを計時し、"
                                 "画像読込・前処理・後処理・モデル読込時間を含めません。")
    else:
        benchmark_description = "速度測定はまだありません。精度評価の進行時間を推論レイテンシとして使用していません。"
    resize_mode = config.get("resize_mode", "square")
    preprocessing = ("OpenCVでRGBへ変換、255で正規化、CUBICで518×518へ直接リサイズ（cropなし）。"
                     "ImageNet mean/stdで正規化してfloat32 NCHWへ。元画像の縦横比は変わります。"
                     "公式の標準前処理は縦横比を保ち、短辺518以上・14の倍数にします。"
                     "本結果は固定正方形入力の比較で、公式標準前処理のベンチマーク値ではありません。"
                     if resize_mode == "square" else
                     "公式OpenCV前処理：縦横比を保つlower_bound=518、14の倍数、CUBIC補間、ImageNet正規化。")
    model_methods = {
        "fp32": "公式Small (vits) の変更していないforwardをONNX export。FP32のまま実行。",
        "int8_dynamic": "定数Bを持つMatMul 48個のみ。重みは出力チャネル別signed INT8、activationは動的uint8。その他の演算はFP32。校正画像は使用しません。",
        "int4_rtn": "定数重みMatMulのみ、symmetric INT4 / block size 32 / round-to-nearest。accuracy_level=1でactivationはFP32を保持。その他の演算はFP32。",
        "int4_fast": "INT4 RTNと同じ4bit重み。accuracy_level=4で内部のINT8 activation近似を許可。実際のカーネル選択はCPUとランタイムに依存。コード名fastは速度を保証しません。",
    }
    method_rows = [
        ("モデル", "Depth Anything V2 Small / ViT-S。相対的な逆深度を予測し、大きい値ほどカメラに近い。"),
        ("評価画像", f"DA-2Kから8カテゴリを層化抽出。各32画像、合計 {split.get('image_count', expected)}画像 / {split.get('pair_count', reference['pairs'])}ペア、seed={dataset.get('seed',42)}。カテゴリ内はペア数が異なります。"),
        ("GTと正答判定", "GTはメートル等の距離ではなく、2点の近遠順序です。元画像座標のpoint1（近い点）の予測逆深度がpoint2を上回れば正答。同値は誤答。"),
        ("前処理", preprocessing),
        ("座標の対応", "公式のbilinear / align_corners=Trueによる元画像サイズへの拡大と等価になるよう、出力マップからGT座標を直接bilinear sample。"),
        ("FP32 export検証", f"公式の変更していないPyTorch forwardとONNX RuntimeのFP32出力をseed42の乱数入力で照合。max_abs={fmt(validation.get('max_abs'),9)}、RMSE={fmt(validation.get('rmse'),9)}、RMSE/std={fmt(validation.get('relative_rmse_std'),9)}。この一致検証は乱数入力1例で、全入力の証明ではありません。"),
        ("量子化範囲", f"{quant_count}個の定数重みMatMul、元FP32 initializerの {quantized_count} / {total_count}要素（{pct(coverage)}）が対象。Convやattention内の可変MatMul等を含む全層integer化ではありません。"),
        ("実行環境", f"Windows PC / Intel Core i7-14700F / ONNX Runtime {environment.get('onnxruntime','—')} / CPUExecutionProvider。intra_op={threads}、inter_op={config.get('inter_op_threads',1)}、sequential、spinning無効。GeForce RTX 5070 Tiは使用せず、GPUは未検証。"),
        ("速度の測り方", benchmark_description),
        ("統計", "FP32と量子化版を同じ画像・ペアで比較。画像単位の対応ありpercentile bootstrap、10,000再標本化、seed42の95% CI。複数ペアを持つ画像はまとめて再抽出し、画像内相関を保持。"),
    ]
    method_html = "".join(f"<tr><th>{e(key)}</th><td>{e(value)}</td></tr>" for key, value in method_rows)
    variant_html = "".join(f"<dt>{e(label(name))}</dt><dd>{e(model_methods.get(name, metadata.get('quantization',{}).get(name, '設定はmetadata.json参照。')))}</dd>" for name in names)
    split_warning = ("この標本のカテゴリ配分は各32画像で、DA-2K全体や利用先の配分とは異なります。"
                     f"{reference['pairs']}ペア程度では小さな変化を見分けにくく、CIが0を含んでも同等性は示せません。"
                     "CIはこの固定画像群からの再標本化であり、未知の利用環境まで保証しません。"
                     "近遠GTは疎なので、境界・細い構造・全画素の距離精度を評価するものではありません。")
    resources = [
        ("summary.json", result_dir/"summary.json"), ("per_image.csv", result_dir/"per_image.csv"),
        ("per_pair.csv", result_dir/"per_pair.csv"), ("models/metadata.json", ROOT/"models/metadata.json"),
        ("data/dataset_summary.json", ROOT/"data/dataset_summary.json"),
        ("評価スクリプト", ROOT/"scripts/evaluate.py"), ("レポート生成スクリプト", ROOT/"scripts/make_report.py"),
    ]
    resource_links = " · ".join(link_file(path, title, report_dir) for title, path in resources if path.is_file())
    manifest_digest = config.get("manifest_sha256", split.get("manifest_sha256", "—"))
    timestamp = summary.get("created_utc", "—")
    result_status = "完了" if complete else f"途中 {completed}/{expected}画像"
    missing_previews = sum(preview_for(path, result_dir, checkpoint) is None for path in index)
    preview_note = (f"保存済みプレビューが {missing_previews}画像で見つかりません。" if missing_previews else
                    "画像をクリックすると保存済みプレビューの原寸を開きます。")
    return f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Depth Anything V2 Small — FP32 / INT8 / INT4 実測比較</title><style>{CSS}</style></head>
<body><main>
<header><div class="eyebrow">LOCAL CPU EXPERIMENT · DEPTH ANYTHING V2 SMALL</div>
<h1>SmallにFP32は必要か。<br>INT8・INT4への量子化を実測。</h1>
<p class="lead">同じ画像で、GTの近遠判定、FP32出力との差、モデルサイズ、CPU推論時間を比較しました。近遠の正答率とマップの見た目は別の指標として読みます。</p>
<div class="facts"><span class="fact">{completed}画像 / {reference['pairs']}ペア</span><span class="fact">8カテゴリ · seed42</span><span class="fact">518×518 · batch 1</span><span class="fact">CPU {threads} threads</span><span class="fact">状態：{e(result_status)}</span></div>
<nav><a href="#findings">結果の読み取り</a><a href="#numbers">比較表</a><a href="#categories">カテゴリ別</a><a href="#visuals">画像比較</a><a href="#method">測定条件</a><a href="#sources">出典とデータ</a></nav></header>
<section class="section summary" id="findings"><h2>この測定で分かったこと</h2><ul>{conclusion_html}</ul>
<p class="note">{e(split_warning)}</p></section>
<section class="section" id="numbers"><h2>同じGTペアでの比較</h2>{overview_table(models,names)}
<p class="note">悪化＝FP32で正答→量子化で誤答、改善＝FP32で誤答→量子化で正答。差の単位ppはpercentage points。CIは正答率差の区間です。サイズはONNXファイルで、実行時RAM/VRAMではありません。速度倍率はFP32中央値÷各モデル中央値。</p>
<div class="chart">{accuracy_chart(models,names)}</div>
<div class="two-col"><dl class="definition"><dt>GT近遠正答率</dt><dd>人が注釈した近い点・遠い点を正しい順序にできた割合。主たる精度指標です。</dd><dt>aligned NRMSE</dt><dd>量子化マップを画像ごとに最小二乗でscale＋shift整合し、RMSEをFP32マップの標準偏差で割った値。0が一致。全画素の真の深度誤差ではありません。</dd></dl><dl class="definition"><dt>Spearman</dt><dd>両出力をbilinearで64×64の格子点にsampleして順位相関を計算。1が順位一致。表は画像ごとの値の単純平均です。</dd><dt>速度</dt><dd>{e(benchmark_description)} CPUでの結果をGPU速度に外挿できません。</dd></dl></div></section>
<section class="section" id="categories"><h2>カテゴリによって変化するか</h2>
<p>全体の正答率は{reference['pairs']}ペアの合計から算出します。カテゴリごとの表は、どの種類の画像に変化が集中したかを確認するためのものです。</p>
{category_table(models,names,dataset)}<p class="note">各セルは正答率、正答/ペア数、同じカテゴリのFP32からの差。各カテゴリの標本は32画像と小さく、この表からカテゴリ間の優劣を断定しません。</p></section>
<section class="section" id="visuals"><h2>画像で見る違い</h2>
<div class="callout">上段は元画像と各モデルの逆深度（値が大きいほど近い）、下段はFP32へscale＋shift整合した差です。上段の深度色は画像ごとにFP32のmin/maxを共通利用し、下段はFP32レンジの±10%で共通表示。差分の青は負、白は0、赤は正です。元画像の緑丸がGTの近い点、赤丸が遠い点です。</div>
<p class="note">{e(preview_note)} プレビュー用の表示縮小・JPEG圧縮を含み、数値指標は圧縮前の出力から計算しています。共通色スケールの上段にはscale/shift差も表れるため、構造差は下段と数値も合わせて確認してください。</p>
<h3>各カテゴリの中央付近の例 · {len(representatives)}画像</h3>
<p class="note">{e(label(selected_variant)) if selected_variant else 'INT4'} のaligned NRMSEが各カテゴリの中央値に最も近い画像を選択。典型例を見るための再現可能な規則です。</p>{examples(representatives)}
<h3>INT8でも差が出た例 · 上位{len(int8_examples)}画像</h3>
<p class="note">INT8 dynamicのaligned NRMSE降順で選びました。GTの正答を維持していても深度マップが同一であることは保証されず、暗い室内と明るい窓を含む例でも差が見られました。全体の平均値だけでなく、使用する画像で確認する必要があります。</p>{examples(int8_examples)}
<h3>FP32とのマップ差が大きい例 · 上位{len(worst)}画像</h3>
<p class="note">{e(label(selected_variant)) if selected_variant else 'INT4'} のaligned NRMSE降順。ここでの「大きい差」はマップ差の順位であり、GT近遠正答率の悪化順位ではありません。誤答化したペア数も表で確認できます。</p>{examples(worst)}
<h3>FP32とのマップ差が小さい例 · 上位{len(best)}画像</h3>
<p class="note">同じ指標の昇順で選び、差の出やすい画像だけに偏らないよう併記します。</p>{examples(best)}</section>
<section class="section" id="method"><h2>測定条件と解釈できる範囲</h2>
<div class="table-wrap"><table class="method-table"><tbody>{method_html}</tbody></table></div>
<h3>量子化の内容</h3><dl class="definition">{variant_html}</dl>
<p class="callout">「INT8」「INT4」は対象となる重み・演算の設定名です。モデル全体の重みとactivationが一律に8bit・4bitであることは意味しません。INT4の内部A8許可版は4bit重みの影響に加えて、activation近似の影響を含む可能性があります。</p>
<p class="note">今回の量子化は重みから作るRTNまたは動的INT8です。別に抽出した校正用64画像はこれらの設定では使用していません。静的INT8・GPTQ・AWQ・量子化を考慮した再学習の結果ではありません。入力サイズ、前処理、量子化対象、ハードウェアが変われば結果も変わり得ます。</p></section>
<section class="section" id="sources"><h2>出典・追試用ファイル</h2>
<ul class="source-list">
<li><a href="https://github.com/DepthAnything/Depth-Anything-V2">Depth Anything V2 公式リポジトリ</a>：Smallの構成、標準前後処理、checkpointとライセンス。</li>
<li><a href="https://huggingface.co/depth-anything/Depth-Anything-V2-Small">作者のSmall checkpoint</a>：本実験のモデル取得元。</li>
<li><a href="https://arxiv.org/html/2406.09414v2">Depth Anything V2 論文</a>：affine-invariant inverse depthとDA-2Kの設計。</li>
<li><a href="https://github.com/DepthAnything/Depth-Anything-V2/blob/main/DA-2K.md">DA-2K 公式説明</a> / <a href="https://huggingface.co/datasets/depth-anything/DA-2K">作者のデータ配布元</a>：8カテゴリと近遠ペア注釈。配布物中の利用可能な注釈画像から抽出。</li>
<li><a href="https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html">ONNX Runtime 量子化ドキュメント</a>：動的INT8とINT4 weight-onlyの対応範囲。</li>
<li><a href="https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulnbits">MatMulNBits 公式演算仕様</a>：accuracy_level=1 / 4の意味。</li>
</ul><p>{resource_links}</p>
<h3>手元の画像で確認する</h3>
<p class="note">同じPC上で下のコマンドを実行すると、任意の画像について各モデルの比較を保存できます。自分の画像にGTがない場合、確認できるのは見た目とFP32出力との差で、GT近遠正答率ではありません。</p>
<pre>.venv/Scripts/python.exe scripts/compare_image.py --image "C:/path/to/image.jpg" --output-dir results/custom</pre>
<p class="note">依存パッケージを固定した再現用環境のインストール：</p>
<pre>.venv/Scripts/python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu</pre>
<details><summary>再現性情報・ハッシュ</summary><pre>Upstream commit: {e(metadata.get('upstream_commit','—'))}
Checkpoint SHA256: {e(metadata.get('checkpoint_sha256','—'))}
Evaluation manifest SHA256: {e(manifest_digest)}
DA-2K archive SHA256: {e(dataset.get('archive_sha256','—'))}
Result timestamp (UTC): {e(timestamp)}
Environment:
{e(json.dumps(environment,ensure_ascii=False,indent=2))}
Models:
{e(json.dumps({name:{'path':result.get('path'),'sha256':result.get('sha256')} for name,result in models.items()},ensure_ascii=False,indent=2))}</pre></details>
<p class="note">HTMLはオンラインJS・フォント・画像を読み込みません。比較画像と測定ファイルはローカルの相対リンクで参照します。外部リンクは公式一次資料を開くためのものです。</p></section>
<p class="footer">測定ファイルから自動生成 · レポート生成 {e(datetime.now().astimezone().isoformat(timespec='seconds'))} · 生成コマンド <code>python scripts/make_report.py</code></p>
</main></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--metadata", type=Path, default=ROOT / "models/metadata.json")
    parser.add_argument("--dataset-summary", type=Path, default=ROOT / "data/dataset_summary.json")
    parser.add_argument("--output", type=Path, help="Default: RESULTS_DIR/report.html")
    args = parser.parse_args()
    result_dir = args.results_dir.resolve()
    report_path = args.output.resolve() if args.output else result_dir / "report.html"
    summary = load_json(result_dir / "summary.json")
    image_rows = load_csv(result_dir / "per_image.csv")
    pair_rows = load_csv(result_dir / "per_pair.csv")
    metadata = load_json(args.metadata)
    dataset = load_json(args.dataset_summary)
    checkpoint_path = result_dir / "checkpoint.json"
    checkpoint = load_json(checkpoint_path) if checkpoint_path.is_file() else {}
    verify_consistency(summary, image_rows, pair_rows)
    html = build_report(summary, image_rows, pair_rows, metadata, dataset,
                        result_dir, report_path, checkpoint)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(report_path)
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
