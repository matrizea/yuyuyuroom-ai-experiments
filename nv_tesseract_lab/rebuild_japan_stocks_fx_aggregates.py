"""Rebuild aggregate tables from saved inference records, without model execution."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from run_japan_stocks_fx_backtest import STOCKS, aggregate_one_stock, aggregate_panel


def main() -> None:
    results = Path(__file__).resolve().parent / "results" / "japan_stocks_fx"
    records = json.loads((results / "per_origin.json").read_text(encoding="utf-8"))
    per_stock: list[dict] = []
    for symbol in STOCKS:
        stock_records = [row for row in records if row["symbol"] == symbol]
        per_stock.extend(
            {"symbol": symbol, **asdict(row)} for row in aggregate_one_stock(stock_records)
        )
    panel = aggregate_panel(records)
    pd.DataFrame(per_stock).to_csv(results / "per_stock_aggregate.csv", index=False)
    pd.DataFrame(panel).to_csv(results / "panel_aggregate.csv", index=False)
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["panel_aggregate"] = panel
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(panel).to_string(index=False))


if __name__ == "__main__":
    main()
