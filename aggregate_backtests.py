"""Aggregate batched backtest outputs into a single dataset and report.

Run after multiple backtest.py --output-suffix batchN passes:

    python aggregate_backtests.py

Outputs:
    reports/backtest_aggregated.json
    reports/backtest_detail_aggregated.json
    reports/dataset_aggregated.parquet
    calibration_aggregated.json
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def aggregate():
    detail_files = sorted(Path("reports").glob("backtest_detail_*.json"))
    dataset_files = sorted(Path("reports").glob("dataset_*.parquet"))

    if not detail_files:
        print("No reports/backtest_detail_*.json files found.")
        return

    all_detail = []
    all_trades = []
    for f in detail_files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        all_detail.extend(data.get("pairs", []))

    for f in dataset_files:
        df = pd.read_parquet(f)
        all_trades.append(df)

    combined_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    rs = combined_df["r"].to_numpy() if not combined_df.empty and "r" in combined_df.columns else np.array([])
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    summary = {
        "trades": int(len(rs)),
        "win_rate": float((rs > 0).mean()) if len(rs) else 0.0,
        "expectancy_r": float(rs.mean()) if len(rs) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf"),
        "max_drawdown_r": 0.0,
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(abs(losses.mean())) if len(losses) else 0.0,
    }
    if len(rs):
        eq = np.cumsum(rs)
        peak = np.maximum.accumulate(eq)
        summary["max_drawdown_r"] = float((peak - eq).max())

    # Per-factor aggregate stats
    factor_stats: dict[str, dict] = defaultdict(lambda: {"wins": 0, "n": 0})
    if not combined_df.empty:
        factor_cols = [c for c in combined_df.columns if c.startswith("factor__")]
        for col in factor_cols:
            name = col.replace("factor__", "")
            mask = combined_df[col].notna() & (combined_df[col].abs() > 1e-9)
            n = int(mask.sum())
            if n == 0:
                continue
            fwins = int(combined_df.loc[mask, "win"].sum())
            factor_stats[name]["wins"] += fwins
            factor_stats[name]["n"] += n

    factor_win_rate = {k: v["wins"] / v["n"] for k, v in factor_stats.items() if v["n"] > 0}

    agg_report = {
        "summary": summary,
        "factor_win_rate": {k: round(v, 4) for k, v in factor_win_rate.items()},
        "n_trades": summary["trades"],
        "n_pairs": len(all_detail),
    }

    Path("reports").mkdir(exist_ok=True)
    with open("reports/backtest_aggregated.json", "w", encoding="utf-8") as f:
        json.dump(agg_report, f, indent=2)

    with open("reports/backtest_detail_aggregated.json", "w", encoding="utf-8") as f:
        json.dump({"pairs": all_detail, "n_pairs": len(all_detail)}, f, indent=2)

    if not combined_df.empty:
        combined_df.to_parquet("reports/dataset_aggregated.parquet", index=False)

    # Write aggregate calibration (simple rates, no shrinkage/recency here)
    calibration = {}
    for k, v in factor_stats.items():
        if v["n"] >= 25:
            wr = v["wins"] / v["n"]
            wl, wu = _wilson(wr, v["n"])
            calibration[k] = {
                "rate": round(wr, 4),
                "raw": round(wr, 4),
                "weighted": round(wr, 4),
                "n": v["n"],
                "eff_n": float(v["n"]),
                "wilson_lower": round(wl, 4),
                "wilson_upper": round(wu, 4),
            }
    calibration["_overall"] = summary["win_rate"]
    calibration["_avg_win_r"] = summary["avg_win_r"]
    calibration["_avg_loss_r"] = summary["avg_loss_r"]
    with open("calibration_aggregated.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print(f"Aggregated {len(all_detail)} pairs -> {summary['trades']} trades")
    print(json.dumps(summary, indent=2))
    print(f"Wrote reports/backtest_aggregated.json, reports/dataset_aggregated.parquet, calibration_aggregated.json")


if __name__ == "__main__":
    aggregate()
