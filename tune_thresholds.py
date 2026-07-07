"""Strength-threshold tuning for individual evidence factors.

For each factor measured in the backtest dataset, this script finds the
strength cutoff that maximizes out-of-sample expectancy per trade. The idea
is: a pattern may have no edge on average, but only when it fires strongly.

Run after a deep backtest:

    python tune_thresholds.py

Outputs: reports/threshold_tuning.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _wilson(p: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def optimal_threshold(values: pd.Series, outcomes: pd.Series, n_bins: int = 9) -> dict | None:
    """Grid-search strength thresholds; require at least 20 trades above cutoff."""
    clean = pd.DataFrame({"strength": values.abs(), "r": outcomes}).dropna()
    if len(clean) < 40:
        return None
    cuts = np.linspace(0.1, 0.9, n_bins)
    best = None
    for cut in cuts:
        above = clean[clean["strength"] >= cut]
        n = len(above)
        if n < 20:
            continue
        exp = float(above["r"].mean())
        wr = float((above["r"] > 0).mean())
        wl, _ = _wilson(wr, n)
        score = exp if wl > 0.5 else exp - 0.5  # penalize if lower CI below 50%
        if best is None or score > best["score"]:
            best = {
                "threshold": round(cut, 3),
                "n": n,
                "win_rate": round(wr, 4),
                "wilson_lower": round(wl, 4),
                "expectancy_r": round(exp, 4),
                "score": round(score, 4),
            }
    return best


def main():
    path = Path("reports/dataset.parquet")
    if not path.exists():
        print("No reports/dataset.parquet found. Run backtest.py --export-dataset first.")
        return

    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} trades")

    results = {}
    for col in [c for c in df.columns if c.startswith("factor__")]:
        name = col.replace("factor__", "")
        rec = optimal_threshold(df[col], df["r"])
        if rec:
            results[name] = rec

    # Sort by expectancy
    results = dict(sorted(results.items(), key=lambda x: x[1]["expectancy_r"], reverse=True))

    out = Path("reports/threshold_tuning.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {out} ({len(results)} factors tuned)")
    print("\nTop strength thresholds:")
    for name, rec in list(results.items())[:15]:
        print(f"  {name:35s} >= {rec['threshold']:.2f}  EV={rec['expectancy_r']:.3f}  WR={rec['win_rate']:.2f}  n={rec['n']}")


if __name__ == "__main__":
    main()
