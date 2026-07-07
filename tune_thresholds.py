"""Strength-threshold tuning for individual evidence factors.

For each factor measured in the backtest dataset, this script finds the
strength cutoff that maximizes expectancy per trade — but it does so with a
NESTED walk-forward so the reported edge is measured out-of-sample. The cutoff
is chosen on training folds and evaluated only on the held-out test folds, so a
threshold that merely fits noise in the sample is not rewarded. (Previously the
cutoff was picked and scored on the same rows — pure data-snooping.)

Run after a deep backtest:

    python tune_thresholds.py

Outputs: reports/threshold_tuning.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from algotrader.backtest.selection import tune_threshold_cv


def optimal_threshold(values: pd.Series, outcomes: pd.Series,
                      times: pd.Series | None = None, n_bins: int = 9) -> dict | None:
    """Nested walk-forward strength-threshold tuning (out-of-sample honest).

    Delegates to selection.tune_threshold_cv: the cutoff is chosen on train
    folds and the returned expectancy/win-rate/Wilson-lower are pooled across
    the held-out test folds only.
    """
    return tune_threshold_cv(values, outcomes, times=times)


def main():
    path = Path("reports/dataset.parquet")
    if not path.exists():
        print("No reports/dataset.parquet found. Run backtest.py --export-dataset first.")
        return

    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} trades")
    times = df["entry_time"] if "entry_time" in df.columns else None

    results = {}
    for col in [c for c in df.columns if c.startswith("factor__")]:
        name = col.replace("factor__", "")
        rec = optimal_threshold(df[col], df["r"], times)
        if rec:
            results[name] = rec

    # Sort by OUT-OF-SAMPLE expectancy.
    results = dict(sorted(results.items(),
                          key=lambda x: x[1]["oos_expectancy_r"], reverse=True))

    out = Path("reports/threshold_tuning.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {out} ({len(results)} factors tuned, OUT-OF-SAMPLE)")
    print("\nTop strength thresholds (OOS expectancy):")
    for name, rec in list(results.items())[:15]:
        print(f"  {name:35s} >= {rec['threshold_median']:.2f}"
              f"  OOS_EV={rec['oos_expectancy_r']:.3f}"
              f"  WR={rec['oos_win_rate']:.2f}"
              f"  n={rec['oos_n']}  folds={rec['n_folds_used']}")


if __name__ == "__main__":
    main()
