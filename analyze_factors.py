"""Factor-accuracy analysis for AlgoTrader.

Reads the backtest dataset and detail report, computes per-factor
performance, identifies weak/redundant factors, and recommends threshold
tuning targets. Run after a deep backtest:

    python analyze_factors.py

Outputs: reports/factor_analysis.json
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


def load_dataset(path: str = "reports/dataset.parquet") -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_detail(path: str = "reports/backtest_detail.json") -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def factor_metrics(df: pd.DataFrame, min_samples: int = 25) -> dict:
    records = []
    factor_cols = [c for c in df.columns if c.startswith("factor__")]
    for col in factor_cols:
        name = col.replace("factor__", "")
        mask = df[col].notna() & (df[col].abs() > 1e-9)
        n = int(mask.sum())
        if n < min_samples:
            continue
        wins = df.loc[mask, "win"].astype(bool)
        rs = df.loc[mask, "r"]
        wr = float(wins.mean())
        exp = float(rs.mean())
        wl, wu = _wilson(wr, n)
        records.append({
            "factor": name,
            "n": n,
            "win_rate": round(wr, 4),
            "wilson_lower": round(wl, 4),
            "wilson_upper": round(wu, 4),
            "expectancy_r": round(exp, 4),
            "avg_win_r": round(float(rs[rs > 0].mean()) if (rs > 0).any() else 0, 4),
            "avg_loss_r": round(float(abs(rs[rs <= 0].mean())) if (rs <= 0).any() else 0, 4),
            "profit_factor": round(float(rs[rs > 0].sum() / abs(rs[rs <= 0].sum())), 4) if (rs <= 0).sum() != 0 else float("inf"),
            "mean_confidence": round(float(df.loc[mask, "confidence"].mean()), 4) if "confidence" in df else None,
        })
    return {r["factor"]: r for r in sorted(records, key=lambda x: x["expectancy_r"], reverse=True)}


def cooccurrence_analysis(df: pd.DataFrame, top_n: int = 50) -> dict:
    factor_cols = [c for c in df.columns if c.startswith("factor__")]
    active = {c.replace("factor__", ""): (df[c].notna() & (df[c].abs() > 1e-9)) for c in factor_cols}
    names = list(active.keys())
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = active[a] & active[b]
            n_both = int(both.sum())
            if n_both < 10:
                continue
            # Pearson-like correlation of occurrence
            corr = float(np.corrcoef(active[a].astype(int), active[b].astype(int))[0, 1])
            if abs(corr) > 0.5:
                pairs.append({
                    "a": a,
                    "b": b,
                    "cooccurrence": n_both,
                    "correlation": round(corr, 3),
                })
    return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:top_n]


def recommend(metrics: dict, min_edge: float = 0.05, min_samples: int = 25) -> dict:
    keep, drop, tune = [], [], []
    for name, m in metrics.items():
        if m["wilson_lower"] < 0.5 and m["expectancy_r"] < min_edge:
            drop.append({"factor": name, "reason": "negative/low edge", **m})
        elif m["wilson_lower"] >= 0.55 and m["expectancy_r"] >= min_edge:
            keep.append({"factor": name, "reason": "solid edge", **m})
        else:
            tune.append({"factor": name, "reason": "marginal — tune or gate", **m})
    return {"keep": keep, "tune": tune, "drop": drop}


def gate_analysis(df: pd.DataFrame) -> dict:
    """Find confidence and n_families thresholds that maximize expectancy."""
    out = {}
    for conf_cut in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = df["confidence"] >= conf_cut
        n = int(mask.sum())
        if n < 20:
            continue
        rs = df.loc[mask, "r"]
        out[f"conf>={conf_cut}"] = {
            "n": n,
            "win_rate": round(float((rs > 0).mean()), 4),
            "expectancy_r": round(float(rs.mean()), 4),
        }

    for fam_cut in [2, 3, 4, 5]:
        mask = df["n_families"] >= fam_cut
        n = int(mask.sum())
        if n < 20:
            continue
        rs = df.loc[mask, "r"]
        out[f"families>={fam_cut}"] = {
            "n": n,
            "win_rate": round(float((rs > 0).mean()), 4),
            "expectancy_r": round(float(rs.mean()), 4),
        }

    for conf_cut in [0.60, 0.70, 0.80]:
        for fam_cut in [3, 4]:
            mask = (df["confidence"] >= conf_cut) & (df["n_families"] >= fam_cut)
            n = int(mask.sum())
            if n < 20:
                continue
            rs = df.loc[mask, "r"]
            out[f"conf>={conf_cut}_fam>={fam_cut}"] = {
                "n": n,
                "win_rate": round(float((rs > 0).mean()), 4),
                "expectancy_r": round(float(rs.mean()), 4),
            }
    return out


def main():
    df = load_dataset()
    detail = load_detail()

    if df is None:
        print("No reports/dataset.parquet found. Run backtest.py --export-dataset first.")
        return

    print(f"Loaded dataset: {len(df)} trades, {len([c for c in df.columns if c.startswith('factor__')])} factor columns")

    metrics = factor_metrics(df)
    pairs = cooccurrence_analysis(df)
    recs = recommend(metrics)
    gates = gate_analysis(df)

    report = {
        "n_trades": len(df),
        "n_factors_measured": len(metrics),
        "factor_metrics": metrics,
        "redundant_pairs": pairs,
        "recommendations": recs,
        "gate_analysis": gates,
    }

    out = Path("reports/factor_analysis.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Wrote {out}")
    print(f"  Keep: {len(recs['keep'])}, Tune: {len(recs['tune'])}, Drop: {len(recs['drop'])}")
    print(f"  Redundant pairs flagged: {len(pairs)}")
    print("\nTop keep factors:")
    for f in recs["keep"][:10]:
        print(f"  {f['factor']:35s} WR={f['win_rate']:.2f} EV={f['expectancy_r']:.3f} n={f['n']}")
    print("\nTop drop factors:")
    for f in recs["drop"][:10]:
        print(f"  {f['factor']:35s} WR={f['win_rate']:.2f} EV={f['expectancy_r']:.3f} n={f['n']}")

    print("\nGate analysis (higher thresholds):")
    for label, m in sorted(gates.items(), key=lambda x: x[1]["expectancy_r"], reverse=True)[:10]:
        print(f"  {label:25s} EV={m['expectancy_r']:.3f} WR={m['win_rate']:.2f} n={m['n']}")


if __name__ == "__main__":
    main()
