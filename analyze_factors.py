"""Factor-accuracy analysis for AlgoTrader.

Reads the backtest dataset and detail report, computes per-factor performance,
and recommends which factors to keep/tune/drop. Two honesty upgrades over a
naive scan:

  * Every factor's win rate is tested against the dataset BASE RATE with a
    one-sided binomial p-value, and those p-values are corrected for the number
    of factors tested with Benjamini-Hochberg (FDR). A factor is only "kept" if
    it survives FDR — this kills the multiple-comparisons illusion where, out of
    dozens of factors, several look significant by chance.
  * Gate tuning (confidence / n_families cutoffs) is chosen on training folds
    and scored out-of-sample, not grid-searched on the full sample.

Run after a deep backtest:

    python analyze_factors.py

Outputs: reports/factor_analysis.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from algotrader.backtest.selection import (
    anchored_folds, benjamini_hochberg, binom_p_greater, time_order,
    wilson_interval)

_PF_CAP = 999.99  # finite profit-factor sentinel (json rejects Infinity)


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


def factor_metrics(df: pd.DataFrame, min_samples: int = 25,
                   fdr_q: float = 0.10) -> dict:
    """Per-factor performance with FDR-corrected significance vs the base rate."""
    base_rate = float(df["win"].mean()) if "win" in df and len(df) else 0.5
    records = []
    pvals: dict[str, float] = {}
    factor_cols = [c for c in df.columns if c.startswith("factor__")]
    for col in factor_cols:
        name = col.replace("factor__", "")
        mask = df[col].notna() & (df[col].abs() > 1e-9)
        n = int(mask.sum())
        if n < min_samples:
            continue
        wins_bool = df.loc[mask, "win"].astype(bool)
        rs = df.loc[mask, "r"]
        wr = float(wins_bool.mean())
        exp = float(rs.mean())
        wl, wu = wilson_interval(wr, n)
        losses_sum = abs(float(rs[rs <= 0].sum()))
        pf = round(float(rs[rs > 0].sum() / losses_sum), 4) if losses_sum != 0 \
            else (_PF_CAP if (rs > 0).any() else 0.0)
        # One-sided binomial p-value that this factor beats the base win rate.
        p_val = binom_p_greater(int(wins_bool.sum()), n, base_rate)
        pvals[name] = p_val
        records.append({
            "factor": name,
            "n": n,
            "win_rate": round(wr, 4),
            "wilson_lower": round(wl, 4),
            "wilson_upper": round(wu, 4),
            "expectancy_r": round(exp, 4),
            "avg_win_r": round(float(rs[rs > 0].mean()) if (rs > 0).any() else 0, 4),
            "avg_loss_r": round(float(abs(rs[rs <= 0].mean())) if (rs <= 0).any() else 0, 4),
            "profit_factor": pf,
            "mean_confidence": round(float(df.loc[mask, "confidence"].mean()), 4) if "confidence" in df else None,
            "base_rate": round(base_rate, 4),
            "p_value": round(p_val, 6),
        })
    # Benjamini-Hochberg across every factor tested -> FDR-significant flag.
    bh = benjamini_hochberg(pvals, q=fdr_q)
    for rec in records:
        info = bh.get(rec["factor"], {})
        rec["p_adjusted"] = info.get("p_adjusted")
        rec["fdr_significant"] = bool(info.get("significant", False))
    return {r["factor"]: r for r in sorted(records, key=lambda x: x["expectancy_r"], reverse=True)}


def cooccurrence_analysis(df: pd.DataFrame, top_n: int = 50) -> list:
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
            corr = float(np.corrcoef(active[a].astype(int), active[b].astype(int))[0, 1])
            if abs(corr) > 0.5:
                pairs.append({"a": a, "b": b, "cooccurrence": n_both,
                              "correlation": round(corr, 3)})
    return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:top_n]


def recommend(metrics: dict, min_edge: float = 0.05) -> dict:
    """Keep only factors that clear the edge thresholds AND survive FDR — a
    factor that looks good but is not FDR-significant is a likely false
    discovery and is routed to 'tune', not 'keep'."""
    keep, drop, tune = [], [], []
    for name, m in metrics.items():
        sig = m.get("fdr_significant", False)
        if m["wilson_lower"] < 0.5 and m["expectancy_r"] < min_edge:
            drop.append({"factor": name, "reason": "negative/low edge", **m})
        elif sig and m["wilson_lower"] >= 0.55 and m["expectancy_r"] >= min_edge:
            keep.append({"factor": name, "reason": "solid edge (FDR-significant)", **m})
        elif m["wilson_lower"] >= 0.55 and m["expectancy_r"] >= min_edge:
            tune.append({"factor": name,
                         "reason": "edge not FDR-significant — likely noise", **m})
        else:
            tune.append({"factor": name, "reason": "marginal — tune or gate", **m})
    return {"keep": keep, "tune": tune, "drop": drop}


def _gate_masks(df: pd.DataFrame) -> dict:
    """Candidate gate configs -> boolean masks over df rows."""
    gates: dict[str, np.ndarray] = {}
    for c in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        gates[f"conf>={c}"] = (df["confidence"] >= c).to_numpy()
    for fam in [2, 3, 4, 5]:
        gates[f"families>={fam}"] = (df["n_families"] >= fam).to_numpy()
    for c in [0.60, 0.70, 0.80]:
        for fam in [3, 4]:
            gates[f"conf>={c}_fam>={fam}"] = (
                (df["confidence"] >= c) & (df["n_families"] >= fam)).to_numpy()
    return gates


def gate_analysis(df: pd.DataFrame) -> dict:
    """In-sample gate table PLUS an out-of-sample nested-CV result: the best gate
    is chosen on train folds and scored on the held-out test folds, so the
    headline number is not the best-of-many in-sample gate (data-snooping)."""
    r = df["r"].to_numpy()
    gates = _gate_masks(df)

    in_sample = {}
    for label, mask in gates.items():
        n = int(mask.sum())
        if n < 20:
            continue
        rs = r[mask]
        in_sample[label] = {"n": n, "win_rate": round(float((rs > 0).mean()), 4),
                            "expectancy_r": round(float(rs.mean()), 4)}

    # Nested CV: pick the highest-expectancy gate on each train fold, score OOS.
    order = time_order(df["entry_time"] if "entry_time" in df.columns else None)
    if order is None:
        order = np.arange(len(df))
    folds = anchored_folds(order, n_folds=5)
    oos_r, picks = [], []
    for train, test in folds:
        best_label, best_exp = None, None
        for label, mask in gates.items():
            tr = train[mask[train]]
            if len(tr) < 20:
                continue
            exp = float(r[tr].mean())
            if best_exp is None or exp > best_exp:
                best_exp, best_label = exp, label
        if best_label is None:
            continue
        picks.append(best_label)
        te = test[gates[best_label][test]]
        if len(te) >= 10:
            oos_r.extend(r[te].tolist())

    oos = None
    if oos_r and picks:
        arr = np.asarray(oos_r, dtype=float)
        wl, _ = wilson_interval(float((arr > 0).mean()), len(arr))
        oos = {
            "selected_gates": picks,
            "oos_n": len(arr),
            "oos_win_rate": round(float((arr > 0).mean()), 4),
            "oos_wilson_lower": round(wl, 4),
            "oos_expectancy_r": round(float(arr.mean()), 4),
        }
    return {"in_sample": in_sample, "out_of_sample_selected": oos}


def main():
    df = load_dataset()
    detail = load_detail()

    if df is None:
        print("No reports/dataset.parquet found. Run backtest.py --export-dataset first.")
        return

    print(f"Loaded dataset: {len(df)} trades, "
          f"{len([c for c in df.columns if c.startswith('factor__')])} factor columns")

    metrics = factor_metrics(df)
    pairs = cooccurrence_analysis(df)
    recs = recommend(metrics)
    gates = gate_analysis(df)

    report = {
        "n_trades": len(df),
        "base_rate": round(float(df["win"].mean()), 4) if "win" in df else None,
        "n_factors_measured": len(metrics),
        "n_fdr_significant": sum(1 for m in metrics.values() if m.get("fdr_significant")),
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
    print(f"  FDR-significant factors: {report['n_fdr_significant']}/{len(metrics)}")
    print(f"  Redundant pairs flagged: {len(pairs)}")
    print("\nTop keep factors (FDR-significant):")
    for f in recs["keep"][:10]:
        print(f"  {f['factor']:35s} WR={f['win_rate']:.2f} EV={f['expectancy_r']:.3f} "
              f"p_adj={f.get('p_adjusted')} n={f['n']}")
    if gates.get("out_of_sample_selected"):
        g = gates["out_of_sample_selected"]
        print(f"\nGate CV (out-of-sample selected): EV={g['oos_expectancy_r']:.3f} "
              f"WR={g['oos_win_rate']:.2f} n={g['oos_n']}")


if __name__ == "__main__":
    main()
