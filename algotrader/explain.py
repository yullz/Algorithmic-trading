"""Generate a plain-English explanation of WHY a trade plan is expected to be
profitable — and, just as importantly, why it might lose.

The goal is transparency: every number the engine acts on is turned into a
sentence a human can sanity-check. Nothing here changes the decision; it only
narrates the decision the risk engine already made.
"""
from __future__ import annotations

from collections import defaultdict

from .models import Bias, SetupKind, Side, TakeProfit
from .signals.confluence import family_of

_FAMILY_LABEL = {
    "trend": "trend", "momentum": "momentum", "mean_reversion": "mean-reversion",
    "volume": "volume", "volatility": "volatility", "candle_reversal":
    "candlestick reversal", "chart": "chart structure", "structure":
    "support/resistance structure", "liquidity": "liquidity/stop-hunt",
    "divergence": "divergence", "derivatives": "funding/open-interest",
    "relative_strength": "relative strength", "other": "other",
}

_COUNTER = {
    SetupKind.BREAKOUT: "the breakout fails and price snaps back into the range (a trap)",
    SetupKind.REVERSAL: "the prior trend resumes and overruns the reversal",
    SetupKind.CONTINUATION: "the trend reverses against the continuation",
    SetupKind.MOMENTUM: "momentum stalls and price mean-reverts",
    SetupKind.MEAN_REVERSION: "price keeps trending instead of reverting",
}


def explain_trade(sig, *, entry: float, stop: float, take_profits: list[TakeProfit],
                  leverage: float, liq: float, win_rate: float,
                  avg_win_r: float, avg_loss_r: float, fees_r: float, ev_r: float,
                  reward_risk: float, calibration: dict, calibrated: bool,
                  warnings: list[str]) -> list[str]:
    dir_word = "higher" if sig.side == Side.LONG else "lower"
    bias_word = "bullish" if sig.side == Side.LONG else "bearish"

    # group agreeing evidence by independent family
    by_fam: dict[str, list] = defaultdict(list)
    for e in sig.evidence:
        by_fam[family_of(e)].append(e)
    fam_bits = [f"{_FAMILY_LABEL.get(f, f)} ({', '.join(e.name for e in evs)})"
                for f, evs in by_fam.items()]

    # strongest historical contributor (by calibrated/known base rate)
    def rate_of(e):
        v = calibration.get(e.name)
        if isinstance(v, dict):
            return v.get("rate", e.base_win_rate)
        return v if v is not None else e.base_win_rate
    best = max(sig.evidence, key=rate_of) if sig.evidence else None
    n_cal = sum(1 for e in sig.evidence if e.name in calibration)
    n_uncal = len(sig.evidence) - n_cal

    stop_pct = abs(entry - stop) / entry
    liq_pct = abs(entry - liq) / entry
    final_r = max((t.r_multiple for t in take_profits), default=1.0)

    out: list[str] = []

    # 1) Thesis
    regime = getattr(sig, "regime", "")
    regime_bit = f" Market regime: {regime.replace('_', ' ')}." if regime else ""
    out.append(
        f"Thesis - {sig.side.value} {sig.symbol} on {sig.timeframe}: {len(by_fam)} "
        f"independent signal {'family' if len(by_fam)==1 else 'families'} agree it should go "
        f"{dir_word} ({bias_word}) - {'; '.join(fam_bits)}. Confidence {sig.confidence:.0%}."
        f"{regime_bit}")

    # 2) Why it may work
    driver = (
        f"winners average {avg_win_r:.2f}R while losers average {avg_loss_r:.2f}R, so the "
        f"positive edge comes from payoff asymmetry (wins run further than losses)"
        if avg_win_r > avg_loss_r else
        f"winners ({avg_win_r:.2f}R) don't outrun losers ({avg_loss_r:.2f}R), so the edge "
        f"leans on the {win_rate:.0%} hit rate")
    best_bit = (f"Its strongest historically-measured factor is {best.name} "
                f"({rate_of(best):.0%} past hit rate). " if best else "")
    ml_prob = getattr(sig, "ml_prob", None)
    ml_bit = ""
    if ml_prob is not None:
        contribs = getattr(sig, "ml_contribs", []) or []
        contrib_txt = f" (top drivers: {', '.join(contribs[:3])})" if contribs else ""
        ml_bit = (f" An ML meta-model trained on past backtest trades rates this "
                  f"setup {ml_prob:.0%}{contrib_txt}, blended at "
                  f"{getattr(sig, 'ml_weight', 0.0):.0%} weight.")
    out.append(
        f"Why it may be profitable - blended win rate {win_rate:.0%}. {best_bit}"
        f"After {fees_r:.2f}R round-trip fees, {driver}; net expected value "
        f"{ev_r:+.2f}R per trade with a {reward_risk:.2f} expected reward:risk and room to "
        f"{final_r:.0f}R if the full target ladder fills.{ml_bit}")

    # 3) Why it may lose
    counter = _COUNTER.get(sig.kind, "price moves against the position to the stop")
    out.append(
        f"Why it may lose - the idea is wrong if price closes back beyond {stop:.4f} "
        f"({stop_pct:.1%} from entry); the classic failure here is that {counter}. "
        f"You lose ~1R (risk is fixed at {abs(entry-stop)/entry:.1%} of price to the stop). "
        f"Liquidation sits at {liq:.4f} ({liq_pct:.1%} away, beyond the stop) so at "
        f"{leverage:.1f}x the stop should trigger before liquidation.")

    # 4) Caveats
    caveats = []
    if not calibrated:
        caveats.append("win rates are UNCALIBRATED priors - run backtest.py first")
    elif n_uncal:
        caveats.append(f"{n_uncal} of {len(sig.evidence)} factors lack >=25 backtest "
                       f"samples and use priors, so treat the win rate as provisional")
    caveats.extend(warnings)
    caveats.append("figures are in-sample historical estimates, not a forecast")
    out.append("Caveats - " + "; ".join(caveats) + ".")
    return out
