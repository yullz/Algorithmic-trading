"""Human-readable rendering of a TradePlan (rich if available, else plain)."""
from __future__ import annotations

from ..models import TradePlan

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    _console = Console()
    _HAS_RICH = True
except Exception:  # rich optional
    _HAS_RICH = False


def format_plan_plain(p: TradePlan) -> str:
    tps = "  ".join(f"TP{i+1} {t.price:.4f}({t.r_multiple:.0f}R,{t.allocation:.0%})"
                    for i, t in enumerate(p.take_profits))
    lines = [
        f"=== {p.symbol} {p.timeframe}  {p.side.value} ===",
        f" entry       : {p.entry:.4f}",
        f" stop-loss   : {p.stop_loss:.4f}   (risk {p.risk_amount:.2f} quote)",
        f" take-profit : {tps}",
        f" leverage    : {p.leverage:.1f}x    liquidation: {p.liquidation_price:.4f}",
        f" size        : {p.qty:.6f} base  |  notional {p.notional:.2f}  |  margin {p.margin:.2f}",
        f" reward:risk : {p.reward_risk:.2f}  |  est win rate {p.expected_win_rate:.0%}  |  EV {p.expected_value_r:+.2f}R",
        f" confidence  : {p.confidence:.0%}   fees~{p.fees_estimate:.2f}",
    ]
    for para in p.explanation:
        lines.append(f"   {para}")
    if p.warnings:
        lines += [f" [!] {w}" for w in p.warnings]
    return "\n".join(lines)


def render_plan(p: TradePlan) -> None:
    if not _HAS_RICH:
        print(format_plan_plain(p))
        return
    color = "green" if p.side.value == "LONG" else "red"
    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_row("Side", f"[bold {color}]{p.side.value}[/] @ {p.entry:.4f}")
    t.add_row("Stop-loss", f"{p.stop_loss:.4f}  (risk {p.risk_amount:.2f})")
    for i, tp in enumerate(p.take_profits):
        t.add_row(f"TP{i+1}", f"{tp.price:.4f}  {tp.r_multiple:.0f}R  ({tp.allocation:.0%})")
    t.add_row("Leverage", f"{p.leverage:.1f}x")
    t.add_row("Liquidation", f"[yellow]{p.liquidation_price:.4f}[/]")
    t.add_row("Size", f"{p.qty:.6f} base | notional {p.notional:.2f} | margin {p.margin:.2f}")
    t.add_row("Reward:Risk", f"{p.reward_risk:.2f}")
    t.add_row("Est. win rate", f"{p.expected_win_rate:.0%}")
    t.add_row("Expected value", f"[bold]{p.expected_value_r:+.2f}R[/]")
    t.add_row("Confidence", f"{p.confidence:.0%}")
    for para in p.explanation:
        label, _, body = para.partition(" - ")
        t.add_row(f"[dim]{label}[/]", body or label)
    subtitle = " | ".join(p.warnings) if p.warnings else "paper-first - not financial advice"
    _console.print(Panel(t, title=f"{p.symbol}  {p.timeframe}",
                         subtitle=f"[dim]{subtitle}[/]", border_style=color))
